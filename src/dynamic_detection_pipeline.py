"""
动态数据检测流水线（核心演示模块）

使用 dataset 数据模拟动态数据流，端到端演示：
采样 -> 特征提取 -> 漂移检测 -> 判断 -> 增量训练(可选) -> 分级 -> 报告生成

数据模拟策略：
1. dataset/train/ 作为历史基准数据（已有基准特征在 baseline_assets/ 中）
2. dataset/val/ 按随机打乱后分为 4 个时间窗口（每窗口约 40 张），模拟不同时间段到达的新数据
3. 对第 3 个窗口的图像施加 HSV 扰动（亮度/饱和度偏移），模拟季节变化导致的数据漂移
4. 第 4 个窗口在增量训练后再检测，验证漂移是否缓解
"""

import os
import sys
import cv2
import json
import pickle
import shutil
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    get_logger,
    compute_mmd,
    feature_level_tests,
    nearest_neighbor_anomaly,
    estimate_gamma,
    BASE_DIR,
    BASELINE_ASSETS_DIR,
    DATASET_DIR,
    REPORTS_DIR,
    WEIGHTS_DIR,
    SAMPLING_DIR
)
from src.online_sampler import OnlineSampler
from src.color_grader import ColorGrader
from src.drift_detector import DriftDetector
from src.drift_report import DriftReportGenerator
from src.drift_visualizer import DriftVisualizer
from src.auto_trainer import AutoTrainer

logger = get_logger(__name__)


class DynamicDetectionPipeline:
    """
    动态数据检测流水线

    使用 dataset 数据模拟动态数据流，端到端演示：
    采样 -> 特征提取 -> 漂移检测 -> 判断 -> 增量训练(可选) -> 分级 -> 报告生成
    """

    def __init__(self, base_model_path: Optional[str] = None, dataset_dir: Optional[str] = None):
        """
        初始化动态检测流水线，加载所有子模块

        Args:
            base_model_path: 基础模型路径，默认使用 runs/classify/train2/weights/best.pt
            dataset_dir: 数据集根目录，默认使用 dataset/
        """
        # 调试信息：详细检查 CUDA 状态
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA 是否可用: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA 版本: {torch.version.cuda}")
            logger.info(f"GPU 数量: {torch.cuda.device_count()}")
            logger.info(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        
        self.device = 'cuda' if cuda_available else 'cpu'
        logger.info(f"最终选择运行设备: {self.device}")

        # 设置路径
        self.base_model_path = base_model_path or os.path.join(WEIGHTS_DIR, "best.pt")
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.train_dir = os.path.join(self.dataset_dir, "train")

        # 确保报告目录存在
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # 初始化子模块
        logger.info("初始化流水线子模块...")

        # 在线采样器 - 用于模拟流式数据
        self.sampler = OnlineSampler(source_dir=self.val_dir)

        # 颜色分级器 - 用于HSV颜色分级
        self.color_grader = ColorGrader(season='spring')

        # 漂移检测器 - 用于四层级漂移检测
        self.drift_detector = DriftDetector()

        # 漂移可视化器 - 用于生成图表
        self.visualizer = DriftVisualizer(output_dir=REPORTS_DIR)

        # 自动训练器 - 用于增量训练
        self.auto_trainer = AutoTrainer(
            base_model_path=self.base_model_path,
            dataset_dir=self.dataset_dir
        )

        # 加载PCA和Scaler（用于特征提取）
        self.pca_scaler_path = os.path.join(BASELINE_ASSETS_DIR, "pca_scaler.pkl")
        self.pca = None
        self.scaler = None
        self._load_pca_scaler()

        # 流水线状态
        self.windows: List[Dict] = []
        self.window_results: List[Dict] = []
        self.current_model_path = self.base_model_path
        self.has_triggered_training = False
        self.training_result = None

        # 特征提取相关
        self.feature_model = None
        self.target_layer_idx = None
        self._hook_handle = None
        self._current_batch_features = []

        logger.info(f"DynamicDetectionPipeline 初始化完成")
        logger.info(f"基础模型: {self.base_model_path}")
        logger.info(f"数据集目录: {self.dataset_dir}")
        logger.info(f"验证集目录: {self.val_dir}")

    def _load_pca_scaler(self):
        """加载已保存的PCA和Scaler"""
        try:
            if os.path.exists(self.pca_scaler_path):
                with open(self.pca_scaler_path, "rb") as f:
                    data = pickle.load(f)
                self.scaler = data["scaler"]
                self.pca = data["pca"]
                logger.info(f"已加载PCA和Scaler: {self.pca_scaler_path}")
            else:
                logger.warning(f"未找到PCA和Scaler文件: {self.pca_scaler_path}")
        except Exception as e:
            logger.error(f"加载PCA和Scaler失败: {e}")

    def _init_feature_extractor(self, model_path: Optional[str] = None):
        """
        初始化特征提取模型

        Args:
            model_path: 模型路径，默认使用当前模型
        """
        model_path = model_path or self.current_model_path

        if self.feature_model is not None:
            # 清理旧的hook
            if self._hook_handle is not None:
                self._hook_handle.remove()
            self._hook_handle = None

        # 加载模型
        self.feature_model = YOLO(model_path)

        # 锁定分类头前一层
        layers = list(self.feature_model.model.model)
        self.target_layer_idx = len(layers) - 2
        logger.info(f"锁定特征层: 索引 [{self.target_layer_idx}], 类型 [{layers[self.target_layer_idx].__class__.__name__}]")

        # 注册hook
        self._register_hook()

    def _hook_fn(self, module, input, output):
        """Hook函数，用于捕获特征"""
        if isinstance(output, (list, tuple)):
            output = output[0]
        feat = output.detach().cpu()
        if feat.dim() == 4:
            feat = torch.mean(feat, dim=[2, 3])
        self._current_batch_features.extend(feat.numpy())

    def _register_hook(self):
        """注册特征提取hook"""
        if self.feature_model is None:
            return
        layer = list(self.feature_model.model.model)[self.target_layer_idx]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def prepare_windows(self, n_windows: int = 4, perturbation_window: int = 2, window_size: int = 40) -> List[Dict]:
        """
        准备模拟时间窗口数据

        使用 OnlineSampler.simulate_stream 分割验证集

        Args:
            n_windows: 窗口数量，默认4个
            perturbation_window: 哪个窗口施加扰动（0-indexed, 默认第3个窗口即index=2）
            window_size: 每个窗口的样本数，默认40张

        Returns:
            List[Dict]: 窗口信息列表
        """
        logger.info("=" * 60)
        logger.info("准备模拟时间窗口数据")
        logger.info("=" * 60)
        logger.info(f"窗口数量: {n_windows}")
        logger.info(f"每窗口大小: {window_size}")
        logger.info(f"扰动窗口索引: {perturbation_window} (第{perturbation_window + 1}个窗口)")

        # 使用OnlineSampler模拟流式数据
        self.windows = self.sampler.simulate_stream(
            dataset_dir=self.val_dir,
            window_size=window_size,
            n_windows=n_windows,
            shuffle=True
        )

        # 标记扰动窗口
        for i, window in enumerate(self.windows):
            window['has_perturbation'] = (i == perturbation_window)
            window['window_id'] = i

        logger.info(f"已创建 {len(self.windows)} 个时间窗口")
        for i, window in enumerate(self.windows):
            perturbation_mark = " [HSV扰动]" if window['has_perturbation'] else ""
            logger.info(f"窗口 {i+1}: {window['size']} 张图片{perturbation_mark}")

        return self.windows

    def apply_hsv_perturbation(self, image_paths: List[str],
                               h_shift: int = 20,
                               s_scale: float = 0.75,
                               v_scale: float = 0.75,
                               output_dir: Optional[str] = None) -> List[str]:
        """
        对图像施加 HSV 扰动，模拟季节/环境变化

        Args:
            image_paths: 原始图像路径列表
            h_shift: 色调(H)偏移量，默认20度
            s_scale: 饱和度(S)缩放因子，默认0.75
            v_scale: 亮度(V)缩放因子，默认0.75
            output_dir: 扰动后图片保存目录，默认使用临时目录

        Returns:
            List[str]: 扰动后图片的路径列表
        """
        if output_dir is None:
            output_dir = os.path.join(BASE_DIR, "temp_perturbed")

        os.makedirs(output_dir, exist_ok=True)
        perturbed_paths = []

        logger.info(f"施加HSV扰动: H偏移={h_shift}, S缩放={s_scale}, V缩放={v_scale}")

        for img_path in image_paths:
            try:
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"无法读取图像: {img_path}")
                    continue

                # 转换到HSV色彩空间
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

                # 施加扰动
                hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180  # H通道偏移 (0-179)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)  # S通道缩放
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)  # V通道缩放

                # 转换回BGR
                result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # 保存扰动后的图像
                out_filename = f"perturbed_{os.path.basename(img_path)}"
                out_path = os.path.join(output_dir, out_filename)
                cv2.imwrite(out_path, result)
                perturbed_paths.append(out_path)

            except Exception as e:
                logger.error(f"处理图像失败 {img_path}: {e}")
                # 如果处理失败，使用原图
                perturbed_paths.append(img_path)

        logger.info(f"HSV扰动完成: {len(perturbed_paths)}/{len(image_paths)} 张图片")
        return perturbed_paths

    def extract_window_features(self, image_paths: List[str], labels: List[int]) -> np.ndarray:
        """
        为一个窗口的图片提取YOLO11特征

        使用 baseline_collector 中类似的逻辑：
        - 加载模型，注册钩子
        - 批量推理提取特征
        - 使用已有的 PCA+Scaler 转换

        Args:
            image_paths: 图片路径列表
            labels: 对应的标签列表

        Returns:
            np.ndarray: 特征数组 (n_samples, 128)
        """
        if self.feature_model is None:
            self._init_feature_extractor()

        logger.info(f"提取特征: {len(image_paths)} 张图片")

        all_records = []

        try:
            # 批量推理提取特征
            results = self.feature_model.predict(
                source=image_paths,
                batch=1,
                imgsz=224,
                stream=True,
                device=self.device,
                verbose=False
            )

            for res in results:
                if not self._current_batch_features:
                    logger.warning(f"未抓取到特征: {res.path}")
                    continue

                img_emb = self._current_batch_features.pop(0)
                record = {
                    "img_name": Path(res.path).name,
                    "image_embedding": img_emb.astype(np.float32),
                    "label": "unknown",
                    "conf": 0.0
                }
                if hasattr(res, 'probs') and res.probs is not None:
                    cls_id = int(res.probs.top1)
                    record["label"] = self.feature_model.names.get(cls_id, f"class_{cls_id}")
                    record["conf"] = float(res.probs.top1conf)
                all_records.append(record)

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise

        if not all_records:
            logger.warning("没有提取到任何特征")
            return np.array([])

        # 转换为DataFrame并应用PCA+Scaler
        df = pd.DataFrame(all_records)
        X = np.stack(df['image_embedding'].values)

        if self.pca is not None and self.scaler is not None:
            X_pca = self.pca.transform(self.scaler.transform(X))
            logger.info(f"特征提取完成: {X_pca.shape}")
            return X_pca.astype(np.float32)
        else:
            logger.warning("未加载PCA和Scaler，返回原始特征")
            return X.astype(np.float32)

    def detect_drift_for_window(self, window_features: np.ndarray,
                                  window_labels: List[int],
                                  window_id: int) -> Dict:
        """
        对单个窗口执行漂移检测

        使用 baseline_assets/baseline_db.pkl 作为基准

        Args:
            window_features: 窗口特征数组 (n_samples, n_features)
            window_labels: 窗口标签列表
            window_id: 窗口ID

        Returns:
            Dict: 漂移检测结果
        """
        logger.info(f"执行窗口 {window_id + 1} 的漂移检测...")

        try:
            # 获取基准数据
            baseline_X = self.drift_detector.baseline_X
            baseline_labels = self.drift_detector.baseline_labels

            # 随机采样窗口大小的数据
            window_size = min(len(baseline_X), len(window_features))
            X_sub = baseline_X[np.random.choice(len(baseline_X), window_size, replace=False)]
            Y_sub = window_features[np.random.choice(len(window_features), window_size, replace=False)]

            # 计算MMD和p-value
            gamma = estimate_gamma(X_sub, Y_sub)
            mmd_score = compute_mmd(X_sub, Y_sub, gamma)

            # 简单的排列检验（减少迭代次数以提高速度）
            iterations = 50
            combined = np.vstack([X_sub, Y_sub])
            n = X_sub.shape[0]
            count = 0
            for _ in range(iterations):
                idx = np.random.permutation(len(combined))
                new_X = combined[idx[:n]]
                new_Y = combined[idx[n:]]
                if compute_mmd(new_X, new_Y, gamma) >= mmd_score:
                    count += 1
            p_value = count / iterations

            is_drift = p_value < 0.05

            # 按类别漂移检测
            per_class = {}
            classes = np.unique(baseline_labels)
            for cls in classes:
                base_cls_X = baseline_X[baseline_labels == cls]
                # 将窗口标签转换为numpy数组进行比较
                window_labels_arr = np.array(window_labels)
                cur_cls_X = window_features[window_labels_arr == int(cls)]
                if len(cur_cls_X) < 3:
                    continue
                mmd_cls = compute_mmd(base_cls_X, cur_cls_X)
                per_class[str(cls)] = {
                    "mmd": float(mmd_cls),
                    "is_drift": mmd_cls > 0.01
                }

            # 特征维度漂移
            feature_level = feature_level_tests(baseline_X, window_features, alpha=0.05)

            result = {
                "window_id": window_id,
                "mmd_score": float(mmd_score),
                "p_value": float(p_value),
                "is_drift": bool(is_drift),
                "status": "DRIFT DETECTED" if is_drift else "DATA STABLE",
                "per_class": per_class,
                "feature_level": {
                    "changed_dims": feature_level['changed_dims'],
                    "n_changed_dims": len(feature_level['changed_dims'])
                }
            }

            status_text = "DRIFT" if is_drift else "STABLE"
            logger.info(f"[{status_text}] 窗口 {window_id + 1} 检测完成: MMD={mmd_score:.4f}, p-value={p_value:.4f}, 状态={result['status']}")

            return result

        except Exception as e:
            logger.error(f"漂移检测失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "window_id": window_id,
                "error": str(e),
                "is_drift": False
            }

    def grade_window_images(self, image_paths: List[str]) -> Dict:
        """
        对窗口内图片进行颜色分级

        Args:
            image_paths: 图片路径列表

        Returns:
            Dict: 分级统计结果
        """
        logger.info(f"对 {len(image_paths)} 张图片进行颜色分级...")

        grade_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        total_confidence = 0
        successful = 0

        for img_path in image_paths:
            try:
                report = self.color_grader.get_color_report(img_path)
                grade = report['grade_result']['grade']
                confidence = report['grade_result']['confidence']
                grade_counts[grade] += 1
                total_confidence += confidence
                successful += 1
            except Exception as e:
                logger.warning(f"分级失败 {img_path}: {e}")

        total = successful if successful > 0 else 1

        result = {
            "grade_distribution": {
                "grade_1": {"count": grade_counts[1], "percentage": round(grade_counts[1] / total * 100, 2)},
                "grade_2": {"count": grade_counts[2], "percentage": round(grade_counts[2] / total * 100, 2)},
                "grade_3": {"count": grade_counts[3], "percentage": round(grade_counts[3] / total * 100, 2)},
                "grade_4": {"count": grade_counts[4], "percentage": round(grade_counts[4] / total * 100, 2)},
            },
            "average_confidence": round(total_confidence / total, 4) if total > 0 else 0,
            "total_processed": successful
        }

        logger.info(f"分级完成: 等级1={grade_counts[1]}, 等级2={grade_counts[2]}, 等级3={grade_counts[3]}, 等级4={grade_counts[4]}")
        return result

    def run_pipeline(self, n_windows: int = 4,
                     perturbation_window: int = 2,
                     window_size: int = 40,
                     drift_threshold: float = 0.05) -> Dict:
        """
        执行完整的动态检测流程

        按窗口顺序处理，记录每个窗口的结果
        在漂移严重时触发增量训练

        Args:
            n_windows: 窗口数量
            perturbation_window: 施加扰动的窗口索引（0-indexed）
            window_size: 每个窗口的样本数
            drift_threshold: 触发训练的MMD阈值

        Returns:
            Dict: 流水线结果
            {
                'windows': [
                    {
                        'window_id': int,
                        'n_samples': int,
                        'has_perturbation': bool,
                        'mmd_score': float,
                        'is_drift': bool,
                        'p_value': float,
                        'per_class_drift': dict,
                        'grade_distribution': dict,
                        'triggered_training': bool,
                        'timestamp': str
                    }, ...
                ],
                'training_result': dict or None,
                'summary': str
            }
        """
        logger.info("=" * 60)
        logger.info("启动动态数据检测流水线")
        logger.info("=" * 60)

        # 步骤1: 准备时间窗口
        self.prepare_windows(n_windows, perturbation_window, window_size)

        if not self.windows:
            logger.error("没有可用的窗口数据")
            return {"error": "No windows available"}

        self.window_results = []
        self.has_triggered_training = False
        self.training_result = None

        # 临时目录用于保存扰动图片
        temp_perturb_dir = os.path.join(BASE_DIR, "temp_perturbed")

        # 在增量训练前，评估基础模型在基础验证集上的准确率
        baseline_accuracy = None
        try:
            logger.info("=" * 60)
            logger.info("评估基础模型在基础验证集上的准确率...")
            logger.info("=" * 60)
            baseline_eval = self.auto_trainer.evaluate_model(
                model_path=self.base_model_path,
                val_dir=self.val_dir
            )
            baseline_accuracy = baseline_eval.get('top1_accuracy', 0)
            logger.info(f"基础模型在基础验证集 Top-1 准确率: {baseline_accuracy * 100:.2f}%")
        except Exception as e:
            logger.error(f"评估基础模型失败: {e}")
            baseline_accuracy = None

        # 处理每个窗口
        for i, window in enumerate(self.windows):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"处理窗口 {i + 1}/{len(self.windows)}")
            logger.info("=" * 60)

            window_id = window['window_id']
            image_paths = window['images']
            labels = window['labels']
            has_perturbation = window.get('has_perturbation', False)

            try:
                # 如果是扰动窗口，施加HSV扰动
                if has_perturbation:
                    logger.info(f"窗口 {i + 1} 是扰动窗口，施加HSV扰动...")
                    image_paths = self.apply_hsv_perturbation(
                        image_paths,
                        h_shift=random.randint(5, 10),  # 降低色调偏移：从15-25降到5-10
                        s_scale=random.uniform(0.92, 0.95),  # 降低饱和度扰动：从0.7-0.8提升到0.92-0.95
                        v_scale=random.uniform(0.92, 0.95),  # 降低亮度扰动：从0.7-0.8提升到0.92-0.95
                        output_dir=temp_perturb_dir
                    )

                # 步骤2: 特征提取
                logger.info(f"窗口 {i + 1}: 提取特征...")
                window_features = self.extract_window_features(image_paths, labels)

                if len(window_features) == 0:
                    logger.warning(f"窗口 {i + 1} 没有提取到特征，跳过")
                    continue

                # 步骤3: 漂移检测
                logger.info(f"窗口 {i + 1}: 执行漂移检测...")
                drift_result = self.detect_drift_for_window(window_features, labels, window_id)

                # 步骤4: 颜色分级
                logger.info(f"窗口 {i + 1}: 执行颜色分级...")
                grade_result = self.grade_window_images(image_paths)

                # 判断是否触发训练（仅在前面的窗口，且未触发过训练）
                triggered_training = False
                if (not self.has_triggered_training and
                        drift_result.get('is_drift', False) and
                        drift_result.get('mmd_score', 0) >= drift_threshold and
                        has_perturbation):

                    logger.info("=" * 60)
                    logger.info(f"窗口 {i + 1} 检测到显著漂移，触发增量训练！")
                    logger.info(f"MMD={drift_result['mmd_score']:.4f} >= 阈值={drift_threshold}")
                    logger.info("=" * 60)

                    try:
                        # 准备增量训练数据
                        # 将当前窗口数据保存为临时数据集
                        temp_data_dir = os.path.join(BASE_DIR, "temp_incremental_data")
                        self._save_window_as_dataset(image_paths, labels, temp_data_dir)

                        # 执行增量训练
                        mixed_data_dir = self.auto_trainer.prepare_incremental_data(
                            new_sample_dir=temp_data_dir,
                            mix_ratio=0.6  #从旧数据中抽取总量的 60%
                        )

                        train_result = self.auto_trainer.incremental_train(
                            data_dir=mixed_data_dir,
                            epochs=80,  # 增量训练轮数（优化为80轮）
                            lr=0.001,
                            model_path=self.current_model_path  # 使用当前模型作为起始点
                        )

                        # 自适应融合策略：评估增量模型质量后决定是否融合及融合比例
                        new_model_path = train_result['model_path']

                        # 评估增量模型在基础验证集上的准确率
                        logger.info("评估增量模型在基础验证集上的准确率...")
                        new_model_base_eval = self.auto_trainer.evaluate_model(
                            model_path=new_model_path,
                            val_dir=self.val_dir
                        )
                        new_model_base_acc = new_model_base_eval.get('top1_accuracy', 0) * 100

                        # 获取基础模型在基础验证集上的准确率（已提前测量）
                        base_model_base_acc = baseline_accuracy * 100 if baseline_accuracy else 0

                        logger.info(f"自适应融合评估 - 增量模型基础验证集准确率: {new_model_base_acc:.2f}%, 基础模型: {base_model_base_acc:.2f}%")

                        # 决策：如果增量模型在基础验证集准确率 >= 基础模型准确率 - 1%，则跳过融合
                        if new_model_base_acc >= base_model_base_acc - 1.0:
                            # 增量模型质量足够高，直接使用
                            logger.info(f"增量模型基础性能良好(退化<1%), 跳过融合直接使用增量模型")
                            # 复制并重命名为fused命名格式以保持一致性
                            fused_model_path = new_model_path.replace('.pt', f'_fused_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
                            shutil.copy2(new_model_path, fused_model_path)
                            logger.info(f"增量模型已复制为融合模型格式: {fused_model_path}")
                        else:
                            # 增量模型有退化，使用alpha=0.7融合（偏向新模型）
                            logger.info(f"增量模型基础性能退化{base_model_base_acc - new_model_base_acc:.2f}%, 使用alpha=0.7融合")
                            fused_model_path = self.auto_trainer.model_fusion(
                                old_model_path=self.base_model_path,
                                new_model_path=new_model_path,
                                alpha=0.7  # 偏向新模型
                            )

                        # 更新当前模型
                        self.current_model_path = fused_model_path
                        self.has_triggered_training = True

                        # 增量训练后，评估多模型多数据集准确率
                        logger.info("=" * 60)
                        logger.info("增量训练完成，开始评估模型性能...")
                        logger.info("=" * 60)

                        accuracy_results = {
                            "baseline_model_on_base_dataset": None,
                            "baseline_model_on_mixed_dataset": None,
                            "fused_model_on_base_dataset": None,
                            "fused_model_on_mixed_dataset": None
                        }

                        # 1. 基础模型在基础验证集上的准确率（已测）
                        if baseline_accuracy is not None:
                            accuracy_results["baseline_model_on_base_dataset"] = baseline_accuracy

                        # 2. 基础模型在混合数据集上的准确率
                        try:
                            logger.info("评估基础模型在混合数据集上的准确率...")
                            mixed_val_dir = os.path.join(mixed_data_dir, "val")
                            if os.path.exists(mixed_val_dir):
                                baseline_on_mixed = self.auto_trainer.evaluate_model(
                                    model_path=self.base_model_path,
                                    val_dir=mixed_val_dir
                                )
                                accuracy_results["baseline_model_on_mixed_dataset"] = baseline_on_mixed.get('top1_accuracy', 0)
                                logger.info(f"基础模型在混合数据集 Top-1 准确率: {accuracy_results['baseline_model_on_mixed_dataset'] * 100:.2f}%")
                            else:
                                logger.warning(f"混合数据集验证目录不存在: {mixed_val_dir}")
                        except Exception as e:
                            logger.error(f"评估基础模型在混合数据集上失败: {e}")

                        # 3. 融合模型在基础验证集上的准确率（检测灾难性遗忘）
                        try:
                            logger.info("评估融合模型在基础验证集上的准确率...")
                            fused_on_base = self.auto_trainer.evaluate_model(
                                model_path=fused_model_path,
                                val_dir=self.val_dir
                            )
                            accuracy_results["fused_model_on_base_dataset"] = fused_on_base.get('top1_accuracy', 0)
                            logger.info(f"融合模型在基础验证集 Top-1 准确率: {accuracy_results['fused_model_on_base_dataset'] * 100:.2f}%")
                        except Exception as e:
                            logger.error(f"评估融合模型在基础验证集上失败: {e}")

                        # 4. 融合模型在混合数据集上的准确率
                        try:
                            logger.info("评估融合模型在混合数据集上的准确率...")
                            mixed_val_dir = os.path.join(mixed_data_dir, "val")
                            if os.path.exists(mixed_val_dir):
                                fused_on_mixed = self.auto_trainer.evaluate_model(
                                    model_path=fused_model_path,
                                    val_dir=mixed_val_dir
                                )
                                accuracy_results["fused_model_on_mixed_dataset"] = fused_on_mixed.get('top1_accuracy', 0)
                                logger.info(f"融合模型在混合数据集 Top-1 准确率: {accuracy_results['fused_model_on_mixed_dataset'] * 100:.2f}%")
                            else:
                                logger.warning(f"混合数据集验证目录不存在: {mixed_val_dir}")
                        except Exception as e:
                            logger.error(f"评估融合模型在混合数据集上失败: {e}")

                        self.training_result = {
                            "triggered_at_window": i,
                            "original_model": self.base_model_path,
                            "new_model": train_result['model_path'],
                            "fused_model": fused_model_path,
                            "metrics": train_result.get('metrics', {}),
                            "accuracy_results": accuracy_results
                        }

                        # 重新初始化特征提取器使用新模型
                        self._init_feature_extractor(fused_model_path)

                        triggered_training = True
                        logger.info(f"增量训练完成，新模型: {fused_model_path}")

                    except Exception as e:
                        logger.error(f"增量训练失败: {e}")
                        import traceback
                        traceback.print_exc()

                # 记录窗口结果
                window_result = {
                    "window_id": window_id,
                    "n_samples": len(image_paths),
                    "has_perturbation": has_perturbation,
                    "mmd_score": drift_result.get('mmd_score', 0),
                    "p_value": drift_result.get('p_value', 1),
                    "is_drift": drift_result.get('is_drift', False),
                    "status": drift_result.get('status', 'UNKNOWN'),
                    "per_class_drift": drift_result.get('per_class', {}),
                    "feature_changed_dims": drift_result.get('feature_level', {}).get('n_changed_dims', 0),
                    "grade_distribution": grade_result.get('grade_distribution', {}),
                    "average_confidence": grade_result.get('average_confidence', 0),
                    "triggered_training": triggered_training,
                    "timestamp": datetime.now().isoformat()
                }

                self.window_results.append(window_result)

                # 输出窗口摘要
                drift_text = "DRIFT" if window_result['is_drift'] else "STABLE"
                logger.info(f"[{drift_text}] 窗口 {i + 1} 完成: MMD={window_result['mmd_score']:.4f}, "
                            f"漂移={window_result['is_drift']}, 训练触发={triggered_training}")

            except Exception as e:
                logger.error(f"窗口 {i + 1} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                # 继续处理下一个窗口
                continue

        # 生成汇总
        summary = self._generate_summary()

        result = {
            "windows": self.window_results,
            "training_result": self.training_result,
            "summary": summary,
            "pipeline_config": {
                "n_windows": n_windows,
                "perturbation_window": perturbation_window,
                "window_size": window_size,
                "drift_threshold": drift_threshold
            }
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info("动态检测流水线执行完成")
        logger.info("=" * 60)
        logger.info(f"处理窗口数: {len(self.window_results)}")
        logger.info(f"触发训练: {self.has_triggered_training}")
        if self.has_triggered_training:
            logger.info(f"训练触发于窗口: {self.training_result.get('triggered_at_window', 'N/A')}")

        return result

    def _save_window_as_dataset(self, image_paths: List[str], labels: List[int], output_dir: str):
        """
        将窗口数据保存为数据集格式

        Args:
            image_paths: 图片路径列表
            labels: 标签列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        for img_path, label in zip(image_paths, labels):
            class_dir = os.path.join(output_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            dst_path = os.path.join(class_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dst_path)

        logger.info(f"窗口数据已保存到: {output_dir}")

    def _generate_summary(self) -> str:
        """生成文本摘要"""
        if not self.window_results:
            return "没有窗口结果"

        lines = []
        lines.append("=" * 60)
        lines.append("动态检测流水线执行摘要")
        lines.append("=" * 60)

        for result in self.window_results:
            window_id = result['window_id'] + 1
            mmd = result['mmd_score']
            is_drift = result['is_drift']
            has_perturbation = result['has_perturbation']
            triggered = result['triggered_training']

            status = "漂移" if is_drift else "稳定"
            perturbation_mark = " [HSV扰动]" if has_perturbation else ""
            training_mark = " [触发训练]" if triggered else ""

            lines.append(f"窗口 {window_id}{perturbation_mark}: MMD={mmd:.4f}, 状态={status}{training_mark}")

        if self.has_triggered_training and self.training_result:
            lines.append("")
            lines.append("增量训练信息:")
            lines.append(f"  触发窗口: {self.training_result['triggered_at_window'] + 1}")
            lines.append(f"  融合模型: {self.training_result['fused_model']}")

        return "\n".join(lines)

    def generate_summary_report(self, pipeline_result: Dict) -> str:
        """
        生成流水线汇总报告

        1. 每个窗口的漂移报告 JSON（保存到 reports/）
        2. 漂移趋势可视化（使用 DriftVisualizer）
        3. 增量训练前后对比（如果有训练）
        4. 文本摘要

        Args:
            pipeline_result: run_pipeline 返回的结果

        Returns:
            str: 报告目录路径
        """
        logger.info("=" * 60)
        logger.info("生成流水线汇总报告")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(REPORTS_DIR, f"pipeline_report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        windows = pipeline_result.get('windows', [])

        # 1. 保存每个窗口的详细报告
        for window_result in windows:
            window_id = window_result['window_id']
            window_report_path = os.path.join(report_dir, f"window_{window_id + 1}_report.json")
            with open(window_report_path, 'w', encoding='utf-8') as f:
                json.dump(window_result, f, ensure_ascii=False, indent=2)

        logger.info(f"已保存 {len(windows)} 个窗口的详细报告")

        # 2. 生成漂移趋势图
        if len(windows) > 0:
            mmd_history = [w.get('mmd_score', 0) for w in windows]
            window_labels = [f"窗口{w['window_id'] + 1}" for w in windows]

            trend_path = os.path.join(report_dir, "drift_trend.png")
            self.visualizer.output_dir = report_dir
            self.visualizer.plot_drift_trend(
                mmd_history=mmd_history,
                window_labels=window_labels,
                threshold=0.05,
                save_name="drift_trend.png"
            )
            logger.info(f"漂移趋势图已保存: {trend_path}")

        # 3. 生成汇总JSON
        summary_path = os.path.join(report_dir, "pipeline_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_result, f, ensure_ascii=False, indent=2)
        logger.info(f"汇总报告已保存: {summary_path}")

        # 4. 生成文本摘要
        summary_text_path = os.path.join(report_dir, "summary.txt")
        with open(summary_text_path, 'w', encoding='utf-8') as f:
            f.write(pipeline_result.get('summary', 'No summary available'))
        logger.info(f"文本摘要已保存: {summary_text_path}")

        # 5. 生成HTML可视化报告
        html_content = self._generate_html_report(pipeline_result, report_dir)
        html_path = os.path.join(report_dir, "pipeline_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML报告已保存: {html_path}")

        logger.info("=" * 60)
        logger.info(f"所有报告已保存到: {report_dir}")
        logger.info("=" * 60)

        return report_dir

    def _generate_html_report(self, pipeline_result: Dict, report_dir: str) -> str:
        """
        生成HTML格式的流水线报告

        Args:
            pipeline_result: 流水线结果
            report_dir: 报告目录

        Returns:
            str: HTML内容
        """
        windows = pipeline_result.get('windows', [])
        training_result = pipeline_result.get('training_result')

        # 构建窗口表格
        window_rows = ""
        for w in windows:
            window_id = w['window_id'] + 1
            has_perturbation = w.get('has_perturbation', False)
            mmd = w.get('mmd_score', 0)
            p_value = w.get('p_value', 1)
            is_drift = w.get('is_drift', False)
            triggered = w.get('triggered_training', False)

            perturbation_cell = "是" if has_perturbation else "否"
            drift_cell = f"{'漂移' if is_drift else '稳定'} ({mmd:.4f})"
            training_cell = "已触发" if triggered else "-"

            # 分级分布
            grade_dist = w.get('grade_distribution', {})
            grade_summary = f"G1:{grade_dist.get('grade_1', {}).get('count', 0)} "
            grade_summary += f"G2:{grade_dist.get('grade_2', {}).get('count', 0)} "
            grade_summary += f"G3:{grade_dist.get('grade_3', {}).get('count', 0)} "
            grade_summary += f"G4:{grade_dist.get('grade_4', {}).get('count', 0)}"

            window_rows += f"""
    <tr>
        <td>窗口 {window_id}</td>
        <td>{w.get('n_samples', 0)}</td>
        <td>{perturbation_cell}</td>
        <td>{drift_cell}</td>
        <td>{p_value:.4f}</td>
        <td>{grade_summary}</td>
        <td>{training_cell}</td>
    </tr>
"""

        # 训练信息和模型性能对比
        training_section = ""
        accuracy_section = ""

        if training_result:
            # 获取准确率结果
            accuracy_results = training_result.get('accuracy_results', {})

            # 格式化准确率显示
            def fmt_acc(acc):
                if acc is None:
                    return "未测量"
                return f"{acc * 100:.2f}%"

            baseline_on_base = fmt_acc(accuracy_results.get('baseline_model_on_base_dataset'))
            baseline_on_mixed = fmt_acc(accuracy_results.get('baseline_model_on_mixed_dataset'))
            fused_on_base = fmt_acc(accuracy_results.get('fused_model_on_base_dataset'))
            fused_on_mixed = fmt_acc(accuracy_results.get('fused_model_on_mixed_dataset'))

            training_section = f"""
    <div class="section">
        <h2>增量训练信息</h2>
        <div class="info-grid">
            <div class="info-item">
                <label>触发窗口</label>
                <value>窗口 {training_result.get('triggered_at_window', 0) + 1}</value>
            </div>
            <div class="info-item">
                <label>融合模型</label>
                <value>{os.path.basename(training_result.get('fused_model', 'N/A'))}</value>
            </div>
        </div>
    </div>
"""

            # 模型性能对比表格
            accuracy_section = f"""
    <div class="section">
        <h2>模型性能对比</h2>
        <p>双模型（基础模型 vs 增强学习模型）在双数据集（基础数据集 vs 混合数据集）上的 Top-1 准确率实测对比</p>
        <table>
            <thead>
                <tr>
                    <th>模型</th>
                    <th>基础数据集 Top-1</th>
                    <th>混合数据集 Top-1</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>基础模型</td>
                    <td>{baseline_on_base}</td>
                    <td>{baseline_on_mixed}</td>
                </tr>
                <tr>
                    <td>增强学习模型(融合)</td>
                    <td>{fused_on_base}</td>
                    <td>{fused_on_mixed}</td>
                </tr>
            </tbody>
        </table>
        <p style="margin-top: 15px; color: #666; font-size: 0.9em;">
            注：融合模型在基础验证集上的准确率用于检测灾难性遗忘。若该准确率显著下降，表明模型在适应新数据时遗忘了旧知识。
        </p>
    </div>
"""
        else:
            # 未触发训练的情况
            accuracy_section = f"""
    <div class="section">
        <h2>模型性能对比</h2>
        <p>本次流水线执行未触发增量训练，因此未进行双模型双数据集准确率对比。</p>
        <p style="color: #666; font-size: 0.9em;">
            当检测到显著数据漂移时，将自动触发增量训练并生成完整的模型性能对比报告。
        </p>
    </div>
"""

        # 检查是否有漂移趋势图
        trend_img = ""
        trend_path = os.path.join(report_dir, "drift_trend.png")
        if os.path.exists(trend_path):
            import base64
            with open(trend_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            trend_img = f'<img src="data:image/png;base64,{img_data}" alt="漂移趋势图">'

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>动态数据检测流水线报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #667eea;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .info-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .info-item label {{
            display: block;
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .info-item value {{
            display: block;
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>动态数据检测流水线报告</h1>
        <p>Dynamic Data Detection Pipeline Report</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>窗口检测结果汇总</h2>
        <table>
            <thead>
                <tr>
                    <th>窗口</th>
                    <th>样本数</th>
                    <th>HSV扰动</th>
                    <th>漂移状态</th>
                    <th>P-Value</th>
                    <th>分级分布</th>
                    <th>训练触发</th>
                </tr>
            </thead>
            <tbody>
                {window_rows}
            </tbody>
        </table>
    </div>

    {training_section}

    {accuracy_section}

    <div class="section">
        <h2>漂移趋势图</h2>
        <div class="chart-container">
            {trend_img if trend_img else '<p>未生成趋势图</p>'}
        </div>
    </div>

    <div class="footer">
        <p>由 DynamicDetectionPipeline 自动生成 | RoseGrade Project</p>
    </div>
</body>
</html>"""

        return html


# ------------------------------
# CLI 入口
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="动态数据检测流水线")
    parser.add_argument("--windows", type=int, default=4, help="窗口数量")
    parser.add_argument("--perturbation-window", type=int, default=2, help="施加扰动的窗口索引(0-indexed)")
    parser.add_argument("--window-size", type=int, default=40, help="每个窗口的样本数")
    parser.add_argument("--drift-threshold", type=float, default=0.05, help="触发训练的MMD阈值")
    parser.add_argument("--model", type=str, default=None, help="基础模型路径")
    parser.add_argument("--dataset", type=str, default=None, help="数据集目录")

    args = parser.parse_args()

    # 创建流水线实例
    pipeline = DynamicDetectionPipeline(
        base_model_path=args.model,
        dataset_dir=args.dataset
    )

    # 执行流水线
    result = pipeline.run_pipeline(
        n_windows=args.windows,
        perturbation_window=args.perturbation_window,
        window_size=args.window_size,
        drift_threshold=args.drift_threshold
    )

    # 生成报告
    report_dir = pipeline.generate_summary_report(result)

    print("\n" + "=" * 60)
    print("流水线执行完成！")
    print(f"报告目录: {report_dir}")
    print("=" * 60)
