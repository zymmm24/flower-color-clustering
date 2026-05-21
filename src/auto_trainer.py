"""
自适应增量训练器 - 检测数据漂移后自动触发模型更新
支持增量学习和模型融合
"""
import os
import json
import shutil
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
import random

from src.utils import get_logger, BASE_DIR, BASELINE_ASSETS_DIR, DATASET_DIR, MODELS_DIR, WEIGHTS_DIR

logger = get_logger(__name__)


class AutoTrainer:
    """
    自适应增量训练器
    检测数据漂移后自动触发模型更新，支持增量学习和模型融合
    """

    def __init__(self, base_model_path: Optional[str] = None, dataset_dir: Optional[str] = None):
        """
        初始化训练器

        Args:
            base_model_path: 基础模型路径 (默认 best.pt)
            dataset_dir: 数据集根目录
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"运行设备: {self.device}")

        # 设置基础模型路径
        if base_model_path is None:
            base_model_path = os.path.join(WEIGHTS_DIR, "best.pt")
        self.base_model_path = base_model_path

        # 设置数据集目录
        if dataset_dir is None:
            dataset_dir = DATASET_DIR
        self.dataset_dir = dataset_dir

        # 确保模型目录存在
        os.makedirs(MODELS_DIR, exist_ok=True)

        # 训练历史记录
        self.history_file = os.path.join(MODELS_DIR, "training_history.json")
        self.training_history = self._load_training_history()

        # 模型版本计数器
        self.model_version = self._get_next_version()

        logger.info(f"AutoTrainer 初始化完成")
        logger.info(f"   基础模型: {self.base_model_path}")
        logger.info(f"   数据集目录: {self.dataset_dir}")
        logger.info(f"   模型保存目录: {MODELS_DIR}")

    def _load_training_history(self) -> List[Dict]:
        """加载训练历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载训练历史失败: {e}")
        return []

    def _save_training_history(self):
        """保存训练历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")

    def _get_next_version(self) -> int:
        """获取下一个模型版本号"""
        if not self.training_history:
            return 1
        versions = [entry.get('version', 0) for entry in self.training_history]
        return max(versions) + 1

    def check_drift_trigger(self, drift_report_path: Optional[str] = None,
                            mmd_threshold: float = 0.05) -> Dict[str, Any]:
        """
        检查是否需要触发重训练

        读取最新的漂移报告JSON，根据MMD值和漂移状态判断

        Args:
            drift_report_path: 漂移报告路径 (默认项目根目录 drift_report.json)
            mmd_threshold: MMD阈值，超过此值认为需要重训练

        Returns:
            dict: {
                'should_retrain': bool,
                'severity': str,  # 'none', 'low', 'medium', 'high'
                'details': dict
            }
        """
        if drift_report_path is None:
            drift_report_path = os.path.join(BASE_DIR, "drift_report.json")

        if not os.path.exists(drift_report_path):
            logger.warning(f"漂移报告不存在: {drift_report_path}")
            return {
                'should_retrain': False,
                'severity': 'none',
                'details': {'error': 'Report not found'}
            }

        try:
            with open(drift_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)

            # 获取MMD分数和漂移状态
            mmd_score = report.get('statistics', {}).get('mmd_score', 0)
            is_drift = report.get('decision', {}).get('is_drift', False)

            # 判断严重程度
            if mmd_score >= 0.1:
                severity = 'high'
            elif mmd_score >= 0.05:
                severity = 'medium'
            elif mmd_score >= 0.01:
                severity = 'low'
            else:
                severity = 'none'

            # 判断是否需要重训练
            should_retrain = is_drift and mmd_score >= mmd_threshold

            # 获取按类别的漂移信息
            per_class_drift = report.get('per_class_drift', {})
            drifted_classes = [
                cls for cls, info in per_class_drift.items()
                if info.get('is_drift', False)
            ]

            result = {
                'should_retrain': should_retrain,
                'severity': severity,
                'details': {
                    'mmd_score': mmd_score,
                    'is_drift': is_drift,
                    'threshold': mmd_threshold,
                    'drifted_classes': drifted_classes,
                    'per_class_drift': per_class_drift,
                    'report_path': drift_report_path,
                    'generated_at': report.get('meta', {}).get('generated_at')
                }
            }

            if should_retrain:
                logger.info(f"检测到数据漂移，建议重训练")
                logger.info(f"   MMD分数: {mmd_score:.4f} (阈值: {mmd_threshold})")
                logger.info(f"   严重程度: {severity}")
                logger.info(f"   漂移类别: {drifted_classes}")
            else:
                logger.info(f"数据分布稳定，无需重训练")
                logger.info(f"   MMD分数: {mmd_score:.4f} (阈值: {mmd_threshold})")

            return result

        except Exception as e:
            logger.error(f"解析漂移报告失败: {e}")
            return {
                'should_retrain': False,
                'severity': 'error',
                'details': {'error': str(e)}
            }

    def prepare_incremental_data(self, new_sample_dir: str,
                                  old_dataset_dir: Optional[str] = None,
                                  mix_ratio: float = 0.3,
                                  output_dir: Optional[str] = None) -> str:
        """
        构建增量训练集

        将新到达的样本与旧数据的子集混合，防止灾难性遗忘

        Args:
            new_sample_dir: 新样本目录 (含类别子目录)
            old_dataset_dir: 旧训练数据目录
            mix_ratio: 旧数据混合比例 (0.3 = 30%旧数据)
            output_dir: 输出的混合数据集目录

        Returns:
            str: 混合数据集路径
        """
        if old_dataset_dir is None:
            old_dataset_dir = os.path.join(self.dataset_dir, "train")

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(BASE_DIR, "incremental_data", f"mixed_{timestamp}")

        logger.info(f"准备增量训练数据...")
        logger.info(f"   新样本目录: {new_sample_dir}")
        logger.info(f"   旧数据目录: {old_dataset_dir}")
        logger.info(f"   混合比例: {mix_ratio * 100:.0f}% 旧数据")
        logger.info(f"   输出目录: {output_dir}")

        # 创建输出目录结构
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # 支持的图片格式
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

        # 获取所有类别
        all_classes = set()

        # 处理新样本
        new_samples_by_class = {}
        if os.path.exists(new_sample_dir):
            for class_name in os.listdir(new_sample_dir):
                class_path = os.path.join(new_sample_dir, class_name)
                if not os.path.isdir(class_path):
                    continue

                all_classes.add(class_name)
                images = [
                    f for f in os.listdir(class_path)
                    if os.path.splitext(f.lower())[1] in img_extensions
                ]
                new_samples_by_class[class_name] = [
                    os.path.join(class_path, img) for img in images
                ]

        # 处理旧数据
        old_samples_by_class = {}
        if os.path.exists(old_dataset_dir):
            for class_name in os.listdir(old_dataset_dir):
                class_path = os.path.join(old_dataset_dir, class_name)
                if not os.path.isdir(class_path):
                    continue

                all_classes.add(class_name)
                images = [
                    f for f in os.listdir(class_path)
                    if os.path.splitext(f.lower())[1] in img_extensions
                ]
                old_samples_by_class[class_name] = [
                    os.path.join(class_path, img) for img in images
                ]

        # 混合数据
        total_new = 0
        total_old = 0

        for class_name in sorted(all_classes):
            class_train_dir = os.path.join(train_dir, class_name)
            class_val_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir, exist_ok=True)

            # 获取新样本
            new_samples = new_samples_by_class.get(class_name, [])

            # 从旧数据中采样
            old_samples = old_samples_by_class.get(class_name, [])
            num_old_to_sample = int(len(old_samples) * mix_ratio)
            if num_old_to_sample > 0 and old_samples:
                sampled_old = random.sample(old_samples, min(num_old_to_sample, len(old_samples)))
            else:
                sampled_old = []

            # 新数据过采样：如果新数据少于旧数据，用有放回抽样补齐
            # 确保新旧数据量平衡，防止模型偏向旧数据
            if len(new_samples) < len(sampled_old) and len(new_samples) > 0:
                oversampled_new = random.choices(new_samples, k=len(sampled_old))
                logger.info(f"   类别 {class_name}: 新数据从 {len(new_samples)} 张过采样至 {len(oversampled_new)} 张")
            else:
                oversampled_new = new_samples

            # 合并并打乱
            all_samples = oversampled_new + sampled_old
            random.shuffle(all_samples)

            # 划分训练集和验证集 (80/20)
            split_idx = int(len(all_samples) * 0.8)
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]

            # 复制文件
            for src_path in train_samples:
                dst_path = os.path.join(class_train_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)

            for src_path in val_samples:
                dst_path = os.path.join(class_val_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)

            total_new += len(new_samples)
            total_old += len(sampled_old)

            logger.info(f"   类别 {class_name}: {len(new_samples)} 新样本 + {len(sampled_old)} 旧样本")

        logger.info(f"增量数据集准备完成")
        logger.info(f"   总新样本: {total_new}, 总旧样本: {total_old}")
        logger.info(f"   输出路径: {output_dir}")

        return output_dir

    def incremental_train(self, data_dir: Optional[str] = None,
                          epochs: int = 10,
                          lr: float = 0.001,
                          imgsz: int = 224,
                          batch: int = 16,
                          model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        增量微调训练

        在现有模型基础上用新数据继续训练

        Args:
            data_dir: 训练数据目录 (含 train/ 和 val/ 子目录)
            epochs: 训练轮数 (增量训练一般 5-15轮)
            lr: 学习率 (增量训练用较小学习率)
            imgsz: 输入图片尺寸
            batch: 批次大小
            model_path: 用于增量训练的起始模型路径 (默认使用 base_model_path)

        Returns:
            dict: {'model_path': str, 'metrics': dict}
        """
        if data_dir is None:
            data_dir = self.dataset_dir

        # 使用指定的模型路径或默认基础模型
        train_start_model = model_path if model_path else self.base_model_path
        
        if not os.path.exists(train_start_model):
            raise FileNotFoundError(f"起始模型不存在: {train_start_model}")

        logger.info(f"开始增量训练...")
        logger.info(f"   起始模型: {train_start_model}")
        logger.info(f"   数据目录: {data_dir}")
        logger.info(f"   训练轮数: {epochs}")
        logger.info(f"   学习率: {lr}")
        logger.info(f"   图片尺寸: {imgsz}")
        logger.info(f"   批次大小: {batch}")

        try:
            # 加载起始模型，并记录训练前的权重用于验证
            model = YOLO(train_start_model)
            try:
                import torch
                pre_train_state = {k: v.clone() for k, v in model.model.state_dict().items()}
            except Exception as _e:
                pre_train_state = None
                logger.debug(f"无法记录训练前权重用于对比: {_e}")

            # 为本次训练生成唯一的输出目录名，避免 exist_ok 复用旧结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_run_name = f"train_incr_v{self.model_version}_{timestamp}"

            # 执行训练 - 使用唯一目录名，exist_ok=False 确保不复用旧目录
            results = model.train(
                data=data_dir,
                epochs=epochs,
                lr0=lr,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                project=os.path.join(BASE_DIR, "runs", "classify"),
                name=train_run_name,
                exist_ok=False,
                verbose=True
            )

            # 生成模型文件名
            model_filename = f"model_v{self.model_version}_{timestamp}.pt"
            model_path = os.path.join(MODELS_DIR, model_filename)

            # 优先从 model.trainer.best 获取本次训练产出的 best.pt 路径
            best_pt_path = None
            if hasattr(model, 'trainer') and model.trainer is not None:
                trainer_best = getattr(model.trainer, 'best', None)
                if trainer_best and os.path.exists(str(trainer_best)):
                    best_pt_path = str(trainer_best)
                    logger.info(f"从 model.trainer.best 获取训练结果: {best_pt_path}")

            # 备用方案：直接定位本次训练的输出目录
            if best_pt_path is None:
                expected_best = os.path.join(
                    BASE_DIR, "runs", "classify", train_run_name, "weights", "best.pt"
                )
                if os.path.exists(expected_best):
                    best_pt_path = expected_best
                    logger.info(f"从预期目录获取训练结果: {best_pt_path}")

            if best_pt_path:
                shutil.copy2(best_pt_path, model_path)
                logger.info(f"模型已保存: {model_path}")
            else:
                # 最终备用：直接保存当前模型对象
                model.save(model_path)
                logger.info(f"模型已保存 (直接导出): {model_path}")

            # 权重变化检测（诊断用）
            if pre_train_state is not None:
                try:
                    import torch
                    post_model = YOLO(model_path)
                    post_state = post_model.model.state_dict()
                    changed = sum(
                        1 for k in pre_train_state
                        if k in post_state and not torch.equal(pre_train_state[k].cpu(), post_state[k].cpu())
                    )
                    total = len(pre_train_state)
                    logger.info(f"权重变化检测: {changed}/{total} 个参数层发生变化")
                    if changed == 0:
                        logger.error(
                            "权重变化检测: 0 个参数层发生变化，训练可能未生效！"
                            f" 请检查训练目录 runs/classify/{train_run_name}"
                        )
                except Exception as _e:
                    logger.debug(f"权重变化检测失败: {_e}")

            # 提取训练指标
            metrics = {
                'epochs': epochs,
                'learning_rate': lr,
                'imgsz': imgsz,
                'batch': batch,
                'top1_accuracy': None,
                'top5_accuracy': None
            }

            # 尝试从 results 中提取指标
            if hasattr(results, 'results_dict'):
                metrics_dict = results.results_dict
                metrics['top1_accuracy'] = metrics_dict.get('metrics/accuracy_top1', None)
                metrics['top5_accuracy'] = metrics_dict.get('metrics/accuracy_top5', None)
            
            # 方法2: 从 metrics 对象中提取
            if metrics['top1_accuracy'] is None and hasattr(results, 'metrics'):
                if hasattr(results.metrics, 'top1'):
                    metrics['top1_accuracy'] = float(results.metrics.top1)
                if hasattr(results.metrics, 'top5'):
                    metrics['top5_accuracy'] = float(results.metrics.top5)
            
            # 方法3: 从本次训练的 results.csv 中读取最后一轮的准确率
            if metrics['top1_accuracy'] is None:
                try:
                    import csv
                    results_csv_path = os.path.join(
                        BASE_DIR, "runs", "classify", train_run_name, "results.csv"
                    )
                    if os.path.exists(results_csv_path):
                        with open(results_csv_path, 'r') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows:
                                last_row = rows[-1]
                                for key in last_row.keys():
                                    if 'accuracy_top1' in key or 'top1_acc' in key:
                                        metrics['top1_accuracy'] = float(last_row[key].strip())
                                    if 'accuracy_top5' in key or 'top5_acc' in key:
                                        metrics['top5_accuracy'] = float(last_row[key].strip())
                except Exception as e:
                    logger.debug(f"从 CSV 读取准确率失败: {e}")
            
            # 方法4: 如果训练时没有验证集，手动在验证集上评估
            if metrics['top1_accuracy'] is None:
                logger.info("在验证集上评估模型性能...")
                try:
                    val_data_dir = os.path.join(os.path.dirname(data_dir.rstrip('/\\')), 'val')
                    if not os.path.exists(val_data_dir):
                        val_data_dir = os.path.join(self.dataset_dir, 'val')
                    
                    if os.path.exists(val_data_dir):
                        val_results = model.val(data=data_dir, imgsz=imgsz, device=self.device)
                        if hasattr(val_results, 'top1'):
                            metrics['top1_accuracy'] = float(val_results.top1)
                        if hasattr(val_results, 'top5'):
                            metrics['top5_accuracy'] = float(val_results.top5)
                        logger.info(f"   Top-1 准确率: {metrics['top1_accuracy']:.4f}")
                    else:
                        logger.warning(f"   未找到验证集，无法评估准确率")
                except Exception as e:
                    logger.warning(f"   验证失败: {e}")

            # 记录训练历史
            history_entry = {
                'version': self.model_version,
                'timestamp': timestamp,
                'model_path': model_path,
                'base_model': self.base_model_path,
                'data_dir': data_dir,
                'metrics': metrics,
                'type': 'incremental'
            }
            self.training_history.append(history_entry)
            self._save_training_history()

            logger.info(f"增量训练完成")
            logger.info(f"   模型版本: v{self.model_version}")
            logger.info(f"   保存路径: {model_path}")

            return {
                'model_path': model_path,
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"增量训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def model_fusion(self, old_model_path: str, new_model_path: str,
                     alpha: float = 0.7) -> str:
        """
        EMA 模型权重融合

        fused = alpha * new_model + (1 - alpha) * old_model

        Args:
            old_model_path: 旧模型路径
            new_model_path: 新模型路径
            alpha: 新模型权重 (0.7 = 70%新模型 + 30%旧模型)

        Returns:
            str: 融合模型保存路径
        """
        logger.info(f"开始模型融合...")
        logger.info(f"   旧模型: {old_model_path}")
        logger.info(f"   新模型: {new_model_path}")
        logger.info(f"   融合系数: alpha={alpha} (新模型权重)")

        if not os.path.exists(old_model_path):
            raise FileNotFoundError(f"旧模型不存在: {old_model_path}")
        if not os.path.exists(new_model_path):
            raise FileNotFoundError(f"新模型不存在: {new_model_path}")

        try:
            # 加载模型
            old_model = YOLO(old_model_path)
            new_model = YOLO(new_model_path)

            # 获取 state_dict
            old_state = old_model.model.state_dict()
            new_state = new_model.model.state_dict()

            # 执行 EMA 融合
            fused_state = {}
            for key in new_state.keys():
                if key in old_state:
                    # 加权平均
                    fused_state[key] = alpha * new_state[key] + (1 - alpha) * old_state[key]
                else:
                    # 如果旧模型中没有该参数，直接使用新模型的
                    fused_state[key] = new_state[key]
                    logger.warning(f"参数 {key} 在旧模型中不存在，使用新模型值")

            # 加载融合后的权重到新模型
            new_model.model.load_state_dict(fused_state)

            # 保存融合模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fused_filename = f"model_v{self.model_version}_fused_{timestamp}.pt"
            fused_path = os.path.join(MODELS_DIR, fused_filename)

            new_model.save(fused_path)

            logger.info(f"模型融合完成")
            logger.info(f"   融合模型: {fused_path}")

            # 记录历史
            history_entry = {
                'version': self.model_version,
                'timestamp': timestamp,
                'model_path': fused_path,
                'old_model': old_model_path,
                'new_model': new_model_path,
                'alpha': alpha,
                'type': 'fused'
            }
            self.training_history.append(history_entry)
            self._save_training_history()

            return fused_path

        except Exception as e:
            logger.error(f"模型融合失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate_model(self, model_path: Optional[str] = None,
                       val_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        评估模型准确率

        在验证集上运行推理，计算 Top-1 和 Top-5 准确率

        Args:
            model_path: 模型路径 (默认使用基础模型)
            val_dir: 验证集目录

        Returns:
            dict: {
                'top1_accuracy': float,
                'top5_accuracy': float,
                'per_class': dict
            }
        """
        if model_path is None:
            model_path = self.base_model_path

        if val_dir is None:
            val_dir = os.path.join(self.dataset_dir, "val")

        logger.info(f"开始评估模型...")
        logger.info(f"   模型: {model_path}")
        logger.info(f"   验证集: {val_dir}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"验证集不存在: {val_dir}")

        try:
            # 加载模型
            model = YOLO(model_path)

            # 获取所有验证图片
            img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            val_images = []
            true_labels = []

            for class_name in sorted(os.listdir(val_dir)):
                class_path = os.path.join(val_dir, class_name)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    if os.path.splitext(img_name.lower())[1] in img_extensions:
                        val_images.append(os.path.join(class_path, img_name))
                        true_labels.append(class_name)

            if not val_images:
                logger.warning(f"验证集中未找到图片")
                return {
                    'top1_accuracy': 0.0,
                    'top5_accuracy': 0.0,
                    'per_class': {}
                }

            logger.info(f"   验证样本数: {len(val_images)}")

            # 运行推理
            results = model.predict(
                source=val_images,
                batch=16,
                imgsz=224,
                device=self.device,
                verbose=False
            )

            # 计算准确率
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            # 按类别统计
            per_class_stats = {}

            for i, result in enumerate(results):
                true_label = true_labels[i]

                if true_label not in per_class_stats:
                    per_class_stats[true_label] = {'correct': 0, 'total': 0}
                per_class_stats[true_label]['total'] += 1

                if hasattr(result, 'probs') and result.probs is not None:
                    # Top-1 预测
                    pred_top1 = result.probs.top1
                    pred_label = model.names[int(pred_top1)]

                    if str(pred_label) == str(true_label):
                        correct_top1 += 1
                        per_class_stats[true_label]['correct'] += 1

                    # Top-5 预测
                    top5_indices = result.probs.top5
                    top5_labels = [model.names[int(idx)] for idx in top5_indices]
                    if str(true_label) in [str(l) for l in top5_labels]:
                        correct_top5 += 1

                total += 1

            # 计算总体准确率
            top1_accuracy = correct_top1 / total if total > 0 else 0
            top5_accuracy = correct_top5 / total if total > 0 else 0

            # 计算每类准确率
            per_class_accuracy = {}
            for class_name, stats in per_class_stats.items():
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                per_class_accuracy[class_name] = {
                    'accuracy': acc,
                    'correct': stats['correct'],
                    'total': stats['total']
                }

            logger.info(f"评估完成")
            logger.info(f"   Top-1 准确率: {top1_accuracy * 100:.2f}%")
            logger.info(f"   Top-5 准确率: {top5_accuracy * 100:.2f}%")

            return {
                'top1_accuracy': top1_accuracy,
                'top5_accuracy': top5_accuracy,
                'per_class': per_class_accuracy
            }

        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def auto_update_loop(self, drift_report_path: Optional[str] = None,
                         new_data_dir: Optional[str] = None,
                         mmd_threshold: float = 0.05,
                         epochs: int = 10,
                         alpha: float = 0.7) -> Dict[str, Any]:
        """
        自动检测-训练-更新一次循环

        1. 检查漂移报告
        2. 如果需要重训练，准备增量数据
        3. 增量训练
        4. 模型融合
        5. 评估新模型
        6. 如果新模型更好，替换为当前模型

        Args:
            drift_report_path: 漂移报告路径
            new_data_dir: 新数据目录
            mmd_threshold: MMD阈值
            epochs: 训练轮数
            alpha: 模型融合系数

        Returns:
            dict: 包含整个流程的结果
        """
        logger.info("=" * 60)
        logger.info("启动自动更新循环")
        logger.info("=" * 60)

        result = {
            'success': False,
            'drift_check': None,
            'data_preparation': None,
            'training': None,
            'fusion': None,
            'evaluation': None,
            'model_updated': False
        }

        try:
            # Step 1: 检查漂移
            logger.info("\n[Step 1/6] 检查数据漂移...")
            drift_result = self.check_drift_trigger(drift_report_path, mmd_threshold)
            result['drift_check'] = drift_result

            if not drift_result['should_retrain']:
                logger.info("无需重训练，流程结束")
                result['success'] = True
                return result

            # Step 2: 准备增量数据
            logger.info("\n[Step 2/6] 准备增量数据...")
            if new_data_dir is None:
                logger.warning("未提供新数据目录，使用原始训练数据")
                mixed_data_dir = self.dataset_dir
            else:
                mixed_data_dir = self.prepare_incremental_data(
                    new_sample_dir=new_data_dir,
                    mix_ratio=0.5  # 新旧数据各占50%
                )
            result['data_preparation'] = {'mixed_data_dir': mixed_data_dir}

            # Step 3: 增量训练
            logger.info("\n[Step 3/6] 增量训练...")
            train_result = self.incremental_train(
                data_dir=mixed_data_dir,
                epochs=epochs
            )
            result['training'] = train_result
            new_model_path = train_result['model_path']

            # Step 4: 模型融合
            logger.info("\n[Step 4/6] 模型融合...")
            fused_model_path = self.model_fusion(
                old_model_path=self.base_model_path,
                new_model_path=new_model_path,
                alpha=alpha
            )
            result['fusion'] = {'fused_model_path': fused_model_path}

            # Step 5: 评估旧模型
            logger.info("\n[Step 5/6] 评估旧模型...")
            old_eval = self.evaluate_model(self.base_model_path)
            logger.info(f"   旧模型 Top-1: {old_eval['top1_accuracy'] * 100:.2f}%")

            # Step 6: 评估融合模型
            logger.info("\n[Step 6/6] 评估融合模型...")
            new_eval = self.evaluate_model(fused_model_path)
            logger.info(f"   融合模型 Top-1: {new_eval['top1_accuracy'] * 100:.2f}%")

            result['evaluation'] = {
                'old_model': old_eval,
                'new_model': new_eval
            }

            # 决定是否更新
            if new_eval['top1_accuracy'] >= old_eval['top1_accuracy']:
                logger.info("\n融合模型表现更好或持平，更新为基础模型")
                # 备份旧模型
                backup_path = self.base_model_path + ".backup"
                shutil.copy2(self.base_model_path, backup_path)
                logger.info(f"   旧模型已备份: {backup_path}")

                # 替换为基础模型
                shutil.copy2(fused_model_path, self.base_model_path)
                result['model_updated'] = True
                result['success'] = True

                logger.info(f"   模型已更新: {self.base_model_path}")
            else:
                logger.info("\n融合模型表现不如旧模型，保持原模型")
                result['model_updated'] = False
                result['success'] = True

            # 更新版本号
            self.model_version = self._get_next_version()

            logger.info("=" * 60)
            logger.info("自动更新循环完成")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"自动更新循环失败: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
            return result

    def get_training_history(self) -> List[Dict]:
        """获取历史训练记录"""
        return self.training_history


# ------------------------------
# CLI 入口
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自适应增量训练器")
    parser.add_argument("--mode", choices=['check', 'train', 'fuse', 'eval', 'auto'],
                        default='auto', help="运行模式")
    parser.add_argument("--model", type=str, default=None, help="基础模型路径")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--new-data", type=str, default=None, help="新数据目录")
    parser.add_argument("--report", type=str, default=None, help="漂移报告路径")
    parser.add_argument("--threshold", type=float, default=0.05, help="MMD阈值")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--alpha", type=float, default=0.7, help="融合系数")
    parser.add_argument("--old-model", type=str, default=None, help="旧模型路径(用于融合)")
    parser.add_argument("--new-model", type=str, default=None, help="新模型路径(用于融合)")

    args = parser.parse_args()

    # 初始化训练器
    trainer = AutoTrainer(base_model_path=args.model, dataset_dir=args.data)

    if args.mode == 'check':
        # 仅检查漂移
        result = trainer.check_drift_trigger(args.report, args.threshold)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == 'train':
        # 仅执行训练
        if args.new_data:
            mixed_dir = trainer.prepare_incremental_data(args.new_data)
        else:
            mixed_dir = args.data
        result = trainer.incremental_train(mixed_dir, epochs=args.epochs, lr=args.lr)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == 'fuse':
        # 仅执行模型融合
        if args.old_model and args.new_model:
            fused_path = trainer.model_fusion(args.old_model, args.new_model, args.alpha)
            print(f"融合模型已保存: {fused_path}")
        else:
            print("错误: 融合模式需要提供 --old-model 和 --new-model")

    elif args.mode == 'eval':
        # 仅执行评估
        model_path = args.model or trainer.base_model_path
        result = trainer.evaluate_model(model_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == 'auto':
        # 执行完整自动更新循环
        result = trainer.auto_update_loop(
            drift_report_path=args.report,
            new_data_dir=args.new_data,
            mmd_threshold=args.threshold,
            epochs=args.epochs,
            alpha=args.alpha
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
