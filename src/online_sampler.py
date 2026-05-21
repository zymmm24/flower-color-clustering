"""
动态在线采样器 - 模拟生产环境中的数据流入，支持多种触发方式
"""
import os
import glob
import random
import shutil
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from src.utils import get_logger, DATASET_DIR, SAMPLING_DIR

logger = get_logger(__name__)


class OnlineSampler:
    """
    动态在线采样器
    模拟生产环境中的数据流入，支持多种触发方式
    """

    def __init__(self, source_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        初始化采样器
        
        Args:
            source_dir: 数据源目录 (默认 dataset/val)
            output_dir: 采样输出目录 (默认 sampling_data/)
        """
        self.source_dir = source_dir or os.path.join(DATASET_DIR, "val")
        self.output_dir = output_dir or SAMPLING_DIR
        self.sampling_history: List[Dict] = []
        self.current_stream: List[Dict] = []
        self.current_window_id: int = 0

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"OnlineSampler 初始化完成: source={self.source_dir}, output={self.output_dir}")

    def scan_source(self) -> Tuple[List[str], List[int]]:
        """
        扫描数据源，返回所有可用图片路径和标签
        
        Returns:
            Tuple[List[str], List[int]]: (图片路径列表, 对应标签列表)
        """
        image_paths = []
        labels = []

        # 支持的图片格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']

        # 扫描每个类别目录
        for class_id in range(1, 5):  # 类别 1-4
            class_dir = os.path.join(self.source_dir, str(class_id))
            if not os.path.exists(class_dir):
                logger.warning(f"类别目录不存在: {class_dir}")
                continue

            for ext in extensions:
                pattern = os.path.join(class_dir, ext)
                files = glob.glob(pattern)
                for f in files:
                    image_paths.append(f)
                    labels.append(class_id)

        logger.info(f"扫描完成: 共发现 {len(image_paths)} 张图片")
        return image_paths, labels

    def time_trigger_sample(
        self,
        interval_seconds: int = 30,
        batch_size: int = 10,
        max_batches: int = 5
    ) -> List[Dict]:
        """
        时间触发采样 - 按固定时间间隔从数据源采样
        
        Args:
            interval_seconds: 采样间隔(秒)
            batch_size: 每批采样数量
            max_batches: 最大采样批次
            
        Returns:
            List[Dict]: 采样结果记录列表
        """
        logger.info(f"开始时间触发采样: 间隔={interval_seconds}s, 批次={max_batches}, 每批={batch_size}")

        image_paths, labels = self.scan_source()
        if not image_paths:
            logger.warning("数据源为空，无法采样")
            return []

        results = []

        for batch_idx in range(max_batches):
            # 随机采样一批数据
            if len(image_paths) < batch_size:
                selected_indices = list(range(len(image_paths)))
            else:
                selected_indices = random.sample(range(len(image_paths)), batch_size)

            batch_samples = []
            for idx in selected_indices:
                img_path = image_paths[idx]
                label = labels[idx]
                quality = self._assess_image_quality(img_path)

                sample = {
                    'timestamp': datetime.now().isoformat(),
                    'image_path': img_path,
                    'label': label,
                    'quality_score': quality,
                    'trigger_type': 'time',
                    'batch_id': batch_idx,
                    'source': 'time_trigger'
                }
                batch_samples.append(sample)

            results.extend(batch_samples)
            self.sampling_history.extend(batch_samples)

            logger.info(f"批次 {batch_idx + 1}/{max_batches} 采样完成: {len(batch_samples)} 张图片")

            # 如果不是最后一批，等待指定间隔
            if batch_idx < max_batches - 1:
                time.sleep(interval_seconds)

        logger.info(f"时间触发采样完成: 共 {len(results)} 个样本")
        return results

    def quality_trigger_sample(
        self,
        quality_threshold: float = 0.5,
        batch_size: int = 10
    ) -> List[Dict]:
        """
        质量触发采样 - 基于图像质量评分筛选
        使用拉普拉斯算子方差评估清晰度
        
        Args:
            quality_threshold: 质量阈值 (0-1)
            batch_size: 采样数量
            
        Returns:
            List[Dict]: 通过质量筛选的样本
        """
        logger.info(f"开始质量触发采样: 阈值={quality_threshold}, 目标数量={batch_size}")

        image_paths, labels = self.scan_source()
        if not image_paths:
            logger.warning("数据源为空，无法采样")
            return []

        # 评估所有图片质量
        quality_scores = []
        for img_path in image_paths:
            score = self._assess_image_quality(img_path)
            quality_scores.append(score)

        # 筛选高质量图片
        qualified_samples = []
        for idx, (img_path, label, score) in enumerate(zip(image_paths, labels, quality_scores)):
            if score >= quality_threshold:
                qualified_samples.append({
                    'idx': idx,
                    'image_path': img_path,
                    'label': label,
                    'quality_score': score
                })

        # 按质量分数排序，选择最好的
        qualified_samples.sort(key=lambda x: x['quality_score'], reverse=True)
        selected = qualified_samples[:batch_size]

        results = []
        for item in selected:
            sample = {
                'timestamp': datetime.now().isoformat(),
                'image_path': item['image_path'],
                'label': item['label'],
                'quality_score': item['quality_score'],
                'trigger_type': 'quality',
                'source': 'quality_trigger'
            }
            results.append(sample)

        self.sampling_history.extend(results)

        logger.info(f"质量触发采样完成: 筛选 {len(qualified_samples)} 张合格图片, "
                    f"选择前 {len(results)} 张高质量图片")
        return results

    def _assess_image_quality(self, image_path: str) -> float:
        """
        评估图像质量
        
        基于:
        - 清晰度: 拉普拉斯算子方差
        - 亮度适中性: 平均亮度接近128的程度
        - 对比度: 标准差
        
        Args:
            image_path: 图片路径
            
        Returns:
            float: 质量评分 (0-1)
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"无法读取图片: {image_path}")
                return 0.0

            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. 清晰度 - 拉普拉斯算子方差
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # 归一化到 0-1 (假设正常范围 0-1000)
            sharpness = min(laplacian_var / 500.0, 1.0)

            # 2. 亮度适中性 - 越接近128越好
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0

            # 3. 对比度 - 标准差
            contrast = np.std(gray)
            contrast_score = min(contrast / 64.0, 1.0)  # 假设正常对比度在64左右

            # 综合评分 (加权平均)
            quality = sharpness * 0.5 + brightness_score * 0.25 + contrast_score * 0.25
            quality = max(0.0, min(1.0, quality))  # 限制在 0-1

            return float(quality)

        except Exception as e:
            logger.error(f"评估图片质量时出错 {image_path}: {e}")
            return 0.0

    def manual_trigger_sample(self, image_paths: List[str]) -> List[Dict]:
        """
        手动触发采样 - 指定具体图片路径
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            List[Dict]: 采样结果
        """
        logger.info(f"开始手动触发采样: {len(image_paths)} 张图片")

        results = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.warning(f"图片不存在: {img_path}")
                continue

            # 从路径推断标签
            try:
                # 假设路径格式: .../class_id/image_name.jpg
                parent_dir = os.path.basename(os.path.dirname(img_path))
                label = int(parent_dir)
            except (ValueError, IndexError):
                label = -1  # 未知标签

            quality = self._assess_image_quality(img_path)

            sample = {
                'timestamp': datetime.now().isoformat(),
                'image_path': img_path,
                'label': label,
                'quality_score': quality,
                'trigger_type': 'manual',
                'source': 'manual_trigger'
            }
            results.append(sample)

        self.sampling_history.extend(results)

        logger.info(f"手动触发采样完成: {len(results)} 个样本")
        return results

    def simulate_stream(
        self,
        dataset_dir: Optional[str] = None,
        window_size: int = 40,
        n_windows: int = 4,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        核心方法: 模拟流式数据到达
        
        将数据集按窗口滑动分批，模拟不同时间段的数据到达
        
        Args:
            dataset_dir: 数据目录 (默认用 source_dir)
            window_size: 每个窗口的样本数
            n_windows: 窗口数量
            shuffle: 是否随机打乱
            
        Returns:
            List[Dict]: 每个窗口的信息
                [
                    {
                        'window_id': 0,
                        'images': [...],
                        'labels': [...],
                        'timestamp': ...,
                        'quality_scores': [...]
                    },
                    ...
                ]
        """
        dataset_dir = dataset_dir or self.source_dir
        logger.info(f"开始模拟流式数据: dir={dataset_dir}, window_size={window_size}, n_windows={n_windows}")

        # 扫描所有图片
        image_paths, labels = self.scan_source()
        if not image_paths:
            logger.warning("数据源为空，无法模拟流")
            return []

        # 打乱数据
        if shuffle:
            combined = list(zip(image_paths, labels))
            random.shuffle(combined)
            image_paths, labels = zip(*combined) if combined else ([], [])
            image_paths, labels = list(image_paths), list(labels)

        # 创建滑动窗口
        windows = []
        total_samples = len(image_paths)

        for i in range(n_windows):
            # 计算窗口起始位置
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, total_samples)

            if start_idx >= total_samples:
                logger.warning(f"窗口 {i} 超出数据范围，停止创建")
                break

            window_images = image_paths[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]

            # 评估每个图片的质量
            quality_scores = [self._assess_image_quality(img) for img in window_images]

            window_info = {
                'window_id': i,
                'images': window_images,
                'labels': window_labels,
                'timestamp': datetime.now().isoformat(),
                'quality_scores': quality_scores,
                'size': len(window_images)
            }
            windows.append(window_info)

            logger.info(f"窗口 {i} 创建完成: {len(window_images)} 张图片")

        self.current_stream = windows
        self.current_window_id = 0

        logger.info(f"流式数据模拟完成: 共 {len(windows)} 个窗口")
        return windows

    def get_sample_batch(self, window_id: Optional[int] = None) -> Optional[Dict]:
        """
        获取指定窗口的采样批次数据
        
        Args:
            window_id: 窗口ID (默认返回当前窗口)
            
        Returns:
            Optional[Dict]: 窗口信息，如果不存在则返回 None
        """
        if not self.current_stream:
            logger.warning("当前没有流数据，请先调用 simulate_stream()")
            return None

        if window_id is None:
            window_id = self.current_window_id

        if window_id < 0 or window_id >= len(self.current_stream):
            logger.warning(f"窗口ID {window_id} 超出范围 (0-{len(self.current_stream)-1})")
            return None

        self.current_window_id = window_id
        window = self.current_stream[window_id]

        logger.info(f"获取窗口 {window_id}: {window['size']} 张图片")
        return window

    def save_samples(self, samples: List[Dict], batch_name: str) -> str:
        """
        保存采样结果到 sampling_data/ 目录
        
        按类别子目录组织: sampling_data/{batch_name}/{class_id}/
        
        Args:
            samples: 采样样本列表
            batch_name: 批次名称
            
        Returns:
            str: 保存的目录路径
        """
        batch_dir = os.path.join(self.output_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        saved_count = 0
        for sample in samples:
            img_path = sample['image_path']
            label = sample.get('label', -1)

            # 创建类别子目录
            class_dir = os.path.join(batch_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            # 复制图片
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(class_dir, img_name)

            try:
                shutil.copy2(img_path, dest_path)
                saved_count += 1
            except Exception as e:
                logger.error(f"复制图片失败 {img_path} -> {dest_path}: {e}")

        # 保存元数据
        metadata = {
            'batch_name': batch_name,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(samples),
            'saved_count': saved_count,
            'samples': samples
        }
        metadata_path = os.path.join(batch_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"采样结果已保存: {batch_dir}, 共 {saved_count} 张图片")
        return batch_dir

    def get_sampling_history(self) -> List[Dict]:
        """
        获取历史采样记录
        
        Returns:
            List[Dict]: 采样历史记录列表
        """
        return self.sampling_history.copy()

    def clear_history(self) -> None:
        """清空采样历史"""
        self.sampling_history = []
        logger.info("采样历史已清空")

    def get_statistics(self) -> Dict:
        """
        获取采样统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.sampling_history:
            return {'total_samples': 0}

        trigger_types = {}
        label_distribution = {}
        quality_scores = []

        for sample in self.sampling_history:
            # 触发类型统计
            trigger = sample.get('trigger_type', 'unknown')
            trigger_types[trigger] = trigger_types.get(trigger, 0) + 1

            # 标签分布
            label = sample.get('label', -1)
            label_distribution[label] = label_distribution.get(label, 0) + 1

            # 质量分数
            quality = sample.get('quality_score', 0)
            quality_scores.append(quality)

        stats = {
            'total_samples': len(self.sampling_history),
            'trigger_type_distribution': trigger_types,
            'label_distribution': label_distribution,
            'quality_statistics': {
                'mean': float(np.mean(quality_scores)) if quality_scores else 0,
                'std': float(np.std(quality_scores)) if quality_scores else 0,
                'min': float(np.min(quality_scores)) if quality_scores else 0,
                'max': float(np.max(quality_scores)) if quality_scores else 0
            }
        }

        return stats


if __name__ == "__main__":
    # 简单测试
    sampler = OnlineSampler()

    # 测试扫描数据源
    paths, labels = sampler.scan_source()
    print(f"扫描到 {len(paths)} 张图片")

    # 测试流式模拟
    windows = sampler.simulate_stream(window_size=20, n_windows=4)
    print(f"创建了 {len(windows)} 个窗口")

    # 测试获取批次
    batch = sampler.get_sample_batch(0)
    if batch:
        print(f"窗口 0 有 {batch['size']} 张图片")

    # 测试质量评估
    if paths:
        quality = sampler._assess_image_quality(paths[0])
        print(f"第一张图片质量评分: {quality:.4f}")

    # 测试质量触发采样
    quality_samples = sampler.quality_trigger_sample(quality_threshold=0.3, batch_size=5)
    print(f"质量采样获得 {len(quality_samples)} 个样本")

    # 测试保存
    if quality_samples:
        save_dir = sampler.save_samples(quality_samples, "test_batch")
        print(f"已保存到: {save_dir}")

    # 测试统计
    stats = sampler.get_statistics()
    print(f"采样统计: {stats}")
