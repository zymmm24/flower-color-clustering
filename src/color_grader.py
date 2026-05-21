"""
基于HSV色彩空间的花卉颜色分级模块

该模块提供HSV特征提取和基于颜色特征的花卉分级功能，
支持季节性参数调整以适应不同季节的花卉特点。
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from src.utils import get_logger, DATASET_DIR

logger = get_logger(__name__)


class ColorGrader:
    """
    基于HSV色彩空间的花卉颜色分级器
    
    通过提取图像的HSV特征（色调、饱和度、亮度），
    结合季节性阈值参数，对花卉进行质量分级。
    
    Attributes:
        season: 当前季节参数，影响分级阈值
        SEASONAL_THRESHOLDS: 季节参数表，不同季节对应不同的HSV阈值
    """
    
    # 季节参数表 - 不同季节对应不同HSV阈值
    # 阈值设计思路：
    # - 春季：花卉色彩鲜艳，饱和度和亮度都较高
    # - 夏季：偏暖色调，亮度较高但饱和度略低
    # - 秋季：偏暗，整体饱和度和亮度都较低
    # - 冬季：偏冷色调，亮度较低
    SEASONAL_THRESHOLDS = {
        'spring': {
            # 春季：色彩鲜艳，饱和度和亮度要求高
            'saturation_high': 0.65,      # 高饱和度阈值
            'saturation_medium': 0.40,    # 中等饱和度阈值
            'saturation_low': 0.20,       # 低饱和度阈值
            'brightness_high': 0.80,      # 高亮度阈值
            'brightness_medium': 0.55,    # 中等亮度阈值
            'brightness_low': 0.30,       # 低亮度阈值
            'hue_std_threshold': 25,      # 色调标准差阈值（越低表示色调越集中）
            'saturation_weight': 0.45,    # 饱和度评分权重
            'brightness_weight': 0.30,    # 亮度评分权重
            'hue_weight': 0.25,           # 色调评分权重
        },
        'summer': {
            # 夏季：偏暖色调，亮度高但允许饱和度稍低
            'saturation_high': 0.60,
            'saturation_medium': 0.35,
            'saturation_low': 0.18,
            'brightness_high': 0.85,
            'brightness_medium': 0.60,
            'brightness_low': 0.35,
            'hue_std_threshold': 28,
            'saturation_weight': 0.40,
            'brightness_weight': 0.35,
            'hue_weight': 0.25,
        },
        'autumn': {
            # 秋季：偏暗，阈值相对宽松
            'saturation_high': 0.55,
            'saturation_medium': 0.32,
            'saturation_low': 0.15,
            'brightness_high': 0.75,
            'brightness_medium': 0.50,
            'brightness_low': 0.28,
            'hue_std_threshold': 30,
            'saturation_weight': 0.40,
            'brightness_weight': 0.30,
            'hue_weight': 0.30,
        },
        'winter': {
            # 冬季：偏冷色调，亮度低，阈值最宽松
            'saturation_high': 0.50,
            'saturation_medium': 0.30,
            'saturation_low': 0.15,
            'brightness_high': 0.70,
            'brightness_medium': 0.45,
            'brightness_low': 0.25,
            'hue_std_threshold': 32,
            'saturation_weight': 0.35,
            'brightness_weight': 0.25,
            'hue_weight': 0.40,
        },
    }
    
    def __init__(self, season: str = 'spring'):
        """
        初始化颜色分级器
        
        Args:
            season: 季节参数，可选 'spring', 'summer', 'autumn', 'winter'
                    默认为 'spring'
        
        Raises:
            ValueError: 当传入的季节参数无效时
        """
        if season not in self.SEASONAL_THRESHOLDS:
            raise ValueError(
                f"无效的季节参数: {season}. "
                f"可选值: {list(self.SEASONAL_THRESHOLDS.keys())}"
            )
        self.season = season
        self.thresholds = self.SEASONAL_THRESHOLDS[season]
        logger.info(f"ColorGrader 初始化完成，当前季节: {season}")
    
    def set_season(self, season: str) -> None:
        """
        动态切换季节参数
        
        Args:
            season: 季节参数，可选 'spring', 'summer', 'autumn', 'winter'
        
        Raises:
            ValueError: 当传入的季节参数无效时
        """
        if season not in self.SEASONAL_THRESHOLDS:
            raise ValueError(
                f"无效的季节参数: {season}. "
                f"可选值: {list(self.SEASONAL_THRESHOLDS.keys())}"
            )
        self.season = season
        self.thresholds = self.SEASONAL_THRESHOLDS[season]
        logger.info(f"季节参数已切换为: {season}")
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            加载的BGR格式图像数组
        
        Raises:
            FileNotFoundError: 图像文件不存在
            ValueError: 图像加载失败
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像：调整大小、去除背景等
        
        Args:
            image: BGR格式输入图像
        
        Returns:
            预处理后的图像
        """
        # 调整图像大小以统一处理（保持长宽比）
        max_size = 512
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        将BGR图像转换为HSV色彩空间
        
        Args:
            image: BGR格式图像
        
        Returns:
            HSV格式图像
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def _compute_histogram(self, channel: np.ndarray, bins: int = 32, 
                          range_max: int = 256) -> np.ndarray:
        """
        计算单通道直方图并归一化
        
        Args:
            channel: 单通道图像数据
            bins: 直方图bin数量
            range_max: 数值范围最大值
        
        Returns:
            归一化后的直方图
        """
        hist = cv2.calcHist([channel], [0], None, [bins], [0, range_max])
        # 归一化
        hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten()
        return hist
    
    def _find_dominant_hue(self, h_channel: np.ndarray, 
                           s_channel: np.ndarray) -> Tuple[float, str]:
        """
        找出主色调
        
        Args:
            h_channel: H通道数据 (0-179)
            s_channel: S通道数据 (0-255)
        
        Returns:
            (主色调值, 色调名称)
        """
        # 只考虑饱和度较高的像素（避免白色/灰色干扰）
        mask = s_channel > 30
        if np.sum(mask) < 100:  # 如果有效像素太少
            return -1, "未知"
        
        valid_hues = h_channel[mask]
        
        # 计算直方图找出峰值
        hist = np.histogram(valid_hues, bins=36, range=(0, 180))[0]
        dominant_bin = np.argmax(hist)
        dominant_hue = (dominant_bin * 5 + 2.5)  # 转换为0-179范围的值
        
        # 色调名称映射
        hue_names = {
            (0, 15): "红色", (15, 25): "橙红", (25, 35): "橙色",
            (35, 50): "黄色", (50, 70): "黄绿", (70, 90): "绿色",
            (90, 110): "青绿", (110, 130): "青色", (130, 150): "蓝紫",
            (150, 170): "紫色", (170, 180): "紫红"
        }
        
        hue_name = "未知"
        for (low, high), name in hue_names.items():
            if low <= dominant_hue < high:
                hue_name = name
                break
        
        return dominant_hue, hue_name
    
    def extract_hsv_features(self, image_path: Union[str, Path]) -> Dict:
        """
        提取单张图片的HSV特征
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            包含HSV特征的字典:
            - h_mean, s_mean, v_mean: 各通道均值 (H: 0-179, S/V: 0-255)
            - h_std, s_std, v_std: 各通道标准差
            - h_hist, s_hist, v_hist: 各通道归一化直方图
            - dominant_hue: 主色调值
            - dominant_hue_name: 主色调名称
            - saturation_score: 饱和度评分 (0-1)
            - brightness_score: 亮度评分 (0-1)
            - hue_concentration: 色调集中度 (0-1, 越高表示色调越集中)
        
        Raises:
            FileNotFoundError: 图像文件不存在
            ValueError: 图像处理失败
        """
        try:
            # 加载和预处理图像
            image = self._load_image(image_path)
            image = self._preprocess_image(image)
            
            # 转换为HSV
            hsv_image = self._convert_to_hsv(image)
            h_channel = hsv_image[:, :, 0].astype(np.float32)
            s_channel = hsv_image[:, :, 1].astype(np.float32)
            v_channel = hsv_image[:, :, 2].astype(np.float32)
            
            # 计算各通道统计特征
            h_mean = float(np.mean(h_channel))
            s_mean = float(np.mean(s_channel))
            v_mean = float(np.mean(v_channel))
            
            h_std = float(np.std(h_channel))
            s_std = float(np.std(s_channel))
            v_std = float(np.std(v_channel))
            
            # 计算直方图
            h_hist = self._compute_histogram(h_channel.astype(np.uint8), bins=36, range_max=180)
            s_hist = self._compute_histogram(s_channel.astype(np.uint8), bins=32, range_max=256)
            v_hist = self._compute_histogram(v_channel.astype(np.uint8), bins=32, range_max=256)
            
            # 找出主色调
            dominant_hue, dominant_hue_name = self._find_dominant_hue(
                h_channel.astype(np.uint8), 
                s_channel.astype(np.uint8)
            )
            
            # 计算饱和度评分 (归一化到0-1)
            saturation_score = s_mean / 255.0
            
            # 计算亮度评分 (归一化到0-1)
            brightness_score = v_mean / 255.0
            
            # 计算色调集中度 (基于标准差的倒数，归一化到0-1)
            # 标准差越小，集中度越高
            hue_concentration = max(0, 1 - (h_std / 60.0))
            
            features = {
                'h_mean': round(h_mean, 2),
                's_mean': round(s_mean, 2),
                'v_mean': round(v_mean, 2),
                'h_std': round(h_std, 2),
                's_std': round(s_std, 2),
                'v_std': round(v_std, 2),
                'h_hist': h_hist.tolist(),
                's_hist': s_hist.tolist(),
                'v_hist': v_hist.tolist(),
                'dominant_hue': round(dominant_hue, 2),
                'dominant_hue_name': dominant_hue_name,
                'saturation_score': round(saturation_score, 4),
                'brightness_score': round(brightness_score, 4),
                'hue_concentration': round(hue_concentration, 4),
            }
            
            logger.debug(f"成功提取HSV特征: {image_path}")
            return features
            
        except Exception as e:
            logger.error(f"提取HSV特征失败: {image_path}, 错误: {str(e)}")
            raise
    
    def grade_by_color(self, features: Dict) -> Dict:
        """
        基于HSV特征进行规则分级
        
        分级规则：
        - 等级1(优质): 高饱和度 + 适中亮度 + 色调集中
        - 等级2(良好): 中高饱和度 + 亮度正常
        - 等级3(一般): 中低饱和度 或 亮度偏暗/过亮
        - 等级4(较差): 低饱和度 + 亮度异常 + 色调分散
        
        Args:
            features: HSV特征字典，由 extract_hsv_features 返回
        
        Returns:
            分级结果字典:
            - grade: 等级 (1-4)
            - confidence: 置信度 (0-1)
            - reason: 分级原因说明
            - scores: 各项评分的详细信息
        """
        sat_score = features['saturation_score']
        bright_score = features['brightness_score']
        hue_conc = features['hue_concentration']
        
        th = self.thresholds
        
        # 计算综合评分
        # 使用加权平均，权重根据季节调整
        weighted_score = (
            sat_score * th['saturation_weight'] +
            bright_score * th['brightness_weight'] +
            hue_conc * th['hue_weight']
        )
        
        # 判断亮度是否适中（避免过暗或过亮）
        brightness_optimal = th['brightness_low'] <= bright_score <= th['brightness_high']
        brightness_penalty = 0
        if not brightness_optimal:
            # 亮度偏离最优范围的惩罚
            if bright_score < th['brightness_low']:
                brightness_penalty = (th['brightness_low'] - bright_score) / th['brightness_low']
            else:
                brightness_penalty = (bright_score - th['brightness_high']) / (1 - th['brightness_high'])
        
        # 根据规则进行分级
        grade = 4
        confidence = 0.5
        reasons = []
        
        # 等级1: 优质
        if (sat_score >= th['saturation_high'] and 
            th['brightness_medium'] <= bright_score <= th['brightness_high'] and
            hue_conc >= 0.6):
            grade = 1
            confidence = min(1.0, 0.8 + (sat_score - th['saturation_high']) * 0.5)
            reasons.append("饱和度高，亮度适中，色调集中")
        
        # 等级2: 良好
        elif (sat_score >= th['saturation_medium'] and 
              th['brightness_low'] <= bright_score <= th['brightness_high'] and
              hue_conc >= 0.4):
            grade = 2
            confidence = min(1.0, 0.7 + (sat_score - th['saturation_medium']) * 0.3)
            if sat_score < th['saturation_high']:
                reasons.append("饱和度良好但未达优质标准")
            if hue_conc < 0.6:
                reasons.append("色调集中度一般")
        
        # 等级3: 一般
        elif (sat_score >= th['saturation_low'] or 
              (th['brightness_low'] <= bright_score <= th['brightness_high'])):
            grade = 3
            confidence = 0.6
            if sat_score < th['saturation_medium']:
                reasons.append("饱和度偏低")
            if bright_score < th['brightness_low']:
                reasons.append("亮度偏暗")
            elif bright_score > th['brightness_high']:
                reasons.append("亮度过高")
            if hue_conc < 0.4:
                reasons.append("色调分散")
        
        # 等级4: 较差
        else:
            grade = 4
            confidence = min(1.0, 0.5 + brightness_penalty * 0.3)
            if sat_score < th['saturation_low']:
                reasons.append("饱和度低，颜色暗淡")
            if bright_score < th['brightness_low']:
                reasons.append("亮度过暗")
            elif bright_score > th['brightness_high']:
                reasons.append("亮度过高，可能过曝")
            if hue_conc < 0.3:
                reasons.append("色调分散，颜色不纯")
        
        # 如果没有特定原因，给出默认说明
        if not reasons:
            if grade == 1:
                reasons.append("整体颜色表现优秀")
            elif grade == 2:
                reasons.append("整体颜色表现良好")
            elif grade == 3:
                reasons.append("整体颜色表现一般")
            else:
                reasons.append("整体颜色表现较差")
        
        result = {
            'grade': grade,
            'confidence': round(confidence, 4),
            'reason': "；".join(reasons),
            'scores': {
                'saturation_score': round(sat_score, 4),
                'brightness_score': round(bright_score, 4),
                'hue_concentration': round(hue_conc, 4),
                'weighted_score': round(weighted_score, 4),
                'brightness_penalty': round(brightness_penalty, 4),
            }
        }
        
        return result
    
    def batch_grade(self, image_dir: Union[str, Path]) -> List[Dict]:
        """
        批量分级目录下所有图片
        
        Args:
            image_dir: 图片目录路径
        
        Returns:
            每张图片的分级结果列表，每个元素包含:
            - image_path: 图片路径
            - image_name: 图片文件名
            - features: HSV特征
            - grade_result: 分级结果
        
        Raises:
            FileNotFoundError: 目录不存在
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"目录不存在: {image_dir}")
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 获取所有图片文件
        image_files = [
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"目录中没有找到图片文件: {image_dir}")
            return []
        
        logger.info(f"开始批量分级，共 {len(image_files)} 张图片")
        
        results = []
        for image_path in sorted(image_files):
            try:
                # 获取完整报告
                report = self.get_color_report(image_path)
                results.append(report)
                logger.debug(f"分级完成: {image_path.name} -> 等级 {report['grade_result']['grade']}")
            except Exception as e:
                logger.error(f"处理图片失败: {image_path}, 错误: {str(e)}")
                # 记录失败信息，继续处理其他图片
                results.append({
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'error': str(e),
                    'features': None,
                    'grade_result': None,
                })
        
        logger.info(f"批量分级完成，成功: {len([r for r in results if 'error' not in r])}/{len(image_files)}")
        return results
    
    def get_color_report(self, image_path: Union[str, Path]) -> Dict:
        """
        获取单张图片的完整颜色分析报告
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            完整分析报告字典，包含:
            - image_path: 图片路径
            - image_name: 图片文件名
            - season: 当前季节参数
            - features: HSV特征
            - grade_result: 分级结果
        """
        image_path = Path(image_path)
        
        # 提取特征
        features = self.extract_hsv_features(image_path)
        
        # 进行分级
        grade_result = self.grade_by_color(features)
        
        report = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'season': self.season,
            'features': features,
            'grade_result': grade_result,
        }
        
        return report
    
    def get_seasonal_statistics(self, image_dir: Union[str, Path]) -> Dict:
        """
        获取某目录下图片的分级统计信息
        
        Args:
            image_dir: 图片目录路径
        
        Returns:
            统计信息字典，包含各等级数量和占比
        """
        results = self.batch_grade(image_dir)
        
        # 过滤掉失败的
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {
                'total': len(results),
                'valid': 0,
                'grade_distribution': {},
                'average_confidence': 0,
            }
        
        # 统计各等级数量
        grade_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        total_confidence = 0
        
        for result in valid_results:
            grade = result['grade_result']['grade']
            confidence = result['grade_result']['confidence']
            grade_counts[grade] += 1
            total_confidence += confidence
        
        total = len(valid_results)
        
        statistics = {
            'total': len(results),
            'valid': total,
            'season': self.season,
            'grade_distribution': {
                'grade_1': {
                    'count': grade_counts[1],
                    'percentage': round(grade_counts[1] / total * 100, 2) if total > 0 else 0
                },
                'grade_2': {
                    'count': grade_counts[2],
                    'percentage': round(grade_counts[2] / total * 100, 2) if total > 0 else 0
                },
                'grade_3': {
                    'count': grade_counts[3],
                    'percentage': round(grade_counts[3] / total * 100, 2) if total > 0 else 0
                },
                'grade_4': {
                    'count': grade_counts[4],
                    'percentage': round(grade_counts[4] / total * 100, 2) if total > 0 else 0
                },
            },
            'average_confidence': round(total_confidence / total, 4) if total > 0 else 0,
        }
        
        return statistics


# 便捷函数，用于快速测试
if __name__ == "__main__":
    # 测试代码
    grader = ColorGrader(season='spring')
    
    # 测试数据集路径
    test_dirs = [
        Path(DATASET_DIR) / "val" / "1",
        Path(DATASET_DIR) / "val" / "2",
        Path(DATASET_DIR) / "val" / "3",
        Path(DATASET_DIR) / "val" / "4",
    ]
    
    print("=" * 60)
    print("HSV颜色分级器测试")
    print("=" * 60)
    
    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"\n测试目录: {test_dir}")
            print("-" * 40)
            
            # 获取统计信息
            stats = grader.get_seasonal_statistics(test_dir)
            print(f"总计: {stats['valid']} 张图片")
            print(f"等级分布:")
            for grade in range(1, 5):
                info = stats['grade_distribution'][f'grade_{grade}']
                print(f"  等级{grade}: {info['count']}张 ({info['percentage']}%)")
            print(f"平均置信度: {stats['average_confidence']}")
            
            # 显示前2张图片的详细报告
            image_files = sorted(list(test_dir.glob("*.jpg")))[:2]
            for img_path in image_files:
                print(f"\n  图片: {img_path.name}")
                try:
                    report = grader.get_color_report(img_path)
                    features = report['features']
                    grade_result = report['grade_result']
                    print(f"    主色调: {features['dominant_hue_name']} ({features['dominant_hue']})")
                    print(f"    饱和度: {features['saturation_score']:.3f}")
                    print(f"    亮度: {features['brightness_score']:.3f}")
                    print(f"    分级: 等级{grade_result['grade']} (置信度: {grade_result['confidence']:.3f})")
                    print(f"    原因: {grade_result['reason']}")
                except Exception as e:
                    print(f"    错误: {e}")
        else:
            print(f"目录不存在: {test_dir}")
    
    print("\n" + "=" * 60)
    print("测试完成")
