"""
RoseGrade 自适应鲜切花卉智能分级系统

该包提供以下核心功能模块：
- YOLO11AutoCollector: 自动收集基准资产
- DriftDetector: 漂移检测器
- DriftReportGenerator: 漂移报告生成器
- ColorGrader: 颜色分级器
- OnlineSampler: 在线采样器
- AutoTrainer: 自动训练器
- QualityGrader: 质量分级器
- DriftVisualizer: 漂移可视化
- DynamicDetectionPipeline: 动态检测流水线

使用延迟导入方式避免重依赖问题。
示例：
    from src import QualityGrader
    # 或
    from src.quality_grader import QualityGrader
"""

__version__ = '1.0.0'
__author__ = 'RoseGrade Team'

# 延迟导入 - 避免在导入包时加载重依赖模块
# 用户可以通过 from src.xxx import Xxx 直接导入

# 定义公共API（用于 IDE 自动补全提示）
__all__ = [
    'YOLO11AutoCollector',
    'DriftDetector',
    'DriftReportGenerator',
    'ColorGrader',
    'OnlineSampler',
    'AutoTrainer',
    'QualityGrader',
    'DriftVisualizer',
    'DynamicDetectionPipeline',
]


def __getattr__(name):
    """
    延迟导入机制 - 只在实际访问时导入对应模块
    这样可以避免在导入 src 包时立即加载 torch, ultralytics 等重依赖
    """
    if name == 'YOLO11AutoCollector':
        from src.baseline_collector import YOLO11AutoCollector
        return YOLO11AutoCollector
    elif name == 'DriftDetector':
        from src.drift_detector import DriftDetector
        return DriftDetector
    elif name == 'DriftReportGenerator':
        from src.drift_report import DriftReportGenerator
        return DriftReportGenerator
    elif name == 'ColorGrader':
        from src.color_grader import ColorGrader
        return ColorGrader
    elif name == 'OnlineSampler':
        from src.online_sampler import OnlineSampler
        return OnlineSampler
    elif name == 'AutoTrainer':
        from src.auto_trainer import AutoTrainer
        return AutoTrainer
    elif name == 'QualityGrader':
        from src.quality_grader import QualityGrader
        return QualityGrader
    elif name == 'DriftVisualizer':
        from src.drift_visualizer import DriftVisualizer
        return DriftVisualizer
    elif name == 'DynamicDetectionPipeline':
        from src.dynamic_detection_pipeline import DynamicDetectionPipeline
        return DynamicDetectionPipeline
    raise AttributeError(f"module 'src' has no attribute '{name}'")
