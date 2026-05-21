"""
RoseGrade 花卉质量分级 REST API 服务

集成 YOLO11 分类模型和 HSV 颜色分级，提供双路融合分级服务
"""
import os
import io
import json
import time
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from src.utils import get_logger, BASE_DIR, BASELINE_ASSETS_DIR, WEIGHTS_DIR, REPORTS_DIR
from src.color_grader import ColorGrader
from src.auto_trainer import AutoTrainer
from src.drift_report import DriftReportGenerator

logger = get_logger(__name__)


class QualityGrader:
    """
    花卉质量分级器
    集成 YOLO11 分类模型和 HSV 颜色分级，提供双路融合分级
    """
    
    # 默认融合权重
    YOLO_WEIGHT = 0.6
    HSV_WEIGHT = 0.4
    
    def __init__(self, model_path: Optional[str] = None, season: str = 'spring'):
        """
        初始化分级器
        加载 YOLO11 模型和 ColorGrader
        
        Args:
            model_path: YOLO11 模型路径，默认使用 runs/classify/train2/weights/best.pt
            season: 季节参数，影响 HSV 分级阈值
        """
        # 设置模型路径
        if model_path is None:
            model_path = os.path.join(WEIGHTS_DIR, "best.pt")
        self.model_path = model_path
        
        # 加载 YOLO11 模型
        logger.info(f"正在加载 YOLO11 模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = YOLO(model_path)
        logger.info("YOLO11 模型加载完成")
        
        # 初始化颜色分级器
        self.color_grader = ColorGrader(season=season)
        logger.info(f"ColorGrader 初始化完成，季节: {season}")
        
        # 漂移报告路径
        self.drift_report_path = os.path.join(BASE_DIR, "drift_report.json")
        
        # 记录初始化时间
        self.initialized_at = datetime.now().isoformat()
    
    def _save_bytes_to_temp(self, image_bytes: bytes) -> str:
        """
        将字节数据保存为临时文件
        
        Args:
            image_bytes: 图片二进制数据
            
        Returns:
            临时文件路径
        """
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(image_bytes)
        temp_file.close()
        return temp_file.name
    
    def _grade_to_score(self, grade: int) -> float:
        """
        将等级转换为分数（用于融合计算）
        等级 1->4 对应分数 4->1（越高越好）
        
        Args:
            grade: 等级 (1-4)
            
        Returns:
            分数 (1.0-4.0)
        """
        # 等级 1 对应最高分 4，等级 4 对应最低分 1
        score_map = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}
        return score_map.get(grade, 2.5)
    
    def _score_to_grade(self, score: float) -> int:
        """
        将融合分数转换为等级
        
        Args:
            score: 融合分数 (1.0-4.0)
            
        Returns:
            等级 (1-4)
        """
        # 四舍五入到最近的等级
        if score >= 3.5:
            return 1
        elif score >= 2.5:
            return 2
        elif score >= 1.5:
            return 3
        else:
            return 4
    
    def grade_single(self, image_path: Optional[str] = None, 
                     image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        单张图片分级
        
        1. YOLO11 推理获取分类结果和置信度
        2. HSV 颜色特征分析
        3. 双路融合：yolo_weight * yolo_grade + hsv_weight * hsv_grade
        
        Args:
            image_path: 图片文件路径（本地文件）
            image_bytes: 图片二进制数据（上传文件）
            
        Returns:
            dict: {
                'grade': int (1-4),
                'confidence': float,
                'yolo_result': {'grade': int, 'confidence': float, 'top5': list},
                'hsv_result': {'grade': int, 'confidence': float, 'features': dict},
                'fusion_score': float,
                'timestamp': str
            }
            
        Raises:
            ValueError: 当未提供图片路径或字节数据时
            FileNotFoundError: 当图片路径不存在时
        """
        temp_path = None
        try:
            # 确定输入方式
            if image_bytes is not None:
                # 字节数据：保存为临时文件
                temp_path = self._save_bytes_to_temp(image_bytes)
                process_path = temp_path
                source_type = "bytes"
            elif image_path is not None:
                # 本地文件路径
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"图片文件不存在: {image_path}")
                process_path = image_path
                source_type = "path"
            else:
                raise ValueError("必须提供 image_path 或 image_bytes 之一")
            
            logger.info(f"开始分级: {process_path} (来源: {source_type})")
            
            # ========== Step 1: YOLO11 推理 ==========
            yolo_results = self.model.predict(process_path, verbose=False)
            yolo_result_obj = yolo_results[0]
            
            if hasattr(yolo_result_obj, 'probs') and yolo_result_obj.probs is not None:
                # YOLO 分类结果（等级从 0 开始，需要 +1）
                yolo_grade = int(yolo_result_obj.probs.top1) + 1
                yolo_confidence = float(yolo_result_obj.probs.top1conf)
                # Top-5 等级（同样 +1）
                top5_indices = yolo_result_obj.probs.top5.tolist()
                top5 = [int(idx) + 1 for idx in top5_indices]
            else:
                # 如果没有概率信息，使用默认值
                yolo_grade = 3
                yolo_confidence = 0.5
                top5 = [3, 2, 4, 1]
            
            yolo_result = {
                'grade': yolo_grade,
                'confidence': round(yolo_confidence, 4),
                'top5': top5
            }
            logger.debug(f"YOLO 结果: 等级={yolo_grade}, 置信度={yolo_confidence:.4f}")
            
            # ========== Step 2: HSV 颜色分析 ==========
            hsv_features = self.color_grader.extract_hsv_features(process_path)
            hsv_grade_result = self.color_grader.grade_by_color(hsv_features)
            
            hsv_grade = hsv_grade_result['grade']
            hsv_confidence = hsv_grade_result['confidence']
            
            hsv_result = {
                'grade': hsv_grade,
                'confidence': round(hsv_confidence, 4),
                'features': {
                    'dominant_hue_name': hsv_features['dominant_hue_name'],
                    'saturation_score': hsv_features['saturation_score'],
                    'brightness_score': hsv_features['brightness_score'],
                    'hue_concentration': hsv_features['hue_concentration']
                }
            }
            logger.debug(f"HSV 结果: 等级={hsv_grade}, 置信度={hsv_confidence:.4f}")
            
            # ========== Step 3: 双路融合 ==========
            # 将等级转换为分数进行融合
            yolo_score = self._grade_to_score(yolo_grade)
            hsv_score = self._grade_to_score(hsv_grade)
            
            # 加权融合
            fusion_score = (
                self.YOLO_WEIGHT * yolo_score +
                self.HSV_WEIGHT * hsv_score
            )
            
            # 融合后的等级
            final_grade = self._score_to_grade(fusion_score)
            
            # 融合置信度（加权平均）
            fusion_confidence = (
                self.YOLO_WEIGHT * yolo_confidence +
                self.HSV_WEIGHT * hsv_confidence
            )
            
            logger.info(f"分级完成: 等级={final_grade}, 融合分数={fusion_score:.2f}")
            
            # 构建返回结果
            result = {
                'grade': final_grade,
                'confidence': round(fusion_confidence, 4),
                'yolo_result': yolo_result,
                'hsv_result': hsv_result,
                'fusion_score': round(fusion_score, 4),
                'fusion_weights': {
                    'yolo': self.YOLO_WEIGHT,
                    'hsv': self.HSV_WEIGHT
                },
                'timestamp': datetime.now().isoformat(),
                'source': source_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分级失败: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug(f"临时文件已清理: {temp_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
    
    def grade_batch(self, image_paths: Optional[List[str]] = None,
                    image_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量分级
        
        Args:
            image_paths: 图片路径列表
            image_dir: 图片目录路径（将分级该目录下所有图片）
            
        Returns:
            list[dict]: 每张图的分级结果列表
            
        Raises:
            ValueError: 当未提供图片路径列表或目录时
            FileNotFoundError: 当目录不存在时
        """
        # 收集所有图片路径
        paths_to_process = []
        
        if image_paths is not None:
            paths_to_process.extend(image_paths)
        
        if image_dir is not None:
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"目录不存在: {image_dir}")
            
            # 支持的图片格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            # 获取目录下所有图片
            for filename in os.listdir(image_dir):
                if Path(filename).suffix.lower() in image_extensions:
                    paths_to_process.append(os.path.join(image_dir, filename))
        
        if not paths_to_process:
            raise ValueError("未找到要分级的图片，请提供 image_paths 或有效的 image_dir")
        
        logger.info(f"开始批量分级，共 {len(paths_to_process)} 张图片")
        
        results = []
        success_count = 0
        
        for i, img_path in enumerate(paths_to_process):
            try:
                result = self.grade_single(image_path=img_path)
                result['image_path'] = img_path
                result['image_name'] = Path(img_path).name
                results.append(result)
                success_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(paths_to_process)}")
                    
            except Exception as e:
                logger.error(f"处理失败 [{img_path}]: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'image_name': Path(img_path).name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"批量分级完成: 成功 {success_count}/{len(paths_to_process)}")
        return results
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        获取最新漂移检测状态
        读取 drift_report.json
        
        Returns:
            dict: 简要状态摘要
        """
        try:
            if not os.path.exists(self.drift_report_path):
                return {
                    'status': 'unknown',
                    'error': '漂移报告不存在',
                    'report_path': self.drift_report_path
                }
            
            with open(self.drift_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # 提取关键信息
            is_drift = report.get('decision', {}).get('is_drift', False)
            status = report.get('decision', {}).get('status', 'UNKNOWN')
            mmd_score = report.get('statistics', {}).get('mmd_score', 0)
            generated_at = report.get('meta', {}).get('generated_at', 'unknown')
            
            # 统计漂移类别
            per_class_drift = report.get('per_class_drift', {})
            drifted_classes = [
                cls for cls, info in per_class_drift.items()
                if info.get('is_drift', False)
            ]
            
            # 判断严重程度
            if mmd_score >= 0.1:
                severity = 'high'
            elif mmd_score >= 0.05:
                severity = 'medium'
            elif mmd_score >= 0.01:
                severity = 'low'
            else:
                severity = 'none'
            
            return {
                'status': 'drift_detected' if is_drift else 'stable',
                'severity': severity,
                'mmd_score': mmd_score,
                'is_drift': is_drift,
                'drifted_classes': drifted_classes,
                'report_generated_at': generated_at,
                'report_path': self.drift_report_path,
                'interpretation': report.get('interpretation', '')
            }
            
        except Exception as e:
            logger.error(f"获取漂移状态失败: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'report_path': self.drift_report_path
            }
    
    def get_drift_report(self) -> Dict[str, Any]:
        """
        获取完整漂移报告 JSON
        
        Returns:
            dict: 完整漂移报告内容
            
        Raises:
            FileNotFoundError: 当漂移报告不存在时
        """
        try:
            if not os.path.exists(self.drift_report_path):
                raise FileNotFoundError(f"漂移报告不存在: {self.drift_report_path}")
            
            with open(self.drift_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            return report
            
        except Exception as e:
            logger.error(f"读取漂移报告失败: {str(e)}")
            raise
    
    def trigger_detection(self) -> Dict[str, Any]:
        """
        手动触发一次漂移检测
        
        Returns:
            dict: 检测结果摘要
        """
        try:
            logger.info("手动触发漂移检测...")
            
            # 检查必要的文件是否存在
            baseline_path = os.path.join(BASELINE_ASSETS_DIR, "baseline_db.pkl")
            test_path = os.path.join(BASELINE_ASSETS_DIR, "val_test_data.pkl")
            
            if not os.path.exists(baseline_path):
                raise FileNotFoundError(f"基准数据不存在: {baseline_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"测试数据不存在: {test_path}")
            
            # 创建报告生成器并生成报告
            generator = DriftReportGenerator(baseline_path, test_path)
            report = generator.generate_report(output_path=self.drift_report_path)
            
            logger.info("漂移检测完成")
            
            # 返回简要结果
            return {
                'success': True,
                'is_drift': report.get('decision', {}).get('is_drift', False),
                'mmd_score': report.get('statistics', {}).get('mmd_score', 0),
                'status': report.get('decision', {}).get('status', 'UNKNOWN'),
                'report_path': self.drift_report_path,
                'generated_at': report.get('meta', {}).get('generated_at')
            }
            
        except Exception as e:
            logger.error(f"漂移检测失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'report_path': self.drift_report_path
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息
        
        Returns:
            dict: 模型信息
        """
        return {
            'model_path': self.model_path,
            'model_type': 'YOLO11 Classification',
            'initialized_at': self.initialized_at,
            'season': self.color_grader.season,
            'fusion_weights': {
                'yolo': self.YOLO_WEIGHT,
                'hsv': self.HSV_WEIGHT
            }
        }


# ===== FastAPI 应用 =====
app = FastAPI(
    title="RoseGrade 花卉智能分级系统",
    description="自适应鲜切花卉智能分级 REST API",
    version="1.0.0"
)

# 全局 grader 实例（延迟初始化）
_grader = None


def get_grader():
    """获取或初始化全局 grader 实例"""
    global _grader
    if _grader is None:
        _grader = QualityGrader()
    return _grader


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "service": "RoseGrade",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/grade_single")
async def api_grade_single(file: UploadFile = File(...)):
    """
    单张图片分级 API
    
    上传单张图片，返回 YOLO + HSV 双路融合分级结果
    """
    try:
        # 验证文件类型
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}，请上传 {allowed_extensions} 格式的图片"
            )
        
        # 读取上传的文件
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        
        # 获取 grader 并执行分级
        grader = get_grader()
        result = grader.grade_single(image_bytes=contents)
        
        # 添加文件名信息
        result['filename'] = file.filename
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"单张图片分级 API 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分级失败: {str(e)}")


@app.post("/api/grade_batch")
async def api_grade_batch(files: List[UploadFile] = File(...)):
    """
    批量分级 API
    
    上传多张图片，返回批量分级结果
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="未上传任何文件")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="单次最多支持 50 张图片")
        
        grader = get_grader()
        results = []
        
        for file in files:
            try:
                # 验证文件类型
                file_ext = Path(file.filename).suffix.lower()
                allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                if file_ext not in allowed_extensions:
                    results.append({
                        'filename': file.filename,
                        'error': f'不支持的文件格式: {file_ext}',
                        'status': 'failed'
                    })
                    continue
                
                # 读取并分级
                contents = await file.read()
                result = grader.grade_single(image_bytes=contents)
                result['filename'] = file.filename
                result['status'] = 'success'
                results.append(result)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # 统计结果
        success_count = len([r for r in results if r.get('status') == 'success'])
        
        return JSONResponse(content={
            'total': len(files),
            'success': success_count,
            'failed': len(files) - success_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量分级 API 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量分级失败: {str(e)}")


@app.get("/api/drift_status")
async def api_drift_status():
    """获取漂移检测状态"""
    try:
        grader = get_grader()
        status = grader.get_drift_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"获取漂移状态 API 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取漂移状态失败: {str(e)}")


@app.get("/api/drift_report")
async def api_drift_report():
    """获取完整漂移报告"""
    try:
        grader = get_grader()
        report = grader.get_drift_report()
        return JSONResponse(content=report)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取漂移报告 API 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取漂移报告失败: {str(e)}")


@app.post("/api/trigger_detection")
async def api_trigger_detection():
    """手动触发漂移检测"""
    try:
        grader = get_grader()
        result = grader.trigger_detection()
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('error', '检测失败'))
        
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"触发漂移检测 API 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"触发检测失败: {str(e)}")


@app.get("/api/model_info")
async def api_model_info():
    """获取当前模型信息"""
    try:
        grader = get_grader()
        info = grader.get_model_info()
        return JSONResponse(content=info)
    except Exception as e:
        logger.error(f"获取模型信息 API 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8080):
    """启动 API 服务器"""
    logger.info(f"启动 RoseGrade API 服务器: http://{host}:{port}")
    logger.info(f"API 文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RoseGrade 分级服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8080, help="服务器端口")
    args = parser.parse_args()
    
    start_server(args.host, args.port)
