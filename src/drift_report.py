import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils import (
    compute_mmd,
    feature_level_tests,
    nearest_neighbor_anomaly,
    get_logger,
    BASELINE_ASSETS_DIR
)

logger = get_logger(__name__)

# -----------------------------
# DriftReportGenerator
# -----------------------------
class DriftReportGenerator:
    def __init__(self, baseline_pkl, test_pkl):
        if not os.path.exists(baseline_pkl):
            raise FileNotFoundError(f"未找到基准数据: {baseline_pkl}")
        if not os.path.exists(test_pkl):
            raise FileNotFoundError(f"未找到测试数据: {test_pkl}")

        self.baseline_df = pd.read_pickle(baseline_pkl)
        self.test_df = pd.read_pickle(test_pkl)
        self.baseline_emb = np.stack(self.baseline_df["embedding_pca"].values)
        self.test_emb = np.stack(self.test_df["embedding_pca"].values)
        self.test_names = self.test_df["img_name"].values

    def generate_report(self, output_path="drift_report.json", alpha=0.05):
        try:
            # 全局 MMD 漂移
            mmd_score = compute_mmd(self.baseline_emb, self.test_emb)
            is_drift_global = bool(mmd_score > 0.01)
            status_global = "DRIFT DETECTED" if is_drift_global else "DATA STABLE"

            # 按类漂移
            per_class = {}
            classes = self.baseline_df["label"].unique()
            for cls in classes:
                try:
                    base_cls_emb = np.stack(self.baseline_df[self.baseline_df["label"]==cls]["embedding_pca"].values)
                    test_cls_emb = np.stack(self.test_df[self.test_df["label"]==cls]["embedding_pca"].values)
                    if len(test_cls_emb) > 0:
                        mmd_cls = compute_mmd(base_cls_emb, test_cls_emb)
                        per_class[int(cls)] = {
                            "baseline_size": int(len(base_cls_emb)),
                            "test_size": int(len(test_cls_emb)),
                            "mmd": float(mmd_cls),
                            "is_drift": bool(mmd_cls > 0.01)
                        }
                except Exception as e:
                    logger.warning(f"Failed to process class {cls}: {e}")
                    continue

            # 特征维度漂移
            try:
                changed_dims, feature_details = feature_level_tests(
                    self.baseline_emb, self.test_emb, alpha=alpha, return_details=True
                )
            except Exception as e:
                logger.warning(f"Feature level tests failed: {e}")
                changed_dims, feature_details = [], []

            # 样本级漂移
            try:
                top_samples = nearest_neighbor_anomaly(
                    self.baseline_emb, self.test_emb, self.test_names, top_k=50
                )
            except Exception as e:
                logger.warning(f"Nearest neighbor anomaly failed: {e}")
                top_samples = []

            # 生成报告
            report = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "YOLO Feature Drift Report",
                    "version": "v1.0"
                },
                "data_info": {
                    "baseline_source": "baseline_assets/baseline_db.pkl",
                    "test_source": "baseline_assets/val_test_data.pkl",
                    "baseline_size": int(len(self.baseline_emb)),
                    "test_size": int(len(self.test_emb))
                },
                "statistics": {
                    "mmd_score": float(mmd_score),
                    "alpha": float(alpha)
                },
                "decision": {
                    "is_drift": is_drift_global,
                    "status": status_global
                },
                "interpretation": (
                    "检测到漂移" if is_drift_global else
                    "当前数据分布与训练阶段保持一致，未发现显著特征漂移，模型运行状态稳定。"
                ),
                "per_class_drift": per_class,
                "feature_level_drift": {
                    "changed_dims": changed_dims,
                    "details": feature_details
                },
                "sample_level_drift": top_samples
            }

            # 保存 JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"漂移报告已生成: {output_path}")
            return report

        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    try:
        baseline_path = os.path.join(BASELINE_ASSETS_DIR, "baseline_db.pkl")
        test_path = os.path.join(BASELINE_ASSETS_DIR, "val_test_data.pkl")
        generator = DriftReportGenerator(baseline_path, test_path)
        generator.generate_report()
        logger.info("漂移报告生成成功！")
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        import traceback
        traceback.print_exc()
