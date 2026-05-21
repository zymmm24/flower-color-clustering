# drift_detector.py
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils import (
    compute_mmd,
    feature_level_tests,
    nearest_neighbor_anomaly,
    get_logger,
    BASELINE_ASSETS_DIR,
    estimate_gamma
)

logger = get_logger(__name__)

# -----------------------------
# 核心检测器
# -----------------------------
class DriftDetector:
    def __init__(self, baseline_path=None):
        if baseline_path is None:
            baseline_path = os.path.join(BASELINE_ASSETS_DIR, "baseline_db.pkl")
        self.baseline_df = pd.read_pickle(baseline_path)
        self.baseline_X = np.stack(self.baseline_df["embedding_pca"].values).astype(np.float32)
        self.baseline_labels = self.baseline_df['label'].values
        logger.info(f"基准库加载成功: {len(self.baseline_X)} 样本")

    # -------- 全局漂移 MMD --------
    def _estimate_gamma(self, X, Y):
        return estimate_gamma(X, Y)

    def calculate_mmd(self, X, Y, gamma):
        return compute_mmd(X, Y, gamma)

    def run_permutation_test(self, X, Y, iterations=100):
        gamma = self._estimate_gamma(X, Y)
        observed_mmd = self.calculate_mmd(X, Y, gamma)
        combined = np.vstack([X, Y])
        n = X.shape[0]
        count = 0
        logger.info(f"执行排列检验 ({iterations} 次迭代)...")
        for _ in range(iterations):
            idx = np.random.permutation(len(combined))
            new_X = combined[idx[:n]]
            new_Y = combined[idx[n:]]
            if self.calculate_mmd(new_X, new_Y, gamma) >= observed_mmd:
                count += 1
        p_value = count / iterations
        return observed_mmd, p_value

    # -------- 主检测流程 --------
    def detect(self, test_pkl_path, window_size=100, alpha=0.05, save_path=None):
        if save_path is None:
            save_path = os.path.join(BASELINE_ASSETS_DIR, "drift_result.pkl")
        test_df = pd.read_pickle(test_pkl_path)
        test_X = np.stack(test_df['embedding_pca'].values).astype(np.float32)
        test_labels = test_df['label'].values

        # 随机窗口采样
        size = min(len(self.baseline_X), len(test_X), window_size)
        X_sub = self.baseline_X[np.random.choice(len(self.baseline_X), size, replace=False)]
        Y_sub = test_X[np.random.choice(len(test_X), size, replace=False)]

        # 全局 MMD
        mmd_score, p_val = self.run_permutation_test(X_sub, Y_sub)
        is_drift = p_val < alpha
        status = "DRIFT DETECTED" if is_drift else "DATA STABLE"
        logger.info(f"全局 MMD={mmd_score:.4f}, p-value={p_val:.4f} -> {status}")

        # -------- 按类别漂移 --------
        per_class = {}
        classes = np.unique(self.baseline_labels)
        for cls in classes:
            base_cls_X = self.baseline_X[self.baseline_labels == cls]
            cur_cls_X = test_X[test_labels == cls]
            if len(cur_cls_X) < 5:
                continue
            mmd_cls = compute_mmd(base_cls_X, cur_cls_X)
            per_class[cls] = "DRIFT" if mmd_cls > 0.01 else "STABLE"

        # -------- 特征维度漂移 --------
        feature_level = feature_level_tests(self.baseline_X, test_X, alpha=alpha)
        logger.info(f"特征维度漂移: {feature_level['changed_dims']}")

        # -------- 样本级漂移 --------
        top_samples, nn_dists = nearest_neighbor_anomaly(self.baseline_X, test_X)
        logger.info(f"样本级漂移 top samples: {top_samples}")

        # -------- 保存中间结果 --------
        result = {
            "timestamp": datetime.now().isoformat(),
            "baseline_size": len(self.baseline_X),
            "test_size": len(test_X),
            "window_size": size,
            "mmd_score": float(mmd_score),
            "p_value": float(p_val),
            "alpha": float(alpha),
            "is_drift": bool(is_drift),
            "status": status,
            "baseline_source": test_pkl_path,
            "test_source": test_pkl_path,
            "per_class": per_class,
            "feature_level": feature_level,
            "sample_level": {"top_samples": top_samples, "nn_dists": nn_dists.tolist()},
        }

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

        logger.info(f"中间漂移检测结果已保存至: {save_path}")
        return result

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    try:
        detector = DriftDetector()
        val_test_path = os.path.join(BASELINE_ASSETS_DIR, "val_test_data.pkl")
        detector.detect(val_test_path)
        logger.info("脚本执行成功完成！")
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        import traceback
        traceback.print_exc()


