"""
公共工具模块 - 包含漂移检测和报告生成中共享的函数和配置
"""
import os
import logging
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances  # pyright: ignore[reportMissingImports]

# -----------------------------
# 路径常量
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_ASSETS_DIR = os.path.join(BASE_DIR, "baseline_assets")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SAMPLING_DIR = os.path.join(BASE_DIR, "sampling_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_DIR = os.path.join(BASE_DIR, "runs", "classify", "train2", "weights")


# -----------------------------
# 统一日志配置
# -----------------------------
def get_logger(name):
    """获取统一配置的 logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# -----------------------------
# MMD 计算函数
# -----------------------------
def compute_mmd(X, Y, gamma=1.0):
    """
    RBF Kernel MMD 计算
    
    Args:
        X: 基准样本嵌入 (n_samples_X, n_features)
        Y: 当前样本嵌入 (n_samples_Y, n_features)
        gamma: RBF 核参数
    
    Returns:
        MMD 距离值
    """
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    m = X.shape[0]
    n = Y.shape[0]
    mmd = K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n)
    return float(np.sqrt(max(mmd, 0.0)))


def estimate_gamma(X, Y):
    """
    基于中位数启发式估计 gamma 参数
    
    Args:
        X: 样本嵌入
        Y: 样本嵌入
    
    Returns:
        估计的 gamma 值
    """
    combined = np.vstack([X, Y])
    dists = pdist(combined, metric="sqeuclidean")
    median_dist = np.median(dists)
    return 1.0 / median_dist if median_dist > 0 else 1.0


# -----------------------------
# 特征级漂移检测
# -----------------------------
def feature_level_tests(baseline_emb, current_emb, alpha=0.05, return_details=False):
    """
    KS-test + Cohen's d 特征维度漂移检测
    
    Args:
        baseline_emb: 基准嵌入 (n_samples, n_features)
        current_emb: 当前嵌入 (n_samples, n_features)
        alpha: 显著性水平
        return_details: 是否返回详细信息（包含 feature 名称）
    
    Returns:
        如果 return_details=False (默认):
            dict: {'changed_dims': list of int, 'pvals': list, 'cohen_d': list}
        如果 return_details=True:
            tuple: (changed_dims: list of str, details: list of dict)
    """
    changed_dims = []
    pvals = []
    cohen_d = []
    details = []
    
    for dim in range(baseline_emb.shape[1]):
        stat, p = ks_2samp(baseline_emb[:, dim], current_emb[:, dim])
        mean_diff = baseline_emb[:, dim].mean() - current_emb[:, dim].mean()
        pooled_std = np.sqrt((baseline_emb[:, dim].var() + current_emb[:, dim].var()) / 2)
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        pvals.append(p)
        cohen_d.append(d)
        
        if return_details:
            details.append({
                "feature": f"dim_{dim}",
                "pval": float(p),
                "cohen_d": float(d)
            })
            if p < alpha and abs(d) > 0.3:
                changed_dims.append(f"dim_{dim}")
        else:
            if p < alpha and abs(d) > 0.3:
                changed_dims.append(dim)
    
    if return_details:
        return changed_dims, details
    else:
        return {'changed_dims': changed_dims, 'pvals': pvals, 'cohen_d': cohen_d}


# -----------------------------
# 样本级异常检测
# -----------------------------
def nearest_neighbor_anomaly(baseline_emb, current_emb, current_names=None, top_k=50):
    """
    基于最近邻距离的样本级异常检测
    
    Args:
        baseline_emb: 基准嵌入
        current_emb: 当前嵌入
        current_names: 当前样本名称列表（可选，用于报告生成）
        top_k: 返回的异常样本数量
    
    Returns:
        如果 current_names 为 None:
            tuple: (top_samples: list of int indices, nn_dists: array)
        如果 current_names 不为 None:
            list: [{"img_name": str, "nn_dist": float}, ...]
    """
    dists = pairwise_distances(current_emb, baseline_emb)
    nn_dist = dists.min(axis=1)
    top_idx = np.argsort(nn_dist)[-top_k:][::-1]
    
    if current_names is not None:
        top_samples = [{"img_name": str(current_names[i]), "nn_dist": float(nn_dist[i])} for i in top_idx]
        return top_samples
    else:
        top_samples = [i for i in top_idx]
        return top_samples, nn_dist[top_idx]
