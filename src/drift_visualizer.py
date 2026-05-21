"""
漂移分析可视化模块 - 生成各类分析图表
"""
import os
import sys
import json
import pickle
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA

# 添加父目录到路径以支持直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, BASELINE_ASSETS_DIR, REPORTS_DIR

logger = get_logger(__name__)


class DriftVisualizer:
    """漂移分析可视化模块 - 生成各类分析图表"""
    
    def __init__(self, output_dir=None):
        """初始化，设置输出目录和matplotlib中文字体"""
        self.output_dir = output_dir if output_dir else REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 配置中文字体
        self._setup_chinese_font()
        
        # 配色方案
        self.colors = {
            'baseline': '#3498db',      # 蓝色 - 基准数据
            'current': '#e74c3c',       # 红色 - 当前/漂移数据
            'stable': '#2ecc71',        # 绿色 - 稳定
            'warning': '#f39c12',       # 橙色 - 警告
            'threshold': '#95a5a6',     # 灰色 - 阈值线
            'background': '#f8f9fa'     # 浅灰背景
        }
    
    def _setup_chinese_font(self):
        """配置matplotlib中文字体支持"""
        self.use_chinese = False
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        
        try:
            # 尝试找到可用的中文字体
            available_fonts = [f.name for f in font_manager.fontManager.ttflist]
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    self.use_chinese = True
                    logger.info(f"使用字体: {font_name}")
                    break
            
            if not self.use_chinese:
                logger.warning("未找到中文字体，将使用英文标签")
        except Exception as e:
            logger.warning(f"字体配置失败: {e}")
    
    def _get_text(self, chinese, english):
        """根据字体支持返回中文或英文文本"""
        return chinese if self.use_chinese else english
    
    def _save_figure(self, fig, save_name):
        """保存图表到指定路径"""
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"图表已保存: {save_path}")
        return save_path
    
    def plot_pca_scatter(self, baseline_emb, current_emb, 
                         baseline_labels=None, current_labels=None,
                         title="PCA特征分布对比", save_name="pca_scatter.png"):
        """
        PCA 2D 散点图 - 基准数据 vs 当前数据分布对比
        不同类别用不同颜色，基准用圆点，当前用三角
        
        Args:
            baseline_emb: 基准数据嵌入 (n_samples, n_features)
            current_emb: 当前数据嵌入 (n_samples, n_features)
            baseline_labels: 基准数据标签 (可选)
            current_labels: 当前数据标签 (可选)
            title: 图表标题
            save_name: 保存文件名
        
        Returns:
            str: 保存的文件路径
        """
        try:
            # 合并数据并进行PCA降维
            combined = np.vstack([baseline_emb, current_emb])
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined)
            
            n_baseline = len(baseline_emb)
            baseline_pca = combined_pca[:n_baseline]
            current_pca = combined_pca[n_baseline:]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            ax.set_facecolor(self.colors['background'])
            
            # 类别颜色映射
            unique_labels = set()
            if baseline_labels is not None:
                unique_labels.update(baseline_labels)
            if current_labels is not None:
                unique_labels.update(current_labels)
            unique_labels = sorted(list(unique_labels))
            
            colors_map = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 4)))
            
            # 绘制基准数据（圆点）
            if baseline_labels is not None:
                for i, label in enumerate(unique_labels):
                    mask = baseline_labels == label
                    ax.scatter(baseline_pca[mask, 0], baseline_pca[mask, 1],
                              c=[colors_map[i]], marker='o', s=50, alpha=0.6,
                              label=f'{self._get_text("基准", "Baseline")} - {self._get_text("类别", "Class")} {label}',
                              edgecolors='white', linewidth=0.5)
            else:
                ax.scatter(baseline_pca[:, 0], baseline_pca[:, 1],
                          c=self.colors['baseline'], marker='o', s=50, alpha=0.6,
                          label=self._get_text('基准数据', 'Baseline Data'),
                          edgecolors='white', linewidth=0.5)
            
            # 绘制当前数据（三角）
            if current_labels is not None:
                for i, label in enumerate(unique_labels):
                    mask = current_labels == label
                    ax.scatter(current_pca[mask, 0], current_pca[mask, 1],
                              c=[colors_map[i]], marker='^', s=80, alpha=0.8,
                              label=f'{self._get_text("当前", "Current")} - {self._get_text("类别", "Class")} {label}',
                              edgecolors='black', linewidth=0.5)
            else:
                ax.scatter(current_pca[:, 0], current_pca[:, 1],
                          c=self.colors['current'], marker='^', s=80, alpha=0.8,
                          label=self._get_text('当前数据', 'Current Data'),
                          edgecolors='black', linewidth=0.5)
            
            # 设置标题和标签
            ax.set_title(title if self.use_chinese else 'PCA Feature Distribution Comparison',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} {self._get_text("方差", "Variance")})',
                         fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} {self._get_text("方差", "Variance")})',
                         fontsize=11)
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            return self._save_figure(fig, save_name)
            
        except Exception as e:
            logger.error(f"PCA散点图生成失败: {e}")
            return None
    
    def plot_drift_trend(self, mmd_history, window_labels=None,
                         threshold=0.05, save_name="drift_trend.png"):
        """
        漂移趋势折线图 - MMD 值随时间窗口变化
        
        Args:
            mmd_history: list[float] 各窗口的MMD值
            window_labels: 窗口标签列表 (可选)
            threshold: 漂移阈值
            save_name: 保存文件名
        
        Returns:
            str: 保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            ax.set_facecolor(self.colors['background'])
            
            x = np.arange(len(mmd_history))
            if window_labels is None:
                window_labels = [f'W{i+1}' for i in range(len(mmd_history))]
            
            # 绘制MMD趋势线
            ax.plot(x, mmd_history, 'o-', color=self.colors['baseline'],
                   linewidth=2, markersize=8, label='MMD Score',
                   markerfacecolor='white', markeredgewidth=2)
            
            # 标记漂移区域
            for i, mmd in enumerate(mmd_history):
                if mmd > threshold:
                    ax.scatter(i, mmd, c=self.colors['current'], s=150, zorder=5,
                              marker='X', edgecolors='black', linewidth=1,
                              label=self._get_text('漂移点', 'Drift Point') if i == 0 else "")
            
            # 绘制阈值线
            ax.axhline(y=threshold, color=self.colors['threshold'], linestyle='--',
                      linewidth=2, label=f'{self._get_text("阈值", "Threshold")} ({threshold})')
            
            # 填充漂移区域
            ax.fill_between(x, threshold, max(mmd_history) * 1.1,
                           where=[m > threshold for m in mmd_history],
                           alpha=0.2, color=self.colors['current'],
                           label=self._get_text('漂移区域', 'Drift Region'))
            
            # 设置标题和标签
            ax.set_title(self._get_text('MMD漂移趋势分析', 'MMD Drift Trend Analysis'),
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(self._get_text('时间窗口', 'Time Window'), fontsize=11)
            ax.set_ylabel(self._get_text('MMD 值', 'MMD Value'), fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(window_labels, rotation=45, ha='right')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            return self._save_figure(fig, save_name)
            
        except Exception as e:
            logger.error(f"漂移趋势图生成失败: {e}")
            return None
    
    def plot_class_drift_heatmap(self, per_class_results, 
                                  save_name="class_drift_heatmap.png"):
        """
        类别漂移热力图
        
        Args:
            per_class_results: dict {class_id: {mmd: float, is_drift: bool}}
            save_name: 保存文件名
        
        Returns:
            str: 保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.set_facecolor(self.colors['background'])
            
            # 提取数据
            classes = sorted([str(c) for c in per_class_results.keys()])
            mmd_values = [per_class_results[c]['mmd'] for c in classes]
            is_drift = [per_class_results[c]['is_drift'] for c in classes]
            
            # 创建热力图数据 (1行多列)
            data = np.array(mmd_values).reshape(1, -1)
            
            # 绘制热力图
            im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=max(mmd_values) * 1.2)
            
            # 设置坐标轴
            ax.set_xticks(np.arange(len(classes)))
            ax.set_xticklabels([f'{self._get_text("类别", "Class")} {c}' for c in classes],
                              fontsize=11)
            ax.set_yticks([0])
            ax.set_yticklabels(['MMD'], fontsize=11)
            
            # 添加数值标签
            for i, (mmd, drift) in enumerate(zip(mmd_values, is_drift)):
                color = 'white' if mmd > max(mmd_values) * 0.5 else 'black'
                text = f'{mmd:.3f}'
                if drift:
                    text += '\n[!]' if not self.use_chinese else '\n[漂移]'
                ax.text(i, 0, text, ha='center', va='center', color=color,
                       fontsize=12, fontweight='bold')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            cbar.set_label(self._get_text('MMD 值', 'MMD Value'), fontsize=10)
            
            # 设置标题
            ax.set_title(self._get_text('各类别漂移程度热力图', 'Per-Class Drift Heatmap'),
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return self._save_figure(fig, save_name)
            
        except Exception as e:
            logger.error(f"类别漂移热力图生成失败: {e}")
            return None
    
    def plot_feature_importance(self, feature_drift_details,
                                 save_name="feature_importance.png"):
        """
        特征维度重要性条形图
        显示漂移最显著的Top-20个维度的KS p-value和Cohen's d
        
        Args:
            feature_drift_details: list of dict [{'feature': str, 'pval': float, 'cohen_d': float}, ...]
            save_name: 保存文件名
        
        Returns:
            str: 保存的文件路径
        """
        try:
            # 按p值排序，选择最显著的Top-20
            sorted_features = sorted(feature_drift_details, key=lambda x: x['pval'])[:20]
            
            if not sorted_features:
                logger.warning("没有特征漂移数据")
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
            fig.patch.set_facecolor('white')
            
            features = [f['feature'].replace('dim_', 'D') for f in sorted_features]
            pvals = [f['pval'] for f in sorted_features]
            cohen_ds = [f['cohen_d'] for f in sorted_features]
            
            y_pos = np.arange(len(features))
            
            # 左图: p-value
            colors1 = [self.colors['current'] if p < 0.05 else self.colors['stable'] for p in pvals]
            bars1 = ax1.barh(y_pos, pvals, color=colors1, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.axvline(x=0.05, color=self.colors['threshold'], linestyle='--', linewidth=2,
                       label='α = 0.05')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features, fontsize=9)
            ax1.invert_yaxis()
            ax1.set_xlabel('KS p-value', fontsize=11)
            ax1.set_title(self._get_text('特征显著性 (p-value)', 'Feature Significance (p-value)'),
                         fontsize=12, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.set_xlim(0, 1)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # 右图: Cohen's d
            colors2 = [self.colors['current'] if abs(d) > 0.3 else self.colors['stable'] for d in cohen_ds]
            bars2 = ax2.barh(y_pos, cohen_ds, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax2.axvline(x=0.3, color=self.colors['threshold'], linestyle='--', linewidth=2)
            ax2.axvline(x=-0.3, color=self.colors['threshold'], linestyle='--', linewidth=2)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("Cohen's d", fontsize=11)
            ax2.set_title(self._get_text('效应量 (Cohen\'s d)', 'Effect Size (Cohen\'s d)'),
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            fig.suptitle(self._get_text('Top-20 漂移特征维度', 'Top-20 Drift Feature Dimensions'),
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            return self._save_figure(fig, save_name)
            
        except Exception as e:
            logger.error(f"特征重要性图生成失败: {e}")
            return None
    
    def plot_sample_anomaly_distribution(self, anomaly_scores,
                                          save_name="anomaly_distribution.png"):
        """
        样本异常分数分布直方图
        
        Args:
            anomaly_scores: list[float] 各样本的最近邻距离
            save_name: 保存文件名
        
        Returns:
            str: 保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.set_facecolor(self.colors['background'])
            
            scores = np.array(anomaly_scores)
            
            # 计算阈值 (95分位数)
            threshold = np.percentile(scores, 95)
            
            # 绘制直方图
            n, bins, patches = ax.hist(scores, bins=30, alpha=0.7, color=self.colors['baseline'],
                                      edgecolor='black', linewidth=0.5, label=self._get_text('样本分布', 'Sample Distribution'))
            
            # 标记异常区域
            for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
                if left_edge > threshold:
                    patch.set_facecolor(self.colors['current'])
                    patch.set_alpha(0.8)
            
            # 绘制阈值线
            ax.axvline(x=threshold, color=self.colors['threshold'], linestyle='--',
                      linewidth=2, label=f'{self._get_text("异常阈值 (95%)", "Anomaly Threshold (95%)")}: {threshold:.2f}')
            
            # 添加统计信息
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            ax.axvline(x=mean_score, color=self.colors['warning'], linestyle=':',
                      linewidth=2, label=f'{self._get_text("均值", "Mean")}: {mean_score:.2f}')
            
            # 设置标题和标签
            ax.set_title(self._get_text('样本异常分数分布', 'Sample Anomaly Score Distribution'),
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(self._get_text('最近邻距离 (异常分数)', 'Nearest Neighbor Distance (Anomaly Score)'),
                         fontsize=11)
            ax.set_ylabel(self._get_text('样本数量', 'Sample Count'), fontsize=11)
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加统计文本
            stats_text = f'{self._get_text("统计", "Statistics")}:\n'
            stats_text += f'{self._get_text("均值", "Mean")}: {mean_score:.2f}\n'
            stats_text += f'{self._get_text("标准差", "Std")}: {std_score:.2f}\n'
            stats_text += f'{self._get_text("最大值", "Max")}: {np.max(scores):.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return self._save_figure(fig, save_name)
            
        except Exception as e:
            logger.error(f"异常分布图生成失败: {e}")
            return None
    
    def plot_training_comparison(self, before_metrics, after_metrics,
                                  save_name="training_comparison.png"):
        """
        增量训练前后模型性能对比柱状图
        
        Args:
            before_metrics: dict {class: accuracy} 训练前各类别准确率
            after_metrics: dict {class: accuracy} 训练后各类别准确率
            save_name: 保存文件名
        
        Returns:
            str: 保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.set_facecolor(self.colors['background'])
            
            # 提取数据
            classes = sorted([str(c) for c in before_metrics.keys()])
            before_vals = [before_metrics[c] for c in classes]
            after_vals = [after_metrics[c] for c in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            # 绘制柱状图
            bars1 = ax.bar(x - width/2, before_vals, width, label=self._get_text('训练前', 'Before Training'),
                          color=self.colors['baseline'], alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, after_vals, width, label=self._get_text('训练后', 'After Training'),
                          color=self.colors['current'], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=9)
            
            # 添加改进箭头
            for i, (b, a) in enumerate(zip(before_vals, after_vals)):
                if a > b:
                    ax.annotate('', xy=(i + width/2, a), xytext=(i - width/2, b),
                               arrowprops=dict(arrowstyle='->', color=self.colors['stable'], lw=2))
            
            # 设置标题和标签
            ax.set_title(self._get_text('增量训练前后性能对比', 'Performance Comparison Before/After Training'),
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(self._get_text('类别', 'Class'), fontsize=11)
            ax.set_ylabel(self._get_text('准确率', 'Accuracy'), fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{self._get_text("类别", "Class")} {c}' for c in classes])
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            plt.tight_layout()
            return self._save_figure(fig, save_name)
            
        except Exception as e:
            logger.error(f"训练对比图生成失败: {e}")
            return None
    
    def generate_full_report(self, drift_result=None, drift_report_path=None,
                              output_dir=None):
        """
        一键生成全部图表
        从漂移检测结果或报告JSON中读取数据，生成所有图表
        同时生成一个汇总 HTML 文件
        
        Args:
            drift_result: dict 漂移检测结果字典 (可选)
            drift_report_path: str 漂移报告JSON文件路径 (可选)
            output_dir: str 输出目录 (可选)
        
        Returns:
            str: HTML报告路径
        """
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        report_data = None
        if drift_result:
            report_data = drift_result
        elif drift_report_path:
            try:
                with open(drift_report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
            except Exception as e:
                logger.error(f"加载报告失败: {e}")
                return None
        else:
            # 尝试加载默认报告
            default_path = os.path.join(BASELINE_ASSETS_DIR, 'drift_result.pkl')
            json_path = os.path.join(os.path.dirname(BASELINE_ASSETS_DIR), 'drift_report.json')
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                except Exception as e:
                    logger.error(f"加载默认报告失败: {e}")
                    return None
            elif os.path.exists(default_path):
                try:
                    with open(default_path, 'rb') as f:
                        report_data = pickle.load(f)
                except Exception as e:
                    logger.error(f"加载默认报告失败: {e}")
                    return None
            else:
                logger.error("未找到漂移报告数据")
                return None
        
        chart_paths = []
        
        try:
            # 1. 类别漂移热力图
            if 'per_class_drift' in report_data:
                path = self.plot_class_drift_heatmap(report_data['per_class_drift'])
                if path:
                    chart_paths.append(('class_drift_heatmap.png', 
                                       self._get_text('类别漂移热力图', 'Per-Class Drift Heatmap')))
            
            # 2. 特征重要性图
            if 'feature_level_drift' in report_data and 'details' in report_data['feature_level_drift']:
                path = self.plot_feature_importance(report_data['feature_level_drift']['details'])
                if path:
                    chart_paths.append(('feature_importance.png',
                                       self._get_text('特征重要性', 'Feature Importance')))
            
            # 3. 样本异常分布
            if 'sample_level_drift' in report_data:
                scores = [s['nn_dist'] for s in report_data['sample_level_drift']]
                path = self.plot_sample_anomaly_distribution(scores)
                if path:
                    chart_paths.append(('anomaly_distribution.png',
                                       self._get_text('异常分数分布', 'Anomaly Score Distribution')))
            
            # 4. 漂移趋势图 (如果有历史数据)
            if 'mmd_history' in report_data:
                path = self.plot_drift_trend(report_data['mmd_history'])
                if path:
                    chart_paths.append(('drift_trend.png',
                                       self._get_text('漂移趋势', 'Drift Trend')))
            
            # 生成HTML报告
            html_path = os.path.join(self.output_dir, 'drift_visualization_report.html')
            self._generate_html_report(chart_paths, report_data, html_path)
            
            logger.info(f"完整报告已生成: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"生成完整报告失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_html_report(self, chart_paths, drift_summary, output_path):
        """生成汇总HTML报告，内嵌所有图表"""
        try:
            # 提取关键信息
            mmd_score = drift_summary.get('statistics', {}).get('mmd_score', 'N/A')
            if mmd_score != 'N/A':
                mmd_score = f"{mmd_score:.4f}"
            
            is_drift = drift_summary.get('decision', {}).get('is_drift', False)
            status = drift_summary.get('decision', {}).get('status', 'UNKNOWN')
            
            # 生成图表HTML
            charts_html = ""
            for filename, title in chart_paths:
                chart_path = os.path.join(self.output_dir, filename)
                if os.path.exists(chart_path):
                    # 读取图片并转为base64
                    with open(chart_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    charts_html += f"""
                    <div class="chart-container">
                        <h3>{title}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{title}">
                    </div>
                    """
            
            # 构建HTML
            html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>漂移分析可视化报告</title>
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
        .status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .status.drift {{
            background-color: #e74c3c;
            color: white;
        }}
        .status.stable {{
            background-color: #2ecc71;
            color: white;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            margin-top: 0;
            color: #667eea;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .summary-item label {{
            display: block;
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .summary-item value {{
            display: block;
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container h3 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
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
        <h1>漂移分析可视化报告</h1>
        <p>Drift Analysis Visualization Report</p>
        <span class="status {'drift' if is_drift else 'stable'}">
            {'检测到漂移' if is_drift else '数据稳定'} | {status}
        </span>
    </div>
    
    <div class="summary">
        <h2>检测摘要</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <label>MMD 分数</label>
                <value>{mmd_score}</value>
            </div>
            <div class="summary-item">
                <label>漂移状态</label>
                <value>{'漂移' if is_drift else '稳定'}</value>
            </div>
            <div class="summary-item">
                <label>生成时间</label>
                <value>{drift_summary.get('meta', {}).get('generated_at', 'N/A')}</value>
            </div>
            <div class="summary-item">
                <label>基准样本数</label>
                <value>{drift_summary.get('data_info', {}).get('baseline_size', 'N/A')}</value>
            </div>
            <div class="summary-item">
                <label>测试样本数</label>
                <value>{drift_summary.get('data_info', {}).get('test_size', 'N/A')}</value>
            </div>
        </div>
    </div>
    
    {charts_html}
    
    <div class="footer">
        <p>由 DriftVisualizer 自动生成 | Generated by DriftVisualizer</p>
    </div>
</body>
</html>"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML报告已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"HTML报告生成失败: {e}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    try:
        visualizer = DriftVisualizer()
        
        # 尝试从默认路径生成报告
        report_path = os.path.join(os.path.dirname(BASELINE_ASSETS_DIR), 'drift_report.json')
        
        if os.path.exists(report_path):
            html_path = visualizer.generate_full_report(drift_report_path=report_path)
            if html_path:
                logger.info(f"可视化报告生成成功: {html_path}")
            else:
                logger.error("报告生成失败")
        else:
            logger.warning(f"未找到报告文件: {report_path}")
            logger.info("请提供 drift_result 字典或 drift_report_path 参数")
            
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
