# RoseGrade - 自适应花卉智能分级系统

> 解决季节变化导致的花卉颜色分级精度下降问题

RoseGrade 是一个面向鲜切花卉的智能分级系统，通过**动态在线采样**、**数据漂移检测**和**自动模型更新**技术，实现对玫瑰花等鲜切花的精准自动化分级。

---

## 🎯 核心功能

| 功能 | 说明 |
|------|------|
|  **双路融合分级** | YOLO11 深度学习 + HSV 颜色分析，分级更准确 |
|  **数据漂移检测** | 自动检测季节/环境变化导致的颜色分布偏移 |
|  **自适应模型更新** | 检测到漂移后自动触发增量训练，保持分级精度 |
|  **季节适应** | 支持春/夏/秋/冬四季分级参数调整 |
|  **实时分级服务** | 提供 REST API，支持单张/批量图片分级 |

---

## 📋 工作流程

```
输入图片 → 特征提取 → 漂移检测 → 自动训练(如需) → 双路分级 → 输出结果
                                    ↓
                              数据分布稳定？
                            ↙              ↘
                          是                否
                          ↓                 ↓
                    直接分级          增量训练模型
```

### 五大核心步骤

#### Step 1: 基准数据收集
- 使用 YOLO11 提取训练集图像特征
- 生成基准数据库（PCA 降维）
- **命令**: `python main.py collect`

#### Step 2: 数据漂移检测
- **全局检测**: MMD 统计检验，判断整体分布是否变化
- **类别检测**: 识别哪些等级发生漂移
- **特征检测**: 找出异常的特征维度
- **样本检测**: 识别离群样本
- **命令**: `python main.py detect`

#### Step 3: 自适应采样
- 检测到漂移后，筛选最有价值的样本
- 新旧数据混合（防止遗忘历史知识）
- 减少训练成本

#### Step 4: 自动模型训练
- 增量微调（小学习率）
- EMA 模型融合（70%新模型 + 30%旧模型）
- 性能验证与自动更新
- **命令**: 检测到漂移后自动触发

#### Step 5: 质量分级
- **YOLO11 路径**: 深度学习分类（权重 60%）
- **HSV 路径**: 颜色特征分析（权重 40%）
- 双路融合输出最终等级
- **命令**: `python main.py grade --image flower.jpg`

---

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.8+ (推荐 3.10/3.11)
python --version

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

按以下结构组织花卉图片：

```
dataset/
├── train/              # 训练集
│   ├── 1/             # 等级1（鲜艳饱满）
│   ├── 2/             # 等级2（颜色鲜艳）
│   ├── 3/             # 等级3（颜色一般）
│   └── 4/             # 等级4（颜色暗淡）
└── val/               # 验证集
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

### 3. 运行系统

```bash
# ① 首次运行：收集基准数据
python main.py collect

# ② 检测数据漂移
python main.py detect

# ③ 分级单张图片
python main.py grade --image path/to/flower.jpg

# ④ 批量分级
python main.py grade --dir dataset/val/1

# ⑤ 启动 API 服务
python main.py serve --port 8080

# ⑥ 运行动态检测演示
python main.py demo

# ⑦ 生成可视化报告
python main.py visualize
```

---

## 💡 使用场景

### 场景 1: 单张图片分级

```bash
# 基础用法
python main.py grade --image rose.jpg

# 指定季节（影响颜色分级参数）
python main.py grade --image rose.jpg --season winter

# 输出示例
============================================================
📊 分级结果
============================================================
图片路径: rose.jpg
最终等级: 1 级
融合置信度: 0.9234
融合分数: 3.8500
------------------------------------------------------------
YOLO 模型结果:
  预测等级: 1 级
  置信度: 0.9500
  Top-5 预测: [1, 2, 3, 4]
------------------------------------------------------------
HSV 颜色分析结果:
  颜色等级: 1 级
  置信度: 0.8800
  主色调: 深红
  饱和度评分: 0.8200
  亮度评分: 0.7500
============================================================
```

### 场景 2: 批量分级

```bash
# 分级整个目录
python main.py grade --dir dataset/val/1

# 输出示例
============================================================
📊 批量分级统计
============================================================
总计处理: 100 张图片
成功: 98 张
失败: 2 张
------------------------------------------------------------
等级分布:
  等级 1:  45 张 ( 45.9%) █████████
  等级 2:  30 张 ( 30.6%) ██████
  等级 3:  15 张 ( 15.3%) ███
  等级 4:   8 张 (  8.2%) █
============================================================
```

### 场景 3: REST API 服务

```bash
# 启动服务
python main.py serve --port 8080

# 访问 API 文档（交互式测试）
http://localhost:8080/docs
```

**可用 API 端点：**
- `POST /api/grade_single` - 单张图片分级
- `POST /api/grade_batch` - 批量分级
- `GET /api/drift_status` - 查看漂移状态
- `POST /api/trigger_detection` - 手动触发检测
- `GET /api/model_info` - 获取模型信息

**使用 curl 测试：**
```bash
curl -X POST "http://localhost:8080/api/grade_single" \
     -F "file=@rose.jpg"
```

### 场景 4: 漂移检测与自适应训练

```bash
# 检测数据漂移
python main.py detect

# 输出示例
==================================================
🚨 检测到数据漂移！
   MMD 分数: 0.0823
   状态: DRIFT DETECTED
==================================================

# 运行动态演示（模拟完整流程）
python main.py demo --windows 4 --perturbation-window 2
```

---

## 🔧 高级用法

### 自定义漂移检测参数

```bash
# 自定义报告输出路径
python main.py detect --output my_report.json

# 演示时调整窗口数量和漂移阈值
python main.py demo \
  --windows 6 \
  --perturbation-window 3 \
  --window-size 50 \
  --drift-threshold 0.03
```

### Python 代码调用

```python
from src.quality_grader import QualityGrader

# 初始化分级器
grader = QualityGrader(season='spring')

# 单张图片分级
result = grader.grade_single(image_path="rose.jpg")
print(f"等级: {result['grade']}")
print(f"置信度: {result['confidence']}")

# 批量分级
results = grader.grade_batch(image_dir="dataset/val/1")
```

### 支持的图片格式

- ✅ JPEG/JPG
- ✅ PNG
- ✅ BMP
- ✅ WebP
- ✅ TIFF

---

## 📊 可视化报告

运行演示或检测后，可生成丰富的可视化报告：

```bash
python main.py visualize
```

**生成内容：**
-  漂移趋势图（MMD 分数变化）
-  类别漂移热力图
-  特征重要性图
-  样本异常分布图
-  HTML 交互式报告

报告保存在 `reports/` 目录。

---

##  项目结构

```
Adaptive_Flower_Grading_Project/
├── src/                          # 核心代码
│   ├── baseline_collector.py     # 基准数据收集
│   ├── drift_detector.py         # 漂移检测器
│   ├── auto_trainer.py           # 自动训练器
│   ├── quality_grader.py         # 分级服务（含API）
│   ├── color_grader.py           # HSV颜色分级
│   ├── online_sampler.py         # 在线采样器
│   ├── drift_visualizer.py       # 可视化模块
│   └── dynamic_detection_pipeline.py  # 演示流水线
├── main.py                       # 统一命令行入口
├── dataset/                      # 数据集
│   ├── train/                   # 训练集
│   └── val/                     # 验证集
├── baseline_assets/              # 基准数据（自动生成）
├── models/                       # 训练模型
├── reports/                      # 可视化报告
└── requirements.txt              # 依赖清单
```

---

##  技术原理

### 双路融合分级

```
图片 → YOLO11分类(60%) ─┐
                        ├→ 加权融合 → 最终等级
图片 → HSV颜色分析(40%) ─┘
```

### 漂移检测算法

| 层级 | 方法 | 作用 |
|------|------|------|
| 全局 | MMD + 排列检验 | 判断整体分布是否变化 |
| 类别 | 按类 MMD | 识别哪些等级发生漂移 |
| 特征 | KS 检验 | 找出异常特征维度 |
| 样本 | 最近邻距离 | 发现离群样本 |

### 自适应训练策略

1. **检测漂移**: MMD 分数 > 阈值
2. **采样数据**: 新样本 + 30%旧样本
3. **增量训练**: 小学习率微调
4. **模型融合**: 70%新 + 30%旧
5. **验证更新**: 性能提升则替换模型

---

##  环境要求

- **Python**: 3.8+ (推荐 3.10/3.11)
- **CUDA**: 11.0+ (可选，支持 CPU 运行)
- **内存**: 建议 8GB+
- **存储**: 建议 10GB+

---


