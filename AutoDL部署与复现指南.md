# EdgeFallNet - AutoDL 算力云部署与复现指南

## 一、项目概述

**项目名称**: EdgeFallNet：面向智能手环的实时跌倒检测轻量化模型

**项目简介**: 基于 SqueezeNet 架构改进的轻量化 1D CNN 模型，用于智能手环加速度计和陀螺仪传感器数据的实时跌倒检测。总参数量仅 **3,731 个**，适合边缘设备部署。

**实验数据集**:
| 数据集 | 输入形状 | 测试准确率 | 灵敏度 | 特异度 | 几何均值 |
|--------|----------|------------|--------|--------|----------|
| FallAllD | (4760, 6) | 91.01% | 93.75% | 90.45% | 92.08% |
| UMAFALL | (300, 6) | 98.17% | 96.77% | 98.72% | 97.74% |

---

## 二、环境要求

### 2.1 推荐镜像（AutoDL 基础镜像）

在 AutoDL 创建实例时，选择以下镜像：

| 配置项 | 推荐值 |
|--------|--------|
| **镜像来源** | PyTorch 官方镜像（含 CUDA） |
| **镜像名称** | `pytorch:2.0.0-cuda11.7-cudnn8-devel-ubuntu22.04` 或类似 |
| **Python 版本** | Python 3.7 ~ 3.10（推荐 3.7/3.8） |
| **GPU** | 任选（RTX 3090 / RTX 4090 / V100 均可，本项目模型极轻量，CPU 也可运行） |
| **CUDA** | 11.7+ |
| **cuDNN** | 8.x |

> **说明**: 本项目使用 TensorFlow/Keras，但 AutoDL 的 PyTorch 镜像已包含完整 Python 环境，只需额外安装 TensorFlow 即可。也可直接选择 TensorFlow 官方镜像。

### 2.2 依赖包清单

创建 `requirements.txt` 文件：

```txt
tensorflow==2.10.0
numpy>=1.21.0,<2.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
PyYAML>=6.0
pickle5; python_version < "3.8"
```

### 2.3 安装命令

连接到 AutoDL 实例后，执行以下命令安装依赖：

```bash
# 进入工作目录
cd /root/autodl-tmp

# 上传或克隆项目代码
# 方式一：从本地上传项目文件夹 EdgeFallNet
# 方式二：如果代码在 Git 仓库中：
# git clone <your-repo-url>

# 创建虚拟环境（推荐）
python -m venv edgefallnet_env
source edgefallnet_env/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install tensorflow==2.10.0 numpy scikit-learn matplotlib PyYAML

# 验证安装
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU可用: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

---

## 三、项目文件结构

```
EdgeFallNet/
├── train.py                  # 主训练脚本（入口）
├── configs/
│   ├── fallld.yaml           # FallAllD 数据集配置
│   └── umafall.yaml          # UMAFALL 数据集配置
├── models/
│   ├── __init__.py            # 模型包初始化
│   └── model.py               # EdgeFallNet 模型定义（核心）
├── datasets/
│   ├── FallAllD/
│   │   ├── FallAllD_Data.pkl  # FallAllD 特征数据
│   │   └── FallAllD_lable.pkl # FallAllD 标签数据
│   ├── UMAFALL/
│   │   ├── UMAFALL_Data.pkl   # UMAFALL 特征数据
│   │   └── UMAFALL_lable.pkl  # UMAFALL 标签数据
│   ├── FallAllDDataset.py     # FallAllD 数据预处理脚本
│   └── UMAFALLDataset.py      # UMAFALL 数据预处理脚本
├── utils/
│   └── __init__.py            # 工具函数
├── saved_models/              # 训练好的模型权重
│   ├── EdgeFallNet_FallAllD.h5
│   └── EdgeFallNet_UMAFALL.h5
└── experiments/               # 实验输出目录（自动生成）
    ├── FallAllD_Experiment/
    │   ├── config.yaml
    │   ├── data/             # 数据划分信息
    │   ├── figures/          # 训练曲线图
    │   ├── logs/             # 训练日志和评估结果
    │   └── models/           # 保存的模型
    └── UMAFALL_Experiment/
        └── ...
```

---

## 四、详细复现步骤

### 步骤 1：上传项目代码到 AutoDL

1. 登录 [AutoDL](https://www.autodl.com)
2. 创建新实例（选择 GPU 实例，推荐 RTX 3090 24GB）
3. 通过 AutoDL 的「JupyterLab」或终端进入实例
4. 使用上传功能将整个 `EdgeFallNet` 文件夹上传至 `/root/autodl-tmp/EdgeFallNet/`

### 步骤 2：准备数据集

**重要**: 项目中的 `datasets/` 目录下已包含预处理好的 `.pkl` 数据文件。如果数据文件缺失，需要先运行数据预处理脚本生成：

#### FallAllD 数据集预处理

原始数据需包含以下 4 个 `.pkl` 文件（放在 `datasets/FallAllD/` 目录下）：
- `ACC_ADL.pkl` - ADL（日常活动）加速度数据
- `ACC_FALL.pkl` - 跌倒加速度数据
- `GCC_ADL.pkl` - ADL 陀螺仪数据
- `GCC_FALL.pkl` - 跌倒陀螺仪数据

运行预处理脚本生成训练用数据：
```bash
cd /root/autodl-tmp/EdgeFallNet/datasets
python FallAllDDataset.py
```

#### UMAFALL 数据集预处理

原始数据需包含以下 2 个 `.txt` 文件（放在 `datasets/UMAFALL/` 目录下）：
- `ADL.txt` - ADL 数据窗口（401 × 3 格式）
- `FALL.txt` - 跌倒数据窗口（401 × 3 格式）

运行预处理脚本：
```bash
cd /root/autodl-tmp/EdgeFallNet/datasets
python UMAFALLDataset.py
```

### 步骤 3：修改配置文件路径

由于 AutoDL 上的路径与本机不同，需要修改 YAML 配置文件中的数据路径：

**编辑 `configs/fallld.yaml`**：
```yaml
dataset:
  name: "FallAllD"
  data_path: "/root/autodl-tmp/EdgeFallNet/datasets/FallAllD/FallAllD_Data.pkl"
  label_path: "/root/autodl-tmp/EdgeFallNet/datasets/FallAllD/FallAllD_lable.pkl"
```

**编辑 `configs/umafall.yaml`**：
```yaml
dataset:
  name: "UMAFALL"
  data_path: "/root/autodl-tmp/EdgeFallNet/datasets/UMAFALL/UMAFALL_Data.pkl"
  label_path: "/root/autodl-tmp/EdgeFallNet/datasets/UMAFALL/UMAFALL_lable.pkl"
```

> **快速替换方法**（在终端执行）：
> ```bash
> cd /root/autodl-tmp/EdgeFallNet
> sed -i 's|F:/PycharmProjects/project1/EdgeFallNet|/root/autodl-tmp/EdgeFallNet|g' configs/fallld.yaml configs/umafall.yaml
> ```

### 步骤 4：训练 FallAllD 模型

```bash
cd /root/autodl-tmp/EdgeFallNet

# 训练 FallAllD 数据集
python train.py --config configs/fallld.yaml --gpu 0
```

**预期输出**:
```
开始跌倒检测模型训练...
默认配置文件: configs/fallld.yaml
默认使用GPU: True (ID: 0)
--------------------------------------------------
使用GPU: 0
实验目录: experiments/FallAllD_Experiment_YYYYMMDD_HHMMSS
============================================================
实验名称: FallAllD_Experiment
数据集: FallAllD
模型: EdgeFallNet_FallAllD
============================================================

训练日志
------------------------------------------------------------
Epoch     Loss         Acc          Val Loss     Val Acc     
------------------------------------------------------------
...
测试准确率: 0.9101
灵敏度: 0.9375
特异度: 0.9045
几何均值: 0.9208
```

### 步骤 5：训练 UMAFALL 模型

```bash
cd /root/autodl-tmp/EdgeFallNet

# 训练 UMAFALL 数据集
python train.py --config configs/umafall.yaml --gpu 0
```

**预期输出**:
```
测试准确率: 0.9817
灵敏度: 0.9677
特异度: 0.9872
几何均值: 0.9774
```

### 步骤 6：（可选）仅 CPU 运行

如果没有 GPU 或需要在 CPU 上运行：

```bash
python train.py --config configs/fallld.yaml --gpu -1
```

---

## 五、输出结果说明

训练完成后，结果保存在 `experiments/` 目录下：

```
experiments/
└── FallAllD_Experiment_时间戳/
    ├── config.yaml              # 本次实验的配置快照
    ├── data/
    │   ├── data_info.json       # 原始数据信息
    │   └── data_split.json      # 数据集划分详情
    ├── figures/
    │   └── training_history.png # 训练/验证 loss 和 accuracy 曲线
    ├── logs/
    │   ├── training.log         # 详细训练日志
    │   ├── training_details.txt # 每 epoch 的指标记录
    │   ├── model_summary.txt    # 模型结构摘要
    │   └── evaluation_results.txt # 最终评估结果（混淆矩阵、各项指标）
    └── models/
        ├── best_model.h5        # 验证集最优模型权重
        └── EdgeFallNet_FallAllD.h5 # 最终模型完整保存
```

---

## 六、模型架构核心参数

| 参数 | FallAllD | UMAFALL |
|------|----------|---------|
| 输入维度 | (4760, 6) | (300, 6) |
| 初始卷积滤波器数 | 16 | 8 |
| Fire1 squeeze/expand1x1/expand3x3 | 8/4/6 | 4/2/6 |
| Fire2 squeeze/expand1x1/expand3x3 | 8/4/6 | 4/2/6 |
| 全连接层单元数 | 8 | 8 |
| Dropout 率 | 0.1 | 0.1 |
| 总参数量 | **3,731** | ~**1,000** |
| 优化器 | Adamax (lr=0.05) | Adamax (lr=0.01) |
| 最大 Epochs | 120 | 120 |
| Batch Size | 32 | 32 |

---

## 七、常见问题排查

### Q1: 提示 "ModuleNotFoundError: No module named 'models'"
确保在 `EdgeFallNet` 项目根目录下运行 `train.py`，不要在其他目录运行。

### Q2: GPU 内存不足
本项目模型极轻量（~3700 参数），几乎不占用 GPU 显存。如遇此问题，可尝试：
```bash
python train.py --config configs/fallld.yaml --gpu -1  # 切换到 CPU
```

### Q3: TensorFlow 安装失败
AutoDL 上建议使用 pip 安装 CPU 版本或匹配 CUDA 版本的 GPU 版本：
```bash
# CUDA 11.x + TF 2.10
pip install tensorflow==2.10.0

# 如果遇到兼容性问题，可尝试
pip install tf-nightly  # 或使用 conda 安装
conda install tensorflow-gpu=2.10
```

### Q4: 数据文件找不到
检查 `configs/*.yaml` 中的 `data_path` 和 `label_path` 是否指向正确的绝对路径。

### Q5: 训练结果与论文不一致
- 确认随机种子设置（默认 random_state=42）
- 确认数据预处理步骤一致
- 不同硬件平台可能有微小浮点差异（<1% 属于正常范围）

---

## 八、一键运行脚本（可选）

创建 `run_all.sh` 一键完成两个数据集的训练：

```bash
#!/bin/bash
set -e

PROJECT_DIR="/root/autodl-tmp/EdgeFallNet"
cd "$PROJECT_DIR"

echo "==========================================="
echo "  EdgeFallNet 自动化训练脚本"
echo "==========================================="

echo "[1/2] 正在训练 FallAllD 数据集..."
python train.py --config configs/fallld.yaml --gpu 0

echo ""
echo "[2/2] 正在训练 UMAFALL 数据集..."
python train.py --config configs/umafall.yaml --gpu 0

echo ""
echo "==========================================="
echo "  全部训练完成！结果保存在 experiments/ 目录"
echo "==========================================="
```

运行方式：
```bash
chmod +x run_all.sh
./run_all.sh
```
