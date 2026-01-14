# 🏥 医学图像分割系统 - AI智能分析平台

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7--3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)
[![GitHub stars](https://img.shields.io/github/stars/1molchuan/Brain-MRI?style=social)](https://github.com/1molchuan/Brain-MRI)
[![许可证](https://img.shields.io/badge/许可证-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

## 概述

一个全面的医学图像分割系统，具有图形界面和API服务，支持多种先进的深度学习模型架构。该系统专为医学图像分割任务设计，特别针对脑肿瘤分割场景进行了优化。

[功能特性](#-主要特性) • [快速开始](#-快速开始) • [使用指南](#-使用指南) • [模型架构](#-支持的模型架构) • [常见问题](#-常见问题)

</div>

---

## 🆕 最新更新

- **架构重构（Architectural Refactoring）**  
  - 将原本超大、单文件的 `worker.py`（约 1.1 万行）和 `utils.py`（约 4.5 千行），重构为模块化包结构 `worker/` 与 `utils/`。  
  - 通过 `__init__.py` 保持向后兼容，原有 `from worker import ...` 和 `from utils import *` 写法仍然可用。  
  - 重构显著提升了代码的 **可维护性**（按职责拆分）、**可执行性**（更快加载、更少循环依赖），并解决了单文件在 IDE/代码审查中带来的性能瓶颈。  

- **基于 HD-BET 的数据优化与模型增强**  
  - 在 LGG-MRI 原始数据集上，引入 **HD-BET 深度学习颅骨剥离（Skull Stripping）** 流水线：自动移除颅骨、脂肪、皮肤等颅外组织，只保留颅内脑组织。  
  - 通过清理强背景噪声，使模型的注意力更集中于 **颅内肿瘤前景**，减少无意义的高亮区域。  
  - 在经过 HD-BET 预处理的高质量数据集上重新训练 DeepLabV3+，显著提升了：  
    - **收敛速度**（更稳定、更少震荡）  
    - **对肿瘤前景的分割鲁棒性**（边界更平滑、小病灶保留更好）  
  - 该高质量 DeepLabV3+ 模型也作为后续实验与集成的 **基础强模型（strong baseline）** 使用。  

---

## ✨ 主要特性

### 🎯 多模型架构支持
- **ImprovedUNet** - 改进的 U-Net 架构，集成注意力机制
- **ResNetUNet** - 基于 ResNet 编码器的 U-Net，支持预训练权重
- **TransUNet** - Transformer + U-Net 混合架构
- **DS-TransUNet** - 双尺度 Transformer U-Net，支持 GWO 优化
- **SwinUNet** - 基于 Swin Transformer 的 U-Net，支持 GWO 优化
- **U-Net++** - 基于 SMP 的密集连接 U-Net
- **DeepLabV3+** - 基于 SMP 的 DeepLabV3+，支持 Grad-CAM
- **NN-Former** - 基于 Transformer 的医学图像分割模型

### 🖥️ 友好的图形界面
- 基于 PyQt5 构建的现代化 GUI
- 实时训练监控和可视化
- 直观的操作流程

### 🚀 高性能训练
- ✅ 混合精度训练（AMP）
- ✅ 学习率调度（Poly/ReduceLROnPlateau）
- ✅ 早停机制
- ✅ EMA（指数移动平均）
- ✅ SWA（随机权重平均）
- ✅ 最佳模型自动保存
- ✅ CuDNN Benchmark 优化
- ✅ DataLoader 多进程优化

### 📊 智能阈值优化
- **GWO（灰狼优化算法）** - 智能搜索最佳分割阈值，替代传统线性扫描
- 自动寻找最优 Dice 系数对应的阈值
- 更快的搜索速度和更优的结果

### 📈 丰富的评估指标
- Dice、IoU、Precision、Recall、Specificity、HD95
- 空 Mask 特殊处理（双空=1.0，单空=0.0）
- 样本级指标统计

### 🔍 测试时增强 (TTA)
- 多尺度推理（0.8x, 1.0x, 1.2x）
- 8 种几何变换（翻转、旋转等）
- 可提升 1-3% 的 Dice 系数

### 💡 智能后处理
- **LCC（最大连通域）** - 保留最大连通区域，去除噪点
- **孔洞填充** - 填补小孔洞，提升分割完整性
- **边缘平滑** - Gaussian 滤波，修正锯齿边缘
- **形态学操作** - 开运算、闭运算，去除毛刺和填充缝隙
- **动态面积阈值** - 根据概率图平均值动态调整过滤阈值
- **高置信度小病灶保护** - 智能保留高置信度的微小病灶

### 📐 数据集支持
- **标准数据集** - 单通道医学图像
- **2.5D 数据集** - TCGA-LGG 格式（三通道堆叠）

### 🌐 API 服务
- RESTful API 模式
- 便于集成到其他系统

---

## 📋 系统要求

### 硬件要求
- **GPU**: 推荐 NVIDIA GPU（支持 CUDA），显存 ≥ 4GB（推荐 ≥ 8GB）
- **内存**: ≥ 8GB RAM（推荐 ≥ 16GB）
- **存储**: ≥ 10GB 可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.7 - 3.12
- **CUDA**: 11.0+（如果使用 GPU）
- **MATLAB**: R2020b+（可选，用于报告生成）

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/medical-segmentation.git
cd medical-segmentation
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**：
- `torch>=1.9.0` – PyTorch深度学习框架
- `PyQt5>=5.15.0` – GUI框架
- `albumentations>=1.1.0` – 数据增强
- `opencv-python>=4.5.0` – 图像处理
- `scikit-image>=0.18.0` – 图像工具
- `segmentation-models-pytorch` – SMP库（U‑Net++、DeepLabV3+）
- `pytorch-grad-cam` – Grad‑CAM可视化（可选）
- `matlab.engine` – MATLAB引擎（可选，用于报告生成）

### 3. 运行应用

#### GUI模式（推荐）

```bash
python main.py
```

#### API模式

```bash
python main.py --mode api --model path/to/model.pth --host 0.0.0.0 --port 8000
```

---

## 📁 项目结构

```
medical-segmentation/
├── main.py                    # 主入口（GUI和应用程序）
├── models.py                  # 所有模型架构
├── dataset.py                 # 2.5D数据集加载器（TCGA2_5DDataset）
├── config.py                  # 集中式模型配置
├── requirements.txt           # 依赖列表
├── README.md                  # 本文档
├── LICENSE                    # 许可证文件
│
├── utils/                     # 工具函数模块（模块化结构）
│   ├── __init__.py            # 向后兼容接口
│   ├── common.py              # 公共导入和配置
│   ├── helpers.py             # 基础工具函数（EarlyStopping等）
│   ├── window_ops.py           # 窗口操作函数
│   ├── image_augmentation.py  # 图像增强类
│   ├── model_loader.py        # 模型加载函数
│   ├── data_processing.py    # 数据处理函数
│   ├── standalone_funcs.py   # 独立函数（多进程）
│   ├── process_pool.py        # 进程池管理器
│   ├── multiprocess_helpers.py # 多进程辅助函数
│   ├── gwo_optimizer.py       # GWO优化器
│   ├── threshold_scan.py      # 阈值扫描
│   ├── dataset.py             # 数据集类（MedicalImageDataset）
│   ├── matlab_bridge.py       # MATLAB相关类
│   ├── visualization.py       # 可视化函数
│   └── README.md              # Utils模块说明
│
├── worker/                    # 工作线程模块（模块化结构）
│   ├── __init__.py            # 向后兼容接口
│   ├── common.py              # 公共导入和配置
│   ├── test_thread.py         # ModelTestThread（模型测试线程）
│   ├── train_thread.py        # TrainThread（训练线程）
│   ├── predict_thread.py      # PredictThread（预测线程）
│   └── README.md              # Worker模块说明
│
├── matlab_reports/            # 生成的MATLAB报告
└── data/                      # 数据目录（用户创建）
    ├── patient_id1/
    │   ├── image1.png
    │   ├── image1_mask.png
    │   └── ...
    └── ...
```

### 文件说明

- **main.py**: 主程序文件，包含 PyQt5 GUI 界面和应用程序入口，支持 MATLAB 引擎预热
- **models.py**: 包含所有模型架构的定义
- **utils/**: 工具函数模块，包含数据处理、模型加载、MATLAB 可视化桥接、GWO 优化器等
  - `helpers.py`: 基础工具函数（EarlyStopping、指标计算等）
  - `gwo_optimizer.py`: 灰狼优化算法阈值优化器
  - `model_loader.py`: 模型加载和参数推断
  - `dataset.py`: 医学图像数据集类
  - `standalone_funcs.py`: 后处理函数（LCC、孔洞填充等）
  - `matlab_bridge.py`: MATLAB 引擎会话和可视化桥接
  - 更多模块详见 `utils/README.md`
- **worker/**: 工作线程模块，包含训练、测试和预测的业务逻辑
  - `train_thread.py`: 训练线程（TrainThread）
  - `test_thread.py`: 测试线程（ModelTestThread）
  - `predict_thread.py`: 预测线程（PredictThread）
  - 更多详情详见 `worker/README.md`
- **dataset.py**: 2.5D 数据集加载器，支持 TCGA-LGG 格式的三通道堆叠输入
- **config.py**: 集中管理所有模型的配置参数

---

## 📖 使用指南

### 训练模型

#### 1. 准备数据

**标准数据集格式：**
```text
data_dir/
├── patient_id1/
│   ├── image1.png          # 原始图像
│   ├── image1_mask.png     # 对应的mask
│   ├── image2.png
│   └── image2_mask.png
├── patient_id2/
│   ├── image1.png
│   └── image1_mask.png
└── ...
```

**2.5D 数据集格式（TCGA-LGG）：**
```text
data_dir/
├── TCGA_CS_5393_19990606_1.tif
├── TCGA_CS_5393_19990606_2.tif
├── TCGA_CS_5393_19990606_3.tif
└── ...
```
系统会自动将相邻切片堆叠为三通道输入。

#### 2. 配置训练参数

在 GUI 界面中：
- 选择数据目录
- 选择数据集类型（标准 / 2.5D）
- 选择模型架构
- 设置训练轮次（Epochs）
- 设置批次大小（Batch Size）
- 选择优化器（Adam/AdamW/SGD）
- 启用/禁用 TTA、GWO 优化等

#### 3. 开始训练

- 点击"开始训练"按钮
- 实时查看训练进度和验证指标
- 训练完成后自动生成性能分析报告

### 测试模型

1. **加载模型**: 选择训练好的模型文件（.pth）
2. **选择测试数据**: 指定测试数据目录
3. **配置选项**:
   - 选择模型架构（或从 checkpoint 自动推断）
   - 启用/禁用 TTA
   - 使用 GWO 优化阈值（推荐）
4. **开始测试**: 点击"开始测试"按钮，查看详细性能指标和可视化结果

### 预测图像

1. **加载模型**: 选择训练好的模型文件
2. **选择图像**: 支持单张或批量图像预测
3. **设置阈值**: 调整二值化阈值（默认 0.5，或使用 GWO 优化结果）
4. **开始预测**: 点击"开始预测"按钮，查看预测结果

---

## 🏗️ 支持的模型架构

### 1. ImprovedUNet
- 改进的 U-Net 架构
- 集成注意力机制
- 适合小数据集训练

### 2. ResNetUNet
- 基于 ResNet 编码器的 U-Net
- 支持 ResNet50/101 预训练权重
- 集成 ASPP 和 CBAM 注意力

### 3. TransUNet
- Transformer + U-Net 混合架构
- 结合 CNN 和 Transformer 优势
- 适合复杂场景分割

### 4. DS-TransUNet
- 双尺度 Transformer U-Net
- 多尺度特征融合
- 提升边界分割精度
- **支持 GWO 超参数优化**

### 5. SwinUNet
- 基于 Swin Transformer 的 U-Net
- 支持窗口注意力机制
- **支持 GWO 超参数优化**

### 6. U-Net++ (SMP)
- 基于 segmentation-models-pytorch
- 支持多种编码器（ResNet、EfficientNet 等）
- ImageNet 预训练权重
- 支持 1/3 通道输入自适应

### 7. DeepLabV3+ (SMP)
- 基于 segmentation-models-pytorch
- ResNet101 编码器
- ImageNet 预训练权重
- 支持 Grad-CAM 可视化
- 支持 1/3 通道输入自适应

### 8. NN-Former
- 基于 Transformer 的医学图像分割模型
- **支持 GWO 超参数优化**

---

## 🎯 核心功能

### 训练功能
- ✅ 自动数据加载和预处理
- ✅ 多种数据增强策略（Albumentations）
- ✅ 混合精度训练（AMP）
- ✅ 学习率调度（Poly/ReduceLROnPlateau）
- ✅ 早停机制
- ✅ EMA（指数移动平均）
- ✅ SWA（随机权重平均）
- ✅ 最佳模型自动保存
- ✅ 实时训练监控
- ✅ CuDNN Benchmark 优化
- ✅ DataLoader 多进程优化

### 评估功能
- ✅ 多指标评估（Dice, IoU, Precision, Recall, Specificity, HD95）
- ✅ **GWO 阈值优化** - 智能搜索最佳阈值
- ✅ 性能分析报告生成（MATLAB）
- ✅ 低 Dice 案例识别
- ✅ 注意力热图可视化（Grad-CAM）
- ✅ 空 Mask 特殊处理（双空=1.0，单空=0.0）
- ✅ 样本级指标统计

### 预测功能
- ✅ 单张/批量图像预测
- ✅ 测试时增强（TTA）
- ✅ 智能后处理（LCC、孔洞填充、边缘平滑、空 Mask 优化）
- ✅ 结果可视化
- ✅ 批量导出

### 可视化功能
- ✅ 训练曲线实时显示
- ✅ 注意力热图（Grad-CAM）
- ✅ 预测结果网格可视化
- ✅ MATLAB 性能分析报告
- ✅ 混淆矩阵可视化

---

## 📊 性能指标

系统支持以下评估指标：

- **Dice 系数**: 衡量分割重叠度（只计算前景类，空 Mask 特殊处理）
- **IoU（交并比）**: 衡量预测与真实掩码的重叠（只计算前景类，空 Mask 特殊处理）
- **Precision（精确率）**: 预测为正样本中真正为正的比例
- **Recall（召回率）**: 真实正样本中被正确预测的比例
- **Specificity（特异度）**: 真实负样本中被正确预测的比例
- **HD95**: 95% Hausdorff 距离，衡量边界精度

### 空 Mask 处理策略

系统实现了完善的空 Mask（无病灶）处理逻辑：

- **双空（GT 为空且 Pred 为空）**: Dice=1.0, IoU=1.0（完美预测）
- **单空（GT 为空但 Pred 不为空）**: Dice=0.0, IoU=0.0（误报）
- **单空（GT 不为空但 Pred 为空）**: Dice=0.0, IoU=0.0（漏报）

后处理函数实现了"绝对最小面积限制"，确保微小噪点（面积 < min_size）被清空，从而触发"双空=1.0"的满分指标。

---

## 🔧 高级功能

### GWO 灰狼优化算法

**功能**: 使用灰狼优化算法（GWO）智能搜索最佳分割阈值，替代传统线性扫描

**优势**:
- 🚀 **更快的搜索速度**: 15 次迭代通常比线性扫描更快
- 🎯 **更优的结果**: 能够跳出局部最优，找到全局更优解
- 📊 **自动优化**: 无需手动设置阈值范围

**使用方法**:
- 在验证阶段自动启用
- 默认参数：`num_wolves=10`, `max_iter=15`
- 搜索空间：[0.05, 0.95]

**适用场景**:
- SwinUNet/DS-TransUNet/NN-Former 的超参数优化
- 阈值优化（替代线性扫描）

**实现位置**: `utils/gwo_optimizer.py`

### 测试时增强 (TTA)
- 多尺度推理（0.8x, 1.0x, 1.2x）
- 8 种几何变换（翻转、旋转等）
- 加权融合策略
- 可将Dice提升1‑3%

### 智能后处理
- **LCC（最大连通域）**: 保留最大连通区域，去除噪点
- **绝对最小面积限制**: 即使最大连通域，如果面积 < min_size，也会被清空（用于空 Mask 优化）
- **孔洞填充**: 填补小孔洞，提升分割完整性
- **边缘平滑**: Gaussian 滤波，修正锯齿边缘
- **形态学操作**: 开运算、闭运算，去除毛刺和填充缝隙
- **动态面积阈值**: 根据概率图平均值动态调整过滤阈值
- **高置信度小病灶保护**: 智能保留高置信度（>0.9）的微小病灶

**实现位置**: `utils/standalone_funcs.py` - `ensemble_post_process_global()` 和 `refine_segmentation_mask()`

### Grad‑CAM可视化
- 支持没有原生注意力图的模型（如DeepLabV3+）
- 使用解码器层作为目标层生成高分辨率热图
- 自动适配二分类/多分类模型
- 仅在验证/测试阶段生成（训练时禁用以节省GPU内存）
- 采样策略：仅处理前5个批次；其余跳过以提高速度。

### 2.5D 数据集支持
- 支持 TCGA-LGG 格式的 2.5D 数据集
- 自动将相邻切片堆叠为三通道输入
- 支持递归搜索子文件夹
- 优雅处理边界缺失切片。

### MATLAB报告生成
- 自动生成性能分析报告（条形图 + 误差条）
- 预测结果网格可视化
- 高质量图表导出（1200×800，300 DPI）
- 优化布局（为X轴标签预留空间）
- 持久保存到 `matlab_reports/` 目录。

**实现位置**: `utils/matlab_bridge.py`

### 性能优化
- **CuDNN Benchmark**: 自动寻找最适合的卷积算法
- **DataLoader 优化**:
  - `num_workers`: 自动设置为 `min(os.cpu_count(), 8)`
  - `pin_memory`: CUDA 设备自动启用
  - `persistent_workers`: 保持子进程存活，避免重复创建
  - `prefetch_factor`: 增加预取因子，提升数据流水线效率
- **内存优化**: 测试阶段仅收集前 5 个样本用于可视化，其余立即释放
- **梯度优化**: 验证阶段仅对前 5 个 batch 启用梯度计算（Grad-CAM），其余使用 `torch.no_grad()`
- **多进程后处理**: 使用 `ProcessPoolExecutor` 并行化 CPU 密集型任务

---

## 🎓 损失函数配置

系统采用**极简且稳健的损失函数配置**：

### 损失函数组合
- **BCE Loss**: 50%
- **Dice Loss**: 50%

这是医学分割的黄金标准组合，经过优化以：
- ✅ 有效约束假阳性
- ✅ 避免过度自信
- ✅ 提升空 Mask 场景下的性能

### 优化器配置
- **默认**: AdamW（weight_decay=5e-4）
- **支持**: Adam、SGD
- **学习率**: Encoder 和 Decoder 使用相同学习率（避免训练初期震荡）

---

## ❓ 常见问题

### Q: 如何选择最适合的模型架构？
**A**: 
- **小数据集**: 推荐 ImprovedUNet 或 ResNetUNet
- **中等数据集**: 推荐 TransUNet、U-Net++ 或 DeepLabV3+
- **大数据集**: 推荐 SwinUNet（支持 GWO 优化）
- **需要高精度**: 推荐 DeepLabV3+（训练稳定，性能优秀）

### Q: 如何提高模型性能？
**A**: 
1. 增加训练数据量
2. 使用数据增强
3. 启用 TTA（测试时）
4. 调整学习率和训练轮次
5. 使用 EMA/SWA 技术
6. 选择合适的模型架构
7. 使用智能后处理（LCC、孔洞填充等）
8. **使用 GWO 优化阈值**（推荐）

### Q: 训练时CUDA内存不足？
**A**: 尝试减小批次大小、降低图像分辨率、禁用混合精度训练、使用CPU训练或启用梯度累积。

### Q: TTA会显著增加推理时间吗？
**A**: 是的，TTA会使推理时间增加约24倍，但可将Dice提升1‑3%。建议仅用于最终评估。

### Q: 如何处理空掩码（无病灶）？
**A**: 系统具有全面的空掩码处理机制：
- 指标计算：两者皆空返回1.0，单边空返回0.0。
- 后处理：微小噪声（面积 < `min_size`）自动清除。
- 确保在空GT场景下，小假阳性不影响Dice分数。

### Q: 如何为DeepLabV3+生成注意力热图？
**A**: DeepLabV3+不支持原生注意力图；系统使用Grad‑CAM。安装 `pytorch-grad-cam`：
```bash
pip install grad-cam
```

### Q: MATLAB报告生成失败？
**A**:
1. 确保已安装MATLAB R2020b+。
2. 安装MATLAB Engine for Python：
   ```bash
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```
3. 检查MATLAB路径配置（`main.py`中的`MATLAB_BIN_PATH`）。

### Q: GWO 优化器如何使用？
**A**: GWO 优化器在验证阶段自动启用，用于：
- 智能搜索最佳分割阈值（替代线性扫描）
- SwinUNet/DS-TransUNet/NN-Former 的超参数优化
- 默认参数已优化，无需手动配置

---

## 📝 开发说明

### 代码架构

项目采用**模块化设计**，将功能拆分为多个主要模块：

#### 核心模块

- **main.py**: GUI 界面和应用程序主入口，包含 MATLAB 引擎预热逻辑
- **models.py**: 所有模型架构的定义
- **config.py**: 模型配置中心
- **dataset.py**: 2.5D 数据集加载器（TCGA2_5DDataset）

#### Utils 模块 (`utils/`)

工具函数模块，包含14个子模块：

- **common.py**: 公共导入和配置
- **helpers.py**: 基础工具函数（EarlyStopping、指标计算等）
- **gwo_optimizer.py**: 灰狼优化算法阈值优化器
- **model_loader.py**: 模型加载和参数推断
- **dataset.py**: 医学图像数据集类（MedicalImageDataset）
- **standalone_funcs.py**: 后处理函数（LCC、孔洞填充、边缘平滑等）
- **matlab_bridge.py**: MATLAB 引擎会话和可视化桥接
- **threshold_scan.py**: 阈值扫描功能
- **visualization.py**: 可视化函数
- 更多模块详见 `utils/README.md`

**向后兼容**: 所有原有导入方式仍然有效
```python
from utils import *  # 仍然有效
from utils import EarlyStopping, GreyWolfThresholdOptimizer  # 仍然有效
```

#### Worker 模块 (`worker/`)

工作线程模块，包含训练、测试和预测的业务逻辑：

- **train_thread.py**: 训练线程（TrainThread）- 约8650行
- **test_thread.py**: 测试线程（ModelTestThread）- 约2000行
- **predict_thread.py**: 预测线程（PredictThread）- 约200行
- **common.py**: 公共导入和配置

**向后兼容**: 所有原有导入方式仍然有效
```python
from worker import TrainThread, ModelTestThread, PredictThread  # 仍然有效
```

更多详情详见 `worker/README.md`

### 关键设计决策

1. **模块化结构**: 将大型文件（`utils.py` 4548行、`worker.py` 11013行）拆分为模块化结构，提高可维护性
2. **向后兼容**: 保持所有原有导入方式不变，无需修改现有代码
3. **通道自适应**: DeepLabV3+ 和 U-Net++ 支持 1/3 通道输入自适应，通过数据加载器自动转换
4. **空 Mask 优化**: 实现了完善的空 Mask 处理逻辑，确保指标计算的准确性
5. **性能优化**: DataLoader 多进程、CuDNN Benchmark、内存优化、多进程后处理等
6. **可视化优化**: Grad-CAM 采样策略、MATLAB 报告优化等
7. **损失函数简化**: 采用 50% BCE + 50% Dice 的黄金标准组合
8. **GWO 优化**: 智能阈值搜索，提升验证效率

### 模块依赖关系

```
main.py
├── models.py
├── utils/ (通过 __init__.py 统一导出)
│   ├── common.py (公共导入)
│   ├── helpers.py
│   ├── gwo_optimizer.py
│   ├── model_loader.py
│   └── ... (其他模块)
└── worker/ (通过 __init__.py 统一导出)
    ├── common.py (公共导入)
    ├── train_thread.py
    ├── test_thread.py
    └── predict_thread.py
```

---

## 📝 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

---

## 🤝 贡献

欢迎提交Issue和Pull Request！请确保您的更改符合项目的编码风格，并包含适当的测试。

### 贡献指南
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📧 联系方式

如有问题或建议，请：
- 提交GitHub Issue
- 发送邮件至：chuan2410450745@sjtu.edu.cn

---

## 🙏 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
- [Albumentations](https://albumentations.ai/) - 数据增强库
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) - 分割模型库
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) - Transformer 架构
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) - 可视化工具

---

## 📚 数据集

模型训练使用了：
- [LGG MRI分割数据集](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## 📈 更新日志

### v2.3 (最新)
- ✅ **HD-BET 深度学习颅骨剥离**：在 LGG-MRI 原始数据上引入 HD-BET 预处理流程，自动移除颅外组织（颅骨、脂肪、皮肤），构建高质量脑实质掩膜。  
- ✅ **数据质量与注意力提升**：在 Skull Stripping 后的数据集上训练，使模型注意力更专注于颅内病灶区域，显著减少背景伪激活。  
- ✅ **DeepLabV3+ 模型增强**：在优化后的数据上重新训练 DeepLabV3+，加快收敛速度并提升对肿瘤前景的分割鲁棒性，形成更强的基础模型。  
- ✅ **与现有流水线集成**：HD-BET 预处理与现有 DataLoader、后处理和 GWO 阈值优化无缝结合，可作为统一的高质量输入源。  

### v2.2
- ✅ **模块化重构** - 将 `utils.py` (4548行) 拆分为 `utils/` 目录（14个模块）
- ✅ **模块化重构** - 将 `worker.py` (11013行) 拆分为 `worker/` 目录（4个模块）
- ✅ **向后兼容** - 保持所有原有导入方式不变
- ✅ **性能优化** - 多进程后处理和指标计算
- ✅ **代码组织** - 提高可维护性和协作友好性

### v2.1
- ✅ **新增 GWO 灰狼优化算法** - 智能阈值搜索，替代线性扫描
- ✅ **简化损失函数配置** - 回归 50% BCE + 50% Dice 的黄金标准
- ✅ **优化学习率配置** - Encoder/Decoder 使用相同学习率
- ✅ **移除 pos_weight 放大** - 减少空 Mask 假阳性
- ✅ **优化后处理策略** - 高置信度小病灶保护

### v2.0
- ✅ 新增 DeepLabV3+ 和 U-Net++ 模型支持
- ✅ 集成 Grad-CAM 可视化
- ✅ 优化空 Mask 处理逻辑
- ✅ 实现智能后处理（绝对最小面积限制）
- ✅ 优化 DataLoader 性能（多进程、pin_memory、persistent_workers）
- ✅ 启用 CuDNN Benchmark
- ✅ 优化 MATLAB 报告生成（高清图表、优化布局）
- ✅ 实现阈值扫描功能
- ✅ 内存优化（测试阶段仅收集少量样本）
- ✅ 支持 2.5D 数据集（TCGA-LGG）

### v1.0
- ✅ 基础模型架构（ImprovedUNet、ResNetUNet、TransUNet、DS‑TransUNet、SwinUNet）
- ✅ GUI界面
- ✅ 训练/测试/预测功能
- ✅ TTA支持
- ✅ 基础后处理

---

<div align="center">

⭐ 如果这个项目对你有帮助，请给个 Star！

Made with ❤️ for Medical Image Segmentation

</div>
