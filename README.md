# 🏥 脑肿瘤分割系统 - AI 医学影像智能分析平台

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7--3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)
![Stars](https://img.shields.io/github/stars/1molchuan/Brain-MRI?style=social)

**🚀 专业的医学图像分割系统 | 🧠 7+ 深度学习模型 | 🌐 现代化 Web 界面 | 🤖 AI 辅助诊断**

[✨ 特性](#-核心特性) • [🚀 快速开始](#-快速开始) • [📖 文档](#-使用指南) • [🔧 API](#-api-文档) • [❓ FAQ](#-常见问题)

</div>

---

## 🎯 项目亮点

> 💡 **为什么选择本系统？**

- ✅ **三端全覆盖** - Web 界面、桌面应用、RESTful API 三合一
- ✅ **7+ 先进模型** - ImprovedUNet、ResNetUNet、TransUNet、DS-TransUNet、SwinUNet、U-Net++、DeepLabV3+
- ✅ **极速阈值优化** - Brent 算法 + 多进程并行，30 秒内完成（vs 传统方法数分钟）
- ✅ **智能后处理** - 数据驱动的自适应后处理策略，自动搜索最优配置
- ✅ **AI 辅助诊断** - 集成 LLM（OpenAI/DeepSeek/Moonshot），自动生成诊断报告
- ✅ **生产就绪** - Docker 支持、API 文档完善、性能优化到位

## 📋 目录

- [核心特性](#-核心特性)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [模型架构](#-支持的模型架构)
- [API 文档](#-api-文档)
- [常见问题](#-常见问题)
- [项目结构](#-项目结构)
- [更新日志](#-更新日志)
- [贡献指南](#-贡献)

## ✨ 核心特性

### 🎨 三端部署方案

| 方式 | 适用场景 | 主要功能 |
|------|---------|----------|
| 🌐 **Web 应用** (`app.py`) | 快速演示、批量处理 | 批量上传、实时调节、AI 诊断、可视化分析 |
| 🖥️ **桌面应用** (`main.py`) | 本地开发、完整工作流 | 模型训练、测试评估、批量预测、性能分析 |
| 🔌 **API 服务** (`server.py`) | 生产部署、系统集成 | RESTful API、模型推理、AI 聊天接口 |

### 🧠 深度学习模型

<details>
<summary><b>点击展开查看 7+ 种支持的模型架构</b></summary>

- **ImprovedUNet** - 改进的 U-Net，集成注意力机制，适合小数据集
- **ResNetUNet** - ResNet 编码器 + U-Net，支持预训练权重
- **TransUNet** - Transformer + U-Net 混合架构
- **DS-TransUNet** - 双尺度 Transformer，支持 GWO 优化
- **SwinUNet** - Swin Transformer + U-Net，支持 GWO 优化
- **U-Net++** (SMP) - 多编码器支持，ImageNet 预训练
- **DeepLabV3+** (SMP) - 高精度分割，推荐用于生产环境

</details>

### 🚀 核心优势

| 功能 | 传统方法 | 本系统 |
|------|---------|--------|
| ⚡ **阈值优化** | 网格搜索，数分钟 | Brent 算法 + 多进程，**30 秒** |
| 🎯 **后处理** | 手动调参 | 数据驱动自动搜索 |
| 🤖 **AI 诊断** | 无 | 集成 LLM，自动生成报告 |
| 📊 **可视化** | 基础图表 | 概率热力图、实时调试面板 |
| 🔧 **部署** | 单一方式 | Web/桌面/API 三端全覆盖 |

### 🔬 智能处理流程

```mermaid
graph LR
    A[图像输入] --> B[模型推理]
    B --> C[Brent 阈值优化]
    C --> D[智能后处理]
    D --> E[结果输出]
    E --> F[AI 诊断报告]
```

**智能后处理策略**
- ✅ LCC（最大连通域）- 去除噪点
- ✅ Remove-Small - 移除小区域
- ✅ 孔洞填充 - 提升完整性
- ✅ 边缘平滑 - 修正锯齿
- ✅ 形态学操作 - 开运算/闭运算
- ✅ 动态阈值 - 自适应调整
- ✅ 高置信度保护 - 保留微小病灶

---

## 📋 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|----------|
| **GPU** | NVIDIA GPU (4GB+) | NVIDIA GPU (8GB+) |
| **内存** | 8GB RAM | 16GB+ RAM |
| **存储** | 10GB | 20GB+ |
| **系统** | Windows 10/Linux/macOS | Windows 11/Ubuntu 20.04+ |
| **Python** | 3.7 | 3.9 - 3.11 |
| **CUDA** | 11.0+ | 11.8+ |

> 💡 **可选**: MATLAB R2020b+（用于高级性能分析报告）

---

## 🚀 快速开始

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/1molchuan/Brain-MRI.git
cd Brain-MRI

# 2. 安装依赖
pip install -r requirements.txt

# 3. 选择启动方式

# 方式 A: Web 应用（推荐新手）
streamlit run app.py

# 方式 B: API 服务（推荐生产环境）
python server.py

# 方式 C: 桌面应用（完整功能）
python main.py
```

### 快速验证

启动 Web 应用后，在浏览器访问 `http://localhost:8501`，按以下步骤测试：

1. **上传图像** - 批量上传 TIF 格式图像（自动配对 `_mask.tif`）
2. **选择模式** - 在侧边栏选择 "API 服务" 或 "本地调试"
3. **查看结果** - 使用滑块浏览切片，实时调节阈值
4. **AI 诊断** - 配置 AI 服务（可选），自动生成诊断报告

> 💡 **提示**: 首次运行需要下载模型权重，请确保网络连接正常

---

## 📖 使用指南

<details>
<summary><b>🌐 Web 应用详细教程</b></summary>

### 基础流程
1. **启动服务** - `streamlit run app.py`
2. **配置模式** - 侧边栏选择 "API 服务" 或 "本地调试"
3. **上传图像** - 批量上传 TIF 图像（自动配对 mask）
4. **调节参数** - 实时调整阈值，查看效果
5. **AI 诊断** - 配置 LLM 服务，生成报告（可选）

### 高级功能
- **智能后处理** - 上传 `best_postprocessing_config.json`
- **深度调试** - 查看概率分布直方图和阈值预览
- **批量分析** - 切片浏览器快速查看所有结果

</details>

<details>
<summary><b>🔌 API 服务使用指南</b></summary>

### 启动服务
```bash
python server.py  # 默认端口 8000
```

### 配置环境变量（AI 功能）
```bash
export OPENAI_API_KEY="your_key"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"
```

### 调用示例
```python
import requests

# 图像分割推理
with open("brain.tif", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/predict?return_prob_map=true",
        files={"file": f}
    )
    result = response.json()

# AI 诊断聊天
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "messages": [{"role": "user", "content": "请分析这张影像"}],
        "context_data": {"文件名": "brain.tif"},
        "llm_config": {
            "api_key": "your_key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        }
    },
    stream=True
)
```

</details>

<details>
<summary><b>🖥️ 桌面应用使用指南</b></summary>

### 训练工作流
1. **选择数据** - 指定训练/验证数据目录
2. **配置模型** - 选择架构（推荐：DeepLabV3+）
3. **设置参数** - 轮次、批次大小、学习率等
4. **开始训练** - 实时监控 Loss 和 Dice 曲线
5. **自动保存** - 最佳模型自动保存为 `best_model.pth`

### 测试评估
1. **加载模型** - 选择训练好的 `.pth` 文件
2. **选择数据** - 指定测试数据目录
3. **配置选项** - 启用 TTA、Grad-CAM 等
4. **开始测试** - 自动优化阈值（Brent 算法）
5. **查看报告** - MATLAB 性能分析报告（如果可用）

### 批量预测
1. **加载模型** - 选择 `.pth` 文件
2. **添加图像** - 批量导入待预测图像
3. **配置后处理** - 上传 `best_postprocessing_config.json`（可选）
4. **开始预测** - 批量处理并导出结果

</details>

---

## 📡 API 文档

### 基础信息

- **Base URL**: `http://127.0.0.1:8000`
- **API 版本**: `1.0.0`

### 端点列表

#### 1. GET `/`

获取 API 信息

**响应示例：**
```json
{
  "message": "Brain Tumor Segmentation API",
  "status": "running",
  "device": "cuda",
  "model_mode": "2.5D"
}
```

#### 2. GET `/health`

健康检查

**响应示例：**
```json
{
  "status": "healthy",
  "device": "cuda"
}
```

#### 3. POST `/predict`

图像分割推理

**请求参数：**
- `file` (File, required): 上传的图像文件（TIF/PNG/JPG）
- `return_prob_map` (bool, optional): 是否返回概率图，默认 `false`

**响应格式：**

如果 `return_prob_map=false`（默认）：
- Content-Type: `image/png`
- 返回二值化 Mask 图像（PNG 格式，0-255）

如果 `return_prob_map=true`：
- Content-Type: `application/json`
- 响应体：
```json
{
  "prob_map_shape": [256, 256],
  "prob_map_data": "base64_encoded_data",
  "prob_map_dtype": "float32",
  "prob_map_min": 0.0,
  "prob_map_max": 1.0,
  "prob_map_mean": 0.1234,
  "binary_mask": "base64_encoded_png",
  "threshold": 0.5
}
```

**使用示例：**
```python
import requests

# 读取图像文件
with open("image.tif", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://127.0.0.1:8000/predict?return_prob_map=true",
        files=files
    )

if response.status_code == 200:
    data = response.json()
    # 解码概率图
    import base64
    import numpy as np
    prob_map_bytes = base64.b64decode(data["prob_map_data"])
    prob_map = np.frombuffer(prob_map_bytes, dtype=np.float32).reshape(data["prob_map_shape"])
```

#### 4. POST `/chat`

AI 辅助诊断聊天接口

**请求体：**
```json
{
  "messages": [
    {"role": "user", "content": "请分析这张影像"}
  ],
  "context_data": {
    "文件名": "image1.tif",
    "肿瘤像素数": "1234",
    "总像素数": "65536",
    "肿瘤占比": "1.88%",
    "图像尺寸": "256x256",
    "Dice系数": "0.8276"
  },
  "llm_config": {
    "api_key": "your_api_key",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**响应格式：**
- Content-Type: `text/event-stream` (Server-Sent Events)
- 流式返回 AI 回复内容

**使用示例：**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "messages": [{"role": "user", "content": "请分析这张影像"}],
        "context_data": {"文件名": "image1.tif", "肿瘤像素数": "1234"},
        "llm_config": {
            "api_key": "your_api_key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        }
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk:
        print(chunk, end="", flush=True)
```

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
- **推荐用于高精度分割任务**

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
- ✅ **Brent 阈值优化 + 多进程并行** - 高效搜索最佳阈值（~15次评估，8核并行，30秒内完成）
- ✅ **MATLAB 性能分析报告生成**（可选，需要 MATLAB R2020b+）
  - 性能分析柱状图：显示全局指标（Dice、IoU、Precision、Recall）的均值和标准差
  - 测试结果可视化：多样本对比展示（原图、GT、预测）
  - 训练历史曲线：Loss 和 Dice 变化趋势
  - 预测网格可视化：批量预测结果展示
- ✅ 低 Dice 案例识别
- ✅ 注意力热图可视化（Grad-CAM，测试时可用，训练过程中已禁用以提升速度）
- ✅ 空 Mask 特殊处理（双空=1.0，单空=0.0）
- ✅ 样本级指标统计
- ✅ **全局指标计算**：基于所有样本（包括空mask）的全局指标，确保与日志一致

### 预测功能
- ✅ 单张/批量图像预测
- ✅ 测试时增强（TTA）
- ✅ 智能后处理（LCC、孔洞填充、边缘平滑、空 Mask 优化）
- ✅ 结果可视化
- ✅ 批量导出

### 智能后处理
- ✅ **LCC（最大连通域）** - 保留最大连通区域，去除噪点
- ✅ **Remove-Small** - 移除小区域（可配置最小面积阈值）
- ✅ **孔洞填充** - 填补小孔洞，提升分割完整性
- ✅ **边缘平滑** - Gaussian 滤波，修正锯齿边缘
- ✅ **形态学操作** - 开运算、闭运算，去除毛刺和填充缝隙
- ✅ **动态面积阈值** - 根据概率图平均值动态调整过滤阈值
- ✅ **高置信度小病灶保护** - 智能保留高置信度的微小病灶
- ✅ **数据驱动自适应后处理** - 基于验证集自动搜索最优策略

### AI 辅助诊断
- ✅ **侧边栏配置** - 在 Web App 中直接配置 LLM 服务
- ✅ **自动上下文提取** - 自动提取预测结果元数据
- ✅ **流式对话** - 实时显示 AI 回复
- ✅ **多 LLM 支持** - 支持 OpenAI、DeepSeek、Moonshot 等
- ✅ **动态配置** - 支持在请求中动态指定 LLM 配置

---

## ❓ 常见问题

### Q: 如何选择最适合的模型架构？
**A**: 
- **小数据集**: 推荐 ImprovedUNet 或 ResNetUNet
- **中等数据集**: 推荐 TransUNet、U-Net++ 或 DeepLabV3+
- **大数据集**: 推荐 SwinUNet（支持 GWO 优化）
- **需要高精度**: 推荐 DeepLabV3+（训练稳定，性能优秀）

### Q: Streamlit Web App 和 PyQt5 GUI 有什么区别？
**A**: 
- **Streamlit Web App** (`app.py`): Web 界面，专注于推理和可视化，支持批量图像处理、AI 辅助诊断，适合快速测试和演示
- **PyQt5 GUI** (`main.py`): 桌面应用，功能完整，支持训练、测试、预测，适合本地开发和完整工作流
- 两者共享相同的模型和后处理逻辑，确保结果一致性

### Q: 如何使用 AI 辅助诊断功能？
**A**: 
1. 启动 FastAPI 后端服务：`python server.py`
2. 启动 Streamlit Web App：`streamlit run app.py`
3. 在 Web App 左侧边栏的"🤖 AI 服务配置"中：
   - 填写 API 服务地址（默认：`http://127.0.0.1:8000`）
   - 填写 LLM API Key
   - 选择 LLM Base URL 和模型名称
4. 完成图像预测后，系统会自动提取元数据并发送给 AI，生成诊断报告

### Q: API 服务如何配置 LLM？
**A**: 
有两种方式：
1. **环境变量**（推荐用于生产环境）：
   ```bash
   export OPENAI_API_KEY=your_api_key
   export LLM_BASE_URL=https://api.deepseek.com/v1
   export LLM_MODEL=deepseek-chat
   ```
2. **请求中动态指定**（推荐用于开发/测试）：
   在 `/chat` 请求的 `llm_config` 字段中提供配置

### Q: 如何提高模型性能？
**A**: 
1. 增加训练数据量
2. 使用数据增强
3. 启用 TTA（测试时）
4. 调整学习率和训练轮次
5. 使用 EMA/SWA 技术
6. 选择合适的模型架构
7. 使用智能后处理（LCC、孔洞填充等）
8. **使用 Brent 方法优化阈值**（自动启用，多进程并行加速）

### Q: 训练时 CUDA 内存不足？
**A**: 尝试减小批次大小、降低图像分辨率、禁用混合精度训练、使用 CPU 训练或启用梯度累积。系统已优化训练流程，禁用训练过程中的注意力热力图生成以节省显存。

### Q: MATLAB 性能分析报告如何启用？
**A**: 
1. 确保已安装 MATLAB R2020b+ 和 Python MATLAB Engine
2. 在 PyQt5 GUI 中，MATLAB 功能会自动检测并启用
3. 如果 MATLAB 不可用，系统会自动使用 Matplotlib 绘图
4. 性能分析报告保存在 `worker/matlab_reports/` 目录下

### Q: 性能分析图表显示的 Dice 值不正确？
**A**: 已修复！现在系统使用全局指标（基于所有样本，包括空mask），与日志中的 "Mean Dice (全样)" 保持一致。如果仍有问题，请检查日志中的调试信息。

### Q: 如何处理空掩码（无病灶）？
**A**: 系统具有全面的空掩码处理机制：
- 指标计算：两者皆空返回 1.0，单边空返回 0.0
- 后处理：微小噪声（面积 < `min_size`）自动清除
- 确保在空 GT 场景下，小假阳性不影响 Dice 分数

### Q: Web App 中阈值调节不生效？
**A**: 已修复！现在确保：
1. 手动调节的阈值会正确应用到预测结果
2. 智能后处理也会使用用户设定的阈值（而非硬编码的 0.5）
3. 如果仍有问题，请检查是否启用了智能后处理，并确认配置文件中是否有阈值设置

---

## 📁 项目结构

```
medical-segmentation/
├── main.py                    # PyQt5 GUI 主入口
├── app.py                     # Streamlit Web App
├── server.py                  # FastAPI 后端服务
├── models.py                  # 所有模型架构
├── dataset.py                 # 2.5D 数据集加载器
├── config.py                  # 集中式模型配置
├── generate_smart_postprocessing_config.py  # 生成智能后处理配置
├── requirements.txt           # 依赖列表
├── README.md                  # 本文档
├── LICENSE                    # 许可证文件
│
├── utils/                     # 工具函数模块
│   ├── __init__.py            # 向后兼容接口
│   ├── common.py              # 公共导入和配置
│   ├── helpers.py             # 基础工具函数
│   ├── model_loader.py        # 模型加载函数
│   ├── smart_postprocessing.py # 智能后处理策略搜索
│   ├── standalone_funcs.py    # 后处理函数
│   └── ...                    # 更多模块
│
├── worker/                    # 工作线程模块
│   ├── __init__.py            # 向后兼容接口
│   ├── train_thread.py        # 训练线程
│   ├── test_thread.py         # 测试线程
│   └── predict_thread.py      # 预测线程
│
├── matlab_reports/            # 生成的 MATLAB 报告
└── data/                      # 数据目录（用户创建）
```

---

## 📈 更新日志

### v2.7 (最新)
- ✅ **MATLAB 性能分析报告优化**
  - 修复性能分析图表 X 轴标签显示问题
  - 修复 Dice 值显示错误（使用全局指标而非前景指标）
  - 优化图表布局和标签可见性
  - 支持 categorical 数组自动标签设置
- ✅ **训练流程优化**
  - 训练过程中禁用注意力热力图生成（提升速度，避免卡死）
  - 优化显存使用，减少 OOM 错误
- ✅ **全局指标计算修复**
  - 确保性能分析图表使用全局指标（基于所有样本，包括空mask）
  - 与日志中的 "Mean Dice (全样)" 保持一致

### v2.6
- ✅ **AI 辅助诊断功能**：集成 LLM（支持 OpenAI/DeepSeek/Moonshot）
  - Web App 侧边栏配置 AI 服务（API Key、Base URL、Model 等）
  - 自动提取预测结果元数据并生成诊断报告
  - 支持流式对话，实时显示 AI 回复
- ✅ **FastAPI 后端服务**：独立的 API 服务（`server.py`）
  - `/predict` - 图像分割推理接口
  - `/chat` - AI 辅助诊断聊天接口
  - 支持动态 LLM 配置
- ✅ **Web App 优化**：改进用户体验和功能完整性

### v2.5
- ✅ **Streamlit Web App**：全新的 Web 界面（`app.py`）
  - 批量图像上传和处理
  - 实时阈值调节和可视化
  - 智能后处理集成
  - 深度调试面板（概率分布直方图、阈值效果预览）
- ✅ **阈值调节功能优化**：修复智能后处理中阈值硬编码问题
- ✅ **智能模型检测**：Web App 自动检测模型模式（2D/2.5D）

### v2.4
- ✅ **Brent 阈值优化方法**：将 GWO 阈值搜索升级为 Brent 方法
- ✅ **多进程并行加速**：测试阶段使用 `ProcessPoolExecutor` 并行计算
- ✅ **性能提升**：Brent 阈值优化总耗时从几分钟压缩到 30 秒以内

### v2.3
- ✅ **HD-BET 深度学习颅骨剥离**：在 LGG-MRI 数据上引入 HD-BET 预处理流程
- ✅ **数据质量与注意力提升**：在 Skull Stripping 后的数据集上训练
- ✅ **DeepLabV3+ 模型增强**：在优化后的数据上重新训练

### v2.2
- ✅ **模块化重构** - 将 `utils.py` 和 `worker.py` 拆分为模块化结构
- ✅ **向后兼容** - 保持所有原有导入方式不变
- ✅ **性能优化** - 多进程后处理和指标计算

---

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！请确保您的更改符合项目的编码风格，并包含适当的测试。

### 贡献指南
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📧 联系方式

如有问题或建议，请：
- 提交 GitHub Issue
- 发送邮件至：chuan2410450745@sjtu.edu.cn

---

## 🙏 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
- [Streamlit](https://streamlit.io/) - Web 框架
- [FastAPI](https://fastapi.tiangolo.com/) - API 框架
- [Albumentations](https://albumentations.ai/) - 数据增强库
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) - 分割模型库
- [OpenAI](https://openai.com/) - LLM API

---

<div align="center">

⭐ 如果这个项目对你有帮助，请给个 Star！

Made with ❤️ for Medical Image Segmentation

</div>
