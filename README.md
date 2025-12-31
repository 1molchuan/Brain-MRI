# 医学图像分割系统 - AI智能分析平台

一个功能完整的医学图像分割系统，提供图形化界面和API服务，支持多种先进的深度学习模型架构。

## ✨ 主要特性

- 🎯 **多模型架构支持**：支持 ImprovedUNet、ResNetUNet、TransUNet、DS-TransUNet、SwinUNet 等多种模型
- 🖥️ **友好的图形界面**：基于 PyQt5 构建的现代化 GUI，操作简单直观
- 🚀 **高性能训练**：支持混合精度训练、学习率调度、早停机制、EMA/SWA 等高级训练技巧
- 📊 **丰富的评估指标**：Dice、IoU、Precision、Recall、Specificity、HD95 等完整指标
- 🔍 **测试时增强 (TTA)**：多尺度测试时增强，提升推理精度
- 📈 **可视化**：训练曲线可视化、性能分析报告、注意力热图等
- 🌐 **API 服务**：支持 RESTful API 模式，便于集成到其他系统
- 💡 **智能后处理**：LCC（最大连通域）、孔洞填充、边缘平滑等后处理技术

## 📋 系统要求

### 硬件要求
- **GPU**：推荐 NVIDIA GPU（支持 CUDA），显存 ≥ 4GB
- **内存**：≥ 8GB RAM
- **存储**：≥ 10GB 可用空间

### 软件要求
- **操作系统**：Windows 10/11, Linux, macOS
- **Python**：3.7 - 3.10
- **CUDA**：11.0+（如果使用 GPU）

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/1molchuan/Brain-MRI.git
cd Brain-MRI
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行程序

#### GUI 模式（默认）

```bash
python "任务2  分割模型 交互版.py"
```

#### API 模式

```bash
python "任务2  分割模型 交互版.py" --mode api --model path/to/model.pth
```

## 📦 依赖安装

主要依赖包：

```bash
pip install torch torchvision
pip install PyQt5
pip install albumentations
pip install opencv-python
pip install scipy
pip install scikit-image
pip install matplotlib
pip install tqdm
pip install pandas
pip install scikit-learn
```

完整依赖列表请参考 `requirements.txt`（如需要，可自行创建）。

## 📖 使用指南

### 训练模型

1. **准备数据**：
   - 数据目录结构：
     ```
     data_dir/
     ├── images/
     │   ├── patient_id1/
     │   │   ├── image1.png
     │   │   └── image2.png
     │   └── patient_id2/
     │       └── ...
     └── masks/
         ├── patient_id1/
         │   ├── mask1.png
         │   └── mask2.png
         └── patient_id2/
             └── ...
   ```

2. **配置训练参数**：
   - 选择数据目录
   - 选择模型架构
   - 设置训练轮次（Epochs）
   - 设置批次大小（Batch Size）
   - 选择优化器（Adam/SGD）
   - 启用/禁用 TTA、GWO 优化等

3. **开始训练**：
   - 点击"开始训练"按钮
   - 实时查看训练进度和验证指标
   - 训练完成后自动生成性能分析报告

### 测试模型

1. **加载模型**：选择训练好的模型文件（.pth）
2. **选择测试数据**：指定测试数据目录
3. **配置选项**：
   - 选择模型架构（或从 checkpoint 自动推断）
   - 启用/禁用 TTA
4. **开始测试**：点击"开始测试"按钮，查看详细性能指标

### 预测图像

1. **加载模型**：选择训练好的模型文件
2. **选择图像**：支持单张或批量图像预测
3. **设置阈值**：调整二值化阈值（默认 0.5）
4. **开始预测**：点击"开始预测"按钮，查看预测结果

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

### 5. SwinUNet
- 基于 Swin Transformer 的 U-Net
- 支持窗口注意力机制
- 支持 GWO 超参数优化

## 🎯 核心功能

### 训练功能
- ✅ 自动数据加载和预处理
- ✅ 多种数据增强策略
- ✅ 混合精度训练（AMP）
- ✅ 学习率调度（Poly/ReduceLROnPlateau）
- ✅ 早停机制
- ✅ EMA（指数移动平均）
- ✅ SWA（随机权重平均）
- ✅ 最佳模型自动保存
- ✅ 实时训练监控

### 评估功能
- ✅ 多指标评估（Dice, IoU, Precision, Recall, Specificity, HD95）
- ✅ 阈值扫描和最优阈值推荐
- ✅ 性能分析报告生成
- ✅ 低 Dice 案例识别
- ✅ 注意力热图可视化

### 预测功能
- ✅ 单张/批量图像预测
- ✅ 测试时增强（TTA）
- ✅ 智能后处理
- ✅ 结果可视化
- ✅ 批量导出

## 📊 性能指标

系统支持以下评估指标：

- **Dice 系数**：衡量分割重叠度
- **IoU（交并比）**：衡量预测与真实掩码的重叠
- **Precision（精确率）**：预测为正样本中真正为正的比例
- **Recall（召回率）**：真实正样本中被正确预测的比例
- **Specificity（特异度）**：真实负样本中被正确预测的比例
- **HD95**：95% Hausdorff 距离，衡量边界精度

## 🔧 高级功能

### 测试时增强 (TTA)
- 多尺度推理（0.8x, 1.0x, 1.2x）
- 8 种几何变换（翻转、旋转等）
- 加权融合策略
- 可提升 1-3% 的 Dice 系数

### 智能后处理
- **LCC（最大连通域）**：保留最大连通区域，去除噪点
- **孔洞填充**：填补小孔洞，提升分割完整性
- **边缘平滑**：Gaussian 滤波，修正锯齿边缘

### GWO 优化（SwinUNet/DS-TransUNet）
- 灰狼优化算法自动搜索最优超参数
- 优化窗口大小、注意力头数等关键参数
- 提升模型性能

## 📁 项目结构

```
medical-segmentation/
├── 任务2  分割模型 交互版.py  # 主程序文件
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包列表（需创建）
└── data/                      # 数据目录（用户创建）
    ├── images/
    └── masks/
```

## 🐛 常见问题

### Q: 训练时出现 CUDA 内存不足？
**A**: 尝试以下方法：
- 减小批次大小（Batch Size）
- 降低图像分辨率
- 关闭混合精度训练
- 使用 CPU 训练（速度较慢）

### Q: 如何选择最适合的模型架构？
**A**: 
- **小数据集**：推荐 ImprovedUNet 或 ResNetUNet
- **中等数据集**：推荐 TransUNet 或 DS-TransUNet
- **大数据集**：推荐 SwinUNet（支持 GWO 优化）

### Q: TTA 会显著增加推理时间吗？
**A**: 是的，TTA 会增加约 24 倍推理时间，但可以提升 1-3% 的 Dice 系数。建议在最终评估时使用，训练和验证阶段可关闭。

### Q: 如何提高模型性能？
**A**: 
1. 增加训练数据量
2. 使用数据增强
3. 启用 TTA（测试时）
4. 调整学习率和训练轮次
5. 使用 EMA/SWA 技术
6. 选择合适的模型架构

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：[chuan2410450745@sjtu.edu.cn]

## 🙏 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [Albumentations](https://albumentations.ai/)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

---

⭐ 如果这个项目对你有帮助，请给个 Star！



