A# main.py 数据后处理相关代码总结

## 1. 数据预处理（Data Preprocessing）

### 位置：`main.py` 第 3556-3607 行

### 关键代码：

```python
# 【修复归一化参数】根据数据集类型动态设置归一化参数
# 2.5D数据集：3通道输入，使用ImageNet 3通道归一化
# 标准数据集：1通道输入，使用单通道归一化（但某些模型可能期望3通道，会在模型内部处理）
if self.dataset_type == "2.5d":
    # 2.5D数据集：3通道输入，使用ImageNet归一化
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)
else:
    # 标准数据集：根据模型类型判断
    # 如果模型是SMP模型（U-Net++或DeepLabV3+），可能使用3通道（如果用户配置了3通道）
    # 其他模型通常使用1通道，但为了兼容性，先使用3通道归一化
    # 实际通道数会在模型构建时根据dataset_type设置
    # 注意：如果模型输入是1通道，归一化参数会被忽略或重复使用
    normalize_mean = (0.485, 0.456, 0.406)  # 默认3通道，兼容性考虑
    normalize_std = (0.229, 0.224, 0.225)

# 训练集数据增强变换
train_transform = A.Compose([
    A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), mode=cv2.BORDER_REFLECT_101, p=0.6),
    # Grid Distortion：模拟非刚体形变，对医学影像非常有效
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.3,  # 增强形变幅度
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.3  # 30%概率应用
    ),
    # Elastic Transform：模拟器官的挤压和变形（医学影像最强增强）
    A.ElasticTransform(
        alpha=50,  # 增强形变强度（从10提升到50）
        sigma=5,   # 增强平滑度（从3提升到5）
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.4  # 提高概率（从0.15提升到0.4）
    ),
    # 其他增强...
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    # 【关键】归一化：使用 ImageNet 标准归一化
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()  # 转换为 PyTorch Tensor，并自动将 HWC -> CHW
])

# 验证集仅做几何归一化，避免引入过多随机性
val_transform = A.Compose([
    A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()
])
```

## 2. 归一化参数说明

### ImageNet 标准归一化参数：
- **均值 (mean)**: `(0.485, 0.456, 0.406)` - 对应 RGB 三个通道
- **标准差 (std)**: `(0.229, 0.224, 0.225)` - 对应 RGB 三个通道

### 归一化公式：
```python
normalized = (pixel_value / 255.0 - mean) / std
```

### 归一化后的值域：
- 理论上：约 `[-2.12, 2.64]`（取决于原始像素值）
- 实际：通常在 `[-2.5, 2.5]` 范围内

## 3. 数据后处理（Post-processing）

**注意**：数据后处理主要在 `worker.py` 中实现，不在 `main.py` 中。

### 主要后处理函数（在 worker.py 中）：

1. **`post_process_mask`**: 基础形态学后处理
   - 移除小连通域
   - 形态学操作（开运算、闭运算）
   - 填充孔洞
   - 保留最大连通域

2. **`smart_post_processing`**: 智能后处理
   - 基于面积和概率的自适应过滤
   - 分级策略：绝对噪音、安全区域、可疑区域

3. **`post_process_refine_for_hd95`**: HD95 优化后处理
   - 高斯模糊平滑边缘
   - 形态学闭运算
   - 严格连通域过滤
   - 动态面积阈值

## 4. 关键要点

1. **归一化策略**：
   - 2.5D 数据集：使用 3 通道 ImageNet 归一化
   - 标准数据集：默认也使用 3 通道归一化（兼容性考虑）

2. **数据增强**：
   - 训练集：包含多种几何和强度增强
   - 验证集：仅做 Resize 和归一化，不做随机增强

3. **Tensor 转换**：
   - `ToTensorV2()` 自动将图像从 `(H, W, C)` 转换为 `(C, H, W)`
   - 同时将像素值从 `[0, 255]` 转换为 `[0.0, 1.0]`（在归一化之前）

4. **后处理位置**：
   - 数据后处理在模型推理后进行，不在数据加载阶段
   - 后处理函数定义在 `worker.py` 的 `TrainThread` 类中

