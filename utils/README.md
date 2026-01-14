# Utils模块拆分说明

## 概述

`utils.py`（原4548行）已拆分为模块化结构，保持向后兼容性。

## 目录结构

```
utils/
├── __init__.py              # 向后兼容接口（141行）
├── common.py                 # 公共导入和配置（34行）
├── helpers.py                # 基础工具函数（447行）
├── window_ops.py             # 窗口操作函数（39行）
├── image_augmentation.py     # 图像增强类（85行）
├── model_loader.py           # 模型加载函数（633行）
├── data_processing.py        # 数据处理函数（90行）
├── standalone_funcs.py       # 独立函数（多进程）（827行）
├── process_pool.py            # 进程池管理器（73行）
├── multiprocess_helpers.py   # 多进程辅助函数（35行）
├── gwo_optimizer.py          # GWO优化器（687行）
├── threshold_scan.py         # 阈值扫描（99行）
├── dataset.py                # 数据集类（271行）
├── matlab_bridge.py          # MATLAB相关类（987行）
└── visualization.py          # 可视化函数（178行）
```

## 模块说明

### 1. `common.py`
- **功能**: 公共导入和配置
- **内容**: 所有模块共享的导入（numpy, torch, cv2等）和MATLAB路径配置

### 2. `helpers.py`
- **功能**: 基础工具函数
- **主要类/函数**:
  - `EarlyStopping`: 早停策略
  - `pure_compute_metrics`: 计算指标
  - `calculate_hd95`: 计算HD95
  - `calculate_custom_score`: 计算自定义分数
  - `calculate_official_total_score`: 计算官方总分
  - `worker_ensemble_logic`: 集成逻辑
  - `global_weight_search_worker`: 全局权重搜索
  - `calculate_metrics_for_weights`: 计算权重指标

### 3. `window_ops.py`
- **功能**: 窗口操作函数
- **主要函数**:
  - `window_partition`: 窗口分割
  - `window_reverse`: 窗口还原

### 4. `image_augmentation.py`
- **功能**: 图像增强类
- **主要类**:
  - `MedicalImageAugmentation`: 医学图像增强

### 5. `model_loader.py`
- **功能**: 模型加载函数
- **主要函数**:
  - `load_ensemble_models`: 加载集成模型
  - `load_model_compatible`: 兼容性模型加载
  - `infer_swin_params_from_state_dict`: 推断Swin参数
  - `infer_dstrans_params_from_state_dict`: 推断DS-Trans参数
  - `read_checkpoint_config`: 读取checkpoint配置

### 6. `data_processing.py`
- **功能**: 数据处理函数
- **主要函数**:
  - `parse_extra_modalities_spec`: 解析额外模态配置
  - `build_extra_modalities_lists`: 构建额外模态列表
  - `normalize_volume_percentile`: 百分位数归一化

### 7. `standalone_funcs.py`
- **功能**: 独立函数（多进程）
- **主要函数**:
  - `_compute_hd95_standalone`: 独立HD95计算
  - `_compute_dice_standalone`: 独立Dice计算
  - `compute_metrics_worker`: 指标计算工作函数
  - `ensemble_masks_global`: 全局掩码集成
  - `ensemble_post_process_global`: 全局后处理
  - `refine_segmentation_mask`: 细化分割掩码
  - `calculate_official_total_score_global`: 全局官方总分
  - `find_optimal_ensemble_weights_global`: 全局最优权重搜索

### 8. `process_pool.py`
- **功能**: 进程池管理器
- **主要类**:
  - `ProcessPoolManager`: 进程池管理器（单例模式）

### 9. `multiprocess_helpers.py`
- **功能**: 多进程辅助函数
- **主要函数**:
  - `_calculate_dice_worker`: Dice计算工作函数

### 10. `gwo_optimizer.py`
- **功能**: GWO优化器
- **主要类**:
  - `GreyWolfThresholdOptimizer`: 灰狼优化算法阈值优化器

### 11. `threshold_scan.py`
- **功能**: 阈值扫描
- **主要函数**:
  - `scan_best_threshold`: 扫描最佳阈值

### 12. `dataset.py`
- **功能**: 数据集类
- **主要类**:
  - `MedicalImageDataset`: 医学图像数据集

### 13. `matlab_bridge.py`
- **功能**: MATLAB相关类
- **主要类**:
  - `MatlabCacheManager`: MATLAB缓存管理器
  - `MatlabCacheDataset`: MATLAB缓存数据集
  - `MatlabEngineSession`: MATLAB引擎会话
  - `MatlabMetricsBridge`: MATLAB指标桥接
  - `MatlabVisualizationBridge`: MATLAB可视化桥接

### 14. `visualization.py`
- **功能**: 可视化函数
- **主要函数**:
  - `render_quick_preview_matplotlib`: 快速预览渲染
  - `save_mat_file`: 保存MAT文件

## 向后兼容性

### 原有导入方式仍然有效

```python
# 方式1: 导入所有
from utils import *

# 方式2: 导入特定函数/类
from utils import (
    EarlyStopping,
    GreyWolfThresholdOptimizer,
    MedicalImageDataset,
    load_model_compatible,
    # ... 等等
)
```

### 新的导入方式（可选）

```python
# 按需导入特定模块
from utils.helpers import EarlyStopping
from utils.gwo_optimizer import GreyWolfThresholdOptimizer
from utils.dataset import MedicalImageDataset
```

## 依赖关系

- 所有模块都依赖 `common.py`（公共导入）
- `helpers.py` 依赖 `standalone_funcs.py`（使用 `ensemble_post_process_global`）
- 其他模块相对独立

## 注意事项

1. **循环依赖**: 已通过合理的导入顺序避免循环依赖
2. **公共接口**: 所有公共接口通过 `__init__.py` 导出
3. **向后兼容**: 保持原有 `from utils import *` 方式不变
4. **备份文件**: 原 `utils.py` 已备份为 `utils.py.backup`

## 拆分优势

1. **可维护性**: 每个模块职责单一，易于定位和修改
2. **协作友好**: 减少合并冲突，多人协作更顺畅
3. **性能优化**: 按需导入，减少启动时间
4. **代码组织**: 模块化结构清晰，便于理解

## 文件统计

- **原文件**: `utils.py` (4548行)
- **拆分后**: 14个模块文件，总计约4500行（保持原有功能）
- **最大模块**: `matlab_bridge.py` (987行)
- **最小模块**: `multiprocess_helpers.py` (35行)

