# -*- coding: utf-8 -*-
"""
Utils模块 - 向后兼容接口
从utils.py拆分而来，保持原有导入方式不变
"""

# 导入公共依赖（必须在最前面）
from utils.common import *

# 按依赖顺序导入各个模块
# 1. 基础工具函数
from utils.helpers import (
    EarlyStopping,
    pure_compute_metrics,
    calculate_hd95,
    calculate_custom_score,
    calculate_official_total_score,
    worker_ensemble_logic,
    global_weight_search_worker,
    calculate_metrics_for_weights,
)

# 2. 窗口操作函数
from utils.window_ops import (
    window_partition,
    window_reverse,
)

# 3. 图像增强类
from utils.image_augmentation import (
    MedicalImageAugmentation,
)

# 4. 模型加载函数
from utils.model_loader import (
    load_ensemble_models,
    load_model_compatible,
    infer_swin_params_from_state_dict,
    infer_dstrans_params_from_state_dict,
    read_checkpoint_config,
)

# 5. 数据处理函数
from utils.data_processing import (
    parse_extra_modalities_spec,
    build_extra_modalities_lists,
    normalize_volume_percentile,
)

# 6. 独立函数（多进程）
from utils.standalone_funcs import (
    _compute_hd95_standalone,
    _compute_dice_standalone,
    compute_metrics_worker,
    ensemble_masks_global,
    ensemble_post_process_global,
    refine_segmentation_mask,
    calculate_official_total_score_global,
    find_optimal_ensemble_weights_global,
)
from utils.smart_postprocessing import (
    find_optimal_postprocessing_strategy,
)

# 7. 进程池管理器
from utils.process_pool import (
    ProcessPoolManager,
)

# 8. 多进程辅助函数
from utils.multiprocess_helpers import (
    _calculate_dice_worker,
)

# 9. GWO优化器
from utils.gwo_optimizer import (
    GreyWolfThresholdOptimizer,
)

# 10. 阈值扫描
from utils.threshold_scan import (
    scan_best_threshold,
)

# 11. 数据集类
from utils.dataset import (
    MedicalImageDataset,
)

# 12. MATLAB相关类
from utils.matlab_bridge import (
    MatlabCacheManager,
    MatlabCacheDataset,
    MatlabEngineSession,
    MatlabMetricsBridge,
    MatlabService,
    MatlabVisualizationBridge,
)

# 13. 可视化函数
from utils.visualization import (
    render_quick_preview_matplotlib,
    save_mat_file,
)

# 导出所有公共接口（用于 from utils import *）
__all__ = [
    # 基础工具函数
    'EarlyStopping',
    'pure_compute_metrics',
    'calculate_hd95',
    'calculate_custom_score',
    'calculate_official_total_score',
    'worker_ensemble_logic',
    'global_weight_search_worker',
    'calculate_metrics_for_weights',
    # 窗口操作
    'window_partition',
    'window_reverse',
    # 图像增强
    'MedicalImageAugmentation',
    # 模型加载
    'load_ensemble_models',
    'load_model_compatible',
    'infer_swin_params_from_state_dict',
    'infer_dstrans_params_from_state_dict',
    'read_checkpoint_config',
    # 数据处理
    'parse_extra_modalities_spec',
    'build_extra_modalities_lists',
    'normalize_volume_percentile',
    # 独立函数（多进程）
    '_compute_hd95_standalone',
    '_compute_dice_standalone',
    'compute_metrics_worker',
    'ensemble_masks_global',
    'ensemble_post_process_global',
    'refine_segmentation_mask',
    'calculate_official_total_score_global',
    'find_optimal_ensemble_weights_global',
    'find_optimal_postprocessing_strategy',
    # 进程池管理器
    'ProcessPoolManager',
    # 多进程辅助函数
    '_calculate_dice_worker',
    # GWO优化器
    'GreyWolfThresholdOptimizer',
    # 阈值扫描
    'scan_best_threshold',
    # 数据集类
    'MedicalImageDataset',
    # MATLAB相关类
    'MatlabCacheManager',
    'MatlabCacheDataset',
    'MatlabEngineSession',
    'MatlabMetricsBridge',
    'MatlabService',
    'MatlabVisualizationBridge',
    # 可视化函数
    'render_quick_preview_matplotlib',
    'save_mat_file',
]

