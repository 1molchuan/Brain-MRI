"""
配置中心模块
集中管理模型参数和训练配置，提升代码可维护性
"""

from typing import Dict, Any, Optional

# ==================== 模型配置 ====================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "resnet_unet": {
        "in_channels": 3,
        "out_channels": 1,
        "pretrained": True,
        "backbone_name": "resnet101",
        "use_aspp": True
    },
    
    "trans_unet": {
        # TransUNet 使用默认参数，无需额外配置
        "in_channels": 3,
        "out_channels": 1
    },
    
    "ds_trans_unet": {
        "in_channels": 3,
        "out_channels": 1,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    },
    
    "swin_unet": {
        "in_channels": 3,
        "out_channels": 1,
        "img_size": 224,
        "patch_size": 4,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 8,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
        "use_attention_gate": True
    },
    
    "nnformer": {
        "in_channels": 3,
        "out_channels": 1
    },
    
    "improved_unet": {
        # ImprovedUNet 使用默认参数
        "in_channels": 3,
        "out_channels": 1
    },
    
    "smp_unetplusplus": {
        # SMP U-Net++ 配置
        # 【2.5D支持】支持2.5D数据集（3通道）和标准数据集（1通道）
        # 输入通道数会根据数据集类型自动设置：2.5D数据集=3，标准数据集=1
        "encoder_name": "resnet101",  # 可切换为 "efficientnet-b5" 等
        "in_channels": 3,  # 默认值（实际会根据数据集类型动态设置）
        "out_channels": 1,  # 二分类
        "encoder_weights": "imagenet",  # ImageNet 预训练
        "activation": None,  # 二分类通常不需要激活（在损失函数中处理）
        "pretrained_weights_path": None  # 自定义预训练权重路径（.pth 文件），默认为 None
    },
    
    "smp_deeplabv3plus": {
        # DeepLabV3+ 配置（降维打击：更接近纯ResNet效果，训练更稳定）
        # 【2.5D支持】支持2.5D数据集（3通道）和标准数据集（1通道）
        # 输入通道数会根据数据集类型自动设置：2.5D数据集=3，标准数据集=1
        "encoder_name": "resnet101",  # 使用ResNet101作为编码器
        "in_channels": 3,  # 默认值（实际会根据数据集类型动态设置）
        "out_channels": 1,  # 二分类
        "encoder_weights": "imagenet",  # ImageNet 预训练
        "activation": None,  # 二分类通常不需要激活（在损失函数中处理）
        "pretrained_weights_path": None  # 自定义预训练权重路径（.pth 文件），默认为 None
    }
}

# ==================== 数据集配置 ====================

DEFAULT_DATASET_CONFIG: Dict[str, Any] = {
    "dataset_type": "2d",  # "2d" 或 "2.5d"
    # "2d": 单通道模式，返回 (1, H, W) 图像，适用于单通道预训练权重
    # "2.5d": 三通道堆叠模式，返回 (3, H, W) 图像，适用于 2.5D 任务或 ImageNet 预训练
}

# ==================== 训练配置 ====================

DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    # 注意：当输入分辨率为 512x512 时，建议保持较小的 batch_size（例如 4），以避免显存溢出
    # 在 Windows 上配合梯度累积，推荐 batch_size=4
    "batch_size": 4,
    "learning_rate": 1e-4,
    "patience": 6,  # 早停轮数
    "min_delta": 5e-4,  # 早停最小改进
    "warmup_epochs": 3,  # 早停预热轮数
    # Windows 下多进程 DataLoader 开销较大，统一使用 0（主进程加载数据）
    "num_workers": 0,
    "pin_memory": True,
    "drop_last": False,
    "weight_decay": 1e-4,
    "optimizer_type": "adam",  # "adam" 或 "sgd"
    "scheduler_type": "poly",  # "poly" 或 "plateau"
    "use_amp": True,  # 混合精度训练
    "use_ema": True,  # 指数移动平均
    "ema_decay": 0.999,
    "use_swa": False,  # 随机权重平均
    "swa_start_epoch": 0.75,  # SWA 开始轮次（相对于总轮次的比例）
    "grad_clip": 1.0,  # 梯度裁剪
    "save_best": True,
    "save_last": False,
}

# ==================== 数据增强配置 ====================

DEFAULT_AUGMENTATION_CONFIG: Dict[str, Any] = {
    "resize_size": (512, 512),  # 提升分辨率以保留更多病灶边缘细节
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.1,
    "affine_prob": 0.6,
    "affine_translate": 0.05,
    "affine_scale": (0.9, 1.1),
    "affine_rotate": (-10, 10),
    "grid_distortion_prob": 0.3,
    "elastic_transform_prob": 0.4,
    "brightness_contrast_prob": 0.4,
    "gamma_correction_prob": 0.3,
    "clahe_prob": 0.3,
    "gaussian_blur_prob": 0.15,
    "normalize_mean": (0.485, 0.456, 0.406),
    "normalize_std": (0.229, 0.224, 0.225),
}

# ==================== 评估配置 ====================

DEFAULT_EVAL_CONFIG: Dict[str, Any] = {
    "use_tta": True,  # 测试时增强
    "tta_scales": [0.8, 1.0, 1.2],
    "threshold_search_range": (0.3, 0.91, 0.05),  # (start, end, step)
    "num_visualization_samples": 8,
    "save_attention_maps": True,
    "save_performance_analysis": True,
}

# ==================== 工具函数 ====================

def get_model_config(model_type: str, **override_params) -> Dict[str, Any]:
    """
    获取模型配置，支持参数覆盖
    
    Args:
        model_type: 模型类型（如 "swin_unet", "resnet_unet" 等）
        **override_params: 要覆盖的参数
    
    Returns:
        模型参数字典
    """
    model_type = model_type.lower()
    
    # 处理别名
    if model_type in ("transunet",):
        model_type = "trans_unet"
    elif model_type in ("dstransunet", "ds-transunet"):
        model_type = "ds_trans_unet"
    elif model_type in ("swinunet",):
        model_type = "swin_unet"
    
    # 获取基础配置
    if model_type in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_type].copy()
    else:
        # 默认使用 ImprovedUNet 配置
        config = MODEL_CONFIGS["improved_unet"].copy()
    
    # 应用参数覆盖
    if override_params:
        config.update(override_params)
    
    return config


def get_train_config(**override_params) -> Dict[str, Any]:
    """
    获取训练配置，支持参数覆盖
    
    Args:
        **override_params: 要覆盖的参数
    
    Returns:
        训练参数字典
    """
    config = DEFAULT_TRAIN_CONFIG.copy()
    if override_params:
        config.update(override_params)
    return config


def get_eval_config(**override_params) -> Dict[str, Any]:
    """
    获取评估配置，支持参数覆盖
    
    Args:
        **override_params: 要覆盖的参数
    
    Returns:
        评估参数字典
    """
    config = DEFAULT_EVAL_CONFIG.copy()
    if override_params:
        config.update(override_params)
    return config


def get_dataset_config(**override_params) -> Dict[str, Any]:
    """
    获取数据集配置，支持参数覆盖
    
    Args:
        **override_params: 要覆盖的参数
    
    Returns:
        数据集参数字典
    """
    config = DEFAULT_DATASET_CONFIG.copy()
    if override_params:
        config.update(override_params)
    return config

