# -*- coding: utf-8 -*-
"""
从utils.multiprocess_helpers模块
"""
from utils.common import *

# ==================== 多进程辅助函数 ====================
def _calculate_dice_worker(args):
    """
    多进程工作函数：计算单个阈值的Dice系数
    
    Args:
        args: 元组 (preds_flat, targets_flat, threshold)
              preds_flat: 展平的概率图（numpy数组）
              targets_flat: 展平的真实标签（numpy数组）
              threshold: 二值化阈值
    
    Returns:
        dice: Dice系数
    """
    preds_flat, targets_flat, threshold = args
    
    # 二值化预测
    pred_mask = (preds_flat >= threshold).astype(np.float32)
    targets_float = targets_flat.astype(np.float32)
    
    # 计算交集和并集
    intersection = np.sum(pred_mask * targets_float)
    union = np.sum(pred_mask) + np.sum(targets_float)
    
    # 避免除零
    if union < 1e-7:
        return 1.0 if intersection < 1e-7 else 0.0
    
    # Dice = 2 * intersection / union
    return (2.0 * intersection) / union


