# -*- coding: utf-8 -*-
"""
从utils.window_ops模块
"""
from utils.common import *

# ==================== 窗口操作函数 ====================

def window_partition(x, window_size):
    """
    将特征图分割成窗口
    Args:
        x: (B, H, W, C)
        window_size: 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口还原为特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: 窗口大小
        H: 特征图高度
        W: 特征图宽度
    Returns:
        x: (B, H, W, C)
    """
    # 修复：更安全的B和C计算，处理可能的维度不匹配
    num_windows_total = windows.shape[0]
    num_windows_per_image = (H // window_size) * (W // window_size)
    B = num_windows_total // num_windows_per_image
    C = windows.shape[-1]
    
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


