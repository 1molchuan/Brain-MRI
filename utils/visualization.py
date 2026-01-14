# -*- coding: utf-8 -*-
"""
从utils.visualization模块
"""
from utils.common import *

def render_quick_preview_matplotlib(images, masks, preds, save_path, num_samples=4, threshold=0.1):
    """
    【向后兼容独立函数】使用 Matplotlib 绘制预测对比图（无依赖，速度快）
    
    这是类方法的独立函数版本，用于向后兼容。
    推荐使用 MatlabVisualizationBridge.render_quick_preview_matplotlib()。
    
    Args:
        images: 图像列表 (List[np.ndarray])，每个元素为 (H, W, 3) 或 (H, W)
        masks: 真实掩码列表 (List[np.ndarray])，每个元素为 (H, W)
        preds: 预测掩码列表 (List[np.ndarray])，每个元素为 (H, W)，可以是概率值或已二值化的掩码
        save_path: 保存路径
        num_samples: 显示的样本数量
        threshold: 二值化阈值，如果preds是概率值则使用此阈值二值化，默认0.1（允许看到低置信度预测）
    
    Returns:
        save_path: 保存的文件路径
    """
    # 如果 MatlabVisualizationBridge 可用，使用类方法
    bridge = MatlabVisualizationBridge.instance()
    if bridge:
        return bridge.render_quick_preview_matplotlib(images, masks, preds, save_path, num_samples, threshold=threshold)
    
    # 否则直接实现（简化版，不依赖类）
    import matplotlib.pyplot as plt
    import numpy as np
    
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = num_samples
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img = images[i]
        true_mask = masks[i]
        pred_mask = preds[i]
        
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        
        if true_mask.ndim == 3:
            true_mask = true_mask[:, :, 0] if true_mask.shape[2] == 1 else true_mask[:, :, 0]
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[:, :, 0] if pred_mask.shape[2] == 1 else pred_mask[:, :, 0]
        
        # 二值化掩码
        # 真实掩码使用0.5阈值（标准）
        true_mask_binary = (true_mask > 0.5).astype(np.float32)
        # 预测掩码使用传入的阈值（允许看到低置信度预测）
        # 如果pred_mask已经是二值化的（只有0和1），则直接使用；否则根据阈值二值化
        if pred_mask.max() <= 1.0 and pred_mask.min() >= 0.0:
            # 可能是概率值，检查是否已经二值化
            unique_vals = np.unique(pred_mask)
            if len(unique_vals) == 2 and (0.0 in unique_vals or 1.0 in unique_vals):
                # 已经是二值化的，直接使用
                pred_mask_binary = pred_mask.astype(np.float32)
            else:
                # 是概率值，根据阈值二值化
                pred_mask_binary = (pred_mask > threshold).astype(np.float32)
        else:
            # 值域不在[0,1]，可能是未归一化的，先归一化再二值化
            pred_mask_normalized = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
            pred_mask_binary = (pred_mask_normalized > threshold).astype(np.float32)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}\nInput', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img)
        if true_mask_binary.sum() > 0:
            overlay_gt = img.copy()
            green_mask = np.zeros_like(overlay_gt)
            green_mask[true_mask_binary > 0.5] = [0, 1, 0]
            overlay_gt = overlay_gt * 0.7 + green_mask * 0.3
            axes[i, 1].imshow(overlay_gt)
        axes[i, 1].set_title('Ground Truth\n(Green)', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(img)
        if pred_mask_binary.sum() > 0:
            overlay_pred = img.copy()
            red_mask = np.zeros_like(overlay_pred)
            red_mask[pred_mask_binary > 0.5] = [1, 0, 0]
            overlay_pred = overlay_pred * 0.7 + red_mask * 0.3
            axes[i, 2].imshow(overlay_pred)
        axes[i, 2].set_title('Prediction\n(Red)', fontsize=10)
        axes[i, 2].axis('off')
        
        overlay = img.copy()
        overlay[true_mask_binary > 0.5, 1] = np.maximum(overlay[true_mask_binary > 0.5, 1], 0.5)
        overlay[pred_mask_binary > 0.5, 0] = np.maximum(overlay[pred_mask_binary > 0.5, 0], 0.5)
        overlap = (true_mask_binary > 0.5) & (pred_mask_binary > 0.5)
        overlay[overlap, 0] = 1.0
        overlay[overlap, 1] = 1.0
        overlay[overlap, 2] = 0.0
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay\n(Green=GT, Red=Pred)', fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return save_path


def save_mat_file(data_dict: dict, file_path: str):
    """
    将数据字典保存为 .mat 文件供 MATLAB 读取
    
    Args:
        data_dict: 要保存的数据字典，键为变量名，值为 numpy 数组或标量
        file_path: 保存路径（.mat 文件）
    
    Example:
        save_mat_file({
            'images': images_array,
            'masks': masks_array,
            'preds': preds_array
        }, 'output.mat')
    
    Note:
        使用 MATLAB v7.3 格式（HDF5）以支持大文件，避免 OverflowError
    """
    try:
        # 首先尝试使用 v7.3 格式（HDF5），支持更大的数据
        # 这可以避免 "Python int too large to convert to C long" 错误
        try:
            savemat(file_path, data_dict, format='7.3', do_compression=False, oned_as='column')
        except (OverflowError, ValueError) as e:
            # 如果 v7.3 格式失败（可能因为缺少 h5py），尝试压缩数据后使用 v5 格式
            print(f"[警告] 使用 v7.3 格式失败: {e}，尝试压缩数据后使用 v5 格式")
            
            # 压缩大数组：如果数组太大，进行降采样或转换为更小的数据类型
            compressed_dict = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    # 检查数组大小
                    size_mb = value.nbytes / (1024 * 1024)
                    if size_mb > 100:  # 如果大于 100MB，进行压缩
                        print(f"[压缩] {key} 数组大小: {size_mb:.2f} MB，进行压缩...")
                        # 尝试转换为 float32（如果原来是 float64）
                        if value.dtype == np.float64:
                            compressed_dict[key] = value.astype(np.float32)
                        else:
                            compressed_dict[key] = value
                    else:
                        compressed_dict[key] = value
                else:
                    compressed_dict[key] = value
            
            # 使用 v5 格式保存压缩后的数据
            savemat(file_path, compressed_dict, do_compression=False, oned_as='column')
    except Exception as e:
        # 如果所有方法都失败，提供更详细的错误信息
        error_msg = f"保存 .mat 文件失败: {e}"
        if "too large" in str(e).lower() or "overflow" in str(e).lower():
            error_msg += "\n\n建议：数据太大，无法保存为 .mat 文件。"
            error_msg += "\n可以尝试："
            error_msg += "\n1. 减少保存的数据量（如只保存部分样本）"
            error_msg += "\n2. 降低数据精度（如使用 float32 代替 float64）"
            error_msg += "\n3. 安装 h5py 库以支持 v7.3 格式：pip install h5py"
        raise RuntimeError(error_msg)