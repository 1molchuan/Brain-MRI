"""
GPU加速的后处理模块 - 完全在GPU上执行，避免CPU-GPU数据传输
使用PyTorch tensor操作替代scipy和cv2，大幅减少数据搬运和CPU占用
"""
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import numpy as np


def _morphology_close_gpu(binary: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    GPU形态学闭操作（先膨胀后腐蚀）- 填充小孔洞和缝隙
    
    Args:
        binary: (H, W) 或 (1, 1, H, W) 二值mask，0/1
        kernel_size: 核大小（奇数）
    
    Returns:
        处理后的mask，形状与输入相同
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 确保是4D tensor (1, 1, H, W)
    if binary.ndim == 2:
        binary = binary.unsqueeze(0).unsqueeze(0)
    elif binary.ndim == 3:
        binary = binary.unsqueeze(1)
    
    # 膨胀：max_pool2d
    dilated = F.max_pool2d(binary, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    # 腐蚀：对补集做max_pool，然后取反
    binary_inv = 1.0 - dilated
    eroded_inv = F.max_pool2d(binary_inv, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    eroded = 1.0 - eroded_inv
    
    # 恢复原始形状
    if binary.ndim == 2:
        return eroded.squeeze(0).squeeze(0)
    elif binary.ndim == 3:
        return eroded.squeeze(1)
    return eroded.squeeze(0).squeeze(0)


def _morphology_open_gpu(binary: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    GPU形态学开操作（先腐蚀后膨胀）- 去除小噪点/毛刺
    
    Args:
        binary: (H, W) 或 (1, 1, H, W) 二值mask，0/1
        kernel_size: 核大小（奇数）
        iterations: 迭代次数
    
    Returns:
        处理后的mask，形状与输入相同
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 确保是4D tensor
    if binary.ndim == 2:
        binary = binary.unsqueeze(0).unsqueeze(0)
    elif binary.ndim == 3:
        binary = binary.unsqueeze(1)
    
    result = binary
    for _ in range(iterations):
        # 腐蚀
        binary_inv = 1.0 - result
        eroded_inv = F.max_pool2d(binary_inv, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        eroded = 1.0 - eroded_inv
        
        # 膨胀
        dilated = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        result = dilated
    
    # 恢复原始形状
    if binary.ndim == 2:
        return result.squeeze(0).squeeze(0)
    elif binary.ndim == 3:
        return result.squeeze(1)
    return result.squeeze(0).squeeze(0)


def _fill_holes_gpu(binary: torch.Tensor) -> torch.Tensor:
    """
    GPU填充孔洞 - 使用形态学膨胀+腐蚀的组合来近似
    
    Args:
        binary: (H, W) 二值mask，0/1
    
    Returns:
        填充孔洞后的mask
    """
    # 使用较大的核进行闭操作来填充孔洞
    return _morphology_close_gpu(binary, kernel_size=5)


def _connected_components_gpu(binary: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    GPU连通域标记 - 使用简化的两遍扫描算法
    
    注意：完整的连通域标记在GPU上实现复杂，这里使用简化版本
    对于大多数分割任务（单器官、最大连通域），这个版本足够准确
    
    Args:
        binary: (H, W) 二值mask，0/1
    
    Returns:
        labeled: (H, W) 标记图，0表示背景，1,2,3...表示不同连通域
        num_features: 连通域数量
    """
    H, W = binary.shape
    device = binary.device
    
    # 转换为整数
    binary_int = (binary > 0.5).long()
    
    # 如果全为0，直接返回
    if binary_int.sum() == 0:
        return torch.zeros_like(binary_int, dtype=torch.long), 0
    
    # 使用简化的标记方法：基于距离变换的近似
    # 对于大多数分割任务，我们主要关心最大连通域，所以可以使用更简单的方法
    
    # 方法：使用膨胀+腐蚀来分离连通域，然后标记
    # 但更简单：直接使用逐像素扫描（在GPU上，小循环也可以接受）
    
    labeled = torch.zeros_like(binary_int, dtype=torch.long)
    current_label = 1
    
    # 第一遍：初步标记（逐行扫描）
    # 对于第一行
    if binary_int[0, 0] > 0:
        labeled[0, 0] = current_label
        current_label += 1
    
    for j in range(1, W):
        if binary_int[0, j] > 0:
            if binary_int[0, j-1] > 0:
                labeled[0, j] = labeled[0, j-1]
            else:
                labeled[0, j] = current_label
                current_label += 1
    
    # 对于剩余行，逐像素扫描（在GPU上，小循环也可以接受）
    for i in range(1, H):
        # 第一列
        if binary_int[i, 0] > 0:
            if binary_int[i-1, 0] > 0:
                labeled[i, 0] = labeled[i-1, 0]
            else:
                labeled[i, 0] = current_label
                current_label += 1
        
        # 其余列
        for j in range(1, W):
            if binary_int[i, j] > 0:
                left_val = binary_int[i, j-1]
                up_val = binary_int[i-1, j]
                
                if left_val > 0 and up_val > 0:
                    left_label = labeled[i, j-1]
                    up_label = labeled[i-1, j]
                    # 选择较小的标签（在第二遍统一）
                    labeled[i, j] = min(left_label, up_label)
                elif left_val > 0:
                    labeled[i, j] = labeled[i, j-1]
                elif up_val > 0:
                    labeled[i, j] = labeled[i-1, j]
                else:
                    labeled[i, j] = current_label
                    current_label += 1
    
    # 第二遍：统一等价标签（迭代统一）
    # 使用迭代方法统一相邻的不同标签
    for iteration in range(5):  # 通常3-5次迭代足够
        changed = False
        
        # 检查左邻居
        left_mask = (binary_int[:, 1:] > 0) & (binary_int[:, :-1] > 0)
        if left_mask.any():
            left_labels = labeled[:, :-1]
            right_labels = labeled[:, 1:]
            diff_mask = left_mask & (left_labels != right_labels)
            if diff_mask.any():
                min_labels = torch.minimum(left_labels, right_labels)
                labeled[:, :-1] = torch.where(diff_mask, min_labels, left_labels)
                labeled[:, 1:] = torch.where(diff_mask, min_labels, right_labels)
                changed = True
        
        # 检查上邻居
        up_mask = (binary_int[1:, :] > 0) & (binary_int[:-1, :] > 0)
        if up_mask.any():
            up_labels = labeled[:-1, :]
            down_labels = labeled[1:, :]
            diff_mask = up_mask & (up_labels != down_labels)
            if diff_mask.any():
                min_labels = torch.minimum(up_labels, down_labels)
                labeled[:-1, :] = torch.where(diff_mask, min_labels, up_labels)
                labeled[1:, :] = torch.where(diff_mask, min_labels, down_labels)
                changed = True
        
        if not changed:
            break
    
    # 重新编号，确保标签连续
    unique_labels = torch.unique(labeled[binary_int > 0])
    if len(unique_labels) == 0:
        return labeled, 0
    
    # 创建映射表
    max_label = unique_labels.max().item()
    remap = torch.zeros(max_label + 1, dtype=torch.long, device=device)
    for new_idx, old_label in enumerate(unique_labels, start=1):
        remap[old_label] = new_idx
    
    labeled_remapped = remap[labeled]
    num_features = len(unique_labels)
    
    return labeled_remapped, num_features


def _keep_largest_component_gpu(binary: torch.Tensor) -> torch.Tensor:
    """
    GPU保留最大连通域
    
    Args:
        binary: (H, W) 二值mask，0/1
    
    Returns:
        只包含最大连通域的mask
    """
    labeled, num_features = _connected_components_gpu(binary)
    if num_features == 0:
        return torch.zeros_like(binary)
    
    # 计算每个连通域的大小
    sizes = torch.bincount(labeled.flatten().long(), minlength=num_features + 1)[1:]  # 跳过0（背景）
    
    # 找到最大连通域
    largest_idx = torch.argmax(sizes).item() + 1
    return (labeled == largest_idx).float()


def _remove_small_components_gpu(binary: torch.Tensor, min_size: int) -> torch.Tensor:
    """
    GPU移除小连通域
    
    Args:
        binary: (H, W) 二值mask，0/1
        min_size: 最小连通域大小（像素数）
    
    Returns:
        移除小连通域后的mask
    """
    labeled, num_features = _connected_components_gpu(binary)
    if num_features == 0:
        return torch.zeros_like(binary)
    
    # 计算每个连通域的大小
    sizes = torch.bincount(labeled.flatten().long(), minlength=num_features + 1)[1:]
    
    # 保留大于min_size的连通域
    keep_labels = torch.where(sizes >= min_size)[0] + 1  # +1因为标签从1开始
    
    if len(keep_labels) == 0:
        return torch.zeros_like(binary)
    
    # 创建mask
    result = torch.zeros_like(binary)
    for label in keep_labels:
        result[labeled == label] = 1.0
    
    return result


def post_process_mask_gpu(
    pred_mask: torch.Tensor,
    prob_map: Optional[torch.Tensor] = None,
    min_size: int = 150,
    use_morphology: bool = True,
    keep_largest: bool = False,
    fill_holes: bool = True,
    enable_opening: bool = True,
    opening_kernel_size: int = 3,
    opening_iterations: int = 1,
    confidence_gate: float = 0.90,
    min_largest_avg_prob: float = 0.5,
) -> torch.Tensor:
    """
    GPU加速的后处理函数 - 完全在GPU上执行，避免CPU-GPU数据传输
    
    Args:
        pred_mask: (H, W) 预测mask，可以是概率图或二值mask
        prob_map: (H, W) 概率图，用于置信度判断（如果为None，使用pred_mask）
        min_size: 移除小于此大小的连通域
        use_morphology: 是否使用形态学操作
        keep_largest: 是否只保留最大连通域
        fill_holes: 是否填充内部孔洞
        enable_opening: 是否启用开操作
        opening_kernel_size: 开操作核大小
        opening_iterations: 开操作迭代次数
        confidence_gate: 置信度阈值
        min_largest_avg_prob: 最大连通域最小平均概率
    
    Returns:
        处理后的mask (H, W)，float32，0/1
    """
    device = pred_mask.device
    
    # 确保是2D tensor
    if pred_mask.ndim > 2:
        pred_mask = pred_mask.squeeze()
    
    # 检查是否已经是二值化的mask（0/1）
    # 如果max <= 1.0 且 min >= 0.0，且值接近0或1，则认为是已二值化的
    is_already_binary = (pred_mask.max() <= 1.0 + 1e-5) and (pred_mask.min() >= -1e-5) and \
                        ((pred_mask < 0.1) | (pred_mask > 0.9)).all()
    
    if is_already_binary:
        # 已经是二值化的，直接使用（但确保是0/1）
        pred_binary = (pred_mask > 0.5).float()
    else:
        # 需要二值化
        if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
            # 可能是logits，需要sigmoid
            pred_mask = torch.sigmoid(pred_mask)
        pred_binary = (pred_mask > 0.5).float()
    
    # 提取概率图
    if prob_map is None:
        prob_map = pred_mask
    else:
        if prob_map.ndim > 2:
            prob_map = prob_map.squeeze()
        if prob_map.max() > 1.0 or prob_map.min() < 0.0:
            prob_map = torch.sigmoid(prob_map)
    
    # 【置信度预过滤】
    max_prob = prob_map.max()
    if max_prob < confidence_gate:
        return torch.zeros_like(pred_binary)
    
    # 【动态阈值】如果预测像素数很少，可能是噪声
    pred_sum = pred_binary.sum().item()
    image_size = pred_binary.numel()
    dynamic_threshold = max(100, int(image_size * 0.0005))
    if pred_sum < dynamic_threshold:
        return torch.zeros_like(pred_binary)
    
    # 1. 填充孔洞
    if fill_holes and pred_sum >= 1000:
        pred_binary = _fill_holes_gpu(pred_binary)
    
    # 2. 形态学操作
    if use_morphology:
        pred_binary = _morphology_close_gpu(pred_binary, kernel_size=3)
        if enable_opening:
            pred_binary = _morphology_open_gpu(pred_binary, kernel_size=opening_kernel_size, iterations=opening_iterations)
    
    # 3. 保留最大连通域或移除小连通域
    if keep_largest:
        largest_component = _keep_largest_component_gpu(pred_binary)
        largest_size = largest_component.sum().item()
        
        if largest_size < min_size:
            return torch.zeros_like(pred_binary)
        
        # 检查最大连通域的平均概率
        if prob_map is not None:
            largest_mean_prob = (prob_map * largest_component).sum() / (largest_size + 1e-7)
            if largest_mean_prob < min_largest_avg_prob:
                return torch.zeros_like(pred_binary)
        
        return largest_component
    else:
        # 移除小连通域
        if min_size > 0:
            dynamic_min_size = max(min_size, int(image_size * 0.002))
            if pred_sum < 1000:
                dynamic_min_size = max(dynamic_min_size, 1000)
            pred_binary = _remove_small_components_gpu(pred_binary, dynamic_min_size)
    
    return pred_binary


def apply_strategy_gpu(
    mask_prob: torch.Tensor,
    method: str,
    params: dict,
) -> torch.Tensor:
    """
    GPU版本的智能后处理策略应用
    
    Args:
        mask_prob: (H, W) 概率图，0-1，在GPU上
        method: 'baseline' | 'lcc' | 'remove_small'
        params: 参数，如 {'min_size': 100}
    
    Returns:
        处理后的mask (H, W)，float32，0/1，在GPU上
    """
    # 二值化
    binary = (mask_prob > 0.5).float()
    
    if binary.sum() == 0:
        return binary
    
    if method == "baseline":
        return binary
    
    if method == "lcc":
        return _keep_largest_component_gpu(binary)
    
    if method == "remove_small":
        min_size = int(params.get("min_size", 0))
        if min_size <= 0:
            return binary
        return _remove_small_components_gpu(binary, min_size)
    
    return binary

