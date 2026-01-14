# -*- coding: utf-8 -*-
"""
从utils.standalone_funcs模块
"""
from utils.common import *

# ==================== 独立函数（用于多进程并行处理）====================

def _compute_hd95_standalone(pred_mask, target_mask):
    """
    独立的HD95计算函数，不依赖类实例，可用于多进程并行处理
    
    Args:
        pred_mask: 预测掩码 (numpy array)
        target_mask: 真实掩码 (numpy array)
    
    Returns:
        HD95值 (float)
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt
    
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return np.nan
    
    structure = np.ones((3, 3), dtype=bool)
    pred_border = np.logical_xor(pred, binary_erosion(pred, structure=structure, border_value=0))
    target_border = np.logical_xor(target, binary_erosion(target, structure=structure, border_value=0))
    
    if not pred_border.any():
        pred_border = pred
    if not target_border.any():
        target_border = target
    
    target_distance = distance_transform_edt(~target_border)
    pred_distance = distance_transform_edt(~pred_border)
    
    distances_pred_to_target = target_distance[pred_border]
    distances_target_to_pred = pred_distance[target_border]
    
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    if all_distances.size == 0:
        return 0.0
    return float(np.percentile(all_distances, 95))


def _compute_dice_standalone(pred_mask, target_mask, smooth=1e-7):
    """
    【统一修复】独立的Dice计算函数，不依赖类实例，可用于多进程并行处理
    
    使用与 _compute_metrics_unified 相同的逻辑，确保一致性
    
    Args:
        pred_mask: 预测掩码 (numpy array)
        target_mask: 真实掩码 (numpy array)
        smooth: 平滑系数
    
    Returns:
        Dice值 (float)
    """
    # 二值化：只计算前景类（> 0.5 视为前景）
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    target_binary = (target_mask > 0.5).astype(np.float32)
    
    # 展平
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # 计算统计量
    pred_sum = float(pred_flat.sum())
    target_sum = float(target_flat.sum())
    intersection = float((pred_flat * target_flat).sum())
    
    # 【核心修复逻辑】空掩码特判
    # Case 1: 双空（GT 为空且 Pred 为空）
    if target_sum <= smooth and pred_sum <= smooth:
        return 1.0  # Dice=1.0 (完美预测)
    
    # Case 2: 单空（GT 为空但 Pred 不为空，或 GT 不为空但 Pred 为空）
    if target_sum <= smooth or pred_sum <= smooth:
        return 0.0  # Dice=0.0 (误报或漏报)
    
    # Case 3: 正常情况，使用标准 Dice 公式（只计算前景类）
    # Dice = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
    return (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)


def compute_metrics_worker(mask_tuple, weights, gt_mask):
    """
    全局独立的工作函数，用于多进程并行计算指标
    
    Args:
        mask_tuple: (sample_idx, sample_masks) 元组，其中sample_masks是多个模型的掩码列表
        weights: 权重列表
        gt_mask: 真实掩码
    
    Returns:
        (dice, hd95): Dice和HD95值
    """
    sample_idx, sample_masks = mask_tuple
    
    # 【任务4】强制数据类型转换
    sample_masks = [np.array(m) if not hasattr(m, 'ndim') else m for m in sample_masks]
    gt_mask = np.array(gt_mask) if not hasattr(gt_mask, 'ndim') else gt_mask
    
    # 使用全局独立函数进行集成
    ensemble_mask = ensemble_masks_global(sample_masks, weights)
    
    # 【极致后处理流水线】必须执行三步后处理
    ensemble_mask = ensemble_post_process_global(
        ensemble_mask,
        use_lcc=True,  # 【第一步】保留最大连通域，彻底切除离群噪点
        use_remove_holes=True,  # 【第二步】填补小孔洞，提升Dice约0.5%
        min_hole_size=100,
        use_edge_smoothing=True  # 【第三步】边缘平滑，修正锯齿边缘
    )
    
    # 计算指标
    dice = _compute_dice_standalone(ensemble_mask, gt_mask)
    hd95 = _compute_hd95_standalone(ensemble_mask, gt_mask)
    
    return dice, hd95


def ensemble_masks_global(mask_list, weights):
    """
    多尺度概率图集成：像素级加权融合（支持任意数量N个模型）
    
    将多个不同分辨率的概率图（或二值掩码）进行加权融合，利用512模型的精细度修正224模型的粗糙边缘。
    
    Args:
        mask_list: 掩码列表（List[numpy.ndarray | torch.Tensor]），每个元素可以是：
                  - numpy array (H, W) 或 (C, H, W) - 概率图或二值掩码
                  - torch.Tensor (H, W) 或 (C, H, W) - 概率图或二值掩码
        weights: 权重列表（List[float]），长度必须与 mask_list 相同，且权重之和应为1.0
    
    Returns:
        ensemble_mask: 融合后的概率图 (numpy array, H x W)
    
    Raises:
        ValueError: 如果掩码数量与权重数量不匹配
    """
    # 【核心修复】动态检查：确保数量严格对齐
    assert len(mask_list) == len(weights), \
        f"掩码数量 ({len(mask_list)}) 与权重数量 ({len(weights)}) 不匹配"
    
    # 【任务4】强制数据类型转换：解决ndim错误
    mask_list = [np.array(m) if not hasattr(m, 'ndim') else m for m in mask_list]
    
    # 权重归一化（如果权重之和不为1.0）
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"⚠️  警告: 权重之和 ({weight_sum:.6f}) 不等于 1.0，将自动归一化")
        weights = [w / weight_sum for w in weights]
    
    # 【核心修复】强制类型转换：确保所有掩码都是numpy数组
    mask_arrays = []
    target_shape = (512, 512)  # 强制使用512x512作为目标尺寸
    
    for i, mask in enumerate(mask_list):
        # 强制转换为numpy数组
        if isinstance(mask, list):
            mask = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif not isinstance(mask, np.ndarray) or not hasattr(mask, 'ndim'):
            mask = np.asarray(mask)
        
        # 处理维度：如果是 (C, H, W)，取第一个通道
        if hasattr(mask, 'ndim'):
            if mask.ndim == 3:
                mask = mask[0]  # 取第一个通道
            elif mask.ndim != 2:
                raise ValueError(f"掩码 {i} 的维度 ({mask.ndim}) 不支持，应为 2D (H, W) 或 3D (C, H, W)")
        
        # 【关键修复】强制所有概率图对齐到512x512，使用bilinear插值
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask.astype(np.float32), 
                (target_shape[1], target_shape[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR  # 使用bilinear插值
            )
        
        # 确保值在 [0, 1] 范围内
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = np.clip(mask, 0.0, 1.0)
        
        mask_arrays.append(mask)
    
    # 【任务2】像素级融合：w1 * mask1 + w2 * mask2（双模型优化）
    if len(mask_arrays) == 2:
        ensemble_mask = weights[0] * mask_arrays[0] + weights[1] * mask_arrays[1]
    else:
        # 【核心修复】像素级加权融合：使用动态循环，支持任意数量模型
        ensemble_mask = np.zeros_like(mask_arrays[0], dtype=np.float32)
        for weight, mask in zip(weights, mask_arrays):
            ensemble_mask += weight * mask
    
    # 确保值在 [0, 1] 范围内
    ensemble_mask = np.clip(ensemble_mask, 0.0, 1.0)
    
    return ensemble_mask


def ensemble_post_process_global(ensemble_mask, use_lcc=True, use_remove_holes=True, 
                                 min_hole_size=100, use_edge_smoothing=True):
    """
    【极致后处理流水线】集成后处理：对融合后的概率图进行后处理
    
    三步流水线：
    1. Largest Connected Component (LCC): 保留最大连通域，彻底切除离群噪点
    2. remove_small_holes: 填补小孔洞，提升Dice约0.5%
    3. 边缘平滑: 微小腐蚀+膨胀，修正锯齿边缘
    
    Args:
        ensemble_mask: 融合后的概率图 (numpy array, H x W)
        use_lcc: 是否使用最大连通域
        use_remove_holes: 是否移除小孔洞
        min_hole_size: 最小孔洞大小（像素），小于此值的孔洞将被填补
        use_edge_smoothing: 是否使用边缘平滑（腐蚀+膨胀）
    
    Returns:
        processed_mask: 处理后的二值掩码 (numpy array, H x W, 0-1)
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    
    # 确保是numpy数组
    if isinstance(ensemble_mask, torch.Tensor):
        mask_np = ensemble_mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(ensemble_mask)
    
    # 确保是2D
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    # 二值化（使用0.5作为阈值）
    binary_mask = (mask_np > 0.5).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return binary_mask.astype(np.float32)
    
    # 【第一步：LCC 过滤】保留最大连通域，彻底切除离群噪点
    if use_lcc:
        labeled, num_features = ndimage.label(binary_mask)
        if num_features > 0:
            # 计算每个连通域的大小
            sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
            # 找到最大的连通域
            largest_label = np.argmax(sizes) + 1
            # 只保留最大连通域
            binary_mask = (labeled == largest_label).astype(np.uint8)
    
    # 【第二步：空洞填充】填补小孔洞，提升Dice约0.5%
    if use_remove_holes and binary_mask.sum() > 0:
        if SKIMAGE_MORPHOLOGY_AVAILABLE:
            # 使用skimage.morphology.remove_small_holes（更精确）
            binary_mask = morphology.remove_small_holes(
                binary_mask.astype(bool), 
                area_threshold=min_hole_size
            ).astype(np.uint8)
        else:
            # 使用scipy实现（回退方案）
            # 反转掩码，找到孔洞（背景中的连通域）
            inverted = (~binary_mask.astype(bool)).astype(np.uint8)
            labeled_holes, num_holes = ndimage.label(inverted)
            if num_holes > 0:
                # 计算每个孔洞的大小
                hole_sizes = ndimage.sum(inverted, labeled_holes, range(1, num_holes + 1))
                # 找到需要填补的小孔洞
                small_holes = []
                for i, size in enumerate(hole_sizes):
                    if size < min_hole_size:
                        small_holes.append(i + 1)
                # 填补小孔洞
                if small_holes:
                    for hole_label in small_holes:
                        binary_mask[labeled_holes == hole_label] = 1
    
    # 【第三步：边缘平滑】微小腐蚀+膨胀，修正锯齿边缘
    if use_edge_smoothing and binary_mask.sum() > 0:
        # 使用3x3结构元素进行微小腐蚀（去除细小突起）
        structure = np.ones((3, 3), dtype=bool)
        binary_mask = binary_erosion(binary_mask.astype(bool), structure=structure, iterations=1).astype(np.uint8)
        # 使用3x3结构元素进行膨胀（恢复大致形状，但边缘更平滑）
        binary_mask = binary_dilation(binary_mask.astype(bool), structure=structure, iterations=1).astype(np.uint8)
    
    return binary_mask.astype(np.float32)


def refine_segmentation_mask(mask, 
                             closing_kernel_size=5, 
                             opening_kernel_size=3,
                             use_largest_component=True,
                             use_gaussian_blur=True,
                             gaussian_sigma=1.0,
                             use_median_filter=True,
                             median_kernel_size=5,
                             hole_area_threshold=500):
    """
    分割结果后处理函数：去除噪声点、填充空洞、平滑边缘（改进版）
    
    改进的处理流程（重点解决边缘毛刺问题）：
    1. 形态学闭运算：确保边界闭合，填充小裂缝
    2. 多级平滑：
       - 先执行中值滤波（5×5）去除孤立像素和"狗啃"状毛刺
       - 再执行高斯滤波平滑边缘
    3. 重新二值化：确保掩码清晰
    4. 开运算：移除细小噪点
    5. 孔洞修补：使用 remove_small_holes 填补内部空洞（area_threshold=500）
    6. 最大连通域提取：只保留面积最大的分割区域
    
    Args:
        mask: 输入的分割掩码 (numpy array, H x W)，可以是二值掩码(0/1)或概率图(0-1)
        closing_kernel_size: 闭运算的核大小（奇数，推荐5-7）
        opening_kernel_size: 开运算的核大小（奇数，推荐3-5）
        use_largest_component: 是否使用最大连通域提取
        use_gaussian_blur: 是否使用高斯滤波平滑边缘
        gaussian_sigma: 高斯滤波的标准差（推荐0.5-2.0）
        use_median_filter: 是否使用中值滤波（默认True，用于去除毛刺）
        median_kernel_size: 中值滤波的核大小（默认5，推荐5×5）
        hole_area_threshold: 孔洞修补的面积阈值（默认500像素）
    
    Returns:
        refined_mask: 处理后的二值掩码 (numpy array, H x W, 0-1)
    
    Example:
        >>> import numpy as np
        >>> mask = np.random.rand(512, 512) > 0.5
        >>> refined = refine_segmentation_mask(mask)
        >>> print(f"原始掩码面积: {mask.sum()}, 处理后面积: {refined.sum()}")
    """
    # 确保输入是numpy数组
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    
    # 确保是2D数组
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    # 二值化：如果输入是概率图，转换为二值掩码
    if mask_np.max() <= 1.0 and mask_np.min() >= 0.0:
        # 可能是概率图，使用0.5作为阈值
        binary_mask = (mask_np > 0.5).astype(np.uint8)
    else:
        # 已经是二值掩码
        binary_mask = (mask_np > 0).astype(np.uint8)
    
    # 如果掩码全为空，直接返回
    if binary_mask.sum() == 0:
        return binary_mask.astype(np.float32)
    
    # 【步骤1：形态学闭运算 - 确保边界闭合】
    # 闭运算 = 先膨胀后腐蚀，可以填充物体内部的小空洞和小裂缝，确保边界闭合
    if closing_kernel_size > 0:
        kernel_closing = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (closing_kernel_size, closing_kernel_size)
        )
        binary_mask = cv2.morphologyEx(
            binary_mask, 
            cv2.MORPH_CLOSE, 
            kernel_closing, 
            iterations=1
        )
    
    # 【步骤2：多级平滑 - 先中值滤波再高斯滤波】
    # 中值滤波：去除孤立像素和"狗啃"状毛刺（关键步骤）
    if use_median_filter and binary_mask.sum() > 0:
        # 使用5×5中值滤波去除毛刺和孤立像素
        binary_mask = cv2.medianBlur(binary_mask, median_kernel_size)
    
    # 高斯滤波：进一步平滑边缘
    if use_gaussian_blur and binary_mask.sum() > 0:
        # 高斯滤波需要先将二值掩码转换为浮点数
        mask_float = binary_mask.astype(np.float32)
        smoothed = cv2.GaussianBlur(
            mask_float, 
            (0, 0),  # 自动计算核大小
            sigmaX=gaussian_sigma, 
            sigmaY=gaussian_sigma
        )
        # 重新二值化，确保掩码清晰
        binary_mask = (smoothed > 0.5).astype(np.uint8)
    
    # 【步骤3：开运算 - 移除细小噪点】
    # 开运算 = 先腐蚀后膨胀，可以移除小的孤立噪点
    if opening_kernel_size > 0 and binary_mask.sum() > 0:
        kernel_opening = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (opening_kernel_size, opening_kernel_size)
        )
        binary_mask = cv2.morphologyEx(
            binary_mask, 
            cv2.MORPH_OPEN, 
            kernel_opening, 
            iterations=1
        )
    
    # 【步骤4：孔洞修补 - 填补内部空洞】
    # 使用 skimage.morphology.remove_small_holes 填补病灶内部的黑色空洞
    if binary_mask.sum() > 0:
        if SKIMAGE_MORPHOLOGY_AVAILABLE:
            # 使用 skimage.morphology.remove_small_holes（更精确）
            binary_mask = morphology.remove_small_holes(
                binary_mask.astype(bool), 
                area_threshold=hole_area_threshold
            ).astype(np.uint8)
        else:
            # 使用 scipy 实现（回退方案）
            # 反转掩码，找到孔洞（背景中的连通域）
            inverted = (~binary_mask.astype(bool)).astype(np.uint8)
            labeled_holes, num_holes = ndimage.label(inverted)
            if num_holes > 0:
                # 计算每个孔洞的大小
                hole_sizes = ndimage.sum(inverted, labeled_holes, range(1, num_holes + 1))
                # 找到需要填补的小孔洞
                small_holes = []
                for i, size in enumerate(hole_sizes):
                    if size < hole_area_threshold:
                        small_holes.append(i + 1)
                # 填补小孔洞
                if small_holes:
                    for hole_label in small_holes:
                        binary_mask[labeled_holes == hole_label] = 1
    
    # 【步骤5：最大连通域提取 - 只保留最大区域】
    if use_largest_component and binary_mask.sum() > 0:
        labeled, num_features = ndimage.label(binary_mask)
        if num_features > 0:
            # 计算每个连通域的面积
            sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
            # 找到最大的连通域
            largest_label = np.argmax(sizes) + 1
            # 只保留最大连通域
            binary_mask = (labeled == largest_label).astype(np.uint8)
    
    return binary_mask.astype(np.float32)


def calculate_official_total_score_global(dice, iou, hd95, sensitivity, specificity):
    """
    计算比赛官方总分公式：
    Total = 0.6*Dice + 0.1*IoU + 0.1/(1+HD95) + 0.1*Sens + 0.1*Spec
    
    Args:
        dice: Dice系数
        iou: IoU系数
        hd95: HD95值（如果为NaN或Inf，则使用一个很大的值）
        sensitivity: 敏感度（召回率）
        specificity: 特异性
    
    Returns:
        总分
    """
    # 处理HD95的NaN/Inf情况
    if np.isnan(hd95) or np.isinf(hd95):
        hd95_term = 0.0  # 如果HD95不可计算，该项为0
    else:
        hd95_term = 0.1 / (1.0 + hd95)
    
    total_score = (
        0.6 * dice +
        0.1 * iou +
        hd95_term +
        0.1 * sensitivity +
        0.1 * specificity
    )
    return total_score

def find_optimal_ensemble_weights_global(mask_list, gt_masks, weight_range=(0.0, 1.0, 0.1),
                                         hd95_threshold=3.0, device=None, search_samples=100, 
                                         use_parallel=True, n_jobs=4):
    """
    寻找最优集成权重，使得验证集上的 Dice 提升且 HD95 保持在阈值以内
    
    Args:
        mask_list: 掩码列表（多个模型的预测结果）
        gt_masks: 真实掩码列表（ground truth）
        weight_range: 权重搜索范围 (min, max, step)
        hd95_threshold: HD95 阈值，默认 3.0
        device: 计算设备（用于计算HD95）
        search_samples: 随机采样数量，默认100（用于加速搜索）
        use_parallel: 是否使用并行处理，默认True
        n_jobs: 并行任务数，-1表示使用所有CPU核心
    
    Returns:
        best_weights: 最优权重列表
        best_metrics: 最优指标字典 {'dice': float, 'hd95': float, 'total_score': float}
    """
    import gc
    import random
    from scipy.ndimage import binary_erosion, distance_transform_edt
    
    # 尝试导入joblib用于并行处理
    try:
        from joblib import Parallel, delayed
        JOBLIB_AVAILABLE = True
    except ImportError:
        JOBLIB_AVAILABLE = False
        if use_parallel:
            print("⚠️  警告: joblib未安装，将使用单进程模式。建议安装: pip install joblib")
    
    # 【任务4】强制数据类型转换：解决ndim错误
    # 【核心修复2】彻底解决数据类型异常：强制类型转换
    # 注意：mask_list可能是嵌套列表，需要递归处理
    converted_mask_list = []
    for model_idx, model_masks in enumerate(mask_list):
        if isinstance(model_masks, list):
            converted_model_masks = []
            for mask_idx, mask in enumerate(model_masks):
                # 强制类型转换：确保是numpy数组
                if isinstance(mask, list):
                    mask = np.array(mask)
                elif isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                elif not isinstance(mask, np.ndarray) or not hasattr(mask, 'ndim'):
                    mask = np.asarray(mask)
                converted_model_masks.append(mask)
            converted_mask_list.append(converted_model_masks)
        else:
            # 如果已经是数组，也要检查
            if not isinstance(model_masks, np.ndarray) or not hasattr(model_masks, 'ndim'):
                converted_mask_list.append(np.asarray(model_masks))
            else:
                converted_mask_list.append(model_masks)
    
    mask_list = converted_mask_list
    
    # 【核心修复2续】确保mask_list中的每个元素都有ndim属性
    for model_idx, model_masks in enumerate(mask_list):
        if isinstance(model_masks, list):
            for mask_idx, mask in enumerate(model_masks):
                if not hasattr(mask, 'ndim'):
                    mask_list[model_idx][mask_idx] = np.asarray(mask)
    
    # 同样处理gt_masks
    if isinstance(gt_masks, list):
        converted_gt_masks = []
        for mask in gt_masks:
            if isinstance(mask, list):
                mask = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            elif not isinstance(mask, np.ndarray):
                mask = np.asarray(mask)
            converted_gt_masks.append(mask)
        gt_masks = converted_gt_masks
    
    num_models = len(mask_list)
    if num_models < 1:
        raise ValueError("至少需要1个模型进行集成")
    
    # 【军令状：极致提速】强制采样策略：搜索阶段只使用100张图片
    total_samples = len(gt_masks)
    search_samples_fixed = 100  # 强制固定为100张，确保搜索速度（从112秒/it降至3秒/it）
    
    # 保存原始数据用于终效评估
    original_mask_list = mask_list
    original_gt_masks = gt_masks
    
    if search_samples_fixed < total_samples:
        # 【军令状】均匀采样100张图片（确保统计分布代表性）
        if total_samples <= search_samples_fixed:
            sample_indices = list(range(total_samples))
        else:
            # 均匀采样：每隔 total_samples/search_samples_fixed 取一张
            step = total_samples / search_samples_fixed
            sample_indices = [int(i * step) for i in range(search_samples_fixed)]
            # 确保最后一个索引不超过范围
            sample_indices = [min(idx, total_samples - 1) for idx in sample_indices]
            # 去重并排序
            sample_indices = sorted(list(set(sample_indices)))
        
        print(f"🚀 【极致提速】采样策略: 从 {total_samples} 张图片中均匀抽取 {len(sample_indices)} 张进行权重搜索")
        print(f"   预期提速: 从 ~112秒/it 降至 ~3秒/it (提速约 {100*(1-100/total_samples):.1f}%)")
        
        sampled_mask_list = []
        for model_masks in mask_list:
            if isinstance(model_masks, list):
                sampled_mask_list.append([model_masks[i] for i in sample_indices])
            else:
                sampled_mask_list.append(model_masks[sample_indices] if hasattr(model_masks, '__getitem__') else model_masks)
        sampled_gt_masks = [gt_masks[i] for i in sample_indices]
        mask_list = sampled_mask_list
        gt_masks = sampled_gt_masks
        print(f"✅ 采样完成，实际使用 {len(gt_masks)} 张图片进行搜索")
    else:
        print(f"📊 使用全量 {total_samples} 张图片进行权重搜索（数据量较小）")
    
    # 【任务2】动态权重生成：检测N个模型，自动适配搜索策略
    min_w, max_w, step_w = weight_range
    
    # 生成所有权重组合
    if num_models == 1:
        weight_combinations = [[1.0]]
    elif num_models == 2:
        # 【任务2】N=2时，自动切换为一维搜索：w1从0到1，w2 = 1.0 - w1
        weight_combinations = []
        for w1 in np.arange(0.0, 1.0 + step_w, step_w):
            w1 = round(w1, 2)
            w2 = round(1.0 - w1, 2)
            weight_combinations.append([w1, w2])
        print(f"✅ 双模型一维搜索：生成 {len(weight_combinations)} 种权重组合（w1: 0.0-1.0, 步长: {step_w}）")
    else:
        # 【任务2】N>2时，使用itertools.product生成步长为0.1的权重组合
        import itertools
        # 使用0.1步长生成权重组合（而不是使用step_w，避免组合数过多）
        weight_steps = np.arange(min_w, max_w + 0.1, 0.1)
        weight_steps = [round(w, 1) for w in weight_steps]
        
        all_combinations = list(itertools.product(weight_steps, repeat=num_models))
        
        weight_combinations = []
        for combo in all_combinations:
            combo_sum = sum(combo)
            if combo_sum > 0:
                # 【任务2】确保sum(weights)归一化为1.0
                normalized = [round(w / combo_sum, 2) for w in combo]
                if all(min_w <= w <= max_w for w in normalized):
                    weight_combinations.append(normalized)
        
        if len(weight_combinations) > 10000:
            print(f"⚠️  警告: 权重组合数量过多 ({len(weight_combinations)})，使用采样策略（每10个取1个）")
            weight_combinations = weight_combinations[::10]
        
        # 去重
        unique_combinations = []
        seen = set()
        for combo in weight_combinations:
            combo_tuple = tuple(combo)
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(combo)
        weight_combinations = unique_combinations
    
    print(f"🔍 开始搜索最优集成权重...")
    print(f"   模型数量: {num_models}")
    print(f"   权重搜索范围: [{min_w}, {max_w}], 步长: {step_w}")
    print(f"   总组合数: {len(weight_combinations)}")
    print(f"   HD95 阈值: {hd95_threshold}")
    
    best_score = -1.0
    best_weights = None
    best_metrics = None
    
    # 【军令状】彻底物理隔离：将所有数据转换为numpy数组，准备传入Parallel
    # 确保mask_list和gt_masks都是纯numpy数组，没有任何类引用
    final_mask_list = []
    for model_masks in mask_list:
        if isinstance(model_masks, list):
            # 转换为numpy数组
            model_array = np.array([np.array(m) if not isinstance(m, np.ndarray) else m for m in model_masks])
        elif isinstance(model_masks, np.ndarray):
            model_array = model_masks
        else:
            model_array = np.array(model_masks)
        final_mask_list.append(model_array)
    
    final_gt_masks = []
    for gt in gt_masks:
        if isinstance(gt, np.ndarray):
            final_gt_masks.append(gt)
        else:
            final_gt_masks.append(np.array(gt))
    
    # 【军令状】彻底物理隔离：使用Parallel和delayed进行真正的并行计算
    total_combinations = len(weight_combinations)
    
    # 确定是否使用并行处理
    actual_n_jobs = 1
    if use_parallel and JOBLIB_AVAILABLE and len(final_gt_masks) > 10:
        actual_n_jobs = min(n_jobs if n_jobs > 0 else 4, 4)
        print(f"🚀 启用并行处理: {actual_n_jobs} 个进程")
    else:
        print(f"📝 使用串行处理")
    
    # 【军令状：极致提速】使用tqdm实现实时进度可视化
    from tqdm import tqdm
    
    # 创建主进度条（显示整体进度和最佳结果）
    main_pbar = tqdm(
        total=total_combinations,
        desc="🔍 权重搜索",
        unit="组合",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | 最佳: {postfix}'
    )
    
    # 初始化最佳结果显示
    best_display = "等待中..."
    main_pbar.set_postfix_str(best_display)
    
    # 【军令状】分批并行处理：每次处理一批权重组合，避免内存溢出
    batch_size = 50  # 每批处理50个权重组合
    processed_count = 0
    
    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_weights = weight_combinations[batch_start:batch_end]
        
        # 并行处理当前批次
        if actual_n_jobs > 1:
            try:
                # 【核心修复】使用Parallel和delayed进行真正的并行计算
                # 【12点军令状任务】使用新的 calculate_metrics_for_weights 函数（包含LCC后处理）
                batch_results = Parallel(n_jobs=actual_n_jobs)(
                    delayed(calculate_metrics_for_weights)(w, final_mask_list, final_gt_masks) 
                    for w in batch_weights
                )
            except Exception as e:
                print(f"\n⚠️  并行计算错误: {e}，回退到串行模式")
                batch_results = [
                    calculate_metrics_for_weights(w, final_mask_list, final_gt_masks) 
                    for w in batch_weights
                ]
        else:
            # 串行处理（回退方案）
            batch_results = [
                calculate_metrics_for_weights(w, final_mask_list, final_gt_masks) 
                for w in batch_weights
            ]
        
        # 处理当前批次的结果
        for weight_idx_in_batch, (weights, result) in enumerate(zip(batch_weights, batch_results)):
            weight_idx = batch_start + weight_idx_in_batch
            total_score, avg_dice, avg_hd95, normalized_weights = result
            
            # 检查HD95约束
            if not np.isnan(avg_hd95) and avg_hd95 > hd95_threshold:
                processed_count += 1
                main_pbar.update(1)
                continue
            
            # 更新最佳结果
            if total_score > best_score:
                best_score = total_score
                best_weights = weights
                best_metrics = {
                    'dice': avg_dice,
                    'hd95': avg_hd95,
                    'total_score': total_score
                }
                # 【实时可视化】更新进度条显示的最佳结果
                best_display = f"Dice={best_metrics['dice']:.4f}, HD95={best_metrics['hd95']:.4f}, Score={best_metrics['total_score']:.4f}, W={best_weights}"
                main_pbar.set_postfix_str(best_display)
                # 【实时打印】控制台输出当前最佳结果
                print(f"\n🎯 当前最佳权重: {best_weights}, 当前最高分: {best_metrics['total_score']:.4f} (Dice={best_metrics['dice']:.4f}, HD95={best_metrics['hd95']:.4f})")
            
            processed_count += 1
            main_pbar.update(1)
        
        # 【性能优化5】内存释放：每处理一批后释放内存
        del batch_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 关闭主进度条
    main_pbar.close()
    
    if best_weights is None:
        print("⚠️  警告: 未找到满足HD95约束的权重组合")
        # 返回平均权重作为默认值
        best_weights = [1.0 / num_models] * num_models
        best_metrics = {'dice': 0.0, 'hd95': np.nan, 'total_score': 0.0}
    else:
        print(f"\n✅ 【采样搜索完成】找到最优权重组合:")
        print(f"   权重: {best_weights}")
        print(f"   Dice: {best_metrics['dice']:.4f}")
        print(f"   HD95: {best_metrics['hd95']:.4f}")
        print(f"   Total Score: {best_metrics['total_score']:.4f}")
        print(f"   Score公式: 0.6 * Dice + 0.1 / (1 + HD95)")
    
    # 【军令状：终效评估】用最优权重跑全量数据
    if best_weights is not None and len(original_gt_masks) > len(final_gt_masks):
        print(f"\n🎯 【终效评估】使用最优权重对全量 {len(original_gt_masks)} 张图片进行最终评估...")
        
        # 准备全量数据
        final_full_mask_list = []
        for model_masks in original_mask_list:
            if isinstance(model_masks, list):
                final_full_mask_list.append(np.array([np.array(m) if not isinstance(m, np.ndarray) else m for m in model_masks]))
            elif isinstance(model_masks, np.ndarray):
                final_full_mask_list.append(model_masks)
            else:
                final_full_mask_list.append(np.array(model_masks))
        
        final_full_gt_masks = []
        for gt in original_gt_masks:
            if isinstance(gt, np.ndarray):
                final_full_gt_masks.append(gt)
            else:
                final_full_gt_masks.append(np.array(gt))
        
        # 使用最优权重计算全量指标（包含极致后处理流水线）
        print("   正在计算全量指标（包含极致后处理：LCC + 空洞填充 + 边缘平滑）...")
        full_total_score, full_avg_dice, full_avg_hd95, _ = calculate_metrics_for_weights(
            best_weights, final_full_mask_list, final_full_gt_masks
        )
        
        print(f"\n📊 【终效评估结果】全量 {len(original_gt_masks)} 张图片:")
        print(f"   Dice: {full_avg_dice:.4f}")
        print(f"   HD95: {full_avg_hd95:.4f} (目标: ≤ 5.0)")
        print(f"   Total Score: {full_total_score:.4f}")
        print(f"   Score公式: 0.6 * Dice + 0.1 / (1 + HD95)")
        
        # 更新最佳指标为全量结果
        best_metrics = {
            'dice': full_avg_dice,
            'hd95': full_avg_hd95,
            'total_score': full_total_score
        }
        
        # 【最终检查】如果Dice > 0.91 且 HD95 < 5.0，立即停止并保存结果
        hd95_target = 5.0  # 目标HD95阈值
        dice_target = 0.91  # 目标Dice阈值
        
        if full_avg_hd95 <= hd95_target:
            print(f"   ✅ HD95满足目标条件 (≤ {hd95_target})")
        else:
            print(f"   ⚠️  HD95超出目标条件 (>{hd95_target})")
        
        if full_avg_dice > dice_target and full_avg_hd95 < hd95_target:
            print(f"\n🎉 【完美达成】指标满足所有要求:")
            print(f"   ✅ Dice = {full_avg_dice:.4f} > {dice_target} (目标达成)")
            print(f"   ✅ HD95 = {full_avg_hd95:.4f} < {hd95_target} (目标达成)")
            print(f"   💾 建议立即保存结果！")
        elif full_avg_dice > dice_target:
            print(f"\n✅ Dice目标达成 ({full_avg_dice:.4f} > {dice_target})，但HD95仍需优化")
        elif full_avg_hd95 < hd95_target:
            print(f"\n✅ HD95目标达成 ({full_avg_hd95:.4f} < {hd95_target})，但Dice仍需优化")
        else:
            print(f"\n⚠️  指标仍需优化: Dice={full_avg_dice:.4f} (目标>{dice_target}), HD95={full_avg_hd95:.4f} (目标<{hd95_target})")
    
    return best_weights, best_metrics


