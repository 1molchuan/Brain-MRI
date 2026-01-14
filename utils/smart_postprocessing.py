from utils.common import *


def _to_numpy(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """安全地将tensor/ndarray转换为2D numpy float32 数组 (H, W, 0-1)."""
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    mask_np = mask_np.astype(np.float32)
    # 如果是logits/概率图，限制到[0,1]
    if mask_np.max() > 1.0 or mask_np.min() < 0.0:
        mask_np = 1.0 / (1.0 + np.exp(-mask_np))  # 简单sigmoid
    return mask_np


def _apply_strategy(mask_prob_np: np.ndarray, method: str, params: Dict) -> np.ndarray:
    """
    在CPU上应用简单后处理策略。
    
    Args:
        mask_prob_np: 概率图 (H, W), 0-1
        method: 'baseline' | 'lcc' | 'remove_small'
        params: 具体参数, 如 {'min_size': 100}
    """
    # 先二值化为0/1
    binary = (mask_prob_np > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(np.float32)
    
    if method == "baseline":
        return binary.astype(np.float32)
    
    # 使用scipy的连通域分析
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return binary.astype(np.float32)
    
    if method == "lcc":
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        return (labeled == largest_label).astype(np.float32)
    
    if method == "remove_small":
        min_size = int(params.get("min_size", 0))
        if min_size <= 0:
            return binary.astype(np.float32)
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        mask_filtered = np.zeros_like(binary, dtype=np.uint8)
        for i, size in enumerate(sizes, start=1):
            if size >= min_size:
                mask_filtered[labeled == i] = 1
        return mask_filtered.astype(np.float32)
    
    # 未知策略则退回baseline
    return binary.astype(np.float32)


def _dice_from_np(pred_np: np.ndarray, gt_np: np.ndarray, smooth: float = 1e-7) -> float:
    """使用与训练阶段一致的Dice定义，在CPU上计算单样本Dice。只计算前景类。"""
    pred_bin = (pred_np > 0.5).astype(np.float32).ravel()
    gt_bin = (gt_np > 0.5).astype(np.float32).ravel()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    intersection = float((pred_bin * gt_bin).sum())
    # 空掩码特判逻辑，保持与训练/评估一致
    if gt_sum <= smooth and pred_sum <= smooth:
        return 1.0
    if gt_sum <= smooth or pred_sum <= smooth:
        return 0.0
    return float((2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth))


def find_optimal_postprocessing_strategy(val_loader, model, device, use_tta: bool = True) -> Dict:
    """
    在验证集上搜索最优后处理策略（数据驱动，自适应）。
    
    搜索空间:
      - baseline: 仅使用模型输出 (可选TTA)，直接二值化
      - lcc: 在baseline基础上保留最大连通域
      - remove_small(k): 在baseline基础上移除面积 < k 的小区域, k ∈ {10, 30, 100, 300}
    """
    import torch.nn.functional as F
    from tqdm import tqdm
    
    model.eval()
    strategies = [{"method": "baseline", "params": {}},
                  {"method": "lcc", "params": {}},]
    for ms in [10, 30, 100, 300]:
        strategies.append({"method": "remove_small", "params": {"min_size": ms}})
    
    # 累积每种策略的Dice
    dice_sums = [0.0 for _ in strategies]
    dice_counts = [0 for _ in strategies]
    
    # 简单TTA：优先使用已有的_tta_inference（如果在TrainThread中有），否则使用原始预测
    def _forward_with_tta_local(images_tensor: torch.Tensor) -> torch.Tensor:
        try:
            # 训练线程/测试线程里已有 _tta_inference 实现，这里尽量复用
            if hasattr(model, "_tta_inference"):
                logits = model._tta_inference(images_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits
        except Exception:
            pass
        # 默认：无TTA，直接前向
        logits = model(images_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[SmartPost] 搜索后处理策略"):
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch
            images = images.to(device)
            masks = masks.to(device).float()
            
            # 获得概率图
            if use_tta:
                logits = _forward_with_tta_local(images)
            else:
                logits = model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
            if logits.shape[2:] != masks.shape[2:]:
                logits = F.interpolate(logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
            probs = torch.sigmoid(logits)
            
            # 遍历batch中每个样本，在CPU上应用不同策略，计算Dice
            b = probs.shape[0]
            for i in range(b):
                prob_i = _to_numpy(probs[i, 0])
                gt_i = _to_numpy(masks[i, 0])
                for idx, strat in enumerate(strategies):
                    processed = _apply_strategy(prob_i, strat["method"], strat["params"])
                    d = _dice_from_np(processed, gt_i)
                    dice_sums[idx] += d
                    dice_counts[idx] += 1
    
    # 汇总结果
    best_idx = 0
    best_dice = -1.0
    for idx, strat in enumerate(strategies):
        if dice_counts[idx] == 0:
            continue
        avg_d = dice_sums[idx] / max(1, dice_counts[idx])
        strategies[idx]["avg_dice"] = float(avg_d)
        if avg_d > best_dice:
            best_dice = avg_d
            best_idx = idx
    
    best_config = strategies[best_idx]
    best_config.setdefault("avg_dice", float(best_dice))
    return best_config


