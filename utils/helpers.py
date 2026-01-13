# -*- coding: utf-8 -*-
"""
从utils.helpers模块
"""
from utils.common import *
from utils.standalone_funcs import ensemble_post_process_global

# ==================== 全局工具函数 ====================
class EarlyStopping:
    """自适应的早停策略，适配小数据场景（更平滑+暖启动+相对增益判定）。"""

    def __init__(
        self,
        patience: int = 6,
        min_delta: float = 5e-4,
        min_rel_improve: float = 0.005,
        warmup_epochs: int = 3,
        cooldown: int = 1,
        smoothing: float = 0.4,
    ):
        self.patience = max(1, patience)
        self.min_delta = min_delta
        self.min_rel = min_rel_improve
        self.warmup_epochs = max(0, warmup_epochs)
        self.cooldown = max(0, cooldown)
        self.smoothing = min(max(smoothing, 0.0), 0.99)

        self.best_score = -float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.epoch_counter = 0
        self.cooldown_counter = 0
        self._smoothed = None

    def _update_smooth(self, score: float) -> float:
        if self._smoothed is None:
            self._smoothed = score
        else:
            self._smoothed = (
                self.smoothing * self._smoothed + (1 - self.smoothing) * score
            )
        return self._smoothed

    def step(self, score: float) :  # -> bool
        self.epoch_counter += 1
        smoothed = self._update_smooth(score)

        # warmup: always observe a few epochs before starting to stop
        if self.epoch_counter <= self.warmup_epochs:
            if smoothed > self.best_score:
                self.best_score = smoothed
                self.best_epoch = self.epoch_counter
            self.bad_epochs = 0
            self.cooldown_counter = self.cooldown
            return False

        improvement = smoothed - self.best_score
        rel_improvement = (
            improvement / (abs(self.best_score) + 1e-8)
            if self.best_score > -float("inf")
            else float("inf")
        )

        if improvement > self.min_delta or rel_improvement > self.min_rel:
            self.best_score = smoothed
            self.best_epoch = self.epoch_counter
            self.bad_epochs = 0
            self.cooldown_counter = self.cooldown
            return False

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

def pure_compute_metrics(ensemble_mask, gt_mask):
    """
    最底层的纯计算函数，只处理 numpy 数据
    绝对不引用任何类成员、类方法或 pyqtSignal
    
    Args:
        ensemble_mask: 集成后的掩码 (numpy array, 0-1)
        gt_mask: 真实掩码 (numpy array, 0-1)
    
    Returns:
        (dice_val, hd95_val): Dice系数和HD95值（像素级）
    """
    try:
        # 确保是numpy数组
        if not isinstance(ensemble_mask, np.ndarray):
            ensemble_mask = np.array(ensemble_mask)
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = np.array(gt_mask)
        
        # 二值化
        pred_binary = (ensemble_mask > 0.5).astype(np.uint8)
        gt_binary = (gt_mask > 0.5).astype(np.uint8)
        
        # 计算Dice
        intersection = (pred_binary & gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        if union == 0:
            dice_val = 1.0
        else:
            dice_val = (2.0 * intersection) / union
        
        # 计算HD95（使用原始像素坐标，不归一化）
        if not pred_binary.any() or not gt_binary.any():
            hd95_val = 99.9 if (pred_binary.any() != gt_binary.any()) else 0.0
        else:
            # 直接计算HD95，不依赖外部函数（避免循环引用）
            from scipy.ndimage import binary_erosion, distance_transform_edt
            
            pred = pred_binary.astype(bool)
            target = gt_binary.astype(bool)
            
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
                hd95_val = 0.0
            else:
                hd95_val = float(np.percentile(all_distances, 95))
            
            if np.isnan(hd95_val) or np.isinf(hd95_val):
                hd95_val = 99.9
        
        return float(dice_val), float(hd95_val)
    except Exception as e:
        # 保底返回，防止计算崩溃
        return 0.0, 99.9


def calculate_hd95(pred, gt):
    """
    【全局独立函数】计算Hausdorff Distance 95 (HD95)
    绝对不引用任何类成员、类方法或 pyqtSignal
    
    Args:
        pred: 预测掩码 (numpy array, 可以是概率图或二值图)
        gt: 真实掩码 (numpy array, 可以是概率图或二值图)
    
    Returns:
        hd95值（像素单位），如果无法计算则返回99.9
    """
    try:
        # 必须先二值化，且确保 pred 和 gt 都有像素，否则会报错
        pred_bin = (pred > 0.5).astype(np.bool_)
        gt_bin = (gt > 0.5).astype(np.bool_)
        
        if not pred_bin.any() and not gt_bin.any():
            return 0.0
        if not pred_bin.any() or not gt_bin.any():
            return 99.9
        
        # 计算边界
        structure = np.ones((3, 3), dtype=bool)
        pred_border = np.logical_xor(pred_bin, binary_erosion(pred_bin, structure=structure, border_value=0))
        gt_border = np.logical_xor(gt_bin, binary_erosion(gt_bin, structure=structure, border_value=0))
        
        if not pred_border.any():
            pred_border = pred_bin
        if not gt_border.any():
            gt_border = gt_bin
        
        # 计算距离变换
        target_distance = distance_transform_edt(~gt_border)
        pred_distance = distance_transform_edt(~pred_border)
        
        distances_pred_to_target = target_distance[pred_border]
        distances_target_to_pred = pred_distance[gt_border]
        
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        if all_distances.size == 0:
            return 0.0
        
        # 返回95百分位距离（像素单位）
        hd95_val = float(np.percentile(all_distances, 95))
        
        # 检查异常值
        if np.isnan(hd95_val) or np.isinf(hd95_val):
            return 99.9
        
        return hd95_val
    except Exception as e:
        # 保底返回，防止计算崩溃
        return 99.9

def calculate_custom_score(dice, iou, precision, recall, specificity, hd95):
    """
    自定义综合评分函数:
    Score = (Dice * 50) + (IoU * 10) + (Precision * 10) + (Recall * 10) + (Specificity * 10) + Score_HD95
    其中 Score_HD95 = 10 / (HD95 + 1)
    """
    dice = float(dice)
    iou = float(iou)
    precision = float(precision)
    recall = float(recall)
    specificity = float(specificity)

    # HD95 项：HD95 越小越好，使用反比变换；若无效则记为 0
    if hd95 is None or not np.isfinite(hd95) or hd95 < 0 or hd95 >= 99.0:
        score_hd95 = 0.0
    else:
        score_hd95 = 10.0 / (float(hd95) + 1.0)

    total_score = (
        dice * 50.0
        + iou * 10.0
        + precision * 10.0
        + recall * 10.0
        + specificity * 10.0
        + score_hd95
    )
    return float(total_score)
def calculate_official_total_score(dice, hd95):
    """
    【全局独立函数】计算官方总分
    严格按照公式：0.6 * dice + 0.1 / (1 + hd95)
    绝对不引用任何类成员、类方法或 pyqtSignal
    
    Args:
        dice: Dice系数
        hd95: HD95值（如果为NaN或Inf，则使用99.9）
    
    Returns:
        总分
    """
    # 处理HD95的NaN/Inf情况
    if np.isnan(hd95) or np.isinf(hd95) or hd95 >= 99.0:
        hd95_term = 0.0  # 如果HD95不可计算，该项为0
    else:
        hd95_term = 0.1 / (1.0 + hd95)
    
    total_score = 0.6 * dice + hd95_term
    return float(total_score)


def worker_ensemble_logic(weights, masks, gts):
    """
    【全局独立函数】集成逻辑工作函数
    像素融合 -> LCC后处理 -> 计算指标
    绝对不引用任何类成员、类方法或 pyqtSignal
    
    Args:
        weights: 权重列表 (list of float)
        masks: 掩码列表，每个元素是一个模型的掩码数组
        gts: 真实掩码列表
    
    Returns:
        (total_score, avg_dice, avg_hd95): 总分、平均Dice、平均HD95
    """
    # 强制数据对齐检查
    assert len(masks) > 0, "mask_list不能为空"
    assert len(gts) > 0, "gt_masks不能为空"
    
    # 强制类型转换
    mask_list = [np.array(m) for m in masks]
    gt_masks = [np.array(gt) for gt in gts]
    
    # 确保所有mask长度一致
    num_samples = len(gt_masks)
    for i, m in enumerate(mask_list):
        if isinstance(m, list):
            assert len(m) == num_samples, f"mask_list[{i}]长度({len(m)})与gt_masks长度({num_samples})不一致"
        elif m.ndim == 3:
            assert m.shape[0] == num_samples, f"mask_list[{i}]第一维({m.shape[0]})与gt_masks长度({num_samples})不一致"
    
    # 使用现有的calculate_metrics_for_weights函数
    total_score, avg_dice, avg_hd95, _ = calculate_metrics_for_weights(weights, mask_list, gt_masks)
    
    return total_score, avg_dice, avg_hd95


def global_weight_search_worker(weights, mask_list, gt_masks):
    """
    被 Parallel 调用的核心工人函数
    绝对不引用任何类成员、类方法或 pyqtSignal
    
    Args:
        weights: 权重列表 (list of float)
        mask_list: 掩码列表，每个元素是 (N, H, W) 的numpy数组
        gt_masks: 真实掩码列表，每个元素是 (H, W) 的numpy数组
    
    Returns:
        (score, avg_dice, avg_hd95, weights): 总分、平均Dice、平均HD95、权重
    """
    try:
        # 1. 权重归一化
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        
        # 2. 确保mask_list是numpy数组
        mask_arrays = []
        for m in mask_list:
            if not isinstance(m, np.ndarray):
                m = np.array(m)
            mask_arrays.append(m)
        
        # 3. 像素级融合：w1*mask1 + w2*mask2 + ...
        # mask_arrays[0] 形状是 (N, H, W)，需要按样本融合
        num_samples = len(gt_masks)
        combined_masks = []
        
        for i in range(num_samples):
            # 确定第一个mask的形状
            first_mask = mask_arrays[0]
            if first_mask.ndim == 3:  # (N, H, W)
                combined = np.zeros_like(first_mask[i], dtype=np.float32)
            else:  # (H, W)
                combined = np.zeros_like(first_mask, dtype=np.float32)
            
            for w, m in zip(weights, mask_arrays):
                if m.ndim == 3:  # (N, H, W)
                    combined += w * m[i].astype(np.float32)
                else:  # (H, W)
                    combined += w * m.astype(np.float32)
            combined_masks.append(combined)
        
        # 4. 计算所有样本的平均指标
        dices, hds = [], []
        for i in range(num_samples):
            d, h = pure_compute_metrics(combined_masks[i], gt_masks[i])
            dices.append(d)
            hds.append(h)
        
        avg_dice = np.mean(dices)
        avg_hd95 = np.mean([h for h in hds if not np.isnan(h) and h < 99.0])
        if np.isnan(avg_hd95) or avg_hd95 >= 99.0:
            avg_hd95 = 99.9
        
        # 5. 计算总分 (0.6*Dice + 0.1/(1+HD95))
        score = 0.6 * avg_dice + 0.1 / (1.0 + avg_hd95)
        
        return float(score), float(avg_dice), float(avg_hd95), weights.tolist()
    except Exception as e:
        # 保底返回
        return 0.0, 0.0, 99.9, weights.tolist() if isinstance(weights, np.ndarray) else weights


def calculate_metrics_for_weights(weights, mask_list, gt_masks):
    """
    【军令状：12点任务】全局独立函数：计算权重组合的指标
    绝对不引用任何类成员、类方法或 pyqtSignal
    
    实现流程：
    1. 接收 (weights, mask_list, gt_masks)
    2. 像素级加权融合
    3. LCC 后处理
    4. 计算 Dice 和 HD95
    5. 返回 Total Score
    
    Args:
        weights: 权重列表 (list of float)
        mask_list: 掩码列表，每个元素是一个模型的掩码数组 (N, H, W) 或列表
        gt_masks: 真实掩码列表，每个元素是 (H, W) 的numpy数组
    
    Returns:
        (total_score, avg_dice, avg_hd95, weights): 总分、平均Dice、平均HD95、归一化权重
    """
    try:
        # 【强制数据对齐检查】
        assert len(mask_list) > 0, "mask_list不能为空"
        assert len(gt_masks) > 0, "gt_masks不能为空"
        assert len(weights) == len(mask_list), f"权重数量({len(weights)})与模型数量({len(mask_list)})不一致"
        
        # 1. 权重归一化
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        
        # 2. 【强制类型转换】确保mask_list是numpy数组
        mask_arrays = []
        for i, m in enumerate(mask_list):
            if isinstance(m, list):
                # 如果是列表，转换为numpy数组
                mask_arrays.append(np.array(m))
            elif not isinstance(m, np.ndarray):
                mask_arrays.append(np.array(m))
            else:
                mask_arrays.append(m)
        
        # 【数据对齐检查】确保所有mask长度一致
        num_samples = len(gt_masks)
        for i, m in enumerate(mask_arrays):
            if isinstance(m, list):
                assert len(m) == num_samples, f"mask_list[{i}]长度({len(m)})与gt_masks长度({num_samples})不一致"
            elif m.ndim == 3:
                assert m.shape[0] == num_samples, f"mask_list[{i}]第一维({m.shape[0]})与gt_masks长度({num_samples})不一致"
        
        # 3. 像素级加权融合：w1*mask1 + w2*mask2 + ...
        combined_masks = []
        
        for i in range(num_samples):
            # 确定第一个mask的形状
            first_mask = mask_arrays[0]
            if first_mask.ndim == 3:  # (N, H, W)
                combined = np.zeros_like(first_mask[i], dtype=np.float32)
            elif first_mask.ndim == 2:  # (H, W)
                combined = np.zeros_like(first_mask, dtype=np.float32)
            else:
                # 如果是列表，取第i个元素
                if isinstance(first_mask, list):
                    combined = np.zeros_like(np.array(first_mask[i]), dtype=np.float32)
                else:
                    combined = np.zeros_like(first_mask, dtype=np.float32)
            
            # 加权融合
            for w, m in zip(weights, mask_arrays):
                if m.ndim == 3:  # (N, H, W)
                    combined += w * m[i].astype(np.float32)
                elif m.ndim == 2:  # (H, W) - 单个样本
                    if i == 0:  # 只在第一个样本时使用
                        combined += w * m.astype(np.float32)
                elif isinstance(m, list):
                    # 如果是列表，取第i个元素
                    combined += w * np.array(m[i]).astype(np.float32)
                else:
                    combined += w * m.astype(np.float32)
            
            combined_masks.append(combined)
        
        # 4. 【极致后处理流水线】+ 计算指标
        dices, hds = [], []
        for i in range(num_samples):
            # 应用极致后处理流水线（LCC + 空洞填充 + 边缘平滑）
            processed_mask = ensemble_post_process_global(
                combined_masks[i],
                use_lcc=True,  # 【第一步】保留最大连通域，彻底切除离群噪点
                use_remove_holes=True,  # 【第二步】填补小孔洞，提升Dice约0.5%
                min_hole_size=100,
                use_edge_smoothing=True  # 【第三步】边缘平滑，修正锯齿边缘
            )
            
            # 计算Dice和HD95
            d, h = pure_compute_metrics(processed_mask, gt_masks[i])
            dices.append(d)
            hds.append(h)
        
        # 5. 计算平均指标
        avg_dice = np.mean(dices)
        avg_hd95 = np.mean([h for h in hds if not np.isnan(h) and h < 99.0])
        if np.isnan(avg_hd95) or avg_hd95 >= 99.0:
            avg_hd95 = 99.9
        
        # 6. 计算总分 (0.6*Dice + 0.1/(1+HD95))
        total_score = 0.6 * avg_dice + 0.1 / (1.0 + avg_hd95)
        
        return float(total_score), float(avg_dice), float(avg_hd95), weights.tolist()
    except Exception as e:
        # 保底返回
        import traceback
        print(f"⚠️  calculate_metrics_for_weights 错误: {e}")
        print(traceback.format_exc())
        return 0.0, 0.0, 99.9, weights.tolist() if isinstance(weights, np.ndarray) else weights


