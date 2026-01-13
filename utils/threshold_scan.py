# -*- coding: utf-8 -*-
"""
从utils.threshold_scan模块
"""
from utils.common import *

def scan_best_threshold(prob_maps, gt_masks):
    """
    在给定的概率图和真实掩膜上扫描阈值，寻找综合评分最高的阈值。
    
    注意：此函数在阈值优化时使用，不包含后处理（smart_post_processing/post_process_mask）。
    验证阶段会使用后处理，这可能导致阈值优化找到的阈值与验证阶段实际效果略有差异。
    但为了保持阈值优化的速度，这里不使用后处理。
    """
    # 确保输入是numpy数组
    if isinstance(prob_maps, torch.Tensor):
        prob_maps = prob_maps.detach().cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()
    
    # 统一维度
    if prob_maps.ndim == 4: prob_maps = prob_maps[:, 0]
    if gt_masks.ndim == 4: gt_masks = gt_masks[:, 0]

    # 【优化】阈值扫描范围：0.3-0.9，步长0.05（共13个阈值点）
    thresholds = np.arange(0.3, 0.95, 0.05)  # 0.95 因为 arange 是左闭右开，所以 0.95 才能包含到 0.9
    best_thresh = 0.5
    best_score = -float("inf")
    best_metrics = {}

    for thr in thresholds:
        pred_bool = (prob_maps >= thr)
        gt_bool = (gt_masks > 0.5)

        # 混淆矩阵统计 (TP, FP, FN, TN)
        tp = np.logical_and(pred_bool, gt_bool).sum()
        fp = np.logical_and(pred_bool, ~gt_bool).sum()
        fn = np.logical_and(~pred_bool, gt_bool).sum()
        tn = np.logical_and(~pred_bool, ~gt_bool).sum()

        # 【统一计算方式】计算基础指标，使用与 worker.py 一致的平滑项和边界处理
        # Dice = 2TP / (2TP + FP + FN)
        dice_den = 2.0 * tp + fp + fn
        dice = 1.0 if dice_den < 1e-7 else (2.0 * tp) / (dice_den + 1e-7)
        
        # IoU = TP / (TP + FP + FN)
        iou_den = tp + fp + fn
        iou = 1.0 if iou_den < 1e-7 else tp / (iou_den + 1e-7)
        
        # Precision = TP / (TP + FP)
        prec_den = tp + fp
        if prec_den < 1e-7:
            precision = 1.0  # 如果没有预测出任何正样本，精确率为1.0
        else:
            precision = tp / (prec_den + 1e-7)  # 使用 1e-7 与 Recall/Specificity 保持一致
        
        # Recall = TP / (TP + FN)
        rec_den = tp + fn
        if rec_den < 1e-7:
            recall = 1.0  # 如果Ground Truth为空，召回率为1.0
        else:
            recall = tp / (rec_den + 1e-7)  # 使用 1e-7 与 Precision/Specificity 保持一致
        
        # Specificity = TN / (TN + FP)
        spec_den = tn + fp
        if spec_den < 1e-7:
            specificity = 1.0  # 如果没有负样本，特异性为1.0
        else:
            specificity = tn / (spec_den + 1e-7)  # 使用 1e-7 与 Precision/Recall 保持一致

        # 计算 HD95 (需要逐样本计算取平均)
        hd95_list = []
        # 【优化】使用更多样本计算HD95以提高准确性
        # 如果样本数少于20，使用全部样本；否则使用最多20个随机样本
        sample_indices = range(len(pred_bool))
        if len(pred_bool) > 20:
            import random
            sample_indices = random.sample(sample_indices, 20)
            
        for i in sample_indices:
            hd = calculate_hd95(pred_bool[i], gt_bool[i])
            if hd < 99.0: # 过滤无效值
                hd95_list.append(hd)
        
        # 【优化】如果所有HD95值都无效，使用一个较大的默认值，但不要太大
        hd95_mean = np.mean(hd95_list) if hd95_list else 50.0

        # 计算综合得分
        # 注意：这里调用的是 utils.py 里的 calculate_custom_score
        total_score = calculate_custom_score(
            dice, iou, precision, recall, specificity, hd95_mean
        )

        if total_score > best_score:
            best_score = total_score
            best_thresh = float(thr)
            best_metrics = {
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "hd95": float(hd95_mean),
                "score": float(total_score),
            }

    return best_thresh, best_metrics
# ==================== 数据集类 ====================

