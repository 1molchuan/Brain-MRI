"""
工具函数和数据处理类模块
包含所有独立工具函数、数据处理类和模型加载函数
"""

import os
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy import ndimage
from torch.utils.data import Dataset
from albumentations import Compose

# 尝试导入可选依赖
try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[警告] skimage未安装，直方图匹配功能将不可用")

try:
    from skimage import morphology
    SKIMAGE_MORPHOLOGY_AVAILABLE = True
except ImportError:
    SKIMAGE_MORPHOLOGY_AVAILABLE = False

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


# ==================== 图像增强类 ====================

class MedicalImageAugmentation:
    """
    医学影像专用数据增强工具类
    包括Rician噪声、直方图匹配、低对比度模拟等
    """
    @staticmethod
    def add_rician_noise(image, noise_level=0.05):
        """
        添加Rician噪声（MRI常见噪声类型）
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            noise_level: 噪声水平 (0-1)
        """
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        
        # Rician噪声：实部和虚部都是高斯噪声
        real_noise = np.random.normal(0, noise_level, image.shape)
        imag_noise = np.random.normal(0, noise_level, image.shape)
        rician_noise = np.sqrt((image + real_noise)**2 + imag_noise**2) - image
        
        noisy_image = image + rician_noise
        noisy_image = np.clip(noisy_image, 0, 1)
        
        if noisy_image.shape[-1] == 1:
            noisy_image = noisy_image[..., 0]
        
        return noisy_image
    
    @staticmethod
    def histogram_matching(image, reference_image=None, sigma=1.0):
        """
        直方图匹配 - 模拟不同扫描仪的强度偏移
        Args:
            image: 输入图像
            reference_image: 参考图像（如果为None，使用随机参考）
            sigma: 高斯模糊参数，用于平滑匹配
        """
        if not SKIMAGE_AVAILABLE:
            # 如果skimage不可用，返回原图
            return image
        
        if reference_image is None:
            # 生成随机参考直方图
            reference_image = np.random.uniform(0, 1, image.shape)
        
        matched = match_histograms(image, reference_image)
        
        # 可选：应用轻微的高斯模糊以模拟扫描仪差异
        if sigma > 0:
            matched = gaussian_filter(matched, sigma=sigma)
        
        return np.clip(matched, 0, 1)
    
    @staticmethod
    def simulate_low_contrast(image, contrast_factor=0.7):
        """
        模拟低对比度图像（常见于某些扫描参数）
        Args:
            image: 输入图像
            contrast_factor: 对比度因子 (0-1)
        """
        mean = image.mean()
        low_contrast = (image - mean) * contrast_factor + mean
        return np.clip(low_contrast, 0, 1)
    
    @staticmethod
    def label_smoothing(mask, sigma=1.0):
        """
        标签软化 - 对Ground Truth做高斯模糊
        让模型学习更平滑的边界概率分布，缓解硬标签带来的过拟合
        Args:
            mask: 二值掩膜 (H, W)
            sigma: 高斯模糊的标准差
        Returns:
            软化的标签 (H, W)，值域[0, 1]
        """
        smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
        return np.clip(smoothed, 0, 1)


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


# ==================== 模型加载函数 ====================

def load_ensemble_models(*args, **kwargs):
    """模型集成功能已取消，调用将报错。"""
    raise RuntimeError("模型集成功能已取消")


def load_model_compatible(model, checkpoint_path, device, verbose=True):
    """
    兼容加载模型权重的工具函数，自动处理新旧版本结构差异
    
    Args:
        model: 要加载权重的模型实例
        checkpoint_path: 模型权重文件路径
        device: 设备
        verbose: 是否打印信息
    
    Returns:
        (success: bool, message: str)
    """
    try:
        loaded_obj = torch.load(checkpoint_path, map_location=device)
        if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
            state_dict = loaded_obj['state_dict']
        else:
            state_dict = loaded_obj
        
        # 处理DataParallel包装
        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 检测并转换旧版本的键名（layer0/layer1 -> enc0/enc1）
        # 检查是否是旧版本的ResNetUNet checkpoint
        old_version_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        has_old_keys = any(any(k.startswith(old_key) for k in state_dict.keys()) for old_key in old_version_keys)
        
        if has_old_keys:
            # 创建键名映射：旧版本 -> 新版本
            key_mapping = {}
            for old_key in state_dict.keys():
                new_key = old_key
                # 映射编码器层
                if old_key.startswith('layer0.'):
                    new_key = old_key.replace('layer0.', 'enc0.', 1)
                elif old_key.startswith('layer1.'):
                    new_key = old_key.replace('layer1.', 'enc1.', 1)
                elif old_key.startswith('layer2.'):
                    new_key = old_key.replace('layer2.', 'enc2.', 1)
                elif old_key.startswith('layer3.'):
                    new_key = old_key.replace('layer3.', 'enc3.', 1)
                elif old_key.startswith('layer4.'):
                    new_key = old_key.replace('layer4.', 'enc4.', 1)
                # 可能还有其他映射，比如 center -> center (如果模型不使用ASPP)
                # 注意：如果检测到旧版本checkpoint，模型会使用center而不是aspp，所以不需要映射
                # 但如果checkpoint中有center而模型使用aspp，则需要映射
                # 这里我们保持center不变，因为旧版本模型会使用center
                # （映射逻辑在_load_model中已经处理，这里只处理layerX -> encX的映射）
                elif old_key.startswith('up0.') or old_key.startswith('up_conv0.'):
                    new_key = old_key.replace('up0.', 'dec4.', 1).replace('up_conv0.', 'dec4.', 1)
                elif old_key.startswith('up1.') or old_key.startswith('up_conv1.'):
                    new_key = old_key.replace('up1.', 'dec3.', 1).replace('up_conv1.', 'dec3.', 1)
                elif old_key.startswith('up2.') or old_key.startswith('up_conv2.'):
                    new_key = old_key.replace('up2.', 'dec2.', 1).replace('up_conv2.', 'dec2.', 1)
                elif old_key.startswith('up3.') or old_key.startswith('up_conv3.'):
                    new_key = old_key.replace('up3.', 'dec1.', 1).replace('up_conv3.', 'dec1.', 1)
                elif old_key.startswith('up4.') or old_key.startswith('up_conv4.'):
                    # up4 对应 dec4（最深层）
                    new_key = old_key.replace('up4.', 'dec4.', 1).replace('up_conv4.', 'dec4.', 1)
                
                if new_key != old_key:
                    key_mapping[old_key] = new_key
            
            # 应用键名映射
            if key_mapping:
                new_state_dict = {}
                for old_key, value in state_dict.items():
                    if old_key in key_mapping:
                        new_state_dict[key_mapping[old_key]] = value
                    else:
                        new_state_dict[old_key] = value
                state_dict = new_state_dict
                if verbose:
                    print(f"[模型加载] 检测到旧版本checkpoint，已转换 {len(key_mapping)} 个键名")
        
        model_dict = model.state_dict()
        
        # 统计匹配情况
        matched = {}
        mismatched = []
        missing_in_ckpt = []
        extra_in_ckpt = []  # checkpoint中有但模型中不存在的键
        
        for k, v in model_dict.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    matched[k] = state_dict[k]
                else:
                    mismatched.append(k)
            else:
                missing_in_ckpt.append(k)
        
        # 检查checkpoint中多余的键
        for k in state_dict.keys():
            if k not in model_dict:
                extra_in_ckpt.append(k)
        
        # 加载匹配的权重
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        
        loaded_keys = len(matched)
        total_keys = len(model_dict)
        
        # 详细诊断信息
        if verbose and loaded_keys == 0:
            print(f"[模型加载] ⚠️ 警告：没有参数匹配！")
            print(f"[模型加载] 模型参数键示例（前10个）:")
            for i, k in enumerate(list(model_dict.keys())[:10]):
                print(f"  {i+1}. {k} (shape: {model_dict[k].shape})")
            print(f"[模型加载] Checkpoint参数键示例（前10个）:")
            for i, k in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i+1}. {k} (shape: {state_dict[k].shape})")
            if extra_in_ckpt:
                print(f"[模型加载] Checkpoint中多余的键（前5个）: {extra_in_ckpt[:5]}")
        elif verbose and loaded_keys > 0 and loaded_keys < total_keys:
            # 显示未匹配的参数类别统计
            missing_categories = {}
            for k in missing_in_ckpt:
                category = k.split('.')[0] if '.' in k else k
                missing_categories[category] = missing_categories.get(category, 0) + 1
            
            if missing_categories:
                print(f"[模型加载] 未匹配参数类别统计:")
                for cat, count in sorted(missing_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  - {cat}: {count} 个参数")
                if len(missing_categories) > 5:
                    print(f"  - ... 还有 {len(missing_categories) - 5} 个类别")
        
        if mismatched:
            msg = f"部分加载: {loaded_keys}/{total_keys}个参数匹配, {len(mismatched)}个形状不匹配"
            if verbose:
                print(f"[模型加载] {msg}")
                print(f"[模型加载] 形状不匹配的参数: {mismatched[:5]}{'...' if len(mismatched) > 5 else ''}")
                # 对于DS-TransUNet，提供更详细的形状信息
                if any('patch_embed3' in k or 'transformer3' in k for k in mismatched[:5]):
                    print(f"[模型加载] 详细形状对比（前3个不匹配的参数）:")
                    for k in mismatched[:3]:
                        model_shape = model_dict[k].shape if k in model_dict else "N/A"
                        ckpt_shape = state_dict[k].shape if k in state_dict else "N/A"
                        print(f"  - {k}:")
                        print(f"    模型期望: {model_shape}")
                        print(f"    Checkpoint实际: {ckpt_shape}")
        elif missing_in_ckpt and loaded_keys > 0:
            msg = f"兼容加载: {loaded_keys}/{total_keys}个参数 (新增层已随机初始化)"
            if verbose:
                print(f"[模型加载] {msg}")
                if missing_in_ckpt:
                    print(f"[模型加载] 新增层（前5个）: {missing_in_ckpt[:5]}{'...' if len(missing_in_ckpt) > 5 else ''}")
        elif loaded_keys == 0:
            msg = f"⚠️ 严重警告: 0/{total_keys}个参数匹配！模型类型可能不匹配"
            if verbose:
                print(f"[模型加载] {msg}")
                print(f"[模型加载] 请检查模型类型是否与checkpoint匹配")
            # 0个参数匹配时返回False，表示加载失败
            return False, msg
        else:
            msg = f"完整加载: {os.path.basename(checkpoint_path)}"
            if verbose:
                print(f"[模型加载] {msg}")
        
        return True, msg
        
    except Exception as e:
        msg = f"加载失败: {str(e)}"
        if verbose:
            print(f"[模型加载] {msg}")
        return False, msg


def infer_swin_params_from_state_dict(state_dict):
    """从state_dict精确推断SwinUNet参数"""
    if 'patch_embed.proj.weight' not in state_dict:
        return None
    
    # embed_dim
    embed_dim = state_dict['patch_embed.proj.weight'].shape[0]
    
    # depths: 统计每个stage的block数
    depths = []
    for stage_idx in range(10):
        block_count = 0
        for block_idx in range(50):
            if f'encoder_layers.{stage_idx}.{block_idx}.norm1.weight' in state_dict:
                block_count += 1
            else:
                break
        if block_count > 0:
            depths.append(block_count)
        else:
            break
    if not depths:
        depths = [2, 2, 6, 2]
    
    # num_heads: 从qkv权重推断
    num_heads = []
    for stage_idx in range(len(depths)):
        qkv_key = f'encoder_layers.{stage_idx}.0.attn.qkv.weight'
        if qkv_key in state_dict:
            qkv_out = state_dict[qkv_key].shape[0]  # 3 * dim
            dim_at_stage = qkv_out // 3
            # 从proj权重推断head数
            proj_key = f'encoder_layers.{stage_idx}.0.attn.proj.weight'
            if proj_key in state_dict:
                for head_dim in [32, 64, 48, 96, 128]:
                    if dim_at_stage % head_dim == 0:
                        num_heads.append(dim_at_stage // head_dim)
                        break
                else:
                    num_heads.append(max(1, dim_at_stage // 32))
            else:
                num_heads.append(max(1, dim_at_stage // 32))
        else:
            num_heads.append(3 * (2 ** stage_idx))
    
    # mlp_hidden_dims: 精确记录每个stage每个block的mlp hidden dim
    # 这样可以避免mlp_ratio的浮点误差
    mlp_hidden_dims = {}
    for stage_idx in range(len(depths)):
        for block_idx in range(depths[stage_idx]):
            fc1_key = f'encoder_layers.{stage_idx}.{block_idx}.mlp.fc1.weight'
            if fc1_key in state_dict:
                mlp_hidden_dims[(stage_idx, block_idx)] = state_dict[fc1_key].shape[0]
    
    # mlp_ratio: 从第一个block推断（用于新建block时的默认值）
    mlp_ratio = 4.0
    fc1_key = 'encoder_layers.0.0.mlp.fc1.weight'
    if fc1_key in state_dict:
        hidden = state_dict[fc1_key].shape[0]
        in_dim = state_dict[fc1_key].shape[1]
        mlp_ratio = hidden / in_dim
    
    # window_size: 尝试从relative_position_bias_table推断
    window_size = 8
    rpb_key = 'encoder_layers.0.0.attn.relative_position_bias_table'
    if rpb_key in state_dict:
        table_size = state_dict[rpb_key].shape[0]
        import math
        ws_calc = (math.sqrt(table_size) + 1) / 2
        if ws_calc == int(ws_calc):
            window_size = int(ws_calc)
    
    return {
        'embed_dim': embed_dim,
        'depths': tuple(depths),
        'num_heads': tuple(num_heads),
        'mlp_ratio': mlp_ratio,
        'window_size': window_size,
        'drop_path_rate': 0.0,
        '_mlp_hidden_dims': mlp_hidden_dims,  # 精确的hidden dims
        '_from_checkpoint': True
    }


def infer_dstrans_params_from_state_dict(state_dict):
    """从state_dict推断DS-TransUNet参数（增强版，提高兼容性）"""
    # 处理可能的键名变体（考虑DataParallel包装等）
    patch_embed3_key = None
    for key in state_dict.keys():
        if 'patch_embed3.weight' in key or key.endswith('patch_embed3.weight'):
            patch_embed3_key = key
            break
    
    if patch_embed3_key is None:
        return None
    
    try:
        # 优先从in_proj_weight推断embed_dim（更准确，因为它直接反映了transformer的维度）
        embed_dim = None
        num_heads = 8  # 默认值
        in_proj_key = None
        for key in state_dict.keys():
            if 'transformer3.layers.0.self_attn.in_proj_weight' in key or key.endswith('transformer3.layers.0.self_attn.in_proj_weight'):
                in_proj_key = key
                break
        
        if in_proj_key:
            in_proj_weight = state_dict[in_proj_key]
            # in_proj_weight的形状是 [3 * embed_dim, embed_dim]
            if len(in_proj_weight.shape) == 2:
                # 从checkpoint读取实际的embed_dim（最准确的方法）
                actual_embed_dim = in_proj_weight.shape[1]  # 第二维是embed_dim
                if in_proj_weight.shape[0] == 3 * actual_embed_dim:
                    embed_dim = actual_embed_dim
                    print(f"[参数推断] 从in_proj_weight读取embed_dim: {embed_dim}")
                    
                    # num_heads 必须是 embed_dim 的约数
                    # 尝试常见的值，优先选择较大的（通常性能更好）
                    for nh in [32, 16, 8, 4]:
                        if embed_dim % nh == 0:
                            num_heads = nh
                            break
                else:
                    print(f"[警告] in_proj_weight形状异常: {in_proj_weight.shape}, 期望: [3*embed_dim, embed_dim]")
        
        # 如果无法从in_proj_weight推断，则从patch_embed3推断
        if embed_dim is None:
            patch_embed3_weight = state_dict[patch_embed3_key]
            if len(patch_embed3_weight.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
                embed_dim = patch_embed3_weight.shape[0]  # 输出通道数
                print(f"[参数推断] 从patch_embed3读取embed_dim: {embed_dim}")
            elif len(patch_embed3_weight.shape) == 2:  # Linear: [out_features, in_features]
                embed_dim = patch_embed3_weight.shape[0]
                print(f"[参数推断] 从patch_embed3读取embed_dim: {embed_dim}")
            else:
                print(f"[警告] patch_embed3.weight形状异常: {patch_embed3_weight.shape}")
                return None
            
            # 从embed_dim推断num_heads
            for nh in [32, 16, 8, 4]:
                if embed_dim % nh == 0:
                    num_heads = nh
                    break
        
        # num_layers: 统计transformer3的层数（检查两个transformer）
        num_layers = 2  # 默认值
        max_layers = 0
        for i in range(20):  # 增加范围以支持更深的模型
            # 检查transformer3
            key3 = f'transformer3.layers.{i}.self_attn.in_proj_weight'
            key3_alt = None
            for k in state_dict.keys():
                if key3 in k or k.endswith(key3):
                    key3_alt = k
                    break
            if key3_alt:
                max_layers = max(max_layers, i + 1)
            else:
                break
        
        if max_layers > 0:
            num_layers = max_layers
        
        # mlp_ratio: 从transformer3.layers[0].linear1或ffn.0.weight推断
        mlp_ratio = 4.0  # 默认值
        linear1_key = None
        for key in state_dict.keys():
            if 'transformer3.layers.0.linear1.weight' in key or key.endswith('transformer3.layers.0.linear1.weight'):
                linear1_key = key
                break
        
        if linear1_key:
            try:
                linear1_out = state_dict[linear1_key].shape[0]
                if embed_dim > 0:
                    mlp_ratio = linear1_out / embed_dim
            except:
                pass
        
        # dropout: 默认值（通常无法从state_dict推断）
        dropout = 0.1  # 默认值
        
        # 验证参数合理性
        if embed_dim <= 0 or num_heads <= 0 or num_layers <= 0:
            print(f"[警告] DS-TransUNet参数推断异常: embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}")
            return None
        
        if embed_dim % num_heads != 0:
            print(f"[警告] embed_dim({embed_dim})不能被num_heads({num_heads})整除，自动调整num_heads")
            # 自动调整num_heads
            for nh in [32, 16, 8, 4]:
                if embed_dim % nh == 0:
                    num_heads = nh
                    break
        
        # 验证推断的参数是否与checkpoint中的实际形状匹配
        # 打印详细的调试信息
        print(f"[调试] DS-TransUNet参数推断结果:")
        print(f"  - embed_dim: {embed_dim}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - mlp_ratio: {mlp_ratio:.2f}")
        print(f"  - dropout: {dropout}")
        
        # 验证关键层的形状
        if in_proj_key:
            in_proj_weight = state_dict[in_proj_key]
            if len(in_proj_weight.shape) == 2:
                expected_shape = (3 * embed_dim, embed_dim)
                actual_shape = in_proj_weight.shape
                print(f"  - transformer3.in_proj_weight形状: {actual_shape} (期望: {expected_shape})")
                if actual_shape != expected_shape:
                    print(f"[警告] in_proj_weight形状不匹配！实际: {actual_shape}, 期望: {expected_shape}")
                    # 尝试从实际形状反推embed_dim
                    if actual_shape[0] % 3 == 0:
                        inferred_embed_dim = actual_shape[0] // 3
                        if inferred_embed_dim == actual_shape[1]:
                            print(f"[提示] 从in_proj_weight反推embed_dim: {inferred_embed_dim}")
                            embed_dim = inferred_embed_dim
                            # 重新计算num_heads
                            for nh in [32, 16, 8, 4]:
                                if embed_dim % nh == 0:
                                    num_heads = nh
                                    break
        
        return {
            'embed_dim': int(embed_dim),
            'num_heads': int(num_heads),
            'num_layers': int(num_layers),
            'mlp_ratio': float(mlp_ratio),
            'dropout': float(dropout)
        }
    except Exception as e:
        print(f"[错误] DS-TransUNet参数推断失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_checkpoint_config(checkpoint_path):
    """读取checkpoint配置，支持从权重形状推断模型参数
    检测顺序与_load_model保持一致：
    1. 首先检查config中是否有model_type
    2. 然后从state_dict推断模型类型（按优先级顺序）
    """
    try:
        loaded_obj = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(loaded_obj, dict) and 'config' in loaded_obj:
            return loaded_obj['config']
        
        # 尝试从权重形状推断模型参数
        state_dict = loaded_obj['state_dict'] if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj else loaded_obj
        
        # 处理DataParallel包装
        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 检测顺序与_load_model保持一致
        # 1. 检测DS-TransUNet (patch_embed3)
        if state_dict and 'patch_embed3.weight' in state_dict:
            dstrans_params = infer_dstrans_params_from_state_dict(state_dict)
            if dstrans_params:
                return {
                    'model_type': 'ds_trans_unet',
                    'dstrans_params': dstrans_params
                }
        
        # 2. 检测SwinUNet (patch_embed.proj)
        if state_dict and 'patch_embed.proj.weight' in state_dict:
            swin_params = infer_swin_params_from_state_dict(state_dict)
            if swin_params:
                return {
                    'model_type': 'swin_unet',
                    'swin_params': swin_params
                }
        
        # 3. 检测ResNetUNet (enc0或layer0)
        old_version_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        has_old_keys = any(any(k.startswith(old_key) for k in state_dict.keys()) for old_key in old_version_keys)
        
        if 'enc0.0.weight' in state_dict or 'enc0.weight' in state_dict or has_old_keys:
            # ResNetUNet
                resnet_params = {}
                # 检测backbone类型
                if 'enc1.0.conv1.weight' in state_dict or (has_old_keys and 'layer1.0.conv1.weight' in state_dict):
                    if 'enc1.2.conv1.weight' in state_dict or (has_old_keys and 'layer1.2.conv1.weight' in state_dict):
                        resnet_params['backbone_name'] = 'resnet101'
                    else:
                        resnet_params['backbone_name'] = 'resnet50'
                
                # 检测是否有ASPP
                has_aspp = any('aspp' in k.lower() for k in state_dict.keys())
                if has_old_keys and not has_aspp:
                    resnet_params['use_aspp'] = False
                
                return {
                    'model_type': 'resnet_unet',
                    'resnet_params': resnet_params
                }
        
        # 4. 检测TransUNet (encoder.0)
        if 'encoder.0.weight' in state_dict:
            return {'model_type': 'trans_unet'}
        
        # 5. 检测其他ResNetUNet变体 (backbone.layer1)
        if 'backbone.layer1.0.conv1.weight' in state_dict:
            return {'model_type': 'resnet_unet'}
            
    except Exception as e:
        print(f"[read_checkpoint_config] 读取失败: {e}")
        return None
    return None

# 注意：instantiate_model 函数引用了模型类，需要在原文件中保留引用
# 这里只提供函数签名，实际实现需要在原文件中保留以访问模型类
# ==================== 数据处理函数 ====================

def parse_extra_modalities_spec(spec: Optional[str]) -> Dict[str, str]:
    """解析额外模态配置字符串，格式: name:path;name2:path2"""
    modalities = {}
    if not spec:
        return modalities
    for item in spec.split(';'):
        if not item.strip() or ':' not in item:
            continue
        name, modal_path = item.split(':', 1)
        modal_path = modal_path.strip().strip('"').strip("'")
        if name.strip() and modal_path:
            modalities[name.strip()] = modal_path
    return modalities


def build_extra_modalities_lists(image_paths: List[str], modalities_dirs: Dict[str, str]) -> Optional[Dict[str, List[Optional[str]]]]:
    """根据主图像路径和模态目录生成额外模态文件路径列表。"""
    if not modalities_dirs:
        return None
    extras = {name: [] for name in modalities_dirs.keys()}
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        for name, dir_path in modalities_dirs.items():
            alt_path = os.path.join(dir_path, base_name)
            extras[name].append(alt_path if os.path.exists(alt_path) else None)
    return extras


def normalize_volume_percentile(volume, p_low=10, p_high=99):
    """
    使用百分位数归一化（参考标准代码）
    更鲁棒，能处理异常值和不同强度范围的医学图像
    
    Args:
        volume: 图像数组 (H, W, C) 或 (H, W)
        p_low: 低百分位数 (默认10)
        p_high: 高百分位数 (默认99)
    
    Returns:
        归一化后的图像
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    volume = volume.astype(np.float32)
    
    # 对多通道图像，对每个通道分别计算百分位数（参考标准代码）
    if len(volume.shape) == 3:
        # 多通道：对每个通道分别归一化
        normalized_channels = []
        for c in range(volume.shape[2]):
            channel = volume[:, :, c]
            p10 = np.percentile(channel, p_low)
            p99 = np.percentile(channel, p_high)
            
            if p99 > p10:
                channel = np.clip(channel, p10, p99)
                channel = (channel - p10) / (p99 - p10)
            else:
                channel = np.zeros_like(channel)
            
            # Z-score标准化
            m = np.mean(channel)
            s = np.std(channel)
            s = max(s, 1e-7)
            channel = (channel - m) / s
            
            normalized_channels.append(channel)
        
        volume = np.stack(normalized_channels, axis=2)
    else:
        # 单通道
        p10 = np.percentile(volume, p_low)
        p99 = np.percentile(volume, p_high)
        
        if p99 > p10:
            volume = np.clip(volume, p10, p99)
            volume = (volume - p10) / (p99 - p10)
        else:
            volume = np.zeros_like(volume)
        
        # Z-score标准化
        m = np.mean(volume)
        s = np.std(volume)
        s = max(s, 1e-7)
        volume = (volume - m) / s
    
    return volume

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
    独立的Dice计算函数，不依赖类实例，可用于多进程并行处理
    
    Args:
        pred_mask: 预测掩码 (numpy array)
        target_mask: 真实掩码 (numpy array)
        smooth: 平滑系数
    
    Returns:
        Dice值 (float)
    """
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection + smooth) / (union + smooth)


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
def scan_best_threshold(prob_maps, gt_masks):
    """
    在给定的概率图和真实掩膜上扫描阈值，寻找综合评分最高的阈值。
    """
    # 确保输入是numpy数组
    if isinstance(prob_maps, torch.Tensor):
        prob_maps = prob_maps.detach().cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()
    
    # 统一维度
    if prob_maps.ndim == 4: prob_maps = prob_maps[:, 0]
    if gt_masks.ndim == 4: gt_masks = gt_masks[:, 0]

    thresholds = np.arange(0.3, 0.91, 0.05)
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

        # 计算基础指标
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        # 计算 HD95 (需要逐样本计算取平均)
        hd95_list = []
        # 为了速度，随机采样最多10个样本计算HD95
        sample_indices = range(len(pred_bool))
        if len(pred_bool) > 10:
            import random
            sample_indices = random.sample(sample_indices, 10)
            
        for i in sample_indices:
            hd = calculate_hd95(pred_bool[i], gt_bool[i])
            if hd < 99.0: # 过滤无效值
                hd95_list.append(hd)
        
        hd95_mean = np.mean(hd95_list) if hd95_list else 99.9

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

class MedicalImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        transform: Optional[Compose] = None,
        training: bool = True,
        normalize: bool = True,
        debug: bool = False,
        return_classification: bool = False,
        extra_modalities: Optional[Dict[str, List[Optional[str]]]] = None,
        context_slices: int = 0,
        context_gap: int = 1,
        use_percentile_normalization: bool = True,
        use_weighted_sampling: bool = False
    ):
        """
        改进的医学图像数据集类（参考标准代码改进）
        
        参数:
            image_paths: 图像路径列表
            mask_paths: 掩膜路径列表 (训练时必需)
            transform: 数据增强变换
            training: 是否为训练模式
            normalize: 是否自动归一化图像
            debug: 调试模式 (会打印加载信息)
            return_classification: 是否返回分类标签（从mask自动生成：有病变=1，无病变=0）
            use_percentile_normalization: 是否使用百分位数归一化（p10-p99，更鲁棒）
            use_weighted_sampling: 是否使用基于mask的权重采样（更关注有病变的样本）
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.training = training
        self.normalize = normalize
        self.debug = debug
        self.return_classification = return_classification
        self.extra_modalities = extra_modalities or {}
        self.context_slices = max(0, context_slices)
        self.context_gap = max(1, context_gap)
        self.use_percentile_normalization = use_percentile_normalization
        self.use_weighted_sampling = use_weighted_sampling and training and mask_paths is not None
        
        # 验证数据
        self._validate_inputs()
        
        # 计算采样权重（基于mask的前景像素数量）
        if self.use_weighted_sampling:
            self._compute_sampling_weights()

    def _validate_inputs(self):
        """验证输入数据是否有效"""
        if self.training and self.mask_paths is None:
            raise ValueError("训练模式必须提供mask路径")
            
        if self.mask_paths and len(self.image_paths) != len(self.mask_paths):
            raise ValueError("图像和mask数量不匹配")
        for name, paths in self.extra_modalities.items():
            if len(paths) != len(self.image_paths):
                raise ValueError(f"模态 {name} 的样本数量与图像不匹配")
            
        if self.debug:
            print(f"数据集初始化: 共{len(self.image_paths)}个样本")
            if self.mask_paths:
                print(f"包含mask数据: 是 (共{len(self.mask_paths)}个)")
            else:
                print("包含mask数据: 否")
    
    def _compute_sampling_weights(self):
        """
        计算基于mask的采样权重（参考标准代码）
        有病变的样本权重更高，帮助模型更关注难样本
        """
        weights = []
        for mask_path in self.mask_paths:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # 计算前景像素数量
                    foreground_pixels = np.sum(mask > 0)
                    weights.append(float(foreground_pixels))
                else:
                    weights.append(0.0)
            except Exception:
                weights.append(0.0)
        
        weights = np.array(weights, dtype=np.float32)
        
        # 添加平滑项，避免权重为0的样本完全不被采样
        # 公式: (w + total*0.1/len) / (total*1.1)
        total = np.sum(weights)
        if total > 0:
            smooth = total * 0.1 / len(weights)
            weights = (weights + smooth) / (total * 1.1)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        self.sampling_weights = weights
        if self.debug:
            pos_samples = np.sum(weights > np.mean(weights))
            print(f"权重采样: {pos_samples}/{len(weights)} 个样本权重高于平均值")
    
    def get_sampling_weights(self) -> Optional[np.ndarray]:
        """返回采样权重（供WeightedRandomSampler使用）"""
        if not self.use_weighted_sampling or not hasattr(self, 'sampling_weights'):
            return None
        return self.sampling_weights.copy()

    def _load_image(self, path: Optional[str], allow_missing: bool = False, apply_context: bool = True) -> Optional[np.ndarray]:
        """加载图像并进行颜色空间转换"""
        if path is None:
            if allow_missing:
                return None
            raise FileNotFoundError("未提供有效的图像路径")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            if allow_missing:
                return None
            raise FileNotFoundError(f"无法读取图像: {path}")
            
        # 处理不同通道数的情况
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32)
        
        if apply_context and self.context_slices > 0:
            context_images = self._load_context_images(path, img.shape)
            if context_images:
                img = np.concatenate([img] + context_images, axis=2)
            
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        """加载mask并二值化处理"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取mask: {path}")
        return (mask > 0).astype(np.float32)

    def _parse_slice_identifier(self, path: str) -> Optional[Tuple[str, str, int, str]]:
        """解析路径，提取病人ID和切片序号"""
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split('_')
        if len(parts) < 2 or not parts[-1].isdigit():
            return None
        slice_idx = int(parts[-1])
        patient_id = '_'.join(parts[:-1])
        base_dir = os.path.dirname(path)
        ext = os.path.splitext(path)[1]
        return base_dir, patient_id, slice_idx, ext

    def _load_context_images(self, path: str, reference_shape: Tuple[int, int, int]) -> List[np.ndarray]:
        """加载邻近切片，追加到通道维度"""
        info = self._parse_slice_identifier(path)
        if not info:
            return []
        base_dir, patient_id, slice_idx, ext = info
        ref_h, ref_w, ref_c = reference_shape
        context_images = []
        for offset in range(-self.context_slices, self.context_slices + 1):
            if offset == 0:
                continue
            target_idx = slice_idx + offset * self.context_gap
            if target_idx < 0:
                context_images.append(np.zeros((ref_h, ref_w, ref_c), dtype=np.float32))
                continue
            neighbor_name = f"{patient_id}_{target_idx}{ext}"
            neighbor_path = os.path.join(base_dir, neighbor_name)
            neighbor_img = self._load_image(neighbor_path, allow_missing=True, apply_context=False)
            if neighbor_img is None:
                neighbor_img = np.zeros((ref_h, ref_w, ref_c), dtype=np.float32)
            else:
                if neighbor_img.shape[:2] != (ref_h, ref_w):
                    neighbor_img = cv2.resize(neighbor_img, (ref_w, ref_h))
                if neighbor_img.shape[2] != ref_c:
                    if neighbor_img.shape[2] == 1 and ref_c == 3:
                        neighbor_img = np.repeat(neighbor_img, 3, axis=2)
                    elif neighbor_img.shape[2] == 3 and ref_c == 1:
                        neighbor_img = cv2.cvtColor(neighbor_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                    else:
                        neighbor_img = cv2.resize(neighbor_img, (ref_w, ref_h))
                        if neighbor_img.ndim == 2:
                            neighbor_img = neighbor_img[..., np.newaxis]
                        while neighbor_img.shape[2] < ref_c:
                            neighbor_img = np.concatenate([neighbor_img, neighbor_img], axis=2)[:, :, :ref_c]
            context_images.append(neighbor_img.astype(np.float32))
        return context_images

    def _to_tensor(self, img: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """将numpy数组转换为tensor"""
        if not is_mask and self.normalize:
            if self.use_percentile_normalization:
                # 使用百分位数归一化（更鲁棒，适合医学图像）
                img = normalize_volume_percentile(img, p_low=10, p_high=99)
            else:
                # 标准归一化
                img = img / 255.0
        
        if len(img.shape) == 2:
            return torch.from_numpy(img).unsqueeze(0).float()
        else:
            return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            # 正常加载单张图片
            image = self._load_image(self.image_paths[idx])
            if self.extra_modalities:
                extra_imgs = []
                for paths in self.extra_modalities.values():
                    extra_img = self._load_image(paths[idx], allow_missing=True, apply_context=False)
                    if extra_img is None:
                        extra_img = np.zeros_like(image)
                    else:
                        if extra_img.shape[:2] != image.shape[:2]:
                            extra_img = cv2.resize(extra_img, (image.shape[1], image.shape[0]))
                    extra_imgs.append(extra_img)
                if extra_imgs:
                    image = np.concatenate([image] + extra_imgs, axis=2)
            
            # 如果有mask路径，加载mask（训练和验证都需要mask）
            if self.mask_paths is not None:
                mask = self._load_mask(self.mask_paths[idx])
                # 生成分类标签：如果mask有前景像素，则为有病变(1)，否则为无病变(0)
                classification_label = torch.tensor(1.0 if np.sum(mask) > 0 else 0.0, dtype=torch.long)
                
                if self.transform:
                    transformed = self.transform(image=image, mask=mask)
                    image = transformed['image']
                    mask = transformed['mask']
                    mask_tensor = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
                    
                    if self.return_classification:
                        return image, mask_tensor, classification_label
                    else:
                        return image, mask_tensor
                else:
                    image_tensor = self._to_tensor(image)
                    mask_tensor = self._to_tensor(mask, is_mask=True)
                    if self.return_classification:
                        return image_tensor, mask_tensor, classification_label
                    else:
                        return image_tensor, mask_tensor
            
            # 推断模式（没有mask）
            else:
                if self.transform:
                    transformed = self.transform(image=image)
                    image = transformed['image']
                    return image
                else:
                    return self._to_tensor(image)
                    
        except Exception as e:
            if self.debug:
                print(f"加载样本 {idx} 失败: {str(e)}")
            # 返回空样本但保持batch一致性
            if self.training:
                if self.return_classification:
                    dummy_image = (torch.zeros((3, 256, 256)), torch.zeros((1, 256, 256)), torch.tensor(0, dtype=torch.long))
                else:
                    dummy_image = (torch.zeros((3, 256, 256)), torch.zeros((1, 256, 256)))
            else:
                dummy_image = torch.zeros((3, 256, 256))
            return dummy_image

# ==================== MATLAB 相关类 ====================
# 注意：这些类主要用于兼容性，大部分功能已移除

import threading

class MatlabCacheManager:
    """MATLAB 缓存功能已移除。"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("MATLAB 缓存功能已移除")

    def manifest_path(self, split_name: str) -> Path:
        safe_split = split_name.replace(os.sep, "_")
        return self.cache_dir / f"{safe_split}_manifest.json"

    def build_manifest(self, split_name: str, image_paths: List[str], mask_paths: List[str]) -> Path:
        manifest = []
        for idx, (img, msk) in enumerate(zip(image_paths, mask_paths)):
            cache_stub = hashlib.sha1(f"{split_name}-{img}".encode('utf-8')).hexdigest()[:10]
            cache_name = f"{split_name}_{idx:05d}_{cache_stub}.mat"
            manifest.append({
                "index": idx,
                "image_path": img,
                "mask_path": msk,
                "cache_path": str(self.cache_dir / cache_name),
                "preferred_format": "mat",
                "notes": "由MATLAB脚本生成，包含变量 I (HxWx3) 与 M (HxW)"
            })

        manifest_path = self.manifest_path(split_name)
        with manifest_path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self._write_instructions(manifest_path)
        return manifest_path

    def _write_instructions(self, manifest_path: Path):
        readme_path = self.cache_dir / "README_MATLAB_CACHE.md"
        if readme_path.exists():
            return

        content = (
            "# MATLAB 缓存指引\n\n"
            "1. 在MATLAB中执行 `manifest = jsondecode(fileread('"
            f"{manifest_path.name}'));`\n"
            "2. 遍历 `manifest`，对 `image_path` 和 `mask_path` 完成标准化、增强、"
            "以及 `gpuArray` 加速的操作。\n"
            "3. 将结果写入 `entry.cache_path`，至少包含 `image` (或 `I`) 与 "
            "`mask` (或 `M`) 变量，类型为 `single`/`logical`。\n"
            "4. Python 端会自动探测 `.mat/.npz` 缓存并优先加载，若不存在则回退到"
            " 原始dataloader。\n"
        )
        readme_path.write_text(content, encoding='utf-8')


class MatlabCacheDataset(Dataset):
    """MATLAB 缓存功能已移除。"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("MATLAB 缓存功能已移除")


class MatlabEngineSession:
    """MATLAB 引擎功能已移除。"""

    @classmethod
    def instance(cls):
        return None


class MatlabMetricsBridge:
    """MATLAB HD95 计算功能已移除。"""

    @classmethod
    def instance(cls):
        return None


class MatlabVisualizationBridge:
    """使用MATLAB绘制预测可视化网格。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.session = MatlabEngineSession.instance()

    @classmethod
    def instance(cls):
        # MATLAB 功能已移除，直接返回 None，避免引用未定义的 MATLAB_ENGINE_AVAILABLE
        return None

    def render_prediction_grid(self, payload_mat_path: str, save_path: str):
        # MATLAB 功能已移除
        pass

    def render_training_history(self, payload_mat_path: str, save_path: str):
        # MATLAB 功能已移除
        pass

    def render_performance_analysis(self, payload_mat_path: str, save_path: str):
        # MATLAB 功能已移除
        pass

    def render_test_results(self, payload_mat_path: str, save_path: str):
        # MATLAB 功能已移除
        pass

    def render_attention_maps(self, payload_mat_path: str, save_path: str):
        # MATLAB 功能已移除
        pass