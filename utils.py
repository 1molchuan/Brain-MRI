"""
工具函数和数据处理类模块
包含所有独立工具函数、数据处理类和模型加载函数
"""
import os
# 注意把路径改成你实际的 MATLAB 安装路径
matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
if matlab_bin_path not in os.environ['PATH']:
    os.environ['PATH'] += ';' + matlab_bin_path
import json
import hashlib
import threading
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


def load_model_compatible(model, checkpoint_path, device, verbose=True, target_model_type=None):
    """
    兼容加载模型权重的工具函数，自动处理新旧版本结构差异
    
    Args:
        model: 要加载权重的模型实例
        checkpoint_path: 模型权重文件路径
        device: 设备
        verbose: 是否打印信息
        target_model_type: 目标模型类型（可选），用于跨架构权重迁移（如从 ResNet-UNet 到 DeepLabV3+）
    
    Returns:
        (success: bool, message: str)
    """
    try:
        loaded_obj = torch.load(checkpoint_path, map_location=device)
        
        # 【跨架构权重迁移】检测 checkpoint 的源架构
        source_model_type = None
        if isinstance(loaded_obj, dict):
            if 'config' in loaded_obj and isinstance(loaded_obj['config'], dict):
                source_model_type = loaded_obj['config'].get('model_type')
            elif 'model_type' in loaded_obj:
                source_model_type = loaded_obj['model_type']
        
        # 如果无法从 config 推断，尝试从 state_dict 键名推断
        if source_model_type is None:
            if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
                temp_state_dict = loaded_obj['state_dict']
            else:
                temp_state_dict = loaded_obj
            
            # 检测 ResNet-UNet 的特征键名
            resnet_unet_keys = ['enc0.', 'enc1.', 'enc2.', 'enc3.', 'enc4.', 'layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.']
            has_resnet_unet_keys = any(any(k.startswith(key) for k in temp_state_dict.keys()) for key in resnet_unet_keys)
            if has_resnet_unet_keys:
                # 检查是否有解码器键名（UNet 特有的）
                unet_decoder_keys = ['dec1.', 'dec2.', 'dec3.', 'dec4.', 'up0.', 'up1.', 'up2.', 'up3.', 'up4.', 'up_conv0.', 'up_conv1.', 'up_conv2.', 'up_conv3.', 'up_conv4.']
                has_unet_decoder = any(any(k.startswith(key) for k in temp_state_dict.keys()) for key in unet_decoder_keys)
                if has_unet_decoder:
                    source_model_type = 'resnet_unet'
        
        # 【跨架构权重迁移】如果源架构是 ResNet-UNet，目标架构是 DeepLabV3+，只加载编码器权重
        if source_model_type and target_model_type:
            source_lower = source_model_type.lower()
            target_lower = target_model_type.lower()
            
            if 'resnet_unet' in source_lower and ('deeplabv3plus' in target_lower or 'smp_deeplabv3plus' in target_lower):
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"🔄 [跨架构权重迁移]")
                    print(f"   源架构: {source_model_type} (ResNet-UNet)")
                    print(f"   目标架构: {target_model_type} (DeepLabV3+)")
                    print(f"   策略: 只加载编码器 (Encoder) 权重，丢弃解码器 (Decoder) 权重")
                    print(f"{'='*60}\n")
        
        if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
            state_dict = loaded_obj['state_dict']
        else:
            state_dict = loaded_obj
        
        # 处理DataParallel包装
        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 【跨架构权重迁移】如果源架构是 ResNet-UNet，目标架构是 DeepLabV3+，过滤掉解码器权重
        encoder_only_state_dict = None
        if source_model_type and target_model_type:
            source_lower = source_model_type.lower()
            target_lower = target_model_type.lower()
            
            if 'resnet_unet' in source_lower and ('deeplabv3plus' in target_lower or 'smp_deeplabv3plus' in target_lower):
                # 定义编码器键名模式（ResNet 编码器层）
                encoder_key_patterns = [
                    'enc0.', 'enc1.', 'enc2.', 'enc3.', 'enc4.',  # 新版本键名
                    'layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.',  # 旧版本键名
                    'encoder.',  # SMP模型的编码器键名
                ]
                
                # 定义解码器键名模式（UNet 解码器层，需要丢弃）
                decoder_key_patterns = [
                    'dec1.', 'dec2.', 'dec3.', 'dec4.',  # 新版本解码器
                    'up0.', 'up1.', 'up2.', 'up3.', 'up4.',  # 旧版本解码器
                    'up_conv0.', 'up_conv1.', 'up_conv2.', 'up_conv3.', 'up_conv4.',  # 旧版本解码器卷积
                    'center.',  # UNet的中心层
                    'final.',  # UNet的最终层
                    'segmentation_head.',  # 分割头（DeepLab有自己的分割头）
                ]
                
                # 过滤：只保留编码器权重，并映射键名到 DeepLabV3+ 格式
                encoder_only_state_dict = {}
                encoder_keys = []
                discarded_keys = []
                
                for k, v in state_dict.items():
                    # 检查是否是编码器键
                    is_encoder = any(k.startswith(pattern) for pattern in encoder_key_patterns)
                    # 检查是否是解码器键
                    is_decoder = any(k.startswith(pattern) for pattern in decoder_key_patterns)
                    
                    if is_encoder and not is_decoder:
                        # 映射 ResNet-UNet 编码器键名到 DeepLabV3+ 格式
                        # ResNet-UNet: enc0 (conv1+bn1+relu), enc1 (layer1), enc2 (layer2), enc3 (layer3), enc4 (layer4)
                        # DeepLabV3+ (SMP): model.encoder.conv1/bn1/relu, model.encoder.layer1, layer2, layer3, layer4
                        mapped_key = k
                        
                        # 处理 enc0: enc0 是 Sequential(conv1, bn1, relu)，需要映射到 model.encoder.conv1/bn1/relu
                        if k.startswith('enc0.'):
                            # enc0.0 -> model.encoder.conv1
                            # enc0.1 -> model.encoder.bn1
                            # enc0.2 -> model.encoder.relu (通常没有权重，但保留映射)
                            if k.startswith('enc0.0.'):
                                mapped_key = k.replace('enc0.0.', 'model.encoder.conv1.', 1)
                            elif k.startswith('enc0.1.'):
                                mapped_key = k.replace('enc0.1.', 'model.encoder.bn1.', 1)
                            elif k.startswith('enc0.2.'):
                                mapped_key = k.replace('enc0.2.', 'model.encoder.relu.', 1)
                            else:
                                # 如果 enc0 不是 Sequential，直接映射
                                mapped_key = k.replace('enc0.', 'model.encoder.conv1.', 1)
                        # 处理 enc1-enc4: 直接映射到 model.encoder.layer1-4
                        elif k.startswith('enc1.'):
                            mapped_key = k.replace('enc1.', 'model.encoder.layer1.', 1)
                        elif k.startswith('enc2.'):
                            mapped_key = k.replace('enc2.', 'model.encoder.layer2.', 1)
                        elif k.startswith('enc3.'):
                            mapped_key = k.replace('enc3.', 'model.encoder.layer3.', 1)
                        elif k.startswith('enc4.'):
                            mapped_key = k.replace('enc4.', 'model.encoder.layer4.', 1)
                        # 处理旧版本键名 (layer0 -> model.encoder.conv1/bn1/relu, layer1-4 -> model.encoder.layer1-4)
                        elif k.startswith('layer0.'):
                            # layer0 通常对应 conv1+bn1+relu，但需要根据具体键名判断
                            # 如果包含 conv1/bn1/relu，直接映射
                            if 'conv1' in k or '0' in k.split('.')[1] if '.' in k else False:
                                mapped_key = k.replace('layer0.', 'model.encoder.conv1.', 1)
                            elif 'bn1' in k or '1' in k.split('.')[1] if '.' in k else False:
                                mapped_key = k.replace('layer0.', 'model.encoder.bn1.', 1)
                            else:
                                mapped_key = k.replace('layer0.', 'model.encoder.conv1.', 1)
                        elif k.startswith('layer1.'):
                            mapped_key = k.replace('layer1.', 'model.encoder.layer1.', 1)
                        elif k.startswith('layer2.'):
                            mapped_key = k.replace('layer2.', 'model.encoder.layer2.', 1)
                        elif k.startswith('layer3.'):
                            mapped_key = k.replace('layer3.', 'model.encoder.layer3.', 1)
                        elif k.startswith('layer4.'):
                            mapped_key = k.replace('layer4.', 'model.encoder.layer4.', 1)
                        
                        encoder_only_state_dict[mapped_key] = v
                        encoder_keys.append(f"{k} -> {mapped_key}")
                    elif is_decoder:
                        discarded_keys.append(k)
                
                if verbose:
                    print(f"[跨架构权重迁移] 编码器权重: {len(encoder_keys)} 个")
                    print(f"[跨架构权重迁移] 已丢弃解码器权重: {len(discarded_keys)} 个")
                    if encoder_keys:
                        print(f"[跨架构权重迁移] 编码器键映射示例（前5个）:")
                        for k in encoder_keys[:5]:
                            print(f"  - {k}")
                    if discarded_keys:
                        print(f"[跨架构权重迁移] 已丢弃键示例（前5个）:")
                        for k in discarded_keys[:5]:
                            print(f"  - {k}")
                
                # 使用过滤后的 state_dict
                state_dict = encoder_only_state_dict
        
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


# ==================== 全局进程池管理器（单例模式）====================
class ProcessPoolManager:
    """
    全局进程池管理器，用于复用进程池（Windows兼容）
    进程池在第一次创建后保留，后续调用复用，避免重复创建的开销
    """
    _instance = None
    _lock = threading.Lock()
    _pool = None
    _pool_size = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProcessPoolManager, cls).__new__(cls)
        return cls._instance
    
    def get_pool(self, pool_size=None):
        """
        获取进程池，如果不存在则创建
        
        Args:
            pool_size: 进程池大小，如果为None则使用CPU核心数-1
            
        Returns:
            multiprocessing.Pool: 进程池对象，如果创建失败则返回None
        """
        if pool_size is None:
            pool_size = max(1, mp.cpu_count() - 1)  # 保留一个核心给主进程
        
        # 如果进程池已存在且大小匹配，直接返回
        if self._pool is not None and self._pool_size == pool_size:
            try:
                # 测试进程池是否仍然有效
                self._pool._check_running()
                return self._pool
            except (ValueError, AssertionError):
                # 进程池已关闭，需要重新创建
                self._pool = None
                self._pool_size = None
        
        # 创建新的进程池
        if self._pool is None:
            try:
                # Windows下使用spawn方式（默认）
                ctx = mp.get_context('spawn')
                self._pool = ctx.Pool(processes=pool_size)
                self._pool_size = pool_size
                print(f"[进程池] 创建进程池，大小: {pool_size} (CPU核心数: {mp.cpu_count()})")
            except Exception as e:
                print(f"[进程池] 创建失败: {e}，将使用单进程模式")
                self._pool = None
                self._pool_size = None
        
        return self._pool
    
    def close_pool(self):
        """关闭进程池（通常不需要手动调用，程序退出时自动清理）"""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
                self._pool = None
                self._pool_size = None
                print("[进程池] 进程池已关闭")
            except Exception as e:
                print(f"[进程池] 关闭失败: {e}")


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


class GreyWolfThresholdOptimizer:
    """
    使用灰狼优化算法(GWO)寻找最佳分割阈值
    目标函数：Dice系数
    
    相比线性扫描，GWO 能够更智能地搜索阈值空间，在更少的迭代次数内找到更优解。
    """
    
    def __init__(self, num_wolves=10, max_iter=20, progress_callback=None, use_multiprocessing=True, 
                 use_mean_dice=False, postprocess_func=None, sample_ratio=1.0):
        """
        Args:
            num_wolves: 灰狼数量（种群大小），默认10
            max_iter: 最大迭代次数，默认20
            progress_callback: 进度回调函数，接收 (iteration, max_iter, best_score, best_threshold) 参数
            use_multiprocessing: 是否在CPU模式下使用多进程（默认True）
            use_mean_dice: 是否使用Mean Dice（每个样本分别计算再平均），默认False（使用Global Dice）
            postprocess_func: 后处理函数，接收(pred_mask, prob_map)返回处理后的mask，默认None（不使用后处理）
            sample_ratio: 采样比例，在迭代过程中只使用部分样本计算Fitness（0.0-1.0），默认1.0（使用全部样本）
        """
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.progress_callback = progress_callback
        self.use_multiprocessing = use_multiprocessing
        self.use_mean_dice = use_mean_dice
        self.postprocess_func = postprocess_func
        self.sample_ratio = sample_ratio
        # 搜索空间 [0.1, 0.9]
        self.lb = 0.1
        self.ub = 0.9
        # 【缓存机制】缓存阈值和对应的Dice分数，避免重复计算
        self._threshold_cache = {}  # {threshold_rounded: dice_score}
        self._cache_tolerance = 0.001  # 阈值差值<0.001时复用结果

    def optimize(self, preds, targets, device=None):
        """
        使用GWO算法优化阈值（GPU加速版本，带自动回退）
        
        Args:
            preds: 模型输出的概率图，可以是 torch.Tensor 或 numpy.ndarray
                  形状为 (N, H, W) 或 (N, C, H, W)
            targets: 真实标签，形状与 preds 相同
            device: 计算设备（如果为None，自动检测preds的设备）
            
        Returns:
            best_threshold: 最佳阈值
            best_dice: 最佳Dice分数
        """
        # 检测设备
        use_gpu = False
        if isinstance(preds, torch.Tensor):
            device = device or preds.device
            use_gpu = device.type == 'cuda'
        elif device is None:
            use_gpu = torch.cuda.is_available()
            device = torch.device('cuda' if use_gpu else 'cpu')
        else:
            device = torch.device(device)
            use_gpu = device.type == 'cuda'
        
        # 检查显存是否充足
        if use_gpu:
            try:
                # 估算数据大小
                if isinstance(preds, np.ndarray):
                    data_size_mb = preds.nbytes / (1024 * 1024)
                else:
                    data_size_mb = preds.numel() * 4 / (1024 * 1024)  # float32 = 4 bytes
                
                # 检查可用显存
                if torch.cuda.is_available():
                    free_memory_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024) - torch.cuda.memory_allocated(device) / (1024 * 1024)
                    # 如果数据大小超过可用显存的30%，使用CPU
                    if data_size_mb > free_memory_mb * 0.3:
                        print(f"[GWO] 显存不足（数据: {data_size_mb:.1f}MB, 可用: {free_memory_mb:.1f}MB），回退到CPU模式")
                        use_gpu = False
                        device = torch.device('cpu')
            except Exception as e:
                print(f"[GWO] 显存检查失败: {e}，回退到CPU模式")
                use_gpu = False
                device = torch.device('cpu')
        
        # 转换为torch tensor并移到指定设备
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()
        
        # 尝试移到GPU，如果失败则回退到CPU
        try:
            preds = preds.to(device)
            targets = targets.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[GWO] GPU显存不足，回退到CPU模式")
                device = torch.device('cpu')
                use_gpu = False
                preds = preds.cpu()
                targets = targets.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
        
        # 统一维度：如果是4D，取第一个通道
        if preds.ndim == 4:
            preds = preds[:, 0]  # (N, H, W)
        if targets.ndim == 4:
            targets = targets[:, 0]  # (N, H, W)
        
        # 【关键修复】保存原始形状（用于Mean Dice计算）
        num_samples = preds.shape[0]
        spatial_shape = preds.shape[1:]  # (H, W)
        
        # 【效率优化】保存全量数据，用于最终评估
        # 在迭代过程中使用采样数据，最终评估时使用全量数据
        if isinstance(preds, torch.Tensor):
            full_preds = preds.clone()
            full_targets = targets.clone()
        else:
            full_preds = preds.copy()
            full_targets = targets.copy()
        
        # 【效率优化】计算采样索引（等间隔采样）
        num_samples_sampled = num_samples
        sample_indices = None
        if self.sample_ratio < 1.0:
            sample_size = max(1, int(num_samples * self.sample_ratio))
            # 等间隔采样
            sample_indices = np.linspace(0, num_samples - 1, sample_size, dtype=np.int32)
            # 在迭代过程中使用采样数据
            if isinstance(preds, torch.Tensor):
                preds = preds[sample_indices]
                targets = targets[sample_indices]
            else:
                preds = preds[sample_indices]
                targets = targets[sample_indices]
            num_samples_sampled = len(sample_indices)
            print(f">>> [GWO效率优化] 采样计算: {num_samples_sampled}/{num_samples} 样本 ({100*self.sample_ratio:.0f}%)")
        
        # 根据是否使用Mean Dice决定数据处理方式
        if self.use_mean_dice:
            # Mean Dice模式：保持(N, H, W)形状，按样本分别计算
            # 如果是CPU模式，转换为numpy数组
            if not use_gpu:
                preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
                targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            else:
                # GPU模式：保持在GPU上，但需要按样本处理
                preds_np = None
                targets_np = None
            preds_flat = None
            targets_flat = None
            preds_flat_np = None
            targets_flat_np = None
            pool = None
        else:
            # Global Dice模式：展平为一维tensor（原有逻辑）
            preds_flat = preds.flatten().float()
            targets_flat = targets.flatten().float()
            
            # 如果是CPU模式，转换为numpy数组（多进程和单进程都需要）
            if not use_gpu:
                preds_flat_np = preds_flat.cpu().numpy()
                targets_flat_np = targets_flat.cpu().numpy()
                # 如果启用多进程，获取全局进程池管理器
                if self.use_multiprocessing:
                    pool_manager = ProcessPoolManager()
                    pool = pool_manager.get_pool(pool_size=min(self.num_wolves, mp.cpu_count() - 1))
                else:
                    pool = None
            else:
                preds_flat_np = None
                targets_flat_np = None
                pool = None
        
        # 初始化狼群位置（阈值）
        positions = np.random.uniform(self.lb, self.ub, self.num_wolves)
        
        if use_gpu:
            positions_tensor = torch.from_numpy(positions).float().to(device)
        else:
            positions_tensor = None
        
        # 【性能优化】Mean Dice模式：如果使用GPU，预先转换到CPU（避免每次迭代都转换）
        if self.use_mean_dice and use_gpu:
            preds_cpu = preds.cpu().numpy()  # (N, H, W) - 采样后的数据
            targets_cpu = targets.cpu().numpy()  # (N, H, W) - 采样后的数据
        else:
            preds_cpu = None
            targets_cpu = None
        
        # 【缓存机制】清空缓存（每次optimize调用时）
        self._threshold_cache.clear()
        
        # 初始化 Alpha, Beta, Delta 狼 (前三名)
        alpha_pos, alpha_score = 0.5, -float('inf')
        beta_pos, beta_score = 0.5, -float('inf')
        delta_pos, delta_score = 0.5, -float('inf')
        
        for t in range(self.max_iter):
            # 线性衰减参数 a 从 2 -> 0
            a = 2.0 - t * (2.0 / self.max_iter)
            
            # 边界处理
            positions = np.clip(positions, self.lb, self.ub)
            
            # 计算所有狼的适应度（Dice分数）
            if self.use_mean_dice:
                # Mean Dice模式：按样本分别计算Dice，然后取平均
                if use_gpu:
                    prob_maps = preds_cpu
                    mask_gts = targets_cpu
                else:
                    prob_maps = preds_np
                    mask_gts = targets_np
                
                scores_np = np.zeros(self.num_wolves)
                for wolf_idx in range(self.num_wolves):
                    threshold = float(positions[wolf_idx])
                    
                    # 【缓存机制】检查是否有缓存的阈值结果
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                        continue
                    
                    empty_dice_scores = []
                    non_empty_dice_scores = []
                    
                    # 对每个样本计算Dice（分类统计）- 使用采样后的数据
                    for sample_idx in range(num_samples_sampled):
                        prob_map = prob_maps[sample_idx]  # (H, W)
                        mask_gt = mask_gts[sample_idx]    # (H, W)
                        
                        # 判断是否为空mask
                        gt_flat = mask_gt.flatten()
                        mask_sum = gt_flat.sum()
                        empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)  # 0.1%像素
                        
                        # 使用阈值二值化
                        pred_mask = (prob_map >= threshold).astype(np.float32)
                        
                        # 应用后处理（如果提供）
                        if self.postprocess_func is not None:
                            try:
                                pred_mask_tensor = torch.from_numpy(pred_mask).float()
                                prob_map_tensor = torch.from_numpy(prob_map).float()
                                pred_mask_processed = self.postprocess_func(pred_mask_tensor, prob_map_tensor)
                                # 转换回numpy
                                if isinstance(pred_mask_processed, torch.Tensor):
                                    pred_mask = pred_mask_processed.detach().cpu().numpy()
                                else:
                                    pred_mask = np.asarray(pred_mask_processed)
                                # 确保是2D数组
                                if pred_mask.ndim > 2:
                                    pred_mask = pred_mask.squeeze()
                                # 确保数据类型和形状正确
                                pred_mask = pred_mask.astype(np.float32)
                                if pred_mask.shape != prob_map.shape:
                                    # 如果形状不匹配，使用原始预测
                                    pred_mask = (prob_map >= threshold).astype(np.float32)
                            except Exception as e:
                                # 如果后处理失败，使用原始预测（避免优化中断）
                                pass
                        
                        # 计算单个样本的Dice
                        pred_flat = pred_mask.flatten()
                        pred_sum = pred_flat.sum()
                        
                        if mask_sum <= empty_threshold:
                            # 空mask样本：GT为空
                            if pred_sum <= 1e-7:
                                dice_sample = 1.0  # GT为空，预测也为空，Dice=1.0
                            else:
                                dice_sample = 0.0  # GT为空，预测不为空（假阳性），Dice=0.0
                            empty_dice_scores.append(float(dice_sample))
                        else:
                            # 有前景样本：计算标准Dice
                            intersection = (pred_flat * gt_flat).sum()
                            dice_den = 2.0 * intersection + pred_sum + mask_sum
                            if dice_den < 1e-7:
                                dice_sample = 0.0
                            else:
                                dice_sample = (2.0 * intersection) / dice_den
                            non_empty_dice_scores.append(float(dice_sample))
                    
                    # 【关键修复】使用Balanced Mean Dice：分别计算前景样本和空样本的平均分，再取均值
                    # 这样可以防止数量众多的空样本带偏阈值搜索方向
                    empty_mean = np.mean(empty_dice_scores) if empty_dice_scores else 1.0
                    non_empty_mean = np.mean(non_empty_dice_scores) if non_empty_dice_scores else 0.0
                    
                    # 如果只有一类样本，使用该类样本的平均分
                    if len(empty_dice_scores) == 0:
                        balanced_dice = non_empty_mean
                    elif len(non_empty_dice_scores) == 0:
                        balanced_dice = empty_mean
                    else:
                        # 两类样本都存在，取均值（平衡权重，防止空样本带偏阈值）
                        balanced_dice = (empty_mean + non_empty_mean) / 2.0
                    
                    scores_np[wolf_idx] = float(balanced_dice)
                    # 【缓存机制】缓存结果
                    self._threshold_cache[threshold_rounded] = float(balanced_dice)
            elif use_gpu:
                # GPU模式：使用批量计算（Global Dice）
                positions_tensor = torch.from_numpy(positions).float().to(device)
                positions_tensor = torch.clamp(positions_tensor, self.lb, self.ub)
                # 【缓存机制】检查缓存，只计算未缓存的阈值
                scores_np = np.zeros(self.num_wolves)
                uncached_indices = []
                uncached_positions = []
                for wolf_idx in range(self.num_wolves):
                    threshold = float(positions[wolf_idx])
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                    else:
                        uncached_indices.append(wolf_idx)
                        uncached_positions.append(threshold)
                
                # 只计算未缓存的阈值
                if uncached_positions:
                    uncached_tensor = torch.tensor(uncached_positions, device=device)
                    uncached_tensor = torch.clamp(uncached_tensor, self.lb, self.ub)
                    uncached_scores = self._calculate_dice_batch(preds_flat, targets_flat, uncached_tensor)
                    uncached_scores_np = uncached_scores.cpu().numpy()
                    # 填充结果并缓存
                    for i, wolf_idx in enumerate(uncached_indices):
                        threshold_rounded = round(uncached_positions[i] / self._cache_tolerance) * self._cache_tolerance
                        scores_np[wolf_idx] = float(uncached_scores_np[i])
                        self._threshold_cache[threshold_rounded] = float(uncached_scores_np[i])
            elif pool is not None:
                # CPU多进程模式：并行计算每只狼的Dice（Global Dice）
                # 【缓存机制】检查缓存，只计算未缓存的阈值
                scores_np = np.zeros(self.num_wolves)
                uncached_args = []
                uncached_indices = []
                for wolf_idx, pos in enumerate(positions):
                    threshold = float(pos)
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                    else:
                        uncached_args.append((preds_flat_np, targets_flat_np, threshold))
                        uncached_indices.append(wolf_idx)
                
                # 只计算未缓存的阈值
                if uncached_args:
                    uncached_scores = np.array(pool.map(_calculate_dice_worker, uncached_args))
                    for i, wolf_idx in enumerate(uncached_indices):
                        threshold_rounded = round(float(uncached_args[i][2]) / self._cache_tolerance) * self._cache_tolerance
                        scores_np[wolf_idx] = float(uncached_scores[i])
                        self._threshold_cache[threshold_rounded] = float(uncached_scores[i])
            else:
                # CPU单进程模式：循环计算（Global Dice）
                # 【缓存机制】检查缓存，只计算未缓存的阈值
                scores_np = np.zeros(self.num_wolves)
                for wolf_idx in range(self.num_wolves):
                    threshold = float(positions[wolf_idx])
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                    else:
                        dice = self._calculate_dice(preds_flat_np, targets_flat_np, threshold)
                        scores_np[wolf_idx] = float(dice)
                        self._threshold_cache[threshold_rounded] = float(dice)
            
            positions_np = positions.copy()
            
            # 更新前三名（Alpha, Beta, Delta）
            for i in range(self.num_wolves):
                score = float(scores_np[i])
                pos = float(positions_np[i])
                
                if score > alpha_score:
                    # 更新 Alpha，原 Alpha 降为 Beta，原 Beta 降为 Delta
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = alpha_score, alpha_pos
                    alpha_score, alpha_pos = score, pos
                elif score > beta_score:
                    # 更新 Beta，原 Beta 降为 Delta
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = score, pos
                elif score > delta_score:
                    # 更新 Delta
                    delta_score, delta_pos = score, pos
            
            # 更新每只狼的位置（基于 Alpha, Beta, Delta 的位置）
            if use_gpu:
                # GPU模式：在GPU上批量计算位置更新
                alpha_pos_tensor = torch.tensor(alpha_pos, device=device)
                beta_pos_tensor = torch.tensor(beta_pos, device=device)
                delta_pos_tensor = torch.tensor(delta_pos, device=device)
                
                # 生成随机数（在GPU上）
                r1 = torch.rand(self.num_wolves, device=device)
                r2 = torch.rand(self.num_wolves, device=device)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = torch.abs(C1 * alpha_pos_tensor - positions_tensor)
                X1 = alpha_pos_tensor - A1 * D_alpha
                
                r1 = torch.rand(self.num_wolves, device=device)
                r2 = torch.rand(self.num_wolves, device=device)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = torch.abs(C2 * beta_pos_tensor - positions_tensor)
                X2 = beta_pos_tensor - A2 * D_beta
                
                r1 = torch.rand(self.num_wolves, device=device)
                r2 = torch.rand(self.num_wolves, device=device)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = torch.abs(C3 * delta_pos_tensor - positions_tensor)
                X3 = delta_pos_tensor - A3 * D_delta
                
                # 狼的位置更新为三者平均
                positions_tensor = (X1 + X2 + X3) / 3.0
                positions = positions_tensor.cpu().numpy()
            else:
                # CPU模式：使用numpy计算
                positions_tensor = torch.from_numpy(positions).float()
                
                r1 = np.random.random(self.num_wolves)
                r2 = np.random.random(self.num_wolves)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos - positions)
                X1 = alpha_pos - A1 * D_alpha
                
                r1 = np.random.random(self.num_wolves)
                r2 = np.random.random(self.num_wolves)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - positions)
                X2 = beta_pos - A2 * D_beta
                
                r1 = np.random.random(self.num_wolves)
                r2 = np.random.random(self.num_wolves)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_pos - positions)
                X3 = delta_pos - A3 * D_delta
                
                # 狼的位置更新为三者平均
                positions = (X1 + X2 + X3) / 3.0
            
            # 调用进度回调函数（如果提供）
            if self.progress_callback is not None:
                try:
                    self.progress_callback(t + 1, self.max_iter, alpha_score, alpha_pos)
                except Exception as e:
                    # 如果回调函数出错，不影响优化过程
                    pass
        
        # 【效率优化】最终评估：使用全量样本重新计算最佳阈值的Dice分数
        if self.sample_ratio < 1.0 and alpha_pos is not None:
            # 保存采样阶段的Dice值，用于日志对比
            sampled_dice = alpha_score
            print(f">>> [GWO最终评估] 使用全量 {num_samples} 个样本重新计算最佳阈值 {alpha_pos:.4f} 的Dice...")
            # 【关键修复】确保使用全量数据（full_preds, full_targets）进行最终评估
            final_dice = self._evaluate_threshold_full(full_preds, full_targets, alpha_pos, use_gpu, device)
            if final_dice is not None:
                alpha_score = final_dice
                print(f">>> [GWO最终评估] 全量样本Dice: {alpha_score:.4f} (采样Dice: {sampled_dice:.4f})")
            else:
                print(f">>> [GWO最终评估警告] 全量评估失败，使用采样Dice: {sampled_dice:.4f}")
        
        return alpha_pos, alpha_score
    
    def _evaluate_threshold_full(self, preds, targets, threshold, use_gpu, device):
        """
        使用全量样本评估阈值的Dice分数（用于最终评估）
        
        Args:
            preds: 全量概率图 (N, H, W)
            targets: 全量真实标签 (N, H, W)
            threshold: 阈值
            use_gpu: 是否使用GPU
            device: 计算设备
            
        Returns:
            dice: Dice分数，如果计算失败返回None
        """
        # 【调试】验证全量数据形状
        if isinstance(preds, torch.Tensor):
            full_num_samples = preds.shape[0]
        else:
            full_num_samples = preds.shape[0]
        
        try:
            if not self.use_mean_dice:
                # Global Dice模式：展平计算
                if isinstance(preds, torch.Tensor):
                    preds_flat = preds.flatten().float()
                    targets_flat = targets.flatten().float()
                else:
                    preds_flat = torch.from_numpy(preds.flatten()).float()
                    targets_flat = torch.from_numpy(targets.flatten()).float()
                
                if use_gpu:
                    preds_flat = preds_flat.to(device)
                    targets_flat = targets_flat.to(device)
                
                dice = self._calculate_dice_batch(preds_flat, targets_flat, 
                                                  torch.tensor([threshold], device=device if use_gpu else None))
                return float(dice[0].item() if isinstance(dice, torch.Tensor) else dice[0])
            else:
                # Mean Dice模式：按样本分别计算
                if isinstance(preds, torch.Tensor):
                    preds_np = preds.cpu().numpy() if use_gpu else preds.numpy()
                    targets_np = targets.cpu().numpy() if use_gpu else targets.numpy()
                else:
                    preds_np = preds
                    targets_np = targets
                
                num_samples = preds_np.shape[0]
                # 【调试】验证样本数量
                if num_samples < 500:  # 如果样本数太少，可能是采样数据而不是全量数据
                    print(f">>> [GWO警告] 全量评估样本数: {num_samples}，可能使用了采样数据而非全量数据")
                empty_dice_scores = []
                non_empty_dice_scores = []
                
                for sample_idx in range(num_samples):
                    prob_map = preds_np[sample_idx]
                    mask_gt = targets_np[sample_idx]
                    
                    # 判断是否为空mask
                    gt_flat = mask_gt.flatten()
                    mask_sum = gt_flat.sum()
                    empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)
                    
                    # 使用阈值二值化
                    pred_mask = (prob_map >= threshold).astype(np.float32)
                    
                    # 应用后处理（如果提供）
                    if self.postprocess_func is not None:
                        try:
                            pred_mask_tensor = torch.from_numpy(pred_mask).float()
                            prob_map_tensor = torch.from_numpy(prob_map).float()
                            pred_mask_processed = self.postprocess_func(pred_mask_tensor, prob_map_tensor)
                            if isinstance(pred_mask_processed, torch.Tensor):
                                pred_mask = pred_mask_processed.detach().cpu().numpy()
                            else:
                                pred_mask = np.asarray(pred_mask_processed)
                            if pred_mask.ndim > 2:
                                pred_mask = pred_mask.squeeze()
                            pred_mask = pred_mask.astype(np.float32)
                            if pred_mask.shape != prob_map.shape:
                                pred_mask = (prob_map >= threshold).astype(np.float32)
                        except Exception:
                            pass
                    
                    # 计算单个样本的Dice
                    pred_flat = pred_mask.flatten()
                    pred_sum = pred_flat.sum()
                    
                    if mask_sum <= empty_threshold:
                        # 空mask样本
                        if pred_sum <= 1e-7:
                            dice_sample = 1.0
                        else:
                            dice_sample = 0.0
                        empty_dice_scores.append(float(dice_sample))
                    else:
                        # 有前景样本
                        intersection = (pred_flat * gt_flat).sum()
                        dice_den = 2.0 * intersection + pred_sum + mask_sum
                        if dice_den < 1e-7:
                            dice_sample = 0.0
                        else:
                            dice_sample = (2.0 * intersection) / dice_den
                        non_empty_dice_scores.append(float(dice_sample))
                
                # 计算Balanced Mean Dice
                empty_mean = np.mean(empty_dice_scores) if empty_dice_scores else 1.0
                non_empty_mean = np.mean(non_empty_dice_scores) if non_empty_dice_scores else 0.0
                
                if len(empty_dice_scores) == 0:
                    balanced_dice = non_empty_mean
                elif len(non_empty_dice_scores) == 0:
                    balanced_dice = empty_mean
                else:
                    balanced_dice = (empty_mean + non_empty_mean) / 2.0
                
                return float(balanced_dice)
        except Exception as e:
            print(f">>> [GWO最终评估] 计算失败: {e}，使用采样Dice")
            return None

    def _calculate_dice(self, preds, targets, threshold):
        """
        快速计算 Dice 系数（CPU版本，用于兼容性）
        
        Args:
            preds: 展平的概率图（一维数组）
            targets: 展平的真实标签（一维数组）
            threshold: 二值化阈值
            
        Returns:
            dice: Dice 系数
        """
        # 二值化预测
        pred_mask = (preds >= threshold).astype(np.float32)
        targets_float = targets.astype(np.float32)
        
        # 计算交集和并集
        intersection = np.sum(pred_mask * targets_float)
        union = np.sum(pred_mask) + np.sum(targets_float)
        
        # 避免除零
        if union < 1e-7:
            return 1.0 if intersection < 1e-7 else 0.0
        
        # Dice = 2 * intersection / union
        return (2.0 * intersection) / union
    
    def _calculate_dice_batch(self, preds_flat, targets_flat, thresholds):
        """
        GPU加速：批量计算多个阈值的Dice系数（显存优化版本）
        
        Args:
            preds_flat: 展平的概率图（一维tensor，在GPU上）
            targets_flat: 展平的真实标签（一维tensor，在GPU上）
            thresholds: 阈值tensor（一维tensor，形状为[num_wolves]，在GPU上）
            
        Returns:
            dice_scores: Dice分数tensor（一维tensor，形状为[num_wolves]，在GPU上）
        """
        num_wolves = thresholds.shape[0]
        num_pixels = preds_flat.shape[0]
        
        # 【显存优化】如果数据量太大，使用循环计算而不是批量扩展
        # 估算显存占用：expand会创建 [M, N] 的tensor，约 4*M*N 字节
        # 如果超过500MB，使用循环方式
        estimated_memory_mb = 4 * num_wolves * num_pixels / (1024 * 1024)
        
        if estimated_memory_mb > 500:
            # 使用循环方式，每次只计算一只狼（节省显存）
            dice_scores = []
            targets_sum = targets_flat.sum()  # 只计算一次
            
            for i in range(num_wolves):
                threshold = thresholds[i]
                # 二值化
                pred_mask = (preds_flat >= threshold).float()
                # 计算交集和并集
                intersection = (pred_mask * targets_flat).sum()
                pred_sum = pred_mask.sum()
                union = pred_sum + targets_sum
                # 计算Dice
                smooth = 1e-7
                if union < smooth:
                    dice = 1.0 if intersection < smooth else 0.0
                else:
                    dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_scores.append(dice)
                # 及时删除中间变量
                del pred_mask, intersection, pred_sum
            
            dice_scores = torch.stack(dice_scores)
        else:
            # 使用批量计算（速度快，但显存占用大）
            thresholds_expanded = thresholds.unsqueeze(1)  # [M, 1]
            
            # 批量二值化：preds >= threshold for each threshold
            # 使用广播，避免expand创建大tensor
            pred_masks = (preds_flat.unsqueeze(0) >= thresholds_expanded).float()  # [M, N]
            
            # 批量计算交集和并集
            targets_expanded = targets_flat.unsqueeze(0)  # [1, N]
            intersections = (pred_masks * targets_expanded).sum(dim=1)  # [M]
            pred_sums = pred_masks.sum(dim=1)  # [M]
            target_sum = targets_flat.sum()  # scalar，只计算一次
            
            # 批量计算Dice
            unions = pred_sums + target_sum  # [M]
            
            # 避免除零（使用smooth项）
            smooth = 1e-7
            dice_scores = (2.0 * intersections + smooth) / (unions + smooth)
            
            # 处理特殊情况：如果union为0，根据intersection判断
            zero_union_mask = unions < smooth
            zero_intersection_mask = intersections < smooth
            dice_scores[zero_union_mask & zero_intersection_mask] = 1.0  # 两者都为0，Dice=1
            dice_scores[zero_union_mask & ~zero_intersection_mask] = 0.0  # union=0但intersection>0，Dice=0
            
            # 清理中间变量
            del pred_masks, targets_expanded, intersections, pred_sums
        
        return dice_scores


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

from scipy.io import savemat
import multiprocessing as mp
from functools import partial

# 尝试导入 MATLAB 引擎
MATLAB_ENGINE_AVAILABLE = False
MATLAB_ENGINE_ERROR = None

try:
    import matlab.engine
    MATLAB_ENGINE_AVAILABLE = True
except ImportError:
    MATLAB_ENGINE_AVAILABLE = False
    MATLAB_ENGINE_ERROR = "matlab.engine 模块未安装"
    print("[提示] matlab.engine 未安装，MATLAB 可视化功能将不可用")
except Exception as e:
    MATLAB_ENGINE_AVAILABLE = False
    MATLAB_ENGINE_ERROR = str(e)
    print(f"[警告] 无法导入 MATLAB 引擎: {e}")

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
    """MATLAB 引擎会话管理类，提供线程安全的 MATLAB 引擎访问。"""
    
    _instance = None
    _instance_lock = threading.Lock()
    _engine = None
    _engine_lock = threading.Lock()
    _start_error = None
    
    def __init__(self):
        if not MATLAB_ENGINE_AVAILABLE:
            self._engine = None
            self._start_error = MATLAB_ENGINE_ERROR or "MATLAB 引擎模块不可用"
            return
        
        try:
            # 尝试启动 MATLAB 引擎
            self._engine = matlab.engine.start_matlab()
            print("[MATLAB] 引擎启动成功")
        except Exception as e:
            error_msg = str(e)
            self._start_error = error_msg
            print(f"[警告] 无法启动 MATLAB 引擎: {error_msg}")
            
            # 提供详细的诊断信息
            if "dll" in error_msg.lower() or "找不到指定的程序" in error_msg or "mwmvmtransport" in error_msg.lower():
                print("\n[MATLAB 诊断] DLL 加载失败，可能的解决方案：")
                print("=" * 60)
                print("1. 安装 Visual C++ Redistributable (2015-2022)")
                print("   下载地址: https://aka.ms/vs/17/release/vc_redist.x64.exe")
                print("   或搜索: Microsoft Visual C++ Redistributable")
                print()
                print("2. 重新安装 MATLAB Engine for Python:")
                print("   打开命令提示符（管理员权限），执行：")
                print("   cd \"C:\\Program Files\\MATLAB\\R2025b\\extern\\engines\\python\"")
                print("   python setup.py install")
                print()
                print("3. 检查 MATLAB 安装完整性：")
                print("   - 确认 MATLAB R2025b 可以正常启动")
                print("   - 检查环境变量 PATH 是否包含 MATLAB bin 目录")
                print()
                print("4. 尝试以管理员权限运行 Python 程序")
                print("=" * 60)
            
            self._engine = None

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if not MATLAB_ENGINE_AVAILABLE:
            return None
        
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        
        # 如果引擎启动失败，返回 None 并打印错误信息
        if cls._instance._engine is None:
            if cls._instance._start_error:
                print(f"[MATLAB] 引擎不可用: {cls._instance._start_error}")
            return None
        
        return cls._instance
    
    def acquire(self):
        """获取 MATLAB 引擎和锁（线程安全）"""
        if self._engine is None:
            raise RuntimeError("MATLAB 引擎不可用")
        return self._engine, self._engine_lock
    
    @staticmethod
    def to_matlab_path(path: str) -> str:
        """将 Windows 路径转换为 MATLAB 兼容路径"""
        # 将反斜杠转换为正斜杠，并转义单引号
        matlab_path = path.replace('\\', '/').replace("'", "''")
        return matlab_path


class MatlabMetricsBridge:
    """MATLAB HD95 计算功能已移除。"""

    @classmethod
    def instance(cls):
        return None


class MatlabVisualizationBridge:
    """
    使用MATLAB绘制预测可视化网格。
    
    【修复版】线程安全设计：
    - 不再使用共享的 MATLAB 引擎实例
    - 每个渲染方法都在当前线程内独立启动和关闭引擎
    - 解决 "state not recoverable" 错误
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        # 【修复】不再需要 session，每个方法会独立启动引擎
        # 仅检查 MATLAB 引擎模块是否可用
        if not MATLAB_ENGINE_AVAILABLE:
            raise RuntimeError("MATLAB 引擎模块不可用，无法创建可视化桥接")

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if not MATLAB_ENGINE_AVAILABLE:
            return None
        
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    try:
                        cls._instance = cls()
                    except RuntimeError:
                        return None
        return cls._instance

    def render_prediction_grid(self, payload_mat_path: str, save_path: str):
        """
        【修复版 V4】针对 Python 3.8+ (含3.12) 的 DLL 白名单修复
        
        策略：
        1. 使用 os.add_dll_directory() 明确告诉 Python DLL 安全目录（Python 3.8+ 必需）
        2. 在子线程内强制注入 MATLAB bin 路径到系统 PATH（兼容旧工具）
        3. 不使用全局引擎，而是每次在当前线程内独立启动一个新引擎
        4. 解决 state not recoverable 问题的关键
        """
        import os
        import sys
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        # Python 3.8+ 必须用这个，否则 PATH 会被无视！
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        # 3. 此时再导入和启动，就能找到 DLL 了
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        print(f"[MATLAB] 正在为当前线程启动独立引擎... (预计耗时 10-15s)")
        eng = None
        
        try:
            # 3. 关键点：在当前线程（Worker Thread）内部启动独立引擎
            eng = matlab.engine.start_matlab()
            
            # 2. 准备数据路径
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_file = MatlabEngineSession.to_matlab_path(save_path)
            
            # V2.0 增强版 MATLAB 绘图脚本
            # 注意：这是 MATLAB 代码字符串，linter 可能误报变量未定义警告
            script = f"""
try
    disp('正在加载数据: {payload}');
    data = load('{payload}');
    images = data.images;
    masks = data.masks;
    preds = data.preds;
    
    % 限制样本数
    numSamples = min(size(images, 4), 4);
    
    % 设置画布：加大分辨率，设置白色背景
    fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 1400, 350 * numSamples]);
    tl = tiledlayout(fig, numSamples, 4, 'Padding','compact', 'TileSpacing','none');
    
    for idx = 1:numSamples
        % --- 数据预处理 ---
        raw_img = images(:,:,:,idx);
        
        % 1. 自动对比度增强 (解决灰蒙蒙的问题)
        if size(raw_img, 3) == 3
            gray_img = rgb2gray(raw_img);
        else
            gray_img = raw_img;
        end
        % 归一化并增强对比度
        base_img = imadjust(mat2gray(gray_img));
        % 转回 RGB 以便彩色叠加
        base_img_rgb = cat(3, base_img, base_img, base_img);
        
        gt_mask = double(masks(:,:,idx));
        pred_mask = double(preds(:,:,idx));
        
        % --- 绘图 1: 原图 ---
        nexttile(tl); 
        imshow(base_img, []); 
        title(sprintf('Sample %d Input', idx), 'FontSize', 12, 'FontWeight', 'bold');
        
        % --- 绘图 2: Ground Truth (绿色风格) ---
        nexttile(tl); 
        imshow(base_img, []); hold on;
        % 创建绿色透明蒙版
        green = cat(3, zeros(size(gt_mask)), ones(size(gt_mask)), zeros(size(gt_mask)));
        h = imshow(green); 
        set(h, 'AlphaData', gt_mask * 0.3); % 30% 透明度
        title('Ground Truth (Green)', 'FontSize', 12);
        
        % --- 绘图 3: Prediction (红色风格) ---
        nexttile(tl); 
        imshow(base_img, []); hold on;
        % 创建红色透明蒙版
        red = cat(3, ones(size(pred_mask)), zeros(size(pred_mask)), zeros(size(pred_mask)));
        h = imshow(red); 
        set(h, 'AlphaData', pred_mask * 0.3); 
        title('Prediction (Red)', 'FontSize', 12);
        
        % --- 绘图 4: 叠加对比 (医学标准) ---
        % 绿色=GT, 红色=Pred, 黄色=重叠(正确预测)
        nexttile(tl); 
        imshow(base_img, []); hold on;
        
        % 绘制 GT (绿色轮廓)
        [B_gt,L_gt] = bwboundaries(gt_mask > 0.5, 'noholes');
        for idx_k = 1:length(B_gt)
            boundary = B_gt{{idx_k}};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1.5);
        end
        
        % 绘制 Pred (红色轮廓)
        [B_pred,L_pred] = bwboundaries(pred_mask > 0.5, 'noholes');
        for idx_k = 1:length(B_pred)
            boundary = B_pred{{idx_k}};
            plot(boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 1.5);
        end
        
        % 添加图例说明
        title('Overlay (Green=GT, Red=Pred)', 'FontSize', 12);
    end
    
    disp('正在高保真导出...');
    % 使用 exportgraphics 的 ContentType='vector' 可以获得更锐利的文字
    exportgraphics(fig, '{save_file}', 'Resolution', 300, 'BackgroundColor','white');
    close(fig);
catch ME
    disp(['MATLAB Error: ', ME.message]);
    rethrow(ME);
end
"""
            
            # 4. 执行脚本
            print(f"[MATLAB] 开始渲染: {save_path}")
            eng.eval(script, nargout=0)
            print("[MATLAB] 渲染完成！")

        except Exception as e:
            print(f"\n[MATLAB 严重错误] {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 5. 务必关闭引擎，防止僵尸进程
            if eng:
                try:
                    eng.quit()
                    print("[MATLAB] 引擎已安全关闭")
                except Exception as e:
                    print(f"[MATLAB] 关闭引擎时出错: {e}")

    def render_training_history(self, payload_mat_path: str, save_path: str):
        """【修复版 V4】针对 Python 3.8+ 的 DLL 白名单修复 - 训练历史曲线渲染"""
        import os
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        eng = None
        try:
            eng = matlab.engine.start_matlab()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_mat = MatlabEngineSession.to_matlab_path(save_path)
            script = f"""
try
    data = load('{payload}');
    epochs = data.epochs;
    trainLoss = data.train_loss;
    valLoss = data.val_loss;
    valDice = data.val_dice;
    fig = figure('Visible','off');
    tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    nexttile;
    plot(epochs, trainLoss, '-ob', 'LineWidth', 2); hold on;
    plot(epochs, valLoss, '-or', 'LineWidth', 2);
    title('训练/验证损失'); xlabel('轮次'); ylabel('Loss');
    legend('训练','验证','Location','best'); grid on;
    nexttile;
    plot(epochs, valDice, '-og', 'LineWidth', 2);
    title('验证Dice'); xlabel('轮次'); ylabel('Dice'); ylim([0 1]); grid on;
    exportgraphics(fig, '{save_mat}', 'Resolution', 300);
    close(fig);
catch ME
    disp(['MATLAB Error: ', ME.message]);
    rethrow(ME);
end
"""
            eng.eval(script, nargout=0)
        except Exception as e:
            print(f"[MATLAB] 训练历史渲染失败: {e}")
        finally:
            if eng:
                try:
                    eng.quit()
                except:
                    pass

    def render_performance_analysis(self, payload_mat_path: str, save_path: str):
        """
        【完全重写版】性能分析绘图
        
        核心改进：
        1. 所有数据处理在 Python 端完成，避免 MATLAB 中的复杂逻辑
        2. 简化 MATLAB 脚本，只负责绘图
        3. 完整的错误处理，确保不会崩溃
        """
        import os
        import numpy as np
        from scipy.io import loadmat
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        # 【关键修复】在最外层添加 try-except，确保任何错误都不会导致程序崩溃
        try:
            import matlab.engine
        except ImportError:
            print("[MATLAB 警告] 未检测到 matlab.engine，跳过性能分析绘图")
            return
        
        eng = None
        try:
            # 【步骤 1】在 Python 端加载和处理数据
            print("[MATLAB] 正在加载性能数据...")
            data = loadmat(payload_mat_path)
            
            # 提取数据（优先使用新格式）
            group_means = None
            group_stds = None
            metric_names = None
            
            if 'avg_metrics_values' in data:
                # 新格式：直接使用数值数组
                group_means = np.array(data['avg_metrics_values']).flatten()
                
                if 'std_metrics_values' in data:
                    group_stds = np.array(data['std_metrics_values']).flatten()
                else:
                    group_stds = np.zeros_like(group_means)
                    print("[MATLAB] 警告: std_metrics_values 不存在，使用默认值 0")
                
                if 'avg_metrics_names' in data:
                    # 处理字符串数组
                    names_data = data['avg_metrics_names']
                    if names_data.dtype.names is None:
                        # 如果是字符数组，转换为字符串列表
                        if names_data.size > 0:
                            metric_names = [str(names_data.flat[i]) for i in range(names_data.size)]
                        else:
                            metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
                    else:
                        metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
                else:
                    metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
            else:
                # 旧格式：尝试从其他字段提取
                print("[MATLAB] 警告: 未找到新格式数据，尝试兼容旧格式...")
                if 'avg_metrics' in data:
                    avg_metrics = data['avg_metrics']
                    # 如果是 struct，转换为数组
                    if isinstance(avg_metrics, np.ndarray) and avg_metrics.dtype.names:
                        fields = avg_metrics.dtype.names
                        group_means = np.array([float(avg_metrics[field][0, 0]) for field in fields])
                        metric_names = [str(field) for field in fields]
                        group_stds = np.zeros_like(group_means)
                    else:
                        print("[MATLAB] 警告: 无法解析旧格式数据，跳过绘图")
                        return
                else:
                    print("[MATLAB] 警告: 未找到有效数据，跳过绘图")
                    return
            
            # 验证数据有效性
            if group_means is None or len(group_means) == 0:
                print("[MATLAB] 警告: 数据为空，跳过绘图")
                return
            
            # 确保 group_stds 和 metric_names 长度匹配
            if group_stds is None or len(group_stds) != len(group_means):
                group_stds = np.zeros_like(group_means)
            
            if metric_names is None or len(metric_names) != len(group_means):
                metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
            
            print(f"[MATLAB] 已加载 {len(group_means)} 个指标")
            
            # 【步骤 2】启动 MATLAB 引擎
            eng = matlab.engine.start_matlab()
            
            # 【步骤 3】将数据传递给 MATLAB（使用 matlab.double 和 matlab.engine 接口）
            eng.workspace['group_means'] = matlab.double(group_means.tolist())
            eng.workspace['group_stds'] = matlab.double(group_stds.tolist())
            eng.workspace['metric_names'] = metric_names  # MATLAB 会自动处理字符串列表
            eng.workspace['save_file'] = MatlabEngineSession.to_matlab_path(save_path)
            
            # 【步骤 4】执行优化的 MATLAB 绘图脚本（修复排版问题）
            script = """
            try
                % 确保数据是列向量
                if size(group_means, 2) > size(group_means, 1)
                    group_means = group_means';
                end
                if size(group_stds, 2) > size(group_stds, 1)
                    group_stds = group_stds';
                end
                
                x_axis = 1:length(group_means);
                
                % 【排版修复】创建高清画布：1200x800 像素
                fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
                
                % 【排版修复】手动锁定绘图区位置，给 X 轴标签预留 25% 的空间
                % [left, bottom, width, height] - 使用归一化坐标 (0-1)
                % bottom=0.25 给 X 轴标签留 25% 的高度，width=0.85 留左右边距，height=0.65 保证图表主体足够大
                ax = axes('Position', [0.10, 0.25, 0.85, 0.65]);
                
                % 绘制柱状图
                b = bar(ax, x_axis, group_means);
                b.FaceColor = [0.2, 0.6, 0.8];
                b.EdgeColor = 'none';
                b.FaceAlpha = 0.7;
                hold(ax, 'on');
                
                % 绘制误差棒
                er = errorbar(ax, x_axis, group_means, group_stds);
                er.Color = [0.2, 0.2, 0.2];
                er.LineStyle = 'none';
                er.LineWidth = 1.5;
                er.CapSize = 10;
                
                % 美化图表
                title(ax, 'Performance Metrics Analysis', 'FontSize', 16, 'FontWeight', 'bold');
                ylabel(ax, 'Metric Value', 'FontSize', 14);
                xlabel(ax, 'Metric Name', 'FontSize', 14);
                grid(ax, 'on');
                set(ax, 'GridAlpha', 0.15);
                set(ax, 'LineWidth', 1.2);
                
                % 设置 x 轴标签（优化字体大小）
                if length(metric_names) == length(x_axis)
                    set(ax, 'XTickLabel', metric_names);
                    set(ax, 'XTick', x_axis);
                    set(ax, 'FontSize', 12);  % 设置坐标轴字体大小
                    xtickangle(ax, 45);
                end
                
                % 自动调整 y 轴范围
                y_max = max(group_means + group_stds) * 1.1;
                y_min = min(group_means - group_stds) * 0.9;
                if y_min < 0
                    y_min = 0;
                end
                ylim(ax, [y_min, y_max]);
                
                % 添加数值标签（优化字体大小）
                xtips = b.XEndPoints;
                ytips = b.YEndPoints;
                labels = string(round(b.YData, 3));
                text(ax, xtips, ytips, labels, 'HorizontalAlignment','center',...
                    'VerticalAlignment','bottom', 'FontSize', 11, 'FontWeight','bold');
                
                % 【排版修复】确保图表布局正确
                set(ax, 'Box', 'on');  % 显示坐标轴边框
                
                % 保存图片（高分辨率）
                disp('正在导出性能分析图...');
                exportgraphics(fig, save_file, 'Resolution', 300);
                close(fig);
                
                disp('性能分析图已成功生成');
            catch ME
                disp(['MATLAB Plot Error: ', ME.message]);
                rethrow(ME);
            end
            """
            
            eng.eval(script, nargout=0)
            print(f"[MATLAB] ✅ 性能分析图已保存: {save_path}")
            
        except Exception as e:
            # 【关键修复】捕获所有异常，确保不会导致程序崩溃
            print(f"[MATLAB 警告] 性能分析绘图失败，已跳过: {str(e)}")
            import traceback
            print(f"[MATLAB] 错误详情: {traceback.format_exc()}")
            # 不抛出异常，让程序继续运行
        
        finally:
            # 确保 MATLAB 引擎被正确关闭
            if eng:
                try:
                    eng.quit()
                except:
                    pass

    def render_test_results(self, payload_mat_path: str, save_path: str):
        """【修复版 V4】针对 Python 3.8+ 的 DLL 白名单修复 - 测试结果可视化渲染"""
        import os
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        eng = None
        try:
            eng = matlab.engine.start_matlab()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_mat = MatlabEngineSession.to_matlab_path(save_path)
            script = f"""
try
    data = load('{payload}');
    images = data.images;
    masks = data.masks;
    preds = data.preds;
    diceVals = data.dice;
    iouVals = data.iou;
    numSamples = size(images, 4);
    fig = figure('Visible','off');
    tiledlayout(fig, numSamples, 4, 'Padding','compact','TileSpacing','compact');
    for idx = 1:numSamples
        img = images(:,:,:,idx);
        mask = masks(:,:,idx) > 0.5;
        pred = preds(:,:,idx) > 0.5;
        overlay = img;
        overlay(:,:,1) = max(overlay(:,:,1), mask);
        overlay(:,:,2) = max(overlay(:,:,2), pred);
        overlay(:,:,3) = max(overlay(:,:,3), mask & pred);
        nexttile; imshow(img, []); title(sprintf('样本 %d 原图', idx));
        nexttile; imshow(mask); title('真实Mask');
        nexttile; imshow(pred); title(sprintf('预测Mask\\nDice %.3f / IoU %.3f', diceVals(idx), iouVals(idx)));
        nexttile; imshow(overlay); title('叠加对比');
    end
    exportgraphics(fig, '{save_mat}', 'Resolution', 300);
    close(fig);
catch ME
    disp(['MATLAB Error: ', ME.message]);
    rethrow(ME);
end
"""
            eng.eval(script, nargout=0)
        except Exception as e:
            print(f"[MATLAB] 测试结果渲染失败: {e}")
        finally:
            if eng:
                try:
                    eng.quit()
                except:
                    pass

    def render_attention_maps(self, payload_mat_path: str, save_path: str):
        """
        【完整实现版】注意力热图渲染
        
        使用 MATLAB 绘制注意力权重热力图，叠加在原图上。
        支持多个注意力层的可视化。
        """
        import os
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        eng = None
        try:
            eng = matlab.engine.start_matlab()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_file = MatlabEngineSession.to_matlab_path(save_path)
            
            script = f"""
            try
                disp('正在加载注意力数据...');
                data = load('{payload}');
                images = data.images;
                masks = data.masks;
                preds = data.preds;
                
                % 检测可用的注意力层
                att_layers = {{}};
                if isfield(data, 'att1')
                    att_layers{{end+1}} = 'att1';
                end
                if isfield(data, 'att2')
                    att_layers{{end+1}} = 'att2';
                end
                if isfield(data, 'att3')
                    att_layers{{end+1}} = 'att3';
                end
                if isfield(data, 'att4')
                    att_layers{{end+1}} = 'att4';
                end
                
                if isempty(att_layers)
                    error('未找到注意力层数据 (att1, att2, att3, att4)');
                end
                
                numSamples = min(size(images, 4), 4);
                numLayers = length(att_layers);
                
                % 设置画布：每行一个样本，每列一个注意力层 + 原图/GT/Pred
                cols = 3 + numLayers;  % Input, GT, Pred, + 各注意力层
                fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 200 * cols, 300 * numSamples]);
                tl = tiledlayout(fig, numSamples, cols, 'Padding','compact', 'TileSpacing','none');
                
                for idx = 1:numSamples
                    % --- 数据预处理 ---
                    raw_img = images(:,:,:,idx);
                    
                    % 转换为灰度图并增强对比度
                    if size(raw_img, 3) == 3
                        gray_img = rgb2gray(raw_img);
                    else
                        gray_img = raw_img;
                    end
                    base_img = imadjust(mat2gray(gray_img));
                    base_img_rgb = cat(3, base_img, base_img, base_img);
                    
                    gt_mask = double(masks(:,:,idx));
                    pred_mask = double(preds(:,:,idx));
                    
                    % --- 绘图 1: 原图 ---
                    nexttile(tl);
                    imshow(base_img, []);
                    title(sprintf('Sample %d\\nInput', idx), 'FontSize', 11, 'FontWeight', 'bold');
                    
                    % --- 绘图 2: Ground Truth ---
                    nexttile(tl);
                    imshow(base_img, []); hold on;
                    green = cat(3, zeros(size(gt_mask)), ones(size(gt_mask)), zeros(size(gt_mask)));
                    h = imshow(green);
                    set(h, 'AlphaData', gt_mask * 0.3);
                    title('Ground Truth', 'FontSize', 11);
                    
                    % --- 绘图 3: Prediction ---
                    nexttile(tl);
                    imshow(base_img, []); hold on;
                    red = cat(3, ones(size(pred_mask)), zeros(size(pred_mask)), zeros(size(pred_mask)));
                    h = imshow(red);
                    set(h, 'AlphaData', pred_mask * 0.3);
                    title('Prediction', 'FontSize', 11);
                    
                    % --- 绘图 4-N: 各注意力层热力图 ---
                    for layer_idx = 1:numLayers
                        layer_name = att_layers{{layer_idx}};
                        att_map = data.(layer_name);
                        
                        % 获取当前样本的注意力图
                        if size(att_map, 3) >= idx
                            att_2d = double(att_map(:,:,idx));
                        else
                            att_2d = double(att_map(:,:,1));  % 回退到第一个
                        end
                        
                        % 使用 imresize 将热力图放大到原图尺寸
                        [H_orig, W_orig] = size(base_img);
                        [H_att, W_att] = size(att_2d);
                        if H_att ~= H_orig || W_att ~= W_orig
                            att_2d = imresize(att_2d, [H_orig, W_orig], 'bilinear');
                        end
                        
                        % 归一化到 [0, 1]
                        att_norm = mat2gray(att_2d);
                        
                        % 使用 jet 配色方案
                        att_colored = ind2rgb(uint8(att_norm * 255), jet(256));
                        
                        % 叠加在原图上 (Alpha=0.5)
                        nexttile(tl);
                        imshow(base_img, []); hold on;
                        h_heat = imshow(att_colored);
                        set(h_heat, 'AlphaData', att_norm * 0.5);  % 50% 透明度
                        title(sprintf('Attention %s', layer_name), 'FontSize', 11);
                    end
                end
                
                disp('正在导出注意力热力图...');
                exportgraphics(fig, '{save_file}', 'Resolution', 300, 'BackgroundColor','white');
                close(fig);
                
            catch ME
                disp(['MATLAB Error: ', ME.message]);
                rethrow(ME);
            end
            """
            
            eng.eval(script, nargout=0)
            print(f"[MATLAB] 注意力热力图已保存: {save_path}")
        except Exception as e:
            print(f"[MATLAB] 注意力热图渲染失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if eng:
                try:
                    eng.quit()
                except:
                    pass
    
    def render_quick_preview_matplotlib(self, images, masks, preds, save_path, num_samples=4, threshold=0.1):
        """
        【快速预览版】使用 Matplotlib 绘制预测对比图（无依赖，速度快）
        
        用于普通 Epoch 的快速预览，避免每次调用 MATLAB 导致训练变慢。
        
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
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_samples = min(num_samples, len(images))
        cols = 4  # Input, GT, Prediction, Overlay
        rows = num_samples
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = images[i]
            true_mask = masks[i]
            pred_mask = preds[i]
            
            # 确保图像是 (H, W, 3) 格式
            if img.ndim == 2:
                # 灰度图转 RGB
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            
            # 确保值在 [0, 1] 范围内
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
            
            # 确保掩码是 (H, W) 格式
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
            
            # --- 绘图 1: Input (原始图像) ---
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Sample {i+1}\nInput', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # --- 绘图 2: Ground Truth (绿色半透明轮廓) ---
            axes[i, 1].imshow(img)
            if true_mask_binary.sum() > 0:
                # 使用轮廓叠加（更清晰）
                from scipy.ndimage import binary_erosion
                try:
                    structure = np.ones((3, 3), dtype=bool)
                    gt_boundary = true_mask_binary - binary_erosion(true_mask_binary.astype(bool), structure=structure).astype(np.float32)
                    if gt_boundary.sum() > 0:
                        # 绘制绿色轮廓
                        axes[i, 1].contour(gt_boundary, levels=[0.5], colors='green', linewidths=2, alpha=0.7)
                except:
                    # 回退：使用透明叠加
                    overlay_gt = img.copy()
                    green_mask = np.zeros_like(overlay_gt)
                    green_mask[true_mask_binary > 0.5] = [0, 1, 0]
                    overlay_gt = overlay_gt * 0.7 + green_mask * 0.3
                    axes[i, 1].imshow(overlay_gt)
            axes[i, 1].set_title('Ground Truth\n(Green)', fontsize=10)
            axes[i, 1].axis('off')
            
            # --- 绘图 3: Prediction (红色半透明轮廓) ---
            axes[i, 2].imshow(img)
            if pred_mask_binary.sum() > 0:
                # 使用轮廓叠加（更清晰）
                try:
                    from scipy.ndimage import binary_erosion
                    structure = np.ones((3, 3), dtype=bool)
                    pred_boundary = pred_mask_binary - binary_erosion(pred_mask_binary.astype(bool), structure=structure).astype(np.float32)
                    if pred_boundary.sum() > 0:
                        # 绘制红色轮廓
                        axes[i, 2].contour(pred_boundary, levels=[0.5], colors='red', linewidths=2, alpha=0.7, linestyles='dashed')
                except:
                    # 回退：使用透明叠加
                    overlay_pred = img.copy()
                    red_mask = np.zeros_like(overlay_pred)
                    red_mask[pred_mask_binary > 0.5] = [1, 0, 0]
                    overlay_pred = overlay_pred * 0.7 + red_mask * 0.3
                    axes[i, 2].imshow(overlay_pred)
            axes[i, 2].set_title('Prediction\n(Red)', fontsize=10)
            axes[i, 2].axis('off')
            
            # --- 绘图 4: Overlay (叠加对比) ---
            # 绿色=GT, 红色=Pred, 黄色=重叠
            overlay = img.copy()
            # GT 区域（绿色）
            overlay[true_mask_binary > 0.5, 1] = np.maximum(overlay[true_mask_binary > 0.5, 1], 0.5)
            # Pred 区域（红色）
            overlay[pred_mask_binary > 0.5, 0] = np.maximum(overlay[pred_mask_binary > 0.5, 0], 0.5)
            # 重叠区域（黄色 = 红+绿）
            overlap = (true_mask_binary > 0.5) & (pred_mask_binary > 0.5)
            overlay[overlap, 0] = 1.0  # 红色
            overlay[overlap, 1] = 1.0  # 绿色
            overlay[overlap, 2] = 0.0  # 蓝色
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay\n(Green=GT, Red=Pred)', fontsize=10)
            axes[i, 3].axis('off')
        
        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # 【关键】必须关闭，防止内存泄漏
        
        return save_path


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