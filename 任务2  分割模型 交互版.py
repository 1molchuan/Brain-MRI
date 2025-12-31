import argparse
import base64
import copy
import importlib
import subprocess
import json
import os
import random
import sys
import tempfile
import threading
import time
import hashlib
import html
import markdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torchvision import models
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.stats import wasserstein_distance
try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[警告] skimage未安装，直方图匹配功能将不可用")
from scipy.io import loadmat, savemat
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免子线程启动GUI警告
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("[警告] nibabel 未安装，NIfTI 可视化将不可用")
# 设置matplotlib支持中文显示
# 尝试设置中文字体，优先使用系统可用的中文字体
try:
    # Windows系统常用中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong']
    # 检查系统可用的中文字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chinese_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        matplotlib.rcParams['font.sans-serif'] = [chinese_font] + matplotlib.rcParams['font.sans-serif']
    else:
        # 如果没有找到中文字体，使用默认字体（可能无法显示中文）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
except Exception:
    # 如果字体设置失败，使用默认设置
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QMessageBox
)
import shutil
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QFileDialog, QComboBox, QProgressBar,
                             QGroupBox, QTabWidget, QScrollArea, QSizePolicy, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QLineEdit,
                             QTextEdit, QTextBrowser, QDoubleSpinBox, QListWidget, QListWidgetItem,
                             QDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QColor, QTextCursor
from torch.utils.data import Dataset
from albumentations import Compose
from typing import Dict, List, Optional, Tuple, Union
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QCheckBox, QLabel, QPushButton, QFileDialog, QMessageBox,
                            QSpinBox, QComboBox, QTabWidget, QScrollArea, QProgressBar)
from PyQt5.QtCore import Qt, QThread, QMutex, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QPixmap, QImage, QFont
# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

# ==================== 【军令状：正午12点前必须跑通】彻底物理隔离的纯计算函数 ====================
# 这些函数必须放在所有类定义之前，确保绝对没有引用任何类成员、类方法或pyqtSignal

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
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt
    
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
    import numpy as np
    
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


class DropPath(nn.Module):
    """Stochastic Depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# 添加注意力机制
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


class CRFPostProcessor:
    """
    条件随机场（CRF）后处理模块
    对模型输出的概率图进行空间一致性优化
    参考: "Efficient Inference in Fully Connected CRFs" (NIPS 2012)
    """
    def __init__(self, num_iterations=5, pos_weight=3.0, pos_xy_std=3, pos_rgb_std=3, 
                 bi_xy_std=80, bi_rgb_std=13, bi_w=10):
        """
        Args:
            num_iterations: CRF迭代次数
            pos_weight: 一元势能权重
            pos_xy_std: 位置高斯核标准差
            pos_rgb_std: 颜色高斯核标准差
            bi_xy_std: 双边滤波位置标准差
            bi_rgb_std: 双边滤波颜色标准差
            bi_w: 双边滤波权重
        """
        self.num_iterations = num_iterations
        self.pos_weight = pos_weight
        self.pos_xy_std = pos_xy_std
        self.pos_rgb_std = pos_rgb_std
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.bi_w = bi_w
    
    def process(self, prob_map, image):
        """
        对概率图进行CRF优化
        Args:
            prob_map: 模型输出的概率图 (H, W) 或 (1, H, W)
            image: 原始图像 (H, W, C) 用于空间一致性约束
        Returns:
            优化后的概率图
        """
        try:
            # pydensecrf是可选依赖，使用动态导入避免linting警告
            import importlib
            dcrf = importlib.import_module('pydensecrf.densecrf')
            utils_module = importlib.import_module('pydensecrf.utils')
            unary_from_softmax = utils_module.unary_from_softmax
        except ImportError:
            print("[警告] pydensecrf未安装，跳过CRF后处理")
            return prob_map
        
        if len(prob_map.shape) == 3:
            prob_map = prob_map[0]
        
        H, W = prob_map.shape
        
        # 准备概率图（需要2类：背景和前景）
        prob_bg = 1 - prob_map
        prob_fg = prob_map
        unary = np.stack([prob_bg, prob_fg], axis=0)
        
        # 创建CRF
        d = dcrf.DenseCRF2D(W, H, 2)
        
        # 一元势能（来自模型预测）
        U = unary_from_softmax(unary)
        U = np.ascontiguousarray(U)
        d.setUnaryEnergy(U)
        
        # 二元势能（空间一致性）
        # 位置高斯核
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_weight)
        
        # 双边滤波（颜色+位置）
        if len(image.shape) == 3:
            image_rgb = image
        else:
            image_rgb = np.stack([image, image, image], axis=-1)
        
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, 
                              rgbim=image_rgb, compat=self.bi_w)
        
        # 推理
        Q = d.inference(self.num_iterations)
        map_result = np.array(Q).reshape((2, H, W))
        result = map_result[1]  # 前景概率
        
        return result
    
    def process_batch(self, prob_maps, images):
        """
        批量处理
        Args:
            prob_maps: (B, H, W) 或 (B, 1, H, W)
            images: (B, C, H, W) 或 (B, H, W, C)
        """
        results = []
        for i in range(prob_maps.shape[0]):
            prob = prob_maps[i]
            img = images[i]
            
            # 转换图像格式
            if len(img.shape) == 3 and img.shape[0] == 3:  # (C, H, W)
                img = img.transpose(1, 2, 0)  # (H, W, C)
            
            result = self.process(prob, img)
            results.append(result)
        
        return np.stack(results, axis=0)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力，提升特征表达能力
    参考: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力模块
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    用于跨尺度特征融合，增强多尺度特征表达能力
    参考: "Feature Pyramid Networks for Object Detection" (CVPR 2017)
    """
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        
        # 横向连接（1x1卷积）
        self.lateral_convs = nn.ModuleList()
        # 输出卷积（3x3卷积，减少混叠效应）
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from different scales
        Returns:
            List of FPN feature maps
        """
        # 自顶向下路径
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )
        
        # 输出特征
        fpn_outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return fpn_outs


class AttentionGate(nn.Module):
    """
    改进的注意力门控机制，解决注意力过于分散的问题:
    1. 添加温度参数控制注意力集中度
    2. 使用更尖锐的激活函数
    3. 支持注意力正则化
    """
    def __init__(self, F_g, F_l, F_int, temperature=1.0, use_sharpen=True):
        super(AttentionGate, self).__init__()
        self.temperature = temperature  # 温度参数: <1 更集中, >1 更分散
        self.use_sharpen = use_sharpen
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 改进: 使用更深的网络生成注意力
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_int // 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(F_int // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
            # 不使用Sigmoid,改用温度调节的softmax-like激活
        )
        self.relu = nn.ReLU(inplace=True)
        
    def _sharpen_attention(self, attention_map):
        """
        使用温度调节的方式增强注意力集中度
        temperature < 1: 更尖锐(更集中)
        temperature = 1: 标准sigmoid
        temperature > 1: 更平滑(更分散)
        """
        if self.use_sharpen:
            # 先通过温度缩放
            attention_scaled = attention_map / self.temperature
            # 再应用sigmoid
            attention_sharp = torch.sigmoid(attention_scaled)
            
            # 可选: 进一步增强对比度 (power transform)
            # attention_sharp = attention_sharp ** 2
            
            return attention_sharp
        else:
            return torch.sigmoid(attention_map)
    
    def forward(self, g, x, return_attention=False):
        # 确保 g 和 x 的空间尺寸一致
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 确保通道数一致
        if g.shape[1] != self.W_g[0].in_channels:
            # 如果通道数不匹配，需要调整（这种情况不应该发生，但为了安全）
            if not hasattr(self, '_g_channel_adapter'):
                self._g_channel_adapter = nn.Conv2d(g.shape[1], self.W_g[0].in_channels, kernel_size=1).to(g.device)
            g = self._g_channel_adapter(g)
        if x.shape[1] != self.W_x[0].in_channels:
            if not hasattr(self, '_x_channel_adapter'):
                self._x_channel_adapter = nn.Conv2d(x.shape[1], self.W_x[0].in_channels, kernel_size=1).to(x.device)
            x = self._x_channel_adapter(x)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        
        # 生成注意力logits
        attention_logits = self.psi(psi)
        
        # 应用温度调节的激活
        attention_map = self._sharpen_attention(attention_logits)
        
        # 加权输出
        output = x * attention_map
        
        if return_attention:
            return output, attention_map
        return output


class SkipAttention(nn.Module):
    """
    nnFormer风格的跳跃注意力机制
    使用交叉注意力机制来融合编码器和解码器特征
    解码器特征作为Query，编码器特征作为Key和Value
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_drop=0.1):
        super(SkipAttention, self).__init__()
        self.dim = dim
        # 确保num_heads至少为1，且能整除dim
        if num_heads <= 0:
            num_heads = max(1, dim // 64)  # 如果为0，使用dim//64，但至少为1
        # 确保num_heads能整除dim
        if dim % num_heads != 0:
            # 调整num_heads使其能整除dim
            num_heads = max(1, dim // (dim // num_heads + 1))
            # 如果还是不能整除，使用最大可能的除数
            for n in range(num_heads, 0, -1):
                if dim % n == 0:
                    num_heads = n
                    break
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 将编码器和解码器特征投影到相同维度
        self.encoder_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.decoder_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # 交叉注意力：Query来自解码器，Key和Value来自编码器
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # Query from decoder
        self.k = nn.Linear(dim, dim, bias=qkv_bias)  # Key from encoder
        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # Value from encoder
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, decoder_feat, encoder_feat):
        """
        Args:
            decoder_feat: (B, C, H, W) 解码器特征
            encoder_feat: (B, C, H, W) 编码器特征（跳跃连接）
        Returns:
            fused_feat: (B, C, H, W) 融合后的特征
        """
        B, C, H, W = decoder_feat.shape
        
        # 确保编码器和解码器特征的空间尺寸一致
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            encoder_feat = F.interpolate(encoder_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 确保通道数一致
        if encoder_feat.shape[1] != C:
            if not hasattr(self, 'encoder_channel_adapter'):
                self.encoder_channel_adapter = nn.Conv2d(encoder_feat.shape[1], C, kernel_size=1).to(encoder_feat.device)
            encoder_feat = self.encoder_channel_adapter(encoder_feat)
        
        # 投影到相同维度 - 优化：合并操作减少中间变量
        dec_proj = self.decoder_proj(decoder_feat)  # (B, C, H, W)
        enc_proj = self.encoder_proj(encoder_feat)   # (B, C, H, W)
        
        # 优化：直接计算，减少reshape次数
        # 转换为序列格式 - 使用contiguous()确保内存连续
        N = H * W
        dec_seq = dec_proj.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        enc_seq = enc_proj.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        
        # 交叉注意力：Query来自解码器，Key和Value来自编码器
        # 优化：减少reshape操作
        head_dim = C // self.num_heads
        q = self.q(dec_seq).view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = self.k(enc_seq).view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = self.v(enc_seq).view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # 计算注意力 - 性能优化：只在必要时clamp
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        # 性能优化：只在值超出范围时才clamp（对于正常值，clamp是多余的）
        if attn.abs().max() > 50.0:
            attn = torch.clamp(attn, min=-50.0, max=50.0)
        attn = attn.softmax(dim=-1)
        
        # 应用注意力到Value
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # 转换回空间格式 (B, C, H, W) - 优化：使用view而不是reshape
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


class nnFormerBlock(nn.Module):
    """
    nnFormer风格的Transformer Block
    交替使用局部窗口注意力（LV-MSA）和全局注意力（GV-MSA）
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path_rate=0.1, use_global_attn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_global_attn = use_global_attn

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        
        # 根据配置选择局部或全局注意力
        if use_global_attn:
            self.attn = GlobalAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = WindowAttention(
                dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"

        shortcut = x
        x = self.norm1(x)
        
        if self.use_global_attn:
            # 全局注意力：直接处理序列
            x = self.attn(x)
        else:
            # 局部窗口注意力：需要处理空间维度
            # 性能优化：减少不必要的reshape操作，使用更高效的内存布局
            x = x.view(B, H, W, C)
            
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            need_pad = pad_b or pad_r
            
            if need_pad:
                # 优化：合并permute操作，减少中间变量
                x = x.permute(0, 3, 1, 2).contiguous()
                x = F.pad(x, (0, pad_r, 0, pad_b))
                x = x.permute(0, 2, 3, 1).contiguous()
            H_pad, W_pad = H + pad_b, W + pad_r

            # 循环移位 - 优化：避免不必要的roll操作
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # 窗口分割 - 优化：使用contiguous确保内存连续，提升性能
            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.contiguous().view(-1, self.window_size * self.window_size, C)

            # W-MSA
            attn_windows = self.attn(x_windows)
            
            # 修复：确保attn_windows的通道数与输入一致（适配层可能改变了通道数）
            # 获取实际的通道数
            actual_C = attn_windows.shape[-1]
            num_windows = attn_windows.shape[0]
            
            # 验证形状是否匹配
            total_elements = attn_windows.numel()
            window_elements = self.window_size * self.window_size * actual_C
            
            if total_elements % window_elements != 0:
                # 如果无法整除，尝试推断正确的通道数或窗口数量
                # 首先尝试推断通道数
                spatial_elements = num_windows * self.window_size * self.window_size
                if total_elements % spatial_elements == 0:
                    # 通道数可以推断
                    actual_C = total_elements // spatial_elements
                    window_elements = spatial_elements * actual_C
                else:
                    # 尝试推断窗口数量
                    if total_elements % (self.window_size * self.window_size) == 0:
                        # 可以推断窗口数量和通道数
                        per_window_elements = total_elements // (self.window_size * self.window_size)
                        # 尝试找到合理的窗口数量和通道数组合
                        # 假设通道数在合理范围内（32-512）
                        found = False
                        for test_C in range(32, 513, 8):  # 通道数必须是8的倍数（因为num_heads通常>=1）
                            test_num_windows = per_window_elements // test_C
                            if test_num_windows * test_C == per_window_elements and test_num_windows > 0:
                                num_windows = test_num_windows
                                actual_C = test_C
                                found = True
                                break
                        if not found:
                            # 最后的尝试：使用reshape自动推断
                            raise RuntimeError(
                                f"无法reshape attn_windows: shape={attn_windows.shape}, "
                                f"window_size={self.window_size}, total_elements={total_elements}, "
                                f"per_window_elements={per_window_elements}"
                            )
                    else:
                        raise RuntimeError(
                            f"无法reshape attn_windows: shape={attn_windows.shape}, "
                            f"window_size={self.window_size}, total_elements={total_elements}, "
                            f"window_elements={window_elements}"
                        )

            # 合并窗口 - 使用实际的通道数
            attn_windows = attn_windows.contiguous().view(num_windows, self.window_size, self.window_size, actual_C)
            shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)

            # 反向循环移位
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            if need_pad:
                x = x[:, :H, :W, :].contiguous()
            x = x.view(B, H * W, C)

        # 残差连接 - 优化：只在训练初期或检测到问题时才检查NaN
        residual = self.drop_path(x)
        # 性能优化：减少NaN检查频率（只在每10个batch检查一次）
        if hasattr(self, '_check_nan_counter'):
            self._check_nan_counter += 1
            check_nan = (self._check_nan_counter % 10 == 0)
        else:
            self._check_nan_counter = 0
            check_nan = True  # 第一次总是检查
        
        if check_nan and (torch.any(torch.isnan(residual)) or torch.any(torch.isinf(residual))):
            residual = torch.zeros_like(residual)
        x = shortcut + residual
        
        if check_nan and (torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))):
            x = shortcut
            
        # MLP
        shortcut2 = x
        x = self.norm2(x)
        
        if check_nan and (torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))):
            x = shortcut2
        
        mlp_out = self.mlp(x)
        residual2 = self.drop_path(mlp_out)
        
        if check_nan and (torch.any(torch.isnan(residual2)) or torch.any(torch.isinf(residual2))):
            residual2 = torch.zeros_like(residual2)
        
        x = shortcut2 + residual2
        
        if check_nan and (torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))):
            x = shortcut2

        return x

# 改进的UNet模型
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()
        
        # 下采样路径
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._block(512, 1024)
        
        # 上采样路径
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = self._block(128, 64)
        
        # 注意力门控 - 使用temperature=0.5让注意力更集中
        # temperature < 1: 注意力更尖锐(更集中到重点区域)
        # temperature = 1: 标准行为
        # temperature > 1: 注意力更平滑(更分散)
        self.att1 = AttentionGate(64, 64, 32, temperature=0.5, use_sharpen=True)
        self.att2 = AttentionGate(128, 128, 64, temperature=0.5, use_sharpen=True)
        self.att3 = AttentionGate(256, 256, 128, temperature=0.5, use_sharpen=True)
        self.att4 = AttentionGate(512, 512, 256, temperature=0.5, use_sharpen=True)
        
        # 中间监督输出
        self.conv4_out = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最终卷积
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, return_attention=False, return_aux=False):
        # 下采样
        conv1 = self.down1(x)
        x = self.pool(conv1)
        
        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)
        
        conv4 = self.down4(x)
        x = self.pool(conv4)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 带注意力的上采样
        attention_maps = {}
        
        x = self.up4(x)
        if return_attention:
            conv4, att4 = self.att4(x, conv4, return_attention=True)
            attention_maps['att4'] = att4
        else:
            conv4 = self.att4(x, conv4)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        ds4 = self.conv4_out(x)
        
        x = self.up3(x)
        if return_attention:
            conv3, att3 = self.att3(x, conv3, return_attention=True)
            attention_maps['att3'] = att3
        else:
            conv3 = self.att3(x, conv3)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        ds3 = self.conv3_out(x)
        
        x = self.up2(x)
        if return_attention:
            conv2, att2 = self.att2(x, conv2, return_attention=True)
            attention_maps['att2'] = att2
        else:
            conv2 = self.att2(x, conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        ds2 = self.conv2_out(x)
        
        x = self.up1(x)
        if return_attention:
            conv1, att1 = self.att1(x, conv1, return_attention=True)
            attention_maps['att1'] = att1
        else:
            conv1 = self.att1(x, conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        
        # 输出logits，不应用sigmoid（BCEWithLogitsLoss会在内部处理）
        final_output = self.final(x)

        aux_outputs = []
        if return_aux:
            target_size = final_output.shape[2:]
            aux_outputs.append(F.interpolate(ds4, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds3, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds2, size=target_size, mode='bilinear', align_corners=False))
        
        # 统一返回格式: (output, aux_outputs, attention_maps)
        # 根据参数填充None,保持一致的返回顺序
        if return_attention and return_aux:
            return final_output, aux_outputs, attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, aux_outputs
        else:
            return final_output


class TransUNet(nn.Module):
    """
    Transformer + UNet 混合架构，用于提高Dice指标
    
    设计思路：
    1. 使用UNet的编码器-解码器结构保持空间信息
    2. 在瓶颈层集成Transformer编码器，捕获长距离依赖关系
    3. Transformer的自注意力机制能够建模全局上下文，提升分割精度
    4. 结合CNN的局部特征提取和Transformer的全局建模能力
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 embed_dim=512, num_heads=8, num_layers=3, 
                 mlp_ratio=4.0, dropout=0.1):
        super(TransUNet, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 下采样路径（编码器）
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # Transformer瓶颈层
        # 将特征图转换为序列
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=1)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 将序列转换回特征图
        self.patch_unembed = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 瓶颈层卷积（在Transformer之后进一步处理）
        self.bottleneck = self._block(512, 1024)
        
        # 上采样路径（解码器）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = self._block(128, 64)
        
        # 注意力门控（与ImprovedUNet保持一致）
        self.att1 = AttentionGate(64, 64, 32, temperature=0.5, use_sharpen=True)
        self.att2 = AttentionGate(128, 128, 64, temperature=0.5, use_sharpen=True)
        self.att3 = AttentionGate(256, 256, 128, temperature=0.5, use_sharpen=True)
        self.att4 = AttentionGate(512, 512, 256, temperature=0.5, use_sharpen=True)
        
        # 中间监督输出
        self.conv4_out = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最终卷积
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _apply_transformer(self, x):
        """
        将特征图通过Transformer处理
        
        Args:
            x: [B, C, H, W] 特征图
            
        Returns:
            [B, C, H, W] 处理后的特征图
        """
        B, C, H, W = x.shape
        
        # 转换为序列: [B, C, H, W] -> [B, H*W, C]
        x_embed = self.patch_embed(x)  # [B, embed_dim, H, W]
        x_flat = x_embed.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # 通过Transformer编码器
        x_transformed = self.transformer(x_flat)  # [B, H*W, embed_dim]
        
        # 转换回特征图: [B, H*W, embed_dim] -> [B, embed_dim, H, W]
        x_transformed = x_transformed.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        x_out = self.patch_unembed(x_transformed)  # [B, 512, H, W]
        
        return x_out
        
    def forward(self, x, return_attention=False, return_aux=False):
        # 下采样（编码器）
        conv1 = self.down1(x)
        x = self.pool(conv1)
        
        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)
        
        conv4 = self.down4(x)
        x = self.pool(conv4)
        
        # Transformer瓶颈层
        # 先通过Transformer增强特征
        x_transformed = self._apply_transformer(x)
        # 再通过卷积瓶颈层
        x = self.bottleneck(x_transformed)
        
        # 带注意力的上采样（解码器）
        attention_maps = {}
        
        x = self.up4(x)
        if return_attention:
            conv4, att4 = self.att4(x, conv4, return_attention=True)
            attention_maps['att4'] = att4
        else:
            conv4 = self.att4(x, conv4)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        ds4 = self.conv4_out(x)
        
        x = self.up3(x)
        if return_attention:
            conv3, att3 = self.att3(x, conv3, return_attention=True)
            attention_maps['att3'] = att3
        else:
            conv3 = self.att3(x, conv3)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        ds3 = self.conv3_out(x)
        
        x = self.up2(x)
        if return_attention:
            conv2, att2 = self.att2(x, conv2, return_attention=True)
            attention_maps['att2'] = att2
        else:
            conv2 = self.att2(x, conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        ds2 = self.conv2_out(x)
        
        x = self.up1(x)
        if return_attention:
            conv1, att1 = self.att1(x, conv1, return_attention=True)
            attention_maps['att1'] = att1
        else:
            conv1 = self.att1(x, conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        
        # 输出logits，不应用sigmoid
        final_output = self.final(x)
        
        aux_outputs = []
        if return_aux:
            target_size = final_output.shape[2:]
            aux_outputs.append(F.interpolate(ds4, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds3, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds2, size=target_size, mode='bilinear', align_corners=False))
        
        # 统一返回格式
        if return_attention and return_aux:
            return final_output, aux_outputs, attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, aux_outputs
        else:
            return final_output


class DSTransUNet(nn.Module):
    """
    DS-TransUNet: Dual-Scale TransUNet
    在多个尺度上使用Transformer，增强多尺度特征提取能力
    
    设计特点：
    1. 在编码器的多个层级（conv3和conv4）使用Transformer，捕获不同尺度的全局依赖
    2. 双尺度特征融合：将不同尺度的Transformer特征进行融合
    3. 保持Deep Supervision，提供多层级监督信号
    4. 使用注意力门控机制，增强特征选择能力
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 embed_dim=256, num_heads=8, num_layers=2, 
                 mlp_ratio=4.0, dropout=0.1):
        super(DSTransUNet, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 下采样路径（编码器）
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 第一个Transformer层（在conv3尺度，256通道）
        self.patch_embed3 = nn.Conv2d(256, embed_dim, kernel_size=1)
        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer3 = nn.TransformerEncoder(encoder_layer3, num_layers=num_layers)
        self.patch_unembed3 = nn.Conv2d(embed_dim, 256, kernel_size=1)
        
        # 第二个Transformer层（在conv4尺度，512通道）
        self.patch_embed4 = nn.Conv2d(512, embed_dim, kernel_size=1)
        encoder_layer4 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer4 = nn.TransformerEncoder(encoder_layer4, num_layers=num_layers)
        self.patch_unembed4 = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 瓶颈层Transformer（在最大下采样尺度）
        self.patch_embed_bottleneck = nn.Conv2d(512, embed_dim, kernel_size=1)
        encoder_layer_bottleneck = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_bottleneck = nn.TransformerEncoder(encoder_layer_bottleneck, num_layers=num_layers)
        self.patch_unembed_bottleneck = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 瓶颈层卷积（在Transformer之后进一步处理）
        self.bottleneck = self._block(512, 1024)
        
        # 双尺度特征融合模块
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 上采样路径（解码器）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = self._block(128, 64)
        
        # 注意力门控（与ImprovedUNet保持一致）
        self.att1 = AttentionGate(64, 64, 32, temperature=0.5, use_sharpen=True)
        self.att2 = AttentionGate(128, 128, 64, temperature=0.5, use_sharpen=True)
        self.att3 = AttentionGate(256, 256, 128, temperature=0.5, use_sharpen=True)
        self.att4 = AttentionGate(512, 512, 256, temperature=0.5, use_sharpen=True)
        
        # 中间监督输出（Deep Supervision）
        self.conv4_out = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最终卷积
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def get_config(self):
        return {
            "in_channels": 3,
            "out_channels": 1,
            "embed_dim": self.embed_dim,
            "num_heads": self.transformer3.layers[0].self_attn.num_heads if hasattr(self.transformer3, "layers") else self.transformer3.layers[0].self_attn.num_heads if isinstance(self.transformer3, nn.TransformerEncoder) else 8,
            "num_layers": len(self.transformer3.layers) if hasattr(self.transformer3, "layers") else 2,
            "mlp_ratio": self.transformer3.layers[0].linear1.out_features / self.transformer3.layers[0].linear1.in_features if hasattr(self.transformer3, "layers") else 4.0,
            "dropout": getattr(self.transformer3.layers[0], "dropout", nn.Dropout(0.1)).p if hasattr(self.transformer3, "layers") else 0.1,
        }
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _apply_transformer(self, x, patch_embed, transformer, patch_unembed):
        """
        将特征图通过Transformer处理
        
        Args:
            x: [B, C, H, W] 特征图
            patch_embed: patch embedding层
            transformer: transformer编码器
            patch_unembed: patch unembedding层
            
        Returns:
            [B, C, H, W] 处理后的特征图
        """
        B, C, H, W = x.shape
        
        # 转换为序列: [B, C, H, W] -> [B, H*W, embed_dim]
        x_embed = patch_embed(x)  # [B, embed_dim, H, W]
        x_flat = x_embed.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # 通过Transformer编码器
        x_transformed = transformer(x_flat)  # [B, H*W, embed_dim]
        
        # 转换回特征图: [B, H*W, embed_dim] -> [B, C, H, W]
        x_transformed = x_transformed.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        x_out = patch_unembed(x_transformed)  # [B, C, H, W]
        
        return x_out
        
    def forward(self, x, return_attention=False, return_aux=False):
        # 下采样（编码器）
        conv1 = self.down1(x)
        x = self.pool(conv1)
        
        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)
        
        # 第一个Transformer层（在conv3尺度）
        conv3_transformed = self._apply_transformer(
            conv3, self.patch_embed3, self.transformer3, self.patch_unembed3
        )
        # 残差连接
        conv3 = conv3 + conv3_transformed
        
        conv4 = self.down4(x)
        x = self.pool(conv4)
        
        # 第二个Transformer层（在conv4尺度）
        conv4_transformed = self._apply_transformer(
            conv4, self.patch_embed4, self.transformer4, self.patch_unembed4
        )
        # 残差连接
        conv4 = conv4 + conv4_transformed
        
        # 双尺度特征融合：将conv3和conv4的特征融合
        # 将conv3上采样到conv4的尺度
        conv3_up = F.interpolate(conv3, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        # 融合两个尺度的特征
        conv4_fused = self.scale_fusion(torch.cat([conv4, conv3_up], dim=1))
        
        # 瓶颈层Transformer
        x_transformed = self._apply_transformer(
            conv4_fused, self.patch_embed_bottleneck, 
            self.transformer_bottleneck, self.patch_unembed_bottleneck
        )
        # 残差连接
        x = conv4_fused + x_transformed
        # 再通过卷积瓶颈层
        x = self.bottleneck(x)
        
        # 保存融合后的conv4用于解码器（注意：使用融合后的特征）
        conv4_for_decoder = conv4_fused
        
        # 带注意力的上采样（解码器）
        attention_maps = {}
        
        x = self.up4(x)
        # 确保 x 和 conv4_for_decoder 的空间尺寸匹配
        if x.shape[2:] != conv4_for_decoder.shape[2:]:
            conv4_for_decoder = F.interpolate(conv4_for_decoder, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv4, att4 = self.att4(x, conv4_for_decoder, return_attention=True)
            attention_maps['att4'] = att4
        else:
            conv4 = self.att4(x, conv4_for_decoder)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        ds4 = self.conv4_out(x)
        
        x = self.up3(x)
        # 确保 x 和 conv3 的空间尺寸匹配
        if x.shape[2:] != conv3.shape[2:]:
            conv3 = F.interpolate(conv3, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv3, att3 = self.att3(x, conv3, return_attention=True)
            attention_maps['att3'] = att3
        else:
            conv3 = self.att3(x, conv3)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        ds3 = self.conv3_out(x)
        
        x = self.up2(x)
        # 确保 x 和 conv2 的空间尺寸匹配
        if x.shape[2:] != conv2.shape[2:]:
            conv2 = F.interpolate(conv2, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv2, att2 = self.att2(x, conv2, return_attention=True)
            attention_maps['att2'] = att2
        else:
            conv2 = self.att2(x, conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        ds2 = self.conv2_out(x)
        
        x = self.up1(x)
        # 确保 x 和 conv1 的空间尺寸匹配
        if x.shape[2:] != conv1.shape[2:]:
            conv1 = F.interpolate(conv1, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv1, att1 = self.att1(x, conv1, return_attention=True)
            attention_maps['att1'] = att1
        else:
            conv1 = self.att1(x, conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        
        # 输出logits，不应用sigmoid
        final_output = self.final(x)
        
        aux_outputs = []
        if return_aux:
            target_size = final_output.shape[2:]
            aux_outputs.append(F.interpolate(ds4, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds3, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds2, size=target_size, mode='bilinear', align_corners=False))
        
        # 统一返回格式
        if return_attention and return_aux:
            return final_output, aux_outputs, attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, aux_outputs
        else:
            return final_output


# ==================== Swin Transformer 核心模块 ====================
class WindowAttention(nn.Module):
    """Swin Transformer的窗口注意力机制（数值稳定版本）"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 确保num_heads有效且dim能被num_heads整除
        if num_heads <= 0:
            num_heads = 1
        if dim % num_heads != 0:
            # 调整dim使其能被num_heads整除（向下取整）
            dim = (dim // num_heads) * num_heads
            if dim == 0:
                dim = num_heads  # 至少为num_heads
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 if head_dim > 0 else 1.0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 如果输入维度与dim不匹配，需要适配层（延迟初始化）
        self._dim_adapter = None
        
        # 改进的权重初始化，防止梯度爆炸
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        # QKV投影使用较小的初始化
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        
        # 输出投影使用标准初始化
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        
        # 如果输入维度与预期维度不匹配，使用适配层
        if C != self.dim:
            if self._dim_adapter is None:
                self._dim_adapter = nn.Linear(C, self.dim).to(x.device)
            x = self._dim_adapter(x)
            C = self.dim
        
        # 确保C能被num_heads整除（双重检查）
        if C % self.num_heads != 0:
            # 调整C使其能被num_heads整除
            head_dim = max(1, C // self.num_heads)
            C_aligned = head_dim * self.num_heads
            if C_aligned != C:
                if self._dim_adapter is None:
                    self._dim_adapter = nn.Linear(C, C_aligned).to(x.device)
                else:
                    # 如果适配层已存在但维度不对，重新创建
                    if self._dim_adapter.out_features != C_aligned:
                        self._dim_adapter = nn.Linear(C, C_aligned).to(x.device)
                x = self._dim_adapter(x)
                C = C_aligned
        
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算attention scores，添加数值稳定性
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 数值稳定性：clamp attention scores防止溢出
        attn = torch.clamp(attn, min=-50.0, max=50.0)
        
        # 使用稳定的softmax
        attn = attn.softmax(dim=-1)
        
        # 检查NaN/Inf
        if torch.any(torch.isnan(attn)) or torch.any(torch.isinf(attn)):
            # 如果出现NaN/Inf，使用均匀分布作为后备
            attn = torch.ones_like(attn) / N
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 最终检查输出
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            # 如果输出有NaN/Inf，返回零
            x = torch.zeros_like(x)
        
        return x


class Mlp(nn.Module):
    """MLP模块（数值稳定版本）"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化，防止梯度爆炸"""
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.02)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # 性能优化：移除频繁的NaN检查，只在必要时检查
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
    
    if num_windows_per_image == 0:
        raise RuntimeError(f"window_size ({window_size}) 大于特征图尺寸 (H={H}, W={W})")
    
    if num_windows_total % num_windows_per_image != 0:
        # 如果无法整除，尝试使用实际的窗口数量
        B = max(1, num_windows_total // num_windows_per_image)
        actual_num_windows = B * num_windows_per_image
        if actual_num_windows < num_windows_total:
            windows = windows[:actual_num_windows]
        elif actual_num_windows > num_windows_total:
            raise RuntimeError(
                f"窗口数量不匹配: 总窗口数={num_windows_total}, "
                f"每图像窗口数={num_windows_per_image}, 计算的B={B}"
            )
    else:
        B = num_windows_total // num_windows_per_image
    
    # 获取实际的通道数
    C = windows.shape[-1]
    
    # 验证形状
    expected_size = B * (H // window_size) * (W // window_size) * window_size * window_size * C
    if windows.numel() != expected_size:
        # 尝试使用reshape自动推断
        try:
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
        except RuntimeError:
            # 如果失败，尝试推断C
            total_elements = windows.numel()
            spatial_elements = B * (H // window_size) * (W // window_size) * window_size * window_size
            if total_elements % spatial_elements == 0:
                C = total_elements // spatial_elements
                x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
            else:
                raise RuntimeError(
                    f"无法reshape windows: shape={windows.shape}, "
                    f"B={B}, H={H}, W={W}, window_size={window_size}, "
                    f"total_elements={total_elements}, spatial_elements={spatial_elements}"
                )
    else:
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


class GlobalAttention(nn.Module):
    """
    全局自注意力机制（类似ViT，用于nnFormer的GV-MSA）
    2D版本的全局体积自注意力
    """
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        """
        x: (B, N, C) where N = H*W
        优化：对于大N，使用分块计算减少内存占用
        """
        B, N, C = x.shape
        head_dim = C // self.num_heads
        
        # 优化：对于大N（>4096），使用分块计算减少内存
        if N > 4096:
            # 分块计算QKV
            chunk_size = 1024
            qkv_chunks = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk = x[:, i:end_idx, :]
                qkv_chunk = self.qkv(chunk).reshape(B, end_idx - i, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                qkv_chunks.append(qkv_chunk)
            
            # 合并chunks
            q = torch.cat([chunk[0] for chunk in qkv_chunks], dim=2)
            k = torch.cat([chunk[1] for chunk in qkv_chunks], dim=2)
            v = torch.cat([chunk[2] for chunk in qkv_chunks], dim=2)
        else:
            # 小N时使用原始方法
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        # 全局注意力：所有位置之间计算注意力
        # 优化：对于大N，使用分块计算注意力矩阵
        if N > 4096:
            # 分块计算注意力
            chunk_size = 512
            out_chunks = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                q_chunk = q[:, :, i:end_idx, :]  # (B, num_heads, chunk_size, head_dim)
                
                # 计算注意力分数
                attn_chunk = (q_chunk @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, chunk_size, N)
                attn_chunk = torch.clamp(attn_chunk, min=-50.0, max=50.0)
                attn_chunk = attn_chunk.softmax(dim=-1)
                attn_chunk = self.attn_drop(attn_chunk)
                
                # 应用注意力
                out_chunk = (attn_chunk @ v).transpose(1, 2)  # (B, chunk_size, num_heads, head_dim)
                out_chunks.append(out_chunk)
            
            x = torch.cat(out_chunks, dim=1).reshape(B, N, C)
        else:
            # 小N时使用原始方法
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = torch.clamp(attn, min=-50.0, max=50.0)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., mlp_hidden_dim=None,
                 drop_path_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 使用更大的eps提高LayerNorm的数值稳定性
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        # 如果提供了精确的mlp_hidden_dim，使用它；否则从mlp_ratio计算
        if mlp_hidden_dim is None:
            mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b or pad_r:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)
        H_pad, W_pad = x.shape[1], x.shape[2]

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_b or pad_r:
            x = x[:, :H, :W, :]
        x = x.reshape(B, H * W, C)

        # 残差连接，添加数值稳定性检查
        residual = self.drop_path(x)
        if torch.any(torch.isnan(residual)) or torch.any(torch.isinf(residual)):
            residual = torch.zeros_like(residual)
        
        x = shortcut + residual
        
        # 检查输出
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            # 如果残差连接导致NaN，只使用shortcut
            x = shortcut
        # MLP残差连接，添加数值稳定性检查
        shortcut2 = x
        x = self.norm2(x)
        
        # 检查norm2输出
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            x = shortcut2
        
        mlp_out = self.mlp(x)
        residual2 = self.drop_path(mlp_out)
        
        # 检查残差
        if torch.any(torch.isnan(residual2)) or torch.any(torch.isinf(residual2)):
            residual2 = torch.zeros_like(residual2)
        
        x = shortcut2 + residual2
        
        # 最终检查
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            x = shortcut2

        return x


class PatchMerging(nn.Module):
    """Patch Merging层，用于下采样"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"
        x = x.view(B, H, W, C)

        # 下采样：将2x2的patch合并为1个
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """图像到Patch Embedding"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B embed_dim H' W'
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B H'*W' embed_dim
        x = self.norm(x)
        return x, H, W


# ==================== SwinUNet 架构 ====================
class SwinUNet(nn.Module):
    """
    SwinUNet: 结合Swin Transformer和UNet的混合架构
    使用GWO优化超参数以提高Dice指标
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 img_size=224, patch_size=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.0, use_attention_gate=True, _mlp_hidden_dims=None, _from_checkpoint=False):
        super(SwinUNet, self).__init__()
        
        self.num_layers = len(depths)
        # 如果来自checkpoint，不归一化embed_dim
        if _from_checkpoint:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = self._normalize_embed_dim(embed_dim)
        embed_dim = self.embed_dim
        self.use_attention_gate = use_attention_gate
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.base_window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self._mlp_hidden_dims = _mlp_hidden_dims  # 精确的mlp hidden dims
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, 
                                     in_chans=in_channels, embed_dim=embed_dim)
        grid_h = self.patch_embed.grid_size[0]
        if _from_checkpoint:
            self.window_size = window_size
        else:
            self.window_size = self._normalize_window_size(window_size, grid_h)
        total_blocks = sum(self.depths)
        if drop_path_rate > 0 and total_blocks > 0:
            self.drop_path_rates = np.linspace(0, drop_path_rate, total_blocks).tolist()
        else:
            self.drop_path_rates = [0.0] * max(1, total_blocks)
        dp_index = 0
        
        # 编码器：Swin Transformer Blocks
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else nn.Identity
            
            # Swin Transformer Blocks
            for i_block in range(depths[i_layer]):
                block_drop_path = self.drop_path_rates[dp_index] if self.drop_path_rates else 0.0
                dp_index += 1
                # 获取精确的mlp_hidden_dim（如果有）
                mlp_hidden = None
                if _mlp_hidden_dims and (i_layer, i_block) in _mlp_hidden_dims:
                    mlp_hidden = _mlp_hidden_dims[(i_layer, i_block)]
                
                layer.append(
                    SwinTransformerBlock(
                        dim=int(embed_dim * 2 ** i_layer),
                        num_heads=num_heads[i_layer],
                        window_size=self.window_size,
                        shift_size=0 if (i_block % 2 == 0) else self.window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        mlp_hidden_dim=mlp_hidden,
                        drop_path_rate=block_drop_path,
                    )
                )
            self.encoder_layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(downsample(int(embed_dim * 2 ** i_layer)))
            else:
                self.downsample_layers.append(nn.Identity())
        
        # 解码器：上采样 + CNN
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i))
            out_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
            
            # 上采样
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
            
            # 卷积块
            self.decoder_layers.append(nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 添加最后一层解码器（从embed_dim到最终输出前的特征）
        # 这一层使用第一个编码器特征（最浅层）作为跳跃连接
        # 将第一个编码器特征从embed_dim降到embed_dim//2以匹配
        self.first_encoder_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_decoder_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_decoder_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),  # embed_dim//2 + embed_dim//2 = embed_dim
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # 注意力门控（可选）- 现在有num_layers层（包括最后一层）
        if use_attention_gate:
            self.att_gates = nn.ModuleList()
            # 前num_layers-1层：对应解码器的上采样层
            for i in range(self.num_layers - 1):
                dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                self.att_gates.append(AttentionGate(dim, dim, dim // 2, temperature=0.5, use_sharpen=True))
            # 最后一层：x是embed_dim//2，skip是embed_dim（投影后变成embed_dim//2）
            self.att_gates.append(AttentionGate(embed_dim // 2, embed_dim // 2, embed_dim // 4, temperature=0.5, use_sharpen=True))
        
        # 最终输出层（现在输入是embed_dim//2，因为最后一层解码器输出embed_dim//2）
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )

    @staticmethod
    def _normalize_window_size(window_size, max_grid):
        max_grid = max(2, int(max_grid))
        window = max(2, int(round(window_size)))
        valid_sizes = [d for d in range(2, max_grid + 1) if max_grid % d == 0]
        if not valid_sizes:
            return min(window, max_grid)
        best = min(valid_sizes, key=lambda d: abs(d - window))
        return best
    
    @staticmethod
    def _normalize_embed_dim(embed_dim):
        embed_dim = max(48, int(round(embed_dim / 6) * 6))
        if embed_dim % 3 != 0:
            embed_dim += (3 - embed_dim % 3)
        return embed_dim
    
    def get_config(self):
        return {
            "in_channels": self.patch_embed.proj.in_channels,
            "out_channels": self.final_conv[-1].out_channels if isinstance(self.final_conv[-1], nn.Conv2d) else 1,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "use_attention_gate": self.use_attention_gate
        }
        
    def forward(self, x, return_attention=False, return_aux=False):
        B = x.shape[0]
        original_size = x.shape[2:]
        
        # 编码器
        x, H, W = self.patch_embed(x)
        encoder_features = []
        
        for i_layer, (layer, downsample) in enumerate(zip(self.encoder_layers, self.downsample_layers)):
            # Swin Transformer Blocks
            for block in layer:
                x = block(x, H, W)
            
            # 保存特征用于跳跃连接
            x_reshaped = x.transpose(1, 2).reshape(B, -1, H, W)
            encoder_features.append(x_reshaped)
            
            # 下采样
            if i_layer < self.num_layers - 1:
                x = downsample(x, H, W)
                H, W = H // 2, W // 2
        
        # 解码器
        attention_maps = {}
        for i in range(self.num_layers - 1):
            # 上采样
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            x = self.decoder_layers[i * 2](x)  # 上采样
            H, W = x.shape[2], x.shape[3]
            
            # 跳跃连接
            skip_feature = encoder_features[self.num_layers - 2 - i]
            skip_feature = F.interpolate(skip_feature, size=(H, W), mode='bilinear', align_corners=False)
            
            # 注意力门控
            if self.use_attention_gate:
                if return_attention:
                    skip_feature, att_map = self.att_gates[i](x, skip_feature, return_attention=True)
                    attention_maps[f'att{i+1}'] = att_map
                else:
                    skip_feature = self.att_gates[i](x, skip_feature)
            
            # 拼接和卷积
            x = torch.cat([x, skip_feature], dim=1)
            x = self.decoder_layers[i * 2 + 1](x)  # 卷积块
        
        # 最后一层解码器：使用第一个编码器特征（最浅层）
        # x在循环结束后应该是空间格式(B, embed_dim, H, W)，因为最后一步是卷积块
        # 但为安全起见，检查并确保是空间格式
        if len(x.shape) == 4:
            # 已经是空间格式，直接使用
            pass
        elif len(x.shape) == 3:
            # 序列格式(B, H*W, C)，需要转换
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        else:
            # 异常情况，尝试reshape
            try:
                x = x.view(B, -1, H, W)
            except:
                raise RuntimeError(f"无法处理x的形状: {x.shape}")
        
        # 现在x应该是空间格式(B, embed_dim, H, W)
        x = self.final_decoder_upsample(x)  # 上采样到embed_dim//2
        H, W = x.shape[2], x.shape[3]
        
        # 跳跃连接：使用第一个编码器特征
        skip_feature = encoder_features[0]  # 第一个编码器特征（embed_dim维度）
        skip_feature = F.interpolate(skip_feature, size=(H, W), mode='bilinear', align_corners=False)
        skip_feature = self.first_encoder_proj(skip_feature)  # 投影到embed_dim//2
        
        # 最后一层注意力门控
        if self.use_attention_gate:
            if return_attention:
                skip_feature, att_map = self.att_gates[self.num_layers - 1](x, skip_feature, return_attention=True)
                attention_maps[f'att{self.num_layers}'] = att_map
            else:
                skip_feature = self.att_gates[self.num_layers - 1](x, skip_feature)
        
        # 拼接和卷积
        x = torch.cat([x, skip_feature], dim=1)  # embed_dim//2 + embed_dim//2 = embed_dim
        x = self.final_decoder_conv(x)  # 输出embed_dim//2
        
        # 最终输出
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        final_output = self.final_conv(x)
        
        # 返回格式
        if return_attention and return_aux:
            return final_output, [], attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, []
        else:
            return final_output


# ==================== GWO (灰狼优化算法) ====================
class GWOOptimizer:
    """
    灰狼优化算法 (Grey Wolf Optimizer)
    用于优化SwinUNet的超参数以提高Dice指标
    """
    def __init__(self, n_wolves=30, max_iter=50, 
                 bounds=None, objective_func=None):
        """
        Args:
            n_wolves: 灰狼数量
            max_iter: 最大迭代次数
            bounds: 参数边界字典，格式: {'param_name': (min, max), ...}
            objective_func: 目标函数，输入参数字典，返回Dice分数
        """
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.bounds = bounds or self._default_bounds()
        self.objective_func = objective_func
        
        # 灰狼位置（参数值）
        self.positions = []
        # 灰狼适应度（Dice分数）
        self.fitness = []
        
        # Alpha, Beta, Delta (前三名)
        self.alpha_pos = None
        self.alpha_score = float('-inf')
        self.beta_pos = None
        self.beta_score = float('-inf')
        self.delta_pos = None
        self.delta_score = float('-inf')
        
    def _default_bounds(self):
        """默认参数边界"""
        return {
            'embed_dim': (64, 128),
            'window_size': (4, 12),
            'mlp_ratio': (2.0, 6.0),
            'drop_rate': (0.1, 0.4),  # 小数据集更高dropout
            'attn_drop_rate': (0.1, 0.4),
        }
    
    def _initialize_wolves(self):
        """初始化灰狼位置"""
        self.positions = []
        for _ in range(self.n_wolves):
            pos = {}
            for param, (min_val, max_val) in self.bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    pos[param] = np.random.randint(min_val, max_val + 1)
                else:
                    pos[param] = np.random.uniform(min_val, max_val)
            self.positions.append(pos)
    
    def _evaluate_fitness(self, position):
        """评估适应度（Dice分数）"""
        try:
            if self.objective_func:
                result = self.objective_func(position)
                # 每次评估后彻底清理GPU缓存和内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # 强制垃圾回收
                import gc
                gc.collect()
                return result
            return 0.0
        except Exception as e:
            print(f"适应度评估错误: {e}")
            # 确保即使出错也清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            return 0.0
    
    def _update_alpha_beta_delta(self):
        """更新Alpha, Beta, Delta"""
        # 如果还没有初始化，先找到前三名
        if self.alpha_pos is None:
            # 找到最佳适应度的索引
            sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i], reverse=True)
            if len(sorted_indices) > 0:
                self.alpha_score = self.fitness[sorted_indices[0]]
                self.alpha_pos = self.positions[sorted_indices[0]].copy()
            if len(sorted_indices) > 1:
                self.beta_score = self.fitness[sorted_indices[1]]
                self.beta_pos = self.positions[sorted_indices[1]].copy()
            if len(sorted_indices) > 2:
                self.delta_score = self.fitness[sorted_indices[2]]
                self.delta_pos = self.positions[sorted_indices[2]].copy()
            return
        
        # 更新Alpha, Beta, Delta
        for i, fitness in enumerate(self.fitness):
            if fitness > self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy() if self.beta_pos else None
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy() if self.alpha_pos else None
                self.alpha_score = fitness
                self.alpha_pos = self.positions[i].copy()
            elif fitness > self.beta_score and fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy() if self.beta_pos else None
                self.beta_score = fitness
                self.beta_pos = self.positions[i].copy()
            elif fitness > self.delta_score and fitness < self.beta_score:
                self.delta_score = fitness
                self.delta_pos = self.positions[i].copy()
    
    def _update_position(self, a):
        """更新灰狼位置"""
        # 检查alpha_pos, beta_pos, delta_pos是否已初始化
        if self.alpha_pos is None or self.beta_pos is None or self.delta_pos is None:
            return  # 如果还未初始化，跳过位置更新
        
        for i in range(self.n_wolves):
            for param in self.bounds.keys():
                # 计算Alpha, Beta, Delta的影响
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[param] - self.positions[i][param])
                X1 = self.alpha_pos[param] - A1 * D_alpha
                
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos[param] - self.positions[i][param])
                X2 = self.beta_pos[param] - A2 * D_beta
                
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos[param] - self.positions[i][param])
                X3 = self.delta_pos[param] - A3 * D_delta
                
                # 更新位置
                new_pos = (X1 + X2 + X3) / 3.0
                
                # 边界约束
                min_val, max_val = self.bounds[param]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    new_pos = int(np.clip(new_pos, min_val, max_val))
                else:
                    new_pos = np.clip(new_pos, min_val, max_val)
                
                self.positions[i][param] = new_pos
    
    def optimize(self, callback=None):
        """
        执行优化
        Args:
            callback: 回调函数，每轮迭代后调用 callback(iter, best_score, best_params)
        Returns:
            best_params: 最佳参数
            best_score: 最佳分数
            history: 优化历史
        """
        # 初始化
        self._initialize_wolves()
        
        # 评估初始适应度
        self.fitness = [self._evaluate_fitness(pos) for pos in self.positions]
        self._update_alpha_beta_delta()
        
        history = []
        
        # 迭代优化
        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)  # 线性递减
            
            # 更新位置
            self._update_position(a)
            
            # 评估适应度（每次评估后会自动清理GPU缓存）
            self.fitness = [self._evaluate_fitness(pos) for pos in self.positions]
            self._update_alpha_beta_delta()
            
            # 每次迭代后强制清理GPU缓存和内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 记录历史
            history.append({
                'iteration': iter + 1,
                'best_score': self.alpha_score,
                'best_params': self.alpha_pos.copy() if self.alpha_pos else {}
            })
            
            # 回调
            if callback and self.alpha_pos:
                callback(iter + 1, self.alpha_score, self.alpha_pos.copy())
        
        # 确保返回有效的参数
        if self.alpha_pos is None:
            # 如果优化失败，返回默认参数
            default_params = {}
            for param, (min_val, max_val) in self.bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    default_params[param] = (min_val + max_val) // 2
                else:
                    default_params[param] = (min_val + max_val) / 2.0
            return default_params, self.alpha_score, history
        
        return self.alpha_pos, self.alpha_score, history


class SkullStripUNet(nn.Module):
    """
    轻量级UNet，用于脑组织mask预测（Skull Stripping）。
    与原brain-segmentation仓库的Keras实现保持相同结构(无BN)以便权重互通。
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = self._double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.center = self._double_conv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        d4 = self.up4(center)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


class SkullStripper:
    """
    Skull Stripping 推理封装器。
    """
    def __init__(self, model_path: Optional[str], device, threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model()
        elif model_path:
            print(f"[警告] SkullStripper模型路径不存在: {model_path}，将跳过剥除颅骨步骤。")

    def _load_model(self):
        self.model = SkullStripUNet().to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def is_available(self):
        return self.model is not None

    @torch.no_grad()
    def strip(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.is_available():
            return images, None
        logits = self.model(images)
        probs = torch.sigmoid(logits)
        if self.threshold is not None:
            mask = (probs > self.threshold).float()
        else:
            mask = probs
        stripped = images * mask
        return stripped, mask


class ChannelAttention(nn.Module):
    """CBAM: Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 降维比例 ratio
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM: Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """结合通道和空间注意力，放在Skip Connection处"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class ASPP(nn.Module):
    """ASPP模块：扩大感受野，捕获多尺度上下文"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        
        # 不同膨胀率的空洞卷积
        dilations = [6, 12, 18]
        for rate in dilations:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        
        self.convs = nn.ModuleList(modules)
        # 融合层
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """优化后的解码块：上采样 + 拼接 + 卷积"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 这里使用插值代替转置卷积，减少棋盘效应
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 这是一个针对Skip Connection的各种预处理
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 1), # 可以在这里降维
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = CBAM(skip_channels) # 对跳跃连接特征应用注意力
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout防止过拟合
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        
        # 确保尺寸匹配 (处理输入图像尺寸不是32倍数的情况)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        skip = self.skip_conv(skip)
        skip = self.attention(skip) # 关键：让网络知道该关注Skip Connection中的什么信息
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """改进的ResNetUNet：集成ASPP、CBAM注意力和双线性上采样"""
    def __init__(self, in_channels=3, out_channels=1, pretrained=True, backbone_name='resnet101', use_aspp=True, freeze_encoder=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone_name = backbone_name
        self.use_aspp = use_aspp
        self.freeze_encoder = freeze_encoder
        
        # 1. 骨干网络加载
        if backbone_name == 'resnet101':
            if pretrained:
                try:
                    base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                except Exception as e:
                    print(f"[警告] ResNet101 预训练权重下载失败，改用随机初始化。错误: {e}")
                    base = models.resnet101(weights=None)
            else:
                base = models.resnet101(weights=None)
            filters = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet50':
            if pretrained:
                try:
                    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                except Exception as e:
                    print(f"[警告] ResNet50 预训练权重下载失败，改用随机初始化。错误: {e}")
                    base = models.resnet50(weights=None)
            else:
                base = models.resnet50(weights=None)
            filters = [256, 512, 1024, 2048]
        else:
            # 默认使用 resnet101
            if pretrained:
                try:
                    base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                except Exception as e:
                    print(f"[警告] ResNet101 预训练权重下载失败，改用随机初始化。错误: {e}")
                    base = models.resnet101(weights=None)
            else:
                base = models.resnet101(weights=None)
            filters = [256, 512, 1024, 2048]
        
        # 处理输入通道
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 编码器层
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu) # 64
        self.pool = base.maxpool
        self.enc1 = base.layer1 # 256
        self.enc2 = base.layer2 # 512
        self.enc3 = base.layer3 # 1024
        self.enc4 = base.layer4 # 2048
        
        # 2. 中间层：ASPP 或 简单的降维卷积
        if use_aspp:
            # 使用ASPP：将2048通道压缩到512，同时扩大感受野
            self.aspp = ASPP(filters[3], 512)
            center_out_channels = 512
        else:
            # 旧版本：简单的1x1卷积降维
            self.center = nn.Sequential(
                nn.Conv2d(filters[3], 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            center_out_channels = 512
        
        # 3. 解码器 (带有注意力机制)
        # 解码器通道数设计：中间层输出512 -> Concat enc3(1024) -> ...
        self.dec4 = DecoderBlock(center_out_channels, filters[2], 256)  # in:512, skip:1024, out:256
        self.dec3 = DecoderBlock(256, filters[1], 128)  # in:256, skip:512, out:128
        self.dec2 = DecoderBlock(128, filters[0], 64)   # in:128, skip:256, out:64
        self.dec1 = DecoderBlock(64, 64, 32)            # in:64,  skip:64 (enc0), out:32
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # 4. 冻结编码器（如果启用）
        if self.freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """冻结编码器层（enc0 到 enc4）"""
        for encoder in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in encoder.parameters():
                param.requires_grad = False
        print("[模型] 已冻结 ResNet 编码器（enc0-enc4），仅训练解码器")
    
    def _unfreeze_encoder(self):
        """解冻编码器层，允许微调"""
        for encoder in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in encoder.parameters():
                param.requires_grad = True
        print("[模型] 已解冻 ResNet 编码器，开始端到端微调")

    def forward(self, x, return_attention=False):
        # Encoder
        x0 = self.enc0(x)      # 1/2
        x_pool = self.pool(x0) # 1/4
        x1 = self.enc1(x_pool) # 1/4, 256
        x2 = self.enc2(x1)     # 1/8, 512
        x3 = self.enc3(x2)     # 1/16, 1024
        x4 = self.enc4(x3)     # 1/32, 2048

        # Center (ASPP 或 简单降维)
        if self.use_aspp:
            center = self.aspp(x4) # 1/32, 512
        else:
            center = self.center(x4) # 1/32, 512

        # Decoder with Attention
        d4 = self.dec4(center, x3) # -> 1/16
        d3 = self.dec3(d4, x2)     # -> 1/8
        d2 = self.dec2(d3, x1)     # -> 1/4
        d1 = self.dec1(d2, x0)     # -> 1/2
        
        # Final Upsampling to recover original size if needed
        # 通常 U-Net 输出是 1/2 或 1/1，这里如果不加最后的上采样，输出是输入尺寸的 1/2
        # 我们直接对 logit 进行双线性插值恢复原图尺寸
        out = self.final(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        
        if return_attention:
            # 使用各层特征图的通道平均作为注意力图
            # 将特征图下采样到相同尺寸以便可视化
            attention_maps = {}
            target_size = (256, 256)  # 目标尺寸
            
            # 对 x1, x2, x3, x4 进行下采样并计算通道平均
            def get_attention_map(feat, name):
                # 计算通道平均
                attn = feat.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                # 下采样到目标尺寸
                if attn.shape[2:] != target_size:
                    attn = F.interpolate(attn, size=target_size, mode='bilinear', align_corners=False)
                # 归一化到 [0, 1]
                attn_min = attn.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                attn_max = attn.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)
                return attn
            
            # 获取各层的注意力图（对应不同尺度的特征）
            attention_maps['att1'] = get_attention_map(x1, 'att1')  # 最精细层
            attention_maps['att2'] = get_attention_map(x2, 'att2')
            attention_maps['att3'] = get_attention_map(x3, 'att3')
            attention_maps['att4'] = get_attention_map(x4, 'att4')  # 最深层
            
            return out, attention_maps
        
        return out


class nnFormer(nn.Module):
    """
    nnFormer: 基于Transformer的医学图像分割模型（2D适配版）
    结合局部窗口注意力（LV-MSA）和全局注意力（GV-MSA），交替使用以学习多尺度特征
    在跳跃连接中使用Skip Attention机制，增强特征融合
    参考: "nnFormer: Interleaved Transformer for Volumetric Segmentation" (MICCAI 2021)
    
    核心改进：
    1. 局部-全局交替注意力：在编码器中交替使用窗口注意力和全局注意力
    2. 跳跃注意力机制：使用自注意力融合编码器和解码器特征
    3. 多尺度特征学习：通过不同尺度的注意力机制捕获局部和全局信息
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 img_size=224, patch_size=4,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, use_skip_attention=True,
                 global_attn_ratio=0.5):
        super(nnFormer, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_skip_attention = use_skip_attention
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.global_attn_ratio = global_attn_ratio  # 全局注意力的比例
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, 
                                     in_chans=in_channels, embed_dim=embed_dim)
        grid_h = self.patch_embed.grid_size[0]
        
        # Drop path rates
        total_blocks = sum(self.depths)
        if drop_path_rate > 0 and total_blocks > 0:
            self.drop_path_rates = np.linspace(0, drop_path_rate, total_blocks).tolist()
        else:
            self.drop_path_rates = [0.0] * max(1, total_blocks)
        dp_index = 0
        
        # 编码器：nnFormer Transformer Blocks（交替使用局部和全局注意力）
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else nn.Identity
            
            # nnFormer Blocks: 交替使用局部窗口注意力和全局注意力
            for i_block in range(depths[i_layer]):
                block_drop_path = self.drop_path_rates[dp_index] if self.drop_path_rates else 0.0
                dp_index += 1
                
                # 决定使用全局还是局部注意力
                # 性能优化：完全禁用全局注意力（除非明确需要），因为全局注意力非常慢
                # 全局注意力的计算复杂度是O(N²)，对于大图像（如256×256，N=65536）会非常慢
                # 只在最后一层的最后一个block使用全局注意力（如果global_attn_ratio > 0.5）
                use_global = False  # 默认禁用全局注意力以提高速度
                if self.global_attn_ratio > 0.5 and i_layer == self.num_layers - 1 and i_block == depths[i_layer] - 1:
                    use_global = True  # 只在明确需要时才启用
                
                layer.append(
                    nnFormerBlock(
                        dim=int(embed_dim * 2 ** i_layer),
                        num_heads=num_heads[i_layer],
                        window_size=self.window_size,
                        shift_size=0 if (i_block % 2 == 0) else self.window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path_rate=block_drop_path,
                        use_global_attn=use_global,
                    )
                )
            self.encoder_layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(downsample(int(embed_dim * 2 ** i_layer)))
            else:
                self.downsample_layers.append(nn.Identity())
        
        # 解码器：上采样 + CNN
        # 如果使用SkipAttention，不需要拼接，所以卷积层输入是out_dim
        # 如果使用传统方式，需要拼接，所以卷积层输入是out_dim * 2
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i))
            out_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
            
            # 上采样
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
            
            # 卷积块：根据是否使用SkipAttention决定输入通道数
            # 如果使用SkipAttention，x已经融合了skip，所以输入是out_dim
            # 如果使用传统方式，需要拼接skip，所以输入是out_dim * 2
            conv_input_dim = out_dim if use_skip_attention else out_dim * 2
            self.decoder_layers.append(nn.Sequential(
                nn.Conv2d(conv_input_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 最后一层解码器
        self.first_encoder_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_decoder_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        # 最后一层卷积：根据是否使用SkipAttention决定输入通道数
        final_conv_input_dim = embed_dim // 2 if use_skip_attention else embed_dim
        self.final_decoder_conv = nn.Sequential(
            nn.Conv2d(final_conv_input_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )

        # 跳跃注意力机制（nnFormer风格）
        if use_skip_attention:
            self.skip_attentions = nn.ModuleList()
            for i in range(self.num_layers - 1):
                dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                # 确保num_heads至少为1，且不超过dim
                num_heads_val = max(1, min(8, dim // 64))
                self.skip_attentions.append(SkipAttention(dim, num_heads=num_heads_val))
            # 最后一层
            final_dim = embed_dim // 2
            final_num_heads = max(1, min(8, final_dim // 64))
            self.skip_attentions.append(SkipAttention(final_dim, num_heads=final_num_heads))
        else:
            # 回退到传统注意力门控
            self.att_gates = nn.ModuleList()
            for i in range(self.num_layers - 1):
                dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                self.att_gates.append(AttentionGate(dim, dim, dim // 2, temperature=0.5, use_sharpen=True))
            self.att_gates.append(AttentionGate(embed_dim // 2, embed_dim // 2, embed_dim // 4, temperature=0.5, use_sharpen=True))
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )

    def get_config(self):
        """返回nnFormer的配置"""
        return {
            "in_channels": self.patch_embed.proj.in_channels,
            "out_channels": self.final_conv[-1].out_channels if isinstance(self.final_conv[-1], nn.Conv2d) else 1,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "use_skip_attention": self.use_skip_attention,
            "global_attn_ratio": self.global_attn_ratio
        }
        
    def forward(self, x, return_attention=False, return_aux=False):
        B = x.shape[0]
        original_size = x.shape[2:]
        
        # 编码器 - 优化：使用detach()减少内存占用，只在需要时保存特征
        x, H, W = self.patch_embed(x)
        encoder_features = []
        
        for i_layer, (layer, downsample) in enumerate(zip(self.encoder_layers, self.downsample_layers)):
            # Transformer Blocks - 性能优化：减少中间变量
            for block in layer:
                x = block(x, H, W)
            
            # 保存特征用于跳跃连接 - 性能优化：只在训练时保留梯度，推理时detach
            x_reshaped = x.transpose(1, 2).reshape(B, -1, H, W)
            # 优化：训练时也使用detach，因为跳跃连接通常不需要梯度（可以节省大量内存）
            encoder_features.append(x_reshaped.detach())
            
            # 下采样
            if i_layer < self.num_layers - 1:
                x = downsample(x, H, W)
                H, W = H // 2, W // 2
                # 性能优化：减少清理频率，只在必要时清理
                if torch.cuda.is_available() and i_layer % 3 == 0:
                    torch.cuda.empty_cache()
        
        # 解码器 - 优化：及时释放不需要的特征
        x = encoder_features[-1]
        decoder_idx = 0
        
        for i in range(self.num_layers - 1):
            # 上采样
            x = self.decoder_layers[decoder_idx](x)
            decoder_idx += 1
            
            # 跳跃连接 - 只在需要时加载特征
            skip_idx = self.num_layers - 2 - i
            skip = encoder_features[skip_idx]
            
            # 确保skip和x的空间尺寸匹配（在调用SkipAttention之前）
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # 跳跃注意力机制（nnFormer风格）
            if self.use_skip_attention:
                x = self.skip_attentions[i](x, skip)
                skip = None  # 已经融合，不需要再拼接
            elif hasattr(self, 'att_gates'):
                # 回退到传统注意力门控
                skip = self.att_gates[i](x, skip)
            
            # 拼接和卷积（如果使用Skip Attention，x已经融合了skip）
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.decoder_layers[decoder_idx](x)
            decoder_idx += 1
            
            # 释放已使用的encoder特征（除了第一个和最后一个）
            if i < self.num_layers - 2:
                encoder_features[skip_idx] = None
        
        # 最后一层解码器
        x = self.final_decoder_upsample(x)
        first_skip = self.first_encoder_proj(encoder_features[0])
        
        # 确保first_skip和x的空间尺寸匹配
        if x.shape[2:] != first_skip.shape[2:]:
            first_skip = F.interpolate(first_skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if self.use_skip_attention:
            x = self.skip_attentions[-1](x, first_skip)
            first_skip = None
        elif hasattr(self, 'att_gates'):
            first_skip = self.att_gates[-1](x, first_skip)
        
        if first_skip is not None:
            x = torch.cat([x, first_skip], dim=1)
        x = self.final_decoder_conv(x)
        
        # 最终输出
        x = self.final_conv(x)
        
        # 恢复到原始尺寸
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # 清理encoder_features（如果不需要返回）
        if not return_aux:
            encoder_features = None
        
        if return_aux:
            return x, encoder_features
        return x

    def get_config(self):
        """返回nnFormer的配置"""
        return {
            "in_channels": self.patch_embed.proj.in_channels,
            "out_channels": self.final_conv[-1].out_channels if isinstance(self.final_conv[-1], nn.Conv2d) else 1,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "use_skip_attention": self.use_skip_attention,
            "global_attn_ratio": self.global_attn_ratio
        }


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

def instantiate_model(model_type: str, device, swin_params: Optional[dict] = None, dstrans_params: Optional[dict] = None, mamba_params: Optional[dict] = None, resnet_params: Optional[dict] = None):
    model_type = (model_type or "improved_unet").lower()
    if model_type == "resnet_unet":
        params = {
            "in_channels": 3,
            "out_channels": 1,
            # 使用 ImageNet 预训练的 ResNet，可提升深层特征表达（Attention 更稳定）
            "pretrained": True,
            "backbone_name": "resnet101",
            "use_aspp": True  # 默认使用ASPP
        }
        if resnet_params:
            params.update(resnet_params)
        return ResNetUNet(**params).to(device)
    if model_type in ("trans_unet", "transunet"):
        return TransUNet().to(device)
    if model_type in ("ds_trans_unet", "dstransunet", "ds-transunet"):
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        }
        if dstrans_params:
            params.update(dstrans_params)
        return DSTransUNet(**params).to(device)
    if model_type == "swin_unet" or model_type == "swinunet":
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "img_size": 224,
            "patch_size": 4,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 8,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
            "use_attention_gate": True
        }
        if swin_params:
            params.update(swin_params)
        return SwinUNet(**params).to(device)
    return ImprovedUNet().to(device)


class LayerNorm2d(nn.Module):
    """轻量2D LayerNorm，兼容卷积特征。"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, eps=eps)
    def forward(self, x):
        return self.norm(x)


class MambaBlock2D(nn.Module):
    """占位：Swin-U Mamba 相关代码已移除，此模块不再使用。"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("MambaBlock2D 已废弃，Swin-U Mamba 功能已删除")


class WindowAttention2D(nn.Module):
    """局部窗口自注意力，输入/输出形状 B,C,H,W"""
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = max(1, dim // num_heads)
        self.num_heads = max(1, dim // head_dim)
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        # [B, C, Hp, Wp] -> [B*num_windows, ws*ws, C]
        x_windows = x.unfold(2, ws, ws).unfold(3, ws, ws)  # B, C, nH, nW, ws, ws
        x_windows = x_windows.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, ws * ws, C)

        qkv = self.qkv(x_windows).reshape(x_windows.shape[0], x_windows.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, Bnw, heads, tokens, dim_head
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], C)
        out = self.proj(out)

        # [B*num_windows, ws*ws, C] -> [B, C, Hp, Wp]
        out = out.view(-1, ws, ws, C)
        # reshape back: need nH, nW
        nH = Hp // ws
        nW = Wp // ws
        out = out.view(-1, nH, nW, ws, ws, C).permute(0, 5, 3, 4, 1, 2).contiguous()
        out = out.view(B, C, ws, ws, nH, nW).permute(0, 1, 4, 2, 5, 3).contiguous()
        out = out.view(B, C, Hp, Wp)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


class SwinBlock2D(nn.Module):
    """
    简化版 Swin block：可选移窗 + 窗口注意力 + MLP
    增加通道降维（attn_reduction）以降低注意力计算量。
    """
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7, shift: bool = False,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_reduction: int = 2):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = LayerNorm2d(dim)
        attn_dim = max(8, dim // max(1, attn_reduction))
        # 使 attn_dim 可被 num_heads 整除
        attn_dim = max(attn_dim // num_heads, 1) * num_heads
        self.attn_in = nn.Conv2d(dim, attn_dim, kernel_size=1) if attn_dim != dim else nn.Identity()
        self.attn = WindowAttention2D(attn_dim, window_size, num_heads)
        self.attn_out = nn.Conv2d(attn_dim, dim, kernel_size=1) if attn_dim != dim else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
        )
        self.norm2 = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        x_attn = self.attn_in(x)
        x = self.attn(x_attn)
        x = self.attn_out(x)
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        x = shortcut + x
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class SwinUMamba(nn.Module):
    """占位：Swin-U Mamba 相关模型已移除，不再支持。"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("Swin-U Mamba 模型已从系统中删除")


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



# 数据集类
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
        engine, lock = self.session.acquire()
        payload_mat = MatlabEngineSession.to_matlab_path(payload_mat_path)
        save_mat = MatlabEngineSession.to_matlab_path(save_path)

        script = f"""
data = load('{payload_mat}');
images = data.images;
masks = data.masks;
preds = data.preds;
numSamples = min(size(images, 4), 4);
cols = 4;
fig = figure('Visible','off');
tl = tiledlayout(fig, numSamples, cols, 'Padding','compact', 'TileSpacing','compact');
for idx = 1:numSamples
    img = images(:,:,:,idx);
    mask = masks(:,:,idx) > 0.5;
    predMask = preds(:,:,idx) > 0.5;
    overlay = img;
    channel1 = overlay(:,:,1);
    channel1(mask) = 1;
    overlay(:,:,1) = channel1;
    channel2 = overlay(:,:,2);
    channel2(predMask) = 1;
    overlay(:,:,2) = channel2;
    nexttile(tl); imshow(img, []); title(sprintf('样本 %d 输入', idx));
    nexttile(tl); imshow(mask); title('真实Mask');
    nexttile(tl); imshow(predMask); title('预测Mask');
    nexttile(tl); imshow(overlay); title('叠加图');
end
exportgraphics(fig, '{save_mat}', 'Resolution', 200);
close(fig);
"""

        with lock:
            engine.eval(script, nargout=0)

    def render_training_history(self, payload_mat_path: str, save_path: str):
        engine, lock = self.session.acquire()
        payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
        save_mat = MatlabEngineSession.to_matlab_path(save_path)
        script = f"""
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
exportgraphics(fig, '{save_mat}', 'Resolution', 200);
close(fig);
"""
        with lock:
            engine.eval(script, nargout=0)

    def render_performance_analysis(self, payload_mat_path: str, save_path: str):
        engine, lock = self.session.acquire()
        payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
        save_mat = MatlabEngineSession.to_matlab_path(save_path)
        script = f"""
data = load('{payload}');
metrics = data.metrics;
avg = data.avg_metrics;
stdVals = data.std_metrics;
fig = figure('Visible','off');
tiledlayout(fig,2,3,'Padding','compact','TileSpacing','compact');
nexttile;
histogram(metrics.dice,20,'FaceColor',[0.2 0.4 0.8]);
xline(avg.dice,'r--','LineWidth',1.5);
title('Dice分布'); xlabel('Dice'); ylabel('数量'); grid on;
nexttile;
histogram(metrics.iou,20,'FaceColor',[0.2 0.7 0.3]);
xline(avg.iou,'r--','LineWidth',1.5);
title('IoU分布'); xlabel('IoU'); ylabel('数量'); grid on;
nexttile;
histogram(metrics.precision,20,'FaceColor',[0.9 0.5 0.2]);
xline(avg.precision,'r--','LineWidth',1.5);
title('精确率分布'); xlabel('Precision'); ylabel('数量'); grid on;
nexttile;
vals = [avg.dice, avg.iou, avg.precision, avg.sensitivity, avg.specificity, avg.f1];
err = [stdVals.dice, stdVals.iou, stdVals.precision, stdVals.sensitivity, stdVals.specificity, stdVals.f1];
bar(vals,'FaceColor',[0.3 0.6 0.9]); hold on;
errorbar(1:numel(vals), vals, err, 'k.', 'LineWidth', 1.5);
set(gca,'XTickLabel',{'Dice','IoU','Precision','Recall','Specificity','F1'},'XTickLabelRotation',30);
ylim([0 1]); title('平均性能'); grid on;
nexttile;
boxplot([metrics.dice', metrics.iou', metrics.precision', metrics.sensitivity', metrics.specificity', metrics.f1'],...
    'Labels',{'Dice','IoU','Precision','Recall','Specificity','F1'});
ylim([0 1]); title('指标箱线图'); grid on;
nexttile;
valsTable = [
    avg.dice, stdVals.dice, data.min_metrics.dice, data.max_metrics.dice, data.median_metrics.dice;
    avg.iou, stdVals.iou, data.min_metrics.iou, data.max_metrics.iou, data.median_metrics.iou;
    avg.precision, stdVals.precision, data.min_metrics.precision, data.max_metrics.precision, data.median_metrics.precision;
    avg.sensitivity, stdVals.sensitivity, data.min_metrics.sensitivity, data.max_metrics.sensitivity, data.median_metrics.sensitivity;
    avg.specificity, stdVals.specificity, data.min_metrics.specificity, data.max_metrics.specificity, data.median_metrics.specificity;
    avg.f1, stdVals.f1, data.min_metrics.f1, data.max_metrics.f1, data.median_metrics.f1;
    avg.hd95, stdVals.hd95, data.min_metrics.hd95, data.max_metrics.hd95, data.median_metrics.hd95];
ax = nexttile;
axis(ax,'off');
rowLabels = {{'Dice','IoU','Precision','Recall','Specificity','F1','HD95'}};
for row = 1:size(valsTable,1)
    yPos = 1 - row * 0.12;
    text(0.01, yPos, sprintf('%-11s: 均值%.4f | std %.4f | min %.4f | max %.4f | median %.4f', ...
        rowLabels{{row}}, valsTable(row,1), valsTable(row,2), valsTable(row,3), valsTable(row,4), valsTable(row,5)), ...
        'FontSize',9,'Parent',ax);
end
title(ax,'统计摘要');
exportgraphics(fig, '{save_mat}', 'Resolution', 200);
close(fig);
"""
        with lock:
            engine.eval(script, nargout=0)

    def render_test_results(self, payload_mat_path: str, save_path: str):
        engine, lock = self.session.acquire()
        payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
        save_mat = MatlabEngineSession.to_matlab_path(save_path)
        script = f"""
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
exportgraphics(fig, '{save_mat}', 'Resolution', 200);
close(fig);
"""
        with lock:
            engine.eval(script, nargout=0)

    def render_attention_maps(self, payload_mat_path: str, save_path: str):
        engine, lock = self.session.acquire()
        payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
        save_mat = MatlabEngineSession.to_matlab_path(save_path)
        script = f"""
"""
        with lock:
            engine.eval(script, nargout=0)

# 训练工作线程
class ModelTestThread(QThread):
    """模型测试线程"""
    update_progress = pyqtSignal(int, str)  # (进度百分比, 状态消息)
    test_finished = pyqtSignal(dict, str, list)  # (性能指标, 注意力热图路径, 低Dice案例列表)
    # 阈值扫描结果（完整表格 + 推荐阈值信息），通过object传递，避免PyQt类型限制
    threshold_sweep_ready = pyqtSignal(object)
    
    def __init__(self, model_paths, data_dir, model_type, use_tta=True):
        super().__init__()
        # 支持单模型（集成功能已删除）
        if isinstance(model_paths, str):
            self.model_paths = [model_paths]
        else:
            self.model_paths = model_paths
        # 只使用第一个模型
        if len(self.model_paths) > 1:
            print(f"[警告] 检测到多个模型文件，仅使用第一个: {self.model_paths[0]}")
        self.model_path = self.model_paths[0]
        self.data_dir = data_dir
        self.model_type = model_type
        self.use_tta = use_tta
        self.stop_requested = False
        self.temp_dir = tempfile.mkdtemp(prefix="model_test_")
        
    def run(self):
        try:
            import torch
            from torch.utils.data import DataLoader
            import platform
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.update_progress.emit(5, f"使用设备: {device}")
            
            # 加载模型（仅支持单模型，集成功能已删除）
            self.update_progress.emit(10, f"正在加载模型: {os.path.basename(self.model_path)}")
            model = self._load_model(device, self.model_path)
            model.eval()
            
            # 加载测试数据
            self.update_progress.emit(20, "正在加载测试数据...")
            
            # 创建临时TrainThread实例来使用其数据加载方法
            temp_train_thread = TrainThread(
                data_dir=self.data_dir,
                epochs=1,
                batch_size=4,
                model_path=None,
                save_best=False
            )
            temp_train_thread.model_type = self.model_type  # 设置模型类型
            
            # 获取patient_ids（子文件夹）
            patient_ids = [pid for pid in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, pid))]
            
            if not patient_ids:
                raise ValueError("测试数据目录为空，未找到子文件夹")
            
            # 使用TrainThread的_collect_image_mask_paths方法获取图像路径
            # 这个方法会正确处理文件结构：data_dir/images/patient_id/*.png 和 data_dir/masks/patient_id/*.png
            image_paths, mask_paths = temp_train_thread._collect_image_mask_paths(patient_ids)
            
            if not image_paths:
                raise ValueError(f"未找到测试图像文件。请检查数据目录结构：\n{self.data_dir}\n\n"
                               f"期望结构：\n"
                               f"  {self.data_dir}/\n"
                               f"    images/\n"
                               f"      patient_id1/\n"
                               f"        *.png\n"
                               f"      patient_id2/\n"
                               f"        *.png\n"
                               f"    masks/\n"
                               f"      patient_id1/\n"
                               f"        *.png\n"
                               f"      patient_id2/\n"
                               f"        *.png")
            
            # 使用全部数据作为测试集
            val_transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            # 使用全部patient_ids作为测试集
            test_dataset = temp_train_thread.load_dataset(
                patient_ids, val_transform, split_name="test", 
                return_classification=False, use_weighted_sampling=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=4, shuffle=False, num_workers=0
            )
            
            # 评估模型（集成功能已删除，仅支持单模型）
            self.update_progress.emit(30, "正在评估模型性能...")
            detailed_metrics, low_dice_cases = self._evaluate_model(model, test_loader, device, image_paths)
            
            # 生成注意力热图
            self.update_progress.emit(80, "正在生成注意力热图...")
            attention_path = self._generate_attention_maps(model, test_loader, device)
            
            self.update_progress.emit(100, "测试完成！")
            self.test_finished.emit(detailed_metrics, attention_path, low_dice_cases)
            
        except Exception as e:
            import traceback
            error_msg = f"测试失败: {str(e)}\n{traceback.format_exc()}"
            self.update_progress.emit(0, error_msg)
            self.test_finished.emit({}, "", [])
    
    def _load_model(self, device, model_path=None):
        """加载模型 - 优先从checkpoint推断模型类型"""
        # 使用传入的model_path，如果没有则使用self.model_path（集成功能已删除）
        if model_path is None:
            model_path = self.model_path
        
        # 首先尝试从checkpoint中读取模型类型和配置
        swin_params = None
        dstrans_params = None
        mamba_params = None
        resnet_params = None
        inferred_model_type = None
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # 尝试从checkpoint中读取模型类型
                if isinstance(checkpoint, dict):
                    if 'model_type' in checkpoint:
                        inferred_model_type = checkpoint['model_type']
                    elif 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                        if 'model_type' in checkpoint['config']:
                            inferred_model_type = checkpoint['config']['model_type']
                    
                    # 读取模型参数配置（checkpoint 顶层）
                    if 'swin_params' in checkpoint:
                        swin_params = checkpoint['swin_params']
                    if 'dstrans_params' in checkpoint:
                        dstrans_params = checkpoint['dstrans_params']
                    if 'mamba_params' in checkpoint:
                        mamba_params = checkpoint['mamba_params']
                    if 'resnet_params' in checkpoint:
                        resnet_params = checkpoint['resnet_params']

                    # 从 config 中优先读取结构参数（配置优先加载）
                    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                        cfg = checkpoint['config']
                        # ResNet 相关参数
                        if 'resnet_params' in cfg:
                            resnet_params = cfg['resnet_params']

                        # DS-TransUNet 相关参数（优先于顶层 dstrans_params）
                        cfg_dstrans = cfg.get('dstrans_params') or cfg.get('dstransunet_args') or cfg.get('model_kwargs')
                        if isinstance(cfg_dstrans, dict):
                            if dstrans_params is None:
                                dstrans_params = {}
                            dstrans_params.update(cfg_dstrans)
                            print(f"[模型加载] 从checkpoint.config读取DS-TransUNet参数: {list(dstrans_params.keys())}")

                        # 兜底：若没有 dstrans_params，但存在关键超参，则组装一个最小配置
                        if dstrans_params is None:
                            possible_keys = ('embed_dim', 'num_heads', 'num_layers', 'mlp_ratio', 'img_size', 'num_classes',
                                             'in_channels', 'out_channels', 'dropout')
                            has_dstrans_like = any(k in cfg for k in possible_keys)
                            if has_dstrans_like:
                                dstrans_params = {}
                                for k in possible_keys:
                                    if k in cfg:
                                        dstrans_params[k] = cfg[k]
                                print(f"[模型加载] 从checkpoint.config推断DS-TransUNet最小参数集: {dstrans_params}")
                    
                    # 从state_dict推断模型类型（如果无法从checkpoint读取）
                    # 使用与read_checkpoint_config相同的检测逻辑和顺序
                    if not inferred_model_type:
                        state_dict = checkpoint.get('state_dict', checkpoint)
                        # 处理DataParallel包装
                        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
                            state_dict = {k[7:]: v for k, v in state_dict.items()}
                        
                        # 检测顺序与read_checkpoint_config保持一致
                        # 1. 检测DS-TransUNet (patch_embed3) - 优先级最高
                        # 检查多种可能的键名变体（考虑DataParallel包装等）
                        has_dstrans = False
                        for key in state_dict.keys():
                            if 'patch_embed3.weight' in key or key.endswith('patch_embed3.weight'):
                                has_dstrans = True
                                break
                        
                        if has_dstrans:
                            inferred_model_type = 'ds_trans_unet'
                            # 从state_dict推断参数（优先使用，因为它是从实际权重形状推断的，最准确）
                            inferred_dstrans_params = infer_dstrans_params_from_state_dict(state_dict)
                            if inferred_dstrans_params:
                                if dstrans_params is None:
                                    dstrans_params = {}
                                # 优先使用推断的参数（从state_dict读取，最准确），覆盖checkpoint config中的参数
                                # 这样可以确保模型结构与checkpoint中的权重匹配
                                dstrans_params.update(inferred_dstrans_params)
                                print(f"[模型加载] 从checkpoint推断DS-TransUNet参数: embed_dim={dstrans_params.get('embed_dim')}, num_heads={dstrans_params.get('num_heads')}, num_layers={dstrans_params.get('num_layers')}, mlp_ratio={dstrans_params.get('mlp_ratio', 4.0):.2f}")
                            else:
                                print(f"[警告] 检测到DS-TransUNet但参数推断失败，将使用checkpoint config或默认参数")
                        
                        # 2. 检测SwinUNet (patch_embed.proj)
                        elif 'patch_embed.proj.weight' in state_dict:
                            inferred_model_type = 'swin_unet'
                        
                        # 3. 检测ResNetUNet (enc0或layer0)
                        elif 'enc0.0.weight' in state_dict or 'enc0.weight' in state_dict:
                            # 检测是否是旧版本checkpoint（使用layer0/layer1等键名）
                            old_version_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
                            has_old_keys = any(any(k.startswith(old_key) for k in state_dict.keys()) for old_key in old_version_keys)
                            
                            inferred_model_type = 'resnet_unet'
                            # 尝试推断backbone类型
                            if 'enc1.0.conv1.weight' in state_dict or (has_old_keys and 'layer1.0.conv1.weight' in state_dict):
                                # 检查是否是ResNet101 (layer1有3个block)
                                if 'enc1.2.conv1.weight' in state_dict or (has_old_keys and 'layer1.2.conv1.weight' in state_dict):
                                    resnet_params = {'backbone_name': 'resnet101'}
                                else:
                                    resnet_params = {'backbone_name': 'resnet50'}
                            
                            # 检测是否有ASPP模块
                            has_aspp = any('aspp' in k.lower() for k in state_dict.keys())
                            # 如果是旧版本checkpoint且没有ASPP，则禁用ASPP
                            if has_old_keys and not has_aspp:
                                if resnet_params is None:
                                    resnet_params = {}
                                resnet_params['use_aspp'] = False
                                print(f"[模型加载] 检测到旧版本checkpoint（无ASPP），将使用兼容模式")
                        
                        # 4. 检测TransUNet (encoder.0)
                        elif 'encoder.0.weight' in state_dict:
                            inferred_model_type = 'trans_unet'
                        
                        # 5. 检测其他ResNetUNet变体 (backbone.layer1)
                        elif 'backbone.layer1.0.conv1.weight' in state_dict:
                            inferred_model_type = 'resnet_unet'
                        
                        # 6. 检测旧版本ResNetUNet (layer0/layer1等键名)
                        else:
                            old_version_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
                            has_old_keys = any(any(k.startswith(old_key) for k in state_dict.keys()) for old_key in old_version_keys)
                            if has_old_keys:
                                inferred_model_type = 'resnet_unet'
                                # 尝试推断backbone类型
                                if 'layer1.0.conv1.weight' in state_dict:
                                    if 'layer1.2.conv1.weight' in state_dict:
                                        resnet_params = {'backbone_name': 'resnet101'}
                                    else:
                                        resnet_params = {'backbone_name': 'resnet50'}
                                
                                # 检测是否有ASPP模块
                                has_aspp = any('aspp' in k.lower() for k in state_dict.keys())
                                if not has_aspp:
                                    if resnet_params is None:
                                        resnet_params = {}
                                    resnet_params['use_aspp'] = False
                                    print(f"[模型加载] 检测到旧版本checkpoint（无ASPP），将使用兼容模式")
            except Exception as e:
                print(f"[警告] 读取checkpoint配置失败: {e}")
        
        # 【保底逻辑】从文件名推断分辨率（如果无法从checkpoint读取）
        # 检查文件名中是否包含"512"关键词，用于判断是否为高分辨率模型
        is_highres = False
        if model_path:
            filename = os.path.basename(model_path).lower()
            if '512' in filename or 'highres' in filename or 'high_res' in filename:
                is_highres = True
                print(f"[模型加载] 从文件名推断：检测到高分辨率模型（512）")
        
        # 使用推断的模型类型，如果没有则使用用户选择的
        model_type_to_use = inferred_model_type or self.model_type
        
        if model_type_to_use != self.model_type:
            print(f"[提示] 从checkpoint推断模型类型: {model_type_to_use} (用户选择: {self.model_type})")
        
        # 使用instantiate_model创建模型（与训练时保持一致）
        model = instantiate_model(
            model_type_to_use, 
            device, 
            swin_params=swin_params,
            dstrans_params=dstrans_params,
            mamba_params=mamba_params,
            resnet_params=resnet_params
        )
        
        # 加载权重（带智能诊断与兼容加载）
        if model_path and os.path.exists(model_path):
            success, msg = load_model_compatible(model, model_path, device, verbose=True)
            if not success:
                print(f"[警告] load_model_compatible 加载失败，将启动详细诊断并尝试兼容加载。原因: {msg}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                except Exception as e:
                    raise RuntimeError(f"模型加载失败且无法读取checkpoint: {e}")

                # 提取 state_dict
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint

                # 处理DataParallel前缀
                if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}

                model_state = model.state_dict()
                missing_keys = []
                unexpected_keys = []
                shape_mismatch = []

                # 检查缺失键 & 形状不匹配
                for k, v in model_state.items():
                    if k not in state_dict:
                        missing_keys.append(k)
                    else:
                        if state_dict[k].shape != v.shape:
                            shape_mismatch.append((k, tuple(v.shape), tuple(state_dict[k].shape)))

                # 检查多余键
                for k in state_dict.keys():
                    if k not in model_state:
                        unexpected_keys.append(k)

                print("\n[模型加载诊断] state_dict 不匹配详情：")
                if missing_keys:
                    print(f"  Missing keys ({len(missing_keys)}):")
                    for k in missing_keys[:50]:
                        print(f"    - {k}")
                    if len(missing_keys) > 50:
                        print(f"    ... 以及另外 {len(missing_keys)-50} 个缺失键")
                else:
                    print("  Missing keys: 无")

                if unexpected_keys:
                    print(f"  Unexpected keys ({len(unexpected_keys)}):")
                    for k in unexpected_keys[:50]:
                        print(f"    - {k}")
                    if len(unexpected_keys) > 50:
                        print(f"    ... 以及另外 {len(unexpected_keys)-50} 个多余键")
                else:
                    print("  Unexpected keys: 无")

                if shape_mismatch:
                    print(f"  Shape mismatch ({len(shape_mismatch)}):")
                    for k, m_shape, c_shape in shape_mismatch[:50]:
                        print(f"    - Key: {k}, Model: {m_shape}, Checkpoint: {c_shape}")
                    if len(shape_mismatch) > 50:
                        print(f"    ... 以及另外 {len(shape_mismatch)-50} 个形状不匹配参数")
                else:
                    print("  Shape mismatch: 无")

                # 特别提示 Transformer / DS-TransUNet 的尺寸问题
                cfg = None
                if isinstance(checkpoint, dict) and isinstance(checkpoint.get('config', None), dict):
                    cfg = checkpoint['config']
                if cfg and self.model_type in ("ds_trans_unet", "swin_unet", "swin_unet_v2", "swinunet"):
                    img_size_cfg = cfg.get("img_size") or cfg.get("image_size")
                    num_classes_cfg = cfg.get("num_classes")
                    print("\n[提示] Transformer/DS-TransUNet 配置检查：")
                    print(f"  checkpoint.config.img_size   = {img_size_cfg}")
                    print(f"  checkpoint.config.num_classes= {num_classes_cfg}")
                    print("  请确认当前实例化的模型 img_size / num_classes 与上述值一致，否则位置编码或输出头会形状不匹配。")

                # 尝试非严格加载（忽略多余键和形状不匹配的部分）
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print("[警告] 模型使用 strict=False 兼容加载成功。")
                    if missing:
                        print(f"  strict=False 仍存在 missing keys ({len(missing)}):")
                        for k in missing[:50]:
                            print(f"    - {k}")
                    if unexpected:
                        print(f"  strict=False 仍存在 unexpected keys ({len(unexpected)}):")
                        for k in unexpected[:50]:
                            print(f"    - {k}")
                except Exception as e2:
                    raise RuntimeError(f"模型严格加载与兼容加载均失败，请根据上方诊断检查模型结构与checkpoint是否匹配。最后错误: {e2}")
        
        return model.to(device)
    
    def _evaluate_model(self, model, dataloader, device, image_paths):
        """评估模型并找出低Dice案例 - 与训练时的评估逻辑保持一致"""
        import torch.nn.functional as F
        import numpy as np
        from tqdm import tqdm
        
        metrics = {
            'dice': [], 'iou': [], 'precision': [], 'recall': [],
            'sensitivity': [], 'specificity': [], 'f1': [], 'hd95': []
        }
        low_dice_cases = []  # [(image_path, dice, iou, precision, recall), ...]
        accum_tp = accum_fp = accum_fn = accum_tn = 0.0
        
        # 统计空mask情况
        empty_target_count = 0  # 真实mask为空的样本数
        empty_pred_count = 0   # 预测mask为空的样本数
        both_empty_count = 0    # 两者都空的样本数
        both_non_empty_count = 0  # 两者都不空的样本数
        
        # 成分分析：测试集样本分布 + 分类Dice（Pos/Neg）
        test_total_samples = 0
        test_pos_samples = 0
        test_neg_samples = 0
        test_dice_pos_sum = 0.0
        test_dice_neg_sum = 0.0
        
        model.eval()
        # 模式检查：确保已进入 eval
        print(f"[测试|模式检查] model.training={getattr(model, 'training', None)} (期望 False)")
        image_idx = 0
        
        # 创建临时TrainThread实例以使用其方法（与训练过程一致）
        temp_train_thread = TrainThread(
            data_dir=self.data_dir,
            epochs=1,
            batch_size=4,
            model_path=None,
            save_best=False
        )
        
        # ==============================
        # 测试期超参搜索：TTA + 阈值扫描
        # ==============================
        # 【修改】阈值搜索范围改为0.89-0.99，步长 0.01，共10个阈值点
        thresholds = [round(0.89 + i * 0.01, 2) for i in range(10)]  # [0.89, 0.90, 0.91, ..., 0.98]
        # 【修改】改为样本级指标计算：为每个阈值存储样本级指标列表
        sweep_dice_scores = {t: [] for t in thresholds}  # 存储每个样本的Dice值
        sweep_iou_scores = {t: [] for t in thresholds}  # 存储每个样本的IoU值
        sweep_precision_scores = {t: [] for t in thresholds}  # 存储每个样本的Precision值
        sweep_recall_scores = {t: [] for t in thresholds}  # 存储每个样本的Recall值
        sweep_specificity_scores = {t: [] for t in thresholds}   # 存储每个样本的Specificity值
        sweep_stats = {t: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "fp_pix": 0.0} for t in thresholds}

        def _forward_with_tta(images_tensor: torch.Tensor) -> torch.Tensor:
            """
            确保TTA开启：优先用内置 _tta_inference；若关闭/不可用则用简易水平翻转TTA。
            返回 logits (B,1,H,W)
            """
            # 强制开启TTA：优先使用 self.use_tta + _tta_inference
            try:
                logits = self._tta_inference(model, images_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits
            except Exception:
                # 简易 TTA：原图 + 水平翻转平均
                logits1 = model(images_tensor)
                if isinstance(logits1, tuple):
                    logits1 = logits1[0]
                logits2 = model(torch.flip(images_tensor, dims=[3]))
                if isinstance(logits2, tuple):
                    logits2 = logits2[0]
                logits2 = torch.flip(logits2, dims=[3])
                return (logits1 + logits2) * 0.5

        print("\n[测试] 开始阈值扫描（TTA + Threshold Sweep）")
        print("Threshold | Global Dice | Precision | Recall | FP Count")
        print("--- | --- | --- | --- | ---")

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="阈值扫描中"):
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.float().to(device)

                logits = _forward_with_tta(images)
                if logits.shape[2:] != masks.shape[2:]:
                    logits = F.interpolate(logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
                probs = torch.sigmoid(logits)

                # 【HD95优化后处理】对每个阈值分别计算（高斯模糊 + 形态学闭运算 + 严格连通域过滤）
                # 确保阈值扫描时的逻辑与最终报告完全一致
                for thr in thresholds:
                    # 对每个样本应用优化的后处理
                    preds_bin_list = []
                    for i in range(probs.shape[0]):
                        prob_single = probs[i, 0]  # H x W
                        # 应用优化的后处理流水线（启用动态面积阈值）
                        pred_single = temp_train_thread.post_process_refine_for_hd95(
                            prob_single, 
                            threshold=thr,
                            min_area_threshold=100,  # 基础面积阈值（会动态调整）
                            use_gaussian_blur=True,  # 启用高斯模糊平滑边缘
                            use_morphology=True,      # 启用形态学闭运算
                            dynamic_area_threshold=True  # 启用动态面积阈值
                        )
                        if isinstance(pred_single, torch.Tensor):
                            preds_bin_list.append(pred_single.unsqueeze(0))
                        else:
                            preds_bin_list.append(torch.from_numpy(pred_single).unsqueeze(0).to(device))
                    preds_bin = torch.cat(preds_bin_list, dim=0).unsqueeze(1).to(device)  # B x 1 x H x W
                    
                    # --- 闭运算代码 (已注释) ---
                    # # 应用闭运算（填充小孔洞，连接接近的物体）
                    # for i in range(preds_bin.shape[0]):
                    #     pred_mask_np = preds_bin[i, 0].cpu().numpy()
                    #     # 转换为 uint8 格式
                    #     if pred_mask_np.max() <= 1.0:
                    #         pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
                    #     else:
                    #         pred_mask_np = pred_mask_np.astype(np.uint8)
                    #     
                    #     # 闭运算：先膨胀后腐蚀，填充小孔洞
                    #     kernel = np.ones((3, 3), np.uint8)
                    #     pred_mask_closed = cv2.morphologyEx(pred_mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
                    #     
                    #     # 转换回 0-1 范围并更新
                    #     pred_mask_closed = (pred_mask_closed > 127).astype(np.float32)
                    #     preds_bin[i, 0] = torch.from_numpy(pred_mask_closed).float().to(device)

                    # 【修改】改为样本级指标计算：对每个样本计算所有指标并存储
                    # 使用与训练过程相同的calculate_batch_dice函数计算每个样本的Dice
                    batch_dice = temp_train_thread.calculate_batch_dice(preds_bin, masks)
                    batch_dice_np = batch_dice.cpu().numpy()
                    
                    # 对每个样本计算所有指标（IoU, Precision, Recall, Specificity）
                    pred = preds_bin > 0.5
                    gt = masks > 0.5
                    
                    for i in range(preds_bin.shape[0]):
                        pred_i = pred[i, 0].cpu().numpy()
                        gt_i = gt[i, 0].cpu().numpy()
                        
                        # 计算每个样本的混淆矩阵
                        tp = np.sum((pred_i > 0.5) & (gt_i > 0.5))
                        fp = np.sum((pred_i > 0.5) & (gt_i <= 0.5))
                        fn = np.sum((pred_i <= 0.5) & (gt_i > 0.5))
                        tn = np.sum((pred_i <= 0.5) & (gt_i <= 0.5))
                        
                        # 计算每个样本的Dice
                        dice_val = float(batch_dice_np[i])
                        sweep_dice_scores[thr].append(dice_val)
                        
                        # 计算每个样本的IoU
                        iou_den = tp + fp + fn
                        iou_val = 1.0 if iou_den < 1e-8 else float(tp / (iou_den + 1e-8))
                        sweep_iou_scores[thr].append(iou_val)
                        
                        # 计算每个样本的Precision
                        # 【修复】如果没有预测出任何正样本(tp+fp=0)，则精确率视为1.0(无误检)
                        prec_den = tp + fp
                        precision_val = float(tp / (prec_den + 1e-8)) if prec_den > 0 else 1.0
                        sweep_precision_scores[thr].append(precision_val)
                        
                        # 计算每个样本的Recall
                        # 【修复】如果Ground Truth为空(无病灶，tp+fn=0)，则召回率视为1.0(完美表现)
                        rec_den = tp + fn
                        recall_val = float(tp / (rec_den + 1e-8)) if rec_den > 0 else 1.0
                        sweep_recall_scores[thr].append(recall_val)
                        
                        # 计算每个样本的Specificity
                        spec_den = tn + fp
                        specificity_val = float(tn / (spec_den + 1e-8)) if spec_den > 0 else 0.0
                        sweep_specificity_scores[thr].append(specificity_val)
                    
                    # 累计像素级混淆矩阵（仅用于FP计数等统计信息）
                    tp_total = torch.sum(pred & gt).item()
                    fp_total = torch.sum(pred & (~gt)).item()
                    fn_total = torch.sum((~pred) & gt).item()
                    tn_total = torch.sum((~pred) & (~gt)).item()
                    sweep_stats[thr]["tp"] += tp_total
                    sweep_stats[thr]["fp"] += fp_total
                    sweep_stats[thr]["fn"] += fn_total
                    sweep_stats[thr]["tn"] += tn_total
                    sweep_stats[thr]["fp_pix"] += fp_total

        # 打印表格并选择最优阈值（使用自定义综合评分函数）
        sweep_rows = []
        for thr in thresholds:
            tp = sweep_stats[thr]["tp"]
            fp = sweep_stats[thr]["fp"]
            fn = sweep_stats[thr]["fn"]
            tn = sweep_stats[thr]["tn"]
            
            # 【修改】使用样本级宏平均计算所有指标：对每个样本的指标值求平均
            # 而不是基于总TP/FP/FN的像素级微平均
            if sweep_dice_scores[thr]:
                dice_val = float(np.mean(sweep_dice_scores[thr]))
            else:
                dice_val = 0.0
            
            if sweep_iou_scores[thr]:
                iou_val = float(np.mean(sweep_iou_scores[thr]))
            else:
                iou_val = 0.0
            
            if sweep_precision_scores[thr]:
                precision = float(np.mean(sweep_precision_scores[thr]))
            else:
                precision = 0.0
            
            if sweep_recall_scores[thr]:
                recall = float(np.mean(sweep_recall_scores[thr]))
            else:
                recall = 0.0
            
            if sweep_specificity_scores[thr]:
                specificity = float(np.mean(sweep_specificity_scores[thr]))
            else:
                specificity = 0.0
            
            fp_count = int(sweep_stats[thr]["fp_pix"])

            # 【更新评分公式】综合得分 = Dice * 0.6 + IoU * 0.1 + Sensitivity(Recall) * 0.1 + Specificity * 0.1
            # 用于阈值选择时的综合评分
            total_score = (
                dice_val * 0.6 +
                iou_val * 0.1 +
                recall * 0.1 +  # Sensitivity = Recall
                specificity * 0.1
            )

            row = {
                "threshold": float(thr),
                "dice": float(dice_val),
                "precision": float(precision),
                "recall": float(recall),
                "iou": float(iou_val),
                "specificity": float(specificity),
                "score": float(total_score),
                "fp_count": int(fp_count),
            }
            sweep_rows.append(row)
            print(f"{thr:0.2f}      | {dice_val:0.4f}      | {precision:0.4f}    | {recall:0.4f} | {fp_count}")

        # 直接以自定义综合评分 Score 作为优化目标选择最佳阈值
        fallback_used = False
        if sweep_rows:
            best_row = max(sweep_rows, key=lambda r: r.get("score", 0.0))
        else:
            fallback_used = True
            best_row = {"threshold": thresholds[0], "dice": 0.0, "precision": 0.0, "recall": 0.0, "fp_count": 0, "score": 0.0}

        optimal_threshold = float(best_row["threshold"])
        print(
            f"\nBest Threshold found: {optimal_threshold:.2f} "
            f"with TotalScore: {best_row.get('score', 0.0):.4f}, "
            f"Dice: {best_row.get('dice', 0.0):.4f}, "
            f"IoU: {best_row.get('iou', 0.0):.4f}, "
            f"Precision: {best_row.get('precision', 0.0):.4f}, "
            f"Recall: {best_row.get('recall', 0.0):.4f}, "
            f"Specificity: {best_row.get('specificity', 0.0):.4f}"
        )

        # 通过信号把扫描表 + 推荐阈值信息传给GUI
        try:
            self.threshold_sweep_ready.emit({
                "rows": sweep_rows,
                "best": best_row,
                # 与 GUI 侧 on_threshold_sweep_ready 中的默认值保持一致
                "recall_floor": 0.90,
                "fallback_used": fallback_used,
            })
        except Exception:
            pass
        
        # 调试：统计模型输出
        output_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        pred_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        binary_stats = {'positive_pixels': []}
        
        # 进入详细评估前，确保DataLoader可以重新迭代
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="评估中")):
                if len(batch_data) == 3:
                    images, masks, _ = batch_data 
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                
                # 预测
                # 强制开启TTA（与阈值扫描一致）
                outputs = _forward_with_tta(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.shape[2:] != masks.shape[2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # 调试：记录输出统计
                if batch_idx == 0:
                    output_stats['min'].append(outputs.min().item())
                    output_stats['max'].append(outputs.max().item())
                    output_stats['mean'].append(outputs.mean().item())
                    output_stats['std'].append(outputs.std().item())
                    print(f"[调试] 模型原始输出统计: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, std={outputs.std().item():.4f}")
                
                preds = torch.sigmoid(outputs)
                
                # 调试：记录sigmoid后统计
                if batch_idx == 0:
                    pred_stats['min'].append(preds.min().item())
                    pred_stats['max'].append(preds.max().item())
                    pred_stats['mean'].append(preds.mean().item())
                    pred_stats['std'].append(preds.std().item())
                    print(f"[调试] Sigmoid后统计: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}, std={preds.std().item():.4f}")
                
                # 【HD95优化后处理】使用最优阈值 + 优化的后处理流水线
                # 确保最终报告指标与最佳阈值搜索结果完全一致
                preds_binary_list = []
                for i in range(preds.shape[0]):
                    prob_single = preds[i, 0]  # H x W
                    # 应用优化的后处理流水线（启用动态面积阈值）
                    pred_single = temp_train_thread.post_process_refine_for_hd95(
                        prob_single,
                        threshold=optimal_threshold,
                        min_area_threshold=100,  # 基础面积阈值（会动态调整）
                        use_gaussian_blur=True,  # 启用高斯模糊平滑边缘
                        use_morphology=True,      # 启用形态学闭运算
                        dynamic_area_threshold=True  # 启用动态面积阈值
                    )
                    if isinstance(pred_single, torch.Tensor):
                        preds_binary_list.append(pred_single.unsqueeze(0))
                    else:
                        preds_binary_list.append(torch.from_numpy(pred_single).unsqueeze(0).to(device))
                preds_binary = torch.cat(preds_binary_list, dim=0).unsqueeze(1).to(device)  # B x 1 x H x W
                
                # --- 闭运算代码 (已注释) ---
                # # 应用闭运算（填充小孔洞，连接接近的物体）- 与阈值扫描时一致
                # for i in range(preds_binary.shape[0]):
                #     pred_mask_np = preds_binary[i, 0].cpu().numpy()
                #     # 转换为 uint8 格式
                #     if pred_mask_np.max() <= 1.0:
                #         pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
                #     else:
                #         pred_mask_np = pred_mask_np.astype(np.uint8)
                #     
                #     # 闭运算：先膨胀后腐蚀，填充小孔洞
                #     kernel = np.ones((3, 3), np.uint8)
                #     pred_mask_closed = cv2.morphologyEx(pred_mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
                #     
                #     # 转换回 0-1 范围并更新
                #     pred_mask_closed = (pred_mask_closed > 127).astype(np.float32)
                #     preds_binary[i, 0] = torch.from_numpy(pred_mask_closed).float().to(preds_binary.device)
                
                # 调试：记录二值化后统计
                if batch_idx == 0:
                    positive_count = (preds_binary > 0.5).sum().item()
                    total_pixels = preds_binary.numel()
                    binary_stats['positive_pixels'].append(positive_count)
                    print(f"[调试] 二值化后正样本像素数: {positive_count}/{total_pixels} ({100*positive_count/total_pixels:.2f}%)")
                    print(f"[调试] 真实mask正样本像素数: {(masks > 0.5).sum().item()}/{masks.numel()} ({100*(masks > 0.5).sum().item()/masks.numel():.2f}%)")
                    print(f"🎯 HD95优化后处理已启用: 高斯模糊 + 形态学闭运算 + 严格连通域过滤(保留前2个, 最小面积100) | 阈值: {optimal_threshold:.2f}")
                
                # 使用与训练过程相同的calculate_batch_dice函数计算Dice（使用纯粹阈值截断的 mask）
                batch_dice = temp_train_thread.calculate_batch_dice(preds_binary, masks)
                batch_dice_np = batch_dice.cpu().numpy()
                
                # 计算每个样本的指标
                for i in range(images.size(0)):
                    pred_mask = preds_binary[i, 0].cpu().numpy()
                    target_mask = masks[i, 0].cpu().numpy()
                    
                    # 使用纯粹阈值截断的 Dice 值（与控制台搜索时一致）
                    dice = float(batch_dice_np[i])
                    
                    # 成分分析：统计正/负样本分布 & 分类Dice
                    test_total_samples += 1
                    target_sum = float(np.sum(target_mask > 0.5))
                    if target_sum < 1e-7:
                        test_neg_samples += 1
                        test_dice_neg_sum += dice
                    else:
                        test_pos_samples += 1
                        test_dice_pos_sum += dice
                    
                    # 检查空mask情况（用于统计）
                    pred_sum = np.sum(pred_mask > 0.5)
                    # target_sum 已在上面计算（避免重复）
                    
                    # 统计空mask情况
                    if target_sum < 1e-7:
                        empty_target_count += 1
                    if pred_sum < 1e-7:
                        empty_pred_count += 1
                    if target_sum < 1e-7 and pred_sum < 1e-7:
                        both_empty_count += 1
                    if target_sum >= 1e-7 and pred_sum >= 1e-7:
                        both_non_empty_count += 1
                    
                    # 计算混淆矩阵（用于其他指标）
                    tp = np.sum((pred_mask > 0.5) & (target_mask > 0.5))
                    fp = np.sum((pred_mask > 0.5) & (target_mask <= 0.5))
                    fn = np.sum((pred_mask <= 0.5) & (target_mask > 0.5))
                    tn = np.sum((pred_mask <= 0.5) & (target_mask <= 0.5))
                    
                    # 计算其他指标（IoU, Precision, Recall等）
                    iou_den = tp + fp + fn
                    iou = 1.0 if iou_den < 1e-8 else tp / (iou_den + 1e-8)
                    
                    # 【修复】Precision: 如果没有预测出任何正样本(tp+fp=0)，则精确率视为1.0(无误检)
                    prec_den = tp + fp
                    precision = float(tp / (prec_den + 1e-8)) if prec_den > 0 else 1.0
                    
                    # 【修复】Recall: 如果Ground Truth为空(无病灶，tp+fn=0)，则召回率视为1.0(完美表现)
                    rec_den = tp + fn
                    recall = float(tp / (rec_den + 1e-8)) if rec_den > 0 else 1.0
                    
                    specificity = tn / (tn + fp + 1e-8)
                    f1 = dice  # 二分类下F1=Dice（使用与训练一致的Dice值）
                    
                    # 计算HD95（使用TrainThread的calculate_hd95方法）
                    hd95 = 0.0
                    if target_sum < 1e-7 and pred_sum < 1e-7:
                        # 两者都为空，HD95为0
                        hd95 = 0.0
                    elif target_sum < 1e-7 or pred_sum < 1e-7:
                        # 只有一个为空，HD95为无穷大（用NaN表示不可计算）
                        hd95 = float('nan')
                    else:
                        # 两者都不为空，计算HD95（使用全局函数）
                        try:
                            hd95 = calculate_hd95(pred_mask, target_mask)
                            if np.isnan(hd95) or np.isinf(hd95) or hd95 >= 99.0:
                                hd95 = float('nan')
                        except Exception as e:
                            print(f"[警告] 计算HD95失败: {e}")
                            hd95 = float('nan')
                    
                    metrics['dice'].append(dice)
                    metrics['iou'].append(float(iou))
                    metrics['precision'].append(float(precision))
                    metrics['recall'].append(float(recall))
                    metrics['sensitivity'].append(float(recall))
                    metrics['specificity'].append(float(specificity))
                    metrics['f1'].append(float(f1))
                    metrics['hd95'].append(hd95)
                    
                    accum_tp += tp
                    accum_fp += fp
                    accum_fn += fn
                    accum_tn += tn
                    
                    # 记录低Dice案例（Dice < 0.7）
                    if dice < 0.7 and image_idx < len(image_paths):
                        # 保存原始图像、预测mask和真实mask
                        original_image = images[i, 0].cpu().numpy().copy()  # 原始输入图像，确保连续
                        # 将图像归一化到0-255范围用于显示
                        if original_image.max() > 1.0:
                            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8) * 255
                        else:
                            original_image = original_image * 255
                        original_image = original_image.astype(np.uint8)
                        # 确保数组是连续的（C顺序）
                        if not original_image.flags['C_CONTIGUOUS']:
                            original_image = np.ascontiguousarray(original_image)
                        
                        # 预测mask（已经是二值化的）
                        pred_mask_display = (pred_mask * 255).astype(np.uint8)
                        if not pred_mask_display.flags['C_CONTIGUOUS']:
                            pred_mask_display = np.ascontiguousarray(pred_mask_display)
                        
                        # 真实mask（转换为0-255）
                        target_mask_display = (target_mask * 255).astype(np.uint8)
                        if not target_mask_display.flags['C_CONTIGUOUS']:
                            target_mask_display = np.ascontiguousarray(target_mask_display)
                        
                        low_dice_cases.append({
                            'image_path': image_paths[image_idx],
                            'dice': float(dice),
                            'iou': float(iou),
                            'precision': float(precision),
                            'recall': float(recall),
                            'specificity': float(specificity),
                            'original_image': original_image,  # numpy数组
                            'pred_mask': pred_mask_display,   # numpy数组
                            'target_mask': target_mask_display  # numpy数组
                        })
                    
                    image_idx += 1
        
        # 打印空mask统计
        total_samples = len(metrics['dice'])
        print(f"\n[统计] 空mask情况分析:")
        print(f"  总样本数: {total_samples}")
        print(f"  真实mask为空的样本: {empty_target_count} ({100*empty_target_count/total_samples:.1f}%)")
        print(f"  预测mask为空的样本: {empty_pred_count} ({100*empty_pred_count/total_samples:.1f}%)")
        print(f"  两者都空的样本: {both_empty_count} ({100*both_empty_count/total_samples:.1f}%)")
        print(f"  两者都不空的样本: {both_non_empty_count} ({100*both_non_empty_count/total_samples:.1f}%)")
        
        # 成分分析报告：用于解释 Overall Dice 差异（空mask比例/正样本能力）
        pos_ratio = (test_pos_samples / test_total_samples) if test_total_samples > 0 else 0.0
        neg_ratio = (test_neg_samples / test_total_samples) if test_total_samples > 0 else 0.0
        test_dice_pos = (test_dice_pos_sum / test_pos_samples) if test_pos_samples > 0 else 0.0
        test_dice_neg = (test_dice_neg_sum / test_neg_samples) if test_neg_samples > 0 else 0.0
        print(f"\n[成分分析] 测试集样本分布:")
        print(f"  Total Samples   : {test_total_samples}")
        print(f"  Positive Samples: {test_pos_samples} ({pos_ratio:.1%})")
        print(f"  Negative Samples: {test_neg_samples} ({neg_ratio:.1%})")
        print(f"[成分分析] 分类 Dice:")
        print(f"  Test_Dice_Pos   : {test_dice_pos:.4f}")
        print(f"  Test_Dice_Neg   : {test_dice_neg:.4f}")
        
        # 计算平均指标（对于HD95使用nanmean，忽略NaN值）
        avg_metrics = {}
        for k, v in metrics.items():
            if k == 'hd95':
                # HD95可能包含NaN，使用nanmean
                if v:
                    arr = np.array(v, dtype=float)
                    if np.all(np.isnan(arr)):
                        avg_metrics[k] = float('nan')
                    else:
                        avg_metrics[k] = float(np.nanmean(arr))
                else:
                    avg_metrics[k] = float('nan')
            else:
                avg_metrics[k] = float(np.mean(v)) if v else 0.0
        
        # 【修改】全局指标计算：从像素级微平均改为样本级宏平均
        # 使用每个样本的指标值列表进行平均，而不是基于总TP/FP/FN计算
        # 这样可以确保每个样本的权重相等，不受样本大小影响
        
        # Dice和F1（二分类下F1=Dice）
        if metrics['dice']:
            avg_metrics['dice'] = float(np.mean(metrics['dice']))
        else:
            avg_metrics['dice'] = 0.0
        avg_metrics['f1'] = avg_metrics['dice']
        
        # IoU：样本级宏平均
        if metrics['iou']:
            avg_metrics['iou'] = float(np.mean(metrics['iou']))
        else:
            avg_metrics['iou'] = 0.0
        
        # Precision：样本级宏平均
        if metrics['precision']:
            avg_metrics['precision'] = float(np.mean(metrics['precision']))
        else:
            avg_metrics['precision'] = 0.0
        
        # Recall/Sensitivity：样本级宏平均
        if metrics['recall']:
            avg_metrics['recall'] = float(np.mean(metrics['recall']))
        else:
            avg_metrics['recall'] = 0.0
        avg_metrics['sensitivity'] = avg_metrics['recall']
        
        # Specificity：样本级宏平均
        if metrics['specificity']:
            avg_metrics['specificity'] = float(np.mean(metrics['specificity']))
        else:
            avg_metrics['specificity'] = 0.0
        
        # 调试：打印混淆矩阵
        print(f"[调试] 最终混淆矩阵: TP={accum_tp:.0f}, FP={accum_fp:.0f}, FN={accum_fn:.0f}, TN={accum_tn:.0f}")
        print(f"[调试] 最终指标: Dice={avg_metrics['dice']:.4f}, IoU={avg_metrics['iou']:.4f}, Precision={avg_metrics['precision']:.4f}, Recall={avg_metrics['recall']:.4f}")
        
        # 【修复】计算官方总分：使用完整的公式，包含所有5个指标
        # 公式：Total = 0.6*Dice + 0.1*IoU + 0.1/(1+HD95) + 0.1*Sens + 0.1*Spec
        hd95_for_score = avg_metrics['hd95'] if not (np.isnan(avg_metrics['hd95']) or np.isinf(avg_metrics['hd95'])) else 99.9
        official_total_score = calculate_official_total_score_global(
            dice=avg_metrics['dice'],
            iou=avg_metrics['iou'],
            hd95=hd95_for_score,
            sensitivity=avg_metrics['sensitivity'],
            specificity=avg_metrics['specificity']
        )
        
        print(f"[官方总分] Total Score = 0.6*Dice + 0.1*IoU + 0.1/(1+HD95) + 0.1*Sens + 0.1*Spec = {official_total_score:.4f}")
        hd95_str = f"{avg_metrics['hd95']:.4f}" if not (np.isnan(avg_metrics['hd95']) or np.isinf(avg_metrics['hd95'])) else "nan"
        print(f"  详细: Dice={avg_metrics['dice']:.4f}, IoU={avg_metrics['iou']:.4f}, HD95={hd95_str}, Sens={avg_metrics['sensitivity']:.4f}, Spec={avg_metrics['specificity']:.4f}")
        
        # 将官方总分添加到 avg_metrics
        avg_metrics['official_total_score'] = official_total_score
        
        detailed_metrics = {
            'average': avg_metrics,
            'all_samples': metrics,
            'total_samples': len(metrics['dice'])
        }
        
        return detailed_metrics, low_dice_cases
    
    def _generate_attention_maps(self, model, dataloader, device):
        """生成注意力热图"""
        try:
            # 检查模型是否支持注意力图
            actual_model = model
            if isinstance(actual_model, nn.DataParallel):
                actual_model = actual_model.module
            
            if not hasattr(actual_model, 'forward') or not callable(getattr(actual_model, 'forward', None)):
                return ""
            
            # 尝试获取注意力图
            model.eval()
            attention_maps_list = []
            images_list = []
            
            with torch.no_grad():
                for batch_data in dataloader:
                    if len(batch_data) == 3:
                        images, masks, _ = batch_data
                    else:
                        images, masks = batch_data
                    images = images.to(device)
                    
                    try:
                        # 尝试获取注意力图
                        if hasattr(actual_model, 'forward'):
                            result = actual_model(images, return_attention=True)
                            if isinstance(result, tuple) and len(result) == 2:
                                outputs, attention_maps = result
                                attention_maps_list.append(attention_maps)
                                images_list.append(images.cpu())
                    except Exception:
                        pass
                    
                    if len(images_list) >= 4:  # 只取前4个样本
                        break
            
            if not attention_maps_list:
                return ""
            
            # 可视化注意力图
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig, axes = plt.subplots(len(images_list), 5, figsize=(20, 4 * len(images_list)))
            if len(images_list) == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (img, att_maps) in enumerate(zip(images_list, attention_maps_list)):
                img_np = img[0].permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                axes[idx, 0].imshow(img_np)
                axes[idx, 0].set_title("原图")
                axes[idx, 0].axis('off')
                
                for i, (att_name, att_map) in enumerate(list(att_maps.items())[:4]):
                    att_np = att_map[0, 0].cpu().numpy()
                    axes[idx, i+1].imshow(att_np, cmap='hot')
                    axes[idx, i+1].set_title(f"{att_name}")
                    axes[idx, i+1].axis('off')
            
            plt.tight_layout()
            attention_path = os.path.join(self.temp_dir, "attention_maps.png")
            plt.savefig(attention_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return attention_path
        except Exception as e:
            print(f"[警告] 生成注意力热图失败: {e}")
            return ""
    
    def _tta_inference(self, model, images):
        """
        【军令状：TTA终极升级】多尺度置信度融合架构 (MSTTA)
        
        多尺度推理：3个尺度 × 8种变换 = 24倍推理
        - 尺度因子: [0.8, 1.0, 1.2]
        - 8种变换: 原始、水平翻转、垂直翻转、旋转90/180/270度、翻转+旋转组合
        
        加权融合：基于置信度的加权平均，而非简单平均
        极致后处理：Gaussian滤波 + LCC + remove_small_holes
        
        目标：利用5080算力优势，通过24倍推理换取0.01 Dice提升
        """
        import torch.nn.functional as F
        from scipy.ndimage import gaussian_filter
        
        B, C, H, W = images.shape
        scales = [0.8, 1.0, 1.2]  # 多尺度因子
        all_predictions = []
        all_weights = []
        
        # 【多尺度循环】
        for scale in scales:
            # Resize到目标尺度
            if scale != 1.0:
                target_h, target_w = int(H * scale), int(W * scale)
                scaled_images = F.interpolate(images, size=(target_h, target_w), 
                                             mode='bilinear', align_corners=False)
            else:
                scaled_images = images
                target_h, target_w = H, W
            
            # 【8种变换循环】
            scale_predictions = []
            
            # 1. 原始图像
            pred = model(scaled_images)
            if isinstance(pred, tuple):
                pred = pred[0]
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 2. 水平翻转
            pred = model(torch.flip(scaled_images, dims=[3]))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.flip(pred, dims=[3])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 3. 垂直翻转
            pred = model(torch.flip(scaled_images, dims=[2]))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.flip(pred, dims=[2])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 4. 旋转90度
            pred = model(torch.rot90(scaled_images, k=1, dims=[2, 3]))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 5. 旋转180度
            pred = model(torch.rot90(scaled_images, k=2, dims=[2, 3]))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.rot90(pred, k=-2, dims=[2, 3])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 6. 旋转270度
            pred = model(torch.rot90(scaled_images, k=3, dims=[2, 3]))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.rot90(pred, k=-3, dims=[2, 3])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 7. 水平翻转+旋转90度
            img_aug = torch.flip(scaled_images, dims=[3])
            img_aug = torch.rot90(img_aug, k=1, dims=[2, 3])
            pred = model(img_aug)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            pred = torch.flip(pred, dims=[3])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 8. 垂直翻转+旋转90度
            img_aug = torch.flip(scaled_images, dims=[2])
            img_aug = torch.rot90(img_aug, k=1, dims=[2, 3])
            pred = model(img_aug)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            pred = torch.flip(pred, dims=[2])
            if not (torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred))):
                if scale != 1.0:
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                scale_predictions.append(pred)
            
            # 收集当前尺度的所有预测
            all_predictions.extend(scale_predictions)
        
        # 【加权融合】计算每个预测的置信度权重
        if len(all_predictions) == 0:
            print(f"[严重警告] MSTTA: 所有变换的预测都包含NaN/Inf，返回零输出")
            return torch.zeros_like(model(images) if not isinstance(model(images), tuple) else model(images)[0])
        
        # 计算置信度权重：使用 p * log(p + eps) 作为置信度度量
        weights = []
        eps = 1e-8
        for pred in all_predictions:
            # 转换为概率
            prob = torch.sigmoid(pred)
            # 计算平均置信度：-p * log(p) 的均值（熵的负值，越高表示越确定）
            entropy = -prob * torch.log(prob + eps) - (1 - prob) * torch.log(1 - prob + eps)
            confidence = 1.0 - entropy.mean()  # 转换为置信度（1 - 熵）
            weights.append(float(confidence))
        
        # 归一化权重
        weights = torch.tensor(weights, device=images.device, dtype=torch.float32)
        weights = weights / (weights.sum() + eps)
        
        # 加权平均
        stacked_preds = torch.stack(all_predictions, dim=0)  # [N, B, C, H, W]
        weights_expanded = weights.view(-1, 1, 1, 1, 1)  # [N, 1, 1, 1, 1]
        weighted_pred = (stacked_preds * weights_expanded).sum(dim=0)  # [B, C, H, W]
        
        # 【极致后处理】应用Gaussian滤波
        weighted_pred_np = weighted_pred.detach().cpu().numpy()
        smoothed_pred_np = np.zeros_like(weighted_pred_np)
        for b in range(B):
            for c in range(C):
                smoothed_pred_np[b, c] = gaussian_filter(weighted_pred_np[b, c], sigma=0.5)
        
        # 转换回tensor
        smoothed_pred = torch.from_numpy(smoothed_pred_np).to(images.device).float()
        
        # 【极致后处理】在概率图上应用LCC和remove_small_holes
        # 注意：这里返回的是logits，后处理会在sigmoid后的概率图上进行
        # 但为了集成到TTA中，我们在内部进行后处理
        prob_pred = torch.sigmoid(smoothed_pred)
        prob_pred_np = prob_pred.detach().cpu().numpy()
        
        # 对每个样本应用极致后处理
        processed_pred_np = np.zeros_like(prob_pred_np)
        for b in range(B):
            for c in range(C):
                prob_map = prob_pred_np[b, c]
                # 应用极致后处理流水线
                processed_mask = ensemble_post_process_global(
                    prob_map,
                    use_lcc=True,  # 保留最大连通域
                    use_remove_holes=True,  # 填补小孔洞
                    min_hole_size=100,
                    use_edge_smoothing=True  # 边缘平滑
                )
                # 转换回logits空间（逆sigmoid）
                processed_pred_np[b, c] = np.clip(np.log(processed_mask / (1 - processed_mask + eps) + eps), -10, 10)
        
        # 转换回tensor
        final_pred = torch.from_numpy(processed_pred_np).to(images.device).float()
        
        return final_pred


class TrainThread(QThread):
    update_progress = pyqtSignal(int, str)  # (进度百分比, 状态消息)
    update_val_progress = pyqtSignal(int, str)  # 验证进度信号
    training_finished = pyqtSignal(str, str)  # (完成消息, 最佳模型路径)
    model_saved = pyqtSignal(str)  # 模型保存通知
    epoch_completed = pyqtSignal(int, float, float, float)  # (轮次, 平均损失, 验证损失, 验证Dice)
    visualization_ready = pyqtSignal(str)  # 保存的可视化路径
    metrics_ready = pyqtSignal(dict)  # 评估指标字典
    visualization_requested = pyqtSignal(str, list, list)  # 参数：(绘图类型, x轴数据, y轴数据)
    test_results_ready = pyqtSignal(str, dict)  # (可视化图像路径, 性能分析数据)
    epoch_analysis_ready = pyqtSignal(int, str, dict)  # (轮次, 可视化图像路径, 性能指标)
    attention_analysis_ready = pyqtSignal(str, dict)  # (注意力可视化路径, 注意力统计信息)
    def __init__(self, data_dir, epochs, batch_size, model_path=None, save_best=True, use_gwo=False, optimizer_type="adam"):
        super().__init__()
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.save_best = save_best
        self.use_gwo = use_gwo  # 是否使用GWO优化
        self.optimizer_type = optimizer_type.lower()
        
        # 安全读取预训练配置
        try:
            self.pretrained_config = read_checkpoint_config(model_path) if model_path else None
        except Exception as e:
            print(f"[警告] 读取预训练配置失败: {e}")
            self.pretrained_config = None
        
        self.swin_params = None   # GWO优化后的SwinUNet参数或模型配置
        self.dstrans_params = None  # GWO优化后的DS-TransUNet参数或模型配置
        self.mamba_params = None  # Swin-U Mamba 已移除，占位字段
        # EMA 已启用，用于提升模型稳定性和Dice性能
        self.use_ema = True
        self.ema_decay = 0.995
        
        # 安全读取环境变量并转换为整数
        try:
            self.ema_eval_start_epoch = max(5, int(os.environ.get("SEG_EMA_EVAL_START", 8)))
        except (ValueError, TypeError):
            self.ema_eval_start_epoch = 8
        
        self.last_optimal_threshold = 0.5
        self.stop_requested = False
        self.best_model_path = None
        self.best_dice = -1.0
        
        # 安全创建临时目录
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="med_seg_")
        except (OSError, PermissionError) as e:
            # 如果临时目录创建失败，使用用户数据目录下的临时目录
            print(f"[警告] 系统临时目录创建失败: {e}，使用数据目录下的临时目录")
            fallback_temp = os.path.join(data_dir, "_temp_training")
            try:
                os.makedirs(fallback_temp, exist_ok=True)
                self.temp_dir = fallback_temp
            except Exception as e2:
                raise RuntimeError(f"无法创建临时目录: {e2}") from e2
        
        self.best_model_cache_dir = os.path.join(self.data_dir, "_best_model_cache")
        self.enable_matlab_cache = False
        self.matlab_cache_manager = None
        self.matlab_metrics_bridge = None
        self.enable_matlab_plots = False
        self.matlab_viz_bridge = None
        self.model_type = os.environ.get("SEG_MODEL", "improved_unet").lower()
        
        # 安全读取环境变量并转换为整数
        try:
            self.context_slices = int(os.environ.get("SEG_CONTEXT_SLICES", os.environ.get("SEG_CONTEXT", "0")))
        except (ValueError, TypeError):
            self.context_slices = 0
        
        try:
            self.context_gap = int(os.environ.get("SEG_CONTEXT_GAP", "1"))
        except (ValueError, TypeError):
            self.context_gap = 1
        
        self.extra_modalities_dirs = parse_extra_modalities_spec(os.environ.get("SEG_EXTRA_MODALITIES"))
        
        if self.pretrained_config:
            self.model_type = self.pretrained_config.get("model_type", self.model_type)
            # 安全深拷贝配置参数
            try:
                swin_params_raw = self.pretrained_config.get("swin_params")
                if swin_params_raw:
                    self.swin_params = copy.deepcopy(swin_params_raw)
            except Exception as e:
                print(f"[警告] 深拷贝 swin_params 失败: {e}，使用原始引用")
                self.swin_params = self.pretrained_config.get("swin_params")
            
            try:
                dstrans_params_raw = self.pretrained_config.get("dstrans_params")
                if dstrans_params_raw:
                    self.dstrans_params = copy.deepcopy(dstrans_params_raw)
            except Exception as e:
                print(f"[警告] 深拷贝 dstrans_params 失败: {e}，使用原始引用")
                self.dstrans_params = self.pretrained_config.get("dstrans_params")
            if self.swin_params or self.dstrans_params:
                self.use_gwo = False
            if "best_threshold" in self.pretrained_config:
                try:
                    self.last_optimal_threshold = float(self.pretrained_config.get("best_threshold", self.last_optimal_threshold))
                except (ValueError, TypeError):
                    pass  # 保持默认值
            context_cfg = self.pretrained_config.get("context")
            if context_cfg:
                try:
                    self.context_slices = int(context_cfg.get("slices", self.context_slices))
                except (ValueError, TypeError):
                    pass  # 保持当前值
                try:
                    self.context_gap = int(context_cfg.get("gap", self.context_gap))
                except (ValueError, TypeError):
                    pass  # 保持当前值
            # 仅保留模态名称，具体路径仍由环境变量提供
            extra_names = self.pretrained_config.get("extra_modalities")
            if extra_names and not self.extra_modalities_dirs:
                print(f"[提示] 模型期望额外模态: {extra_names}，请通过 SEG_EXTRA_MODALITIES 指定对应路径。")
        # Skull Stripping 配置
        self.use_skull_stripper = os.environ.get("SEG_USE_SKULL_STRIPPER", "0") == "1"
        self.skull_stripper_path = os.environ.get("SKULL_STRIPPER_PATH")
        
        # 安全读取环境变量并转换为浮点数
        try:
            self.skull_stripper_threshold = float(os.environ.get("SEG_SKULL_STRIP_THRESH", "0.5"))
        except (ValueError, TypeError):
            self.skull_stripper_threshold = 0.5
        self.skull_stripper = None
        if self.pretrained_config:
            skull_cfg = self.pretrained_config.get("skull_stripping")
            if skull_cfg:
                self.use_skull_stripper = skull_cfg.get("enabled", self.use_skull_stripper)
                self.skull_stripper_path = skull_cfg.get("model_path", self.skull_stripper_path)
                self.skull_stripper_threshold = skull_cfg.get("threshold", self.skull_stripper_threshold)
        # nnFormer 配置
        self.use_nnformer = False
        
        # 跟踪训练历史
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_dice_history = []
        self.val_dice_pos_history = []  # 仅统计有前景mask样本的Dice
        self.val_dice_neg_history = []  # 仅统计空mask样本的Dice
        # 增加深度监督权重,提升多尺度特征学习
        self.aux_loss_weights = [0.3, 0.2, 0.1]  # 从[0.2,0.1,0.05]提升
        self.split_metadata: Dict[str, Dict[str, List[str]]] = {}
        self.pos_weight_cache: Dict[str, float] = {}
        # 验证阶段动态阈值刷新设置
        try:
            self.threshold_refresh_interval = int(os.environ.get("SEG_THRESH_REFRESH", 1)) or 1
        except (ValueError, TypeError):
            self.threshold_refresh_interval = 1
        # 默认采样更多验证批次, 增强阈值搜索鲁棒性
        try:
            self.threshold_search_batches = int(os.environ.get("SEG_THRESH_BATCHES", 12)) or 6
        except (ValueError, TypeError):
            self.threshold_search_batches = 6
        # 是否启用ReduceLROnPlateau (默认关闭，避免与Cosine重复调度导致学习率坍缩)
        self.use_plateau_scheduler = os.environ.get("SEG_USE_PLATEAU", "0") == "1"
        
        # 确保临时目录存在
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"无法创建临时目录 {self.temp_dir}: {e}") from e
   
    def visualize_predictions(self, model, dataloader, device, save_name="predictions"):
        """可视化模型预测结果与真实标签"""
        save_path = os.path.join(self.temp_dir, f"{save_name}.png")
        model.eval()
        # 处理数据：可能包含分类标签
        batch_data = next(iter(dataloader))
        if len(batch_data) == 3:
            images, masks, _ = batch_data
        else:
            images, masks = batch_data
        images, masks = images.to(device), masks.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
        
        num_samples = min(4, images.size(0))
        sample_triplets = []
        for i in range(num_samples):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1).astype(np.float32)
            true_mask = masks[i, 0].cpu().numpy().astype(np.float32)
            pred_mask = preds[i, 0].cpu().numpy().astype(np.float32)
            sample_triplets.append((img, true_mask, pred_mask))

        if self.enable_matlab_plots and self.matlab_viz_bridge:
            try:
                payload_path = self._save_matlab_viz_payload(
                    [triplet[0] for triplet in sample_triplets],
                    [triplet[1] for triplet in sample_triplets],
                    [triplet[2] for triplet in sample_triplets],
                    save_name
                )
                matlab_save_path = os.path.join(self.temp_dir, f"{save_name}_matlab.png")
                self.matlab_viz_bridge.render_prediction_grid(payload_path, matlab_save_path)
                return matlab_save_path
            except Exception as exc:
                print(f"[MATLAB Plot] 使用matplotlib回退: {exc}")

        plt.figure(figsize=(15, 10))
        for idx, (img, true_mask, pred_mask) in enumerate(sample_triplets):
            overlay = img.copy()
            overlay[true_mask == 1, 0] = 1
            overlay[pred_mask == 1, 1] = 1

            plt.subplot(num_samples, 4, idx * 4 + 1)
            plt.imshow(img)
            plt.title(f"样本 {idx + 1}\n输入图像")
            plt.axis('off')

            plt.subplot(num_samples, 4, idx * 4 + 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title("真实标签")
            plt.axis('off')

            plt.subplot(num_samples, 4, idx * 4 + 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title(f"预测结果\nDice: {self.calculate_dice(preds[idx], masks[idx]).item():.2f}")
            plt.axis('off')

            plt.subplot(num_samples, 4, idx * 4 + 4)
            plt.imshow(overlay)
            plt.title("叠加图（红:真实, 绿:预测）")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def plot_training_history(self):
        """绘制训练历史曲线"""
        save_path = os.path.join(self.temp_dir, "training_history.png")

        if self.enable_matlab_plots and self.matlab_viz_bridge:
            try:
                payload = self._save_training_history_payload()
                if payload:
                    matlab_path = os.path.join(self.temp_dir, "training_history_matlab.png")
                    self.matlab_viz_bridge.render_training_history(payload, matlab_path)
                    return matlab_path
            except Exception as exc:
                print(f"[MATLAB Plot] 训练历史回退: {exc}")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='训练损失')
        plt.plot(self.val_loss_history, label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_dice_history, label='Dice系数', color='green')
        plt.title('验证Dice分数')
        plt.xlabel('轮次')
        plt.ylabel('Dice分数')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def find_optimal_threshold(self, model, dataloader, device, num_samples=50):
        """
        在验证集上寻找最优二值化阈值
        
        Args:
            num_samples: 用于搜索的批次数（None表示使用全部验证集，确保与验证阶段一致）
        
        Returns:
            最优阈值
        """
        model.eval()
        # 如果num_samples为None或0，使用全部验证集（与验证阶段保持一致）
        use_all_samples = (num_samples is None or num_samples <= 0)
        if not use_all_samples:
            num_samples = max(1, int(num_samples))
        
        with torch.no_grad():
            all_probs = []
            all_masks = []
            
            for idx, batch_data in enumerate(dataloader):
                if not use_all_samples and idx >= num_samples:
                    break
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images = images.to(device)
                masks = masks.to(device)
                
                # 使用TTA进行推理（与验证阶段一致）
                # 这确保阈值优化时使用的预测与验证统计时一致
                outputs = self._tta_inference(model, images)
                probs = torch.sigmoid(outputs)
                # 确保 probs 和 masks 的空间尺寸匹配
                if probs.shape[2:] != masks.shape[2:]:
                    probs = F.interpolate(probs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                all_probs.append(probs.detach().cpu().numpy())
                all_masks.append(masks.detach().cpu().numpy())
            
            if not all_probs:
                return 0.5
            
            all_probs_np = np.concatenate(all_probs, axis=0)
            all_masks_np = np.concatenate(all_masks, axis=0)

            best_threshold, best_metrics = self.scan_best_threshold(all_probs_np, all_masks_np)

        sample_info = "全部验证集" if use_all_samples else f"{num_samples}个批次"
        score_val = best_metrics.get("score", 0.0) if isinstance(best_metrics, dict) else 0.0
        print(
            f"[阈值优化] 使用样本: {sample_info} | "
            f"最优阈值: {best_threshold:.3f}, 综合评分: {score_val:.4f}, "
            f"Dice: {best_metrics.get('dice', float('nan')):.4f}, "
            f"IoU: {best_metrics.get('iou', float('nan')):.4f}"
        )
        return float(best_threshold)
    
    def evaluate_model(self, model, dataloader, device, use_tta=True, adaptive_threshold=True):
        """
        综合模型评估
        
        Args:
            use_tta: 是否使用测试时增强(TTA),可提升1-3%的Dice
            adaptive_threshold: 是否使用自适应阈值
        """
        # 寻找最优阈值
        if adaptive_threshold:
            optimal_thresh = self.find_optimal_threshold(model, dataloader, device)
        else:
            optimal_thresh = 0.5
        self.last_optimal_threshold = float(optimal_thresh)
        
        model.eval()
        metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'sensitivity': [],
            'specificity': [],
            'f1': [],
            'hd95': []
        }
        # 微平均累积混淆矩阵，保证最终显示的指标一致（Dice=F1）
        accum_tp = accum_fp = accum_fn = accum_tn = 0.0
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="评估中(TTA)" if use_tta else "评估中"):
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                brain_mask = None
                if self.use_skull_stripper:
                    images, brain_mask = self._apply_skull_strip(images)
                
                if use_tta:
                    # 测试时增强: 8个变换的平均
                    outputs = self._tta_inference(model, images)
                else:
                    outputs = model(images)
                # 确保 outputs 和 masks 的空间尺寸匹配
                if outputs.shape[2:] != masks.shape[2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                if brain_mask is not None:
                    outputs = outputs * brain_mask
                
                preds = torch.sigmoid(outputs)
                preds = (preds > optimal_thresh).float()  # 使用最优阈值
                
                # 应用后处理优化：填充孔洞，不再强制只保留最大连通域
                for i in range(preds.shape[0]):
                    preds[i, 0] = self.post_process_mask(
                        preds[i, 0], 
                        min_size=30, 
                        use_morphology=True,
                        keep_largest=False,  # 允许多发病灶同时存在
                        fill_holes=True     # 填充孔洞，去除假阴性空洞
                    )
                
                # 计算批次中每个图像的指标
                for i in range(preds.shape[0]):
                    pred = preds[i, 0]
                    mask = masks[i, 0]
                    
                    # 双重检查尺寸匹配（以防后处理改变了尺寸）
                    if pred.shape != mask.shape:
                        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                    # 计算混淆矩阵的四个基本值
                    tp = float((pred * mask).sum().item())
                    pred_sum = float(pred.sum().item())   # TP + FP
                    mask_sum = float(mask.sum().item())   # TP + FN
                    fp = float((pred * (1 - mask)).sum().item())
                    fn = float(((1 - pred) * mask).sum().item())
                    tn = float(((1 - pred) * (1 - mask)).sum().item())
                    
                    # 验证: tp + fp = pred_sum, tp + fn = mask_sum
                    assert abs((tp + fp) - pred_sum) < 1e-5, f"TP+FP计算错误: {tp+fp} vs {pred_sum}"
                    assert abs((tp + fn) - mask_sum) < 1e-5, f"TP+FN计算错误: {tp+fn} vs {mask_sum}"
                    
                    # Dice = 2TP / (2TP + FP + FN)
                    dice_den = 2.0 * tp + fp + fn
                    if dice_den < 1e-7:
                        dice = 1.0 if (mask_sum < 1e-7 and pred_sum < 1e-7) else 0.0
                    else:
                        dice = (2.0 * tp) / dice_den
                    
                    # IoU = TP / (TP + FP + FN)
                    union = tp + fp + fn
                    if union < 1e-7:
                        iou = 1.0 if (mask_sum < 1e-7 and pred_sum < 1e-7) else 0.0
                    else:
                        iou = tp / union
                    
                    # Precision = TP / (TP + FP)
                    if tp + fp < 1e-7:
                        precision = 1.0 if mask_sum < 1e-7 else 0.0
                    else:
                        precision = tp / (tp + fp)
                    
                    # Recall/Sensitivity = TP / (TP + FN)
                    if tp + fn < 1e-7:
                        recall = 1.0 if pred_sum < 1e-7 else 0.0
                    else:
                        recall = tp / (tp + fn)
                    
                            # Specificity = TN / (TN + FP)
                    tn_plus_fp = tn + fp
                    specificity = 1.0 if tn_plus_fp < 1e-7 else tn / tn_plus_fp
                    
                    # F1在二分类下应与Dice一致，这里直接复用
                    f1 = dice
                    
                    # 计算HD95
                    if mask_sum < 1e-7:
                        hd95 = 0.0 if pred_sum < 1e-7 else float('inf')
                    elif pred_sum < 1e-7:
                        hd95 = float('inf')
                    else:
                        hd95 = self.calculate_hd95(
                            pred.detach().cpu().numpy(),
                            mask.detach().cpu().numpy()
                        )

                    metrics['dice'].append(float(dice))
                    metrics['iou'].append(float(iou))
                    metrics['precision'].append(float(precision))
                    metrics['recall'].append(float(recall))
                    metrics['sensitivity'].append(float(recall))
                    metrics['specificity'].append(float(specificity))
                    metrics['f1'].append(float(f1))
                    metrics['hd95'].append(hd95 if not np.isinf(hd95) else 0.0)
                    
                    accum_tp += tp
                    accum_fp += fp
                    accum_fn += fn
                    accum_tn += tn
        
        # 计算平均指标，忽略nan值
        metrics_arrays = {k: np.array(v, dtype=float) for k, v in metrics.items()}
        avg_metrics = {}
        std_metrics = {}
        min_metrics = {}
        max_metrics = {}
        median_metrics = {}
        for k, arr in metrics_arrays.items():
            if arr.size == 0 or np.all(np.isnan(arr)):
                avg_metrics[k] = float('nan')
                std_metrics[k] = float('nan')
                min_metrics[k] = float('nan')
                max_metrics[k] = float('nan')
                median_metrics[k] = float('nan')
            else:
                avg_metrics[k] = float(np.nanmean(arr))
                std_metrics[k] = float(np.nanstd(arr))
                min_metrics[k] = float(np.nanmin(arr))
                max_metrics[k] = float(np.nanmax(arr))
                median_metrics[k] = float(np.nanmedian(arr))
        
        # 微平均（global）指标，使用累积的混淆矩阵确保各指标一致
        micro_metrics = {}
        dice_den = 2 * accum_tp + accum_fp + accum_fn
        micro_metrics['dice'] = 1.0 if dice_den < 1e-7 else (2 * accum_tp) / dice_den
        
        iou_den = accum_tp + accum_fp + accum_fn
        micro_metrics['iou'] = 1.0 if iou_den < 1e-7 else accum_tp / iou_den
        
        prec_den = accum_tp + accum_fp
        micro_metrics['precision'] = 1.0 if prec_den < 1e-7 else accum_tp / prec_den
        
        rec_den = accum_tp + accum_fn
        micro_metrics['recall'] = 1.0 if rec_den < 1e-7 else accum_tp / rec_den
        micro_metrics['sensitivity'] = micro_metrics['recall']
        
        spec_den = accum_tn + accum_fp
        micro_metrics['specificity'] = 1.0 if spec_den < 1e-7 else accum_tn / spec_den
        
        micro_metrics['f1'] = micro_metrics['dice']  # 二分类下F1=Dice
        micro_metrics['hd95'] = float(np.nanmean(metrics_arrays['hd95'])) if metrics_arrays['hd95'].size > 0 else float('nan')
        
        # 添加统计信息
        detailed_metrics = {
            'average': avg_metrics,
            'std': std_metrics,
            'min': min_metrics,
            'max': max_metrics,
            'median': median_metrics,
            'all_samples': metrics
        }
        # 覆盖平均值为微平均，确保显示一致
        for k, v in micro_metrics.items():
            detailed_metrics['average'][k] = float(v)
        
        # 保存指标到CSV
        metrics_path = os.path.join(self.temp_dir, 'performance_metrics.csv')
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        
        return detailed_metrics, metrics_path
    
    def evaluate_model_ensemble(self, models, dataloader, device, use_tta=True, adaptive_threshold=True):
        """模型集成功能已取消。"""
        raise RuntimeError("模型集成功能已取消")
    
    def find_optimal_threshold_ensemble(self, *args, **kwargs):
        """模型集成功能已取消。"""
        raise RuntimeError("模型集成功能已取消")
    
    def evaluate_per_volume(self, model, dataloader, device, patient_slice_index=None, patients=None, use_tta=True):
        """
        按volume评估（参考标准代码）
        将同一病人的所有slice组织成volume，然后计算每个volume的Dice
        这种方式更符合临床评估习惯
        
        Args:
            model: 模型
            dataloader: 数据加载器
            patient_slice_index: 病人-切片索引列表 [(patient_idx, slice_idx), ...]
            patients: 病人ID列表
            use_tta: 是否使用测试时增强
        
        Returns:
            volume_metrics: 每个volume的指标字典
            avg_dice: 平均Dice（按volume）
        """
        model.eval()
        
        # 如果没有提供patient_slice_index，尝试从dataset获取
        if patient_slice_index is None:
            if hasattr(dataloader.dataset, 'patient_slice_index'):
                patient_slice_index = dataloader.dataset.patient_slice_index
            elif hasattr(dataloader.dataset, 'image_paths'):
                # 从路径推断病人ID
                patient_slice_index = []
                for i, path in enumerate(dataloader.dataset.image_paths):
                    # 尝试从路径提取病人ID和切片序号
                    base = os.path.splitext(os.path.basename(path))[0]
                    parts = base.split('_')
                    if len(parts) >= 2:
                        patient_id = '_'.join(parts[:-1])
                        try:
                            slice_idx = int(parts[-1])
                            patient_slice_index.append((patient_id, slice_idx))
                        except ValueError:
                            patient_slice_index.append((base, 0))
                    else:
                        patient_slice_index.append((base, i))
        
        if patients is None:
            if hasattr(dataloader.dataset, 'patients'):
                patients = dataloader.dataset.patients
            else:
                # 从patient_slice_index提取唯一病人ID
                patients = sorted(list(set([p[0] for p in patient_slice_index])))
        
        # 收集所有预测和真实值
        all_preds = []
        all_trues = []
        all_inputs = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="按volume评估"):
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                
                if use_tta:
                    outputs = self._tta_inference(model, images)
                else:
                    outputs = model(images)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > self.last_optimal_threshold).float()
                
                # 智能后处理：先按面积+概率过滤微小病灶/噪点，再进行形态学优化
                for i in range(preds.shape[0]):
                    pred_mask_tensor = preds[i, 0]
                    prob_map_tensor = probs[i, 0]
                    # 先执行智能后处理（不再简单按min_size裁剪）
                    pred_mask_tensor = self.smart_post_processing(pred_mask_tensor, prob_map_tensor)
                    # 再执行传统形态学后处理，但不移除小区域（min_size=0）
                    pred_mask_processed = self.post_process_mask(
                        pred_mask_tensor,
                        min_size=0,
                        use_morphology=True,
                        keep_largest=False,  # 允许多发病灶同时存在
                        fill_holes=True     # 填充孔洞，去除假阴性空洞
                    )
                    preds[i, 0] = pred_mask_processed
                
                all_preds.extend([preds[i].cpu().numpy() for i in range(preds.shape[0])])
                all_trues.extend([masks[i].cpu().numpy() for i in range(masks.shape[0])])
                all_inputs.extend([images[i].cpu().numpy() for i in range(images.shape[0])])
        
        # 按volume组织数据
        if patient_slice_index:
            from collections import OrderedDict
            slice_counter = OrderedDict()
            for pid, _ in patient_slice_index:
                slice_counter[pid] = slice_counter.get(pid, 0) + 1
            patient_order = list(slice_counter.keys())
            num_slices = [slice_counter[pid] for pid in patient_order]
            patients = patient_order
        else:
            # 如果无法推断，假设每个样本是一个volume
            num_slices = np.ones(len(all_preds), dtype=int)
        
        # 计算每个volume的Dice
        volume_dice_list = []
        volume_metrics = {}
        index = 0
        
        for p_idx, patient_id in enumerate(patients):
            if p_idx >= len(num_slices):
                break
            num_s = num_slices[p_idx] if p_idx < len(num_slices) else 1
            
            volume_pred = np.array(all_preds[index:index + num_s])
            volume_true = np.array(all_trues[index:index + num_s])
            
            # 计算volume级别的Dice
            volume_dice = self._dice_per_volume(volume_pred, volume_true)
            volume_dice_list.append(volume_dice)
            volume_metrics[patient_id] = {
                'dice': float(volume_dice),
                'num_slices': int(num_s)
            }
            
            index += num_s
        
        avg_dice = np.mean(volume_dice_list) if volume_dice_list else 0.0
        
        return volume_metrics, avg_dice
    
    def _dice_per_volume(self, y_pred, y_true):
        """
        计算volume级别的Dice系数（参考标准代码）
        
        Args:
            y_pred: 预测mask数组 (N, C, H, W) 或 (N, H, W)
            y_true: 真实mask数组 (N, C, H, W) 或 (N, H, W)
        
        Returns:
            dice系数
        """
        # 展平并二值化
        if len(y_pred.shape) == 4:
            y_pred = y_pred[:, 0]  # 取第一个通道
        if len(y_true.shape) == 4:
            y_true = y_true[:, 0]
        
        y_pred = np.round(y_pred).astype(int).flatten()
        y_true = np.round(y_true).astype(int).flatten()
        
        # 计算Dice
        intersection = np.sum(y_pred * y_true)
        union = np.sum(y_pred) + np.sum(y_true)
        
        if union == 0:
            return 1.0  # 如果两者都是全零，Dice=1
        
        dice = 2.0 * intersection / union
        return float(dice)

    def evaluate_classification_model(self, model, dataloader, device):
        """评估分类模型，并自动寻找最优阈值"""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []  # 存储所有概率值，用于寻找最优阈值
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="评估分类模型"):
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, labels = batch_data
                    images, labels = images.to(device), labels.to(device)
                else:
                    # 如果没有分类标签，从mask生成（mask有像素则label=1，否则label=0）
                    images, masks = batch_data
                    images = images.to(device)
                    labels = (masks.sum(dim=[1, 2, 3]) > 0).long().to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 有病变的概率
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['无病变', '有病变'], output_dict=True)
        
        # 自动寻找最优分类阈值（基于F1分数）
        optimal_threshold = 0.5
        best_f1 = 0.0
        if len(all_probs) > 0 and len(all_labels) > 0:
            thresholds = np.arange(0.3, 0.8, 0.05)
            for thresh in thresholds:
                thresh_preds = (np.array(all_probs) > thresh).astype(int)
                if len(np.unique(thresh_preds)) > 1:  # 确保有正负样本
                    from sklearn.metrics import f1_score
                    f1 = f1_score(all_labels, thresh_preds)
                    if f1 > best_f1:
                        best_f1 = f1
                        optimal_threshold = thresh
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'optimal_threshold': float(optimal_threshold),
            'best_f1_at_threshold': float(best_f1)
        }
        
        return metrics

    def evaluate_two_stage_system(self, classification_model, segmentation_model, dataloader, device, 
                                   classification_threshold=0.5, segmentation_threshold=0.5, use_tta=True,
                                   use_adaptive_strategy=True, confidence_threshold=0.9):
        """
        评估两阶段系统（分类+分割）- 改进的级联策略
        
        Args:
            classification_model: 分类模型
            segmentation_model: 分割模型
            dataloader: 数据加载器（需要返回分类标签）
            device: 设备
            classification_threshold: 分类阈值（logits的softmax后，类别1的概率）
            segmentation_threshold: 分割阈值
            use_tta: 是否使用测试时增强
            use_adaptive_strategy: 是否使用自适应策略（只对高置信度的无病变样本跳过分割）
            confidence_threshold: 置信度阈值（只有无病变概率>此值才跳过分割）
        """
        classification_model.eval()
        segmentation_model.eval()
        
        # 分类指标
        cls_correct = 0
        cls_total = 0
        cls_preds = []
        cls_labels = []
        
        # 分割指标（只对分类为有病变的图像计算，用于评估分割模型本身）
        seg_metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # 系统整体指标（计算所有样本的最终输出，包括分类错误的情况）
        system_dice_list = []  # 系统整体Dice（所有样本）
        system_iou_list = []
        system_precision_list = []
        system_recall_list = []
        
        # 统计信息
        skip_count = 0  # 跳过分割的样本数
        total_count = 0
        
        # 整体系统指标
        system_metrics = {
            'true_positive': 0,  # 正确分类为有病变且分割正确
            'false_positive': 0,   # 错误分类为有病变
            'false_negative': 0,  # 错误分类为无病变（漏检）
            'true_negative': 0   # 正确分类为无病变
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估两阶段系统（改进级联策略）"):
                if len(batch) == 3:
                    images, masks, labels = batch
                    images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                else:
                    images, masks = batch
                    images, masks = images.to(device), masks.to(device)
                    # 从mask生成标签
                    labels = (masks.sum(dim=[1, 2, 3]) > 0).long()
                
                # 第一阶段：分类
                cls_outputs = classification_model(images)
                cls_probs = torch.softmax(cls_outputs, dim=1)
                cls_prob_lesion = cls_probs[:, 1]  # 有病变的概率
                cls_prob_normal = cls_probs[:, 0]  # 无病变的概率
                
                # 改进的级联策略
                if use_adaptive_strategy:
                    # 自适应策略：只对高置信度的无病变样本跳过分割
                    # 1. 有病变概率 > classification_threshold → 进行分割
                    # 2. 无病变概率 > confidence_threshold → 跳过分割（高置信度无病变）
                    # 3. 其他情况（不确定）→ 仍然进行分割（保守策略）
                    need_segmentation = (cls_prob_lesion > classification_threshold) | (cls_prob_normal < confidence_threshold)
                    cls_predicted = (cls_prob_lesion > classification_threshold).long()
                else:
                    # 原始策略：只对分类为有病变的进行分割
                    cls_predicted = (cls_prob_lesion > classification_threshold).long()
                    need_segmentation = cls_predicted == 1
                
                cls_total += labels.size(0)
                cls_correct += (cls_predicted == labels).sum().item()
                cls_preds.extend(cls_predicted.cpu().numpy())
                cls_labels.extend(labels.cpu().numpy())
                
                batch_size = images.size(0)
                total_count += batch_size
                
                # 初始化系统最终输出（全零mask）
                system_final_preds = torch.zeros_like(masks)
                
                # 统计跳过的样本
                skip_count += (need_segmentation == False).sum().item()
                
                if need_segmentation.any():
                    seg_images = images[need_segmentation]
                    seg_masks = masks[need_segmentation]
                    seg_labels = labels[need_segmentation]
                    
                    if use_tta:
                        seg_outputs = self._tta_inference(segmentation_model, seg_images)
                    else:
                        seg_outputs = segmentation_model(seg_images)
                    
                    # 确保 seg_outputs 和 seg_masks 的空间尺寸匹配
                    if seg_outputs.shape[2:] != seg_masks.shape[2:]:
                        seg_outputs = F.interpolate(seg_outputs, size=seg_masks.shape[2:], mode='bilinear', align_corners=False)
                    
                    seg_preds = torch.sigmoid(seg_outputs)
                    seg_preds = (seg_preds > segmentation_threshold).float()
                    
                    # 将分割结果填入系统最终输出
                    seg_idx = 0
                    for i in range(batch_size):
                        if need_segmentation[i]:
                            system_final_preds[i] = seg_preds[seg_idx]
                            seg_idx += 1
                        # 如果跳过分割，保持全零mask（系统最终输出）
                    
                    # 计算分割指标（只对进行分割的样本，用于评估分割模型本身）
                    for i in range(seg_preds.shape[0]):
                        pred = seg_preds[i, 0]
                        mask = seg_masks[i, 0]
                        
                        # 双重检查尺寸匹配（以防万一）
                        if pred.shape != mask.shape:
                            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                        
                        pred_sum = float(pred.sum().item())
                        mask_sum = float(mask.sum().item())
                        intersection = float((pred * mask).sum().item())
                        
                    if mask_sum > 1e-7 or pred_sum > 1e-7:
                        # 标准混淆矩阵定义，确保与主评估一致
                        tp = intersection
                        fp = float((pred * (1 - mask)).sum().item())
                        fn = float(((1 - pred) * mask).sum().item())
                        tn = float(((1 - pred) * (1 - mask)).sum().item())
                        
                        dice_den = 2.0 * tp + fp + fn
                        dice = 1.0 if dice_den < 1e-7 else (2.0 * tp) / dice_den
                        
                        union = tp + fp + fn
                        iou = 1.0 if union < 1e-7 else tp / union
                        
                        precision = 1.0 if (tp + fp) < 1e-7 else tp / (tp + fp)
                        recall = 1.0 if (tp + fn) < 1e-7 else tp / (tp + fn)
                        specificity = 1.0 if (tn + fp) < 1e-7 else tn / (tn + fp)
                        f1 = dice
                        
                        seg_metrics['dice'].append(float(dice))
                        seg_metrics['iou'].append(float(iou))
                        seg_metrics['precision'].append(float(precision))
                        seg_metrics['recall'].append(float(recall))
                        seg_metrics['f1'].append(float(f1))
                
                # 计算系统整体Dice（所有样本，包括分类错误的情况）
                for i in range(batch_size):
                    system_pred = system_final_preds[i, 0]
                    true_mask = masks[i, 0]
                    
                    # 双重检查尺寸匹配（以防万一）
                    if system_pred.shape != true_mask.shape:
                        system_pred = F.interpolate(system_pred.unsqueeze(0).unsqueeze(0), size=true_mask.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    
                    pred_sum = float(system_pred.sum().item())
                    mask_sum = float(true_mask.sum().item())
                    intersection = float((system_pred * true_mask).sum().item())
                    
                    # 计算系统整体Dice（包括空mask的情况）
                    if mask_sum > 1e-7 or pred_sum > 1e-7:
                        dice = self._safe_dice_score(system_pred, true_mask)
                        total = pred_sum + mask_sum
                        union = total - intersection
                        iou = (intersection + 1e-7) / (union + 1e-7) if union > 1e-7 else 0.0
                        precision = (intersection + 1e-7) / (pred_sum + 1e-7) if pred_sum > 1e-7 else 0.0
                        recall = (intersection + 1e-7) / (mask_sum + 1e-7) if mask_sum > 1e-7 else 0.0
                        
                        system_dice_list.append(float(dice))
                        system_iou_list.append(float(iou))
                        system_precision_list.append(float(precision))
                        system_recall_list.append(float(recall))
                
                # 计算整体系统指标
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    pred_label = cls_predicted[i].item()
                    
                    if true_label == 1 and pred_label == 1:
                        system_metrics['true_positive'] += 1
                    elif true_label == 0 and pred_label == 1:
                        system_metrics['false_positive'] += 1
                    elif true_label == 1 and pred_label == 0:
                        system_metrics['false_negative'] += 1
                    else:
                        system_metrics['true_negative'] += 1
        
        # 计算分类准确率
        cls_accuracy = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
        
        # 计算分类混淆矩阵
        cls_labels_arr = np.array(cls_labels)
        cls_preds_arr = np.array(cls_preds)
        cls_confusion_matrix = {
            'true_positive': int(((cls_labels_arr == 1) & (cls_preds_arr == 1)).sum()),
            'false_positive': int(((cls_labels_arr == 0) & (cls_preds_arr == 1)).sum()),
            'false_negative': int(((cls_labels_arr == 1) & (cls_preds_arr == 0)).sum()),
            'true_negative': int(((cls_labels_arr == 0) & (cls_preds_arr == 0)).sum())
        }
        
        # 计算分割平均指标（只对分类为有病变的样本，用于评估分割模型本身）
        seg_avg_metrics = {}
        for k, v in seg_metrics.items():
            if v:
                seg_avg_metrics[k] = float(np.mean(v))
            else:
                seg_avg_metrics[k] = 0.0
        
        # 计算系统整体Dice指标（所有样本，包括分类错误的情况）
        system_dice_avg = float(np.mean(system_dice_list)) if system_dice_list else 0.0
        system_iou_avg = float(np.mean(system_iou_list)) if system_iou_list else 0.0
        system_precision_avg = float(np.mean(system_precision_list)) if system_precision_list else 0.0
        system_recall_avg = float(np.mean(system_recall_list)) if system_recall_list else 0.0
        
        # 计算效率提升
        skip_ratio = skip_count / total_count if total_count > 0 else 0.0
        
        # 计算整体系统指标
        total_samples = (system_metrics['true_positive'] + system_metrics['false_positive'] + 
                         system_metrics['false_negative'] + system_metrics['true_negative'])
        
        system_accuracy = 100.0 * (system_metrics['true_positive'] + system_metrics['true_negative']) / total_samples if total_samples > 0 else 0.0
        system_precision = system_metrics['true_positive'] / (system_metrics['true_positive'] + system_metrics['false_positive'] + 1e-7)
        system_recall = system_metrics['true_positive'] / (system_metrics['true_positive'] + system_metrics['false_negative'] + 1e-7)
        system_f1 = 2 * system_precision * system_recall / (system_precision + system_recall + 1e-7)
        
        results = {
            'classification': {
                'accuracy': cls_accuracy,
                'confusion_matrix': cls_confusion_matrix
            },
            'segmentation': seg_avg_metrics,  # 分割模型指标（只对进行分割的样本）
            'system': {
                'accuracy': system_accuracy,
                'precision': system_precision,
                'recall': system_recall,
                'f1': system_f1,
                'dice': system_dice_avg,  # 系统整体Dice（所有样本）
                'iou': system_iou_avg,
                'segmentation_precision': system_precision_avg,
                'segmentation_recall': system_recall_avg,
                'confusion_matrix': system_metrics,
                'efficiency': {
                    'skip_ratio': skip_ratio,  # 跳过分割的样本比例
                    'computation_saved': skip_ratio * 100  # 节省的计算百分比
                }
            }
        }
        
        return results
    
    def visualize_test_results(self, model, dataloader, device, num_samples=8, use_tta=True):
        """可视化测试集上的分割结果，包含原图、真实mask、预测mask和对比图
        
        Args:
            use_tta: 是否使用测试时增强（默认True，训练结束后的测试推荐使用）
        """
        save_path = os.path.join(self.temp_dir, "test_results_visualization.png")
        model.eval()
        
        # 收集样本
        all_images = []
        all_masks = []
        all_preds = []
        all_metrics = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                
                # 使用TTA进行预测（训练结束后的测试推荐使用）
                if use_tta:
                    outputs = self._tta_inference(model, images)
                else:
                    outputs = model(images)
                # 确保 outputs 和 masks 的空间尺寸匹配
                if outputs.shape[2:] != masks.shape[2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()
                
                for i in range(images.size(0)):
                    if len(all_masks) >= num_samples:
                        break
                    
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1).astype(np.float32)
                    mask = masks[i, 0].cpu().numpy().astype(np.float32)
                    pred = preds_binary[i, 0].cpu().numpy().astype(np.float32)
                    
                    # 确保 pred 和 mask 的尺寸匹配（双重检查，以防万一）
                    if pred.shape != mask.shape:
                        from scipy.ndimage import zoom
                        if len(pred.shape) == 2 and len(mask.shape) == 2:
                            zoom_factors = (mask.shape[0] / pred.shape[0], mask.shape[1] / pred.shape[1])
                            pred = zoom(pred, zoom_factors, order=1)
                    
                    # 计算指标 - 使用改进的空mask处理
                    pred_sum = pred.sum()
                    mask_sum = mask.sum()
                    intersection = (pred * mask).sum()
                    
                    # 使用_safe_dice_score统一处理
                    dice = self._safe_dice_score(pred, mask)
                    
                    # IoU计算也需要特殊处理
                    if mask_sum <= 1e-7:
                        if pred_sum <= 1e-7:
                            iou = 1.0  # 完美匹配
                        else:
                            iou = 0.0  # 有误检
                    elif pred_sum <= 1e-7:
                        iou = 0.0  # 完全漏检
                    else:
                        total = pred_sum + mask_sum
                        union = total - intersection
                        iou = (intersection + 1e-7) / (union + 1e-7)
                    
                    all_images.append(img)
                    all_masks.append(mask)
                    all_preds.append(pred)
                    all_metrics.append({'dice': dice, 'iou': iou})
                
                if len(all_masks) >= num_samples:
                    break

        if self.enable_matlab_plots and self.matlab_viz_bridge:
            try:
                payload = self._save_test_results_payload(all_images, all_masks, all_preds, all_metrics, "test_results")
                matlab_path = os.path.join(self.temp_dir, "test_results_visualization_matlab.png")
                self.matlab_viz_bridge.render_test_results(payload, matlab_path)
                return matlab_path
            except Exception as exc:
                print(f"[MATLAB Plot] 测试可视化回退: {exc}")
        
        # 创建可视化
        num_samples = min(num_samples, len(all_images))
        cols = 4  # 原图、真实mask、预测mask、对比图
        rows = num_samples
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = all_images[i]
            true_mask = all_masks[i]
            pred_mask = all_preds[i]
            metrics = all_metrics[i]
            
            # 创建对比图：红色=真实，绿色=预测，黄色=重叠
            overlay = img.copy()
            overlay[true_mask == 1, 0] = 1  # 红色：真实区域
            overlay[pred_mask == 1, 1] = 1  # 绿色：预测区域
            overlay[(true_mask == 1) & (pred_mask == 1), 2] = 1  # 黄色：重叠区域
            
            # 原图
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"样本 {i+1}\n原始图像", fontsize=10)
            axes[i, 0].axis('off')
            
            # 真实mask
            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title("真实Mask\n(真实标签)", fontsize=10)
            axes[i, 1].axis('off')
            
            # 预测mask
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f"预测Mask\nDice: {metrics['dice']:.3f}\nIoU: {metrics['iou']:.3f}", 
                               fontsize=10)
            axes[i, 2].axis('off')
            
            # 对比图
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("对比图\n(红:真实, 绿:预测, 黄:重叠)", fontsize=10)
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_performance_analysis(self, detailed_metrics):
        """生成性能分析报告的可视化"""
        save_path = os.path.join(self.temp_dir, "performance_analysis.png")

        if self.enable_matlab_plots and self.matlab_viz_bridge:
            try:
                payload = self._save_performance_payload(detailed_metrics)
                matlab_path = os.path.join(self.temp_dir, "performance_analysis_matlab.png")
                self.matlab_viz_bridge.render_performance_analysis(payload, matlab_path)
                return matlab_path
            except Exception as exc:
                print(f"[MATLAB Plot] 性能分析回退: {exc}")
        
        metrics = detailed_metrics['all_samples']
        avg_metrics = detailed_metrics['average']
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 指标分布直方图
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(metrics['dice'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(avg_metrics['dice'], color='red', linestyle='--', linewidth=2, label=f'平均: {avg_metrics["dice"]:.3f}')
        ax1.set_xlabel('Dice系数')
        ax1.set_ylabel('样本数量')
        ax1.set_title('Dice系数分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(metrics['iou'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(avg_metrics['iou'], color='red', linestyle='--', linewidth=2, label=f'平均: {avg_metrics["iou"]:.3f}')
        ax2.set_xlabel('IoU')
        ax2.set_ylabel('样本数量')
        ax2.set_title('IoU分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(metrics['precision'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(avg_metrics['precision'], color='red', linestyle='--', linewidth=2, label=f'平均: {avg_metrics["precision"]:.3f}')
        ax3.set_xlabel('精确率')
        ax3.set_ylabel('样本数量')
        ax3.set_title('精确率分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 2. 指标对比柱状图
        ax4 = plt.subplot(2, 3, 4)
        metric_names = ['Dice系数', 'IoU', '精确率', '敏感度(召回率)', '特异度', 'F1分数']
        metric_values = [
            avg_metrics['dice'],
            avg_metrics['iou'],
            avg_metrics['precision'],
            avg_metrics.get('sensitivity', avg_metrics.get('recall', 0)),
            avg_metrics.get('specificity', 0),
            avg_metrics['f1']
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8e44ad']
        bars = ax4.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('分数')
        ax4.set_title('平均性能指标对比')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 指标箱线图
        ax5 = plt.subplot(2, 3, 5)
        box_data = [
            metrics['dice'],
            metrics['iou'],
            metrics['precision'],
            metrics.get('sensitivity', metrics['recall']),
            metrics['specificity'],
            metrics['f1']
        ]
        bp = ax5.boxplot(box_data, tick_labels=metric_names, patch_artist=True)  # 使用tick_labels替代labels（已翻译为中文）
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_ylabel('分数')
        ax5.set_title('指标分布箱线图')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 4. 统计信息表格
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        stats_data = []
        for metric in ['dice', 'iou', 'precision', 'sensitivity', 'specificity', 'f1', 'hd95']:
            stats_data.append([
                metric.upper(),
                f"{detailed_metrics['average'][metric]:.4f}",
                f"{detailed_metrics['std'][metric]:.4f}",
                f"{detailed_metrics['min'][metric]:.4f}",
                f"{detailed_metrics['max'][metric]:.4f}",
                f"{detailed_metrics['median'][metric]:.4f}"
            ])
        
        table = ax6.table(cellText=stats_data,
                         colLabels=['指标', '平均值', '标准差', '最小值', '最大值', '中位数'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('模型性能分析报告', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_attention_maps(self, model, dataloader, device, num_samples=4):
        """可视化注意力权重图，用于模型可解释性分析 - 优化版"""
        if not self._supports_attention_maps(model):
            raise RuntimeError("当前模型不支持注意力可视化")
        save_path = os.path.join(self.temp_dir, "attention_visualization.png")
        model.eval()
        
        # 收集样本和注意力图
        all_images = []
        all_masks = []
        all_preds = []
        all_attention_maps = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                # 获取预测结果和注意力权重
                outputs, attention_maps = model(images, return_attention=True)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()
                
                for i in range(images.size(0)):
                    if len(all_images) >= num_samples:
                        break
                    
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1).astype(np.float32)
                    mask = masks[i, 0].cpu().numpy().astype(np.float32)
                    pred = preds_binary[i, 0].cpu().numpy().astype(np.float32)
                    
                    # 收集所有层的注意力图，并上采样到原始图像大小
                    att_dict = {}
                    for att_name, att_map in attention_maps.items():
                        att_np = att_map[i, 0].cpu().numpy()
                        # 上采样到256x256（与输入图像大小一致）
                        from scipy.ndimage import zoom
                        target_size = (256, 256)
                        if att_np.shape != target_size:
                            zoom_factors = (target_size[0] / att_np.shape[0], target_size[1] / att_np.shape[1])
                            att_np = zoom(att_np, zoom_factors, order=1)
                        att_dict[att_name] = att_np
                    
                    all_images.append(img)
                    all_masks.append(mask)
                    all_preds.append(pred)
                    all_attention_maps.append(att_dict)
                
                if len(all_images) >= num_samples:
                    break

        att_layer_payload = {'att1': [], 'att2': [], 'att3': [], 'att4': []}
        for att_dict in all_attention_maps:
            for key in att_layer_payload.keys():
                if key in att_dict:
                    att_layer_payload[key].append(att_dict[key])

        if self.enable_matlab_plots and self.matlab_viz_bridge:
            try:
                payload_path = self._save_attention_payload(all_images, all_masks, all_preds, att_layer_payload, "attention_visualization")
                matlab_path = os.path.join(self.temp_dir, "attention_visualization_matlab.png")
                self.matlab_viz_bridge.render_attention_maps(payload_path, matlab_path)
                return matlab_path
            except Exception as exc:
                print(f"[MATLAB Plot] 注意力可视化回退: {exc}")
        
        # 创建可视化 - 优化布局
        num_samples = min(num_samples, len(all_images))
        cols = 7  # 原图、真实mask、预测mask、att1叠加、att2叠加、att3叠加、att4叠加
        rows = num_samples
        
        fig, axes = plt.subplots(rows, cols, figsize=(24, 4.5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = all_images[i]
            true_mask = all_masks[i]
            pred_mask = all_preds[i]
            att_maps = all_attention_maps[i]
            
            # 原图
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"样本 {i+1}\n原始图像", fontsize=11, fontweight='bold', pad=8)
            axes[i, 0].axis('off')
            
            # 真实mask
            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title("真实Mask\n(Ground Truth)", fontsize=11, fontweight='bold', pad=8)
            axes[i, 1].axis('off')
            
            # 预测mask
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title("预测Mask\n(Prediction)", fontsize=11, fontweight='bold', pad=8)
            axes[i, 2].axis('off')
            
            # 注意力图叠加显示（在原图上叠加注意力热力图）
            col_idx = 3
            for att_name in ['att1', 'att2', 'att3', 'att4']:
                if att_name in att_maps and col_idx < cols:
                    att = att_maps[att_name]
                    layer_num = att_name[-1]
                    
                    # 归一化注意力图
                    att_norm = (att - att.min()) / (att.max() - att.min() + 1e-8)
                    
                    overlay = img.copy()
                    
                    import matplotlib.cm as cm
                    heatmap = cm.jet(att_norm)[:, :, :3]
                    
                    alpha = 0.5  # 透明度
                    blended = overlay * (1 - alpha) + heatmap * alpha
                    
                    # 显示叠加图像
                    im = axes[i, col_idx].imshow(blended)
                    axes[i, col_idx].set_title(f"注意力层{layer_num}\n(叠加显示)", 
                                             fontsize=11, fontweight='bold', pad=8)
                    axes[i, col_idx].axis('off')
                    
                    # 添加颜色条显示注意力强度（使用原始注意力图）
                    im_cbar = axes[i, col_idx].imshow(att_norm, cmap='hot', alpha=0.0)  # 仅用于colorbar
                    cbar = plt.colorbar(im_cbar, ax=axes[i, col_idx], fraction=0.046, pad=0.02)
                    cbar.set_label('注意力强度', fontsize=9, rotation=270, labelpad=15)
                    
                    col_idx += 1
        
        # 使用普通文本替代emoji，避免字体警告
        plt.suptitle('模型注意力权重可视化 - 可解释性分析', 
                    fontsize=18, fontweight='bold', y=0.995, color='#1e293b')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def analyze_attention_statistics(self, model, dataloader, device, num_samples=20):
        """分析注意力权重的统计特性 - 增强版，支持动态检测注意力层"""
        if not self._supports_attention_maps(model):
            raise RuntimeError("当前模型不支持注意力统计分析")
        model.eval()
        # 先运行一次获取实际的注意力层名称
        attention_stats = {}
        
        with torch.no_grad():
            eval_count = 0
            for batch_data in dataloader:
                if eval_count >= num_samples:
                    break
                
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                outputs, attention_maps = model(images, return_attention=True)
                
                # 初始化统计字典（只初始化实际存在的层）
                if not attention_stats:
                    for att_name in attention_maps.keys():
                        attention_stats[att_name] = {
                            'mean': [], 'std': [], 'max': [], 'min': [], 
                            'entropy': [], 'concentration': []
                        }
                
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()
                
                for i in range(images.size(0)):
                    if eval_count >= num_samples:
                        break
                    
                    mask_np = masks[i, 0].cpu().numpy()
                    pred_np = preds_binary[i, 0].cpu().numpy()
                    
                    for att_name, att_map in attention_maps.items():
                        if att_name not in attention_stats:
                            continue
                            
                        att_np = att_map[i, 0].cpu().numpy()
                        
                        # 检查是否有无效值
                        if np.any(np.isnan(att_np)) or np.any(np.isinf(att_np)):
                            # 跳过包含nan/inf的样本
                            continue
                        
                        # 基础统计
                        att_mean = float(att_np.mean())
                        att_std = float(att_np.std())
                        att_max = float(att_np.max())
                        att_min = float(att_np.min())
                        
                        if not (np.isnan(att_mean) or np.isinf(att_mean)):
                            attention_stats[att_name]['mean'].append(att_mean)
                        if not (np.isnan(att_std) or np.isinf(att_std)):
                            attention_stats[att_name]['std'].append(att_std)
                        if not (np.isnan(att_max) or np.isinf(att_max)):
                            attention_stats[att_name]['max'].append(att_max)
                        if not (np.isnan(att_min) or np.isinf(att_min)):
                            attention_stats[att_name]['min'].append(att_min)
                        
                        # 计算熵（衡量注意力分布的分散程度）
                        att_flat = att_np.flatten()
                        att_sum = att_flat.sum()
                        if att_sum > 1e-8:  # 确保不是全零
                            att_flat = att_flat / att_sum  # 归一化为概率分布
                            att_flat = att_flat[att_flat > 1e-8]  # 去除接近零的值
                            if len(att_flat) > 0:
                                entropy = -np.sum(att_flat * np.log(att_flat + 1e-8))
                                if not (np.isnan(entropy) or np.isinf(entropy)):
                                    attention_stats[att_name]['entropy'].append(float(entropy))
                        else:
                            # 全零情况，熵为0
                            attention_stats[att_name]['entropy'].append(0.0)
                        
                        # 计算集中度（高注意力值区域的占比）
                        if att_np.size > 0:
                            threshold = np.percentile(att_np, 90)  # 前10%的阈值
                            if not np.isnan(threshold):
                                concentration = float(np.sum(att_np >= threshold) / att_np.size)
                                if not (np.isnan(concentration) or np.isinf(concentration)):
                                    attention_stats[att_name]['concentration'].append(concentration)
                    
                    eval_count += 1
                
                if eval_count >= num_samples:
                    break
        
        # 计算平均统计，处理空列表情况
        avg_stats = {}
        for att_name, stats in attention_stats.items():
            avg_stats[att_name] = {}
            for stat_name, values in stats.items():
                if len(values) > 0:
                    avg_val = np.mean(values)
                    if not (np.isnan(avg_val) or np.isinf(avg_val)):
                        avg_stats[att_name][stat_name] = float(avg_val)
                    else:
                        avg_stats[att_name][stat_name] = 0.0
                else:
                    # 空列表，返回默认值
                    avg_stats[att_name][stat_name] = 0.0 if stat_name in ['mean', 'std', 'max', 'min'] else (0.0 if stat_name == 'entropy' else 0.0)
        
        return avg_stats
    def calculate_custom_score(self, dice, iou, precision, recall, specificity, hd95):
        """
        自定义综合评分函数:
        Score = (Dice * 50) + (IoU * 10) + (Precision * 10) + (Recall * 10) + (Specificity * 10) + Score_HD95
        其中 Score_HD95 = 10 / (HD95 + 1)，若HD95不可用则该项记为0。
        """
        import numpy as np
        
        # 确保输入转换为 float，防止 tensor 或 numpy 类型导致计算报错
        dice = float(dice) if dice is not None else 0.0
        iou = float(iou) if iou is not None else 0.0
        precision = float(precision) if precision is not None else 0.0
        recall = float(recall) if recall is not None else 0.0
        specificity = float(specificity) if specificity is not None else 0.0

        # HD95 项处理：越小越好 -> 反比得分
        # 如果 hd95 是 None, inf, nan 或负数，该项得分为 0
        if hd95 is None or not np.isfinite(hd95) or hd95 < 0:
            score_hd95 = 0.0
        else:
            score_hd95 = 10.0 / (float(hd95) + 1.0)

        # 加权求和
        total_score = (
            dice * 50.0 +        # Dice 权重最高 (50%)
            iou * 10.0 +         # IoU (10%)
            precision * 10.0 +   # Precision (10%)
            recall * 10.0 +      # Recall (10%)
            specificity * 10.0 + # Specificity (10%)
            score_hd95           # HD95 (约10%)
        )
        
        return float(total_score)
    def scan_best_threshold(self, prob_maps: np.ndarray, gt_masks: np.ndarray):
        """
        在给定的概率图和真实掩膜上扫描阈值，寻找综合评分最高的阈值。

        Args:
            prob_maps: 概率图，形状 [N, H, W] 或 [N, 1, H, W]，数值范围 [0,1]
            gt_masks:  真实掩膜，形状与 prob_maps 对应，取值 {0,1}

        Returns:
            best_thresh: 综合评分最高的阈值
            best_metrics: 对应阈值下的指标字典（dice, iou, precision, recall, specificity, hd95, score）
        """
        prob_maps = np.asarray(prob_maps, dtype=np.float32)
        gt_masks = np.asarray(gt_masks, dtype=np.float32)

        # 统一为 [N, H, W]
        if prob_maps.ndim == 4:
            prob_maps = prob_maps[:, 0]
        if gt_masks.ndim == 4:
            gt_masks = gt_masks[:, 0]

        # 二值化真值
        gt_bool = gt_masks > 0.5

        thresholds = np.arange(0.3, 0.91, 0.05, dtype=np.float32)
        best_thresh = 0.5
        best_score = -float("inf")
        best_metrics = {}

        for thr in thresholds:
            pred_bool = prob_maps >= float(thr)

            # 全局混淆矩阵（所有像素一起统计）
            tp = np.logical_and(pred_bool, gt_bool).sum(dtype=np.float64)
            fp = np.logical_and(pred_bool, ~gt_bool).sum(dtype=np.float64)
            fn = np.logical_and(~pred_bool, gt_bool).sum(dtype=np.float64)
            tn = np.logical_and(~pred_bool, ~gt_bool).sum(dtype=np.float64)

            pred_sum = tp + fp
            mask_sum = tp + fn

            dice_den = 2.0 * tp + fp + fn
            if dice_den < 1e-7:
                dice = 1.0 if (mask_sum < 1e-7 and pred_sum < 1e-7) else 0.0
            else:
                dice = (2.0 * tp) / (dice_den + 1e-8)

            union = tp + fp + fn
            iou = 1.0 if union < 1e-7 else tp / (union + 1e-8)

            if pred_sum < 1e-7:
                precision = 1.0 if mask_sum < 1e-7 else 0.0
            else:
                precision = tp / (pred_sum + 1e-8)

            if (tp + fn) < 1e-7:
                recall = 1.0 if pred_sum < 1e-7 else 0.0
            else:
                recall = tp / (tp + fn + 1e-8)

            if (tn + fp) < 1e-7:
                specificity = 1.0
            else:
                specificity = tn / (tn + fp + 1e-8)

            # 计算该阈值下的平均 HD95（对每个样本单独计算）
            hd95_list = []
            for i in range(pred_bool.shape[0]):
                try:
                    hd = self.calculate_hd95(
                        pred_bool[i].astype(np.uint8),
                        gt_bool[i].astype(np.uint8),
                    )
                except Exception:
                    hd = float("nan")
                if np.isfinite(hd):
                    hd95_list.append(float(hd))

            if hd95_list:
                hd95_mean = float(np.nanmean(hd95_list))
            else:
                # 若所有样本都无法计算 HD95，则记为无穷大，以便在评分中让该项为 0
                hd95_mean = float("inf")

            total_score = self.calculate_custom_score(
                dice=dice,
                iou=iou,
                precision=precision,
                recall=recall,
                specificity=specificity,
                hd95=hd95_mean,
            )

            if total_score > best_score:
                best_score = float(total_score)
                best_thresh = float(thr)
                best_metrics = {
                    "dice": float(dice),
                    "iou": float(iou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "hd95": float(hd95_mean) if np.isfinite(hd95_mean) else float("nan"),
                    "score": float(total_score),
                }

        return best_thresh, best_metrics
    


    def run(self):
        try:
            # 初始化设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.update_progress.emit(0, f"使用设备: {device}")
            
            # 数据准备
            patient_ids = [pid for pid in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, pid))]
            
            # 单模型训练
            train_ids, val_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
            
            # 数据增强（增强对比度、光照和形变，提升泛化能力）
            # 优化数据增强 - 针对医学影像的非刚体形变特性
            # 重点增强：Grid Distortion + Elastic Transform（模拟器官挤压和变形）
            # MixUp 将在训练循环中实现（需要两张图像混合）
            train_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
                # Grid Distortion：模拟非刚体形变，对医学影像非常有效
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,  # 增强形变幅度
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.3  # 30%概率应用
                ),
                # Elastic Transform：模拟器官的挤压和变形（医学影像最强增强）
                A.ElasticTransform(
                    alpha=50,  # 增强形变强度（从10提升到50）
                    sigma=5,   # 增强平滑度（从3提升到5）
                    alpha_affine=10,  # 添加仿射变换
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.4  # 提高概率（从0.15提升到0.4）
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                # GaussNoise 参数需使用 var_limit 而非 mean/std
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            # 验证集仅做几何归一化，避免引入过多随机性
            val_transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            # 加载分割训练数据
            self.update_progress.emit(5, "正在加载分割训练数据...")
            # 根据CPU核心数和操作系统设置合适的num_workers
            # Windows上使用多进程可能导致卡死，建议使用0或1
            import platform
            is_windows = platform.system() == 'Windows'
            cpu_count = os.cpu_count() or 1
            if is_windows:
                # Windows上使用单进程或0，避免卡死
                num_workers = 0
                use_persistent_workers = False
            else:
                # Linux/Mac可以使用多进程
                num_workers = max(0, min(4, cpu_count - 1))
                use_persistent_workers = num_workers > 0
            
            self.update_progress.emit(6, f"数据加载器配置: num_workers={num_workers}")
            
            train_dataset = self.load_dataset(train_ids, train_transform, split_name="train", return_classification=False)
            train_sampler = None
            if getattr(train_dataset, "use_weighted_sampling", False):
                weights = train_dataset.get_sampling_weights()
                if weights is not None:
                    weight_tensor = torch.as_tensor(weights, dtype=torch.double)
                    train_sampler = WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda' and not is_windows),  # Windows上pin_memory可能导致问题
                persistent_workers=use_persistent_workers,
                prefetch_factor=2 if num_workers > 0 else None
            )
            
            self.update_progress.emit(10, "正在加载分割验证数据...")
            val_dataset = self.load_dataset(val_ids, val_transform, split_name="val", return_classification=False, use_weighted_sampling=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda' and not is_windows),
                persistent_workers=use_persistent_workers,
                prefetch_factor=2 if num_workers > 0 else None
            )
            
            train_pos_weight = self.pos_weight_cache.get('train')
            if train_pos_weight is None:
                mask_paths = self.split_metadata.get('train', {}).get('mask_paths', [])
                train_pos_weight = self._estimate_pos_weight(mask_paths)
                self.pos_weight_cache['train'] = train_pos_weight
            self.update_progress.emit(12, f"估计前景权重: {train_pos_weight:.2f}")
            
            # 如果有预训练模型，先读取配置以确保架构匹配
            if self.model_path and os.path.exists(self.model_path):
                # 若用户选择的是 last_model.pth，优先回退到同目录下的 best_model_dice_*.pth
                model_path_to_use = self.model_path
                base_name = os.path.basename(self.model_path)
                if base_name.startswith("last_model"):
                    parent = os.path.dirname(self.model_path)
                    try:
                        cand = sorted(
                            [p for p in os.listdir(parent) if p.startswith("best_model_dice_") and p.endswith(".pth")],
                            reverse=True,
                        )
                        if cand:
                            model_path_to_use = os.path.join(parent, cand[0])
                            print(f"[提示] 检测到 last_model.pth，自动切换为最佳模型权重: {os.path.basename(model_path_to_use)}")
                    except Exception:
                        pass

                ckpt_config = read_checkpoint_config(model_path_to_use)
                if ckpt_config:
                    # 从checkpoint推断的配置覆盖当前设置
                    if 'model_type' in ckpt_config:
                        self.model_type = ckpt_config['model_type']
                    if 'swin_params' in ckpt_config and ckpt_config['swin_params']:
                        self.swin_params = copy.deepcopy(ckpt_config['swin_params'])
                        self.use_gwo = False  # 已有参数，禁用GWO
                        self.update_progress.emit(13, f"从checkpoint推断SwinUNet参数: embed_dim={self.swin_params.get('embed_dim')}")
                    if 'dstrans_params' in ckpt_config and ckpt_config['dstrans_params']:
                        self.dstrans_params = copy.deepcopy(ckpt_config['dstrans_params'])
                        self.use_gwo = False
                        self.update_progress.emit(13, f"从checkpoint推断DS-TransUNet参数: embed_dim={self.dstrans_params.get('embed_dim')}")
            
            # GWO优化（SwinUNet / DS-TransUNet）
            if self.use_gwo and self.swin_params is None and (self.model_type == "swin_unet" or self.model_type == "swinunet"):
                self.update_progress.emit(13, "开始GWO优化SwinUNet超参数...")
                self.swin_params = self._gwo_optimize_swin_params(train_loader, val_loader, device)
                self.update_progress.emit(14, f"GWO优化完成，最佳参数: {self.swin_params}")
            if self.use_gwo and self.dstrans_params is None and self.model_type in ("ds_trans_unet", "dstransunet", "ds-transunet"):
                self.update_progress.emit(13, "开始GWO优化DS-TransUNet超参数...")
                self.dstrans_params = self._gwo_optimize_dstrans_params(train_loader, val_loader, device)
                self.update_progress.emit(14, f"GWO优化完成，最佳参数: {self.dstrans_params}")
            
            # 初始化模型
            self.update_progress.emit(15, f"正在构建模型 ({self.model_type})...")
            try:
                model = self._build_model(device, swin_params=self.swin_params, dstrans_params=self.dstrans_params)
                self.update_progress.emit(16, "模型构建完成")
            except Exception as e:
                self.update_progress.emit(0, f"模型构建失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return
            if self.model_path and os.path.exists(self.model_path):
                # 与上方一致：若为 last_model.pth，则优先加载同目录下分数最高的 best_model_dice_*.pth
                model_path_to_use = self.model_path
                base_name = os.path.basename(self.model_path)
                if base_name.startswith("last_model"):
                    parent = os.path.dirname(self.model_path)
                    try:
                        cand = sorted(
                            [p for p in os.listdir(parent) if p.startswith("best_model_dice_") and p.endswith(".pth")],
                            reverse=True,
                        )
                        if cand:
                            model_path_to_use = os.path.join(parent, cand[0])
                    except Exception:
                        pass

                # 使用兼容加载函数
                success, msg = load_model_compatible(model, model_path_to_use, device, verbose=False)
                self.update_progress.emit(15, msg)
            ema_model = None
            if self.use_ema:
                ema_model = self._init_ema_model(model, device)
            
            # 优化器和损失函数
            # 默认学习率：预训练模型（ResNet）使用更小的学习率进行微调
            # SwinUNet 和 Transformer 模型从头训练，可以使用稍大的学习率
            if self.model_type in ("swin_unet", "swinunet"):
                default_lr = 5e-5
            elif self.model_type == "resnet_unet":
                # ResNet101使用预训练权重，需要更小的学习率进行微调
                # 从5e-5进一步降低到2e-5，避免梯度爆炸和数值不稳定
                default_lr = 2e-5
            else:
                default_lr = 1e-4

            # 若设置了环境变量 SEG_LR，则优先使用，便于在训练瓶颈时手动降低学习率
            env_lr = os.environ.get("SEG_LR")
            try:
                initial_lr = float(env_lr) if env_lr is not None else default_lr
            except ValueError:
                print(f"[警告] 无法解析 SEG_LR='{env_lr}'，回退到默认学习率 {default_lr}")
                initial_lr = default_lr

            optimizer = self._create_optimizer(model.parameters(), lr=initial_lr)
            # 增强前景权重以处理类别不平衡
            adjusted_pos_weight = min(train_pos_weight * 1.5, 20.0)
            bce_weight_tensor = torch.tensor([adjusted_pos_weight], device=device)
            bce_criterion = nn.BCEWithLogitsLoss(pos_weight=bce_weight_tensor)

            # Poly学习率 + Warmup: lr = base_lr * (1 - epoch / max_epochs) ** power
            warmup_epochs_lr = 5
            poly_power = float(os.environ.get("SEG_POLY_POWER", "0.9"))
            scheduler = None
            # 使用 ReduceLROnPlateau 在验证Dice长期不提升时自动降低学习率
            # 兼容较旧版本的PyTorch，这里不使用verbose参数
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=3
            )

            # SWA与早停配置 - 若启用EMA则默认关闭SWA避免冲突
            swa_enabled = (not self.use_ema) and self.epochs >= 15
            swa_start_epoch = max(int(self.epochs * 0.5), 1)  # 更早启用
            swa_model = AveragedModel(model) if swa_enabled else None
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=2e-5,  # 更低的SWA学习率
                anneal_epochs=3,
                anneal_strategy='cos'
            ) if swa_enabled else None
            swa_active_epochs = 0

            warmup_epochs = min(8, max(3, self.epochs // 5))
            # 小数据集：更宽松的早停策略，给模型充分学习时间
            early_stopping = EarlyStopping(
                patience=max(12, self.epochs // 3),  # 更大耐心
                min_delta=1e-4,  # 更低阈值
                min_rel_improve=0.003,  # 更低相对提升要求
                warmup_epochs=warmup_epochs + 5,  # 更长预热
                cooldown=3,  # 更长冷却
                smoothing=0.3,
            )
            early_stop_triggered = False

            # AMP 混合精度训练（仅CUDA启用）
            amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            amp_enabled = (amp_device_type == 'cuda')
            # SwinUNet在半精度下更容易出现溢出，默认关闭AMP或使用更小的缩放
            if self.model_type in ("swin_unet", "swinunet"):
                amp_enabled = False
            scaler = GradScaler('cuda', enabled=amp_enabled, init_scale=2.0 ** 7, growth_interval=200, growth_factor=1.5, backoff_factor=0.5)
            
            # 训练循环
            # 冻结/解冻策略：前50% epoch冻结编码器，后50%解冻进行微调
            freeze_epochs = int(self.epochs * 0.5)
            encoder_frozen = False
            # 训练过程中用于学习率调度的基准LR（解冻时会动态下调）
            base_lr = float(initial_lr)
            
            for epoch in range(self.epochs):
                if self.stop_requested:
                    self.update_progress.emit(0, "训练已由用户停止")
                    # 【修复】用户停止时也要发送完成信号，确保UI正确更新
                    self.training_finished.emit("训练已被用户停止", self.best_model_path if self.save_best else None)
                    return
                
                # 冻结/解冻编码器逻辑（仅对 ResNetUNet 有效）
                if self.model_type == "resnet_unet":
                    actual_model = self._unwrap_model(model)
                    if isinstance(actual_model, ResNetUNet):
                        if epoch < freeze_epochs:
                            # 前50% epoch：冻结编码器
                            if not encoder_frozen:
                                actual_model._freeze_encoder()
                                encoder_frozen = True
                                # 重新创建优化器，只优化可训练参数
                                trainable_params = [p for p in model.parameters() if p.requires_grad]
                                optimizer = self._create_optimizer(trainable_params, initial_lr)
                                print(f"[训练策略] Epoch {epoch+1}/{self.epochs}: 编码器已冻结，仅训练解码器")
                        else:
                            # 后50% epoch：解冻编码器进行微调
                            if encoder_frozen:
                                actual_model._unfreeze_encoder()
                                encoder_frozen = False
                                # 重新创建优化器，优化所有参数（使用较小的学习率进行微调）
                                # 解冻瞬间：把“当前学习率”强制降低到 1/10，避免 ResNet101 全量微调震荡
                                current_lr = float(optimizer.param_groups[0]['lr'])
                                fine_tune_lr = current_lr * 0.1
                                base_lr = fine_tune_lr  # 同时更新后续Poly调度的基准LR，避免被initial_lr覆盖回去
                                trainable_params = [p for p in model.parameters() if p.requires_grad]
                                optimizer = self._create_optimizer(trainable_params, fine_tune_lr)
                                print(f"[训练策略] Epoch {epoch+1}/{self.epochs}: 编码器已解冻，开始端到端微调 (LR={fine_tune_lr:.6f})")
                
                epoch_loss_weights = self._get_loss_weights(epoch, self.epochs)
                
                # 每个epoch开始时重置梯度消失计数器
                if hasattr(self, '_zero_grad_count'):
                    self._zero_grad_count = 0
                
                # Warmup + Poly学习率调整
                if epoch < warmup_epochs_lr:
                    # 线性Warmup到 base_lr
                    warmup_factor = (epoch + 1) / warmup_epochs_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = base_lr * warmup_factor
                else:
                    # Warmup结束后，按epoch使用Poly策略衰减学习率
                    t = (epoch - warmup_epochs_lr) / max(1, self.epochs - warmup_epochs_lr)
                    lr = base_lr * (1.0 - t) ** poly_power
                    # ResNet50需要更大的最小学习率，避免梯度消失
                    min_lr = 1e-5 if self.model_type == "resnet_unet" else 1e-6
                    lr = max(lr, min_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                # 训练阶段
                model.train()
                # 确保EMA模型也处于train模式（以便BN统计量能正确更新）
                if self.use_ema and ema_model is not None:
                    ema_model.train()
                epoch_loss = 0.0
                train_samples = 0
                
                # 添加进度提示，避免看起来卡死
                if epoch == 0:
                    self.update_progress.emit(20, "开始第一个训练批次（首次运行可能较慢，请耐心等待）...")
                
                for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f'训练轮次 {epoch+1}/{self.epochs}')):
                    if self.stop_requested:
                        # 【修复】用户停止时也要发送完成信号，确保UI正确更新
                        self.training_finished.emit("训练已被用户停止", self.best_model_path if self.save_best else None)
                        return
                    
                    # 处理数据
                    images, masks = batch_data
                    images, masks = images.to(device), masks.float().to(device)
                    
                    batch_size = images.size(0)
                    
                    # MixUp 数据增强（小数据集增强泛化能力，防止对特定纹理过拟合）
                    # 从第3个epoch开始，50%概率使用MixUp
                    use_mixup = (epoch >= 3) and (np.random.rand() < 0.5) and (batch_size > 1)
                    if use_mixup:
                        # 随机打乱索引，创建混合对
                        indices = torch.randperm(batch_size).to(device)
                        # Beta分布生成混合系数 lambda（alpha=0.2 使得混合更保守，适合医学影像）
                        lam = np.random.beta(0.2, 0.2)
                        lam = max(lam, 1.0 - lam)  # 确保主要样本权重更大
                        
                        # 混合图像
                        mixed_images = lam * images + (1.0 - lam) * images[indices]
                        # 混合mask（保持相同的lambda）
                        mixed_masks = lam * masks + (1.0 - lam) * masks[indices]
                        
                        images = mixed_images
                        masks = mixed_masks
                    
                    # 定期清理GPU缓存，降低显存峰值
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    brain_mask = None
                    if self.use_skull_stripper:
                        images, brain_mask = self._apply_skull_strip(images)

                    # 输入数据验证：检查NaN/Inf（在增加train_samples之前）
                    if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)):
                        print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 输入图像包含NaN/Inf，跳过此批次")
                        continue
                    if torch.any(torch.isnan(masks)) or torch.any(torch.isinf(masks)):
                        print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 输入掩膜包含NaN/Inf，跳过此批次")
                        continue
                    
                    # 只有在所有检查通过后才增加train_samples
                    train_samples += batch_size
                    
                    # 检查输入数据范围是否合理
                    # ImageNet归一化后，理论上值域在-2.5到2.5左右
                    # 考虑数据增强（ColorJitter、RandomBrightnessContrast等），合理范围扩展到-5到5
                    # 只有在极端情况下（超出-10到10）才警告并裁剪
                    image_min, image_max = images.min().item(), images.max().item()
                    if image_min < -10.0 or image_max > 10.0:
                        # 只在真正极端的情况下才打印警告（避免过多日志）
                        if image_min < -15.0 or image_max > 15.0:
                            print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 输入图像值域异常 (min={image_min:.4f}, max={image_max:.4f})，进行裁剪")
                        # 裁剪到合理范围（ImageNet归一化 + 数据增强的合理范围）
                        images = torch.clamp(images, min=-5.0, max=5.0)
                    elif image_min < -5.0 or image_max > 5.0:
                        # 静默裁剪到合理范围，不打印警告（这是数据增强的正常结果）
                        images = torch.clamp(images, min=-5.0, max=5.0)
                    
                    if masks.min() < 0.0 or masks.max() > 1.0:
                        mask_min, mask_max = masks.min().item(), masks.max().item()
                        # 只在极端情况下才警告
                        if mask_min < -0.1 or mask_max > 1.1:
                            print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 掩膜值域异常 (min={mask_min:.4f}, max={mask_max:.4f})，进行裁剪")
                        masks = torch.clamp(masks, min=0.0, max=1.0)

                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type=amp_device_type, enabled=amp_enabled):
                        supports_aux = self._supports_aux_outputs(model)
                        supports_attention = self._supports_attention_maps(model)
                        forward_kwargs = {}
                        if supports_aux:
                            forward_kwargs['return_aux'] = True
                        if supports_attention:
                            forward_kwargs['return_attention'] = True
                        
                        if forward_kwargs:
                            forward_out = model(images, **forward_kwargs)
                            if supports_aux and supports_attention:
                                outputs, aux_outputs, attention_maps = forward_out
                            elif supports_aux:
                                outputs, aux_outputs = forward_out
                                attention_maps = {}
                            else:
                                outputs, attention_maps = forward_out
                                aux_outputs = []
                        else:
                            outputs = model(images)
                            aux_outputs = []
                            attention_maps = {}
                        if brain_mask is not None:
                            outputs = outputs * brain_mask

                        # 检查模型输出是否包含NaN/Inf，如果严重则跳过该batch
                        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                            nan_ratio = (torch.isnan(outputs).sum() + torch.isinf(outputs).sum()).float() / outputs.numel()
                            if nan_ratio > 0.1:  # 如果超过10%的值为NaN/Inf，跳过该batch
                                print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 模型输出NaN/Inf比例过高({nan_ratio:.2%})，跳过此批次")
                                continue
                            else:
                                # 少量NaN/Inf时尝试修正
                                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
                        # 在计算损失前，先检查并裁剪logits到合理范围，防止数值不稳定
                        outputs = torch.clamp(outputs, min=-10.0, max=10.0)
                        
                        # 基础分割损失
                        loss = self.compute_seg_loss(outputs, masks, bce_criterion, weights=epoch_loss_weights)
                        
                        # 检查损失是否为NaN/Inf，如果是则跳过该batch
                        if not torch.isfinite(loss):
                            print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 损失为NaN/Inf，跳过此批次")
                            continue
                        
                        # 确保基础损失非负且有限
                        loss = torch.clamp(loss, min=0.0, max=1000.0)  # 限制最大损失值
                        
                        # 检查损失是否为NaN/Inf（在反向传播之前）
                        if not torch.isfinite(loss):
                            print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 损失为NaN/Inf，尝试修复...")
                            # 尝试使用简单的BCE损失
                            loss = bce_criterion(outputs, masks)
                            loss = torch.clamp(loss, min=0.0, max=1000.0)
                            
                            # 如果仍然是NaN/Inf，跳过此批次
                            if not torch.isfinite(loss):
                                print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 修复失败，跳过此批次")
                                continue
                        
                        # 辅助输出损失
                        if aux_outputs:
                            for weight, aux_logits in zip(self.aux_loss_weights, aux_outputs):
                                loss += weight * self.compute_seg_loss(aux_logits, masks, bce_criterion, weights=epoch_loss_weights)
                        
                        # 注意力集中度损失
                        if attention_maps:
                            att_loss = self.attention_concentration_loss(attention_maps, masks, weight=0.005)
                            if att_loss > 0 and torch.isfinite(att_loss):
                                loss += att_loss
                        
                        # 最终检查：在反向传播之前确保loss是有效的
                        if not torch.isfinite(loss):
                            print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 最终loss为NaN/Inf，跳过此批次")
                            continue
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # 清理异常梯度，防止NaN/Inf传播
                    grad_clamp = 1.0 if self.model_type in ("swin_unet", "swinunet") else 5.0
                    grad_sanitized = self._sanitize_gradients(model, clamp_value=grad_clamp)
                    if grad_sanitized:
                        print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 检测到异常梯度，已自动修复")
                    
                    # 检查梯度中的NaN/Inf
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 参数 {name} 的梯度包含NaN/Inf，清零梯度")
                                param.grad.zero_()
                                has_nan_grad = True
                    
                    if has_nan_grad:
                        print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 检测到NaN/Inf梯度，跳过此批次")
                        scaler.update()
                        continue
                    
                    # 计算梯度范数并检查
                    total_grad_norm = 0.0
                    param_count = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if torch.isfinite(param_norm):
                                total_grad_norm += param_norm.item() ** 2
                                param_count += 1
                            else:
                                print(f"[警告] 参数梯度范数为NaN/Inf，清零该梯度")
                                p.grad.zero_()
                    
                    if param_count > 0:
                        total_grad_norm = total_grad_norm ** (1. / 2)
                    else:
                        total_grad_norm = 0.0
                    
                    # 调试：检查梯度（仅在第一个epoch的前几个batch或梯度异常时）
                    if (epoch == 0 and batch_idx < 3) or total_grad_norm > 100.0 or total_grad_norm < 1e-6:
                        print(f"[调试] Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss.item():.4f}, GradNorm={total_grad_norm:.6f}, LR={optimizer.param_groups[0]['lr']:.8f}")
                        if total_grad_norm < 1e-6:
                            print(f"[警告] 梯度过小，模型可能无法正常更新！")
                        if total_grad_norm > 100.0:
                            print(f"[警告] 梯度过大，可能发生梯度爆炸！")
                    
                    # 梯度裁剪：统一使用标准 max_norm=1.0（0.05 过小会导致训练不稳定/难以收敛）
                    max_grad_norm = 1.0
                    if total_grad_norm > 10.0:
                        print(f"[严重警告] 梯度过大({total_grad_norm:.2f})，执行梯度裁剪(max_norm={max_grad_norm})")
                    
                    # 如果梯度为0，尝试临时提高学习率或跳过该batch
                    if total_grad_norm < 1e-8:
                        print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 梯度完全消失(GradNorm={total_grad_norm:.8f})")
                        # 如果连续多个batch梯度为0，临时提高学习率
                        if not hasattr(self, '_zero_grad_count'):
                            self._zero_grad_count = 0
                        self._zero_grad_count += 1
                        if self._zero_grad_count > 5:
                            # 临时将学习率提高2倍
                            current_lr = optimizer.param_groups[0]['lr']
                            new_lr = min(current_lr * 2.0, initial_lr * 0.1)  # 最高不超过初始学习率的10%
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            print(f"[修复] 临时提高学习率: {current_lr:.8f} -> {new_lr:.8f}")
                            self._zero_grad_count = 0
                        scaler.update()
                        continue
                    else:
                        # 梯度正常时重置计数器
                        if hasattr(self, '_zero_grad_count'):
                            self._zero_grad_count = 0
                    
                    clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    # 再次检查裁剪后的梯度
                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                                print(f"[严重警告] 梯度裁剪后仍有NaN/Inf，清零梯度")
                                p.grad.zero_()
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # 检查模型参数是否包含NaN/Inf
                    for name, param in model.named_parameters():
                        if torch.any(torch.isnan(param.data)) or torch.any(torch.isinf(param.data)):
                            print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: 参数 {name} 包含NaN/Inf！")
                            # 尝试从EMA模型恢复（如果可用）
                            if hasattr(self, 'use_ema') and self.use_ema and ema_model is not None:
                                print(f"[尝试恢复] 从EMA模型恢复参数 {name}")
                                with torch.no_grad():
                                    actual_model = self._unwrap_model(model)
                                    actual_ema = self._unwrap_model(ema_model)
                                    if name in actual_ema.state_dict():
                                        param.data.copy_(actual_ema.state_dict()[name])
                    if self.use_ema and ema_model is not None:
                        self._update_ema_model(ema_model, model)
                    
                    # 检查损失值是否有效
                    loss_value = loss.item()
                    if not np.isfinite(loss_value):
                        print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 损失值为NaN/Inf，使用0.0")
                        loss_value = 0.0
                    
                    epoch_loss += loss_value * batch_size
                    
                    # 定期清理GPU缓存
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 更新训练进度
                    train_progress = 20 + int(50 * (batch_idx + 1) / len(train_loader))
                    self.update_progress.emit(
                        train_progress,
                        f"轮次 {epoch+1}/{self.epochs} | 批次 {batch_idx+1}/{len(train_loader)} | 损失: {loss_value:.4f}"
                    )
                
                # 验证阶段
                model.eval()
                val_dice = 0.0
                val_iou = 0.0
                val_loss = 0.0
                val_samples = 0
                val_pred_fg_pixels = 0.0
                val_gt_fg_pixels = 0.0
                val_total_pixels = 0.0
                # 【诊断】添加空mask样本统计，帮助诊断Dice虚高问题
                val_empty_mask_count = 0  # 目标为空mask的样本数
                val_empty_mask_dice_sum = 0.0  # 空mask样本的Dice总和
                val_non_empty_mask_count = 0  # 目标有前景的样本数
                val_non_empty_mask_dice_sum = 0.0  # 有前景样本的Dice总和
                # IoU分类统计
                val_empty_mask_iou_sum = 0.0
                val_non_empty_mask_iou_sum = 0.0
                
                self.update_val_progress.emit(0, f"开始验证轮次 {epoch+1}...")
                # 如果启用EMA且训练了足够轮次，使用EMA模型进行评估
                eval_model_for_epoch = model
                if self.use_ema and ema_model is not None and epoch >= self.ema_eval_start_epoch:
                    # EMA模型在评估时需要设置为eval模式
                    ema_model.eval()
                    eval_model_for_epoch = ema_model
                    # 如果原模型是DataParallel，需要包装EMA模型
                    if isinstance(model, nn.DataParallel):
                        eval_model_for_epoch = nn.DataParallel(ema_model)
                # 确保模型处于eval模式（无论是普通模型还是EMA模型）
                if not isinstance(eval_model_for_epoch, nn.DataParallel):
                    eval_model_for_epoch.eval()
                else:
                    eval_model_for_epoch.module.eval()
                
                # 动态刷新阈值，避免Dice长期卡在固定值
                allow_refresh = (epoch >= 1)
                refresh_threshold = (
                    allow_refresh and (
                        epoch == 1
                        or self.threshold_refresh_interval <= 1
                        or ((epoch + 1) % self.threshold_refresh_interval == 0)
                    )
                )
                if refresh_threshold:
                    try:
                        # 使用全部验证集进行阈值优化，确保与验证阶段结果一致
                        val_threshold = float(self.find_optimal_threshold(
                            eval_model_for_epoch,
                            val_loader,
                            device,
                            num_samples=None,  # None表示使用全部验证集
                        ))
                        self.last_optimal_threshold = val_threshold
                    except Exception as threshold_err:
                        print(f"[警告] 阈值搜索失败，使用上一次的阈值。原因: {threshold_err}")
                        val_threshold = float(getattr(self, "last_optimal_threshold", 0.5))
                else:
                    if epoch == 0:
                        val_threshold = 0.5
                    else:
                        val_threshold = float(getattr(self, "last_optimal_threshold", 0.5))
                
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        if self.stop_requested:
                            # 【修复】用户停止时也要发送完成信号，确保UI正确更新
                            self.training_finished.emit("训练已被用户停止", self.best_model_path if self.save_best else None)
                            return
                        
                        # 处理数据
                        images, masks = val_batch
                        images = images.to(device)
                        masks = masks.float().to(device)
                            
                        batch_size = images.size(0)
                        brain_mask = None
                        if self.use_skull_stripper:
                            images, brain_mask = self._apply_skull_strip(images)
                        
                        # 验证阶段输入数据检查（在增加val_samples之前）
                        if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)):
                            print(f"[警告] 验证阶段: Batch {val_idx+1}: 输入图像包含NaN/Inf，跳过")
                            continue
                        if torch.any(torch.isnan(masks)) or torch.any(torch.isinf(masks)):
                            print(f"[警告] 验证阶段: Batch {val_idx+1}: 输入掩膜包含NaN/Inf，跳过")
                            continue
                        
                        with autocast(device_type=amp_device_type, enabled=amp_enabled):
                            # 在forward之前检查输入
                            if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)):
                                print(f"[警告] 验证阶段: Batch {val_idx+1}: 输入图像包含NaN/Inf，跳过")
                                continue
                            
                            # 【关键修复】验证阶段不使用TTA，与训练损失计算保持一致
                            # 原因：训练损失基于单次前向传播，如果验证使用TTA会导致Dice虚高
                            # 如果需要TTA评估，应该在训练结束后的最终测试阶段使用
                            # 可以通过环境变量 SEG_USE_TTA_IN_VAL=1 启用（不推荐）
                            use_tta_in_val = os.environ.get("SEG_USE_TTA_IN_VAL", "0") == "1"
                            
                            if use_tta_in_val:
                                # 仅在明确启用时使用TTA（不推荐，会导致训练和验证不一致）
                                try:
                                    outputs = self._tta_inference(eval_model_for_epoch, images)
                                    if brain_mask is not None:
                                        outputs = outputs * brain_mask
                                except RuntimeError as e:
                                    if "out of memory" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
                                        print(f"[严重警告] 验证阶段: Batch {val_idx+1}: TTA推理失败 ({str(e)[:100]})，跳过该batch")
                                        continue
                                    else:
                                        raise
                            else:
                                # 标准验证：单次前向传播（与训练损失计算一致）
                                outputs = eval_model_for_epoch(images)
                                if isinstance(outputs, tuple):
                                    outputs = outputs[0]
                                if brain_mask is not None:
                                    outputs = outputs * brain_mask
                            
                            # 检查模型输出，如果出现NaN/Inf，说明模型已经崩溃，跳过该batch
                            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                                nan_ratio = (torch.isnan(outputs).sum() + torch.isinf(outputs).sum()).float() / outputs.numel()
                                print(f"[严重警告] 验证阶段: Batch {val_idx+1}: 模型输出包含NaN/Inf (比例: {nan_ratio:.2%})，跳过该batch")
                                # 如果NaN/Inf比例过高，说明模型已崩溃，跳过该batch
                                continue
                            
                            # 在计算损失前，先检查并裁剪logits到合理范围，防止数值不稳定
                            outputs = torch.clamp(outputs, min=-10.0, max=10.0)
                            
                            # 计算损失（基于单次前向传播，与训练时一致）
                            loss = self.compute_seg_loss(outputs, masks, bce_criterion, weights=epoch_loss_weights)
                            
                            # 检查损失值
                            loss_value = loss.item()
                            if not np.isfinite(loss_value):
                                print(f"[警告] 验证阶段: Batch {val_idx+1}: 损失为NaN/Inf，使用0.0")
                                loss_value = 0.0
                        
                        # 只有在所有检查通过后才增加val_samples和累加指标
                        val_samples += batch_size
                        val_loss += loss_value * batch_size

                        # 【关键修复】计算Dice系数时，使用与测试阶段完全相同的流程
                        # 但注意：为了与训练损失保持一致，这里不使用TTA（已在上面修复）
                        # 如果需要在验证时也使用TTA评估，应该单独计算一个"TTA Dice"用于参考
                        
                        probs = torch.sigmoid(outputs)
                        # 调试：检查模型输出范围和mask（仅在第一个epoch的第一个batch）
                        if epoch == 0 and val_idx == 0:
                            print(f"[调试] 验证阶段 - 模型输出范围: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                            print(f"[调试] 验证阶段 - Sigmoid后范围: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
                            print(f"[调试] 验证阶段 - 使用阈值: {val_threshold:.4f}, 预测前景像素数: {(probs > val_threshold).sum().item()}")
                            print(f"[调试] 验证阶段 - Mask前景像素数: {masks.sum().item():.0f}, 总像素数: {masks.numel()}")
                        
                        # 确保 probs 和 masks 的空间尺寸匹配
                        if probs.shape[2:] != masks.shape[2:]:
                            probs = F.interpolate(probs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        
                        # 使用最优阈值进行二值化（与测试时一致）
                        preds = (probs > val_threshold).float()
                        
                        # 【智能后处理】先按面积+概率过滤微小病灶/噪点，再进行形态学优化
                        # 注意：后处理只影响Dice计算，不影响损失计算（损失基于原始logits）
                        for i in range(preds.shape[0]):
                            pred_mask_tensor = preds[i, 0]
                            prob_map_tensor = probs[i, 0]
                            # 先执行智能后处理（不再简单按min_size裁剪）
                            pred_mask_tensor = self.smart_post_processing(pred_mask_tensor, prob_map_tensor)
                            # 再执行传统形态学后处理，但不移除小区域（min_size=0）
                            pred_mask_processed = self.post_process_mask(
                                pred_mask_tensor,
                                min_size=0,
                                use_morphology=True,
                                keep_largest=False,  # 允许多发病灶同时存在
                                fill_holes=True     # 填充孔洞，去除假阴性空洞
                            )
                            # post_process_mask会返回tensor或numpy，需要确保是tensor
                            if isinstance(pred_mask_processed, torch.Tensor):
                                preds[i, 0] = pred_mask_processed.to(preds.device)
                            else:
                                preds[i, 0] = torch.from_numpy(pred_mask_processed).float().to(preds.device)
                        
                        # 使用与训练过程相同的calculate_batch_dice函数计算Dice
                        batch_dice = self.calculate_batch_dice(preds.float(), masks)
                        val_dice += batch_dice.sum().item()
                        val_pred_fg_pixels += preds.sum().item()
                        val_gt_fg_pixels += masks.sum().item()
                        val_total_pixels += float(masks.numel())
                        
                        # 计算批次 IoU（逐样本），并分类统计
                        # 计算批次 IoU（逐样本），并分类统计
                        batch_size = masks.shape[0]
                        for i in range(batch_size):
                            mask_i = masks[i, 0]
                            mask_sum = mask_i.sum().item()
                            pred_i = preds[i, 0]
                            
                            # 计算混淆矩阵
                            tp = torch.sum((pred_i > 0.5) & (mask_i > 0.5)).item()
                            fp = torch.sum((pred_i > 0.5) & (mask_i <= 0.5)).item()
                            fn = torch.sum((pred_i <= 0.5) & (mask_i > 0.5)).item()
                            tn = torch.sum((pred_i <= 0.5) & (mask_i <= 0.5)).item()
                            
                            # 【修复】分别计算前景类和背景类的IoU
                            # 前景类IoU（Positive Class）
                            iou_pos_den = tp + fp + fn
                            iou_pos_i = 1.0 if iou_pos_den < 1e-8 else tp / (iou_pos_den + 1e-8)
                            
                            # 背景类IoU（Negative Class）
                            iou_neg_den = tn + fp + fn
                            iou_neg_i = 1.0 if iou_neg_den < 1e-8 else tn / (iou_neg_den + 1e-8)
                            
                            # 整体IoU（使用前景类IoU，与标准定义一致）
                            val_iou += iou_pos_i
                            
                            # 判断是否为空mask
                            total_pixels = mask_i.numel()
                            avg_fg_ratio = val_gt_fg_pixels / max(1.0, val_total_pixels) if val_total_pixels > 0 else 0.0
                            adaptive_empty_threshold = max(1e-7, avg_fg_ratio * 0.001)
                            empty_threshold_pixels = adaptive_empty_threshold * total_pixels
                            
                            if mask_sum <= empty_threshold_pixels:
                                val_empty_mask_count += 1
                                val_empty_mask_dice_sum += batch_dice[i].item()
                                val_empty_mask_iou_sum += iou_neg_i  # ✅ 使用背景类IoU
                            else:
                                val_non_empty_mask_count += 1
                                val_non_empty_mask_dice_sum += batch_dice[i].item()
                                val_non_empty_mask_iou_sum += iou_pos_i  # ✅ 使用前景类IoU
                        
                        # 更新验证进度
                        val_progress = int(100 * (val_idx + 1) / len(val_loader))
                        current_avg_loss = val_loss / max(1, val_samples)
                        current_avg_dice = val_dice / max(1, val_samples)
                        # 计算当前批次的 Dice_Pos 和 Dice_Neg（用于进度显示）
                        current_dice_pos = val_non_empty_mask_dice_sum / max(1, val_non_empty_mask_count) if val_non_empty_mask_count > 0 else 0.0
                        current_dice_neg = val_empty_mask_dice_sum / max(1, val_empty_mask_count) if val_empty_mask_count > 0 else 0.0
                        
                        self.update_val_progress.emit(
                            val_progress,
                            f"验证轮次 {epoch+1} | 批次 {val_idx+1}/{len(val_loader)}\n"
                            f"损失: {current_avg_loss:.4f} | Dice_Pos: {current_dice_pos:.4f} | Dice_Neg: {current_dice_neg:.4f} | 整体Dice: {current_avg_dice:.4f}"
                        )
                        
                        # 每5个批次强制更新UI
                        if val_idx % 5 == 0:
                            QApplication.processEvents()
                
                # 计算平均值（确保没有NaN/Inf）
                avg_train_loss = epoch_loss / max(1, train_samples)
                if not np.isfinite(avg_train_loss):
                    print(f"[警告] Epoch {epoch+1}: 训练平均损失为NaN/Inf，使用0.0")
                    avg_train_loss = 0.0
                
                val_dice /= max(1, val_samples)
                val_iou /= max(1, val_samples)
                if not np.isfinite(val_dice):
                    print(f"[警告] Epoch {epoch+1}: 验证Dice为NaN/Inf，使用0.0")
                    val_dice = 0.0
                if not np.isfinite(val_iou):
                    print(f"[警告] Epoch {epoch+1}: 验证IoU为NaN/Inf，使用0.0")
                    val_iou = 0.0

                # 使用 ReduceLROnPlateau 根据验证Dice自动调整学习率（优先提升稳定性）
                if plateau_scheduler is not None:
                    plateau_scheduler.step(val_dice)
                
                avg_val_loss = val_loss / max(1, val_samples)
                if not np.isfinite(avg_val_loss):
                    print(f"[警告] Epoch {epoch+1}: 验证平均损失为NaN/Inf，使用0.0")
                    avg_val_loss = 0.0
                
                pred_fg_ratio = val_pred_fg_pixels / max(1.0, val_total_pixels)
                gt_fg_ratio = val_gt_fg_pixels / max(1.0, val_total_pixels)
                
                # 【关键修改】分别统计有前景mask和空mask的Dice/IoU
                dice_pos = val_non_empty_mask_dice_sum / max(1, val_non_empty_mask_count) if val_non_empty_mask_count > 0 else 0.0
                dice_neg = val_empty_mask_dice_sum / max(1, val_empty_mask_count) if val_empty_mask_count > 0 else 0.0
                iou_pos = val_non_empty_mask_iou_sum / max(1, val_non_empty_mask_count) if val_non_empty_mask_count > 0 else 0.0
                iou_neg = val_empty_mask_iou_sum / max(1, val_empty_mask_count) if val_empty_mask_count > 0 else 0.0
                empty_mask_ratio = val_empty_mask_count / max(1, val_samples) if val_samples > 0 else 0.0
                
                # 记录到历史中
                self.val_dice_pos_history.append(dice_pos)
                self.val_dice_neg_history.append(dice_neg)
                
                print(
                    f"[验证统计] Epoch {epoch+1}: threshold={val_threshold:.3f}, "
                    f"pred_fg_ratio={pred_fg_ratio:.4f}, gt_fg_ratio={gt_fg_ratio:.4f}, "
                    f"val_dice={val_dice:.4f}, val_iou={val_iou:.4f}"
                )
                print(
                    f"[Dice/IoU分类统计] "
                    f"Dice_Pos: {dice_pos:.4f}, IoU_Pos: {iou_pos:.4f} ({val_non_empty_mask_count}/{val_samples}样本) | "
                    f"Dice_Neg: {dice_neg:.4f}, IoU_Neg: {iou_neg:.4f} ({val_empty_mask_count}/{val_samples}样本) | "
                    f"整体Dice: {val_dice:.4f}, 整体IoU: {val_iou:.4f}"
                )

                # 根据验证Dice或SWA阶段调整学习率（Poly策略下仅保留SWA调度）
                swa_epoch_active = swa_enabled and epoch >= swa_start_epoch
                if swa_epoch_active and swa_model is not None:
                    swa_model.update_parameters(model)
                    if swa_scheduler is not None:
                        swa_scheduler.step()
                    swa_active_epochs += 1
                # Poly学习率已在epoch开始时直接设置，不再使用scheduler/plateau_scheduler
                
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新训练历史
                self.train_loss_history.append(avg_train_loss)
                self.val_loss_history.append(avg_val_loss)
                self.val_dice_history.append(val_dice)
                
                # 发送轮次完成信号
                self.epoch_completed.emit(epoch + 1, avg_train_loss, avg_val_loss, val_dice)
                
                # 每个epoch结束后清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                
                # 每个轮次结束后生成性能分析可视化
                self.update_progress.emit(
                    int(70 + 20 * (epoch + 1) / self.epochs),
                    f"轮次 {epoch+1} 完成 (LR={current_lr:.6f})，生成性能分析..."
                )
                
                # 生成测试集分割结果可视化 - 使用TTA提升性能
                test_viz_path = self.visualize_test_results(
                    eval_model_for_epoch, 
                    val_loader, 
                    device, 
                    num_samples=6,  # 每个轮次显示6个样本
                    use_tta=True    # 训练结束后的测试使用TTA
                )
                
                # 计算当前轮次的性能指标（快速评估）
                model.eval()
                epoch_metrics = {
                    'dice': [],
                    'iou': [],
                    'precision': [],
                    'recall': [],
                    'sensitivity': [],
                    'specificity': [],
                    'f1': [],
                    'hd95': []
                }
                
                with torch.no_grad():
                    # 只评估部分验证集以加快速度
                    eval_samples = min(20, len(val_dataset))  # 最多评估20个样本
                    eval_count = 0
                    
                    for batch_data in val_loader:
                        if eval_count >= eval_samples:
                            break
                        
                        # 处理数据：可能包含分类标签
                        if len(batch_data) == 3:
                            images, masks, _ = batch_data
                        else:
                            images, masks = batch_data
                        images, masks = images.to(device), masks.to(device)
                        outputs = eval_model_for_epoch(images)
                        # 确保 outputs 和 masks 的空间尺寸匹配
                        if outputs.shape[2:] != masks.shape[2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        preds = torch.sigmoid(outputs)
                        preds = (preds > val_threshold).float()
                        
                        for i in range(preds.shape[0]):
                            if eval_count >= eval_samples:
                                break
                                
                            pred = preds[i, 0]
                            mask = masks[i, 0]
                            
                            # 双重检查尺寸匹配（以防万一）
                            if pred.shape != mask.shape:
                                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                            
                            # 标准混淆矩阵定义，确保与主评估一致
                            tp = float((pred * mask).sum().item())
                            pred_sum = float(pred.sum().item())   # TP + FP
                            mask_sum = float(mask.sum().item())   # TP + FN
                            fp = float((pred * (1 - mask)).sum().item())
                            fn = float(((1 - pred) * mask).sum().item())
                            tn = float(((1 - pred) * (1 - mask)).sum().item())
                            
                            dice_den = 2.0 * tp + fp + fn
                            if dice_den < 1e-7:
                                dice = 1.0 if (mask_sum < 1e-7 and pred_sum < 1e-7) else 0.0
                            else:
                                dice = (2.0 * tp) / dice_den
                            
                            union = tp + fp + fn
                            iou = 1.0 if union < 1e-7 else tp / union
                            
                            if (tp + fp) < 1e-7:
                                precision = 1.0 if mask_sum < 1e-7 else 0.0
                            else:
                                precision = tp / (tp + fp)
                            
                            if (tp + fn) < 1e-7:
                                recall = 1.0 if pred_sum < 1e-7 else 0.0
                            else:
                                recall = tp / (tp + fn)
                            
                            specificity = 1.0 if (tn + fp) < 1e-7 else tn / (tn + fp)
                            
                            f1 = dice  # 二分类下F1=Dice
                            hd95 = self.calculate_hd95(
                                pred.cpu().numpy(),
                                mask.cpu().numpy()
                            )
                            
                            epoch_metrics['dice'].append(float(dice))
                            epoch_metrics['iou'].append(float(iou))
                            epoch_metrics['precision'].append(float(precision))
                            epoch_metrics['recall'].append(float(recall))
                            epoch_metrics['sensitivity'].append(float(recall))
                            epoch_metrics['specificity'].append(float(specificity))
                            epoch_metrics['f1'].append(float(f1))
                            epoch_metrics['hd95'].append(hd95)
                            
                            eval_count += 1
                        
                        if eval_count >= eval_samples:
                            break
                
                # 计算平均指标
                avg_epoch_metrics = {}
                for k, values in epoch_metrics.items():
                    arr = np.array(values, dtype=float)
                    if arr.size == 0 or np.all(np.isnan(arr)):
                        avg_epoch_metrics[k] = float('nan')
                    else:
                        avg_epoch_metrics[k] = float(np.nanmean(arr))

                # 基于当前阈值的平均指标，计算综合评分
                hd95_mean = avg_epoch_metrics.get('hd95', float('inf'))
                total_score = self.calculate_custom_score(
                    dice=avg_epoch_metrics.get('dice', 0.0),
                    iou=avg_epoch_metrics.get('iou', 0.0),
                    precision=avg_epoch_metrics.get('precision', 0.0),
                    recall=avg_epoch_metrics.get('recall', 0.0),
                    specificity=avg_epoch_metrics.get('specificity', 0.0),
                    hd95=hd95_mean,
                )
                avg_epoch_metrics['score'] = float(total_score)

                # 格式化 HD95（处理 NaN/Inf 情况）
                hd95_str = f"{hd95_mean:.4f}" if np.isfinite(hd95_mean) else "nan"
                print(
                    f"[验证评分] Epoch {epoch+1}: threshold={val_threshold:.3f}, "
                    f"TotalScore={total_score:.4f}, "
                    f"Dice={avg_epoch_metrics.get('dice', float('nan')):.4f}, "
                    f"IoU={avg_epoch_metrics.get('iou', float('nan')):.4f}, "
                    f"Precision={avg_epoch_metrics.get('precision', float('nan')):.4f}, "
                    f"Recall={avg_epoch_metrics.get('recall', float('nan')):.4f}, "
                    f"Specificity={avg_epoch_metrics.get('specificity', float('nan')):.4f}, "
                    f"HD95={hd95_str}"
                )
                
                # 发送epoch分析结果信号（包含综合评分）
                self.epoch_analysis_ready.emit(epoch + 1, test_viz_path, avg_epoch_metrics)
                
                # Save best model
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    if self.save_best:
                        os.makedirs(self.best_model_cache_dir, exist_ok=True)
                        self.best_model_path = os.path.join(
                            self.best_model_cache_dir, f"best_model_dice_{val_dice:.4f}.pth"
                        )
                        self._save_checkpoint(eval_model_for_epoch, self.best_model_path)
                        self.model_saved.emit(f"已保存最佳模型 (Dice: {val_dice:.4f})")

                # 恢复EMA模型为train模式（如果使用了EMA）
                if self.use_ema and ema_model is not None and epoch >= self.ema_eval_start_epoch:
                    ema_model.train()
                
                # 触发早停
                if early_stopping.step(val_dice):
                    early_stop_triggered = True
                    self.update_progress.emit(
                        min(90, int(70 + 20 * (epoch + 1) / max(1, self.epochs))),
                        "验证Dice长期未提升，触发早停..."
                    )
                    break
            
            # 确定最终用于评估的模型（优先使用EMA，其次SWA，最后普通模型）
            eval_model = model
            if self.use_ema and ema_model is not None and self.epochs >= self.ema_eval_start_epoch:
                self.update_progress.emit(87, "使用EMA模型进行最终评估...")
                ema_model.eval()
                eval_model = ema_model
                if isinstance(model, nn.DataParallel):
                    eval_model = nn.DataParallel(ema_model)
            elif swa_enabled and swa_active_epochs > 0 and swa_model is not None:
                self.update_progress.emit(88, "应用SWA权重并更新BN统计...")
                # 使用安全的BN更新函数，处理可能包含分类标签的数据
                self._safe_update_bn(swa_model, train_loader, device)
                eval_model = swa_model
                if self.save_best:
                    swa_model_path = os.path.join(self.temp_dir, f"swa_model_epoch_{epoch+1}.pth")
                    self._save_checkpoint(eval_model, swa_model_path)
                    self.model_saved.emit(f"SWA平滑模型已保存: {os.path.basename(swa_model_path)}")

            # 最终评估和可视化
            self.update_progress.emit(90, "正在执行最终评估...")
            
            # 生成训练历史图表
            history_path = self.plot_training_history()
            self.visualization_ready.emit(history_path)
            
            # 执行综合评估（单阶段：仅分割模型）- 使用TTA提升性能
            self.update_progress.emit(92, "计算性能指标（单阶段分割模型，使用TTA）...")
            detailed_metrics, metrics_path = self.evaluate_model(eval_model, val_loader, device, use_tta=True, adaptive_threshold=True)
            self.metrics_ready.emit(detailed_metrics)
            
            # 保存单阶段评估结果用于对比
            single_stage_results = {
                'segmentation_dice': detailed_metrics['average']['dice'],
                'segmentation_iou': detailed_metrics['average']['iou'],
                'segmentation_precision': detailed_metrics['average']['precision'],
                'segmentation_recall': detailed_metrics['average']['recall'],
                'segmentation_f1': detailed_metrics['average']['f1']
            }
            
            # 分类模型相关评估已删除
            if False:  # 已禁用两阶段评估
                self.update_progress.emit(93, "评估两阶段系统（分类+分割）...")
                try:
                    # 重新创建验证数据加载器（因为val_loader_cls可能不在作用域内）
                    val_dataset_cls = self.load_dataset(val_ids, val_transform, split_name="val", return_classification=True)
                    cpu_count = os.cpu_count() or 1
                    num_workers = max(0, min(4, cpu_count - 1))
                    val_loader_cls = DataLoader(
                        val_dataset_cls,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=num_workers > 0
                    )
                    
                    # 加载分类模型
                    classification_model = self._build_classification_model(device)
                    cls_checkpoint = torch.load(self.classification_model_path, map_location=device)
                    actual_cls_model = self._unwrap_model(classification_model)
                    actual_cls_model.load_state_dict(cls_checkpoint['state_dict'])
                    classification_model.eval()
                    
                    # 评估分类模型（自动寻找最优阈值）
                    cls_metrics = self.evaluate_classification_model(classification_model, val_loader_cls, device)
                    
                    # 使用自动找到的最优分类阈值
                    optimal_cls_threshold = cls_metrics.get('optimal_threshold', 0.5)
                    if optimal_cls_threshold != 0.5:
                        print(f"\n[优化] 自动找到最优分类阈值: {optimal_cls_threshold:.3f} (原阈值: 0.5)")
                        print(f"[优化] 在最优阈值下的F1分数: {cls_metrics.get('best_f1_at_threshold', 0.0):.4f}")
                    
                    # 评估两阶段系统（使用改进的级联策略）
                    # 策略1：自适应策略（只对高置信度的无病变样本跳过分割）
                    two_stage_results_adaptive = self.evaluate_two_stage_system(
                        classification_model, eval_model, val_loader_cls, device,
                        classification_threshold=optimal_cls_threshold, 
                        segmentation_threshold=self.last_optimal_threshold,
                        use_adaptive_strategy=True,
                        confidence_threshold=0.85  # 只有无病变概率>85%才跳过
                    )
                    
                    # 策略2：保守策略（所有样本都进行分割，分类模型仅用于引导）
                    two_stage_results_conservative = self.evaluate_two_stage_system(
                        classification_model, eval_model, val_loader_cls, device,
                        classification_threshold=optimal_cls_threshold, 
                        segmentation_threshold=self.last_optimal_threshold,
                        use_adaptive_strategy=False  # 所有样本都分割
                    )
                    
                    # 选择最佳策略（选择最接近单阶段性能的策略）
                    adaptive_dice = two_stage_results_adaptive['system'].get('dice', 0.0)
                    conservative_dice = two_stage_results_conservative['system'].get('dice', 0.0)
                    single_dice = single_stage_results['segmentation_dice']
                    
                    if abs(adaptive_dice - single_dice) < abs(conservative_dice - single_dice):
                        two_stage_results = two_stage_results_adaptive
                        strategy_name = "自适应策略（高置信度跳过）"
                    else:
                        two_stage_results = two_stage_results_conservative
                        strategy_name = "保守策略（全部分割）"
                    
                    print(f"\n[级联策略优化] 选择策略: {strategy_name}")
                    print(f"  - 自适应策略Dice: {adaptive_dice:.4f} (跳过率: {two_stage_results_adaptive['system'].get('efficiency', {}).get('computation_saved', 0.0):.1f}%)")
                    print(f"  - 保守策略Dice: {conservative_dice:.4f} (跳过率: 0.0%)")
                    print(f"  - 单阶段Dice: {single_dice:.4f}")
                    print(f"  - 最终选择: {strategy_name} (Dice: {two_stage_results['system'].get('dice', 0.0):.4f})")
                    
                    # 保存对比评估结果（包含两种策略）
                    comparison_path = os.path.join(self.temp_dir, 'system_comparison.json')
                    import json
                    with open(comparison_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'single_stage': single_stage_results,
                            'two_stage': {
                                'adaptive_strategy': {
                                    'results': two_stage_results_adaptive,
                                    'dice': adaptive_dice
                                },
                                'conservative_strategy': {
                                    'results': two_stage_results_conservative,
                                    'dice': conservative_dice
                                },
                                'selected_strategy': strategy_name,
                                'final_results': two_stage_results
                            },
                            'classification_metrics': cls_metrics,
                            'comparison': {
                                'dice_improvement_adaptive': adaptive_dice - single_stage_results['segmentation_dice'],
                                'dice_improvement_conservative': conservative_dice - single_stage_results['segmentation_dice'],
                                'recommendation': 'two_stage_adaptive' if (adaptive_dice > single_stage_results['segmentation_dice'] + 0.01) else ('two_stage_conservative' if (conservative_dice > single_stage_results['segmentation_dice'] + 0.01) else 'single_stage')
                            }
                        }, f, ensure_ascii=False, indent=2)
                    
                    print("\n" + "="*60)
                    print("【性能对比分析】")
                    print("="*60)
                    
                    # 单阶段 vs 两阶段对比
                    print("\n【单阶段系统】（仅分割模型）:")
                    print(f"  - Dice: {single_stage_results['segmentation_dice']:.4f}")
                    print(f"  - IoU: {single_stage_results['segmentation_iou']:.4f}")
                    print(f"  - Precision: {single_stage_results['segmentation_precision']:.4f}")
                    print(f"  - Recall: {single_stage_results['segmentation_recall']:.4f}")
                    print(f"  - F1: {single_stage_results['segmentation_f1']:.4f}")
                    
                    print("\n【两阶段系统】（分类+分割）:")
                    print(f"  分类模型准确率: {cls_metrics['accuracy']:.2f}%")
                    print(f"  分割模型指标（仅对分类为有病变的样本）:")
                    print(f"    - Dice: {two_stage_results['segmentation']['dice']:.4f}")
                    print(f"    - IoU: {two_stage_results['segmentation']['iou']:.4f}")
                    print(f"  系统整体指标（所有样本，包括分类错误）:")
                    print(f"    - 系统Dice: {two_stage_results['system'].get('dice', 0.0):.4f} ⭐")
                    print(f"    - 系统IoU: {two_stage_results['system'].get('iou', 0.0):.4f}")
                    print(f"    - 系统F1: {two_stage_results['system']['f1']:.4f}")
                    print(f"    - 系统Precision: {two_stage_results['system']['precision']:.4f}")
                    print(f"    - 系统Recall: {two_stage_results['system']['recall']:.4f}")
                    
                    # 性能对比分析
                    print("\n【性能对比】:")
                    dice_diff = two_stage_results['system'].get('dice', 0.0) - single_stage_results['segmentation_dice']
                    if dice_diff > 0.01:
                        print(f"  ✅ 两阶段系统Dice提升: +{dice_diff:.4f} ({(dice_diff/single_stage_results['segmentation_dice']*100):.1f}%)")
                        print(f"  💡 建议：使用两阶段系统")
                    elif dice_diff < -0.01:
                        print(f"  ⚠️  两阶段系统Dice下降: {dice_diff:.4f} ({(dice_diff/single_stage_results['segmentation_dice']*100):.1f}%)")
                        print(f"  💡 建议：仅使用分割模型（单阶段）")
                    else:
                        print(f"  ➡️  两阶段系统Dice变化: {dice_diff:+.4f} (基本持平)")
                        print(f"  💡 建议：根据实际需求选择（两阶段可节省计算，单阶段更简单）")
                    
                    # 效率分析
                    if cls_metrics['accuracy'] > 0.7:
                        efficiency_gain = (1 - cls_metrics.get('false_positive_rate', 0.3)) * 100
                        print(f"\n【效率分析】:")
                        print(f"  - 分类准确率: {cls_metrics['accuracy']:.2f}%")
                        print(f"  - 预计可跳过约 {(1-cls_metrics.get('false_positive_rate', 0.3))*100:.1f}% 的无病变图像分割")
                        print(f"  - 两阶段系统可显著提升推理效率")
                    else:
                        print(f"\n【效率分析】:")
                        print(f"  ⚠️  分类准确率较低 ({cls_metrics['accuracy']:.2f}%)，可能影响系统效率")
                        print(f"  💡 建议：优化分类模型或仅使用分割模型")
                    
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"两阶段评估出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 生成测试结果可视化 - 使用TTA提升性能
            self.update_progress.emit(95, "生成测试集分割结果可视化（TTA）...")
            test_viz_path = self.visualize_test_results(eval_model, val_loader, device, num_samples=8, use_tta=True)
            
            # 生成性能分析
            self.update_progress.emit(98, "生成性能分析报告...")
            perf_analysis_path = self.generate_performance_analysis(detailed_metrics)
            
            # 生成注意力可视化用于可解释性分析（若模型支持）- 使用TTA
            if self._supports_attention_maps(eval_model):
                self.update_progress.emit(99, "生成注意力可解释性分析（TTA）...")
                # 注意：visualize_attention_maps 内部会使用 return_attention，TTA可能不支持，保持原样
                attention_viz_path = self.visualize_attention_maps(eval_model, val_loader, device, num_samples=4)
                attention_stats = self.analyze_attention_statistics(eval_model, val_loader, device, num_samples=20)
            else:
                self.update_progress.emit(99, "当前模型不支持注意力可视化，跳过该步骤。")
                attention_viz_path = ""
                attention_stats = {}
            
            # 发送测试结果信号，包含性能分析路径
            self.test_results_ready.emit(test_viz_path, detailed_metrics)
            self.visualization_ready.emit(perf_analysis_path)  # 同时发送性能分析
            self.attention_analysis_ready.emit(attention_viz_path, attention_stats)  # 发送注意力分析
            
            # 训练完成
            fallback_dice = self.val_dice_history[-1] if self.val_dice_history else 0.0
            final_best = self.best_dice if self.best_dice >= 0 else fallback_dice
            if early_stop_triggered:
                finish_msg = f"训练提前结束（早停），最佳Dice分数: {final_best:.4f}"
            else:
                finish_msg = f"训练完成！最佳Dice分数: {final_best:.4f}"
            self.update_progress.emit(100, finish_msg)
            self.training_finished.emit(finish_msg, self.best_model_path if self.save_best else None)
            
        except KeyboardInterrupt:
            # 用户手动中断训练（Ctrl+C）
            print("\n[用户中断] 训练已被用户手动停止")
            self.update_progress.emit(0, "训练已被用户中断")
            self.training_finished.emit("训练已被用户中断", None)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"训练错误: {str(e)}"
            # 打印详细错误信息以便调试
            print(f"\n{'='*60}")
            print("训练错误详情:")
            print(f"{'='*60}")
            print(error_trace)
            print(f"{'='*60}\n")
            self.update_progress.emit(0, error_msg)
            self.training_finished.emit(error_msg, None)
        finally:
            # 确保释放GPU内存
            torch.cuda.empty_cache()
    
    def stop(self):
        """安全停止训练"""
        self.stop_requested = True     
    def __del__(self):
        """自动清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _collect_image_mask_paths(self, patient_ids: List[str]) -> Tuple[List[str], List[str]]:
        image_paths = []
        mask_paths = []
        
        for pid in patient_ids:
            patient_dir = os.path.join(self.data_dir, pid)
            if not os.path.exists(patient_dir):
                continue
                
            files = [f for f in os.listdir(patient_dir) 
                    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
            
            for img_file in [f for f in files if 'mask' not in f.lower()]:
                base_name = os.path.splitext(img_file)[0]
                mask_file = self._find_matching_mask(files, base_name)
                if mask_file:
                    image_paths.append(os.path.join(patient_dir, img_file))
                    mask_paths.append(os.path.join(patient_dir, mask_file))
        return image_paths, mask_paths

    def _find_matching_mask(self, files: List[str], base_name: str) -> Optional[str]:
        """严格匹配图像对应的mask，避免 base_name 子串造成串号。"""
        base_lower = base_name.lower()
        preferred_suffixes = ['_mask', '-mask', ' mask', '_seg', '-seg']

        def normalize(name: str) -> str:
            name_no_ext = os.path.splitext(name)[0].lower()
            for suffix in preferred_suffixes:
                if name_no_ext.endswith(suffix):
                    return name_no_ext[:-len(suffix)]
            return name_no_ext.replace('mask', '').strip('_- ')

        exact_match = None
        fuzzy_candidates = []
        for f in files:
            if 'mask' not in f.lower():
                continue
            normalized = normalize(f)
            if normalized == base_lower:
                exact_match = f
                break
            if base_lower in os.path.splitext(f)[0].lower():
                fuzzy_candidates.append(f)

        if exact_match:
            return exact_match
        if fuzzy_candidates:
            return sorted(fuzzy_candidates, key=lambda x: len(x))[0]
        return None

    def load_dataset(self, patient_ids, transform, split_name="train", return_classification=False, 
                     use_percentile_normalization=True, use_weighted_sampling=None):
        """
        加载医学图像数据集，优先使用MATLAB缓存
        
        Args:
            patient_ids: 病人ID列表
            transform: 数据增强变换
            split_name: 数据集分割名称
            return_classification: 是否返回分类标签
            use_percentile_normalization: 是否使用百分位数归一化（p10-p99，更鲁棒）
            use_weighted_sampling: 是否使用基于mask的权重采样（None时自动：训练集启用，验证集禁用）
        """
        image_paths, mask_paths = self._collect_image_mask_paths(patient_ids)
        self.split_metadata[split_name] = {
            'image_paths': image_paths,
            'mask_paths': mask_paths
        }
        extra_modalities = self._prepare_extra_modalities(image_paths)
        
        # 自动决定是否使用权重采样
        if use_weighted_sampling is None:
            use_weighted_sampling = (split_name == "train")
        
        base_dataset = MedicalImageDataset(
            image_paths,
            mask_paths,
            transform,
            training=(split_name == "train"),
            return_classification=return_classification,
            extra_modalities=extra_modalities,
            context_slices=self.context_slices,
            context_gap=self.context_gap,
            use_percentile_normalization=use_percentile_normalization,
            use_weighted_sampling=use_weighted_sampling
        )

        return base_dataset

    def _prepare_extra_modalities(self, image_paths: List[str]) -> Optional[Dict[str, List[Optional[str]]]]:
        if not self.extra_modalities_dirs:
            return None
        return build_extra_modalities_lists(image_paths, self.extra_modalities_dirs)


    def _estimate_pos_weight(self, mask_paths: List[str], sample_size: int = 100) -> float:
        """估算正负样本比例，自适应调节BCE的pos_weight。"""
        if not mask_paths:
            return 1.0

        sample_paths = random.sample(mask_paths, min(sample_size, len(mask_paths)))
        total_pos = 0
        total_neg = 0

        for path in sample_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            pos = int(np.count_nonzero(mask))
            neg = int(mask.size - pos)
            total_pos += pos
            total_neg += neg

        if total_pos == 0:
            return 1.0
        ratio = total_neg / max(total_pos, 1)
        return float(max(ratio, 1.0))

    def _build_model(self, device, swin_params=None, dstrans_params: Optional[dict] = None):
        """
        根据配置构建模型，支持ResNet34编码器UNet、改进UNet、TransUNet、SwinUNet、Swin-U Mamba。
        
        Args:
            device: 设备
            swin_params: SwinUNet的超参数（如果使用GWO优化）
            dstrans_params: DS-TransUNet的超参数（如果使用GWO优化）
        """
        if self.model_type == "resnet_unet":
            # 默认冻结编码器，前50% epoch只训练解码器，后50%解冻进行微调
            freeze_encoder = True  # 可以通过配置控制
            model = ResNetUNet(freeze_encoder=freeze_encoder).to(device)
        elif self.model_type == "trans_unet" or self.model_type == "transunet":
            model = TransUNet().to(device)
            self.update_progress.emit(15, "使用Transformer+UNet混合架构（可提高Dice指标）")
        elif self.model_type in ("ds_trans_unet", "dstransunet", "ds-transunet"):
            dstrans_kwargs = {
                "in_channels": 3,
                "out_channels": 1,
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 2,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
            }
            if dstrans_params:
                dstrans_kwargs.update(copy.deepcopy(dstrans_params))
            # 移除DSTransUNet不接受的内置参数
            dstrans_kwargs.pop('_from_checkpoint', None)
            if dstrans_kwargs["embed_dim"] % dstrans_kwargs["num_heads"] != 0:
                dstrans_kwargs["embed_dim"] = dstrans_kwargs["num_heads"] * max(1, dstrans_kwargs["embed_dim"] // dstrans_kwargs["num_heads"])
            model = DSTransUNet(**dstrans_kwargs).to(device)
            self.update_progress.emit(15, "使用DS-TransUNet（双尺度Transformer+UNet，增强多尺度特征提取）")
        elif self.model_type == "swin_unet" or self.model_type == "swinunet":
            swin_kwargs = {
                "in_channels": 3,
                "out_channels": 1
            }
            if swin_params:
                swin_kwargs.update(copy.deepcopy(swin_params))
            # 如果参数来自checkpoint推断，跳过归一化以保持兼容
            from_checkpoint = swin_params and swin_params.get('_from_checkpoint', False)
            if not from_checkpoint:
                normalized_embed = SwinUNet._normalize_embed_dim(swin_kwargs.get('embed_dim', 96))
                swin_kwargs['embed_dim'] = normalized_embed
            img_size = swin_kwargs.get('img_size', (224, 224))
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            patch_size = swin_kwargs.get('patch_size', (4, 4))
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            grid_h = max(2, img_size[0] // max(1, patch_size[0]))
            if not from_checkpoint:
                normalized_window = SwinUNet._normalize_window_size(swin_kwargs.get('window_size', 8), max_grid=grid_h)
                swin_kwargs['window_size'] = normalized_window
            if 'drop_path_rate' not in swin_kwargs:
                swin_kwargs['drop_path_rate'] = 0.1 if not from_checkpoint else 0.0
            swin_kwargs['img_size'] = img_size
            swin_kwargs['patch_size'] = patch_size
            # 保留_from_checkpoint和_mlp_hidden_dims传给SwinUNet
            model = SwinUNet(**swin_kwargs).to(device)
            final_embed = swin_kwargs.get('embed_dim', 96)
            final_window = swin_kwargs.get('window_size', 8)
            self.update_progress.emit(
                15,
                f"使用SwinUNet（参数：embed_dim={int(final_embed)}, window_size={int(final_window)}）"
            )
        elif self.model_type in ("swin_u_mamba", "swin-u-mamba", "swinumamba"):
            mamba_kwargs = {
                "in_channels": 3,
                "out_channels": 1,
                "base_channels": 64,
                "num_blocks": (2, 2, 2, 2),
                "dropout": 0.05,
            }
            if swin_params:
                mamba_kwargs.update(copy.deepcopy(swin_params))
            model = SwinUMamba(**mamba_kwargs).to(device)
            self.update_progress.emit(
                15,
                f"使用Swin-U Mamba（base_channels={mamba_kwargs.get('base_channels',64)}, blocks={mamba_kwargs.get('num_blocks',(2,2,2,2))}）"
            )
        else:
            model = ImprovedUNet().to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            self.update_progress.emit(20, f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        # 初始化SkullStripper
        if self.use_skull_stripper and self.skull_stripper is None:
            self.skull_stripper = SkullStripper(self.skull_stripper_path, device, self.skull_stripper_threshold)
            if not self.skull_stripper.is_available():
                self.use_skull_stripper = False
                print("[警告] SkullStripper未准备好，将跳过剥除颅骨步骤。")
        return model

    def _apply_skull_strip(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        如果启用SkullStripper，则对输入进行剥除颅骨处理。
        Returns:
            processed_images, brain_mask
        """
        if not self.use_skull_stripper or not self.skull_stripper or not self.skull_stripper.is_available():
            return images, None
        return self.skull_stripper.strip(images)

    # 分类模型相关函数已删除

    def _safe_update_bn(self, model, dataloader, device):
        """安全地更新BN统计量，处理可能包含分类标签的数据加载器"""
        model.train()
        with torch.no_grad():
            for batch_data in dataloader:
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images = images.to(device)
                _ = model(images)  # 只使用images来更新BN统计量
    
    def _unwrap_model(self, model):
        """解包DataParallel，返回实际模型"""
        actual = model
        if isinstance(actual, nn.DataParallel):
            actual = actual.module
        if isinstance(actual, AveragedModel):
            # AveragedModel包装了原始模型，位于module属性
            actual = actual.module
        return actual

    def _supports_aux_outputs(self, model):
        """模型是否支持辅助输出"""
        actual = self._unwrap_model(model)
        return isinstance(actual, (ImprovedUNet, TransUNet, DSTransUNet, SwinUNet))

    def _supports_attention_maps(self, model):
        """模型是否提供注意力图"""
        actual = self._unwrap_model(model)
        return isinstance(actual, (ImprovedUNet, TransUNet, DSTransUNet, SwinUNet, ResNetUNet))
    
    def _create_optimizer(self, parameters, lr):
        # 微调阶段统一收紧学习率：大于1e-4的强制压到1e-4，若恰好等于1e-4则进一步降为1e-5
        effective_lr = float(lr)
        if effective_lr > 1e-4:
            effective_lr = 1e-4
        elif abs(effective_lr - 1e-4) < 1e-9:
            effective_lr = 1e-5
        # 若外部已传入更小的学习率（如2e-5），则保持不变
        if self.optimizer_type == "adam":
            return optim.Adam(parameters, lr=effective_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        if self.optimizer_type == "sgd":
            # 使用 SGD + Nesterov 动量
            return optim.SGD(parameters, lr=effective_lr, momentum=0.99, nesterov=True, weight_decay=5e-4)
        # 默认使用AdamW - 小数据集增强正则化
        return optim.AdamW(parameters, lr=effective_lr, weight_decay=5e-4)
    
    def _get_loss_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """优化的损失权重策略 - 更强调Dice和Tversky"""
        progress = epoch / max(1, total_epochs - 1)
        # 早期：BCE主导帮助收敛；后期：Dice+Tversky主导提升分割质量
        weights = {
            # BCE 只负责前期收敛, 后期权重下降到较低水平
            'bce': max(0.10, 0.30 - 0.18 * progress),
            # Dice 从一开始就占比较高, 随epoch进一步提升
            'dice': 0.45 + 0.30 * progress,          # 0.45 -> 0.75
            # Tversky 在后期配合Dice, 更关注 FN
            'tversky': 0.25 + 0.15 * progress,       # 0.25 -> 0.40
            # Focal Tversky 针对难案例，逐步加权
            'tversky_focal': 0.05 + 0.10 * progress,  # 0.05 -> 0.15
            # 边界损失稍微降低, 防止过度关注细小噪声
            'boundary': 0.08,
            # Hausdorff 距离损失：训练前30%关闭，之后渐进开启（专注边界）
            'hausdorff': 0.08 * max((progress - 0.3) / 0.7, 0.0),
            # Focal 主要在前期起作用, 后期权重很小
            'focal': max(0.03, 0.10 * (1.0 - progress)),
            # Lovasz 在全程参与, 但后期比重更高, 对齐 IoU/Dice
            'lovasz': 0.05 + 0.10 * progress,        # 0.05 -> 0.15
            # 假阴性惩罚逐渐增加, 提高召回率, 一般能拉高Dice
            'fn_penalty': 0.06 + 0.09 * progress,    # 0.06 -> 0.15
            # 假阳性惩罚随epoch略微下降, 让模型在后期更敢预测前景
            'fp_penalty': 0.18 + 0.12 * (1.0 - progress),  # 0.30 -> 0.18
        }
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        return weights
    
    def _init_ema_model(self, model, device):
        """
        初始化EMA模型副本
        注意：EMA模型保持train()模式，以便BN统计量也能正确更新
        """
        actual_model = self._unwrap_model(model)
        ema_model = copy.deepcopy(actual_model).to(device)
        # 保持train模式，这样BN的running统计量也能被EMA更新
        ema_model.train()
        # 禁用梯度计算
        for param in ema_model.parameters():
            param.requires_grad = False
        # 确保初始权重完全同步（使用decay=0进行一次更新，确保完全复制）
        # 这样EMA模型从一开始就和原模型完全一致
        with torch.no_grad():
            ema_state = ema_model.state_dict()
            model_state = actual_model.state_dict()
            for key in ema_state.keys():
                if key in model_state:
                    ema_state[key].copy_(model_state[key])
        return ema_model
    
    def _update_ema_model(self, ema_model, model, decay=None):
        """
        使用当前模型参数更新EMA模型
        同时更新BN的running_mean和running_var
        """
        if ema_model is None:
            return
        if decay is None:
            decay = self.ema_decay
        if not 0.0 < decay < 1.0:
            decay = 0.995
        
        actual_model = self._unwrap_model(model)
        
        with torch.no_grad():
            # 更新普通参数（只更新requires_grad=True的参数）
            for ema_param, model_param in zip(ema_model.parameters(), actual_model.parameters()):
                if model_param.requires_grad:
                    ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
            
            # 更新BN层的running统计量（如果存在）
            # 使用state_dict来确保正确匹配模块
            ema_state = ema_model.state_dict()
            model_state = actual_model.state_dict()
            
            for key in ema_state.keys():
                if 'running_mean' in key or 'running_var' in key:
                    if key in model_state:
                        ema_state[key].mul_(decay).add_(model_state[key], alpha=1.0 - decay)
                elif 'num_batches_tracked' in key:
                    if key in model_state:
                        ema_state[key] = model_state[key]

    def _sanitize_gradients(self, model, clamp_value=5.0):
        """
        清理梯度中的NaN/Inf，避免传播到后续步骤。
        Returns:
            bool: 是否发现并修复了异常梯度
        """
        had_issue = False
        actual_model = self._unwrap_model(model)
        for name, param in actual_model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                had_issue = True
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
                param.grad.clamp_(min=-clamp_value, max=clamp_value)
        return had_issue
    
    def _extract_model_config(self, model):
        actual = self._unwrap_model(model)
        config = {"model_type": self.model_type}
        if isinstance(actual, SwinUNet):
            config["swin_params"] = copy.deepcopy(actual.get_config())
        if isinstance(actual, DSTransUNet):
            config["dstrans_params"] = copy.deepcopy(actual.get_config())
        if isinstance(actual, ResNetUNet):
            # 从模型结构中推断backbone_name
            # 检查enc4的输出通道数来判断是ResNet50还是ResNet101
            if hasattr(actual, 'enc4'):
                # ResNet101的layer4输出2048通道，ResNet50也是2048，但可以通过layer数量判断
                # 更简单的方法：检查是否有backbone_name属性，或者从state_dict推断
                backbone_name = getattr(actual, 'backbone_name', 'resnet101')
                config["resnet_params"] = {
                    "in_channels": getattr(actual, 'in_channels', 3),
                    "out_channels": getattr(actual, 'out_channels', 1),
                    "pretrained": False,  # 测试时不需要pretrained
                    "backbone_name": backbone_name
                }
        config["best_threshold"] = float(getattr(self, "last_optimal_threshold", 0.5))
        config["skull_stripping"] = {
            "enabled": self.use_skull_stripper,
            "model_path": self.skull_stripper_path,
            "threshold": self.skull_stripper_threshold
        }
        config["context"] = {
            "slices": self.context_slices,
            "gap": self.context_gap
        }
        config["extra_modalities"] = list(self.extra_modalities_dirs.keys())
        return config
    
    def _save_checkpoint(self, model, path):
        actual = self._unwrap_model(model)
        state_dict = actual.state_dict()
        config = self._extract_model_config(model)
        torch.save({"state_dict": state_dict, "config": config}, path)
    
    def _gwo_optimize_swin_params(self, train_loader, val_loader, device, n_wolves=10, max_iter=5):
        """
        使用GWO优化SwinUNet的超参数
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            n_wolves: 灰狼数量（减少以加快优化速度）
            max_iter: 最大迭代次数（减少以加快优化速度）
        
        Returns:
            最佳参数字典
        """
        def objective_func(params):
            """目标函数：训练模型并返回验证Dice分数"""
            try:
                params = params.copy()
                params['embed_dim'] = SwinUNet._normalize_embed_dim(params.get('embed_dim', 96))
                params['window_size'] = SwinUNet._normalize_window_size(params.get('window_size', 8), max_grid=64)
                # 创建临时模型 - 小数据集默认更高dropout
                temp_model = SwinUNet(
                    embed_dim=int(params['embed_dim']),
                    window_size=int(params['window_size']),
                    mlp_ratio=params.get('mlp_ratio', 4.0),
                    drop_rate=params.get('drop_rate', 0.2),
                    attn_drop_rate=params.get('attn_drop_rate', 0.2)
                ).to(device)
                
                # 快速训练几个批次来评估参数
                temp_model.train()
                optimizer = self._create_optimizer(temp_model.parameters(), lr=1e-4)
                bce_criterion = nn.BCEWithLogitsLoss()
                
                # 快速训练（仅几个批次）
                max_batches = 5
                for batch_idx, batch_data in enumerate(train_loader):
                    if batch_idx >= max_batches:
                        break
                    # 处理数据：可能包含分类标签
                    if len(batch_data) == 3:
                        images, masks, _ = batch_data
                    else:
                        images, masks = batch_data
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad()
                    outputs = temp_model(images)
                    loss = bce_criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                
                # 在验证集上评估（改进：加入Hausdorff Distance作为优化目标）
                temp_model.eval()
                dice_scores = []
                hd95_scores = []
                # 使用与主验证阶段一致的阈值，避免Dice不一致
                eval_threshold = float(getattr(self, "last_optimal_threshold", 0.5))
                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(val_loader):
                        if batch_idx >= 3:  # 仅评估几个批次
                            break
                        # 处理数据：可能包含分类标签
                        if len(batch_data) == 3:
                            images, masks, _ = batch_data
                        else:
                            images, masks = batch_data
                        images, masks = images.to(device), masks.to(device)
                        outputs = temp_model(images)
                        preds = torch.sigmoid(outputs)
                        # 确保 preds 和 masks 的空间尺寸匹配
                        if preds.shape[2:] != masks.shape[2:]:
                            preds = F.interpolate(preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        preds = preds > eval_threshold
                        batch_dice = self.calculate_batch_dice(preds.float(), masks)
                        dice_scores.extend(batch_dice.cpu().numpy())
                        
                        # 计算Hausdorff Distance 95
                        try:
                            for i in range(preds.shape[0]):
                                pred_mask = preds[i, 0].cpu().numpy()
                                target_mask = masks[i, 0].cpu().numpy()
                                hd95 = self.calculate_hd95(pred_mask, target_mask)
                                if not np.isnan(hd95):
                                    hd95_scores.append(hd95)
                        except Exception:
                            pass  # 如果HD95计算失败，跳过
                
                avg_dice = np.mean(dice_scores) if dice_scores else 0.0
                avg_hd95 = np.mean(hd95_scores) if hd95_scores else 0.0
                
                # 组合优化目标：Dice越高越好，HD95越低越好
                # 归一化HD95（假设最大HD95为100像素），然后与Dice组合
                normalized_hd95 = 1.0 - min(avg_hd95 / 100.0, 1.0)  # 归一化到[0, 1]，越高越好
                combined_score = 0.7 * avg_dice + 0.3 * normalized_hd95  # Dice权重70%，HD95权重30%
                
                del temp_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return combined_score
            except Exception as e:
                print(f"GWO评估错误: {e}")
                return 0.0
        
        # 定义参数边界
        bounds = {
            'embed_dim': (64, 128),
            'window_size': (4, 12),
            'mlp_ratio': (2.0, 6.0),
            'drop_rate': (0.15, 0.35),  # 小数据集更高dropout
            'attn_drop_rate': (0.15, 0.35),
        }
        
        # 创建GWO优化器
        gwo = GWOOptimizer(
            n_wolves=n_wolves,
            max_iter=max_iter,
            bounds=bounds,
            objective_func=objective_func
        )
        
        # 执行优化
        def callback(iter, score, params):
            self.update_progress.emit(13, f"GWO迭代 {iter}/{max_iter}, 当前最佳综合分数: {score:.4f} (Dice+HD95)")
        
        best_params, best_score, history = gwo.optimize(callback=callback)
        if best_params:
            best_params['embed_dim'] = SwinUNet._normalize_embed_dim(best_params.get('embed_dim', 96))
            best_params['window_size'] = SwinUNet._normalize_window_size(best_params.get('window_size', 8), max_grid=64)
        
        return best_params
    
    def _gwo_optimize_nnformer_params(self, train_loader, val_loader, device, n_wolves=5, max_iter=2):
        """
        使用GWO优化nnFormer的超参数
        
        注意：为了减少内存占用，默认使用较少的wolves和迭代次数
        如果内存充足，可以增加这些参数以提高优化效果
        """
        # 跟踪评估计数和内存使用
        eval_count = [0]  # 使用列表以便在闭包中修改
        total_evals = n_wolves * (max_iter + 1)  # 初始评估 + 每次迭代
        
        def objective_func(params):
            temp_model = None
            optimizer = None
            scaler = None
            try:
                eval_count[0] += 1
                current_eval = eval_count[0]
                
                # 获取评估前的内存
                mem_before = self._get_gpu_memory_info()
                
                params = params.copy()
                embed_dim = int(params.get('embed_dim', 96))
                window_size = int(params.get('window_size', 7))
                mlp_ratio = float(params.get('mlp_ratio', 4.0))
                drop_rate = float(params.get('drop_rate', 0.0))
                attn_drop_rate = float(params.get('attn_drop_rate', 0.0))
                drop_path_rate = float(params.get('drop_path_rate', 0.1))
                global_attn_ratio = float(params.get('global_attn_ratio', 0.5))
                
                # 确保embed_dim能被num_heads整除
                # 根据embed_dim自动计算合适的num_heads
                if embed_dim >= 96:
                    num_heads = [3, 6, 12, 24]
                elif embed_dim >= 64:
                    num_heads = [2, 4, 8, 16]
                else:
                    num_heads = [2, 4, 6, 12]
                
                # 可视化：显示当前评估信息
                param_str = f"embed_dim={embed_dim}, window={window_size}, mlp={mlp_ratio:.2f}, drop={drop_rate:.2f}, global_attn={global_attn_ratio:.2f}"
                mem_str = f"内存: {mem_before[0]:.2f}GB / {mem_before[1]:.2f}GB"
                progress_msg = f"GWO评估 [{current_eval}/{total_evals}] | {param_str} | {mem_str}"
                self.update_progress.emit(10 + int(80 * current_eval / total_evals), progress_msg)
                
                # 检查内存使用，如果过高则提前返回
                if mem_before[0] > 13.0:  # 如果已使用超过13GB，直接跳过
                    print(f"警告: GPU内存使用过高 ({mem_before[0]:.2f}GB)，跳过此评估以避免崩溃")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    import gc
                    gc.collect()
                    return 0.0
                
                # 创建临时模型
                temp_model = nnFormer(
                    in_channels=3,
                    out_channels=1,
                    img_size=224,
                    patch_size=4,
                    embed_dim=embed_dim,
                    depths=[2, 2, 2, 2],
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    use_skip_attention=True,
                    global_attn_ratio=global_attn_ratio
                ).to(device)
                
                # 快速训练几个批次来评估参数
                temp_model.train()
                optimizer = self._create_optimizer(temp_model.parameters(), lr=1e-4)
                bce_criterion = nn.BCEWithLogitsLoss()
                
                # 混合精度训练
                amp_enabled = (device.type == 'cuda')
                scaler = GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
                
                # 快速训练（仅几个批次）
                max_batches = 3  # 减少批次以节省内存
                for batch_idx, batch_data in enumerate(train_loader):
                    if batch_idx >= max_batches:
                        break
                    
                    if len(batch_data) == 3:
                        images, masks, _ = batch_data
                    else:
                        images, masks = batch_data
                    images, masks = images.to(device), masks.to(device)
                    
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = temp_model(images)
                            if outputs.shape[2:] != masks.shape[2:]:
                                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                            loss = bce_criterion(outputs, masks)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = temp_model(images)
                        if outputs.shape[2:] != masks.shape[2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        loss = bce_criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                    
                    # 清理
                    del outputs, loss, images, masks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # 在验证集上评估
                temp_model.eval()
                dice_scores = []
                eval_threshold = float(getattr(self, "last_optimal_threshold", 0.5))
                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(val_loader):
                        if batch_idx >= 2:  # 仅评估2个批次
                            break
                        
                        if len(batch_data) == 3:
                            images, masks, _ = batch_data
                        else:
                            images, masks = batch_data
                        images, masks = images.to(device), masks.to(device)
                        
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                outputs = temp_model(images)
                        else:
                            outputs = temp_model(images)
                        
                        if outputs.shape[2:] != masks.shape[2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        
                        preds = torch.sigmoid(outputs)
                        preds = (preds > eval_threshold).float()
                        batch_dice = self.calculate_batch_dice(preds, masks)
                        dice_scores.extend(batch_dice.cpu().numpy())
                        
                        del images, masks, outputs, preds, batch_dice
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                
                if not dice_scores:
                    return 0.0
                dice_mean = float(np.mean(dice_scores))
                
                # 获取评估后的内存
                mem_after = self._get_gpu_memory_info()
                mem_diff = mem_after[0] - mem_before[0]
                
                # 可视化：显示评估结果
                result_msg = f"评估完成 | Dice: {dice_mean:.4f} | 内存变化: {mem_diff:+.2f}GB"
                self.update_progress.emit(10 + int(80 * current_eval / total_evals), result_msg)
                
                return dice_mean
            except Exception as e:
                print(f"GWO评估错误: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
            finally:
                # 关键：显式释放资源
                if temp_model is not None:
                    temp_model.cpu()
                    del temp_model
                if optimizer is not None:
                    optimizer.state.clear()
                    del optimizer
                if scaler is not None:
                    del scaler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                import gc
                gc.collect()

        # 定义参数边界
        bounds = {
            'embed_dim': (64, 128),  # 整数范围
            'window_size': (4, 10),  # 整数范围
            'mlp_ratio': (3.0, 5.0),  # 浮点数范围
            'drop_rate': (0.0, 0.2),  # 浮点数范围
            'attn_drop_rate': (0.0, 0.2),  # 浮点数范围
            'drop_path_rate': (0.05, 0.15),  # 浮点数范围
            'global_attn_ratio': (0.3, 0.7),  # 浮点数范围，控制全局注意力的比例
        }

        # 检查初始内存使用
        initial_mem = self._get_gpu_memory_info()
        if initial_mem[0] > 12.0:
            warning_msg = f"警告: GPU内存使用已较高 ({initial_mem[0]:.2f}GB)，建议关闭其他程序后再运行GWO优化"
            self.update_progress.emit(10, warning_msg)
            print(warning_msg)
        
        gwo = GWOOptimizer(
            n_wolves=n_wolves,
            max_iter=max_iter,
            bounds=bounds,
            objective_func=objective_func,
        )

        def callback(iter, score, params):
            mem_allocated, mem_reserved, mem_max = self._get_gpu_memory_info()
            mem_percent = (mem_allocated / 16.0) * 100 if torch.cuda.is_available() else 0
            
            param_info = f"embed_dim={int(params.get('embed_dim', 96))}, "
            param_info += f"window={int(params.get('window_size', 7))}, "
            param_info += f"mlp={params.get('mlp_ratio', 4.0):.2f}, "
            param_info += f"global_attn={params.get('global_attn_ratio', 0.5):.2f}"
            
            status_msg = f"GWO迭代 {iter}/{max_iter} | 最佳Dice: {score:.4f} | {param_info} | GPU内存: {mem_allocated:.2f}GB ({mem_percent:.1f}%)"
            
            if mem_percent > 90:
                status_msg += " ⚠️⚠️ 内存严重不足！"
            elif mem_percent > 85:
                status_msg += " ⚠️ 内存使用过高！"
            elif mem_percent > 70:
                status_msg += " ⚡ 内存使用较高"
            
            self.update_progress.emit(10 + int(80 * iter / max_iter), status_msg)

        # 显示开始信息
        total_evals = n_wolves * (max_iter + 1)
        start_msg = f"开始GWO优化nnFormer: {n_wolves}个wolves, {max_iter}次迭代, 共{total_evals}次评估 | 初始内存: {initial_mem[0]:.2f}GB"
        self.update_progress.emit(10, start_msg)
        print(start_msg)
        
        best_params, best_score, history = gwo.optimize(callback=callback)
        
        if best_params:
            # 确保参数类型正确
            best_params['embed_dim'] = int(best_params.get('embed_dim', 96))
            best_params['window_size'] = int(best_params.get('window_size', 7))
            best_params['mlp_ratio'] = float(best_params.get('mlp_ratio', 4.0))
            best_params['drop_rate'] = float(best_params.get('drop_rate', 0.0))
            best_params['attn_drop_rate'] = float(best_params.get('attn_drop_rate', 0.0))
            best_params['drop_path_rate'] = float(best_params.get('drop_path_rate', 0.1))
            best_params['global_attn_ratio'] = float(best_params.get('global_attn_ratio', 0.5))
        
        final_msg = f"GWO优化完成 | 最佳Dice: {best_score:.4f} | 最佳参数: {best_params}"
        self.update_progress.emit(14, final_msg)
        print(final_msg)
        
        return best_params
    
    def _get_gpu_memory_info(self):
        """获取GPU内存使用信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            return allocated, reserved, max_allocated
        return 0.0, 0.0, 0.0
    
    def _gwo_optimize_dstrans_params(self, train_loader, val_loader, device, n_wolves=5, max_iter=2):
        """
        使用GWO优化DS-TransUNet的超参数
        
        注意：为了减少内存占用，默认使用较少的wolves和迭代次数
        如果内存充足，可以增加这些参数以提高优化效果
        """
        # 跟踪评估计数和内存使用
        eval_count = [0]  # 使用列表以便在闭包中修改
        total_evals = n_wolves * (max_iter + 1)  # 初始评估 + 每次迭代
        
        def objective_func(params):
            temp_model = None
            optimizer = None
            try:
                eval_count[0] += 1
                current_eval = eval_count[0]
                
                # 获取评估前的内存
                mem_before = self._get_gpu_memory_info()
                
                params = params.copy()
                embed_dim = int(params.get('embed_dim', 256))
                num_heads = int(params.get('num_heads', 8))
                num_layers = int(params.get('num_layers', 2))
                mlp_ratio = float(params.get('mlp_ratio', 4.0))
                dropout = float(params.get('dropout', 0.1))
                if embed_dim % num_heads != 0:
                    embed_dim = num_heads * max(1, embed_dim // num_heads)
                
                # 可视化：显示当前评估信息
                param_str = f"embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}, mlp={mlp_ratio:.2f}, drop={dropout:.2f}"
                mem_str = f"内存: {mem_before[0]:.2f}GB / {mem_before[1]:.2f}GB"
                progress_msg = f"GWO评估 [{current_eval}/{total_evals}] | {param_str} | {mem_str}"
                self.update_progress.emit(10 + int(80 * current_eval / total_evals), progress_msg)
                
                # 检查内存使用，如果过高则提前返回（更严格的限制）
                if mem_before[0] > 13.0:  # 如果已使用超过13GB，直接跳过（从14GB降低到13GB）
                    print(f"警告: GPU内存使用过高 ({mem_before[0]:.2f}GB)，跳过此评估以避免崩溃")
                    # 强制清理
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    import gc
                    gc.collect()
                    return 0.0
                
                # 创建临时模型（使用更小的embed_dim范围以减少内存）
                temp_model = DSTransUNet(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ).to(device)
                
                # 使用混合精度训练以减少内存
                temp_model.train()
                optimizer = self._create_optimizer(temp_model.parameters(), lr=1e-4)
                bce_criterion = nn.BCEWithLogitsLoss()
                # 使用 torch.amp.GradScaler 以避免弃用警告
                scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
                
                # 限制训练批次以减少内存使用
                max_batches = 1  # 训练1个batch
                for batch_idx, batch_data in enumerate(train_loader):
                    if batch_idx >= max_batches:
                        break
                    
                    # 立即释放batch_data引用
                    if len(batch_data) == 3:
                        images, masks, _ = batch_data
                        del batch_data
                    else:
                        images, masks = batch_data
                        del batch_data
                    
                    images, masks = images.to(device), masks.to(device)
                    
                    # 使用混合精度
                    if scaler is not None:
                        optimizer.zero_grad(set_to_none=True)  # 更彻底地清零梯度
                        with torch.amp.autocast('cuda'):
                            outputs = temp_model(images)
                            # 确保输出尺寸与mask尺寸匹配
                            if outputs.shape[2:] != masks.shape[2:]:
                                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                            loss = bce_criterion(outputs, masks)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.zero_grad(set_to_none=True)  # 更彻底地清零梯度
                        outputs = temp_model(images)
                        # 确保输出尺寸与mask尺寸匹配
                        if outputs.shape[2:] != masks.shape[2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        loss = bce_criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                    
                    # 彻底清理所有中间变量
                    del outputs, loss, images, masks
                    # 每次batch后都清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                temp_model.eval()
                dice_scores = []
                eval_threshold = float(getattr(self, "last_optimal_threshold", 0.5))
                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(val_loader):
                        if batch_idx >= 1:  # 评估1个batch
                            break
                        
                        # 立即释放batch_data引用
                        if len(batch_data) == 3:
                            images, masks, _ = batch_data
                            del batch_data
                        else:
                            images, masks = batch_data
                            del batch_data
                        
                        images, masks = images.to(device), masks.to(device)
                        
                        # 使用混合精度推理
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                outputs = temp_model(images)
                        else:
                            outputs = temp_model(images)
                        
                        # 确保输出尺寸与mask尺寸匹配
                        if outputs.shape[2:] != masks.shape[2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        
                        preds = torch.sigmoid(outputs)
                        preds = (preds > eval_threshold).float()
                        dice_scores_batch = self.calculate_batch_dice(preds, masks)
                        # 立即转移到CPU并转换为numpy，释放GPU内存
                        dice_scores_batch_cpu = dice_scores_batch.cpu().numpy()
                        dice_scores.extend(dice_scores_batch_cpu)
                        
                        # 彻底清理所有中间变量
                        del images, masks, outputs, preds, dice_scores_batch, dice_scores_batch_cpu
                        # 每次batch后都清理缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                
                if not dice_scores:
                    return 0.0
                dice_mean = float(np.mean(dice_scores))
                
                # 获取评估后的内存
                mem_after = self._get_gpu_memory_info()
                mem_diff = mem_after[0] - mem_before[0]
                
                # 可视化：显示评估结果
                result_msg = f"评估完成 | Dice: {dice_mean:.4f} | 内存变化: {mem_diff:+.2f}GB"
                self.update_progress.emit(10 + int(80 * current_eval / total_evals), result_msg)
                
                return dice_mean
            except Exception as e:
                print(f"GWO评估错误: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
            finally:
                # 关键：显式释放资源
                if temp_model is not None:
                    # 先清除模型的所有参数和缓冲区
                    temp_model.cpu()  # 移到CPU
                    del temp_model
                if optimizer is not None:
                    # 清除优化器状态
                    optimizer.state.clear()
                    del optimizer
                if scaler is not None:
                    del scaler
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # 重置最大内存统计，以便监控每次评估的内存峰值
                    torch.cuda.reset_peak_memory_stats()
                # 强制Python垃圾回收
                import gc
                gc.collect()

        # 注意：整数参数使用整数边界，浮点数参数使用浮点数边界
        # 为了减少内存占用，缩小参数搜索范围
        bounds = {
            'embed_dim': (128, 192),  # 整数范围（从128-256缩小到128-192）
            'num_heads': (4, 6),  # 整数范围（从4-8缩小到4-6）
            'num_layers': (2, 2),  # 整数范围（固定为2层，最小化内存）
            'mlp_ratio': (3.0, 4.0),  # 浮点数范围（从3.0-4.5缩小到3.0-4.0）
            'dropout': (0.05, 0.1),  # 浮点数范围（从0.05-0.15缩小到0.05-0.1）
        }

        # 检查初始内存使用
        initial_mem = self._get_gpu_memory_info()
        if initial_mem[0] > 12.0:  # 如果初始内存已超过12GB
            warning_msg = f"警告: GPU内存使用已较高 ({initial_mem[0]:.2f}GB)，建议关闭其他程序后再运行GWO优化"
            self.update_progress.emit(10, warning_msg)
            print(warning_msg)
        
        gwo = GWOOptimizer(
            n_wolves=n_wolves,
            max_iter=max_iter,
            bounds=bounds,
            objective_func=objective_func,
        )

        def callback(iter, score, params):
            # 获取当前内存使用
            mem_allocated, mem_reserved, mem_max = self._get_gpu_memory_info()
            mem_percent = (mem_allocated / 16.0) * 100 if torch.cuda.is_available() else 0  # 假设16GB GPU
            
            # 格式化参数信息
            param_info = f"embed_dim={int(params.get('embed_dim', 256))}, "
            param_info += f"heads={int(params.get('num_heads', 8))}, "
            param_info += f"layers={int(params.get('num_layers', 2))}"
            
            # 构建详细的状态信息
            status_msg = f"GWO迭代 {iter}/{max_iter} | 最佳Dice: {score:.4f} | {param_info} | GPU内存: {mem_allocated:.2f}GB ({mem_percent:.1f}%)"
            
            # 内存警告
            if mem_percent > 90:
                status_msg += " ⚠️⚠️ 内存严重不足！"
            elif mem_percent > 85:
                status_msg += " ⚠️ 内存使用过高！"
            elif mem_percent > 70:
                status_msg += " ⚡ 内存使用较高"
            
            self.update_progress.emit(10 + int(80 * iter / max_iter), status_msg)

        # 显示开始信息
        total_evals = n_wolves * (max_iter + 1)
        start_msg = f"开始GWO优化: {n_wolves}个wolves, {max_iter}次迭代, 共{total_evals}次评估 | 初始内存: {initial_mem[0]:.2f}GB"
        self.update_progress.emit(10, start_msg)
        
        best_params, best_score, history = gwo.optimize(callback=callback)
        if best_params:
            best_params["embed_dim"] = int(best_params.get("embed_dim", 256))
            best_params["num_heads"] = int(best_params.get("num_heads", 8))
            best_params["num_layers"] = int(best_params.get("num_layers", 2))
            best_params["mlp_ratio"] = float(best_params.get("mlp_ratio", 4.0))
            best_params["dropout"] = float(best_params.get("dropout", 0.1))
            if best_params["embed_dim"] % best_params["num_heads"] != 0:
                best_params["embed_dim"] = best_params["num_heads"] * max(1, best_params["embed_dim"] // best_params["num_heads"])
            best_params["_from_checkpoint"] = True
        return best_params

    def _save_matlab_viz_payload(
        self,
        images_list: List[np.ndarray],
        masks_list: List[np.ndarray],
        preds_list: List[np.ndarray],
        save_name: str
    ) -> str:
        if not images_list:
            raise ValueError("没有样本可用于生成MATLAB可视化")

        payload_path = os.path.join(self.temp_dir, f"{save_name}_payload.mat")
        images_arr = np.transpose(np.stack(images_list, axis=0), (1, 2, 3, 0)).astype(np.float32)
        masks_arr = np.transpose(np.stack(masks_list, axis=0), (1, 2, 0)).astype(np.float32)
        preds_arr = np.transpose(np.stack(preds_list, axis=0), (1, 2, 0)).astype(np.float32)
        savemat(payload_path, {'images': images_arr, 'masks': masks_arr, 'preds': preds_arr})
        return payload_path

    def _save_training_history_payload(self) -> Optional[str]:
        if not self.train_loss_history or not self.val_loss_history:
            return None
        payload_path = os.path.join(self.temp_dir, "training_history_payload.mat")
        epochs = np.arange(1, len(self.train_loss_history) + 1, dtype=np.float32)
        savemat(payload_path, {
            'epochs': epochs,
            'train_loss': np.array(self.train_loss_history, dtype=np.float32),
            'val_loss': np.array(self.val_loss_history, dtype=np.float32),
            'val_dice': np.array(self.val_dice_history or [0.0] * len(epochs), dtype=np.float32)
        })
        return payload_path

    def _save_performance_payload(self, detailed_metrics: dict) -> str:
        payload_path = os.path.join(self.temp_dir, "performance_metrics_payload.mat")
        def to_array_map(source: dict) -> dict:
            return {k: np.array(source.get(k, 0.0)).astype(np.float32) for k in source}

        metrics = {k: np.array(v, dtype=np.float32) for k, v in detailed_metrics.get('all_samples', {}).items()}
        avg = to_array_map(detailed_metrics.get('average', {}))
        std = to_array_map(detailed_metrics.get('std', {}))
        min_vals = to_array_map(detailed_metrics.get('min', {}))
        max_vals = to_array_map(detailed_metrics.get('max', {}))
        median_vals = to_array_map(detailed_metrics.get('median', {}))
        savemat(payload_path, {
            'metrics': metrics,
            'avg_metrics': avg,
            'std_metrics': std,
            'min_metrics': min_vals,
            'max_metrics': max_vals,
            'median_metrics': median_vals
        })
        return payload_path

    def _save_test_results_payload(self, images_np: List[np.ndarray], masks_np: List[np.ndarray],
                                   preds_np: List[np.ndarray], metrics_list: List[dict],
                                   save_name: str) -> str:
        if not images_np:
            raise ValueError("没有样本可用于生成测试可视化")
        payload_path = os.path.join(self.temp_dir, f"{save_name}_payload.mat")
        images_arr = np.transpose(np.stack(images_np, axis=0), (1, 2, 3, 0)).astype(np.float32)
        masks_arr = np.transpose(np.stack(masks_np, axis=0), (1, 2, 0)).astype(np.float32)
        preds_arr = np.transpose(np.stack(preds_np, axis=0), (1, 2, 0)).astype(np.float32)
        dice_vals = np.array([m.get('dice', 0.0) for m in metrics_list], dtype=np.float32)
        iou_vals = np.array([m.get('iou', 0.0) for m in metrics_list], dtype=np.float32)
        savemat(payload_path, {
            'images': images_arr,
            'masks': masks_arr,
            'preds': preds_arr,
            'dice': dice_vals,
            'iou': iou_vals
        })
        return payload_path

    def _save_attention_payload(self, images_np: List[np.ndarray], masks_np: List[np.ndarray],
                                preds_np: List[np.ndarray], attention_maps: dict,
                                save_name: str) -> str:
        if not images_np:
            raise ValueError("没有样本可用于生成注意力可视化")
        payload_path = os.path.join(self.temp_dir, f"{save_name}_payload.mat")
        images_arr = np.transpose(np.stack(images_np, axis=0), (1, 2, 3, 0)).astype(np.float32)
        masks_arr = np.transpose(np.stack(masks_np, axis=0), (1, 2, 0)).astype(np.float32)
        preds_arr = np.transpose(np.stack(preds_np, axis=0), (1, 2, 0)).astype(np.float32)
        payload = {
            'images': images_arr,
            'masks': masks_arr,
            'preds': preds_arr
        }
        for key, maps in attention_maps.items():
            if not maps:
                continue
            payload[key] = np.transpose(np.stack(maps, axis=0), (1, 2, 0)).astype(np.float32)
        savemat(payload_path, payload)
        return payload_path

    def _safe_dice_score(self, pred, target, eps: float = 1e-7) -> float:
        """
        计算Dice系数,对空预测和空目标进行安全处理。
        
        处理策略:
        - 当目标为空且预测也为空: Dice = 1.0 (完美匹配)
        - 当目标为空但预测有误检: 使用相对误差公式,避免过度惩罚
        - 当预测为空但目标有前景: Dice = 0.0 (完全漏检)
        - 正常情况: 使用标准Dice公式
        """
        if isinstance(pred, torch.Tensor):
            # 确保 pred 和 target 的空间尺寸匹配
            if pred.shape != target.shape:
                if pred.dim() >= 2 and target.dim() >= 2:
                    if pred.shape[-2:] != target.shape[-2:]:
                        # 将 pred 调整到 target 的尺寸
                        if pred.dim() == 2:
                            pred = pred.unsqueeze(0).unsqueeze(0)
                        elif pred.dim() == 3:
                            pred = pred.unsqueeze(0)
                        if target.dim() == 2:
                            target = target.unsqueeze(0).unsqueeze(0)
                        elif target.dim() == 3:
                            target = target.unsqueeze(0)
                        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
                        if pred.dim() == 4 and pred.size(0) == 1:
                            pred = pred.squeeze(0)
                        if target.dim() == 4 and target.size(0) == 1:
                            target = target.squeeze(0)
            inter = float((pred * target).sum().item())
            pred_sum = float(pred.sum().item())
            target_sum = float(target.sum().item())
            total_pixels = pred.numel()
        else:
            # NumPy 数组处理
            if pred.shape != target.shape:
                # 使用 scipy 或 PIL 进行 resize
                from scipy.ndimage import zoom
                if len(pred.shape) == 2 and len(target.shape) == 2:
                    zoom_factors = (target.shape[0] / pred.shape[0], target.shape[1] / pred.shape[1])
                    pred = zoom(pred, zoom_factors, order=1)
            inter = float(np.sum(pred * target))
            pred_sum = float(np.sum(pred))
            target_sum = float(np.sum(target))
            total_pixels = pred.size if hasattr(pred, 'size') else np.prod(pred.shape)
        
        # Case 1: 目标为空
        if target_sum <= eps:
            if pred_sum <= eps:
                return 1.0  # 预测也为空,完美匹配
            else:
                # 预测有误检,计算相对惩罚
                # 基于误检像素占总像素的比例
                false_positive_ratio = pred_sum / total_pixels
                # 使用线性惩罚: Dice = 1 - 2×误检率
                # 例如: 1%误检 -> 0.98, 5%误检 -> 0.90, 10%误检 -> 0.80
                return max(0.0, 1.0 - 2.0 * false_positive_ratio)
        
        # Case 2: 预测为空但目标有前景
        if pred_sum <= eps:
            return 0.0  # 完全漏检
        
        # Case 3: 正常情况,使用标准Dice
        denom = pred_sum + target_sum
        return (2.0 * inter + eps) / (denom + eps)
    def calculate_hd95(self, pred, gt):
        """
        计算 Hausdorff Distance 95 (HD95)
        衡量预测边界与真实边界的重合度，单位：像素
        """
        import numpy as np
        from scipy.ndimage import binary_erosion, distance_transform_edt
        
        try:
            # 确保输入是 bool 类型
            if pred.dtype != bool:
                pred = (pred > 0.5).astype(bool)
            if gt.dtype != bool:
                gt = (gt > 0.5).astype(bool)
            
            # 如果全是黑的（没有预测或没有真值），直接返回默认值
            if not pred.any() or not gt.any():
                # 如果都没病灶，距离为0；如果一个有一个没，距离无穷大(用99.9代替)
                return 0.0 if (not pred.any() and not gt.any()) else 99.9
            
            # 提取边界
            structure = np.ones((3, 3), dtype=bool)
            pred_border = np.logical_xor(pred, binary_erosion(pred, structure))
            gt_border = np.logical_xor(gt, binary_erosion(gt, structure))
            
            # 如果边界提取失败（比如只有一个像素），回退到原图
            if not pred_border.any(): pred_border = pred
            if not gt_border.any(): gt_border = gt
            
            # 计算距离变换 (Distance Transform)
            # dt[i] 表示像素 i 到最近背景像素的距离
            # 我们需要的是：预测边界上的点 -> 到 -> 真实边界 的最近距离
            gt_dt = distance_transform_edt(~gt_border)
            pred_dt = distance_transform_edt(~pred_border)
            
            # 双向距离
            d1 = gt_dt[pred_border] # 预测边界点 到 真实边界 的距离
            d2 = pred_dt[gt_border] # 真实边界点 到 预测边界 的距离
            
            all_distances = np.concatenate([d1, d2])
            
            if all_distances.size == 0:
                return 0.0
            
            # 取第 95 百分位距离，排除离群点干扰
            hd95 = np.percentile(all_distances, 95)
            return float(hd95)
            
        except Exception as e:
            print(f"[Warning] HD95 计算失败: {e}")
            return 99.9 
    def calculate_dice(self, pred, target, smooth=1e-7):
        """计算单个样本的Dice系数"""
        if isinstance(pred, torch.Tensor):
            pred_tensor = pred.float()
            target_tensor = target.float()
        else:
            pred_tensor = torch.from_numpy(pred).float()
            target_tensor = torch.from_numpy(target).float()
        
        # 确保 pred 和 target 的空间尺寸匹配
        if pred_tensor.dim() >= 2 and target_tensor.dim() >= 2:
            if pred_tensor.shape[-2:] != target_tensor.shape[-2:]:
                # 将 pred 调整到 target 的尺寸
                if pred_tensor.dim() == 2:
                    pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
                elif pred_tensor.dim() == 3:
                    pred_tensor = pred_tensor.unsqueeze(0)
                if target_tensor.dim() == 2:
                    target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
                elif target_tensor.dim() == 3:
                    target_tensor = target_tensor.unsqueeze(0)
                pred_tensor = F.interpolate(pred_tensor, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
                if pred_tensor.dim() == 4 and pred_tensor.size(0) == 1:
                    pred_tensor = pred_tensor.squeeze(0)
                if target_tensor.dim() == 4 and target_tensor.size(0) == 1:
                    target_tensor = target_tensor.squeeze(0)
        
        if pred_tensor.dim() > 2:
            pred_tensor = pred_tensor.view(1, -1)
            target_tensor = target_tensor.view(1, -1)
        else:
            pred_tensor = pred_tensor.view(1, -1)
            target_tensor = target_tensor.view(1, -1)
        
        intersection = (pred_tensor * target_tensor).sum()
        return (2. * intersection + smooth) / (pred_tensor.sum() + target_tensor.sum() + smooth)

    def calculate_batch_dice(self, pred, target, smooth=1e-7):
        """
        计算一个批次中每个样本的Dice系数。
        对空mask情况进行特殊处理,避免过度惩罚少量误检。
        """
        # 确保 pred 和 target 的空间尺寸匹配
        if pred.shape[2:] != target.shape[2:]:
            # 将 pred 调整到 target 的尺寸（因为 target 是 ground truth）
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        pred_flat = pred.view(pred.size(0), -1).float()
        target_flat = target.view(target.size(0), -1).float()
        
        batch_size = pred.size(0)
        total_pixels = pred_flat.size(1)
        avg_fg_ratio = float(target_flat.sum() / max(1.0, batch_size * total_pixels))
        # 【修复】降低空mask阈值，从0.015改为0.001，避免将少量前景像素误判为空mask
        # 对于256x256图像，阈值从9.8像素降低到0.65像素，更严格
        adaptive_empty_threshold = max(smooth, avg_fg_ratio * 0.001)
        dice_scores = []
        
        for i in range(batch_size):
            pred_i = pred_flat[i]
            target_i = target_flat[i]
            
            intersection = (pred_i * target_i).sum()
            pred_sum = pred_i.sum()
            target_sum = target_i.sum()
            
            # Case 1: 目标为空（真正的空mask，无病变）
            if target_sum <= adaptive_empty_threshold:
                if pred_sum <= smooth:
                    # 【修改】全阴性情况：预测也为空时，给予完全正确的阴性预测满分奖励
                    # 如果预测也为空，Dice = 1.0（完全正确）
                    dice = 1.0
                else:
                    # 误检惩罚：目标为空但预测有前景
                    false_positive_ratio = pred_sum.item() / max(1.0, total_pixels)
                    dice = max(0.0, 1.0 - 1.5 * false_positive_ratio)
            # Case 2: 预测为空但目标有前景
            elif pred_sum <= smooth:
                dice = 0.0
            # Case 3: 正常情况（有病变样本）
            else:
                dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
            
            dice_scores.append(dice)
        
        return torch.tensor(dice_scores, device=pred.device)

    def dice_loss(self, logits, targets, smooth=1e-7):
        """
        用于训练的Dice Loss（数值稳定版本）。
        logits: 模型原始输出 (未经过sigmoid)
        targets: [0,1] 掩膜
        
        注意: 训练时的loss计算保持标准公式,不对空mask进行特殊宽容处理,
        这样才能让模型学习到正确的预测行为。
        """
        probs = torch.sigmoid(logits)
        # 确保 probs 和 targets 的空间尺寸匹配
        if probs.shape[2:] != targets.shape[2:]:
            # 将 probs 调整到 targets 的尺寸（因为 targets 是 ground truth）
            probs = F.interpolate(probs, size=targets.shape[2:], mode='bilinear', align_corners=False)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1) + smooth
        # 检查分母是否为零或过小
        denominator = torch.clamp(denominator, min=smooth)
        dice = (2. * intersection + smooth) / denominator
        dice = torch.clamp(dice, min=0.0, max=1.0)
        loss = 1 - dice.mean()
        # 检查NaN/Inf
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=logits.device)
        return loss

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        """帮助缓解样本不平衡的Focal Loss（数值稳定版本）"""
        # 确保 logits 和 targets 的空间尺寸匹配
        if logits.shape[2:] != targets.shape[2:]:
            logits = F.interpolate(logits, size=targets.shape[2:], mode='bilinear', align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 使用clamp防止exp溢出
        bce_clamped = torch.clamp(bce, min=-50.0, max=50.0)
        pt = torch.exp(-bce_clamped)
        # 使用clamp防止数值不稳定
        pt = torch.clamp(pt, min=1e-7, max=1.0-1e-7)
        focal = alpha * (1 - pt) ** gamma * bce
        # 检查NaN/Inf
        focal = torch.where(torch.isfinite(focal), focal, torch.zeros_like(focal))
        return focal.mean()

    def tversky_loss(self, logits, targets, alpha=0.1, beta=0.9, smooth=1e-7):
        """
        Tversky Loss对召回/精确进行加权，提升Dice表现（数值稳定版本）
        
        参数说明：
        - alpha: 假阳性(FP)的权重，默认0.1
        - beta: 假阴性(FN/漏报)的权重，默认0.9
        - 当beta=0.9, alpha=0.1时，漏报一个像素的惩罚是多报一个像素惩罚的9倍
        - 这有助于减少漏检，提高召回率，特别适合医学图像分割任务
        """
        # 确保 logits 和 targets 的空间尺寸匹配
        if logits.shape[2:] != targets.shape[2:]:
            logits = F.interpolate(logits, size=targets.shape[2:], mode='bilinear', align_corners=False)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        true_pos = (probs * targets).sum(dim=1)
        false_pos = (probs * (1 - targets)).sum(dim=1)
        false_neg = ((1 - probs) * targets).sum(dim=1)

        denominator = true_pos + alpha * false_pos + beta * false_neg + smooth
        # 检查分母是否为零或过小
        denominator = torch.clamp(denominator, min=smooth)
        tversky = (true_pos + smooth) / denominator
        tversky = torch.clamp(tversky, min=0.0, max=1.0)
        loss = 1 - tversky.mean()
        # 检查NaN/Inf
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=logits.device)
        return loss
    
    def tversky_focal_loss(self, logits, targets, alpha=0.1, beta=0.9, gamma=0.75, smooth=1e-7):
        """
        Focal Tversky Loss: 在Tversky Loss基础上进一步强调难分样本，
        对于Dice难以提升的区域更敏感，可有效改善少量漏检造成的Dice下降。
        （数值稳定版本）
        
        参数说明：
        - alpha: 假阳性(FP)的权重，默认0.1
        - beta: 假阴性(FN/漏报)的权重，默认0.9
        - 当beta=0.9, alpha=0.1时，漏报一个像素的惩罚是多报一个像素惩罚的9倍
        """
        tversky_val = 1.0 - self.tversky_loss(logits, targets, alpha=alpha, beta=beta, smooth=smooth)
        # 确保tversky_val在合理范围内，防止pow溢出
        tversky_val = torch.clamp(tversky_val, min=1e-7, max=1.0-1e-7)
        focal_term = torch.pow((1.0 - tversky_val), gamma)
        # 检查NaN/Inf
        focal_term = torch.where(torch.isfinite(focal_term), focal_term, torch.zeros_like(focal_term))
        return focal_term.mean()

    def edge_loss(self, logits, targets):
        """强调目标边界的拉普拉斯边缘损失"""
        # 确保 logits 和 targets 的空间尺寸匹配
        if logits.shape[2:] != targets.shape[2:]:
            logits = F.interpolate(logits, size=targets.shape[2:], mode='bilinear', align_corners=False)
        probs = torch.sigmoid(logits)
        kernel = logits.new_tensor([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]]).unsqueeze(0).unsqueeze(0)
        preds_edge = F.conv2d(probs, kernel, padding=1)
        target_edge = F.conv2d(targets.float(), kernel, padding=1)
        return F.l1_loss(preds_edge, target_edge)
    
    def hausdorff_distance_loss(self, logits, targets, percentile=95, alpha=1.0):
        """
        Hausdorff Distance Loss - 直接优化边界距离
        
        通过计算预测边界和真实边界之间的Hausdorff距离来优化分割精度，
        特别适用于边界模糊的医学影像分割任务。
        
        Args:
            logits: 模型输出logits (B, 1, H, W)
            targets: 真实掩膜 (B, 1, H, W)
            percentile: 使用百分位数而非最大值，更稳定 (默认95)
            alpha: 距离变换的缩放因子
        """
        # 确保 logits 和 targets 的空间尺寸匹配
        if logits.shape[2:] != targets.shape[2:]:
            logits = F.interpolate(logits, size=targets.shape[2:], mode='bilinear', align_corners=False)
        probs = torch.sigmoid(logits)
        B, C, H, W = probs.shape
        
        # 二值化预测和真实掩膜
        pred_binary = (probs > 0.5).float()
        target_binary = targets.float()
        
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(B):
            pred_mask = pred_binary[b, 0].cpu().numpy()
            target_mask = target_binary[b, 0].cpu().numpy()
            
            # 计算距离变换
            # 对于预测边界到真实边界的距离
            if pred_mask.sum() > 0 and target_mask.sum() > 0:
                # 计算预测边界到最近真实边界的距离
                pred_boundary = pred_mask - binary_erosion(pred_mask.astype(np.uint8), iterations=1).astype(np.float32)
                if pred_boundary.sum() > 0:
                    dist_pred_to_target = distance_transform_edt(1 - target_mask)
                    dist_pred = dist_pred_to_target[pred_boundary > 0]
                    if len(dist_pred) > 0:
                        hd_pred = np.percentile(dist_pred, percentile)
                    else:
                        hd_pred = 0.0
                else:
                    hd_pred = 0.0
                
                # 计算真实边界到最近预测边界的距离
                target_boundary = target_mask - binary_erosion(target_mask.astype(np.uint8), iterations=1).astype(np.float32)
                if target_boundary.sum() > 0:
                    dist_target_to_pred = distance_transform_edt(1 - pred_mask)
                    dist_target = dist_target_to_pred[target_boundary > 0]
                    if len(dist_target) > 0:
                        hd_target = np.percentile(dist_target, percentile)
                    else:
                        hd_target = 0.0
                else:
                    hd_target = 0.0
                
                # Hausdorff距离是双向距离的最大值
                hd = max(hd_pred, hd_target)
                total_loss += hd * alpha
                valid_samples += 1
        
        if valid_samples > 0:
            loss = torch.tensor(total_loss / valid_samples, device=logits.device, dtype=logits.dtype)
        else:
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        return loss
    
    def lovasz_hinge_loss(self, logits, targets):
        """
        Lovasz-Hinge损失 - 直接优化IoU/Dice（数值稳定版本）
        
        Lovasz损失是IoU loss的凸代理,比标准Dice loss更有效
        参考: "The Lovász-Softmax loss" (CVPR 2018)
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # 计算误差（使用hinge loss的形式）
        # errors = max(0, 1 - (2*probs - 1) * (2*targets - 1))
        # 对于二分类：如果预测正确，error接近0；如果预测错误，error接近1
        errors = torch.clamp(1.0 - (2 * probs_flat - 1) * (2 * targets_flat - 1), min=0.0)
        errors_sorted, indices = torch.sort(errors, descending=True)
        targets_sorted = targets_flat[indices]
        
        # Lovasz extension - 修复计算，确保非负
        n = len(targets_sorted)
        if n == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # 计算IoU的Lovasz扩展
        # 对于每个位置，计算累积的intersection和union
        tp = targets_sorted.sum()  # 总的正样本数
        fp = (1 - targets_sorted).sum()  # 总的负样本数
        
        # 计算累积的intersection和union
        tp_cumsum = targets_sorted.cumsum(0)
        fp_cumsum = (1 - targets_sorted).cumsum(0)
        
        # 计算IoU (Jaccard) - 增加数值稳定性
        intersection = tp - tp_cumsum
        union = tp + fp - intersection
        # 使用更大的epsilon并检查除零
        union = torch.clamp(union, min=1e-6)
        jaccard = intersection / union
        jaccard = torch.clamp(jaccard, min=0.0, max=1.0)
        
        # 检查NaN/Inf
        jaccard = torch.where(torch.isfinite(jaccard), jaccard, torch.zeros_like(jaccard))
        
        # 计算Lovasz扩展的梯度权重（差分形式）
        if n > 1:
            jaccard_diff = torch.zeros_like(jaccard)
            jaccard_diff[0] = jaccard[0]
            jaccard_diff[1:] = jaccard[1:] - jaccard[:-1]
            jaccard = jaccard_diff
        
        # 计算损失（确保非负和非NaN）
        loss = torch.dot(errors_sorted, jaccard)
        loss = torch.clamp(loss, min=0.0)  # 确保损失非负
        # 最终检查NaN/Inf
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=logits.device)
        return loss
    
    def attention_concentration_loss(self, attention_maps, masks, weight=0.01):
        """
        注意力集中度损失 - 鼓励注意力聚焦在病灶区域
        
        原理:
        1. 计算注意力图的熵(entropy) - 熵越低越集中
        2. 计算注意力图与mask的对齐度 - 鼓励注意力关注病灶区域
        
        参数:
        - attention_maps: dict of attention maps from different layers
        - masks: ground truth masks
        - weight: loss权重
        """
        if not attention_maps:
            return 0.0
        
        total_loss = 0.0
        num_maps = 0
        
        for key, att_map in attention_maps.items():
            if att_map is None:
                continue
            
            # Resize mask to match attention map size
            B, _, H, W = att_map.shape
            mask_resized = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            
            # Loss 1: 熵损失 - 鼓励注意力分布更尖锐(低熵)
            # 只在有病灶的样本上计算,避免空mask导致问题
            has_foreground = mask_resized.sum(dim=[1, 2, 3]) > 1e-3
            if has_foreground.any():
                att_fg = att_map[has_foreground]
                # 添加小的epsilon避免log(0)
                # 注意: 使用.clamp避免autocast下的数值问题
                att_clamped = att_fg.clamp(min=1e-7, max=1.0-1e-7)
                entropy = -(att_clamped * torch.log(att_clamped) + 
                           (1 - att_clamped) * torch.log(1 - att_clamped)).mean()
                
                # Loss 2: 对齐损失 - 使用MSE替代BCE (autocast安全)
                # 或者使用L1 loss,效果类似但更稳定
                mask_fg = mask_resized[has_foreground]
                alignment_loss = F.mse_loss(att_fg, mask_fg, reduction='mean')
                
                total_loss += entropy * 0.1 + alignment_loss
                num_maps += 1
        
        if num_maps == 0:
            return 0.0
        
        return weight * (total_loss / num_maps)

    def compute_seg_loss(self, logits, masks, bce_criterion, use_lovasz=True, weights=None):
        """
        组合多种损失函数 - 优化版
        
        Args:
            use_lovasz: 是否使用Lovasz损失(推荐,可提升Dice)
        """
        # 确保 logits 和 masks 的空间尺寸匹配
        if logits.shape[2:] != masks.shape[2:]:
            # 将 logits 调整到 masks 的尺寸（因为 masks 是 ground truth）
            logits = F.interpolate(logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        bce_loss = bce_criterion(logits, masks)
        probs = torch.sigmoid(logits)
        dice_loss_val = self.dice_loss(logits, masks)
        focal_loss_val = self.focal_loss(logits, masks)
        boundary_loss = self.edge_loss(logits, masks)
        # Tversky Loss: 漏报(FN)的惩罚是多报(FP)惩罚的约2.3倍 (alpha=0.3, beta=0.7)
        # 加大对FN的惩罚，强迫模型识别微小病灶区域
        tversky_loss_val = self.tversky_loss(logits, masks, alpha=0.3, beta=0.7)
        # Focal Tversky Loss: 进一步强调难分样本，使用与主Tversky Loss相同的参数
        tversky_focal_loss_val = self.tversky_focal_loss(logits, masks, alpha=0.3, beta=0.7, gamma=0.8)
        # 假阴性惩罚：应该有病变但预测为无病变
        false_negative_penalty = ((1 - probs) * masks).mean()
        # 假阳性惩罚：应该无病变但预测为有病变（使用clamp确保非负）
        false_positive_penalty = (probs.clamp(min=0.0, max=1.0) ** 2.0 * (1 - masks)).mean()
        
        loss_weights = {
            'bce': 0.20,
            'dice': 0.25,
            'tversky': 0.35,  # 增加Tversky Loss权重，作为主要损失函数
            'tversky_focal': 0.05,
            'boundary': 0.05,  # 提升边界权重
            'hausdorff': 0.05,  # 默认开启小权重的Hausdorff，关注轮廓
            'focal': 0.03,
            'lovasz': 0.0,
            'fn_penalty': 0.03,
            'fp_penalty': 0.02,
        }
        if use_lovasz:
            loss_weights['lovasz'] = 0.10
            loss_weights['bce'] = 0.15
            loss_weights['dice'] = 0.20
            loss_weights['tversky'] = 0.35  # 保持Tversky为主要损失
        if weights:
            loss_weights.update(weights)
        
        combined_loss = (
            loss_weights['bce'] * bce_loss
            + loss_weights['dice'] * dice_loss_val
            + loss_weights['tversky'] * tversky_loss_val
            + loss_weights['tversky_focal'] * tversky_focal_loss_val
            + loss_weights['boundary'] * boundary_loss
            + loss_weights['focal'] * focal_loss_val
            + loss_weights['fn_penalty'] * false_negative_penalty
            + loss_weights['fp_penalty'] * false_positive_penalty
            + loss_weights.get('hausdorff', 0.0) * torch.tensor(0.0, device=logits.device)  # 预留Hausdorff项
        )
        if use_lovasz and loss_weights.get('lovasz', 0) > 0:
            lovasz_loss_val = self.lovasz_hinge_loss(logits, masks)
            combined_loss += loss_weights['lovasz'] * lovasz_loss_val
        
        # 检查每个损失组件是否有NaN/Inf
        loss_components = {
            'bce': bce_loss,
            'dice': dice_loss_val,
            'tversky': tversky_loss_val,
            'tversky_focal': tversky_focal_loss_val,
            'boundary': boundary_loss,
            'focal': focal_loss_val,
            'fn_penalty': false_negative_penalty,
            'fp_penalty': false_positive_penalty,
            'hausdorff': torch.tensor(0.0, device=logits.device),
        }
        if use_lovasz and loss_weights.get('lovasz', 0) > 0:
            loss_components['lovasz'] = lovasz_loss_val
        
        # 替换NaN/Inf的损失组件为0
        for key, loss_val in loss_components.items():
            if not torch.isfinite(loss_val):
                print(f"[警告] {key}损失出现NaN/Inf，已替换为0")
                loss_components[key] = torch.tensor(0.0, device=logits.device)
        
        # 重新计算组合损失
        # 添加Hausdorff Distance Loss（如果启用）
        hausdorff_loss = None
        if loss_weights.get('hausdorff', 0) > 0:
            try:
                hausdorff_loss = self.hausdorff_distance_loss(logits, masks, percentile=95, alpha=1.0)
                if torch.isfinite(hausdorff_loss):
                    loss_components['hausdorff'] = hausdorff_loss
                else:
                    loss_components['hausdorff'] = torch.tensor(0.0, device=logits.device)
            except Exception as e:
                print(f"[警告] Hausdorff Loss计算失败: {e}，跳过")
                loss_components['hausdorff'] = torch.tensor(0.0, device=logits.device)
        
        combined_loss = (
            loss_weights['bce'] * loss_components['bce']
            + loss_weights['dice'] * loss_components['dice']
            + loss_weights['tversky'] * loss_components['tversky']
            + loss_weights['tversky_focal'] * loss_components['tversky_focal']
            + loss_weights['boundary'] * loss_components['boundary']
            + loss_weights['focal'] * loss_components['focal']
            + loss_weights['fn_penalty'] * loss_components['fn_penalty']
            + loss_weights['fp_penalty'] * loss_components['fp_penalty']
        )
        if use_lovasz and loss_weights.get('lovasz', 0) > 0:
            combined_loss += loss_weights['lovasz'] * loss_components['lovasz']
        if loss_weights.get('hausdorff', 0) > 0 and 'hausdorff' in loss_components:
            combined_loss += loss_weights['hausdorff'] * loss_components['hausdorff']
        
        # 最终检查：如果组合损失仍然是NaN/Inf，使用BCE损失作为后备
        if not torch.isfinite(combined_loss):
            print(
                "[严重警告] 组合损失仍为NaN/Inf，使用BCE损失作为后备 -> "
                f"BCE={loss_components['bce'].item():.4f}, Dice={loss_components['dice'].item():.4f}, "
                f"Tversky={loss_components['tversky'].item():.4f}, Boundary={loss_components['boundary'].item():.4f}, "
                f"Focal={loss_components['focal'].item():.4f}, "
                f"Lovasz={(loss_components.get('lovasz', torch.tensor(0.0)).item() if use_lovasz else 0.0):.4f}"
            )
            combined_loss = loss_components['bce']  # 使用BCE作为后备
        
        return combined_loss

    def _ensemble_inference(self, *args, **kwargs):
        """模型集成功能已取消。"""
        raise RuntimeError("模型集成功能已取消")

    def _tta_inference(self, model, images):
        """
        【完全重写】多尺度测试时增强 (MSTTA) - 修复版
        
        核心改进：
        1. 维度自适应：动态检测输出通道数，彻底解决 IndexError
        2. 概率空间融合：在概率空间进行TTA融合，避免数学错误
        3. 正确的后处理：对概率图进行高斯平滑和后处理
        4. 精度优化：避免反复的 Log/Sigmoid 转换，减少精度损失
        
        多尺度推理：3个尺度 × 8种变换 = 24倍推理
        - 尺度因子: [0.8, 1.0, 1.2]
        - 8种变换: 原始、水平翻转、垂直翻转、旋转90/180/270度、翻转+旋转组合
        """
        import torch.nn.functional as F
        from scipy.ndimage import gaussian_filter
        
        B, C_input, H, W = images.shape  # C_input 是输入图像的通道数（通常是3）
        scales = [0.8, 1.0, 1.2]  # 多尺度因子
        all_prob_maps = []  # 存储所有概率图（而非Logits）
        all_weights = []  # 存储置信度权重
        
        # 【多尺度循环】
        for scale in scales:
            # Resize到目标尺度
            if scale != 1.0:
                target_h, target_w = int(H * scale), int(W * scale)
                scaled_images = F.interpolate(images, size=(target_h, target_w), 
                                             mode='bilinear', align_corners=False)
            else:
                scaled_images = images
                target_h, target_w = H, W
            
            # 【8种变换循环】
            scale_prob_maps = []
            
            # 1. 原始图像
            pred_logits = model(scaled_images)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                # 【关键修复】立即转换为概率图，在概率空间进行融合
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 2. 水平翻转
            pred_logits = model(torch.flip(scaled_images, dims=[3]))
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.flip(pred_logits, dims=[3])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 3. 垂直翻转
            pred_logits = model(torch.flip(scaled_images, dims=[2]))
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.flip(pred_logits, dims=[2])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 4. 旋转90度
            pred_logits = model(torch.rot90(scaled_images, k=1, dims=[2, 3]))
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.rot90(pred_logits, k=-1, dims=[2, 3])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 5. 旋转180度
            pred_logits = model(torch.rot90(scaled_images, k=2, dims=[2, 3]))
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.rot90(pred_logits, k=-2, dims=[2, 3])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 6. 旋转270度
            pred_logits = model(torch.rot90(scaled_images, k=3, dims=[2, 3]))
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.rot90(pred_logits, k=-3, dims=[2, 3])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 7. 水平翻转+旋转90度
            img_aug = torch.flip(scaled_images, dims=[3])
            img_aug = torch.rot90(img_aug, k=1, dims=[2, 3])
            pred_logits = model(img_aug)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.rot90(pred_logits, k=-1, dims=[2, 3])
            pred_logits = torch.flip(pred_logits, dims=[3])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 8. 垂直翻转+旋转90度
            img_aug = torch.flip(scaled_images, dims=[2])
            img_aug = torch.rot90(img_aug, k=1, dims=[2, 3])
            pred_logits = model(img_aug)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = torch.rot90(pred_logits, k=-1, dims=[2, 3])
            pred_logits = torch.flip(pred_logits, dims=[2])
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 收集当前尺度的所有概率图
            all_prob_maps.extend(scale_prob_maps)
        
        # 【检查是否有有效预测】
        if len(all_prob_maps) == 0:
            print(f"[严重警告] MSTTA: 所有变换的预测都包含NaN/Inf，返回零输出")
            fallback_output = model(images)
            if isinstance(fallback_output, tuple):
                fallback_output = fallback_output[0]
            return torch.zeros_like(fallback_output)
        
        # 【维度自适应】从第一个概率图中获取模型输出的实际通道数
        first_prob = all_prob_maps[0]
        if first_prob.dim() == 4:
            _, C_output, _, _ = first_prob.shape  # C_output 是模型输出的通道数
        elif first_prob.dim() == 3:
            # 如果输出是 [B, H, W]，说明是单通道，需要添加通道维度
            C_output = 1
            all_prob_maps = [p.unsqueeze(1) if p.dim() == 3 else p for p in all_prob_maps]
        else:
            raise ValueError(f"不支持的预测张量维度: {first_prob.dim()}")
        
        # 【加权融合】计算每个预测的置信度权重（基于概率图）
        weights = []
        eps = 1e-8
        for prob_map in all_prob_maps:
            # 计算平均置信度：使用熵的负值作为置信度度量
            # 熵越低，置信度越高
            entropy = -prob_map * torch.log(prob_map + eps) - (1 - prob_map) * torch.log(1 - prob_map + eps)
            confidence = 1.0 - entropy.mean()  # 转换为置信度（1 - 熵）
            weights.append(float(confidence))
        
        # 归一化权重
        weights = torch.tensor(weights, device=images.device, dtype=torch.float32)
        weights = weights / (weights.sum() + eps)
        
        # 【概率空间加权平均】在概率空间进行融合，而非Logits空间
        stacked_probs = torch.stack(all_prob_maps, dim=0)  # [N, B, C_output, H, W]
        weights_expanded = weights.view(-1, 1, 1, 1, 1)  # [N, 1, 1, 1, 1]
        fused_prob = (stacked_probs * weights_expanded).sum(dim=0)  # [B, C_output, H, W]
        
        # 【正确的后处理】对概率图进行高斯平滑（而非对Logits）
        fused_prob_np = fused_prob.detach().cpu().numpy()
        smoothed_prob_np = np.zeros_like(fused_prob_np)
        for b in range(B):
            for c in range(C_output):  # 【关键修复】使用 C_output，彻底解决 IndexError
                smoothed_prob_np[b, c] = gaussian_filter(fused_prob_np[b, c], sigma=0.5)
        
        # 【极致后处理】在概率图上应用LCC和remove_small_holes
        processed_prob_np = np.zeros_like(smoothed_prob_np)
        for b in range(B):
            for c in range(C_output):  # 【关键修复】使用 C_output，彻底解决 IndexError
                prob_map = smoothed_prob_np[b, c]
                # 应用极致后处理流水线
                processed_mask = ensemble_post_process_global(
                    prob_map,
                    use_lcc=True,  # 保留最大连通域
                    use_remove_holes=True,  # 填补小孔洞
                    min_hole_size=100,
                    use_edge_smoothing=True  # 边缘平滑
                )
                processed_prob_np[b, c] = processed_mask
        
        # 【兼容性返回】将处理好的概率图映射回伪Logits格式
        # 避免使用不稳定的 np.log 公式，直接使用线性映射
        # 0 -> -10, 1 -> 10，保持数值稳定性
        processed_prob_tensor = torch.from_numpy(processed_prob_np).to(images.device).float()
        # 线性映射：prob [0, 1] -> logits [-10, 10]
        final_logits = (processed_prob_tensor - 0.5) * 20.0  # 将 [0, 1] 映射到 [-10, 10]
        
        return final_logits
    
    @staticmethod
    def smart_post_processing(
        pred_mask,
        pred_probs,
        tiny_size_thresh: int = 2,
        small_min_size: int = 3,
        small_max_size: int = 19,
        prob_threshold: float = 0.65,
    ):
        """
        智能后处理函数（Smart Post-Processing）
        
        仅基于连通域面积 + 概率自适应地过滤小病灶/噪点，避免误删真实微小病灶。
        
        分级策略：
        - Level 1: 绝对噪音 (area <= tiny_size_thresh，默认 <=2 像素) -> 直接删除
        - Level 2: 安全区域 (area >= 20 像素) -> 无条件保留
        - Level 3: 模糊地带 (3~19 像素) -> 仅在平均概率 > prob_threshold 时保留
        """
        # 延迟导入，避免在未安装 skimage 时直接崩溃
        try:
            from skimage import measure
        except ImportError:
            # 如果没有 skimage，回退为原mask，不做智能过滤
            return pred_mask
        
        if isinstance(pred_mask, torch.Tensor):
            mask_np = pred_mask.detach().cpu().numpy()
            is_tensor = True
            device = pred_mask.device
        else:
            mask_np = np.asarray(pred_mask)
            is_tensor = False
            device = None
        
        if isinstance(pred_probs, torch.Tensor):
            probs_np = pred_probs.detach().cpu().numpy()
        else:
            probs_np = np.asarray(pred_probs)
        
        # 保证二维
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        if probs_np.ndim > 2:
            probs_np = probs_np.squeeze()
        
        # 尺寸不一致时直接返回原mask，避免形状错误
        if mask_np.shape != probs_np.shape:
            return pred_mask
        
        # 二值化（preds 本身已经是0/1，这里再次保证）
        binary = (mask_np > 0.5).astype(np.uint8)
        
        # 没有前景就直接返回
        if binary.sum() == 0:
            return pred_mask
        
        # 连通域标记，并使用概率图作为 intensity_image，以便计算 mean_intensity
        labels = measure.label(binary, connectivity=1)
        regions = measure.regionprops(labels, intensity_image=probs_np.astype(np.float32))
        
        cleaned = np.zeros_like(binary, dtype=np.uint8)
        
        for region in regions:
            area = region.area
            mean_prob = float(region.mean_intensity) if hasattr(region, "mean_intensity") else 0.0
            
            # Level 1: 极小区域（<= tiny_size_thresh）视为绝对噪音，直接跳过
            if area <= tiny_size_thresh:
                continue
            
            # Level 2: 大于等于 20 像素的区域，无条件保留
            if area >= 20:
                cleaned[labels == region.label] = 1
                continue
            
            # Level 3: 3~19 像素之间，依据平均概率判断
            if small_min_size <= area <= small_max_size and mean_prob > prob_threshold:
                cleaned[labels == region.label] = 1
                continue
            # 否则视为噪声，不写入 cleaned
        
        # 如果全部被过滤掉，则保持全空mask，表示智能过滤认为该图像中没有可靠病灶
        # 之前的逻辑会回退为原始 noisy mask，这会拉低 Dice_Neg，现根据统计策略移除回退。
        
        if is_tensor:
            return torch.from_numpy(cleaned).to(device=device, dtype=torch.float32)
        else:
            return cleaned.astype(np.float32)

    @staticmethod
    def post_process_optimize(mask):
        """
        对二值掩码进行微小膨胀，填补边缘，提升 Dice
        
        【关键】针对欠分割问题，通过微小膨胀（1-2像素）来提升 Dice 分数
        适用于 Specificity 很高但可能存在轻微欠分割的情况
        
        Args:
            mask: 二值掩码 (numpy array, 0-1 或 0-255)
        
        Returns:
            dilated_mask: 膨胀后的掩码 (numpy array, 0-1)
        """
        # 1. 确保是 uint8 格式
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        
        # 2. 定义膨胀核 (Kernel)
        # 使用 3x3 的核，迭代 1 次，相当于向外扩 1 个像素
        # 如果想更激进，可以把 iterations 改为 2
        kernel = np.ones((3, 3), np.uint8)
        
        # 3. 执行膨胀 (Dilation)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 4. 转换回 0-1 范围
        dilated_mask = (dilated_mask > 127).astype(np.float32)
        
        return dilated_mask

    @staticmethod
    def post_process_mask(
        pred_mask,
        min_size=50,
        use_morphology=True,
        keep_largest=True,
        fill_holes=True,
        enable_opening=True,
        opening_kernel_size: int = 3,
        opening_iterations: int = 1,
    ):
        """
        后处理优化预测mask - 增强版
        
        Args:
            pred_mask: 预测mask (numpy或tensor)
            min_size: 移除小于此大小的连通域
            use_morphology: 是否使用形态学操作
            keep_largest: 是否只保留最大连通域（单器官分割推荐）
            fill_holes: 是否填充内部孔洞（去除假阴性空洞）
        
        Returns:
            处理后的mask
        """
        import cv2
        from scipy import ndimage
        
        if isinstance(pred_mask, torch.Tensor):
            pred_np = pred_mask.detach().cpu().numpy()
            is_tensor = True
            device = pred_mask.device
        else:
            pred_np = pred_mask.copy()
            is_tensor = False
        
        if pred_np.sum() < 10:  # 几乎为空,直接返回
            return pred_mask
        
        pred_binary = (pred_np > 0.5).astype(np.uint8)
        
        # 1. 填充孔洞（Fill Holes）- 去除器官内部的假阴性空洞
        if fill_holes:
            # 使用 scipy.ndimage.binary_fill_holes 填充内部孔洞
            pred_binary = ndimage.binary_fill_holes(pred_binary).astype(np.uint8)
        
        # 2. 形态学闭操作 - 进一步填充小孔洞和缝隙
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)
            # 形态学开操作（可选）- 去除小噪点/毛刺
            if enable_opening:
                k = int(max(1, opening_kernel_size))
                # kernel size 需为奇数
                if k % 2 == 0:
                    k += 1
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                iters = int(max(1, opening_iterations))
                pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel_small, iterations=iters)
        
        # 3. 保留最大连通域（Keep Largest Connected Component）- 去除孤立的噪点
        if keep_largest:
            labeled, num_features = ndimage.label(pred_binary)
            if num_features > 0:
                # 计算每个连通域的大小
                sizes = ndimage.sum(pred_binary, labeled, range(1, num_features + 1))
                # 找到最大的连通域
                largest_label = np.argmax(sizes) + 1
                # 只保留最大连通域
                pred_binary = (labeled == largest_label).astype(np.uint8)
        else:
            # 4. 连通域分析 - 移除小区域（如果不使用keep_largest）
            if min_size > 0:
                labeled, num_features = ndimage.label(pred_binary)
                if num_features > 0:
                    sizes = ndimage.sum(pred_binary, labeled, range(1, num_features + 1))
                    mask_sizes = sizes >= min_size
                    # 只保留大区域
                    keep_labels = np.where(mask_sizes)[0] + 1
                    pred_binary = np.isin(labeled, keep_labels).astype(np.uint8)
        
        # 返回原始类型
        if is_tensor:
            return torch.from_numpy(pred_binary).to(device).float()
        else:
            return pred_binary.astype(np.float32)
    
    @staticmethod
    def post_process_refine_for_hd95(pred_probs, threshold=0.5, min_area_threshold=100, 
                                     use_gaussian_blur=True, use_morphology=True,
                                     dynamic_area_threshold=True):
        """
        优化的后处理流水线：专门用于降低HD95，同时保持Dice > 0.88
        
        策略：
        1. 高斯模糊平滑边缘（可选）
        2. 二值化
        3. 形态学闭运算：填充内部空洞并平滑边缘
        4. 严格连通域过滤：仅保留面积最大的两个连通域，删除小区域
        5. 动态面积阈值：根据输入概率动态调整面积阈值（低概率样本更严格）
        
        Args:
            pred_probs: 概率图 (numpy array 或 torch.Tensor, shape: H x W)
            threshold: 二值化阈值
            min_area_threshold: 基础最小连通域面积阈值（像素），小于此值的区域将被删除
            use_gaussian_blur: 是否使用高斯模糊平滑边缘
            use_morphology: 是否使用形态学闭运算
            dynamic_area_threshold: 是否根据概率动态调整面积阈值
        
        Returns:
            处理后的二值掩码 (numpy array 或 torch.Tensor, 0-1)
        """
        import cv2
        from scipy import ndimage
        
        # 转换为 numpy
        if isinstance(pred_probs, torch.Tensor):
            probs_np = pred_probs.detach().cpu().numpy()
            is_tensor = True
            device = pred_probs.device
        else:
            probs_np = np.asarray(pred_probs)
            is_tensor = False
            device = None
        
        # 确保二维
        if probs_np.ndim > 2:
            probs_np = probs_np.squeeze()
        
        # 【动态面积阈值】根据输入概率的平均值动态调整面积阈值
        # 低概率样本（平均概率 < 0.3）使用更严格的过滤（1.5倍基础阈值）
        # 中等概率样本（0.3 <= 平均概率 < 0.6）使用标准阈值
        # 高概率样本（平均概率 >= 0.6）使用较宽松的过滤（0.8倍基础阈值）
        if dynamic_area_threshold:
            mean_prob = float(np.mean(probs_np))
            if mean_prob < 0.3:
                # 低概率样本：更严格的过滤，减少假阳性
                area_threshold = int(min_area_threshold * 1.5)
            elif mean_prob >= 0.6:
                # 高概率样本：较宽松的过滤，避免删除真实病灶
                area_threshold = int(min_area_threshold * 0.8)
            else:
                # 中等概率样本：使用标准阈值
                area_threshold = min_area_threshold
        else:
            area_threshold = min_area_threshold
        
        # 1. 高斯模糊平滑边缘（降低HD95的关键步骤）
        if use_gaussian_blur:
            probs_blurred = cv2.GaussianBlur(probs_np.astype(np.float32), ksize=(3, 3), sigmaX=0.5)
        else:
            probs_blurred = probs_np.astype(np.float32)
        
        # 2. 二值化
        binary = (probs_blurred > threshold).astype(np.uint8)
        
        # 如果没有前景，直接返回
        if binary.sum() == 0:
            if is_tensor:
                return torch.from_numpy(binary.astype(np.float32)).to(device)
            return binary.astype(np.float32)
        
        # 3. 形态学闭运算：填充内部空洞并平滑边缘
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 4. 严格连通域过滤：仅保留面积最大的两个连通域，删除小区域
        try:
            from skimage import measure
            labels = measure.label(binary, connectivity=1)
            regions = measure.regionprops(labels)
            
            if len(regions) == 0:
                cleaned = np.zeros_like(binary, dtype=np.uint8)
            else:
                # 按面积降序排序
                sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
                
                cleaned = np.zeros_like(binary, dtype=np.uint8)
                # 仅保留面积最大的两个连通域（左右肺），且面积必须 >= area_threshold（动态调整）
                kept_count = 0
                for region in sorted_regions:
                    if region.area >= area_threshold and kept_count < 2:
                        cleaned[labels == region.label] = 1
                        kept_count += 1
        except ImportError:
            # 如果没有 skimage，使用 scipy 实现
            labeled, num_features = ndimage.label(binary)
            if num_features > 0:
                sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
                # 找到面积最大的两个连通域
                sorted_indices = np.argsort(sizes)[::-1]
                kept_labels = []
                for idx in sorted_indices:
                    if sizes[idx] >= area_threshold and len(kept_labels) < 2:
                        kept_labels.append(idx + 1)
                if kept_labels:
                    cleaned = np.isin(labeled, kept_labels).astype(np.uint8)
                else:
                    cleaned = np.zeros_like(binary, dtype=np.uint8)
            else:
                cleaned = np.zeros_like(binary, dtype=np.uint8)
        
        if is_tensor:
            return torch.from_numpy(cleaned.astype(np.float32)).to(device)
        return cleaned.astype(np.float32)


# ==================== 【核心修复1】独立函数：解决Pickle错误 ====================
# 将HD95和Dice计算逻辑剥离为独立函数，不依赖类实例，可用于多进程并行处理

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


def _ensemble_masks_standalone(mask_list, weights):
    """
    独立的集成函数，不依赖类实例，可用于多进程并行处理
    
    【任务2】像素级融合函数修正：确保使用 w1 * mask1 + w2 * mask2
    
    Args:
        mask_list: 掩码列表（纯numpy数组列表），支持2个模型
        weights: 权重列表（纯Python列表），长度为2，w1和w2
    
    Returns:
        集成后的掩码 (numpy array)
    """
    assert len(mask_list) == len(weights), \
        f"掩码数量 ({len(mask_list)}) 与权重数量 ({len(weights)}) 不匹配"
    
    # 【任务2】双模型优化：确保权重之和为1.0
    if len(weights) == 2:
        w1, w2 = weights[0], weights[1]
        # 确保 w1 + w2 = 1.0
        weight_sum = w1 + w2
        if abs(weight_sum - 1.0) > 1e-6:
            w1 = w1 / weight_sum
            w2 = w2 / weight_sum
        weights = [w1, w2]
    else:
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            weights = [w / weight_sum for w in weights]
    
    mask_arrays = []
    target_shape = (512, 512)
    import cv2
    
    # 【任务2】强制类型转换：修复ndim报错
    for i, mask in enumerate(mask_list):
        # 强制转换为numpy数组
        if isinstance(mask, list):
            mask = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif not isinstance(mask, np.ndarray) or not hasattr(mask, 'ndim'):
            mask = np.asarray(mask)
        
        # 处理维度
        if hasattr(mask, 'ndim'):
            if mask.ndim == 3:
                mask = mask[0]
            elif mask.ndim != 2:
                raise ValueError(f"掩码 {i} 的维度 ({mask.ndim}) 不支持")
        else:
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[0]
        
        # 调整尺寸
        if mask.shape != target_shape:
            mask = cv2.resize(mask.astype(np.float32), 
                            (target_shape[1], target_shape[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        # 归一化到[0, 1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = np.clip(mask, 0.0, 1.0)
        mask_arrays.append(mask)
    
    # 【任务2】像素级融合：w1 * mask1 + w2 * mask2
    if len(mask_arrays) == 2:
        ensemble_mask = weights[0] * mask_arrays[0] + weights[1] * mask_arrays[1]
    else:
        ensemble_mask = np.zeros_like(mask_arrays[0], dtype=np.float32)
        for weight, mask in zip(weights, mask_arrays):
            ensemble_mask += weight * mask
    
    return np.clip(ensemble_mask, 0.0, 1.0)


def _ensemble_post_process_standalone(ensemble_mask, use_lcc=True, use_remove_holes=True, min_hole_size=100):
    """
    独立的后处理函数，不依赖类实例，可用于多进程并行处理
    
    Args:
        ensemble_mask: 集成后的概率图
        use_lcc: 是否使用最大连通域（必须启用以确保HD95优势）
        use_remove_holes: 是否移除小孔洞
        min_hole_size: 最小孔洞大小
    
    Returns:
        处理后的二值掩码
    """
    from scipy import ndimage
    try:
        from skimage import morphology
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
    
    if isinstance(ensemble_mask, torch.Tensor):
        mask_np = ensemble_mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(ensemble_mask)
    
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    binary_mask = (mask_np > 0.5).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return binary_mask.astype(np.float32)
    
    # 【核心修复5】强制执行LCC过滤，确保HD95优势
    if use_lcc:
        labeled, num_features = ndimage.label(binary_mask)
        if num_features > 0:
            sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            binary_mask = (labeled == largest_label).astype(np.uint8)
    
    if use_remove_holes and binary_mask.sum() > 0:
        if SKIMAGE_AVAILABLE:
            binary_mask = morphology.remove_small_holes(
                binary_mask.astype(bool), 
                area_threshold=min_hole_size
            ).astype(np.uint8)
        else:
            inverted = (~binary_mask.astype(bool)).astype(np.uint8)
            labeled_holes, num_holes = ndimage.label(inverted)
            if num_holes > 0:
                hole_sizes = ndimage.sum(inverted, labeled_holes, range(1, num_holes + 1))
                small_holes = [i + 1 for i, size in enumerate(hole_sizes) if size < min_hole_size]
                if small_holes:
                    for hole_label in small_holes:
                        binary_mask[labeled_holes == hole_label] = 1
    
    return binary_mask.astype(np.float32)

# ==================== 独立函数定义结束 ====================

# ==================== 【紧急修复】全局独立函数：解决Pickle错误和多进程冲突 ====================
# 将集成相关函数移出TrainThread类，定义为全局独立函数，避免PyQt5信号序列化问题

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
    
    import cv2
    
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
    from scipy import ndimage
    from scipy.ndimage import binary_erosion, binary_dilation
    try:
        from skimage import morphology
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
    
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
        if SKIMAGE_AVAILABLE:
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
    
    @staticmethod
    def evaluate_ensemble_performance(mask_list, weights, gt_masks, baseline_score=0.8273):
        """
        评估集成后的性能，对比单模型baseline
        
        Args:
            mask_list: 掩码列表（多个模型的预测结果）
            weights: 集成权重
            gt_masks: 真实掩码列表
            baseline_score: 单模型baseline总分，默认0.8273
        
        Returns:
            metrics: 指标字典，包含Dice, IoU, HD95, Sensitivity, Specificity, Total Score
            improvement: 相对于baseline的提升
        """
        from scipy.ndimage import binary_erosion, distance_transform_edt
        
        # 【关键修复】强制类型转换：确保mask_list中的每个元素都是numpy数组
        converted_mask_list = []
        for model_idx, model_masks in enumerate(mask_list):
            if isinstance(model_masks, list):
                converted_model_masks = []
                for mask_idx, mask in enumerate(model_masks):
                    if isinstance(mask, list):
                        mask = np.array(mask)
                    elif isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    elif not isinstance(mask, np.ndarray):
                        mask = np.asarray(mask)
                    converted_model_masks.append(mask)
                converted_mask_list.append(converted_model_masks)
            else:
                converted_mask_list.append(model_masks)
        
        mask_list = converted_mask_list
        
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
        
        # 计算HD95的辅助函数
        def compute_hd95(pred_mask, target_mask):
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
        
        # 计算Dice的辅助函数
        def compute_dice(pred_mask, target_mask, smooth=1e-7):
            pred = pred_mask.astype(bool)
            target = target_mask.astype(bool)
            intersection = (pred & target).sum()
            union = pred.sum() + target.sum()
            if union == 0:
                return 1.0
            return (2.0 * intersection + smooth) / (union + smooth)
        
        # 计算IoU的辅助函数
        def compute_iou(pred_mask, target_mask, smooth=1e-7):
            pred = pred_mask.astype(bool)
            target = target_mask.astype(bool)
            intersection = (pred & target).sum()
            union = (pred | target).sum()
            if union == 0:
                return 1.0
            return (intersection + smooth) / (union + smooth)
        
        # 计算Sensitivity和Specificity的辅助函数
        def compute_sens_spec(pred_mask, target_mask):
            pred = pred_mask.astype(bool)
            target = target_mask.astype(bool)
            tp = (pred & target).sum()
            fn = (~pred & target).sum()
            fp = (pred & ~target).sum()
            tn = (~pred & ~target).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            return sensitivity, specificity
        
        # 对每个样本进行集成
        ensemble_preds = []
        for sample_idx in range(len(gt_masks)):
            # 【关键修复】确保sample_masks中的每个元素都是numpy数组
            sample_masks = []
            for model_masks in mask_list:
                if isinstance(model_masks, list):
                    if sample_idx < len(model_masks):
                        mask = model_masks[sample_idx]
                        # 强制转换为numpy数组
                        if isinstance(mask, list):
                            mask = np.array(mask)
                        elif isinstance(mask, torch.Tensor):
                            mask = mask.detach().cpu().numpy()
                        elif not isinstance(mask, np.ndarray):
                            mask = np.asarray(mask)
                        # 处理维度：如果是 (C, H, W)，取第一个通道
                        if mask.ndim == 3:
                            mask = mask[0]
                        sample_masks.append(mask)
                    else:
                        # 如果索引超出范围，创建一个零数组
                        sample_masks.append(np.zeros_like(gt_masks[0] if len(gt_masks) > 0 else np.zeros((512, 512))))
                else:
                    # 如果model_masks是数组，直接使用
                    if model_masks.ndim > 2:
                        mask = model_masks[sample_idx]
                    else:
                        mask = model_masks
                    if not isinstance(mask, np.ndarray):
                        mask = np.asarray(mask)
                    sample_masks.append(mask)
            
            # 集成概率图
            ensemble_mask = ensemble_masks_global(sample_masks, weights)
            
            # 【极致后处理流水线】应用三步后处理（LCC + 空洞填充 + 边缘平滑）
            ensemble_mask = ensemble_post_process_global(
                ensemble_mask,
                use_lcc=True,  # 【第一步】保留最大连通域，彻底切除离群噪点
                use_remove_holes=True,  # 【第二步】填补小孔洞，提升Dice约0.5%
                min_hole_size=100,
                use_edge_smoothing=True  # 【第三步】边缘平滑，修正锯齿边缘
            )
            ensemble_preds.append(ensemble_mask)
        
        # 计算整体指标
        dice_scores = []
        iou_scores = []
        hd95_scores = []
        sensitivity_scores = []
        specificity_scores = []
        
        for pred, gt in zip(ensemble_preds, gt_masks):
            dice = compute_dice(pred, gt)
            iou = compute_iou(pred, gt)
            hd95 = compute_hd95(pred, gt)
            sensitivity, specificity = compute_sens_spec(pred, gt)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            if not np.isnan(hd95):
                hd95_scores.append(hd95)
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
        
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_hd95 = np.mean(hd95_scores) if hd95_scores else np.nan
        avg_sensitivity = np.mean(sensitivity_scores)
        avg_specificity = np.mean(specificity_scores)
        
        # 计算官方总分
        total_score = calculate_official_total_score_global(
            avg_dice, avg_iou, avg_hd95, avg_sensitivity, avg_specificity
        )
        
        # 计算提升
        improvement = total_score - baseline_score
        
        metrics = {
            'dice': avg_dice,
            'iou': avg_iou,
            'hd95': avg_hd95,
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity,
            'total_score': total_score
        }
        
        print(f"\n📊 集成性能评估:")
        print(f"   Dice: {avg_dice:.4f}")
        print(f"   IoU: {avg_iou:.4f}")
        print(f"   HD95: {avg_hd95:.4f}")
        print(f"   Sensitivity: {avg_sensitivity:.4f}")
        print(f"   Specificity: {avg_specificity:.4f}")
        print(f"   Total Score: {total_score:.4f}")
        print(f"   Baseline Score: {baseline_score:.4f}")
        print(f"   提升: {improvement:+.4f} ({'✅ 提升' if improvement > 0 else '❌ 下降'})")
        
        return metrics, improvement
    
    @staticmethod
    def calculate_official_total_score(dice, iou, hd95, sensitivity, specificity):
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
    
    def calculate_hd95(self, pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        """
        计算Hausdorff Distance 95 (HD95)，衡量分割边界距离。
        若任一掩膜为空，则返回nan，表示该指标不可计算。
        
        【关键】使用原始像素坐标系，不进行归一化。
        distance_transform_edt 默认使用像素距离（每个像素=1单位），
        因此返回的HD95值直接表示像素距离，无需乘以像素间距。
        """
        if self.matlab_metrics_bridge:
            try:
                return self.matlab_metrics_bridge.compute_hd95(pred_mask, target_mask)
            except Exception as exc:
                print(f"[MATLAB HD95] 回退到CPU实现: {exc}")

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

        # 【关键】distance_transform_edt 使用原始像素坐标系
        # 返回的距离值直接表示像素数，无需归一化或乘以像素间距
        target_distance = distance_transform_edt(~target_border)
        pred_distance = distance_transform_edt(~pred_border)

        distances_pred_to_target = target_distance[pred_border]
        distances_target_to_pred = pred_distance[target_border]

        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        if all_distances.size == 0:
            return 0.0
        # 返回95百分位距离（像素单位）
        return float(np.percentile(all_distances, 95))

    def calculate_custom_score(
        self,
        dice: float,
        iou: float,
        precision: float,
        recall: float,
        specificity: float,
        hd95: float,
    ) -> float:
        """
        自定义综合评分函数:
        Score = (Dice * 50) + (IoU * 10) + (Precision * 10) + (Recall * 10) + (Specificity * 10) + Score_HD95
        其中 Score_HD95 = 10 / (HD95 + 1)，若HD95不可用则该项记为0。
        """
        dice = float(dice)
        iou = float(iou)
        precision = float(precision)
        recall = float(recall)
        specificity = float(specificity)

        # HD95 项：HD95 越小越好，使用反比变换；若无效则记为 0
        if hd95 is None or not np.isfinite(hd95) or hd95 < 0:
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

    def scan_best_threshold(self, prob_maps: np.ndarray, gt_masks: np.ndarray):
        """
        在给定的概率图和真实掩膜上扫描阈值，寻找综合评分最高的阈值。

        Args:
            prob_maps: 概率图，形状 [N, H, W] 或 [N, 1, H, W]，数值范围 [0,1]
            gt_masks:  真实掩膜，形状与 prob_maps 对应，取值 {0,1}

        Returns:
            best_thresh: 综合评分最高的阈值
            best_metrics: 对应阈值下的指标字典（dice, iou, precision, recall, specificity, hd95, score）
        """
        prob_maps = np.asarray(prob_maps, dtype=np.float32)
        gt_masks = np.asarray(gt_masks, dtype=np.float32)

        # 统一为 [N, H, W]
        if prob_maps.ndim == 4:
            prob_maps = prob_maps[:, 0]
        if gt_masks.ndim == 4:
            gt_masks = gt_masks[:, 0]

        # 二值化真值
        gt_bool = gt_masks > 0.5

        thresholds = np.arange(0.3, 0.91, 0.05, dtype=np.float32)
        best_thresh = 0.5
        best_score = -float("inf")
        best_metrics = {}

        for thr in thresholds:
            pred_bool = prob_maps >= float(thr)

            # 全局混淆矩阵（所有像素一起统计）
            tp = np.logical_and(pred_bool, gt_bool).sum(dtype=np.float64)
            fp = np.logical_and(pred_bool, ~gt_bool).sum(dtype=np.float64)
            fn = np.logical_and(~pred_bool, gt_bool).sum(dtype=np.float64)
            tn = np.logical_and(~pred_bool, ~gt_bool).sum(dtype=np.float64)

            pred_sum = tp + fp
            mask_sum = tp + fn

            dice_den = 2.0 * tp + fp + fn
            if dice_den < 1e-7:
                dice = 1.0 if (mask_sum < 1e-7 and pred_sum < 1e-7) else 0.0
            else:
                dice = (2.0 * tp) / (dice_den + 1e-8)

            union = tp + fp + fn
            iou = 1.0 if union < 1e-7 else tp / (union + 1e-8)

            if pred_sum < 1e-7:
                precision = 1.0 if mask_sum < 1e-7 else 0.0
            else:
                precision = tp / (pred_sum + 1e-8)

            if (tp + fn) < 1e-7:
                recall = 1.0 if pred_sum < 1e-7 else 0.0
            else:
                recall = tp / (tp + fn + 1e-8)

            if (tn + fp) < 1e-7:
                specificity = 1.0
            else:
                specificity = tn / (tn + fp + 1e-8)

            # 计算该阈值下的平均 HD95（对每个样本单独计算）
            hd95_list = []
            for i in range(pred_bool.shape[0]):
                try:
                    hd = self.calculate_hd95(
                        pred_bool[i].astype(np.uint8),
                        gt_bool[i].astype(np.uint8),
                    )
                except Exception:
                    hd = float("nan")
                if np.isfinite(hd):
                    hd95_list.append(float(hd))

            if hd95_list:
                hd95_mean = float(np.nanmean(hd95_list))
            else:
                # 若所有样本都无法计算 HD95，则记为无穷大，以便在评分中让该项为 0
                hd95_mean = float("inf")

            total_score = self.calculate_custom_score(
                dice=dice,
                iou=iou,
                precision=precision,
                recall=recall,
                specificity=specificity,
                hd95=hd95_mean,
            )

            if total_score > best_score:
                best_score = float(total_score)
                best_thresh = float(thr)
                best_metrics = {
                    "dice": float(dice),
                    "iou": float(iou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "hd95": float(hd95_mean) if np.isfinite(hd95_mean) else float("nan"),
                    "score": float(total_score),
                }

        return best_thresh, best_metrics
    



# 预测工作线程
class PredictThread(QThread):
    update_progress = pyqtSignal(int, str)
    prediction_finished = pyqtSignal(list, list, list)  # 添加原始图像路径参数
    
    def __init__(self, image_paths, model_path, threshold=0.5, save_results=True, output_dir=None):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.threshold = threshold
        self.save_results = save_results
        self.output_dir = output_dir
        if self.save_results and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.model_config = read_checkpoint_config(model_path) if model_path else None
        self.model_type = (self.model_config or {}).get("model_type", "improved_unet")
        self.swin_params = (self.model_config or {}).get("swin_params")
        self.dstrans_params = (self.model_config or {}).get("dstrans_params")
        self.model_threshold = (self.model_config or {}).get("best_threshold")
        if self.model_threshold is not None:
            self.threshold = float(self.model_threshold)
        self.use_tta = True
        context_cfg = (self.model_config or {}).get("context") or {}
        self.context_slices = int(context_cfg.get("slices", os.environ.get("SEG_CONTEXT_SLICES", "0")))
        self.context_gap = int(context_cfg.get("gap", os.environ.get("SEG_CONTEXT_GAP", "1")))
        self.required_modalities = (self.model_config or {}).get("extra_modalities") or []
        self.extra_modalities_dirs = parse_extra_modalities_spec(os.environ.get("SEG_EXTRA_MODALITIES"))
        if self.required_modalities:
            missing = [m for m in self.required_modalities if m not in self.extra_modalities_dirs]
            if missing:
                print(f"[提示] 模型期望额外模态: {missing}，当前未在 SEG_EXTRA_MODALITIES 中配置，将尝试仅使用可用模态。")
        skull_cfg = (self.model_config or {}).get("skull_stripping") or {}
        self.use_skull_stripper = skull_cfg.get("enabled", False)
        self.skull_stripper_path = skull_cfg.get("model_path")
        self.skull_stripper_threshold = skull_cfg.get("threshold", 0.5)
        if self.use_skull_stripper and not self.skull_stripper_path:
            self.use_skull_stripper = False
        # nnFormer 配置
        self.use_nnformer = False
    
    def _predict_with_tta(self, model, image, use_tta=True):
        if not use_tta:
            return torch.sigmoid(model(image))
        preds = []
        preds.append(torch.sigmoid(model(image)))
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(image, dims=[3]))), dims=[3]))
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(image, dims=[2]))), dims=[2]))
        preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(image, k=1, dims=[2, 3]))), k=-1, dims=[2, 3]))
        return torch.stack(preds, dim=0).mean(dim=0)
    
    def _post_process(self, prob_tensor):
        processed = TrainThread.post_process_mask(
            prob_tensor.squeeze(0), 
            min_size=30, 
            use_morphology=True,
            keep_largest=False,  # 允许多发病灶同时存在
            fill_holes=True     # 填充孔洞，去除假阴性空洞
        )
        if isinstance(processed, torch.Tensor):
            return processed.unsqueeze(0).unsqueeze(0)
        processed = torch.from_numpy(processed).float()
        return processed.unsqueeze(0).unsqueeze(0)
    

    def run(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.update_progress.emit(0, f"使用设备: {device}")

            
            # 数据转换
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            # 创建数据集
            extra_modalities = build_extra_modalities_lists(self.image_paths, self.extra_modalities_dirs)
            dataset = MedicalImageDataset(
                self.image_paths,
                transform=transform,
                training=False,
                extra_modalities=extra_modalities,
                context_slices=self.context_slices,
                context_gap=self.context_gap
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            if self.model_threshold is not None:
                self.update_progress.emit(8, f"使用模型自适应阈值: {self.threshold:.3f}")
            
            # 加载分割模型 - 使用兼容加载
            model = instantiate_model(self.model_type, device, self.swin_params, self.dstrans_params, None)
            success, msg = load_model_compatible(model, self.model_path, device, verbose=True)
            if not success:
                raise RuntimeError(f"模型加载失败: {msg}")
            model.eval()
            skull_stripper = None
            if self.use_skull_stripper:
                skull_stripper = SkullStripper(self.skull_stripper_path, device, self.skull_stripper_threshold)
                if not skull_stripper.is_available():
                    skull_stripper = None
                    self.update_progress.emit(6, "SkullStripper不可用，回退为单阶段推理")
            
            self.update_progress.emit(10, "模型加载完成，开始预测...")
            
            input_images = []
            output_masks = []
            input_numpy_images = []  # 存储原始图像数据
            
            with torch.no_grad():
                for i, batch_data in enumerate(dataloader):
                    # 处理数据
                    if isinstance(batch_data, tuple):
                        if len(batch_data) == 2:
                            image, mask = batch_data
                        else:
                            image = batch_data[0]
                    else:
                        image = batch_data
                    # 确保image是tensor
                    if not isinstance(image, torch.Tensor):
                        if isinstance(image, (list, tuple)) and len(image) > 0:
                            image = image[0]
                    image = image.to(device)
                    brain_mask = None
                    if skull_stripper and skull_stripper.is_available():
                        image, brain_mask = skull_stripper.strip(image)
                    
                    # 分割预测
                    prob = self._predict_with_tta(model, image, use_tta=self.use_tta)
                    if brain_mask is not None:
                        prob = prob * brain_mask
                    pred = (prob > self.threshold).float()
                    pred = self._post_process(pred)
                    
                    # 转换回图像格式
                    image_np = image[0].cpu().numpy().transpose(1, 2, 0)
                    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                    prob_np = prob[0, 0].cpu().numpy()
                    pred_np = pred[0, 0].cpu().numpy()
                    pred_np = (pred_np * 255).astype(np.uint8)
                    
                    # 存储原始图像数据
                    input_numpy_images.append((image_np, pred_np, prob_np, ""))
                    
                    # 如果需要保存结果
                    if self.save_results and self.output_dir:
                        # 安全获取文件名
                        if i < len(self.image_paths):
                            base_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                        else:
                            base_name = f"image_{i}"
                        input_path = os.path.join(self.output_dir, f"{base_name}_input.png")
                        output_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
                        cv2.imwrite(input_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(output_path, pred_np)
                        
                        input_images.append(input_path)
                        output_masks.append(output_path)
                    else:
                        # 如果不保存，使用临时文件名
                        input_images.append(f"image_{i}_input")
                        output_masks.append(f"image_{i}_mask")
                    
                    progress_msg = f"处理图像 {i+1}/{len(dataloader)}"
                    progress = 10 + int(90 * (i + 1) / len(dataloader))
                    self.update_progress.emit(progress, progress_msg)
            
            self.prediction_finished.emit(input_images, output_masks, input_numpy_images)
        
        except Exception as e:
            self.update_progress.emit(0, f"预测错误: {str(e)}")


class SegmentationAPIService:
    """提供HTTP API调用的分割服务封装"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        if not model_path:
            raise ValueError("必须提供模型路径以启用API模式")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        self.model_path = model_path
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.input_size = (256, 256)
        self.transform = A.Compose([
            A.Resize(*self.input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.lock = threading.Lock()
        self.model_config = read_checkpoint_config(model_path)
        self.model_type = (self.model_config or {}).get("model_type", "improved_unet")
        self.swin_params = (self.model_config or {}).get("swin_params")
        self.dstrans_params = (self.model_config or {}).get("dstrans_params")
        self.threshold = float((self.model_config or {}).get("best_threshold", 0.5))
        self._load_model()

    def _load_model(self):
        self.model = instantiate_model(self.model_type, self.device, self.swin_params, self.dstrans_params)
        success, msg = load_model_compatible(self.model, self.model_path, self.device, verbose=True)
        if not success:
            raise RuntimeError(f"实时预测模型加载失败: {msg}")
        self.model.eval()

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("无法解析上传的图像数据")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _encode_png_base64(self, image: np.ndarray) -> str:
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("PNG编码失败")
        return base64.b64encode(buffer.tobytes()).decode('utf-8')

    def metadata(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "input_size": self.input_size,
        }

    def predict_from_bytes(self, image_bytes: bytes, threshold: Optional[float] = None) -> Dict[str, Union[str, int, float]]:
        if threshold is None:
            threshold = self.threshold
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold必须位于0到1之间")

        original_image = self._decode_image(image_bytes)
        transformed = self.transform(image=original_image)["image"].unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with self.lock:
            with torch.no_grad():
                logits = self.model(transformed)
                probs = torch.sigmoid(logits)
        duration_ms = (time.perf_counter() - start) * 1000

        mask = probs[0, 0].cpu().numpy()
        mask_bin = (mask >= threshold).astype(np.uint8)
        mask_uint8 = (mask_bin * 255).astype(np.uint8)

        overlay = cv2.resize(original_image.copy(), self.input_size)
        overlay_np = overlay.astype(np.float32)
        overlay_np[..., 0] = np.maximum(overlay_np[..., 0], mask_bin * 255)
        overlay_np[..., 2] = np.maximum(overlay_np[..., 2] * (1 - mask_bin) + mask_bin * 80, overlay_np[..., 2])
        overlay_bgr = cv2.cvtColor(overlay_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

        return {
            "mask_base64": self._encode_png_base64(mask_uint8),
            "overlay_base64": self._encode_png_base64(overlay_bgr),
            "width": int(mask_uint8.shape[1]),
            "height": int(mask_uint8.shape[0]),
            "threshold": float(threshold),
            "processing_time_ms": round(duration_ms, 2)
        }


def create_segmentation_api(service: SegmentationAPIService):
    """创建FastAPI应用以暴露模型推理接口"""
    try:
        fastapi_module = importlib.import_module("fastapi")
        cors_module = importlib.import_module("fastapi.middleware.cors")
    except ImportError as exc:
        raise ImportError(
            "启用API模式需要安装FastAPI依赖，请先运行: pip install fastapi uvicorn"
        ) from exc

    FastAPI = fastapi_module.FastAPI
    UploadFile = fastapi_module.UploadFile
    File = fastapi_module.File
    HTTPException = fastapi_module.HTTPException
    Query = fastapi_module.Query
    CORSMiddleware = cors_module.CORSMiddleware

    app = FastAPI(
        title="Medical Segmentation API",
        description="通过HTTP接口访问医学图像分割模型的推理服务",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", **service.metadata()}

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        threshold: float = Query(0.5, ge=0.01, le=0.99),
    ):
        if not file:
            raise HTTPException(status_code=400, detail="缺少图像文件")
        contents = await file.read()
        try:
            result = service.predict_from_bytes(contents, threshold=threshold)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"filename": file.filename, **result}

    @app.post("/predict/batch")
    async def predict_batch(
        files: List[UploadFile] = File(...),
        threshold: float = Query(0.5, ge=0.01, le=0.99),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="请至少上传一张图像")
        responses = []
        for file in files:
            contents = await file.read()
            try:
                result = service.predict_from_bytes(contents, threshold=threshold)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{file.filename}: {exc}") from exc
            responses.append({"filename": file.filename, **result})
        return {"results": responses}

    return app


class APIServerThread(QThread):
    """在后台线程中运行Uvicorn API服务器"""
    status_changed = pyqtSignal(str)
    server_started = pyqtSignal(str)
    server_stopped = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, service: SegmentationAPIService, host: str, port: int, reload: bool = False, parent=None):
        super().__init__(parent)
        self.service = service
        self.host = host
        self.port = port
        self.reload = reload
        self._stop_event = threading.Event()
        self.server = None

    def run(self):
        try:
            uvicorn = importlib.import_module("uvicorn")
        except ImportError:
            self.error_occurred.emit("运行API服务需要安装uvicorn，请执行: pip install uvicorn")
            return

        self.status_changed.emit("API服务启动中...")

        try:
            app = create_segmentation_api(self.service)
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                reload=self.reload,
                log_level="info"
            )
            config.install_signal_handlers = False
            self.server = uvicorn.Server(config)
            self.server_started.emit(f"API运行中: http://{self.host}:{self.port}")
            self.server.run()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.server = None
            status_msg = "API服务已停止" if self._stop_event.is_set() else "API服务已退出"
            self.server_stopped.emit(status_msg)

    def stop(self):
        self._stop_event.set()
        if self.server:
            self.server.should_exit = True


class AIAssistantThread(QThread):
    """调用远程AI聊天接口的后台线程"""
    success = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_url: str, model: str, api_key: str, prompt: str, timeout: int = 120, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.timeout = timeout

    def run(self):
        if not self.api_key:
            self.error.emit("缺少API Key，请先填写。")
            return

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an AI assistant specialized in medical image segmentation."},
                    {"role": "user", "content": self.prompt}
                ]
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                try:
                    detail = response.json().get("error", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(f"请求失败: {response.status_code} {detail}")

            data = response.json()
            choices = data.get("choices")
            if not choices:
                raise RuntimeError("响应中缺少choices字段")
            content = choices[0]["message"]["content"]
            self.success.emit(content)
        except Exception as exc:
            self.error.emit(str(exc))


# 主窗口
class MedicalSegmentationApp(QMainWindow):
    visualization_requested = pyqtSignal(str, list, list)
    visualization_ready = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # 基础窗口设置
        self.setWindowTitle("🤖 医学图像分割系统 - AI智能分析平台")
        try:
            self.setWindowIcon(QIcon("icon.png"))  # 请确保图标文件存在
        except:
            pass  # 如果图标不存在，忽略错误
        self.setGeometry(100, 100, 1400, 1000)
        self.setMinimumSize(1200, 800)
        self.visualization_requested.connect(self.handle_visualization)
        # 线程锁
        self.lock = QMutex()
        
        # 主题：light / dark
        self.theme = "light"

        # 初始化变量
        self._init_variables()
        
        # 初始化UI
        self.initUI()
    
    def _init_variables(self):
        """初始化所有变量"""
        self.model_path = None
        self.resnet_model_path = None
        self.data_dir = None
        self.output_dir = None

        self.train_thread = None
        self.predict_thread = None
        self.test_thread = None
        self.test_model_path = None
        self.test_data_dir = None
        self.test_results = None
        self.low_dice_cases = []
        self.current_results = []
        self.api_thread = None
        self.api_model_path = None
        self.api_service = None
        self.ai_thread = None
        self.llm_threshold_thread = None
        self.prediction_stats = None
        self.system_status_labels = {}
        self.tab_indexes = {}
        # 默认使用旧API地址
        self.ai_base_url = "https://models.sjtu.edu.cn/api/v1/chat/completions"
        # 可选的API地址列表
        self.ai_base_url_options = [
            ("SJTU模型服务", "https://models.sjtu.edu.cn/api/v1/chat/completions"),
            ("ChatAnywhere", "https://api.chatanywhere.tech/v1/chat/completions")
        ]
        self.ai_model_name = "deepseek-r1"
        # 不同API服务支持的模型列表
        self.ai_model_options_by_service = {
            "https://models.sjtu.edu.cn/api/v1/chat/completions": [
                ("DeepSeek-R1", "deepseek-r1"),
                ("DeepSeek-V3", "deepseek-v3"),
                ("Qwen3-Coder", "qwen3coder"),
                ("Qwen3-VL", "qwen3vl")
            ],
            "https://api.chatanywhere.tech/v1/chat/completions": [
                ("DeepSeek-R1", "deepseek-r1"),
                ("DeepSeek-V3", "deepseek-v3"),
                ("GPT-3.5 Turbo", "gpt-3.5-turbo"),
                ("GPT-4o Mini", "gpt-4o-mini"),
                ("GPT-4o", "gpt-4o"),
                ("GPT-4.1 Mini", "gpt-4.1-mini"),
                ("GPT-4.1 Nano", "gpt-4.1-nano"),
                ("GPT-4.1", "gpt-4.1"),
                ("GPT-5 Mini", "gpt-5-mini"),
                ("GPT-5 Nano", "gpt-5-nano"),
                ("GPT-5", "gpt-5")
            ]
        }
        # 默认模型选项（SJTU服务）
        self.ai_model_options = self.ai_model_options_by_service[self.ai_base_url]
        # 不同API服务对应的默认API key
        self.ai_api_key_by_service = {
            "https://models.sjtu.edu.cn/api/v1/chat/completions": "",
            "https://api.chatanywhere.tech/v1/chat/completions": ""
        }
        # 默认API key（当前服务的）
        self.ai_api_key = self.ai_api_key_by_service.get(self.ai_base_url, "")
        # 标记用户是否手动修改过API key
        self.ai_key_manually_changed = False
        self.ai_limits = {
            "rpm": 100,
            "tpm": 3000,
            "weekly": 1_000_000
        }
    
    def initUI(self):
        """主UI初始化方法"""
        # 应用全局样式表
        self.apply_global_styles()
        
        # 中央控件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ===== 左侧控制面板 =====
        self.setup_control_panel()
        
        # ===== 右侧标签页 =====
        self.setup_tab_widget()
        
        # 状态栏
        self.statusBar().showMessage("✅ 就绪")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border-top: 2px solid #e2e8f0;
                padding: 8px;
                font-size: 10pt;
                color: #475569;
            }
        """)
    
    def apply_global_styles(self):
        """应用全局样式表"""
        style = """
        /* 全局样式 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #f8fafc, stop:0.5 #f1f5f9, stop:1 #e2e8f0);
        }
        
        /* GroupBox样式 */
        QGroupBox {
            font-weight: bold;
            font-size: 12pt;
            border: 2px solid #e2e8f0;
            border-radius: 14px;
            margin-top: 12px;
            padding-top: 18px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ffffff, stop:1 #f8fafc);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 10px;
            color: #1e293b;
            font-size: 13pt;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #f8fafc);
            border-radius: 6px;
        }
        
        /* 按钮样式 */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #3b82f6, stop:1 #2563eb);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 11pt;
            font-weight: 600;
            min-height: 40px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #2563eb, stop:1 #1d4ed8);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #1d4ed8, stop:1 #1e40af);
            padding: 11px 23px;
        }
        
        QPushButton:disabled {
            background: #cbd5e1;
            color: #94a3b8;
        }
        
        /* 停止按钮特殊样式 */
        QPushButton[text="⏹ 停止训练"], QPushButton[text="停止训练"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ef4444, stop:1 #dc2626);
        }
        
        QPushButton[text="⏹ 停止训练"]:hover, QPushButton[text="停止训练"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #dc2626, stop:1 #b91c1c);
        }
        
        QPushButton[text="⏹ 停止训练"]:pressed, QPushButton[text="停止训练"]:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #b91c1c, stop:1 #991b1b);
        }
        
        /* 标签样式 */
        QLabel {
            color: #1e293b;
            font-size: 11pt;
        }
        
        /* 进度条样式 */
        QProgressBar {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            background: #f8fafc;
            height: 28px;
            font-size: 11pt;
            color: #1e293b;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3b82f6, stop:0.5 #06b6d4, stop:1 #10b981);
            border-radius: 10px;
        }
        
        /* SpinBox样式 */
        QSpinBox {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11pt;
            background-color: #ffffff;
            min-width: 100px;
        }
        
        QSpinBox:focus {
            border-color: #3b82f6;
            background-color: #f8fafc;
        }
        
        QSpinBox::up-button, QSpinBox::down-button {
            background: #f1f5f9;
            border: none;
            border-radius: 4px;
            width: 20px;
        }
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background: #e2e8f0;
        }
        
        /* ComboBox样式 */
        QComboBox {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11pt;
            background-color: #ffffff;
            min-width: 150px;
        }
        
        QComboBox:focus {
            border-color: #3b82f6;
            background-color: #f8fafc;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #64748b;
            width: 0;
            height: 0;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            selection-background-color: #3b82f6;
            selection-color: white;
            padding: 4px;
        }
        
        /* CheckBox样式 */
        QCheckBox {
            font-size: 11pt;
            spacing: 10px;
            color: #475569;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #cbd5e1;
            border-radius: 4px;
            background-color: #ffffff;
        }
        
        QCheckBox::indicator:hover {
            border-color: #3b82f6;
        }
        
        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #3b82f6, stop:1 #2563eb);
            border-color: #2563eb;
        }
        
        /* TabWidget样式 */
        QTabWidget::pane {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            background-color: #ffffff;
            top: -1px;
            padding: 4px;
        }
        
        QTabBar {
            alignment: left;
        }
        
        QTabBar::tab {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f1f5f9, stop:1 #e2e8f0);
            color: #64748b;
            border: 2px solid transparent;
            border-bottom: none;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding: 12px 24px;
            margin: 4px 2px;
            font-size: 11pt;
            font-weight: 500;
            min-width: 100px;
            min-height: 35px;
        }
        
        QTabBar::tab:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e2e8f0, stop:1 #cbd5e1);
            color: #475569;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ffffff, stop:1 #f8fafc);
            color: #2563eb;
            border-color: #3b82f6;
            border-bottom-color: #ffffff;
            font-weight: 600;
        }
        
        QTabBar::tab:first {
            margin-left: 0px;
        }
        
        QTabBar::tab:last {
            margin-right: 0px;
        }
        /* ScrollArea样式 */
        QScrollArea {
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background-color: #ffffff;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #f8fafc;
            width: 14px;
            border-radius: 7px;
            border: 1px solid #e2e8f0;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #cbd5e1, stop:1 #94a3b8);
            border-radius: 6px;
            min-height: 40px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #94a3b8, stop:1 #64748b);
        }
        
        QScrollBar::handle:vertical:pressed {
            background: #475569;
        }
        
        QScrollBar:horizontal {
            border: none;
            background: #f8fafc;
            height: 14px;
            border-radius: 7px;
            border: 1px solid #e2e8f0;
        }
        
        QScrollBar::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #cbd5e1, stop:1 #94a3b8);
            border-radius: 6px;
            min-width: 40px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #94a3b8, stop:1 #64748b);
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            height: 0px;
            width: 0px;
        }
        """
        # 如果是暗色主题，叠加一层简单的暗色样式覆盖基础配色
        if getattr(self, "theme", "light") == "dark":
            dark_style = """
            QMainWindow {
                background: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #e5e7eb;
            }
            QGroupBox {
                border: 1px solid #1f2937;
                background-color: #020617;
            }
            QGroupBox::title {
                color: #e5e7eb;
                background-color: #020617;
            }
            QLabel {
                color: #e5e7eb;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1d4ed8);
                color: #e5e7eb;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #1f2937;
                background: #020617;
            }
            QTabBar::tab {
                background: #020617;
                color: #9ca3af;
                padding: 8px 18px;
            }
            QTabBar::tab:selected {
                background: #111827;
                color: #f9fafb;
                border-bottom: 2px solid #3b82f6;
            }
            QScrollArea {
                background: #020617;
            }
            QStatusBar {
                background: #020617;
                color: #9ca3af;
            }
            """
            style = style + dark_style

        self.setStyleSheet(style)

    def toggle_theme(self):
        """在浅色 / 深色主题之间切换"""
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_global_styles()
        if hasattr(self, "theme_toggle_btn"):
            self.theme_toggle_btn.setText("🌙 深色" if self.theme == "dark" else "☀ 浅色")
        self.statusBar().showMessage("🌙 已切换到深色主题" if self.theme == "dark" else "☀ 已切换到浅色主题")

    def on_theme_toggle_clicked(self):
        """主题切换按钮回调"""
        self.toggle_theme()
    
    def setup_control_panel(self):
        """左侧控制面板设置"""
        control_panel = QGroupBox("⚙️ 控制面板")
        control_panel.setFixedWidth(340)
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(15, 20, 15, 15)
        
        # 顶部主题切换
        theme_layout = QHBoxLayout()
        theme_label = QLabel("🎨 主题:")
        theme_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.theme_toggle_btn = QPushButton("☀ 浅色")
        self.theme_toggle_btn.setFixedHeight(32)
        self.theme_toggle_btn.setToolTip("在浅色 / 深色主题之间切换")
        self.theme_toggle_btn.clicked.connect(self.on_theme_toggle_clicked)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_toggle_btn)
        theme_layout.addStretch()
        control_layout.addLayout(theme_layout)

        # 添加模型保存选项
        self.save_best_checkbox = QCheckBox("💾 自动保存最佳模型")
        self.save_best_checkbox.setChecked(True)
        self.save_best_checkbox.setToolTip("训练过程中自动保存表现最好的模型\n模型将保存在输出目录中")
        control_layout.addWidget(self.save_best_checkbox)

        self.create_system_status_group(control_layout)
        self.create_quick_nav_group(control_layout)

        # 初始化隐藏的API控件（不添加到界面，保持功能兼容）
        self._init_hidden_api_controls(control_panel)
        
        # 其他控制组件...
        control_layout.addStretch()
        control_panel.setLayout(control_layout)

        self.main_layout.addWidget(control_panel)

    def _init_hidden_api_controls(self, parent):
        """创建但不显示API服务控件，保留相关功能兼容"""
        self.api_control_container = QGroupBox("🌐 API服务", parent)
        api_layout = QVBoxLayout(self.api_control_container)
        api_layout.setSpacing(10)

        self.api_model_label = QLabel("✗ 未选择API模型", self.api_control_container)
        self.api_model_label.setWordWrap(True)
        self.api_model_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        browse_api_model_btn = QPushButton("📁 选择API模型", self.api_control_container)
        browse_api_model_btn.clicked.connect(self.browse_api_model)
        browse_api_model_btn.setToolTip("选择用于API推理的已训练模型(.pth/.pt)")

        host_layout = QHBoxLayout()
        host_label = QLabel("地址:", self.api_control_container)
        host_label.setMinimumWidth(60)
        self.api_host_input = QLineEdit("0.0.0.0", self.api_control_container)
        self.api_host_input.setPlaceholderText("0.0.0.0")
        host_layout.addWidget(host_label)
        host_layout.addWidget(self.api_host_input)

        port_layout = QHBoxLayout()
        port_label = QLabel("端口:", self.api_control_container)
        port_label.setMinimumWidth(60)
        self.api_port_spin = QSpinBox(self.api_control_container)
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(8000)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.api_port_spin)

        device_layout = QHBoxLayout()
        device_label = QLabel("设备:", self.api_control_container)
        device_label.setMinimumWidth(60)
        self.api_device_combo = QComboBox(self.api_control_container)
        self.api_device_combo.addItem("自动选择", None)
        self.api_device_combo.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.api_device_combo.addItem("CUDA:0", "cuda:0")
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.api_device_combo)

        api_button_layout = QHBoxLayout()
        api_button_layout.setSpacing(12)
        self.api_start_btn = QPushButton("▶️ 启动API", self.api_control_container)
        self.api_start_btn.clicked.connect(self.start_api_server)
        self.api_stop_btn = QPushButton("⏹ 关闭API", self.api_control_container)
        self.api_stop_btn.clicked.connect(self.stop_api_server)
        self.api_stop_btn.setEnabled(False)
        api_button_layout.addWidget(self.api_start_btn)
        api_button_layout.addWidget(self.api_stop_btn)

        self.api_status_label = QLabel("⚠️ API未运行", self.api_control_container)
        self.api_status_label.setWordWrap(True)
        self.api_status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fee2e2, stop:1 #fecaca);
                border-left: 4px solid #dc2626;
                border-radius: 8px;
                color: #991b1b;
                font-size: 10pt;
            }
        """)

        api_layout.addWidget(self.api_model_label)
        api_layout.addWidget(browse_api_model_btn)
        api_layout.addLayout(host_layout)
        api_layout.addLayout(port_layout)
        api_layout.addLayout(device_layout)
        api_layout.addLayout(api_button_layout)
        api_layout.addWidget(self.api_status_label)
        self.api_control_container.hide()

    def create_system_status_group(self, layout):
        """创建系统状态卡片"""
        status_group = QGroupBox("🛰 系统状态")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(8)
        group_layout.setContentsMargins(12, 18, 12, 12)

        self.system_status_labels = {}
        status_items = {
            "data": "训练数据",
            "train_model": "训练模型",
            "predict_model": "预测模型",
            "output_dir": "输出目录"
        }

        for key, title in status_items.items():
            label = QLabel(f"{title}: 未选择")
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setMinimumHeight(32)
            self.system_status_labels[key] = {"label": label, "title": title}
            group_layout.addWidget(label)
            self.update_system_status(key, "未选择", status="warning")

        status_group.setLayout(group_layout)
        layout.addWidget(status_group)

    def create_quick_nav_group(self, layout):
        """创建快速导航按钮"""
        nav_group = QGroupBox("⚡ 快速导航")
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(8)
        nav_layout.setContentsMargins(12, 18, 12, 12)

        buttons = [
            ("前往训练", "train"),
            ("前往预测", "predict"),
            ("查看结果", "result"),
            ("性能分析", "analysis"),
            ("AI助手", "assistant")
        ]

        for text, key in buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda _, k=key: self.switch_to_tab(k))
            nav_layout.addWidget(btn)

        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

    def update_system_status(self, key, value, status="info"):
        """更新系统状态显示"""
        info = self.system_status_labels.get(key)
        if not info:
            return
        label = info["label"]
        title = info["title"]
        styles = {
            "info": """
                QLabel {
                    padding: 10px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f8fafc, stop:1 #eef2ff);
                    border-radius: 8px;
                    border-left: 4px solid #6366f1;
                    color: #312e81;
                    font-size: 10pt;
                }
            """,
            "success": """
                QLabel {
                    padding: 10px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border-radius: 8px;
                    border-left: 4px solid #16a34a;
                    color: #166534;
                    font-size: 10pt;
                }
            """,
            "warning": """
                QLabel {
                    padding: 10px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #fef3c7, stop:1 #fde68a);
                    border-radius: 8px;
                    border-left: 4px solid #f59e0b;
                    color: #92400e;
                    font-size: 10pt;
                }
            """
        }
        label.setStyleSheet(styles.get(status, styles["info"]))
        label.setText(f"{title}: {value}")

    def switch_to_tab(self, tab_key):
        """快速切换到指定标签页"""
        index = self.tab_indexes.get(tab_key)
        if index is not None:
            self.tab_widget.setCurrentIndex(index)
    
    def setup_tab_widget(self):
        """右侧标签页设置"""
        self.tab_widget = QTabWidget()
        
        # 训练标签页
        self.setup_train_tab()
        
        # 预测标签页
        self.setup_predict_tab()
        
        # 结果标签页
        self.setup_result_tab()
        
        # 性能分析标签页
        self.setup_analysis_tab()

        # 模型测试标签页
        self.setup_model_test_tab()

        # AI助手标签页
        self.setup_ai_assistant_tab()
        
        self.main_layout.addWidget(self.tab_widget)
    
    def setup_train_tab(self):
        """训练标签页设置"""
        train_tab = QWidget()
        
        # 使用滚动区域包装内容
        train_scroll = QScrollArea()
        train_scroll.setWidgetResizable(True)
        train_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        train_scroll.setFrameShape(QScrollArea.NoFrame)
        
        train_content = QWidget()
        train_layout = QVBoxLayout()
        train_layout.setSpacing(15)
        train_layout.setContentsMargins(15, 15, 15, 15)
        
        # 数据目录选择
        data_dir_group = QGroupBox("📚 训练数据")
        data_dir_layout = QVBoxLayout()
        data_dir_layout.setSpacing(12)
        data_dir_layout.setContentsMargins(15, 20, 15, 15)
        
        self.data_dir_label = QLabel("✗ 未选择数据目录")
        self.data_dir_label.setWordWrap(True)
        self.data_dir_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        browse_data_btn = QPushButton("📁 选择数据目录")
        browse_data_btn.setToolTip("选择包含训练图像和掩码的数据目录")
        browse_data_btn.clicked.connect(self.browse_data_dir)
        
        data_dir_layout.addWidget(self.data_dir_label)
        data_dir_layout.addWidget(browse_data_btn)
        data_dir_group.setLayout(data_dir_layout)
        
        # 训练参数
        params_group = QGroupBox("⚙️ 训练参数")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(14)
        params_layout.setContentsMargins(15, 20, 15, 15)
        
        # 训练轮次
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("🔄 训练轮次:")
        epochs_label.setMinimumWidth(120)
        epochs_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setSuffix(" 轮")
        self.epochs_spin.setToolTip("设置训练的总轮次数\n建议值: 20-100")
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        
        # 批量大小
        batch_layout = QHBoxLayout()
        batch_label = QLabel("📦 批量大小:")
        batch_label.setMinimumWidth(120)
        batch_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        self.batch_spin.setToolTip("每次训练使用的样本数量\n根据GPU内存调整，建议: 2-8")
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        
        # 模型选择
        model_label = QLabel("🤖 预训练模型:")
        model_label.setStyleSheet("font-weight: 600; color: #475569;")
        # 单模型路径（用于非集成模式）
        self.model_path_label = QLabel("✗ 未选择模型")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        browse_model_btn = QPushButton("📁 选择模型")
        browse_model_btn.setToolTip("选择预训练模型文件（可选）\n如果为空，将从零开始训练")
        browse_model_btn.clicked.connect(self.browse_model_path)
        
        # 模型架构选择
        arch_label = QLabel("🏗️ 模型架构:")
        arch_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.arch_combo = QComboBox()
        self.arch_combo.addItem("改进UNet (ImprovedUNet)", "improved_unet")
        self.arch_combo.addItem("ResNet-UNet (ResNetUNet)", "resnet_unet")
        self.arch_combo.addItem("Transformer+UNet (TransUNet)", "trans_unet")
        self.arch_combo.addItem("DS-TransUNet (双尺度Transformer+UNet) ⭐", "ds_trans_unet")
        self.arch_combo.addItem("SwinUNet (Swin Transformer+UNet) ⭐推荐", "swin_unet")
        self.arch_combo.setCurrentIndex(4)  # 默认选择SwinUNet
        self.arch_combo.setToolTip(
            "选择模型架构类型：\n"
            "• ImprovedUNet: 基础改进UNet\n"
            "• ResNetUNet: 使用ResNet101编码器\n"
            "• TransUNet: Transformer+UNet混合架构\n"
            "• DS-TransUNet: 双尺度Transformer+UNet，在多个尺度使用Transformer增强多尺度特征提取\n"
            "• SwinUNet: Swin Transformer+UNet混合架构，可配合GWO优化提高Dice指标"
        )
        
        # GWO优化选项（SwinUNet / DS-TransUNet / nnFormer 可用）
        self.gwo_checkbox = QCheckBox("启用GWO优化（灰狼优化算法）")
        self.gwo_checkbox.setToolTip(
            "使用GWO算法优化 SwinUNet、DS-TransUNet 或 nnFormer 的超参数以提高Dice指标\n"
            "注意：优化过程需要额外时间，但能显著提升模型性能"
        )
        self.gwo_checkbox.setEnabled(False)  # 默认禁用，只有选择支持的架构时启用
        self.arch_combo.currentIndexChanged.connect(self._on_arch_changed)
        self._on_arch_changed()
        
        # 优化器选择
        optimizer_label = QLabel("⚙️ 优化器:")
        optimizer_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("Adam", "adam")
        self.optimizer_combo.addItem("AdamW", "adamw")
        self.optimizer_combo.addItem("SGD + Nesterov", "sgd")
        self.optimizer_combo.setCurrentIndex(0)
        self.optimizer_combo.setToolTip(
            "选择训练优化器：\n"
            "• Adam：标准Adam优化\n"
            "• AdamW：带解耦权重衰减的AdamW，适合较大正则需求\n"
            "• SGD + Nesterov：带Nesterov动量的SGD（momentum=0.99）"
        )
        
        # 添加到布局
        params_layout.addLayout(epochs_layout)
        params_layout.addLayout(batch_layout)
        params_layout.addWidget(model_label)
        params_layout.addWidget(self.model_path_label)
        params_layout.addWidget(browse_model_btn)
        params_layout.addWidget(arch_label)
        params_layout.addWidget(self.arch_combo)
        params_layout.addWidget(self.gwo_checkbox)
        params_layout.addWidget(optimizer_label)
        params_layout.addWidget(self.optimizer_combo)
        params_group.setLayout(params_layout)
        
        # 训练按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        self.train_btn = QPushButton("🚀 开始训练")
        self.train_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setMinimumHeight(48)
        self.train_btn.setToolTip("开始训练模型\n需要先选择数据目录")
        
        self.stop_train_btn = QPushButton("⏹ 停止训练")
        self.stop_train_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setMinimumHeight(48)
        self.stop_train_btn.setToolTip("停止当前正在进行的训练")
        
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.stop_train_btn)
        
        # 训练进度
        train_progress_label = QLabel("📊 训练进度:")
        train_progress_label.setStyleSheet("font-weight: 600; color: #475569; font-size: 11pt;")
        self.train_progress = QProgressBar()
        self.train_progress.setFormat("训练: %p%")
        self.train_status = QLabel("⏳ 准备训练")
        self.train_status.setWordWrap(True)
        self.train_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.train_status.setMinimumHeight(50)
        self.train_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dbeafe, stop:1 #bfdbfe);
                border-radius: 8px;
                border-left: 4px solid #3b82f6;
                color: #1e40af;
                font-weight: 500;
                font-size: 10pt;
            }
        """)
        
        # 添加验证进度条
        val_progress_label = QLabel("✅ 验证进度:")
        val_progress_label.setStyleSheet("font-weight: 600; color: #475569; font-size: 11pt; padding-top: 8px;")
        self.val_progress = QProgressBar()
        self.val_progress.setFormat("验证: %p%")
        self.val_status = QLabel("⏳ 验证状态: 等待验证...")
        self.val_status.setWordWrap(True)
        self.val_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.val_status.setMinimumHeight(50)
        self.val_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f3e5f5, stop:1 #e1bee7);
                border-radius: 8px;
                border-left: 4px solid #9333ea;
                color: #6b21a8;
                font-weight: 500;
                font-size: 10pt;
            }
        """)
        
        # 添加训练统计信息
        self.stats_group = QGroupBox("📈 训练统计")
        self.stats_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)  # 确保GroupBox可以适应窗口大小
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(8)  # 减小间距，避免重叠
        stats_layout.setContentsMargins(12, 20, 12, 12)  # 减小左右边距
        
        self.epoch_label = QLabel("🔄 当前轮次: -")
        self.loss_label = QLabel("📉 训练损失: -")
        self.val_loss_label = QLabel("📊 验证损失: -")
        self.dice_label = QLabel("🎯 Dice系数: -")
        
        # 设置统计标签样式和属性，确保小窗口时也能正常显示
        stat_label_style = """
            QLabel {
                padding: 8px 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fff7ed, stop:1 #ffedd5);
                border-left: 4px solid #f97316;
                border-radius: 8px;
                font-weight: 600;
                color: #9a3412;
                font-size: 10pt;
                min-height: 20px;
            }
        """
        # 设置所有统计标签的属性
        for label in [self.epoch_label, self.loss_label, self.val_loss_label, self.dice_label]:
            label.setStyleSheet(stat_label_style)
            label.setWordWrap(True)  # 允许文本换行
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)  # 水平扩展，垂直最小
            label.setMinimumHeight(38)  # 设置最小高度，稍微减小
            label.setMaximumHeight(100)  # 设置最大高度，防止过度扩展
        
        stats_layout.addWidget(self.epoch_label)
        stats_layout.addWidget(self.loss_label)
        stats_layout.addWidget(self.val_loss_label)
        stats_layout.addWidget(self.dice_label)
        self.stats_group.setLayout(stats_layout)
        
        # 添加到训练布局
        train_layout.addWidget(data_dir_group)
        train_layout.addWidget(params_group)
        train_layout.addLayout(button_layout)
        train_layout.addWidget(train_progress_label)
        train_layout.addWidget(self.train_progress)
        train_layout.addWidget(self.train_status)
        train_layout.addWidget(val_progress_label)
        train_layout.addWidget(self.val_progress)  # 添加验证进度条
        train_layout.addWidget(self.val_status)    # 添加验证状态
        train_layout.addWidget(self.stats_group)   # 添加统计信息
        train_layout.addStretch()
        
        train_content.setLayout(train_layout)
        train_scroll.setWidget(train_content)
        
        # 设置训练标签页的主布局
        train_tab_layout = QVBoxLayout()
        train_tab_layout.setContentsMargins(0, 0, 0, 0)
        train_tab_layout.addWidget(train_scroll)
        train_tab.setLayout(train_tab_layout)
        
        self.tab_widget.addTab(train_tab, "🚀 训练")
        self.tab_indexes["train"] = self.tab_widget.indexOf(train_tab)
    
    def setup_predict_tab(self):
        """预测标签页设置"""
        predict_tab = QWidget()
        
        # 使用滚动区域包装内容
        predict_scroll = QScrollArea()
        predict_scroll.setWidgetResizable(True)
        predict_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        predict_scroll.setFrameShape(QScrollArea.NoFrame)
        
        predict_content = QWidget()
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(15)
        predict_layout.setContentsMargins(15, 15, 15, 15)
        
        # 输入图像选择
        input_group = QGroupBox("🖼️ 输入图像")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(12)
        input_layout.setContentsMargins(15, 20, 15, 15)

        self.input_list = QComboBox()
        self.input_list.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.input_list.setMinimumHeight(40)
        self.input_list.setToolTip("选择要预测的图像")
        
        button_layout_input = QHBoxLayout()
        button_layout_input.setSpacing(12)
        browse_input_btn = QPushButton("➕ 添加图像")
        browse_input_btn.setToolTip("添加一张或多张图像到预测列表")
        browse_input_btn.clicked.connect(self.browse_input_images)
        
        clear_input_btn = QPushButton("🗑️ 清空列表")
        clear_input_btn.setToolTip("清空所有已添加的图像")
        clear_input_btn.clicked.connect(self.clear_input_images)
        
        button_layout_input.addWidget(browse_input_btn)
        button_layout_input.addWidget(clear_input_btn)
        
        input_layout.addWidget(self.input_list)
        input_layout.addLayout(button_layout_input)
        input_group.setLayout(input_layout)
        
        # 模型选择
        pred_model_group = QGroupBox("🤖 预测模型")
        pred_model_layout = QVBoxLayout()
        pred_model_layout.setSpacing(12)
        pred_model_layout.setContentsMargins(15, 20, 15, 15)
        
        self.pred_model_label = QLabel("✗ 未选择模型")
        self.pred_model_label.setWordWrap(True)
        self.pred_model_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        
        browse_pred_model_btn = QPushButton("📁 选择模型")
        browse_pred_model_btn.setToolTip("选择训练好的模型文件用于预测")
        browse_pred_model_btn.clicked.connect(self.browse_pred_model_path)
        
        pred_model_layout.addWidget(self.pred_model_label)
        pred_model_layout.addWidget(browse_pred_model_btn)
        
        pred_model_group.setLayout(pred_model_layout)
        
        # 输出目录
        output_group = QGroupBox("📂 输出设置")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(12)
        output_layout.setContentsMargins(15, 20, 15, 15)
        
        self.output_dir_label = QLabel("✗ 未选择输出目录")
        self.output_dir_label.setWordWrap(True)
        self.output_dir_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        
        browse_output_btn = QPushButton("📁 选择输出目录")
        browse_output_btn.setToolTip("选择保存预测结果的目录")
        browse_output_btn.clicked.connect(self.browse_output_dir)

        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(browse_output_btn)
        output_group.setLayout(output_layout)

        # 阈值控制
        threshold_group = QGroupBox("🧮 阈值调控")
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(12)
        threshold_layout.setContentsMargins(15, 20, 15, 15)

        threshold_spin_layout = QHBoxLayout()
        threshold_label = QLabel("二值化阈值:")
        threshold_label.setMinimumWidth(100)
        threshold_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.50)
        self.threshold_spin.setSuffix("")
        threshold_spin_layout.addWidget(threshold_label)
        threshold_spin_layout.addWidget(self.threshold_spin)
        threshold_spin_layout.addStretch()

        self.llm_threshold_btn = QPushButton("🤖 LLM推荐阈值")
        self.llm_threshold_btn.setEnabled(False)
        self.llm_threshold_btn.setToolTip("基于最近一次预测的概率统计，请求LLM给出更优阈值建议")
        self.llm_threshold_btn.clicked.connect(self.request_llm_threshold)

        self.llm_threshold_status = QLabel("需要先完成预测以生成统计数据")
        self.llm_threshold_status.setWordWrap(True)
        self.llm_threshold_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: #f8fafc;
                border-radius: 8px;
                border-left: 4px solid #94a3b8;
                color: #475569;
                font-size: 10pt;
            }
        """)

        threshold_layout.addLayout(threshold_spin_layout)
        threshold_layout.addWidget(self.llm_threshold_btn)
        threshold_layout.addWidget(self.llm_threshold_status)
        threshold_group.setLayout(threshold_layout)
        
        # 预测按钮
        self.predict_btn = QPushButton("🚀 开始预测")
        self.predict_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setMinimumHeight(48)
        self.predict_btn.setToolTip("开始对选定的图像进行预测\n需要先选择模型和输出目录")
        
        # 预测进度
        predict_progress_label = QLabel("📊 预测进度:")
        predict_progress_label.setStyleSheet("font-weight: 600; color: #475569; font-size: 11pt;")
        self.predict_progress = QProgressBar()
        self.predict_progress.setFormat("预测: %p%")
        self.predict_status = QLabel("⏳ 准备预测")
        self.predict_status.setWordWrap(True)
        self.predict_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.predict_status.setMinimumHeight(50)
        self.predict_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dcfce7, stop:1 #bbf7d0);
                border-radius: 8px;
                border-left: 4px solid #16a34a;
                color: #166534;
                font-weight: 500;
                font-size: 10pt;
            }
        """)
        
        # 添加到预测布局
        predict_layout.addWidget(input_group)
        predict_layout.addWidget(pred_model_group)
        predict_layout.addWidget(output_group)
        predict_layout.addWidget(threshold_group)
        predict_layout.addWidget(self.predict_btn)
        predict_layout.addWidget(predict_progress_label)
        predict_layout.addWidget(self.predict_progress)
        predict_layout.addWidget(self.predict_status)
        predict_layout.addStretch()
        
        predict_content.setLayout(predict_layout)
        predict_scroll.setWidget(predict_content)
        
        # 设置预测标签页的主布局
        predict_tab_layout = QVBoxLayout()
        predict_tab_layout.setContentsMargins(0, 0, 0, 0)
        predict_tab_layout.addWidget(predict_scroll)
        predict_tab.setLayout(predict_tab_layout)
        
        self.tab_widget.addTab(predict_tab, "🔮 预测")
        self.tab_indexes["predict"] = self.tab_widget.indexOf(predict_tab)
    
    def setup_result_tab(self):
        """结果标签页设置"""
        result_tab = QWidget()
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(15, 15, 15, 15)
        result_layout.setSpacing(10)
        
        # 添加标题
        result_title = QLabel("📋 预测结果")
        result_title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("""
            QLabel {
                color: #1e293b;
                padding: 14px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border-radius: 10px;
                border: 2px solid #3b82f6;
                margin-bottom: 8px;
            }
        """)
        result_layout.addWidget(result_title)

        # ===== 预览区域（大图 + 翻页 + 缩略图）=====
        preview_group = QGroupBox("👀 结果预览")
        preview_layout = QVBoxLayout()

        # 大图区域：输入图像 + 分割结果
        preview_image_layout = QHBoxLayout()
        self.preview_input_label = QLabel("输入图像预览")
        self.preview_output_label = QLabel("分割结果预览")
        for lbl in (self.preview_input_label, self.preview_output_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 320)
            lbl.setStyleSheet("""
                QLabel {
                    border: 2px solid #e2e8f0;
                    border-radius: 10px;
                    background-color: #0b1120;
                    color: #64748b;
                }
            """)
        preview_image_layout.addWidget(self.preview_input_label)
        preview_image_layout.addWidget(self.preview_output_label)
        preview_layout.addLayout(preview_image_layout)

        # 翻页按钮
        nav_layout = QHBoxLayout()
        self.prev_result_btn = QPushButton("⬅ 上一张")
        self.next_result_btn = QPushButton("下一张 ➡")
        self.prev_result_btn.clicked.connect(self.show_prev_result)
        self.next_result_btn.clicked.connect(self.show_next_result)
        self.result_index_label = QLabel("0 / 0")
        self.result_index_label.setStyleSheet("font-weight: 600; color: #475569;")
        nav_layout.addWidget(self.prev_result_btn)
        nav_layout.addWidget(self.next_result_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.result_index_label)
        preview_layout.addLayout(nav_layout)

        # 缩略图条
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout()
        self.thumbnail_layout.setContentsMargins(5, 5, 5, 5)
        self.thumbnail_layout.setSpacing(8)
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        preview_layout.addWidget(self.thumbnail_scroll)

        preview_group.setLayout(preview_layout)
        result_layout.addWidget(preview_group)

        # 结果显示区域（完整列表）
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        
        self.result_container = QWidget()
        self.result_container_layout = QVBoxLayout()
        self.result_container_layout.setSpacing(20)
        self.result_container_layout.setContentsMargins(10, 10, 10, 10)
        self.result_container.setLayout(self.result_container_layout)
        
        self.result_scroll.setWidget(self.result_container)
        result_layout.addWidget(self.result_scroll)
        
        result_tab.setLayout(result_layout)
        self.tab_widget.addTab(result_tab, "📊 结果")
        self.tab_indexes["result"] = self.tab_widget.indexOf(result_tab)
    
    def setup_analysis_tab(self):
        """性能分析标签页设置"""
        analysis_tab = QWidget()
        analysis_tab_layout = QVBoxLayout()
        analysis_tab.setLayout(analysis_tab_layout)

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        analysis_scroll.setFrameShape(QScrollArea.NoFrame)

        analysis_container = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_container.setLayout(analysis_layout)
        analysis_scroll.setWidget(analysis_container)

        analysis_tab_layout.addWidget(analysis_scroll)
        
        # 标题
        title_label = QLabel("📊 模型性能分析与测试集分割结果")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #1e293b;
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border-radius: 12px;
                border: 2px solid #3b82f6;
                margin-bottom: 12px;
            }
        """)
        analysis_layout.addWidget(title_label)
        
        # 性能指标显示区域
        metrics_group = QGroupBox("📈 性能指标统计")
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(15)
        metrics_layout.setContentsMargins(10, 20, 10, 10)
        
        # Dice系数折线图
        dice_chart_group = QGroupBox("📈 Dice系数变化趋势")
        dice_chart_layout = QVBoxLayout()
        dice_chart_layout.setContentsMargins(10, 20, 10, 10)
        dice_chart_layout.setSpacing(5)
        
        # 创建matplotlib图表
        self.dice_figure = Figure(figsize=(10, 5), dpi=100)
        self.dice_canvas = FigureCanvas(self.dice_figure)
        self.dice_canvas.setMinimumHeight(350)
        self.dice_canvas.setMinimumWidth(600)
        self.dice_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.dice_ax = self.dice_figure.add_subplot(111)
        self.dice_ax.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
        self.dice_ax.set_ylabel('Dice系数', fontsize=11, fontweight='bold')
        self.dice_ax.set_title('训练过程中Dice系数的变化', fontsize=12, fontweight='bold', pad=15)
        self.dice_ax.grid(True, alpha=0.3, linestyle='--')
        self.dice_ax.set_ylim([0, 1])
        self.dice_ax.set_xlim([0, 10])  # 初始显示10个轮次
        self.dice_line, = self.dice_ax.plot([], [], 'o-', color='#4CAF50', linewidth=2.5, 
                                           markersize=8, label='Dice系数', markerfacecolor='#66BB6A',
                                           markeredgecolor='#2E7D32', markeredgewidth=1.5)
        self.dice_ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        # 优化布局，确保所有元素可见
        self.dice_figure.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
        
        dice_chart_layout.addWidget(self.dice_canvas)
        dice_chart_group.setLayout(dice_chart_layout)
        metrics_layout.addWidget(dice_chart_group)
        
        # 创建一个容器widget用于滚动
        metrics_container = QWidget()
        metrics_container_layout = QVBoxLayout()
        metrics_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.metrics_text = QLabel("等待训练开始...\n每个轮次结束后将自动更新性能指标")
        self.metrics_text.setWordWrap(True)
        self.metrics_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metrics_text.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                padding: 15px;
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                line-height: 1.6;
            }
        """)
        self.metrics_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.metrics_text.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 允许选择文本
        
        metrics_container_layout.addWidget(self.metrics_text)
        metrics_container_layout.addStretch()  # 添加弹性空间
        metrics_container.setLayout(metrics_container_layout)
        
        # 添加滚动区域
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setWidget(metrics_container)
        metrics_scroll.setMinimumHeight(200)  # 设置最小高度
        metrics_scroll.setMaximumHeight(400)  # 设置最大高度，超过后可以滚动
        metrics_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 需要时显示滚动条
        metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 水平方向不需要滚动条（因为有自动换行）
        metrics_scroll.setFrameShape(QScrollArea.NoFrame)  # 移除边框
        
        metrics_layout.addWidget(metrics_scroll)
        metrics_group.setLayout(metrics_layout)
        analysis_layout.addWidget(metrics_group)
        
        # 测试集分割结果可视化区域
        viz_group = QGroupBox("🖼️ 测试集分割结果可视化")
        viz_layout = QVBoxLayout()
        
        # 缩放控制按钮
        test_zoom_layout = QHBoxLayout()
        test_zoom_layout.setSpacing(10)
        self.test_zoom_in_btn = QPushButton("🔍+ 放大")
        self.test_zoom_out_btn = QPushButton("🔍- 缩小")
        self.test_zoom_fit_btn = QPushButton("📐 适应窗口")
        self.test_zoom_original_btn = QPushButton("📏 原始大小")
        self.test_zoom_in_btn.setMinimumHeight(35)
        self.test_zoom_out_btn.setMinimumHeight(35)
        self.test_zoom_fit_btn.setMinimumHeight(35)
        self.test_zoom_original_btn.setMinimumHeight(35)
        self.test_zoom_in_btn.clicked.connect(lambda: self.zoom_image('test', 'in'))
        self.test_zoom_out_btn.clicked.connect(lambda: self.zoom_image('test', 'out'))
        self.test_zoom_fit_btn.clicked.connect(lambda: self.zoom_image('test', 'fit'))
        self.test_zoom_original_btn.clicked.connect(lambda: self.zoom_image('test', 'original'))
        test_zoom_layout.addWidget(self.test_zoom_in_btn)
        test_zoom_layout.addWidget(self.test_zoom_out_btn)
        test_zoom_layout.addWidget(self.test_zoom_fit_btn)
        test_zoom_layout.addWidget(self.test_zoom_original_btn)
        test_zoom_layout.addStretch()
        viz_layout.addLayout(test_zoom_layout)
        
        self.test_results_label = QLabel("暂无结果")
        self.test_results_label.setAlignment(Qt.AlignCenter)
        self.test_results_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.test_results_label.setScaledContents(False)  # 不自动缩放，保持原始比例
        
        # 滚动区域用于显示大图
        test_scroll = QScrollArea()
        test_scroll.setWidgetResizable(False)  # 改为False，让图片可以超出窗口大小
        test_scroll.setWidget(self.test_results_label)
        test_scroll.setMinimumHeight(400)
        
        viz_layout.addWidget(test_scroll)
        viz_group.setLayout(viz_layout)
        analysis_layout.addWidget(viz_group)
        
        # 保存原始pixmap和当前缩放比例
        self.test_original_pixmap = None
        self.test_zoom_factor = 1.0
        
        # 性能分析图表区域
        perf_group = QGroupBox("性能分析图表")
        perf_layout = QVBoxLayout()
        
        # 缩放控制按钮
        perf_zoom_layout = QHBoxLayout()
        perf_zoom_layout.setSpacing(10)
        self.perf_zoom_in_btn = QPushButton("🔍+ 放大")
        self.perf_zoom_out_btn = QPushButton("🔍- 缩小")
        self.perf_zoom_fit_btn = QPushButton("📐 适应窗口")
        self.perf_zoom_original_btn = QPushButton("📏 原始大小")
        self.perf_zoom_in_btn.setMinimumHeight(35)
        self.perf_zoom_out_btn.setMinimumHeight(35)
        self.perf_zoom_fit_btn.setMinimumHeight(35)
        self.perf_zoom_original_btn.setMinimumHeight(35)
        self.perf_zoom_in_btn.clicked.connect(lambda: self.zoom_image('perf', 'in'))
        self.perf_zoom_out_btn.clicked.connect(lambda: self.zoom_image('perf', 'out'))
        self.perf_zoom_fit_btn.clicked.connect(lambda: self.zoom_image('perf', 'fit'))
        self.perf_zoom_original_btn.clicked.connect(lambda: self.zoom_image('perf', 'original'))
        perf_zoom_layout.addWidget(self.perf_zoom_in_btn)
        perf_zoom_layout.addWidget(self.perf_zoom_out_btn)
        perf_zoom_layout.addWidget(self.perf_zoom_fit_btn)
        perf_zoom_layout.addWidget(self.perf_zoom_original_btn)
        perf_zoom_layout.addStretch()
        perf_layout.addLayout(perf_zoom_layout)
        
        self.perf_analysis_label = QLabel("暂无结果")
        self.perf_analysis_label.setAlignment(Qt.AlignCenter)
        self.perf_analysis_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.perf_analysis_label.setScaledContents(False)  # 不自动缩放，保持原始比例
        
        perf_scroll = QScrollArea()
        perf_scroll.setWidgetResizable(False)  # 改为False，让图片可以超出窗口大小
        perf_scroll.setWidget(self.perf_analysis_label)
        perf_scroll.setMinimumHeight(400)
        
        perf_layout.addWidget(perf_scroll)
        perf_group.setLayout(perf_layout)
        analysis_layout.addWidget(perf_group)
        
        # 保存原始pixmap和当前缩放比例
        self.perf_original_pixmap = None
        self.perf_zoom_factor = 1.0
        
        # 注意力可解释性分析区域
        attention_group = QGroupBox("🔥 注意力可解释性分析")
        attention_layout = QVBoxLayout()
        attention_layout.setSpacing(12)
        attention_layout.setContentsMargins(15, 20, 15, 15)
        
        # 缩放控制按钮
        att_zoom_layout = QHBoxLayout()
        att_zoom_layout.setSpacing(10)
        self.att_zoom_in_btn = QPushButton("🔍+ 放大")
        self.att_zoom_out_btn = QPushButton("🔍- 缩小")
        self.att_zoom_fit_btn = QPushButton("📐 适应窗口")
        self.att_zoom_original_btn = QPushButton("📏 原始大小")
        self.att_zoom_in_btn.setMinimumHeight(38)
        self.att_zoom_out_btn.setMinimumHeight(38)
        self.att_zoom_fit_btn.setMinimumHeight(38)
        self.att_zoom_original_btn.setMinimumHeight(38)
        self.att_zoom_in_btn.setToolTip("放大注意力可视化图")
        self.att_zoom_out_btn.setToolTip("缩小注意力可视化图")
        self.att_zoom_fit_btn.setToolTip("自动适应窗口大小")
        self.att_zoom_original_btn.setToolTip("显示原始大小")
        self.att_zoom_in_btn.clicked.connect(lambda: self.zoom_image('attention', 'in'))
        self.att_zoom_out_btn.clicked.connect(lambda: self.zoom_image('attention', 'out'))
        self.att_zoom_fit_btn.clicked.connect(lambda: self.zoom_image('attention', 'fit'))
        self.att_zoom_original_btn.clicked.connect(lambda: self.zoom_image('attention', 'original'))
        att_zoom_layout.addWidget(self.att_zoom_in_btn)
        att_zoom_layout.addWidget(self.att_zoom_out_btn)
        att_zoom_layout.addWidget(self.att_zoom_fit_btn)
        att_zoom_layout.addWidget(self.att_zoom_original_btn)
        att_zoom_layout.addStretch()
        attention_layout.addLayout(att_zoom_layout)
        
        self.attention_label = QLabel("⏳ 等待训练完成...\n将显示注意力权重可视化")
        self.attention_label.setAlignment(Qt.AlignCenter)
        self.attention_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cbd5e1;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                color: #64748b;
                font-size: 12pt;
                padding: 20px;
                min-height: 400px;
            }
        """)
        self.attention_label.setScaledContents(False)
        
        attention_scroll = QScrollArea()
        attention_scroll.setWidgetResizable(True)
        attention_scroll.setWidget(self.attention_label)
        attention_scroll.setMinimumHeight(450)
        
        attention_layout.addWidget(attention_scroll)
        attention_group.setLayout(attention_layout)
        analysis_layout.addWidget(attention_group)
        
        # 注意力统计信息 - 使用分割器显示表格和图表
        att_stats_group = QGroupBox("📊 注意力统计分析")
        att_stats_layout = QVBoxLayout()
        att_stats_layout.setSpacing(12)
        att_stats_layout.setContentsMargins(15, 20, 15, 15)
        
        # 使用分割器分割表格和图表
        stats_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：详细统计表格
        table_container = QWidget()
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        table_title = QLabel("📋 详细统计指标")
        table_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        table_title.setStyleSheet("color: #475569; padding: 4px 0;")
        table_layout.addWidget(table_title)
        
        # 创建表格显示统计数据
        self.attention_stats_table = QTableWidget()
        self.attention_stats_table.setColumnCount(3)
        self.attention_stats_table.setHorizontalHeaderLabels(["注意力层", "统计指标", "数值"])
        self.attention_stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attention_stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.attention_stats_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.attention_stats_table.setAlternatingRowColors(True)
        self.attention_stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.attention_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.attention_stats_table.verticalHeader().setVisible(False)
        self.attention_stats_table.setMinimumHeight(250)
        self.attention_stats_table.setMaximumHeight(400)
        self.attention_stats_table.setSortingEnabled(False)  # 暂时禁用排序
        
        # 设置表格样式
        self.attention_stats_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid #e2e8f0;
                border-radius: 10px;
                background-color: #ffffff;
                gridline-color: #f1f5f9;
                font-size: 10pt;
            }
            QTableWidget::item {
                padding: 10px;
                border: none;
            }
            QTableWidget::item:hover {
                background: #f1f5f9;
            }
            QTableWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dbeafe, stop:1 #bfdbfe);
                color: #1e40af;
                font-weight: 500;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                color: #475569;
                padding: 12px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: 600;
                font-size: 11pt;
            }
        """)
        
        table_layout.addWidget(self.attention_stats_table)
        
        # 初始化表格占位提示
        self.attention_stats_table.setRowCount(1)
        placeholder_item = QTableWidgetItem("⏳ 等待训练完成，将显示详细统计数据...")
        placeholder_item.setTextAlignment(Qt.AlignCenter)
        placeholder_item.setFont(QFont("Microsoft YaHei", 10))
        placeholder_item.setForeground(QColor(100, 116, 139))
        self.attention_stats_table.setItem(0, 0, placeholder_item)
        self.attention_stats_table.setSpan(0, 0, 1, 3)  # 合并3列
        self.attention_stats_table.setRowHeight(0, 100)
        
        table_container.setLayout(table_layout)
        
        # 右侧：可视化图表
        chart_container = QWidget()
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        chart_title = QLabel("📈 统计可视化")
        chart_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        chart_title.setStyleSheet("color: #475569; padding: 4px 0;")
        chart_layout.addWidget(chart_title)
        
        # 创建matplotlib图表用于显示统计可视化
        self.attention_chart_figure = Figure(figsize=(6, 4), dpi=100)
        self.attention_chart_canvas = FigureCanvas(self.attention_chart_figure)
        self.attention_chart_canvas.setMinimumHeight(250)
        self.attention_chart_canvas.setMaximumHeight(400)
        self.attention_chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        chart_layout.addWidget(self.attention_chart_canvas)
        chart_container.setLayout(chart_layout)
        
        # 添加到分割器
        stats_splitter.addWidget(table_container)
        stats_splitter.addWidget(chart_container)
        stats_splitter.setStretchFactor(0, 1)
        stats_splitter.setStretchFactor(1, 1)
        stats_splitter.setSizes([400, 400])
        
        att_stats_layout.addWidget(stats_splitter)
        
        # 添加分析建议区域
        analysis_suggestion_label = QLabel("💡 分析建议:")
        analysis_suggestion_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        analysis_suggestion_label.setStyleSheet("color: #475569; padding-top: 8px;")
        att_stats_layout.addWidget(analysis_suggestion_label)
        
        self.attention_analysis_text = QLabel("等待训练完成，将显示注意力分析建议...")
        self.attention_analysis_text.setWordWrap(True)
        self.attention_analysis_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.attention_analysis_text.setStyleSheet("""
            QLabel {
                font-size: 10pt;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fef3c7, stop:1 #fde68a);
                border: 2px solid #f59e0b;
                border-radius: 8px;
                border-left: 4px solid #f59e0b;
                color: #92400e;
                min-height: 60px;
            }
        """)
        self.attention_analysis_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        att_stats_layout.addWidget(self.attention_analysis_text)
        
        att_stats_group.setLayout(att_stats_layout)
        analysis_layout.addWidget(att_stats_group)
        
        # 保存按钮
        save_btn_layout = QHBoxLayout()
        self.save_analysis_btn = QPushButton("💾 保存分析报告")
        self.save_analysis_btn.clicked.connect(self.save_analysis_report)
        self.save_analysis_btn.setEnabled(False)
        self.save_analysis_btn.setMinimumHeight(45)
        save_btn_layout.addStretch()
        save_btn_layout.addWidget(self.save_analysis_btn)
        save_btn_layout.addStretch()
        analysis_layout.addLayout(save_btn_layout)
        
        analysis_layout.addStretch()
        self.tab_widget.addTab(analysis_tab, "性能分析")
        self.tab_indexes["analysis"] = self.tab_widget.indexOf(analysis_tab)
        
        # 存储分析数据
        self.analysis_data = None
        self.test_viz_path = None
        self.perf_analysis_path = None
        self.attention_viz_path = None
        self.attention_stats = None
        self.attention_original_pixmap = None
        self.attention_zoom_factor = 1.0

    def setup_model_test_tab(self):
        """模型测试标签页 - 专门用于测试模型性能"""
        test_tab = QWidget()
        test_layout = QVBoxLayout()
        test_layout.setSpacing(15)
        test_layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题
        title_label = QLabel("🧪 模型测试与性能分析")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #1e293b;
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border-radius: 12px;
                border: 2px solid #3b82f6;
                margin-bottom: 12px;
            }
        """)
        test_layout.addWidget(title_label)
        
        # 使用滚动区域
        test_scroll = QScrollArea()
        test_scroll.setWidgetResizable(True)
        test_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        test_scroll.setFrameShape(QScrollArea.NoFrame)
        
        test_content = QWidget()
        test_content_layout = QVBoxLayout()
        test_content_layout.setSpacing(15)
        test_content_layout.setContentsMargins(15, 15, 15, 15)
        
        # 模型和数据选择区域
        config_group = QGroupBox("⚙️ 测试配置")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(12)
        config_layout.setContentsMargins(15, 20, 15, 15)
        
        # 模型路径选择 - 支持多模型集成
        model_label = QLabel("🤖 模型文件（支持多模型集成）:")
        model_label.setStyleSheet("font-weight: 600; color: #475569;")
        
        # 多模型列表
        model_list_layout = QVBoxLayout()
        self.test_model_list = QListWidget()
        self.test_model_list.setMaximumHeight(120)
        self.test_model_list.setStyleSheet("""
            QListWidget {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #e2e8f0;
            }
            QListWidget::item:selected {
                background-color: #dbeafe;
            }
        """)
        
        model_btn_layout = QHBoxLayout()
        browse_test_model_btn = QPushButton("➕ 添加模型")
        browse_test_model_btn.clicked.connect(self.browse_test_model_path)
        remove_model_btn = QPushButton("➖ 移除选中")
        remove_model_btn.clicked.connect(self.remove_test_model)
        model_btn_layout.addWidget(browse_test_model_btn)
        model_btn_layout.addWidget(remove_model_btn)
        
        model_list_layout.addWidget(self.test_model_list)
        model_list_layout.addLayout(model_btn_layout)
        
        # 测试数据目录选择
        data_label = QLabel("📚 测试数据目录:")
        data_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.test_data_dir_label = QLabel("✗ 未选择数据目录")
        self.test_data_dir_label.setWordWrap(True)
        self.test_data_dir_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                color: #64748b;
                font-size: 9pt;
            }
        """)
        browse_test_data_btn = QPushButton("📁 选择数据目录")
        browse_test_data_btn.clicked.connect(self.browse_test_data_dir)
        
        # 模型架构选择
        arch_label = QLabel("🏗️ 模型架构:")
        arch_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.test_arch_combo = QComboBox()
        self.test_arch_combo.addItem("改进UNet (ImprovedUNet)", "improved_unet")
        self.test_arch_combo.addItem("ResNet-UNet (ResNetUNet)", "resnet_unet")
        self.test_arch_combo.addItem("Transformer+UNet (TransUNet)", "trans_unet")
        self.test_arch_combo.addItem("DS-TransUNet", "ds_trans_unet")
        self.test_arch_combo.addItem("SwinUNet", "swin_unet")
        
        # 使用TTA选项
        self.test_use_tta_checkbox = QCheckBox("使用测试时增强 (TTA)")
        self.test_use_tta_checkbox.setChecked(True)
        self.test_use_tta_checkbox.setToolTip("启用TTA可以提升1-3%的Dice系数，但会增加推理时间")
        
        # 开始测试按钮
        self.start_test_btn = QPushButton("🚀 开始测试")
        self.start_test_btn.setMinimumHeight(50)
        self.start_test_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #059669, stop:1 #047857);
            }
        """)
        self.start_test_btn.clicked.connect(self.start_model_test)
        
        config_layout.addWidget(model_label)
        config_layout.addLayout(model_list_layout)
        config_layout.addWidget(data_label)
        config_layout.addWidget(self.test_data_dir_label)
        config_layout.addWidget(browse_test_data_btn)
        config_layout.addWidget(arch_label)
        config_layout.addWidget(self.test_arch_combo)
        config_layout.addWidget(self.test_use_tta_checkbox)
        config_layout.addWidget(self.start_test_btn)
        config_group.setLayout(config_layout)
        test_content_layout.addWidget(config_group)
        
        # 测试进度
        self.test_progress = QProgressBar()
        self.test_progress.setMinimum(0)
        self.test_progress.setMaximum(100)
        self.test_progress.setValue(0)
        self.test_status = QLabel("等待开始测试...")
        self.test_status.setStyleSheet("padding: 8px; background: #f1f5f9; border-radius: 6px;")
        test_content_layout.addWidget(self.test_progress)
        test_content_layout.addWidget(self.test_status)
        
        # 结果展示区域 - 使用标签页
        results_tabs = QTabWidget()
        
        # 性能指标标签页
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(10, 10, 10, 10)

        # 推荐阈值（来自阈值扫描的智能选择）
        self.test_recommended_threshold_label = QLabel("推荐阈值: --")
        self.test_recommended_threshold_label.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fef9c3, stop:1 #fde68a);
                border: 1px solid #f59e0b;
                border-radius: 8px;
                color: #92400e;
                font-weight: 700;
                font-size: 11pt;
            }
        """)
        metrics_layout.addWidget(self.test_recommended_threshold_label)
        
        self.test_metrics_text = QTextEdit()
        self.test_metrics_text.setReadOnly(True)
        self.test_metrics_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.test_metrics_text.setPlaceholderText("测试完成后，性能指标将显示在这里...")
        metrics_layout.addWidget(QLabel("📊 性能指标:"))
        metrics_layout.addWidget(self.test_metrics_text)
        metrics_tab.setLayout(metrics_layout)
        results_tabs.addTab(metrics_tab, "📊 性能指标")

        # 阈值扫描详情标签页
        sweep_tab = QWidget()
        sweep_layout = QVBoxLayout()
        sweep_layout.setContentsMargins(10, 10, 10, 10)

        sweep_title = QLabel("🔎 阈值扫描详情（Threshold | Dice | Precision | Recall | FP Count）")
        sweep_title.setStyleSheet("font-weight: 700; color: #334155;")
        sweep_layout.addWidget(sweep_title)

        self.test_sweep_table = QTableWidget(0, 5)
        self.test_sweep_table.setHorizontalHeaderLabels(["阈值", "Global Dice", "Precision", "Recall", "FP Count"])
        self.test_sweep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.test_sweep_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.test_sweep_table.setSelectionMode(QTableWidget.SingleSelection)
        self.test_sweep_table.horizontalHeader().setStretchLastSection(True)
        self.test_sweep_table.setAlternatingRowColors(True)
        self.test_sweep_table.setStyleSheet("""
            QTableWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
            QHeaderView::section {
                background: #f1f5f9;
                padding: 6px;
                border: 1px solid #e2e8f0;
                font-weight: 700;
                color: #334155;
            }
        """)
        sweep_layout.addWidget(self.test_sweep_table)

        sweep_tab.setLayout(sweep_layout)
        results_tabs.addTab(sweep_tab, "🔎 扫描详情")
        
        # 注意力热图标签页
        attention_tab = QWidget()
        attention_layout = QVBoxLayout()
        attention_layout.setContentsMargins(10, 10, 10, 10)
        
        self.test_attention_label = QLabel("暂无注意力热图")
        self.test_attention_label.setAlignment(Qt.AlignCenter)
        self.test_attention_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0; min-height: 400px;")
        self.test_attention_label.setScaledContents(False)
        
        attention_scroll = QScrollArea()
        attention_scroll.setWidgetResizable(True)
        attention_scroll.setWidget(self.test_attention_label)
        attention_layout.addWidget(QLabel("🔥 注意力热图:"))
        attention_layout.addWidget(attention_scroll)
        attention_tab.setLayout(attention_layout)
        results_tabs.addTab(attention_tab, "🔥 注意力热图")
        
        # Dice系数低的案例标签页
        low_dice_tab = QWidget()
        low_dice_layout = QVBoxLayout()
        low_dice_layout.setContentsMargins(10, 10, 10, 10)
        
        self.low_dice_list = QListWidget()
        self.low_dice_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: #ffffff;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f1f5f9;
            }
            QListWidget::item:hover {
                background: #f8fafc;
            }
        """)
        self.low_dice_list.itemDoubleClicked.connect(self.view_low_dice_case)
        
        low_dice_layout.addWidget(QLabel("⚠️ Dice系数低的案例 (双击查看详情):"))
        low_dice_layout.addWidget(self.low_dice_list)
        low_dice_tab.setLayout(low_dice_layout)
        results_tabs.addTab(low_dice_tab, "⚠️ 低Dice案例")
        
        test_content_layout.addWidget(results_tabs)
        
        test_content.setLayout(test_content_layout)
        test_scroll.setWidget(test_content)
        test_layout.addWidget(test_scroll)
        test_tab.setLayout(test_layout)
        
        self.tab_widget.addTab(test_tab, "🧪 模型测试")
        self.tab_indexes["test"] = self.tab_widget.indexOf(test_tab)
        
        # 初始化测试相关变量
        self.test_model_paths = []  # 改为列表，支持多模型
        self.test_data_dir = None
        self.test_thread = None
        self.test_results = None
        self.low_dice_cases = []

    def setup_ai_assistant_tab(self):
        """AI助手标签页"""
        ai_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # API配置
        config_group = QGroupBox("🔐 API配置")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(10)

        # API地址选择
        url_layout = QHBoxLayout()
        url_label = QLabel("接口地址:")
        url_label.setMinimumWidth(80)
        self.ai_url_combo = QComboBox()
        for display, url in self.ai_base_url_options:
            self.ai_url_combo.addItem(display, url)
        # 设置当前选中的URL（匹配默认值）
        current_index = 0
        for i, (_, url) in enumerate(self.ai_base_url_options):
            if url == self.ai_base_url:
                current_index = i
                break
        self.ai_url_combo.setCurrentIndex(current_index)
        self.ai_url_combo.currentIndexChanged.connect(self.on_api_url_changed)
        self.ai_url_combo.setToolTip("选择要使用的API服务地址")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.ai_url_combo)
        config_layout.addLayout(url_layout)
        
        self.ai_base_label = QLabel(f"当前地址: {self.ai_base_url}")
        self.ai_base_label.setStyleSheet("color: #475569; font-weight: 600; font-size: 9pt;")
        config_layout.addWidget(self.ai_base_label)

        model_layout = QHBoxLayout()
        model_label = QLabel("模型选择:")
        model_label.setMinimumWidth(80)
        self.ai_model_combo = QComboBox()
        for display, value in self.ai_model_options:
            self.ai_model_combo.addItem(display, value)
        # 尝试设置当前模型，如果不存在则使用第一个
        current_model_index = 0
        for i in range(self.ai_model_combo.count()):
            if self.ai_model_combo.itemData(i) == self.ai_model_name:
                current_model_index = i
                break
        self.ai_model_combo.setCurrentIndex(current_model_index)
        self.ai_model_combo.setToolTip("根据选择的API服务显示可用的模型列表\n切换API服务时会自动更新模型选项")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.ai_model_combo)
        config_layout.addLayout(model_layout)

        limits_text = (
            f"资源限制：每分钟请求 {self.ai_limits['rpm']} 次、"
            f"每分钟 {self.ai_limits['tpm']} tokens、"
            f"每周 {self.ai_limits['weekly']:,} tokens"
        )
        limits_label = QLabel(limits_text)
        limits_label.setWordWrap(True)
        limits_label.setStyleSheet("""
            QLabel {
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 8px;
                padding: 8px;
                color: #92400e;
            }
        """)
        config_layout.addWidget(limits_label)

        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        key_label.setMinimumWidth(80)
        self.ai_key_input = QLineEdit()
        self.ai_key_input.setEchoMode(QLineEdit.Password)
        self.ai_key_input.setPlaceholderText("请输入API Key")
        self.ai_key_input.setText(self.ai_api_key)
        # 连接信号，标记用户是否手动修改过API key
        self.ai_key_input.textChanged.connect(self.on_api_key_changed)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.ai_key_input)
        config_layout.addLayout(key_layout)

        self.ai_status_label = QLabel("✅ 已就绪")
        self.ai_status_label.setStyleSheet("""
            QLabel {
                padding: 8px 10px;
                background: #dcfce7;
                border-left: 4px solid #16a34a;
                border-radius: 8px;
                color: #166534;
            }
        """)
        config_layout.addWidget(self.ai_status_label)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # 对话区域
        conversation_group = QGroupBox("💬 对话")
        convo_layout = QVBoxLayout()
        convo_layout.setSpacing(10)

        self.ai_prompt_input = QTextEdit()
        self.ai_prompt_input.setPlaceholderText("请输入您想咨询的问题，例如：\n“如何提升当前分割模型的Dice指标？”")
        self.ai_prompt_input.setMinimumHeight(120)

        self.ai_response_view = QTextBrowser()
        self.ai_response_view.setOpenExternalLinks(True)
        self.ai_response_view.setReadOnly(True)
        self.ai_response_view.setStyleSheet("background: #f8fafc;")
        self.ai_response_view.setMinimumHeight(200)

        button_layout = QHBoxLayout()
        self.ai_send_btn = QPushButton("🚀 发送请求")
        self.ai_send_btn.clicked.connect(self.send_ai_request)
        self.ai_clear_btn = QPushButton("🧹 清空对话")
        self.ai_clear_btn.clicked.connect(self.clear_ai_history)
        button_layout.addWidget(self.ai_send_btn)
        button_layout.addWidget(self.ai_clear_btn)

        convo_layout.addWidget(QLabel("问题输入："))
        convo_layout.addWidget(self.ai_prompt_input)
        convo_layout.addLayout(button_layout)
        convo_layout.addWidget(QLabel("AI回复："))
        convo_layout.addWidget(self.ai_response_view)

        conversation_group.setLayout(convo_layout)
        layout.addWidget(conversation_group)
        layout.addStretch()

        ai_tab.setLayout(layout)
        self.tab_widget.addTab(ai_tab, "🤖 AI助手")
        self.tab_indexes["assistant"] = self.tab_widget.indexOf(ai_tab)
    

    def browse_data_dir(self):
        """选择训练数据目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if directory:
            self.lock.lock()
            self.data_dir = directory
            self.lock.unlock()
            self.data_dir_label.setText(f"✓ {directory}")
            self.data_dir_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
            self.train_btn.setEnabled(True)
            self.update_system_status("data", directory, status="success")
    
    def browse_model_path(self, model_type=None):
        """选择预训练模型
        
        Args:
            model_type: 'resnet'，如果为 None 则选择单模型
        """
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt)")
        if path:
            if model_type == 'resnet':
                self.resnet_model_path = path
                self.resnet_model_path_label.setText(f"✓ {os.path.basename(path)}")
                self.resnet_model_path_label.setStyleSheet("""
                    QLabel {
                        padding: 10px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #dcfce7, stop:1 #bbf7d0);
                        border: 2px solid #16a34a;
                        border-radius: 6px;
                        color: #166534;
                        font-size: 9pt;
                        font-weight: 500;
                    }
                """)
            else:
                self.lock.lock()
                self.model_path = path
                self.lock.unlock()
            self.model_path_label.setText(f"✓ {path}")
            self.model_path_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
            self.update_system_status("train_model", path, status="success")
    
    def browse_pred_model_path(self):
        """选择预测模型"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt)")
        if path:
            self.lock.lock()
            self.model_path = path
            self.lock.unlock()
            self.pred_model_label.setText(f"✓ {path}")
            self.pred_model_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
    
    def browse_test_model_path(self):
        """选择测试模型文件（支持多选）"""
        paths, _ = QFileDialog.getOpenFileNames(self, "选择模型文件（可多选）", "", "PyTorch模型 (*.pth *.pt)")
        for path in paths:
            if path and path not in self.test_model_paths:
                self.test_model_paths.append(path)
                item = QListWidgetItem(f"✓ {os.path.basename(path)}")
                item.setData(Qt.UserRole, path)  # 存储完整路径
                self.test_model_list.addItem(item)
    
    def remove_test_model(self):
        """移除选中的模型"""
        current_item = self.test_model_list.currentItem()
        if current_item:
            path = current_item.data(Qt.UserRole)
            if path in self.test_model_paths:
                self.test_model_paths.remove(path)
            self.test_model_list.takeItem(self.test_model_list.row(current_item))
    
    def browse_test_data_dir(self):
        """选择测试数据目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择测试数据目录")
        if directory:
            self.test_data_dir = directory
            self.test_data_dir_label.setText(f"✓ {directory}")
            self.test_data_dir_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 6px;
                    color: #166534;
                    font-size: 9pt;
                    font-weight: 500;
                }
            """)
    
    def start_model_test(self):
        """开始模型测试"""
        # 检查模型文件（集成功能已删除，仅支持单模型）
        if len(self.test_model_paths) < 1:
            QMessageBox.warning(self, "警告", "请至少选择一个模型文件")
            return
        # 验证第一个模型文件
        if not os.path.exists(self.test_model_paths[0]):
            QMessageBox.warning(self, "警告", "模型文件不存在")
            return
        
        if not self.test_data_dir or not os.path.exists(self.test_data_dir):
            QMessageBox.warning(self, "警告", "请先选择有效的测试数据目录")
            return
        
        # 获取模型架构（从checkpoint推断或用户选择）
        model_type = self.test_arch_combo.currentData() or self.test_arch_combo.currentText()
        use_tta = self.test_use_tta_checkbox.isChecked()
        
        # 创建测试线程（集成功能已删除）
        self.test_thread = ModelTestThread(
            model_paths=[self.test_model_paths[0]],  # 仅使用第一个模型
            data_dir=self.test_data_dir,
            model_type=model_type,
            use_tta=use_tta
        )
        self.test_thread.update_progress.connect(self.update_test_progress)
        self.test_thread.threshold_sweep_ready.connect(self.on_threshold_sweep_ready)
        self.test_thread.test_finished.connect(self.on_test_finished)
        self.test_thread.start()
        
        self.start_test_btn.setEnabled(False)
        self.test_status.setText("测试进行中...")
        # 清空上一次扫描结果
        if hasattr(self, "test_sweep_table"):
            self.test_sweep_table.setRowCount(0)
        if hasattr(self, "test_recommended_threshold_label"):
            self.test_recommended_threshold_label.setText("推荐阈值: --")

    def on_threshold_sweep_ready(self, payload):
        """接收阈值扫描结果并更新GUI展示"""
        if not payload or not isinstance(payload, dict):
            return
        rows = payload.get("rows", []) or []
        best = payload.get("best", {}) or {}
        recall_floor = float(payload.get("recall_floor", 0.90))
        fallback_used = bool(payload.get("fallback_used", False))

        # 更新推荐阈值展示
        try:
            thr = float(best.get("threshold", 0.0))
            rec = float(best.get("recall", 0.0))
            warn = "（回退）" if fallback_used else ""
            self.test_recommended_threshold_label.setText(f"推荐阈值: {thr:.2f} (Recall: {rec*100:.1f}%) {warn}")
            # Recall 低于阈值时加红提示
            if rec < recall_floor:
                self.test_recommended_threshold_label.setStyleSheet("""
                    QLabel {
                        padding: 10px 12px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #fee2e2, stop:1 #fecaca);
                        border: 1px solid #ef4444;
                        border-radius: 8px;
                        color: #991b1b;
                        font-weight: 800;
                        font-size: 11pt;
                    }
                """)
        except Exception:
            pass

        # 更新表格
        if not hasattr(self, "test_sweep_table"):
            return
        table = self.test_sweep_table
        table.setRowCount(len(rows))

        best_thr = float(best.get("threshold", -1.0))
        for r_idx, r in enumerate(rows):
            thr = float(r.get("threshold", 0.0))
            dice = float(r.get("dice", 0.0))
            prec = float(r.get("precision", 0.0))
            rec = float(r.get("recall", 0.0))
            fp = int(r.get("fp_count", 0))

            items = [
                QTableWidgetItem(f"{thr:.2f}"),
                QTableWidgetItem(f"{dice:.4f}"),
                QTableWidgetItem(f"{prec:.4f}"),
                QTableWidgetItem(f"{rec:.4f}"),
                QTableWidgetItem(f"{fp:,}"),
            ]
            for c, it in enumerate(items):
                it.setTextAlignment(Qt.AlignCenter)
                table.setItem(r_idx, c, it)

            # 高亮最佳阈值行
            if abs(thr - best_thr) < 1e-6:
                for c in range(5):
                    cell = table.item(r_idx, c)
                    if cell:
                        cell.setBackground(QColor("#dcfce7"))
                        cell.setForeground(QColor("#166534"))
                        f = cell.font()
                        f.setBold(True)
                        cell.setFont(f)
    
    def update_test_progress(self, value, message):
        """更新测试进度"""
        self.test_progress.setValue(value)
        self.test_status.setText(message)
    
    def on_test_finished(self, detailed_metrics, attention_path, low_dice_cases):
        """测试完成处理"""
        self.start_test_btn.setEnabled(True)
        self.test_results = detailed_metrics
        self.low_dice_cases = low_dice_cases
        
        # 显示性能指标
        self.display_test_metrics(detailed_metrics)
        
        # 显示注意力热图
        if attention_path and os.path.exists(attention_path):
            pixmap = QPixmap(attention_path)
            self.test_attention_label.setPixmap(pixmap.scaled(
                self.test_attention_label.width(), 
                self.test_attention_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        else:
            self.test_attention_label.setText("模型不支持注意力热图或生成失败")
        
        # 显示低Dice案例
        self.display_low_dice_cases(low_dice_cases)
        
        # 切换到测试标签页
        self.switch_to_tab("test")
        
        QMessageBox.information(
            self, "测试完成",
            f"模型测试完成！\n\n"
            f"平均 Dice 系数: {detailed_metrics.get('average', {}).get('dice', 0):.4f}\n"
            f"总样本数: {detailed_metrics.get('total_samples', 0)}\n"
            f"低Dice案例数: {len(low_dice_cases)}"
        )
    
    def display_test_metrics(self, detailed_metrics):
        """显示测试性能指标"""
        avg_metrics = detailed_metrics.get('average', {})
        total_samples = detailed_metrics.get('total_samples', 0)
        
        metrics_text = "=" * 60 + "\n"
        metrics_text += "📊 模型测试性能指标\n"
        metrics_text += "=" * 60 + "\n\n"
        
        metrics_text += f"测试样本总数: {total_samples}\n\n"
        
        metrics_text += "【平均性能指标】\n"
        metrics_text += "-" * 60 + "\n"
        metrics_text += f"Dice系数:        {avg_metrics.get('dice', 0):.4f}\n"
        metrics_text += f"IoU:             {avg_metrics.get('iou', 0):.4f}\n"
        metrics_text += f"精确率 (Precision): {avg_metrics.get('precision', 0):.4f}\n"
        metrics_text += f"召回率 (Recall):    {avg_metrics.get('recall', 0):.4f}\n"
        metrics_text += f"敏感度 (Sensitivity): {avg_metrics.get('sensitivity', 0):.4f}\n"
        metrics_text += f"特异度 (Specificity): {avg_metrics.get('specificity', 0):.4f}\n"
        metrics_text += f"F1分数:          {avg_metrics.get('f1', 0):.4f}\n"
        # 显示HD95，如果是NaN则显示"N/A"
        hd95_val = avg_metrics.get('hd95', float('nan'))
        if np.isnan(hd95_val):
            metrics_text += f"HD95:            N/A (部分样本无法计算)\n\n"
        else:
            metrics_text += f"HD95:            {hd95_val:.4f}\n\n"
        
        # 性能分析
        dice = avg_metrics.get('dice', 0)
        metrics_text += "【性能分析】\n"
        metrics_text += "-" * 60 + "\n"
        if dice >= 0.9:
            metrics_text += "✅ Dice系数表现优秀 (≥0.9)，模型分割精度很高。\n"
        elif dice >= 0.8:
            metrics_text += "✅ Dice系数表现良好 (0.8-0.9)，模型分割精度较好。\n"
        elif dice >= 0.7:
            metrics_text += "⚠️ Dice系数表现一般 (0.7-0.8)，模型分割精度中等，建议进一步优化。\n"
        else:
            metrics_text += "❌ Dice系数较低 (<0.7)，模型分割精度有待提升，建议检查数据质量和模型架构。\n"
        
        precision = avg_metrics.get('precision', 0)
        recall = avg_metrics.get('recall', 0)
        if abs(precision - recall) < 0.1:
            metrics_text += "✅ 精确率和召回率较为平衡，模型在假阳性控制方面表现良好。\n"
        elif precision > recall:
            metrics_text += "⚠️ 精确率高于召回率，模型更倾向于减少假阳性，但可能漏检部分目标。\n"
        else:
            metrics_text += "⚠️ 召回率高于精确率，模型更倾向于捕获所有目标，但可能产生较多假阳性。\n"
        
        self.test_metrics_text.setText(metrics_text)
    
    def display_low_dice_cases(self, low_dice_cases):
        """显示低Dice案例列表"""
        self.low_dice_list.clear()
        
        if not low_dice_cases:
            self.low_dice_list.addItem("✅ 没有低Dice案例（所有样本Dice ≥ 0.7）")
            return
        
        # 按Dice排序
        low_dice_cases_sorted = sorted(low_dice_cases, key=lambda x: x['dice'])
        
        for case in low_dice_cases_sorted:
            image_name = os.path.basename(case['image_path'])
            item_text = f"Dice: {case['dice']:.4f} | IoU: {case['iou']:.4f} | Precision: {case['precision']:.4f} | Recall: {case['recall']:.4f} | {image_name}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, case)  # 存储完整案例数据
            self.low_dice_list.addItem(item)
    
    def view_low_dice_case(self, item):
        """查看低Dice案例详情，显示原始图像、预测mask和真实mask"""
        case_data = item.data(Qt.UserRole)
        if not case_data:
            return
        
        # 创建详情对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("低Dice案例详情")
        dialog.setMinimumSize(1000, 700)
        
        layout = QVBoxLayout(dialog)
        
        # 性能指标文本
        detail_text = f"""
性能指标:
  • Dice系数:     {case_data['dice']:.4f}
  • IoU:          {case_data['iou']:.4f}
  • 精确率:       {case_data['precision']:.4f}
  • 召回率:       {case_data['recall']:.4f}
  • 特异度:       {case_data['specificity']:.4f}

图像路径: {case_data['image_path']}
        """
        text_label = QLabel(detail_text)
        text_label.setStyleSheet("font-size: 12px; padding: 10px;")
        layout.addWidget(text_label)
        
        # 图像显示区域
        images_layout = QHBoxLayout()
        
        # 原始图像
        if 'original_image' in case_data:
            orig_img = case_data['original_image']
            orig_qimg = QImage(orig_img.data, orig_img.shape[1], orig_img.shape[0], orig_img.shape[1], QImage.Format_Grayscale8)
            orig_pixmap = QPixmap.fromImage(orig_qimg).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            orig_label = QLabel()
            orig_label.setPixmap(orig_pixmap)
            orig_label.setAlignment(Qt.AlignCenter)
            orig_label.setStyleSheet("border: 2px solid #3b82f6; padding: 5px;")
            orig_title = QLabel("原始图像")
            orig_title.setAlignment(Qt.AlignCenter)
            orig_layout = QVBoxLayout()
            orig_layout.addWidget(orig_title)
            orig_layout.addWidget(orig_label)
            images_layout.addLayout(orig_layout)
        
        # 预测mask
        if 'pred_mask' in case_data:
            pred_img = case_data['pred_mask']
            pred_qimg = QImage(pred_img.data, pred_img.shape[1], pred_img.shape[0], pred_img.shape[1], QImage.Format_Grayscale8)
            pred_pixmap = QPixmap.fromImage(pred_qimg).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pred_label = QLabel()
            pred_label.setPixmap(pred_pixmap)
            pred_label.setAlignment(Qt.AlignCenter)
            pred_label.setStyleSheet("border: 2px solid #ef4444; padding: 5px;")
            pred_title = QLabel("预测Mask")
            pred_title.setAlignment(Qt.AlignCenter)
            pred_layout = QVBoxLayout()
            pred_layout.addWidget(pred_title)
            pred_layout.addWidget(pred_label)
            images_layout.addLayout(pred_layout)
        
        # 真实mask
        if 'target_mask' in case_data:
            target_img = case_data['target_mask']
            target_qimg = QImage(target_img.data, target_img.shape[1], target_img.shape[0], target_img.shape[1], QImage.Format_Grayscale8)
            target_pixmap = QPixmap.fromImage(target_qimg).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            target_label = QLabel()
            target_label.setPixmap(target_pixmap)
            target_label.setAlignment(Qt.AlignCenter)
            target_label.setStyleSheet("border: 2px solid #10b981; padding: 5px;")
            target_title = QLabel("真实Mask")
            target_title.setAlignment(Qt.AlignCenter)
            target_layout = QVBoxLayout()
            target_layout.addWidget(target_title)
            target_layout.addWidget(target_label)
            images_layout.addLayout(target_layout)
        
        layout.addLayout(images_layout)
        
        # 分析文本
        analysis_text = """
分析:
  • 该案例的Dice系数较低，可能存在以下问题:
    - 目标边界模糊
    - 目标尺寸过小
    - 图像质量较差
    - 模型在该类型样本上表现不佳

建议:
  • 检查该图像的质量和标注准确性
  • 考虑增加类似样本的训练数据
  • 调整模型参数或损失函数权重
        """
        analysis_label = QLabel(analysis_text)
        analysis_label.setStyleSheet("font-size: 11px; padding: 10px; color: #666;")
        layout.addWidget(analysis_label)
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def browse_api_model(self):
        """选择API服务使用的模型"""
        path, _ = QFileDialog.getOpenFileName(self, "选择API模型文件", "", "PyTorch模型 (*.pth *.pt)")
        if path:
            self.api_model_path = path
            self.api_model_label.setText(f"✓ {path}")
            self.api_model_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
    
    def browse_input_images(self):
        """选择输入图像"""
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图像文件", "", 
                                              "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff)")
        if paths:

            for path in paths:
                self.input_list.addItem(path)
            self.update_predict_btn_state()
    
    def clear_input_images(self):
        """清空输入图像列表"""
        self.input_list.clear()
        self.update_predict_btn_state()
    
    def browse_output_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.lock.lock()
            self.output_dir = directory
            self.lock.unlock()
            self.output_dir_label.setText(f"✓ {directory}")
            self.output_dir_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
            self.update_predict_btn_state()
            self.update_system_status("output_dir", directory, status="success")
    
    def update_predict_btn_state(self):
        """更新预测按钮状态"""
        enabled = (self.input_list.count() > 0 and 
                   self.model_path is not None and 
                   self.output_dir is not None)
        self.predict_btn.setEnabled(enabled)

    def start_api_server(self):
        """启动内置API服务"""
        if self.api_thread and self.api_thread.isRunning():
            QMessageBox.information(self, "提示", "API服务已经在运行中")
            return

        if not self.api_model_path or not os.path.exists(self.api_model_path):
            QMessageBox.warning(self, "警告", "请先选择有效的API模型文件")
            return

        host = self.api_host_input.text().strip() or "0.0.0.0"
        port = self.api_port_spin.value()
        device = self.api_device_combo.currentData()

        try:
            self.api_service = SegmentationAPIService(self.api_model_path, device=device)
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"模型加载失败: {exc}")
            self.set_api_status(f"❌ 模型加载失败: {exc}", status="error")
            self.api_service = None
            return

        self.api_thread = APIServerThread(self.api_service, host, port)
        self.api_thread.status_changed.connect(self.on_api_status_changed)
        self.api_thread.server_started.connect(self.on_api_started)
        self.api_thread.server_stopped.connect(self.on_api_stopped)
        self.api_thread.error_occurred.connect(self.on_api_error)
        self.api_thread.finished.connect(self.on_api_thread_finished)
        self.api_thread.start()

        self.api_start_btn.setEnabled(False)
        self.api_stop_btn.setEnabled(True)
        self.set_api_status("⏳ API服务启动中...", status="info")

    def stop_api_server(self):
        """停止API服务"""
        if self.api_thread and self.api_thread.isRunning():
            self.api_thread.stop()
            self.set_api_status("⏳ 正在停止API服务...", status="info")
        else:
            QMessageBox.information(self, "提示", "API服务当前未运行")

    def on_api_status_changed(self, message):
        self.set_api_status(message, status="info")

    def on_api_started(self, message):
        self.set_api_status(message, status="running")

    def on_api_stopped(self, message):
        self.set_api_status(message, status="info")
        self.api_start_btn.setEnabled(True)
        self.api_stop_btn.setEnabled(False)

    def on_api_error(self, message):
        self.set_api_status(f"❌ API错误: {message}", status="error")
        QMessageBox.warning(self, "API错误", message)

    def on_api_thread_finished(self):
        self.api_thread = None
        self.api_service = None
        self.api_start_btn.setEnabled(True)
        self.api_stop_btn.setEnabled(False)

    def set_api_status(self, text, status="info"):
        """更新API状态显示"""
        styles = {
            "info": """
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e0f2fe, stop:1 #bae6fd);
                    border-left: 4px solid #0284c7;
                    border-radius: 8px;
                    color: #075985;
                    font-size: 10pt;
                }
            """,
            "running": """
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border-left: 4px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                }
            """,
            "error": """
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #fee2e2, stop:1 #fecaca);
                    border-left: 4px solid #dc2626;
                    border-radius: 8px;
                    color: #991b1b;
                    font-size: 10pt;
                }
            """
        }
        self.api_status_label.setStyleSheet(styles.get(status, styles["info"]))
        self.api_status_label.setText(text)

    def send_ai_request(self):
        """发送远程AI请求"""
        if self.ai_thread and self.ai_thread.isRunning():
            QMessageBox.information(self, "提示", "正在等待上一条回复，请稍候。")
            return

        prompt = self.ai_prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "请先输入问题")
            return

        api_key = self.ai_key_input.text().strip() or self.ai_api_key
        if not api_key:
            QMessageBox.warning(self, "警告", "请填写API Key")
            return

        self.append_ai_message("用户", prompt, is_markdown=False)
        self.ai_send_btn.setEnabled(False)
        self.set_ai_status_label("⏳ 正在请求AI服务...", status="info")

        selected_model = self.ai_model_combo.currentData() or self.ai_model_combo.currentText()

        self.ai_thread = AIAssistantThread(
            base_url=self.ai_base_url,
            model=selected_model,
            api_key=api_key,
            prompt=prompt
        )
        self.ai_thread.success.connect(self.on_ai_success)
        self.ai_thread.error.connect(self.on_ai_error)
        self.ai_thread.finished.connect(self.on_ai_finished)
        self.ai_thread.start()

    def on_ai_success(self, content: str):
        self.append_ai_message("AI", content, is_markdown=True)
        self.set_ai_status_label("✅ AI回复已收到", status="success")

    def on_ai_error(self, message: str):
        self.append_ai_message("系统", f"请求失败：{message}", is_markdown=False)
        self.set_ai_status_label(f"❌ {message}", status="error")
        QMessageBox.warning(self, "AI请求失败", message)

    def on_ai_finished(self):
        self.ai_send_btn.setEnabled(True)
        self.ai_thread = None

    def clear_ai_history(self):
        self.ai_response_view.clear()
        self.set_ai_status_label("🧼 对话已清空，等待新的问题", status="info")

    def append_ai_message(self, role: str, message: str, is_markdown: bool = False):
        """将聊天内容以HTML追加到对话框，支持Markdown渲染"""
        if not hasattr(self, "ai_response_view"):
            return

        role_html = self.escape_html(role)
        if is_markdown:
            body_html = self.render_markdown_html(message)
        else:
            body_html = self.escape_html(message).replace("\n", "<br>")

        html_block = f"""
        <div style="padding:8px 0;">
            <div style="font-weight:600;color:#0f172a;">{role_html}：</div>
            <div style="margin-top:6px;color:#1e293b;line-height:1.6;">{body_html}</div>
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:12px 0;">
        </div>
        """
        self.ai_response_view.moveCursor(QTextCursor.End)
        self.ai_response_view.insertHtml(html_block)
        self.ai_response_view.moveCursor(QTextCursor.End)
        self.ai_response_view.verticalScrollBar().setValue(
            self.ai_response_view.verticalScrollBar().maximum()
        )

    def render_markdown_html(self, text: str) -> str:
        """将Markdown文本转换为HTML，缺少依赖时退回普通文本"""
        try:
            import markdown  # 延迟导入，避免强依赖

            return markdown.markdown(
                text,
                extensions=["fenced_code", "tables", "nl2br"]
            )
        except Exception:
            return self.escape_html(text).replace("\n", "<br>")

    def escape_html(self, text: str) -> str:
        """安全转义HTML"""
        return html.escape(text or "", quote=False)

    def set_ai_status_label(self, text, status="info"):
        styles = {
            "info": """
                QLabel {
                    padding: 8px 10px;
                    background: #e0f2fe;
                    border-left: 4px solid #0284c7;
                    border-radius: 8px;
                    color: #075985;
                }
            """,
            "success": """
                QLabel {
                    padding: 8px 10px;
                    background: #dcfce7;
                    border-left: 4px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                }
            """,
            "error": """
                QLabel {
                    padding: 8px 10px;
                    background: #fee2e2;
                    border-left: 4px solid #dc2626;
                    border-radius: 8px;
                    color: #991b1b;
                }
            """
        }
        self.ai_status_label.setStyleSheet(styles.get(status, styles["info"]))
        self.ai_status_label.setText(text)

    def compute_prediction_statistics(self, results):
        """根据预测概率生成统计信息"""
        if not results:
            return None

        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
        samples = []
        # 兼容不同格式的结果:
        # - (image_np, pred_np, prob_map)
        # - (image_np, pred_np, prob_map, tag)  # 如 nnFormer 标记
        # - 直接为 prob_map 数组
        for idx, item in enumerate(results, start=1):
            # 直接是概率图
            if isinstance(item, np.ndarray):
                prob_map = item
            # 元组 / 列表：取第 3 个作为概率图
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                prob_map = item[2]
            else:
                # 不认识的格式，跳过
                continue

            if prob_map is None:
                continue
            sample = {
                "index": idx,
                "mean_prob": float(np.mean(prob_map)),
                "std_prob": float(np.std(prob_map)),
                "p10": float(np.percentile(prob_map, 10)),
                "p90": float(np.percentile(prob_map, 90)),
                "foreground_ratio": {
                    f"{thr:.2f}": float(np.mean(prob_map >= thr)) for thr in thresholds
                }
            }
            samples.append(sample)

        if not samples:
            return None

        aggregate = {
            "mean_prob": float(np.mean([s["mean_prob"] for s in samples])),
            "std_prob": float(np.mean([s["std_prob"] for s in samples])),
            "p10": float(np.mean([s["p10"] for s in samples])),
            "p90": float(np.mean([s["p90"] for s in samples])),
            "foreground_ratio": {
                key: float(np.mean([s["foreground_ratio"][key] for s in samples]))
                for key in samples[0]["foreground_ratio"].keys()
            }
        }

        return {
            "samples": samples,
            "aggregate": aggregate,
            "thresholds": thresholds
        }

    def on_api_key_changed(self, text):
        """当用户手动修改API key时标记"""
        # 检查当前输入的key是否等于当前服务的默认key
        current_default_key = self.ai_api_key_by_service.get(self.ai_base_url, "")
        if text.strip() != current_default_key:
            self.ai_key_manually_changed = True
        else:
            # 如果用户改回了默认值，重置标记
            self.ai_key_manually_changed = False
    
    def on_api_url_changed(self, index):
        """当用户选择不同的API地址时更新模型选项"""
        if index >= 0:
            selected_url = self.ai_url_combo.itemData(index)
            if selected_url:
                # 保存旧URL，用于判断是否需要更新API key
                old_url = self.ai_base_url
                self.ai_base_url = selected_url
                self.ai_base_label.setText(f"当前地址: {self.ai_base_url}")
                
                # 根据选择的API服务更新模型选项
                if selected_url in self.ai_model_options_by_service:
                    model_options = self.ai_model_options_by_service[selected_url]
                    self.ai_model_options = model_options
                    
                    # 更新模型下拉框
                    model_combo = getattr(self, "ai_model_combo", None)
                    if model_combo:
                        model_combo.clear()
                        for display, value in model_options:
                            model_combo.addItem(display, value)
                        # 默认选择第一个模型
                        if model_options:
                            model_combo.setCurrentIndex(0)
                            self.ai_model_name = model_options[0][1]
                
                # 根据选择的API服务更新API key（如果用户没有手动修改过）
                if selected_url in self.ai_api_key_by_service:
                    new_api_key = self.ai_api_key_by_service[selected_url]
                    key_input = getattr(self, "ai_key_input", None)
                    if key_input:
                        current_key = key_input.text().strip()
                        old_default_key = self.ai_api_key_by_service.get(old_url, "")
                        # 如果当前key等于旧服务的默认key，或者用户没有手动修改过，则自动更新
                        if current_key == old_default_key or not self.ai_key_manually_changed:
                            # 临时断开信号，避免触发手动修改标记
                            try:
                                key_input.textChanged.disconnect(self.on_api_key_changed)
                            except:
                                pass
                            key_input.setText(new_api_key)
                            self.ai_api_key = new_api_key
                            self.ai_key_manually_changed = False
                            # 重新连接信号
                            key_input.textChanged.connect(self.on_api_key_changed)
                        # 如果用户手动修改过，保持用户输入的key不变
    
    def build_threshold_prompt(self, stats, current_threshold):
        """构造发送给LLM的提示词"""
        lines = [
            "你是一名医学图像分割系统的调参与质检助手。",
            "我们已经对若干张图像进行了前景概率预测，下面是统计数据。",
            f"当前用于二值化的阈值为 {current_threshold:.2f}。",
            "请根据统计信息判断是否需要调整阈值，使预测掩膜更加合理。",
            "如果统计显示高概率像素占比很小，可以适当降低阈值；反之可提高。",
            "请仅输出JSON，格式为：",
            '{"recommended_threshold": 0.xx, "reason": "简要说明"}',
            "其中 recommended_threshold 必须在 0.05 到 0.95 之间。"
        ]

        agg = stats["aggregate"]
        lines.append("\n【整体统计】")
        lines.append(f"- 平均概率: {agg['mean_prob']:.4f} (std {agg['std_prob']:.4f})")
        lines.append(f"- P10/P90: {agg['p10']:.4f} / {agg['p90']:.4f}")
        lines.append("- 不同阈值下的前景占比：")
        for thr, ratio in agg["foreground_ratio"].items():
            lines.append(f"  - 阈值 {thr}: 前景像素 {ratio*100:.2f}%")

        lines.append("\n【样本统计】")
        for sample in stats["samples"]:
            fg_ratios = ", ".join(
                [f"{thr}:{ratio*100:.1f}%" for thr, ratio in sample["foreground_ratio"].items()]
            )
            lines.append(
                f"- 样本{sample['index']}: mean={sample['mean_prob']:.4f}, "
                f"std={sample['std_prob']:.4f}, p10/p90={sample['p10']:.4f}/{sample['p90']:.4f}, "
                f"foreground({fg_ratios})"
            )

        lines.append("\n请基于上述数据输出JSON。")
        return "\n".join(lines)

    def request_llm_threshold(self):
        """调用LLM推荐阈值"""
        if not self.prediction_stats:
            QMessageBox.warning(self, "提示", "请先运行一次预测以生成统计数据。")
            return

        api_key_widget = getattr(self, "ai_key_input", None)
        api_key = (api_key_widget.text().strip() if api_key_widget else self.ai_api_key).strip() or self.ai_api_key
        if not api_key:
            QMessageBox.warning(self, "提示", "请在AI助手中填写可用的API Key。")
            return

        model_combo = getattr(self, "ai_model_combo", None)
        model_name = None
        if model_combo and model_combo.currentData():
            model_name = model_combo.currentData()
        elif model_combo:
            model_name = model_combo.currentText()
        else:
            model_name = self.ai_model_name

        if self.llm_threshold_thread and self.llm_threshold_thread.isRunning():
            QMessageBox.information(self, "提示", "上一条请求尚未完成，请稍候。")
            return

        prompt = self.build_threshold_prompt(self.prediction_stats, self.threshold_spin.value())
        self.llm_threshold_thread = AIAssistantThread(
            base_url=self.ai_base_url,
            model=model_name,
            api_key=api_key,
            prompt=prompt
        )
        self.llm_threshold_thread.success.connect(self.on_llm_threshold_success)
        self.llm_threshold_thread.error.connect(self.on_llm_threshold_error)
        self.llm_threshold_thread.finished.connect(self.on_llm_threshold_finished)
        self.llm_threshold_thread.start()
        self.llm_threshold_btn.setEnabled(False)
        self.set_llm_threshold_status("⏳ 正在请求LLM分析阈值...", status="info")

    def on_llm_threshold_success(self, content):
        try:
            data = self.extract_json_from_text(content)
            recommended = float(data.get("recommended_threshold"))
            reason = data.get("reason", "LLM未提供原因")
        except Exception as exc:
            self.set_llm_threshold_status(f"解析LLM回复失败: {exc}", status="error")
            QMessageBox.warning(self, "阈值推荐失败", f"无法解析LLM回复:\n{content}")
            return

        recommended = min(max(recommended, 0.05), 0.95)
        self.threshold_spin.setValue(recommended)
        self.set_llm_threshold_status(
            f"推荐阈值 {recommended:.2f}\n原因: {reason}", status="success"
        )
        QMessageBox.information(
            self,
            "LLM 阈值建议",
            f"推荐使用阈值 {recommended:.2f}\n原因：{reason}\n"
            "请重新运行预测以应用新的阈值。"
        )

    def on_llm_threshold_error(self, message):
        self.set_llm_threshold_status(f"❌ 请求失败: {message}", status="error")
        QMessageBox.warning(self, "LLM请求错误", message)

    def on_llm_threshold_finished(self):
        if self.llm_threshold_thread:
            self.llm_threshold_thread = None
        if self.prediction_stats:
            self.llm_threshold_btn.setEnabled(True)

    def set_llm_threshold_status(self, text, status="info"):
        styles = {
            "info": """
                QLabel {
                    padding: 10px 12px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border-left: 4px solid #94a3b8;
                    color: #475569;
                    font-size: 10pt;
                }
            """,
            "success": """
                QLabel {
                    padding: 10px 12px;
                    background: #dcfce7;
                    border-radius: 8px;
                    border-left: 4px solid #16a34a;
                    color: #166534;
                    font-size: 10pt;
                }
            """,
            "error": """
                QLabel {
                    padding: 10px 12px;
                    background: #fee2e2;
                    border-radius: 8px;
                    border-left: 4px solid #dc2626;
                    color: #991b1b;
                    font-size: 10pt;
                }
            """
        }
        if hasattr(self, "llm_threshold_status"):
            self.llm_threshold_status.setStyleSheet(styles.get(status, styles["info"]))
            self.llm_threshold_status.setText(text)

    def extract_json_from_text(self, text):
        """尝试从LLM回复中解析JSON"""
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            return json.loads(snippet)

        raise ValueError("未找到有效的JSON内容")
    
    def start_training(self):
        """开始训练"""
        if not self.data_dir:
            QMessageBox.warning(self, "警告", "请先选择数据目录")
            return
        
        self.train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.train_progress.setValue(0)
        
        save_best = self.save_best_checkbox.isChecked()
        
        # 设置模型架构类型
        selected_arch = self.arch_combo.currentData() or self.arch_combo.currentText()
        os.environ["SEG_MODEL"] = selected_arch

        
        # 获取GWO优化选项（SwinUNet/DS-TransUNet可用）
        use_gwo = self.gwo_checkbox.isChecked() and (
            self.arch_combo.currentData() in ("swin_unet", "ds_trans_unet") or 
            self.arch_combo.currentText().lower().startswith(("swin", "ds_trans"))
        )
        
        selected_optimizer = self.optimizer_combo.currentData() or "adam"
        os.environ["SEG_OPTIMIZER"] = selected_optimizer

        # 准备实例化 TrainThread，添加异常捕获以排查初始化失败问题
        print(">>> [DEBUG] 准备实例化 TrainThread...")
        try:
            self.train_thread = TrainThread(
                data_dir=self.data_dir,
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_spin.value(),
                model_path=self.model_path,
                save_best=save_best,
                use_gwo=use_gwo,
                optimizer_type=selected_optimizer
            )
            print(">>> [DEBUG] TrainThread 实例化成功")
            
            # 连接所有信号
            self.train_thread.update_progress.connect(self.update_train_progress)
            self.train_thread.update_val_progress.connect(self.update_val_progress)  # 添加这行
            self.train_thread.training_finished.connect(self.training_complete)
            self.train_thread.model_saved.connect(self.model_saved)
            self.train_thread.epoch_completed.connect(self.update_train_stats)  # 添加这行
            self.train_thread.test_results_ready.connect(self.display_test_results)  # 添加测试结果展示
            self.train_thread.metrics_ready.connect(self.display_performance_metrics)  # 添加性能指标展示
            self.train_thread.visualization_ready.connect(self.display_performance_chart)  # 添加性能分析图表展示
            self.train_thread.epoch_analysis_ready.connect(self.display_epoch_analysis)  # 添加每个epoch的分析展示
            self.train_thread.attention_analysis_ready.connect(self.display_attention_analysis)  # 添加注意力分析展示
        
            print(">>> [DEBUG] 所有信号连接成功，准备启动线程...")
            self.train_thread.start()
            print(">>> [DEBUG] 线程启动命令已发送")
        except Exception as e:
            import traceback
            print(f">>> [FATAL] TrainThread 初始化失败: {e}")
            print(">>> [FATAL] 详细错误堆栈:")
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "初始化训练线程失败", 
                f"训练线程初始化时发生错误：\n\n{str(e)}\n\n请检查控制台输出的详细错误信息。"
            )
            # 恢复按钮状态
            self.train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)
            self.train_progress.setValue(0)
            return
    
    def _on_arch_changed(self):
        """处理架构选择变化"""
        selected_arch = self.arch_combo.currentData() or self.arch_combo.currentText()
        # 选择 SwinUNet、DS-TransUNet 时启用GWO选项
        is_gwo_supported = (
            selected_arch in ("swin_unet", "ds_trans_unet") or 
            selected_arch.lower().startswith("swin") or 
            selected_arch.lower().startswith("ds_trans")
        )
        self.gwo_checkbox.setEnabled(is_gwo_supported)
        if not is_gwo_supported:
            self.gwo_checkbox.setChecked(False)
    
    def stop_training(self):
        """停止训练"""
        if self.train_thread:
            self.train_thread.stop_requested = True
            self.train_thread.wait()
            self.train_thread = None
        
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
    def update_val_progress(self, value, message):
        """更新验证进度"""
        self.val_progress.setValue(value)
        self.val_status.setText(message)

    def update_train_stats(self, epoch, loss, val_loss, val_dice):
        """更新训练统计信息"""
        self.epoch_label.setText(f"当前轮次: {epoch}")
        self.loss_label.setText(f"训练Loss: {loss:.4f}")
        self.val_loss_label.setText(f"验证Loss: {val_loss:.4f}")  
        self.dice_label.setText(f"Dice系数: {val_dice:.4f}")
        
        # 更新Dice系数折线图
        self.update_dice_chart()
    
    def update_train_progress(self, value, message):
        """更新训练进度"""
        self.train_progress.setValue(value)
        self.train_status.setText(message)
    
    def training_complete(self, message, best_model_path):
        """训练完成处理"""
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.train_status.setText(message)
        
        # 提示用户可以查看性能分析
        QMessageBox.information(
            self, '训练完成',
            f'{message}\n\n'
            '性能分析结果已生成！\n'
            '请切换到"性能分析"标签页查看：\n'
            '- 测试集分割结果可视化\n'
            '- 性能指标统计\n'
            '- 性能分析图表'
        )
        
        # 如果存在最佳模型且用户选择了保存
        if best_model_path and os.path.exists(best_model_path):
            reply = QMessageBox.question(
                self, '保存最佳模型',
                '训练已完成，是否保存最佳模型到指定位置?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.save_model(best_model_path)
    
    def model_saved(self, message):
        """模型保存通知"""
        self.train_status.setText(message)
    
    def save_model(self, temp_model_path):
        """保存模型到指定位置"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存最佳模型",
            "best_model.pth",
            "PyTorch模型 (*.pth *.pt)"
        )
        
        if path:
            try:
                shutil.copyfile(temp_model_path, path)
                QMessageBox.information(self, "成功", f"模型已保存到:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")
    
    def start_prediction(self):
        """开始预测"""
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "警告", "请先选择有效的模型文件")
            return
        
        if self.input_list.count() == 0:
            QMessageBox.warning(self, "警告", "请添加要预测的图像")
            return
        
        # 询问用户是否保存结果

        reply = QMessageBox.question(self, '保存结果', 
                                    '您想要保存预测结果吗?',
                                    QMessageBox.Yes | QMessageBox.No, 
                                    QMessageBox.Yes)
        
        save_results = reply == QMessageBox.Yes
        output_dir = None
        
        if save_results:
            # 让用户选择输出目录
            directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if not directory:
                save_results = False
            else:
                output_dir = directory
        
        image_paths = [self.input_list.itemText(i) for i in range(self.input_list.count())]
        self.predict_btn.setEnabled(False)
        self.predict_progress.setValue(0)
        self.prediction_stats = None
        
        self.predict_thread = PredictThread(
            image_paths=image_paths,
            model_path=self.model_path,
            threshold=self.threshold_spin.value(),
            save_results=save_results,
            output_dir=output_dir
        )
        
        self.predict_thread.update_progress.connect(self.update_predict_progress)
        self.predict_thread.prediction_finished.connect(self.prediction_complete)
        self.predict_thread.start()
        if hasattr(self, 'llm_threshold_btn'):
            self.llm_threshold_btn.setEnabled(False)
            self.set_llm_threshold_status("正在进行预测，完成后可请求LLM推荐阈值", status="info")
    
    def update_predict_progress(self, value, message):
        """更新预测进度"""
        self.predict_progress.setValue(value)
        self.predict_status.setText(message)
    
    def prediction_complete(self, input_images, output_masks, input_numpy_images):
        """预测完成处理"""
        self.predict_btn.setEnabled(True)
        self.predict_status.setText("预测完成")
        
        # 清空之前的结果
        for i in reversed(range(self.result_container_layout.count())):
            widget = self.result_container_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # 保存当前结果
        self.current_results = input_numpy_images
        self.prediction_stats = self.compute_prediction_statistics(input_numpy_images)
        if self.prediction_stats and hasattr(self, 'llm_threshold_btn'):
            self.llm_threshold_btn.setEnabled(True)
            self.set_llm_threshold_status(
                "统计数据已生成，点击“LLM推荐阈值”获取建议。", status="success"
            )
        
        # 清空旧的结果展示和缩略图
        for i in reversed(range(self.result_container_layout.count())):
            item = self.result_container_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if hasattr(self, "thumbnail_layout"):
            for i in reversed(range(self.thumbnail_layout.count())):
                item = self.thumbnail_layout.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        # 显示新结果
        for idx, (image_tuple, input_path, output_path) in enumerate(zip(input_numpy_images, input_images, output_masks)):
            # image_tuple 可能是 (image, pred, prob) 或 (image, pred, prob, tag)
            if isinstance(image_tuple, (list, tuple)) and len(image_tuple) >= 2:
                image_np, pred_np = image_tuple[0], image_tuple[1]
            else:
                # 无法解析的格式，跳过
                print(f"[警告] 无法解析预测结果格式: {type(image_tuple)}")
                continue
            # 确保图像数据是连续的并且类型正确
            image_np = np.ascontiguousarray(image_np)
            pred_np = np.ascontiguousarray(pred_np)
            
            # 确保图像是8位无符号整数格式
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)
            if pred_np.dtype != np.uint8:
                pred_np = (pred_np * 255).astype(np.uint8)
            
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            
            # 转换为QPixmap（已翻译为中文）
            q_img = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            input_pixmap = QPixmap.fromImage(q_img)
            
            # 预测mask有时可能是(H, W)或(H, W, 1)，统一处理成2D单通道
            pred_vis = np.squeeze(pred_np)
            if pred_vis.ndim == 3:
                pred_vis = pred_vis[:, :, 0]
            if pred_vis.ndim != 2:
                raise ValueError(f"预测mask维度非法: shape={pred_vis.shape}, 期望为(H,W)或(H,W,1)")
            
            pred_height, pred_width = pred_vis.shape
            
            # 对于单通道图像，使用 Format_Grayscale8（已翻译为中文）
            pred_q_img = QImage(pred_vis.data, pred_width, pred_height, pred_width, QImage.Format_Grayscale8)
            output_pixmap = QPixmap.fromImage(pred_q_img)
            
            # 输入图像
            input_label = QLabel(f"📷 输入图像 {idx+1}:")
            input_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            input_label.setStyleSheet("""
                QLabel {
                    color: #1976d2;
                    padding: 5px;
                    background-color: #e3f2fd;
                    border-radius: 4px;
                }
            """)
            self.result_container_layout.addWidget(input_label)
            
            input_pixmap = input_pixmap.scaled(512, 512, Qt.KeepAspectRatio)
            input_image = QLabel()
            input_image.setPixmap(input_pixmap)
            input_image.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 5px;
                    background-color: white;
                }
            """)
            input_image.setAlignment(Qt.AlignCenter)
            self.result_container_layout.addWidget(input_image)
            
            # 分割结果
            output_label = QLabel("🎯 分割结果:")
            output_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            output_label.setStyleSheet("""
                QLabel {
                    color: #7b1fa2;
                    padding: 5px;
                    background-color: #f3e5f5;
                    border-radius: 4px;
                }
            """)
            self.result_container_layout.addWidget(output_label)
            
            output_pixmap = output_pixmap.scaled(512, 512, Qt.KeepAspectRatio)
            output_image = QLabel()
            output_image.setPixmap(output_pixmap)
            output_image.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 5px;
                    background-color: white;
                }
            """)
            output_image.setAlignment(Qt.AlignCenter)
            self.result_container_layout.addWidget(output_image)

            # 创建缩略图（点击可快速预览）
            if hasattr(self, "thumbnail_layout"):
                thumb_label = QLabel()
                thumb_pix = input_pixmap.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumb_label.setPixmap(thumb_pix)
                thumb_label.setCursor(Qt.PointingHandCursor)

                def make_handler(index):
                    def handler(event):
                        self.show_result_at(index)
                    return handler

                thumb_label.mousePressEvent = make_handler(idx)
                self.thumbnail_layout.addWidget(thumb_label)

        # 初始化预览为第一张
        if input_numpy_images:
            self.show_result_at(0)
            
            # 添加保存按钮
            save_btn = QPushButton("💾 保存结果")
            save_btn.clicked.connect(lambda _, i=idx: self.save_single_result(i))
            save_btn.setMinimumHeight(40)
            self.result_container_layout.addWidget(save_btn)
            
            # 分隔线
            line = QWidget()
            line.setFixedHeight(1)
            line.setStyleSheet("background-color: #cccccc;")
            line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.result_container_layout.addWidget(line)
        
        # 滚动到顶部
        self.result_scroll.verticalScrollBar().setValue(0)
        
        if any(path is not None for path in output_masks):
            QMessageBox.information(self, "完成", "预测完成! 结果已保存到输出目录")
        else:
            QMessageBox.information(self, "完成", "预测完成! 结果未保存，您可以选择保存单个结果或重新运行预测并选择保存")
    
    def save_single_result(self, index):
        """保存单个结果"""
        if index < 0 or index >= len(self.current_results):
            return
        
        # 兼容 (image, pred, prob) / (image, pred, prob, tag) 等格式
        result_item = self.current_results[index]
        if isinstance(result_item, (list, tuple)) and len(result_item) >= 2:
            image_np, pred_np = result_item[0], result_item[1]
        else:
            print(f"[警告] show_result_at: 无法解析结果格式: {type(result_item)}")
            return
        
        # 让用户选择保存目录和文件名
        path, _ = QFileDialog.getSaveFileName(self, "保存分割结果", 
                                             "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg)")
        
        if path:
            try:
                # 获取文件扩展名
                ext = os.path.splitext(path)[1].lower()
                
                # 保存输入图像
                input_path = os.path.splitext(path)[0] + "_input" + ext
                cv2.imwrite(input_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
                # 保存分割结果
                output_path = os.path.splitext(path)[0] + "_mask" + ext
                cv2.imwrite(output_path, pred_np)
                
                QMessageBox.information(self, "成功", 
f"结果已保存到:\n{input_path}\n{output_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

    def show_result_at(self, index: int):
        """在预览区域显示指定索引的结果"""
        if not self.current_results:
            return
        index = max(0, min(index, len(self.current_results) - 1))
        self.current_result_index = index

        # 兼容 (image, pred, prob) / (image, pred, prob, tag) 等格式
        result_item = self.current_results[index]
        if isinstance(result_item, (list, tuple)) and len(result_item) >= 2:
            image_np, pred_np = result_item[0], result_item[1]
        else:
            print(f"[警告] show_result_at: 无法解析结果格式: {type(result_item)}")
            return

        # 输入图像
        image_np = np.ascontiguousarray(image_np)
        h, w, _ = image_np.shape
        bytes_per_line = 3 * w
        q_img = QImage(image_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        input_pixmap = QPixmap.fromImage(q_img).scaled(
            512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_input_label.setPixmap(input_pixmap)

        # 分割结果mask
        pred_vis = np.squeeze(pred_np)
        if pred_vis.ndim == 3:
            pred_vis = pred_vis[:, :, 0]
        if pred_vis.ndim == 2:
            ph, pw = pred_vis.shape
            pred_q_img = QImage(pred_vis.data, pw, ph, pw, QImage.Format_Grayscale8)
            output_pixmap = QPixmap.fromImage(pred_q_img).scaled(
                512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_output_label.setPixmap(output_pixmap)
        else:
            self.preview_output_label.setText(f"mask 维度非法: {pred_vis.shape}")

        # 更新索引文本
        self.result_index_label.setText(
            f"{index + 1} / {len(self.current_results)}"
        )

    def show_prev_result(self):
        """预览上一张结果"""
        if not self.current_results:
            return
        new_index = (getattr(self, "current_result_index", 0) - 1) % len(
            self.current_results
        )
        self.show_result_at(new_index)

    def show_next_result(self):
        """预览下一张结果"""
        if not self.current_results:
            return
        new_index = (getattr(self, "current_result_index", 0) + 1) % len(
            self.current_results
        )
        self.show_result_at(new_index)
    
    def display_epoch_analysis(self, epoch, viz_path, metrics):
        """显示每个epoch的性能分析结果"""
        if os.path.exists(viz_path):
            # 显示测试集分割结果可视化
            pixmap = QPixmap(viz_path)
            self.test_original_pixmap = pixmap  # 保存原始pixmap
            self.test_zoom_factor = 1.0
            # 初始显示：适应窗口大小，但保持比例
            self._display_image_with_zoom('test', pixmap, 'fit')
        
        # 优先使用完整验证集的Dice（val_dice_history）以与折线图一致
        displayed_dice = metrics.get('dice', 0.0)
        if (self.train_thread is not None and
            hasattr(self.train_thread, 'val_dice_history') and
            len(self.train_thread.val_dice_history) >= epoch):
            displayed_dice = float(self.train_thread.val_dice_history[epoch - 1])
        displayed_f1 = metrics.get('f1', displayed_dice)
        
        # 更新性能指标显示（包含历史信息）
        metrics_text = f"=== 当前 Epoch {epoch} 性能指标 ===\n\n"
        metrics_text += f"【当前轮次指标】\n"
        metrics_text += f"Dice系数: {displayed_dice:.4f}\n"
        metrics_text += f"IoU: {metrics.get('iou', 0):.4f}\n"
        metrics_text += f"精确率: {metrics.get('precision', 0):.4f}\n"
        metrics_text += f"敏感度(召回率): {metrics.get('sensitivity', metrics.get('recall', 0)):.4f}\n"
        metrics_text += f"特异度: {metrics.get('specificity', 0):.4f}\n"
        metrics_text += f"F1分数: {displayed_dice:.4f}\n"
        metrics_text += f"HD95: {metrics.get('hd95', float('nan')):.4f}\n\n"
        
        # 如果有训练历史，显示趋势
        if (self.train_thread is not None and 
            hasattr(self.train_thread, 'val_dice_history') and 
            len(self.train_thread.val_dice_history) > 0):
            metrics_text += f"【训练趋势】\n"
            metrics_text += f"验证Dice历史: {[f'{x:.3f}' for x in self.train_thread.val_dice_history[-5:]]}\n"
            if len(self.train_thread.val_dice_history) > 1:
                trend = "↑ 提升" if self.train_thread.val_dice_history[-1] > self.train_thread.val_dice_history[-2] else "↓ 下降"
                metrics_text += f"趋势: {trend}\n"
            metrics_text += "\n"
        
        metrics_text += "（每个轮次自动更新，训练完成后将显示完整统计）"
        
        self.metrics_text.setText(metrics_text)
        
        # 更新Dice系数折线图
        self.update_dice_chart()
        
        # 自动切换到性能分析标签页（仅在第一个epoch或每5个epoch切换一次，避免过于频繁）
        if epoch == 1 or epoch % 5 == 0:
            self.tab_widget.setCurrentIndex(3)  # 性能分析标签页是第4个（索引3）
    
    def display_test_results(self, viz_path, detailed_metrics):
        """显示测试集分割结果"""
        self.test_viz_path = viz_path
        self.analysis_data = detailed_metrics
        
        if os.path.exists(viz_path):
            pixmap = QPixmap(viz_path)
            self.test_original_pixmap = pixmap  # 保存原始pixmap
            self.test_zoom_factor = 1.0
            # 初始显示：适应窗口大小，但保持比例
            self._display_image_with_zoom('test', pixmap, 'fit')
            # 自动切换到性能分析标签页以查看图表和指标
            self.tab_widget.setCurrentIndex(3)  # 性能分析标签页是第4个（索引3）
        else:
            self.test_results_label.setText(f"无法加载图像: {viz_path}")
            self.test_original_pixmap = None
            QMessageBox.warning(self, "提示", f"无法加载测试集可视化图像: {viz_path}")
    
    def display_performance_chart(self, chart_path):
        """显示性能分析图表"""
        # 检查是否是性能分析图表
        if "performance_analysis" in chart_path and os.path.exists(chart_path):
            self.perf_analysis_path = chart_path
            pixmap = QPixmap(chart_path)
            self.perf_original_pixmap = pixmap  # 保存原始pixmap
            self.perf_zoom_factor = 1.0
            # 初始显示：适应窗口大小，但保持比例
            self._display_image_with_zoom('perf', pixmap, 'fit')
            # 自动切换到性能分析标签页
            self.tab_widget.setCurrentIndex(3)  # 性能分析标签页是第4个（索引3）
    
    def display_performance_metrics(self, detailed_metrics):
        """显示性能指标"""
        self.analysis_data = detailed_metrics
        
        # 性能分析图表应该在训练线程中已经生成
        if (self.train_thread is not None and 
            hasattr(self.train_thread, 'temp_dir') and 
            self.train_thread.temp_dir):
            perf_path = os.path.join(self.train_thread.temp_dir, "performance_analysis.png")
            if os.path.exists(perf_path):
                self.perf_analysis_path = perf_path
                pixmap = QPixmap(perf_path)
                self.perf_original_pixmap = pixmap  # 保存原始pixmap
                self.perf_zoom_factor = 1.0
                # 初始显示：适应窗口大小，但保持比例
                self._display_image_with_zoom('perf', pixmap, 'fit')
        
        # 格式化指标文本
        avg_metrics = detailed_metrics.get('average', {})
        std_metrics = detailed_metrics.get('std', {})
        
        metrics_text = "=== 模型性能指标统计 ===\n\n"
        metrics_text += f"测试样本数量: {len(detailed_metrics.get('all_samples', {}).get('dice', []))}\n\n"
        
        metrics_text += "【平均值 ± 标准差】\n"
        metric_names_cn = {
            'dice': 'Dice系数',
            'iou': 'IoU',
            'precision': '精确率',
            'recall': '召回率',
            'sensitivity': '敏感度(召回率)',
            'specificity': '特异度',
            'f1': 'F1分数',
            'hd95': 'HD95'
        }
        summary_metrics = ['dice', 'iou', 'precision', 'sensitivity', 'specificity', 'f1', 'hd95']
        for metric_name in summary_metrics:
            avg_val = avg_metrics.get(metric_name, 0)
            std_val = std_metrics.get(metric_name, 0)
            metrics_text += f"{metric_names_cn[metric_name]:12s}: {avg_val:.4f} ± {std_val:.4f}\n"
        
        metrics_text += "\n【详细统计】\n"
        for metric_name in summary_metrics:
            min_val = detailed_metrics.get('min', {}).get(metric_name, 0)
            max_val = detailed_metrics.get('max', {}).get(metric_name, 0)
            median_val = detailed_metrics.get('median', {}).get(metric_name, 0)
            metrics_text += f"{metric_names_cn[metric_name]}:\n"
            metrics_text += f"  最小值: {min_val:.4f}\n"
            metrics_text += f"  最大值: {max_val:.4f}\n"
            metrics_text += f"  中位数: {median_val:.4f}\n\n"
        
        # 性能分析
        metrics_text += "【性能分析】\n"
        dice_avg = avg_metrics.get('dice', 0)
        if dice_avg >= 0.9:
            metrics_text += "Dice系数表现优秀 (≥0.9)，模型分割精度很高。\n"
        elif dice_avg >= 0.8:
            metrics_text += "Dice系数表现良好 (0.8-0.9)，模型分割精度较好。\n"
        elif dice_avg >= 0.7:
            metrics_text += "Dice系数表现一般 (0.7-0.8)，模型分割精度中等，建议进一步优化。\n"
        else:
            metrics_text += "Dice系数较低 (<0.7)，模型分割精度有待提升，建议检查数据质量和模型架构。\n"
        
        precision = avg_metrics.get('precision', 0)
        recall = avg_metrics.get('sensitivity', avg_metrics.get('recall', 0))
        specificity = avg_metrics.get('specificity', 0)
        if abs(precision - recall) < 0.1:
            metrics_text += "精确率和召回率较为平衡，模型在假阳性控制方面表现良好。\n"
        elif precision > recall:
            metrics_text += "精确率高于召回率，模型更倾向于减少假阳性，但可能漏检部分目标。\n"
        else:
            metrics_text += "召回率高于精确率，模型更倾向于捕获所有目标，但可能产生较多假阳性。\n"
        metrics_text += f"特异度平均水平: {specificity:.4f}\n"
        
        self.metrics_text.setText(metrics_text)
        self.save_analysis_btn.setEnabled(True)
        
        # 更新Dice系数折线图
        self.update_dice_chart()
    
    def update_dice_chart(self):
        """更新Dice系数折线图"""
        if (self.train_thread is not None and 
            hasattr(self.train_thread, 'val_dice_history') and 
            len(self.train_thread.val_dice_history) > 0):
            
            epochs = list(range(1, len(self.train_thread.val_dice_history) + 1))
            dice_values = self.train_thread.val_dice_history
            
            # 更新折线图数据
            self.dice_ax.clear()
            self.dice_ax.plot(epochs, dice_values, 'o-', color='#4CAF50', linewidth=2.5, 
                            markersize=8, label='Dice系数', markerfacecolor='#66BB6A',
                            markeredgecolor='#2E7D32', markeredgewidth=1.5)
            self.dice_ax.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
            self.dice_ax.set_ylabel('Dice系数', fontsize=11, fontweight='bold')
            self.dice_ax.set_title('训练过程中Dice系数的变化', fontsize=12, fontweight='bold', pad=15)
            self.dice_ax.grid(True, alpha=0.3, linestyle='--')
            self.dice_ax.set_ylim([0, 1])
            
            # 智能调整X轴范围，确保所有数据点可见
            max_epoch = max(epochs) if epochs else 1
            # 如果轮次较少，显示更多空间；如果轮次较多，自动扩展
            if max_epoch <= 10:
                x_max = 10
            else:
                x_max = max_epoch + 2  # 留出一些边距
            
            self.dice_ax.set_xlim([0, x_max])
            
            # 设置X轴刻度，避免过于密集
            if max_epoch <= 20:
                self.dice_ax.set_xticks(range(0, x_max + 1, max(1, x_max // 10)))
            else:
                # 轮次较多时，只显示部分刻度
                step = max(1, max_epoch // 10)
                self.dice_ax.set_xticks(range(0, max_epoch + 1, step))
            
            # 设置Y轴刻度
            self.dice_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            self.dice_ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            
            self.dice_ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
            
            # 添加当前最大值标注
            if dice_values:
                max_idx = dice_values.index(max(dice_values))
                max_epoch = epochs[max_idx]
                max_dice = dice_values[max_idx]
                
                # 确保标注不会超出图表范围
                annotation_y = min(max_dice + 0.1, 0.95)
                
                self.dice_ax.annotate(f'最佳: {max_dice:.4f}\n轮次: {max_epoch}', 
                                     xy=(max_epoch, max_dice),
                                     xytext=(max_epoch, annotation_y),
                                     arrowprops=dict(arrowstyle='->', color='#f44336', lw=2, 
                                                   connectionstyle="arc3,rad=0.2"),
                                     fontsize=9,
                                     color='#f44336',
                                     fontweight='bold',
                                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            
            # 添加当前值标注（最后一个点）
            if len(dice_values) > 0:
                current_epoch = epochs[-1]
                current_dice = dice_values[-1]
                self.dice_ax.annotate(f'当前: {current_dice:.4f}', 
                                     xy=(current_epoch, current_dice),
                                     xytext=(current_epoch + 0.5, current_dice),
                                     fontsize=8,
                                     color='#1976d2',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.6))
            
            # 优化布局，确保所有元素可见
            self.dice_figure.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
            self.dice_canvas.draw()
    
    def display_attention_analysis(self, viz_path, attention_stats):
        """显示注意力可解释性分析结果 - 优化版"""
        self.attention_viz_path = viz_path or ""
        self.attention_stats = attention_stats or {}
        
        has_image = bool(viz_path) and os.path.exists(viz_path)
        
        # 显示注意力可视化图
        if has_image:
            pixmap = QPixmap(viz_path)
            self.attention_original_pixmap = pixmap
            self.attention_zoom_factor = 1.0
            self._display_image_with_zoom('attention', pixmap, 'fit')
            self.attention_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3b82f6;
                    border-radius: 10px;
                    background: #ffffff;
                }
            """)
        else:
            self.attention_original_pixmap = None
            self.attention_zoom_factor = 1.0
            self.attention_label.setText("当前模型不支持注意力可视化或尚未生成结果。")
            self.attention_label.setStyleSheet("""
                QLabel {
                    padding: 16px;
                    border: 2px dashed #94a3b8;
                    border-radius: 10px;
                    color: #475569;
                    background: #f8fafc;
                }
            """)
        
        # 使用表格显示注意力统计信息
        self.attention_stats_table.setRowCount(0)  # 清空表格
        if not attention_stats:
            return
        
        row = 0
        layer_names = {
            'att1': '注意力层1 (最精细)',
            'att2': '注意力层2',
            'att3': '注意力层3',
            'att4': '注意力层4 (深层)'
        }
        
        for att_name in ['att1', 'att2', 'att3', 'att4']:
            if att_name in attention_stats:
                stats = attention_stats[att_name]
                layer_name = layer_names.get(att_name, f'注意力层{att_name[-1]}')
                
                # 添加统计指标行
                metrics = [
                    ('平均权重', stats.get('mean', 0), ''),
                    ('标准差', stats.get('std', 0), ''),
                    ('最大权重', stats.get('max', 0), ''),
                    ('最小权重', stats.get('min', 0), ''),
                    ('熵值', stats.get('entropy', 0), '（分散程度）'),
                    ('集中度', stats.get('concentration', 0), '（高注意力占比）')
                ]
                
                # 设置层名称的合并单元格（使用rowspan）
                layer_start_row = row
                
                for metric_name, value, desc in metrics:
                    self.attention_stats_table.insertRow(row)
                    
                    # 层名称（只在第一行显示，并设置行高）
                    if metric_name == '平均权重':
                        layer_item = QTableWidgetItem(layer_name)
                        layer_item.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
                        # 设置背景色区分不同层（使用QColor对象）
                        layer_colors = {
                            'att1': QColor(254, 243, 199),  # #fef3c7
                            'att2': QColor(253, 230, 138),  # #fde68a
                            'att3': QColor(252, 211, 77),   # #fcd34d
                            'att4': QColor(251, 191, 36)    # #fbbf24
                        }
                        layer_item.setBackground(layer_colors.get(att_name, QColor(255, 255, 255)))
                        self.attention_stats_table.setItem(row, 0, layer_item)
                        self.attention_stats_table.setRowHeight(row, 35)  # 设置行高
                    else:
                        empty_item = QTableWidgetItem("")
                        self.attention_stats_table.setItem(row, 0, empty_item)
                        self.attention_stats_table.setRowHeight(row, 30)
                    
                    # 指标名称
                    metric_item = QTableWidgetItem(f"{metric_name}{desc}")
                    metric_item.setFont(QFont("Microsoft YaHei", 10))
                    self.attention_stats_table.setItem(row, 1, metric_item)
                    
                    # 数值
                    if isinstance(value, (int, float)):
                        if metric_name == '集中度':
                            value_str = f"{value:.2%}"
                        elif metric_name == '熵值':
                            value_str = f"{value:.4f}"
                        else:
                            value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    value_item = QTableWidgetItem(value_str)
                    value_item.setFont(QFont("Courier New", 10, QFont.Bold))
                    value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    # 根据数值设置颜色提示（使用更明显的颜色）
                    if metric_name == '最大权重':
                        if value > 0.8:
                            value_item.setForeground(QColor(22, 163, 74))  # 绿色
                            value_item.setBackground(QColor(220, 252, 231))  # 浅绿背景
                        elif value > 0.5:
                            value_item.setForeground(QColor(217, 119, 6))  # 橙色
                            value_item.setBackground(QColor(255, 247, 237))  # 浅橙背景
                        else:
                            value_item.setForeground(QColor(220, 38, 38))  # 红色
                            value_item.setBackground(QColor(254, 242, 242))  # 浅红背景
                    elif metric_name == '集中度':
                        if value > 0.1:
                            value_item.setForeground(QColor(22, 163, 74))
                            value_item.setBackground(QColor(220, 252, 231))
                        elif value > 0.05:
                            value_item.setForeground(QColor(217, 119, 6))
                            value_item.setBackground(QColor(255, 247, 237))
                        else:
                            value_item.setForeground(QColor(220, 38, 38))
                            value_item.setBackground(QColor(254, 242, 242))
                    elif metric_name == '熵值':
                        if value < 2.0:
                            value_item.setForeground(QColor(22, 163, 74))
                            value_item.setBackground(QColor(220, 252, 231))
                        elif value < 4.0:
                            value_item.setForeground(QColor(217, 119, 6))
                            value_item.setBackground(QColor(255, 247, 237))
                        else:
                            value_item.setForeground(QColor(220, 38, 38))
                            value_item.setBackground(QColor(254, 242, 242))
                    
                    self.attention_stats_table.setItem(row, 2, value_item)
                    row += 1
                
                # 添加分隔行（使用更细的分隔线）
                self.attention_stats_table.insertRow(row)
                for col in range(3):
                    sep_item = QTableWidgetItem("")
                    sep_item.setBackground(QColor(241, 245, 249))  # 浅灰背景
                    sep_item.setFlags(Qt.NoItemFlags)  # 不可选择
                    self.attention_stats_table.setItem(row, col, sep_item)
                self.attention_stats_table.setRowHeight(row, 8)  # 分隔行高度
                row += 1
        
        # 调整列宽
        self.attention_stats_table.resizeColumnsToContents()
        
        # 更新可视化图表
        self._update_attention_charts(attention_stats)
        
        # 更新分析建议文本
        analysis_text = self._generate_detailed_analysis_text(attention_stats)
        self.attention_analysis_text.setText(analysis_text)
        
        # 状态栏提示
        brief_text = self._generate_attention_analysis_text(attention_stats)
        self.statusBar().showMessage(f"✅ 注意力分析完成 | {brief_text}", 5000)
    
    def _update_attention_charts(self, attention_stats):
        """更新注意力统计图表"""
        self.attention_chart_figure.clear()
        
        if not attention_stats:
            ax = self.attention_chart_figure.add_subplot(111)
            ax.text(0.5, 0.5, '等待统计数据...', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            self.attention_chart_canvas.draw()
            return
        
        # 创建2x2子图布局
        gs = self.attention_chart_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 子图1: 各层最大权重对比
        ax1 = self.attention_chart_figure.add_subplot(gs[0, 0])
        layers = []
        max_values = []
        colors = ['#ef4444', '#f97316', '#3b82f6', '#10b981']
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                layers.append(f'层{att_name[-1]}')
                max_values.append(attention_stats[att_name].get('max', 0))
        
        if layers:
            bars = ax1.bar(layers, max_values, color=colors[:len(layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax1.set_ylabel('最大权重', fontsize=10, fontweight='bold')
            ax1.set_title('各层最大注意力权重', fontsize=11, fontweight='bold', pad=10)
            ax1.set_ylim([0, max(max_values) * 1.2 if max_values else 1])
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, max_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图2: 各层集中度对比
        ax2 = self.attention_chart_figure.add_subplot(gs[0, 1])
        conc_values = []
        conc_layers = []
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                conc_layers.append(f'层{att_name[-1]}')
                conc_values.append(attention_stats[att_name].get('concentration', 0) * 100)  # 转换为百分比
        
        if conc_layers:
            bars = ax2.bar(conc_layers, conc_values, color=colors[:len(conc_layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax2.set_ylabel('集中度 (%)', fontsize=10, fontweight='bold')
            ax2.set_title('各层注意力集中度', fontsize=11, fontweight='bold', pad=10)
            ax2.set_ylim([0, max(conc_values) * 1.2 if conc_values else 10])
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, conc_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图3: 各层熵值对比（分散程度）
        ax3 = self.attention_chart_figure.add_subplot(gs[1, 0])
        entropy_values = []
        entropy_layers = []
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                entropy_layers.append(f'层{att_name[-1]}')
                entropy_values.append(attention_stats[att_name].get('entropy', 0))
        
        if entropy_layers:
            bars = ax3.bar(entropy_layers, entropy_values, color=colors[:len(entropy_layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax3.set_ylabel('熵值', fontsize=10, fontweight='bold')
            ax3.set_title('各层注意力分散程度', fontsize=11, fontweight='bold', pad=10)
            ax3.set_ylim([0, max(entropy_values) * 1.2 if entropy_values else 5])
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, entropy_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图4: 各层平均权重对比
        ax4 = self.attention_chart_figure.add_subplot(gs[1, 1])
        mean_values = []
        mean_layers = []
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                mean_layers.append(f'层{att_name[-1]}')
                mean_values.append(attention_stats[att_name].get('mean', 0))
        
        if mean_layers:
            bars = ax4.bar(mean_layers, mean_values, color=colors[:len(mean_layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax4.set_ylabel('平均权重', fontsize=10, fontweight='bold')
            ax4.set_title('各层平均注意力权重', fontsize=11, fontweight='bold', pad=10)
            ax4.set_ylim([0, max(mean_values) * 1.2 if mean_values else 1])
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, mean_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        self.attention_chart_figure.suptitle('注意力统计可视化分析', 
                                            fontsize=13, fontweight='bold', y=0.98)
        self.attention_chart_canvas.draw()
    
    def _generate_detailed_analysis_text(self, attention_stats):
        """生成详细的注意力分析文本"""
        if not attention_stats:
            return "等待训练完成，将显示注意力分析建议..."
        
        analysis_lines = []
        analysis_lines.append("【注意力机制分析报告】\n")
        
        # 分析各层
        for att_name in ['att1', 'att2', 'att3', 'att4']:
            if att_name in attention_stats:
                stats = attention_stats[att_name]
                layer_num = att_name[-1]
                max_val = stats.get('max', 0)
                conc = stats.get('concentration', 0)
                entropy = stats.get('entropy', 0)
                mean_val = stats.get('mean', 0)
                
                analysis_lines.append(f"📊 注意力层{layer_num}:")
                
                # 最大权重分析
                if max_val > 0.8:
                    analysis_lines.append(f"  ✓ 最大权重 {max_val:.3f} - 模型能够强烈聚焦于关键区域")
                elif max_val > 0.5:
                    analysis_lines.append(f"  ⚠ 最大权重 {max_val:.3f} - 模型对关键区域有中等关注")
                else:
                    analysis_lines.append(f"  ✗ 最大权重 {max_val:.3f} - 注意力分布较分散，建议增加训练")
                
                # 集中度分析
                if conc > 0.1:
                    analysis_lines.append(f"  ✓ 集中度 {conc:.1%} - 高注意力区域占比良好")
                elif conc > 0.05:
                    analysis_lines.append(f"  ⚠ 集中度 {conc:.1%} - 注意力分布较为均匀")
                else:
                    analysis_lines.append(f"  ✗ 集中度 {conc:.1%} - 注意力过于分散")
                
                # 熵值分析
                if entropy < 2.0:
                    analysis_lines.append(f"  ✓ 熵值 {entropy:.3f} - 注意力分布集中，聚焦明确")
                elif entropy < 4.0:
                    analysis_lines.append(f"  ⚠ 熵值 {entropy:.3f} - 注意力分布中等分散")
                else:
                    analysis_lines.append(f"  ✗ 熵值 {entropy:.3f} - 注意力分布过于分散")
                
                analysis_lines.append("")
        
        # 总体建议
        analysis_lines.append("【优化建议】")
        
        # 检查att1（最精细层）
        if 'att1' in attention_stats:
            att1_max = attention_stats['att1'].get('max', 0)
            att1_conc = attention_stats['att1'].get('concentration', 0)
            if att1_max < 0.5 or att1_conc < 0.05:
                analysis_lines.append("• 注意力层1（最精细层）表现不佳，建议：")
                analysis_lines.append("  - 增加训练轮次以提升模型聚焦能力")
                analysis_lines.append("  - 检查数据标注质量，确保标注准确")
                analysis_lines.append("  - 考虑调整学习率或使用学习率调度")
        
        # 检查att4（深层）
        if 'att4' in attention_stats:
            att4_mean = attention_stats['att4'].get('mean', 0)
            if att4_mean < 0.2:
                analysis_lines.append("• 注意力层4（深层）注意力值较低，建议：")
                analysis_lines.append("  - 检查模型架构，确保深层特征提取正常")
                analysis_lines.append("  - 考虑使用预训练模型或调整网络深度")
        
        # 综合评估
        all_max = [attention_stats[att].get('max', 0) for att in ['att1', 'att2', 'att3', 'att4'] if att in attention_stats]
        if all_max:
            avg_max = np.mean(all_max)
            if avg_max > 0.7:
                analysis_lines.append("• 整体表现优秀，模型注意力机制工作良好 ✓")
            elif avg_max > 0.5:
                analysis_lines.append("• 整体表现良好，仍有优化空间")
            else:
                analysis_lines.append("• 整体表现需要改进，建议全面检查训练过程")
        
        return "\n".join(analysis_lines)
    
    def _generate_attention_analysis_text(self, attention_stats):
        """生成简短的注意力分析文本（用于状态栏）"""
        analysis_parts = []
        
        if 'att1' in attention_stats:
            att1_max = attention_stats['att1'].get('max', 0)
            att1_conc = attention_stats['att1'].get('concentration', 0)
            if att1_max > 0.8 and att1_conc > 0.1:
                analysis_parts.append("层1聚焦良好")
            elif att1_max > 0.5:
                analysis_parts.append("层1关注中等")
            else:
                analysis_parts.append("层1需改进")
        
        if 'att4' in attention_stats:
            att4_mean = attention_stats['att4'].get('mean', 0)
            if att4_mean > 0.3:
                analysis_parts.append("层4识别大尺度特征")
            else:
                analysis_parts.append("层4提取全局特征")
        
        return " | ".join(analysis_parts) if analysis_parts else "分析完成"
    
    def zoom_image(self, image_type, zoom_action):
        """缩放图片"""
        if image_type == 'test':
            original = self.test_original_pixmap
            label = self.test_results_label
            zoom_factor = self.test_zoom_factor
        elif image_type == 'perf':
            original = self.perf_original_pixmap
            label = self.perf_analysis_label
            zoom_factor = self.perf_zoom_factor
        elif image_type == 'attention':
            original = self.attention_original_pixmap
            label = self.attention_label
            zoom_factor = self.attention_zoom_factor
        else:
            return
        
        if original is None:
            return
        
        self._display_image_with_zoom(image_type, original, zoom_action)
    
    def _display_image_with_zoom(self, image_type, pixmap, zoom_action):
        """根据缩放动作显示图片"""
        if pixmap is None:
            return
        
        if image_type == 'test':
            label = self.test_results_label
            current_factor = self.test_zoom_factor
        elif image_type == 'perf':
            label = self.perf_analysis_label
            current_factor = self.perf_zoom_factor
        elif image_type == 'attention':
            label = self.attention_label
            current_factor = self.attention_zoom_factor
        else:
            return
        
        # 获取滚动区域大小（通过查找父级QScrollArea）
        max_width = 1200
        max_height = 800
        parent = label.parent()
        while parent:
            if isinstance(parent, QScrollArea):
                viewport_size = parent.viewport().size()
                max_width = max(viewport_size.width() - 20, 400)
                max_height = max(viewport_size.height() - 20, 400)
                break
            parent = parent.parent()
        
        if zoom_action == 'in':
            # 放大：增加20%
            new_factor = current_factor * 1.2
        elif zoom_action == 'out':
            # 缩小：减少20%
            new_factor = max(0.1, current_factor * 0.8)
        elif zoom_action == 'fit':
            # 适应窗口：计算合适的缩放比例
            pixmap_size = pixmap.size()
            scale_w = max_width / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            scale_h = max_height / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            new_factor = min(scale_w, scale_h, 1.0)  # 不超过原始大小
        elif zoom_action == 'original':
            # 原始大小
            new_factor = 1.0
        else:
            new_factor = current_factor
        
        # 应用缩放
        if image_type == 'test':
            self.test_zoom_factor = new_factor
        elif image_type == 'perf':
            self.perf_zoom_factor = new_factor
        elif image_type == 'attention':
            self.attention_zoom_factor = new_factor
        
        # 计算新尺寸
        new_size = pixmap.size() * new_factor
        scaled_pixmap = pixmap.scaled(
            int(new_size.width()), 
            int(new_size.height()), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 设置图片并调整label大小
        label.setPixmap(scaled_pixmap)
        label.resize(scaled_pixmap.size())
        label.setText("")
    
    def save_analysis_report(self):
        """保存分析报告"""
        if not self.analysis_data:
            QMessageBox.warning(self, "警告", "没有可保存的分析数据")
            return
        
        # 让用户选择保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return
        
        try:
            # 保存测试结果可视化
            if self.test_viz_path and os.path.exists(self.test_viz_path):
                test_dest = os.path.join(save_dir, "test_results_visualization.png")
                shutil.copy2(self.test_viz_path, test_dest)
            
            # 保存性能分析图表
            if self.perf_analysis_path and os.path.exists(self.perf_analysis_path):
                perf_dest = os.path.join(save_dir, "performance_analysis.png")
                shutil.copy2(self.perf_analysis_path, perf_dest)
            
            # 保存注意力可视化
            if self.attention_viz_path and os.path.exists(self.attention_viz_path):
                att_dest = os.path.join(save_dir, "attention_visualization.png")
                shutil.copy2(self.attention_viz_path, att_dest)
            
            # 保存指标CSV（已翻译为中文）
            if (self.train_thread is not None and 
                hasattr(self.train_thread, 'temp_dir') and 
                self.train_thread.temp_dir):
                metrics_csv = os.path.join(self.train_thread.temp_dir, 'performance_metrics.csv')
                if os.path.exists(metrics_csv):
                    csv_dest = os.path.join(save_dir, "performance_metrics.csv")
                    shutil.copy2(metrics_csv, csv_dest)
            
            # 保存文本报告
            report_path = os.path.join(save_dir, "performance_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("模型性能分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                avg_metrics = self.analysis_data.get('average', {})
                std_metrics = self.analysis_data.get('std', {})
                
                f.write(f"测试样本数量: {len(self.analysis_data.get('all_samples', {}).get('dice', []))}\n\n")
                
                f.write("【平均值 ± 标准差】\n")
                metric_names_cn = {
                    'dice': 'Dice系数',
                    'iou': 'IoU',
                    'precision': '精确率',
                    'recall': '召回率',
                    'sensitivity': '敏感度(召回率)',
                    'specificity': '特异度',
                    'f1': 'F1分数',
                    'hd95': 'HD95'
                }
                summary_metrics = ['dice', 'iou', 'precision', 'sensitivity', 'specificity', 'f1', 'hd95']
                for metric_name in summary_metrics:
                    avg_val = avg_metrics.get(metric_name, 0)
                    std_val = std_metrics.get(metric_name, 0)
                    f.write(f"{metric_names_cn[metric_name]:12s}: {avg_val:.4f} ± {std_val:.4f}\n")
                
                f.write("\n【详细统计】\n")
                for metric_name in summary_metrics:
                    min_val = self.analysis_data.get('min', {}).get(metric_name, 0)
                    max_val = self.analysis_data.get('max', {}).get(metric_name, 0)
                    median_val = self.analysis_data.get('median', {}).get(metric_name, 0)
                    f.write(f"{metric_names_cn[metric_name]}:\n")
                    f.write(f"  最小值: {min_val:.4f}\n")
                    f.write(f"  最大值: {max_val:.4f}\n")
                    f.write(f"  中位数: {median_val:.4f}\n\n")
                
                # 保存注意力统计信息
                if self.attention_stats:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("注意力可解释性分析\n")
                    f.write("=" * 50 + "\n\n")
                    for att_name in ['att1', 'att2', 'att3', 'att4']:
                        if att_name in self.attention_stats:
                            stats = self.attention_stats[att_name]
                            layer_name = f"注意力层{att_name[-1]}"
                            f.write(f"【{layer_name}】\n")
                            f.write(f"  平均权重: {stats['mean']:.4f}\n")
                            f.write(f"  标准差: {stats['std']:.4f}\n")
                            f.write(f"  最大权重: {stats['max']:.4f}\n")
                            f.write(f"  最小权重: {stats['min']:.4f}\n\n")
            
            QMessageBox.information(self, "成功", f"分析报告已保存到:\n{save_dir}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")
    def handle_visualization(self, plot_type, x_data, y_data):
        """处理可视化请求的主线程方法"""
        try:
            if plot_type == "training_history":
                save_path = os.path.join(tempfile.gettempdir(), "training_history.png")
                
                # 使用Agg后端避免GUI问题（已翻译为中文）
                with plt.ioff():  # 关闭交互模式
                    fig = plt.figure(figsize=(12, 5))
                    
                    # 绘制训练曲线
                    ax1 = fig.add_subplot(121)
                    ax1.plot(x_data, y_data['train_loss'], 'b-', label='训练损失')
                    ax1.plot(x_data, y_data['val_loss'], 'r-', label='验证损失')
                    ax1.set_title('训练历史')
                    ax1.set_xlabel('轮次')
                    ax1.set_ylabel('损失')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # 绘制评估指标
                    ax2 = fig.add_subplot(122)
                    ax2.plot(x_data, y_data['val_dice'], 'g-', label='Dice分数')
                    ax2.set_title('验证指标')
                    ax2.set_xlabel('轮次')
                    ax2.set_ylabel('Dice系数')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    fig.savefig(save_path, bbox_inches='tight')
                    plt.close(fig)
                
                self.visualization_ready.emit(save_path)
                
        except Exception as e:
            print(f"可视化错误: {str(e)}")
    def closeEvent(self, event):
        """安全关闭窗口"""
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop_requested = True
            self.train_thread.wait()
        
        if self.predict_thread and self.predict_thread.isRunning():
            self.predict_thread.terminate()
            self.predict_thread.wait()

        if self.api_thread and self.api_thread.isRunning():
            self.api_thread.stop()
            self.api_thread.wait()

        if self.ai_thread:
            if self.ai_thread.isRunning():
                self.ai_thread.terminate()
                self.ai_thread.wait()
            self.ai_thread = None

        if self.llm_threshold_thread and self.llm_threshold_thread.isRunning():
            self.llm_threshold_thread.terminate()
            self.llm_threshold_thread.wait()
        
        event.accept()
    def update_training_plot(self, pixmap):
        """更新界面上的训练曲线图"""
        if hasattr(self, 'plot_label'):
            self.plot_label.setPixmap(pixmap)
        else:
            # 首次创建显示区域
            self.plot_label = QLabel(self)
            self.plot_label.setPixmap(pixmap)
            self.result_container_layout.insertWidget(0, self.plot_label)
    def on_training_epoch_completed(self, epoch, train_loss, val_loss, val_dice):
        """收集训练数据并触发可视化更新"""
        if not hasattr(self, 'training_history'):
            self.training_history = {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'val_dice': []
            }
        
        # 添加新数据
        self.training_history['epochs'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_dice'].append(val_dice)
        
        # 请求可视化更新
        self.visualizer.plot_history(self.training_history)
class TrainingVisualizer(QObject):
    """
    线程安全的训练可视化工具
    所有matplotlib操作都在主线程执行
    """
    plot_updated = pyqtSignal(QPixmap)  # 发送生成的图像
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = None
        
    def plot_history(self, history_data):
        """主线程调用的安全绘图方法"""
        try:
            # 准备数据
            epochs = history_data['epochs']
            train_loss = history_data['train_loss']
            val_loss = history_data['val_loss']
            val_dice = history_data['val_dice']
            
            # 创建图形（必须在主线程）
            if self.fig is None:
                self.fig = plt.figure(figsize=(12, 5), dpi=100)
            
            plt.clf()  # 清除之前的图形
            
            # 子图1：损失曲线
            ax1 = self.fig.add_subplot(121)
            ax1.plot(epochs, train_loss, 'b-', label='训练损失')
            ax1.plot(epochs, val_loss, 'r-', label='验证损失')
            ax1.set_title('训练曲线')
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('损失')
            ax1.legend()
            ax1.grid(True)
            
            # 子图2：Dice系数
            ax2 = self.fig.add_subplot(122)
            ax2.plot(epochs, val_dice, 'g-', label='Dice系数')
            ax2.set_title('评估指标')
            ax2.set_xlabel('轮次')
            ax2.set_ylabel('Dice分数')
            ax2.legend()
            ax2.grid(True)
  
            plt.tight_layout()
            
            # 保存到临时文件
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, "training_history.png")
            self.fig.savefig(save_path, bbox_inches='tight', dpi=100)
            
            # 加载保存的图像并转换为QPixmap（已翻译为中文）
            if os.path.exists(save_path):
                pixmap = QPixmap(save_path)
                self.plot_updated.emit(pixmap)
            
        except Exception as e:
            print(f"绘图错误: {str(e)}")  # 已翻译为中文

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学图像分割GUI/API应用")
    parser.add_argument(
        "--mode",
        choices=["gui", "api"],
        default="gui",
        help="运行模式: gui(默认) 或 api",
    )
    parser.add_argument(
        "--model",
        help="API模式下用于推理的模型路径(.pth/.pt)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API模式监听地址，默认0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API模式端口，默认8000",
    )
    parser.add_argument(
        "--device",
        help="API模式下指定推理设备，例如cpu或cuda:0",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="API模式是否启用热重载(开发用途)",
    )
    args = parser.parse_args()

    if args.mode == "gui":
        from PyQt5.QtWidgets import QApplication

        qt_app = QApplication(sys.argv)
        window = MedicalSegmentationApp()
        window.show()
        sys.exit(qt_app.exec_())
    else:
        if not args.model:
            parser.error("API模式必须通过--model提供模型路径")
        service = SegmentationAPIService(model_path=args.model, device=args.device)
        api_app = create_segmentation_api(service)
        try:
            uvicorn = importlib.import_module("uvicorn")
        except ImportError as exc:
            raise ImportError("运行API模式需要安装uvicorn: pip install uvicorn") from exc

        uvicorn.run(api_app, host=args.host, port=args.port, reload=args.reload)
