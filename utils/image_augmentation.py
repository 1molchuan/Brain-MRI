# -*- coding: utf-8 -*-
"""
从utils.image_augmentation模块
"""
from utils.common import *

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


