"""
Streamlit Web App for Brain Tumor Segmentation
支持批量上传 2D 图像序列，兼容 2D 和 2.5D 模型
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import json
import os
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re
from scipy import ndimage
import requests
import io

# === Matplotlib 中文显示配置 ===
# 尝试设置中文字体，按优先级尝试 Windows/Linux 常见中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 导入项目模块
from models import instantiate_model
from utils.model_loader import read_checkpoint_config, load_model_compatible
from utils.smart_postprocessing import _apply_strategy

# ==================== 配置 ====================
st.set_page_config(
    page_title="脑肿瘤分割系统",
    page_icon="🏥",
    layout="wide"
)

# ==================== 工具函数 ====================

def detect_model_mode(model) -> str:
    """
    检测模型模式（2D 或 2.5D）
    通过检查第一层卷积的输入通道数判断
    """
    try:
        # 获取实际模型（处理 DataParallel 包装）
        actual_model = model
        if isinstance(actual_model, torch.nn.DataParallel):
            actual_model = actual_model.module
        
        # 尝试多种方式获取第一层卷积
        first_conv = None
        
        # 方式1: SMP 模型 (DeepLabV3+, U-Net++)
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'encoder'):
            encoder = actual_model.model.encoder
            if hasattr(encoder, 'conv1'):
                first_conv = encoder.conv1
            elif hasattr(encoder, 'stem') and hasattr(encoder.stem, 'conv1'):
                first_conv = encoder.stem.conv1
        
        # 方式2: ResNetUNet
        elif hasattr(actual_model, 'enc0'):
            if hasattr(actual_model.enc0, '0'):
                first_conv = actual_model.enc0[0]
            elif hasattr(actual_model.enc0, 'conv1'):
                first_conv = actual_model.enc0.conv1
        
        # 方式3: ImprovedUNet
        elif hasattr(actual_model, 'conv1'):
            first_conv = actual_model.conv1
        
        # 方式4: TransUNet
        elif hasattr(actual_model, 'encoder') and hasattr(actual_model.encoder, '0'):
            first_conv = actual_model.encoder[0]
        
        if first_conv is None:
            st.warning("⚠️ 无法检测模型输入通道数，默认使用 2D 模式")
            return "2D"
        
        # 检查输入通道数
        in_channels = first_conv.in_channels
        if in_channels == 3:
            return "2.5D"
        elif in_channels == 1:
            return "2D"
        else:
            st.warning(f"⚠️ 检测到非常规输入通道数 ({in_channels})，默认使用 2D 模式")
            return "2D"
    
    except Exception as e:
        st.error(f"❌ 检测模型模式时出错: {e}")
        return "2D"


def natural_sort_key(filename: str) -> List:
    """
    自然排序键：支持数字排序（如 slice_1.png, slice_2.png, slice_10.png）
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    return [convert(c) for c in re.split(r'(\d+)', filename)]


def load_image_file(file) -> Tuple[np.ndarray, Dict]:
    """
    加载单个图像文件并返回数组和元数据
    
    Args:
        file: Streamlit UploadedFile 对象
    
    Returns:
        image: (H, W) 或 (H, W, 3) numpy array (已归一化到 [0, 1])
        metadata: 包含原始尺寸、文件名等信息的字典
    """
    try:
        # 读取图像
        img = Image.open(file)
        
        # 【关键修复】自动检测图像模式，不强制转为灰度
        # 如果是RGB/RGBA，保持RGB；如果是单通道，保持灰度
        if img.mode in ('RGB', 'RGBA'):
            img = img.convert('RGB')
            is_rgb = True
        elif img.mode == 'L':
            is_rgb = False
        else:
            # 其他模式（如P模式）转为RGB
            img = img.convert('RGB')
            is_rgb = True
        
        # 转换为 numpy 数组
        img_array = np.array(img, dtype=np.float32)
        
        # 保存原始尺寸和通道信息
        if len(img_array.shape) == 3:
            original_h, original_w, original_c = img_array.shape
        else:
            original_h, original_w = img_array.shape
            original_c = 1
        
        # Resize 到 256x256（模型输入尺寸）
        target_size = (256, 256)
        if len(img_array.shape) == 3:
            # RGB图像：需要分别resize每个通道
            img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            # 灰度图像
            img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
        
        img_resized = img_resized.astype(np.float32)
        
        # 【关键修复】鲁棒的归一化逻辑
        # 如果最大值 > 1.0，说明是 0-127, 0-255, 0-65535 等范围，归一化到 0-1
        img_max = img_resized.max()
        
        if img_max > 1.0:
            # 使用最大值归一化（更简单直接）
            img_normalized = img_resized / img_max
        else:
            # 如果已经在 [0, 1] 范围，使用 Min-Max 归一化确保完全在 [0, 1]
            img_min = img_resized.min()
            if img_max > img_min:
                img_normalized = (img_resized - img_min) / (img_max - img_min)
            else:
                # 如果所有值相同（全黑或全白），设为0
                img_normalized = np.zeros_like(img_resized)
        
        # 确保值在 [0, 1] 范围内
        img_normalized = np.clip(img_normalized, 0.0, 1.0)
        
        metadata = {
            'filename': file.name,
            'original_shape': img_array.shape,
            'processed_shape': img_normalized.shape,
            'is_rgb': is_rgb,
            'raw_min': float(img_array.min()),
            'raw_max': float(img_array.max()),
            'raw_mean': float(img_array.mean()),
        }
        
        return img_normalized, metadata
    
    except Exception as e:
        st.error(f"❌ 加载图像文件失败 ({file.name}): {e}")
        return None, {}


def preprocess_image_for_2d(image: np.ndarray) -> torch.Tensor:
    """
    2D 模式：智能通道适配 + ImageNet 标准化
    - RGB 图像：保留原样，归一化到 [0, 1] 后应用 ImageNet 标准化
    - 灰度图像：归一化后复制3份，应用 ImageNet 标准化

    【关键修复】添加 ImageNet 标准化，与 server.py 保持一致
    """
    # 确保是 float32 类型
    img_float = image.astype(np.float32)

    # 处理RGB图像：如果已经是RGB，直接使用
    if len(img_float.shape) == 3 and img_float.shape[2] == 3:
        # RGB图像：保留原样，归一化到 [0, 1]
        img_max = img_float.max()
        img_min = img_float.min()

        # 【修复除零风险】统一的鲁棒归一化逻辑
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        elif img_max > 0:
            img_normalized = img_float / img_max
        else:
            img_normalized = np.zeros_like(img_float)

        img_normalized = np.clip(img_normalized, 0.0, 1.0)

        # 转换为 (3, H, W) 格式: (H, W, 3) -> (3, H, W)
        stacked = np.transpose(img_normalized, (2, 0, 1))
    elif len(img_float.shape) == 3 and img_float.shape[2] == 4:
        # RGBA图像：只取RGB通道，忽略Alpha
        img_rgb = img_float[:, :, :3]
        img_max = img_rgb.max()
        img_min = img_rgb.min()

        if img_max > img_min:
            img_normalized = (img_rgb - img_min) / (img_max - img_min)
        elif img_max > 0:
            img_normalized = img_rgb / img_max
        else:
            img_normalized = np.zeros_like(img_rgb)

        img_normalized = np.clip(img_normalized, 0.0, 1.0)
        stacked = np.transpose(img_normalized, (2, 0, 1))
    else:
        # 灰度图像：归一化后复制3份
        # 确保是 (H, W) 形状
        if len(img_float.shape) == 3:
            # 如果是 (H, W, 1)，降维
            if img_float.shape[2] == 1:
                img_float = img_float.squeeze(2)
            else:
                # 其他情况：转换为灰度（取平均值）
                img_float = np.mean(img_float, axis=2)
        elif len(img_float.shape) != 2:
            raise ValueError(f"Expected 2D or 3D image, got shape {img_float.shape}")

        # 归一化
        img_max = img_float.max()
        img_min = img_float.min()

        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        elif img_max > 0:
            img_normalized = img_float / img_max
        else:
            img_normalized = np.zeros_like(img_float)

        img_normalized = np.clip(img_normalized, 0.0, 1.0)

        # 复制3份：stack = [img, img, img]
        stacked = np.stack([img_normalized, img_normalized, img_normalized], axis=0)  # (3, H, W)

    # 转换为 Tensor: (1, 3, H, W)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0)

    # 【关键修复】应用 ImageNet 标准化（与 server.py 保持一致）
    # ImageNet 预训练模型的标准化参数
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    # 标准化公式: (x - mean) / std
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    # 【Debug】打印 Tensor shape 和统计信息
    print(f"[Debug] preprocess_image_for_2d: Input Tensor Shape = {tensor.shape}")
    print(f"[Debug] preprocess_image_for_2d: Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}, Mean={tensor.mean().item():.4f}")

    return tensor


def preprocess_image_for_2_5d(image: np.ndarray) -> torch.Tensor:
    """
    2.5D 模式：智能通道适配 + ImageNet 标准化
    - RGB 图像：保留原样，归一化到 [0, 1] 后应用 ImageNet 标准化
    - 灰度图像：归一化后复制3份，应用 ImageNet 标准化

    【关键修复】添加 ImageNet 标准化，与 server.py 保持一致
    """
    # 确保是 float32 类型
    img_float = image.astype(np.float32)

    # 处理RGB图像：如果已经是RGB，直接使用
    if len(img_float.shape) == 3 and img_float.shape[2] == 3:
        # RGB图像：保留原样，归一化到 [0, 1]
        img_max = img_float.max()
        img_min = img_float.min()

        # 【修复除零风险】统一的鲁棒归一化逻辑
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        elif img_max > 0:
            img_normalized = img_float / img_max
        else:
            img_normalized = np.zeros_like(img_float)

        img_normalized = np.clip(img_normalized, 0.0, 1.0)

        # 转换为 (3, H, W) 格式: (H, W, 3) -> (3, H, W)
        stacked = np.transpose(img_normalized, (2, 0, 1))
    else:
        # 灰度图像：归一化后复制3份
        # 确保是 (H, W) 形状
        if len(img_float.shape) == 3:
            # 如果是 (H, W, 1)，降维
            if img_float.shape[2] == 1:
                img_float = img_float.squeeze(2)
            else:
                # 其他情况：转换为灰度（取平均值）
                img_float = np.mean(img_float, axis=2)
        elif len(img_float.shape) != 2:
            raise ValueError(f"Expected 2D or 3D image, got shape {img_float.shape}")

        # 归一化
        img_max = img_float.max()
        img_min = img_float.min()

        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        elif img_max > 0:
            img_normalized = img_float / img_max
        else:
            img_normalized = np.zeros_like(img_float)

        img_normalized = np.clip(img_normalized, 0.0, 1.0)

        # 复制3份：stack = [img, img, img]
        stacked = np.stack([img_normalized, img_normalized, img_normalized], axis=0)  # (3, H, W)

    # 转换为 Tensor: (1, 3, H, W)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0)

    # 【关键修复】应用 ImageNet 标准化（与 server.py 保持一致）
    # ImageNet 预训练模型的标准化参数
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    # 标准化公式: (x - mean) / std
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    # 【Debug】打印 Tensor shape 和统计信息
    print(f"[Debug] preprocess_image_for_2_5d: Input Tensor Shape = {tensor.shape}")
    print(f"[Debug] preprocess_image_for_2_5d: Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}, Mean={tensor.mean().item():.4f}")

    return tensor


def robust_normalize_for_display(img_array: np.ndarray) -> np.ndarray:
    """
    将任意范围的图像归一化到 0-255 的 uint8 格式，用于显示。
    使用 99% 分位数截断，防止高亮噪点导致整体变暗。
    
    Args:
        img_array: 输入图像数组 (H, W) 或任意形状
    
    Returns:
        归一化后的图像数组，范围 [0, 255]，dtype=uint8
    """
    img_array = np.nan_to_num(img_array)  # 处理 NaN
    
    # 诊断：如果已经是 0-1 或 0-255 的 mask，直接返回
    unique_vals = np.unique(img_array)
    if unique_vals.size <= 2:
        # 可能是二值 mask
        if img_array.max() <= 1.0:
            return (img_array * 255).astype(np.uint8)
        else:
            return img_array.astype(np.uint8)
    
    # 鲁棒归一化：使用 1% 和 99% 分位数截断异常值
    p01 = np.percentile(img_array, 1)
    p99 = np.percentile(img_array, 99)
    
    # 截断异常值
    img_clipped = np.clip(img_array, p01, p99)
    
    # 线性映射到 0-255
    if p99 > p01:
        img_normalized = (img_clipped - p01) / (p99 - p01)
    else:
        # 如果所有值相同，返回全零
        img_normalized = np.zeros_like(img_clipped)
    
    # 转换为 uint8
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    return img_uint8


def calculate_dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-7) -> float:
    """
    计算 Dice 系数
    """
    pred_bin = (pred > 0.5).astype(np.float32).ravel()
    gt_bin = (gt > 0.5).astype(np.float32).ravel()
    
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    intersection = (pred_bin * gt_bin).sum()
    
    # 空掩码特殊处理
    if gt_sum <= smooth and pred_sum <= smooth:
        return 1.0
    if gt_sum <= smooth or pred_sum <= smooth:
        return 0.0
    
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    return float(dice)


# ==================== 模型加载（缓存） ====================

@st.cache_resource
def load_model(model_path: str, device: str):
    """
    加载模型并缓存
    """
    try:
        # 读取 checkpoint 配置
        config = read_checkpoint_config(model_path)
        if config is None:
            st.error("❌ 无法读取模型配置，请检查模型文件")
            return None, None, None
        
        model_type = config.get('model_type', 'improved_unet')
        
        # 实例化模型
        model = instantiate_model(
            model_type=model_type,
            device=device,
            swin_params=config.get('swin_params'),
            dstrans_params=config.get('dstrans_params'),
            resnet_params=config.get('resnet_params')
        )
        
        # 加载权重
        success, msg = load_model_compatible(model, model_path, device, verbose=False)
        if not success:
            st.error(f"❌ 模型权重加载失败: {msg}")
            return None, None, None
        
        model.eval()
        
        # 检测模型模式
        mode = detect_model_mode(model)
        
        return model, model_type, mode
    
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


# ==================== 推理函数（提取为独立函数） ====================

def predict_local(
    model, 
    original_image: np.ndarray, 
    mode: str, 
    device: str,
    threshold: float,
    use_smart_post: bool,
    smart_post_cfg: Optional[Dict]
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    本地推理函数
    
    Returns:
        prob_map: 概率图 (H, W)
        binary_mask: 二值化 Mask (H, W) uint8
        metadata: 包含统计信息的字典
    """
    with torch.no_grad():
        # 根据模式预处理
        if mode == "2.5D":
            input_tensor = preprocess_image_for_2_5d(original_image)
        else:
            input_tensor = preprocess_image_for_2d(original_image)
        
        # 获取输入 Tensor 的统计信息
        input_tensor_min = input_tensor.min().item()
        input_tensor_max = input_tensor.max().item()
        input_tensor_mean = input_tensor.mean().item()
        
        # 移动到设备
        input_tensor = input_tensor.to(device)
        
        # 推理
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        # Sigmoid 激活
        prob_map = torch.sigmoid(output[0, 0]).cpu().numpy()  # (H, W)
    
    # 后处理
    actual_threshold = threshold
    
    if use_smart_post and smart_post_cfg is not None:
        method = smart_post_cfg.get('method', 'baseline')
        params = smart_post_cfg.get('params', {})
        
        if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
            actual_threshold = smart_post_cfg.get('threshold', 0.5)
        else:
            actual_threshold = threshold
        
        binary_mask_pre = (prob_map > actual_threshold).astype(np.uint8)
        
        if method == "baseline":
            binary_mask = binary_mask_pre
        elif method == "lcc":
            labeled, num_features = ndimage.label(binary_mask_pre)
            if num_features > 0:
                sizes = ndimage.sum(binary_mask_pre, labeled, range(1, num_features + 1))
                largest_label = int(np.argmax(sizes)) + 1
                binary_mask = (labeled == largest_label).astype(np.uint8)
            else:
                binary_mask = binary_mask_pre
        elif method == "remove_small":
            min_size = int(params.get("min_size", 0))
            if min_size > 0:
                labeled, num_features = ndimage.label(binary_mask_pre)
                if num_features > 0:
                    sizes = ndimage.sum(binary_mask_pre, labeled, range(1, num_features + 1))
                    mask_filtered = np.zeros_like(binary_mask_pre, dtype=np.uint8)
                    for i, size in enumerate(sizes, start=1):
                        if size >= min_size:
                            mask_filtered[labeled == i] = 1
                    binary_mask = mask_filtered
                else:
                    binary_mask = binary_mask_pre
            else:
                binary_mask = binary_mask_pre
        else:
            binary_mask = binary_mask_pre
    else:
        actual_threshold = threshold
        binary_mask = (prob_map > actual_threshold).astype(np.uint8)
    
    # 【关键修复】确保 binary_mask 是 0-255 范围（用于显示）
    # 如果 binary_mask 是 0/1 范围，乘以 255
    if binary_mask.max() <= 1.0:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    else:
        # 如果已经是 0-255 范围，确保是 uint8
        binary_mask = np.clip(binary_mask, 0, 255).astype(np.uint8)
    
    metadata = {
        'input_tensor_min': input_tensor_min,
        'input_tensor_max': input_tensor_max,
        'input_tensor_mean': input_tensor_mean,
        'actual_threshold': actual_threshold
    }
    
    return prob_map, binary_mask, metadata


def predict_api(image_bytes: bytes, api_url: str = "http://127.0.0.1:8000/predict", return_prob_map: bool = True) -> tuple:
    """
    API 推理函数
    
    Args:
        image_bytes: 图像文件的二进制数据
        api_url: API 服务地址
        return_prob_map: 是否返回概率图（用于调试）
    
    Returns:
        (prob_map, binary_mask, metadata): 
        - prob_map: 概率图 (H, W) float32 [0, 1]，如果失败返回 None
        - binary_mask: 二值化 Mask (H, W) uint8，如果失败返回 None
        - metadata: 包含阈值等信息的字典
    """
    try:
        files = {"file": ("image.tif", image_bytes, "image/tiff")}
        params = {"return_prob_map": return_prob_map}
        response = requests.post(api_url, files=files, params=params, timeout=30)
        
        if response.status_code == 200:
            if return_prob_map:
                # 解析 JSON 响应
                import base64
                # 检查响应内容类型
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    # 如果返回的不是 JSON，可能是服务器不支持 return_prob_map 参数
                    st.warning("⚠️ 服务器不支持返回概率图，降级为仅返回 mask")
                    mask_image = Image.open(io.BytesIO(response.content))
                    binary_mask = np.array(mask_image, dtype=np.uint8)
                    return None, binary_mask, {'mode': 'api', 'actual_threshold': 0.5}
                
                try:
                    data = response.json()
                except ValueError as e:
                    st.error(f"❌ JSON 解析失败: {e}")
                    st.error(f"响应内容: {response.text[:200]}")  # 显示前200个字符
                    return None, None, {}
                
                # 解码概率图
                prob_map_b64 = data.get("prob_map_data")
                prob_map_shape = tuple(data.get("prob_map_shape"))
                prob_map_bytes = base64.b64decode(prob_map_b64)
                prob_map = np.frombuffer(prob_map_bytes, dtype=np.float32).reshape(prob_map_shape)
                
                # 解码二值化 mask
                mask_b64 = data.get("binary_mask")
                mask_bytes = base64.b64decode(mask_b64)
                mask_image = Image.open(io.BytesIO(mask_bytes))
                binary_mask = np.array(mask_image, dtype=np.uint8)
                
                # 提取元数据
                metadata = {
                    'mode': 'api',
                    'actual_threshold': data.get('threshold', 0.5),
                    'prob_map_min': data.get('prob_map_min', 0.0),
                    'prob_map_max': data.get('prob_map_max', 1.0),
                    'prob_map_mean': data.get('prob_map_mean', 0.0)
                }
                
                return prob_map, binary_mask, metadata
            else:
                # 向后兼容：只返回 PNG 图像
                mask_image = Image.open(io.BytesIO(response.content))
                binary_mask = np.array(mask_image, dtype=np.uint8)
                return None, binary_mask, {'mode': 'api', 'actual_threshold': 0.5}
        else:
            st.error(f"❌ API 请求失败: {response.status_code} - {response.text}")
            return None, None, {}
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到 API 服务，请确保 `server.py` 正在运行")
        return None, None, {}
    except Exception as e:
        st.error(f"❌ API 调用失败: {e}")
        return None, None, {}


# ==================== 主应用 ====================

def main():
    st.title("🏥 脑肿瘤分割系统")
    st.markdown("---")
    
    # ==================== Sidebar ====================
    with st.sidebar:
        st.header("📋 配置")
        
        # 【新增】推理模式选择器
        inference_mode = st.radio(
            "推理模式 (Inference Mode)",
            options=["API 服务 (推荐)", "本地调试 (Local)"],
            index=0,  # 默认选择 API 服务
            help="API 服务：使用 FastAPI 后端（快速，无需加载模型）\n本地调试：在本地加载模型（适合测试不同权重）"
        )
        
        st.markdown("---")
        
        # 模型文件上传（仅在本地模式下显示）
        model_file = None
        if inference_mode == "本地调试 (Local)":
            model_file = st.file_uploader(
                "上传模型文件 (.pth)",
                type=['pth'],
                help="请上传训练好的模型权重文件"
            )
        else:
            # API 模式：显示提示信息
            st.info("🌐 **API 模式**：使用后端服务，无需上传模型")
            st.caption("确保 `server.py` 正在运行（`python server.py`）")
        
        # 设备选择（仅在本地模式下使用）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if inference_mode == "本地调试 (Local)" and device == "cpu":
            st.warning("⚠️ 未检测到 GPU，将使用 CPU 推理（速度较慢）")
        
        # 设备选择
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            st.warning("⚠️ 未检测到 GPU，将使用 CPU 推理（速度较慢）")
        
        # 加载模型（仅在本地模式下）
        model = None
        model_type = None
        mode = None
        
        if inference_mode == "本地调试 (Local)":
            if model_file is not None:
                # 保存上传的文件到临时目录
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                model_path = temp_dir / model_file.name
                
                with open(model_path, "wb") as f:
                    f.write(model_file.getbuffer())
                
                with st.spinner("🔄 正在加载模型..."):
                    model, model_type, mode = load_model(str(model_path), device)
                
                if model is not None:
                    st.success("✅ 模型加载成功")
                    st.info(f"**模型类型**: {model_type}")
                    st.info(f"**当前模式**: {mode}")
                else:
                    st.error("❌ 模型加载失败")
                    return
            else:
                st.warning("⚠️ 请上传模型文件以开始")
                return
        else:
            # API 模式：检查服务状态
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                if response.status_code == 200:
                    st.success("✅ API 服务连接正常")
                else:
                    st.error("❌ API 服务异常")
            except:
                st.error("❌ 无法连接到 API 服务，请运行 `python server.py`")
        
        st.markdown("---")
        
        # 后处理配置
        st.subheader("🔧 后处理设置")
        
        # 【修复问题A】允许用户上传后处理配置文件
        cfg_file = st.file_uploader(
            "上传后处理配置文件 (可选)",
            type=['json'],
            help="上传 best_postprocessing_config.json 以启用智能后处理"
        )
        
        smart_post_cfg = None
        
        # 优先使用用户上传的配置文件
        if cfg_file is not None:
            try:
                smart_post_cfg = json.load(cfg_file)
                st.success("✅ 已加载后处理配置文件")
            except Exception as e:
                st.error(f"❌ 解析配置文件失败: {e}")
                smart_post_cfg = None
        
        # 如果用户未上传配置文件，尝试在模型目录查找（向后兼容）
        elif model_file is not None and model_path is not None:
            model_dir = Path(model_path).parent
            cfg_path = model_dir / "best_postprocessing_config.json"
            
            if cfg_path.exists():
                try:
                    with open(cfg_path, 'r', encoding='utf-8') as f:
                        smart_post_cfg = json.load(f)
                    st.info("ℹ️ 检测到模型目录中的后处理配置")
                except Exception as e:
                    st.warning(f"⚠️ 读取后处理配置失败: {e}")
        
        # 启用智能后处理复选框（暂时默认禁用，用于排查问题）
        use_smart_post = st.checkbox(
            "启用智能后处理",
            value=False,  # 【暂时禁用】默认 False，用于排查问题
            disabled=(smart_post_cfg is None),
            help="需要上传 best_postprocessing_config.json 才能启用"
        )
        
        # 阈值选择逻辑
        threshold = 0.5  # 默认值
        post_threshold = None
        
        if use_smart_post and smart_post_cfg is not None:
            # 智能后处理已启用
            method = smart_post_cfg.get('method', 'baseline')
            params = smart_post_cfg.get('params', {})
            
            # 检查配置文件中是否有阈值
            if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
                # 配置文件中有阈值，使用配置的阈值
                post_threshold = smart_post_cfg.get('threshold', 0.5)
                threshold = post_threshold  # 保持兼容性
                st.info(f"**智能后处理已启用**")
                st.info(f"  - **方法**: {method}")
                st.info(f"  - **阈值**: {post_threshold:.3f} (来自配置文件)")
                if params:
                    st.info(f"  - **参数**: {params}")
            else:
                # 配置文件中没有阈值，允许用户手动选择
                st.info(f"**智能后处理已启用**")
                st.info(f"  - **方法**: {method}")
                st.warning("⚠️ 配置文件中未找到阈值，请手动选择")
                if params:
                    st.info(f"  - **参数**: {params}")
                
                # 显示阈值滑块供用户选择
                threshold = st.slider(
                    "阈值（手动选择）",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help="配置文件中没有阈值，请手动选择二值化阈值"
                )
        else:
            # 未启用智能后处理，使用手动阈值
            threshold = st.slider(
                "阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="二值化阈值（默认 0.5）"
            )
        
        st.markdown("---")
        
        # 可视化设置
        st.subheader("🎨 可视化设置")
        alpha = st.slider(
            "透明度",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Mask 叠加透明度"
        )
        
        # 调试模式：显示概率热力图
        show_prob_map = st.checkbox(
            "显示概率热力图 (调试模式)",
            value=False,
            help="勾选后，右侧显示原始概率图而非 Overlay，便于调试"
        )
        
        st.markdown("---")
        
        # ==================== AI 服务配置 ====================
        st.subheader("🤖 AI 服务配置")
        
        # 初始化 AI 配置（使用 session_state 保存）
        if "ai_config" not in st.session_state:
            st.session_state.ai_config = {
                "api_base_url": "http://127.0.0.1:8000",
                "api_key": "",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 2000
            }
        
        # API 服务地址
        ai_api_base_url = st.text_input(
            "API 服务地址",
            value=st.session_state.ai_config.get("api_base_url", "http://127.0.0.1:8000"),
            help="后端 API 服务地址（默认: http://127.0.0.1:8000）",
            key="ai_api_base_url_input"
        )
        st.session_state.ai_config["api_base_url"] = ai_api_base_url
        
        # LLM API Key（从环境变量或用户输入）
        ai_api_key = st.text_input(
            "LLM API Key",
            value=st.session_state.ai_config.get("api_key", os.getenv("OPENAI_API_KEY", "")),
            type="password",
            help="OpenAI/DeepSeek/Moonshot API Key（也可通过环境变量 OPENAI_API_KEY 设置）",
            key="ai_api_key_input"
        )
        st.session_state.ai_config["api_key"] = ai_api_key
        
        # LLM Base URL（可选，用于自定义 API 服务）
        ai_llm_base_url = st.text_input(
            "LLM Base URL（可选）",
            value=st.session_state.ai_config.get("llm_base_url", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")),
            help="LLM API 基础地址（默认: https://api.openai.com/v1）\n"
                 "DeepSeek: https://api.deepseek.com/v1\n"
                 "Moonshot: https://api.moonshot.cn/v1",
            key="ai_llm_base_url_input"
        )
        st.session_state.ai_config["llm_base_url"] = ai_llm_base_url
        
        # 模型选择
        ai_model_name = st.text_input(
            "模型名称",
            value=st.session_state.ai_config.get("model_name", os.getenv("LLM_MODEL", "gpt-3.5-turbo")),
            help="LLM 模型名称（默认: gpt-3.5-turbo）\n"
                 "DeepSeek: deepseek-chat\n"
                 "Moonshot: moonshot-v1-8k",
            key="ai_model_name_input"
        )
        st.session_state.ai_config["model_name"] = ai_model_name
        
        # 高级参数（可折叠）
        with st.expander("⚙️ 高级参数", expanded=False):
            ai_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.ai_config.get("temperature", 0.7),
                step=0.1,
                help="控制输出的随机性（0.0-2.0），值越大越随机"
            )
            st.session_state.ai_config["temperature"] = ai_temperature
            
            ai_max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=8000,
                value=st.session_state.ai_config.get("max_tokens", 2000),
                step=100,
                help="最大生成 token 数"
            )
            st.session_state.ai_config["max_tokens"] = ai_max_tokens
        
        # 状态显示
        if ai_api_key:
            st.success("✅ AI 服务已配置")
        else:
            st.warning("⚠️ 请填写 API Key 以启用 AI 服务")
    
    # ==================== Main Area ====================
    
    # 检查模式状态
    if inference_mode == "本地调试 (Local)" and model is None:
        st.info("👈 请在左侧上传模型文件以开始")
        return
    elif inference_mode == "API 服务 (推荐)":
        # API 模式：检查服务状态
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code != 200:
                st.error("❌ API 服务异常，请检查 `server.py` 是否正常运行")
                return
        except:
            st.error("❌ 无法连接到 API 服务，请运行 `python server.py`")
            return
    
    # 批量图像文件上传
    st.subheader("📁 数据上传")
    
    uploaded_files = st.file_uploader(
        "请选择文件夹内的所有 TIF 文件 (Ctrl+A 全选)",
        type=['tif', 'tiff'],
        accept_multiple_files=True,
        help="支持格式: TIF, TIFF。请上传原图和对应的 _mask.tif 文件。"
    )
    
    if not uploaded_files or len(uploaded_files) == 0:
        st.info("👆 请上传 TIF 图像文件（支持批量上传）")
        return
    
    # ==================== 优化后的文件匹配算法 ====================
    # 智能配对：区分原图和 Mask
    original_files = []
    mask_files = []
    
    for f in uploaded_files:
        # 使用 os.path.splitext 安全地获取文件名和扩展名
        name_lower = f.name.lower()
        if '_mask.tif' in name_lower or '_mask.tiff' in name_lower:
            mask_files.append(f)
        else:
            original_files.append(f)
    
    # 构建 Mask 文件字典映射（优化性能）
    # 键：不带扩展名的文件名（去除 _mask 后缀）
    # 值：Mask 文件对象
    mask_dict = {}
    for mask_file in mask_files:
        # 获取不带扩展名的文件名
        name_without_ext, _ = os.path.splitext(mask_file.name)
        # 转换为小写以便匹配
        name_without_ext_lower = name_without_ext.lower()
        
        # 移除 _mask 后缀（如果存在）
        if name_without_ext_lower.endswith('_mask'):
            base_name = name_without_ext_lower[:-5]  # 移除 '_mask'
        else:
            # 如果文件名中间包含 _mask，尝试其他方式
            base_name = name_without_ext_lower.replace('_mask', '')
        
        # 存储映射（使用原始大小写的 base_name 作为键）
        mask_dict[base_name] = mask_file
    
    # 创建配对列表（优化后的匹配逻辑）
    paired_data = []
    for orig_file in original_files:
        # 获取原图文件名（不带扩展名）
        orig_name_without_ext, _ = os.path.splitext(orig_file.name)
        orig_base_name = orig_name_without_ext.lower()
        
        # 在字典中查找对应的 Mask
        mask_file = mask_dict.get(orig_base_name, None)
        
        paired_data.append({
            'original': orig_file,
            'mask': mask_file,
            'original_name': orig_file.name
        })
    
    # 按文件名排序
    paired_data = sorted(paired_data, key=lambda x: natural_sort_key(x['original_name']))
    
    num_pairs = len(paired_data)
    
    if num_pairs == 0:
        st.warning("⚠️ 未检测到有效的原图文件")
        return
    
    st.success(f"✅ 已配对 {num_pairs} 组数据 | 原图: {len(original_files)} 个 | Mask: {len(mask_files)} 个")
    
    # 显示后处理策略信息（在主区域顶部）
    if use_smart_post and smart_post_cfg is not None:
        method = smart_post_cfg.get('method', 'baseline')
        params = smart_post_cfg.get('params', {})
        # 检查配置文件中是否有阈值
        if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
            post_threshold_display = smart_post_cfg.get('threshold', 0.5)
            threshold_source = "配置文件"
        else:
            post_threshold_display = threshold
            threshold_source = "手动选择"
        st.success(f"🔧 **智能后处理已启用** | 方法: **{method}** | 阈值: **{post_threshold_display:.3f}** ({threshold_source})")
        if params:
            st.caption(f"参数: {params}")
    else:
        st.info(f"🔧 **手动阈值模式** | 当前阈值: **{threshold:.3f}**")
    
    # 显示配对信息
    with st.expander("📋 配对信息", expanded=False):
        for i, pair in enumerate(paired_data):
            mask_status = f"✅ `{pair['mask'].name}`" if pair['mask'] else "❌ 无"
            st.write(f"{i+1}. **{pair['original_name']}** → {mask_status}")
    
    # ==================== 切片浏览器模式 ====================
    st.markdown("---")
    st.subheader("🔍 切片浏览器")
    
    # 切片选择器（在主区域顶部）
    if num_pairs == 1:
        current_idx = 0
        st.info(f"**当前文件**: `{paired_data[0]['original_name']}` (仅1个文件)")
    else:
        current_idx = st.slider(
            "选择切片索引",
            min_value=0,
            max_value=num_pairs - 1,
            value=num_pairs // 2,  # 默认选择中间位置
            step=1,
            help=f"共 {num_pairs} 个切片，使用滑块浏览"
        )
    
    # 获取当前选中的配对数据
    current_pair = paired_data[current_idx]
    
    # 在 Sidebar 显示当前切片信息
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 当前切片信息")
        st.write(f"**索引**: {current_idx + 1}/{num_pairs}")
        st.write(f"**原图**: `{current_pair['original_name']}`")
        if current_pair['mask']:
            st.write(f"**GT Mask**: `{current_pair['mask'].name}`")
        else:
            st.warning("**GT Mask**: 无")
    
    # ==================== 单张图片推理和展示 ====================
    st.markdown("---")
    st.subheader("📊 三列对比展示")
    
    # 加载当前原图
    with st.spinner(f"🔄 正在加载: {current_pair['original_name']}..."):
        original_image, orig_metadata = load_image_file(current_pair['original'])
        if original_image is None:
            st.error(f"❌ 加载原图失败: {current_pair['original_name']}")
            return
        
        # 加载 Mask（如果存在）
        gt_image = None
        if current_pair['mask'] is not None:
            gt_image, _ = load_image_file(current_pair['mask'])
            if gt_image is None:
                st.warning(f"⚠️ 加载 Mask 失败: {current_pair['mask'].name}")
    
    # 推理（根据模式选择）
    prob_map = None
    binary_mask = None
    inference_metadata = {}
    
    if inference_mode == "API 服务 (推荐)":
        # API 模式：调用后端服务
        with st.spinner("🔄 正在调用 API 服务..."):
            # 【关键修复】智能通道适配：优先保留 RGB 信息
            # 1. 如果是 RGB 图像 (H, W, 3)，保留原样
            if original_image.ndim == 3 and original_image.shape[2] == 3:
                # RGB 图像：确保值在 [0, 255] 范围
                if original_image.max() <= 1.0:
                    img_for_api = (original_image * 255).astype(np.uint8)
                else:
                    img_for_api = np.clip(original_image, 0, 255).astype(np.uint8)
                # 转换为 PIL Image (RGB 模式)
                img_pil = Image.fromarray(img_for_api, mode='RGB')
            elif original_image.ndim == 3 and original_image.shape[2] == 4:
                # RGBA 图像：只取 RGB 通道
                img_rgb = original_image[:, :, :3]
                if img_rgb.max() <= 1.0:
                    img_for_api = (img_rgb * 255).astype(np.uint8)
                else:
                    img_for_api = np.clip(img_rgb, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_for_api, mode='RGB')
            else:
                # 灰度图像：转换为灰度模式
                if original_image.ndim == 3:
                    # 如果是 (H, W, 1)，降维
                    if original_image.shape[2] == 1:
                        original_image = original_image.squeeze(2)
                    else:
                        # 其他情况：转换为灰度（取平均值）
                        original_image = np.mean(original_image, axis=2)
                
                # 确保是 2D 数组
                if original_image.ndim != 2:
                    raise ValueError(f"Expected 2D array after dimension cleaning, got shape {original_image.shape}")
                
                # 归一化到 [0, 255]
                if original_image.max() <= 1.0:
                    img_for_api = (original_image * 255).astype(np.uint8)
                else:
                    img_for_api = np.clip(original_image, 0, 255).astype(np.uint8)
                
                # 转换为 PIL Image (灰度模式)
                img_pil = Image.fromarray(img_for_api, mode='L')
            
            # 将图像转换为字节流
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='TIFF')
            img_byte_arr.seek(0)
            
            # 【新增】API 模式支持返回概率图
            prob_map, binary_mask, api_metadata = predict_api(img_byte_arr.getvalue(), return_prob_map=True)
            
            if binary_mask is None:
                st.error("❌ API 推理失败")
                return
            
            # 使用 API 返回的元数据
            inference_metadata = api_metadata.copy()
            if prob_map is not None:
                inference_metadata['prob_map_available'] = True
            else:
                inference_metadata['prob_map_available'] = False
            
            st.success("✅ API 推理完成")
    else:
        # 本地模式：使用本地模型
        with st.spinner("🔄 正在推理..."):
            prob_map, binary_mask, inference_metadata = predict_local(
                model=model,
                original_image=original_image,
                mode=mode,
                device=device,
                threshold=threshold,
                use_smart_post=use_smart_post,
                smart_post_cfg=smart_post_cfg
            )
            
            actual_threshold = inference_metadata.get('actual_threshold', threshold)
            input_tensor_min = inference_metadata.get('input_tensor_min', 0)
            input_tensor_max = inference_metadata.get('input_tensor_max', 1)
            input_tensor_mean = inference_metadata.get('input_tensor_mean', 0)
            
            st.info(f"🔧 **当前生效阈值**: {actual_threshold:.4f}")
            
            # 【调试】验证阈值是否生效
            positive_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            st.caption(f"📊 **阈值验证**: 正像素数 = {positive_pixels} / {total_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
    
    # 【关键修复】使用动态阈值生成用于 Dice 计算的 mask
    # 如果有概率图，使用用户选择的阈值重新生成 mask；否则使用服务器返回的 binary_mask
    dice_mask = None
    if prob_map is not None:
        # 确定实际使用的阈值（考虑智能后处理配置）
        if use_smart_post and smart_post_cfg is not None:
            if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
                dice_threshold = smart_post_cfg.get('threshold', threshold)
            else:
                dice_threshold = threshold
        else:
            dice_threshold = threshold
        
        # 从概率图生成二值化 mask（使用用户选择的阈值）
        dice_mask = (prob_map > dice_threshold).astype(np.uint8)
        
        # 如果启用了智能后处理，应用后处理逻辑
        if use_smart_post and smart_post_cfg is not None:
            method = smart_post_cfg.get('method', 'baseline')
            params = smart_post_cfg.get('params', {})
            
            if method == "lcc":
                # LCC方法：保留最大连通域
                labeled, num_features = ndimage.label(dice_mask)
                if num_features > 0:
                    sizes = ndimage.sum(dice_mask, labeled, range(1, num_features + 1))
                    largest_label = int(np.argmax(sizes)) + 1
                    dice_mask = (labeled == largest_label).astype(np.uint8)
            elif method == "remove_small":
                # remove_small方法：移除小区域
                min_size = int(params.get("min_size", 0))
                if min_size > 0:
                    labeled, num_features = ndimage.label(dice_mask)
                    if num_features > 0:
                        sizes = ndimage.sum(dice_mask, labeled, range(1, num_features + 1))
                        mask_filtered = np.zeros_like(dice_mask, dtype=np.uint8)
                        for i, size in enumerate(sizes, start=1):
                            if size >= min_size:
                                mask_filtered[labeled == i] = 1
                        dice_mask = mask_filtered
    else:
        # 如果没有概率图，使用服务器返回的 binary_mask（降级方案）
        # 需要转换为 0/1 格式用于 Dice 计算
        if binary_mask.max() > 1.0:
            dice_mask = (binary_mask > 127).astype(np.uint8)
        else:
            dice_mask = binary_mask.astype(np.uint8)
    
    # 计算 Dice（如果有 GT）
    dice_score = None
    if gt_image is not None and dice_mask is not None:
        dice_score = calculate_dice(dice_mask, gt_image)
        # 在 Sidebar 显示 Dice
        with st.sidebar:
            st.markdown("---")
            st.metric("🎯 Dice 系数", f"{dice_score:.4f}")
    
    # 三列对比展示
    col1, col2, col3 = st.columns(3)
    
    # 【关键修复】鲁棒的尺寸获取：只取前两个维度，兼容 2D (灰度) 和 3D (RGB) 数组
    H, W = original_image.shape[:2]
    
    with col1:
        st.markdown("**Original Image**")
        # 显示原图
        orig_display = robust_normalize_for_display(original_image)
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.imshow(orig_display, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(current_pair['original_name'], fontsize=10)
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)
        
        # 【Debug】显示图像统计信息
        st.caption("**🔍 Debug 信息**")
        if orig_metadata:
            st.caption(f"Raw Min: {orig_metadata.get('raw_min', 'N/A'):.2f} | "
                      f"Raw Max: {orig_metadata.get('raw_max', 'N/A'):.2f} | "
                      f"Raw Mean: {orig_metadata.get('raw_mean', 'N/A'):.2f}")
        
        # 显示输入 Tensor 的统计信息（仅本地模式）
        if inference_mode == "本地调试 (Local)":
            input_tensor_min = inference_metadata.get('input_tensor_min', 0)
            input_tensor_max = inference_metadata.get('input_tensor_max', 1)
            input_tensor_mean = inference_metadata.get('input_tensor_mean', 0)
            
            st.caption(f"Input Tensor Min: {input_tensor_min:.6f} | "
                      f"Max: {input_tensor_max:.6f} | "
                      f"Mean: {input_tensor_mean:.6f}")
            
            # 检查是否在 [0, 1] 范围内
            if input_tensor_min < 0.0 or input_tensor_max > 1.0:
                st.warning("⚠️ **警告**: 输入 Tensor 不在 [0, 1] 范围内！")
            elif input_tensor_min == 0.0 and input_tensor_max == 0.0:
                st.error("❌ **错误**: 输入 Tensor 全为 0（全黑）！")
            elif input_tensor_min == 1.0 and input_tensor_max == 1.0:
                st.error("❌ **错误**: 输入 Tensor 全为 1（全白）！")
            else:
                st.success("✅ 输入 Tensor 范围正常 [0, 1]")
    
    with col2:
        st.markdown("**Ground Truth**")
        if gt_image is not None:
            # 【关键修复】强制将GT Mask乘以255，确保可见
            # 无论原始值范围如何，都乘以255以确保白色区域可见
            if gt_image.max() <= 1.0:
                # 如果是0/1二值图像，乘以255
                gt_display = (gt_image * 255).astype(np.uint8)
            else:
                # 如果已经是0-255范围，直接使用
                gt_display = gt_image.astype(np.uint8)
            
            # 确保值在0-255范围内
            gt_display = np.clip(gt_display, 0, 255).astype(np.uint8)
            
            # 【调试】显示GT统计信息
            st.caption(f"GT Min: {gt_image.min():.4f}, Max: {gt_image.max():.4f}, 正像素: {(gt_image > 0.5).sum()}")
            
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            # 【关键修复】明确指定vmin=0, vmax=255，确保Mask可见
            ax2.imshow(gt_display, cmap='gray', vmin=0, vmax=255)
            ax2.set_title(current_pair['mask'].name if current_pair['mask'] else "无", fontsize=10)
            ax2.axis('off')
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("无")
    
    with col3:
        if show_prob_map:
            # 调试模式：显示概率热力图（仅本地模式）
            if prob_map is None:
                st.warning("⚠️ API 模式下不提供概率热力图")
                st.markdown("**Prediction**")
            else:
                st.markdown("**概率热力图 (调试模式)**")
                fig3, ax3 = plt.subplots(figsize=(5, 5))
                im = ax3.imshow(prob_map, cmap='plasma', vmin=0, vmax=1)
                ax3.set_title("Probability Map", fontsize=10)
                ax3.axis('off')
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                st.pyplot(fig3)
                plt.close(fig3)
                
                # 显示概率统计
                st.caption(f"**概率统计**: Min={prob_map.min():.4f}, Max={prob_map.max():.4f}, Mean={prob_map.mean():.4f}")
                st.caption(f"**正样本像素数** (>0.5): {(prob_map > 0.5).sum()} / {prob_map.size} ({(prob_map > 0.5).sum() / prob_map.size * 100:.2f}%)")
                actual_threshold = inference_metadata.get('actual_threshold', threshold)
                st.caption(f"**正样本像素数** (>{actual_threshold:.2f}): {(prob_map > actual_threshold).sum()} / {prob_map.size} ({(prob_map > actual_threshold).sum() / prob_map.size * 100:.2f}%)")
        
        # 显示预测结果（如果不在概率图模式，或API模式下）
        if not show_prob_map or (show_prob_map and prob_map is None):
            # 正常模式：显示预测结果（二值化 Mask）
            st.markdown("**Prediction**")
            
            # 【关键修复】动态阈值：优先使用用户选择的阈值，从概率图重新生成 mask
            if prob_map is not None:
                # 如果有概率图，使用用户选择的阈值重新生成 mask
                # 确定实际使用的阈值（考虑智能后处理配置）
                if use_smart_post and smart_post_cfg is not None:
                    # 智能后处理模式：优先使用配置文件中的阈值，否则使用用户选择的阈值
                    if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
                        display_threshold = smart_post_cfg.get('threshold', threshold)
                    else:
                        display_threshold = threshold
                else:
                    # 手动阈值模式：使用用户选择的阈值
                    display_threshold = threshold
                
                # 从概率图生成二值化 mask（使用用户选择的阈值）
                pred_mask = (prob_map > display_threshold).astype(np.uint8)
                
                # 如果启用了智能后处理，应用后处理逻辑
                if use_smart_post and smart_post_cfg is not None:
                    method = smart_post_cfg.get('method', 'baseline')
                    params = smart_post_cfg.get('params', {})
                    
                    if method == "lcc":
                        # LCC方法：保留最大连通域
                        labeled, num_features = ndimage.label(pred_mask)
                        if num_features > 0:
                            sizes = ndimage.sum(pred_mask, labeled, range(1, num_features + 1))
                            largest_label = int(np.argmax(sizes)) + 1
                            pred_mask = (labeled == largest_label).astype(np.uint8)
                    elif method == "remove_small":
                        # remove_small方法：移除小区域
                        min_size = int(params.get("min_size", 0))
                        if min_size > 0:
                            labeled, num_features = ndimage.label(pred_mask)
                            if num_features > 0:
                                sizes = ndimage.sum(pred_mask, labeled, range(1, num_features + 1))
                                mask_filtered = np.zeros_like(pred_mask, dtype=np.uint8)
                                for i, size in enumerate(sizes, start=1):
                                    if size >= min_size:
                                        mask_filtered[labeled == i] = 1
                                pred_mask = mask_filtered
                
                # 转换为 0-255 用于显示
                pred_display = (pred_mask * 255).astype(np.uint8)
                actual_threshold_used = display_threshold
            else:
                # 如果没有概率图，使用服务器返回的 binary_mask（降级方案）
                if binary_mask.max() <= 1.0:
                    pred_display = (binary_mask * 255).astype(np.uint8)
                else:
                    pred_display = binary_mask.astype(np.uint8)
                # 使用服务器返回的阈值（可能不是用户选择的）
                actual_threshold_used = inference_metadata.get('actual_threshold', threshold)
                st.warning("⚠️ 无法获取概率图，使用服务器返回的 mask（阈值可能不匹配）")
            
            # 确保值在0-255范围内
            pred_display = np.clip(pred_display, 0, 255).astype(np.uint8)
            
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            
            # 根据后处理模式设置标题
            if inference_mode == "API 服务 (推荐)":
                if use_smart_post and smart_post_cfg is not None:
                    method = smart_post_cfg.get('method', 'baseline')
                    post_title = f"Prediction (API, {method}, 阈值={actual_threshold_used:.4f})"
                else:
                    post_title = f"Prediction (API, 阈值={actual_threshold_used:.4f})"
            elif use_smart_post and smart_post_cfg is not None:
                method = smart_post_cfg.get('method', 'baseline')
                post_title = f"Prediction ({method}, 阈值={actual_threshold_used:.4f})"
            else:
                post_title = f"Prediction (阈值={actual_threshold_used:.4f})"
            
            # 【关键修复】明确指定vmin=0, vmax=255，确保Mask可见
            ax3.imshow(pred_display, cmap='gray', vmin=0, vmax=255)
            ax3.set_title(post_title, fontsize=10)
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close(fig3)
            
            # 显示当前使用的阈值信息
            st.caption(f"**当前显示阈值**: {actual_threshold_used:.4f} | 正像素: {(pred_display > 127).sum()} / {pred_display.size} ({(pred_display > 127).sum() / pred_display.size * 100:.2f}%)")
        
        # 在主区域也显示 Dice（如果存在）
        if dice_score is not None:
            st.metric("Dice 系数", f"{dice_score:.4f}")
    
    # ==================== 深度调试面板 ====================
    st.markdown("---")
    with st.expander("🔍 深度调试面板 (Debug Info)", expanded=True):
        # API 模式下检查概率图是否可用
        if inference_mode == "API 服务 (推荐)":
            if prob_map is None or not inference_metadata.get('prob_map_available', False):
                st.info("ℹ️ API 模式下概率图不可用（请确保服务器支持返回概率图）")
                return
        
        # 确定实际使用的阈值
        if use_smart_post and smart_post_cfg is not None:
            if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
                actual_threshold = smart_post_cfg.get('threshold', 0.5)
            else:
                actual_threshold = threshold
        else:
            actual_threshold = threshold
        
        # 1. 显示概率统计数据
        if prob_map is not None:
            prob_min = prob_map.min()
            prob_max = prob_map.max()
            prob_mean = prob_map.mean()
            prob_median = np.median(prob_map)
        else:
            st.warning("⚠️ 概率图不可用（API 模式）")
            return
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown("**📊 概率统计**")
            st.write(f"**Min**: {prob_min:.6f}")
            st.write(f"**Max**: {prob_max:.6f}")
            st.write(f"**Mean**: {prob_mean:.6f}")
            st.write(f"**Median**: {prob_median:.6f}")
        
        with col_stat2:
            st.markdown("**🎯 阈值分析**")
            st.write(f"**当前阈值**: {actual_threshold:.4f}")
            st.write(f"**阈值位置**: {'✅ 在范围内' if prob_min <= actual_threshold <= prob_max else '⚠️ 超出范围'}")
            
            # 计算阈值两侧的像素数
            pixels_below = (prob_map < actual_threshold).sum()
            pixels_above = (prob_map >= actual_threshold).sum()
            total_pixels = prob_map.size
            
            st.write(f"**阈值以下**: {pixels_below} ({pixels_below/total_pixels*100:.2f}%)")
            st.write(f"**阈值以上**: {pixels_above} ({pixels_above/total_pixels*100:.2f}%)")
            
            # 警告信息
            if actual_threshold > prob_max:
                st.error(f"⚠️ **警告**: 阈值 ({actual_threshold:.4f}) 大于最大概率值 ({prob_max:.4f})，预测结果将全为0！")
            elif actual_threshold < prob_min:
                st.warning(f"⚠️ **提示**: 阈值 ({actual_threshold:.4f}) 小于最小概率值 ({prob_min:.4f})，预测结果将全为1！")
        
        # 2. 概率分布直方图
        st.markdown("**📈 概率分布直方图**")
        
        # 创建直方图
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        
        # 计算直方图（使用50个bins）
        hist, bins = np.histogram(prob_map.flatten(), bins=50, range=(0.0, 1.0))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 绘制直方图（使用对数坐标）
        ax_hist.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.7, color='steelblue', edgecolor='black')
        ax_hist.set_yscale('log')  # Y轴使用对数坐标
        ax_hist.set_xlabel('概率值', fontsize=12)
        ax_hist.set_ylabel('像素数量 (对数坐标)', fontsize=12)
        ax_hist.set_title('概率分布直方图', fontsize=14)
        ax_hist.grid(True, alpha=0.3)
        
        # 用红线标出当前阈值位置
        ax_hist.axvline(x=actual_threshold, color='red', linestyle='--', linewidth=2, 
                        label=f'当前阈值 = {actual_threshold:.4f}')
        ax_hist.legend()
        
        # 添加统计信息文本（修复网页版黑框问题：使用白色背景和黑色文字）
        stats_text = f'Min: {prob_min:.4f} | Max: {prob_max:.4f} | Mean: {prob_mean:.4f} | Median: {prob_median:.4f}'
        ax_hist.text(0.5, 0.95, stats_text, transform=ax_hist.transAxes, 
                    fontsize=10, verticalalignment='top', color='black',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)
        
        # 3. 阈值效果预览（仅本地模式，需要概率图）
        if prob_map is not None:
            st.markdown("**👁️ 阈值效果预览**")
            col_preview1, col_preview2, col_preview3 = st.columns(3)
            
            # 预览不同阈值的效果
            preview_thresholds = [actual_threshold * 0.5, actual_threshold, actual_threshold * 1.5]
            preview_thresholds = [min(max(t, 0.0), 1.0) for t in preview_thresholds]  # 限制在[0, 1]
            
            for idx, (col, prev_thresh) in enumerate(zip([col_preview1, col_preview2, col_preview3], preview_thresholds)):
                with col:
                    prev_mask = (prob_map > prev_thresh).astype(np.uint8) * 255
                    prev_display = np.clip(prev_mask, 0, 255).astype(np.uint8)
                    
                    fig_prev, ax_prev = plt.subplots(figsize=(3, 3))
                    ax_prev.imshow(prev_display, cmap='gray', vmin=0, vmax=255)
                    ax_prev.set_title(f'阈值={prev_thresh:.3f}', fontsize=9)
                    ax_prev.axis('off')
                    st.pyplot(fig_prev)
                    plt.close(fig_prev)
                    
                    # 显示像素统计
                    positive_pixels = (prob_map > prev_thresh).sum()
                    st.caption(f"正像素: {positive_pixels} ({positive_pixels/prob_map.size*100:.1f}%)")
    
    # ==================== AI 影像诊断助手 ====================
    st.markdown("---")
    st.subheader("🤖 AI 影像诊断助手")
    
    # 初始化对话历史
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # 初始化上下文数据标志
    if "context_sent" not in st.session_state:
        st.session_state.context_sent = False
    
    # 如果完成了预测且还没有发送上下文，自动发送上下文
    if binary_mask is not None and not st.session_state.context_sent:
        # 提取元数据
        tumor_pixels = (binary_mask > 127).sum() if binary_mask.max() > 1.0 else binary_mask.sum()
        total_pixels = binary_mask.size
        tumor_percentage = (tumor_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        
        context_data = {
            "文件名": current_pair['original_name'],
            "肿瘤像素数": f"{tumor_pixels}",
            "总像素数": f"{total_pixels}",
            "肿瘤占比": f"{tumor_percentage:.2f}%",
            "图像尺寸": f"{H}x{W}"
        }
        
        if dice_score is not None:
            context_data["Dice系数"] = f"{dice_score:.4f}"
        
        if prob_map is not None:
            context_data["概率图最大值"] = f"{prob_map.max():.4f}"
            context_data["概率图平均值"] = f"{prob_map.mean():.4f}"
        
        # 构建自动消息
        auto_message = f"我刚刚完成了一张影像的分析。文件名：{current_pair['original_name']}，检测到肿瘤区域像素数：{tumor_pixels}，肿瘤占比：{tumor_percentage:.2f}%"
        if dice_score is not None:
            auto_message += f"，Dice置信度：{dice_score:.4f}"
        auto_message += "。请帮我生成一份简要的分析报告。"
        
        # 添加到对话历史
        st.session_state.chat_messages.append({
            "role": "user",
            "content": auto_message
        })
        
        # 标记上下文已发送
        st.session_state.context_sent = True
        
        # 发送到后端
        try:
            # 使用侧边栏配置的 API 地址
            api_base_url = st.session_state.ai_config.get("api_base_url", "http://127.0.0.1:8000")
            api_url = f"{api_base_url}/chat"
            
            # 准备 LLM 配置（如果用户提供了）
            llm_config = None
            if st.session_state.ai_config.get("api_key"):
                llm_config = {
                    "api_key": st.session_state.ai_config.get("api_key"),
                    "base_url": st.session_state.ai_config.get("llm_base_url", "https://api.openai.com/v1"),
                    "model": st.session_state.ai_config.get("model_name", "gpt-3.5-turbo"),
                    "temperature": st.session_state.ai_config.get("temperature", 0.7),
                    "max_tokens": st.session_state.ai_config.get("max_tokens", 2000)
                }
            
            response = requests.post(
                api_url,
                json={
                    "messages": st.session_state.chat_messages,
                    "context_data": context_data,
                    "llm_config": llm_config
                },
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
                # 流式接收回复
                assistant_reply = ""
                message_placeholder = st.empty()
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        assistant_reply += chunk
                        message_placeholder.markdown(assistant_reply)
                
                # 添加到对话历史
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": assistant_reply
                })
            else:
                st.error(f"❌ AI 服务错误: {response.status_code}")
        except Exception as e:
            st.error(f"❌ 无法连接到 AI 服务: {e}")
            st.info("💡 提示：请确保 `server.py` 正在运行，并且已配置 LLM API Key")
    
    # 显示对话历史
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 准备上下文数据（如果有新的预测结果）
        context_data = None
        if binary_mask is not None:
            tumor_pixels = (binary_mask > 127).sum() if binary_mask.max() > 1.0 else binary_mask.sum()
            total_pixels = binary_mask.size
            tumor_percentage = (tumor_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
            
            context_data = {
                "文件名": current_pair['original_name'],
                "肿瘤像素数": f"{tumor_pixels}",
                "总像素数": f"{total_pixels}",
                "肿瘤占比": f"{tumor_percentage:.2f}%",
                "图像尺寸": f"{H}x{W}"
            }
            
            if dice_score is not None:
                context_data["Dice系数"] = f"{dice_score:.4f}"
            
            if prob_map is not None:
                context_data["概率图最大值"] = f"{prob_map.max():.4f}"
                context_data["概率图平均值"] = f"{prob_map.mean():.4f}"
        
        # 发送到后端
        with st.chat_message("assistant"):
            try:
                # 使用侧边栏配置的 API 地址
                api_base_url = st.session_state.ai_config.get("api_base_url", "http://127.0.0.1:8000")
                api_url = f"{api_base_url}/chat"
                
                # 准备 LLM 配置（如果用户提供了）
                llm_config = None
                if st.session_state.ai_config.get("api_key"):
                    llm_config = {
                        "api_key": st.session_state.ai_config.get("api_key"),
                        "base_url": st.session_state.ai_config.get("llm_base_url", "https://api.openai.com/v1"),
                        "model": st.session_state.ai_config.get("model_name", "gpt-3.5-turbo"),
                        "temperature": st.session_state.ai_config.get("temperature", 0.7),
                        "max_tokens": st.session_state.ai_config.get("max_tokens", 2000)
                    }
                
                response = requests.post(
                    api_url,
                    json={
                        "messages": st.session_state.chat_messages,
                        "context_data": context_data,
                        "llm_config": llm_config
                    },
                    stream=True,
                    timeout=60
                )
                
                if response.status_code == 200:
                    # 流式接收回复
                    assistant_reply = ""
                    message_placeholder = st.empty()
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            assistant_reply += chunk
                            message_placeholder.markdown(assistant_reply)
                    
                    # 添加到对话历史
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": assistant_reply
                    })
                else:
                    error_msg = f"❌ AI 服务错误: {response.status_code}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            except requests.exceptions.ConnectionError:
                error_msg = "❌ 无法连接到 AI 服务，请确保 `server.py` 正在运行"
                st.error(error_msg)
                st.info("💡 提示：请运行 `python server.py` 启动后端服务")
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"❌ 请求失败: {e}"
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话历史"):
        st.session_state.chat_messages = []
        st.session_state.context_sent = False
        st.rerun()


if __name__ == "__main__":
    main()
