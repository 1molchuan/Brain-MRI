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
    2D 模式：单张图像预处理
    【关键修复】处理RGB和灰度图像，确保输入在 [0, 1] 范围，正确转换为 PyTorch 需要的 (B, C, H, W) 格式
    """
    # 确保是 float32 类型
    img_float = image.astype(np.float32)
    
    # 处理RGB图像：转换为灰度（取平均值）
    if len(img_float.shape) == 3 and img_float.shape[2] == 3:
        # RGB图像：转换为灰度 (H, W, 3) -> (H, W)
        img_float = np.mean(img_float, axis=2)
    elif len(img_float.shape) == 3 and img_float.shape[2] == 4:
        # RGBA图像：转换为灰度（只取RGB通道，忽略Alpha）
        img_float = np.mean(img_float[:, :, :3], axis=2)
    
    # 再次检查并强制归一化（双重保险）
    img_max = img_float.max()
    
    if img_max > 1.0:
        img_float = img_float / img_max
    else:
        img_min = img_float.min()
        if img_max > img_min:
            img_float = (img_float - img_min) / (img_max - img_min)
        else:
            img_float = np.zeros_like(img_float)
    
    img_float = np.clip(img_float, 0.0, 1.0)
    
    # 确保是 (H, W) 形状
    if len(img_float.shape) != 2:
        raise ValueError(f"Expected 2D image after processing, got shape {img_float.shape}")
    
    # 【关键修复】正确转换为 PyTorch Tensor 格式: (B, C, H, W)
    # 步骤：1. (H, W) -> 2. (1, H, W) -> 3. (1, 1, H, W)
    tensor = torch.from_numpy(img_float).float()  # (H, W)
    tensor = tensor.unsqueeze(0)  # (1, H, W) - 添加通道维度
    tensor = tensor.unsqueeze(0)  # (1, 1, H, W) - 添加批次维度
    
    # 【Debug】打印 Tensor shape
    print(f"[Debug] preprocess_image_for_2d: Input Tensor Shape = {tensor.shape}")
    
    return tensor


def preprocess_image_for_2_5d(image: np.ndarray) -> torch.Tensor:
    """
    2.5D 模式：将单张图像复制3份堆叠（伪3D）
    为了代码健壮性，不根据文件名寻找相邻切片，统一使用复制堆叠
    【关键修复】处理RGB和灰度图像，确保输入在 [0, 1] 范围
    """
    # 确保是 float32 类型
    img_float = image.astype(np.float32)
    
    # 处理RGB图像：如果已经是RGB，直接使用；否则转换为灰度后复制3份
    if len(img_float.shape) == 3 and img_float.shape[2] == 3:
        # RGB图像：直接使用RGB通道 (H, W, 3)
        # 归一化每个通道
        for c in range(3):
            channel = img_float[:, :, c]
            ch_max = channel.max()
            if ch_max > 1.0:
                img_float[:, :, c] = channel / ch_max
            else:
                ch_min = channel.min()
                if ch_max > ch_min:
                    img_float[:, :, c] = (channel - ch_min) / (ch_max - ch_min)
                else:
                    img_float[:, :, c] = np.zeros_like(channel)
        
        img_float = np.clip(img_float, 0.0, 1.0)
        # 转换为 (3, H, W) 格式
        stacked = np.transpose(img_float, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
    else:
        # 灰度图像：归一化后复制3份
        img_max = img_float.max()
        
        if img_max > 1.0:
            img_float = img_float / img_max
        else:
            img_min = img_float.min()
            if img_max > img_min:
                img_float = (img_float - img_min) / (img_max - img_min)
            else:
                img_float = np.zeros_like(img_float)
        
        img_float = np.clip(img_float, 0.0, 1.0)
        
        # 确保是 (H, W) 形状
        if len(img_float.shape) != 2:
            raise ValueError(f"Expected 2D image, got shape {img_float.shape}")
        
        # 复制3份：stack = [img, img, img]
        stacked = np.stack([img_float, img_float, img_float], axis=0)  # (3, H, W)
    
    # 转换为 Tensor: (1, 3, H, W)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0)
    
    # 【Debug】打印 Tensor shape
    print(f"[Debug] preprocess_image_for_2_5d: Input Tensor Shape = {tensor.shape}")
    
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


# ==================== 主应用 ====================

def main():
    st.title("🏥 脑肿瘤分割系统")
    st.markdown("---")
    
    # ==================== Sidebar ====================
    with st.sidebar:
        st.header("📋 配置")
        
        # 模型文件上传
        model_file = st.file_uploader(
            "上传模型文件 (.pth)",
            type=['pth'],
            help="请上传训练好的模型权重文件"
        )
        
        # 设备选择
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            st.warning("⚠️ 未检测到 GPU，将使用 CPU 推理（速度较慢）")
        
        # 加载模型
        model = None
        model_type = None
        mode = None
        
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
    
    # ==================== Main Area ====================
    
    if model is None:
        st.info("👈 请在左侧上传模型文件以开始")
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
    
    # 推理
    with st.spinner("🔄 正在推理..."):
        with torch.no_grad():
            # 根据模式预处理
            if mode == "2.5D":
                input_tensor = preprocess_image_for_2_5d(original_image)
            else:
                input_tensor = preprocess_image_for_2d(original_image)
            
            # 获取输入 Tensor 的统计信息（用于调试）
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
    # 【关键修复】确定实际使用的阈值（优先使用手动选择的阈值进行调试）
    actual_threshold = threshold  # 默认使用手动选择的阈值
    
    if use_smart_post and smart_post_cfg is not None:
        method = smart_post_cfg.get('method', 'baseline')
        params = smart_post_cfg.get('params', {})
        # 如果配置中有阈值，使用配置的阈值；否则使用用户手动选择的阈值
        if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
            actual_threshold = smart_post_cfg.get('threshold', 0.5)
            st.info(f"🔧 **当前生效阈值**: {actual_threshold:.4f} (来自配置文件)")
        else:
            actual_threshold = threshold  # 使用用户手动选择的阈值
            st.info(f"🔧 **当前生效阈值**: {actual_threshold:.4f} (手动选择)")
        
        # 【关键修复】_apply_strategy 内部硬编码了 0.5 阈值，所以我们需要绕过它
        # 解决方案：先手动应用用户设定的阈值，然后对二值化结果直接应用后处理逻辑
        binary_mask_pre = (prob_map > actual_threshold).astype(np.uint8)
        
        if method == "baseline":
            # baseline方法：直接使用二值化结果
            binary_mask = binary_mask_pre
        elif method == "lcc":
            # LCC方法：保留最大连通域
            labeled, num_features = ndimage.label(binary_mask_pre)
            if num_features > 0:
                sizes = ndimage.sum(binary_mask_pre, labeled, range(1, num_features + 1))
                largest_label = int(np.argmax(sizes)) + 1
                binary_mask = (labeled == largest_label).astype(np.uint8)
            else:
                binary_mask = binary_mask_pre
        elif method == "remove_small":
            # remove_small方法：移除小区域
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
            # 未知方法，回退到baseline
            binary_mask = binary_mask_pre
    else:
        # 【关键修复】直接使用手动选择的阈值，确保变量正确传递
        actual_threshold = threshold
        st.info(f"🔧 **当前生效阈值**: {actual_threshold:.4f} (手动选择)")
        binary_mask = (prob_map > actual_threshold).astype(np.uint8)
    
    # 【调试】验证阈值是否生效
    positive_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    st.caption(f"📊 **阈值验证**: 正像素数 = {positive_pixels} / {total_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
    
    # 计算 Dice（如果有 GT）
    dice_score = None
    if gt_image is not None:
        dice_score = calculate_dice(binary_mask, gt_image)
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
        
        # 显示输入 Tensor 的统计信息
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
            # 调试模式：显示概率热力图
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
            st.caption(f"**正样本像素数** (>{threshold:.2f}): {(prob_map > threshold).sum()} / {prob_map.size} ({(prob_map > threshold).sum() / prob_map.size * 100:.2f}%)")
        else:
            # 正常模式：显示预测结果（二值化 Mask）
            st.markdown("**Prediction**")
            # 【关键修复】确保Mask乘以255，并明确指定vmin/vmax
            pred_display = (binary_mask * 255).astype(np.uint8)
            # 确保值在0-255范围内
            pred_display = np.clip(pred_display, 0, 255).astype(np.uint8)
            
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            
            # 根据后处理模式设置标题（使用实际生效的阈值）
            if use_smart_post and smart_post_cfg is not None:
                method = smart_post_cfg.get('method', 'baseline')
                post_title = f"Prediction ({method}, 阈值={actual_threshold:.4f})"
            else:
                post_title = f"Prediction (阈值={actual_threshold:.4f})"
            
            # 【关键修复】明确指定vmin=0, vmax=255，确保Mask可见
            ax3.imshow(pred_display, cmap='gray', vmin=0, vmax=255)
            ax3.set_title(post_title, fontsize=10)
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close(fig3)
        
        # 在主区域也显示 Dice（如果存在）
        if dice_score is not None:
            st.metric("Dice 系数", f"{dice_score:.4f}")
    
    # ==================== 深度调试面板 ====================
    st.markdown("---")
    with st.expander("🔍 深度调试面板 (Debug Info)", expanded=True):
        # 确定实际使用的阈值
        if use_smart_post and smart_post_cfg is not None:
            if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
                actual_threshold = smart_post_cfg.get('threshold', 0.5)
            else:
                actual_threshold = threshold
        else:
            actual_threshold = threshold
        
        # 1. 显示概率统计数据
        prob_min = prob_map.min()
        prob_max = prob_map.max()
        prob_mean = prob_map.mean()
        prob_median = np.median(prob_map)
        
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
        
        # 3. 阈值效果预览
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


if __name__ == "__main__":
    main()
