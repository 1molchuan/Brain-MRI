"""
FastAPI 后端服务 - 固定模型推理服务
模拟实际部署环境，启动时自动加载 best_model.pth
"""

import os
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
import io
import cv2
from scipy import ndimage

# === Matplotlib 中文显示配置 ===
# 尝试设置中文字体，按优先级尝试 Windows/Linux 常见中文字体
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    pass  # 如果 matplotlib 未安装，忽略

# 导入项目模块
from models import instantiate_model
from utils.model_loader import read_checkpoint_config, load_model_compatible

# ==================== 配置 ====================
MODEL_PATH = Path("best_model.pth")
CONFIG_PATH = Path("best_postprocessing_config.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (256, 256)

# ==================== 全局变量 ====================
app = FastAPI(title="Brain Tumor Segmentation API", version="1.0.0")
model = None
model_mode = None  # "2D" or "2.5D"
smart_post_cfg = None


# ==================== 预处理函数 ====================

def preprocess_image_for_api(image_bytes: bytes) -> torch.Tensor:
    """
    预处理图像：智能通道适配 + ImageNet 标准化
    - RGB 图像：保留原样，归一化到 [0, 1] 后应用 ImageNet 标准化
    - 灰度图像：归一化后复制3份，应用 ImageNet 标准化
    
    Args:
        image_bytes: 图像文件的二进制数据
    
    Returns:
        input_tensor: (1, 3, 256, 256) torch.Tensor，已应用 ImageNet 标准化
    """
    # 1. 读取图像，统一转换为 RGB 模式（确保兼容性）
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 2. Resize 到 256x256（使用 PIL 保持质量）
    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    # 3. 转换为 numpy 数组 (H, W, 3)，范围 [0, 255]
    img_array = np.array(img_resized, dtype=np.float32)
    
    # 4. 归一化到 [0, 1]
    # 前端传来的图片通常是 0-255 (uint8)，必须除以 255.0
    if img_array.max() > 1.0:
        img_normalized = img_array / 255.0
    else:
        # 如果已经在 [0, 1] 范围，直接使用
        img_normalized = img_array
    
    # 确保值在 [0, 1] 范围内
    img_normalized = np.clip(img_normalized, 0.0, 1.0)
    
    # 5. 转换为 PyTorch Tensor 格式: (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
    # RGB 图像：直接转置通道维度
    if len(img_normalized.shape) == 3 and img_normalized.shape[2] == 3:
        # (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1)
        # (3, H, W) -> (1, 3, H, W)
        input_tensor = img_tensor.unsqueeze(0)
    else:
        # 灰度图像：复制3份
        if len(img_normalized.shape) == 2:
            # (H, W) -> (1, H, W) -> (1, 1, H, W) -> (1, 3, H, W)
            input_tensor = torch.from_numpy(img_normalized).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unexpected image shape: {img_normalized.shape}")
    
    # 6. 【关键修复】应用 ImageNet 标准化（Z-Score 标准化）
    # ImageNet 预训练模型的标准化参数
    # 注意：这些 tensor 在 CPU 上，会在移动到设备时自动处理
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    
    # 标准化公式: (x - mean) / std
    input_tensor = (input_tensor - IMAGENET_MEAN) / IMAGENET_STD
    
    return input_tensor


def apply_postprocessing(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    应用后处理：根据配置应用智能后处理或简单阈值
    
    Args:
        prob_map: 概率图 (H, W), 0-1
        threshold: 二值化阈值
    
    Returns:
        binary_mask: (H, W) uint8, 0-255
    """
    global smart_post_cfg
    
    if smart_post_cfg is not None:
        method = smart_post_cfg.get('method', 'baseline')
        params = smart_post_cfg.get('params', {})
        
        # 如果配置中有阈值，使用配置的阈值
        if 'threshold' in smart_post_cfg and smart_post_cfg['threshold'] is not None:
            threshold = smart_post_cfg.get('threshold', 0.5)
        
        # 先应用阈值得到二值化结果
        binary_mask_pre = (prob_map > threshold).astype(np.uint8)
        
        if method == "baseline":
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
            binary_mask = binary_mask_pre
    else:
        # 简单阈值
        binary_mask = (prob_map > threshold).astype(np.uint8)
    
    # 转换为 0-255
    return (binary_mask * 255).astype(np.uint8)


# ==================== 模型加载和初始化 ====================

def load_model_on_startup():
    """启动时加载模型"""
    global model, model_mode, smart_post_cfg
    
    if not MODEL_PATH.exists():
        print(f"⚠️ 警告: 未找到模型文件 {MODEL_PATH}，API 将无法工作")
        return
    
    print(f"🔄 正在加载模型: {MODEL_PATH}")
    
    try:
        # 读取 checkpoint 配置
        config = read_checkpoint_config(str(MODEL_PATH))
        if config is None:
            print(f"❌ 无法读取模型配置")
            return
        
        model_type = config.get('model_type', 'deeplabv3plus')
        
        # 实例化模型（默认使用 DeepLabV3+）
        model = instantiate_model(
            model_type=model_type,
            device=DEVICE,
            swin_params=config.get('swin_params'),
            dstrans_params=config.get('dstrans_params'),
            resnet_params=config.get('resnet_params')
        )
        
        # 加载权重
        success, msg = load_model_compatible(model, str(MODEL_PATH), DEVICE, verbose=False)
        if not success:
            print(f"❌ 模型权重加载失败: {msg}")
            return
        
        model.eval()
        
        # 检测模型模式（简化：API 统一使用 2.5D 模式）
        model_mode = "2.5D"
        
        print(f"✅ 模型加载成功: {model_type}, 模式: {model_mode}")
        
        # 预热：进行一次推理
        print("🔥 正在预热模型...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
            _ = model(dummy_input)
            if isinstance(_, tuple):
                _ = _[0]
        print("✅ 模型预热完成")
        
        # 加载后处理配置
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    smart_post_cfg = json.load(f)
                print(f"✅ 已加载后处理配置: {smart_post_cfg.get('method', 'baseline')}")
            except Exception as e:
                print(f"⚠️ 读取后处理配置失败: {e}")
                smart_post_cfg = None
        else:
            print("ℹ️ 未找到后处理配置文件，将使用默认阈值")
            smart_post_cfg = None
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """启动事件：加载模型"""
    load_model_on_startup()


# ==================== API 接口 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Brain Tumor Segmentation API",
        "status": "running" if model is not None else "model_not_loaded",
        "device": DEVICE,
        "model_mode": model_mode
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...), return_prob_map: bool = Query(False)):
    """
    预测接口
    
    Args:
        file: 上传的图像文件
        return_prob_map: 是否返回概率图（用于调试）
    
    Returns:
        如果 return_prob_map=True: JSON 格式，包含概率图和二值化 mask
        如果 return_prob_map=False: PNG 格式的二进制流（Mask 图像，0-255）
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # 1. 读取图像
        image_bytes = await file.read()
        
        # 2. 预处理
        input_tensor = preprocess_image_for_api(image_bytes)
        input_tensor = input_tensor.to(DEVICE)
        
        # 【输入数据侦探】打印详细的输入 Tensor 统计信息
        print(f"🔍 [Debug] Input Tensor Stats:")
        print(f"  - Shape: {input_tensor.shape}")
        print(f"  - Min: {input_tensor.min().item():.6f}")
        print(f"  - Max: {input_tensor.max().item():.6f}")
        print(f"  - Mean: {input_tensor.mean().item():.6f}")
        print(f"  - Std: {input_tensor.std().item():.6f}")
        # 打印每个通道的统计信息
        for c in range(3):
            channel_data = input_tensor[0, c, :, :]
            print(f"  - Channel {c}: Min={channel_data.min().item():.6f}, Max={channel_data.max().item():.6f}, Mean={channel_data.mean().item():.6f}")
        
        # 3. 推理
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            
            # Sigmoid 激活
            prob_map = torch.sigmoid(output[0, 0]).cpu().numpy()  # (H, W)
        
        # 4. 后处理
        binary_mask = apply_postprocessing(prob_map)
        
        # 5. 根据参数决定返回格式
        if return_prob_map:
            # 返回 JSON 格式，包含概率图和二值化 mask
            from fastapi.responses import JSONResponse
            import base64
            
            # 将概率图编码为 base64（0-1 范围，float32）
            prob_map_bytes = prob_map.tobytes()
            prob_map_b64 = base64.b64encode(prob_map_bytes).decode('utf-8')
            
            # 将二值化 mask 编码为 PNG base64
            mask_image = Image.fromarray(binary_mask, mode='L')
            mask_byte_arr = io.BytesIO()
            mask_image.save(mask_byte_arr, format='PNG')
            mask_byte_arr.seek(0)
            mask_b64 = base64.b64encode(mask_byte_arr.read()).decode('utf-8')
            
            return JSONResponse(content={
                "prob_map_shape": prob_map.shape,
                "prob_map_data": prob_map_b64,
                "prob_map_dtype": "float32",
                "prob_map_min": float(prob_map.min()),
                "prob_map_max": float(prob_map.max()),
                "prob_map_mean": float(prob_map.mean()),
                "binary_mask": mask_b64,
                "threshold": smart_post_cfg.get('threshold', 0.5) if smart_post_cfg and 'threshold' in smart_post_cfg else 0.5
            })
        else:
            # 返回 PNG 格式（向后兼容）
            mask_image = Image.fromarray(binary_mask, mode='L')
            img_byte_arr = io.BytesIO()
            mask_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.read(), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

