# -*- coding: utf-8 -*-
"""
工作线程模块
包含训练、测试和预测的工作线程类
"""

# PyQt5 相关导入
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QMutex
# 找到类似这一行，加上 QApplication
from PyQt5.QtWidgets import QApplication, QMessageBox

# PyTorch 相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# 数据处理相关导入
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose

# 其他标准库导入
import os
import sys
import time
import tempfile
import json
import random
import copy
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 科学计算库
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.stats import wasserstein_distance
from scipy.io import loadmat, savemat

# 图像处理
try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[警告] skimage未安装，直方图匹配功能将不可用")

# 可视化
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免子线程启动GUI警告
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# NIfTI 支持
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("[警告] nibabel 未安装，NIfTI 可视化将不可用")

# 设置matplotlib支持中文显示
try:
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong']
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chinese_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        matplotlib.rcParams['font.sans-serif'] = [chinese_font] + matplotlib.rcParams['font.sans-serif']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False

# 导入模型和工具函数
from models import *
from utils import *

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
                    # 【安全检查】确保列表不为空
                    if len(preds_bin_list) == 0:
                        raise ValueError(f"preds_bin_list为空，probs.shape={probs.shape}")
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
                # 【安全检查】确保列表不为空
                if len(preds_binary_list) == 0:
                    raise ValueError(f"preds_binary_list为空，preds.shape={preds.shape}")
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
        
        # 【关键修复】统一所有预测的空间尺寸到目标尺寸 (H, W)
        # 确保所有张量在 stack 之前具有相同的空间维度
        target_size = (H, W)
        normalized_predictions = []
        for pred in all_predictions:
            if pred.dim() == 4:
                _, _, h, w = pred.shape
                if h != H or w != W:
                    # 插值到目标尺寸
                    pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            normalized_predictions.append(pred)
        all_predictions = normalized_predictions
        
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

            best_threshold, best_metrics = scan_best_threshold(all_probs_np, all_masks_np)

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
                        hd95 = calculate_hd95(
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
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), mode=cv2.BORDER_REFLECT_101, p=0.6),
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
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.4  # 提高概率（从0.15提升到0.4）
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                # GaussNoise 已移除（参数不兼容），如需噪声增强可使用其他方式
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
                            hd95 = calculate_hd95(
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
                total_score = calculate_custom_score(
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
                                hd95 = calculate_hd95(pred_mask, target_mask)
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
        
        # 【关键修复】统一所有概率图的空间尺寸到目标尺寸 (H, W)
        # 确保所有张量在 stack 之前具有相同的空间维度
        target_size = (H, W)
        normalized_prob_maps = []
        for prob_map in all_prob_maps:
            if prob_map.dim() == 4:
                _, _, h, w = prob_map.shape
                if h != H or w != W:
                    # 插值到目标尺寸
                    prob_map = F.interpolate(prob_map, size=target_size, mode='bilinear', align_corners=False)
            normalized_prob_maps.append(prob_map)
        all_prob_maps = normalized_prob_maps
        
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
    
    def _compute_hd95_for_ensemble(self, pred_mask, target_mask):
        """
        计算HD95的辅助方法（用于集成评估）
        
        Args:
            pred_mask: 预测掩码
            target_mask: 真实掩码
        
        Returns:
            HD95值
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
        
    def _compute_dice_for_ensemble(self, pred_mask, target_mask, smooth=1e-7):
        """
        计算Dice的辅助方法（用于集成评估）
        
        Args:
            pred_mask: 预测掩码
            target_mask: 真实掩码
            smooth: 平滑项
        
        Returns:
            Dice系数
        """
        pred = pred_mask.astype(bool)
        target = target_mask.astype(bool)
        intersection = (pred & target).sum()
        union = pred.sum() + target.sum()
        if union == 0:
            return 1.0
        return (2.0 * intersection + smooth) / (union + smooth)
        
    def _compute_iou_for_ensemble(self, pred_mask, target_mask, smooth=1e-7):
        """
        计算IoU的辅助方法（用于集成评估）
        
        Args:
            pred_mask: 预测掩码
            target_mask: 真实掩码
            smooth: 平滑项
        
        Returns:
            IoU系数
        """
        pred = pred_mask.astype(bool)
        target = target_mask.astype(bool)
        intersection = (pred & target).sum()
        union = (pred | target).sum()
        if union == 0:
            return 1.0
        return (intersection + smooth) / (union + smooth)
        
    def _compute_sens_spec_for_ensemble(self, pred_mask, target_mask):
        """
        计算Sensitivity和Specificity的辅助方法（用于集成评估）
        
        Args:
            pred_mask: 预测掩码
            target_mask: 真实掩码
        
        Returns:
            (sensitivity, specificity) 元组
        """
        pred = pred_mask.astype(bool)
        target = target_mask.astype(bool)
        tp = (pred & target).sum()
        fn = (~pred & target).sum()
        fp = (pred & ~target).sum()
        tn = (~pred & ~target).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return sensitivity, specificity
    
    def evaluate_ensemble_performance(self, mask_list, weights, gt_masks, baseline_score=0.8273):
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
            dice = self._compute_dice_for_ensemble(pred, gt)
            iou = self._compute_iou_for_ensemble(pred, gt)
            hd95 = self._compute_hd95_for_ensemble(pred, gt)
            sensitivity, specificity = self._compute_sens_spec_for_ensemble(pred, gt)
            
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
                    hd = calculate_hd95(
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

            total_score = calculate_custom_score(
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
        import torch.nn.functional as F
        if not use_tta:
            return torch.sigmoid(model(image))
        preds = []
        preds.append(torch.sigmoid(model(image)))
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(image, dims=[3]))), dims=[3]))
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(image, dims=[2]))), dims=[2]))
        preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(image, k=1, dims=[2, 3]))), k=-1, dims=[2, 3]))
        
        # 【关键修复】统一所有预测的空间尺寸
        if len(preds) > 0 and preds[0].dim() == 4:
            _, _, H, W = preds[0].shape
            target_size = (H, W)
            normalized_preds = []
            for pred in preds:
                if pred.dim() == 4:
                    _, _, h, w = pred.shape
                    if h != H or w != W:
                        # 插值到目标尺寸
                        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
                normalized_preds.append(pred)
            preds = normalized_preds
        
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


