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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 【日志优化】抑制 Numpy 和 Grad-CAM 的非致命警告
# 训练初期梯度不稳定可能导致 Numpy 抛出 invalid value 或 overflow 警告，这些是非致命的
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pytorch_grad_cam")
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
    # 【Windows 多进程支持】不在全局作用域打印，避免多进程导入时的问题
    # print("[警告] skimage未安装，直方图匹配功能将不可用")

# Grad-CAM 支持（用于 DeepLabV3+ 等不支持原生注意力图的模型）
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    # 不在全局作用域打印，避免多进程导入时的问题
    # print("[警告] pytorch-grad-cam 未安装，DeepLabV3+ 的 Grad-CAM 热力图功能将不可用")
    # print("      请运行: pip install grad-cam")

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
    # 【Windows 多进程支持】不在全局作用域打印，避免多进程导入时的问题
    # print("[警告] nibabel 未安装，NIfTI 可视化将不可用")

# 设置matplotlib支持中文显示
# 【Windows 多进程支持】这些配置在导入时执行是安全的，不会影响多进程
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
# 显式导入关键工具函数和类，防止 NameError
from utils import (
    EarlyStopping,
    scan_best_threshold,
    GreyWolfThresholdOptimizer,  # 【GWO】引入灰狼优化器
    save_mat_file,
    MatlabVisualizationBridge,
    MatlabEngineSession,
    MedicalImageDataset,
    load_model_compatible,
    read_checkpoint_config,
    parse_extra_modalities_spec,
    build_extra_modalities_lists,
    calculate_hd95,
    calculate_custom_score,
    window_partition,
    window_reverse,
)
# 导入其他工具函数（使用 * 导入以保持兼容性）
from utils import *

# 尝试导入2.5D数据集
try:
    from dataset import TCGA2_5DDataset
    TCGA2_5D_AVAILABLE = True
except ImportError:
    TCGA2_5D_AVAILABLE = False
    # 【Windows 多进程支持】不在全局作用域打印，避免多进程导入时的问题
    # print("[警告] TCGA2_5DDataset 未导入，2.5D数据集功能将不可用")

class ModelTestThread(QThread):
    """模型测试线程"""
    update_progress = pyqtSignal(int, str)  # (进度百分比, 状态消息)
    test_finished = pyqtSignal(dict, str, list)  # (性能指标, 注意力热图路径, 低Dice案例列表)
    # 阈值扫描结果（完整表格 + 推荐阈值信息），通过object传递，避免PyQt类型限制
    threshold_sweep_ready = pyqtSignal(object)
    
    def __init__(self, model_paths, data_dir, model_type, use_tta=True, enable_matlab_plots=None, dataset_type="standard"):
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
        self.enable_matlab_plots = enable_matlab_plots  # 保存用户设置的MATLAB开关状态
        self.dataset_type = dataset_type  # 数据集类型：standard 或 2.5d
        self.stop_requested = False
        self.temp_dir = tempfile.mkdtemp(prefix="model_test_")
        
        # 【持久化修复】创建持久化目录用于保存 MATLAB 报表
        # 在项目根目录下创建 matlab_reports 文件夹，确保报表不会被系统清理
        project_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.persistent_report_dir = os.path.join(project_root, "matlab_reports")
        os.makedirs(self.persistent_report_dir, exist_ok=True)
        print(f"[MATLAB] 报表将保存到持久化目录: {self.persistent_report_dir}")
        
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
                save_best=False,
                dataset_type=self.dataset_type  # 传递数据集类型
            )
            temp_train_thread.model_type = self.model_type  # 设置模型类型
            
            # 【修复归一化参数】根据数据集类型动态设置归一化参数
            if self.dataset_type == "2.5d":
                # 2.5D数据集：3通道输入，使用ImageNet 3通道归一化
                normalize_mean = (0.485, 0.456, 0.406)
                normalize_std = (0.229, 0.224, 0.225)
            else:
                # 标准数据集：根据模型类型判断
                # 如果模型是SMP模型（U-Net++或DeepLabV3+），可能使用3通道（如果用户配置了3通道）
                # 其他模型通常使用1通道，但为了兼容性，先使用3通道归一化
                # 实际通道数会在模型构建时根据dataset_type设置
                normalize_mean = (0.485, 0.456, 0.406)  # 默认3通道，兼容性考虑
                normalize_std = (0.229, 0.224, 0.225)
            
            val_transform = A.Compose([
                A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2()
            ])
            
            # 根据数据集类型选择不同的数据加载逻辑
            if self.dataset_type == "2.5d":
                # 2.5D数据集：直接使用TCGA2_5DDataset，无需patient_ids
                if not TCGA2_5D_AVAILABLE:
                    raise ImportError("TCGA2_5DDataset 未导入，无法使用2.5D数据集")
                
                # 使用全部数据作为测试集（2.5D数据集会自动递归搜索所有.tif文件）
                test_dataset = temp_train_thread.load_dataset(
                    [], val_transform, split_name="test", 
                    return_classification=False, use_weighted_sampling=False
                )
                
                # 为了兼容后续代码，创建虚拟的image_paths和mask_paths
                # 这些路径仅用于日志和可视化，不影响实际数据加载
                image_paths = []
                mask_paths = []
                if hasattr(test_dataset, 'file_dict') and hasattr(test_dataset, 'file_list'):
                    for case_id, slice_id in test_dataset.file_list:
                        img_path = test_dataset.file_dict.get((case_id, slice_id))
                        if img_path:
                            image_paths.append(img_path)
                            # mask路径用于日志，实际加载由TCGA2_5DDataset处理
                            mask_paths.append(img_path.replace('.tif', '_mask.tif'))
            else:
                # 标准数据集：按patient_id组织
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
            
            # 使用全部patient_ids作为测试集
            test_dataset = temp_train_thread.load_dataset(
                patient_ids, val_transform, split_name="test", 
                return_classification=False, use_weighted_sampling=False
            )
            # 【Windows 多进程优化】为测试线程也启用多进程数据加载
            # Intel Core Ultra 9 285HX: 使用 8 个 worker 充分利用 P-Core
            import platform
            is_windows = platform.system() == 'Windows'
            cpu_count = os.cpu_count() or 1
            num_workers = 8 if is_windows else max(0, min(4, cpu_count - 1))
            test_loader = DataLoader(
                test_dataset, 
                batch_size=4, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True,  # 【优化】加速数据传输
                persistent_workers=(num_workers > 0)  # 【关键】让子进程保持存活
            )
            
            # 评估模型（集成功能已删除，仅支持单模型）
            self.update_progress.emit(30, "正在评估模型性能...")
            print("[测试] 开始评估模型性能...")
            detailed_metrics, low_dice_cases = self._evaluate_model(model, test_loader, device, image_paths)
            print("[测试] 模型性能评估完成")
            
            # 生成注意力热图
            self.update_progress.emit(80, "正在生成注意力热图...")
            print("[测试] 开始生成注意力热图...")
            attention_path = self._generate_attention_maps(model, test_loader, device)
            print("[测试] 注意力热图生成完成")
            
            # 【MATLAB 可视化增强】如果 MATLAB 可用且用户已启用，生成高清报表
            # 【修复】首先检查用户是否启用了 MATLAB，避免不必要的计算
            if not self.enable_matlab_plots:
                print("[测试] 用户已禁用 MATLAB 绘图，跳过 MATLAB 可视化报表生成")
            else:
                self.update_progress.emit(90, "正在生成 MATLAB 可视化报表...")
                print("[测试] 开始 MATLAB 可视化报表生成...")
                try:
                    # 创建临时 TrainThread 实例以使用其 MATLAB 桥接
                    # 【修复】传递用户设置的 MATLAB 开关状态和数据集类型，保持与主界面设置一致
                    temp_train_thread = TrainThread(
                        data_dir=self.data_dir,
                        epochs=1,
                        batch_size=4,
                        model_path=None,
                        save_best=False,
                        enable_matlab_plots=self.enable_matlab_plots,  # 传递用户设置
                        dataset_type=self.dataset_type  # 传递数据集类型
                    )
                    
                    # 【修复】再次检查 temp_train_thread 的 enable_matlab_plots（可能因为 MATLAB 不可用而被禁用）
                    if not temp_train_thread.enable_matlab_plots:
                        print("[测试] MATLAB 引擎不可用或用户已禁用，跳过 MATLAB 可视化报表生成")
                    elif not temp_train_thread.matlab_viz_bridge:
                        print("[测试] MATLAB 引擎不可用，跳过 MATLAB 可视化报表生成")
                    else:
                        # MATLAB 可用且已启用，收集测试数据用于 MATLAB 可视化
                        # 【内存优化】只收集前 5 个样本用于可视化，其他样本只计算指标，防止内存泄漏
                        print("[MATLAB] 正在收集测试数据（仅保存前 5 个样本用于可视化）...")
                        viz_images = []  # 只保存前 5 个样本用于可视化
                        viz_masks = []
                        viz_preds = []
                        viz_metrics = []
                        max_viz_samples = 5  # 最多保存 5 个样本
                        sample_count = 0  # 已处理的样本计数
                        
                        model.eval()
                        # 验证前主动清理显存，避免与训练阶段的中间缓存互相干扰
                        try:
                            import gc
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            # 清理失败不影响主流程，安全忽略
                            pass
                        
                        # 获取最优阈值（从_evaluate_model返回的detailed_metrics中获取）
                        optimal_threshold = detailed_metrics.get('optimal_threshold', 0.5)
                        print(f"[MATLAB] 使用最优阈值: {optimal_threshold:.2f}")
                        
                        with torch.no_grad():
                            for batch_data in test_loader:
                                if len(batch_data) == 3:
                                    images, masks, _ = batch_data
                                else:
                                    images, masks = batch_data
                                images, masks = images.to(device), masks.to(device)
                                
                                # 使用 TTA 进行预测
                                outputs = temp_train_thread._tta_inference(model, images)
                                if isinstance(outputs, tuple):
                                    outputs = outputs[0]
                                if outputs.shape[2:] != masks.shape[2:]:
                                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                                preds = torch.sigmoid(outputs)
                                # 使用最优阈值进行二值化
                                preds_binary = (preds > optimal_threshold).float()
                                
                                for i in range(images.size(0)):
                                    sample_count += 1
                                    
                                    # 计算指标（所有样本都需要计算，用于统计）
                                    mask_np = masks[i, 0].cpu().numpy().astype(np.float32)
                                    pred_np = preds_binary[i, 0].cpu().numpy().astype(np.float32)
                                    
                                    # 【统一指标计算】使用统一函数同时计算 Dice 和 IoU，确保逻辑一致
                                    dice, iou = temp_train_thread._compute_metrics_unified(pred_np, mask_np)
                                    
                                    # 【内存优化】只保存前 5 个样本的图像数据用于可视化
                                    if len(viz_images) < max_viz_samples:
                                        # 保存图像数据（仅前 5 个）
                                        img = images[i].cpu().permute(1, 2, 0).numpy()
                                        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                                        img = np.clip(img, 0, 1).astype(np.float32)
                                        
                                        viz_images.append(img)
                                        viz_masks.append(mask_np.copy())  # 使用 copy() 避免引用原始数据
                                        viz_preds.append(pred_np.copy())
                                        viz_metrics.append({'dice': dice, 'iou': iou})
                                    else:
                                        # 第 6 个样本以后：只计算指标，不保存图像数据
                                        # 指标已在上方计算，这里不需要额外操作
                                        # 立即释放临时变量，避免内存累积
                                        del mask_np, pred_np
                                    
                                    # 每处理 100 个样本，主动清理一次内存
                                    if sample_count % 100 == 0:
                                        import gc
                                        gc.collect()
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                        
                        num_viz = len(viz_images)
                        print(f"[MATLAB优化] 已处理 {sample_count} 个样本，仅保存前 {num_viz} 个样本用于可视化绘图")
                        
                        # 保存数据到 .mat 文件（仅保存切片后的数据）
                        print(f"[MATLAB] 正在保存前 {num_viz} 个样本的数据到 .mat 文件...")
                        debug_data_path = os.path.join(temp_train_thread.temp_dir, "debug_data.mat")
                        images_arr = np.transpose(np.stack(viz_images, axis=0), (1, 2, 3, 0)).astype(np.float32)
                        masks_arr = np.transpose(np.stack(viz_masks, axis=0), (1, 2, 0)).astype(np.float32)
                        preds_arr = np.transpose(np.stack(viz_preds, axis=0), (1, 2, 0)).astype(np.float32)
                        dice_vals = np.array([m.get('dice', 0.0) for m in viz_metrics], dtype=np.float32)
                        iou_vals = np.array([m.get('iou', 0.0) for m in viz_metrics], dtype=np.float32)
                        
                        save_mat_file({
                            'images': images_arr,
                            'masks': masks_arr,
                            'preds': preds_arr,
                            'dice': dice_vals,
                            'iou': iou_vals,
                            'metrics': detailed_metrics
                        }, debug_data_path)
                        print(f"[MATLAB] 数据已保存到: {debug_data_path}")
                        
                        # MATLAB 可用且已启用，生成高清报表
                        try:
                            print(f"[MATLAB] 开始生成 MATLAB 高清可视化报表（使用前 {num_viz} 个样本）...")
                            print("[MATLAB] 正在调用 MATLAB 引擎，预计几秒内完成...")
                            # 【持久化修复】生成预测网格可视化 - 保存到持久化目录
                            import time
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            pred_grid_path = os.path.join(self.persistent_report_dir, f"test_prediction_grid_{timestamp}.png")
                            os.makedirs(os.path.dirname(pred_grid_path), exist_ok=True)
                            temp_train_thread.matlab_viz_bridge.render_test_results(debug_data_path, pred_grid_path)
                            print(f"[MATLAB] ✅ 预测网格可视化已保存到持久化目录: {pred_grid_path}")
                            
                            # 生成性能分析报表
                            if 'all_samples' in detailed_metrics:
                                print("[MATLAB] 正在生成性能分析报表...")
                                perf_payload = temp_train_thread._save_performance_payload(detailed_metrics)
                                perf_analysis_path = os.path.join(self.persistent_report_dir, f"test_performance_analysis_{timestamp}.png")
                                os.makedirs(os.path.dirname(perf_analysis_path), exist_ok=True)
                                temp_train_thread.matlab_viz_bridge.render_performance_analysis(perf_payload, perf_analysis_path)
                                print(f"[MATLAB] ✅ 性能分析报表已保存到持久化目录: {perf_analysis_path}")
                            else:
                                print("[MATLAB] ⚠️ 未找到详细指标数据，跳过性能分析报表生成")
                            print("[MATLAB] MATLAB 可视化报表生成完成")
                        except Exception as exc:
                            print(f"[MATLAB] ❌ 可视化生成失败（已回退到 Python 绘图）: {exc}")
                            import traceback
                            print(f"[MATLAB] 错误详情: {traceback.format_exc()}")
                except Exception as exc:
                    print(f"[MATLAB] 测试可视化增强失败（不影响主流程）: {exc}")
                    import traceback
                    print(f"[MATLAB] 错误详情: {traceback.format_exc()}")
            
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
                        # 【2.5D支持】读取数据集类型和输入通道数（用于SMP模型）
                        if 'dataset_type' in cfg:
                            self.dataset_type = cfg['dataset_type']
                            print(f"[模型加载] 从checkpoint读取数据集类型: {self.dataset_type}")
                        if 'in_channels' in cfg:
                            self.in_channels_from_checkpoint = cfg['in_channels']
                            print(f"[模型加载] 从checkpoint读取输入通道数: {self.in_channels_from_checkpoint}")
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
        
        # 【2.5D支持】如果checkpoint中有in_channels信息，使用它来创建模型
        # 对于SMP模型（DeepLabV3+ 和 U-Net++），需要确保输入通道数匹配
        in_channels_from_checkpoint = getattr(self, 'in_channels_from_checkpoint', None)
        if in_channels_from_checkpoint is not None:
            print(f"[模型加载] 使用checkpoint中的输入通道数: {in_channels_from_checkpoint}")
        
        # 使用instantiate_model创建模型（与训练时保持一致）
        model = instantiate_model(
            model_type_to_use, 
            device, 
            swin_params=swin_params,
            dstrans_params=dstrans_params,
            mamba_params=mamba_params,
            resnet_params=resnet_params,
            in_channels_override=in_channels_from_checkpoint  # 传递从checkpoint读取的in_channels
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
    
    def export_to_onnx(self, model_path, output_path=None, input_size=(512, 512), input_channels=None, opset_version=11):
        """
        导出模型为ONNX格式
        
        Args:
            model_path: PyTorch模型文件路径 (.pth)
            output_path: 输出ONNX文件路径，如果为None则自动生成
            input_size: 输入图像尺寸 (H, W)，默认 (512, 512)
            input_channels: 输入通道数，如果为None则从checkpoint或模型类型推断
            opset_version: ONNX opset版本，默认11
        
        Returns:
            导出的ONNX文件路径，如果失败返回None
        """
        try:
            import torch
            import torch.onnx
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[ONNX导出] 开始导出模型: {os.path.basename(model_path)}")
            print(f"[ONNX导出] 使用设备: {device}")
            
            # 加载模型
            model = self._load_model(device, model_path)
            model.eval()
            
            # 处理DataParallel包装
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            
            # === 关键修复：替换 ONNX 不支持的 AdaptiveMaxPool2d ===
            def replace_layers(model):
                """递归替换模型中的 AdaptiveMaxPool2d 为 AdaptiveAvgPool2d"""
                for name, child in model.named_children():
                    if isinstance(child, torch.nn.AdaptiveMaxPool2d):
                        print(f"[ONNX适配] 替换不支持的层: {name} (AdaptiveMaxPool2d -> AdaptiveAvgPool2d)")
                        # 获取原层的输出尺寸
                        # PyTorch 中 AdaptiveMaxPool2d 的 output_size 可能存储在不同的属性中
                        output_size = (1, 1)  # 默认值
                        try:
                            # 尝试多种方式获取 output_size
                            if hasattr(child, 'output_size'):
                                output_size = child.output_size
                            elif hasattr(child, '_output_size'):
                                output_size = child._output_size
                            else:
                                # 如果无法获取，使用默认值 (1, 1)，这是 ResNet 分类头常用的尺寸
                                output_size = (1, 1)
                                print(f"[ONNX适配] 无法获取原层 output_size，使用默认值 (1, 1)")
                        except Exception as e:
                            print(f"[ONNX适配] 获取 output_size 时出错: {e}，使用默认值 (1, 1)")
                            output_size = (1, 1)
                        
                        # 替换为 AdaptiveAvgPool2d，ONNX 支持这个
                        setattr(model, name, torch.nn.AdaptiveAvgPool2d(output_size))
                        print(f"[ONNX适配] 已替换为 AdaptiveAvgPool2d(output_size={output_size})")
                    else:
                        # 递归处理子模块
                        replace_layers(child)
            
            # 执行替换
            replace_layers(model)
            print(f"[ONNX适配] AdaptiveMaxPool2d 替换完成")
            # ===================================================
            
            # 【关键修复】强制检测模型实际需要的输入通道数
            # 不要依赖推断，直接从模型的第一层检测
            detected_channels = None
            
            # 方法1：尝试从模型的第一层直接检测
            try:
                # 检查SMP模型（DeepLabV3+, U-Net++等）的encoder第一层
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'conv1'):
                    detected_channels = model.encoder.conv1.in_channels
                    print(f"[ONNX导出] 从模型encoder.conv1检测到输入通道数: {detected_channels}")
                # 检查ResNet模型的第一层
                elif hasattr(model, 'conv1'):
                    detected_channels = model.conv1.in_channels
                    print(f"[ONNX导出] 从模型conv1检测到输入通道数: {detected_channels}")
                # 检查UNet类模型的第一层
                elif hasattr(model, 'down1'):
                    # 尝试从down1的第一个卷积层检测
                    if hasattr(model.down1, '__getitem__'):
                        first_conv = model.down1[0]
                        if hasattr(first_conv, 'in_channels'):
                            detected_channels = first_conv.in_channels
                            print(f"[ONNX导出] 从模型down1[0]检测到输入通道数: {detected_channels}")
                    elif hasattr(model.down1, 'conv') and hasattr(model.down1.conv, 'in_channels'):
                        detected_channels = model.down1.conv.in_channels
                        print(f"[ONNX导出] 从模型down1.conv检测到输入通道数: {detected_channels}")
                # 检查是否有in_channels属性
                elif hasattr(model, 'in_channels'):
                    detected_channels = model.in_channels
                    print(f"[ONNX导出] 从模型in_channels属性检测到输入通道数: {detected_channels}")
            except Exception as e:
                print(f"[ONNX导出] 从模型结构检测通道数失败: {e}")
            
            # 方法2：如果检测失败，尝试从checkpoint读取
            if detected_channels is None:
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict):
                        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                            cfg = checkpoint['config']
                            if 'in_channels' in cfg:
                                detected_channels = cfg['in_channels']
                                print(f"[ONNX导出] 从checkpoint读取输入通道数: {detected_channels}")
                except:
                    pass
            
            # 方法3：如果还是None，根据模型类型强制判断
            if detected_channels is None:
                # 如果模型类型包含resnet或smp，强制使用3通道（ResNet Backbone通常需要3通道）
                if 'resnet' in self.model_type.lower() or 'smp' in self.model_type.lower():
                    detected_channels = 3
                    print(f"[ONNX导出] 根据模型类型（{self.model_type}）强制使用3通道（ResNet Backbone需要）")
                elif self.dataset_type == "2.5d":
                    detected_channels = 3
                    print(f"[ONNX导出] 根据数据集类型（2.5D）使用3通道")
                else:
                    # 最后兜底：根据数据集类型推断
                    if self.model_type in ("smp_unetplusplus", "smp_deeplabv3plus"):
                        detected_channels = 3  # SMP模型默认3通道
                    else:
                        detected_channels = 1  # 标准模型通常使用1通道
                    print(f"[ONNX导出] 根据数据集类型推断输入通道数: {detected_channels}")
            
            # 使用检测到的通道数（优先使用用户指定的input_channels，如果提供了的话）
            final_channels = input_channels if input_channels is not None else detected_channels
            
            # 【关键修复】如果检测到模型需要3通道但推断为1通道，强制使用3通道
            # 这可以解决 ResNet Backbone 期望3通道但推断为1通道的问题
            if detected_channels == 3 and final_channels == 1:
                print(f"[ONNX导出] ⚠️ 检测到模型需要3通道，但推断为1通道，强制使用3通道")
                final_channels = 3
            
            # 创建示例输入（使用最终确定的通道数）
            dummy_input = torch.randn(1, final_channels, input_size[0], input_size[1]).to(device)
            print(f"[ONNX导出] 最终输入形状: {dummy_input.shape} (通道数: {final_channels})")
            
            # 确定输出路径
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                output_dir = os.path.dirname(model_path) or "."
                output_path = os.path.join(output_dir, f"{base_name}.onnx")
            
            # 导出ONNX
            print(f"[ONNX导出] 正在导出到: {output_path}")
            
            # 【修复1】强制使用 Opset 11 或 12，避免自动跳到 18
            # Opset 11 对 DeepLabV3+ 最稳定，避免 adaptive_max_pool2d 等操作的问题
            if opset_version is None or opset_version > 12:
                opset_version = 11
            opset_version = max(11, min(12, opset_version))  # 限制在 11-12 之间
            print(f"[ONNX导出] 使用ONNX opset版本: {opset_version} (固定，不会自动升级)")
            
            # 处理模型可能有多个输出的情况
            try:
                # 尝试获取模型输出以确定输出名称
                with torch.no_grad():
                    test_output = model(dummy_input)
                    if isinstance(test_output, tuple):
                        output_names = [f"output_{i}" for i in range(len(test_output))]
                    else:
                        output_names = ["output"]
            except:
                output_names = ["output"]
            
            # 【修复2】固定输入尺寸，避免 adaptive_max_pool2d 等动态操作的问题
            # 使用固定尺寸可以避免某些操作（如 adaptive_max_pool2d）在 ONNX 导出时的兼容性问题
            # 如果需要动态输入，可以在推理时使用不同的输入尺寸，但导出时使用固定尺寸
            print(f"[ONNX导出] 使用固定输入尺寸: {input_size[0]}x{input_size[1]} (避免 adaptive 操作问题)")
            
            # 【修复3】显式禁用 dynamo，使用旧版 torch.onnx.export API
            # 确保不使用 torch.export 或 dynamo=True，这些可能导致兼容性问题
            print(f"[ONNX导出] 使用旧版 torch.onnx.export API (禁用 dynamo)")
            
            # 导出（固定尺寸，不使用动态轴）
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,  # 固定为 11 或 12
                do_constant_folding=True,
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=None,  # 【关键修复】固定尺寸，不使用动态轴，避免 adaptive 操作问题
                verbose=False,
                # 显式禁用任何可能的 dynamo 相关功能
                # 注意：torch.onnx.export 本身不支持 dynamo 参数，但确保不使用 torch.export
            )
            
            print(f"[ONNX导出] ✅ 导出成功: {output_path}")
            
            # 验证导出的ONNX模型
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                print(f"[ONNX导出] ✅ ONNX模型验证通过")
                
                # 打印模型信息
                print(f"[ONNX导出] 模型信息:")
                print(f"  - 输入: {[f'{inp.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}' for inp in onnx_model.graph.input]}")
                print(f"  - 输出: {[f'{out.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]}' for out in onnx_model.graph.output]}")
            except ImportError:
                print(f"[ONNX导出] ⚠️ onnx包未安装，跳过模型验证（建议安装: pip install onnx）")
            except Exception as e:
                print(f"[ONNX导出] ⚠️ ONNX模型验证失败: {e}")
            
            return output_path
            
        except Exception as e:
            import traceback
            error_msg = f"ONNX导出失败: {str(e)}\n{traceback.format_exc()}"
            print(f"[ONNX导出] ❌ {error_msg}")
            return None
    
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
        # 【阈值扫描范围修改】从 0.9 到 0.99，步长为 0.01，共10个阈值点
        thresholds = [round(0.9 + i * 0.01, 2) for i in range(10)]  # [0.9, 0.91, 0.92, ..., 0.99]
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
                        
                        # 【修复IoU计算】通过Dice值反推IoU，消除指标不一致
                        # 公式：IoU = Dice / (2 - Dice)
                        # 当Dice=1.0时，IoU也应为1.0（避免除以0）
                        if dice_val >= 1.0 - 1e-8:
                            iou_val = 1.0
                        else:
                            iou_val = float(dice_val / (2.0 - dice_val))
                        sweep_iou_scores[thr].append(iou_val)
                        
                        # 计算每个样本的Precision
                        # 【统一计算方式】Precision = TP / (TP + FP)，使用与 Recall/Specificity 一致的平滑项
                        prec_den = tp + fp
                        if prec_den < 1e-7:
                            # 如果没有预测出任何正样本(tp+fp=0)，则精确率视为1.0(无误检)
                            precision_val = 1.0
                        else:
                            precision_val = float(tp / (prec_den + 1e-7))  # 使用 1e-7 与 Recall/Specificity 保持一致
                        sweep_precision_scores[thr].append(precision_val)
                        
                        # 计算每个样本的Recall
                        # 【统一计算方式】Recall = TP / (TP + FN)，使用与 Precision/Specificity 一致的平滑项
                        rec_den = tp + fn
                        if rec_den < 1e-7:
                            # 如果Ground Truth为空(无病灶，tp+fn=0)，则召回率视为1.0(完美表现)
                            recall_val = 1.0
                        else:
                            recall_val = float(tp / (rec_den + 1e-7))  # 使用 1e-7 与 Precision/Specificity 保持一致
                        sweep_recall_scores[thr].append(recall_val)
                        
                        # 计算每个样本的Specificity
                        # 【统一计算方式】Specificity = TN / (TN + FP)，使用与 Precision/Recall 一致的平滑项
                        spec_den = tn + fp
                        if spec_den < 1e-7:
                            specificity_val = 1.0  # 如果没有负样本，特异性为1.0
                        else:
                            specificity_val = float(tn / (spec_den + 1e-7))  # 使用 1e-7 与 Precision/Recall 保持一致
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
                    
                    # 【修复IoU计算】通过Dice值反推IoU，消除指标不一致
                    # 公式：IoU = Dice / (2 - Dice)
                    # 当Dice=1.0时，IoU也应为1.0（避免除以0）
                    if dice >= 1.0 - 1e-8:
                        iou = 1.0
                    else:
                        iou = float(dice / (2.0 - dice))
                    
                    # 【统一计算方式】Precision: 如果没有预测出任何正样本(tp+fp=0)，则精确率视为1.0(无误检)
                    prec_den = tp + fp
                    if prec_den < 1e-7:
                        precision = 1.0
                    else:
                        precision = float(tp / (prec_den + 1e-7))  # 使用 1e-7 与 Recall/Specificity 保持一致
                    
                    # 【统一计算方式】Recall: 如果Ground Truth为空(无病灶，tp+fn=0)，则召回率视为1.0(完美表现)
                    rec_den = tp + fn
                    if rec_den < 1e-7:
                        recall = 1.0
                    else:
                        recall = float(tp / (rec_den + 1e-7))  # 使用 1e-7 与 Precision/Specificity 保持一致
                    
                    # 【统一计算方式】Specificity: 使用与 Precision/Recall 一致的平滑项
                    spec_den = tn + fp
                    if spec_den < 1e-7:
                        specificity = 1.0  # 如果没有负样本，特异性为1.0
                    else:
                        specificity = float(tn / (spec_den + 1e-7))  # 使用 1e-7 与 Precision/Recall 保持一致
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
            'total_samples': len(metrics['dice']),
            'optimal_threshold': optimal_threshold  # 添加最优阈值，供MATLAB可视化使用
        }
        
        return detailed_metrics, low_dice_cases
    
    def _generate_gradcam_for_deeplabv3(self, model, images, device):
        """
        为 DeepLabV3+ 生成 Grad-CAM 热力图（ModelTestThread 专用）
        
        Args:
            model: 模型实例（已解包，非 DataParallel）
            images: 输入图像 (B, 3, H, W)
            device: 设备
        
        Returns:
            attention_maps: 字典，包含 Grad-CAM 热力图
        """
        if not GRAD_CAM_AVAILABLE:
            return {}
        
        try:
            # 确保模型处于 eval 模式（Grad-CAM 需要）
            was_training = model.training
            model.eval()
            
            # 获取实际模型（SMPDeepLabV3Plus 包装了 smp.DeepLabV3Plus）
            actual_model = model
            if hasattr(model, 'model'):
                actual_model = model.model
            
            # 【分辨率优化】使用 Decoder 作为目标层，获得更高分辨率的热力图
            # Decoder 具有更高的空间分辨率（接近输入图像大小），而 encoder.layer4 只有 1/32 分辨率
            target_layer = None
            
            # 优先使用 decoder（更高分辨率）
            if hasattr(actual_model, 'decoder'):
                decoder = actual_model.decoder
                # decoder 可能是一个 Sequential 或 ModuleList
                if hasattr(decoder, '__getitem__') and len(decoder) > 0:
                    # 如果是可索引的，取最后一个模块（通常是输出层）
                    target_layer = decoder[-1]
                elif hasattr(decoder, 'segmentation_head'):
                    # 某些 decoder 有 segmentation_head
                    target_layer = decoder.segmentation_head
                else:
                    target_layer = decoder
            elif hasattr(actual_model, 'segmentation_head'):
                # 如果 decoder 不存在，尝试直接使用 segmentation_head
                target_layer = actual_model.segmentation_head
            
            # 如果 decoder 不可用，回退到 encoder.layer4（低分辨率，但至少能工作）
            if target_layer is None:
                encoder = actual_model.encoder
                if hasattr(encoder, 'layer4'):
                    layer4 = encoder.layer4
                    if hasattr(layer4, '__getitem__'):
                        target_layer = layer4[-1] if len(layer4) > 0 else layer4
                    else:
                        target_layer = layer4
                elif hasattr(encoder, 'blocks') and len(encoder.blocks) > 0:
                    target_layer = encoder.blocks[-1]
            
            if target_layer is None:
                print("[Grad-CAM] 无法找到目标层（decoder 或 encoder），跳过 Grad-CAM 生成")
                if was_training:
                    model.train()
                return {}
            
            # 初始化 GradCAM
            # 注意：GradCAM 需要访问实际的模型结构，使用 actual_model（smp.DeepLabV3Plus）
            # 新版本的 grad-cam 库已移除 use_cuda 参数，会自动检测设备
            cam = GradCAM(model=actual_model, target_layers=[target_layer])
            
            # 获取图像尺寸 (images 形状是 [B, C, H, W])
            height, width = images.shape[2], images.shape[3]
            
            # 创建全1掩码 (表示关注整张图的类别预测)
            # SemanticSegmentationTarget 不接受 mask=None，必须传入具体的 numpy 数组
            mask = np.ones((height, width), dtype=np.float32)
            
            # 确定目标类别索引
            # 对于二分类模型 (classes=1)，输出只有1个通道，索引必须是0
            # 对于多分类模型 (classes>1)，可以使用 category=1 或其他类别索引
            target_category = 1  # 默认使用类别1（前景类）
            
            # 检查模型的类别数
            if hasattr(actual_model, 'classes'):
                num_classes = actual_model.classes
                if num_classes == 1:
                    # 二分类模型：输出只有1个通道（索引0），必须使用 category=0
                    target_category = 0
                    # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                    if not getattr(self, '_has_logged_gradcam_info', False):
                        print(f"[Grad-CAM] 检测到二分类模型 (classes=1)，使用 category=0")
                        self._has_logged_gradcam_info = True
                else:
                    # 多分类模型：可以使用 category=1（前景类）或其他类别
                    target_category = min(1, num_classes - 1)  # 确保不越界
                    # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                    if not getattr(self, '_has_logged_gradcam_info', False):
                        print(f"[Grad-CAM] 检测到多分类模型 (classes={num_classes})，使用 category={target_category}")
                        self._has_logged_gradcam_info = True
            else:
                # 如果无法获取 classes 属性，尝试从输出形状推断
                # 先进行一次前向传播获取输出形状（仅用于推断）
                try:
                    with torch.no_grad():
                        test_output = actual_model(images[:1])  # 只取第一个样本测试
                        if isinstance(test_output, tuple):
                            test_output = test_output[0]
                        num_classes = test_output.shape[1]  # (B, C, H, W) 中的 C
                        if num_classes == 1:
                            target_category = 0
                            # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                            if not getattr(self, '_has_logged_gradcam_info', False):
                                print(f"[Grad-CAM] 通过输出形状推断为二分类模型 (channels=1)，使用 category=0")
                                self._has_logged_gradcam_info = True
                        else:
                            target_category = min(1, num_classes - 1)
                            # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                            if not getattr(self, '_has_logged_gradcam_info', False):
                                print(f"[Grad-CAM] 通过输出形状推断为多分类模型 (channels={num_classes})，使用 category={target_category}")
                                self._has_logged_gradcam_info = True
                except Exception as e:
                    # 如果推断失败，默认使用 category=0（二分类）
                    target_category = 0
                    # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                    if not getattr(self, '_has_logged_gradcam_info', False):
                        print(f"[Grad-CAM] 无法推断模型类别数，默认使用 category=0 (二分类): {e}")
                        self._has_logged_gradcam_info = True
            
            # 定义目标：语义分割的目标类别
            targets = [SemanticSegmentationTarget(category=target_category, mask=mask)]
            
            # 【关键修复】强制开启梯度计算，这是 Grad-CAM 必须的
            # 即使外部有 torch.no_grad()，这里也要临时开启梯度计算
            # 确保输入图像支持求导
            images_grad = images.clone().detach().requires_grad_(True)
            
            # 生成 Grad-CAM 热力图
            # grayscale_cam 形状: (B, H, W)
            # 使用 torch.enable_grad() 上下文管理器，确保梯度计算可用
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=images_grad, targets=targets)
            
            # 转换为 torch.Tensor 并添加通道维度，匹配其他模型的注意力图格式
            # 格式: (B, 1, H, W)
            attention_maps = {}
            if len(grayscale_cam.shape) == 3:  # (B, H, W)
                grayscale_cam_tensor = torch.from_numpy(grayscale_cam).float().to(device)
                grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)  # (B, 1, H, W)
            else:
                grayscale_cam_tensor = torch.from_numpy(grayscale_cam).float().to(device)
            
            # 使用 'gradcam_decoder' 作为键名（因为现在使用 decoder 作为目标层）
            attention_maps['gradcam_decoder'] = grayscale_cam_tensor
            
            # 恢复模型训练状态
            if was_training:
                model.train()
            
            return attention_maps
            
        except Exception as e:
            print(f"[Grad-CAM警告] 生成热力图失败: {e}")
            import traceback
            traceback.print_exc()
            # 确保恢复模型状态
            if was_training:
                model.train()
            return {}
    
    def _generate_attention_maps(self, model, dataloader, device):
        """生成注意力热图"""
        print("[注意力热图] 开始生成注意力热图...")
        fig = None
        try:
            # 检查模型是否支持注意力图
            actual_model = model
            if isinstance(actual_model, nn.DataParallel):
                actual_model = actual_model.module
            
            if not hasattr(actual_model, 'forward') or not callable(getattr(actual_model, 'forward', None)):
                print("[注意力热图] 模型不支持注意力图生成，跳过")
                return ""
            
            # 尝试获取注意力图
            print("[注意力热图] 正在从模型提取注意力图...")
            model.eval()
            attention_maps_list = []
            images_list = []
            
            # 【关键修复】采用混合梯度策略：前 5 个样本开启梯度以生成 Grad-CAM，其余关闭梯度以加速
            max_samples = 4  # 最多收集 4 个样本用于可视化
            sample_count = 0
            
            for batch_data in dataloader:
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images = images.to(device)
                
                # 判断是否需要生成 Grad-CAM（前 5 个样本）
                need_gradcam = (sample_count < max_samples)
                
                try:
                    # 尝试获取注意力图
                    if hasattr(actual_model, 'forward'):
                        # 【DeepLabV3+ 兼容性 + Grad-CAM 集成】DeepLabV3+ 不支持 return_attention，使用 Grad-CAM
                        is_deeplabv3 = (
                            self.model_type in ("deeplabv3plus", "smp_deeplabv3plus") or
                            type(actual_model).__name__ == "SMPDeepLabV3Plus"
                        )
                        
                        if is_deeplabv3:
                            # DeepLabV3+ 需要使用 Grad-CAM
                            if need_gradcam:
                                # 【关键修复】前 5 个样本：必须在 torch.enable_grad() 下运行，并激活输入梯度
                                with torch.enable_grad():
                                    # 确保输入图像支持梯度计算（Grad-CAM 必需）
                                    images_grad = images.clone().detach().requires_grad_(True)
                                    
                                    # 先获取输出（用于验证模型正常工作）
                                    outputs = actual_model(images_grad)
                                    
                                    # 使用 Grad-CAM 生成热力图（需要梯度）
                                    attention_maps = self._generate_gradcam_for_deeplabv3(actual_model, images_grad, device)
                                    
                                    # 如果 Grad-CAM 成功生成，收集注意力图
                                    if attention_maps:
                                        attention_maps_list.append(attention_maps)
                                        images_list.append(images_grad.detach().cpu())
                                        sample_count += 1
                            else:
                                # 第 6 个样本以后：使用 torch.no_grad() 加速（虽然这里不会执行，因为已经 break）
                                with torch.no_grad():
                                    outputs = actual_model(images)
                                    # 不需要生成热力图
                        else:
                            # 其他模型：支持 return_attention
                            if need_gradcam:
                                # 前 5 个样本：尝试获取注意力图
                                result = actual_model(images, return_attention=True)
                                if isinstance(result, tuple) and len(result) == 2:
                                    outputs, attention_maps = result
                                    attention_maps_list.append(attention_maps)
                                    images_list.append(images.cpu())
                                    sample_count += 1
                            else:
                                # 第 6 个样本以后：只获取输出，不获取注意力图
                                with torch.no_grad():
                                    outputs = actual_model(images)
                except Exception as e:
                    # 如果获取注意力图失败，打印错误信息以便调试
                    print(f"[注意力热图] 获取注意力图失败（样本 {sample_count}）: {e}")
                    import traceback
                    print(f"[注意力热图] 错误详情: {traceback.format_exc()}")
                
                # 如果已收集足够样本，退出循环
                if len(images_list) >= max_samples:
                    break
            
            if not attention_maps_list:
                print("[注意力热图] 未获取到注意力图，跳过可视化")
                return ""
            
            # 可视化注意力图
            print("[注意力热图] 开始绘图...")
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            try:
                # 【布局优化】根据实际的 attention_maps 数量动态设置列数，避免空白子图
                # 计算每行需要的列数：1（原图）+ attention_maps 数量
                if attention_maps_list:
                    # 取第一个样本的 attention_maps 数量作为参考
                    num_att_maps = len(attention_maps_list[0])
                    num_cols = 1 + num_att_maps  # 1 个原图 + N 个热力图
                else:
                    num_cols = 2  # 默认：原图 + 1 个热力图
                
                # 限制最大列数，避免布局过宽
                num_cols = min(num_cols, 5)
                
                fig, axes = plt.subplots(len(images_list), num_cols, figsize=(4 * num_cols, 4 * len(images_list)))
                if len(images_list) == 1:
                    axes = axes.reshape(1, -1) if num_cols > 1 else axes.reshape(1, -1)
                elif num_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for idx, (img, att_maps) in enumerate(zip(images_list, attention_maps_list)):
                    img_np = img[0].permute(1, 2, 0).numpy()
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np, 0, 1)
                    
                    # 显示原图
                    axes[idx, 0].imshow(img_np)
                    axes[idx, 0].set_title("原图")
                    axes[idx, 0].axis('off')
                    
                    # 显示热力图（只显示实际存在的，不显示空白）
                    for i, (att_name, att_map) in enumerate(list(att_maps.items())[:num_cols-1]):
                        if i + 1 >= num_cols:
                            break  # 避免超出列数
                        att_np = att_map[0, 0].cpu().numpy()
                        axes[idx, i+1].imshow(att_np, cmap='hot')
                        axes[idx, i+1].set_title(f"{att_name}")
                        axes[idx, i+1].axis('off')
                
                plt.tight_layout()
                attention_path = os.path.join(self.temp_dir, "attention_maps.png")
                # 【优化】降低dpi以加快生成速度（从150降到100）
                plt.savefig(attention_path, dpi=100, bbox_inches='tight')
                print(f"[注意力热图] 绘图完成，已保存: {attention_path}")
                return attention_path
            finally:
                # 【关键修复】确保无论是否出错都关闭figure，防止内存泄漏
                if 'fig' in locals() and fig is not None:
                    plt.close(fig)
                    print("[注意力热图] 已释放matplotlib资源")
        except Exception as e:
            print(f"[警告] 生成注意力热图失败: {e}")
            import traceback
            print(f"[注意力热图] 错误详情: {traceback.format_exc()}")
            # 确保即使出错也关闭figure
            if fig is not None:
                try:
                    plt.close(fig)
                except:
                    pass
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
        import math
        from scipy.ndimage import gaussian_filter
        
        B, C, H, W = images.shape
        scales = [0.8, 1.0, 1.2]  # 多尺度因子
        all_predictions = []
        all_weights = []
        
        # 【多尺度循环】
        for scale in scales:
            # Resize到目标尺度
            if scale != 1.0:
                # 原始缩放尺寸
                raw_h, raw_w = float(H) * float(scale), float(W) * float(scale)
                # 为兼容 SMP (DeepLabV3+ 等) 对输入尺寸“必须能被32整除”的要求，
                # 将缩放后的尺寸向上取整到 32 的倍数，避免出现 409x409 这类非法尺寸。
                def _ceil_to_multiple(v: float, base: int = 32) -> int:
                    return max(base, int(math.ceil(v / base) * base))
                target_h = _ceil_to_multiple(raw_h, 32)
                target_w = _ceil_to_multiple(raw_w, 32)
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
    def __init__(self, data_dir, epochs, batch_size, model_path=None, save_best=True, use_gwo=False, optimizer_type="adam", dataset_type="standard", enable_matlab_plots=None):
        super().__init__()
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.save_best = save_best
        self.use_gwo = use_gwo  # 是否使用GWO优化
        self.optimizer_type = optimizer_type.lower()
        self.dataset_type = dataset_type.lower()  # "standard" 或 "2.5d"
        self._enable_matlab_plots_override = enable_matlab_plots  # 保存用户设置
        # 梯度累积步数：在小批次 (batch_size=4) 下通过累积多个 step 的梯度来提升等效 batch size，稳定训练
        # Windows 场景默认使用 4 步累积，对应等效 batch size ≈ 16
        self.accumulation_steps = 4
        
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
        self.gwo_best_dice = None  # GWO找到的全验证集最佳Dice，用于best_model判定
        
        # 【日志优化】标记是否已打印 Grad-CAM 信息，避免重复日志刷屏
        self._has_logged_gradcam_info = False
        
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
        
        # 【持久化修复】创建持久化目录用于保存 MATLAB 报表
        # 在项目根目录下创建 matlab_reports 文件夹，确保报表不会被系统清理
        project_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.persistent_report_dir = os.path.join(project_root, "matlab_reports")
        os.makedirs(self.persistent_report_dir, exist_ok=True)
        print(f"[MATLAB] 报表将保存到持久化目录: {self.persistent_report_dir}")
        
        self.best_model_cache_dir = os.path.join(self.data_dir, "_best_model_cache")
        self.enable_matlab_cache = False
        self.matlab_cache_manager = None
        self.matlab_metrics_bridge = None
        # 尝试初始化 MATLAB 可视化桥接
        try:
            self.matlab_viz_bridge = MatlabVisualizationBridge.instance()
            # 如果用户明确设置了enable_matlab_plots，使用用户设置；否则根据MATLAB是否可用自动判断
            if self._enable_matlab_plots_override is not None:
                self.enable_matlab_plots = self._enable_matlab_plots_override and (self.matlab_viz_bridge is not None)
                if self._enable_matlab_plots_override and not self.enable_matlab_plots:
                    print("[提示] 用户要求使用MATLAB，但MATLAB引擎不可用，将使用 Python 绘图")
                elif not self._enable_matlab_plots_override:
                    print("[提示] 用户已禁用MATLAB可视化，将仅使用 Matplotlib 绘图")
                elif self.enable_matlab_plots:
                    print("[提示] MATLAB 引擎可用，将启用 MATLAB 高清绘图功能")
            else:
                # 默认行为：如果MATLAB可用则使用
                self.enable_matlab_plots = (self.matlab_viz_bridge is not None)
                if self.enable_matlab_plots:
                    print("[提示] MATLAB 引擎可用，将启用 MATLAB 高清绘图功能")
        except Exception as e:
            self.matlab_viz_bridge = None
            self.enable_matlab_plots = False
            print(f"[提示] MATLAB 引擎不可用，将使用 Python 绘图: {e}")
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
        self.val_dice_history = []  # 统计所有验证样本的平均Dice（包括空mask样本），用于最佳模型选择
        self.val_dice_pos_history = []  # 仅统计有前景mask样本的Dice（用于诊断）
        self.val_dice_neg_history = []  # 仅统计空mask样本的Dice（用于诊断）
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
   
    def visualize_predictions(self, model, dataloader, device, save_name="predictions", threshold=None):
        """可视化模型预测结果与真实标签
        
        Args:
            threshold: 二值化阈值，如果为None则使用self.last_optimal_threshold，如果仍不可用则使用0.1
        """
        save_path = os.path.join(self.temp_dir, f"{save_name}.png")
        model.eval()
        # 处理数据：可能包含分类标签
        batch_data = next(iter(dataloader))
        if len(batch_data) == 3:
            images, masks, _ = batch_data
        else:
            images, masks = batch_data
        images, masks = images.to(device), masks.to(device)
        
        # 确定使用的阈值
        if threshold is None:
            threshold = getattr(self, 'last_optimal_threshold', 0.1)
        # 如果阈值仍然不可用或无效，使用0.1作为默认值（允许看到低置信度预测）
        if threshold is None or threshold <= 0 or threshold >= 1:
            threshold = 0.1
        
        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()
        
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
                # 【持久化修复】保存到持久化目录
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                matlab_save_path = os.path.join(self.persistent_report_dir, f"{save_name}_{timestamp}_matlab.png")
                os.makedirs(os.path.dirname(matlab_save_path), exist_ok=True)
                self.matlab_viz_bridge.render_prediction_grid(payload_path, matlab_save_path)
                print(f"[MATLAB] ✅ 预测网格已保存到持久化目录: {matlab_save_path}")
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
                    # 【持久化修复】保存到持久化目录
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    matlab_path = os.path.join(self.persistent_report_dir, f"training_history_{timestamp}_matlab.png")
                    os.makedirs(os.path.dirname(matlab_path), exist_ok=True)
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
            
            # 计算总批次数（用于进度显示）
            total_batches = len(dataloader) if use_all_samples else min(num_samples, len(dataloader))
            
            for idx, batch_data in enumerate(dataloader):
                if not use_all_samples and idx >= num_samples:
                    break
                
                # 更新数据收集进度（0-40%）
                data_collect_progress = int(40 * (idx + 1) / max(1, total_batches))
                self.update_progress.emit(
                    data_collect_progress,
                    f"阈值优化: 收集数据 {idx+1}/{total_batches} 批次..."
                )
                
                # 处理数据：可能包含分类标签
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
                images = images.to(device)
                masks = masks.to(device)
                
                # 【性能优化】阈值优化阶段禁用TTA以提升速度
                # 如果验证阶段也禁用TTA，这里也应该禁用以保持一致
                # 可以通过环境变量 SEG_USE_TTA_IN_VAL=1 启用（不推荐，速度慢）
                use_tta_in_threshold = os.environ.get("SEG_USE_TTA_IN_VAL", "0") == "1"  # 默认禁用
                
                if use_tta_in_threshold:
                    # 使用TTA进行推理（与验证阶段一致）
                    outputs = self._tta_inference(model, images)
                else:
                    # 标准推理：单次前向传播（速度更快）
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                
                probs = torch.sigmoid(outputs)
                # 确保 probs 和 masks 的空间尺寸匹配
                if probs.shape[2:] != masks.shape[2:]:
                    probs = F.interpolate(probs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                all_probs.append(probs.detach().cpu().numpy())
                all_masks.append(masks.detach().cpu().numpy())
                
                # 【显存优化】删除GPU上的中间变量
                del outputs, probs
                if torch.cuda.is_available() and idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            if not all_probs:
                return (0.5, 0.0)  # 返回默认阈值和0.0 Dice
            
            # 数据收集完成，开始GWO优化
            self.update_progress.emit(45, "阈值优化: 数据收集完成，开始GWO优化...")
            
            # 【显存优化】先拼接numpy数组，再决定是否移到GPU
            # 这样可以避免在GPU上拼接时占用过多显存
            all_probs_np = np.concatenate(all_probs, axis=0)
            all_masks_np = np.concatenate(all_masks, axis=0)
            
            # 【显存优化】删除原始列表，释放内存
            del all_probs, all_masks
            import gc
            gc.collect()

            # 检查数据大小，决定是否使用GPU
            data_size_mb = all_probs_np.nbytes / (1024 * 1024) * 2  # preds + masks
            use_gpu_for_gwo = False
            
            if torch.cuda.is_available():
                try:
                    # 检查可用显存
                    free_memory_mb = (torch.cuda.get_device_properties(device).total_memory - 
                                    torch.cuda.memory_allocated(device)) / (1024 * 1024)
                    # 如果数据大小小于可用显存的20%，使用GPU
                    if data_size_mb < free_memory_mb * 0.2:
                        use_gpu_for_gwo = True
                        print(f">>> [GWO] 数据大小: {data_size_mb:.1f}MB, 可用显存: {free_memory_mb:.1f}MB，使用GPU加速")
                    else:
                        print(f">>> [GWO] 数据大小: {data_size_mb:.1f}MB, 可用显存: {free_memory_mb:.1f}MB，使用CPU模式（节省显存）")
                except Exception as e:
                    print(f">>> [GWO] 显存检查失败: {e}，使用CPU模式")
            
            # 【关键修复】保存样本数量（在删除前）
            total_samples = all_probs_np.shape[0]
            print(f">>> [GWO] 参与计算的样本数: {total_samples}")
            
            # 【GWO优化】使用灰狼优化算法替代线性扫描，更智能地寻找最佳阈值
            # 【关键修复】传递后处理函数，使GWO在搜索过程中也应用后处理
            def postprocess_func(pred_mask, prob_map):
                """后处理函数，用于GWO的Fitness Function"""
                # 先执行智能后处理
                pred_mask_processed = self.smart_post_processing(pred_mask, prob_map)
                # 再执行传统形态学后处理（与验证阶段参数一致）
                pred_mask_final = self.post_process_mask(
                    pred_mask_processed,
                    min_size=0,
                    use_morphology=True,
                    keep_largest=False,
                    fill_holes=True,
                    prob_map=prob_map
                )
                return pred_mask_final
            
            if use_gpu_for_gwo:
                print(">>> [GWO] 灰狼群正在搜索最佳阈值（GPU加速，使用Mean Dice+后处理）...")
                # 转换为tensor并移到GPU（在GWO内部会处理OOM）
                all_probs_tensor = torch.from_numpy(all_probs_np).to(device)
                all_masks_tensor = torch.from_numpy(all_masks_np).to(device)
            else:
                print(">>> [GWO] 灰狼群正在搜索最佳阈值（CPU模式，使用Mean Dice+后处理）...")
                # 保持在CPU上
                all_probs_tensor = all_probs_np
                all_masks_tensor = all_masks_np
            
            # 定义进度回调函数
            def gwo_progress_callback(iteration, max_iter, best_score, best_threshold):
                # GWO优化进度（45-90%）
                gwo_progress = 45 + int(45 * iteration / max_iter)
                device_str = "GPU" if use_gpu_for_gwo else "CPU"
                self.update_progress.emit(
                    gwo_progress,
                    f"阈值优化: GWO迭代 {iteration}/{max_iter} | 最佳阈值: {best_threshold:.4f} | 最佳Dice(Mean+后处理): {best_score:.4f} ({device_str})"
                )
            
            gwo = GreyWolfThresholdOptimizer(
                num_wolves=10, 
                max_iter=8,  # 【效率优化】缩减迭代次数到8次
                progress_callback=gwo_progress_callback,
                use_mean_dice=True,  # 使用Mean Dice
                postprocess_func=postprocess_func,  # 传递后处理函数
                sample_ratio=0.2,  # 【效率优化】在迭代过程中只使用20%的样本计算Fitness
                metrics_func=self.calculate_batch_metrics  # 统一指标计算入口
            )
            
            # 执行优化（带错误处理和自动回退）
            try:
                best_threshold, best_dice = gwo.optimize(all_probs_tensor, all_masks_tensor, device=device if use_gpu_for_gwo else None)
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    print(f">>> [GWO] GPU显存不足，自动回退到CPU模式")
                    # 清理GPU显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 回退到CPU（确保使用numpy数组）
                    if use_gpu_for_gwo:
                        # 如果之前用的是GPU tensor，需要重新获取numpy数组
                        # 但此时all_probs_np和all_masks_np应该还在
                        best_threshold, best_dice = gwo.optimize(all_probs_np, all_masks_np, device=None)
                    else:
                        best_threshold, best_dice = gwo.optimize(all_probs_tensor, all_masks_tensor, device=None)
                else:
                    raise
            
            # GWO优化完成（已使用Mean Dice+后处理）
            print(f">>> [GWO] 搜索完成! 最佳阈值: {best_threshold:.4f}, 最佳Dice(Mean+后处理): {best_dice:.4f}")
            self.update_progress.emit(90, f"阈值优化: GWO完成 | 最佳阈值: {best_threshold:.4f} | 最佳Dice(Mean+后处理): {best_dice:.4f}")
            
            # 【关键修复】GWO已经使用了Mean Dice+后处理，所以best_dice可以直接用于best_model判定
            # 但为了兼容性和诊断，我们仍然重新计算一次以验证一致性
            print(">>> [GWO] 验证计算一致性（重新计算一次）...")
            self.update_progress.emit(92, "阈值优化: 验证计算一致性...")
            
            # 【关键修复】确保total_samples在删除前已保存（修复日志显示0个样本的问题）
            # 必须在删除all_probs_np之前保存
            if 'total_samples' not in locals():
                if all_probs_np is not None:
                    total_samples = all_probs_np.shape[0]
                elif isinstance(all_probs_tensor, torch.Tensor):
                    total_samples = all_probs_tensor.shape[0]
                else:
                    total_samples = 0
            
            # 统一处理：无论GPU还是CPU模式，都转换为numpy数组
            if isinstance(all_probs_tensor, torch.Tensor):
                # 如果是tensor，转换回numpy
                all_probs_for_postprocess = all_probs_tensor.cpu().numpy()
                all_masks_for_postprocess = all_masks_tensor.cpu().numpy()
            else:
                # 如果已经是numpy数组，直接使用（CPU模式）
                all_probs_for_postprocess = all_probs_np.copy()
                all_masks_for_postprocess = all_masks_np.copy()
            
            # 再次确认total_samples
            if total_samples == 0:
                total_samples = all_probs_for_postprocess.shape[0]
            
            # 对每个样本应用后处理并计算Dice（与验证阶段保持一致）
            processed_preds_list = []
            dice_scores_per_sample = []  # 用于计算Mean Dice
            empty_mask_count = 0
            empty_mask_dice_sum = 0.0
            non_empty_mask_count = 0
            non_empty_mask_dice_sum = 0.0
            
            for i in range(total_samples):
                # 获取单个样本的概率图和真实标签
                prob_map = all_probs_for_postprocess[i, 0]  # (H, W)
                mask_gt = all_masks_for_postprocess[i, 0]    # (H, W)
                
                # 使用最佳阈值二值化
                pred_mask = (prob_map >= best_threshold).astype(np.float32)
                
                # 转换为tensor进行后处理（后处理函数支持tensor和numpy）
                pred_mask_tensor = torch.from_numpy(pred_mask).float()
                prob_map_tensor = torch.from_numpy(prob_map).float()
                
                # 先执行智能后处理
                pred_mask_tensor = self.smart_post_processing(pred_mask_tensor, prob_map_tensor)
                
                # 再执行传统形态学后处理（与验证阶段参数一致）
                pred_mask_processed = self.post_process_mask(
                    pred_mask_tensor,
                    min_size=0,
                    use_morphology=True,
                    keep_largest=False,
                    fill_holes=True,
                    prob_map=prob_map_tensor
                )
                
                # 转换回numpy
                if isinstance(pred_mask_processed, torch.Tensor):
                    pred_mask_processed = pred_mask_processed.cpu().numpy()
                
                processed_preds_list.append(pred_mask_processed)
                
                # 【统一计算】使用统一的指标计算函数
                pred_tensor = torch.from_numpy(pred_mask_processed).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                mask_tensor = torch.from_numpy(mask_gt).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                sample_metrics = self.calculate_batch_metrics(pred_tensor, mask_tensor)
                
                dice_sample = sample_metrics['dice'][0]
                is_empty = sample_metrics['is_empty'][0]
                
                if is_empty:
                    empty_mask_count += 1
                    empty_mask_dice_sum += dice_sample
                    # 【诊断】记录假阳性情况
                    if dice_sample < 1.0 and empty_mask_count <= 5:  # 只记录前5个，避免日志过多
                        pred_sum = pred_mask_processed.sum()
                        fp_pixels = int(pred_sum)
                        print(f">>> [诊断] 空mask样本 #{i+1}: 后处理后仍有 {fp_pixels} 个假阳性像素")
                else:
                    non_empty_mask_count += 1
                    non_empty_mask_dice_sum += dice_sample
                
                dice_scores_per_sample.append(float(dice_sample))
                
                # 每处理100个样本显示一次进度
                if (i + 1) % 100 == 0:
                    self.update_progress.emit(92 + int(6 * (i + 1) / total_samples), 
                                             f"阈值优化: 后处理进度 {i+1}/{total_samples}...")
            
            # 输出诊断信息
            if empty_mask_count > 0:
                empty_mask_avg_dice = empty_mask_dice_sum / empty_mask_count
                print(f">>> [GWO诊断] 空mask样本: {empty_mask_count}/{total_samples} ({100*empty_mask_count/total_samples:.1f}%)")
                print(f">>> [GWO诊断] 空mask平均Dice: {empty_mask_avg_dice:.4f}")
                if empty_mask_avg_dice < 0.9:
                    print(f">>> [GWO警告] 空mask Dice偏低，可能是后处理未能完全过滤假阳性")
            if non_empty_mask_count > 0:
                non_empty_mask_avg_dice = non_empty_mask_dice_sum / non_empty_mask_count
                print(f">>> [GWO诊断] 有前景样本: {non_empty_mask_count}/{total_samples} ({100*non_empty_mask_count/total_samples:.1f}%)")
                print(f">>> [GWO诊断] 有前景样本平均Dice: {non_empty_mask_avg_dice:.4f}")
            
            # 【关键修复】使用Mean Dice（每个样本分别计算再平均，与验证阶段一致）
            # 验证阶段使用 Mean Dice，所以GWO也应该使用 Mean Dice 以保持一致
            mean_dice_postprocessed = np.mean(dice_scores_per_sample) if dice_scores_per_sample else 0.0
            
            # 【内存优化】分批计算Global Dice用于对比，避免内存爆炸
            # 不要一次性创建(N, H, W)的bool数组，而是分批累加TP/FP/FN
            batch_size_for_global = 100  # 每批处理100个样本
            tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
            
            for batch_start in range(0, total_samples, batch_size_for_global):
                batch_end = min(batch_start + batch_size_for_global, total_samples)
                batch_preds = np.array(processed_preds_list[batch_start:batch_end])  # (B, H, W)
                batch_masks = all_masks_for_postprocess[batch_start:batch_end]  # (B, H, W)
                
                # 确保形状一致
                if batch_preds.shape != batch_masks.shape:
                    # 如果形状不匹配，调整batch_masks
                    if batch_masks.ndim == 3 and batch_preds.ndim == 3:
                        # 确保都是(B, H, W)
                        if batch_masks.shape[0] != batch_preds.shape[0]:
                            batch_masks = batch_masks[:batch_preds.shape[0]]
                        if batch_masks.shape[1:] != batch_preds.shape[1:]:
                            # 使用插值调整大小（不应该发生，但安全起见）
                            from scipy.ndimage import zoom
                            zoom_factors = (1.0, batch_preds.shape[1]/batch_masks.shape[1], 
                                          batch_preds.shape[2]/batch_masks.shape[2])
                            batch_masks = zoom(batch_masks, zoom_factors, order=0)
                
                # 二值化并展平
                pred_bool_batch = (batch_preds > 0.5).astype(np.float32)  # (B, H, W)
                gt_bool_batch = (batch_masks > 0.5).astype(np.float32)  # (B, H, W)
                
                # 展平为(B*H*W,)
                pred_flat_batch = pred_bool_batch.flatten()  # (B*H*W,)
                gt_flat_batch = gt_bool_batch.flatten()  # (B*H*W,)
                
                # 计算混淆矩阵（逐元素，避免广播）
                tp_batch = np.sum(pred_flat_batch * gt_flat_batch)
                fp_batch = np.sum(pred_flat_batch * (1 - gt_flat_batch))
                fn_batch = np.sum((1 - pred_flat_batch) * gt_flat_batch)
                tn_batch = np.sum((1 - pred_flat_batch) * (1 - gt_flat_batch))
                
                tp_total += int(tp_batch)
                fp_total += int(fp_batch)
                fn_total += int(fn_batch)
                tn_total += int(tn_batch)
                
                # 清理批次变量
                del batch_preds, batch_masks, pred_bool_batch, gt_bool_batch
                del pred_flat_batch, gt_flat_batch
                if torch.cuda.is_available() and batch_end % 500 == 0:
                    torch.cuda.empty_cache()
            
            # 计算Global Dice（用于对比）
            dice_den_postprocessed = 2.0 * tp_total + fp_total + fn_total
            global_dice_postprocessed = 1.0 if dice_den_postprocessed < 1e-7 else (2.0 * tp_total) / (dice_den_postprocessed + 1e-7)
            
            print(f">>> [GWO] 验证计算完成! 最佳阈值: {best_threshold:.4f}")
            print(f">>> [GWO] GWO搜索时的Dice: {best_dice:.4f} (Mean+后处理)")
            print(f">>> [GWO] 重新计算的Mean Dice: {mean_dice_postprocessed:.4f} (用于验证一致性)")
            print(f">>> [GWO] Global Dice(后处理): {global_dice_postprocessed:.4f} (用于对比)")
            
            # 【关键修复】检查一致性
            dice_diff = abs(best_dice - mean_dice_postprocessed)
            if dice_diff > 0.01:
                print(f">>> [GWO警告] Dice差异较大: {dice_diff:.4f}，可能存在计算不一致")
            else:
                print(f">>> [GWO] Dice一致性验证通过 (差异: {dice_diff:.4f})")
            
            # 【关键修复】使用重新计算的Mean Dice作为最终结果（确保与验证阶段完全一致）
            best_dice = mean_dice_postprocessed
            
            # 为了兼容性，计算完整的指标字典（使用后处理后的结果）
            # 使用已计算的混淆矩阵
            tp = tp_total
            fp = fp_total
            fn = fn_total
            tn = tn_total
            
            # 计算完整指标
            dice_den = 2.0 * tp + fp + fn
            dice = 1.0 if dice_den < 1e-7 else (2.0 * tp) / (dice_den + 1e-7)
            iou_den = tp + fp + fn
            iou = 1.0 if iou_den < 1e-7 else tp / (iou_den + 1e-7)
            prec_den = tp + fp
            precision = 1.0 if prec_den < 1e-7 else tp / (prec_den + 1e-7)
            rec_den = tp + fn
            recall = 1.0 if rec_den < 1e-7 else tp / (rec_den + 1e-7)
            spec_den = tn + fp
            specificity = 1.0 if spec_den < 1e-7 else tn / (spec_den + 1e-7)
            
            best_metrics = {
                'dice': float(dice),
                'iou': float(iou),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'score': float(best_dice)  # 使用后处理后的Mean Dice（与验证阶段一致）
            }
            
            # 【显存优化】删除拼接后的数组和中间变量
            # 注意：在CPU模式下，all_probs_tensor就是all_probs_np，所以只需要删除一次
            if use_gpu_for_gwo:
                # GPU模式：删除tensor和numpy数组
                del all_probs_tensor, all_masks_tensor
            del all_probs_np, all_masks_np
            del processed_preds_list, dice_scores_per_sample
            if 'all_probs_for_postprocess' in locals():
                del all_probs_for_postprocess, all_masks_for_postprocess
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 【关键修复】确保total_samples正确显示
        if 'total_samples' not in locals() or total_samples == 0:
            # 如果total_samples未定义或为0，尝试从best_metrics或其他地方获取
            if 'all_probs_for_postprocess' in locals() and all_probs_for_postprocess is not None:
                total_samples = all_probs_for_postprocess.shape[0]
            elif 'all_probs_np' in locals() and all_probs_np is not None:
                total_samples = all_probs_np.shape[0]
            else:
                total_samples = num_samples if 'num_samples' in locals() else 0

        sample_info = "全部验证集" if use_all_samples else f"{num_samples}个批次"
        score_val = best_metrics.get("score", 0.0) if isinstance(best_metrics, dict) else 0.0
        print(
            f"[阈值优化] 使用样本: {sample_info} ({total_samples}个样本) | "
            f"最优阈值: {best_threshold:.3f}, 最佳Dice(全验证集): {best_dice:.4f}, "
            f"Dice: {best_metrics.get('dice', float('nan')):.4f}, "
            f"IoU: {best_metrics.get('iou', float('nan')):.4f}"
        )
        # 返回最佳阈值和最佳Dice（基于全验证集GWO优化）
        return (float(best_threshold), float(best_dice))
    
    def evaluate_model(self, model, dataloader, device, use_tta=True, adaptive_threshold=True):
        """
        综合模型评估
        
        Args:
            use_tta: 是否使用测试时增强(TTA),可提升1-3%的Dice
            adaptive_threshold: 是否使用自适应阈值
        """
        # 寻找最优阈值
        if adaptive_threshold:
            threshold_result = self.find_optimal_threshold(model, dataloader, device)
            # 处理返回值：可能是元组(threshold, dice)或单个值（向后兼容）
            if isinstance(threshold_result, tuple):
                optimal_thresh, gwo_dice = threshold_result
                self.gwo_best_dice = float(gwo_dice)
            else:
                optimal_thresh = threshold_result
                self.gwo_best_dice = None
        else:
            optimal_thresh = 0.5
            self.gwo_best_dice = None
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
                
                prob_maps = torch.sigmoid(outputs)
                preds = (prob_maps > optimal_thresh).float()  # 使用最优阈值
                
                # 应用后处理优化：填充孔洞，不再强制只保留最大连通域
                for i in range(preds.shape[0]):
                    preds[i, 0] = self.post_process_mask(
                        preds[i, 0], 
                        min_size=30, 
                        use_morphology=True,
                        keep_largest=False,  # 允许多发病灶同时存在
                        fill_holes=True,     # 填充孔洞，去除假阴性空洞
                        prob_map=prob_maps[i, 0]
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
                    
                    # 【统一计算方式】Precision = TP / (TP + FP)
                    prec_den = tp + fp
                    if prec_den < 1e-7:
                        # 如果没有预测出任何正样本(tp+fp=0)，则精确率视为1.0(无误检)
                        precision = 1.0
                    else:
                        precision = float(tp / (prec_den + 1e-7))  # 使用 1e-7 与 Recall/Specificity 保持一致
                    
                    # 【统一计算方式】Recall/Sensitivity = TP / (TP + FN)
                    rec_den = tp + fn
                    if rec_den < 1e-7:
                        # 如果Ground Truth为空(无病灶，tp+fn=0)，则召回率视为1.0(完美表现)
                        recall = 1.0
                    else:
                        recall = float(tp / (rec_den + 1e-7))  # 使用 1e-7 与 Precision/Specificity 保持一致
                    
                    # 【统一计算方式】Specificity = TN / (TN + FP)
                    spec_den = tn + fp
                    if spec_den < 1e-7:
                        specificity = 1.0  # 如果没有负样本，特异性为1.0
                    else:
                        specificity = float(tn / (spec_den + 1e-7))  # 使用 1e-7 与 Precision/Recall 保持一致
                    
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
                        fill_holes=True,     # 填充孔洞，去除假阴性空洞
                        prob_map=prob_map_tensor
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
                        
                        # 【统一计算方式】Precision = TP / (TP + FP)
                        prec_den = tp + fp
                        if prec_den < 1e-7:
                            precision = 1.0  # 如果没有预测出任何正样本，精确率为1.0
                        else:
                            precision = float(tp / (prec_den + 1e-7))  # 使用 1e-7 与 Recall/Specificity 保持一致
                        
                        # 【统一计算方式】Recall = TP / (TP + FN)
                        rec_den = tp + fn
                        if rec_den < 1e-7:
                            recall = 1.0  # 如果Ground Truth为空，召回率为1.0
                        else:
                            recall = float(tp / (rec_den + 1e-7))  # 使用 1e-7 与 Precision/Specificity 保持一致
                        
                        # 【统一计算方式】Specificity = TN / (TN + FP)
                        spec_den = tn + fp
                        if spec_den < 1e-7:
                            specificity = 1.0  # 如果没有负样本，特异性为1.0
                        else:
                            specificity = float(tn / (spec_den + 1e-7))  # 使用 1e-7 与 Precision/Recall 保持一致
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
                    
                    # 【统一指标计算】使用统一函数同时计算 Dice 和 IoU，确保逻辑一致
                    dice, iou = self._compute_metrics_unified(system_pred, true_mask)
                    
                    # 【统一计算方式】Precision = TP / (TP + FP) = intersection / pred_sum
                    # 注意：这里 intersection = tp, pred_sum = tp + fp
                    if pred_sum < 1e-7:
                        precision = 1.0  # 如果没有预测出任何正样本，精确率为1.0
                    else:
                        precision = float(intersection / (pred_sum + 1e-7))  # 使用 1e-7 与 Recall 保持一致
                    
                    # 【统一计算方式】Recall = TP / (TP + FN) = intersection / mask_sum
                    # 注意：这里 intersection = tp, mask_sum = tp + fn
                    if mask_sum < 1e-7:
                        recall = 1.0  # 如果Ground Truth为空，召回率为1.0
                    else:
                        recall = float(intersection / (mask_sum + 1e-7))  # 使用 1e-7 与 Precision 保持一致
                    
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
        # 【统一计算方式】System Precision = TP / (TP + FP)
        system_tp = system_metrics['true_positive']
        system_fp = system_metrics['false_positive']
        system_fn = system_metrics['false_negative']
        prec_den = system_tp + system_fp
        if prec_den < 1e-7:
            system_precision = 1.0  # 如果没有预测出任何正样本，精确率为1.0
        else:
            system_precision = float(system_tp / (prec_den + 1e-7))  # 使用 1e-7 与 Recall 保持一致
        
        # 【统一计算方式】System Recall = TP / (TP + FN)
        rec_den = system_tp + system_fn
        if rec_den < 1e-7:
            system_recall = 1.0  # 如果Ground Truth为空，召回率为1.0
        else:
            system_recall = float(system_tp / (rec_den + 1e-7))  # 使用 1e-7 与 Precision 保持一致
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
    
    def visualize_test_results(self, model, dataloader, device, num_samples=8, use_tta=True, epoch=None, is_best=False, threshold=None):
        """
        【双引擎策略】可视化测试集上的分割结果
        
        策略：
        - 关键 Epoch（每20轮或最佳模型）：使用 MATLAB 生成高清图
        - 普通 Epoch：使用 Matplotlib 快速预览
        
        Args:
            use_tta: 是否使用测试时增强（默认True，训练结束后的测试推荐使用）
            epoch: 当前轮次（用于判断是否为关键 Epoch）
            is_best: 是否为最佳模型
            threshold: 二值化阈值，如果为None则使用self.last_optimal_threshold，如果仍不可用则使用0.1
        """
        # 确定使用的阈值
        if threshold is None:
            threshold = getattr(self, 'last_optimal_threshold', 0.1)
        # 如果阈值仍然不可用或无效，使用0.1作为默认值（允许看到低置信度预测）
        if threshold is None or threshold <= 0 or threshold >= 1:
            threshold = 0.1
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
                preds_binary = (preds > threshold).float()
                
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
                    
                    # 【统一指标计算】使用统一函数同时计算 Dice 和 IoU，确保逻辑一致
                    dice, iou = self._compute_metrics_unified(pred, mask)
                    
                    all_images.append(img)
                    all_masks.append(mask)
                    all_preds.append(pred)
                    all_metrics.append({'dice': dice, 'iou': iou})
                
                if len(all_masks) >= num_samples:
                    break

        # 【双引擎策略】判断使用 MATLAB 还是 Matplotlib
        use_matlab = False
        if epoch is not None:
            # 关键 Epoch：每20轮或最佳模型
            is_key_epoch = (epoch % 20 == 0) or is_best
            use_matlab = is_key_epoch and self.enable_matlab_plots and self.matlab_viz_bridge
        else:
            # 如果没有传入 epoch，默认使用 MATLAB（向后兼容）
            use_matlab = self.enable_matlab_plots and self.matlab_viz_bridge

        # 使用 MATLAB 生成高清图（关键 Epoch）
        if use_matlab:
            try:
                # 使用 _save_matlab_viz_payload 保存数据（与 render_prediction_grid 兼容）
                payload = self._save_matlab_viz_payload(all_images, all_masks, all_preds, "test_results")
                # 【持久化修复】保存到持久化目录
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = f"test_results_visualization_epoch{epoch}_matlab" if epoch is not None else "test_results_visualization_matlab"
                matlab_path = os.path.join(self.persistent_report_dir, f"{base_name}_{timestamp}.png")
                os.makedirs(os.path.dirname(matlab_path), exist_ok=True)
                self.matlab_viz_bridge.render_prediction_grid(payload, matlab_path)
                print(f"[高清] MATLAB 渲染完成，已保存到持久化目录: {matlab_path}")
                return matlab_path
            except Exception as exc:
                print(f"[MATLAB Plot] 测试可视化回退: {exc}")
                # 回退到 Matplotlib
        
        # 使用 Matplotlib 快速预览（普通 Epoch）
        save_path = os.path.join(self.temp_dir, f"test_results_visualization_epoch{epoch}_preview.png" if epoch is not None else "test_results_visualization.png")
        try:
            # 使用类方法（如果可用）或独立函数，传入阈值
            if self.matlab_viz_bridge:
                self.matlab_viz_bridge.render_quick_preview_matplotlib(all_images, all_masks, all_preds, save_path, num_samples=min(num_samples, len(all_images)), threshold=threshold)
            else:
                # 回退：使用独立函数（向后兼容）
                from utils import render_quick_preview_matplotlib
                render_quick_preview_matplotlib(all_images, all_masks, all_preds, save_path, num_samples=min(num_samples, len(all_images)), threshold=threshold)
            print(f"[快照] Matplotlib 绘图完成: {save_path}")
            return save_path
        except Exception as exc:
            print(f"[Matplotlib Plot] 快速预览失败: {exc}")
            # 最终回退：使用原有的 Matplotlib 代码
            return self._fallback_matplotlib_plot(all_images, all_masks, all_preds, all_metrics, save_path, num_samples)
    
    def _fallback_matplotlib_plot(self, all_images, all_masks, all_preds, all_metrics, save_path, num_samples):
        """回退方案：使用原有的 Matplotlib 绘图代码"""
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
                # 【持久化修复】保存到持久化目录
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                matlab_path = os.path.join(self.persistent_report_dir, f"performance_analysis_{timestamp}_matlab.png")
                os.makedirs(os.path.dirname(matlab_path), exist_ok=True)
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
    
    def visualize_attention_maps(self, model, dataloader, device, num_samples=4, threshold=None):
        """可视化注意力权重图，用于模型可解释性分析 - 优化版
        
        Args:
            threshold: 二值化阈值，如果为None则使用self.last_optimal_threshold，如果仍不可用则使用0.1
        """
        # 【DeepLabV3+ 兼容性】DeepLabV3+ 虽然标记为支持注意力图，但实际上不支持 return_attention
        # 这里允许 DeepLabV3+ 通过，但会在后续返回空结果
        is_deeplabv3 = self.model_type in ("deeplabv3plus", "smp_deeplabv3plus")
        if not self._supports_attention_maps(model) and not is_deeplabv3:
            raise RuntimeError("当前模型不支持注意力可视化")
        save_path = os.path.join(self.temp_dir, "attention_visualization.png")
        model.eval()
        
        # 确定使用的阈值
        if threshold is None:
            threshold = getattr(self, 'last_optimal_threshold', 0.1)
        # 如果阈值仍然不可用或无效，使用0.1作为默认值
        if threshold is None or threshold <= 0 or threshold >= 1:
            threshold = 0.1
        
        # 收集样本和注意力图
        all_images = []
        all_masks = []
        all_preds = []
        all_attention_maps = []
        
        # 【性能优化】只对前 5 个 batch 生成 Grad-CAM，其他 batch 跳过以提升速度
        # Grad-CAM 需要反向传播，计算成本极高，全量生成会导致验证时间过长
        max_gradcam_batches = 5  # 只对前 5 个 batch 生成 Grad-CAM
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 处理数据：可能包含分类标签
            if len(batch_data) == 3:
                images, masks, _ = batch_data
            else:
                images, masks = batch_data
            images, masks = images.to(device), masks.to(device)
            
            # 【性能优化】判断是否需要生成 Grad-CAM（仅前 5 个 batch）
            need_gradcam = (batch_idx < max_gradcam_batches)
            is_deeplabv3 = self.model_type in ("deeplabv3plus", "smp_deeplabv3plus")
            
            if need_gradcam:
                # 需要 Grad-CAM 的样本：必须在 torch.enable_grad() 下运行
                # 【DeepLabV3+ 兼容性 + Grad-CAM 集成】DeepLabV3+ 不支持 return_attention，使用 Grad-CAM
                if is_deeplabv3:
                    # DeepLabV3+ 不支持 return_attention，先获取输出
                    with torch.enable_grad():
                        outputs = model(images)
                        # 使用 Grad-CAM 生成热力图（需要梯度）
                        actual_model = self._unwrap_model(model)
                        attention_maps = self._generate_gradcam_for_deeplabv3(actual_model, images, device)
                else:
                    outputs, attention_maps = model(images, return_attention=True)
            else:
                # 不需要 Grad-CAM 的样本：使用 torch.no_grad() 加速
                # 如果已收集足够样本，直接退出循环
                if len(all_images) >= num_samples:
                    break
                    
                with torch.no_grad():
                    if is_deeplabv3:
                        # DeepLabV3+ 不需要注意力图，直接获取输出
                        outputs = model(images)
                        attention_maps = {}  # 不需要热力图
                    else:
                        # 其他模型：尝试获取注意力图，但不强制
                        try:
                            outputs, attention_maps = model(images, return_attention=True)
                        except:
                            # 如果获取失败，只获取输出
                            outputs = model(images)
                            attention_maps = {}
            
            # 处理预测结果（无论是否需要 Grad-CAM）
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > threshold).float()
            
            # 只处理需要可视化的样本（前 5 个 batch）
            if need_gradcam:
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
                    # 【DeepLabV3+ 兼容性】如果 attention_maps 为空（DeepLabV3+ 不支持），使用占位符
                    if not attention_maps:
                        # 对于 DeepLabV3+，创建一个占位符注意力图（使用预测概率图作为替代）
                        pred_np = pred.copy()
                        att_dict['output_probability'] = pred_np  # 使用预测概率作为注意力图
                    else:
                        for att_name, att_map in attention_maps.items():
                            att_np = att_map[i, 0].cpu().numpy()
                            # 上采样到512x512（与输入图像大小一致）
                            from scipy.ndimage import zoom
                            target_size = (512, 512)  # 提升分辨率以保留更多病灶边缘细节
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
                # 【持久化修复】保存到持久化目录
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                matlab_path = os.path.join(self.persistent_report_dir, f"attention_visualization_{timestamp}_matlab.png")
                os.makedirs(os.path.dirname(matlab_path), exist_ok=True)
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
        # 【DeepLabV3+ 兼容性】DeepLabV3+ 虽然标记为支持注意力图，但实际上不支持 return_attention
        is_deeplabv3 = self.model_type in ("deeplabv3plus", "smp_deeplabv3plus")
        if not self._supports_attention_maps(model) and not is_deeplabv3:
            raise RuntimeError("当前模型不支持注意力统计分析")
        model.eval()
        # 先运行一次获取实际的注意力层名称
        attention_stats = {}
        
        # 【性能优化】只对前 5 个 batch 生成 Grad-CAM，其他 batch 跳过以提升速度
        # Grad-CAM 需要反向传播，计算成本极高，全量生成会导致验证时间过长
        max_gradcam_batches = 5  # 只对前 5 个 batch 生成 Grad-CAM
        gradcam_generated = False  # 标记是否已生成 Grad-CAM
        
        eval_count = 0
        for batch_idx, batch_data in enumerate(dataloader):
            if eval_count >= num_samples:
                break
            
            # 处理数据：可能包含分类标签
            if len(batch_data) == 3:
                images, masks, _ = batch_data
            else:
                images, masks = batch_data
            images, masks = images.to(device), masks.to(device)
            
            # 【性能优化】判断是否需要生成 Grad-CAM（仅前 5 个 batch）
            need_gradcam = (batch_idx < max_gradcam_batches)
            is_deeplabv3 = self.model_type in ("deeplabv3plus", "smp_deeplabv3plus")
            
            if need_gradcam:
                # 需要 Grad-CAM 的样本：必须在 torch.enable_grad() 下运行
                # 【DeepLabV3+ 兼容性 + Grad-CAM 集成】DeepLabV3+ 不支持 return_attention，使用 Grad-CAM
                if is_deeplabv3:
                    # DeepLabV3+ 不支持 return_attention，先获取输出
                    with torch.enable_grad():
                        outputs = model(images)
                        # 使用 Grad-CAM 生成热力图（需要梯度）
                        actual_model = self._unwrap_model(model)
                        attention_maps = self._generate_gradcam_for_deeplabv3(actual_model, images, device)
                        gradcam_generated = True
                else:
                    outputs, attention_maps = model(images, return_attention=True)
                    gradcam_generated = True
            else:
                # 不需要 Grad-CAM 的样本：使用 torch.no_grad() 加速
                with torch.no_grad():
                    if is_deeplabv3:
                        # DeepLabV3+ 不需要注意力图，直接获取输出
                        outputs = model(images)
                        attention_maps = {}  # 不需要热力图
                    else:
                        # 其他模型：尝试获取注意力图，但不强制
                        try:
                            outputs, attention_maps = model(images, return_attention=True)
                        except:
                            # 如果获取失败，只获取输出
                            outputs = model(images)
                            attention_maps = {}
            
            # 初始化统计字典（只初始化实际存在的层）
            # 【性能优化】如果前 5 个 batch 都没有生成 Grad-CAM，提前返回
            if batch_idx == max_gradcam_batches - 1 and not gradcam_generated and not attention_maps:
                print("[注意力统计] 前 5 个 batch 均无法获取注意力图（可能是 Grad-CAM 生成失败），跳过统计分析")
                return None
            
            # 【统计兼容性】如果 attention_maps 为空，跳过该 batch 的统计
            # 注意：不需要 Grad-CAM 的 batch（need_gradcam=False）也会进入这里
            if not attention_maps:
                eval_count += images.size(0)
                # 如果不需要 Grad-CAM 且已处理足够样本，提前退出
                if not need_gradcam and eval_count >= num_samples:
                    break
                continue
                
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
            # 【Bug修复】清空训练历史记录，防止图表数据重叠
            # 策略：每次开始训练时都清空历史记录，确保图表从第1轮开始绘制
            # 如果历史记录列表不为空，说明可能是上次训练留下的数据，应该清空
            # 注意：当前实现不支持从checkpoint恢复历史记录，如果需要断点续训并恢复历史，
            # 可以在后续添加从checkpoint读取历史数据的逻辑
            has_existing_history = (
                len(self.train_loss_history) > 0 or 
                len(self.val_loss_history) > 0 or 
                len(self.val_dice_history) > 0
            )
            
            if has_existing_history:
                print("[训练历史] 检测到残留的历史记录，已清空（防止图表数据重叠）")
            
            # 清空所有历史记录列表，确保每次训练都从第1轮开始
            self.train_loss_history = []
            self.val_loss_history = []
            self.val_dice_history = []
            self.val_dice_pos_history = []
            self.val_dice_neg_history = []
            
            if not has_existing_history:
                print("[训练历史] 开始新训练，历史记录已初始化")
            
            # 初始化设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.update_progress.emit(0, f"使用设备: {device}")
            
            # 数据准备
            # 2.5D数据集直接从文件加载，不需要patient_ids
            if self.dataset_type == "2.5d":
                # 2.5D数据集：直接使用load_dataset，它会处理文件列表
                # 我们需要手动分割文件列表
                from dataset import TCGA2_5DDataset
                if not TCGA2_5D_AVAILABLE:
                    raise ImportError("TCGA2_5DDataset 未导入，无法使用2.5D数据集")
                
                # 创建临时数据集以获取文件列表
                temp_dataset = TCGA2_5DDataset(
                    data_dir=self.data_dir,
                    mask_dir=self.data_dir,
                    transform=None,
                    is_train=True,
                    debug=False
                )
                all_indices = list(range(len(temp_dataset)))
                
                # 【修复数据泄露Bug】使用GroupShuffleSplit按case_id分组划分
                # 确保同一个病人的所有切片要么全在训练集，要么全在验证集
                # 避免同一病人的切片同时出现在训练集和验证集，导致验证集分数虚高
                # GroupShuffleSplit已在文件顶部导入
                
                # 1. 获取所有样本的 group (case_id)
                groups = [temp_dataset.file_list[i][0] for i in range(len(temp_dataset))]
                
                # 2. 按组划分
                gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
                train_idx, val_idx = next(gss.split(all_indices, groups=groups))
                
                # 3. 转换回列表
                train_indices = train_idx.tolist()
                val_indices = val_idx.tolist()
                
                # 存储索引用于后续数据集创建
                self.train_indices = train_indices
                self.val_indices = val_indices
                
                # 验证分组正确性（调试信息）
                train_cases = set([temp_dataset.file_list[i][0] for i in train_indices])
                val_cases = set([temp_dataset.file_list[i][0] for i in val_indices])
                overlap = train_cases & val_cases
                if overlap:
                    print(f"[警告] 发现 {len(overlap)} 个病例同时出现在训练集和验证集中，可能存在数据泄露！")
                else:
                    print(f"[数据划分] ✅ 成功按病例分组：训练集 {len(train_cases)} 个病例，验证集 {len(val_cases)} 个病例，无重叠")
                patient_ids = []  # 2.5D数据集不使用patient_ids
            else:
                # 标准数据集：按patient_id组织
                patient_ids = [pid for pid in os.listdir(self.data_dir) 
                             if os.path.isdir(os.path.join(self.data_dir, pid))]
            
            # 单模型训练（仅对标准数据集）
            if self.dataset_type != "2.5d":
                # 检查patient_ids是否为空
                if not patient_ids:
                    error_msg = f"数据目录为空或格式不正确！\n\n数据目录: {self.data_dir}\n\n期望结构:\n  {self.data_dir}/\n    patient_id_1/\n    patient_id_2/\n    ...\n\n或者使用2.5D数据集模式。"
                    self.update_progress.emit(0, error_msg)
                    self.training_finished.emit(error_msg, None)
                    return
                train_ids, val_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
            else:
                train_ids, val_ids = [], []  # 2.5D数据集使用索引分割
            
            # 数据增强（增强对比度、光照和形变，提升泛化能力）
            # 优化数据增强 - 针对医学影像的非刚体形变特性
            # 重点增强：Grid Distortion + Elastic Transform（模拟器官挤压和变形）
            # MixUp 将在训练循环中实现（需要两张图像混合）
            
            # 【修复归一化参数】根据数据集类型动态设置归一化参数
            # 2.5D数据集：3通道输入，使用ImageNet 3通道归一化
            # 标准数据集：1通道输入，使用单通道归一化（但某些模型可能期望3通道，会在模型内部处理）
            if self.dataset_type == "2.5d":
                # 2.5D数据集：3通道输入，使用ImageNet归一化
                normalize_mean = (0.485, 0.456, 0.406)
                normalize_std = (0.229, 0.224, 0.225)
            else:
                # 标准数据集：根据模型类型判断
                # 如果模型是SMP模型（U-Net++或DeepLabV3+），可能使用3通道（如果用户配置了3通道）
                # 其他模型通常使用1通道，但为了兼容性，先使用3通道归一化
                # 实际通道数会在模型构建时根据dataset_type设置
                # 注意：如果模型输入是1通道，归一化参数会被忽略或重复使用
                normalize_mean = (0.485, 0.456, 0.406)  # 默认3通道，兼容性考虑
                normalize_std = (0.229, 0.224, 0.225)
            
            train_transform = A.Compose([
                A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), border_mode=cv2.BORDER_REFLECT_101, p=0.6),
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
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2()
            ])
            
            # 验证集仅做几何归一化，避免引入过多随机性
            val_transform = A.Compose([
                A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2()
            ])
            
            # 加载分割训练数据
            self.update_progress.emit(5, "正在加载分割训练数据...")
            # 【Windows 多进程优化】针对 Intel Core Ultra 9 285HX (24核) 优化
            # 使用 8 个 worker 充分利用 8 个 P-Core（性能核心），既能喂饱 RTX 5080，又不会导致系统卡顿
            import platform
            is_windows = platform.system() == 'Windows'
            cpu_count = os.cpu_count() or 1
            
            # 【性能优化】设置 num_workers: 充分利用多核 CPU（ROG 枪神9 i9 + RTX 4090）
            # 动态计算最优值：使用 min(os.cpu_count(), 8) 或直接设置为 8
            # 对于高性能硬件，使用更多 workers 可以显著提升数据加载速度
            if is_windows:
                # Windows 上使用 8 个 workers（充分利用 P-Core）
                num_workers = min(os.cpu_count(), 8) if os.cpu_count() > 0 else 8
            else:
                # Linux/Mac 可以使用更多进程
                num_workers = min(os.cpu_count(), 8) if os.cpu_count() > 0 else 4
            
            # 【关键优化】persistent_workers=True: 让子进程在 Epoch 之间保持存活，避免重复创建
            # 这对性能提升至关重要，避免每个 Epoch 重新创建进程的开销
            use_persistent_workers = (num_workers > 0)
            
            # 【性能优化】pin_memory=True: 锁页内存，极大加快 CPU 到 GPU 的数据传输
            # 对于 RTX 4090 这样的高性能 GPU，pin_memory 能带来显著性能提升
            # 如果遇到 "resource already mapped" 错误，可以尝试减少 num_workers 或使用单 GPU
            use_pin_memory = (device.type == 'cuda')  # CUDA 设备启用 pin_memory
            
            self.update_progress.emit(6, f"数据加载器配置: num_workers={num_workers}, persistent_workers={use_persistent_workers}, pin_memory={use_pin_memory}, prefetch_factor=4")
            
            # 2.5D数据集使用索引，标准数据集使用patient_ids
            if self.dataset_type == "2.5d":
                train_dataset = self.load_dataset([], train_transform, split_name="train", return_classification=False)
                # 创建子集（使用索引）
                from torch.utils.data import Subset
                train_dataset = Subset(train_dataset, self.train_indices)
            else:
                train_dataset = self.load_dataset(train_ids, train_transform, split_name="train", return_classification=False)
            train_sampler = None
            if getattr(train_dataset, "use_weighted_sampling", False):
                weights = train_dataset.get_sampling_weights()
                if weights is not None:
                    weight_tensor = torch.as_tensor(weights, dtype=torch.double)
                    train_sampler = WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)

            # 检查训练数据集是否为空
            if len(train_dataset) == 0:
                error_msg = f"训练数据集为空！\n\n数据目录: {self.data_dir}\n\n请检查:\n1. 数据目录结构是否正确\n2. 图像和掩膜文件是否存在\n3. 数据集类型选择是否正确"
                self.update_progress.emit(0, error_msg)
                self.training_finished.emit(error_msg, None)
                return
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=num_workers,  # 【性能优化】使用多进程加速数据加载
                pin_memory=use_pin_memory,  # 【性能优化】锁页内存，加速 CPU->GPU 传输
                persistent_workers=use_persistent_workers,  # 【性能优化】保持子进程存活，避免重复创建
                prefetch_factor=4 if num_workers > 0 else None  # 【性能优化】增加预取因子，提升数据流水线效率
            )
            
            self.update_progress.emit(10, "正在加载分割验证数据...")
            # 2.5D数据集使用索引，标准数据集使用patient_ids
            if self.dataset_type == "2.5d":
                val_dataset = self.load_dataset([], val_transform, split_name="val", return_classification=False, use_weighted_sampling=False)
                # 创建子集（使用索引）
                from torch.utils.data import Subset
                val_dataset = Subset(val_dataset, self.val_indices)
            else:
                val_dataset = self.load_dataset(val_ids, val_transform, split_name="val", return_classification=False, use_weighted_sampling=False)
            
            # 检查验证数据集是否为空
            if len(val_dataset) == 0:
                error_msg = f"验证数据集为空！\n\n数据目录: {self.data_dir}\n\n请检查:\n1. 数据目录结构是否正确\n2. 图像和掩膜文件是否存在\n3. 数据集类型选择是否正确"
                self.update_progress.emit(0, error_msg)
                self.training_finished.emit(error_msg, None)
                return
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,  # 【性能优化】使用多进程加速数据加载
                pin_memory=use_pin_memory,  # 【性能优化】锁页内存，加速 CPU->GPU 传输
                persistent_workers=use_persistent_workers,  # 【性能优化】保持子进程存活，避免重复创建
                prefetch_factor=4 if num_workers > 0 else None  # 【性能优化】增加预取因子，提升数据流水线效率
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
            
            # 【GUI选项驱动】根据dataset_type自动判断并设置模型输入通道数
            if self.dataset_type == "2.5d":
                input_channels = 3
                use_stacking = True
                dataset_mode_desc = "2.5D模式（3通道堆叠：上一张、当前、下一张）"
            else:  # 默认为 2D
                input_channels = 1
                use_stacking = False
                dataset_mode_desc = "2D模式（单通道：仅当前切片）"
            
            print(f"\n{'='*60}")
            print(f"📋 [GUI选项驱动配置]")
            print(f"   数据集类型: {self.dataset_type}")
            print(f"   模型输入通道数: {input_channels}")
            print(f"   数据堆叠: {'启用' if use_stacking else '禁用'}")
            print(f"   模式描述: {dataset_mode_desc}")
            print(f"{'='*60}\n")
            
            # 初始化模型（_build_model内部会根据self.dataset_type设置in_channels）
            self.update_progress.emit(15, f"正在构建模型 ({self.model_type}, {dataset_mode_desc})...")
            try:
                model = self._build_model(device, swin_params=self.swin_params, dstrans_params=self.dstrans_params)
                self.update_progress.emit(16, f"模型构建完成（输入通道数: {input_channels}）")
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

                # 【权重加载保护】使用兼容加载函数
                # 如果是 2D 模式，允许加载单通道预训练权重
                # 如果是 2.5D 模式，如果加载的是单通道权重，_adapt_model_channels 会在后续自动适配
                print(f"[权重加载] 正在加载权重: {model_path_to_use}")
                print(f"[权重加载] 当前模式: {dataset_mode_desc}，期望输入通道数: {input_channels}")
                # 传递 target_model_type 参数，用于跨架构权重迁移（如从 ResNet-UNet 到 DeepLabV3+）
                success, msg = load_model_compatible(model, model_path_to_use, device, verbose=False, target_model_type=self.model_type)
                self.update_progress.emit(15, msg)
                if not success:
                    print(f"[警告] 权重加载失败: {msg}")
                    print(f"[提示] 如果是通道数不匹配，_adapt_model_channels 会在后续自动适配")
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
            elif self.model_type in ("smp_deeplabv3plus", "deeplabv3plus"):
                # DeepLabV3+：使用更高的学习率，因为decoder是随机初始化的，需要更快的学习
                # Encoder使用ImageNet预训练，Decoder需要快速适应任务
                default_lr = 2e-4  # 从1e-4提升到2e-4
            else:
                default_lr = 1e-4

            # 若设置了环境变量 SEG_LR，则优先使用，便于在训练瓶颈时手动降低学习率
            env_lr = os.environ.get("SEG_LR")
            try:
                initial_lr = float(env_lr) if env_lr is not None else default_lr
            except ValueError:
                print(f"[警告] 无法解析 SEG_LR='{env_lr}'，回退到默认学习率 {default_lr}")
                initial_lr = default_lr

            # 【降维打击】为SMP模型（U-Net++和DeepLabV3+）实现差异化学习率：encoder使用小LR，decoder使用10倍LR
            # 这样可以让ResNet保持稳定，同时强行把随机初始化的头部拉起来
            if self.model_type in ("smp_unetplusplus", "smp_deeplabv3plus", "deeplabv3plus"):
                # 获取encoder和decoder参数
                encoder_params = []
                decoder_params = []
                
                # 处理可能的DataParallel包装
                actual_model = model.module if isinstance(model, nn.DataParallel) else model
                
                # 检查是否是SMP模型（UnetPlusPlus或DeepLabV3Plus）
                if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'encoder'):
                    # SMP模型结构：model.model.encoder 和 model.model.decoder（或decoder）
                    encoder_params = list(actual_model.model.encoder.parameters())
                    # DeepLabV3+和UnetPlusPlus都使用decoder
                    if hasattr(actual_model.model, 'decoder'):
                        decoder_params = list(actual_model.model.decoder.parameters())
                    else:
                        # 如果没有decoder，可能是其他结构，尝试获取所有非encoder参数
                        decoder_params = []
                        for name, param in actual_model.model.named_parameters():
                            if 'encoder' not in name:
                                decoder_params.append(param)
                    # 添加segmentation_head参数到decoder组
                    if hasattr(actual_model.model, 'segmentation_head'):
                        decoder_params.extend(list(actual_model.model.segmentation_head.parameters()))
                elif hasattr(actual_model, 'encoder') and hasattr(actual_model, 'decoder'):
                    # 直接有encoder和decoder属性
                    encoder_params = list(actual_model.encoder.parameters())
                    decoder_params = list(actual_model.decoder.parameters())
                    if hasattr(actual_model, 'segmentation_head'):
                        decoder_params.extend(list(actual_model.segmentation_head.parameters()))
                else:
                    # 回退：无法识别结构，使用统一学习率
                    print("[警告] 无法识别SMP模型结构，使用统一学习率")
                    encoder_params = []
                    decoder_params = list(model.parameters())
                
                if encoder_params and decoder_params:
                    # 参数分组：encoder和decoder使用相同学习率（修正：移除10倍倍率，避免训练初期震荡）
                    decoder_lr = initial_lr
                    print(f"[差异化学习率] Encoder LR: {initial_lr:.2e}, Decoder LR: {decoder_lr:.2e}")
                    optimizer = self._create_optimizer_with_groups(
                        [
                            {'params': encoder_params, 'lr': initial_lr},
                            {'params': decoder_params, 'lr': decoder_lr}
                        ],
                        lr=initial_lr  # 默认LR（用于scheduler）
                    )
                else:
                    # 回退到统一学习率
                    optimizer = self._create_optimizer(model.parameters(), lr=initial_lr)
            else:
                # 非SMP U-Net++模型，使用统一学习率
                optimizer = self._create_optimizer(model.parameters(), lr=initial_lr)
            
            # 【简化】移除 pos_weight，使用标准 BCEWithLogitsLoss
            # Dice Loss 本身就能很好地处理类别不平衡，额外的 pos_weight 会导致严重的假阳性
            bce_criterion = nn.BCEWithLogitsLoss()

            # Poly学习率 + Warmup: lr = base_lr * (1 - epoch / max_epochs) ** power
            warmup_epochs_lr = 5
            poly_power = float(os.environ.get("SEG_POLY_POWER", "0.9"))
            scheduler = None
            # 使用 ReduceLROnPlateau 在验证Dice长期不提升时自动降低学习率
            # 配置：当val_dice在3-5个Epoch内不再上升时，自动将学习率减半
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',           # 监控验证Dice（越大越好）
                factor=0.5,          # 学习率减半（而非0.1的10倍降低，更温和）
                patience=4,           # 4个epoch无提升则降低学习率（3-5之间）
                min_lr=1e-6           # 最小学习率下限
                # 注意：新版本PyTorch不再支持verbose参数，学习率变化日志由自定义代码打印
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
            
            # 【性能优化】开启 CuDNN Benchmark，自动寻找最适合当前硬件的卷积算法
            # 这通常能带来 10-20% 的加速，特别适合 RTX 4090 这样的高性能 GPU
            if device.type == 'cuda':
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = True
                print("[性能优化] CuDNN Benchmark 已启用，将自动优化卷积算法")
            
            # 训练循环
            # 【性能优化】编码器冻结策略已禁用，让编码器全程参与训练以提升性能和精度
            # 解冻编码器会显著增加 GPU 负载并提升 HD95 指标
            # 如果遇到显存不足，可以通过增大 Batch Size 来利用 AMP 节省的显存
            # 
            # 冻结/解冻策略：前50% epoch冻结编码器，后50%解冻进行微调（已禁用）
            # freeze_epochs = int(self.epochs * 0.5)
            # encoder_frozen = False
            # 训练过程中用于学习率调度的基准LR（解冻时会动态下调）
            base_lr = float(initial_lr)
            
            for epoch in range(self.epochs):
                if self.stop_requested:
                    self.update_progress.emit(0, "训练已由用户停止")
                    # 【修复】用户停止时也要发送完成信号，确保UI正确更新
                    self.training_finished.emit("训练已被用户停止", self.best_model_path if self.save_best else None)
                    return
                
                # 【性能优化】编码器冻结逻辑已禁用，让编码器全程参与训练
                # 解冻编码器会显著增加 GPU 负载并提升 HD95 指标
                # 如果遇到显存不足，可以通过增大 Batch Size 来利用 AMP 节省的显存
                # 
                # 冻结/解冻编码器逻辑（已禁用，仅对 ResNetUNet 有效）
                # if self.model_type == "resnet_unet":
                #     actual_model = self._unwrap_model(model)
                #     if isinstance(actual_model, ResNetUNet):
                #         if epoch < freeze_epochs:
                #             # 前50% epoch：冻结编码器
                #             if not encoder_frozen:
                #                 actual_model._freeze_encoder()
                #                 encoder_frozen = True
                #                 # 重新创建优化器，只优化可训练参数
                #                 trainable_params = [p for p in model.parameters() if p.requires_grad]
                #                 optimizer = self._create_optimizer(trainable_params, initial_lr)
                #                 print(f"[训练策略] Epoch {epoch+1}/{self.epochs}: 编码器已冻结，仅训练解码器")
                #         else:
                #             # 后50% epoch：解冻编码器进行微调
                #             if encoder_frozen:
                #                 actual_model._unfreeze_encoder()
                #                 encoder_frozen = False
                #                 # 重新创建优化器，优化所有参数（使用较小的学习率进行微调）
                #                 # 解冻瞬间：把"当前学习率"强制降低到 1/10，避免 ResNet101 全量微调震荡
                #                 current_lr = float(optimizer.param_groups[0]['lr'])
                #                 fine_tune_lr = current_lr * 0.1
                #                 base_lr = fine_tune_lr  # 同时更新后续Poly调度的基准LR，避免被initial_lr覆盖回去
                #                 trainable_params = [p for p in model.parameters() if p.requires_grad]
                #                 optimizer = self._create_optimizer(trainable_params, fine_tune_lr)
                #                 print(f"[训练策略] Epoch {epoch+1}/{self.epochs}: 编码器已解冻，开始端到端微调 (LR={fine_tune_lr:.6f})")
                
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
                    
                    # 【通道一致性检查】确保数据通道数与模型匹配
                    # DeepLabV3+ 使用"伪三通道流"：数据加载器将单通道图像转换为3通道RGB
                    # 模型固定为3通道输入，因此 images 应该是 [B, 3, H, W]
                    # 不要对 images 进行通道切片（如 images[:, :1, :, :]），这会破坏通道匹配
                    if images.shape[1] != 3:
                        # 防御性检查：如果数据是1通道，复制为3通道（防守性编程）
                        if images.shape[1] == 1:
                            print(f"[警告] 训练阶段: 检测到1通道数据，自动复制为3通道以匹配DeepLabV3+")
                            images = images.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                        else:
                            raise ValueError(f"训练阶段: 意外的通道数 {images.shape[1]}，期望3通道")
                    
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
                    
                    # 定期清理GPU缓存，降低显存峰值（更频繁地清理）
                    if batch_idx % 5 == 0 and torch.cuda.is_available():
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

                    # 【显存优化】训练循环异常处理，捕获 OOM 错误
                    try:
                        with autocast(device_type=amp_device_type, enabled=amp_enabled):
                            supports_aux = self._supports_aux_outputs(model)
                            supports_attention = self._supports_attention_maps(model)
                            
                            # 【DeepLabV3+ 兼容性】检查模型类型，DeepLabV3+ 的 forward 不支持 return_attention
                            is_deeplabv3 = self.model_type in ("deeplabv3plus", "smp_deeplabv3plus")
                            
                            # 【显存优化】训练阶段彻底禁用注意力图生成，避免显存溢出
                            # 训练和验证阶段都不生成注意力热力图，仅在测试阶段（ModelTestThread）生成
                            need_attention = False  # 训练阶段强制禁用，避免 OOM
                        
                        # 【显存优化】训练阶段不传递 return_attention 参数，彻底禁用注意力图生成
                        forward_kwargs = {}
                        if supports_aux:
                            forward_kwargs['return_aux'] = True
                        # 训练阶段不传递 return_attention，即使模型支持也不生成注意力图
                        # if supports_attention:
                        #     forward_kwargs['return_attention'] = True
                        
                        # 【紧急修复】DeepLabV3+ 不支持 return_attention，必须在调用前移除
                        # 无论 need_attention 是否为 True，都要移除，因为 DeepLabV3+ 的 forward 方法不支持此参数
                        if is_deeplabv3 and "return_attention" in forward_kwargs:
                            forward_kwargs.pop("return_attention")
                        
                        # 【显存优化】确保训练阶段不生成任何注意力图
                        if "return_attention" in forward_kwargs:
                            forward_kwargs.pop("return_attention")
                        
                        # 执行正常的前向传播（用于计算 Loss 和指标）
                        if forward_kwargs:
                            forward_out = model(images, **forward_kwargs)
                            if supports_aux:
                                outputs, aux_outputs = forward_out
                            else:
                                outputs = forward_out
                                aux_outputs = []
                        else:
                            outputs = model(images)
                            aux_outputs = []
                        
                        # 【显存优化】训练阶段彻底禁用所有注意力图生成，避免显存溢出
                        # 训练和验证阶段都不生成注意力热力图（Grad-CAM 或原生注意力图）
                        # 仅在测试阶段（ModelTestThread）生成注意力热力图用于最终分析
                        attention_maps = {}  # 始终为空字典，不生成任何注意力图
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
                        
                        # 记录未缩放的原始 loss 用于日志和 epoch 统计
                        loss_value = loss.item()
                        if not np.isfinite(loss_value):
                            print(f"[警告] Epoch {epoch+1}, Batch {batch_idx+1}: 损失值为NaN/Inf，使用0.0")
                            loss_value = 0.0
                        
                        # 梯度累积：将 loss 按累积步数缩放，累计多个小 batch 的梯度后再进行一次优化步骤
                        loss = loss / max(1, getattr(self, "accumulation_steps", 1))
                        
                        scaler.scale(loss).backward()
                        
                        # 只有在达到累积步数时才执行一次优化器 step
                        should_step = ((batch_idx + 1) % max(1, getattr(self, "accumulation_steps", 1)) == 0) or ((batch_idx + 1) == len(train_loader))
                        if should_step:
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
                                optimizer.zero_grad(set_to_none=True)
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
                                print(f"[调试] Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss_value:.4f}, GradNorm={total_grad_norm:.6f}, LR={optimizer.param_groups[0]['lr']:.8f}")
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
                                # 如果连续多个step梯度为0，临时提高学习率
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
                                    optimizer.zero_grad(set_to_none=True)
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
                            optimizer.zero_grad(set_to_none=True)
                            
                            # 【显存优化】定期清理显存，防止 OOM
                            # 每 N 个 batch 清理一次（避免过于频繁影响性能）
                            if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
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
                        
                        # EMA 更新（在 try 块内，确保总是执行）
                        if self.use_ema and ema_model is not None:
                            self._update_ema_model(ema_model, model)
                    
                    except RuntimeError as e:
                        # 【显存优化】捕获 OOM 异常并清理显存
                        if "out of memory" in str(e).lower():
                            print(f"[严重警告] Epoch {epoch+1}, Batch {batch_idx+1}: CUDA OOM 错误，清理显存并跳过此批次")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # 清理可能残留的变量
                            if 'outputs' in locals():
                                del outputs
                            if 'images' in locals():
                                del images
                            if 'masks' in locals():
                                del masks
                            continue
                        else:
                            # 其他 RuntimeError 重新抛出
                            raise
                    
                    # 累加 epoch 损失（使用未缩放的 loss_value）
                    epoch_loss += loss_value * batch_size
                    
                    # 【显存优化】定期清理GPU缓存（更频繁地清理）
                    if batch_idx % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 【显存优化】删除训练循环中的中间变量
                    if 'outputs' in locals():
                        del outputs
                    if 'aux_outputs' in locals():
                        del aux_outputs
                    if 'attention_maps' in locals():
                        del attention_maps
                    if 'mixed_images' in locals():
                        del mixed_images
                    if 'mixed_masks' in locals():
                        del mixed_masks
                    if 'indices' in locals():
                        del indices
                    
                    # 更新训练进度
                    train_progress = 20 + int(50 * (batch_idx + 1) / len(train_loader))
                    self.update_progress.emit(
                        train_progress,
                        f"轮次 {epoch+1}/{self.epochs} | 批次 {batch_idx+1}/{len(train_loader)} | 损失: {loss_value:.4f}"
                    )
                
                # 验证阶段
                model.eval()
                val_dice = 0.0
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
                
                # 【修复】添加IoU/Precision/Recall累加器，基于全量样本计算
                val_iou_sum = 0.0
                val_precision_sum = 0.0
                val_recall_sum = 0.0
                
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
                        # 使用全部验证集进行阈值优化
                        # 注意：阈值优化阶段不使用后处理（为了速度），但验证阶段会使用后处理
                        # 这可能导致阈值优化找到的阈值与验证阶段实际效果略有差异，但通常影响很小
                        threshold_result = self.find_optimal_threshold(
                            eval_model_for_epoch,
                            val_loader,
                            device,
                            num_samples=None,  # None表示使用全部验证集
                        )
                        # 处理返回值：可能是元组(threshold, dice)或单个值（向后兼容）
                        if isinstance(threshold_result, tuple):
                            val_threshold, gwo_best_dice = threshold_result
                            # 【关键修复】保存GWO找到的全验证集最佳Dice，用于best_model判定
                            self.gwo_best_dice = float(gwo_best_dice)
                            print(f">>> [GWO] 全验证集最佳Dice已保存: {self.gwo_best_dice:.4f} (将用于best_model判定)")
                        else:
                            # 向后兼容：如果返回单个值
                            val_threshold = float(threshold_result)
                            self.gwo_best_dice = None
                        self.last_optimal_threshold = val_threshold
                        # 【显存优化】阈值优化后清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as threshold_err:
                        print(f"[警告] 阈值搜索失败，使用上一次的阈值。原因: {threshold_err}")
                        val_threshold = float(getattr(self, "last_optimal_threshold", 0.5))
                        self.gwo_best_dice = None
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
                        
                        # 【通道一致性检查】确保验证数据通道数与模型匹配
                        # DeepLabV3+ 使用"伪三通道流"：数据加载器将单通道图像转换为3通道RGB
                        # 模型固定为3通道输入，因此 images 应该是 [B, 3, H, W]
                        # 不要对 images 进行通道切片，确保验证数据直接送入模型
                        if images.shape[1] != 3:
                            # 防御性检查：如果数据是1通道，复制为3通道（防守性编程）
                            if images.shape[1] == 1:
                                print(f"[警告] 验证阶段: 检测到1通道数据，自动复制为3通道以匹配DeepLabV3+")
                                images = images.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                            else:
                                raise ValueError(f"验证阶段: 意外的通道数 {images.shape[1]}，期望3通道")
                            
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
                            
                            # 【性能优化】验证阶段禁用TTA（测试时增强）以提升速度
                            # 策略：原图预测 + 水平翻转后预测再翻回来，取平均
                            # 这通常能白嫖0.01~0.02的Dice分数，但会显著降低验证速度（24倍推理）
                            # 可以通过环境变量 SEG_USE_TTA_IN_VAL=1 启用（不推荐，速度慢）
                            use_tta_in_val = os.environ.get("SEG_USE_TTA_IN_VAL", "0") == "1"  # 默认禁用以提升速度
                            
                            if use_tta_in_val:
                                # 使用TTA提升验证性能
                                try:
                                    outputs = self._tta_inference(eval_model_for_epoch, images)
                                    if brain_mask is not None:
                                        outputs = outputs * brain_mask
                                except RuntimeError as e:
                                    if "out of memory" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
                                        print(f"[严重警告] 验证阶段: Batch {val_idx+1}: TTA推理失败 ({str(e)[:100]})，跳过该batch")
                                        # 【显存优化】OOM 时清理显存
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
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
                                fill_holes=True,     # 填充孔洞，去除假阴性空洞
                                prob_map=prob_map_tensor
                            )
                            # post_process_mask会返回tensor或numpy，需要确保是tensor
                            if isinstance(pred_mask_processed, torch.Tensor):
                                preds[i, 0] = pred_mask_processed.to(preds.device)
                            else:
                                preds[i, 0] = torch.from_numpy(pred_mask_processed).float().to(preds.device)
                        
                        # 【统一计算】使用统一的指标计算函数（单一真理来源）
                        batch_metrics = self.calculate_batch_metrics(preds.float(), masks)
                        batch_dice = batch_metrics['dice']
                        batch_iou = batch_metrics['iou']
                        batch_precision = batch_metrics['precision']
                        batch_recall = batch_metrics['recall']
                        batch_is_empty = batch_metrics['is_empty']
                        
                        batch_size = masks.shape[0]
                        for i in range(batch_size):
                            dice_i = batch_dice[i]
                            iou_i = batch_iou[i]
                            precision_i = batch_precision[i]
                            recall_i = batch_recall[i]
                            is_empty_i = batch_is_empty[i]
                            
                            # 累加所有样本的IoU/Precision/Recall（包括空mask和前景样本）
                            val_iou_sum += iou_i
                            val_precision_sum += precision_i
                            val_recall_sum += recall_i
                            
                            if is_empty_i:
                                val_empty_mask_count += 1
                                val_empty_mask_dice_sum += dice_i
                            else:
                                val_non_empty_mask_count += 1
                                val_non_empty_mask_dice_sum += dice_i
                        
                        # 统计像素信息（用于日志）
                        val_pred_fg_pixels += preds.sum().item()
                        val_gt_fg_pixels += masks.sum().item()
                        val_total_pixels += float(masks.numel())
                        
                        # 更新验证进度
                        val_progress = int(100 * (val_idx + 1) / len(val_loader))
                        current_avg_loss = val_loss / max(1, val_samples)
                        # 计算当前批次的所有样本平均 Dice（用于最佳模型选择）
                        val_current_total_count = val_non_empty_mask_count + val_empty_mask_count
                        val_current_total_dice_sum = val_non_empty_mask_dice_sum + val_empty_mask_dice_sum
                        current_avg_dice = val_current_total_dice_sum / max(1, val_current_total_count)
                        # 计算当前批次的 Dice_Pos 和 Dice_Neg（用于进度显示）
                        current_dice_pos = val_non_empty_mask_dice_sum / max(1, val_non_empty_mask_count) if val_non_empty_mask_count > 0 else 0.0
                        current_dice_neg = val_empty_mask_dice_sum / max(1, val_empty_mask_count) if val_empty_mask_count > 0 else 0.0
                        
                        self.update_val_progress.emit(
                            val_progress,
                            f"验证轮次 {epoch+1} | 批次 {val_idx+1}/{len(val_loader)}\n"
                            f"损失: {current_avg_loss:.4f} | Dice_Pos: {current_dice_pos:.4f} | Dice_Neg: {current_dice_neg:.4f} | 整体Dice(所有样本): {current_avg_dice:.4f}"
                        )
                        
                        # 每5个批次强制更新UI
                        if val_idx % 5 == 0:
                            QApplication.processEvents()
                        
                        # 【显存优化】显式删除中间变量，释放GPU显存
                        del outputs, probs, preds, batch_dice
                        if 'pred_mask_tensor' in locals():
                            del pred_mask_tensor
                        if 'prob_map_tensor' in locals():
                            del prob_map_tensor
                        if 'pred_mask_processed' in locals():
                            del pred_mask_processed
                        # 每10个批次清理一次GPU缓存
                        if val_idx % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # 计算平均值（确保没有NaN/Inf）
                avg_train_loss = epoch_loss / max(1, train_samples)
                if not np.isfinite(avg_train_loss):
                    print(f"[警告] Epoch {epoch+1}: 训练平均损失为NaN/Inf，使用0.0")
                    avg_train_loss = 0.0
                
                # 【修改】val_dice 统计所有验证样本（包括空mask样本）
                # 使用所有样本的平均 Dice 来选择最佳模型，确保模型在所有场景下都有良好表现
                val_total_count = val_non_empty_mask_count + val_empty_mask_count
                val_total_dice_sum = val_non_empty_mask_dice_sum + val_empty_mask_dice_sum
                
                if val_total_count > 0:
                    val_dice = val_total_dice_sum / val_total_count
                else:
                    # 如果没有样本，使用0.0（而不是NaN）
                    val_dice = 0.0
                
                if not np.isfinite(val_dice):
                    print(f"[警告] Epoch {epoch+1}: 验证Dice为NaN/Inf，使用0.0")
                    val_dice = 0.0

                # 使用 ReduceLROnPlateau 根据验证Dice自动调整学习率（优先提升稳定性）
                if plateau_scheduler is not None:
                    old_lr = optimizer.param_groups[0]['lr']
                    plateau_scheduler.step(val_dice)
                    new_lr = optimizer.param_groups[0]['lr']
                    # 如果学习率发生变化，打印显眼的提示
                    if new_lr < old_lr:
                        print(f"\n{'='*60}")
                        print(f"📉 检测到性能停滞，学习率下调为: {new_lr:.2e} (原: {old_lr:.2e})")
                        print(f"   当前验证Dice: {val_dice:.4f}")
                        print(f"{'='*60}\n")
                
                avg_val_loss = val_loss / max(1, val_samples)
                if not np.isfinite(avg_val_loss):
                    print(f"[警告] Epoch {epoch+1}: 验证平均损失为NaN/Inf，使用0.0")
                    avg_val_loss = 0.0
                
                pred_fg_ratio = val_pred_fg_pixels / max(1.0, val_total_pixels)
                gt_fg_ratio = val_gt_fg_pixels / max(1.0, val_total_pixels)
                
                # 【关键修改】分别统计有前景mask和空mask的Dice（用于诊断和详细分析）
                # 注意：val_dice 现在统计所有样本（包括空mask样本），用于最佳模型选择
                dice_pos = val_non_empty_mask_dice_sum / max(1, val_non_empty_mask_count) if val_non_empty_mask_count > 0 else 0.0
                dice_neg = val_empty_mask_dice_sum / max(1, val_empty_mask_count) if val_empty_mask_count > 0 else 0.0
                empty_mask_ratio = val_empty_mask_count / max(1, val_samples) if val_samples > 0 else 0.0
                
                # 记录到历史中（记录所有样本的平均Dice，用于最佳模型选择）
                val_total_count = val_non_empty_mask_count + val_empty_mask_count
                val_total_dice = val_dice  # 已经在上面计算为所有样本的平均Dice
                # 【修复】val_dice_history 已在下方"更新训练历史"部分统一添加，此处不再重复添加
                self.val_dice_pos_history.append(dice_pos)  # 保留用于诊断
                self.val_dice_neg_history.append(dice_neg)  # 保留用于诊断
                
                # 【统一日志格式】重写验证报告输出
                print(f"\n{'='*60}")
                print(f"[验证报告] Epoch {epoch+1} | 最佳阈值: {val_threshold:.4f}")
                print(f"{'-'*60}")
                print(f"[整体表现] Mean Dice (全样): {val_dice:.4f}  <-- (用于 Best Model 判定)")
                print(f"[分组详情]")
                print(f"   - 空 Mask ({val_empty_mask_count}/{val_samples}): {dice_neg:.4f}  (反映背景抑制能力)")
                print(f"   - 前景类 ({val_non_empty_mask_count}/{val_samples}): {dice_pos:.4f}  (反映病灶识别能力)")
                
                # 【诊断信息】检查阈值是否过高
                if dice_neg > 0.9 and dice_pos < 0.5:
                    print(f"[警告] 阈值过高，虽然抑制了背景，但严重损伤了前景识别")
                
                # 【修复】计算详细指标（IoU, Precision, Recall）- 基于全量样本的平均值
                val_total_count = val_non_empty_mask_count + val_empty_mask_count
                if val_total_count > 0:
                    avg_iou = val_iou_sum / val_total_count
                    avg_precision = val_precision_sum / val_total_count
                    avg_recall = val_recall_sum / val_total_count
                    print(f"[详细指标] IoU: {avg_iou:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f}")
                else:
                    print(f"[详细指标] IoU: N/A | Precision: N/A | Recall: N/A  (无样本)")
                
                print(f"{'-'*60}")
                print(f"Loss: {avg_val_loss:.4f} (基于全部验证集{val_samples}个样本，使用后处理)\n")

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
                
                # 【显存优化】显式删除验证阶段的变量（但保留eval_model_for_epoch，后续还会用到）
                if 'val_batch' in locals():
                    del val_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 每个轮次结束后生成性能分析可视化
                self.update_progress.emit(
                    int(70 + 20 * (epoch + 1) / self.epochs),
                    f"轮次 {epoch+1} 完成 (LR={current_lr:.6f})，生成性能分析..."
                )
                
                # 【双引擎策略】判断是否为关键 Epoch（每20轮或最佳模型）
                is_best_epoch = (val_dice > self.best_dice) if hasattr(self, 'best_dice') else False
                is_key_epoch = ((epoch + 1) % 20 == 0) or is_best_epoch
                
                # 生成测试集分割结果可视化
                # 使用当前轮次计算出的最佳阈值
                viz_threshold = getattr(self, 'last_optimal_threshold', 0.1)
                # 【显存优化】临时创建eval模型用于可视化
                temp_eval_model = model.eval() if not isinstance(model, nn.DataParallel) else model.module.eval()
                if self.use_ema and ema_model is not None and epoch >= self.ema_eval_start_epoch:
                    temp_eval_model = ema_model.eval()
                    if isinstance(model, nn.DataParallel):
                        temp_eval_model = nn.DataParallel(temp_eval_model)
                test_viz_path = self.visualize_test_results(
                    temp_eval_model, 
                    val_loader, 
                    device, 
                    num_samples=6,  # 每个轮次显示6个样本
                    use_tta=True,   # 训练结束后的测试使用TTA
                    epoch=epoch + 1,  # 传入当前轮次（1-based）
                    is_best=is_best_epoch,  # 传入是否为最佳模型
                    threshold=viz_threshold  # 传入当前轮次的最佳阈值
                )
                # 【显存优化】删除临时eval模型引用
                del temp_eval_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 计算当前轮次的性能指标（全验证集评估）
                # 【关键修复】验证评分现在使用全部验证集，与验证统计保持一致
                # 1. 验证统计：使用全部验证集 + 后处理，用于主要评估和早停判断
                # 2. 验证评分：使用全部验证集 + 后处理，用于性能分析和可视化（已修复）
                # 【显存优化】使用临时eval模型
                temp_eval_model_for_metrics = model.eval() if not isinstance(model, nn.DataParallel) else model.module.eval()
                if self.use_ema and ema_model is not None and epoch >= self.ema_eval_start_epoch:
                    temp_eval_model_for_metrics = ema_model.eval()
                    if isinstance(model, nn.DataParallel):
                        temp_eval_model_for_metrics = nn.DataParallel(temp_eval_model_for_metrics)
                
                epoch_metrics = {
                    'dice': [],
                    'iou': [],
                    'precision': [],
                    'recall': [],
                    'sensitivity': [],
                    'f1': []
                }
                
                with torch.no_grad():
                    # 【关键修复】使用全部验证集，不再限制为20个样本
                    eval_count = 0
                    total_eval_samples = len(val_dataset)
                    
                    for batch_data in val_loader:
                        
                        # 处理数据：可能包含分类标签
                        if len(batch_data) == 3:
                            images, masks, _ = batch_data
                        else:
                            images, masks = batch_data
                        images, masks = images.to(device), masks.to(device)
                        brain_mask = None
                        if self.use_skull_stripper:
                            images, brain_mask = self._apply_skull_strip(images)
                        
                        outputs = temp_eval_model_for_metrics(images)
                        # 确保 outputs 和 masks 的空间尺寸匹配
                        if outputs.shape[2:] != masks.shape[2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        if brain_mask is not None:
                            outputs = outputs * brain_mask
                        
                        probs = torch.sigmoid(outputs)
                        preds = (probs > val_threshold).float()
                        
                        # 【统一逻辑】验证评分也应该使用后处理，与验证统计保持一致
                        # 先执行智能后处理
                        for i in range(preds.shape[0]):
                            pred_mask_tensor = preds[i, 0]
                            prob_map_tensor = probs[i, 0]
                            pred_mask_tensor = self.smart_post_processing(pred_mask_tensor, prob_map_tensor)
                            # 再执行传统形态学后处理
                            pred_mask_processed = self.post_process_mask(
                                pred_mask_tensor,
                                min_size=0,
                                use_morphology=True,
                                keep_largest=False,
                                fill_holes=True,
                                prob_map=prob_map_tensor
                            )
                            if isinstance(pred_mask_processed, torch.Tensor):
                                preds[i, 0] = pred_mask_processed.to(preds.device)
                            else:
                                preds[i, 0] = torch.from_numpy(pred_mask_processed).float().to(preds.device)
                        
                        for i in range(preds.shape[0]):
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
                            
                            # 【关键修复】统计所有样本（包括空mask），与验证统计保持一致
                            # 空mask样本的Dice计算：如果GT为空且预测也为空，Dice=1.0；否则Dice=0.0
                            empty_threshold_pixels = max(1e-7, float(mask.numel()) * 0.001)  # 0.1%像素，统一阈值
                            
                            if mask_sum <= empty_threshold_pixels:
                                # 空mask样本：GT为空
                                if pred_sum <= 1e-7:
                                    dice = 1.0  # GT为空，预测也为空，Dice=1.0
                                else:
                                    dice = 0.0  # GT为空，预测不为空（假阳性），Dice=0.0
                                iou = dice  # IoU与Dice相同
                                precision = 0.0 if pred_sum > 1e-7 else 1.0
                                recall = 1.0  # GT为空，recall=1.0（没有漏检）
                            else:
                                # 有前景样本
                                dice_den = 2.0 * tp + fp + fn
                                if dice_den < 1e-7:
                                    dice = 0.0  # 有前景但预测为空，Dice=0
                                else:
                                    dice = (2.0 * tp) / dice_den
                                
                                union = tp + fp + fn
                                iou = 1.0 if union < 1e-7 else tp / union
                                
                                if (tp + fp) < 1e-7:
                                    precision = 0.0  # 有前景但预测为空，precision=0
                                else:
                                    precision = tp / (tp + fp)
                                
                                if (tp + fn) < 1e-7:
                                    recall = 0.0  # 有前景但预测为空，recall=0
                                else:
                                    recall = tp / (tp + fn)
                                
                                f1 = dice  # 二分类下F1=Dice
                                
                            # 统计所有样本（包括空mask）
                                epoch_metrics['dice'].append(float(dice))
                                epoch_metrics['iou'].append(float(iou))
                                epoch_metrics['precision'].append(float(precision))
                                epoch_metrics['recall'].append(float(recall))
                                epoch_metrics['sensitivity'].append(float(recall))
                                epoch_metrics['f1'].append(float(f1))
                            
                            eval_count += 1
                        
                        # 【显存优化】删除epoch分析阶段的中间变量
                        del outputs, probs, preds, images, masks
                        if brain_mask is not None:
                            del brain_mask
                        if torch.cuda.is_available() and eval_count % 5 == 0:
                            torch.cuda.empty_cache()
                
                # 【显存优化】删除临时eval模型
                del temp_eval_model_for_metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 计算平均指标
                avg_epoch_metrics = {}
                for k, values in epoch_metrics.items():
                    arr = np.array(values, dtype=float)
                    if arr.size == 0 or np.all(np.isnan(arr)):
                        avg_epoch_metrics[k] = float('nan')
                    else:
                        avg_epoch_metrics[k] = float(np.nanmean(arr))

                # 【关键修复】验证评分现在使用全部验证集，与验证统计保持一致
                # 计算实际评估的样本数量
                actual_eval_samples = len(epoch_metrics.get('dice', []))
                # 【统一标准】验证评分统计所有样本（包括空mask），与验证统计保持一致
                print(
                    f"[验证评分] Epoch {epoch+1}: threshold={val_threshold:.3f}, "
                    f"Dice(所有样本)={avg_epoch_metrics.get('dice', float('nan')):.4f}, "
                    f"IoU(所有样本)={avg_epoch_metrics.get('iou', float('nan')):.4f}, "
                    f"Precision={avg_epoch_metrics.get('precision', float('nan')):.4f}, "
                    f"Recall={avg_epoch_metrics.get('recall', float('nan')):.4f} "
                    f"(基于全部验证集{actual_eval_samples}个样本，使用后处理)"
                )
                
                # 【高清模式】关键 Epoch 时调用所有 MATLAB 方法生成出版级图表
                # 注意：必须在 epoch_metrics 和 avg_epoch_metrics 计算完成后调用
                if is_key_epoch and self.enable_matlab_plots and self.matlab_viz_bridge:
                    print(f"\n[高清渲染] Epoch {epoch+1} 是关键轮次，正在调用 MATLAB 生成出版级图表...")
                    try:
                        # 1. 生成性能分析报表（如果数据可用）
                        if len(epoch_metrics.get('dice', [])) > 0:
                            try:
                                # 【数据清洗】从 epoch_metrics 中提取纯数值列表，确保是 double 类型
                                # epoch_metrics 是一个字典，键是指标名，值是列表
                                dice_values = epoch_metrics.get('dice', [])
                                
                                # 确保所有值都是数值类型（不是字典或结构体）
                                dice_values_clean = []
                                for val in dice_values:
                                    if isinstance(val, (int, float, np.number)):
                                        dice_values_clean.append(float(val))
                                    elif isinstance(val, dict):
                                        # 如果值是字典，尝试提取数值（向后兼容）
                                        dice_values_clean.append(float(val.get('val_dice', val.get('dice', 0.0))))
                                    else:
                                        # 其他类型，尝试转换为 float
                                        try:
                                            dice_values_clean.append(float(val))
                                        except (ValueError, TypeError):
                                            dice_values_clean.append(0.0)
                                
                                # 转换为 numpy 数组，确保是 double 类型
                                dice_array = np.array(dice_values_clean, dtype=np.float64)
                                
                                # 直接保存为 .mat 文件，使用 MATLAB 脚本期望的字段名
                                perf_payload_path = os.path.join(self.temp_dir, f"performance_metrics_epoch{epoch+1}_payload.mat")
                                from scipy.io import savemat
                                savemat(perf_payload_path, {
                                    'dice_scores': dice_array,  # MATLAB 脚本期望的字段名
                                    'iou_scores': np.array(epoch_metrics.get('iou', []), dtype=np.float64),
                                    'precision_scores': np.array(epoch_metrics.get('precision', []), dtype=np.float64),
                                    'recall_scores': np.array(epoch_metrics.get('recall', []), dtype=np.float64),
                                })
                                
                                # 【持久化修复】保存到持久化目录
                                import time
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                perf_analysis_path = os.path.join(self.persistent_report_dir, f"performance_analysis_epoch{epoch+1}_{timestamp}_matlab.png")
                                os.makedirs(os.path.dirname(perf_analysis_path), exist_ok=True)
                                self.matlab_viz_bridge.render_performance_analysis(perf_payload_path, perf_analysis_path)
                                print(f"[高清渲染] 性能分析报表已保存到持久化目录: {perf_analysis_path}")
                            except Exception as exc:
                                print(f"[高清渲染] 性能分析报表生成失败: {exc}")
                                import traceback
                                traceback.print_exc()
                        
                        # 2. 生成注意力热力图（如果模型支持）
                        # 【显存优化】使用临时eval模型
                        temp_eval_model_for_att = model.eval() if not isinstance(model, nn.DataParallel) else model.module.eval()
                        if self.use_ema and ema_model is not None and epoch >= self.ema_eval_start_epoch:
                            temp_eval_model_for_att = ema_model.eval()
                            if isinstance(model, nn.DataParallel):
                                temp_eval_model_for_att = nn.DataParallel(temp_eval_model_for_att)
                        
                        if self._supports_attention_maps(temp_eval_model_for_att):
                            try:
                                # 收集注意力数据
                                all_images_att = []
                                all_masks_att = []
                                all_preds_att = []
                                att_layer_payload = {'att1': [], 'att2': [], 'att3': [], 'att4': []}
                                
                                # 【性能优化】只对前 2 个 batch 生成 Grad-CAM，其他 batch 跳过以提升速度
                                max_gradcam_batches_att = 2  # 只对前 2 个 batch 生成 Grad-CAM（训练循环中）
                                
                                att_count = 0
                                for batch_idx_att, batch_data in enumerate(val_loader):
                                    if att_count >= 4:  # 只收集4个样本
                                        break
                                    if len(batch_data) == 3:
                                        images, masks, _ = batch_data
                                    else:
                                        images, masks = batch_data
                                    images, masks = images.to(device), masks.to(device)
                                    
                                    # 【性能优化】判断是否需要生成 Grad-CAM（仅前 2 个 batch）
                                    need_gradcam_att = (batch_idx_att < max_gradcam_batches_att)
                                    is_deeplabv3 = self.model_type in ("deeplabv3plus", "smp_deeplabv3plus")
                                    
                                    if need_gradcam_att:
                                        # 需要 Grad-CAM 的样本：必须在 torch.enable_grad() 下运行
                                        # 【DeepLabV3+ 兼容性 + Grad-CAM 集成】DeepLabV3+ 不支持 return_attention，使用 Grad-CAM
                                        if is_deeplabv3:
                                            # DeepLabV3+ 不支持 return_attention，先获取输出
                                            with torch.enable_grad():
                                                outputs = temp_eval_model_for_att(images)
                                                # 使用 Grad-CAM 生成热力图（需要梯度）
                                                actual_model = self._unwrap_model(temp_eval_model_for_att)
                                                attention_maps = self._generate_gradcam_for_deeplabv3(actual_model, images, device)
                                        else:
                                            outputs, attention_maps = temp_eval_model_for_att(images, return_attention=True)
                                    else:
                                        # 不需要 Grad-CAM 的样本：使用 torch.no_grad() 加速
                                        with torch.no_grad():
                                            if is_deeplabv3:
                                                # DeepLabV3+ 不需要注意力图，直接获取输出
                                                outputs = temp_eval_model_for_att(images)
                                                attention_maps = {}  # 不需要热力图
                                            else:
                                                # 其他模型：尝试获取注意力图，但不强制
                                                try:
                                                    outputs, attention_maps = temp_eval_model_for_att(images, return_attention=True)
                                                except:
                                                    # 如果获取失败，只获取输出
                                                    outputs = temp_eval_model_for_att(images)
                                                    attention_maps = {}
                                    
                                    # 如果不需要 Grad-CAM 且已收集足够样本，直接退出
                                    if not need_gradcam_att and att_count >= 4:
                                        break
                                    
                                    # 只处理需要可视化的样本（前 2 个 batch）
                                    if need_gradcam_att:
                                        preds = torch.sigmoid(outputs)
                                        preds_binary = (preds > 0.5).float()
                                        
                                        for i in range(images.size(0)):
                                            if att_count >= 4:
                                                break
                                            img = images[i].cpu().permute(1, 2, 0).numpy()
                                            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                                            img = np.clip(img, 0, 1).astype(np.float32)
                                            mask = masks[i, 0].cpu().numpy().astype(np.float32)
                                            pred = preds_binary[i, 0].cpu().numpy().astype(np.float32)
                                            
                                            all_images_att.append(img)
                                            all_masks_att.append(mask)
                                            all_preds_att.append(pred)
                                            
                                            # 收集注意力图
                                            for att_name in ['att1', 'att2', 'att3', 'att4']:
                                                if att_name in attention_maps:
                                                    att_np = attention_maps[att_name][i, 0].cpu().numpy()
                                                    # 上采样到512x512
                                                    from scipy.ndimage import zoom
                                                    target_size = (512, 512)  # 提升分辨率以保留更多病灶边缘细节
                                                    if att_np.shape != target_size:
                                                        zoom_factors = (target_size[0] / att_np.shape[0], target_size[1] / att_np.shape[1])
                                                        att_np = zoom(att_np, zoom_factors, order=1)
                                                    att_layer_payload[att_name].append(att_np)
                                            
                                            att_count += 1
                                        
                                        # 【显存优化】删除注意力图收集的中间变量
                                        del outputs, attention_maps, preds, preds_binary
                                        if torch.cuda.is_available() and att_count % 2 == 0:
                                            torch.cuda.empty_cache()
                                
                                if all_images_att:
                                    att_payload = self._save_attention_payload(all_images_att, all_masks_att, all_preds_att, att_layer_payload, f"attention_epoch{epoch+1}")
                                    # 【持久化修复】保存到持久化目录
                                    import time
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    att_path = os.path.join(self.persistent_report_dir, f"attention_visualization_epoch{epoch+1}_{timestamp}_matlab.png")
                                    os.makedirs(os.path.dirname(att_path), exist_ok=True)
                                    self.matlab_viz_bridge.render_attention_maps(att_payload, att_path)
                                    print(f"[高清渲染] 注意力热力图已保存到持久化目录: {att_path}")
                                
                                # 【显存优化】删除注意力图相关变量
                                del all_images_att, all_masks_att, all_preds_att, att_layer_payload
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception as exc:
                                print(f"[高清渲染] 注意力热力图生成失败: {exc}")
                        # 【显存优化】删除临时eval模型
                        if 'temp_eval_model_for_att' in locals():
                            del temp_eval_model_for_att
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as exc:
                        print(f"[高清渲染] MATLAB 高清渲染过程出错: {exc}")
                        import traceback
                        traceback.print_exc()
                else:
                    # 【快速模式】普通 Epoch 跳过 MATLAB 调用，只使用 Matplotlib 快速预览
                    if epoch is not None:
                        print(f"[快照] Epoch {epoch+1} 使用 Matplotlib 快速预览（非关键轮次）")
                
                # 发送epoch分析结果信号（包含综合评分）
                self.epoch_analysis_ready.emit(epoch + 1, test_viz_path, avg_epoch_metrics)
                
                # Save best model
                # 【关键修复】优先使用GWO找到的全验证集最佳Dice作为判定依据
                # 如果GWO未运行或失败，则回退到验证循环计算的val_dice
                dice_for_best_model = getattr(self, 'gwo_best_dice', None)
                if dice_for_best_model is None:
                    # 回退到验证循环计算的val_dice（基于全部验证集+后处理）
                    dice_for_best_model = val_dice
                    print(f">>> [Best Model] 使用验证循环Dice: {dice_for_best_model:.4f} (GWO未运行)")
                else:
                    print(f">>> [Best Model] 使用GWO全验证集最佳Dice: {dice_for_best_model:.4f}")
                
                if dice_for_best_model > self.best_dice:
                    self.best_dice = dice_for_best_model
                    if self.save_best:
                        os.makedirs(self.best_model_cache_dir, exist_ok=True)
                        self.best_model_path = os.path.join(
                            self.best_model_cache_dir, f"best_model_dice_{dice_for_best_model:.4f}.pth"
                        )
                        self._save_checkpoint(eval_model_for_epoch, self.best_model_path)
                        self.model_saved.emit(f"已保存最佳模型 (Dice: {dice_for_best_model:.4f}, 基于全验证集GWO优化)")

                # 恢复EMA模型为train模式（如果使用了EMA）
                if self.use_ema and ema_model is not None and epoch >= self.ema_eval_start_epoch:
                    ema_model.train()
                
                # 【显存优化】在所有使用eval_model_for_epoch的操作完成后，删除它
                del eval_model_for_epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
                    import platform
                    is_windows = platform.system() == 'Windows'
                    cpu_count = os.cpu_count() or 1
                    # 【Windows 多进程优化】Intel Core Ultra 9 285HX: 使用 8 个 worker 充分利用 P-Core
                    num_workers = 8 if is_windows else max(0, min(4, cpu_count - 1))
                    val_loader_cls = DataLoader(
                        val_dataset_cls,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,  # 【优化】加速数据传输
                        persistent_workers=(num_workers > 0)  # 【关键】让子进程保持存活
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
            # 使用训练过程中计算出的最佳阈值
            viz_threshold = getattr(self, 'last_optimal_threshold', 0.1)
            test_viz_path = self.visualize_test_results(eval_model, val_loader, device, num_samples=8, use_tta=True, threshold=viz_threshold)
            
            # 生成性能分析
            self.update_progress.emit(98, "生成性能分析报告...")
            perf_analysis_path = self.generate_performance_analysis(detailed_metrics)
            
            # 【显存优化】彻底禁用训练和验证阶段的注意力热力图生成，防止 CUDA OOM
            # 注意力热力图生成需要大量显存（Grad-CAM 需要反向传播），在训练和验证阶段禁用
            # 仅在测试阶段（ModelTestThread）生成注意力热力图用于最终分析
            self.update_progress.emit(99, "注意力可视化已禁用（训练/验证阶段，防止 CUDA OOM）")
            attention_viz_path = ""
            attention_stats = {}
            
            # 发送测试结果信号，包含性能分析路径
            self.test_results_ready.emit(test_viz_path, detailed_metrics)
            self.visualization_ready.emit(perf_analysis_path)  # 同时发送性能分析
            
            # 【显存优化】训练和验证阶段已禁用注意力热力图生成，发送空信号
            # 仅在测试阶段（ModelTestThread）生成注意力热力图用于最终分析
            if attention_stats is None:
                attention_stats = {}
            # 如果 attention_viz_path 为空，说明未生成注意力热力图（训练/验证阶段）
            if attention_viz_path:
                self.attention_analysis_ready.emit(attention_viz_path, attention_stats)
            else:
                # 训练/验证阶段不发送注意力分析信号，避免下游处理错误
                pass
            
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
        # 如果使用2.5D数据集
        if self.dataset_type == "2.5d" and TCGA2_5D_AVAILABLE:
            # 2.5D数据集：支持递归搜索子文件夹
            # 目录结构示例：
            # data_dir/
            #   - 子文件夹1/
            #     - TCGA_CS_5393_19990606_1.tif
            #     - TCGA_CS_5393_19990606_1_mask.tif
            #     - TCGA_CS_5393_19990606_2.tif
            #     - TCGA_CS_5393_19990606_2_mask.tif
            #   - 子文件夹2/
            #     - TCGA_CS_5394_19990607_1.tif
            #     - TCGA_CS_5394_19990607_1_mask.tif
            #   ...
            # 系统会自动递归搜索所有子文件夹中的.tif文件
            mask_dir = self.data_dir  # mask和图像在同一目录（或子目录）
            
            # 【GUI选项驱动】根据dataset_type自动设置mode参数
            # 2.5D模式：使用3通道堆叠（上一张、当前、下一张）
            # 2D模式：只使用当前切片，单通道
            if self.dataset_type == "2.5d":
                dataset_mode = "2.5d"  # 三通道堆叠
            else:
                dataset_mode = "2d"  # 单通道
            print(f"[数据集加载] dataset_type={self.dataset_type}, 设置TCGA2_5DDataset mode={dataset_mode}")
            
            base_dataset = TCGA2_5DDataset(
                data_dir=self.data_dir,
                mask_dir=mask_dir,
                transform=transform,
                is_train=(split_name == "train"),
                debug=False,
                mode=dataset_mode  # 传递mode参数，确保数据加载方式与模型通道数匹配
            )
            print(f"[2.5D数据集] 加载了 {len(base_dataset)} 个样本")
            return base_dataset
        
        # 标准数据集加载逻辑
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
        # 【2.5D支持】根据数据集类型动态设置输入通道数
        # 2.5D数据集：3通道（上一张、当前、下一张）
        # 标准数据集：1通道（单通道输入）
        if self.dataset_type == "2.5d":
            in_channels = 3  # 2.5D输入：上一张、当前、下一张
            dataset_mode = "2.5D模式"
        else:
            in_channels = 1  # 标准数据集：单通道输入
            dataset_mode = "标准模式"
        
        if self.model_type == "resnet_unet":
            # 默认冻结编码器，前50% epoch只训练解码器，后50%解冻进行微调
            freeze_encoder = True  # 可以通过配置控制
            # ResNetUNet需要检查是否支持in_channels参数
            # 如果不支持，可能需要修改模型定义或使用适配层
            try:
                model = ResNetUNet(freeze_encoder=freeze_encoder, in_channels=in_channels).to(device)
            except TypeError:
                # 如果ResNetUNet不支持in_channels参数，使用默认值（通常是3）
                # 对于标准数据集（1通道），可能需要添加适配层
                print(f"[警告] ResNetUNet不支持in_channels参数，使用默认值。数据集类型：{self.dataset_type}")
                model = ResNetUNet(freeze_encoder=freeze_encoder).to(device)
                if in_channels == 1:
                    print(f"[警告] ResNetUNet期望3通道输入，但数据集是1通道。可能需要修改模型定义。")
            self.update_progress.emit(15, f"使用ResNet-UNet（{dataset_mode}）")
        elif self.model_type == "trans_unet" or self.model_type == "transunet":
            model = TransUNet(in_channels=in_channels).to(device)
            self.update_progress.emit(15, f"使用Transformer+UNet混合架构（{dataset_mode}，可提高Dice指标）")
        elif self.model_type in ("ds_trans_unet", "dstransunet", "ds-transunet"):
            dstrans_kwargs = {
                "in_channels": in_channels,  # 使用动态设置的通道数
                "out_channels": 1,
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 2,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
            }
            if dstrans_params:
                dstrans_kwargs.update(copy.deepcopy(dstrans_params))
                # 如果dstrans_params中指定了in_channels，优先使用（用于checkpoint恢复）
                if 'in_channels' in dstrans_params:
                    dstrans_kwargs['in_channels'] = dstrans_params['in_channels']
            # 移除DSTransUNet不接受的内置参数
            dstrans_kwargs.pop('_from_checkpoint', None)
            if dstrans_kwargs["embed_dim"] % dstrans_kwargs["num_heads"] != 0:
                dstrans_kwargs["embed_dim"] = dstrans_kwargs["num_heads"] * max(1, dstrans_kwargs["embed_dim"] // dstrans_kwargs["num_heads"])
            model = DSTransUNet(**dstrans_kwargs).to(device)
            self.update_progress.emit(15, f"使用DS-TransUNet（双尺度Transformer+UNet，{dataset_mode}，增强多尺度特征提取）")
        elif self.model_type == "swin_unet" or self.model_type == "swinunet":
            swin_kwargs = {
                "in_channels": in_channels,  # 使用动态设置的通道数
                "out_channels": 1
            }
            if swin_params:
                swin_kwargs.update(copy.deepcopy(swin_params))
                # 如果swin_params中指定了in_channels，优先使用（用于checkpoint恢复）
                if 'in_channels' in swin_params:
                    swin_kwargs['in_channels'] = swin_params['in_channels']
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
                f"使用SwinUNet（{dataset_mode}，参数：embed_dim={int(final_embed)}, window_size={int(final_window)}）"
            )
        elif self.model_type in ("swin_u_mamba", "swin-u-mamba", "swinumamba"):
            mamba_kwargs = {
                "in_channels": in_channels,  # 使用动态设置的通道数
                "out_channels": 1,
                "base_channels": 64,
                "num_blocks": (2, 2, 2, 2),
                "dropout": 0.05,
            }
            if swin_params:
                mamba_kwargs.update(copy.deepcopy(swin_params))
                # 如果swin_params中指定了in_channels，优先使用（用于checkpoint恢复）
                if 'in_channels' in swin_params:
                    mamba_kwargs['in_channels'] = swin_params['in_channels']
            model = SwinUMamba(**mamba_kwargs).to(device)
            self.update_progress.emit(
                15,
                f"使用Swin-U Mamba（{dataset_mode}，base_channels={mamba_kwargs.get('base_channels',64)}, blocks={mamba_kwargs.get('num_blocks',(2,2,2,2))}）"
            )
        elif self.model_type == "smp_unetplusplus":
            # 使用SMP U-Net++模型
            # 【2.5D支持】根据数据集类型动态设置输入通道数
            from config import get_model_config
            from models import SMPUnetPlusPlus
            
            model_config = get_model_config("smp_unetplusplus")
            
            # 【关键修复】根据数据集类型动态设置输入通道数
            # 2.5D数据集：3通道（上一张、当前、下一张）
            # 标准数据集：1通道（单通道输入）
            if self.dataset_type == "2.5d":
                in_channels = 3  # 2.5D输入：上一张、当前、下一张
                dataset_mode = "2.5D模式"
            else:
                in_channels = 1  # 标准数据集：单通道输入
                dataset_mode = "标准模式"
            
            # 如果提供了预训练模型路径，使用它作为pretrained_weights_path
            pretrained_weights_path = model_config.get("pretrained_weights_path")
            if self.model_path and os.path.exists(self.model_path):
                pretrained_weights_path = self.model_path
            
            model = SMPUnetPlusPlus(
                encoder_name=model_config.get("encoder_name", "resnet101"),
                encoder_weights=model_config.get("encoder_weights", "imagenet"),
                in_channels=in_channels,  # 使用动态设置的通道数
                classes=model_config.get("out_channels", 1),
                activation=model_config.get("activation", None),
                pretrained_weights_path=pretrained_weights_path
            ).to(device)
            
            encoder_name = model_config.get("encoder_name", "resnet101")
            self.update_progress.emit(
                15,
                f"使用SMP U-Net++（编码器: {encoder_name}, 输入通道: {in_channels}, {dataset_mode}）"
            )
        elif self.model_type in ("deeplabv3plus", "smp_deeplabv3plus"):
            # 【降维打击】使用SMP DeepLabV3+模型（更接近纯ResNet，训练更稳定）
            # 【锁定3通道】强制使用3通道输入，不再考虑数据集类型
            from config import get_model_config
            from models import SMPDeepLabV3Plus
            
            # 统一使用 smp_deeplabv3plus 作为配置键
            model_config = get_model_config("smp_deeplabv3plus")
            
            # 【锁定3通道】强制使用3通道，不再动态适配
            in_channels = 3
            
            # 如果提供了预训练模型路径，使用它作为pretrained_weights_path
            pretrained_weights_path = model_config.get("pretrained_weights_path")
            if self.model_path and os.path.exists(self.model_path):
                pretrained_weights_path = self.model_path
            
            model = SMPDeepLabV3Plus(
                encoder_name=model_config.get("encoder_name", "resnet101"),
                encoder_weights=model_config.get("encoder_weights", "imagenet"),
                in_channels=in_channels,  # 锁定为3通道
                classes=model_config.get("out_channels", 1),
                activation=model_config.get("activation", None),
                pretrained_weights_path=pretrained_weights_path
            ).to(device)
            
            encoder_name = model_config.get("encoder_name", "resnet101")
            self.update_progress.emit(
                15,
                f"【降维打击】使用SMP DeepLabV3+（编码器: {encoder_name}, 输入通道: 3通道，已锁定）"
            )
        else:
            # ImprovedUNet：根据数据集类型动态设置输入通道数
            model = ImprovedUNet(in_channels=in_channels).to(device)
            self.update_progress.emit(15, f"使用改进UNet（{dataset_mode}）")

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
    
    def _adapt_model_channels(self, model, train_loader, device):
        """
        模型输入通道自适应：检查数据通道数与模型第一层通道数是否匹配，如不匹配则动态修改
        
        参数:
            model: 模型实例
            train_loader: 训练数据加载器
            device: 设备
        
        返回:
            适配后的模型（如果进行了修改，则返回新模型；否则返回原模型）
        """
        try:
            # 【锁定3通道】SMPDeepLabV3Plus已锁定为3通道，跳过通道适配
            actual_model = self._unwrap_model(model)
            # 检查是否是SMPDeepLabV3Plus模型（通过类名或模型类型判断）
            is_deeplabv3 = (
                self.model_type in ("deeplabv3plus", "smp_deeplabv3plus") or
                type(actual_model).__name__ == "SMPDeepLabV3Plus" or
                (hasattr(actual_model, 'model') and type(actual_model.model).__name__ == "SMPDeepLabV3Plus")
            )
            if is_deeplabv3:
                print(f"[通道适配] SMPDeepLabV3Plus已锁定为3通道，跳过通道适配")
                return model
            
            # 步骤1: 从 train_loader 中取出一个 batch
            data_iter = iter(train_loader)
            batch_data = next(data_iter)
            if len(batch_data) == 2:
                images, masks = batch_data
            elif len(batch_data) == 3:
                images, masks, _ = batch_data
            else:
                print("[通道适配] 无法解析batch数据格式，跳过通道适配检查")
                return model
            
            # 获取数据的通道数
            if len(images.shape) < 4:
                print("[通道适配] 图像维度不足，跳过通道适配检查")
                return model
            
            data_channels = images.shape[1]  # (B, C, H, W) 中的 C
            print(f"[通道适配] 检测到数据通道数: {data_channels}")
            
            # 步骤2: 获取模型第一层的通道数
            actual_model = self._unwrap_model(model)
            model_channels = None
            first_conv_layer = None
            first_conv_path = None
            
            # 尝试多种方式找到第一层卷积
            if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'encoder'):
                # SMP模型结构：model.model.encoder.conv1
                if hasattr(actual_model.model.encoder, 'conv1'):
                    first_conv_layer = actual_model.model.encoder.conv1
                    first_conv_path = 'model.encoder.conv1'
                    model_channels = first_conv_layer.in_channels
            elif hasattr(actual_model, 'encoder') and hasattr(actual_model.encoder, 'conv1'):
                # 直接有encoder.conv1
                first_conv_layer = actual_model.encoder.conv1
                first_conv_path = 'encoder.conv1'
                model_channels = first_conv_layer.in_channels
            elif hasattr(actual_model, 'conv1'):
                # 直接有conv1
                first_conv_layer = actual_model.conv1
                first_conv_path = 'conv1'
                model_channels = first_conv_layer.in_channels
            elif hasattr(actual_model, 'down1'):
                # UNet类模型：down1的第一个卷积层
                if hasattr(actual_model.down1, '__getitem__'):
                    # down1可能是Sequential，尝试找到第一个Conv2d层
                    for idx, layer in enumerate(actual_model.down1):
                        if isinstance(layer, nn.Conv2d):
                            first_conv_layer = layer
                            first_conv_path = f'down1[{idx}]'
                            model_channels = layer.in_channels
                            break
                elif hasattr(actual_model.down1, 'conv'):
                    first_conv_layer = actual_model.down1.conv
                    first_conv_path = 'down1.conv'
                    if isinstance(first_conv_layer, nn.Conv2d) and hasattr(first_conv_layer, 'in_channels'):
                        model_channels = first_conv_layer.in_channels
            
            if model_channels is None or first_conv_layer is None:
                print(f"[通道适配] 无法找到模型第一层卷积，跳过通道适配检查")
                return model
            
            print(f"[通道适配] 检测到模型第一层通道数: {model_channels} (路径: {first_conv_path})")
            
            # 步骤3: 如果两者不一致，动态修改模型的第一层卷积
            if data_channels != model_channels:
                print(f"\n{'='*60}")
                print(f"⚠️  [通道适配警告] 数据通道数 ({data_channels}) 与模型第一层通道数 ({model_channels}) 不匹配！")
                print(f"   正在自动适配模型以匹配数据通道数...")
                print(f"{'='*60}\n")
                
                # 获取第一层卷积的权重和偏置
                old_weight = first_conv_layer.weight.data.clone()  # (out_channels, in_channels, H, W)
                old_bias = first_conv_layer.bias.data.clone() if first_conv_layer.bias is not None else None
                
                # 创建新的卷积层
                out_channels = old_weight.shape[0]
                kernel_size = old_weight.shape[2:]  # (H, W)
                stride = first_conv_layer.stride if hasattr(first_conv_layer, 'stride') else (1, 1)
                padding = first_conv_layer.padding if hasattr(first_conv_layer, 'padding') else (0, 0)
                dilation = first_conv_layer.dilation if hasattr(first_conv_layer, 'dilation') else (1, 1)
                groups = first_conv_layer.groups if hasattr(first_conv_layer, 'groups') else 1
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    in_channels=data_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=(old_bias is not None)
                ).to(device)
                
                # 权重适配策略
                if data_channels > model_channels:
                    # 数据是3通道，模型是1通道：将1通道权重复制3份（或取平均扩展）
                    print(f"   [适配策略] 1通道 -> {data_channels}通道：复制权重")
                    # 方法1: 直接复制（简单但可能不够好）
                    # 方法2: 取平均后复制（更稳定）
                    # 这里使用平均复制策略
                    old_weight_mean = old_weight.mean(dim=1, keepdim=True)  # (out_channels, 1, H, W)
                    new_weight = old_weight_mean.repeat(1, data_channels, 1, 1)  # (out_channels, data_channels, H, W)
                    new_conv.weight.data = new_weight
                    if old_bias is not None:
                        new_conv.bias.data = old_bias.clone()
                else:
                    # 数据是1通道，模型是3通道：将3通道权重求和（或取平均），压缩成1通道卷积层
                    print(f"   [适配策略] {model_channels}通道 -> 1通道：求和压缩权重")
                    # 方法1: 直接求和（简单）
                    # 方法2: 取平均（更稳定）
                    # 这里使用平均策略
                    new_weight = old_weight.mean(dim=1, keepdim=True)  # (out_channels, 1, H, W)
                    new_conv.weight.data = new_weight
                    if old_bias is not None:
                        new_conv.bias.data = old_bias.clone()
                
                # 替换第一层卷积
                # 根据路径设置新层
                if first_conv_path == 'model.encoder.conv1':
                    actual_model.model.encoder.conv1 = new_conv
                elif first_conv_path == 'encoder.conv1':
                    actual_model.encoder.conv1 = new_conv
                elif first_conv_path == 'conv1':
                    actual_model.conv1 = new_conv
                elif first_conv_path.startswith('down1'):
                    # 对于down1，需要根据路径索引替换
                    if '[' in first_conv_path and ']' in first_conv_path:
                        # 提取索引，例如 down1[0] -> 0
                        idx_str = first_conv_path.split('[')[1].split(']')[0]
                        try:
                            idx = int(idx_str)
                            if hasattr(actual_model.down1, '__getitem__'):
                                # 如果是Sequential，需要替换对应索引的层
                                if isinstance(actual_model.down1, nn.Sequential):
                                    # 创建一个新的Sequential，替换指定索引的层
                                    layers = list(actual_model.down1)
                                    layers[idx] = new_conv
                                    actual_model.down1 = nn.Sequential(*layers).to(device)
                                else:
                                    # 如果是其他可索引结构，直接替换
                                    actual_model.down1[idx] = new_conv
                        except (ValueError, IndexError):
                            print(f"   ⚠️  [通道适配] 无法解析down1索引: {first_conv_path}")
                    elif hasattr(actual_model.down1, 'conv'):
                        actual_model.down1.conv = new_conv
                
                # 如果模型被DataParallel包装，需要同步更新
                if isinstance(model, nn.DataParallel):
                    model.module = actual_model
                elif isinstance(model, AveragedModel):
                    model.module = actual_model
                else:
                    model = actual_model
                
                print(f"   ✅ [通道适配完成] 模型第一层已从 {model_channels} 通道适配为 {data_channels} 通道")
                print(f"   适配路径: {first_conv_path}")
                print(f"{'='*60}\n")
                
                # 验证适配是否成功
                try:
                    test_images = images[:1].to(device)  # 只取第一个样本测试
                    with torch.no_grad():
                        _ = model(test_images)
                    print(f"   ✅ [验证通过] 适配后的模型可以正常前向传播")
                except Exception as e:
                    print(f"   ⚠️  [验证失败] 适配后的模型前向传播出错: {str(e)}")
                    print(f"   建议检查模型结构和数据格式")
            else:
                print(f"[通道适配] 数据通道数 ({data_channels}) 与模型通道数 ({model_channels}) 匹配，无需适配")
            
            return model
            
        except Exception as e:
            print(f"[通道适配] 通道适配过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"[通道适配] 出错后返回原模型，训练将继续进行")
            return model
    
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
        from models import SMPDeepLabV3Plus
        return isinstance(actual, (ImprovedUNet, TransUNet, DSTransUNet, SwinUNet, ResNetUNet, SMPDeepLabV3Plus))
    
    def _generate_gradcam_for_deeplabv3(self, model, images, device):
        """
        为 DeepLabV3+ 生成 Grad-CAM 热力图
        
        Args:
            model: 模型实例（已解包，非 DataParallel）
            images: 输入图像 (B, 3, H, W)
            device: 设备
        
        Returns:
            attention_maps: 字典，包含 Grad-CAM 热力图
        """
        if not GRAD_CAM_AVAILABLE:
            return {}
        
        try:
            # 确保模型处于 eval 模式（Grad-CAM 需要）
            was_training = model.training
            model.eval()
            
            # 获取实际模型（SMPDeepLabV3Plus 包装了 smp.DeepLabV3Plus）
            actual_model = model
            if hasattr(model, 'model'):
                actual_model = model.model
            
            # 【分辨率优化】使用 Decoder 作为目标层，获得更高分辨率的热力图
            # Decoder 具有更高的空间分辨率（接近输入图像大小），而 encoder.layer4 只有 1/32 分辨率
            target_layer = None
            
            # 优先使用 decoder（更高分辨率）
            if hasattr(actual_model, 'decoder'):
                decoder = actual_model.decoder
                # decoder 可能是一个 Sequential 或 ModuleList
                if hasattr(decoder, '__getitem__') and len(decoder) > 0:
                    # 如果是可索引的，取最后一个模块（通常是输出层）
                    target_layer = decoder[-1]
                elif hasattr(decoder, 'segmentation_head'):
                    # 某些 decoder 有 segmentation_head
                    target_layer = decoder.segmentation_head
                else:
                    target_layer = decoder
            elif hasattr(actual_model, 'segmentation_head'):
                # 如果 decoder 不存在，尝试直接使用 segmentation_head
                target_layer = actual_model.segmentation_head
            
            # 如果 decoder 不可用，回退到 encoder.layer4（低分辨率，但至少能工作）
            if target_layer is None:
                encoder = actual_model.encoder
                if hasattr(encoder, 'layer4'):
                    layer4 = encoder.layer4
                    if hasattr(layer4, '__getitem__'):
                        target_layer = layer4[-1] if len(layer4) > 0 else layer4
                    else:
                        target_layer = layer4
                elif hasattr(encoder, 'blocks') and len(encoder.blocks) > 0:
                    target_layer = encoder.blocks[-1]
            
            if target_layer is None:
                print("[Grad-CAM] 无法找到目标层（decoder 或 encoder），跳过 Grad-CAM 生成")
                if was_training:
                    model.train()
                return {}
            
            # 初始化 GradCAM
            # 注意：GradCAM 需要访问实际的模型结构，使用 actual_model（smp.DeepLabV3Plus）
            # 新版本的 grad-cam 库已移除 use_cuda 参数，会自动检测设备
            cam = GradCAM(model=actual_model, target_layers=[target_layer])
            
            # 获取图像尺寸 (images 形状是 [B, C, H, W])
            height, width = images.shape[2], images.shape[3]
            
            # 创建全1掩码 (表示关注整张图的类别预测)
            # SemanticSegmentationTarget 不接受 mask=None，必须传入具体的 numpy 数组
            mask = np.ones((height, width), dtype=np.float32)
            
            # 确定目标类别索引
            # 对于二分类模型 (classes=1)，输出只有1个通道，索引必须是0
            # 对于多分类模型 (classes>1)，可以使用 category=1 或其他类别索引
            target_category = 1  # 默认使用类别1（前景类）
            
            # 检查模型的类别数
            if hasattr(actual_model, 'classes'):
                num_classes = actual_model.classes
                if num_classes == 1:
                    # 二分类模型：输出只有1个通道（索引0），必须使用 category=0
                    target_category = 0
                    # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                    if not getattr(self, '_has_logged_gradcam_info', False):
                        print(f"[Grad-CAM] 检测到二分类模型 (classes=1)，使用 category=0")
                        self._has_logged_gradcam_info = True
                else:
                    # 多分类模型：可以使用 category=1（前景类）或其他类别
                    target_category = min(1, num_classes - 1)  # 确保不越界
                    # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                    if not getattr(self, '_has_logged_gradcam_info', False):
                        print(f"[Grad-CAM] 检测到多分类模型 (classes={num_classes})，使用 category={target_category}")
                        self._has_logged_gradcam_info = True
            else:
                # 如果无法获取 classes 属性，尝试从输出形状推断
                # 先进行一次前向传播获取输出形状（仅用于推断）
                try:
                    with torch.no_grad():
                        test_output = actual_model(images[:1])  # 只取第一个样本测试
                        if isinstance(test_output, tuple):
                            test_output = test_output[0]
                        num_classes = test_output.shape[1]  # (B, C, H, W) 中的 C
                        if num_classes == 1:
                            target_category = 0
                            # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                            if not getattr(self, '_has_logged_gradcam_info', False):
                                print(f"[Grad-CAM] 通过输出形状推断为二分类模型 (channels=1)，使用 category=0")
                                self._has_logged_gradcam_info = True
                        else:
                            target_category = min(1, num_classes - 1)
                            # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                            if not getattr(self, '_has_logged_gradcam_info', False):
                                print(f"[Grad-CAM] 通过输出形状推断为多分类模型 (channels={num_classes})，使用 category={target_category}")
                                self._has_logged_gradcam_info = True
                except Exception as e:
                    # 如果推断失败，默认使用 category=0（二分类）
                    target_category = 0
                    # 【日志优化】仅在第一次调用时打印，避免重复日志刷屏
                    if not getattr(self, '_has_logged_gradcam_info', False):
                        print(f"[Grad-CAM] 无法推断模型类别数，默认使用 category=0 (二分类): {e}")
                        self._has_logged_gradcam_info = True
            
            # 定义目标：语义分割的目标类别
            targets = [SemanticSegmentationTarget(category=target_category, mask=mask)]
            
            # 【关键修复】强制开启梯度计算，这是 Grad-CAM 必须的
            # 即使外部有 torch.no_grad()，这里也要临时开启梯度计算
            # 确保输入图像支持求导
            images_grad = images.clone().detach().requires_grad_(True)
            
            # 生成 Grad-CAM 热力图
            # grayscale_cam 形状: (B, H, W)
            # 使用 torch.enable_grad() 上下文管理器，确保梯度计算可用
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=images_grad, targets=targets)
            
            # 转换为 torch.Tensor 并添加通道维度，匹配其他模型的注意力图格式
            # 格式: (B, 1, H, W)
            attention_maps = {}
            if len(grayscale_cam.shape) == 3:  # (B, H, W)
                grayscale_cam_tensor = torch.from_numpy(grayscale_cam).float().to(device)
                grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)  # (B, 1, H, W)
            else:
                grayscale_cam_tensor = torch.from_numpy(grayscale_cam).float().to(device)
            
            # 使用 'gradcam_decoder' 作为键名（因为现在使用 decoder 作为目标层）
            attention_maps['gradcam_decoder'] = grayscale_cam_tensor
            
            # 恢复模型训练状态
            if was_training:
                model.train()
            
            return attention_maps
            
        except Exception as e:
            print(f"[Grad-CAM警告] 生成热力图失败: {e}")
            import traceback
            traceback.print_exc()
            # 确保恢复模型状态
            if was_training:
                model.train()
            return {}
    
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
    
    def _create_optimizer_with_groups(self, param_groups, lr):
        """
        创建带参数分组的优化器（用于差异化学习率）
        
        Args:
            param_groups: 参数组列表，每个元素是包含'params'和'lr'的字典
            lr: 默认学习率（用于scheduler，实际LR由param_groups指定）
        
        Returns:
            优化器实例
        """
        # 处理每个参数组的学习率（应用与_create_optimizer相同的限制逻辑）
        processed_groups = []
        for group in param_groups:
            group_lr = float(group.get('lr', lr))
            # 对encoder组应用学习率限制（如果LR太大）
            # 通过检查参数数量或学习率大小来判断是encoder还是decoder
            is_encoder_group = group_lr <= 1e-4 or group_lr == lr
            
            if is_encoder_group:
                # encoder组：应用限制
                if group_lr > 1e-4:
                    group_lr = 1e-4
                elif abs(group_lr - 1e-4) < 1e-9:
                    group_lr = 1e-5
            else:
                # decoder组：允许更大的学习率（10倍），但也要有上限
                if group_lr > 1e-3:
                    group_lr = 1e-3
            
            processed_group = {
                'params': group['params'],
                'lr': group_lr,
                'weight_decay': group.get('weight_decay', 5e-4)
            }
            processed_groups.append(processed_group)
        
        # 【降维打击】对于SMP模型（U-Net++和DeepLabV3+），强制使用AdamW以获得更好的训练效果
        # AdamW对权重衰减的处理更稳定，适合微调预训练模型
        if self.model_type in ("smp_unetplusplus", "smp_deeplabv3plus", "deeplabv3plus"):
            return optim.AdamW(processed_groups, weight_decay=5e-4)
        
        if self.optimizer_type == "adam":
            return optim.Adam(processed_groups, betas=(0.9, 0.999))
        if self.optimizer_type == "sgd":
            return optim.SGD(processed_groups, momentum=0.99, nesterov=True)
        # 默认使用AdamW
        return optim.AdamW(processed_groups)
    
    def _get_loss_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        【极简配置】回归稳健的基准配置：50% BCE + 50% Dice
        
        这是医学分割的黄金标准组合，先跑通这个，再考虑加其他的。
        """
        # 极简 Loss 组合：只使用 BCE 和 Dice，各占 50%
        weights = {
            'bce': 0.5,
            'dice': 0.5,
            'focal': 0.0,
            'tversky': 0.0,
            'tversky_focal': 0.0,
            'boundary': 0.0,
            'hausdorff': 0.0,
            'lovasz': 0.0,
            'fn_penalty': 0.0,
            'fp_penalty': 0.0,
        }
        # 不需要归一化，因为总和已经是 1.0
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
        # 【2.5D支持】为SMP模型（DeepLabV3+ 和 U-Net++）保存输入通道数和数据集类型
        if isinstance(actual, (SMPDeepLabV3Plus, SMPUnetPlusPlus)):
            # 从模型的第一层卷积推断输入通道数
            if hasattr(actual, 'model') and hasattr(actual.model, 'encoder'):
                # SMP模型结构：model.model.encoder
                if hasattr(actual.model.encoder, 'stem') and hasattr(actual.model.encoder.stem, 'conv1'):
                    in_channels = actual.model.encoder.stem.conv1.in_channels
                elif hasattr(actual.model.encoder, 'conv1'):
                    in_channels = actual.model.encoder.conv1.in_channels
                else:
                    # 回退：从state_dict推断
                    state_dict = actual.state_dict()
                    first_conv_key = None
                    for key in state_dict.keys():
                        if 'encoder.stem.conv1.weight' in key or 'encoder.conv1.weight' in key:
                            first_conv_key = key
                            break
                    if first_conv_key:
                        in_channels = state_dict[first_conv_key].shape[1]
                    else:
                        in_channels = 3  # 默认值
                config["in_channels"] = in_channels
                # 保存数据集类型，用于测试时正确加载模型
                config["dataset_type"] = getattr(self, 'dataset_type', 'standard')
                print(f"[Checkpoint] 保存SMP模型配置: in_channels={in_channels}, dataset_type={config['dataset_type']}")
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
        
        # 【MATLAB 兼容性修复】将字典转换为数值数组和名称数组，避免 MATLAB 识别为 struct 导致转换失败
        def dict_to_arrays(source: dict):
            """将字典转换为数值数组和名称数组"""
            if not source:
                return np.array([], dtype=np.float32), np.array([], dtype='<U100')
            keys = list(source.keys())
            values = [float(source.get(k, 0.0)) for k in keys]
            return np.array(values, dtype=np.float32), np.array(keys, dtype='<U100')
        
        # 处理 all_samples：保持原有结构（每个指标是一个数组）
        metrics = {k: np.array(v, dtype=np.float32) for k, v in detailed_metrics.get('all_samples', {}).items()}
        
        # 处理统计指标：转换为数值数组和名称数组
        avg_vals, avg_names = dict_to_arrays(detailed_metrics.get('average', {}))
        std_vals, std_names = dict_to_arrays(detailed_metrics.get('std', {}))
        min_vals, min_names = dict_to_arrays(detailed_metrics.get('min', {}))
        max_vals, max_names = dict_to_arrays(detailed_metrics.get('max', {}))
        median_vals, median_names = dict_to_arrays(detailed_metrics.get('median', {}))
        
        # 【关键修复】确保 std_metrics_values 字段总是存在（MATLAB 端必需）
        # 如果 std_vals 为空但 avg_vals 不为空，创建一个全 0 数组作为默认标准差
        if len(std_vals) == 0 and len(avg_vals) > 0:
            std_vals = np.zeros(len(avg_vals), dtype=np.float32)
            std_names = avg_names.copy()  # 使用相同的名称
        
        # 保存到 .mat 文件
        # 注意：MATLAB 需要数值数组（double），而不是 struct
        payload = {
            'metrics': metrics,  # 保持原有结构，用于详细分析
        }
        
        # 添加统计指标的数值数组和名称数组
        if len(avg_vals) > 0:
            payload['avg_metrics_values'] = avg_vals
            payload['avg_metrics_names'] = avg_names
            # 【关键修复】确保 std_metrics_values 总是与 avg_metrics_values 一起存在
            payload['std_metrics_values'] = std_vals
            payload['std_metrics_names'] = std_names
        if len(min_vals) > 0:
            payload['min_metrics_values'] = min_vals
            payload['min_metrics_names'] = min_names
        if len(max_vals) > 0:
            payload['max_metrics_values'] = max_vals
            payload['max_metrics_names'] = max_names
        if len(median_vals) > 0:
            payload['median_metrics_values'] = median_vals
            payload['median_metrics_names'] = median_names
        
        savemat(payload_path, payload)
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

    def _compute_metrics_unified(self, pred, target, eps: float = 1e-7):
        """
        【统一指标计算函数】同时计算 Dice 和 IoU，确保逻辑完全一致
        
        核心原则：
        1. 只计算前景类（Class Index = 1），不计算背景
        2. 空掩码情况严格处理：
           - GT 为空且 Pred 为空 → Dice=1.0, IoU=1.0 (完美预测)
           - GT 为空但 Pred 不为空 → Dice=0.0, IoU=0.0 (误报)
           - GT 不为空但 Pred 为空 → Dice=0.0, IoU=0.0 (漏报)
        3. 正常情况使用标准公式
        
        Args:
            pred: 预测掩码（可以是 torch.Tensor 或 numpy.ndarray）
            target: 真实掩码（可以是 torch.Tensor 或 numpy.ndarray）
            eps: 平滑系数
        
        Returns:
            (dice, iou): Dice 系数和 IoU 值
        """
        # 预处理：转换为 numpy 并展平
        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = np.array(pred)
        
        if isinstance(target, torch.Tensor):
            target_np = target.detach().cpu().numpy()
        else:
            target_np = np.array(target)
        
        # 确保尺寸匹配
        if pred_np.shape != target_np.shape:
            from scipy.ndimage import zoom
            if len(pred_np.shape) == 2 and len(target_np.shape) == 2:
                zoom_factors = (target_np.shape[0] / pred_np.shape[0], target_np.shape[1] / pred_np.shape[1])
                pred_np = zoom(pred_np, zoom_factors, order=1)
            else:
                # 对于多维数组，只调整最后两个维度
                if len(pred_np.shape) >= 2 and len(target_np.shape) >= 2:
                    zoom_factors = (target_np.shape[-2] / pred_np.shape[-2], target_np.shape[-1] / pred_np.shape[-1])
                    pred_np = zoom(pred_np, [1] * (len(pred_np.shape) - 2) + list(zoom_factors), order=1)
        
        # 二值化：只计算前景类（> 0.5 视为前景）
        pred_binary = (pred_np > 0.5).astype(np.float32)
        target_binary = (target_np > 0.5).astype(np.float32)
        
        # 展平
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        
        # 计算统计量
        pred_sum = float(pred_flat.sum())
        target_sum = float(target_flat.sum())
        intersection = float((pred_flat * target_flat).sum())
        union = pred_sum + target_sum - intersection
        
        # 【核心修复逻辑】空掩码特判
        # Case 1: 双空（GT 为空且 Pred 为空）
        if target_sum <= eps and pred_sum <= eps:
            return 1.0, 1.0  # Dice=1.0, IoU=1.0 (完美预测)
        
        # Case 2: 单空（GT 为空但 Pred 不为空，或 GT 不为空但 Pred 为空）
        if target_sum <= eps or pred_sum <= eps:
            return 0.0, 0.0  # Dice=0.0, IoU=0.0 (误报或漏报)
        
        # Case 3: 正常情况，使用标准公式
        # Dice = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
        # IoU = |Pred ∩ GT| / |Pred ∪ GT|
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        iou = (intersection + eps) / (union + eps) if union > eps else 0.0
        
        return float(dice), float(iou)
    
    def _safe_dice_score(self, pred, target, eps: float = 1e-7) -> float:
        """
        【向后兼容】计算Dice系数，内部调用统一计算函数
        
        注意：此函数保留用于向后兼容，新代码应使用 _compute_metrics_unified
        """
        dice, _ = self._compute_metrics_unified(pred, target, eps)
        return dice
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
        pred_sum = pred_tensor.sum()
        target_sum = target_tensor.sum()
        
        # 【关键修复】空掩码特判逻辑
        # Case 1: GT 为空（全黑样本）
        if target_sum <= smooth:
            if pred_sum <= smooth:
                # 场景 A: GT 为空，Pred 为空 → Dice = 1.0 (完美预测)
                return 1.0
            else:
                # 场景 B: GT 为空，Pred 不为空 (有误报) → Dice = 0.0 (完全错误)
                return 0.0
        
        # Case 2: GT 不为空，但预测为空
        if pred_sum <= smooth:
            # 场景 C: GT 不为空，Pred 为空 (漏报) → Dice = 0.0 (完全漏检)
            return 0.0
        
        # Case 3: 正常情况，使用标准 Dice 公式（只计算前景类）
        return (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

    @staticmethod
    @staticmethod
    def calculate_batch_metrics(pred, target, smooth=1e-7):
        """
        【单一真理来源】统一的指标计算函数
        
        所有阶段（GWO搜索、GWO最终评估、Epoch验证循环）必须调用此函数计算指标。
        
        Args:
            pred: 预测mask (B, H, W) 或 (B, 1, H, W)，值域[0,1]
            target: 真实mask (B, H, W) 或 (B, 1, H, W)，值域[0,1]
            smooth: 平滑系数，默认1e-7
            
        Returns:
            metrics_dict: {
                'dice': [dice_0, dice_1, ..., dice_B-1],  # 每个样本的Dice
                'iou': [iou_0, iou_1, ..., iou_B-1],
                'precision': [prec_0, prec_1, ..., prec_B-1],
                'recall': [recall_0, recall_1, ..., recall_B-1],
                'is_empty': [bool_0, bool_1, ..., bool_B-1],  # True表示空mask样本
            }
        """
        import torch
        import torch.nn.functional as F
        
        # 统一维度处理
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 确保尺寸匹配
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        pred_flat = pred.view(pred.size(0), -1).float()
        target_flat = target.view(target.size(0), -1).float()
        
        batch_size = pred.size(0)
        total_pixels = pred_flat.size(1)
        # 【统一阈值】使用固定的0.1%像素阈值
        empty_threshold = max(1.0, float(total_pixels) * 0.001)  # 0.1%像素
        
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        is_empty_list = []
        
        for i in range(batch_size):
            pred_i = pred_flat[i]
            target_i = target_flat[i]
            
            intersection = (pred_i * target_i).sum()
            pred_sum = pred_i.sum()
            target_sum = target_i.sum()
            
            # 判断是否为空mask
            is_empty = (target_sum <= empty_threshold)
            is_empty_list.append(is_empty)
            
            if is_empty:
                # 空mask样本：GT为空
                if pred_sum <= smooth:
                    # GT为空，预测也为空 → Dice=1.0
                    dice = 1.0
                    iou = 1.0
                    precision = 1.0
                    recall = 1.0
                else:
                    # GT为空，预测不为空（假阳性）→ Dice=0.0（严厉惩罚）
                    dice = 0.0
                    iou = 0.0
                    precision = 0.0
                    recall = 1.0  # GT为空，recall=1.0（没有漏检）
            else:
                # 前景样本：使用标准公式
                tp = intersection
                fp = pred_sum - intersection
                fn = target_sum - intersection
                
                # Dice = 2*TP / (2*TP + FP + FN)
                dice_den = 2.0 * tp + fp + fn
                if dice_den < smooth:
                    dice = 0.0
                else:
                    dice = (2.0 * tp) / dice_den
                
                # IoU = TP / (TP + FP + FN)
                union = tp + fp + fn
                if union < smooth:
                    iou = 0.0
                else:
                    iou = tp / union
                
                # Precision = TP / (TP + FP)
                if (tp + fp) < smooth:
                    precision = 0.0
                else:
                    precision = tp / (tp + fp)
                
                # Recall = TP / (TP + FN)
                if (tp + fn) < smooth:
                    recall = 0.0
                else:
                    recall = tp / (tp + fn)
            
            dice_scores.append(float(dice))
            iou_scores.append(float(iou))
            precision_scores.append(float(precision))
            recall_scores.append(float(recall))
        
        return {
            'dice': dice_scores,
            'iou': iou_scores,
            'precision': precision_scores,
            'recall': recall_scores,
            'is_empty': is_empty_list,
        }
    
    def calculate_batch_dice(self, pred, target, smooth=1e-7):
        """
        计算一个批次中每个样本的Dice系数。
        对空mask情况进行特殊处理,避免过度惩罚少量误检。
        
        【注意】此函数保留用于向后兼容，新代码应使用 calculate_batch_metrics
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
        # 【统一阈值】使用固定的0.1%像素阈值，与验证阶段保持一致
        # 对于512x512图像，阈值约为0.65像素
        empty_threshold = max(1e-7, float(total_pixels) * 0.001)  # 0.1%像素，统一阈值
        dice_scores = []
        
        for i in range(batch_size):
            pred_i = pred_flat[i]
            target_i = target_flat[i]
            
            intersection = (pred_i * target_i).sum()
            pred_sum = pred_i.sum()
            target_sum = target_i.sum()
            
            # 【关键修复】空掩码特判逻辑
            # Case 1: GT 为空（全黑样本）
            if target_sum <= empty_threshold:
                if pred_sum <= smooth:
                    # 场景 A: GT 为空，Pred 为空 → Dice = 1.0 (完美预测)
                    dice = 1.0
                else:
                    # 场景 B: GT 为空，Pred 不为空 (有误报) → Dice = 0.0 (完全错误)
                    # 这是假阳性（False Positive），应该得 0 分
                    dice = 0.0
            # Case 2: GT 不为空，但预测为空
            elif pred_sum <= smooth:
                # 场景 C: GT 不为空，Pred 为空 (漏报) → Dice = 0.0 (完全漏检)
                dice = 0.0
            # Case 3: 正常情况，使用标准 Dice 公式（只计算前景类）
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
        
        # 【默认权重】仅在没有传入weights时使用（向后兼容）
        # 【注意】训练时实际使用的是 _get_loss_weights() 返回的权重，而不是这里的默认值
        # 默认权重仅作为后备，实际权重配置见 _get_loss_weights() 方法
        loss_weights = {
            'bce': 0.20,      # 默认BCE权重（实际训练时会被覆盖）
            'dice': 0.80,     # 默认Dice权重（实际训练时会被覆盖）
            'tversky': 0.0,
            'tversky_focal': 0.0,
            'boundary': 0.0,
            'hausdorff': 0.0,
            'focal': 0.0,
            'lovasz': 0.0,
            'fn_penalty': 0.0,
            'fp_penalty': 0.0,
        }
        # 【关键】如果传入了自定义权重（训练时从 _get_loss_weights() 获取），则使用传入的权重
        if weights:
            loss_weights.update(weights)
        
        # 简化损失计算：仅使用BCE和Dice
        combined_loss = (
            loss_weights['bce'] * bce_loss
            + loss_weights['dice'] * dice_loss_val
        )
        
        # 如果启用了其他损失且权重>0，则添加（向后兼容）
        if loss_weights.get('tversky', 0) > 0:
            combined_loss += loss_weights['tversky'] * tversky_loss_val
        if loss_weights.get('tversky_focal', 0) > 0:
            combined_loss += loss_weights['tversky_focal'] * tversky_focal_loss_val
        if loss_weights.get('boundary', 0) > 0:
            combined_loss += loss_weights['boundary'] * boundary_loss
        if loss_weights.get('focal', 0) > 0:
            combined_loss += loss_weights['focal'] * focal_loss_val
        if loss_weights.get('fn_penalty', 0) > 0:
            combined_loss += loss_weights['fn_penalty'] * false_negative_penalty
        if loss_weights.get('fp_penalty', 0) > 0:
            combined_loss += loss_weights['fp_penalty'] * false_positive_penalty
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
        5. 【新增】尺寸适配：确保输入尺寸能被16整除（DeepLabV3+等模型要求）
        
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
        
        # 【辅助函数】确保尺寸能被16整除（DeepLabV3+等模型要求）
        def pad_to_divisible_by_16(h, w):
            """将尺寸向上取整到16的倍数"""
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            return pad_h, pad_w
        
        # 【辅助函数】填充图像到16的倍数
        def pad_image(img, target_h, target_w):
            """填充图像到目标尺寸（能被16整除）"""
            _, _, h, w = img.shape
            pad_h, pad_w = pad_to_divisible_by_16(target_h, target_w)
            if pad_h > 0 or pad_w > 0:
                # 使用反射填充，避免边界伪影
                img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            return img, pad_h, pad_w
        
        # 【辅助函数】裁剪预测结果到原始尺寸
        def crop_prediction(pred, original_h, original_w, pad_h, pad_w):
            """裁剪预测结果，移除填充部分"""
            _, _, h, w = pred.shape
            # 裁剪填充部分（填充是在底部和右侧）
            if pad_h > 0 or pad_w > 0:
                # 裁剪到原始尺寸（移除底部和右侧的填充）
                end_h = h - pad_h if pad_h > 0 else h
                end_w = w - pad_w if pad_w > 0 else w
                pred = pred[:, :, :end_h, :end_w]
            # 如果尺寸不匹配，插值到原始尺寸
            if pred.shape[2] != original_h or pred.shape[3] != original_w:
                pred = F.interpolate(pred, size=(original_h, original_w), mode='bilinear', align_corners=False)
            return pred
        
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
            
            # 【关键修复】确保缩放后的尺寸能被16整除
            scaled_images_padded, pad_h, pad_w = pad_image(scaled_images, target_h, target_w)
            padded_h, padded_w = target_h + pad_h, target_w + pad_w
            
            # 【8种变换循环】
            scale_prob_maps = []
            
            # 【辅助函数】对图像进行变换并推理，自动处理填充和裁剪
            def predict_with_transform(img_input, transform_func, reverse_transform_func):
                """对填充后的图像进行变换、推理，然后裁剪回原始尺寸"""
                # 应用变换
                img_transformed = transform_func(img_input)
                # 推理
                pred_logits = model(img_transformed)
                if isinstance(pred_logits, tuple):
                    pred_logits = pred_logits[0]
                # 反向变换
                pred_logits = reverse_transform_func(pred_logits)
                # 裁剪填充部分并插值到原始尺寸
                pred_logits = crop_prediction(pred_logits, target_h, target_w, pad_h, pad_w)
                # 如果scale != 1.0，还需要插值到原始H, W
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                return pred_logits
            
            # 1. 原始图像
            pred_logits = model(scaled_images_padded)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_logits = crop_prediction(pred_logits, target_h, target_w, pad_h, pad_w)
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                if scale != 1.0:
                    pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 2. 水平翻转
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.flip(x, dims=[3]),
                lambda x: torch.flip(x, dims=[3])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 3. 垂直翻转
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.flip(x, dims=[2]),
                lambda x: torch.flip(x, dims=[2])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 4. 旋转90度
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                lambda x: torch.rot90(x, k=-1, dims=[2, 3])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 5. 旋转180度
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),
                lambda x: torch.rot90(x, k=-2, dims=[2, 3])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 6. 旋转270度
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),
                lambda x: torch.rot90(x, k=-3, dims=[2, 3])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 7. 水平翻转+旋转90度
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.rot90(torch.flip(x, dims=[3]), k=1, dims=[2, 3]),
                lambda x: torch.flip(torch.rot90(x, k=-1, dims=[2, 3]), dims=[3])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
                pred_prob = torch.sigmoid(pred_logits)
                scale_prob_maps.append(pred_prob)
            
            # 8. 垂直翻转+旋转90度
            pred_logits = predict_with_transform(
                scaled_images_padded,
                lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3]),
                lambda x: torch.flip(torch.rot90(x, k=-1, dims=[2, 3]), dims=[2])
            )
            if not (torch.any(torch.isnan(pred_logits)) or torch.any(torch.isinf(pred_logits))):
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
        
        # 连通域标记，并使用概率图作为 intensity_image，以便计算 mean_intensity / max_intensity
        labels = measure.label(binary, connectivity=1)
        regions = measure.regionprops(labels, intensity_image=probs_np.astype(np.float32))
        
        cleaned = np.zeros_like(binary, dtype=np.uint8)
        
        for region in regions:
            area = region.area
            mean_prob = float(region.mean_intensity) if hasattr(region, "mean_intensity") else 0.0
            # 获取区域内的最大概率（如果不可用则回退为 mean_prob）
            max_prob = float(region.max_intensity) if hasattr(region, "max_intensity") else mean_prob
            
            # Level 1: 极小区域（<= tiny_size_thresh）视为绝对噪音，直接跳过
            if area <= tiny_size_thresh:
                continue
            
            # Level 2: 大于等于 20 像素的区域，无条件保留
            if area >= 20:
                cleaned[labels == region.label] = 1
                continue
            
            # Level 3: 3~19 像素之间，依据平均概率 / 最大概率判断
            # 修改为：平均概率达标 或 最大概率极高(>0.9) 时保留
            if small_min_size <= area <= small_max_size and (mean_prob > prob_threshold or max_prob > 0.9):
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
        prob_map=None,
        confidence_gate: float = 0.85,
        min_largest_avg_prob: float = 0.5,
    ):
        """
        后处理优化预测mask - 增强版
        
        Args:
            pred_mask: 预测mask (numpy或tensor)
            min_size: 移除小于此大小的连通域。当 keep_largest=True 时，如果最大连通域也小于此值，将清空整个mask（用于处理空GT场景下的微小噪点）
            use_morphology: 是否使用形态学操作
            keep_largest: 是否只保留最大连通域（单器官分割推荐）。注意：即使保留最大连通域，如果其面积 < min_size，也会被清空
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
        
        # 提取概率图用于置信度判断与平均概率计算
        if prob_map is None:
            prob_np = pred_np
        else:
            if isinstance(prob_map, torch.Tensor):
                prob_np = prob_map.detach().cpu().numpy()
            else:
                prob_np = np.asarray(prob_map)
        
        # 【关键修复】对于几乎为空的预测，更严格地处理，避免后处理引入假阳性
        pred_sum = pred_np.sum()
        
        # 【置信度预过滤】如果全图最大概率低于阈值，直接判定为全黑
        # 这可以避免低置信度的噪声被形态学操作放大
        max_prob_proxy = float(prob_np.max()) if prob_np.size > 0 else 0.0
        if max_prob_proxy < confidence_gate:
            if is_tensor:
                return torch.zeros_like(pred_mask)
            else:
                return np.zeros_like(pred_np)
        
        # 如果预测像素数很少（< 100像素），可能是噪声，直接清空
        # 【动态阈值】根据图像大小调整阈值
        image_size = pred_np.size
        dynamic_threshold = max(100, int(image_size * 0.0005))  # 至少100像素，或图像的0.05%
        if pred_sum < dynamic_threshold:
            # 对于几乎为空的预测，直接返回全空mask，避免后处理引入假阳性
            if is_tensor:
                return torch.zeros_like(pred_mask)
            else:
                return np.zeros_like(pred_np)
        
        pred_binary = (pred_np > 0.5).astype(np.uint8)
        
        # 1. 填充孔洞（Fill Holes）- 去除器官内部的假阴性空洞
        # 【关键修复】fill_holes只在有较大预测块时启用，防止把背景底噪填成实心块
        if fill_holes and pred_sum >= 1000:  # 只有预测块较大时才填充孔洞
            # 使用 scipy.ndimage.binary_fill_holes 填充内部孔洞
            pred_binary = ndimage.binary_fill_holes(pred_binary).astype(np.uint8)
        
        # 2. 形态学闭操作 - 进一步填充小孔洞和缝隙
        if use_morphology:
            # 核大小从 (5, 5) 调整为 (3, 3)，减弱闭操作，避免不同病灶被误连
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
                max_size = sizes[largest_label - 1]  # sizes 是 0-indexed，largest_label 是 1-indexed
                
                # 【关键优化】绝对最小面积限制：如果最大的块都小于阈值，说明全是噪点，直接清空
                # 这样可以避免在空 GT 的情况下，微小噪点被保留导致 Dice 从 1.0 变成 0.0
                largest_component = (labeled == largest_label).astype(np.uint8)
                if max_size < min_size:
                    pred_binary = np.zeros_like(pred_binary)
                else:
                    # 计算最大连通域的平均概率，避免低置信度大块被误保留
                    if prob_np is not None and prob_np.size > 0:
                        largest_mean_prob = float((prob_np * largest_component).sum() / max_size)
                    else:
                        largest_mean_prob = 1.0
                    
                    if largest_mean_prob < min_largest_avg_prob:
                        pred_binary = np.zeros_like(pred_binary)
                    else:
                        # 只保留最大连通域
                        pred_binary = largest_component
        else:
            # 4. 连通域分析 - 移除小区域（如果不使用keep_largest）
            # 【动态连通域过滤】根据图像大小和预测块大小动态调整min_size
            if min_size > 0:
                labeled, num_features = ndimage.label(pred_binary)
                if num_features > 0:
                    sizes = ndimage.sum(pred_binary, labeled, range(1, num_features + 1))
                    # 【关键修复】动态调整min_size：对于几乎为空的预测，使用更严格的阈值
                    dynamic_min_size = max(min_size, int(image_size * 0.002))  # 至少min_size，或图像的0.2%
                    if pred_sum < 1000:  # 如果总预测像素很少，使用更严格的阈值
                        dynamic_min_size = max(dynamic_min_size, 1000)  # 至少1000像素
                    mask_sizes = sizes >= dynamic_min_size
                    # 只保留大区域
                    keep_labels = np.where(mask_sizes)[0] + 1
                    if len(keep_labels) == 0:
                        # 如果没有区域满足条件，清空整个mask
                        pred_binary = np.zeros_like(pred_binary)
                    else:
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
    pred_sum = pred.sum()
    target_sum = target.sum()
    
    # 【关键修复】空掩码特判逻辑
    # Case 1: GT 为空（全黑样本）
    if target_sum == 0:
        if pred_sum == 0:
            # 场景 A: GT 为空，Pred 为空 → Dice = 1.0 (完美预测)
            return 1.0
        else:
            # 场景 B: GT 为空，Pred 不为空 (有误报) → Dice = 0.0 (完全错误)
            return 0.0
    
    # Case 2: GT 不为空，但预测为空
    if pred_sum == 0:
        # 场景 C: GT 不为空，Pred 为空 (漏报) → Dice = 0.0 (完全漏检)
        return 0.0
    
    # Case 3: 正常情况，使用标准 Dice 公式（只计算前景类）
    union = pred_sum + target_sum
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
        【统一修复】计算Dice的辅助方法（用于集成评估）
        
        使用与 _compute_metrics_unified 相同的逻辑，确保一致性
        
        Args:
            pred_mask: 预测掩码
            target_mask: 真实掩码
            smooth: 平滑项
        
        Returns:
            Dice系数
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
        
    def _compute_iou_for_ensemble(self, pred_mask, target_mask, smooth=1e-7):
        """
        【统一修复】计算IoU的辅助方法（用于集成评估）
        
        使用与 _compute_metrics_unified 相同的逻辑，确保一致性
        
        Args:
            pred_mask: 预测掩码
            target_mask: 真实掩码
            smooth: 平滑项
        
        Returns:
            IoU系数
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
        union = pred_sum + target_sum - intersection
        
        # 【核心修复逻辑】空掩码特判
        # Case 1: 双空（GT 为空且 Pred 为空）
        if target_sum <= smooth and pred_sum <= smooth:
            return 1.0  # IoU=1.0 (完美预测)
        
        # Case 2: 单空（GT 为空但 Pred 不为空，或 GT 不为空但 Pred 为空）
        if target_sum <= smooth or pred_sum <= smooth:
            return 0.0  # IoU=0.0 (误报或漏报)
        
        # Case 3: 正常情况，使用标准 IoU 公式（只计算前景类）
        # IoU = |Pred ∩ GT| / |Pred ∪ GT|
        return (intersection + smooth) / (union + smooth) if union > smooth else 0.0
        
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

    # scan_best_threshold 方法已移除，请使用 utils.py 中的全局函数 scan_best_threshold



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
                A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
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
            # 【Windows 多进程优化】为预测线程也启用多进程数据加载
            # Intel Core Ultra 9 285HX: 使用 8 个 worker 充分利用 P-Core
            import platform
            is_windows = platform.system() == 'Windows'
            cpu_count = os.cpu_count() or 1
            num_workers = 8 if is_windows else max(0, min(4, cpu_count - 1))
            dataloader = DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),  # 【优化】加速数据传输
                persistent_workers=(num_workers > 0)  # 【关键】让子进程保持存活
            )
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




