# -*- coding: utf-8 -*-
"""
工作线程公共模块
包含所有线程类共享的导入和配置
"""

# PyQt5 相关导入
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QMutex
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# 【日志优化】抑制 Numpy 和 Grad-CAM 的非致命警告
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

# Grad-CAM 支持
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

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
from utils import (
    EarlyStopping,
    scan_best_threshold,
    GreyWolfThresholdOptimizer,
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

