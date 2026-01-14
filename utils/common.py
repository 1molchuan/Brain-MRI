"""
工具函数和数据处理类模块
包含所有独立工具函数、数据处理类和模型加载函数
"""
import utils.logging_setup
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

