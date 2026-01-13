"""
TCGA-LGG 数据集加载器
支持 2.5D 输入(上一张、当前、下一张切片堆叠)
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List

# 尝试导入 cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    raise ImportError(
        "OpenCV (cv2) 未安装。请运行: pip install opencv-python-headless"
    )

# 尝试导入 albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    raise ImportError(
        "albumentations 未安装。请运行: pip install albumentations"
    )


class TCGA2_5DDataset(Dataset):
    """
    TCGA-LGG 2.5D 数据集(支持递归搜索子文件夹)
    
    文件名格式: TCGA_CS_5393_19990606_1.tif
    解析: {Case_ID}_{Slice_ID}.tif
    
    2.5D 策略:
    - Channel 0: 上一张切片 (Previous)
    - Channel 1: 当前切片 (Current)
    - Channel 2: 下一张切片 (Next)
    
    目录结构支持：
    - 扁平结构：所有 .tif 文件在同一目录
    - 嵌套结构：主文件夹包含多个子文件夹，每个子文件夹包含 .tif 文件
    - 系统会自动递归搜索所有子文件夹中的 .tif 文件
    - Mask 文件会在图像文件同一目录或 mask_dir 中递归搜索
    """
    
    def __init__(
        self,
        data_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        is_train: bool = True,
        debug: bool = False,
        mode: str = "2.5d"  # "2d" 或 "2.5d"
    ):
        """
        Args:
            data_dir: 图像目录路径(支持递归搜索所有子文件夹)
            mask_dir: 掩码目录路径(如果为None，则从data_dir同级目录查找，也支持递归搜索)
            transform: albumentations 变换
            is_train: 是否为训练集
            debug: 是否启用调试模式(打印shape信息)
            mode: 数据集模式，"2d" 返回单通道 (1, H, W)，"2.5d" 返回三通道堆叠 (3, H, W)
        
        目录结构支持:
        - 扁平结构: data_dir/ 下直接包含 .tif 文件
        - 嵌套结构: data_dir/ 下有多个子文件夹，每个子文件夹包含 .tif 文件
        - 系统会自动递归搜索所有子文件夹中的 .tif 文件
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir if mask_dir else data_dir.replace('images', 'masks')
        self.transform = transform
        self.is_train = is_train
        self.debug = debug
        self.mode = mode.lower()  # 统一转换为小写
        
        # 验证 mode 参数
        if self.mode not in ("2d", "2.5d"):
            raise ValueError(f"mode 必须是 '2d' 或 '2.5d'，当前为: {mode}")
        
        print(f"[数据集] 模式: {self.mode.upper()} ({'单通道' if self.mode == '2d' else '三通道堆叠'})")
        
        # 解析并组织文件
        self.file_dict, self.file_list = self._parse_and_organize_files()
        
        if len(self.file_list) == 0:
            raise ValueError(
                f"在 {data_dir} 及其所有子文件夹中未找到任何 .tif 文件\n"
                f"请确保：\n"
                f"  1. 目录路径正确\n"
                f"  2. 目录中包含 .tif 格式的图像文件\n"
                f"  3. 文件名格式为: {{Case_ID}}_{{Slice_ID}}.tif (例如: TCGA_CS_5393_19990606_1.tif)"
            )
        
        print(f"[数据集] 加载了 {len(self.file_list)} 个样本")
        print(f"[数据集] 包含 {len(self.file_dict)} 个病例")
    
    def _parse_filename(self, filename: str) -> Tuple[str, int]:
        """
        解析文件名，提取 Case_ID 和 Slice_ID
        
        Args:
            filename: 文件名，如 "TCGA_CS_5393_19990606_1.tif"
        
        Returns:
            (case_id, slice_id): 病例ID和切片序号
        """
        # 移除扩展名
        basename = os.path.splitext(filename)[0]
        
        # 匹配格式: TCGA_CS_5393_19990606_1
        # 最后的下划线和数字是 Slice_ID
        match = re.match(r'^(.+)_(\d+)$', basename)
        if match:
            case_id = match.group(1)  # TCGA_CS_5393_19990606
            slice_id = int(match.group(2))  # 1
            return case_id, slice_id
        else:
            # 如果格式不匹配，尝试其他解析方式
            parts = basename.split('_')
            if len(parts) >= 2:
                case_id = '_'.join(parts[:-1])
                try:
                    slice_id = int(parts[-1])
                    return case_id, slice_id
                except ValueError:
                    pass
        
        # 如果无法解析，使用文件名作为 case_id，0 作为 slice_id
        print(f"[警告] 无法解析文件名格式: {filename}，使用默认值")
        return basename, 0
    
    def _parse_and_organize_files(self) -> Tuple[Dict[Tuple[str, int], str], List[Tuple[str, int]]]:
        """
        解析并组织文件(递归搜索所有子文件夹)
        
        Returns:
            (file_dict, file_list):
                - file_dict: {(case_id, slice_id): file_path} 查找表
                - file_list: [(case_id, slice_id), ...] 排序后的文件列表
        """
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        # 递归收集所有 .tif 文件(排除 _mask.tif)
        all_files = []
        
        # 递归遍历所有子文件夹
        for root, dirs, files in os.walk(self.data_dir):
            for filename in files:
                if filename.endswith('.tif') and '_mask' not in filename:
                    file_path = os.path.join(root, filename)
                    case_id, slice_id = self._parse_filename(filename)
                    all_files.append((case_id, slice_id, file_path))
        
        if len(all_files) == 0:
            raise ValueError(f"在 {self.data_dir} 及其子文件夹中未找到任何 .tif 文件")
        
        # 构建查找表
        # 如果同一 (case_id, slice_id) 出现多次，保留最后一个(或可以报错)
        file_dict = {}
        duplicate_keys = []
        for case_id, slice_id, file_path in all_files:
            key = (case_id, slice_id)
            if key in file_dict:
                duplicate_keys.append((key, file_path, file_dict[key]))
            file_dict[key] = file_path
        
        if duplicate_keys:
            print(f"[警告] 发现 {len(duplicate_keys)} 个重复的 (case_id, slice_id) 组合，将使用最后找到的文件")
            for key, new_path, old_path in duplicate_keys[:5]:  # 只显示前5个
                print(f"  - {key}: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
        
        # 排序：先按 case_id 字符串排序，再按 slice_id 数值排序
        sorted_files = sorted(all_files, key=lambda x: (x[0], x[1]))
        
        # 提取排序后的 (case_id, slice_id) 列表
        file_list = [(case_id, slice_id) for case_id, slice_id, _ in sorted_files]
        
        return file_dict, file_list
    
    def _get_adjacent_slice_path(self, case_id: str, slice_id: int, offset: int) -> Optional[str]:
        """
        获取相邻切片的路径
        
        Args:
            case_id: 病例ID
            slice_id: 当前切片ID
            offset: 偏移量(-1 表示上一张，+1 表示下一张)
        
        Returns:
            文件路径，如果不存在则返回 None
        """
        target_slice_id = slice_id + offset
        key = (case_id, target_slice_id)
        return self.file_dict.get(key)
    
    def _load_image(self, file_path: str) -> np.ndarray:
        """
        加载图像(确保是单通道灰度图)
        
        Args:
            file_path: 图像路径
        
        Returns:
            灰度图像数组 (H, W)
        """
        # 使用 cv2 读取，确保是灰度图
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"无法读取图像: {file_path}")
        
        # 确保是单通道
        if len(img.shape) == 3:
            # 如果是伪彩色，转换为灰度
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img.astype(np.float32)
    
    def _load_mask(self, case_id: str, slice_id: int, image_path: str) -> np.ndarray:
        """
        加载掩码(支持递归搜索子文件夹)
        
        Args:
            case_id: 病例ID
            slice_id: 切片ID
            image_path: 对应的图像文件路径(用于确定mask的相对位置)
        
        Returns:
            掩码数组 (H, W)
        """
        # 策略1: 在图像文件同一目录下查找mask
        image_dir = os.path.dirname(image_path)
        mask_filename = f"{case_id}_{slice_id}_mask.tif"
        mask_path = os.path.join(image_dir, mask_filename)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.float32)
                return mask
        
        # 策略2: 在mask_dir中递归搜索
        if os.path.exists(self.mask_dir):
            # 递归搜索mask文件
            for root, dirs, files in os.walk(self.mask_dir):
                for filename in files:
                    if filename == mask_filename:
                        mask_path = os.path.join(root, filename)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            mask = (mask > 127).astype(np.float32)
                            return mask
        
        # 策略3: 尝试其他可能的命名格式(在同一目录)
        alt_mask_filename = f"{case_id}_{slice_id}.tif"
        alt_mask_path = os.path.join(image_dir, alt_mask_filename)
        if os.path.exists(alt_mask_path):
            mask = cv2.imread(alt_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.float32)
                return mask
        
        # 如果都找不到，抛出错误
        raise FileNotFoundError(
            f"未找到掩码文件: {mask_filename}\n"
            f"  图像路径: {image_path}\n"
            f"  搜索目录: {image_dir}, {self.mask_dir}"
        )
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据项(支持 2D 和 2.5D 模式)
        
        Args:
            index: 样本索引
        
        Returns:
            (image, mask): 
                - 2D 模式: image (1, H, W) 单通道图像, mask (1, H, W) 单通道掩码
                - 2.5D 模式: image (3, H, W) 三通道堆叠图像, mask (1, H, W) 单通道掩码
        """
        # 获取当前帧信息
        case_id, slice_id = self.file_list[index]
        
        # 加载当前帧
        current_path = self.file_dict[(case_id, slice_id)]
        current_img = self._load_image(current_path)
        h, w = current_img.shape
        
        # 根据模式决定返回格式
        if self.mode == "2d":
            # 2D 模式：只返回当前切片，单通道 (1, H, W)
            stacked_img = current_img  # (H, W)
        else:
            # 2.5D 模式：堆叠上一张、当前、下一张，三通道 (3, H, W)
            # 尝试加载上一张和下一张
            prev_path = self._get_adjacent_slice_path(case_id, slice_id, -1)
            next_path = self._get_adjacent_slice_path(case_id, slice_id, +1)
            
            # 边界处理：如果相邻切片不存在，复制当前帧
            if prev_path is None:
                prev_img = current_img.copy()
            else:
                prev_img = self._load_image(prev_path)
            
            if next_path is None:
                next_img = current_img.copy()
            else:
                next_img = self._load_image(next_path)
            
            # 确保三张图像尺寸一致
            if prev_img.shape != (h, w):
                prev_img = cv2.resize(prev_img, (w, h), interpolation=cv2.INTER_LINEAR)
            if next_img.shape != (h, w):
                next_img = cv2.resize(next_img, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 堆叠为 3 通道 (H, W, 3)
            # Channel 0: Previous, Channel 1: Current, Channel 2: Next
            stacked_img = np.dstack([prev_img, current_img, next_img])
        
        # 【修复双重归一化Bug】不再手动除以255.0
        # 保持0-255的数值范围，让albumentations.Normalize处理归一化
        # 如果手动除以255，然后Normalize又会除以255，导致像素值过小（约0.00xxx）
        # stacked_img = stacked_img / 255.0  # 已删除：避免双重归一化
        
        # 加载掩码(传入图像路径以便在同一目录查找)
        mask = self._load_mask(case_id, slice_id, current_path)
        
        # 确保掩码尺寸与图像一致
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 应用数据增强
        if self.transform is not None:
            transformed = self.transform(image=stacked_img, mask=mask)
            stacked_img = transformed['image']
            mask = transformed['mask']
            
            # 【关键修复】确保 mask 有通道维度
            # albumentations 的 ToTensorV2 对单通道 mask 不会自动添加通道维度
            # 需要手动确保 mask 是 (1, H, W) 而不是 (H, W)
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    # 如果是 2D tensor (H, W)，添加通道维度
                    mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    # 如果是 3D tensor 但通道维度不在第一维，需要调整
                    # 这种情况应该不会发生，但为了安全起见
                    if mask.shape[2] == 1:
                        mask = mask.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)
            elif isinstance(mask, np.ndarray):
                # 如果是 numpy array，转换为 tensor 并添加通道维度
                if mask.ndim == 2:
                    mask = torch.from_numpy(mask).unsqueeze(0).float()  # (H, W) -> (1, H, W)
                elif mask.ndim == 3:
                    mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # (H, W, 1) -> (1, H, W)
        else:
            # 如果没有变换，手动转换为 tensor
            if self.mode == "2d":
                # 2D 模式：单通道 (H, W) -> (1, H, W)
                stacked_img = torch.from_numpy(stacked_img).unsqueeze(0).float()  # (H, W) -> (1, H, W)
            else:
                # 2.5D 模式：三通道 (H, W, 3) -> (3, H, W)
                stacked_img = torch.from_numpy(stacked_img).permute(2, 0, 1).float()  # (H, W, 3) -> (3, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # (H, W) -> (1, H, W)
        
        # 【维度验证】确保返回的 tensor 维度正确
        if isinstance(stacked_img, torch.Tensor):
            expected_channels = 1 if self.mode == "2d" else 3
            if stacked_img.ndim != 3 or stacked_img.shape[0] != expected_channels:
                raise ValueError(f"图像维度错误: 期望 ({expected_channels}, H, W)，实际 {stacked_img.shape}")
        if isinstance(mask, torch.Tensor):
            if mask.ndim != 3 or mask.shape[0] != 1:
                raise ValueError(f"掩码维度错误: 期望 (1, H, W)，实际 {mask.shape}")
        
        # 调试输出
        if self.debug:
            expected_channels = 1 if self.mode == "2d" else 3
            print(f"[调试] 样本 {index} ({self.mode.upper()}): image shape={stacked_img.shape}, mask shape={mask.shape}")
            assert stacked_img.shape[0] == expected_channels, f"图像通道数应为{expected_channels}，实际为{stacked_img.shape[0]}"
            assert mask.shape[0] == 1, f"掩码通道数应为1，实际为{mask.shape[0]}"
        
        return stacked_img, mask


def get_tcga_transforms(
    is_train: bool = True,
    img_size: Tuple[int, int] = (256, 256),
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    获取 TCGA 数据集的数据增强变换
    
    Args:
        is_train: 是否为训练集
        img_size: 目标图像尺寸
        normalize_mean: 归一化均值(ImageNet)
        normalize_std: 归一化标准差(ImageNet)
    
    Returns:
        albumentations.Compose 变换
    """
    if is_train:
        transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                mode=cv2.BORDER_REFLECT_101,
                p=0.6
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            ),
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    else:
        # 验证/测试集：只做 resize 和归一化
        transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    
    return transform

