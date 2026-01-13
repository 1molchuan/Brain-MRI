# -*- coding: utf-8 -*-
"""
从utils.dataset模块
"""
from utils.common import *

# ==================== 数据集类 ====================

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

