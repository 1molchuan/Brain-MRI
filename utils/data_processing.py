# -*- coding: utf-8 -*-
"""
从utils.data_processing模块
"""
from utils.common import *

# ==================== 数据处理函数 ====================

def parse_extra_modalities_spec(spec: Optional[str]) -> Dict[str, str]:
    """解析额外模态配置字符串，格式: name:path;name2:path2"""
    modalities = {}
    if not spec:
        return modalities
    for item in spec.split(';'):
        if not item.strip() or ':' not in item:
            continue
        name, modal_path = item.split(':', 1)
        modal_path = modal_path.strip().strip('"').strip("'")
        if name.strip() and modal_path:
            modalities[name.strip()] = modal_path
    return modalities


def build_extra_modalities_lists(image_paths: List[str], modalities_dirs: Dict[str, str]) -> Optional[Dict[str, List[Optional[str]]]]:
    """根据主图像路径和模态目录生成额外模态文件路径列表。"""
    if not modalities_dirs:
        return None
    extras = {name: [] for name in modalities_dirs.keys()}
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        for name, dir_path in modalities_dirs.items():
            alt_path = os.path.join(dir_path, base_name)
            extras[name].append(alt_path if os.path.exists(alt_path) else None)
    return extras


def normalize_volume_percentile(volume, p_low=10, p_high=99):
    """
    使用百分位数归一化（参考标准代码）
    更鲁棒，能处理异常值和不同强度范围的医学图像
    
    Args:
        volume: 图像数组 (H, W, C) 或 (H, W)
        p_low: 低百分位数 (默认10)
        p_high: 高百分位数 (默认99)
    
    Returns:
        归一化后的图像
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    volume = volume.astype(np.float32)
    
    # 对多通道图像，对每个通道分别计算百分位数（参考标准代码）
    if len(volume.shape) == 3:
        # 多通道：对每个通道分别归一化
        normalized_channels = []
        for c in range(volume.shape[2]):
            channel = volume[:, :, c]
            p10 = np.percentile(channel, p_low)
            p99 = np.percentile(channel, p_high)
            
            if p99 > p10:
                channel = np.clip(channel, p10, p99)
                channel = (channel - p10) / (p99 - p10)
            else:
                channel = np.zeros_like(channel)
            
            # Z-score标准化
            m = np.mean(channel)
            s = np.std(channel)
            s = max(s, 1e-7)
            channel = (channel - m) / s
            
            normalized_channels.append(channel)
        
        volume = np.stack(normalized_channels, axis=2)
    else:
        # 单通道
        p10 = np.percentile(volume, p_low)
        p99 = np.percentile(volume, p_high)
        
        if p99 > p10:
            volume = np.clip(volume, p10, p99)
            volume = (volume - p10) / (p99 - p10)
        else:
            volume = np.zeros_like(volume)
        
        # Z-score标准化
        m = np.mean(volume)
        s = np.std(volume)
        s = max(s, 1e-7)
        volume = (volume - m) / s
    
    return volume

