import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from typing import Optional, Tuple
from utils import window_partition, window_reverse

class DropPath(nn.Module):
    """Stochastic Depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
# MedicalImageAugmentation 类已移动到 utils.py

class CRFPostProcessor:
    """
    条件随机场（CRF）后处理模块
    对模型输出的概率图进行空间一致性优化
    参考: "Efficient Inference in Fully Connected CRFs" (NIPS 2012)
    """
    def __init__(self, num_iterations=5, pos_weight=3.0, pos_xy_std=3, pos_rgb_std=3, 
                 bi_xy_std=80, bi_rgb_std=13, bi_w=10):
        """
        Args:
            num_iterations: CRF迭代次数
            pos_weight: 一元势能权重
            pos_xy_std: 位置高斯核标准差
            pos_rgb_std: 颜色高斯核标准差
            bi_xy_std: 双边滤波位置标准差
            bi_rgb_std: 双边滤波颜色标准差
            bi_w: 双边滤波权重
        """
        self.num_iterations = num_iterations
        self.pos_weight = pos_weight
        self.pos_xy_std = pos_xy_std
        self.pos_rgb_std = pos_rgb_std
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.bi_w = bi_w
    
    def process(self, prob_map, image):
        """
        对概率图进行CRF优化
        Args:
            prob_map: 模型输出的概率图 (H, W) 或 (1, H, W)
            image: 原始图像 (H, W, C) 用于空间一致性约束
        Returns:
            优化后的概率图
        """
        try:
            # pydensecrf是可选依赖，使用动态导入避免linting警告
            import importlib
            dcrf = importlib.import_module('pydensecrf.densecrf')
            utils_module = importlib.import_module('pydensecrf.utils')
            unary_from_softmax = utils_module.unary_from_softmax
        except ImportError:
            print("[警告] pydensecrf未安装，跳过CRF后处理")
            return prob_map
        
        if len(prob_map.shape) == 3:
            prob_map = prob_map[0]
        
        H, W = prob_map.shape
        
        # 准备概率图（需要2类：背景和前景）
        prob_bg = 1 - prob_map
        prob_fg = prob_map
        unary = np.stack([prob_bg, prob_fg], axis=0)
        
        # 创建CRF
        d = dcrf.DenseCRF2D(W, H, 2)
        
        # 一元势能（来自模型预测）
        U = unary_from_softmax(unary)
        U = np.ascontiguousarray(U)
        d.setUnaryEnergy(U)
        
        # 二元势能（空间一致性）
        # 位置高斯核
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_weight)
        
        # 双边滤波（颜色+位置）
        if len(image.shape) == 3:
            image_rgb = image
        else:
            image_rgb = np.stack([image, image, image], axis=-1)
        
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, 
                              rgbim=image_rgb, compat=self.bi_w)
        
        # 推理
        Q = d.inference(self.num_iterations)
        map_result = np.array(Q).reshape((2, H, W))
        result = map_result[1]  # 前景概率
        
        return result
    
    def process_batch(self, prob_maps, images):
        """
        批量处理
        Args:
            prob_maps: (B, H, W) 或 (B, 1, H, W)
            images: (B, C, H, W) 或 (B, H, W, C)
        """
        results = []
        for i in range(prob_maps.shape[0]):
            prob = prob_maps[i]
            img = images[i]
            
            # 转换图像格式
            if len(img.shape) == 3 and img.shape[0] == 3:  # (C, H, W)
                img = img.transpose(1, 2, 0)  # (H, W, C)
            
            result = self.process(prob, img)
            results.append(result)
        
        return np.stack(results, axis=0)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力，提升特征表达能力
    参考: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力模块
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    用于跨尺度特征融合，增强多尺度特征表达能力
    参考: "Feature Pyramid Networks for Object Detection" (CVPR 2017)
    """
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        
        # 横向连接（1x1卷积）
        self.lateral_convs = nn.ModuleList()
        # 输出卷积（3x3卷积，减少混叠效应）
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from different scales
        Returns:
            List of FPN feature maps
        """
        # 自顶向下路径
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )
        
        # 输出特征
        fpn_outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return fpn_outs


class AttentionGate(nn.Module):
    """
    改进的注意力门控机制，解决注意力过于分散的问题:
    1. 添加温度参数控制注意力集中度
    2. 使用更尖锐的激活函数
    3. 支持注意力正则化
    """
    def __init__(self, F_g, F_l, F_int, temperature=1.0, use_sharpen=True):
        super(AttentionGate, self).__init__()
        self.temperature = temperature  # 温度参数: <1 更集中, >1 更分散
        self.use_sharpen = use_sharpen
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 改进: 使用更深的网络生成注意力
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_int // 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(F_int // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
            # 不使用Sigmoid,改用温度调节的softmax-like激活
        )
        self.relu = nn.ReLU(inplace=True)
        
    def _sharpen_attention(self, attention_map):
        """
        使用温度调节的方式增强注意力集中度
        temperature < 1: 更尖锐(更集中)
        temperature = 1: 标准sigmoid
        temperature > 1: 更平滑(更分散)
        """
        if self.use_sharpen:
            # 先通过温度缩放
            attention_scaled = attention_map / self.temperature
            # 再应用sigmoid
            attention_sharp = torch.sigmoid(attention_scaled)
            
            # 可选: 进一步增强对比度 (power transform)
            # attention_sharp = attention_sharp ** 2
            
            return attention_sharp
        else:
            return torch.sigmoid(attention_map)
    
    def forward(self, g, x, return_attention=False):
        # 确保 g 和 x 的空间尺寸一致
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 确保通道数一致
        if g.shape[1] != self.W_g[0].in_channels:
            # 如果通道数不匹配，需要调整（这种情况不应该发生，但为了安全）
            if not hasattr(self, '_g_channel_adapter'):
                self._g_channel_adapter = nn.Conv2d(g.shape[1], self.W_g[0].in_channels, kernel_size=1).to(g.device)
            g = self._g_channel_adapter(g)
        if x.shape[1] != self.W_x[0].in_channels:
            if not hasattr(self, '_x_channel_adapter'):
                self._x_channel_adapter = nn.Conv2d(x.shape[1], self.W_x[0].in_channels, kernel_size=1).to(x.device)
            x = self._x_channel_adapter(x)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        
        # 生成注意力logits
        attention_logits = self.psi(psi)
        
        # 应用温度调节的激活
        attention_map = self._sharpen_attention(attention_logits)
        
        # 加权输出
        output = x * attention_map
        
        if return_attention:
            return output, attention_map
        return output


class SkipAttention(nn.Module):
    """
    nnFormer风格的跳跃注意力机制
    使用交叉注意力机制来融合编码器和解码器特征
    解码器特征作为Query，编码器特征作为Key和Value
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_drop=0.1):
        super(SkipAttention, self).__init__()
        self.dim = dim
        # 确保num_heads至少为1，且能整除dim
        if num_heads <= 0:
            num_heads = max(1, dim // 64)  # 如果为0，使用dim//64，但至少为1
        # 确保num_heads能整除dim
        if dim % num_heads != 0:
            # 调整num_heads使其能整除dim
            num_heads = max(1, dim // (dim // num_heads + 1))
            # 如果还是不能整除，使用最大可能的除数
            for n in range(num_heads, 0, -1):
                if dim % n == 0:
                    num_heads = n
                    break
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 将编码器和解码器特征投影到相同维度
        self.encoder_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.decoder_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # 交叉注意力：Query来自解码器，Key和Value来自编码器
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # Query from decoder
        self.k = nn.Linear(dim, dim, bias=qkv_bias)  # Key from encoder
        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # Value from encoder
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, decoder_feat, encoder_feat):
        """
        Args:
            decoder_feat: (B, C, H, W) 解码器特征
            encoder_feat: (B, C, H, W) 编码器特征（跳跃连接）
        Returns:
            fused_feat: (B, C, H, W) 融合后的特征
        """
        B, C, H, W = decoder_feat.shape
        
        # 确保编码器和解码器特征的空间尺寸一致
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            encoder_feat = F.interpolate(encoder_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 确保通道数一致
        if encoder_feat.shape[1] != C:
            if not hasattr(self, 'encoder_channel_adapter'):
                self.encoder_channel_adapter = nn.Conv2d(encoder_feat.shape[1], C, kernel_size=1).to(encoder_feat.device)
            encoder_feat = self.encoder_channel_adapter(encoder_feat)
        
        # 投影到相同维度 - 优化：合并操作减少中间变量
        dec_proj = self.decoder_proj(decoder_feat)  # (B, C, H, W)
        enc_proj = self.encoder_proj(encoder_feat)   # (B, C, H, W)
        
        # 优化：直接计算，减少reshape次数
        # 转换为序列格式 - 使用contiguous()确保内存连续
        N = H * W
        dec_seq = dec_proj.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        enc_seq = enc_proj.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        
        # 交叉注意力：Query来自解码器，Key和Value来自编码器
        # 优化：减少reshape操作
        head_dim = C // self.num_heads
        q = self.q(dec_seq).view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = self.k(enc_seq).view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = self.v(enc_seq).view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # 计算注意力 - 性能优化：只在必要时clamp
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        # 性能优化：只在值超出范围时才clamp（对于正常值，clamp是多余的）
        if attn.abs().max() > 50.0:
            attn = torch.clamp(attn, min=-50.0, max=50.0)
        attn = attn.softmax(dim=-1)
        
        # 应用注意力到Value
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # 转换回空间格式 (B, C, H, W) - 优化：使用view而不是reshape
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


class nnFormerBlock(nn.Module):
    """
    nnFormer风格的Transformer Block
    交替使用局部窗口注意力（LV-MSA）和全局注意力（GV-MSA）
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path_rate=0.1, use_global_attn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_global_attn = use_global_attn

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        
        # 根据配置选择局部或全局注意力
        if use_global_attn:
            self.attn = GlobalAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = WindowAttention(
                dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"

        shortcut = x
        x = self.norm1(x)
        
        if self.use_global_attn:
            # 全局注意力：直接处理序列
            x = self.attn(x)
        else:
            # 局部窗口注意力：需要处理空间维度
            # 性能优化：减少不必要的reshape操作，使用更高效的内存布局
            x = x.view(B, H, W, C)
            
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            need_pad = pad_b or pad_r
            
            if need_pad:
                # 优化：合并permute操作，减少中间变量
                x = x.permute(0, 3, 1, 2).contiguous()
                x = F.pad(x, (0, pad_r, 0, pad_b))
                x = x.permute(0, 2, 3, 1).contiguous()
            H_pad, W_pad = H + pad_b, W + pad_r

            # 循环移位 - 优化：避免不必要的roll操作
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # 窗口分割 - 优化：使用contiguous确保内存连续，提升性能
            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.contiguous().view(-1, self.window_size * self.window_size, C)

            # W-MSA
            attn_windows = self.attn(x_windows)
            
            # 修复：确保attn_windows的通道数与输入一致（适配层可能改变了通道数）
            # 获取实际的通道数
            actual_C = attn_windows.shape[-1]
            num_windows = attn_windows.shape[0]
            
            # 验证形状是否匹配
            total_elements = attn_windows.numel()
            window_elements = self.window_size * self.window_size * actual_C
            
            if total_elements % window_elements != 0:
                # 如果无法整除，尝试推断正确的通道数或窗口数量
                # 首先尝试推断通道数
                spatial_elements = num_windows * self.window_size * self.window_size
                if total_elements % spatial_elements == 0:
                    # 通道数可以推断
                    actual_C = total_elements // spatial_elements
                    window_elements = spatial_elements * actual_C
                else:
                    # 尝试推断窗口数量
                    if total_elements % (self.window_size * self.window_size) == 0:
                        # 可以推断窗口数量和通道数
                        per_window_elements = total_elements // (self.window_size * self.window_size)
                        # 尝试找到合理的窗口数量和通道数组合
                        # 假设通道数在合理范围内（32-512）
                        found = False
                        for test_C in range(32, 513, 8):  # 通道数必须是8的倍数（因为num_heads通常>=1）
                            test_num_windows = per_window_elements // test_C
                            if test_num_windows * test_C == per_window_elements and test_num_windows > 0:
                                num_windows = test_num_windows
                                actual_C = test_C
                                found = True
                                break
                        if not found:
                            # 最后的尝试：使用reshape自动推断
                            raise RuntimeError(
                                f"无法reshape attn_windows: shape={attn_windows.shape}, "
                                f"window_size={self.window_size}, total_elements={total_elements}, "
                                f"per_window_elements={per_window_elements}"
                            )
                    else:
                        raise RuntimeError(
                            f"无法reshape attn_windows: shape={attn_windows.shape}, "
                            f"window_size={self.window_size}, total_elements={total_elements}, "
                            f"window_elements={window_elements}"
                        )

            # 合并窗口 - 使用实际的通道数
            attn_windows = attn_windows.contiguous().view(num_windows, self.window_size, self.window_size, actual_C)
            shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)

            # 反向循环移位
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            if need_pad:
                x = x[:, :H, :W, :].contiguous()
            x = x.view(B, H * W, C)

        # 残差连接 - 优化：只在训练初期或检测到问题时才检查NaN
        residual = self.drop_path(x)
        # 性能优化：减少NaN检查频率（只在每10个batch检查一次）
        if hasattr(self, '_check_nan_counter'):
            self._check_nan_counter += 1
            check_nan = (self._check_nan_counter % 10 == 0)
        else:
            self._check_nan_counter = 0
            check_nan = True  # 第一次总是检查
        
        if check_nan and (torch.any(torch.isnan(residual)) or torch.any(torch.isinf(residual))):
            residual = torch.zeros_like(residual)
        x = shortcut + residual
        
        if check_nan and (torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))):
            x = shortcut
            
        # MLP
        shortcut2 = x
        x = self.norm2(x)
        
        if check_nan and (torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))):
            x = shortcut2
        
        mlp_out = self.mlp(x)
        residual2 = self.drop_path(mlp_out)
        
        if check_nan and (torch.any(torch.isnan(residual2)) or torch.any(torch.isinf(residual2))):
            residual2 = torch.zeros_like(residual2)
        
        x = shortcut2 + residual2
        
        if check_nan and (torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))):
            x = shortcut2

        return x

# 改进的UNet模型
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()
        
        # 下采样路径
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._block(512, 1024)
        
        # 上采样路径
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = self._block(128, 64)
        
        # 注意力门控 - 使用temperature=0.5让注意力更集中
        # temperature < 1: 注意力更尖锐(更集中到重点区域)
        # temperature = 1: 标准行为
        # temperature > 1: 注意力更平滑(更分散)
        self.att1 = AttentionGate(64, 64, 32, temperature=0.5, use_sharpen=True)
        self.att2 = AttentionGate(128, 128, 64, temperature=0.5, use_sharpen=True)
        self.att3 = AttentionGate(256, 256, 128, temperature=0.5, use_sharpen=True)
        self.att4 = AttentionGate(512, 512, 256, temperature=0.5, use_sharpen=True)
        
        # 中间监督输出
        self.conv4_out = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最终卷积
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, return_attention=False, return_aux=False):
        # 下采样
        conv1 = self.down1(x)
        x = self.pool(conv1)
        
        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)
        
        conv4 = self.down4(x)
        x = self.pool(conv4)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 带注意力的上采样
        attention_maps = {}
        
        x = self.up4(x)
        if return_attention:
            conv4, att4 = self.att4(x, conv4, return_attention=True)
            attention_maps['att4'] = att4
        else:
            conv4 = self.att4(x, conv4)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        ds4 = self.conv4_out(x)
        
        x = self.up3(x)
        if return_attention:
            conv3, att3 = self.att3(x, conv3, return_attention=True)
            attention_maps['att3'] = att3
        else:
            conv3 = self.att3(x, conv3)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        ds3 = self.conv3_out(x)
        
        x = self.up2(x)
        if return_attention:
            conv2, att2 = self.att2(x, conv2, return_attention=True)
            attention_maps['att2'] = att2
        else:
            conv2 = self.att2(x, conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        ds2 = self.conv2_out(x)
        
        x = self.up1(x)
        if return_attention:
            conv1, att1 = self.att1(x, conv1, return_attention=True)
            attention_maps['att1'] = att1
        else:
            conv1 = self.att1(x, conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        
        # 输出logits，不应用sigmoid（BCEWithLogitsLoss会在内部处理）
        final_output = self.final(x)

        aux_outputs = []
        if return_aux:
            target_size = final_output.shape[2:]
            aux_outputs.append(F.interpolate(ds4, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds3, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds2, size=target_size, mode='bilinear', align_corners=False))
        
        # 统一返回格式: (output, aux_outputs, attention_maps)
        # 根据参数填充None,保持一致的返回顺序
        if return_attention and return_aux:
            return final_output, aux_outputs, attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, aux_outputs
        else:
            return final_output


class TransUNet(nn.Module):
    """
    Transformer + UNet 混合架构，用于提高Dice指标
    
    设计思路：
    1. 使用UNet的编码器-解码器结构保持空间信息
    2. 在瓶颈层集成Transformer编码器，捕获长距离依赖关系
    3. Transformer的自注意力机制能够建模全局上下文，提升分割精度
    4. 结合CNN的局部特征提取和Transformer的全局建模能力
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 embed_dim=512, num_heads=8, num_layers=3, 
                 mlp_ratio=4.0, dropout=0.1):
        super(TransUNet, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 下采样路径（编码器）
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # Transformer瓶颈层
        # 将特征图转换为序列
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=1)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 将序列转换回特征图
        self.patch_unembed = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 瓶颈层卷积（在Transformer之后进一步处理）
        self.bottleneck = self._block(512, 1024)
        
        # 上采样路径（解码器）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = self._block(128, 64)
        
        # 注意力门控（与ImprovedUNet保持一致）
        self.att1 = AttentionGate(64, 64, 32, temperature=0.5, use_sharpen=True)
        self.att2 = AttentionGate(128, 128, 64, temperature=0.5, use_sharpen=True)
        self.att3 = AttentionGate(256, 256, 128, temperature=0.5, use_sharpen=True)
        self.att4 = AttentionGate(512, 512, 256, temperature=0.5, use_sharpen=True)
        
        # 中间监督输出
        self.conv4_out = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最终卷积
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _apply_transformer(self, x):
        """
        将特征图通过Transformer处理
        
        Args:
            x: [B, C, H, W] 特征图
            
        Returns:
            [B, C, H, W] 处理后的特征图
        """
        B, C, H, W = x.shape
        
        # 转换为序列: [B, C, H, W] -> [B, H*W, C]
        x_embed = self.patch_embed(x)  # [B, embed_dim, H, W]
        x_flat = x_embed.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # 通过Transformer编码器
        x_transformed = self.transformer(x_flat)  # [B, H*W, embed_dim]
        
        # 转换回特征图: [B, H*W, embed_dim] -> [B, embed_dim, H, W]
        x_transformed = x_transformed.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        x_out = self.patch_unembed(x_transformed)  # [B, 512, H, W]
        
        return x_out
        
    def forward(self, x, return_attention=False, return_aux=False):
        # 下采样（编码器）
        conv1 = self.down1(x)
        x = self.pool(conv1)
        
        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)
        
        conv4 = self.down4(x)
        x = self.pool(conv4)
        
        # Transformer瓶颈层
        # 先通过Transformer增强特征
        x_transformed = self._apply_transformer(x)
        # 再通过卷积瓶颈层
        x = self.bottleneck(x_transformed)
        
        # 带注意力的上采样（解码器）
        attention_maps = {}
        
        x = self.up4(x)
        if return_attention:
            conv4, att4 = self.att4(x, conv4, return_attention=True)
            attention_maps['att4'] = att4
        else:
            conv4 = self.att4(x, conv4)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        ds4 = self.conv4_out(x)
        
        x = self.up3(x)
        if return_attention:
            conv3, att3 = self.att3(x, conv3, return_attention=True)
            attention_maps['att3'] = att3
        else:
            conv3 = self.att3(x, conv3)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        ds3 = self.conv3_out(x)
        
        x = self.up2(x)
        if return_attention:
            conv2, att2 = self.att2(x, conv2, return_attention=True)
            attention_maps['att2'] = att2
        else:
            conv2 = self.att2(x, conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        ds2 = self.conv2_out(x)
        
        x = self.up1(x)
        if return_attention:
            conv1, att1 = self.att1(x, conv1, return_attention=True)
            attention_maps['att1'] = att1
        else:
            conv1 = self.att1(x, conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        
        # 输出logits，不应用sigmoid
        final_output = self.final(x)
        
        aux_outputs = []
        if return_aux:
            target_size = final_output.shape[2:]
            aux_outputs.append(F.interpolate(ds4, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds3, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds2, size=target_size, mode='bilinear', align_corners=False))
        
        # 统一返回格式
        if return_attention and return_aux:
            return final_output, aux_outputs, attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, aux_outputs
        else:
            return final_output


class DSTransUNet(nn.Module):
    """
    DS-TransUNet: Dual-Scale TransUNet
    在多个尺度上使用Transformer，增强多尺度特征提取能力
    
    设计特点：
    1. 在编码器的多个层级（conv3和conv4）使用Transformer，捕获不同尺度的全局依赖
    2. 双尺度特征融合：将不同尺度的Transformer特征进行融合
    3. 保持Deep Supervision，提供多层级监督信号
    4. 使用注意力门控机制，增强特征选择能力
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 embed_dim=256, num_heads=8, num_layers=2, 
                 mlp_ratio=4.0, dropout=0.1):
        super(DSTransUNet, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 下采样路径（编码器）
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 第一个Transformer层（在conv3尺度，256通道）
        self.patch_embed3 = nn.Conv2d(256, embed_dim, kernel_size=1)
        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer3 = nn.TransformerEncoder(encoder_layer3, num_layers=num_layers)
        self.patch_unembed3 = nn.Conv2d(embed_dim, 256, kernel_size=1)
        
        # 第二个Transformer层（在conv4尺度，512通道）
        self.patch_embed4 = nn.Conv2d(512, embed_dim, kernel_size=1)
        encoder_layer4 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer4 = nn.TransformerEncoder(encoder_layer4, num_layers=num_layers)
        self.patch_unembed4 = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 瓶颈层Transformer（在最大下采样尺度）
        self.patch_embed_bottleneck = nn.Conv2d(512, embed_dim, kernel_size=1)
        encoder_layer_bottleneck = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_bottleneck = nn.TransformerEncoder(encoder_layer_bottleneck, num_layers=num_layers)
        self.patch_unembed_bottleneck = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 瓶颈层卷积（在Transformer之后进一步处理）
        self.bottleneck = self._block(512, 1024)
        
        # 双尺度特征融合模块
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 上采样路径（解码器）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = self._block(128, 64)
        
        # 注意力门控（与ImprovedUNet保持一致）
        self.att1 = AttentionGate(64, 64, 32, temperature=0.5, use_sharpen=True)
        self.att2 = AttentionGate(128, 128, 64, temperature=0.5, use_sharpen=True)
        self.att3 = AttentionGate(256, 256, 128, temperature=0.5, use_sharpen=True)
        self.att4 = AttentionGate(512, 512, 256, temperature=0.5, use_sharpen=True)
        
        # 中间监督输出（Deep Supervision）
        self.conv4_out = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最终卷积
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def get_config(self):
        return {
            "in_channels": 3,
            "out_channels": 1,
            "embed_dim": self.embed_dim,
            "num_heads": self.transformer3.layers[0].self_attn.num_heads if hasattr(self.transformer3, "layers") else self.transformer3.layers[0].self_attn.num_heads if isinstance(self.transformer3, nn.TransformerEncoder) else 8,
            "num_layers": len(self.transformer3.layers) if hasattr(self.transformer3, "layers") else 2,
            "mlp_ratio": self.transformer3.layers[0].linear1.out_features / self.transformer3.layers[0].linear1.in_features if hasattr(self.transformer3, "layers") else 4.0,
            "dropout": getattr(self.transformer3.layers[0], "dropout", nn.Dropout(0.1)).p if hasattr(self.transformer3, "layers") else 0.1,
        }
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _apply_transformer(self, x, patch_embed, transformer, patch_unembed):
        """
        将特征图通过Transformer处理
        
        Args:
            x: [B, C, H, W] 特征图
            patch_embed: patch embedding层
            transformer: transformer编码器
            patch_unembed: patch unembedding层
            
        Returns:
            [B, C, H, W] 处理后的特征图
        """
        B, C, H, W = x.shape
        
        # 转换为序列: [B, C, H, W] -> [B, H*W, embed_dim]
        x_embed = patch_embed(x)  # [B, embed_dim, H, W]
        x_flat = x_embed.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # 通过Transformer编码器
        x_transformed = transformer(x_flat)  # [B, H*W, embed_dim]
        
        # 转换回特征图: [B, H*W, embed_dim] -> [B, C, H, W]
        x_transformed = x_transformed.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        x_out = patch_unembed(x_transformed)  # [B, C, H, W]
        
        return x_out
        
    def forward(self, x, return_attention=False, return_aux=False):
        # 下采样（编码器）
        conv1 = self.down1(x)
        x = self.pool(conv1)
        
        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)
        
        # 第一个Transformer层（在conv3尺度）
        conv3_transformed = self._apply_transformer(
            conv3, self.patch_embed3, self.transformer3, self.patch_unembed3
        )
        # 残差连接
        conv3 = conv3 + conv3_transformed
        
        conv4 = self.down4(x)
        x = self.pool(conv4)
        
        # 第二个Transformer层（在conv4尺度）
        conv4_transformed = self._apply_transformer(
            conv4, self.patch_embed4, self.transformer4, self.patch_unembed4
        )
        # 残差连接
        conv4 = conv4 + conv4_transformed
        
        # 双尺度特征融合：将conv3和conv4的特征融合
        # 将conv3上采样到conv4的尺度
        conv3_up = F.interpolate(conv3, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        # 融合两个尺度的特征
        conv4_fused = self.scale_fusion(torch.cat([conv4, conv3_up], dim=1))
        
        # 瓶颈层Transformer
        x_transformed = self._apply_transformer(
            conv4_fused, self.patch_embed_bottleneck, 
            self.transformer_bottleneck, self.patch_unembed_bottleneck
        )
        # 残差连接
        x = conv4_fused + x_transformed
        # 再通过卷积瓶颈层
        x = self.bottleneck(x)
        
        # 保存融合后的conv4用于解码器（注意：使用融合后的特征）
        conv4_for_decoder = conv4_fused
        
        # 带注意力的上采样（解码器）
        attention_maps = {}
        
        x = self.up4(x)
        # 确保 x 和 conv4_for_decoder 的空间尺寸匹配
        if x.shape[2:] != conv4_for_decoder.shape[2:]:
            conv4_for_decoder = F.interpolate(conv4_for_decoder, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv4, att4 = self.att4(x, conv4_for_decoder, return_attention=True)
            attention_maps['att4'] = att4
        else:
            conv4 = self.att4(x, conv4_for_decoder)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        ds4 = self.conv4_out(x)
        
        x = self.up3(x)
        # 确保 x 和 conv3 的空间尺寸匹配
        if x.shape[2:] != conv3.shape[2:]:
            conv3 = F.interpolate(conv3, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv3, att3 = self.att3(x, conv3, return_attention=True)
            attention_maps['att3'] = att3
        else:
            conv3 = self.att3(x, conv3)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        ds3 = self.conv3_out(x)
        
        x = self.up2(x)
        # 确保 x 和 conv2 的空间尺寸匹配
        if x.shape[2:] != conv2.shape[2:]:
            conv2 = F.interpolate(conv2, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv2, att2 = self.att2(x, conv2, return_attention=True)
            attention_maps['att2'] = att2
        else:
            conv2 = self.att2(x, conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        ds2 = self.conv2_out(x)
        
        x = self.up1(x)
        # 确保 x 和 conv1 的空间尺寸匹配
        if x.shape[2:] != conv1.shape[2:]:
            conv1 = F.interpolate(conv1, size=x.shape[2:], mode='bilinear', align_corners=False)
        if return_attention:
            conv1, att1 = self.att1(x, conv1, return_attention=True)
            attention_maps['att1'] = att1
        else:
            conv1 = self.att1(x, conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        
        # 输出logits，不应用sigmoid
        final_output = self.final(x)
        
        aux_outputs = []
        if return_aux:
            target_size = final_output.shape[2:]
            aux_outputs.append(F.interpolate(ds4, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds3, size=target_size, mode='bilinear', align_corners=False))
            aux_outputs.append(F.interpolate(ds2, size=target_size, mode='bilinear', align_corners=False))
        
        # 统一返回格式
        if return_attention and return_aux:
            return final_output, aux_outputs, attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, aux_outputs
        else:
            return final_output


# ==================== Swin Transformer 核心模块 ====================
class WindowAttention(nn.Module):
    """Swin Transformer的窗口注意力机制（数值稳定版本）"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 确保num_heads有效且dim能被num_heads整除
        if num_heads <= 0:
            num_heads = 1
        if dim % num_heads != 0:
            # 调整dim使其能被num_heads整除（向下取整）
            dim = (dim // num_heads) * num_heads
            if dim == 0:
                dim = num_heads  # 至少为num_heads
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 if head_dim > 0 else 1.0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 如果输入维度与dim不匹配，需要适配层（延迟初始化）
        self._dim_adapter = None
        
        # 改进的权重初始化，防止梯度爆炸
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        # QKV投影使用较小的初始化
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        
        # 输出投影使用标准初始化
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        
        # 如果输入维度与预期维度不匹配，使用适配层
        if C != self.dim:
            if self._dim_adapter is None:
                self._dim_adapter = nn.Linear(C, self.dim).to(x.device)
            x = self._dim_adapter(x)
            C = self.dim
        
        # 确保C能被num_heads整除（双重检查）
        if C % self.num_heads != 0:
            # 调整C使其能被num_heads整除
            head_dim = max(1, C // self.num_heads)
            C_aligned = head_dim * self.num_heads
            if C_aligned != C:
                if self._dim_adapter is None:
                    self._dim_adapter = nn.Linear(C, C_aligned).to(x.device)
                else:
                    # 如果适配层已存在但维度不对，重新创建
                    if self._dim_adapter.out_features != C_aligned:
                        self._dim_adapter = nn.Linear(C, C_aligned).to(x.device)
                x = self._dim_adapter(x)
                C = C_aligned
        
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算attention scores，添加数值稳定性
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 数值稳定性：clamp attention scores防止溢出
        attn = torch.clamp(attn, min=-50.0, max=50.0)
        
        # 使用稳定的softmax
        attn = attn.softmax(dim=-1)
        
        # 检查NaN/Inf
        if torch.any(torch.isnan(attn)) or torch.any(torch.isinf(attn)):
            # 如果出现NaN/Inf，使用均匀分布作为后备
            attn = torch.ones_like(attn) / N
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 最终检查输出
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            # 如果输出有NaN/Inf，返回零
            x = torch.zeros_like(x)
        
        return x


class Mlp(nn.Module):
    """MLP模块（数值稳定版本）"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化，防止梯度爆炸"""
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.02)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # 性能优化：移除频繁的NaN检查，只在必要时检查
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# window_partition 和 window_reverse 函数已移动到 utils.py


class GlobalAttention(nn.Module):
    """
    全局自注意力机制（类似ViT，用于nnFormer的GV-MSA）
    2D版本的全局体积自注意力
    """
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        """
        x: (B, N, C) where N = H*W
        优化：对于大N，使用分块计算减少内存占用
        """
        B, N, C = x.shape
        head_dim = C // self.num_heads
        
        # 优化：对于大N（>4096），使用分块计算减少内存
        if N > 4096:
            # 分块计算QKV
            chunk_size = 1024
            qkv_chunks = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk = x[:, i:end_idx, :]
                qkv_chunk = self.qkv(chunk).reshape(B, end_idx - i, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                qkv_chunks.append(qkv_chunk)
            
            # 合并chunks
            q = torch.cat([chunk[0] for chunk in qkv_chunks], dim=2)
            k = torch.cat([chunk[1] for chunk in qkv_chunks], dim=2)
            v = torch.cat([chunk[2] for chunk in qkv_chunks], dim=2)
        else:
            # 小N时使用原始方法
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        # 全局注意力：所有位置之间计算注意力
        # 优化：对于大N，使用分块计算注意力矩阵
        if N > 4096:
            # 分块计算注意力
            chunk_size = 512
            out_chunks = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                q_chunk = q[:, :, i:end_idx, :]  # (B, num_heads, chunk_size, head_dim)
                
                # 计算注意力分数
                attn_chunk = (q_chunk @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, chunk_size, N)
                attn_chunk = torch.clamp(attn_chunk, min=-50.0, max=50.0)
                attn_chunk = attn_chunk.softmax(dim=-1)
                attn_chunk = self.attn_drop(attn_chunk)
                
                # 应用注意力
                out_chunk = (attn_chunk @ v).transpose(1, 2)  # (B, chunk_size, num_heads, head_dim)
                out_chunks.append(out_chunk)
            
            x = torch.cat(out_chunks, dim=1).reshape(B, N, C)
        else:
            # 小N时使用原始方法
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = torch.clamp(attn, min=-50.0, max=50.0)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., mlp_hidden_dim=None,
                 drop_path_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 使用更大的eps提高LayerNorm的数值稳定性
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        # 如果提供了精确的mlp_hidden_dim，使用它；否则从mlp_ratio计算
        if mlp_hidden_dim is None:
            mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b or pad_r:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)
        H_pad, W_pad = x.shape[1], x.shape[2]

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_b or pad_r:
            x = x[:, :H, :W, :]
        x = x.reshape(B, H * W, C)

        # 残差连接，添加数值稳定性检查
        residual = self.drop_path(x)
        if torch.any(torch.isnan(residual)) or torch.any(torch.isinf(residual)):
            residual = torch.zeros_like(residual)
        
        x = shortcut + residual
        
        # 检查输出
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            # 如果残差连接导致NaN，只使用shortcut
            x = shortcut
        # MLP残差连接，添加数值稳定性检查
        shortcut2 = x
        x = self.norm2(x)
        
        # 检查norm2输出
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            x = shortcut2
        
        mlp_out = self.mlp(x)
        residual2 = self.drop_path(mlp_out)
        
        # 检查残差
        if torch.any(torch.isnan(residual2)) or torch.any(torch.isinf(residual2)):
            residual2 = torch.zeros_like(residual2)
        
        x = shortcut2 + residual2
        
        # 最终检查
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            x = shortcut2

        return x


class PatchMerging(nn.Module):
    """Patch Merging层，用于下采样"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"
        x = x.view(B, H, W, C)

        # 下采样：将2x2的patch合并为1个
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        
        # 处理奇数尺寸：确保所有张量具有相同的空间维度
        # 获取所有张量的最小高度和宽度
        H_new = min(x0.shape[1], x1.shape[1], x2.shape[1], x3.shape[1])
        W_new = min(x0.shape[2], x1.shape[2], x2.shape[2], x3.shape[2])
        
        # 裁剪所有张量到相同尺寸
        x0 = x0[:, :H_new, :W_new, :]
        x1 = x1[:, :H_new, :W_new, :]
        x2 = x2[:, :H_new, :W_new, :]
        x3 = x3[:, :H_new, :W_new, :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """图像到Patch Embedding"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B embed_dim H' W'
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B H'*W' embed_dim
        x = self.norm(x)
        return x, H, W


# ==================== SwinUNet 架构 ====================
class SwinUNet(nn.Module):
    """
    SwinUNet: 结合Swin Transformer和UNet的混合架构
    使用GWO优化超参数以提高Dice指标
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 img_size=224, patch_size=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.0, use_attention_gate=True, _mlp_hidden_dims=None, _from_checkpoint=False):
        super(SwinUNet, self).__init__()
        
        self.num_layers = len(depths)
        # 如果来自checkpoint，不归一化embed_dim
        if _from_checkpoint:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = self._normalize_embed_dim(embed_dim)
        embed_dim = self.embed_dim
        self.use_attention_gate = use_attention_gate
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.base_window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self._mlp_hidden_dims = _mlp_hidden_dims  # 精确的mlp hidden dims
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, 
                                     in_chans=in_channels, embed_dim=embed_dim)
        grid_h = self.patch_embed.grid_size[0]
        if _from_checkpoint:
            self.window_size = window_size
        else:
            self.window_size = self._normalize_window_size(window_size, grid_h)
        total_blocks = sum(self.depths)
        if drop_path_rate > 0 and total_blocks > 0:
            self.drop_path_rates = np.linspace(0, drop_path_rate, total_blocks).tolist()
        else:
            self.drop_path_rates = [0.0] * max(1, total_blocks)
        dp_index = 0
        
        # 编码器：Swin Transformer Blocks
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else nn.Identity
            
            # Swin Transformer Blocks
            for i_block in range(depths[i_layer]):
                block_drop_path = self.drop_path_rates[dp_index] if self.drop_path_rates else 0.0
                dp_index += 1
                # 获取精确的mlp_hidden_dim（如果有）
                mlp_hidden = None
                if _mlp_hidden_dims and (i_layer, i_block) in _mlp_hidden_dims:
                    mlp_hidden = _mlp_hidden_dims[(i_layer, i_block)]
                
                layer.append(
                    SwinTransformerBlock(
                        dim=int(embed_dim * 2 ** i_layer),
                        num_heads=num_heads[i_layer],
                        window_size=self.window_size,
                        shift_size=0 if (i_block % 2 == 0) else self.window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        mlp_hidden_dim=mlp_hidden,
                        drop_path_rate=block_drop_path,
                    )
                )
            self.encoder_layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(downsample(int(embed_dim * 2 ** i_layer)))
            else:
                self.downsample_layers.append(nn.Identity())
        
        # 解码器：上采样 + CNN
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i))
            out_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
            
            # 上采样
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
            
            # 卷积块
            self.decoder_layers.append(nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 添加最后一层解码器（从embed_dim到最终输出前的特征）
        # 这一层使用第一个编码器特征（最浅层）作为跳跃连接
        # 将第一个编码器特征从embed_dim降到embed_dim//2以匹配
        self.first_encoder_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_decoder_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_decoder_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),  # embed_dim//2 + embed_dim//2 = embed_dim
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # 注意力门控（可选）- 现在有num_layers层（包括最后一层）
        if use_attention_gate:
            self.att_gates = nn.ModuleList()
            # 前num_layers-1层：对应解码器的上采样层
            for i in range(self.num_layers - 1):
                dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                self.att_gates.append(AttentionGate(dim, dim, dim // 2, temperature=0.5, use_sharpen=True))
            # 最后一层：x是embed_dim//2，skip是embed_dim（投影后变成embed_dim//2）
            self.att_gates.append(AttentionGate(embed_dim // 2, embed_dim // 2, embed_dim // 4, temperature=0.5, use_sharpen=True))
        
        # 最终输出层（现在输入是embed_dim//2，因为最后一层解码器输出embed_dim//2）
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )

    @staticmethod
    def _normalize_window_size(window_size, max_grid):
        max_grid = max(2, int(max_grid))
        window = max(2, int(round(window_size)))
        valid_sizes = [d for d in range(2, max_grid + 1) if max_grid % d == 0]
        if not valid_sizes:
            return min(window, max_grid)
        best = min(valid_sizes, key=lambda d: abs(d - window))
        return best
    
    @staticmethod
    def _normalize_embed_dim(embed_dim):
        embed_dim = max(48, int(round(embed_dim / 6) * 6))
        if embed_dim % 3 != 0:
            embed_dim += (3 - embed_dim % 3)
        return embed_dim
    
    def get_config(self):
        return {
            "in_channels": self.patch_embed.proj.in_channels,
            "out_channels": self.final_conv[-1].out_channels if isinstance(self.final_conv[-1], nn.Conv2d) else 1,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "use_attention_gate": self.use_attention_gate
        }
        
    def forward(self, x, return_attention=False, return_aux=False):
        B = x.shape[0]
        original_size = x.shape[2:]
        
        # 编码器
        x, H, W = self.patch_embed(x)
        encoder_features = []
        
        for i_layer, (layer, downsample) in enumerate(zip(self.encoder_layers, self.downsample_layers)):
            # Swin Transformer Blocks
            for block in layer:
                x = block(x, H, W)
            
            # 保存特征用于跳跃连接
            x_reshaped = x.transpose(1, 2).reshape(B, -1, H, W)
            encoder_features.append(x_reshaped)
            
            # 下采样
            if i_layer < self.num_layers - 1:
                x = downsample(x, H, W)
                H, W = H // 2, W // 2
        
        # 解码器
        attention_maps = {}
        for i in range(self.num_layers - 1):
            # 上采样
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            x = self.decoder_layers[i * 2](x)  # 上采样
            H, W = x.shape[2], x.shape[3]
            
            # 跳跃连接
            skip_feature = encoder_features[self.num_layers - 2 - i]
            skip_feature = F.interpolate(skip_feature, size=(H, W), mode='bilinear', align_corners=False)
            
            # 注意力门控
            if self.use_attention_gate:
                if return_attention:
                    skip_feature, att_map = self.att_gates[i](x, skip_feature, return_attention=True)
                    attention_maps[f'att{i+1}'] = att_map
                else:
                    skip_feature = self.att_gates[i](x, skip_feature)
            
            # 拼接和卷积
            x = torch.cat([x, skip_feature], dim=1)
            x = self.decoder_layers[i * 2 + 1](x)  # 卷积块
        
        # 最后一层解码器：使用第一个编码器特征（最浅层）
        # x在循环结束后应该是空间格式(B, embed_dim, H, W)，因为最后一步是卷积块
        # 但为安全起见，检查并确保是空间格式
        if len(x.shape) == 4:
            # 已经是空间格式，直接使用
            pass
        elif len(x.shape) == 3:
            # 序列格式(B, H*W, C)，需要转换
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        else:
            # 异常情况，尝试reshape
            try:
                x = x.view(B, -1, H, W)
            except:
                raise RuntimeError(f"无法处理x的形状: {x.shape}")
        
        # 现在x应该是空间格式(B, embed_dim, H, W)
        x = self.final_decoder_upsample(x)  # 上采样到embed_dim//2
        H, W = x.shape[2], x.shape[3]
        
        # 跳跃连接：使用第一个编码器特征
        skip_feature = encoder_features[0]  # 第一个编码器特征（embed_dim维度）
        skip_feature = F.interpolate(skip_feature, size=(H, W), mode='bilinear', align_corners=False)
        skip_feature = self.first_encoder_proj(skip_feature)  # 投影到embed_dim//2
        
        # 最后一层注意力门控
        if self.use_attention_gate:
            if return_attention:
                skip_feature, att_map = self.att_gates[self.num_layers - 1](x, skip_feature, return_attention=True)
                attention_maps[f'att{self.num_layers}'] = att_map
            else:
                skip_feature = self.att_gates[self.num_layers - 1](x, skip_feature)
        
        # 拼接和卷积
        x = torch.cat([x, skip_feature], dim=1)  # embed_dim//2 + embed_dim//2 = embed_dim
        x = self.final_decoder_conv(x)  # 输出embed_dim//2
        
        # 最终输出
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        final_output = self.final_conv(x)
        
        # 返回格式
        if return_attention and return_aux:
            return final_output, [], attention_maps
        elif return_attention:
            return final_output, attention_maps
        elif return_aux:
            return final_output, []
        else:
            return final_output


# ==================== GWO (灰狼优化算法) ====================
class GWOOptimizer:
    """
    灰狼优化算法 (Grey Wolf Optimizer)
    用于优化SwinUNet的超参数以提高Dice指标
    """
    def __init__(self, n_wolves=30, max_iter=50, 
                 bounds=None, objective_func=None):
        """
        Args:
            n_wolves: 灰狼数量
            max_iter: 最大迭代次数
            bounds: 参数边界字典，格式: {'param_name': (min, max), ...}
            objective_func: 目标函数，输入参数字典，返回Dice分数
        """
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.bounds = bounds or self._default_bounds()
        self.objective_func = objective_func
        
        # 灰狼位置（参数值）
        self.positions = []
        # 灰狼适应度（Dice分数）
        self.fitness = []
        
        # Alpha, Beta, Delta (前三名)
        self.alpha_pos = None
        self.alpha_score = float('-inf')
        self.beta_pos = None
        self.beta_score = float('-inf')
        self.delta_pos = None
        self.delta_score = float('-inf')
        
    def _default_bounds(self):
        """默认参数边界"""
        return {
            'embed_dim': (64, 128),
            'window_size': (4, 12),
            'mlp_ratio': (2.0, 6.0),
            'drop_rate': (0.1, 0.4),  # 小数据集更高dropout
            'attn_drop_rate': (0.1, 0.4),
        }
    
    def _initialize_wolves(self):
        """初始化灰狼位置"""
        self.positions = []
        for _ in range(self.n_wolves):
            pos = {}
            for param, (min_val, max_val) in self.bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    pos[param] = np.random.randint(min_val, max_val + 1)
                else:
                    pos[param] = np.random.uniform(min_val, max_val)
            self.positions.append(pos)
    
    def _evaluate_fitness(self, position):
        """评估适应度（Dice分数）"""
        try:
            if self.objective_func:
                result = self.objective_func(position)
                # 每次评估后彻底清理GPU缓存和内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # 强制垃圾回收
                import gc
                gc.collect()
                return result
            return 0.0
        except Exception as e:
            print(f"适应度评估错误: {e}")
            # 确保即使出错也清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            return 0.0
    
    def _update_alpha_beta_delta(self):
        """更新Alpha, Beta, Delta"""
        # 如果还没有初始化，先找到前三名
        if self.alpha_pos is None:
            # 找到最佳适应度的索引
            sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i], reverse=True)
            if len(sorted_indices) > 0:
                self.alpha_score = self.fitness[sorted_indices[0]]
                self.alpha_pos = self.positions[sorted_indices[0]].copy()
            if len(sorted_indices) > 1:
                self.beta_score = self.fitness[sorted_indices[1]]
                self.beta_pos = self.positions[sorted_indices[1]].copy()
            if len(sorted_indices) > 2:
                self.delta_score = self.fitness[sorted_indices[2]]
                self.delta_pos = self.positions[sorted_indices[2]].copy()
            return
        
        # 更新Alpha, Beta, Delta
        for i, fitness in enumerate(self.fitness):
            if fitness > self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy() if self.beta_pos else None
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy() if self.alpha_pos else None
                self.alpha_score = fitness
                self.alpha_pos = self.positions[i].copy()
            elif fitness > self.beta_score and fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy() if self.beta_pos else None
                self.beta_score = fitness
                self.beta_pos = self.positions[i].copy()
            elif fitness > self.delta_score and fitness < self.beta_score:
                self.delta_score = fitness
                self.delta_pos = self.positions[i].copy()
    
    def _update_position(self, a):
        """更新灰狼位置"""
        # 检查alpha_pos, beta_pos, delta_pos是否已初始化
        if self.alpha_pos is None or self.beta_pos is None or self.delta_pos is None:
            return  # 如果还未初始化，跳过位置更新
        
        for i in range(self.n_wolves):
            for param in self.bounds.keys():
                # 计算Alpha, Beta, Delta的影响
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[param] - self.positions[i][param])
                X1 = self.alpha_pos[param] - A1 * D_alpha
                
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos[param] - self.positions[i][param])
                X2 = self.beta_pos[param] - A2 * D_beta
                
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos[param] - self.positions[i][param])
                X3 = self.delta_pos[param] - A3 * D_delta
                
                # 更新位置
                new_pos = (X1 + X2 + X3) / 3.0
                
                # 边界约束
                min_val, max_val = self.bounds[param]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    new_pos = int(np.clip(new_pos, min_val, max_val))
                else:
                    new_pos = np.clip(new_pos, min_val, max_val)
                
                self.positions[i][param] = new_pos
    
    def optimize(self, callback=None):
        """
        执行优化
        Args:
            callback: 回调函数，每轮迭代后调用 callback(iter, best_score, best_params)
        Returns:
            best_params: 最佳参数
            best_score: 最佳分数
            history: 优化历史
        """
        # 初始化
        self._initialize_wolves()
        
        # 评估初始适应度
        self.fitness = [self._evaluate_fitness(pos) for pos in self.positions]
        self._update_alpha_beta_delta()
        
        history = []
        
        # 迭代优化
        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)  # 线性递减
            
            # 更新位置
            self._update_position(a)
            
            # 评估适应度（每次评估后会自动清理GPU缓存）
            self.fitness = [self._evaluate_fitness(pos) for pos in self.positions]
            self._update_alpha_beta_delta()
            
            # 每次迭代后强制清理GPU缓存和内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 记录历史
            history.append({
                'iteration': iter + 1,
                'best_score': self.alpha_score,
                'best_params': self.alpha_pos.copy() if self.alpha_pos else {}
            })
            
            # 回调
            if callback and self.alpha_pos:
                callback(iter + 1, self.alpha_score, self.alpha_pos.copy())
        
        # 确保返回有效的参数
        if self.alpha_pos is None:
            # 如果优化失败，返回默认参数
            default_params = {}
            for param, (min_val, max_val) in self.bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    default_params[param] = (min_val + max_val) // 2
                else:
                    default_params[param] = (min_val + max_val) / 2.0
            return default_params, self.alpha_score, history
        
        return self.alpha_pos, self.alpha_score, history


class SkullStripUNet(nn.Module):
    """
    轻量级UNet，用于脑组织mask预测（Skull Stripping）。
    与原brain-segmentation仓库的Keras实现保持相同结构(无BN)以便权重互通。
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = self._double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.center = self._double_conv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        d4 = self.up4(center)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


class SkullStripper:
    """
    Skull Stripping 推理封装器。
    """
    def __init__(self, model_path: Optional[str], device, threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model()
        elif model_path:
            print(f"[警告] SkullStripper模型路径不存在: {model_path}，将跳过剥除颅骨步骤。")

    def _load_model(self):
        self.model = SkullStripUNet().to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def is_available(self):
        return self.model is not None

    @torch.no_grad()
    def strip(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.is_available():
            return images, None
        logits = self.model(images)
        probs = torch.sigmoid(logits)
        if self.threshold is not None:
            mask = (probs > self.threshold).float()
        else:
            mask = probs
        stripped = images * mask
        return stripped, mask


class ChannelAttention(nn.Module):
    """CBAM: Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 降维比例 ratio
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM: Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """结合通道和空间注意力，放在Skip Connection处"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class ASPP(nn.Module):
    """ASPP模块：扩大感受野，捕获多尺度上下文"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        
        # 不同膨胀率的空洞卷积
        dilations = [6, 12, 18]
        for rate in dilations:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        
        self.convs = nn.ModuleList(modules)
        # 融合层
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """优化后的解码块：上采样 + 拼接 + 卷积"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 这里使用插值代替转置卷积，减少棋盘效应
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 这是一个针对Skip Connection的各种预处理
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 1), # 可以在这里降维
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = CBAM(skip_channels) # 对跳跃连接特征应用注意力
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout防止过拟合
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        
        # 确保尺寸匹配 (处理输入图像尺寸不是32倍数的情况)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        skip = self.skip_conv(skip)
        skip = self.attention(skip) # 关键：让网络知道该关注Skip Connection中的什么信息
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """改进的ResNetUNet：集成ASPP、CBAM注意力和双线性上采样"""
    def __init__(self, in_channels=3, out_channels=1, pretrained=True, backbone_name='resnet101', use_aspp=True, freeze_encoder=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone_name = backbone_name
        self.use_aspp = use_aspp
        self.freeze_encoder = freeze_encoder
        
        # 1. 骨干网络加载
        if backbone_name == 'resnet101':
            if pretrained:
                try:
                    base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                except Exception as e:
                    print(f"[警告] ResNet101 预训练权重下载失败，改用随机初始化。错误: {e}")
                    base = models.resnet101(weights=None)
            else:
                base = models.resnet101(weights=None)
            filters = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet50':
            if pretrained:
                try:
                    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                except Exception as e:
                    print(f"[警告] ResNet50 预训练权重下载失败，改用随机初始化。错误: {e}")
                    base = models.resnet50(weights=None)
            else:
                base = models.resnet50(weights=None)
            filters = [256, 512, 1024, 2048]
        else:
            # 默认使用 resnet101
            if pretrained:
                try:
                    base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                except Exception as e:
                    print(f"[警告] ResNet101 预训练权重下载失败，改用随机初始化。错误: {e}")
                    base = models.resnet101(weights=None)
            else:
                base = models.resnet101(weights=None)
            filters = [256, 512, 1024, 2048]
        
        # 处理输入通道
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 编码器层
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu) # 64
        self.pool = base.maxpool
        self.enc1 = base.layer1 # 256
        self.enc2 = base.layer2 # 512
        self.enc3 = base.layer3 # 1024
        self.enc4 = base.layer4 # 2048
        
        # 2. 中间层：ASPP 或 简单的降维卷积
        if use_aspp:
            # 使用ASPP：将2048通道压缩到512，同时扩大感受野
            self.aspp = ASPP(filters[3], 512)
            center_out_channels = 512
        else:
            # 旧版本：简单的1x1卷积降维
            self.center = nn.Sequential(
                nn.Conv2d(filters[3], 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            center_out_channels = 512
        
        # 3. 解码器 (带有注意力机制)
        # 解码器通道数设计：中间层输出512 -> Concat enc3(1024) -> ...
        self.dec4 = DecoderBlock(center_out_channels, filters[2], 256)  # in:512, skip:1024, out:256
        self.dec3 = DecoderBlock(256, filters[1], 128)  # in:256, skip:512, out:128
        self.dec2 = DecoderBlock(128, filters[0], 64)   # in:128, skip:256, out:64
        self.dec1 = DecoderBlock(64, 64, 32)            # in:64,  skip:64 (enc0), out:32
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # 4. 冻结编码器（如果启用）
        if self.freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """冻结编码器层（enc0 到 enc4）"""
        for encoder in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in encoder.parameters():
                param.requires_grad = False
        print("[模型] 已冻结 ResNet 编码器（enc0-enc4），仅训练解码器")
    
    def _unfreeze_encoder(self):
        """解冻编码器层，允许微调"""
        for encoder in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in encoder.parameters():
                param.requires_grad = True
        print("[模型] 已解冻 ResNet 编码器，开始端到端微调")

    def forward(self, x, return_attention=False):
        # Encoder
        x0 = self.enc0(x)      # 1/2
        x_pool = self.pool(x0) # 1/4
        x1 = self.enc1(x_pool) # 1/4, 256
        x2 = self.enc2(x1)     # 1/8, 512
        x3 = self.enc3(x2)     # 1/16, 1024
        x4 = self.enc4(x3)     # 1/32, 2048

        # Center (ASPP 或 简单降维)
        if self.use_aspp:
            center = self.aspp(x4) # 1/32, 512
        else:
            center = self.center(x4) # 1/32, 512

        # Decoder with Attention
        d4 = self.dec4(center, x3) # -> 1/16
        d3 = self.dec3(d4, x2)     # -> 1/8
        d2 = self.dec2(d3, x1)     # -> 1/4
        d1 = self.dec1(d2, x0)     # -> 1/2
        
        # Final Upsampling to recover original size if needed
        # 通常 U-Net 输出是 1/2 或 1/1，这里如果不加最后的上采样，输出是输入尺寸的 1/2
        # 我们直接对 logit 进行双线性插值恢复原图尺寸
        out = self.final(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        
        if return_attention:
            # 使用各层特征图的通道平均作为注意力图
            # 将特征图下采样到相同尺寸以便可视化
            attention_maps = {}
            target_size = (256, 256)  # 目标尺寸
            
            # 对 x1, x2, x3, x4 进行下采样并计算通道平均
            def get_attention_map(feat, name):
                # 计算通道平均
                attn = feat.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                # 下采样到目标尺寸
                if attn.shape[2:] != target_size:
                    attn = F.interpolate(attn, size=target_size, mode='bilinear', align_corners=False)
                # 归一化到 [0, 1]
                attn_min = attn.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                attn_max = attn.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)
                return attn
            
            # 获取各层的注意力图（对应不同尺度的特征）
            attention_maps['att1'] = get_attention_map(x1, 'att1')  # 最精细层
            attention_maps['att2'] = get_attention_map(x2, 'att2')
            attention_maps['att3'] = get_attention_map(x3, 'att3')
            attention_maps['att4'] = get_attention_map(x4, 'att4')  # 最深层
            
            return out, attention_maps
        
        return out


class nnFormer(nn.Module):
    """
    nnFormer: 基于Transformer的医学图像分割模型（2D适配版）
    结合局部窗口注意力（LV-MSA）和全局注意力（GV-MSA），交替使用以学习多尺度特征
    在跳跃连接中使用Skip Attention机制，增强特征融合
    参考: "nnFormer: Interleaved Transformer for Volumetric Segmentation" (MICCAI 2021)
    
    核心改进：
    1. 局部-全局交替注意力：在编码器中交替使用窗口注意力和全局注意力
    2. 跳跃注意力机制：使用自注意力融合编码器和解码器特征
    3. 多尺度特征学习：通过不同尺度的注意力机制捕获局部和全局信息
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 img_size=224, patch_size=4,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, use_skip_attention=True,
                 global_attn_ratio=0.5):
        super(nnFormer, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_skip_attention = use_skip_attention
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.global_attn_ratio = global_attn_ratio  # 全局注意力的比例
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, 
                                     in_chans=in_channels, embed_dim=embed_dim)
        grid_h = self.patch_embed.grid_size[0]
        
        # Drop path rates
        total_blocks = sum(self.depths)
        if drop_path_rate > 0 and total_blocks > 0:
            self.drop_path_rates = np.linspace(0, drop_path_rate, total_blocks).tolist()
        else:
            self.drop_path_rates = [0.0] * max(1, total_blocks)
        dp_index = 0
        
        # 编码器：nnFormer Transformer Blocks（交替使用局部和全局注意力）
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else nn.Identity
            
            # nnFormer Blocks: 交替使用局部窗口注意力和全局注意力
            for i_block in range(depths[i_layer]):
                block_drop_path = self.drop_path_rates[dp_index] if self.drop_path_rates else 0.0
                dp_index += 1
                
                # 决定使用全局还是局部注意力
                # 性能优化：完全禁用全局注意力（除非明确需要），因为全局注意力非常慢
                # 全局注意力的计算复杂度是O(N²)，对于大图像（如256×256，N=65536）会非常慢
                # 只在最后一层的最后一个block使用全局注意力（如果global_attn_ratio > 0.5）
                use_global = False  # 默认禁用全局注意力以提高速度
                if self.global_attn_ratio > 0.5 and i_layer == self.num_layers - 1 and i_block == depths[i_layer] - 1:
                    use_global = True  # 只在明确需要时才启用
                
                layer.append(
                    nnFormerBlock(
                        dim=int(embed_dim * 2 ** i_layer),
                        num_heads=num_heads[i_layer],
                        window_size=self.window_size,
                        shift_size=0 if (i_block % 2 == 0) else self.window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path_rate=block_drop_path,
                        use_global_attn=use_global,
                    )
                )
            self.encoder_layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(downsample(int(embed_dim * 2 ** i_layer)))
            else:
                self.downsample_layers.append(nn.Identity())
        
        # 解码器：上采样 + CNN
        # 如果使用SkipAttention，不需要拼接，所以卷积层输入是out_dim
        # 如果使用传统方式，需要拼接，所以卷积层输入是out_dim * 2
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i))
            out_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
            
            # 上采样
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
            
            # 卷积块：根据是否使用SkipAttention决定输入通道数
            # 如果使用SkipAttention，x已经融合了skip，所以输入是out_dim
            # 如果使用传统方式，需要拼接skip，所以输入是out_dim * 2
            conv_input_dim = out_dim if use_skip_attention else out_dim * 2
            self.decoder_layers.append(nn.Sequential(
                nn.Conv2d(conv_input_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 最后一层解码器
        self.first_encoder_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_decoder_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        # 最后一层卷积：根据是否使用SkipAttention决定输入通道数
        final_conv_input_dim = embed_dim // 2 if use_skip_attention else embed_dim
        self.final_decoder_conv = nn.Sequential(
            nn.Conv2d(final_conv_input_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )

        # 跳跃注意力机制（nnFormer风格）
        if use_skip_attention:
            self.skip_attentions = nn.ModuleList()
            for i in range(self.num_layers - 1):
                dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                # 确保num_heads至少为1，且不超过dim
                num_heads_val = max(1, min(8, dim // 64))
                self.skip_attentions.append(SkipAttention(dim, num_heads=num_heads_val))
            # 最后一层
            final_dim = embed_dim // 2
            final_num_heads = max(1, min(8, final_dim // 64))
            self.skip_attentions.append(SkipAttention(final_dim, num_heads=final_num_heads))
        else:
            # 回退到传统注意力门控
            self.att_gates = nn.ModuleList()
            for i in range(self.num_layers - 1):
                dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                self.att_gates.append(AttentionGate(dim, dim, dim // 2, temperature=0.5, use_sharpen=True))
            self.att_gates.append(AttentionGate(embed_dim // 2, embed_dim // 2, embed_dim // 4, temperature=0.5, use_sharpen=True))
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )

    def get_config(self):
        """返回nnFormer的配置"""
        return {
            "in_channels": self.patch_embed.proj.in_channels,
            "out_channels": self.final_conv[-1].out_channels if isinstance(self.final_conv[-1], nn.Conv2d) else 1,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "use_skip_attention": self.use_skip_attention,
            "global_attn_ratio": self.global_attn_ratio
        }
        
    def forward(self, x, return_attention=False, return_aux=False):
        B = x.shape[0]
        original_size = x.shape[2:]
        
        # 编码器 - 优化：使用detach()减少内存占用，只在需要时保存特征
        x, H, W = self.patch_embed(x)
        encoder_features = []
        
        for i_layer, (layer, downsample) in enumerate(zip(self.encoder_layers, self.downsample_layers)):
            # Transformer Blocks - 性能优化：减少中间变量
            for block in layer:
                x = block(x, H, W)
            
            # 保存特征用于跳跃连接 - 性能优化：只在训练时保留梯度，推理时detach
            x_reshaped = x.transpose(1, 2).reshape(B, -1, H, W)
            # 优化：训练时也使用detach，因为跳跃连接通常不需要梯度（可以节省大量内存）
            encoder_features.append(x_reshaped.detach())
            
            # 下采样
            if i_layer < self.num_layers - 1:
                x = downsample(x, H, W)
                H, W = H // 2, W // 2
                # 性能优化：减少清理频率，只在必要时清理
                if torch.cuda.is_available() and i_layer % 3 == 0:
                    torch.cuda.empty_cache()
        
        # 解码器 - 优化：及时释放不需要的特征
        x = encoder_features[-1]
        decoder_idx = 0
        
        for i in range(self.num_layers - 1):
            # 上采样
            x = self.decoder_layers[decoder_idx](x)
            decoder_idx += 1
            
            # 跳跃连接 - 只在需要时加载特征
            skip_idx = self.num_layers - 2 - i
            skip = encoder_features[skip_idx]
            
            # 确保skip和x的空间尺寸匹配（在调用SkipAttention之前）
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # 跳跃注意力机制（nnFormer风格）
            if self.use_skip_attention:
                x = self.skip_attentions[i](x, skip)
                skip = None  # 已经融合，不需要再拼接
            elif hasattr(self, 'att_gates'):
                # 回退到传统注意力门控
                skip = self.att_gates[i](x, skip)
            
            # 拼接和卷积（如果使用Skip Attention，x已经融合了skip）
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.decoder_layers[decoder_idx](x)
            decoder_idx += 1
            
            # 释放已使用的encoder特征（除了第一个和最后一个）
            if i < self.num_layers - 2:
                encoder_features[skip_idx] = None
        
        # 最后一层解码器
        x = self.final_decoder_upsample(x)
        first_skip = self.first_encoder_proj(encoder_features[0])
        
        # 确保first_skip和x的空间尺寸匹配
        if x.shape[2:] != first_skip.shape[2:]:
            first_skip = F.interpolate(first_skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if self.use_skip_attention:
            x = self.skip_attentions[-1](x, first_skip)
            first_skip = None
        elif hasattr(self, 'att_gates'):
            first_skip = self.att_gates[-1](x, first_skip)
        
        if first_skip is not None:
            x = torch.cat([x, first_skip], dim=1)
        x = self.final_decoder_conv(x)
        
        # 最终输出
        x = self.final_conv(x)
        
        # 恢复到原始尺寸
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # 清理encoder_features（如果不需要返回）
        if not return_aux:
            encoder_features = None
        
        if return_aux:
            return x, encoder_features
        return x

    def get_config(self):
        """返回nnFormer的配置"""
        return {
            "in_channels": self.patch_embed.proj.in_channels,
            "out_channels": self.final_conv[-1].out_channels if isinstance(self.final_conv[-1], nn.Conv2d) else 1,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "use_skip_attention": self.use_skip_attention,
            "global_attn_ratio": self.global_attn_ratio
        }


# 工具函数和数据处理类已移动到 utils.py

def instantiate_model(model_type: str, device, swin_params: Optional[dict] = None, dstrans_params: Optional[dict] = None, mamba_params: Optional[dict] = None, resnet_params: Optional[dict] = None):
    model_type = (model_type or "improved_unet").lower()
    if model_type == "resnet_unet":
        params = {
            "in_channels": 3,
            "out_channels": 1,
            # 使用 ImageNet 预训练的 ResNet，可提升深层特征表达（Attention 更稳定）
            "pretrained": True,
            "backbone_name": "resnet101",
            "use_aspp": True  # 默认使用ASPP
        }
        if resnet_params:
            params.update(resnet_params)
        return ResNetUNet(**params).to(device)
    if model_type in ("trans_unet", "transunet"):
        return TransUNet().to(device)
    if model_type in ("ds_trans_unet", "dstransunet", "ds-transunet"):
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        }
        if dstrans_params:
            params.update(dstrans_params)
        return DSTransUNet(**params).to(device)
    if model_type == "swin_unet" or model_type == "swinunet":
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "img_size": 224,
            "patch_size": 4,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 8,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
            "use_attention_gate": True
        }
        if swin_params:
            params.update(swin_params)
        return SwinUNet(**params).to(device)
    return ImprovedUNet().to(device)


class LayerNorm2d(nn.Module):
    """轻量2D LayerNorm，兼容卷积特征。"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, eps=eps)
    def forward(self, x):
        return self.norm(x)


class MambaBlock2D(nn.Module):
    """占位：Swin-U Mamba 相关代码已移除，此模块不再使用。"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("MambaBlock2D 已废弃，Swin-U Mamba 功能已删除")


class WindowAttention2D(nn.Module):
    """局部窗口自注意力，输入/输出形状 B,C,H,W"""
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = max(1, dim // num_heads)
        self.num_heads = max(1, dim // head_dim)
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        # [B, C, Hp, Wp] -> [B*num_windows, ws*ws, C]
        x_windows = x.unfold(2, ws, ws).unfold(3, ws, ws)  # B, C, nH, nW, ws, ws
        x_windows = x_windows.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, ws * ws, C)

        qkv = self.qkv(x_windows).reshape(x_windows.shape[0], x_windows.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, Bnw, heads, tokens, dim_head
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], C)
        out = self.proj(out)

        # [B*num_windows, ws*ws, C] -> [B, C, Hp, Wp]
        out = out.view(-1, ws, ws, C)
        # reshape back: need nH, nW
        nH = Hp // ws
        nW = Wp // ws
        out = out.view(-1, nH, nW, ws, ws, C).permute(0, 5, 3, 4, 1, 2).contiguous()
        out = out.view(B, C, ws, ws, nH, nW).permute(0, 1, 4, 2, 5, 3).contiguous()
        out = out.view(B, C, Hp, Wp)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


class SwinBlock2D(nn.Module):
    """
    简化版 Swin block：可选移窗 + 窗口注意力 + MLP
    增加通道降维（attn_reduction）以降低注意力计算量。
    """
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7, shift: bool = False,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_reduction: int = 2):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = LayerNorm2d(dim)
        attn_dim = max(8, dim // max(1, attn_reduction))
        # 使 attn_dim 可被 num_heads 整除
        attn_dim = max(attn_dim // num_heads, 1) * num_heads
        self.attn_in = nn.Conv2d(dim, attn_dim, kernel_size=1) if attn_dim != dim else nn.Identity()
        self.attn = WindowAttention2D(attn_dim, window_size, num_heads)
        self.attn_out = nn.Conv2d(attn_dim, dim, kernel_size=1) if attn_dim != dim else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
        )
        self.norm2 = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        x_attn = self.attn_in(x)
        x = self.attn(x_attn)
        x = self.attn_out(x)
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        x = shortcut + x
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class SwinUMamba(nn.Module):
    """占位：Swin-U Mamba 相关模型已移除，不再支持。"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("Swin-U Mamba 模型已从系统中删除")
# ==================== 模型实例化工厂函数 ====================

def instantiate_model(model_type: str, device, swin_params=None, dstrans_params=None, mamba_params=None, resnet_params=None):
    """
    实例化模型的工厂函数
    根据 model_type 创建对应的模型实例
    """
    model_type = (model_type or "improved_unet").lower()
    
    if model_type == "resnet_unet":
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "pretrained": True,
            "backbone_name": "resnet101",
            "use_aspp": True
        }
        if resnet_params:
            params.update(resnet_params)
        return ResNetUNet(**params).to(device)
        
    if model_type in ("trans_unet", "transunet"):
        return TransUNet().to(device)
        
    if model_type in ("ds_trans_unet", "dstransunet", "ds-transunet"):
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        }
        if dstrans_params:
            params.update(dstrans_params)
        return DSTransUNet(**params).to(device)
        
    if model_type == "swin_unet" or model_type == "swinunet":
        params = {
            "in_channels": 3,
            "out_channels": 1,
            "img_size": 224,
            "patch_size": 4,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 8,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
            "use_attention_gate": True
        }
        if swin_params:
            params.update(swin_params)
        return SwinUNet(**params).to(device)
    
    if model_type == "nnformer":
        return nnFormer(in_channels=3, out_channels=1).to(device)
        
    # 默认返回 ImprovedUNet
    return ImprovedUNet().to(device)