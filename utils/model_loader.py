# -*- coding: utf-8 -*-
"""
从utils.model_loader模块
"""
from utils.common import *

# ==================== 模型加载函数 ====================

def load_ensemble_models(*args, **kwargs):
    """模型集成功能已取消，调用将报错。"""
    raise RuntimeError("模型集成功能已取消")


def load_model_compatible(model, checkpoint_path, device, verbose=True, target_model_type=None):
    """
    兼容加载模型权重的工具函数，自动处理新旧版本结构差异
    
    Args:
        model: 要加载权重的模型实例
        checkpoint_path: 模型权重文件路径
        device: 设备
        verbose: 是否打印信息
        target_model_type: 目标模型类型（可选），用于跨架构权重迁移（如从 ResNet-UNet 到 DeepLabV3+）
    
    Returns:
        (success: bool, message: str)
    """
    try:
        loaded_obj = torch.load(checkpoint_path, map_location=device)
        
        # 【跨架构权重迁移】检测 checkpoint 的源架构
        source_model_type = None
        if isinstance(loaded_obj, dict):
            if 'config' in loaded_obj and isinstance(loaded_obj['config'], dict):
                source_model_type = loaded_obj['config'].get('model_type')
            elif 'model_type' in loaded_obj:
                source_model_type = loaded_obj['model_type']
        
        # 如果无法从 config 推断，尝试从 state_dict 键名推断
        if source_model_type is None:
            if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
                temp_state_dict = loaded_obj['state_dict']
            else:
                temp_state_dict = loaded_obj
            
            # 检测 ResNet-UNet 的特征键名
            resnet_unet_keys = ['enc0.', 'enc1.', 'enc2.', 'enc3.', 'enc4.', 'layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.']
            has_resnet_unet_keys = any(any(k.startswith(key) for k in temp_state_dict.keys()) for key in resnet_unet_keys)
            if has_resnet_unet_keys:
                # 检查是否有解码器键名（UNet 特有的）
                unet_decoder_keys = ['dec1.', 'dec2.', 'dec3.', 'dec4.', 'up0.', 'up1.', 'up2.', 'up3.', 'up4.', 'up_conv0.', 'up_conv1.', 'up_conv2.', 'up_conv3.', 'up_conv4.']
                has_unet_decoder = any(any(k.startswith(key) for k in temp_state_dict.keys()) for key in unet_decoder_keys)
                if has_unet_decoder:
                    source_model_type = 'resnet_unet'
        
        # 【跨架构权重迁移】如果源架构是 ResNet-UNet，目标架构是 DeepLabV3+，只加载编码器权重
        if source_model_type and target_model_type:
            source_lower = source_model_type.lower()
            target_lower = target_model_type.lower()
            
            if 'resnet_unet' in source_lower and ('deeplabv3plus' in target_lower or 'smp_deeplabv3plus' in target_lower):
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"🔄 [跨架构权重迁移]")
                    print(f"   源架构: {source_model_type} (ResNet-UNet)")
                    print(f"   目标架构: {target_model_type} (DeepLabV3+)")
                    print(f"   策略: 只加载编码器 (Encoder) 权重，丢弃解码器 (Decoder) 权重")
                    print(f"{'='*60}\n")
        
        if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
            state_dict = loaded_obj['state_dict']
        else:
            state_dict = loaded_obj
        
        # 处理DataParallel包装
        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 【跨架构权重迁移】如果源架构是 ResNet-UNet，目标架构是 DeepLabV3+，过滤掉解码器权重
        encoder_only_state_dict = None
        if source_model_type and target_model_type:
            source_lower = source_model_type.lower()
            target_lower = target_model_type.lower()
            
            if 'resnet_unet' in source_lower and ('deeplabv3plus' in target_lower or 'smp_deeplabv3plus' in target_lower):
                # 定义编码器键名模式（ResNet 编码器层）
                encoder_key_patterns = [
                    'enc0.', 'enc1.', 'enc2.', 'enc3.', 'enc4.',  # 新版本键名
                    'layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.',  # 旧版本键名
                    'encoder.',  # SMP模型的编码器键名
                ]
                
                # 定义解码器键名模式（UNet 解码器层，需要丢弃）
                decoder_key_patterns = [
                    'dec1.', 'dec2.', 'dec3.', 'dec4.',  # 新版本解码器
                    'up0.', 'up1.', 'up2.', 'up3.', 'up4.',  # 旧版本解码器
                    'up_conv0.', 'up_conv1.', 'up_conv2.', 'up_conv3.', 'up_conv4.',  # 旧版本解码器卷积
                    'center.',  # UNet的中心层
                    'final.',  # UNet的最终层
                    'segmentation_head.',  # 分割头（DeepLab有自己的分割头）
                ]
                
                # 过滤：只保留编码器权重，并映射键名到 DeepLabV3+ 格式
                encoder_only_state_dict = {}
                encoder_keys = []
                discarded_keys = []
                
                for k, v in state_dict.items():
                    # 检查是否是编码器键
                    is_encoder = any(k.startswith(pattern) for pattern in encoder_key_patterns)
                    # 检查是否是解码器键
                    is_decoder = any(k.startswith(pattern) for pattern in decoder_key_patterns)
                    
                    if is_encoder and not is_decoder:
                        # 映射 ResNet-UNet 编码器键名到 DeepLabV3+ 格式
                        # ResNet-UNet: enc0 (conv1+bn1+relu), enc1 (layer1), enc2 (layer2), enc3 (layer3), enc4 (layer4)
                        # DeepLabV3+ (SMP): model.encoder.conv1/bn1/relu, model.encoder.layer1, layer2, layer3, layer4
                        mapped_key = k
                        
                        # 处理 enc0: enc0 是 Sequential(conv1, bn1, relu)，需要映射到 model.encoder.conv1/bn1/relu
                        if k.startswith('enc0.'):
                            # enc0.0 -> model.encoder.conv1
                            # enc0.1 -> model.encoder.bn1
                            # enc0.2 -> model.encoder.relu (通常没有权重，但保留映射)
                            if k.startswith('enc0.0.'):
                                mapped_key = k.replace('enc0.0.', 'model.encoder.conv1.', 1)
                            elif k.startswith('enc0.1.'):
                                mapped_key = k.replace('enc0.1.', 'model.encoder.bn1.', 1)
                            elif k.startswith('enc0.2.'):
                                mapped_key = k.replace('enc0.2.', 'model.encoder.relu.', 1)
                            else:
                                # 如果 enc0 不是 Sequential，直接映射
                                mapped_key = k.replace('enc0.', 'model.encoder.conv1.', 1)
                        # 处理 enc1-enc4: 直接映射到 model.encoder.layer1-4
                        elif k.startswith('enc1.'):
                            mapped_key = k.replace('enc1.', 'model.encoder.layer1.', 1)
                        elif k.startswith('enc2.'):
                            mapped_key = k.replace('enc2.', 'model.encoder.layer2.', 1)
                        elif k.startswith('enc3.'):
                            mapped_key = k.replace('enc3.', 'model.encoder.layer3.', 1)
                        elif k.startswith('enc4.'):
                            mapped_key = k.replace('enc4.', 'model.encoder.layer4.', 1)
                        # 处理旧版本键名 (layer0 -> model.encoder.conv1/bn1/relu, layer1-4 -> model.encoder.layer1-4)
                        elif k.startswith('layer0.'):
                            # layer0 通常对应 conv1+bn1+relu，但需要根据具体键名判断
                            # 如果包含 conv1/bn1/relu，直接映射
                            if 'conv1' in k or '0' in k.split('.')[1] if '.' in k else False:
                                mapped_key = k.replace('layer0.', 'model.encoder.conv1.', 1)
                            elif 'bn1' in k or '1' in k.split('.')[1] if '.' in k else False:
                                mapped_key = k.replace('layer0.', 'model.encoder.bn1.', 1)
                            else:
                                mapped_key = k.replace('layer0.', 'model.encoder.conv1.', 1)
                        elif k.startswith('layer1.'):
                            mapped_key = k.replace('layer1.', 'model.encoder.layer1.', 1)
                        elif k.startswith('layer2.'):
                            mapped_key = k.replace('layer2.', 'model.encoder.layer2.', 1)
                        elif k.startswith('layer3.'):
                            mapped_key = k.replace('layer3.', 'model.encoder.layer3.', 1)
                        elif k.startswith('layer4.'):
                            mapped_key = k.replace('layer4.', 'model.encoder.layer4.', 1)
                        
                        encoder_only_state_dict[mapped_key] = v
                        encoder_keys.append(f"{k} -> {mapped_key}")
                    elif is_decoder:
                        discarded_keys.append(k)
                
                if verbose:
                    print(f"[跨架构权重迁移] 编码器权重: {len(encoder_keys)} 个")
                    print(f"[跨架构权重迁移] 已丢弃解码器权重: {len(discarded_keys)} 个")
                    if encoder_keys:
                        print(f"[跨架构权重迁移] 编码器键映射示例（前5个）:")
                        for k in encoder_keys[:5]:
                            print(f"  - {k}")
                    if discarded_keys:
                        print(f"[跨架构权重迁移] 已丢弃键示例（前5个）:")
                        for k in discarded_keys[:5]:
                            print(f"  - {k}")
                
                # 使用过滤后的 state_dict
                state_dict = encoder_only_state_dict
        
        # 检测并转换旧版本的键名（layer0/layer1 -> enc0/enc1）
        # 检查是否是旧版本的ResNetUNet checkpoint
        old_version_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        has_old_keys = any(any(k.startswith(old_key) for k in state_dict.keys()) for old_key in old_version_keys)
        
        if has_old_keys:
            # 创建键名映射：旧版本 -> 新版本
            key_mapping = {}
            for old_key in state_dict.keys():
                new_key = old_key
                # 映射编码器层
                if old_key.startswith('layer0.'):
                    new_key = old_key.replace('layer0.', 'enc0.', 1)
                elif old_key.startswith('layer1.'):
                    new_key = old_key.replace('layer1.', 'enc1.', 1)
                elif old_key.startswith('layer2.'):
                    new_key = old_key.replace('layer2.', 'enc2.', 1)
                elif old_key.startswith('layer3.'):
                    new_key = old_key.replace('layer3.', 'enc3.', 1)
                elif old_key.startswith('layer4.'):
                    new_key = old_key.replace('layer4.', 'enc4.', 1)
                # 可能还有其他映射，比如 center -> center (如果模型不使用ASPP)
                # 注意：如果检测到旧版本checkpoint，模型会使用center而不是aspp，所以不需要映射
                # 但如果checkpoint中有center而模型使用aspp，则需要映射
                # 这里我们保持center不变，因为旧版本模型会使用center
                # （映射逻辑在_load_model中已经处理，这里只处理layerX -> encX的映射）
                elif old_key.startswith('up0.') or old_key.startswith('up_conv0.'):
                    new_key = old_key.replace('up0.', 'dec4.', 1).replace('up_conv0.', 'dec4.', 1)
                elif old_key.startswith('up1.') or old_key.startswith('up_conv1.'):
                    new_key = old_key.replace('up1.', 'dec3.', 1).replace('up_conv1.', 'dec3.', 1)
                elif old_key.startswith('up2.') or old_key.startswith('up_conv2.'):
                    new_key = old_key.replace('up2.', 'dec2.', 1).replace('up_conv2.', 'dec2.', 1)
                elif old_key.startswith('up3.') or old_key.startswith('up_conv3.'):
                    new_key = old_key.replace('up3.', 'dec1.', 1).replace('up_conv3.', 'dec1.', 1)
                elif old_key.startswith('up4.') or old_key.startswith('up_conv4.'):
                    # up4 对应 dec4（最深层）
                    new_key = old_key.replace('up4.', 'dec4.', 1).replace('up_conv4.', 'dec4.', 1)
                
                if new_key != old_key:
                    key_mapping[old_key] = new_key
            
            # 应用键名映射
            if key_mapping:
                new_state_dict = {}
                for old_key, value in state_dict.items():
                    if old_key in key_mapping:
                        new_state_dict[key_mapping[old_key]] = value
                    else:
                        new_state_dict[old_key] = value
                state_dict = new_state_dict
                if verbose:
                    print(f"[模型加载] 检测到旧版本checkpoint，已转换 {len(key_mapping)} 个键名")
        
        model_dict = model.state_dict()
        
        # 统计匹配情况
        matched = {}
        mismatched = []
        missing_in_ckpt = []
        extra_in_ckpt = []  # checkpoint中有但模型中不存在的键
        
        for k, v in model_dict.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    matched[k] = state_dict[k]
                else:
                    mismatched.append(k)
            else:
                missing_in_ckpt.append(k)
        
        # 检查checkpoint中多余的键
        for k in state_dict.keys():
            if k not in model_dict:
                extra_in_ckpt.append(k)
        
        # 加载匹配的权重
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        
        loaded_keys = len(matched)
        total_keys = len(model_dict)
        
        # 详细诊断信息
        if verbose and loaded_keys == 0:
            print(f"[模型加载] ⚠️ 警告：没有参数匹配！")
            print(f"[模型加载] 模型参数键示例（前10个）:")
            for i, k in enumerate(list(model_dict.keys())[:10]):
                print(f"  {i+1}. {k} (shape: {model_dict[k].shape})")
            print(f"[模型加载] Checkpoint参数键示例（前10个）:")
            for i, k in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i+1}. {k} (shape: {state_dict[k].shape})")
            if extra_in_ckpt:
                print(f"[模型加载] Checkpoint中多余的键（前5个）: {extra_in_ckpt[:5]}")
        elif verbose and loaded_keys > 0 and loaded_keys < total_keys:
            # 显示未匹配的参数类别统计
            missing_categories = {}
            for k in missing_in_ckpt:
                category = k.split('.')[0] if '.' in k else k
                missing_categories[category] = missing_categories.get(category, 0) + 1
            
            if missing_categories:
                print(f"[模型加载] 未匹配参数类别统计:")
                for cat, count in sorted(missing_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  - {cat}: {count} 个参数")
                if len(missing_categories) > 5:
                    print(f"  - ... 还有 {len(missing_categories) - 5} 个类别")
        
        if mismatched:
            msg = f"部分加载: {loaded_keys}/{total_keys}个参数匹配, {len(mismatched)}个形状不匹配"
            if verbose:
                print(f"[模型加载] {msg}")
                print(f"[模型加载] 形状不匹配的参数: {mismatched[:5]}{'...' if len(mismatched) > 5 else ''}")
                # 对于DS-TransUNet，提供更详细的形状信息
                if any('patch_embed3' in k or 'transformer3' in k for k in mismatched[:5]):
                    print(f"[模型加载] 详细形状对比（前3个不匹配的参数）:")
                    for k in mismatched[:3]:
                        model_shape = model_dict[k].shape if k in model_dict else "N/A"
                        ckpt_shape = state_dict[k].shape if k in state_dict else "N/A"
                        print(f"  - {k}:")
                        print(f"    模型期望: {model_shape}")
                        print(f"    Checkpoint实际: {ckpt_shape}")
        elif missing_in_ckpt and loaded_keys > 0:
            msg = f"兼容加载: {loaded_keys}/{total_keys}个参数 (新增层已随机初始化)"
            if verbose:
                print(f"[模型加载] {msg}")
                if missing_in_ckpt:
                    print(f"[模型加载] 新增层（前5个）: {missing_in_ckpt[:5]}{'...' if len(missing_in_ckpt) > 5 else ''}")
        elif loaded_keys == 0:
            msg = f"⚠️ 严重警告: 0/{total_keys}个参数匹配！模型类型可能不匹配"
            if verbose:
                print(f"[模型加载] {msg}")
                print(f"[模型加载] 请检查模型类型是否与checkpoint匹配")
            # 0个参数匹配时返回False，表示加载失败
            return False, msg
        else:
            msg = f"完整加载: {os.path.basename(checkpoint_path)}"
            if verbose:
                print(f"[模型加载] {msg}")
        
        return True, msg
        
    except Exception as e:
        msg = f"加载失败: {str(e)}"
        if verbose:
            print(f"[模型加载] {msg}")
        return False, msg


def infer_swin_params_from_state_dict(state_dict):
    """从state_dict精确推断SwinUNet参数"""
    if 'patch_embed.proj.weight' not in state_dict:
        return None
    
    # embed_dim
    embed_dim = state_dict['patch_embed.proj.weight'].shape[0]
    
    # depths: 统计每个stage的block数
    depths = []
    for stage_idx in range(10):
        block_count = 0
        for block_idx in range(50):
            if f'encoder_layers.{stage_idx}.{block_idx}.norm1.weight' in state_dict:
                block_count += 1
            else:
                break
        if block_count > 0:
            depths.append(block_count)
        else:
            break
    if not depths:
        depths = [2, 2, 6, 2]
    
    # num_heads: 从qkv权重推断
    num_heads = []
    for stage_idx in range(len(depths)):
        qkv_key = f'encoder_layers.{stage_idx}.0.attn.qkv.weight'
        if qkv_key in state_dict:
            qkv_out = state_dict[qkv_key].shape[0]  # 3 * dim
            dim_at_stage = qkv_out // 3
            # 从proj权重推断head数
            proj_key = f'encoder_layers.{stage_idx}.0.attn.proj.weight'
            if proj_key in state_dict:
                for head_dim in [32, 64, 48, 96, 128]:
                    if dim_at_stage % head_dim == 0:
                        num_heads.append(dim_at_stage // head_dim)
                        break
                else:
                    num_heads.append(max(1, dim_at_stage // 32))
            else:
                num_heads.append(max(1, dim_at_stage // 32))
        else:
            num_heads.append(3 * (2 ** stage_idx))
    
    # mlp_hidden_dims: 精确记录每个stage每个block的mlp hidden dim
    # 这样可以避免mlp_ratio的浮点误差
    mlp_hidden_dims = {}
    for stage_idx in range(len(depths)):
        for block_idx in range(depths[stage_idx]):
            fc1_key = f'encoder_layers.{stage_idx}.{block_idx}.mlp.fc1.weight'
            if fc1_key in state_dict:
                mlp_hidden_dims[(stage_idx, block_idx)] = state_dict[fc1_key].shape[0]
    
    # mlp_ratio: 从第一个block推断（用于新建block时的默认值）
    mlp_ratio = 4.0
    fc1_key = 'encoder_layers.0.0.mlp.fc1.weight'
    if fc1_key in state_dict:
        hidden = state_dict[fc1_key].shape[0]
        in_dim = state_dict[fc1_key].shape[1]
        mlp_ratio = hidden / in_dim
    
    # window_size: 尝试从relative_position_bias_table推断
    window_size = 8
    rpb_key = 'encoder_layers.0.0.attn.relative_position_bias_table'
    if rpb_key in state_dict:
        table_size = state_dict[rpb_key].shape[0]
        import math
        ws_calc = (math.sqrt(table_size) + 1) / 2
        if ws_calc == int(ws_calc):
            window_size = int(ws_calc)
    
    return {
        'embed_dim': embed_dim,
        'depths': tuple(depths),
        'num_heads': tuple(num_heads),
        'mlp_ratio': mlp_ratio,
        'window_size': window_size,
        'drop_path_rate': 0.0,
        '_mlp_hidden_dims': mlp_hidden_dims,  # 精确的hidden dims
        '_from_checkpoint': True
    }


def infer_dstrans_params_from_state_dict(state_dict):
    """从state_dict推断DS-TransUNet参数（增强版，提高兼容性）"""
    # 处理可能的键名变体（考虑DataParallel包装等）
    patch_embed3_key = None
    for key in state_dict.keys():
        if 'patch_embed3.weight' in key or key.endswith('patch_embed3.weight'):
            patch_embed3_key = key
            break
    
    if patch_embed3_key is None:
        return None
    
    try:
        # 优先从in_proj_weight推断embed_dim（更准确，因为它直接反映了transformer的维度）
        embed_dim = None
        num_heads = 8  # 默认值
        in_proj_key = None
        for key in state_dict.keys():
            if 'transformer3.layers.0.self_attn.in_proj_weight' in key or key.endswith('transformer3.layers.0.self_attn.in_proj_weight'):
                in_proj_key = key
                break
        
        if in_proj_key:
            in_proj_weight = state_dict[in_proj_key]
            # in_proj_weight的形状是 [3 * embed_dim, embed_dim]
            if len(in_proj_weight.shape) == 2:
                # 从checkpoint读取实际的embed_dim（最准确的方法）
                actual_embed_dim = in_proj_weight.shape[1]  # 第二维是embed_dim
                if in_proj_weight.shape[0] == 3 * actual_embed_dim:
                    embed_dim = actual_embed_dim
                    print(f"[参数推断] 从in_proj_weight读取embed_dim: {embed_dim}")
                    
                    # num_heads 必须是 embed_dim 的约数
                    # 尝试常见的值，优先选择较大的（通常性能更好）
                    for nh in [32, 16, 8, 4]:
                        if embed_dim % nh == 0:
                            num_heads = nh
                            break
                else:
                    print(f"[警告] in_proj_weight形状异常: {in_proj_weight.shape}, 期望: [3*embed_dim, embed_dim]")
        
        # 如果无法从in_proj_weight推断，则从patch_embed3推断
        if embed_dim is None:
            patch_embed3_weight = state_dict[patch_embed3_key]
            if len(patch_embed3_weight.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
                embed_dim = patch_embed3_weight.shape[0]  # 输出通道数
                print(f"[参数推断] 从patch_embed3读取embed_dim: {embed_dim}")
            elif len(patch_embed3_weight.shape) == 2:  # Linear: [out_features, in_features]
                embed_dim = patch_embed3_weight.shape[0]
                print(f"[参数推断] 从patch_embed3读取embed_dim: {embed_dim}")
            else:
                print(f"[警告] patch_embed3.weight形状异常: {patch_embed3_weight.shape}")
                return None
            
            # 从embed_dim推断num_heads
            for nh in [32, 16, 8, 4]:
                if embed_dim % nh == 0:
                    num_heads = nh
                    break
        
        # num_layers: 统计transformer3的层数（检查两个transformer）
        num_layers = 2  # 默认值
        max_layers = 0
        for i in range(20):  # 增加范围以支持更深的模型
            # 检查transformer3
            key3 = f'transformer3.layers.{i}.self_attn.in_proj_weight'
            key3_alt = None
            for k in state_dict.keys():
                if key3 in k or k.endswith(key3):
                    key3_alt = k
                    break
            if key3_alt:
                max_layers = max(max_layers, i + 1)
            else:
                break
        
        if max_layers > 0:
            num_layers = max_layers
        
        # mlp_ratio: 从transformer3.layers[0].linear1或ffn.0.weight推断
        mlp_ratio = 4.0  # 默认值
        linear1_key = None
        for key in state_dict.keys():
            if 'transformer3.layers.0.linear1.weight' in key or key.endswith('transformer3.layers.0.linear1.weight'):
                linear1_key = key
                break
        
        if linear1_key:
            try:
                linear1_out = state_dict[linear1_key].shape[0]
                if embed_dim > 0:
                    mlp_ratio = linear1_out / embed_dim
            except:
                pass
        
        # dropout: 默认值（通常无法从state_dict推断）
        dropout = 0.1  # 默认值
        
        # 验证参数合理性
        if embed_dim <= 0 or num_heads <= 0 or num_layers <= 0:
            print(f"[警告] DS-TransUNet参数推断异常: embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}")
            return None
        
        if embed_dim % num_heads != 0:
            print(f"[警告] embed_dim({embed_dim})不能被num_heads({num_heads})整除，自动调整num_heads")
            # 自动调整num_heads
            for nh in [32, 16, 8, 4]:
                if embed_dim % nh == 0:
                    num_heads = nh
                    break
        
        # 验证推断的参数是否与checkpoint中的实际形状匹配
        # 打印详细的调试信息
        print(f"[调试] DS-TransUNet参数推断结果:")
        print(f"  - embed_dim: {embed_dim}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - mlp_ratio: {mlp_ratio:.2f}")
        print(f"  - dropout: {dropout}")
        
        # 验证关键层的形状
        if in_proj_key:
            in_proj_weight = state_dict[in_proj_key]
            if len(in_proj_weight.shape) == 2:
                expected_shape = (3 * embed_dim, embed_dim)
                actual_shape = in_proj_weight.shape
                print(f"  - transformer3.in_proj_weight形状: {actual_shape} (期望: {expected_shape})")
                if actual_shape != expected_shape:
                    print(f"[警告] in_proj_weight形状不匹配！实际: {actual_shape}, 期望: {expected_shape}")
                    # 尝试从实际形状反推embed_dim
                    if actual_shape[0] % 3 == 0:
                        inferred_embed_dim = actual_shape[0] // 3
                        if inferred_embed_dim == actual_shape[1]:
                            print(f"[提示] 从in_proj_weight反推embed_dim: {inferred_embed_dim}")
                            embed_dim = inferred_embed_dim
                            # 重新计算num_heads
                            for nh in [32, 16, 8, 4]:
                                if embed_dim % nh == 0:
                                    num_heads = nh
                                    break
        
        return {
            'embed_dim': int(embed_dim),
            'num_heads': int(num_heads),
            'num_layers': int(num_layers),
            'mlp_ratio': float(mlp_ratio),
            'dropout': float(dropout)
        }
    except Exception as e:
        print(f"[错误] DS-TransUNet参数推断失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_checkpoint_config(checkpoint_path):
    """读取checkpoint配置，支持从权重形状推断模型参数
    检测顺序与_load_model保持一致：
    1. 首先检查config中是否有model_type
    2. 然后从state_dict推断模型类型（按优先级顺序）
    """
    try:
        loaded_obj = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(loaded_obj, dict) and 'config' in loaded_obj:
            return loaded_obj['config']
        
        # 尝试从权重形状推断模型参数
        state_dict = loaded_obj['state_dict'] if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj else loaded_obj
        
        # 处理DataParallel包装
        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 检测顺序与_load_model保持一致
        # 1. 检测DS-TransUNet (patch_embed3)
        if state_dict and 'patch_embed3.weight' in state_dict:
            dstrans_params = infer_dstrans_params_from_state_dict(state_dict)
            if dstrans_params:
                return {
                    'model_type': 'ds_trans_unet',
                    'dstrans_params': dstrans_params
                }
        
        # 2. 检测SwinUNet (patch_embed.proj)
        if state_dict and 'patch_embed.proj.weight' in state_dict:
            swin_params = infer_swin_params_from_state_dict(state_dict)
            if swin_params:
                return {
                    'model_type': 'swin_unet',
                    'swin_params': swin_params
                }
        
        # 3. 检测ResNetUNet (enc0或layer0)
        old_version_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        has_old_keys = any(any(k.startswith(old_key) for k in state_dict.keys()) for old_key in old_version_keys)
        
        if 'enc0.0.weight' in state_dict or 'enc0.weight' in state_dict or has_old_keys:
            # ResNetUNet
                resnet_params = {}
                # 检测backbone类型
                if 'enc1.0.conv1.weight' in state_dict or (has_old_keys and 'layer1.0.conv1.weight' in state_dict):
                    if 'enc1.2.conv1.weight' in state_dict or (has_old_keys and 'layer1.2.conv1.weight' in state_dict):
                        resnet_params['backbone_name'] = 'resnet101'
                    else:
                        resnet_params['backbone_name'] = 'resnet50'
                
                # 检测是否有ASPP
                has_aspp = any('aspp' in k.lower() for k in state_dict.keys())
                if has_old_keys and not has_aspp:
                    resnet_params['use_aspp'] = False
                
                return {
                    'model_type': 'resnet_unet',
                    'resnet_params': resnet_params
                }
        
        # 4. 检测TransUNet (encoder.0)
        if 'encoder.0.weight' in state_dict:
            return {'model_type': 'trans_unet'}
        
        # 5. 检测其他ResNetUNet变体 (backbone.layer1)
        if 'backbone.layer1.0.conv1.weight' in state_dict:
            return {'model_type': 'resnet_unet'}
            
    except Exception as e:
        print(f"[read_checkpoint_config] 读取失败: {e}")
        return None
    return None

# 注意：instantiate_model 函数引用了模型类，需要在原文件中保留引用
# 这里只提供函数签名，实际实现需要在原文件中保留以访问模型类
