#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为已有模型生成智能后处理配置 JSON 文件的独立脚本

用法:
    python generate_smart_postprocessing_config.py --model_path <模型路径> --data_dir <数据目录> [--model_type <模型类型>] [--use_tta]

示例:
    python generate_smart_postprocessing_config.py --model_path ./checkpoints/best_model.pth --data_dir ./data
    python generate_smart_postprocessing_config.py --model_path ./checkpoints/best_model.pth --data_dir ./data --model_type deeplabv3plus --use_tta
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置 NumExpr 线程数（避免警告）
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count() or 1, 24))

from utils import find_optimal_postprocessing_strategy, load_model_compatible, MedicalImageDataset
from utils.model_loader import read_checkpoint_config
from worker.train_thread import TrainThread
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import albumentations as A


def create_val_dataloader(data_dir, batch_size=4, dataset_type="standard"):
    """创建验证集 DataLoader"""
    # 创建临时 TrainThread 实例以使用其数据加载方法
    temp_train_thread = TrainThread(
        data_dir=data_dir,
        epochs=1,
        batch_size=batch_size,
        model_path=None,
        save_best=False,
        dataset_type=dataset_type
    )
    
    # 根据数据集类型设置归一化参数
    if dataset_type == "2.5d":
        normalize_mean = (0.485, 0.456, 0.406)
        normalize_std = (0.229, 0.224, 0.225)
    else:
        normalize_mean = (0.485, 0.456, 0.406)
        normalize_std = (0.229, 0.224, 0.225)
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2()
    ])
    
    # 获取 patient_ids
    if dataset_type == "2.5d":
        # 2.5D 数据集：使用全部数据
        val_dataset = temp_train_thread.load_dataset(
            [], val_transform, split_name="val",
            return_classification=False, use_weighted_sampling=False
        )
    else:
        # 标准数据集：按 patient_id 组织
        patient_ids = [pid for pid in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, pid))]
        
        if not patient_ids:
            raise ValueError(f"数据目录为空，未找到子文件夹: {data_dir}")
        
        # 使用 TrainThread 的数据加载方法
        val_dataset = temp_train_thread.load_dataset(
            patient_ids, val_transform, split_name="val",
            return_classification=False, use_weighted_sampling=False
        )
    
    # 创建 DataLoader
    import platform
    is_windows = platform.system() == 'Windows'
    cpu_count = os.cpu_count() or 1
    num_workers = 8 if is_windows else max(0, min(4, cpu_count - 1))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    
    return val_loader


def load_model_from_checkpoint(model_path, device, model_type=None):
    """从检查点加载模型"""
    print(f"[加载模型] 正在加载模型: {model_path}")
    
    # 读取检查点配置以确定模型类型
    checkpoint_config = read_checkpoint_config(model_path)
    if checkpoint_config and 'model_type' in checkpoint_config:
        detected_model_type = checkpoint_config['model_type']
        print(f"[加载模型] 从检查点检测到模型类型: {detected_model_type}")
        if model_type is None:
            model_type = detected_model_type
        elif model_type != detected_model_type:
            print(f"[警告] 指定的模型类型 ({model_type}) 与检查点中的类型 ({detected_model_type}) 不一致，使用检查点中的类型")
            model_type = detected_model_type
    else:
        if model_type is None:
            raise ValueError("无法从检查点推断模型类型，请使用 --model_type 参数指定")
        print(f"[加载模型] 使用指定的模型类型: {model_type}")
    
    # 创建临时 TrainThread 以使用其模型构建方法
    temp_train_thread = TrainThread(
        data_dir="",  # 不需要数据目录
        epochs=1,
        batch_size=4,
        model_path=None,
        save_best=False
    )
    temp_train_thread.model_type = model_type
    
    # 构建模型
    model = temp_train_thread._build_model(device)
    
    # 加载权重
    success, msg = load_model_compatible(model, model_path, device, verbose=True, target_model_type=model_type)
    if not success:
        raise RuntimeError(f"模型加载失败: {msg}")
    
    model.eval()
    print(f"[加载模型] 模型加载成功: {model_type}")
    
    return model, model_type


def main():
    parser = argparse.ArgumentParser(
        description="为已有模型生成智能后处理配置 JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（自动从检查点推断模型类型）
  python generate_smart_postprocessing_config.py --model_path ./checkpoints/best_model.pth --data_dir ./data

  # 指定模型类型
  python generate_smart_postprocessing_config.py --model_path ./checkpoints/best_model.pth --data_dir ./data --model_type deeplabv3plus

  # 启用 TTA（测试时增强）
  python generate_smart_postprocessing_config.py --model_path ./checkpoints/best_model.pth --data_dir ./data --use_tta

  # 2.5D 数据集
  python generate_smart_postprocessing_config.py --model_path ./checkpoints/best_model.pth --data_dir ./data --dataset_type 2.5d
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重文件路径（.pth 文件）')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='验证数据目录路径')
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['unet', 'unetpp', 'deeplabv3plus', 'swin_unet', 'dstrans', 'resnet_unet'],
                       help='模型类型（如果检查点中没有配置信息，需要手动指定）')
    parser.add_argument('--use_tta', action='store_true',
                       help='启用测试时增强（TTA）')
    parser.add_argument('--dataset_type', type=str, default='standard',
                       choices=['standard', '2.5d'],
                       help='数据集类型：standard（标准）或 2.5d（2.5D）')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小（默认：4）')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"[错误] 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"[错误] 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用设备: {device}")
    
    try:
        # 1. 加载模型
        model, model_type = load_model_from_checkpoint(args.model_path, device, args.model_type)
        
        # 2. 创建验证集 DataLoader
        print(f"\n[数据加载] 正在创建验证集 DataLoader...")
        val_loader = create_val_dataloader(args.data_dir, batch_size=args.batch_size, dataset_type=args.dataset_type)
        print(f"[数据加载] 验证集 DataLoader 创建成功，共 {len(val_loader)} 个批次")
        
        # 3. 搜索最佳后处理策略
        print(f"\n[策略搜索] 开始搜索最佳后处理策略（使用 {'TTA' if args.use_tta else '无TTA'}）...")
        best_cfg = find_optimal_postprocessing_strategy(
            val_loader,
            model,
            device,
            use_tta=args.use_tta
        )
        
        method = best_cfg.get("method", "baseline")
        avg_dice = best_cfg.get("avg_dice", 0.0)
        params = best_cfg.get("params", {})
        
        print(f"\n[策略搜索] 搜索完成！")
        print(f"  最佳策略: {method}")
        print(f"  平均 Dice: {avg_dice:.4f}")
        print(f"  参数: {params}")
        
        # 4. 保存配置到模型目录
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        cfg_path = os.path.join(model_dir, "best_postprocessing_config.json")
        
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, ensure_ascii=False, indent=2)
        
        print(f"\n[保存配置] 已保存最佳后处理配置到: {cfg_path}")
        print(f"\n✅ 完成！现在可以在测试和推理阶段使用此配置文件。")
        
    except Exception as e:
        print(f"\n[错误] 生成配置失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

