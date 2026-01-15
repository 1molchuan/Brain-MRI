# -*- coding: utf-8 -*-
"""
从utils.gwo_optimizer模块
"""
from utils.common import *
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 【内存安全多进程】顶层worker函数，用于分块计算适应度
def _calculate_fitness_chunk(args):
    """
    计算一个数据块的适应度统计量（顶层函数，兼容Windows多进程）
    
    Args:
        args: 元组 (probs_chunk, masks_chunk, threshold, post_processing_params)
            - probs_chunk: (N, H, W) float16 numpy数组
            - masks_chunk: (N, H, W) uint8 numpy数组
            - threshold: float，阈值
            - post_processing_params: dict，后处理参数
    
    Returns:
        dict: {
            'tp': int,  # True Positive总数
            'fp': int,  # False Positive总数
            'fn': int,  # False Negative总数
            'tn': int,  # True Negative总数
            'empty_mask_count': int,  # 空mask样本数
            'non_empty_mask_count': int,  # 非空mask样本数
            'empty_dice_sum': float,  # 空mask样本的Dice总和
            'non_empty_dice_sum': float  # 非空mask样本的Dice总和
        }
    """
    import os
    probs_chunk, masks_chunk, threshold, post_processing_params = args
    
    # 【PID验证】打印进程ID，验证多进程是否启动
    chunk_size = len(probs_chunk) if hasattr(probs_chunk, '__len__') else 0
    print(f"[Process PID={os.getpid()}] 正在处理 Chunk (样本数={chunk_size}, 阈值={threshold:.4f})...", flush=True)
    
    try:
        # 确保数据类型正确
        probs_chunk = np.asarray(probs_chunk, dtype=np.float16)
        masks_chunk = np.asarray(masks_chunk, dtype=np.uint8)
        
        # 二值化
        pred_mask = (probs_chunk >= threshold).astype(np.uint8)
        
        # 应用后处理（如果启用）
        has_postprocess = post_processing_params.get('has_postprocess', False)
        smart_post_cfg_dict = post_processing_params.get('smart_post_cfg_dict', None)
        
        if has_postprocess:
            try:
                from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
                from skimage.morphology import remove_small_objects
                
                # 对每个样本应用后处理
                for i in range(len(pred_mask)):
                    pred_single = pred_mask[i].astype(bool)
                    prob_single = probs_chunk[i]
                    
                    if smart_post_cfg_dict is not None:
                        # 有智能配置：最小干预 + 智能策略
                        if smart_post_cfg_dict.get('fill_holes', True):
                            pred_single = binary_fill_holes(pred_single).astype(np.uint8)
                        
                        method = smart_post_cfg_dict.get('method', 'baseline')
                        params = smart_post_cfg_dict.get('params', {})
                        if method != 'baseline':
                            from utils.smart_postprocessing import _apply_strategy
                            pred_single = _apply_strategy(pred_single.astype(np.float32), method, params).astype(np.uint8)
                    else:
                        # 无智能配置：使用默认的保守后处理
                        min_size = 150
                        pred_single = remove_small_objects(pred_single, min_size=min_size).astype(np.uint8)
                        pred_single = binary_fill_holes(pred_single.astype(bool)).astype(np.uint8)
                        pred_single = binary_opening(pred_single.astype(bool), structure=np.ones((3,3))).astype(np.uint8)
                        pred_single = binary_closing(pred_single.astype(bool), structure=np.ones((3,3))).astype(np.uint8)
                    
                    pred_mask[i] = pred_single
            except Exception:
                # 后处理失败则回退原始预测
                pred_mask = (probs_chunk >= threshold).astype(np.uint8)
        
        # 计算统计量（按样本分别计算，用于Mean Dice）
        num_samples = len(pred_mask)
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0
        empty_mask_count = 0
        non_empty_mask_count = 0
        empty_dice_sum = 0.0
        non_empty_dice_sum = 0.0
        
        for i in range(num_samples):
            pred_flat = pred_mask[i].flatten().astype(np.float32)
            gt_flat = masks_chunk[i].flatten().astype(np.float32)
            
            # 计算TP, FP, FN, TN
            tp = int(np.sum(pred_flat * gt_flat))
            fp = int(np.sum(pred_flat * (1.0 - gt_flat)))
            fn = int(np.sum((1.0 - pred_flat) * gt_flat))
            tn = int(np.sum((1.0 - pred_flat) * (1.0 - gt_flat)))
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
            
            # 计算Dice（用于Mean Dice）
            mask_sum = gt_flat.sum()
            pred_sum = pred_flat.sum()
            empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)
            
            if mask_sum <= empty_threshold:
                # 空mask
                empty_mask_count += 1
                dice_sample = 1.0 if pred_sum <= 1e-7 else 0.0
                empty_dice_sum += dice_sample
            else:
                # 非空mask
                non_empty_mask_count += 1
                intersection = float(tp)
                dice_den = 2.0 * intersection + pred_sum + mask_sum
                dice_sample = 0.0 if dice_den < 1e-7 else (2.0 * intersection) / dice_den
                non_empty_dice_sum += dice_sample
        
        return {
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total,
            'tn': tn_total,
            'empty_mask_count': empty_mask_count,
            'non_empty_mask_count': non_empty_mask_count,
            'empty_dice_sum': empty_dice_sum,
            'non_empty_dice_sum': non_empty_dice_sum
        }
    except Exception as e:
        import traceback
        print(f"[多进程] 计算适应度块失败: {e}")
        traceback.print_exc()
        # 返回零统计量
        return {
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'empty_mask_count': 0, 'non_empty_mask_count': 0,
            'empty_dice_sum': 0.0, 'non_empty_dice_sum': 0.0
        }

def _process_wolf_cpu(args):
    """
    处理单只狼的适应度计算（CPU模式，用于多进程）
    
    Args:
        args: 元组 (wolf_idx, threshold, prob_maps, mask_gts, has_postprocess, smart_post_cfg_dict, num_samples)
              注意：postprocess_func不能直接传递（无法pickle），改为传递标志和配置
    
    Returns:
        (wolf_idx, threshold_rounded, dice_score)
    """
    wolf_idx, threshold, prob_maps, mask_gts, has_postprocess, smart_post_cfg_dict, num_samples = args
    
    try:
        threshold_rounded = round(threshold / 0.001) * 0.001  # 缓存容差
        
        # 处理所有样本（单进程处理样本，避免嵌套多进程）
        dice_scores = []
        for i in range(num_samples):
            prob_map = prob_maps[i]
            mask_gt = mask_gts[i]
            
            # 二值化
            pred_mask = (prob_map >= threshold).astype(np.float32)
            
            # 应用后处理（如果启用）
            if has_postprocess:
                try:
                    # 在子进程中重新导入后处理模块（避免pickle问题）
                    from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
                    from skimage.morphology import remove_small_objects
                    
                    if smart_post_cfg_dict is not None:
                        # 有智能配置：最小干预 + 智能策略
                        # 基本后处理（最小干预）
                        if smart_post_cfg_dict.get('fill_holes', True):
                            pred_mask = binary_fill_holes(pred_mask.astype(bool)).astype(np.float32)
                        
                        # 智能策略
                        method = smart_post_cfg_dict.get('method', 'baseline')
                        params = smart_post_cfg_dict.get('params', {})
                        if method != 'baseline':
                            from utils.smart_postprocessing import _apply_strategy
                            pred_mask = _apply_strategy(pred_mask, method, params)
                    else:
                        # 无智能配置：使用默认的保守后处理
                        if smart_post_cfg_dict is None or smart_post_cfg_dict.get('min_size', 150) > 0:
                            min_size = smart_post_cfg_dict.get('min_size', 150) if smart_post_cfg_dict else 150
                            pred_mask = remove_small_objects(pred_mask.astype(bool), min_size=min_size).astype(np.float32)
                        
                        if smart_post_cfg_dict is None or smart_post_cfg_dict.get('fill_holes', True):
                            pred_mask = binary_fill_holes(pred_mask.astype(bool)).astype(np.float32)
                        
                        if smart_post_cfg_dict is None or smart_post_cfg_dict.get('use_morphology', True):
                            pred_mask = binary_opening(pred_mask.astype(bool), structure=np.ones((3,3))).astype(np.float32)
                            pred_mask = binary_closing(pred_mask.astype(bool), structure=np.ones((3,3))).astype(np.float32)
                except Exception as e:
                    # 后处理失败则回退原始预测
                    pred_mask = (prob_map >= threshold).astype(np.float32)
            
            # 计算Dice
            pred_flat = pred_mask.flatten()
            gt_flat = mask_gt.flatten()
            mask_sum = gt_flat.sum()
            empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)
            pred_sum = pred_flat.sum()
            
            if mask_sum <= empty_threshold:
                dice_sample = 1.0 if pred_sum <= 1e-7 else 0.0
            else:
                intersection = (pred_flat * gt_flat).sum()
                dice_den = 2.0 * intersection + pred_sum + mask_sum
                dice_sample = 0.0 if dice_den < 1e-7 else (2.0 * intersection) / dice_den
            
            dice_scores.append(float(dice_sample))
        
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        return (wolf_idx, threshold_rounded, mean_dice)
    except Exception as e:
        import traceback
        print(f"[多进程] 处理灰狼 {wolf_idx} 失败: {e}")
        traceback.print_exc()
        return (wolf_idx, round(threshold / 0.001) * 0.001, 0.0)

class GreyWolfThresholdOptimizer:
    """
    使用灰狼优化算法(GWO)寻找最佳分割阈值
    目标函数：Dice系数
    
    相比线性扫描，GWO 能够更智能地搜索阈值空间，在更少的迭代次数内找到更优解。
    """
    
    def __init__(self, num_wolves=10, max_iter=20, progress_callback=None, use_multiprocessing=True, 
                 use_mean_dice=False, postprocess_func=None, sample_ratio=1.0, metrics_func=None, smart_post_cfg=None):
        """
        Args:
            num_wolves: 灰狼数量（种群大小），默认10
            max_iter: 最大迭代次数，默认20
            progress_callback: 进度回调函数，接收 (iteration, max_iter, best_score, best_threshold) 参数
            use_multiprocessing: 是否在CPU模式下使用多进程（默认True）
            use_mean_dice: 是否使用Mean Dice（每个样本分别计算再平均），默认False（使用Global Dice）
            postprocess_func: 后处理函数，接收(pred_mask, prob_map)返回处理后的mask，默认None（不使用后处理）
            sample_ratio: 采样比例，在迭代过程中只使用部分样本计算Fitness（0.0-1.0），默认1.0（使用全部样本）
            smart_post_cfg: 智能后处理配置字典，用于多进程模式（可序列化）
        """
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.progress_callback = progress_callback
        self.use_multiprocessing = use_multiprocessing
        self.use_mean_dice = use_mean_dice
        self.postprocess_func = postprocess_func
        self.sample_ratio = sample_ratio
        self.metrics_func = metrics_func
        self.smart_post_cfg = smart_post_cfg  # 保存智能后处理配置
        # 搜索空间 [0.05, 0.95]（扩大范围，覆盖更广的阈值空间）
        self.lb = 0.05
        self.ub = 0.95
        # 【缓存机制】缓存阈值和对应的Dice分数，避免重复计算
        self._threshold_cache = {}  # {threshold_rounded: dice_score}
        self._cache_tolerance = 0.001  # 阈值差值<0.001时复用结果

    def optimize(self, preds, targets, device=None):
        """
        使用GWO算法优化阈值（内存安全 + CPU并行模式）
        
        Args:
            preds: 模型输出的概率图，可以是 torch.Tensor 或 numpy.ndarray
                  形状为 (N, H, W) 或 (N, C, H, W)
            targets: 真实标签，形状与 preds 相同
            device: 计算设备（强制使用CPU多进程模式）
            
        Returns:
            best_threshold: 最佳阈值
            best_dice: 最佳Dice分数
        """
        import platform
        import os
        import gc
        
        # 【进度确认】立即打印，确认optimize方法已开始执行
        print(f"[GWO] 开始执行阈值优化，初始化中...")
        
        # 【内存优化】强制CPU模式，转换为最小精度的numpy数组
        # 将数据转换为numpy数组（CPU模式）
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 统一维度：如果是4D，取第一个通道
        if preds.ndim == 4:
            preds = preds[:, 0]  # (N, H, W)
        if targets.ndim == 4:
            targets = targets[:, 0]  # (N, H, W)
        
        # 【内存优化】强制数据瘦身：转换为最小精度
        print(f"[GWO内存优化] 原始数据类型: preds={preds.dtype}, targets={targets.dtype}")
        print(f"[GWO内存优化] 原始数据形状: preds={preds.shape}, targets={targets.shape}")
        
        # 转换为float16（概率图）
        preds = np.asarray(preds, dtype=np.float16)
        # 转换为uint8（mask，0或1）
        targets = np.asarray(targets > 0.5, dtype=np.uint8)
        
        # 释放原始引用，触发垃圾回收
        gc.collect()
        
        print(f"[GWO内存优化] 优化后数据类型: preds={preds.dtype}, targets={targets.dtype}")
        print(f"[GWO内存优化] 内存占用: preds={preds.nbytes / (1024**2):.2f}MB, targets={targets.nbytes / (1024**2):.2f}MB")
        
        # 强制使用CPU模式
        use_gpu = False
        device = torch.device('cpu')
        
        # 【关键修复】保存原始形状（用于Mean Dice计算）
        num_samples = preds.shape[0]
        spatial_shape = preds.shape[1:]  # (H, W)
        
        # 【效率优化】保存全量数据，用于最终评估
        full_preds = preds.copy()
        full_targets = targets.copy()
        
        # 【效率优化】计算采样索引（等间隔采样）
        num_samples_sampled = num_samples
        sample_indices = None
        if self.sample_ratio < 1.0:
            sample_size = max(1, int(num_samples * self.sample_ratio))
            # 等间隔采样
            sample_indices = np.linspace(0, num_samples - 1, sample_size, dtype=np.int32)
            # 在迭代过程中使用采样数据
            preds = preds[sample_indices]
            targets = targets[sample_indices]
            num_samples_sampled = len(sample_indices)
            print(f">>> [GWO效率优化] 采样计算: {num_samples_sampled}/{num_samples} 样本 ({100*self.sample_ratio:.0f}%)")
        
        # 准备后处理参数（可序列化）
        post_processing_params = {
            'has_postprocess': (self.postprocess_func is not None),
            'smart_post_cfg_dict': None
        }
        if hasattr(self, 'smart_post_cfg') and self.smart_post_cfg:
            post_processing_params['smart_post_cfg_dict'] = {
                'method': self.smart_post_cfg.get('method', 'baseline'),
                'params': self.smart_post_cfg.get('params', {}),
                'min_size': 150 if not self.smart_post_cfg else 0,
                'use_morphology': True if not self.smart_post_cfg else False,
                'fill_holes': True
            }
        
        # 【分块参数】设置分块大小（每个chunk的样本数）
        chunk_size = 500  # 每个chunk 500个样本，可根据内存调整
        is_windows = platform.system() == 'Windows'
        
        # 【多进程参数】限制并发数，防止内存爆炸
        cpu_count = os.cpu_count() or 1
        max_workers = min(8, cpu_count)
        if is_windows:
            # Windows下可能需要更保守的并发数
            max_workers = min(4, max_workers)
        
        # 【调试信息】打印多进程配置
        print(f"[GWO多进程配置] CPU核心数: {cpu_count}, max_workers: {max_workers}, is_windows: {is_windows}")
        print(f"[GWO多进程配置] use_multiprocessing: {self.use_multiprocessing}")
        print(f"[GWO] 分块参数: chunk_size={chunk_size}, max_workers={max_workers}, is_windows={is_windows}")
        print(f"[GWO主进程] PID={os.getpid()}", flush=True)
        
        # 【调试日志】初始化种群
        print(f"[GWO-Debug] 初始化种群...")
        print(f"[GWO-Debug] 样本数: {num_samples}, 空间形状: {spatial_shape}, 使用CPU多进程模式")
        
        # 初始化狼群位置（阈值）
        positions = np.random.uniform(self.lb, self.ub, self.num_wolves)
        
        # 【缓存机制】清空缓存（每次optimize调用时）
        self._threshold_cache.clear()
        
        # 初始化 Alpha, Beta, Delta 狼 (前三名)
        alpha_pos, alpha_score = 0.5, -float('inf')
        beta_pos, beta_score = 0.5, -float('inf')
        delta_pos, delta_score = 0.5, -float('inf')
        
        # 【进度确认】初始化完成，开始迭代优化
        print(f"[GWO] 初始化完成，使用CPU多进程模式，开始迭代优化（{self.max_iter}次迭代，{self.num_wolves}只灰狼）...")
        print(f"[GWO-Debug] 种群初始化完成，初始位置范围: [{positions.min():.4f}, {positions.max():.4f}]")
        
        for t in range(self.max_iter):
            # 【调试日志】每次迭代开始
            print(f"[GWO-Debug] ========== Iteration {t+1}/{self.max_iter} ==========")
            print(f"[GWO-Debug] 迭代 {t+1}/{self.max_iter} 开始...")
            
            # 线性衰减参数 a 从 2 -> 0
            a = 2.0 - t * (2.0 / self.max_iter)
            
            # 边界处理
            positions = np.clip(positions, self.lb, self.ub)
            
            # 【调试日志】开始计算适应度
            print(f"[GWO-Debug] 开始计算适应度（Fitness）...")
            print(f"[GWO-Debug] 使用Mean Dice模式: {self.use_mean_dice}, 后处理函数: {self.postprocess_func is not None}")
            
            # 【分块多进程】计算所有狼的适应度（Dice分数）
            scores_np = np.zeros(self.num_wolves)
            
            # 准备需要计算的灰狼任务
            wolf_tasks = []
            for wolf_idx in range(self.num_wolves):
                threshold = float(positions[wolf_idx])
                threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                
                # 【缓存机制】检查是否有缓存的阈值结果
                if threshold_rounded in self._threshold_cache:
                    scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                else:
                    # 需要计算的任务
                    wolf_tasks.append((wolf_idx, threshold, threshold_rounded))
            
            print(f"[GWO-Debug] 缓存命中: {self.num_wolves - len(wolf_tasks)}/{self.num_wolves}, 需要计算: {len(wolf_tasks)}")
            
            # 【分块多进程处理】对每个阈值，将数据分块并行计算
            if len(wolf_tasks) > 0:
                # 将数据分块
                num_chunks = (num_samples_sampled + chunk_size - 1) // chunk_size
                print(f"[GWO-Debug] 数据分块: {num_samples_sampled} 样本分为 {num_chunks} 个chunk（每个chunk最多{chunk_size}样本）")
                
                # 对每个需要计算的阈值，使用分块多进程计算
                for task_idx, (wolf_idx, threshold, threshold_rounded) in enumerate(wolf_tasks):
                    if task_idx % max(1, len(wolf_tasks) // 5) == 0:
                        print(f"[GWO-Debug] 处理灰狼 {task_idx+1}/{len(wolf_tasks)} (阈值: {threshold:.4f})")
                    
                    try:
                        # 准备chunk任务
                        chunk_tasks = []
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, num_samples_sampled)
                            probs_chunk = preds[start_idx:end_idx]  # (chunk_size, H, W)
                            masks_chunk = targets[start_idx:end_idx]  # (chunk_size, H, W)
                            chunk_tasks.append((probs_chunk, masks_chunk, threshold, post_processing_params))
                        
                        # 【多进程并行计算chunk】
                        # 检查是否应该使用多进程
                        should_use_multiprocessing = (
                            self.use_multiprocessing and 
                            len(chunk_tasks) > 1 and 
                            max_workers > 1
                        )
                        
                        if should_use_multiprocessing:
                            # 使用多进程并行处理chunk（包括Windows）
                            print(f"[GWO-Debug] 启动多进程处理: {len(chunk_tasks)} 个chunk, max_workers={max_workers}", flush=True)
                            try:
                                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                                    print(f"[GWO-Debug] ProcessPoolExecutor 已创建，提交 {len(chunk_tasks)} 个任务...", flush=True)
                                    futures = {executor.submit(_calculate_fitness_chunk, task): i for i, task in enumerate(chunk_tasks)}
                                    print(f"[GWO-Debug] 所有任务已提交，等待结果...", flush=True)
                                    
                                    # 聚合所有chunk的统计量
                                    total_stats = {
                                        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                                        'empty_mask_count': 0, 'non_empty_mask_count': 0,
                                        'empty_dice_sum': 0.0, 'non_empty_dice_sum': 0.0
                                    }
                                    
                                    completed_count = 0
                                    for future in as_completed(futures):
                                        chunk_idx = futures[future]
                                        completed_count += 1
                                        try:
                                            chunk_stats = future.result()
                                            print(f"[GWO-Debug] Chunk {chunk_idx} 完成 ({completed_count}/{len(chunk_tasks)})", flush=True)
                                            # 聚合统计量
                                            for key in total_stats:
                                                total_stats[key] += chunk_stats[key]
                                        except Exception as e:
                                            print(f"[GWO-Debug] Chunk {chunk_idx} 计算失败: {e}", flush=True)
                                            import traceback
                                            traceback.print_exc()
                                    
                                    print(f"[GWO-Debug] 所有chunk处理完成，开始聚合统计量...", flush=True)
                                
                                # 计算Mean Dice
                                total_samples = total_stats['empty_mask_count'] + total_stats['non_empty_mask_count']
                                if total_samples > 0:
                                    # Mean Dice = (空mask的Dice总和 + 非空mask的Dice总和) / 总样本数
                                    mean_dice = (total_stats['empty_dice_sum'] + total_stats['non_empty_dice_sum']) / total_samples
                                else:
                                    mean_dice = 0.0
                                
                                print(f"[GWO-Debug] 多进程计算完成，Mean Dice: {mean_dice:.4f}", flush=True)
                                
                            except Exception as e:
                                print(f"[GWO-Debug] 多进程执行失败，回退到单进程: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                                # 回退到单进程模式
                                mean_dice = self._calculate_fitness_single_process(preds, targets, threshold, post_processing_params, num_samples_sampled)
                        else:
                            # 单进程模式：直接在主进程计算
                            reason = []
                            if not self.use_multiprocessing:
                                reason.append("use_multiprocessing=False")
                            if len(chunk_tasks) <= 1:
                                reason.append(f"chunk数量={len(chunk_tasks)}")
                            if max_workers <= 1:
                                reason.append(f"max_workers={max_workers}")
                            print(f"[GWO-Debug] 使用单进程模式（原因: {', '.join(reason)}）", flush=True)
                            mean_dice = self._calculate_fitness_single_process(preds, targets, threshold, post_processing_params, num_samples_sampled)
                        
                        scores_np[wolf_idx] = mean_dice
                        self._threshold_cache[threshold_rounded] = mean_dice
                        
                    except Exception as e:
                        print(f"[GWO-Debug] 处理灰狼 {wolf_idx} 失败: {e}")
                        import traceback
                        traceback.print_exc()
                        scores_np[wolf_idx] = 0.0
            
            # 【调试日志】适应度计算完成
            print(f"[GWO-Debug] 适应度计算完成，分数范围: [{scores_np.min():.4f}, {scores_np.max():.4f}]")
            
            # 更新前三名（Alpha, Beta, Delta）
            print(f"[GWO-Debug] 更新前三名（Alpha, Beta, Delta）...")
            positions_np = positions.copy()
            for i in range(self.num_wolves):
                score = float(scores_np[i])
                pos = float(positions_np[i])
                
                if score > alpha_score:
                    # 更新 Alpha，原 Alpha 降为 Beta，原 Beta 降为 Delta
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = alpha_score, alpha_pos
                    alpha_score, alpha_pos = score, pos
                elif score > beta_score:
                    # 更新 Beta，原 Beta 降为 Delta
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = score, pos
                elif score > delta_score:
                    # 更新 Delta
                    delta_score, delta_pos = score, pos
            
            # 更新每只狼的位置（基于 Alpha, Beta, Delta 的位置）
            # CPU模式：使用numpy计算
            r1 = np.random.random(self.num_wolves)
            r2 = np.random.random(self.num_wolves)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - positions)
            X1 = alpha_pos - A1 * D_alpha
            
            r1 = np.random.random(self.num_wolves)
            r2 = np.random.random(self.num_wolves)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - positions)
            X2 = beta_pos - A2 * D_beta
            
            r1 = np.random.random(self.num_wolves)
            r2 = np.random.random(self.num_wolves)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - positions)
            X3 = delta_pos - A3 * D_delta
            
            # 狼的位置更新为三者平均
            positions = (X1 + X2 + X3) / 3.0
            
            # 【调试日志】迭代完成
            print(f"[GWO-Debug] 迭代 {t+1} 完成，最佳阈值: {alpha_pos:.4f}, 最佳分数: {alpha_score:.4f}")
            
            # 调用进度回调函数（如果提供）
            if self.progress_callback is not None:
                try:
                    print(f"[GWO-Debug] 调用进度回调函数...")
                    self.progress_callback(t + 1, self.max_iter, alpha_score, alpha_pos)
                    print(f"[GWO-Debug] 进度回调函数执行完成")
                except Exception as e:
                    # 如果回调函数出错，不影响优化过程
                    print(f"[GWO-Debug] 进度回调函数出错: {e}")
                    pass
        
        # 【效率优化】最终评估：使用全量样本重新计算最佳阈值的Dice分数
        if self.sample_ratio < 1.0 and alpha_pos is not None:
            # 保存采样阶段的Dice值，用于日志对比
            sampled_dice = alpha_score
            print(f">>> [GWO最终评估] 使用全量 {num_samples} 个样本重新计算最佳阈值 {alpha_pos:.4f} 的Dice...")
            # 【关键修复】确保使用全量数据（full_preds, full_targets）进行最终评估
            final_dice = self._evaluate_threshold_full(full_preds, full_targets, alpha_pos, use_gpu, device)
            if final_dice is not None:
                alpha_score = final_dice
                print(f">>> [GWO最终评估] 全量样本Dice: {alpha_score:.4f} (采样Dice: {sampled_dice:.4f})")
            else:
                print(f">>> [GWO最终评估警告] 全量评估失败，使用采样Dice: {sampled_dice:.4f}")
        
        return alpha_pos, alpha_score
    
    def _calculate_fitness_single_process(self, preds, targets, threshold, post_processing_params, num_samples):
        """
        单进程模式：计算适应度（用于回退或Windows系统）
        
        Args:
            preds: (N, H, W) float16 numpy数组
            targets: (N, H, W) uint8 numpy数组
            threshold: float，阈值
            post_processing_params: dict，后处理参数
            num_samples: int，样本数
        
        Returns:
            mean_dice: float，Mean Dice分数
        """
        try:
            has_postprocess = post_processing_params.get('has_postprocess', False)
            smart_post_cfg_dict = post_processing_params.get('smart_post_cfg_dict', None)
            
            dice_scores = []
            for i in range(num_samples):
                prob_map = preds[i]  # (H, W)
                mask_gt = targets[i]  # (H, W)
                
                # 二值化
                pred_mask = (prob_map >= threshold).astype(np.uint8)
                
                # 应用后处理（如果启用）
                if has_postprocess:
                    try:
                        from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
                        from skimage.morphology import remove_small_objects
                        
                        pred_mask_bool = pred_mask.astype(bool)
                        
                        if smart_post_cfg_dict is not None:
                            # 有智能配置：最小干预 + 智能策略
                            if smart_post_cfg_dict.get('fill_holes', True):
                                pred_mask_bool = binary_fill_holes(pred_mask_bool)
                            
                            method = smart_post_cfg_dict.get('method', 'baseline')
                            params = smart_post_cfg_dict.get('params', {})
                            if method != 'baseline':
                                from utils.smart_postprocessing import _apply_strategy
                                pred_mask_bool = _apply_strategy(pred_mask_bool.astype(np.float32), method, params).astype(bool)
                        else:
                            # 无智能配置：使用默认的保守后处理
                            min_size = 150
                            pred_mask_bool = remove_small_objects(pred_mask_bool, min_size=min_size)
                            pred_mask_bool = binary_fill_holes(pred_mask_bool)
                            pred_mask_bool = binary_opening(pred_mask_bool, structure=np.ones((3,3)))
                            pred_mask_bool = binary_closing(pred_mask_bool, structure=np.ones((3,3)))
                        
                        pred_mask = pred_mask_bool.astype(np.uint8)
                    except Exception:
                        # 后处理失败则回退原始预测
                        pred_mask = (prob_map >= threshold).astype(np.uint8)
                
                # 计算Dice
                pred_flat = pred_mask.flatten().astype(np.float32)
                gt_flat = mask_gt.flatten().astype(np.float32)
                mask_sum = gt_flat.sum()
                empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)
                pred_sum = pred_flat.sum()
                
                if mask_sum <= empty_threshold:
                    dice_sample = 1.0 if pred_sum <= 1e-7 else 0.0
                else:
                    intersection = (pred_flat * gt_flat).sum()
                    dice_den = 2.0 * intersection + pred_sum + mask_sum
                    dice_sample = 0.0 if dice_den < 1e-7 else (2.0 * intersection) / dice_den
                
                dice_scores.append(float(dice_sample))
            
            mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
            return mean_dice
        except Exception as e:
            print(f"[GWO-Debug] 单进程计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _evaluate_threshold_full(self, preds, targets, threshold, use_gpu, device):
        """
        使用全量样本评估阈值的Dice分数（用于最终评估）
        
        Args:
            preds: 全量概率图 (N, H, W)
            targets: 全量真实标签 (N, H, W)
            threshold: 阈值
            use_gpu: 是否使用GPU
            device: 计算设备
            
        Returns:
            dice: Dice分数，如果计算失败返回None
        """
        # 【调试】验证全量数据形状
        if isinstance(preds, torch.Tensor):
            full_num_samples = preds.shape[0]
        else:
            full_num_samples = preds.shape[0]
        
        try:
            if not self.use_mean_dice:
                # Global Dice模式：展平计算
                if isinstance(preds, torch.Tensor):
                    preds_flat = preds.flatten().float()
                    targets_flat = targets.flatten().float()
                else:
                    preds_flat = torch.from_numpy(preds.flatten()).float()
                    targets_flat = torch.from_numpy(targets.flatten()).float()
                
                if use_gpu:
                    preds_flat = preds_flat.to(device)
                    targets_flat = targets_flat.to(device)
                
                dice = self._calculate_dice_batch(preds_flat, targets_flat, 
                                                  torch.tensor([threshold], device=device if use_gpu else None))
                return float(dice[0].item() if isinstance(dice, torch.Tensor) else dice[0])
            else:
                # Mean Dice模式：按样本分别计算
                if isinstance(preds, torch.Tensor):
                    preds_np = preds.cpu().numpy() if use_gpu else preds.numpy()
                    targets_np = targets.cpu().numpy() if use_gpu else targets.numpy()
                else:
                    preds_np = preds
                    targets_np = targets
                
                num_samples = preds_np.shape[0]
                # 【调试】验证样本数量
                if num_samples < 500:  # 如果样本数太少，可能是采样数据而不是全量数据
                    print(f">>> [GWO警告] 全量评估样本数: {num_samples}，可能使用了采样数据而非全量数据")
                
                # 使用单进程计算（全量数据通常较大，避免多进程开销）
                post_processing_params = {
                    'has_postprocess': (self.postprocess_func is not None),
                    'smart_post_cfg_dict': None
                }
                if hasattr(self, 'smart_post_cfg') and self.smart_post_cfg:
                    post_processing_params['smart_post_cfg_dict'] = {
                        'method': self.smart_post_cfg.get('method', 'baseline'),
                        'params': self.smart_post_cfg.get('params', {}),
                        'min_size': 150 if not self.smart_post_cfg else 0,
                        'use_morphology': True if not self.smart_post_cfg else False,
                        'fill_holes': True
                    }
                
                mean_dice = self._calculate_fitness_single_process(preds_np, targets_np, threshold, post_processing_params, num_samples)
                return mean_dice
        except Exception as e:
            print(f">>> [GWO最终评估错误] 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_dice(self, preds, targets, threshold):
        """
        快速计算 Dice 系数（CPU版本，用于兼容性）
        
        Args:
            preds: 展平的概率图（一维数组）
            targets: 展平的真实标签（一维数组）
            threshold: 二值化阈值
            
        Returns:
            dice: Dice 系数
        """
        # 二值化预测
        pred_mask = (preds >= threshold).astype(np.float32)
        targets_float = targets.astype(np.float32)
        
        # 计算交集和并集
        intersection = np.sum(pred_mask * targets_float)
        union = np.sum(pred_mask) + np.sum(targets_float)
        
        # 避免除零
        if union < 1e-7:
            return 1.0 if intersection < 1e-7 else 0.0
        
        # Dice = 2 * intersection / union
        return (2.0 * intersection) / union
    
    def _calculate_dice_batch(self, preds_flat, targets_flat, thresholds):
        """
        GPU加速：批量计算多个阈值的Dice系数（显存优化版本）
        
        Args:
            preds_flat: 展平的概率图（一维tensor，在GPU上）
            targets_flat: 展平的真实标签（一维tensor，在GPU上）
            thresholds: 阈值tensor（一维tensor，形状为[num_wolves]，在GPU上）
            
        Returns:
            dice_scores: Dice分数tensor（一维tensor，形状为[num_wolves]，在GPU上）
        """
        num_wolves = thresholds.shape[0]
        num_pixels = preds_flat.shape[0]
        
        # 【显存优化】如果数据量太大，使用循环计算而不是批量扩展
        # 估算显存占用：expand会创建 [M, N] 的tensor，约 4*M*N 字节
        # 如果超过500MB，使用循环方式
        estimated_memory_mb = 4 * num_wolves * num_pixels / (1024 * 1024)
        
        if estimated_memory_mb > 500:
            # 使用循环方式，每次只计算一只狼（节省显存）
            dice_scores = []
            targets_sum = targets_flat.sum()  # 只计算一次
            
            for i in range(num_wolves):
                threshold = thresholds[i]
                # 二值化
                pred_mask = (preds_flat >= threshold).float()
                # 计算交集和并集
                intersection = (pred_mask * targets_flat).sum()
                pred_sum = pred_mask.sum()
                union = pred_sum + targets_sum
                # 计算Dice
                smooth = 1e-7
                if union < smooth:
                    dice = 1.0 if intersection < smooth else 0.0
                else:
                    dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_scores.append(dice)
                # 及时删除中间变量
                del pred_mask, intersection, pred_sum
            
            dice_scores = torch.stack(dice_scores)
        else:
            # 使用批量计算（速度快，但显存占用大）
            thresholds_expanded = thresholds.unsqueeze(1)  # [M, 1]
            
            # 批量二值化：preds >= threshold for each threshold
            # 使用广播，避免expand创建大tensor
            pred_masks = (preds_flat.unsqueeze(0) >= thresholds_expanded).float()  # [M, N]
            
            # 批量计算交集和并集
            targets_expanded = targets_flat.unsqueeze(0)  # [1, N]
            intersections = (pred_masks * targets_expanded).sum(dim=1)  # [M]
            pred_sums = pred_masks.sum(dim=1)  # [M]
            target_sum = targets_flat.sum()  # scalar，只计算一次
            
            # 批量计算Dice
            unions = pred_sums + target_sum  # [M]
            
            # 避免除零（使用smooth项）
            smooth = 1e-7
            dice_scores = (2.0 * intersections + smooth) / (unions + smooth)
            
            # 处理特殊情况：如果union为0，根据intersection判断
            zero_union_mask = unions < smooth
            zero_intersection_mask = intersections < smooth
            dice_scores[zero_union_mask & zero_intersection_mask] = 1.0  # 两者都为0，Dice=1
            dice_scores[zero_union_mask & ~zero_intersection_mask] = 0.0  # union=0但intersection>0，Dice=0
            
            # 清理中间变量
            del pred_masks, targets_expanded, intersections, pred_sums
        
        return dice_scores


