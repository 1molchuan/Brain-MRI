# -*- coding: utf-8 -*-
"""
从utils.gwo_optimizer模块
"""
from utils.common import *

class GreyWolfThresholdOptimizer:
    """
    使用灰狼优化算法(GWO)寻找最佳分割阈值
    目标函数：Dice系数
    
    相比线性扫描，GWO 能够更智能地搜索阈值空间，在更少的迭代次数内找到更优解。
    """
    
    def __init__(self, num_wolves=10, max_iter=20, progress_callback=None, use_multiprocessing=True, 
                 use_mean_dice=False, postprocess_func=None, sample_ratio=1.0, metrics_func=None):
        """
        Args:
            num_wolves: 灰狼数量（种群大小），默认10
            max_iter: 最大迭代次数，默认20
            progress_callback: 进度回调函数，接收 (iteration, max_iter, best_score, best_threshold) 参数
            use_multiprocessing: 是否在CPU模式下使用多进程（默认True）
            use_mean_dice: 是否使用Mean Dice（每个样本分别计算再平均），默认False（使用Global Dice）
            postprocess_func: 后处理函数，接收(pred_mask, prob_map)返回处理后的mask，默认None（不使用后处理）
            sample_ratio: 采样比例，在迭代过程中只使用部分样本计算Fitness（0.0-1.0），默认1.0（使用全部样本）
        """
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.progress_callback = progress_callback
        self.use_multiprocessing = use_multiprocessing
        self.use_mean_dice = use_mean_dice
        self.postprocess_func = postprocess_func
        self.sample_ratio = sample_ratio
        self.metrics_func = metrics_func
        # 搜索空间 [0.05, 0.95]（扩大范围，覆盖更广的阈值空间）
        self.lb = 0.05
        self.ub = 0.95
        # 【缓存机制】缓存阈值和对应的Dice分数，避免重复计算
        self._threshold_cache = {}  # {threshold_rounded: dice_score}
        self._cache_tolerance = 0.001  # 阈值差值<0.001时复用结果

    def optimize(self, preds, targets, device=None):
        """
        使用GWO算法优化阈值（GPU加速版本，带自动回退）
        
        Args:
            preds: 模型输出的概率图，可以是 torch.Tensor 或 numpy.ndarray
                  形状为 (N, H, W) 或 (N, C, H, W)
            targets: 真实标签，形状与 preds 相同
            device: 计算设备（如果为None，自动检测preds的设备）
            
        Returns:
            best_threshold: 最佳阈值
            best_dice: 最佳Dice分数
        """
        # 检测设备
        use_gpu = False
        if isinstance(preds, torch.Tensor):
            device = device or preds.device
            use_gpu = device.type == 'cuda'
        elif device is None:
            use_gpu = torch.cuda.is_available()
            device = torch.device('cuda' if use_gpu else 'cpu')
        else:
            device = torch.device(device)
            use_gpu = device.type == 'cuda'
        
        # 检查显存是否充足
        if use_gpu:
            try:
                # 估算数据大小
                if isinstance(preds, np.ndarray):
                    data_size_mb = preds.nbytes / (1024 * 1024)
                else:
                    data_size_mb = preds.numel() * 4 / (1024 * 1024)  # float32 = 4 bytes
                
                # 检查可用显存
                if torch.cuda.is_available():
                    free_memory_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024) - torch.cuda.memory_allocated(device) / (1024 * 1024)
                    # 如果数据大小超过可用显存的30%，使用CPU
                    if data_size_mb > free_memory_mb * 0.3:
                        print(f"[GWO] 显存不足（数据: {data_size_mb:.1f}MB, 可用: {free_memory_mb:.1f}MB），回退到CPU模式")
                        use_gpu = False
                        device = torch.device('cpu')
            except Exception as e:
                print(f"[GWO] 显存检查失败: {e}，回退到CPU模式")
                use_gpu = False
                device = torch.device('cpu')
        
        # 转换为torch tensor并移到指定设备
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()
        
        # 尝试移到GPU，如果失败则回退到CPU
        try:
            preds = preds.to(device)
            targets = targets.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[GWO] GPU显存不足，回退到CPU模式")
                device = torch.device('cpu')
                use_gpu = False
                preds = preds.cpu()
                targets = targets.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
        
        # 统一维度：如果是4D，取第一个通道
        if preds.ndim == 4:
            preds = preds[:, 0]  # (N, H, W)
        if targets.ndim == 4:
            targets = targets[:, 0]  # (N, H, W)
        
        # 【关键修复】保存原始形状（用于Mean Dice计算）
        num_samples = preds.shape[0]
        spatial_shape = preds.shape[1:]  # (H, W)
        
        # 【效率优化】保存全量数据，用于最终评估
        # 在迭代过程中使用采样数据，最终评估时使用全量数据
        if isinstance(preds, torch.Tensor):
            full_preds = preds.clone()
            full_targets = targets.clone()
        else:
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
            if isinstance(preds, torch.Tensor):
                preds = preds[sample_indices]
                targets = targets[sample_indices]
            else:
                preds = preds[sample_indices]
                targets = targets[sample_indices]
            num_samples_sampled = len(sample_indices)
            print(f">>> [GWO效率优化] 采样计算: {num_samples_sampled}/{num_samples} 样本 ({100*self.sample_ratio:.0f}%)")
        
        # 根据是否使用Mean Dice决定数据处理方式
        if self.use_mean_dice:
            # Mean Dice模式：保持(N, H, W)形状，按样本分别计算
            # 如果是CPU模式，转换为numpy数组
            if not use_gpu:
                preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
                targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            else:
                # GPU模式：保持在GPU上，但需要按样本处理
                preds_np = None
                targets_np = None
            preds_flat = None
            targets_flat = None
            preds_flat_np = None
            targets_flat_np = None
            pool = None
        else:
            # Global Dice模式：展平为一维tensor（原有逻辑）
            preds_flat = preds.flatten().float()
            targets_flat = targets.flatten().float()
            
            # 如果是CPU模式，转换为numpy数组（多进程和单进程都需要）
            if not use_gpu:
                preds_flat_np = preds_flat.cpu().numpy()
                targets_flat_np = targets_flat.cpu().numpy()
                # 如果启用多进程，获取全局进程池管理器
                if self.use_multiprocessing:
                    pool_manager = ProcessPoolManager()
                    pool = pool_manager.get_pool(pool_size=min(self.num_wolves, mp.cpu_count() - 1))
                else:
                    pool = None
            else:
                preds_flat_np = None
                targets_flat_np = None
                pool = None
        
        # 初始化狼群位置（阈值）
        positions = np.random.uniform(self.lb, self.ub, self.num_wolves)
        
        if use_gpu:
            positions_tensor = torch.from_numpy(positions).float().to(device)
        else:
            positions_tensor = None
        
        # 【性能优化】Mean Dice模式：如果使用GPU，预先转换到CPU（避免每次迭代都转换）
        if self.use_mean_dice and use_gpu:
            preds_cpu = preds.cpu().numpy()  # (N, H, W) - 采样后的数据
            targets_cpu = targets.cpu().numpy()  # (N, H, W) - 采样后的数据
        else:
            preds_cpu = None
            targets_cpu = None
        
        # 【缓存机制】清空缓存（每次optimize调用时）
        self._threshold_cache.clear()
        
        # 初始化 Alpha, Beta, Delta 狼 (前三名)
        alpha_pos, alpha_score = 0.5, -float('inf')
        beta_pos, beta_score = 0.5, -float('inf')
        delta_pos, delta_score = 0.5, -float('inf')
        
        for t in range(self.max_iter):
            # 线性衰减参数 a 从 2 -> 0
            a = 2.0 - t * (2.0 / self.max_iter)
            
            # 边界处理
            positions = np.clip(positions, self.lb, self.ub)
            
            # 计算所有狼的适应度（Dice分数）
            if self.use_mean_dice:
                # Mean Dice模式：按样本分别计算Dice，然后取平均
                if use_gpu:
                    prob_maps = preds_cpu
                    mask_gts = targets_cpu
                else:
                    prob_maps = preds_np
                    mask_gts = targets_np
                
                scores_np = np.zeros(self.num_wolves)
                for wolf_idx in range(self.num_wolves):
                    threshold = float(positions[wolf_idx])
                    
                    # 【缓存机制】检查是否有缓存的阈值结果
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                        continue
                    
                    # 构建当前阈值下的预测mask列表
                    pred_masks_list = []
                    for sample_idx in range(num_samples_sampled):
                        prob_map = prob_maps[sample_idx]  # (H, W)
                        pred_mask = (prob_map >= threshold).astype(np.float32)
                        
                        # 应用后处理（如果提供）
                        if self.postprocess_func is not None:
                            try:
                                pred_mask_tensor = torch.from_numpy(pred_mask).float()
                                prob_map_tensor = torch.from_numpy(prob_map).float()
                                pred_mask_processed = self.postprocess_func(pred_mask_tensor, prob_map_tensor)
                                # 转换回numpy
                                if isinstance(pred_mask_processed, torch.Tensor):
                                    pred_mask = pred_mask_processed.detach().cpu().numpy()
                                else:
                                    pred_mask = np.asarray(pred_mask_processed)
                                # 确保是2D数组
                                if pred_mask.ndim > 2:
                                    pred_mask = pred_mask.squeeze()
                                pred_mask = pred_mask.astype(np.float32)
                                if pred_mask.shape != prob_map.shape:
                                    pred_mask = (prob_map >= threshold).astype(np.float32)
                            except Exception:
                                # 后处理失败则回退原始预测
                                pred_mask = (prob_map >= threshold).astype(np.float32)
                        pred_masks_list.append(pred_mask)
                    
                    # 使用统一的指标计算函数（calculate_batch_metrics）作为单一真理源
                    if self.metrics_func is not None:
                        preds_batch_np = np.stack(pred_masks_list, axis=0)  # (N, H, W)
                        targets_batch_np = mask_gts[:num_samples_sampled]
                        preds_batch = torch.from_numpy(preds_batch_np).float().unsqueeze(1)   # (N,1,H,W)
                        targets_batch = torch.from_numpy(targets_batch_np).float().unsqueeze(1)  # (N,1,H,W)
                        metrics = self.metrics_func(preds_batch, targets_batch)
                        dice_scores = metrics.get("dice", [])
                        balanced_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
                    else:
                        # 回退逻辑：保持原有的平衡Dice计算
                        empty_dice_scores = []
                        non_empty_dice_scores = []
                        for sample_idx in range(num_samples_sampled):
                            mask_gt = mask_gts[sample_idx]    # (H, W)
                            gt_flat = mask_gt.flatten()
                            mask_sum = gt_flat.sum()
                            empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)  # 0.1%像素
                            
                            pred_mask = pred_masks_list[sample_idx]
                            pred_flat = pred_mask.flatten()
                            pred_sum = pred_flat.sum()
                            
                            if mask_sum <= empty_threshold:
                                dice_sample = 1.0 if pred_sum <= 1e-7 else 0.0
                                empty_dice_scores.append(float(dice_sample))
                            else:
                                intersection = (pred_flat * gt_flat).sum()
                                dice_den = 2.0 * intersection + pred_sum + mask_sum
                                dice_sample = 0.0 if dice_den < 1e-7 else (2.0 * intersection) / dice_den
                                non_empty_dice_scores.append(float(dice_sample))
                        
                        empty_mean = np.mean(empty_dice_scores) if empty_dice_scores else 1.0
                        non_empty_mean = np.mean(non_empty_dice_scores) if non_empty_dice_scores else 0.0
                        if len(empty_dice_scores) == 0:
                            balanced_dice = non_empty_mean
                        elif len(non_empty_dice_scores) == 0:
                            balanced_dice = empty_mean
                        else:
                            balanced_dice = (empty_mean + non_empty_mean) / 2.0
                    
                    scores_np[wolf_idx] = float(balanced_dice)
                    # 【缓存机制】缓存结果
                    self._threshold_cache[threshold_rounded] = float(balanced_dice)
            elif use_gpu:
                # GPU模式：使用批量计算（Global Dice）
                positions_tensor = torch.from_numpy(positions).float().to(device)
                positions_tensor = torch.clamp(positions_tensor, self.lb, self.ub)
                # 【缓存机制】检查缓存，只计算未缓存的阈值
                scores_np = np.zeros(self.num_wolves)
                uncached_indices = []
                uncached_positions = []
                for wolf_idx in range(self.num_wolves):
                    threshold = float(positions[wolf_idx])
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                    else:
                        uncached_indices.append(wolf_idx)
                        uncached_positions.append(threshold)
                
                # 只计算未缓存的阈值
                if uncached_positions:
                    uncached_tensor = torch.tensor(uncached_positions, device=device)
                    uncached_tensor = torch.clamp(uncached_tensor, self.lb, self.ub)
                    uncached_scores = self._calculate_dice_batch(preds_flat, targets_flat, uncached_tensor)
                    uncached_scores_np = uncached_scores.cpu().numpy()
                    # 填充结果并缓存
                    for i, wolf_idx in enumerate(uncached_indices):
                        threshold_rounded = round(uncached_positions[i] / self._cache_tolerance) * self._cache_tolerance
                        scores_np[wolf_idx] = float(uncached_scores_np[i])
                        self._threshold_cache[threshold_rounded] = float(uncached_scores_np[i])
            elif pool is not None:
                # CPU多进程模式：并行计算每只狼的Dice（Global Dice）
                # 【缓存机制】检查缓存，只计算未缓存的阈值
                scores_np = np.zeros(self.num_wolves)
                uncached_args = []
                uncached_indices = []
                for wolf_idx, pos in enumerate(positions):
                    threshold = float(pos)
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                    else:
                        uncached_args.append((preds_flat_np, targets_flat_np, threshold))
                        uncached_indices.append(wolf_idx)
                
                # 只计算未缓存的阈值
                if uncached_args:
                    uncached_scores = np.array(pool.map(_calculate_dice_worker, uncached_args))
                    for i, wolf_idx in enumerate(uncached_indices):
                        threshold_rounded = round(float(uncached_args[i][2]) / self._cache_tolerance) * self._cache_tolerance
                        scores_np[wolf_idx] = float(uncached_scores[i])
                        self._threshold_cache[threshold_rounded] = float(uncached_scores[i])
            else:
                # CPU单进程模式：循环计算（Global Dice）
                # 【缓存机制】检查缓存，只计算未缓存的阈值
                scores_np = np.zeros(self.num_wolves)
                for wolf_idx in range(self.num_wolves):
                    threshold = float(positions[wolf_idx])
                    threshold_rounded = round(threshold / self._cache_tolerance) * self._cache_tolerance
                    if threshold_rounded in self._threshold_cache:
                        scores_np[wolf_idx] = self._threshold_cache[threshold_rounded]
                    else:
                        dice = self._calculate_dice(preds_flat_np, targets_flat_np, threshold)
                        scores_np[wolf_idx] = float(dice)
                        self._threshold_cache[threshold_rounded] = float(dice)
            
            positions_np = positions.copy()
            
            # 更新前三名（Alpha, Beta, Delta）
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
            if use_gpu:
                # GPU模式：在GPU上批量计算位置更新
                alpha_pos_tensor = torch.tensor(alpha_pos, device=device)
                beta_pos_tensor = torch.tensor(beta_pos, device=device)
                delta_pos_tensor = torch.tensor(delta_pos, device=device)
                
                # 生成随机数（在GPU上）
                r1 = torch.rand(self.num_wolves, device=device)
                r2 = torch.rand(self.num_wolves, device=device)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = torch.abs(C1 * alpha_pos_tensor - positions_tensor)
                X1 = alpha_pos_tensor - A1 * D_alpha
                
                r1 = torch.rand(self.num_wolves, device=device)
                r2 = torch.rand(self.num_wolves, device=device)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = torch.abs(C2 * beta_pos_tensor - positions_tensor)
                X2 = beta_pos_tensor - A2 * D_beta
                
                r1 = torch.rand(self.num_wolves, device=device)
                r2 = torch.rand(self.num_wolves, device=device)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = torch.abs(C3 * delta_pos_tensor - positions_tensor)
                X3 = delta_pos_tensor - A3 * D_delta
                
                # 狼的位置更新为三者平均
                positions_tensor = (X1 + X2 + X3) / 3.0
                positions = positions_tensor.cpu().numpy()
            else:
                # CPU模式：使用numpy计算
                positions_tensor = torch.from_numpy(positions).float()
                
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
            
            # 调用进度回调函数（如果提供）
            if self.progress_callback is not None:
                try:
                    self.progress_callback(t + 1, self.max_iter, alpha_score, alpha_pos)
                except Exception as e:
                    # 如果回调函数出错，不影响优化过程
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
                pred_masks_list = []
                for sample_idx in range(num_samples):
                    prob_map = preds_np[sample_idx]
                    pred_mask = (prob_map >= threshold).astype(np.float32)
                    
                    # 应用后处理（如果提供）
                    if self.postprocess_func is not None:
                        try:
                            pred_mask_tensor = torch.from_numpy(pred_mask).float()
                            prob_map_tensor = torch.from_numpy(prob_map).float()
                            pred_mask_processed = self.postprocess_func(pred_mask_tensor, prob_map_tensor)
                            if isinstance(pred_mask_processed, torch.Tensor):
                                pred_mask = pred_mask_processed.detach().cpu().numpy()
                            else:
                                pred_mask = np.asarray(pred_mask_processed)
                            if pred_mask.ndim > 2:
                                pred_mask = pred_mask.squeeze()
                            pred_mask = pred_mask.astype(np.float32)
                            if pred_mask.shape != prob_map.shape:
                                pred_mask = (prob_map >= threshold).astype(np.float32)
                        except Exception:
                            pred_mask = (prob_map >= threshold).astype(np.float32)
                    pred_masks_list.append(pred_mask)
                
                if self.metrics_func is not None:
                    preds_batch_np = np.stack(pred_masks_list, axis=0)
                    preds_batch = torch.from_numpy(preds_batch_np).float().unsqueeze(1)
                    targets_batch = torch.from_numpy(targets_np).float().unsqueeze(1)
                    metrics = self.metrics_func(preds_batch, targets_batch)
                    dice_scores = metrics.get("dice", [])
                    balanced_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
                else:
                    # 回退到原有的平衡Dice计算
                    empty_dice_scores = []
                    non_empty_dice_scores = []
                    for sample_idx in range(num_samples):
                        mask_gt = targets_np[sample_idx]
                        gt_flat = mask_gt.flatten()
                        mask_sum = gt_flat.sum()
                        empty_threshold = max(1e-7, float(gt_flat.size) * 0.001)
                        pred_mask = pred_masks_list[sample_idx]
                        pred_flat = pred_mask.flatten()
                        pred_sum = pred_flat.sum()
                        if mask_sum <= empty_threshold:
                            dice_sample = 1.0 if pred_sum <= 1e-7 else 0.0
                            empty_dice_scores.append(float(dice_sample))
                        else:
                            intersection = (pred_flat * gt_flat).sum()
                            dice_den = 2.0 * intersection + pred_sum + mask_sum
                            dice_sample = 0.0 if dice_den < 1e-7 else (2.0 * intersection) / dice_den
                            non_empty_dice_scores.append(float(dice_sample))
                    empty_mean = np.mean(empty_dice_scores) if empty_dice_scores else 1.0
                    non_empty_mean = np.mean(non_empty_dice_scores) if non_empty_dice_scores else 0.0
                    if len(empty_dice_scores) == 0:
                        balanced_dice = non_empty_mean
                    elif len(non_empty_dice_scores) == 0:
                        balanced_dice = empty_mean
                    else:
                        balanced_dice = (empty_mean + non_empty_mean) / 2.0
                
                return float(balanced_dice)
        except Exception as e:
            print(f">>> [GWO最终评估] 计算失败: {e}，使用采样Dice")
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


