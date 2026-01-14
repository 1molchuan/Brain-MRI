# -*- coding: utf-8 -*-
"""
从utils.process_pool模块
"""
from utils.common import *

# ==================== 全局进程池管理器（单例模式）====================
class ProcessPoolManager:
    """
    全局进程池管理器，用于复用进程池（Windows兼容）
    进程池在第一次创建后保留，后续调用复用，避免重复创建的开销
    """
    _instance = None
    _lock = threading.Lock()
    _pool = None
    _pool_size = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProcessPoolManager, cls).__new__(cls)
        return cls._instance
    
    def get_pool(self, pool_size=None):
        """
        获取进程池，如果不存在则创建
        
        Args:
            pool_size: 进程池大小，如果为None则使用CPU核心数-1
            
        Returns:
            multiprocessing.Pool: 进程池对象，如果创建失败则返回None
        """
        if pool_size is None:
            pool_size = max(1, mp.cpu_count() - 1)  # 保留一个核心给主进程
        
        # 如果进程池已存在且大小匹配，直接返回
        if self._pool is not None and self._pool_size == pool_size:
            try:
                # 测试进程池是否仍然有效
                self._pool._check_running()
                return self._pool
            except (ValueError, AssertionError):
                # 进程池已关闭，需要重新创建
                self._pool = None
                self._pool_size = None
        
        # 创建新的进程池
        if self._pool is None:
            try:
                # Windows下使用spawn方式（默认）
                ctx = mp.get_context('spawn')
                self._pool = ctx.Pool(processes=pool_size)
                self._pool_size = pool_size
                print(f"[进程池] 创建进程池，大小: {pool_size} (CPU核心数: {mp.cpu_count()})")
            except Exception as e:
                print(f"[进程池] 创建失败: {e}，将使用单进程模式")
                self._pool = None
                self._pool_size = None
        
        return self._pool
    
    def close_pool(self):
        """关闭进程池（通常不需要手动调用，程序退出时自动清理）"""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
                self._pool = None
                self._pool_size = None
                print("[进程池] 进程池已关闭")
            except Exception as e:
                print(f"[进程池] 关闭失败: {e}")


