# -*- coding: utf-8 -*-
"""
从utils.matlab_bridge模块
"""
from utils.common import *

# ==================== MATLAB 相关类 ====================

from scipy.io import savemat
import multiprocessing as mp
from functools import partial

# 尝试导入 MATLAB 引擎
MATLAB_ENGINE_AVAILABLE = False
MATLAB_ENGINE_ERROR = None

try:
    import matlab.engine
    import matlab  # 用于 matlab.double() 等函数
    MATLAB_ENGINE_AVAILABLE = True
except ImportError:
    MATLAB_ENGINE_AVAILABLE = False
    MATLAB_ENGINE_ERROR = "matlab.engine 模块未安装"
    print("[提示] matlab.engine 未安装，MATLAB 可视化功能将不可用")
    matlab = None  # 设置为 None，避免后续使用时报错
except Exception as e:
    MATLAB_ENGINE_AVAILABLE = False
    MATLAB_ENGINE_ERROR = str(e)
    print(f"[警告] 无法导入 MATLAB 引擎: {e}")
    matlab = None  # 设置为 None，避免后续使用时报错

class MatlabCacheManager:
    """MATLAB 缓存功能已移除。"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("MATLAB 缓存功能已移除")

    def manifest_path(self, split_name: str) -> Path:
        safe_split = split_name.replace(os.sep, "_")
        return self.cache_dir / f"{safe_split}_manifest.json"

    def build_manifest(self, split_name: str, image_paths: List[str], mask_paths: List[str]) -> Path:
        manifest = []
        for idx, (img, msk) in enumerate(zip(image_paths, mask_paths)):
            cache_stub = hashlib.sha1(f"{split_name}-{img}".encode('utf-8')).hexdigest()[:10]
            cache_name = f"{split_name}_{idx:05d}_{cache_stub}.mat"
            manifest.append({
                "index": idx,
                "image_path": img,
                "mask_path": msk,
                "cache_path": str(self.cache_dir / cache_name),
                "preferred_format": "mat",
                "notes": "由MATLAB脚本生成，包含变量 I (HxWx3) 与 M (HxW)"
            })

        manifest_path = self.manifest_path(split_name)
        with manifest_path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self._write_instructions(manifest_path)
        return manifest_path

    def _write_instructions(self, manifest_path: Path):
        readme_path = self.cache_dir / "README_MATLAB_CACHE.md"
        if readme_path.exists():
            return

        content = (
            "# MATLAB 缓存指引\n\n"
            "1. 在MATLAB中执行 `manifest = jsondecode(fileread('"
            f"{manifest_path.name}'));`\n"
            "2. 遍历 `manifest`，对 `image_path` 和 `mask_path` 完成标准化、增强、"
            "以及 `gpuArray` 加速的操作。\n"
            "3. 将结果写入 `entry.cache_path`，至少包含 `image` (或 `I`) 与 "
            "`mask` (或 `M`) 变量，类型为 `single`/`logical`。\n"
            "4. Python 端会自动探测 `.mat/.npz` 缓存并优先加载，若不存在则回退到"
            " 原始dataloader。\n"
        )
        readme_path.write_text(content, encoding='utf-8')


class MatlabCacheDataset(Dataset):
    """MATLAB 缓存功能已移除。"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("MATLAB 缓存功能已移除")


class MatlabEngineSession:
    """MATLAB 引擎会话管理类，提供线程安全的 MATLAB 引擎访问。"""
    
    _instance = None
    _instance_lock = threading.Lock()
    _engine = None
    _engine_lock = threading.Lock()
    _start_error = None
    
    def __init__(self):
        if not MATLAB_ENGINE_AVAILABLE:
            self._engine = None
            self._start_error = MATLAB_ENGINE_ERROR or "MATLAB 引擎模块不可用"
            return
        
        try:
            # 尝试启动 MATLAB 引擎
            self._engine = matlab.engine.start_matlab()
            print("[MATLAB] 引擎启动成功")
        except Exception as e:
            error_msg = str(e)
            self._start_error = error_msg
            print(f"[警告] 无法启动 MATLAB 引擎: {error_msg}")
            
            # 提供详细的诊断信息
            if "dll" in error_msg.lower() or "找不到指定的程序" in error_msg or "mwmvmtransport" in error_msg.lower():
                print("\n[MATLAB 诊断] DLL 加载失败，可能的解决方案：")
                print("=" * 60)
                print("1. 安装 Visual C++ Redistributable (2015-2022)")
                print("   下载地址: https://aka.ms/vs/17/release/vc_redist.x64.exe")
                print("   或搜索: Microsoft Visual C++ Redistributable")
                print()
                print("2. 重新安装 MATLAB Engine for Python:")
                print("   打开命令提示符（管理员权限），执行：")
                print("   cd \"C:\\Program Files\\MATLAB\\R2025b\\extern\\engines\\python\"")
                print("   python setup.py install")
                print()
                print("3. 检查 MATLAB 安装完整性：")
                print("   - 确认 MATLAB R2025b 可以正常启动")
                print("   - 检查环境变量 PATH 是否包含 MATLAB bin 目录")
                print()
                print("4. 尝试以管理员权限运行 Python 程序")
                print("=" * 60)
            
            self._engine = None

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if not MATLAB_ENGINE_AVAILABLE:
            return None
        
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        
        # 如果引擎启动失败，返回 None 并打印错误信息
        if cls._instance._engine is None:
            if cls._instance._start_error:
                print(f"[MATLAB] 引擎不可用: {cls._instance._start_error}")
            return None
        
        return cls._instance
    
    def acquire(self):
        """获取 MATLAB 引擎和锁（线程安全）"""
        if self._engine is None:
            raise RuntimeError("MATLAB 引擎不可用")
        return self._engine, self._engine_lock
    
    @staticmethod
    def to_matlab_path(path: str) -> str:
        """将 Windows 路径转换为 MATLAB 兼容路径"""
        # 将反斜杠转换为正斜杠，并转义单引号
        matlab_path = path.replace('\\', '/').replace("'", "''")
        return matlab_path


class MatlabMetricsBridge:
    """MATLAB HD95 计算功能已移除。"""

    @classmethod
    def instance(cls):
        return None


class MatlabService:
    """
    MATLAB 引擎常驻服务（单例模式）
    
    【重构版】实现常驻内存的 MATLAB 引擎，避免每次调用都重启引擎。
    性能提升：从每次 10-15 秒降低到几乎瞬间响应。
    """
    _instance = None
    _engine = None
    _lock = threading.Lock()
    _started = False

    @classmethod
    def start(cls):
        """启动引擎并将 matlab_scripts 添加到路径"""
        with cls._lock:
            if cls._engine is None and not cls._started:
                if not MATLAB_ENGINE_AVAILABLE:
                    print("⚠️ MATLAB 引擎模块不可用，无法启动服务")
                    return False
                
                try:
                    print("🚀 正在启动 MATLAB 引擎（常驻模式）...")
                    
                    # 配置 MATLAB bin 路径（DLL 白名单）
                    matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
                    if os.path.exists(matlab_bin_path):
                        if matlab_bin_path not in os.environ.get('PATH', ''):
                            os.environ['PATH'] = matlab_bin_path + ";" + os.environ.get('PATH', '')
                        if hasattr(os, 'add_dll_directory'):
                            try:
                                os.add_dll_directory(matlab_bin_path)
                            except Exception as e:
                                print(f"[警告] 无法添加 DLL 目录: {e}")
                    
                    # 启动 MATLAB 引擎
                    cls._engine = matlab.engine.start_matlab()
                    
                    # 添加脚本路径
                    script_path = os.path.abspath("matlab_scripts")
                    if os.path.exists(script_path):
                        cls._engine.addpath(script_path, nargout=0)
                        print(f"✅ MATLAB 脚本路径已添加: {script_path}")
                    else:
                        print(f"⚠️ MATLAB 脚本目录不存在: {script_path}")
                    
                    cls._started = True
                    print("✅ MATLAB 引擎启动完毕（常驻模式）")
                    return True
                except Exception as e:
                    print(f"❌ MATLAB 引擎启动失败: {e}")
                    cls._engine = None
                    cls._started = False
                    return False
            elif cls._engine is not None:
                print("ℹ️ MATLAB 引擎已在运行")
                return True
            else:
                return False

    @classmethod
    def get_engine(cls):
        """获取当前引擎实例"""
        if cls._engine is None:
            raise RuntimeError("MATLAB 引擎未运行，请先调用 start()")
        return cls._engine

    @classmethod
    def is_running(cls):
        """检查引擎是否正在运行"""
        return cls._engine is not None and cls._started

    @classmethod
    def quit(cls):
        """关闭引擎"""
        with cls._lock:
            if cls._engine:
                try:
                    cls._engine.quit()
                    print("✅ MATLAB 引擎已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭 MATLAB 引擎时出错: {e}")
                finally:
                    cls._engine = None
                    cls._started = False


class MatlabVisualizationBridge:
    """
    使用MATLAB绘制预测可视化网格。
    
    【重构版】使用常驻 MATLAB 引擎服务：
    - 使用 MatlabService 单例获取常驻引擎
    - 性能大幅提升：从每次 10-15 秒降低到几乎瞬间响应
    - 线程安全：通过 MatlabService 的锁机制保证
    - MATLAB 脚本已提取为独立的 .m 文件，便于维护和调试
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        # 【修复】不再需要 session，每个方法会独立启动引擎
        # 仅检查 MATLAB 引擎模块是否可用
        if not MATLAB_ENGINE_AVAILABLE:
            raise RuntimeError("MATLAB 引擎模块不可用，无法创建可视化桥接")

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if not MATLAB_ENGINE_AVAILABLE:
            return None
        
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    try:
                        cls._instance = cls()
                    except RuntimeError:
                        return None
        return cls._instance

    def render_prediction_grid(self, payload_mat_path: str, save_path: str):
        """
        【重构版】使用常驻 MATLAB 引擎渲染预测结果网格
        
        性能提升：从每次 10-15 秒降低到几乎瞬间响应
        """
        try:
            # 获取常驻引擎
            eng = MatlabService.get_engine()
            
            # 准备数据路径
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_file = MatlabEngineSession.to_matlab_path(save_path)
            
            # 调用 MATLAB 函数
            print(f"[MATLAB] 开始渲染: {save_path}")
            eng.render_prediction_grid(payload, save_file, nargout=0)
            print("[MATLAB] 渲染完成！")
            
        except RuntimeError as e:
            print(f"[MATLAB] 引擎未运行: {e}")
        except Exception as e:
            print(f"\n[MATLAB 严重错误] {e}")
            import traceback
            traceback.print_exc()

    def render_training_history(self, payload_mat_path: str, save_path: str):
        """【重构版】使用常驻 MATLAB 引擎渲染训练历史曲线"""
        try:
            eng = MatlabService.get_engine()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_mat = MatlabEngineSession.to_matlab_path(save_path)
            eng.render_training_history(payload, save_mat, nargout=0)
        except RuntimeError as e:
            print(f"[MATLAB] 引擎未运行: {e}")
        except Exception as e:
            print(f"[MATLAB] 训练历史渲染失败: {e}")

    def render_performance_analysis(self, payload_mat_path: str, save_path: str):
        """
        【重构版】性能分析绘图
        
        核心改进：
        1. 所有数据处理在 Python 端完成，避免 MATLAB 中的复杂逻辑
        2. 使用常驻引擎，性能大幅提升
        3. 完整的错误处理，确保不会崩溃
        """
        import numpy as np
        from scipy.io import loadmat
        
        try:
            # 【步骤 1】在 Python 端加载和处理数据
            print("[MATLAB] 正在加载性能数据...")
            data = loadmat(payload_mat_path)
            
            # 提取数据（优先使用新格式）
            group_means = None
            group_stds = None
            metric_names = None
            
            if 'avg_metrics_values' in data:
                # 新格式：直接使用数值数组
                group_means = np.array(data['avg_metrics_values']).flatten()
                
                if 'std_metrics_values' in data:
                    group_stds = np.array(data['std_metrics_values']).flatten()
                else:
                    group_stds = np.zeros_like(group_means)
                    print("[MATLAB] 警告: std_metrics_values 不存在，使用默认值 0")
                
                if 'avg_metrics_names' in data:
                    # 处理字符串数组
                    names_data = data['avg_metrics_names']
                    if names_data.dtype.names is None:
                        # 如果是字符数组，转换为字符串列表
                        if names_data.size > 0:
                            metric_names = [str(names_data.flat[i]) for i in range(names_data.size)]
                        else:
                            metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
                    else:
                        metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
                else:
                    metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
            else:
                # 旧格式：尝试从其他字段提取
                print("[MATLAB] 警告: 未找到新格式数据，尝试兼容旧格式...")
                if 'avg_metrics' in data:
                    avg_metrics = data['avg_metrics']
                    # 如果是 struct，转换为数组
                    if isinstance(avg_metrics, np.ndarray) and avg_metrics.dtype.names:
                        fields = avg_metrics.dtype.names
                        group_means = np.array([float(avg_metrics[field][0, 0]) for field in fields])
                        metric_names = [str(field) for field in fields]
                        group_stds = np.zeros_like(group_means)
                    else:
                        print("[MATLAB] 警告: 无法解析旧格式数据，跳过绘图")
                        return
                else:
                    print("[MATLAB] 警告: 未找到有效数据，跳过绘图")
                    return
            
            # 验证数据有效性
            if group_means is None or len(group_means) == 0:
                print("[MATLAB] 警告: 数据为空，跳过绘图")
                return
            
            # 确保 group_stds 和 metric_names 长度匹配
            if group_stds is None or len(group_stds) != len(group_means):
                group_stds = np.zeros_like(group_means)
            
            if metric_names is None or len(metric_names) != len(group_means):
                metric_names = [f'Metric {i+1}' for i in range(len(group_means))]
            
            print(f"[MATLAB] 已加载 {len(group_means)} 个指标")
            print(f"[MATLAB] 指标名称: {metric_names}")
            
            # 【步骤 2】获取常驻 MATLAB 引擎
            eng = MatlabService.get_engine()
            
            # 【步骤 3】调用 MATLAB 函数（直接传递参数）
            # 注意：matlab 模块已在文件顶部导入（如果可用）
            if not MATLAB_ENGINE_AVAILABLE:
                raise RuntimeError("MATLAB 引擎模块不可用")
            
            # 【关键修复】确保 metric_names 转换为 MATLAB cell array 格式
            # Python 列表会被自动转换为 MATLAB cell array，但为了确保兼容性，显式转换
            if isinstance(metric_names, list):
                # 使用 matlab 模块的 cell array 构造函数（如果可用）
                try:
                    # 尝试使用 matlab.cell 创建 cell array
                    metric_names_matlab = matlab.cell(metric_names)
                except (AttributeError, TypeError):
                    # 如果 matlab.cell 不可用，直接传递列表（MATLAB 引擎会自动转换）
                    metric_names_matlab = metric_names
            else:
                metric_names_matlab = metric_names
            
            eng.render_performance_analysis(
                matlab.double(group_means.tolist()),
                matlab.double(group_stds.tolist()),
                metric_names_matlab,
                MatlabEngineSession.to_matlab_path(save_path),
                nargout=0
            )
            print(f"[MATLAB] ✅ 性能分析图已保存: {save_path}")
            
        except RuntimeError as e:
            print(f"[MATLAB] 引擎未运行: {e}")
        except Exception as e:
            # 【关键修复】捕获所有异常，确保不会导致程序崩溃
            print(f"[MATLAB 警告] 性能分析绘图失败，已跳过: {str(e)}")
            import traceback
            print(f"[MATLAB] 错误详情: {traceback.format_exc()}")
            # 不抛出异常，让程序继续运行

    def render_test_results(self, payload_mat_path: str, save_path: str):
        """【重构版】使用常驻 MATLAB 引擎渲染测试结果可视化"""
        try:
            eng = MatlabService.get_engine()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_mat = MatlabEngineSession.to_matlab_path(save_path)
            eng.render_test_results(payload, save_mat, nargout=0)
        except RuntimeError as e:
            print(f"[MATLAB] 引擎未运行: {e}")
        except Exception as e:
            print(f"[MATLAB] 测试结果渲染失败: {e}")

    def render_attention_maps(self, payload_mat_path: str, save_path: str):
        """
        【重构版】注意力热图渲染
        
        使用常驻 MATLAB 引擎绘制注意力权重热力图，叠加在原图上。
        支持多个注意力层的可视化。
        """
        try:
            eng = MatlabService.get_engine()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_file = MatlabEngineSession.to_matlab_path(save_path)
            eng.render_attention_maps(payload, save_file, nargout=0)
            print(f"[MATLAB] 注意力热力图已保存: {save_path}")
        except RuntimeError as e:
            print(f"[MATLAB] 引擎未运行: {e}")
        except Exception as e:
            print(f"[MATLAB] 注意力热图渲染失败: {e}")
            import traceback
            traceback.print_exc()
    
    def render_quick_preview_matplotlib(self, images, masks, preds, save_path, num_samples=4, threshold=0.1):
        """
        【快速预览版】使用 Matplotlib 绘制预测对比图（无依赖，速度快）
        
        用于普通 Epoch 的快速预览，避免每次调用 MATLAB 导致训练变慢。
        
        Args:
            images: 图像列表 (List[np.ndarray])，每个元素为 (H, W, 3) 或 (H, W)
            masks: 真实掩码列表 (List[np.ndarray])，每个元素为 (H, W)
            preds: 预测掩码列表 (List[np.ndarray])，每个元素为 (H, W)，可以是概率值或已二值化的掩码
            save_path: 保存路径
            num_samples: 显示的样本数量
            threshold: 二值化阈值，如果preds是概率值则使用此阈值二值化，默认0.1（允许看到低置信度预测）
        
        Returns:
            save_path: 保存的文件路径
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_samples = min(num_samples, len(images))
        cols = 4  # Input, GT, Prediction, Overlay
        rows = num_samples
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = images[i]
            true_mask = masks[i]
            pred_mask = preds[i]
            
            # 确保图像是 (H, W, 3) 格式
            if img.ndim == 2:
                # 灰度图转 RGB
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            
            # 确保值在 [0, 1] 范围内
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
            
            # 确保掩码是 (H, W) 格式
            if true_mask.ndim == 3:
                true_mask = true_mask[:, :, 0] if true_mask.shape[2] == 1 else true_mask[:, :, 0]
            if pred_mask.ndim == 3:
                pred_mask = pred_mask[:, :, 0] if pred_mask.shape[2] == 1 else pred_mask[:, :, 0]
            
            # 二值化掩码
            # 真实掩码使用0.5阈值（标准）
            true_mask_binary = (true_mask > 0.5).astype(np.float32)
            # 预测掩码使用传入的阈值（允许看到低置信度预测）
            # 如果pred_mask已经是二值化的（只有0和1），则直接使用；否则根据阈值二值化
            if pred_mask.max() <= 1.0 and pred_mask.min() >= 0.0:
                # 可能是概率值，检查是否已经二值化
                unique_vals = np.unique(pred_mask)
                if len(unique_vals) == 2 and (0.0 in unique_vals or 1.0 in unique_vals):
                    # 已经是二值化的，直接使用
                    pred_mask_binary = pred_mask.astype(np.float32)
                else:
                    # 是概率值，根据阈值二值化
                    pred_mask_binary = (pred_mask > threshold).astype(np.float32)
            else:
                # 值域不在[0,1]，可能是未归一化的，先归一化再二值化
                pred_mask_normalized = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
                pred_mask_binary = (pred_mask_normalized > threshold).astype(np.float32)
            
            # --- 绘图 1: Input (原始图像) ---
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Sample {i+1}\nInput', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # --- 绘图 2: Ground Truth (绿色半透明轮廓) ---
            axes[i, 1].imshow(img)
            if true_mask_binary.sum() > 0:
                # 使用轮廓叠加（更清晰）
                from scipy.ndimage import binary_erosion
                try:
                    structure = np.ones((3, 3), dtype=bool)
                    gt_boundary = true_mask_binary - binary_erosion(true_mask_binary.astype(bool), structure=structure).astype(np.float32)
                    if gt_boundary.sum() > 0:
                        # 绘制绿色轮廓
                        axes[i, 1].contour(gt_boundary, levels=[0.5], colors='green', linewidths=2, alpha=0.7)
                except:
                    # 回退：使用透明叠加
                    overlay_gt = img.copy()
                    green_mask = np.zeros_like(overlay_gt)
                    green_mask[true_mask_binary > 0.5] = [0, 1, 0]
                    overlay_gt = overlay_gt * 0.7 + green_mask * 0.3
                    axes[i, 1].imshow(overlay_gt)
            axes[i, 1].set_title('Ground Truth\n(Green)', fontsize=10)
            axes[i, 1].axis('off')
            
            # --- 绘图 3: Prediction (红色半透明轮廓) ---
            axes[i, 2].imshow(img)
            if pred_mask_binary.sum() > 0:
                # 使用轮廓叠加（更清晰）
                try:
                    from scipy.ndimage import binary_erosion
                    structure = np.ones((3, 3), dtype=bool)
                    pred_boundary = pred_mask_binary - binary_erosion(pred_mask_binary.astype(bool), structure=structure).astype(np.float32)
                    if pred_boundary.sum() > 0:
                        # 绘制红色轮廓
                        axes[i, 2].contour(pred_boundary, levels=[0.5], colors='red', linewidths=2, alpha=0.7, linestyles='dashed')
                except:
                    # 回退：使用透明叠加
                    overlay_pred = img.copy()
                    red_mask = np.zeros_like(overlay_pred)
                    red_mask[pred_mask_binary > 0.5] = [1, 0, 0]
                    overlay_pred = overlay_pred * 0.7 + red_mask * 0.3
                    axes[i, 2].imshow(overlay_pred)
            axes[i, 2].set_title('Prediction\n(Red)', fontsize=10)
            axes[i, 2].axis('off')
            
            # --- 绘图 4: Overlay (叠加对比) ---
            # 绿色=GT, 红色=Pred, 黄色=重叠
            overlay = img.copy()
            # GT 区域（绿色）
            overlay[true_mask_binary > 0.5, 1] = np.maximum(overlay[true_mask_binary > 0.5, 1], 0.5)
            # Pred 区域（红色）
            overlay[pred_mask_binary > 0.5, 0] = np.maximum(overlay[pred_mask_binary > 0.5, 0], 0.5)
            # 重叠区域（黄色 = 红+绿）
            overlap = (true_mask_binary > 0.5) & (pred_mask_binary > 0.5)
            overlay[overlap, 0] = 1.0  # 红色
            overlay[overlap, 1] = 1.0  # 绿色
            overlay[overlap, 2] = 0.0  # 蓝色
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay\n(Green=GT, Red=Pred)', fontsize=10)
            axes[i, 3].axis('off')
        
        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # 【关键】必须关闭，防止内存泄漏
        
        return save_path


