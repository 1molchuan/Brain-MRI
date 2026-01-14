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
    MATLAB_ENGINE_AVAILABLE = True
except ImportError:
    MATLAB_ENGINE_AVAILABLE = False
    MATLAB_ENGINE_ERROR = "matlab.engine 模块未安装"
    print("[提示] matlab.engine 未安装，MATLAB 可视化功能将不可用")
except Exception as e:
    MATLAB_ENGINE_AVAILABLE = False
    MATLAB_ENGINE_ERROR = str(e)
    print(f"[警告] 无法导入 MATLAB 引擎: {e}")

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


class MatlabVisualizationBridge:
    """
    使用MATLAB绘制预测可视化网格。
    
    【修复版】线程安全设计：
    - 不再使用共享的 MATLAB 引擎实例
    - 每个渲染方法都在当前线程内独立启动和关闭引擎
    - 解决 "state not recoverable" 错误
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
        【修复版 V4】针对 Python 3.8+ (含3.12) 的 DLL 白名单修复
        
        策略：
        1. 使用 os.add_dll_directory() 明确告诉 Python DLL 安全目录（Python 3.8+ 必需）
        2. 在子线程内强制注入 MATLAB bin 路径到系统 PATH（兼容旧工具）
        3. 不使用全局引擎，而是每次在当前线程内独立启动一个新引擎
        4. 解决 state not recoverable 问题的关键
        """
        import os
        import sys
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        # Python 3.8+ 必须用这个，否则 PATH 会被无视！
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        # 3. 此时再导入和启动，就能找到 DLL 了
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        print(f"[MATLAB] 正在为当前线程启动独立引擎... (预计耗时 10-15s)")
        eng = None
        
        try:
            # 3. 关键点：在当前线程（Worker Thread）内部启动独立引擎
            eng = matlab.engine.start_matlab()
            
            # 2. 准备数据路径
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_file = MatlabEngineSession.to_matlab_path(save_path)
            
            # V2.0 增强版 MATLAB 绘图脚本
            # 注意：这是 MATLAB 代码字符串，linter 可能误报变量未定义警告
            script = f"""
try
    disp('正在加载数据: {payload}');
    data = load('{payload}');
    images = data.images;
    masks = data.masks;
    preds = data.preds;
    
    % 限制样本数
    numSamples = min(size(images, 4), 4);
    
    % 设置画布：加大分辨率，设置白色背景
    fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 1400, 350 * numSamples]);
    tl = tiledlayout(fig, numSamples, 4, 'Padding','compact', 'TileSpacing','none');
    
    for idx = 1:numSamples
        % --- 数据预处理 ---
        raw_img = images(:,:,:,idx);
        
        % 1. 自动对比度增强 (解决灰蒙蒙的问题)
        if size(raw_img, 3) == 3
            gray_img = rgb2gray(raw_img);
        else
            gray_img = raw_img;
        end
        % 归一化并增强对比度
        base_img = imadjust(mat2gray(gray_img));
        % 转回 RGB 以便彩色叠加
        base_img_rgb = cat(3, base_img, base_img, base_img);
        
        gt_mask = double(masks(:,:,idx));
        pred_mask = double(preds(:,:,idx));
        
        % --- 绘图 1: 原图 ---
        nexttile(tl); 
        imshow(base_img, []); 
        title(sprintf('Sample %d Input', idx), 'FontSize', 12, 'FontWeight', 'bold');
        
        % --- 绘图 2: Ground Truth (绿色风格) ---
        nexttile(tl); 
        imshow(base_img, []); hold on;
        % 创建绿色透明蒙版
        green = cat(3, zeros(size(gt_mask)), ones(size(gt_mask)), zeros(size(gt_mask)));
        h = imshow(green); 
        set(h, 'AlphaData', gt_mask * 0.3); % 30% 透明度
        title('Ground Truth (Green)', 'FontSize', 12);
        
        % --- 绘图 3: Prediction (红色风格) ---
        nexttile(tl); 
        imshow(base_img, []); hold on;
        % 创建红色透明蒙版
        red = cat(3, ones(size(pred_mask)), zeros(size(pred_mask)), zeros(size(pred_mask)));
        h = imshow(red); 
        set(h, 'AlphaData', pred_mask * 0.3); 
        title('Prediction (Red)', 'FontSize', 12);
        
        % --- 绘图 4: 叠加对比 (医学标准) ---
        % 绿色=GT, 红色=Pred, 黄色=重叠(正确预测)
        nexttile(tl); 
        imshow(base_img, []); hold on;
        
        % 绘制 GT (绿色轮廓)
        [B_gt,L_gt] = bwboundaries(gt_mask > 0.5, 'noholes');
        for idx_k = 1:length(B_gt)
            boundary = B_gt{{idx_k}};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1.5);
        end
        
        % 绘制 Pred (红色轮廓)
        [B_pred,L_pred] = bwboundaries(pred_mask > 0.5, 'noholes');
        for idx_k = 1:length(B_pred)
            boundary = B_pred{{idx_k}};
            plot(boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 1.5);
        end
        
        % 添加图例说明
        title('Overlay (Green=GT, Red=Pred)', 'FontSize', 12);
    end
    
    disp('正在高保真导出...');
    % 使用 exportgraphics 的 ContentType='vector' 可以获得更锐利的文字
    exportgraphics(fig, '{save_file}', 'Resolution', 300, 'BackgroundColor','white');
    close(fig);
catch ME
    disp(['MATLAB Error: ', ME.message]);
    rethrow(ME);
end
"""
            
            # 4. 执行脚本
            print(f"[MATLAB] 开始渲染: {save_path}")
            eng.eval(script, nargout=0)
            print("[MATLAB] 渲染完成！")

        except Exception as e:
            print(f"\n[MATLAB 严重错误] {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 5. 务必关闭引擎，防止僵尸进程
            if eng:
                try:
                    eng.quit()
                    print("[MATLAB] 引擎已安全关闭")
                except Exception as e:
                    print(f"[MATLAB] 关闭引擎时出错: {e}")

    def render_training_history(self, payload_mat_path: str, save_path: str):
        """【修复版 V4】针对 Python 3.8+ 的 DLL 白名单修复 - 训练历史曲线渲染"""
        import os
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        eng = None
        try:
            eng = matlab.engine.start_matlab()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_mat = MatlabEngineSession.to_matlab_path(save_path)
            script = f"""
try
    data = load('{payload}');
    epochs = data.epochs;
    trainLoss = data.train_loss;
    valLoss = data.val_loss;
    valDice = data.val_dice;
    fig = figure('Visible','off');
    tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    nexttile;
    plot(epochs, trainLoss, '-ob', 'LineWidth', 2); hold on;
    plot(epochs, valLoss, '-or', 'LineWidth', 2);
    title('训练/验证损失'); xlabel('轮次'); ylabel('Loss');
    legend('训练','验证','Location','best'); grid on;
    nexttile;
    plot(epochs, valDice, '-og', 'LineWidth', 2);
    title('验证Dice'); xlabel('轮次'); ylabel('Dice'); ylim([0 1]); grid on;
    exportgraphics(fig, '{save_mat}', 'Resolution', 300);
    close(fig);
catch ME
    disp(['MATLAB Error: ', ME.message]);
    rethrow(ME);
end
"""
            eng.eval(script, nargout=0)
        except Exception as e:
            print(f"[MATLAB] 训练历史渲染失败: {e}")
        finally:
            if eng:
                try:
                    eng.quit()
                except:
                    pass

    def render_performance_analysis(self, payload_mat_path: str, save_path: str):
        """
        【完全重写版】性能分析绘图
        
        核心改进：
        1. 所有数据处理在 Python 端完成，避免 MATLAB 中的复杂逻辑
        2. 简化 MATLAB 脚本，只负责绘图
        3. 完整的错误处理，确保不会崩溃
        """
        import os
        import numpy as np
        from scipy.io import loadmat
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        # 【关键修复】在最外层添加 try-except，确保任何错误都不会导致程序崩溃
        try:
            import matlab.engine
        except ImportError:
            print("[MATLAB 警告] 未检测到 matlab.engine，跳过性能分析绘图")
            return
        
        eng = None
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
            
            # 【步骤 2】启动 MATLAB 引擎
            eng = matlab.engine.start_matlab()
            
            # 【步骤 3】将数据传递给 MATLAB（使用 matlab.double 和 matlab.engine 接口）
            eng.workspace['group_means'] = matlab.double(group_means.tolist())
            eng.workspace['group_stds'] = matlab.double(group_stds.tolist())
            eng.workspace['metric_names'] = metric_names  # MATLAB 会自动处理字符串列表
            eng.workspace['save_file'] = MatlabEngineSession.to_matlab_path(save_path)
            
            # 【步骤 4】执行优化的 MATLAB 绘图脚本（修复排版问题）
            script = """
            try
                % 确保数据是列向量
                if size(group_means, 2) > size(group_means, 1)
                    group_means = group_means';
                end
                if size(group_stds, 2) > size(group_stds, 1)
                    group_stds = group_stds';
                end
                
                x_axis = 1:length(group_means);
                
                % 【排版修复】创建高清画布：1200x800 像素
                fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
                
                % 【排版修复】手动锁定绘图区位置，给 X 轴标签预留 25% 的空间
                % [left, bottom, width, height] - 使用归一化坐标 (0-1)
                % bottom=0.25 给 X 轴标签留 25% 的高度，width=0.85 留左右边距，height=0.65 保证图表主体足够大
                ax = axes('Position', [0.10, 0.25, 0.85, 0.65]);
                
                % 绘制柱状图
                b = bar(ax, x_axis, group_means);
                b.FaceColor = [0.2, 0.6, 0.8];
                b.EdgeColor = 'none';
                b.FaceAlpha = 0.7;
                hold(ax, 'on');
                
                % 绘制误差棒
                er = errorbar(ax, x_axis, group_means, group_stds);
                er.Color = [0.2, 0.2, 0.2];
                er.LineStyle = 'none';
                er.LineWidth = 1.5;
                er.CapSize = 10;
                
                % 美化图表
                title(ax, 'Performance Metrics Analysis', 'FontSize', 16, 'FontWeight', 'bold');
                ylabel(ax, 'Metric Value', 'FontSize', 14);
                xlabel(ax, 'Metric Name', 'FontSize', 14);
                grid(ax, 'on');
                set(ax, 'GridAlpha', 0.15);
                set(ax, 'LineWidth', 1.2);
                
                % 设置 x 轴标签（优化字体大小）
                if length(metric_names) == length(x_axis)
                    set(ax, 'XTickLabel', metric_names);
                    set(ax, 'XTick', x_axis);
                    set(ax, 'FontSize', 12);  % 设置坐标轴字体大小
                    xtickangle(ax, 45);
                end
                
                % 自动调整 y 轴范围
                y_max = max(group_means + group_stds) * 1.1;
                y_min = min(group_means - group_stds) * 0.9;
                if y_min < 0
                    y_min = 0;
                end
                ylim(ax, [y_min, y_max]);
                
                % 添加数值标签（优化字体大小）
                xtips = b.XEndPoints;
                ytips = b.YEndPoints;
                labels = string(round(b.YData, 3));
                text(ax, xtips, ytips, labels, 'HorizontalAlignment','center',...
                    'VerticalAlignment','bottom', 'FontSize', 11, 'FontWeight','bold');
                
                % 【排版修复】确保图表布局正确
                set(ax, 'Box', 'on');  % 显示坐标轴边框
                
                % 保存图片（高分辨率）
                disp('正在导出性能分析图...');
                exportgraphics(fig, save_file, 'Resolution', 300);
                close(fig);
                
                disp('性能分析图已成功生成');
            catch ME
                disp(['MATLAB Plot Error: ', ME.message]);
                rethrow(ME);
            end
            """
            
            eng.eval(script, nargout=0)
            print(f"[MATLAB] ✅ 性能分析图已保存: {save_path}")
            
        except Exception as e:
            # 【关键修复】捕获所有异常，确保不会导致程序崩溃
            print(f"[MATLAB 警告] 性能分析绘图失败，已跳过: {str(e)}")
            import traceback
            print(f"[MATLAB] 错误详情: {traceback.format_exc()}")
            # 不抛出异常，让程序继续运行
        
        finally:
            # 确保 MATLAB 引擎被正确关闭
            if eng:
                try:
                    eng.quit()
                except:
                    pass

    def render_test_results(self, payload_mat_path: str, save_path: str):
        """【修复版 V4】针对 Python 3.8+ 的 DLL 白名单修复 - 测试结果可视化渲染"""
        import os
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        eng = None
        try:
            eng = matlab.engine.start_matlab()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_mat = MatlabEngineSession.to_matlab_path(save_path)
            script = f"""
try
    data = load('{payload}');
    images = data.images;
    masks = data.masks;
    preds = data.preds;
    diceVals = data.dice;
    iouVals = data.iou;
    numSamples = size(images, 4);
    fig = figure('Visible','off');
    tiledlayout(fig, numSamples, 4, 'Padding','compact','TileSpacing','compact');
    for idx = 1:numSamples
        img = images(:,:,:,idx);
        mask = masks(:,:,idx) > 0.5;
        pred = preds(:,:,idx) > 0.5;
        overlay = img;
        overlay(:,:,1) = max(overlay(:,:,1), mask);
        overlay(:,:,2) = max(overlay(:,:,2), pred);
        overlay(:,:,3) = max(overlay(:,:,3), mask & pred);
        nexttile; imshow(img, []); title(sprintf('样本 %d 原图', idx));
        nexttile; imshow(mask); title('真实Mask');
        nexttile; imshow(pred); title(sprintf('预测Mask\\nDice %.3f / IoU %.3f', diceVals(idx), iouVals(idx)));
        nexttile; imshow(overlay); title('叠加对比');
    end
    exportgraphics(fig, '{save_mat}', 'Resolution', 300);
    close(fig);
catch ME
    disp(['MATLAB Error: ', ME.message]);
    rethrow(ME);
end
"""
            eng.eval(script, nargout=0)
        except Exception as e:
            print(f"[MATLAB] 测试结果渲染失败: {e}")
        finally:
            if eng:
                try:
                    eng.quit()
                except:
                    pass

    def render_attention_maps(self, payload_mat_path: str, save_path: str):
        """
        【完整实现版】注意力热图渲染
        
        使用 MATLAB 绘制注意力权重热力图，叠加在原图上。
        支持多个注意力层的可视化。
        """
        import os
        
        # ================== 核心修复 (针对 Python 3.12) ==================
        matlab_bin_path = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        # 1. 传统的 PATH 设置 (为了兼容旧工具)
        if matlab_bin_path not in os.environ['PATH']:
            os.environ['PATH'] = matlab_bin_path + ";" + os.environ['PATH']
            print(f"[MATLAB] 已在线程内强制添加路径: {matlab_bin_path}")
        
        # 2. 【关键一步】添加 DLL 安全目录白名单
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(matlab_bin_path)
                print(f"[系统] 已添加 DLL 安全目录: {matlab_bin_path}")
            except Exception as e:
                print(f"[警告] 无法添加 DLL 目录: {e}")
        # =============================================================
        
        try:
            import matlab.engine
        except ImportError:
            print("[错误] 未检测到 matlab.engine，无法绘图")
            return

        eng = None
        try:
            eng = matlab.engine.start_matlab()
            payload = MatlabEngineSession.to_matlab_path(payload_mat_path)
            save_file = MatlabEngineSession.to_matlab_path(save_path)
            
            script = f"""
            try
                disp('正在加载注意力数据...');
                data = load('{payload}');
                images = data.images;
                masks = data.masks;
                preds = data.preds;
                
                % 检测可用的注意力层
                att_layers = {{}};
                if isfield(data, 'att1')
                    att_layers{{end+1}} = 'att1';
                end
                if isfield(data, 'att2')
                    att_layers{{end+1}} = 'att2';
                end
                if isfield(data, 'att3')
                    att_layers{{end+1}} = 'att3';
                end
                if isfield(data, 'att4')
                    att_layers{{end+1}} = 'att4';
                end
                
                if isempty(att_layers)
                    error('未找到注意力层数据 (att1, att2, att3, att4)');
                end
                
                numSamples = min(size(images, 4), 4);
                numLayers = length(att_layers);
                
                % 设置画布：每行一个样本，每列一个注意力层 + 原图/GT/Pred
                cols = 3 + numLayers;  % Input, GT, Pred, + 各注意力层
                fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 200 * cols, 300 * numSamples]);
                tl = tiledlayout(fig, numSamples, cols, 'Padding','compact', 'TileSpacing','none');
                
                for idx = 1:numSamples
                    % --- 数据预处理 ---
                    raw_img = images(:,:,:,idx);
                    
                    % 转换为灰度图并增强对比度
                    if size(raw_img, 3) == 3
                        gray_img = rgb2gray(raw_img);
                    else
                        gray_img = raw_img;
                    end
                    base_img = imadjust(mat2gray(gray_img));
                    base_img_rgb = cat(3, base_img, base_img, base_img);
                    
                    gt_mask = double(masks(:,:,idx));
                    pred_mask = double(preds(:,:,idx));
                    
                    % --- 绘图 1: 原图 ---
                    nexttile(tl);
                    imshow(base_img, []);
                    title(sprintf('Sample %d\\nInput', idx), 'FontSize', 11, 'FontWeight', 'bold');
                    
                    % --- 绘图 2: Ground Truth ---
                    nexttile(tl);
                    imshow(base_img, []); hold on;
                    green = cat(3, zeros(size(gt_mask)), ones(size(gt_mask)), zeros(size(gt_mask)));
                    h = imshow(green);
                    set(h, 'AlphaData', gt_mask * 0.3);
                    title('Ground Truth', 'FontSize', 11);
                    
                    % --- 绘图 3: Prediction ---
                    nexttile(tl);
                    imshow(base_img, []); hold on;
                    red = cat(3, ones(size(pred_mask)), zeros(size(pred_mask)), zeros(size(pred_mask)));
                    h = imshow(red);
                    set(h, 'AlphaData', pred_mask * 0.3);
                    title('Prediction', 'FontSize', 11);
                    
                    % --- 绘图 4-N: 各注意力层热力图 ---
                    for layer_idx = 1:numLayers
                        layer_name = att_layers{{layer_idx}};
                        att_map = data.(layer_name);
                        
                        % 获取当前样本的注意力图
                        if size(att_map, 3) >= idx
                            att_2d = double(att_map(:,:,idx));
                        else
                            att_2d = double(att_map(:,:,1));  % 回退到第一个
                        end
                        
                        % 使用 imresize 将热力图放大到原图尺寸
                        [H_orig, W_orig] = size(base_img);
                        [H_att, W_att] = size(att_2d);
                        if H_att ~= H_orig || W_att ~= W_orig
                            att_2d = imresize(att_2d, [H_orig, W_orig], 'bilinear');
                        end
                        
                        % 归一化到 [0, 1]
                        att_norm = mat2gray(att_2d);
                        
                        % 使用 jet 配色方案
                        att_colored = ind2rgb(uint8(att_norm * 255), jet(256));
                        
                        % 叠加在原图上 (Alpha=0.5)
                        nexttile(tl);
                        imshow(base_img, []); hold on;
                        h_heat = imshow(att_colored);
                        set(h_heat, 'AlphaData', att_norm * 0.5);  % 50% 透明度
                        title(sprintf('Attention %s', layer_name), 'FontSize', 11);
                    end
                end
                
                disp('正在导出注意力热力图...');
                exportgraphics(fig, '{save_file}', 'Resolution', 300, 'BackgroundColor','white');
                close(fig);
                
            catch ME
                disp(['MATLAB Error: ', ME.message]);
                rethrow(ME);
            end
            """
            
            eng.eval(script, nargout=0)
            print(f"[MATLAB] 注意力热力图已保存: {save_path}")
        except Exception as e:
            print(f"[MATLAB] 注意力热图渲染失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if eng:
                try:
                    eng.quit()
                except:
                    pass
    
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


