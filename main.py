import os
import sys

# ============================================================================
# 【核弹级修复】DLL 冲突终极解决方案：预热启动 (Warm-up Launch)
# ============================================================================
# 问题：torch、PyQt5 等库在 MATLAB 之前加载时，它们的 DLL 会抢占进程空间，
#       导致 MATLAB 引擎无法加载自己的 DLL，报错 "找不到指定的程序"。
#
# 解决方案：必须在导入 torch 或 PyQt 之前执行，否则 Anaconda 的 DLL 会抢占位置。
#           仅 import 是不够的，必须真实启动一次引擎才能强制加载底层 DLL。
#           通过启动 -> 锁定 DLL -> 立即关闭的方式，将 MATLAB 核心 DLL 锁定在进程内存中。
# ============================================================================

# 标记 MATLAB 引擎是否可用（全局变量，但初始化在 if __name__ == '__main__' 中）
MATLAB_ENGINE_AVAILABLE = False

def _warmup_matlab_engine():
    """预热 MATLAB 引擎，防止 DLL 冲突"""
    global MATLAB_ENGINE_AVAILABLE
    try:
        # 1. 配置路径
        # 请根据实际安装路径修改，当前环境为 R2025b
        MATLAB_BIN_PATH = r"C:\Program Files\MATLAB\R2025b\bin\win64"
        
        if os.path.exists(MATLAB_BIN_PATH):
            # 传统 PATH 设置
            if MATLAB_BIN_PATH not in os.environ.get('PATH', ''):
                os.environ['PATH'] = MATLAB_BIN_PATH + ";" + os.environ.get('PATH', '')
            
            # Python 3.8+ 安全目录白名单
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(MATLAB_BIN_PATH)
                except: 
                    pass

        # 2. 【关键步骤】真实启动一次引擎
        # 仅 import 是不够的，必须 start_matlab 才能强制加载底层 DLL
        print("[System] 正在预热 MATLAB 引擎 (防止 DLL 冲突，约需 5-10秒)...")
        import matlab.engine
        
        # 启动临时会话 -> 锁定 DLL -> 立即关闭
        temp_eng = matlab.engine.start_matlab()
        temp_eng.quit()
        
        MATLAB_ENGINE_AVAILABLE = True
        print("✅ MATLAB 引擎预热成功！核心 DLL 已锁定。")
        
    except ImportError:
        MATLAB_ENGINE_AVAILABLE = False
        print("⚠️ 未检测到 matlab.engine，跳过预热。")
    except Exception as e:
        MATLAB_ENGINE_AVAILABLE = False
        print(f"❌ MATLAB 预热失败: {e}")
        print("提示：如果后续报错 '找不到指定的程序'，请检查 Visual C++ 运行库或路径配置。")

# ============================================================================
# 其他库必须在此之后导入
# ============================================================================

# ============================================================================
# NumExpr 线程数配置（避免警告信息）
# ============================================================================
# 设置 NumExpr 最大线程数，避免 "NUMEXPR_MAX_THREADS not set" 警告
# NumExpr 是 pandas/numpy 使用的表达式求值库
if 'NUMEXPR_MAX_THREADS' not in os.environ:
    import multiprocessing
    # 设置为 CPU 核心数，但不超过 24（避免过度占用）
    max_threads = min(multiprocessing.cpu_count(), 24)
    os.environ['NUMEXPR_MAX_THREADS'] = str(max_threads)

import argparse
import random
import shutil   # 文件操作可能用到
import logging  # 日志可能用到
import html     # HTML转义
import tempfile # 临时文件
import importlib # 动态导入模块
import json     # JSON处理
import hashlib  # 哈希计算
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset  # Dataset基类
import cv2  # OpenCV图像处理

# === GUI 相关的 PyQt5 ===
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSpinBox, QFileDialog, QComboBox, QProgressBar, 
    QGroupBox, QTabWidget, QScrollArea, QMessageBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QSplitter, QTextEdit, QDialog,
    QCheckBox, QLineEdit, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QTextBrowser, QSizePolicy, QInputDialog, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, QMutex, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QColor, QTextCursor

# === 画图相关的 Matplotlib (你刚报错缺的就是这个) ===
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# === 你的自定义模块 ===
from utils import *
from models import *
from worker import TrainThread, ModelTestThread, PredictThread

# === 可选功能占位符类（如果未实现，将使用占位符） ===
try:
    # 尝试导入API相关类（如果存在）
    from api_service import SegmentationAPIService, APIServerThread, create_segmentation_api
except ImportError:
    # 如果不存在，创建占位符类
    class SegmentationAPIService:
        """API服务占位符类"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SegmentationAPIService 需要实现。请创建 api_service.py 文件。")
    
    class APIServerThread(QThread):
        """API服务器线程占位符类"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise NotImplementedError("APIServerThread 需要实现。请创建 api_service.py 文件。")
    
    def create_segmentation_api(service):
        """创建API应用的占位符函数"""
        raise NotImplementedError("create_segmentation_api 需要实现。请创建 api_service.py 文件。")

try:
    # 尝试导入AI助手相关类（如果存在）
    from ai_assistant import AIAssistantThread
except ImportError:
    # 如果不存在，创建占位符类
    class AIAssistantThread(QThread):
        """AI助手线程占位符类"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise NotImplementedError("AIAssistantThread 需要实现。请创建 ai_assistant.py 文件。")

# 设置随机种子
random.seed(42)
np.random.seed(42)  # numpy 已在上面导入
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
class MedicalSegmentationApp(QMainWindow):
    visualization_requested = pyqtSignal(str, list, list)
    visualization_ready = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # 基础窗口设置
        self.setWindowTitle("🤖 医学图像分割系统 - AI智能分析平台")
        try:
            self.setWindowIcon(QIcon("icon.png"))  # 请确保图标文件存在
        except:
            pass  # 如果图标不存在，忽略错误
        self.setGeometry(100, 100, 1400, 1000)
        self.setMinimumSize(1200, 800)
        self.visualization_requested.connect(self.handle_visualization)
        # 线程锁
        self.lock = QMutex()
        
        # 主题：light / dark
        self.theme = "light"

        # 初始化变量
        self._init_variables()
        
        # 初始化UI
        self.initUI()
    
    def _init_variables(self):
        """初始化所有变量"""
        self.model_path = None
        self.resnet_model_path = None
        self.data_dir = None
        self.output_dir = None
        
        # 【Bug修复】图表防抖去重：记录上一次绘制的轮次，防止重复绘制
        self.last_plotted_epoch = -1
        self.epoch_history = []  # 存储真实的epoch值，用于X轴

        self.train_thread = None
        self.predict_thread = None
        self.test_thread = None
        self.test_model_path = None
        self.test_data_dir = None
        self.test_results = None
        self.low_dice_cases = []
        self.current_results = []
        self.api_thread = None
        self.api_model_path = None
        self.api_service = None
        self.ai_thread = None
        self.llm_threshold_thread = None
        self.prediction_stats = None
        self.system_status_labels = {}
        self.tab_indexes = {}
        # 默认使用旧API地址
        self.ai_base_url = "https://models.sjtu.edu.cn/api/v1/chat/completions"
        # 可选的API地址列表
        self.ai_base_url_options = [
            ("SJTU模型服务", "https://models.sjtu.edu.cn/api/v1/chat/completions"),
            ("ChatAnywhere", "https://api.chatanywhere.tech/v1/chat/completions")
        ]
        self.ai_model_name = "deepseek-r1"
        # 不同API服务支持的模型列表
        self.ai_model_options_by_service = {
            "https://models.sjtu.edu.cn/api/v1/chat/completions": [
                ("DeepSeek-R1", "deepseek-r1"),
                ("DeepSeek-V3", "deepseek-v3"),
                ("Qwen3-Coder", "qwen3coder"),
                ("Qwen3-VL", "qwen3vl")
            ],
            "https://api.chatanywhere.tech/v1/chat/completions": [
                ("DeepSeek-R1", "deepseek-r1"),
                ("DeepSeek-V3", "deepseek-v3"),
                ("GPT-3.5 Turbo", "gpt-3.5-turbo"),
                ("GPT-4o Mini", "gpt-4o-mini"),
                ("GPT-4o", "gpt-4o"),
                ("GPT-4.1 Mini", "gpt-4.1-mini"),
                ("GPT-4.1 Nano", "gpt-4.1-nano"),
                ("GPT-4.1", "gpt-4.1"),
                ("GPT-5 Mini", "gpt-5-mini"),
                ("GPT-5 Nano", "gpt-5-nano"),
                ("GPT-5", "gpt-5")
            ]
        }
        # 默认模型选项（SJTU服务）
        self.ai_model_options = self.ai_model_options_by_service[self.ai_base_url]
        # 不同API服务对应的默认API key
        self.ai_api_key_by_service = {
            "https://models.sjtu.edu.cn/api/v1/chat/completions": "",
            "https://api.chatanywhere.tech/v1/chat/completions": ""
        }
        # 默认API key（当前服务的）
        self.ai_api_key = self.ai_api_key_by_service.get(self.ai_base_url, "")
        # 标记用户是否手动修改过API key
        self.ai_key_manually_changed = False
        self.ai_limits = {
            "rpm": 100,
            "tpm": 3000,
            "weekly": 1_000_000
        }
    
    def initUI(self):
        """主UI初始化方法"""
        # 应用全局样式表
        self.apply_global_styles()
        
        # 中央控件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ===== 左侧控制面板 =====
        self.setup_control_panel()
        
        # ===== 右侧标签页 =====
        self.setup_tab_widget()
        
        # 状态栏
        self.statusBar().showMessage("✅ 就绪")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border-top: 2px solid #e2e8f0;
                padding: 8px;
                font-size: 10pt;
                color: #475569;
            }
        """)
    
    def apply_global_styles(self):
        """应用全局样式表"""
        style = """
        /* 全局样式 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #f8fafc, stop:0.5 #f1f5f9, stop:1 #e2e8f0);
        }
        
        /* GroupBox样式 */
        QGroupBox {
            font-weight: bold;
            font-size: 12pt;
            border: 2px solid #e2e8f0;
            border-radius: 14px;
            margin-top: 12px;
            padding-top: 18px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ffffff, stop:1 #f8fafc);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 10px;
            color: #1e293b;
            font-size: 13pt;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #f8fafc);
            border-radius: 6px;
        }
        
        /* 按钮样式 */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #3b82f6, stop:1 #2563eb);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 11pt;
            font-weight: 600;
            min-height: 40px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #2563eb, stop:1 #1d4ed8);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #1d4ed8, stop:1 #1e40af);
            padding: 11px 23px;
        }
        
        QPushButton:disabled {
            background: #cbd5e1;
            color: #94a3b8;
        }
        
        /* 停止按钮特殊样式 */
        QPushButton[text="⏹ 停止训练"], QPushButton[text="停止训练"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ef4444, stop:1 #dc2626);
        }
        
        QPushButton[text="⏹ 停止训练"]:hover, QPushButton[text="停止训练"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #dc2626, stop:1 #b91c1c);
        }
        
        QPushButton[text="⏹ 停止训练"]:pressed, QPushButton[text="停止训练"]:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #b91c1c, stop:1 #991b1b);
        }
        
        /* 标签样式 */
        QLabel {
            color: #1e293b;
            font-size: 11pt;
        }
        
        /* 进度条样式 */
        QProgressBar {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            background: #f8fafc;
            height: 28px;
            font-size: 11pt;
            color: #1e293b;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3b82f6, stop:0.5 #06b6d4, stop:1 #10b981);
            border-radius: 10px;
        }
        
        /* SpinBox样式 */
        QSpinBox {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11pt;
            background-color: #ffffff;
            min-width: 100px;
        }
        
        QSpinBox:focus {
            border-color: #3b82f6;
            background-color: #f8fafc;
        }
        
        QSpinBox::up-button, QSpinBox::down-button {
            background: #f1f5f9;
            border: none;
            border-radius: 4px;
            width: 20px;
        }
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background: #e2e8f0;
        }
        
        /* ComboBox样式 */
        QComboBox {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11pt;
            background-color: #ffffff;
            min-width: 150px;
        }
        
        QComboBox:focus {
            border-color: #3b82f6;
            background-color: #f8fafc;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #64748b;
            width: 0;
            height: 0;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            selection-background-color: #3b82f6;
            selection-color: white;
            padding: 4px;
        }
        
        /* CheckBox样式 */
        QCheckBox {
            font-size: 11pt;
            spacing: 10px;
            color: #475569;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #cbd5e1;
            border-radius: 4px;
            background-color: #ffffff;
        }
        
        QCheckBox::indicator:hover {
            border-color: #3b82f6;
        }
        
        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #3b82f6, stop:1 #2563eb);
            border-color: #2563eb;
        }
        
        /* TabWidget样式 */
        QTabWidget::pane {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            background-color: #ffffff;
            top: -1px;
            padding: 4px;
        }
        
        QTabBar {
            alignment: left;
        }
        
        QTabBar::tab {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f1f5f9, stop:1 #e2e8f0);
            color: #64748b;
            border: 2px solid transparent;
            border-bottom: none;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding: 12px 24px;
            margin: 4px 2px;
            font-size: 11pt;
            font-weight: 500;
            min-width: 100px;
            min-height: 35px;
        }
        
        QTabBar::tab:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e2e8f0, stop:1 #cbd5e1);
            color: #475569;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ffffff, stop:1 #f8fafc);
            color: #2563eb;
            border-color: #3b82f6;
            border-bottom-color: #ffffff;
            font-weight: 600;
        }
        
        QTabBar::tab:first {
            margin-left: 0px;
        }
        
        QTabBar::tab:last {
            margin-right: 0px;
        }
        /* ScrollArea样式 */
        QScrollArea {
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background-color: #ffffff;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #f8fafc;
            width: 14px;
            border-radius: 7px;
            border: 1px solid #e2e8f0;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #cbd5e1, stop:1 #94a3b8);
            border-radius: 6px;
            min-height: 40px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #94a3b8, stop:1 #64748b);
        }
        
        QScrollBar::handle:vertical:pressed {
            background: #475569;
        }
        
        QScrollBar:horizontal {
            border: none;
            background: #f8fafc;
            height: 14px;
            border-radius: 7px;
            border: 1px solid #e2e8f0;
        }
        
        QScrollBar::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #cbd5e1, stop:1 #94a3b8);
            border-radius: 6px;
            min-width: 40px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #94a3b8, stop:1 #64748b);
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            height: 0px;
            width: 0px;
        }
        """
        # 如果是暗色主题，叠加一层简单的暗色样式覆盖基础配色
        if getattr(self, "theme", "light") == "dark":
            dark_style = """
            QMainWindow {
                background: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #e5e7eb;
            }
            QGroupBox {
                border: 1px solid #1f2937;
                background-color: #020617;
            }
            QGroupBox::title {
                color: #e5e7eb;
                background-color: #020617;
            }
            QLabel {
                color: #e5e7eb;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1d4ed8);
                color: #e5e7eb;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #1f2937;
                background: #020617;
            }
            QTabBar::tab {
                background: #020617;
                color: #9ca3af;
                padding: 8px 18px;
            }
            QTabBar::tab:selected {
                background: #111827;
                color: #f9fafb;
                border-bottom: 2px solid #3b82f6;
            }
            QScrollArea {
                background: #020617;
            }
            QStatusBar {
                background: #020617;
                color: #9ca3af;
            }
            """
            style = style + dark_style

        self.setStyleSheet(style)

    def toggle_theme(self):
        """在浅色 / 深色主题之间切换"""
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_global_styles()
        if hasattr(self, "theme_toggle_btn"):
            self.theme_toggle_btn.setText("🌙 深色" if self.theme == "dark" else "☀ 浅色")
        self.statusBar().showMessage("🌙 已切换到深色主题" if self.theme == "dark" else "☀ 已切换到浅色主题")

    def on_theme_toggle_clicked(self):
        """主题切换按钮回调"""
        self.toggle_theme()
    
    def setup_control_panel(self):
        """左侧控制面板设置"""
        control_panel = QGroupBox("⚙️ 控制面板")
        control_panel.setFixedWidth(340)
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(15, 20, 15, 15)
        
        # 顶部主题切换
        theme_layout = QHBoxLayout()
        theme_label = QLabel("🎨 主题:")
        theme_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.theme_toggle_btn = QPushButton("☀ 浅色")
        self.theme_toggle_btn.setFixedHeight(32)
        self.theme_toggle_btn.setToolTip("在浅色 / 深色主题之间切换")
        self.theme_toggle_btn.clicked.connect(self.on_theme_toggle_clicked)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_toggle_btn)
        theme_layout.addStretch()
        control_layout.addLayout(theme_layout)

        # 添加模型保存选项
        self.save_best_checkbox = QCheckBox("💾 自动保存最佳模型")
        self.save_best_checkbox.setChecked(True)
        self.save_best_checkbox.setToolTip("训练过程中自动保存表现最好的模型\n模型将保存在输出目录中")
        control_layout.addWidget(self.save_best_checkbox)

        # 添加MATLAB可视化开关
        self.use_matlab_checkbox = QCheckBox("📊 使用MATLAB可视化")
        # 默认根据MATLAB_ENGINE_AVAILABLE设置初始状态
        self.use_matlab_checkbox.setChecked(MATLAB_ENGINE_AVAILABLE)
        self.use_matlab_checkbox.setEnabled(MATLAB_ENGINE_AVAILABLE)
        if not MATLAB_ENGINE_AVAILABLE:
            self.use_matlab_checkbox.setToolTip("MATLAB引擎不可用（已禁用）\n系统将仅使用Matplotlib绘图")
            self.use_matlab_checkbox.setStyleSheet("color: #94a3b8;")  # 灰色显示，表示禁用
        else:
            self.use_matlab_checkbox.setToolTip("启用MATLAB高清绘图功能\n取消勾选将仅使用Matplotlib绘图\n建议：MATLAB绘图质量更高但速度较慢")
        control_layout.addWidget(self.use_matlab_checkbox)
        
        # 添加关闭MATLAB引擎按钮
        # QPushButton已在文件顶部导入，无需重复导入
        self.close_matlab_btn = QPushButton("🔴 关闭MATLAB引擎")
        self.close_matlab_btn.setEnabled(MATLAB_ENGINE_AVAILABLE)
        self.close_matlab_btn.setToolTip("完全关闭MATLAB引擎进程\n关闭后需要重启程序才能重新使用MATLAB功能")
        if not MATLAB_ENGINE_AVAILABLE:
            self.close_matlab_btn.setStyleSheet("color: #94a3b8; background-color: #f1f5f9;")
        else:
            self.close_matlab_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
                QPushButton:pressed {
                    background-color: #b91c1c;
                }
            """)
        self.close_matlab_btn.clicked.connect(self._close_matlab_engine)
        control_layout.addWidget(self.close_matlab_btn)

        self.create_system_status_group(control_layout)
        self.create_quick_nav_group(control_layout)

        # 初始化隐藏的API控件（不添加到界面，保持功能兼容）
        self._init_hidden_api_controls(control_panel)
        
        # 其他控制组件...
        control_layout.addStretch()
        control_panel.setLayout(control_layout)

        self.main_layout.addWidget(control_panel)

    def _close_matlab_engine(self):
        """
        关闭MATLAB引擎
        禁用MATLAB相关功能，强制使用Matplotlib
        """
        global MATLAB_ENGINE_AVAILABLE
        
        reply = QMessageBox.question(
            self, 
            "关闭MATLAB引擎",
            "确定要关闭MATLAB引擎吗？\n\n"
            "关闭后：\n"
            "• 将禁用所有MATLAB可视化功能\n"
            "• 系统将仅使用Matplotlib绘图\n"
            "• 需要重启程序才能重新启用MATLAB\n\n"
            "是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 尝试关闭所有MATLAB引擎会话
                # 注意：由于MATLAB引擎是全局的，这里主要是禁用功能
                MATLAB_ENGINE_AVAILABLE = False
                
                # 更新UI状态
                self.use_matlab_checkbox.setChecked(False)
                self.use_matlab_checkbox.setEnabled(False)
                self.use_matlab_checkbox.setToolTip("MATLAB引擎已关闭（已禁用）\n系统将仅使用Matplotlib绘图\n需要重启程序才能重新启用")
                self.use_matlab_checkbox.setStyleSheet("color: #94a3b8;")
                
                self.close_matlab_btn.setEnabled(False)
                self.close_matlab_btn.setStyleSheet("color: #94a3b8; background-color: #f1f5f9;")
                self.close_matlab_btn.setText("🔴 MATLAB引擎已关闭")
                self.close_matlab_btn.setToolTip("MATLAB引擎已关闭\n需要重启程序才能重新启用")
                
                QMessageBox.information(
                    self,
                    "MATLAB引擎已关闭",
                    "MATLAB引擎功能已禁用。\n\n"
                    "系统将仅使用Matplotlib进行可视化。\n"
                    "如需重新启用MATLAB，请重启程序。"
                )
                
                print("[MATLAB] 用户已手动关闭MATLAB引擎功能")
                
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "关闭失败",
                    f"关闭MATLAB引擎时发生错误：\n{str(e)}\n\n"
                    "MATLAB功能可能仍在使用中。"
                )
                print(f"[MATLAB] 关闭引擎时出错: {e}")

    def _init_hidden_api_controls(self, parent):
        """创建但不显示API服务控件，保留相关功能兼容"""
        self.api_control_container = QGroupBox("🌐 API服务", parent)
        api_layout = QVBoxLayout(self.api_control_container)
        api_layout.setSpacing(10)

        self.api_model_label = QLabel("✗ 未选择API模型", self.api_control_container)
        self.api_model_label.setWordWrap(True)
        self.api_model_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        browse_api_model_btn = QPushButton("📁 选择API模型", self.api_control_container)
        browse_api_model_btn.clicked.connect(self.browse_api_model)
        browse_api_model_btn.setToolTip("选择用于API推理的已训练模型(.pth/.pt)")

        host_layout = QHBoxLayout()
        host_label = QLabel("地址:", self.api_control_container)
        host_label.setMinimumWidth(60)
        self.api_host_input = QLineEdit("0.0.0.0", self.api_control_container)
        self.api_host_input.setPlaceholderText("0.0.0.0")
        host_layout.addWidget(host_label)
        host_layout.addWidget(self.api_host_input)

        port_layout = QHBoxLayout()
        port_label = QLabel("端口:", self.api_control_container)
        port_label.setMinimumWidth(60)
        self.api_port_spin = QSpinBox(self.api_control_container)
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(8000)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.api_port_spin)

        device_layout = QHBoxLayout()
        device_label = QLabel("设备:", self.api_control_container)
        device_label.setMinimumWidth(60)
        self.api_device_combo = QComboBox(self.api_control_container)
        self.api_device_combo.addItem("自动选择", None)
        self.api_device_combo.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.api_device_combo.addItem("CUDA:0", "cuda:0")
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.api_device_combo)

        api_button_layout = QHBoxLayout()
        api_button_layout.setSpacing(12)
        self.api_start_btn = QPushButton("▶️ 启动API", self.api_control_container)
        self.api_start_btn.clicked.connect(self.start_api_server)
        self.api_stop_btn = QPushButton("⏹ 关闭API", self.api_control_container)
        self.api_stop_btn.clicked.connect(self.stop_api_server)
        self.api_stop_btn.setEnabled(False)
        api_button_layout.addWidget(self.api_start_btn)
        api_button_layout.addWidget(self.api_stop_btn)

        self.api_status_label = QLabel("⚠️ API未运行", self.api_control_container)
        self.api_status_label.setWordWrap(True)
        self.api_status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fee2e2, stop:1 #fecaca);
                border-left: 4px solid #dc2626;
                border-radius: 8px;
                color: #991b1b;
                font-size: 10pt;
            }
        """)

        api_layout.addWidget(self.api_model_label)
        api_layout.addWidget(browse_api_model_btn)
        api_layout.addLayout(host_layout)
        api_layout.addLayout(port_layout)
        api_layout.addLayout(device_layout)
        api_layout.addLayout(api_button_layout)
        api_layout.addWidget(self.api_status_label)
        self.api_control_container.hide()

    def create_system_status_group(self, layout):
        """创建系统状态卡片"""
        status_group = QGroupBox("🛰 系统状态")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(8)
        group_layout.setContentsMargins(12, 18, 12, 12)

        self.system_status_labels = {}
        status_items = {
            "data": "训练数据",
            "train_model": "训练模型",
            "predict_model": "预测模型",
            "output_dir": "输出目录"
        }

        for key, title in status_items.items():
            label = QLabel(f"{title}: 未选择")
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setMinimumHeight(32)
            self.system_status_labels[key] = {"label": label, "title": title}
            group_layout.addWidget(label)
            self.update_system_status(key, "未选择", status="warning")

        status_group.setLayout(group_layout)
        layout.addWidget(status_group)

    def create_quick_nav_group(self, layout):
        """创建快速导航按钮"""
        nav_group = QGroupBox("⚡ 快速导航")
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(8)
        nav_layout.setContentsMargins(12, 18, 12, 12)

        buttons = [
            ("前往训练", "train"),
            ("前往预测", "predict"),
            ("查看结果", "result"),
            ("性能分析", "analysis"),
            ("AI助手", "assistant")
        ]

        for text, key in buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda _, k=key: self.switch_to_tab(k))
            nav_layout.addWidget(btn)

        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

    def update_system_status(self, key, value, status="info"):
        """更新系统状态显示"""
        info = self.system_status_labels.get(key)
        if not info:
            return
        label = info["label"]
        title = info["title"]
        styles = {
            "info": """
                QLabel {
                    padding: 10px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f8fafc, stop:1 #eef2ff);
                    border-radius: 8px;
                    border-left: 4px solid #6366f1;
                    color: #312e81;
                    font-size: 10pt;
                }
            """,
            "success": """
                QLabel {
                    padding: 10px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border-radius: 8px;
                    border-left: 4px solid #16a34a;
                    color: #166534;
                    font-size: 10pt;
                }
            """,
            "warning": """
                QLabel {
                    padding: 10px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #fef3c7, stop:1 #fde68a);
                    border-radius: 8px;
                    border-left: 4px solid #f59e0b;
                    color: #92400e;
                    font-size: 10pt;
                }
            """
        }
        label.setStyleSheet(styles.get(status, styles["info"]))
        label.setText(f"{title}: {value}")

    def switch_to_tab(self, tab_key):
        """快速切换到指定标签页"""
        index = self.tab_indexes.get(tab_key)
        if index is not None:
            self.tab_widget.setCurrentIndex(index)
    
    def setup_tab_widget(self):
        """右侧标签页设置"""
        self.tab_widget = QTabWidget()
        
        # 训练标签页
        self.setup_train_tab()
        
        # 预测标签页
        self.setup_predict_tab()
        
        # 结果标签页
        self.setup_result_tab()
        
        # 性能分析标签页
        self.setup_analysis_tab()

        # 模型测试标签页
        self.setup_model_test_tab()

        # AI助手标签页
        self.setup_ai_assistant_tab()
        
        self.main_layout.addWidget(self.tab_widget)
    
    def setup_train_tab(self):
        """训练标签页设置"""
        train_tab = QWidget()
        
        # 使用滚动区域包装内容
        train_scroll = QScrollArea()
        train_scroll.setWidgetResizable(True)
        train_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        train_scroll.setFrameShape(QScrollArea.NoFrame)
        
        train_content = QWidget()
        train_layout = QVBoxLayout()
        train_layout.setSpacing(15)
        train_layout.setContentsMargins(15, 15, 15, 15)
        
        # 数据目录选择
        data_dir_group = QGroupBox("📚 训练数据")
        data_dir_layout = QVBoxLayout()
        data_dir_layout.setSpacing(12)
        data_dir_layout.setContentsMargins(15, 20, 15, 15)
        
        self.data_dir_label = QLabel("✗ 未选择数据目录")
        self.data_dir_label.setWordWrap(True)
        self.data_dir_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        browse_data_btn = QPushButton("📁 选择数据目录")
        browse_data_btn.setToolTip("选择包含训练图像和掩码的数据目录")
        browse_data_btn.clicked.connect(self.browse_data_dir)
        
        data_dir_layout.addWidget(self.data_dir_label)
        data_dir_layout.addWidget(browse_data_btn)
        data_dir_group.setLayout(data_dir_layout)
        
        # 训练参数
        params_group = QGroupBox("⚙️ 训练参数")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(14)
        params_layout.setContentsMargins(15, 20, 15, 15)
        
        # 训练轮次
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("🔄 训练轮次:")
        epochs_label.setMinimumWidth(120)
        epochs_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setSuffix(" 轮")
        self.epochs_spin.setToolTip("设置训练的总轮次数\n建议值: 20-100")
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        
        # 批量大小
        batch_layout = QHBoxLayout()
        batch_label = QLabel("📦 批量大小:")
        batch_label.setMinimumWidth(120)
        batch_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        self.batch_spin.setToolTip("每次训练使用的样本数量\n根据GPU内存调整，建议: 2-8")
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        
        # 模型选择
        model_label = QLabel("🤖 预训练模型:")
        model_label.setStyleSheet("font-weight: 600; color: #475569;")
        # 单模型路径（用于非集成模式）
        self.model_path_label = QLabel("✗ 未选择模型")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        browse_model_btn = QPushButton("📁 选择模型")
        browse_model_btn.setToolTip("选择预训练模型文件（可选）\n如果为空，将从零开始训练")
        browse_model_btn.clicked.connect(self.browse_model_path)
        
        # 模型架构选择
        arch_label = QLabel("🏗️ 模型架构:")
        arch_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.arch_combo = QComboBox()
        self.arch_combo.addItem("改进UNet (ImprovedUNet)", "improved_unet")
        self.arch_combo.addItem("ResNet-UNet (ResNetUNet)", "resnet_unet")
        self.arch_combo.addItem("Transformer+UNet (TransUNet)", "trans_unet")
        self.arch_combo.addItem("DS-TransUNet (双尺度Transformer+UNet) ⭐", "ds_trans_unet")
        self.arch_combo.addItem("SwinUNet (Swin Transformer+UNet) ⭐推荐", "swin_unet")
        self.arch_combo.addItem("SMP U-Net++ (ResNet101) 🚀高级", "smp_unetplusplus")
        self.arch_combo.addItem("DeepLabV3+ (ResNet101) 🔥降维打击", "smp_deeplabv3plus")
        # 【降维打击】默认使用DeepLabV3+，更接近纯ResNet效果
        self.arch_combo.setCurrentIndex(6)  # 默认选择DeepLabV3+
        self.arch_combo.setToolTip(
            "选择模型架构类型：\n"
            "• ImprovedUNet: 基础改进UNet\n"
            "• ResNetUNet: 使用ResNet101编码器\n"
            "• TransUNet: Transformer+UNet混合架构\n"
            "• DS-TransUNet: 双尺度Transformer+UNet，在多个尺度使用Transformer增强多尺度特征提取\n"
            "• SwinUNet: Swin Transformer+UNet混合架构，可配合GWO优化提高Dice指标\n"
            "• SMP U-Net++: 使用segmentation-models-pytorch的U-Net++，支持标准数据集（1通道）和2.5D数据集（3通道）\n"
            "• DeepLabV3+: 使用segmentation-models-pytorch的DeepLabV3+，锁定3通道输入，更接近纯ResNet效果（降维打击，默认选择）"
        )
        
        # GWO优化选项（SwinUNet / DS-TransUNet / nnFormer 可用）
        self.gwo_checkbox = QCheckBox("启用GWO优化（灰狼优化算法）")
        self.gwo_checkbox.setToolTip(
            "使用GWO算法优化 SwinUNet、DS-TransUNet 或 nnFormer 的超参数以提高Dice指标\n"
            "注意：优化过程需要额外时间，但能显著提升模型性能"
        )
        self.gwo_checkbox.setEnabled(False)  # 默认禁用，只有选择支持的架构时启用
        self.arch_combo.currentIndexChanged.connect(self._on_arch_changed)
        self._on_arch_changed()
        
        # 【GUI优化】训练时自动切换到分析页选项（默认关闭）
        self.auto_switch_tab_checkbox = QCheckBox("训练时自动切换到分析页")
        self.auto_switch_tab_checkbox.setChecked(False)  # 默认关闭
        self.auto_switch_tab_checkbox.setToolTip(
            "勾选后，训练过程中会在第1个epoch或每5个epoch自动切换到性能分析标签页\n"
            "取消勾选则只更新图表数据，不自动切换标签页（避免打断用户）"
        )
        
        # 数据集类型选择（2.5D vs 普通）
        dataset_type_label = QLabel("📊 数据集类型:")
        dataset_type_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.addItem("普通数据集 (单通道输入)", "standard")
        self.dataset_type_combo.addItem("2.5D数据集 (TCGA-LGG格式) 🚀", "2.5d")
        self.dataset_type_combo.setCurrentIndex(0)  # 默认普通数据集
        self.dataset_type_combo.setToolTip(
            "选择数据集类型：\n"
            "• 普通数据集: 标准单通道图像输入，适用于大多数分割任务\n"
            "• 2.5D数据集: TCGA-LGG格式，使用相邻切片堆叠为3通道输入\n"
            "  注意：2.5D数据集需要文件名格式为 TCGA_CS_5393_19990606_1.tif\n"
            "  所有模型都支持2.5D数据集，系统会自动调整输入通道数"
        )
        self.dataset_type_combo.currentIndexChanged.connect(self._on_dataset_type_changed)
        self._on_dataset_type_changed()
        
        # 优化器选择
        optimizer_label = QLabel("⚙️ 优化器:")
        optimizer_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("Adam", "adam")
        self.optimizer_combo.addItem("AdamW", "adamw")
        self.optimizer_combo.addItem("SGD + Nesterov", "sgd")
        self.optimizer_combo.setCurrentIndex(0)
        self.optimizer_combo.setToolTip(
            "选择训练优化器：\n"
            "• Adam：标准Adam优化\n"
            "• AdamW：带解耦权重衰减的AdamW，适合较大正则需求\n"
            "• SGD + Nesterov：带Nesterov动量的SGD（momentum=0.99）"
        )
        
        # 添加到布局
        params_layout.addLayout(epochs_layout)
        params_layout.addLayout(batch_layout)
        params_layout.addWidget(model_label)
        params_layout.addWidget(self.model_path_label)
        params_layout.addWidget(browse_model_btn)
        params_layout.addWidget(arch_label)
        params_layout.addWidget(self.arch_combo)
        params_layout.addWidget(self.gwo_checkbox)
        params_layout.addWidget(self.auto_switch_tab_checkbox)
        params_layout.addWidget(dataset_type_label)
        params_layout.addWidget(self.dataset_type_combo)
        params_layout.addWidget(optimizer_label)
        params_layout.addWidget(self.optimizer_combo)
        params_group.setLayout(params_layout)
        
        # 训练按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        self.train_btn = QPushButton("🚀 开始训练")
        self.train_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setMinimumHeight(48)
        self.train_btn.setToolTip("开始训练模型\n需要先选择数据目录")
        
        self.stop_train_btn = QPushButton("⏹ 停止训练")
        self.stop_train_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setMinimumHeight(48)
        self.stop_train_btn.setToolTip("停止当前正在进行的训练")
        
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.stop_train_btn)
        
        # 训练进度
        train_progress_label = QLabel("📊 训练进度:")
        train_progress_label.setStyleSheet("font-weight: 600; color: #475569; font-size: 11pt;")
        self.train_progress = QProgressBar()
        self.train_progress.setFormat("训练: %p%")
        self.train_status = QLabel("⏳ 准备训练")
        self.train_status.setWordWrap(True)
        self.train_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.train_status.setMinimumHeight(50)
        self.train_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dbeafe, stop:1 #bfdbfe);
                border-radius: 8px;
                border-left: 4px solid #3b82f6;
                color: #1e40af;
                font-weight: 500;
                font-size: 10pt;
            }
        """)
        
        # 添加验证进度条
        val_progress_label = QLabel("✅ 验证进度:")
        val_progress_label.setStyleSheet("font-weight: 600; color: #475569; font-size: 11pt; padding-top: 8px;")
        self.val_progress = QProgressBar()
        self.val_progress.setFormat("验证: %p%")
        self.val_status = QLabel("⏳ 验证状态: 等待验证...")
        self.val_status.setWordWrap(True)
        self.val_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.val_status.setMinimumHeight(50)
        self.val_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f3e5f5, stop:1 #e1bee7);
                border-radius: 8px;
                border-left: 4px solid #9333ea;
                color: #6b21a8;
                font-weight: 500;
                font-size: 10pt;
            }
        """)
        
        # 添加训练统计信息
        self.stats_group = QGroupBox("📈 训练统计")
        self.stats_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)  # 确保GroupBox可以适应窗口大小
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(8)  # 减小间距，避免重叠
        stats_layout.setContentsMargins(12, 20, 12, 12)  # 减小左右边距
        
        self.epoch_label = QLabel("🔄 当前轮次: -")
        self.loss_label = QLabel("📉 训练损失: -")
        self.val_loss_label = QLabel("📊 验证损失: -")
        self.dice_label = QLabel("🎯 Dice系数: -")
        
        # 设置统计标签样式和属性，确保小窗口时也能正常显示
        stat_label_style = """
            QLabel {
                padding: 8px 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fff7ed, stop:1 #ffedd5);
                border-left: 4px solid #f97316;
                border-radius: 8px;
                font-weight: 600;
                color: #9a3412;
                font-size: 10pt;
                min-height: 20px;
            }
        """
        # 设置所有统计标签的属性
        for label in [self.epoch_label, self.loss_label, self.val_loss_label, self.dice_label]:
            label.setStyleSheet(stat_label_style)
            label.setWordWrap(True)  # 允许文本换行
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)  # 水平扩展，垂直最小
            label.setMinimumHeight(38)  # 设置最小高度，稍微减小
            label.setMaximumHeight(100)  # 设置最大高度，防止过度扩展
        
        stats_layout.addWidget(self.epoch_label)
        stats_layout.addWidget(self.loss_label)
        stats_layout.addWidget(self.val_loss_label)
        stats_layout.addWidget(self.dice_label)
        self.stats_group.setLayout(stats_layout)
        
        # 添加到训练布局
        train_layout.addWidget(data_dir_group)
        train_layout.addWidget(params_group)
        train_layout.addLayout(button_layout)
        train_layout.addWidget(train_progress_label)
        train_layout.addWidget(self.train_progress)
        train_layout.addWidget(self.train_status)
        train_layout.addWidget(val_progress_label)
        train_layout.addWidget(self.val_progress)  # 添加验证进度条
        train_layout.addWidget(self.val_status)    # 添加验证状态
        train_layout.addWidget(self.stats_group)   # 添加统计信息
        train_layout.addStretch()
        
        train_content.setLayout(train_layout)
        train_scroll.setWidget(train_content)
        
        # 设置训练标签页的主布局
        train_tab_layout = QVBoxLayout()
        train_tab_layout.setContentsMargins(0, 0, 0, 0)
        train_tab_layout.addWidget(train_scroll)
        train_tab.setLayout(train_tab_layout)
        
        self.tab_widget.addTab(train_tab, "🚀 训练")
        self.tab_indexes["train"] = self.tab_widget.indexOf(train_tab)
    
    def setup_predict_tab(self):
        """预测标签页设置"""
        predict_tab = QWidget()
        
        # 使用滚动区域包装内容
        predict_scroll = QScrollArea()
        predict_scroll.setWidgetResizable(True)
        predict_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        predict_scroll.setFrameShape(QScrollArea.NoFrame)
        
        predict_content = QWidget()
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(15)
        predict_layout.setContentsMargins(15, 15, 15, 15)
        
        # 输入图像选择
        input_group = QGroupBox("🖼️ 输入图像")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(12)
        input_layout.setContentsMargins(15, 20, 15, 15)

        self.input_list = QComboBox()
        self.input_list.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.input_list.setMinimumHeight(40)
        self.input_list.setToolTip("选择要预测的图像")
        
        button_layout_input = QHBoxLayout()
        button_layout_input.setSpacing(12)
        browse_input_btn = QPushButton("➕ 添加图像")
        browse_input_btn.setToolTip("添加一张或多张图像到预测列表")
        browse_input_btn.clicked.connect(self.browse_input_images)
        
        clear_input_btn = QPushButton("🗑️ 清空列表")
        clear_input_btn.setToolTip("清空所有已添加的图像")
        clear_input_btn.clicked.connect(self.clear_input_images)
        
        button_layout_input.addWidget(browse_input_btn)
        button_layout_input.addWidget(clear_input_btn)
        
        input_layout.addWidget(self.input_list)
        input_layout.addLayout(button_layout_input)
        input_group.setLayout(input_layout)
        
        # 模型选择
        pred_model_group = QGroupBox("🤖 预测模型")
        pred_model_layout = QVBoxLayout()
        pred_model_layout.setSpacing(12)
        pred_model_layout.setContentsMargins(15, 20, 15, 15)
        
        self.pred_model_label = QLabel("✗ 未选择模型")
        self.pred_model_label.setWordWrap(True)
        self.pred_model_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        
        browse_pred_model_btn = QPushButton("📁 选择模型")
        browse_pred_model_btn.setToolTip("选择训练好的模型文件用于预测")
        browse_pred_model_btn.clicked.connect(self.browse_pred_model_path)
        
        pred_model_layout.addWidget(self.pred_model_label)
        pred_model_layout.addWidget(browse_pred_model_btn)
        
        pred_model_group.setLayout(pred_model_layout)
        
        # 输出目录
        output_group = QGroupBox("📂 输出设置")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(12)
        output_layout.setContentsMargins(15, 20, 15, 15)
        
        # 【GUI优化】添加保存结果checkbox
        self.save_results_checkbox = QCheckBox("💾 保存/导出结果")
        self.save_results_checkbox.setChecked(True)  # 默认勾选
        self.save_results_checkbox.setToolTip("勾选后将保存预测结果到输出目录")
        self.save_results_checkbox.stateChanged.connect(self.on_save_results_changed)
        
        self.output_dir_label = QLabel("✗ 未选择输出目录")
        self.output_dir_label.setWordWrap(True)
        self.output_dir_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                font-size: 10pt;
            }
        """)
        
        browse_output_btn = QPushButton("📁 选择输出目录")
        browse_output_btn.setToolTip("选择保存预测结果的目录")
        browse_output_btn.clicked.connect(self.browse_output_dir)

        output_layout.addWidget(self.save_results_checkbox)
        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(browse_output_btn)
        output_group.setLayout(output_layout)

        # 阈值控制
        threshold_group = QGroupBox("🧮 阈值调控")
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(12)
        threshold_layout.setContentsMargins(15, 20, 15, 15)

        threshold_spin_layout = QHBoxLayout()
        threshold_label = QLabel("二值化阈值:")
        threshold_label.setMinimumWidth(100)
        threshold_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.50)
        self.threshold_spin.setSuffix("")
        threshold_spin_layout.addWidget(threshold_label)
        threshold_spin_layout.addWidget(self.threshold_spin)
        threshold_spin_layout.addStretch()

        self.llm_threshold_btn = QPushButton("🤖 LLM推荐阈值")
        self.llm_threshold_btn.setEnabled(False)
        self.llm_threshold_btn.setToolTip("基于最近一次预测的概率统计，请求LLM给出更优阈值建议")
        self.llm_threshold_btn.clicked.connect(self.request_llm_threshold)

        self.llm_threshold_status = QLabel("需要先完成预测以生成统计数据")
        self.llm_threshold_status.setWordWrap(True)
        self.llm_threshold_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: #f8fafc;
                border-radius: 8px;
                border-left: 4px solid #94a3b8;
                color: #475569;
                font-size: 10pt;
            }
        """)

        threshold_layout.addLayout(threshold_spin_layout)
        threshold_layout.addWidget(self.llm_threshold_btn)
        threshold_layout.addWidget(self.llm_threshold_status)
        threshold_group.setLayout(threshold_layout)
        
        # 预测按钮
        self.predict_btn = QPushButton("🚀 开始预测")
        self.predict_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setMinimumHeight(48)
        self.predict_btn.setToolTip("开始对选定的图像进行预测\n需要先选择模型和输出目录")
        
        # 预测进度
        predict_progress_label = QLabel("📊 预测进度:")
        predict_progress_label.setStyleSheet("font-weight: 600; color: #475569; font-size: 11pt;")
        self.predict_progress = QProgressBar()
        self.predict_progress.setFormat("预测: %p%")
        self.predict_status = QLabel("⏳ 准备预测")
        self.predict_status.setWordWrap(True)
        self.predict_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.predict_status.setMinimumHeight(50)
        self.predict_status.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dcfce7, stop:1 #bbf7d0);
                border-radius: 8px;
                border-left: 4px solid #16a34a;
                color: #166534;
                font-weight: 500;
                font-size: 10pt;
            }
        """)
        
        # 添加到预测布局
        predict_layout.addWidget(input_group)
        predict_layout.addWidget(pred_model_group)
        predict_layout.addWidget(output_group)
        predict_layout.addWidget(threshold_group)
        predict_layout.addWidget(self.predict_btn)
        predict_layout.addWidget(predict_progress_label)
        predict_layout.addWidget(self.predict_progress)
        predict_layout.addWidget(self.predict_status)
        predict_layout.addStretch()
        
        predict_content.setLayout(predict_layout)
        predict_scroll.setWidget(predict_content)
        
        # 设置预测标签页的主布局
        predict_tab_layout = QVBoxLayout()
        predict_tab_layout.setContentsMargins(0, 0, 0, 0)
        predict_tab_layout.addWidget(predict_scroll)
        predict_tab.setLayout(predict_tab_layout)
        
        self.tab_widget.addTab(predict_tab, "🔮 预测")
        self.tab_indexes["predict"] = self.tab_widget.indexOf(predict_tab)
    
    def setup_result_tab(self):
        """结果标签页设置"""
        result_tab = QWidget()
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(15, 15, 15, 15)
        result_layout.setSpacing(10)
        
        # 添加标题
        result_title = QLabel("📋 预测结果")
        result_title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("""
            QLabel {
                color: #1e293b;
                padding: 14px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border-radius: 10px;
                border: 2px solid #3b82f6;
                margin-bottom: 8px;
            }
        """)
        result_layout.addWidget(result_title)

        # ===== 预览区域（大图 + 翻页 + 缩略图）=====
        preview_group = QGroupBox("👀 结果预览")
        preview_layout = QVBoxLayout()

        # 大图区域：输入图像 + 分割结果
        preview_image_layout = QHBoxLayout()
        self.preview_input_label = QLabel("输入图像预览")
        self.preview_output_label = QLabel("分割结果预览")
        for lbl in (self.preview_input_label, self.preview_output_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 320)
            lbl.setStyleSheet("""
                QLabel {
                    border: 2px solid #e2e8f0;
                    border-radius: 10px;
                    background-color: #0b1120;
                    color: #64748b;
                }
            """)
        preview_image_layout.addWidget(self.preview_input_label)
        preview_image_layout.addWidget(self.preview_output_label)
        preview_layout.addLayout(preview_image_layout)

        # 翻页按钮
        nav_layout = QHBoxLayout()
        self.prev_result_btn = QPushButton("⬅ 上一张")
        self.next_result_btn = QPushButton("下一张 ➡")
        self.prev_result_btn.clicked.connect(self.show_prev_result)
        self.next_result_btn.clicked.connect(self.show_next_result)
        self.result_index_label = QLabel("0 / 0")
        self.result_index_label.setStyleSheet("font-weight: 600; color: #475569;")
        nav_layout.addWidget(self.prev_result_btn)
        nav_layout.addWidget(self.next_result_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.result_index_label)
        preview_layout.addLayout(nav_layout)

        # 缩略图条
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout()
        self.thumbnail_layout.setContentsMargins(5, 5, 5, 5)
        self.thumbnail_layout.setSpacing(8)
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        preview_layout.addWidget(self.thumbnail_scroll)

        preview_group.setLayout(preview_layout)
        result_layout.addWidget(preview_group)

        # 结果显示区域（完整列表）
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        
        self.result_container = QWidget()
        self.result_container_layout = QVBoxLayout()
        self.result_container_layout.setSpacing(20)
        self.result_container_layout.setContentsMargins(10, 10, 10, 10)
        self.result_container.setLayout(self.result_container_layout)
        
        self.result_scroll.setWidget(self.result_container)
        result_layout.addWidget(self.result_scroll)
        
        result_tab.setLayout(result_layout)
        self.tab_widget.addTab(result_tab, "📊 结果")
        self.tab_indexes["result"] = self.tab_widget.indexOf(result_tab)
    
    def setup_analysis_tab(self):
        """性能分析标签页设置"""
        analysis_tab = QWidget()
        analysis_tab_layout = QVBoxLayout()
        analysis_tab.setLayout(analysis_tab_layout)

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        analysis_scroll.setFrameShape(QScrollArea.NoFrame)

        analysis_container = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_container.setLayout(analysis_layout)
        analysis_scroll.setWidget(analysis_container)

        analysis_tab_layout.addWidget(analysis_scroll)
        
        # 标题
        title_label = QLabel("📊 模型性能分析与测试集分割结果")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #1e293b;
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border-radius: 12px;
                border: 2px solid #3b82f6;
                margin-bottom: 12px;
            }
        """)
        analysis_layout.addWidget(title_label)
        
        # 性能指标显示区域
        metrics_group = QGroupBox("📈 性能指标统计")
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(15)
        metrics_layout.setContentsMargins(10, 20, 10, 10)
        
        # Dice系数折线图
        dice_chart_group = QGroupBox("📈 Dice系数变化趋势")
        dice_chart_layout = QVBoxLayout()
        dice_chart_layout.setContentsMargins(10, 20, 10, 10)
        dice_chart_layout.setSpacing(5)
        
        # 创建matplotlib图表
        self.dice_figure = Figure(figsize=(10, 5), dpi=100)
        self.dice_canvas = FigureCanvas(self.dice_figure)
        self.dice_canvas.setMinimumHeight(350)
        self.dice_canvas.setMinimumWidth(600)
        self.dice_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.dice_ax = self.dice_figure.add_subplot(111)
        self.dice_ax.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
        self.dice_ax.set_ylabel('Dice系数', fontsize=11, fontweight='bold')
        self.dice_ax.set_title('训练过程中Dice系数的变化', fontsize=12, fontweight='bold', pad=15)
        self.dice_ax.grid(True, alpha=0.3, linestyle='--')
        self.dice_ax.set_ylim([0, 1])
        self.dice_ax.set_xlim([0, 10])  # 初始显示10个轮次
        self.dice_line, = self.dice_ax.plot([], [], 'o-', color='#4CAF50', linewidth=2.5, 
                                           markersize=8, label='Dice系数', markerfacecolor='#66BB6A',
                                           markeredgecolor='#2E7D32', markeredgewidth=1.5)
        self.dice_ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        # 优化布局，确保所有元素可见
        self.dice_figure.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
        
        dice_chart_layout.addWidget(self.dice_canvas)
        dice_chart_group.setLayout(dice_chart_layout)
        metrics_layout.addWidget(dice_chart_group)
        
        # 创建一个容器widget用于滚动
        metrics_container = QWidget()
        metrics_container_layout = QVBoxLayout()
        metrics_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.metrics_text = QLabel("等待训练开始...\n每个轮次结束后将自动更新性能指标")
        self.metrics_text.setWordWrap(True)
        self.metrics_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metrics_text.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                padding: 15px;
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                line-height: 1.6;
            }
        """)
        self.metrics_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.metrics_text.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 允许选择文本
        
        metrics_container_layout.addWidget(self.metrics_text)
        metrics_container_layout.addStretch()  # 添加弹性空间
        metrics_container.setLayout(metrics_container_layout)
        
        # 添加滚动区域
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setWidget(metrics_container)
        metrics_scroll.setMinimumHeight(200)  # 设置最小高度
        metrics_scroll.setMaximumHeight(400)  # 设置最大高度，超过后可以滚动
        metrics_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 需要时显示滚动条
        metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 水平方向不需要滚动条（因为有自动换行）
        metrics_scroll.setFrameShape(QScrollArea.NoFrame)  # 移除边框
        
        metrics_layout.addWidget(metrics_scroll)
        metrics_group.setLayout(metrics_layout)
        analysis_layout.addWidget(metrics_group)
        
        # 测试集分割结果可视化区域
        viz_group = QGroupBox("🖼️ 测试集分割结果可视化")
        viz_layout = QVBoxLayout()
        
        # 缩放控制按钮
        test_zoom_layout = QHBoxLayout()
        test_zoom_layout.setSpacing(10)
        self.test_zoom_in_btn = QPushButton("🔍+ 放大")
        self.test_zoom_out_btn = QPushButton("🔍- 缩小")
        self.test_zoom_fit_btn = QPushButton("📐 适应窗口")
        self.test_zoom_original_btn = QPushButton("📏 原始大小")
        self.test_zoom_in_btn.setMinimumHeight(35)
        self.test_zoom_out_btn.setMinimumHeight(35)
        self.test_zoom_fit_btn.setMinimumHeight(35)
        self.test_zoom_original_btn.setMinimumHeight(35)
        self.test_zoom_in_btn.clicked.connect(lambda: self.zoom_image('test', 'in'))
        self.test_zoom_out_btn.clicked.connect(lambda: self.zoom_image('test', 'out'))
        self.test_zoom_fit_btn.clicked.connect(lambda: self.zoom_image('test', 'fit'))
        self.test_zoom_original_btn.clicked.connect(lambda: self.zoom_image('test', 'original'))
        test_zoom_layout.addWidget(self.test_zoom_in_btn)
        test_zoom_layout.addWidget(self.test_zoom_out_btn)
        test_zoom_layout.addWidget(self.test_zoom_fit_btn)
        test_zoom_layout.addWidget(self.test_zoom_original_btn)
        test_zoom_layout.addStretch()
        viz_layout.addLayout(test_zoom_layout)
        
        self.test_results_label = QLabel("暂无结果")
        self.test_results_label.setAlignment(Qt.AlignCenter)
        self.test_results_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.test_results_label.setScaledContents(False)  # 不自动缩放，保持原始比例
        
        # 滚动区域用于显示大图
        test_scroll = QScrollArea()
        test_scroll.setWidgetResizable(False)  # 改为False，让图片可以超出窗口大小
        test_scroll.setWidget(self.test_results_label)
        test_scroll.setMinimumHeight(400)
        
        viz_layout.addWidget(test_scroll)
        viz_group.setLayout(viz_layout)
        analysis_layout.addWidget(viz_group)
        
        # 保存原始pixmap和当前缩放比例
        self.test_original_pixmap = None
        self.test_zoom_factor = 1.0
        
        # 性能分析图表区域
        perf_group = QGroupBox("性能分析图表")
        perf_layout = QVBoxLayout()
        
        # 缩放控制按钮
        perf_zoom_layout = QHBoxLayout()
        perf_zoom_layout.setSpacing(10)
        self.perf_zoom_in_btn = QPushButton("🔍+ 放大")
        self.perf_zoom_out_btn = QPushButton("🔍- 缩小")
        self.perf_zoom_fit_btn = QPushButton("📐 适应窗口")
        self.perf_zoom_original_btn = QPushButton("📏 原始大小")
        self.perf_zoom_in_btn.setMinimumHeight(35)
        self.perf_zoom_out_btn.setMinimumHeight(35)
        self.perf_zoom_fit_btn.setMinimumHeight(35)
        self.perf_zoom_original_btn.setMinimumHeight(35)
        self.perf_zoom_in_btn.clicked.connect(lambda: self.zoom_image('perf', 'in'))
        self.perf_zoom_out_btn.clicked.connect(lambda: self.zoom_image('perf', 'out'))
        self.perf_zoom_fit_btn.clicked.connect(lambda: self.zoom_image('perf', 'fit'))
        self.perf_zoom_original_btn.clicked.connect(lambda: self.zoom_image('perf', 'original'))
        perf_zoom_layout.addWidget(self.perf_zoom_in_btn)
        perf_zoom_layout.addWidget(self.perf_zoom_out_btn)
        perf_zoom_layout.addWidget(self.perf_zoom_fit_btn)
        perf_zoom_layout.addWidget(self.perf_zoom_original_btn)
        perf_zoom_layout.addStretch()
        perf_layout.addLayout(perf_zoom_layout)
        
        self.perf_analysis_label = QLabel("暂无结果")
        self.perf_analysis_label.setAlignment(Qt.AlignCenter)
        self.perf_analysis_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.perf_analysis_label.setScaledContents(False)  # 不自动缩放，保持原始比例
        
        perf_scroll = QScrollArea()
        perf_scroll.setWidgetResizable(False)  # 改为False，让图片可以超出窗口大小
        perf_scroll.setWidget(self.perf_analysis_label)
        perf_scroll.setMinimumHeight(400)
        
        perf_layout.addWidget(perf_scroll)
        perf_group.setLayout(perf_layout)
        analysis_layout.addWidget(perf_group)
        
        # 保存原始pixmap和当前缩放比例
        self.perf_original_pixmap = None
        self.perf_zoom_factor = 1.0
        
        # 注意力可解释性分析区域
        attention_group = QGroupBox("🔥 注意力可解释性分析")
        attention_layout = QVBoxLayout()
        attention_layout.setSpacing(12)
        attention_layout.setContentsMargins(15, 20, 15, 15)
        
        # 缩放控制按钮
        att_zoom_layout = QHBoxLayout()
        att_zoom_layout.setSpacing(10)
        self.att_zoom_in_btn = QPushButton("🔍+ 放大")
        self.att_zoom_out_btn = QPushButton("🔍- 缩小")
        self.att_zoom_fit_btn = QPushButton("📐 适应窗口")
        self.att_zoom_original_btn = QPushButton("📏 原始大小")
        self.att_zoom_in_btn.setMinimumHeight(38)
        self.att_zoom_out_btn.setMinimumHeight(38)
        self.att_zoom_fit_btn.setMinimumHeight(38)
        self.att_zoom_original_btn.setMinimumHeight(38)
        self.att_zoom_in_btn.setToolTip("放大注意力可视化图")
        self.att_zoom_out_btn.setToolTip("缩小注意力可视化图")
        self.att_zoom_fit_btn.setToolTip("自动适应窗口大小")
        self.att_zoom_original_btn.setToolTip("显示原始大小")
        self.att_zoom_in_btn.clicked.connect(lambda: self.zoom_image('attention', 'in'))
        self.att_zoom_out_btn.clicked.connect(lambda: self.zoom_image('attention', 'out'))
        self.att_zoom_fit_btn.clicked.connect(lambda: self.zoom_image('attention', 'fit'))
        self.att_zoom_original_btn.clicked.connect(lambda: self.zoom_image('attention', 'original'))
        att_zoom_layout.addWidget(self.att_zoom_in_btn)
        att_zoom_layout.addWidget(self.att_zoom_out_btn)
        att_zoom_layout.addWidget(self.att_zoom_fit_btn)
        att_zoom_layout.addWidget(self.att_zoom_original_btn)
        att_zoom_layout.addStretch()
        attention_layout.addLayout(att_zoom_layout)
        
        self.attention_label = QLabel("⏳ 等待训练完成...\n将显示注意力权重可视化")
        self.attention_label.setAlignment(Qt.AlignCenter)
        self.attention_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cbd5e1;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                color: #64748b;
                font-size: 12pt;
                padding: 20px;
                min-height: 400px;
            }
        """)
        self.attention_label.setScaledContents(False)
        
        attention_scroll = QScrollArea()
        attention_scroll.setWidgetResizable(True)
        attention_scroll.setWidget(self.attention_label)
        attention_scroll.setMinimumHeight(450)
        
        attention_layout.addWidget(attention_scroll)
        attention_group.setLayout(attention_layout)
        analysis_layout.addWidget(attention_group)
        
        # 注意力统计信息 - 使用分割器显示表格和图表
        att_stats_group = QGroupBox("📊 注意力统计分析")
        att_stats_layout = QVBoxLayout()
        att_stats_layout.setSpacing(12)
        att_stats_layout.setContentsMargins(15, 20, 15, 15)
        
        # 使用分割器分割表格和图表
        stats_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：详细统计表格
        table_container = QWidget()
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        table_title = QLabel("📋 详细统计指标")
        table_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        table_title.setStyleSheet("color: #475569; padding: 4px 0;")
        table_layout.addWidget(table_title)
        
        # 创建表格显示统计数据
        self.attention_stats_table = QTableWidget()
        self.attention_stats_table.setColumnCount(3)
        self.attention_stats_table.setHorizontalHeaderLabels(["注意力层", "统计指标", "数值"])
        self.attention_stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attention_stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.attention_stats_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.attention_stats_table.setAlternatingRowColors(True)
        self.attention_stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.attention_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.attention_stats_table.verticalHeader().setVisible(False)
        self.attention_stats_table.setMinimumHeight(250)
        self.attention_stats_table.setMaximumHeight(400)
        self.attention_stats_table.setSortingEnabled(False)  # 暂时禁用排序
        
        # 设置表格样式
        self.attention_stats_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid #e2e8f0;
                border-radius: 10px;
                background-color: #ffffff;
                gridline-color: #f1f5f9;
                font-size: 10pt;
            }
            QTableWidget::item {
                padding: 10px;
                border: none;
            }
            QTableWidget::item:hover {
                background: #f1f5f9;
            }
            QTableWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dbeafe, stop:1 #bfdbfe);
                color: #1e40af;
                font-weight: 500;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                color: #475569;
                padding: 12px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: 600;
                font-size: 11pt;
            }
        """)
        
        table_layout.addWidget(self.attention_stats_table)
        
        # 初始化表格占位提示
        self.attention_stats_table.setRowCount(1)
        placeholder_item = QTableWidgetItem("⏳ 等待训练完成，将显示详细统计数据...")
        placeholder_item.setTextAlignment(Qt.AlignCenter)
        placeholder_item.setFont(QFont("Microsoft YaHei", 10))
        placeholder_item.setForeground(QColor(100, 116, 139))
        self.attention_stats_table.setItem(0, 0, placeholder_item)
        self.attention_stats_table.setSpan(0, 0, 1, 3)  # 合并3列
        self.attention_stats_table.setRowHeight(0, 100)
        
        table_container.setLayout(table_layout)
        
        # 右侧：可视化图表
        chart_container = QWidget()
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        chart_title = QLabel("📈 统计可视化")
        chart_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        chart_title.setStyleSheet("color: #475569; padding: 4px 0;")
        chart_layout.addWidget(chart_title)
        
        # 创建matplotlib图表用于显示统计可视化
        self.attention_chart_figure = Figure(figsize=(6, 4), dpi=100)
        self.attention_chart_canvas = FigureCanvas(self.attention_chart_figure)
        self.attention_chart_canvas.setMinimumHeight(250)
        self.attention_chart_canvas.setMaximumHeight(400)
        self.attention_chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        chart_layout.addWidget(self.attention_chart_canvas)
        chart_container.setLayout(chart_layout)
        
        # 添加到分割器
        stats_splitter.addWidget(table_container)
        stats_splitter.addWidget(chart_container)
        stats_splitter.setStretchFactor(0, 1)
        stats_splitter.setStretchFactor(1, 1)
        stats_splitter.setSizes([400, 400])
        
        att_stats_layout.addWidget(stats_splitter)
        
        # 添加分析建议区域
        analysis_suggestion_label = QLabel("💡 分析建议:")
        analysis_suggestion_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        analysis_suggestion_label.setStyleSheet("color: #475569; padding-top: 8px;")
        att_stats_layout.addWidget(analysis_suggestion_label)
        
        self.attention_analysis_text = QLabel("等待训练完成，将显示注意力分析建议...")
        self.attention_analysis_text.setWordWrap(True)
        self.attention_analysis_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.attention_analysis_text.setStyleSheet("""
            QLabel {
                font-size: 10pt;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fef3c7, stop:1 #fde68a);
                border: 2px solid #f59e0b;
                border-radius: 8px;
                border-left: 4px solid #f59e0b;
                color: #92400e;
                min-height: 60px;
            }
        """)
        self.attention_analysis_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        att_stats_layout.addWidget(self.attention_analysis_text)
        
        att_stats_group.setLayout(att_stats_layout)
        analysis_layout.addWidget(att_stats_group)
        
        # 保存按钮
        save_btn_layout = QHBoxLayout()
        self.save_analysis_btn = QPushButton("💾 保存分析报告")
        self.save_analysis_btn.clicked.connect(self.save_analysis_report)
        self.save_analysis_btn.setEnabled(False)
        self.save_analysis_btn.setMinimumHeight(45)
        save_btn_layout.addStretch()
        save_btn_layout.addWidget(self.save_analysis_btn)
        save_btn_layout.addStretch()
        analysis_layout.addLayout(save_btn_layout)
        
        analysis_layout.addStretch()
        self.tab_widget.addTab(analysis_tab, "性能分析")
        self.tab_indexes["analysis"] = self.tab_widget.indexOf(analysis_tab)
        
        # 存储分析数据
        self.analysis_data = None
        self.test_viz_path = None
        self.perf_analysis_path = None
        self.attention_viz_path = None
        self.attention_stats = None
        self.attention_original_pixmap = None
        self.attention_zoom_factor = 1.0

    def setup_model_test_tab(self):
        """模型测试标签页 - 专门用于测试模型性能"""
        test_tab = QWidget()
        test_layout = QVBoxLayout()
        test_layout.setSpacing(15)
        test_layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题
        title_label = QLabel("🧪 模型测试与性能分析")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #1e293b;
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border-radius: 12px;
                border: 2px solid #3b82f6;
                margin-bottom: 12px;
            }
        """)
        test_layout.addWidget(title_label)
        
        # 使用滚动区域
        test_scroll = QScrollArea()
        test_scroll.setWidgetResizable(True)
        test_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        test_scroll.setFrameShape(QScrollArea.NoFrame)
        
        test_content = QWidget()
        test_content_layout = QVBoxLayout()
        test_content_layout.setSpacing(15)
        test_content_layout.setContentsMargins(15, 15, 15, 15)
        
        # 模型和数据选择区域
        config_group = QGroupBox("⚙️ 测试配置")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(12)
        config_layout.setContentsMargins(15, 20, 15, 15)
        
        # 模型路径选择 - 支持多模型集成
        model_label = QLabel("🤖 模型文件（支持多模型集成）:")
        model_label.setStyleSheet("font-weight: 600; color: #475569;")
        
        # 多模型列表
        model_list_layout = QVBoxLayout()
        self.test_model_list = QListWidget()
        self.test_model_list.setMaximumHeight(120)
        self.test_model_list.setStyleSheet("""
            QListWidget {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #e2e8f0;
            }
            QListWidget::item:selected {
                background-color: #dbeafe;
            }
        """)
        
        model_btn_layout = QHBoxLayout()
        browse_test_model_btn = QPushButton("➕ 添加模型")
        browse_test_model_btn.clicked.connect(self.browse_test_model_path)
        remove_model_btn = QPushButton("➖ 移除选中")
        remove_model_btn.clicked.connect(self.remove_test_model)
        model_btn_layout.addWidget(browse_test_model_btn)
        model_btn_layout.addWidget(remove_model_btn)
        
        model_list_layout.addWidget(self.test_model_list)
        model_list_layout.addLayout(model_btn_layout)
        
        # 测试数据目录选择
        data_label = QLabel("📚 测试数据目录:")
        data_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.test_data_dir_label = QLabel("✗ 未选择数据目录")
        self.test_data_dir_label.setWordWrap(True)
        self.test_data_dir_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                color: #64748b;
                font-size: 9pt;
            }
        """)
        browse_test_data_btn = QPushButton("📁 选择数据目录")
        browse_test_data_btn.clicked.connect(self.browse_test_data_dir)
        
        # 模型架构选择
        arch_label = QLabel("🏗️ 模型架构:")
        arch_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.test_arch_combo = QComboBox()
        self.test_arch_combo.addItem("改进UNet (ImprovedUNet)", "improved_unet")
        self.test_arch_combo.addItem("ResNet-UNet (ResNetUNet)", "resnet_unet")
        self.test_arch_combo.addItem("Transformer+UNet (TransUNet)", "trans_unet")
        self.test_arch_combo.addItem("DS-TransUNet (双尺度Transformer+UNet) ⭐", "ds_trans_unet")
        self.test_arch_combo.addItem("SwinUNet (Swin Transformer+UNet) ⭐推荐", "swin_unet")
        self.test_arch_combo.addItem("SMP U-Net++ (ResNet101) 🚀高级", "smp_unetplusplus")
        self.test_arch_combo.addItem("DeepLabV3+ (ResNet101) 🔥降维打击", "smp_deeplabv3plus")
        self.test_arch_combo.setToolTip(
            "选择模型架构类型：\n"
            "• ImprovedUNet: 基础改进UNet\n"
            "• ResNetUNet: 使用ResNet101编码器\n"
            "• TransUNet: Transformer+UNet混合架构\n"
            "• DS-TransUNet: 双尺度Transformer+UNet，在多个尺度使用Transformer增强多尺度特征提取\n"
            "• SwinUNet: Swin Transformer+UNet混合架构，可配合GWO优化提高Dice指标\n"
            "• SMP U-Net++: 使用segmentation-models-pytorch的U-Net++，支持标准数据集（1通道）和2.5D数据集（3通道）\n"
            "• DeepLabV3+: 使用segmentation-models-pytorch的DeepLabV3+，锁定3通道输入，更接近纯ResNet效果（降维打击，默认选择）"
        )
        
        # 数据集类型选择（测试模块）
        test_dataset_type_label = QLabel("📊 数据集类型:")
        test_dataset_type_label.setStyleSheet("font-weight: 600; color: #475569;")
        self.test_dataset_type_combo = QComboBox()
        self.test_dataset_type_combo.addItem("普通数据集 (单通道输入)", "standard")
        self.test_dataset_type_combo.addItem("2.5D数据集 (TCGA-LGG格式) 🚀", "2.5d")
        self.test_dataset_type_combo.setCurrentIndex(0)  # 默认普通数据集
        self.test_dataset_type_combo.setToolTip(
            "选择测试数据集类型：\n"
            "• 普通数据集: 标准单通道图像输入，适用于大多数分割任务\n"
            "• 2.5D数据集: TCGA-LGG格式，使用相邻切片堆叠为3通道输入\n"
            "  注意：2.5D数据集需要文件名格式为 TCGA_CS_5393_19990606_1.tif\n"
            "  所有模型都支持2.5D数据集，系统会自动调整输入通道数"
        )
        
        # 使用TTA选项
        self.test_use_tta_checkbox = QCheckBox("使用测试时增强 (TTA)")
        self.test_use_tta_checkbox.setChecked(True)
        self.test_use_tta_checkbox.setToolTip("启用TTA可以提升1-3%的Dice系数，但会增加推理时间")
        
        # 导出ONNX按钮
        self.export_onnx_btn = QPushButton("📦 导出ONNX模型")
        self.export_onnx_btn.setMinimumHeight(40)
        self.export_onnx_btn.setToolTip(
            "将选中的PyTorch模型导出为ONNX格式\n"
            "• 支持动态batch size和输入尺寸\n"
            "• 自动从checkpoint读取输入通道数\n"
            "• 默认输入尺寸: 512x512"
        )
        self.export_onnx_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                font-size: 11pt;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1d4ed8);
            }
            QPushButton:disabled {
                background: #94a3b8;
                color: #64748b;
            }
        """)
        self.export_onnx_btn.clicked.connect(self.export_model_to_onnx)
        self.export_onnx_btn.setEnabled(False)  # 默认禁用，需要先选择模型
        
        # 开始测试按钮
        self.start_test_btn = QPushButton("🚀 开始测试")
        self.start_test_btn.setMinimumHeight(50)
        self.start_test_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #059669, stop:1 #047857);
            }
        """)
        self.start_test_btn.clicked.connect(self.start_model_test)
        
        config_layout.addWidget(model_label)
        config_layout.addLayout(model_list_layout)
        config_layout.addWidget(data_label)
        config_layout.addWidget(self.test_data_dir_label)
        config_layout.addWidget(browse_test_data_btn)
        config_layout.addWidget(arch_label)
        config_layout.addWidget(self.test_arch_combo)
        config_layout.addWidget(test_dataset_type_label)
        config_layout.addWidget(self.test_dataset_type_combo)
        config_layout.addWidget(self.test_use_tta_checkbox)
        config_layout.addWidget(self.export_onnx_btn)
        config_layout.addWidget(self.start_test_btn)
        config_group.setLayout(config_layout)
        test_content_layout.addWidget(config_group)
        
        # 测试进度
        self.test_progress = QProgressBar()
        self.test_progress.setMinimum(0)
        self.test_progress.setMaximum(100)
        self.test_progress.setValue(0)
        self.test_status = QLabel("等待开始测试...")
        self.test_status.setStyleSheet("padding: 8px; background: #f1f5f9; border-radius: 6px;")
        test_content_layout.addWidget(self.test_progress)
        test_content_layout.addWidget(self.test_status)
        
        # 结果展示区域 - 使用标签页
        results_tabs = QTabWidget()
        
        # 性能指标标签页
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(10, 10, 10, 10)

        # 推荐阈值（来自阈值扫描的智能选择）
        self.test_recommended_threshold_label = QLabel("推荐阈值: --")
        self.test_recommended_threshold_label.setStyleSheet("""
            QLabel {
                padding: 10px 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fef9c3, stop:1 #fde68a);
                border: 1px solid #f59e0b;
                border-radius: 8px;
                color: #92400e;
                font-weight: 700;
                font-size: 11pt;
            }
        """)
        metrics_layout.addWidget(self.test_recommended_threshold_label)
        
        self.test_metrics_text = QTextEdit()
        self.test_metrics_text.setReadOnly(True)
        self.test_metrics_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.test_metrics_text.setPlaceholderText("测试完成后，性能指标将显示在这里...")
        metrics_layout.addWidget(QLabel("📊 性能指标:"))
        metrics_layout.addWidget(self.test_metrics_text)
        metrics_tab.setLayout(metrics_layout)
        results_tabs.addTab(metrics_tab, "📊 性能指标")

        # 阈值扫描详情标签页
        sweep_tab = QWidget()
        sweep_layout = QVBoxLayout()
        sweep_layout.setContentsMargins(10, 10, 10, 10)

        sweep_title = QLabel("🔎 阈值扫描详情（Threshold | Dice | Precision | Recall | FP Count | Score）")
        sweep_title.setStyleSheet("font-weight: 700; color: #334155;")
        sweep_layout.addWidget(sweep_title)

        # 增加一列显示综合评分 Score，便于观察 GWO 搜索的目标函数
        self.test_sweep_table = QTableWidget(0, 6)
        self.test_sweep_table.setHorizontalHeaderLabels(["阈值", "Global Dice", "Precision", "Recall", "FP Count", "Score"])
        self.test_sweep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.test_sweep_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.test_sweep_table.setSelectionMode(QTableWidget.SingleSelection)
        self.test_sweep_table.horizontalHeader().setStretchLastSection(True)
        self.test_sweep_table.setAlternatingRowColors(True)
        self.test_sweep_table.setStyleSheet("""
            QTableWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
            QHeaderView::section {
                background: #f1f5f9;
                padding: 6px;
                border: 1px solid #e2e8f0;
                font-weight: 700;
                color: #334155;
            }
        """)
        sweep_layout.addWidget(self.test_sweep_table)

        sweep_tab.setLayout(sweep_layout)
        results_tabs.addTab(sweep_tab, "🔎 扫描详情")
        
        # 注意力热图标签页
        attention_tab = QWidget()
        attention_layout = QVBoxLayout()
        attention_layout.setContentsMargins(10, 10, 10, 10)
        
        self.test_attention_label = QLabel("暂无注意力热图")
        self.test_attention_label.setAlignment(Qt.AlignCenter)
        self.test_attention_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0; min-height: 400px;")
        self.test_attention_label.setScaledContents(False)
        
        attention_scroll = QScrollArea()
        attention_scroll.setWidgetResizable(True)
        attention_scroll.setWidget(self.test_attention_label)
        attention_layout.addWidget(QLabel("🔥 注意力热图:"))
        attention_layout.addWidget(attention_scroll)
        attention_tab.setLayout(attention_layout)
        results_tabs.addTab(attention_tab, "🔥 注意力热图")
        
        # Dice系数低的案例标签页
        low_dice_tab = QWidget()
        low_dice_layout = QVBoxLayout()
        low_dice_layout.setContentsMargins(10, 10, 10, 10)
        
        self.low_dice_list = QListWidget()
        self.low_dice_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: #ffffff;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f1f5f9;
            }
            QListWidget::item:hover {
                background: #f8fafc;
            }
        """)
        self.low_dice_list.itemDoubleClicked.connect(self.view_low_dice_case)
        
        low_dice_layout.addWidget(QLabel("⚠️ Dice系数低的案例 (双击查看详情):"))
        low_dice_layout.addWidget(self.low_dice_list)
        low_dice_tab.setLayout(low_dice_layout)
        results_tabs.addTab(low_dice_tab, "⚠️ 低Dice案例")
        
        test_content_layout.addWidget(results_tabs)
        
        test_content.setLayout(test_content_layout)
        test_scroll.setWidget(test_content)
        test_layout.addWidget(test_scroll)
        test_tab.setLayout(test_layout)
        
        self.tab_widget.addTab(test_tab, "🧪 模型测试")
        self.tab_indexes["test"] = self.tab_widget.indexOf(test_tab)
        
        # 初始化测试相关变量
        self.test_model_paths = []  # 改为列表，支持多模型
        self.test_data_dir = None
        self.test_thread = None
        self.test_results = None
        self.low_dice_cases = []

    def setup_ai_assistant_tab(self):
        """AI助手标签页"""
        ai_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # API配置
        config_group = QGroupBox("🔐 API配置")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(10)

        # API地址选择
        url_layout = QHBoxLayout()
        url_label = QLabel("接口地址:")
        url_label.setMinimumWidth(80)
        self.ai_url_combo = QComboBox()
        for display, url in self.ai_base_url_options:
            self.ai_url_combo.addItem(display, url)
        # 设置当前选中的URL（匹配默认值）
        current_index = 0
        for i, (_, url) in enumerate(self.ai_base_url_options):
            if url == self.ai_base_url:
                current_index = i
                break
        self.ai_url_combo.setCurrentIndex(current_index)
        self.ai_url_combo.currentIndexChanged.connect(self.on_api_url_changed)
        self.ai_url_combo.setToolTip("选择要使用的API服务地址")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.ai_url_combo)
        config_layout.addLayout(url_layout)
        
        self.ai_base_label = QLabel(f"当前地址: {self.ai_base_url}")
        self.ai_base_label.setStyleSheet("color: #475569; font-weight: 600; font-size: 9pt;")
        config_layout.addWidget(self.ai_base_label)

        model_layout = QHBoxLayout()
        model_label = QLabel("模型选择:")
        model_label.setMinimumWidth(80)
        self.ai_model_combo = QComboBox()
        for display, value in self.ai_model_options:
            self.ai_model_combo.addItem(display, value)
        # 尝试设置当前模型，如果不存在则使用第一个
        current_model_index = 0
        for i in range(self.ai_model_combo.count()):
            if self.ai_model_combo.itemData(i) == self.ai_model_name:
                current_model_index = i
                break
        self.ai_model_combo.setCurrentIndex(current_model_index)
        self.ai_model_combo.setToolTip("根据选择的API服务显示可用的模型列表\n切换API服务时会自动更新模型选项")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.ai_model_combo)
        config_layout.addLayout(model_layout)

        limits_text = (
            f"资源限制：每分钟请求 {self.ai_limits['rpm']} 次、"
            f"每分钟 {self.ai_limits['tpm']} tokens、"
            f"每周 {self.ai_limits['weekly']:,} tokens"
        )
        limits_label = QLabel(limits_text)
        limits_label.setWordWrap(True)
        limits_label.setStyleSheet("""
            QLabel {
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 8px;
                padding: 8px;
                color: #92400e;
            }
        """)
        config_layout.addWidget(limits_label)

        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        key_label.setMinimumWidth(80)
        self.ai_key_input = QLineEdit()
        self.ai_key_input.setEchoMode(QLineEdit.Password)
        self.ai_key_input.setPlaceholderText("请输入API Key")
        self.ai_key_input.setText(self.ai_api_key)
        # 连接信号，标记用户是否手动修改过API key
        self.ai_key_input.textChanged.connect(self.on_api_key_changed)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.ai_key_input)
        config_layout.addLayout(key_layout)

        self.ai_status_label = QLabel("✅ 已就绪")
        self.ai_status_label.setStyleSheet("""
            QLabel {
                padding: 8px 10px;
                background: #dcfce7;
                border-left: 4px solid #16a34a;
                border-radius: 8px;
                color: #166534;
            }
        """)
        config_layout.addWidget(self.ai_status_label)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # 对话区域
        conversation_group = QGroupBox("💬 对话")
        convo_layout = QVBoxLayout()
        convo_layout.setSpacing(10)

        self.ai_prompt_input = QTextEdit()
        self.ai_prompt_input.setPlaceholderText("请输入您想咨询的问题，例如：\n“如何提升当前分割模型的Dice指标？”")
        self.ai_prompt_input.setMinimumHeight(120)

        self.ai_response_view = QTextBrowser()
        self.ai_response_view.setOpenExternalLinks(True)
        self.ai_response_view.setReadOnly(True)
        self.ai_response_view.setStyleSheet("background: #f8fafc;")
        self.ai_response_view.setMinimumHeight(200)

        button_layout = QHBoxLayout()
        self.ai_send_btn = QPushButton("🚀 发送请求")
        self.ai_send_btn.clicked.connect(self.send_ai_request)
        self.ai_clear_btn = QPushButton("🧹 清空对话")
        self.ai_clear_btn.clicked.connect(self.clear_ai_history)
        button_layout.addWidget(self.ai_send_btn)
        button_layout.addWidget(self.ai_clear_btn)

        convo_layout.addWidget(QLabel("问题输入："))
        convo_layout.addWidget(self.ai_prompt_input)
        convo_layout.addLayout(button_layout)
        convo_layout.addWidget(QLabel("AI回复："))
        convo_layout.addWidget(self.ai_response_view)

        conversation_group.setLayout(convo_layout)
        layout.addWidget(conversation_group)
        layout.addStretch()

        ai_tab.setLayout(layout)
        self.tab_widget.addTab(ai_tab, "🤖 AI助手")
        self.tab_indexes["assistant"] = self.tab_widget.indexOf(ai_tab)
    

    def browse_data_dir(self):
        """选择训练数据目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if directory:
            self.lock.lock()
            self.data_dir = directory
            self.lock.unlock()
            self.data_dir_label.setText(f"✓ {directory}")
            self.data_dir_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
            self.train_btn.setEnabled(True)
            self.update_system_status("data", directory, status="success")
    
    def browse_model_path(self, model_type=None):
        """选择预训练模型
        
        Args:
            model_type: 'resnet'，如果为 None 则选择单模型
        """
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt)")
        if path:
            if model_type == 'resnet':
                self.resnet_model_path = path
                self.resnet_model_path_label.setText(f"✓ {os.path.basename(path)}")
                self.resnet_model_path_label.setStyleSheet("""
                    QLabel {
                        padding: 10px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #dcfce7, stop:1 #bbf7d0);
                        border: 2px solid #16a34a;
                        border-radius: 6px;
                        color: #166534;
                        font-size: 9pt;
                        font-weight: 500;
                    }
                """)
            else:
                self.lock.lock()
                self.model_path = path
                self.lock.unlock()
            self.model_path_label.setText(f"✓ {path}")
            self.model_path_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
            self.update_system_status("train_model", path, status="success")
    
    def browse_pred_model_path(self):
        """选择预测模型"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt)")
        if path:
            self.lock.lock()
            self.model_path = path
            self.lock.unlock()
            self.pred_model_label.setText(f"✓ {path}")
            self.pred_model_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
    
    def browse_test_model_path(self):
        """选择测试模型文件（支持多选）"""
        paths, _ = QFileDialog.getOpenFileNames(self, "选择模型文件（可多选）", "", "PyTorch模型 (*.pth *.pt)")
        for path in paths:
            if path and path not in self.test_model_paths:
                self.test_model_paths.append(path)
                item = QListWidgetItem(f"✓ {os.path.basename(path)}")
                item.setData(Qt.UserRole, path)  # 存储完整路径
                self.test_model_list.addItem(item)
        
        # 如果有模型，启用导出ONNX按钮
        if self.test_model_paths:
            self.export_onnx_btn.setEnabled(True)
    
    def remove_test_model(self):
        """移除选中的模型"""
        current_item = self.test_model_list.currentItem()
        if current_item:
            path = current_item.data(Qt.UserRole)
            if path in self.test_model_paths:
                self.test_model_paths.remove(path)
            self.test_model_list.takeItem(self.test_model_list.row(current_item))
        
        # 如果没有模型了，禁用导出ONNX按钮
        if not self.test_model_paths:
            self.export_onnx_btn.setEnabled(False)
    
    def browse_test_data_dir(self):
        """选择测试数据目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择测试数据目录")
        if directory:
            self.test_data_dir = directory
            self.test_data_dir_label.setText(f"✓ {directory}")
            self.test_data_dir_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 6px;
                    color: #166534;
                    font-size: 9pt;
                    font-weight: 500;
                }
            """)
    
    def export_model_to_onnx(self):
        """导出模型为ONNX格式"""
        if not self.test_model_paths:
            QMessageBox.warning(self, "警告", "请先选择要导出的模型文件")
            return
        
        # 使用第一个选中的模型，或者让用户选择
        model_path = self.test_model_paths[0]
        if len(self.test_model_paths) > 1:
            # 如果有多个模型，让用户选择
            items = [os.path.basename(p) for p in self.test_model_paths]
            item, ok = QInputDialog.getItem(self, "选择模型", "请选择要导出的模型:", items, 0, False)
            if not ok:
                return
            model_path = self.test_model_paths[items.index(item)]
        
        # 获取模型架构和数据集类型
        model_type = self.test_arch_combo.currentData() or self.test_arch_combo.currentText()
        dataset_type = self.test_dataset_type_combo.currentData() or self.test_dataset_type_combo.currentText()
        
        # 询问输入尺寸
        input_size_str, ok = QInputDialog.getText(
            self, 
            "输入尺寸", 
            "请输入输入图像尺寸 (格式: HxW，例如: 512x512):",
            text="512x512"
        )
        if not ok:
            return
        
        try:
            h, w = map(int, input_size_str.split('x'))
            input_size = (h, w)
        except:
            QMessageBox.warning(self, "错误", "输入尺寸格式错误，请使用 HxW 格式（例如: 512x512）")
            return
        
        # 选择输出路径
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        default_path = os.path.join(os.path.dirname(model_path) or ".", f"{base_name}.onnx")
        output_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存ONNX模型", 
            default_path,
            "ONNX模型 (*.onnx)"
        )
        
        if not output_path:
            return
        
        # 显示进度对话框
        progress_dialog = QProgressDialog("正在导出ONNX模型...", "取消", 0, 0, self)
        progress_dialog.setWindowTitle("导出ONNX")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setCancelButton(None)  # 不允许取消
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # 创建临时ModelTestThread来使用其导出方法
            from worker import ModelTestThread
            temp_test_thread = ModelTestThread(
                model_paths=[model_path],
                data_dir=".",  # 不需要数据目录
                model_type=model_type,
                use_tta=False,
                enable_matlab_plots=False,
                dataset_type=dataset_type
            )
            
            # 执行导出
            result_path = temp_test_thread.export_to_onnx(
                model_path=model_path,
                output_path=output_path,
                input_size=input_size,
                input_channels=None,  # 自动推断
                opset_version=11
            )
            
            progress_dialog.close()
            
            if result_path:
                QMessageBox.information(
                    self, 
                    "导出成功", 
                    f"ONNX模型已成功导出到:\n{result_path}\n\n"
                    f"输入尺寸: {input_size[0]}x{input_size[1]}\n"
                    f"模型架构: {model_type}\n"
                    f"数据集类型: {dataset_type}"
                )
            else:
                QMessageBox.critical(self, "导出失败", "ONNX模型导出失败，请查看控制台输出获取详细信息")
                
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self, 
                "导出失败", 
                f"导出过程中发生错误:\n{str(e)}\n\n请查看控制台输出获取详细信息"
            )
            import traceback
            print(f"[ONNX导出] 错误详情:\n{traceback.format_exc()}")
    
    def start_model_test(self):
        """开始模型测试"""
        # 检查模型文件（集成功能已删除，仅支持单模型）
        if len(self.test_model_paths) < 1:
            QMessageBox.warning(self, "警告", "请至少选择一个模型文件")
            return
        
        # 验证第一个模型文件
        if not os.path.exists(self.test_model_paths[0]):
            QMessageBox.warning(self, "警告", "模型文件不存在")
            return
        
        if not self.test_data_dir or not os.path.exists(self.test_data_dir):
            QMessageBox.warning(self, "警告", "请先选择有效的测试数据目录")
            return
        
        # 获取模型架构（从checkpoint推断或用户选择）
        model_type = self.test_arch_combo.currentData() or self.test_arch_combo.currentText()
        use_tta = self.test_use_tta_checkbox.isChecked()
        # 获取数据集类型
        dataset_type = self.test_dataset_type_combo.currentData() or self.test_dataset_type_combo.currentText()
        # 获取MATLAB开关状态（与训练模块保持一致）
        use_matlab = self.use_matlab_checkbox.isChecked() if hasattr(self, 'use_matlab_checkbox') else MATLAB_ENGINE_AVAILABLE
        
        # 创建测试线程（集成功能已删除）
        self.test_thread = ModelTestThread(
            model_paths=[self.test_model_paths[0]],  # 仅使用第一个模型
            data_dir=self.test_data_dir,
            model_type=model_type,
            use_tta=use_tta,
            enable_matlab_plots=use_matlab,  # 传递MATLAB开关状态，保持与主界面设置一致
            dataset_type=dataset_type  # 传递数据集类型
        )
        self.test_thread.update_progress.connect(self.update_test_progress)
        self.test_thread.threshold_sweep_ready.connect(self.on_threshold_sweep_ready)
        self.test_thread.test_finished.connect(self.on_test_finished)
        self.test_thread.start()
        
        self.start_test_btn.setEnabled(False)
        self.test_status.setText("测试进行中...")
        # 清空上一次扫描结果
        if hasattr(self, "test_sweep_table"):
            self.test_sweep_table.setRowCount(0)
        if hasattr(self, "test_recommended_threshold_label"):
            self.test_recommended_threshold_label.setText("推荐阈值: --")

    def on_threshold_sweep_ready(self, payload):
        """接收阈值扫描结果并更新GUI展示"""
        if not payload or not isinstance(payload, dict):
            return
        rows = payload.get("rows", []) or []
        best = payload.get("best", {}) or {}
        recall_floor = float(payload.get("recall_floor", 0.90))
        fallback_used = bool(payload.get("fallback_used", False))

        # 更新推荐阈值展示（显示来自 Brent 搜索的最佳阈值 + 关键指标）
        try:
            thr = float(best.get("threshold", 0.0))
            rec = float(best.get("recall", 0.0))
            dice = float(best.get("dice", 0.0))
            score = float(best.get("score", 0.0))
            warn = "（回退）" if fallback_used else ""
            self.test_recommended_threshold_label.setText(
                f"推荐阈值(Brent): {thr:.2f} | Dice: {dice:.4f} | Recall: {rec*100:.1f}% | Score: {score:.4f} {warn}"
            )
            # Recall 低于阈值时加红提示
            if rec < recall_floor:
                self.test_recommended_threshold_label.setStyleSheet("""
                    QLabel {
                        padding: 10px 12px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #fee2e2, stop:1 #fecaca);
                        border: 1px solid #ef4444;
                        border-radius: 8px;
                        color: #991b1b;
                        font-weight: 800;
                        font-size: 11pt;
                    }
                """)
        except Exception:
            pass

        # 更新表格
        if not hasattr(self, "test_sweep_table"):
            return
        table = self.test_sweep_table
        table.setRowCount(len(rows))

        best_thr = float(best.get("threshold", -1.0))
        for r_idx, r in enumerate(rows):
            thr = float(r.get("threshold", 0.0))
            dice = float(r.get("dice", 0.0))
            prec = float(r.get("precision", 0.0))
            rec = float(r.get("recall", 0.0))
            fp = int(r.get("fp_count", 0))
            score = float(r.get("score", 0.0))

            items = [
                QTableWidgetItem(f"{thr:.2f}"),
                QTableWidgetItem(f"{dice:.4f}"),
                QTableWidgetItem(f"{prec:.4f}"),
                QTableWidgetItem(f"{rec:.4f}"),
                QTableWidgetItem(f"{fp:,}"),
                QTableWidgetItem(f"{score:.4f}"),
            ]
            for c, it in enumerate(items):
                it.setTextAlignment(Qt.AlignCenter)
                table.setItem(r_idx, c, it)

            # 高亮最佳阈值行（包括Score列）
            if abs(thr - best_thr) < 1e-6:
                for c in range(table.columnCount()):
                    cell = table.item(r_idx, c)
                    if cell:
                        cell.setBackground(QColor("#dcfce7"))
                        cell.setForeground(QColor("#166534"))
                        f = cell.font()
                        f.setBold(True)
                        cell.setFont(f)
    
    def update_test_progress(self, value, message):
        """更新测试进度"""
        self.test_progress.setValue(value)
        self.test_status.setText(message)
    
    def on_test_finished(self, detailed_metrics, attention_path, low_dice_cases):
        """测试完成处理"""
        self.start_test_btn.setEnabled(True)
        self.test_results = detailed_metrics
        self.low_dice_cases = low_dice_cases
        
        # 显示性能指标
        self.display_test_metrics(detailed_metrics)
        
        # 显示注意力热图
        if attention_path and os.path.exists(attention_path):
            pixmap = QPixmap(attention_path)
            self.test_attention_label.setPixmap(pixmap.scaled(
                self.test_attention_label.width(), 
                self.test_attention_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        else:
            self.test_attention_label.setText("模型不支持注意力热图或生成失败")
        
        # 显示低Dice案例
        self.display_low_dice_cases(low_dice_cases)
        
        # 切换到测试标签页
        self.switch_to_tab("test")
        
        QMessageBox.information(
            self, "测试完成",
            f"模型测试完成！\n\n"
            f"平均 Dice 系数: {detailed_metrics.get('average', {}).get('dice', 0):.4f}\n"
            f"总样本数: {detailed_metrics.get('total_samples', 0)}\n"
            f"低Dice案例数: {len(low_dice_cases)}"
        )
    
    def display_test_metrics(self, detailed_metrics):
        """显示测试性能指标"""
        avg_metrics = detailed_metrics.get('average', {})
        total_samples = detailed_metrics.get('total_samples', 0)
        
        metrics_text = "=" * 60 + "\n"
        metrics_text += "📊 模型测试性能指标\n"
        metrics_text += "=" * 60 + "\n\n"
        
        metrics_text += f"测试样本总数: {total_samples}\n\n"
        
        metrics_text += "【平均性能指标】\n"
        metrics_text += "-" * 60 + "\n"
        metrics_text += f"Dice系数:        {avg_metrics.get('dice', 0):.4f}\n"
        metrics_text += f"IoU:             {avg_metrics.get('iou', 0):.4f}\n"
        metrics_text += f"精确率 (Precision): {avg_metrics.get('precision', 0):.4f}\n"
        metrics_text += f"召回率 (Recall):    {avg_metrics.get('recall', 0):.4f}\n"
        metrics_text += f"敏感度 (Sensitivity): {avg_metrics.get('sensitivity', 0):.4f}\n"
        metrics_text += f"特异度 (Specificity): {avg_metrics.get('specificity', 0):.4f}\n"
        metrics_text += f"F1分数:          {avg_metrics.get('f1', 0):.4f}\n"
        # 显示HD95，如果是NaN则显示"N/A"
        hd95_val = avg_metrics.get('hd95', float('nan'))
        if np.isnan(hd95_val):
            metrics_text += f"HD95:            N/A (部分样本无法计算)\n\n"
        else:
            metrics_text += f"HD95:            {hd95_val:.4f}\n\n"
        
        # 性能分析
        dice = avg_metrics.get('dice', 0)
        metrics_text += "【性能分析】\n"
        metrics_text += "-" * 60 + "\n"
        if dice >= 0.9:
            metrics_text += "✅ Dice系数表现优秀 (≥0.9)，模型分割精度很高。\n"
        elif dice >= 0.8:
            metrics_text += "✅ Dice系数表现良好 (0.8-0.9)，模型分割精度较好。\n"
        elif dice >= 0.7:
            metrics_text += "⚠️ Dice系数表现一般 (0.7-0.8)，模型分割精度中等，建议进一步优化。\n"
        else:
            metrics_text += "❌ Dice系数较低 (<0.7)，模型分割精度有待提升，建议检查数据质量和模型架构。\n"
        
        precision = avg_metrics.get('precision', 0)
        recall = avg_metrics.get('recall', 0)
        if abs(precision - recall) < 0.1:
            metrics_text += "✅ 精确率和召回率较为平衡，模型在假阳性控制方面表现良好。\n"
        elif precision > recall:
            metrics_text += "⚠️ 精确率高于召回率，模型更倾向于减少假阳性，但可能漏检部分目标。\n"
        else:
            metrics_text += "⚠️ 召回率高于精确率，模型更倾向于捕获所有目标，但可能产生较多假阳性。\n"
        
        self.test_metrics_text.setText(metrics_text)
    
    def display_low_dice_cases(self, low_dice_cases):
        """显示低Dice案例列表"""
        self.low_dice_list.clear()
        
        if not low_dice_cases:
            self.low_dice_list.addItem("✅ 没有低Dice案例（所有样本Dice ≥ 0.7）")
            return
        
        # 按Dice排序
        low_dice_cases_sorted = sorted(low_dice_cases, key=lambda x: x['dice'])
        
        for case in low_dice_cases_sorted:
            image_name = os.path.basename(case['image_path'])
            item_text = f"Dice: {case['dice']:.4f} | IoU: {case['iou']:.4f} | Precision: {case['precision']:.4f} | Recall: {case['recall']:.4f} | {image_name}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, case)  # 存储完整案例数据
            self.low_dice_list.addItem(item)
    
    def view_low_dice_case(self, item):
        """查看低Dice案例详情，显示原始图像、预测mask和真实mask"""
        case_data = item.data(Qt.UserRole)
        if not case_data:
            return
        
        # 创建详情对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("低Dice案例详情")
        dialog.setMinimumSize(1000, 700)
        
        layout = QVBoxLayout(dialog)
        
        # 性能指标文本
        detail_text = f"""
性能指标:
  • Dice系数:     {case_data['dice']:.4f}
  • IoU:          {case_data['iou']:.4f}
  • 精确率:       {case_data['precision']:.4f}
  • 召回率:       {case_data['recall']:.4f}
  • 特异度:       {case_data['specificity']:.4f}

图像路径: {case_data['image_path']}
        """
        text_label = QLabel(detail_text)
        text_label.setStyleSheet("font-size: 12px; padding: 10px;")
        layout.addWidget(text_label)
        
        # 图像显示区域
        images_layout = QHBoxLayout()
        
        # 原始图像
        if 'original_image' in case_data:
            orig_img = case_data['original_image']
            orig_qimg = QImage(orig_img.data, orig_img.shape[1], orig_img.shape[0], orig_img.shape[1], QImage.Format_Grayscale8)
            orig_pixmap = QPixmap.fromImage(orig_qimg).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            orig_label = QLabel()
            orig_label.setPixmap(orig_pixmap)
            orig_label.setAlignment(Qt.AlignCenter)
            orig_label.setStyleSheet("border: 2px solid #3b82f6; padding: 5px;")
            orig_title = QLabel("原始图像")
            orig_title.setAlignment(Qt.AlignCenter)
            orig_layout = QVBoxLayout()
            orig_layout.addWidget(orig_title)
            orig_layout.addWidget(orig_label)
            images_layout.addLayout(orig_layout)
        
        # 预测mask
        if 'pred_mask' in case_data:
            pred_img = case_data['pred_mask']
            pred_qimg = QImage(pred_img.data, pred_img.shape[1], pred_img.shape[0], pred_img.shape[1], QImage.Format_Grayscale8)
            pred_pixmap = QPixmap.fromImage(pred_qimg).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pred_label = QLabel()
            pred_label.setPixmap(pred_pixmap)
            pred_label.setAlignment(Qt.AlignCenter)
            pred_label.setStyleSheet("border: 2px solid #ef4444; padding: 5px;")
            pred_title = QLabel("预测Mask")
            pred_title.setAlignment(Qt.AlignCenter)
            pred_layout = QVBoxLayout()
            pred_layout.addWidget(pred_title)
            pred_layout.addWidget(pred_label)
            images_layout.addLayout(pred_layout)
        
        # 真实mask
        if 'target_mask' in case_data:
            target_img = case_data['target_mask']
            target_qimg = QImage(target_img.data, target_img.shape[1], target_img.shape[0], target_img.shape[1], QImage.Format_Grayscale8)
            target_pixmap = QPixmap.fromImage(target_qimg).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            target_label = QLabel()
            target_label.setPixmap(target_pixmap)
            target_label.setAlignment(Qt.AlignCenter)
            target_label.setStyleSheet("border: 2px solid #10b981; padding: 5px;")
            target_title = QLabel("真实Mask")
            target_title.setAlignment(Qt.AlignCenter)
            target_layout = QVBoxLayout()
            target_layout.addWidget(target_title)
            target_layout.addWidget(target_label)
            images_layout.addLayout(target_layout)
        
        layout.addLayout(images_layout)
        
        # 分析文本
        analysis_text = """
分析:
  • 该案例的Dice系数较低，可能存在以下问题:
    - 目标边界模糊
    - 目标尺寸过小
    - 图像质量较差
    - 模型在该类型样本上表现不佳

建议:
  • 检查该图像的质量和标注准确性
  • 考虑增加类似样本的训练数据
  • 调整模型参数或损失函数权重
        """
        analysis_label = QLabel(analysis_text)
        analysis_label.setStyleSheet("font-size: 11px; padding: 10px; color: #666;")
        layout.addWidget(analysis_label)
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def browse_api_model(self):
        """选择API服务使用的模型"""
        path, _ = QFileDialog.getOpenFileName(self, "选择API模型文件", "", "PyTorch模型 (*.pth *.pt)")
        if path:
            self.api_model_path = path
            self.api_model_label.setText(f"✓ {path}")
            self.api_model_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
    
    def browse_input_images(self):
        """选择输入图像"""
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图像文件", "", 
                                              "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff)")
        if paths:

            for path in paths:
                self.input_list.addItem(path)
            self.update_predict_btn_state()
    
    def clear_input_images(self):
        """清空输入图像列表"""
        self.input_list.clear()
        self.update_predict_btn_state()
    
    def browse_output_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.lock.lock()
            self.output_dir = directory
            self.lock.unlock()
            self.output_dir_label.setText(f"✓ {directory}")
            self.output_dir_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border: 2px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                    font-weight: 500;
                }
            """)
            # 【GUI优化】如果勾选了保存结果，确保checkbox也勾选
            if hasattr(self, 'save_results_checkbox'):
                self.save_results_checkbox.setChecked(True)
            self.update_predict_btn_state()
            self.update_system_status("output_dir", directory, status="success")
    
    def update_predict_btn_state(self):
        """更新预测按钮状态"""
        # 【GUI优化】如果勾选了保存结果，必须选择输出目录；否则不要求
        save_results = self.save_results_checkbox.isChecked() if hasattr(self, 'save_results_checkbox') else True
        output_dir_required = save_results and (self.output_dir is not None)
        enabled = (self.input_list.count() > 0 and 
                   self.model_path is not None and 
                   (not save_results or output_dir_required))
        self.predict_btn.setEnabled(enabled)
    
    def on_save_results_changed(self, state):
        """处理保存结果checkbox状态变化"""
        # 如果取消勾选，允许不选输出目录；如果勾选，必须选择输出目录
        self.update_predict_btn_state()

    def start_api_server(self):
        """启动内置API服务"""
        if self.api_thread and self.api_thread.isRunning():
            QMessageBox.information(self, "提示", "API服务已经在运行中")
            return

        if not self.api_model_path or not os.path.exists(self.api_model_path):
            QMessageBox.warning(self, "警告", "请先选择有效的API模型文件")
            return

        host = self.api_host_input.text().strip() or "0.0.0.0"
        port = self.api_port_spin.value()
        device = self.api_device_combo.currentData()

        try:
            self.api_service = SegmentationAPIService(self.api_model_path, device=device)
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"模型加载失败: {exc}")
            self.set_api_status(f"❌ 模型加载失败: {exc}", status="error")
            self.api_service = None
            return

        self.api_thread = APIServerThread(self.api_service, host, port)
        self.api_thread.status_changed.connect(self.on_api_status_changed)
        self.api_thread.server_started.connect(self.on_api_started)
        self.api_thread.server_stopped.connect(self.on_api_stopped)
        self.api_thread.error_occurred.connect(self.on_api_error)
        self.api_thread.finished.connect(self.on_api_thread_finished)
        self.api_thread.start()

        self.api_start_btn.setEnabled(False)
        self.api_stop_btn.setEnabled(True)
        self.set_api_status("⏳ API服务启动中...", status="info")

    def stop_api_server(self):
        """停止API服务"""
        if self.api_thread and self.api_thread.isRunning():
            self.api_thread.stop()
            self.set_api_status("⏳ 正在停止API服务...", status="info")
        else:
            QMessageBox.information(self, "提示", "API服务当前未运行")

    def on_api_status_changed(self, message):
        self.set_api_status(message, status="info")

    def on_api_started(self, message):
        self.set_api_status(message, status="running")

    def on_api_stopped(self, message):
        self.set_api_status(message, status="info")
        self.api_start_btn.setEnabled(True)
        self.api_stop_btn.setEnabled(False)

    def on_api_error(self, message):
        self.set_api_status(f"❌ API错误: {message}", status="error")
        QMessageBox.warning(self, "API错误", message)

    def on_api_thread_finished(self):
        self.api_thread = None
        self.api_service = None
        self.api_start_btn.setEnabled(True)
        self.api_stop_btn.setEnabled(False)

    def set_api_status(self, text, status="info"):
        """更新API状态显示"""
        styles = {
            "info": """
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e0f2fe, stop:1 #bae6fd);
                    border-left: 4px solid #0284c7;
                    border-radius: 8px;
                    color: #075985;
                    font-size: 10pt;
                }
            """,
            "running": """
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #dcfce7, stop:1 #bbf7d0);
                    border-left: 4px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                    font-size: 10pt;
                }
            """,
            "error": """
                QLabel {
                    padding: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #fee2e2, stop:1 #fecaca);
                    border-left: 4px solid #dc2626;
                    border-radius: 8px;
                    color: #991b1b;
                    font-size: 10pt;
                }
            """
        }
        self.api_status_label.setStyleSheet(styles.get(status, styles["info"]))
        self.api_status_label.setText(text)

    def send_ai_request(self):
        """发送远程AI请求"""
        if self.ai_thread and self.ai_thread.isRunning():
            QMessageBox.information(self, "提示", "正在等待上一条回复，请稍候。")
            return

        prompt = self.ai_prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "请先输入问题")
            return

        api_key = self.ai_key_input.text().strip() or self.ai_api_key
        if not api_key:
            QMessageBox.warning(self, "警告", "请填写API Key")
            return

        self.append_ai_message("用户", prompt, is_markdown=False)
        self.ai_send_btn.setEnabled(False)
        self.set_ai_status_label("⏳ 正在请求AI服务...", status="info")

        selected_model = self.ai_model_combo.currentData() or self.ai_model_combo.currentText()

        self.ai_thread = AIAssistantThread(
            base_url=self.ai_base_url,
            model=selected_model,
            api_key=api_key,
            prompt=prompt
        )
        self.ai_thread.success.connect(self.on_ai_success)
        self.ai_thread.error.connect(self.on_ai_error)
        self.ai_thread.finished.connect(self.on_ai_finished)
        self.ai_thread.start()

    def on_ai_success(self, content: str):
        self.append_ai_message("AI", content, is_markdown=True)
        self.set_ai_status_label("✅ AI回复已收到", status="success")

    def on_ai_error(self, message: str):
        self.append_ai_message("系统", f"请求失败：{message}", is_markdown=False)
        self.set_ai_status_label(f"❌ {message}", status="error")
        QMessageBox.warning(self, "AI请求失败", message)

    def on_ai_finished(self):
        self.ai_send_btn.setEnabled(True)
        self.ai_thread = None

    def clear_ai_history(self):
        self.ai_response_view.clear()
        self.set_ai_status_label("🧼 对话已清空，等待新的问题", status="info")

    def append_ai_message(self, role: str, message: str, is_markdown: bool = False):
        """将聊天内容以HTML追加到对话框，支持Markdown渲染"""
        if not hasattr(self, "ai_response_view"):
            return

        role_html = self.escape_html(role)
        if is_markdown:
            body_html = self.render_markdown_html(message)
        else:
            body_html = self.escape_html(message).replace("\n", "<br>")

        html_block = f"""
        <div style="padding:8px 0;">
            <div style="font-weight:600;color:#0f172a;">{role_html}：</div>
            <div style="margin-top:6px;color:#1e293b;line-height:1.6;">{body_html}</div>
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:12px 0;">
        </div>
        """
        self.ai_response_view.moveCursor(QTextCursor.End)
        self.ai_response_view.insertHtml(html_block)
        self.ai_response_view.moveCursor(QTextCursor.End)
        self.ai_response_view.verticalScrollBar().setValue(
            self.ai_response_view.verticalScrollBar().maximum()
        )

    def render_markdown_html(self, text: str) -> str:
        """将Markdown文本转换为HTML，缺少依赖时退回普通文本"""
        try:
            import markdown  # 延迟导入，避免强依赖

            return markdown.markdown(
                text,
                extensions=["fenced_code", "tables", "nl2br"]
            )
        except Exception:
            return self.escape_html(text).replace("\n", "<br>")

    def escape_html(self, text: str) -> str:
        """安全转义HTML"""
        return html.escape(text or "", quote=False)

    def set_ai_status_label(self, text, status="info"):
        styles = {
            "info": """
                QLabel {
                    padding: 8px 10px;
                    background: #e0f2fe;
                    border-left: 4px solid #0284c7;
                    border-radius: 8px;
                    color: #075985;
                }
            """,
            "success": """
                QLabel {
                    padding: 8px 10px;
                    background: #dcfce7;
                    border-left: 4px solid #16a34a;
                    border-radius: 8px;
                    color: #166534;
                }
            """,
            "error": """
                QLabel {
                    padding: 8px 10px;
                    background: #fee2e2;
                    border-left: 4px solid #dc2626;
                    border-radius: 8px;
                    color: #991b1b;
                }
            """
        }
        self.ai_status_label.setStyleSheet(styles.get(status, styles["info"]))
        self.ai_status_label.setText(text)

    def compute_prediction_statistics(self, results):
        """根据预测概率生成统计信息"""
        if not results:
            return None

        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
        samples = []
        # 兼容不同格式的结果:
        # - (image_np, pred_np, prob_map)
        # - (image_np, pred_np, prob_map, tag)  # 如 nnFormer 标记
        # - 直接为 prob_map 数组
        for idx, item in enumerate(results, start=1):
            # 直接是概率图
            if isinstance(item, np.ndarray):
                prob_map = item
            # 元组 / 列表：取第 3 个作为概率图
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                prob_map = item[2]
            else:
                # 不认识的格式，跳过
                continue

            if prob_map is None:
                continue
            sample = {
                "index": idx,
                "mean_prob": float(np.mean(prob_map)),
                "std_prob": float(np.std(prob_map)),
                "p10": float(np.percentile(prob_map, 10)),
                "p90": float(np.percentile(prob_map, 90)),
                "foreground_ratio": {
                    f"{thr:.2f}": float(np.mean(prob_map >= thr)) for thr in thresholds
                }
            }
            samples.append(sample)

        if not samples:
            return None

        aggregate = {
            "mean_prob": float(np.mean([s["mean_prob"] for s in samples])),
            "std_prob": float(np.mean([s["std_prob"] for s in samples])),
            "p10": float(np.mean([s["p10"] for s in samples])),
            "p90": float(np.mean([s["p90"] for s in samples])),
            "foreground_ratio": {
                key: float(np.mean([s["foreground_ratio"][key] for s in samples]))
                for key in samples[0]["foreground_ratio"].keys()
            }
        }

        return {
            "samples": samples,
            "aggregate": aggregate,
            "thresholds": thresholds
        }

    def on_api_key_changed(self, text):
        """当用户手动修改API key时标记"""
        # 检查当前输入的key是否等于当前服务的默认key
        current_default_key = self.ai_api_key_by_service.get(self.ai_base_url, "")
        if text.strip() != current_default_key:
            self.ai_key_manually_changed = True
        else:
            # 如果用户改回了默认值，重置标记
            self.ai_key_manually_changed = False
    
    def on_api_url_changed(self, index):
        """当用户选择不同的API地址时更新模型选项"""
        if index >= 0:
            selected_url = self.ai_url_combo.itemData(index)
            if selected_url:
                # 保存旧URL，用于判断是否需要更新API key
                old_url = self.ai_base_url
                self.ai_base_url = selected_url
                self.ai_base_label.setText(f"当前地址: {self.ai_base_url}")
                
                # 根据选择的API服务更新模型选项
                if selected_url in self.ai_model_options_by_service:
                    model_options = self.ai_model_options_by_service[selected_url]
                    self.ai_model_options = model_options
                    
                    # 更新模型下拉框
                    model_combo = getattr(self, "ai_model_combo", None)
                    if model_combo:
                        model_combo.clear()
                        for display, value in model_options:
                            model_combo.addItem(display, value)
                        # 默认选择第一个模型
                        if model_options:
                            model_combo.setCurrentIndex(0)
                            self.ai_model_name = model_options[0][1]
                
                # 根据选择的API服务更新API key（如果用户没有手动修改过）
                if selected_url in self.ai_api_key_by_service:
                    new_api_key = self.ai_api_key_by_service[selected_url]
                    key_input = getattr(self, "ai_key_input", None)
                    if key_input:
                        current_key = key_input.text().strip()
                        old_default_key = self.ai_api_key_by_service.get(old_url, "")
                        # 如果当前key等于旧服务的默认key，或者用户没有手动修改过，则自动更新
                        if current_key == old_default_key or not self.ai_key_manually_changed:
                            # 临时断开信号，避免触发手动修改标记
                            try:
                                key_input.textChanged.disconnect(self.on_api_key_changed)
                            except:
                                pass
                            key_input.setText(new_api_key)
                            self.ai_api_key = new_api_key
                            self.ai_key_manually_changed = False
                            # 重新连接信号
                            key_input.textChanged.connect(self.on_api_key_changed)
                        # 如果用户手动修改过，保持用户输入的key不变
    
    def build_threshold_prompt(self, stats, current_threshold):
        """构造发送给LLM的提示词"""
        lines = [
            "你是一名医学图像分割系统的调参与质检助手。",
            "我们已经对若干张图像进行了前景概率预测，下面是统计数据。",
            f"当前用于二值化的阈值为 {current_threshold:.2f}。",
            "请根据统计信息判断是否需要调整阈值，使预测掩膜更加合理。",
            "如果统计显示高概率像素占比很小，可以适当降低阈值；反之可提高。",
            "请仅输出JSON，格式为：",
            '{"recommended_threshold": 0.xx, "reason": "简要说明"}',
            "其中 recommended_threshold 必须在 0.05 到 0.95 之间。"
        ]

        agg = stats["aggregate"]
        lines.append("\n【整体统计】")
        lines.append(f"- 平均概率: {agg['mean_prob']:.4f} (std {agg['std_prob']:.4f})")
        lines.append(f"- P10/P90: {agg['p10']:.4f} / {agg['p90']:.4f}")
        lines.append("- 不同阈值下的前景占比：")
        for thr, ratio in agg["foreground_ratio"].items():
            lines.append(f"  - 阈值 {thr}: 前景像素 {ratio*100:.2f}%")

        lines.append("\n【样本统计】")
        for sample in stats["samples"]:
            fg_ratios = ", ".join(
                [f"{thr}:{ratio*100:.1f}%" for thr, ratio in sample["foreground_ratio"].items()]
            )
            lines.append(
                f"- 样本{sample['index']}: mean={sample['mean_prob']:.4f}, "
                f"std={sample['std_prob']:.4f}, p10/p90={sample['p10']:.4f}/{sample['p90']:.4f}, "
                f"foreground({fg_ratios})"
            )

        lines.append("\n请基于上述数据输出JSON。")
        return "\n".join(lines)

    def request_llm_threshold(self):
        """调用LLM推荐阈值"""
        if not self.prediction_stats:
            QMessageBox.warning(self, "提示", "请先运行一次预测以生成统计数据。")
            return

        api_key_widget = getattr(self, "ai_key_input", None)
        api_key = (api_key_widget.text().strip() if api_key_widget else self.ai_api_key).strip() or self.ai_api_key
        if not api_key:
            QMessageBox.warning(self, "提示", "请在AI助手中填写可用的API Key。")
            return

        model_combo = getattr(self, "ai_model_combo", None)
        model_name = None
        if model_combo and model_combo.currentData():
            model_name = model_combo.currentData()
        elif model_combo:
            model_name = model_combo.currentText()
        else:
            model_name = self.ai_model_name

        if self.llm_threshold_thread and self.llm_threshold_thread.isRunning():
            QMessageBox.information(self, "提示", "上一条请求尚未完成，请稍候。")
            return

        prompt = self.build_threshold_prompt(self.prediction_stats, self.threshold_spin.value())
        self.llm_threshold_thread = AIAssistantThread(
            base_url=self.ai_base_url,
            model=model_name,
            api_key=api_key,
            prompt=prompt
        )
        self.llm_threshold_thread.success.connect(self.on_llm_threshold_success)
        self.llm_threshold_thread.error.connect(self.on_llm_threshold_error)
        self.llm_threshold_thread.finished.connect(self.on_llm_threshold_finished)
        self.llm_threshold_thread.start()
        self.llm_threshold_btn.setEnabled(False)
        self.set_llm_threshold_status("⏳ 正在请求LLM分析阈值...", status="info")

    def on_llm_threshold_success(self, content):
        try:
            data = self.extract_json_from_text(content)
            recommended = float(data.get("recommended_threshold"))
            reason = data.get("reason", "LLM未提供原因")
        except Exception as exc:
            self.set_llm_threshold_status(f"解析LLM回复失败: {exc}", status="error")
            QMessageBox.warning(self, "阈值推荐失败", f"无法解析LLM回复:\n{content}")
            return

        recommended = min(max(recommended, 0.05), 0.95)
        self.threshold_spin.setValue(recommended)
        self.set_llm_threshold_status(
            f"推荐阈值 {recommended:.2f}\n原因: {reason}", status="success"
        )
        QMessageBox.information(
            self,
            "LLM 阈值建议",
            f"推荐使用阈值 {recommended:.2f}\n原因：{reason}\n"
            "请重新运行预测以应用新的阈值。"
        )

    def on_llm_threshold_error(self, message):
        self.set_llm_threshold_status(f"❌ 请求失败: {message}", status="error")
        QMessageBox.warning(self, "LLM请求错误", message)

    def on_llm_threshold_finished(self):
        if self.llm_threshold_thread:
            self.llm_threshold_thread = None
        if self.prediction_stats:
            self.llm_threshold_btn.setEnabled(True)

    def set_llm_threshold_status(self, text, status="info"):
        styles = {
            "info": """
                QLabel {
                    padding: 10px 12px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border-left: 4px solid #94a3b8;
                    color: #475569;
                    font-size: 10pt;
                }
            """,
            "success": """
                QLabel {
                    padding: 10px 12px;
                    background: #dcfce7;
                    border-radius: 8px;
                    border-left: 4px solid #16a34a;
                    color: #166534;
                    font-size: 10pt;
                }
            """,
            "error": """
                QLabel {
                    padding: 10px 12px;
                    background: #fee2e2;
                    border-radius: 8px;
                    border-left: 4px solid #dc2626;
                    color: #991b1b;
                    font-size: 10pt;
                }
            """
        }
        if hasattr(self, "llm_threshold_status"):
            self.llm_threshold_status.setStyleSheet(styles.get(status, styles["info"]))
            self.llm_threshold_status.setText(text)

    def extract_json_from_text(self, text):
        """尝试从LLM回复中解析JSON"""
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            return json.loads(snippet)

        raise ValueError("未找到有效的JSON内容")
    
    def start_training(self):
        """开始训练"""
        if not self.data_dir:
            QMessageBox.warning(self, "警告", "请先选择数据目录")
            return
        
        # 【Bug修复】清空图表，确保每次训练都从空白图表开始
        if hasattr(self, 'dice_ax'):
            self.dice_ax.clear()
            self.dice_ax.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
            self.dice_ax.set_ylabel('Dice系数', fontsize=11, fontweight='bold')
            self.dice_ax.set_title('训练过程中Dice系数的变化', fontsize=12, fontweight='bold', pad=15)
            self.dice_ax.grid(True, alpha=0.3, linestyle='--')
            self.dice_ax.set_ylim([0, 1])
            self.dice_ax.set_xlim([0, 10])
            if hasattr(self, 'dice_canvas'):
                self.dice_canvas.draw()
        
        # 【Bug修复】重置图表防抖状态，确保每次新训练都从第1轮开始
        self.last_plotted_epoch = -1
        self.epoch_history = []
        
        self.train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.train_progress.setValue(0)
        
        save_best = self.save_best_checkbox.isChecked()
        
        # 设置模型架构类型
        selected_arch = self.arch_combo.currentData() or self.arch_combo.currentText()
        os.environ["SEG_MODEL"] = selected_arch

        
        # 获取GWO优化选项（SwinUNet/DS-TransUNet可用）
        use_gwo = self.gwo_checkbox.isChecked() and (
            self.arch_combo.currentData() in ("swin_unet", "ds_trans_unet") or 
            self.arch_combo.currentText().lower().startswith(("swin", "ds_trans"))
        )
        
        selected_optimizer = self.optimizer_combo.currentData() or "adam"
        os.environ["SEG_OPTIMIZER"] = selected_optimizer
        
        # 获取数据集类型
        dataset_type = self.dataset_type_combo.currentData() or self.dataset_type_combo.currentText()
        
        # 【已移除限制】SMP模型（U-Net++和DeepLabV3+）现在支持标准数据集和2.5D数据集
        # 系统会根据数据集类型自动调整输入通道数（标准=1通道，2.5D=3通道）

        # 准备实例化 TrainThread，添加异常捕获以排查初始化失败问题
        print(">>> [DEBUG] 准备实例化 TrainThread...")
        print(f">>> [DEBUG] 数据集类型: {dataset_type}")
        print(f">>> [DEBUG] 模型架构: {selected_arch}")
        try:
            # 获取MATLAB开关状态
            use_matlab = self.use_matlab_checkbox.isChecked() if hasattr(self, 'use_matlab_checkbox') else MATLAB_ENGINE_AVAILABLE
            
            # 【GUI优化】修复训练线程信号重复连接问题
            # 在创建新线程之前，先断开旧线程的所有信号连接（如果存在）
            old_thread = getattr(self, 'train_thread', None)
            if old_thread is not None:
                try:
                    # 断开旧线程的所有信号连接
                    old_thread.update_progress.disconnect()
                    old_thread.update_val_progress.disconnect()
                    old_thread.training_finished.disconnect()
                    old_thread.model_saved.disconnect()
                    old_thread.epoch_completed.disconnect()
                    old_thread.test_results_ready.disconnect()
                    old_thread.metrics_ready.disconnect()
                    old_thread.visualization_ready.disconnect()
                    old_thread.epoch_analysis_ready.disconnect()
                    old_thread.attention_analysis_ready.disconnect()
                    print(">>> [DEBUG] 已断开旧线程的信号连接")
                except (TypeError, RuntimeError):
                    # 如果信号未连接或已断开，忽略错误（这是正常的）
                    pass
                # 等待旧线程完全结束
                if old_thread.isRunning():
                    old_thread.wait()
            
            # 创建新线程
            self.train_thread = TrainThread(
                data_dir=self.data_dir,
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_spin.value(),
                model_path=self.model_path,
                save_best=save_best,
                use_gwo=use_gwo,
                optimizer_type=selected_optimizer,
                dataset_type=dataset_type,  # 传递数据集类型
                enable_matlab_plots=use_matlab  # 传递MATLAB开关状态
            )
            print(">>> [DEBUG] TrainThread 实例化成功")
            
            # 连接所有信号（确保只连接一次）
            self.train_thread.update_progress.connect(self.update_train_progress)
            self.train_thread.update_val_progress.connect(self.update_val_progress)
            self.train_thread.training_finished.connect(self.training_complete)
            self.train_thread.model_saved.connect(self.model_saved)
            self.train_thread.epoch_completed.connect(self.update_train_stats)
            self.train_thread.test_results_ready.connect(self.display_test_results)
            self.train_thread.metrics_ready.connect(self.display_performance_metrics)
            self.train_thread.visualization_ready.connect(self.display_performance_chart)
            self.train_thread.epoch_analysis_ready.connect(self.display_epoch_analysis)
            self.train_thread.attention_analysis_ready.connect(self.display_attention_analysis)
            
            print(">>> [DEBUG] 所有信号连接成功，准备启动线程...")
            self.train_thread.start()
            print(">>> [DEBUG] 线程启动命令已发送")
        except Exception as e:
            import traceback
            print(f">>> [FATAL] TrainThread 初始化失败: {e}")
            print(">>> [FATAL] 详细错误堆栈:")
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "初始化训练线程失败", 
                f"训练线程初始化时发生错误：\n\n{str(e)}\n\n请检查控制台输出的详细错误信息。"
            )
            # 恢复按钮状态
            self.train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)
            self.train_progress.setValue(0)
            return
    
    def _on_arch_changed(self):
        """处理架构选择变化"""
        selected_arch = self.arch_combo.currentData() or self.arch_combo.currentText()
        # 选择 SwinUNet、DS-TransUNet 时启用GWO选项
        is_gwo_supported = (
            selected_arch in ("swin_unet", "ds_trans_unet") or 
            selected_arch.lower().startswith("swin") or 
            selected_arch.lower().startswith("ds_trans")
        )
        self.gwo_checkbox.setEnabled(is_gwo_supported)
        if not is_gwo_supported:
            self.gwo_checkbox.setChecked(False)
        
        # 【已移除限制】SMP模型（U-Net++和DeepLabV3+）现在支持标准数据集和2.5D数据集
        # 用户可以根据需要自由选择数据集类型，系统会自动调整输入通道数
    
    def _on_dataset_type_changed(self):
        """处理数据集类型选择变化"""
        # 【已移除限制】所有模型都支持标准数据集和2.5D数据集
        # 系统会根据数据集类型自动调整输入通道数，无需强制配对
        dataset_type = self.dataset_type_combo.currentData() or self.dataset_type_combo.currentText()
        selected_arch = self.arch_combo.currentData() or self.arch_combo.currentText()
        
        # 可选：显示信息提示（不强制）
        if dataset_type == "2.5d":
            # 2.5D数据集会使用3通道输入（上一张、当前、下一张切片堆叠）
            # 所有模型都支持，系统会自动调整
            pass
    
    def stop_training(self):
        """停止训练"""
        if self.train_thread:
            self.train_thread.stop_requested = True
            self.train_thread.wait()
            self.train_thread = None
        
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
    def update_val_progress(self, value, message):
        """更新验证进度"""
        self.val_progress.setValue(value)
        self.val_status.setText(message)

    def update_train_stats(self, epoch, loss, val_loss, val_dice):
        """更新训练统计信息"""
        self.epoch_label.setText(f"当前轮次: {epoch}")
        self.loss_label.setText(f"训练Loss: {loss:.4f}")
        self.val_loss_label.setText(f"验证Loss: {val_loss:.4f}")  
        self.dice_label.setText(f"Dice系数: {val_dice:.4f}")
        
        # 更新Dice系数折线图（传入当前epoch，用于防抖去重）
        self.update_dice_chart(epoch)
    
    def update_train_progress(self, value, message):
        """更新训练进度"""
        self.train_progress.setValue(value)
        self.train_status.setText(message)
    
    def training_complete(self, message, best_model_path):
        """训练完成处理"""
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.train_status.setText(message)
        
        # 提示用户可以查看性能分析
        QMessageBox.information(
            self, '训练完成',
            f'{message}\n\n'
            '性能分析结果已生成！\n'
            '请切换到"性能分析"标签页查看：\n'
            '- 测试集分割结果可视化\n'
            '- 性能指标统计\n'
            '- 性能分析图表'
        )
        
        # 如果存在最佳模型且用户选择了保存
        if best_model_path and os.path.exists(best_model_path):
            reply = QMessageBox.question(
                self, '保存最佳模型',
                '训练已完成，是否保存最佳模型到指定位置?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.save_model(best_model_path)
    
    def model_saved(self, message):
        """模型保存通知"""
        self.train_status.setText(message)
    
    def save_model(self, temp_model_path):
        """保存模型到指定位置"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存最佳模型",
            "best_model.pth",
            "PyTorch模型 (*.pth *.pt)"
        )
        
        if path:
            try:
                shutil.copyfile(temp_model_path, path)
                QMessageBox.information(self, "成功", f"模型已保存到:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")
    
    def start_prediction(self):
        """开始预测"""
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "警告", "请先选择有效的模型文件")
            return
        
        if self.input_list.count() == 0:
            QMessageBox.warning(self, "警告", "请添加要预测的图像")
            return
        
        # 【GUI优化】统一预测输出目录流程：不再弹窗询问，直接使用checkbox状态和已选择的目录
        save_results = self.save_results_checkbox.isChecked() if hasattr(self, 'save_results_checkbox') else True
        
        # 如果勾选了保存但未选择输出目录，提示用户
        if save_results:
            if not self.output_dir:
                QMessageBox.warning(self, "警告", "您已勾选保存结果，请先选择输出目录")
                return
            output_dir = self.output_dir
        else:
            output_dir = None
        
        image_paths = [self.input_list.itemText(i) for i in range(self.input_list.count())]
        self.predict_btn.setEnabled(False)
        self.predict_progress.setValue(0)
        self.prediction_stats = None
        
        self.predict_thread = PredictThread(
            image_paths=image_paths,
            model_path=self.model_path,
            threshold=self.threshold_spin.value(),
            save_results=save_results,
            output_dir=output_dir
        )
        
        self.predict_thread.update_progress.connect(self.update_predict_progress)
        self.predict_thread.prediction_finished.connect(self.prediction_complete)
        self.predict_thread.start()
        if hasattr(self, 'llm_threshold_btn'):
            self.llm_threshold_btn.setEnabled(False)
            self.set_llm_threshold_status("正在进行预测，完成后可请求LLM推荐阈值", status="info")
    
    def update_predict_progress(self, value, message):
        """更新预测进度"""
        self.predict_progress.setValue(value)
        self.predict_status.setText(message)
    
    def prediction_complete(self, input_images, output_masks, input_numpy_images):
        """预测完成处理"""
        self.predict_btn.setEnabled(True)
        self.predict_status.setText("预测完成")
        
        # 清空之前的结果
        for i in reversed(range(self.result_container_layout.count())):
            widget = self.result_container_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # 保存当前结果
        self.current_results = input_numpy_images
        self.prediction_stats = self.compute_prediction_statistics(input_numpy_images)
        if self.prediction_stats and hasattr(self, 'llm_threshold_btn'):
            self.llm_threshold_btn.setEnabled(True)
            self.set_llm_threshold_status(
                "统计数据已生成，点击“LLM推荐阈值”获取建议。", status="success"
            )
        
        # 清空旧的结果展示和缩略图
        for i in reversed(range(self.result_container_layout.count())):
            item = self.result_container_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if hasattr(self, "thumbnail_layout"):
            for i in reversed(range(self.thumbnail_layout.count())):
                item = self.thumbnail_layout.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        # 显示新结果
        for idx, (image_tuple, input_path, output_path) in enumerate(zip(input_numpy_images, input_images, output_masks)):
            # image_tuple 可能是 (image, pred, prob) 或 (image, pred, prob, tag)
            if isinstance(image_tuple, (list, tuple)) and len(image_tuple) >= 2:
                image_np, pred_np = image_tuple[0], image_tuple[1]
            else:
                # 无法解析的格式，跳过
                print(f"[警告] 无法解析预测结果格式: {type(image_tuple)}")
                continue
            # 确保图像数据是连续的并且类型正确
            image_np = np.ascontiguousarray(image_np)
            pred_np = np.ascontiguousarray(pred_np)
            
            # 确保图像是8位无符号整数格式
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)
            if pred_np.dtype != np.uint8:
                pred_np = (pred_np * 255).astype(np.uint8)
            
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            
            # 转换为QPixmap（已翻译为中文）
            q_img = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            input_pixmap = QPixmap.fromImage(q_img)
            
            # 预测mask有时可能是(H, W)或(H, W, 1)，统一处理成2D单通道
            pred_vis = np.squeeze(pred_np)
            if pred_vis.ndim == 3:
                pred_vis = pred_vis[:, :, 0]
            if pred_vis.ndim != 2:
                raise ValueError(f"预测mask维度非法: shape={pred_vis.shape}, 期望为(H,W)或(H,W,1)")
            
            pred_height, pred_width = pred_vis.shape
            
            # 对于单通道图像，使用 Format_Grayscale8（已翻译为中文）
            pred_q_img = QImage(pred_vis.data, pred_width, pred_height, pred_width, QImage.Format_Grayscale8)
            output_pixmap = QPixmap.fromImage(pred_q_img)
            
            # 输入图像
            input_label = QLabel(f"📷 输入图像 {idx+1}:")
            input_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            input_label.setStyleSheet("""
                QLabel {
                    color: #1976d2;
                    padding: 5px;
                    background-color: #e3f2fd;
                    border-radius: 4px;
                }
            """)
            self.result_container_layout.addWidget(input_label)
            
            input_pixmap = input_pixmap.scaled(512, 512, Qt.KeepAspectRatio)
            input_image = QLabel()
            input_image.setPixmap(input_pixmap)
            input_image.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 5px;
                    background-color: white;
                }
            """)
            input_image.setAlignment(Qt.AlignCenter)
            self.result_container_layout.addWidget(input_image)
            
            # 分割结果
            output_label = QLabel("🎯 分割结果:")
            output_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            output_label.setStyleSheet("""
                QLabel {
                    color: #7b1fa2;
                    padding: 5px;
                    background-color: #f3e5f5;
                    border-radius: 4px;
                }
            """)
            self.result_container_layout.addWidget(output_label)
            
            output_pixmap = output_pixmap.scaled(512, 512, Qt.KeepAspectRatio)
            output_image = QLabel()
            output_image.setPixmap(output_pixmap)
            output_image.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 5px;
                    background-color: white;
                }
            """)
            output_image.setAlignment(Qt.AlignCenter)
            self.result_container_layout.addWidget(output_image)

            # 创建缩略图（点击可快速预览）
            if hasattr(self, "thumbnail_layout"):
                thumb_label = QLabel()
                thumb_pix = input_pixmap.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumb_label.setPixmap(thumb_pix)
                thumb_label.setCursor(Qt.PointingHandCursor)

                def make_handler(index):
                    def handler(event):
                        self.show_result_at(index)
                    return handler

                thumb_label.mousePressEvent = make_handler(idx)
                self.thumbnail_layout.addWidget(thumb_label)

        # 初始化预览为第一张
        if input_numpy_images:
            self.show_result_at(0)
            
            # 添加保存按钮
            save_btn = QPushButton("💾 保存结果")
            save_btn.clicked.connect(lambda _, i=idx: self.save_single_result(i))
            save_btn.setMinimumHeight(40)
            self.result_container_layout.addWidget(save_btn)
            
            # 分隔线
            line = QWidget()
            line.setFixedHeight(1)
            line.setStyleSheet("background-color: #cccccc;")
            line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.result_container_layout.addWidget(line)
        
        # 滚动到顶部
        self.result_scroll.verticalScrollBar().setValue(0)
        
        if any(path is not None for path in output_masks):
            QMessageBox.information(self, "完成", "预测完成! 结果已保存到输出目录")
        else:
            QMessageBox.information(self, "完成", "预测完成! 结果未保存，您可以选择保存单个结果或重新运行预测并选择保存")
    
    def save_single_result(self, index):
        """保存单个结果"""
        if index < 0 or index >= len(self.current_results):
            return
        
        # 兼容 (image, pred, prob) / (image, pred, prob, tag) 等格式
        result_item = self.current_results[index]
        if isinstance(result_item, (list, tuple)) and len(result_item) >= 2:
            image_np, pred_np = result_item[0], result_item[1]
        else:
            print(f"[警告] show_result_at: 无法解析结果格式: {type(result_item)}")
            return
        
        # 让用户选择保存目录和文件名
        path, _ = QFileDialog.getSaveFileName(self, "保存分割结果", 
                                             "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg)")
        
        if path:
            try:
                # 获取文件扩展名
                ext = os.path.splitext(path)[1].lower()
                
                # 保存输入图像
                input_path = os.path.splitext(path)[0] + "_input" + ext
                cv2.imwrite(input_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
                # 保存分割结果
                output_path = os.path.splitext(path)[0] + "_mask" + ext
                cv2.imwrite(output_path, pred_np)
                
                QMessageBox.information(self, "成功", 
f"结果已保存到:\n{input_path}\n{output_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

    def show_result_at(self, index: int):
        """在预览区域显示指定索引的结果"""
        if not self.current_results:
            return
        index = max(0, min(index, len(self.current_results) - 1))
        self.current_result_index = index

        # 兼容 (image, pred, prob) / (image, pred, prob, tag) 等格式
        result_item = self.current_results[index]
        if isinstance(result_item, (list, tuple)) and len(result_item) >= 2:
            image_np, pred_np = result_item[0], result_item[1]
        else:
            print(f"[警告] show_result_at: 无法解析结果格式: {type(result_item)}")
            return

        # 输入图像
        image_np = np.ascontiguousarray(image_np)
        h, w, _ = image_np.shape
        bytes_per_line = 3 * w
        q_img = QImage(image_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        input_pixmap = QPixmap.fromImage(q_img).scaled(
            512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_input_label.setPixmap(input_pixmap)

        # 分割结果mask
        pred_vis = np.squeeze(pred_np)
        if pred_vis.ndim == 3:
            pred_vis = pred_vis[:, :, 0]
        if pred_vis.ndim == 2:
            ph, pw = pred_vis.shape
            pred_q_img = QImage(pred_vis.data, pw, ph, pw, QImage.Format_Grayscale8)
            output_pixmap = QPixmap.fromImage(pred_q_img).scaled(
                512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_output_label.setPixmap(output_pixmap)
        else:
            self.preview_output_label.setText(f"mask 维度非法: {pred_vis.shape}")

        # 更新索引文本
        self.result_index_label.setText(
            f"{index + 1} / {len(self.current_results)}"
        )

    def show_prev_result(self):
        """预览上一张结果"""
        if not self.current_results:
            return
        new_index = (getattr(self, "current_result_index", 0) - 1) % len(
            self.current_results
        )
        self.show_result_at(new_index)

    def show_next_result(self):
        """预览下一张结果"""
        if not self.current_results:
            return
        new_index = (getattr(self, "current_result_index", 0) + 1) % len(
            self.current_results
        )
        self.show_result_at(new_index)
    
    def display_epoch_analysis(self, epoch, viz_path, metrics):
        """显示每个epoch的性能分析结果"""
        if os.path.exists(viz_path):
            # 显示测试集分割结果可视化
            pixmap = QPixmap(viz_path)
            self.test_original_pixmap = pixmap  # 保存原始pixmap
            self.test_zoom_factor = 1.0
            # 初始显示：适应窗口大小，但保持比例
            self._display_image_with_zoom('test', pixmap, 'fit')
        
        # 优先使用完整验证集的Dice（val_dice_history）以与折线图一致
        displayed_dice = metrics.get('dice', 0.0)
        if (self.train_thread is not None and
            hasattr(self.train_thread, 'val_dice_history') and
            len(self.train_thread.val_dice_history) >= epoch):
            displayed_dice = float(self.train_thread.val_dice_history[epoch - 1])
        displayed_f1 = metrics.get('f1', displayed_dice)
        
        # 更新性能指标显示（包含历史信息）
        metrics_text = f"=== 当前 Epoch {epoch} 性能指标 ===\n\n"
        metrics_text += f"【当前轮次指标】\n"
        metrics_text += f"Dice系数: {displayed_dice:.4f}\n"
        metrics_text += f"IoU: {metrics.get('iou', 0):.4f}\n"
        metrics_text += f"精确率: {metrics.get('precision', 0):.4f}\n"
        metrics_text += f"敏感度(召回率): {metrics.get('sensitivity', metrics.get('recall', 0)):.4f}\n"
        metrics_text += f"特异度: {metrics.get('specificity', 0):.4f}\n"
        metrics_text += f"F1分数: {displayed_dice:.4f}\n"
        metrics_text += f"HD95: {metrics.get('hd95', float('nan')):.4f}\n\n"
        
        # 如果有训练历史，显示趋势
        if (self.train_thread is not None and 
            hasattr(self.train_thread, 'val_dice_history') and 
            len(self.train_thread.val_dice_history) > 0):
            metrics_text += f"【训练趋势】\n"
            metrics_text += f"验证Dice历史: {[f'{x:.3f}' for x in self.train_thread.val_dice_history[-5:]]}\n"
            if len(self.train_thread.val_dice_history) > 1:
                trend = "↑ 提升" if self.train_thread.val_dice_history[-1] > self.train_thread.val_dice_history[-2] else "↓ 下降"
                metrics_text += f"趋势: {trend}\n"
            metrics_text += "\n"
        
        metrics_text += "（每个轮次自动更新，训练完成后将显示完整统计）"
        
        self.metrics_text.setText(metrics_text)
        
        # 【Bug修复】移除重复的图表更新调用
        # update_dice_chart() 已由 update_train_stats (epoch_completed 信号) 负责更新
        # 这里不再重复调用，避免每个 epoch 的数据点被绘制两次
        # self.update_dice_chart()  # 已移除
        
        # 【GUI优化】训练过程中不要自动切换Tab（避免打断用户）
        # 只有在用户勾选了"训练时自动切换到分析页"时才自动切换
        auto_switch = getattr(self, 'auto_switch_tab_checkbox', None)
        if auto_switch and auto_switch.isChecked():
            # 仅在第一个epoch或每5个epoch切换一次，避免过于频繁
            if epoch == 1 or epoch % 5 == 0:
                self.tab_widget.setCurrentIndex(3)  # 性能分析标签页是第4个（索引3）
    
    def display_test_results(self, viz_path, detailed_metrics):
        """显示测试集分割结果"""
        self.test_viz_path = viz_path
        self.analysis_data = detailed_metrics
        
        if os.path.exists(viz_path):
            pixmap = QPixmap(viz_path)
            self.test_original_pixmap = pixmap  # 保存原始pixmap
            self.test_zoom_factor = 1.0
            # 初始显示：适应窗口大小，但保持比例
            self._display_image_with_zoom('test', pixmap, 'fit')
            # 【GUI优化】不再自动切换Tab，只更新数据（避免打断用户）
            # 如果需要查看结果，用户可以手动切换到性能分析标签页
        else:
            self.test_results_label.setText(f"无法加载图像: {viz_path}")
            self.test_original_pixmap = None
            QMessageBox.warning(self, "提示", f"无法加载测试集可视化图像: {viz_path}")
    
    def display_performance_chart(self, chart_path):
        """显示性能分析图表"""
        # 检查是否是性能分析图表
        if "performance_analysis" in chart_path and os.path.exists(chart_path):
            self.perf_analysis_path = chart_path
            pixmap = QPixmap(chart_path)
            self.perf_original_pixmap = pixmap  # 保存原始pixmap
            self.perf_zoom_factor = 1.0
            # 初始显示：适应窗口大小，但保持比例
            self._display_image_with_zoom('perf', pixmap, 'fit')
            # 【GUI优化】不再自动切换Tab，只更新数据（避免打断用户）
            # 如果需要查看结果，用户可以手动切换到性能分析标签页
    
    def display_performance_metrics(self, detailed_metrics):
        """显示性能指标"""
        self.analysis_data = detailed_metrics
        
        # 性能分析图表应该在训练线程中已经生成
        if (self.train_thread is not None and 
            hasattr(self.train_thread, 'temp_dir') and 
            self.train_thread.temp_dir):
            perf_path = os.path.join(self.train_thread.temp_dir, "performance_analysis.png")
            if os.path.exists(perf_path):
                self.perf_analysis_path = perf_path
                pixmap = QPixmap(perf_path)
                self.perf_original_pixmap = pixmap  # 保存原始pixmap
                self.perf_zoom_factor = 1.0
                # 初始显示：适应窗口大小，但保持比例
                self._display_image_with_zoom('perf', pixmap, 'fit')
        
        # 格式化指标文本
        avg_metrics = detailed_metrics.get('average', {})
        std_metrics = detailed_metrics.get('std', {})
        
        metrics_text = "=== 模型性能指标统计 ===\n\n"
        metrics_text += f"测试样本数量: {len(detailed_metrics.get('all_samples', {}).get('dice', []))}\n\n"
        
        metrics_text += "【平均值 ± 标准差】\n"
        metric_names_cn = {
            'dice': 'Dice系数',
            'iou': 'IoU',
            'precision': '精确率',
            'recall': '召回率',
            'sensitivity': '敏感度(召回率)',
            'specificity': '特异度',
            'f1': 'F1分数',
            'hd95': 'HD95'
        }
        summary_metrics = ['dice', 'iou', 'precision', 'sensitivity', 'specificity', 'f1', 'hd95']
        for metric_name in summary_metrics:
            avg_val = avg_metrics.get(metric_name, 0)
            std_val = std_metrics.get(metric_name, 0)
            metrics_text += f"{metric_names_cn[metric_name]:12s}: {avg_val:.4f} ± {std_val:.4f}\n"
        
        metrics_text += "\n【详细统计】\n"
        for metric_name in summary_metrics:
            min_val = detailed_metrics.get('min', {}).get(metric_name, 0)
            max_val = detailed_metrics.get('max', {}).get(metric_name, 0)
            median_val = detailed_metrics.get('median', {}).get(metric_name, 0)
            metrics_text += f"{metric_names_cn[metric_name]}:\n"
            metrics_text += f"  最小值: {min_val:.4f}\n"
            metrics_text += f"  最大值: {max_val:.4f}\n"
            metrics_text += f"  中位数: {median_val:.4f}\n\n"
        
        # 性能分析
        metrics_text += "【性能分析】\n"
        dice_avg = avg_metrics.get('dice', 0)
        if dice_avg >= 0.9:
            metrics_text += "Dice系数表现优秀 (≥0.9)，模型分割精度很高。\n"
        elif dice_avg >= 0.8:
            metrics_text += "Dice系数表现良好 (0.8-0.9)，模型分割精度较好。\n"
        elif dice_avg >= 0.7:
            metrics_text += "Dice系数表现一般 (0.7-0.8)，模型分割精度中等，建议进一步优化。\n"
        else:
            metrics_text += "Dice系数较低 (<0.7)，模型分割精度有待提升，建议检查数据质量和模型架构。\n"
        
        precision = avg_metrics.get('precision', 0)
        recall = avg_metrics.get('sensitivity', avg_metrics.get('recall', 0))
        specificity = avg_metrics.get('specificity', 0)
        if abs(precision - recall) < 0.1:
            metrics_text += "精确率和召回率较为平衡，模型在假阳性控制方面表现良好。\n"
        elif precision > recall:
            metrics_text += "精确率高于召回率，模型更倾向于减少假阳性，但可能漏检部分目标。\n"
        else:
            metrics_text += "召回率高于精确率，模型更倾向于捕获所有目标，但可能产生较多假阳性。\n"
        metrics_text += f"特异度平均水平: {specificity:.4f}\n"
        
        self.metrics_text.setText(metrics_text)
        self.save_analysis_btn.setEnabled(True)
        
        # 更新Dice系数折线图
        self.update_dice_chart()
    
    def update_dice_chart(self, epoch=None):
        """更新Dice系数折线图
        
        Args:
            epoch: 当前epoch值（用于防抖去重）。如果为None，则从历史记录长度推断
        """
        if (self.train_thread is None or 
            not hasattr(self.train_thread, 'val_dice_history') or 
            len(self.train_thread.val_dice_history) == 0):
            return
        
        # 【核心修复】防抖去重：如果这个 epoch 已经画过了，直接忽略
        # 如果未传入epoch，从历史记录长度推断（向后兼容）
        if epoch is None:
            epoch = len(self.train_thread.val_dice_history)
        
        if epoch == self.last_plotted_epoch:
            print(f"[UI防抖] 忽略重复的绘图请求: Epoch {epoch}")
            return
        
        # 更新记录
        self.last_plotted_epoch = epoch
        
        dice_values = self.train_thread.val_dice_history
        
        # 【核心修复】确保X轴使用真实的epoch值，而不是基于数据点数量自动生成
        # 如果历史记录长度与epoch不匹配，说明数据可能有问题，使用实际长度
        actual_length = len(dice_values)
        if actual_length != epoch:
            print(f"[UI警告] Epoch {epoch} 与历史记录长度 {actual_length} 不匹配，使用实际长度")
            epoch = actual_length
        
        # 确保 epoch_history 与 dice_values 同步
        # 如果 epoch_history 长度小于 dice_values，补齐缺失的epoch值
        while len(self.epoch_history) < len(dice_values):
            self.epoch_history.append(len(self.epoch_history) + 1)
        
        # 使用真实的epoch值作为X轴（确保每个epoch只对应一个数据点）
        epochs = self.epoch_history[:len(dice_values)]
        
        # 更新折线图数据
        self.dice_ax.clear()
        self.dice_ax.plot(epochs, dice_values, 'o-', color='#4CAF50', linewidth=2.5, 
                        markersize=8, label='Dice系数', markerfacecolor='#66BB6A',
                        markeredgecolor='#2E7D32', markeredgewidth=1.5)
        self.dice_ax.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
        self.dice_ax.set_ylabel('Dice系数', fontsize=11, fontweight='bold')
        self.dice_ax.set_title('训练过程中Dice系数的变化', fontsize=12, fontweight='bold', pad=15)
        self.dice_ax.grid(True, alpha=0.3, linestyle='--')
        self.dice_ax.set_ylim([0, 1])
        
        # 智能调整X轴范围，确保所有数据点可见
        max_epoch = max(epochs) if epochs else 1
        # 如果轮次较少，显示更多空间；如果轮次较多，自动扩展
        if max_epoch <= 10:
            x_max = 10
        else:
            x_max = max_epoch + 2  # 留出一些边距
        
        self.dice_ax.set_xlim([0, x_max])
        
        # 设置X轴刻度，避免过于密集
        if max_epoch <= 20:
            self.dice_ax.set_xticks(range(0, x_max + 1, max(1, x_max // 10)))
        else:
            # 轮次较多时，只显示部分刻度
            step = max(1, max_epoch // 10)
            self.dice_ax.set_xticks(range(0, max_epoch + 1, step))
        
        # 设置Y轴刻度
        self.dice_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.dice_ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        self.dice_ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        # 添加当前最大值标注
        if dice_values:
            max_idx = dice_values.index(max(dice_values))
            max_epoch = epochs[max_idx]
            max_dice = dice_values[max_idx]
            
            # 确保标注不会超出图表范围
            annotation_y = min(max_dice + 0.1, 0.95)
            
            self.dice_ax.annotate(f'最佳: {max_dice:.4f}\n轮次: {max_epoch}', 
                                 xy=(max_epoch, max_dice),
                                 xytext=(max_epoch, annotation_y),
                                 arrowprops=dict(arrowstyle='->', color='#f44336', lw=2, 
                                               connectionstyle="arc3,rad=0.2"),
                                 fontsize=9,
                                 color='#f44336',
                                 fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # 添加当前值标注（最后一个点）
        if len(dice_values) > 0:
            current_epoch = epochs[-1]
            current_dice = dice_values[-1]
            self.dice_ax.annotate(f'当前: {current_dice:.4f}', 
                                 xy=(current_epoch, current_dice),
                                 xytext=(current_epoch + 0.5, current_dice),
                                 fontsize=8,
                                 color='#1976d2',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.6))
        
        # 优化布局，确保所有元素可见
        self.dice_figure.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
        self.dice_canvas.draw()
    
    def display_attention_analysis(self, viz_path, attention_stats):
        """显示注意力可解释性分析结果 - 优化版"""
        self.attention_viz_path = viz_path or ""
        self.attention_stats = attention_stats or {}
        
        has_image = bool(viz_path) and os.path.exists(viz_path)
        
        # 显示注意力可视化图
        if has_image:
            pixmap = QPixmap(viz_path)
            self.attention_original_pixmap = pixmap
            self.attention_zoom_factor = 1.0
            self._display_image_with_zoom('attention', pixmap, 'fit')
            self.attention_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3b82f6;
                    border-radius: 10px;
                    background: #ffffff;
                }
            """)
        else:
            self.attention_original_pixmap = None
            self.attention_zoom_factor = 1.0
            self.attention_label.setText("当前模型不支持注意力可视化或尚未生成结果。")
            self.attention_label.setStyleSheet("""
                QLabel {
                    padding: 16px;
                    border: 2px dashed #94a3b8;
                    border-radius: 10px;
                    color: #475569;
                    background: #f8fafc;
                }
            """)
        
        # 使用表格显示注意力统计信息
        self.attention_stats_table.setRowCount(0)  # 清空表格
        if not attention_stats:
            return
        
        row = 0
        layer_names = {
            'att1': '注意力层1 (最精细)',
            'att2': '注意力层2',
            'att3': '注意力层3',
            'att4': '注意力层4 (深层)'
        }
        
        for att_name in ['att1', 'att2', 'att3', 'att4']:
            if att_name in attention_stats:
                stats = attention_stats[att_name]
                layer_name = layer_names.get(att_name, f'注意力层{att_name[-1]}')
                
                # 添加统计指标行
                metrics = [
                    ('平均权重', stats.get('mean', 0), ''),
                    ('标准差', stats.get('std', 0), ''),
                    ('最大权重', stats.get('max', 0), ''),
                    ('最小权重', stats.get('min', 0), ''),
                    ('熵值', stats.get('entropy', 0), '（分散程度）'),
                    ('集中度', stats.get('concentration', 0), '（高注意力占比）')
                ]
                
                # 设置层名称的合并单元格（使用rowspan）
                layer_start_row = row
                
                for metric_name, value, desc in metrics:
                    self.attention_stats_table.insertRow(row)
                    
                    # 层名称（只在第一行显示，并设置行高）
                    if metric_name == '平均权重':
                        layer_item = QTableWidgetItem(layer_name)
                        layer_item.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
                        # 设置背景色区分不同层（使用QColor对象）
                        layer_colors = {
                            'att1': QColor(254, 243, 199),  # #fef3c7
                            'att2': QColor(253, 230, 138),  # #fde68a
                            'att3': QColor(252, 211, 77),   # #fcd34d
                            'att4': QColor(251, 191, 36)    # #fbbf24
                        }
                        layer_item.setBackground(layer_colors.get(att_name, QColor(255, 255, 255)))
                        self.attention_stats_table.setItem(row, 0, layer_item)
                        self.attention_stats_table.setRowHeight(row, 35)  # 设置行高
                    else:
                        empty_item = QTableWidgetItem("")
                        self.attention_stats_table.setItem(row, 0, empty_item)
                        self.attention_stats_table.setRowHeight(row, 30)
                    
                    # 指标名称
                    metric_item = QTableWidgetItem(f"{metric_name}{desc}")
                    metric_item.setFont(QFont("Microsoft YaHei", 10))
                    self.attention_stats_table.setItem(row, 1, metric_item)
                    
                    # 数值
                    if isinstance(value, (int, float)):
                        if metric_name == '集中度':
                            value_str = f"{value:.2%}"
                        elif metric_name == '熵值':
                            value_str = f"{value:.4f}"
                        else:
                            value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    value_item = QTableWidgetItem(value_str)
                    value_item.setFont(QFont("Courier New", 10, QFont.Bold))
                    value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    # 根据数值设置颜色提示（使用更明显的颜色）
                    if metric_name == '最大权重':
                        if value > 0.8:
                            value_item.setForeground(QColor(22, 163, 74))  # 绿色
                            value_item.setBackground(QColor(220, 252, 231))  # 浅绿背景
                        elif value > 0.5:
                            value_item.setForeground(QColor(217, 119, 6))  # 橙色
                            value_item.setBackground(QColor(255, 247, 237))  # 浅橙背景
                        else:
                            value_item.setForeground(QColor(220, 38, 38))  # 红色
                            value_item.setBackground(QColor(254, 242, 242))  # 浅红背景
                    elif metric_name == '集中度':
                        if value > 0.1:
                            value_item.setForeground(QColor(22, 163, 74))
                            value_item.setBackground(QColor(220, 252, 231))
                        elif value > 0.05:
                            value_item.setForeground(QColor(217, 119, 6))
                            value_item.setBackground(QColor(255, 247, 237))
                        else:
                            value_item.setForeground(QColor(220, 38, 38))
                            value_item.setBackground(QColor(254, 242, 242))
                    elif metric_name == '熵值':
                        if value < 2.0:
                            value_item.setForeground(QColor(22, 163, 74))
                            value_item.setBackground(QColor(220, 252, 231))
                        elif value < 4.0:
                            value_item.setForeground(QColor(217, 119, 6))
                            value_item.setBackground(QColor(255, 247, 237))
                        else:
                            value_item.setForeground(QColor(220, 38, 38))
                            value_item.setBackground(QColor(254, 242, 242))
                    
                    self.attention_stats_table.setItem(row, 2, value_item)
                    row += 1
                
                # 添加分隔行（使用更细的分隔线）
                self.attention_stats_table.insertRow(row)
                for col in range(3):
                    sep_item = QTableWidgetItem("")
                    sep_item.setBackground(QColor(241, 245, 249))  # 浅灰背景
                    sep_item.setFlags(Qt.NoItemFlags)  # 不可选择
                    self.attention_stats_table.setItem(row, col, sep_item)
                self.attention_stats_table.setRowHeight(row, 8)  # 分隔行高度
                row += 1
        
        # 调整列宽
        self.attention_stats_table.resizeColumnsToContents()
        
        # 更新可视化图表
        self._update_attention_charts(attention_stats)
        
        # 更新分析建议文本
        analysis_text = self._generate_detailed_analysis_text(attention_stats)
        self.attention_analysis_text.setText(analysis_text)
        
        # 状态栏提示
        brief_text = self._generate_attention_analysis_text(attention_stats)
        self.statusBar().showMessage(f"✅ 注意力分析完成 | {brief_text}", 5000)
    
    def _update_attention_charts(self, attention_stats):
        """更新注意力统计图表"""
        self.attention_chart_figure.clear()
        
        if not attention_stats:
            ax = self.attention_chart_figure.add_subplot(111)
            ax.text(0.5, 0.5, '等待统计数据...', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            self.attention_chart_canvas.draw()
            return
        
        # 创建2x2子图布局
        gs = self.attention_chart_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 子图1: 各层最大权重对比
        ax1 = self.attention_chart_figure.add_subplot(gs[0, 0])
        layers = []
        max_values = []
        colors = ['#ef4444', '#f97316', '#3b82f6', '#10b981']
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                layers.append(f'层{att_name[-1]}')
                max_values.append(attention_stats[att_name].get('max', 0))
        
        if layers:
            bars = ax1.bar(layers, max_values, color=colors[:len(layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax1.set_ylabel('最大权重', fontsize=10, fontweight='bold')
            ax1.set_title('各层最大注意力权重', fontsize=11, fontweight='bold', pad=10)
            ax1.set_ylim([0, max(max_values) * 1.2 if max_values else 1])
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, max_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图2: 各层集中度对比
        ax2 = self.attention_chart_figure.add_subplot(gs[0, 1])
        conc_values = []
        conc_layers = []
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                conc_layers.append(f'层{att_name[-1]}')
                conc_values.append(attention_stats[att_name].get('concentration', 0) * 100)  # 转换为百分比
        
        if conc_layers:
            bars = ax2.bar(conc_layers, conc_values, color=colors[:len(conc_layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax2.set_ylabel('集中度 (%)', fontsize=10, fontweight='bold')
            ax2.set_title('各层注意力集中度', fontsize=11, fontweight='bold', pad=10)
            ax2.set_ylim([0, max(conc_values) * 1.2 if conc_values else 10])
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, conc_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图3: 各层熵值对比（分散程度）
        ax3 = self.attention_chart_figure.add_subplot(gs[1, 0])
        entropy_values = []
        entropy_layers = []
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                entropy_layers.append(f'层{att_name[-1]}')
                entropy_values.append(attention_stats[att_name].get('entropy', 0))
        
        if entropy_layers:
            bars = ax3.bar(entropy_layers, entropy_values, color=colors[:len(entropy_layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax3.set_ylabel('熵值', fontsize=10, fontweight='bold')
            ax3.set_title('各层注意力分散程度', fontsize=11, fontweight='bold', pad=10)
            ax3.set_ylim([0, max(entropy_values) * 1.2 if entropy_values else 5])
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, entropy_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图4: 各层平均权重对比
        ax4 = self.attention_chart_figure.add_subplot(gs[1, 1])
        mean_values = []
        mean_layers = []
        
        for idx, att_name in enumerate(['att1', 'att2', 'att3', 'att4']):
            if att_name in attention_stats:
                mean_layers.append(f'层{att_name[-1]}')
                mean_values.append(attention_stats[att_name].get('mean', 0))
        
        if mean_layers:
            bars = ax4.bar(mean_layers, mean_values, color=colors[:len(mean_layers)], alpha=0.8, edgecolor='white', linewidth=2)
            ax4.set_ylabel('平均权重', fontsize=10, fontweight='bold')
            ax4.set_title('各层平均注意力权重', fontsize=11, fontweight='bold', pad=10)
            ax4.set_ylim([0, max(mean_values) * 1.2 if mean_values else 1])
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_axisbelow(True)
            
            # 添加数值标签
            for bar, val in zip(bars, mean_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        self.attention_chart_figure.suptitle('注意力统计可视化分析', 
                                            fontsize=13, fontweight='bold', y=0.98)
        self.attention_chart_canvas.draw()
    
    def _generate_detailed_analysis_text(self, attention_stats):
        """生成详细的注意力分析文本"""
        if not attention_stats:
            return "等待训练完成，将显示注意力分析建议..."
        
        analysis_lines = []
        analysis_lines.append("【注意力机制分析报告】\n")
        
        # 分析各层
        for att_name in ['att1', 'att2', 'att3', 'att4']:
            if att_name in attention_stats:
                stats = attention_stats[att_name]
                layer_num = att_name[-1]
                max_val = stats.get('max', 0)
                conc = stats.get('concentration', 0)
                entropy = stats.get('entropy', 0)
                mean_val = stats.get('mean', 0)
                
                analysis_lines.append(f"📊 注意力层{layer_num}:")
                
                # 最大权重分析
                if max_val > 0.8:
                    analysis_lines.append(f"  ✓ 最大权重 {max_val:.3f} - 模型能够强烈聚焦于关键区域")
                elif max_val > 0.5:
                    analysis_lines.append(f"  ⚠ 最大权重 {max_val:.3f} - 模型对关键区域有中等关注")
                else:
                    analysis_lines.append(f"  ✗ 最大权重 {max_val:.3f} - 注意力分布较分散，建议增加训练")
                
                # 集中度分析
                if conc > 0.1:
                    analysis_lines.append(f"  ✓ 集中度 {conc:.1%} - 高注意力区域占比良好")
                elif conc > 0.05:
                    analysis_lines.append(f"  ⚠ 集中度 {conc:.1%} - 注意力分布较为均匀")
                else:
                    analysis_lines.append(f"  ✗ 集中度 {conc:.1%} - 注意力过于分散")
                
                # 熵值分析
                if entropy < 2.0:
                    analysis_lines.append(f"  ✓ 熵值 {entropy:.3f} - 注意力分布集中，聚焦明确")
                elif entropy < 4.0:
                    analysis_lines.append(f"  ⚠ 熵值 {entropy:.3f} - 注意力分布中等分散")
                else:
                    analysis_lines.append(f"  ✗ 熵值 {entropy:.3f} - 注意力分布过于分散")
                
                analysis_lines.append("")
        
        # 总体建议
        analysis_lines.append("【优化建议】")
        
        # 检查att1（最精细层）
        if 'att1' in attention_stats:
            att1_max = attention_stats['att1'].get('max', 0)
            att1_conc = attention_stats['att1'].get('concentration', 0)
            if att1_max < 0.5 or att1_conc < 0.05:
                analysis_lines.append("• 注意力层1（最精细层）表现不佳，建议：")
                analysis_lines.append("  - 增加训练轮次以提升模型聚焦能力")
                analysis_lines.append("  - 检查数据标注质量，确保标注准确")
                analysis_lines.append("  - 考虑调整学习率或使用学习率调度")
        
        # 检查att4（深层）
        if 'att4' in attention_stats:
            att4_mean = attention_stats['att4'].get('mean', 0)
            if att4_mean < 0.2:
                analysis_lines.append("• 注意力层4（深层）注意力值较低，建议：")
                analysis_lines.append("  - 检查模型架构，确保深层特征提取正常")
                analysis_lines.append("  - 考虑使用预训练模型或调整网络深度")
        
        # 综合评估
        all_max = [attention_stats[att].get('max', 0) for att in ['att1', 'att2', 'att3', 'att4'] if att in attention_stats]
        if all_max:
            avg_max = np.mean(all_max)
            if avg_max > 0.7:
                analysis_lines.append("• 整体表现优秀，模型注意力机制工作良好 ✓")
            elif avg_max > 0.5:
                analysis_lines.append("• 整体表现良好，仍有优化空间")
            else:
                analysis_lines.append("• 整体表现需要改进，建议全面检查训练过程")
        
        return "\n".join(analysis_lines)
    
    def _generate_attention_analysis_text(self, attention_stats):
        """生成简短的注意力分析文本（用于状态栏）"""
        analysis_parts = []
        
        if 'att1' in attention_stats:
            att1_max = attention_stats['att1'].get('max', 0)
            att1_conc = attention_stats['att1'].get('concentration', 0)
            if att1_max > 0.8 and att1_conc > 0.1:
                analysis_parts.append("层1聚焦良好")
            elif att1_max > 0.5:
                analysis_parts.append("层1关注中等")
            else:
                analysis_parts.append("层1需改进")
        
        if 'att4' in attention_stats:
            att4_mean = attention_stats['att4'].get('mean', 0)
            if att4_mean > 0.3:
                analysis_parts.append("层4识别大尺度特征")
            else:
                analysis_parts.append("层4提取全局特征")
        
        return " | ".join(analysis_parts) if analysis_parts else "分析完成"
    
    def zoom_image(self, image_type, zoom_action):
        """缩放图片"""
        if image_type == 'test':
            original = self.test_original_pixmap
            label = self.test_results_label
            zoom_factor = self.test_zoom_factor
        elif image_type == 'perf':
            original = self.perf_original_pixmap
            label = self.perf_analysis_label
            zoom_factor = self.perf_zoom_factor
        elif image_type == 'attention':
            original = self.attention_original_pixmap
            label = self.attention_label
            zoom_factor = self.attention_zoom_factor
        else:
            return
        
        if original is None:
            return
        
        self._display_image_with_zoom(image_type, original, zoom_action)
    
    def _display_image_with_zoom(self, image_type, pixmap, zoom_action):
        """根据缩放动作显示图片"""
        if pixmap is None:
            return
        
        if image_type == 'test':
            label = self.test_results_label
            current_factor = self.test_zoom_factor
        elif image_type == 'perf':
            label = self.perf_analysis_label
            current_factor = self.perf_zoom_factor
        elif image_type == 'attention':
            label = self.attention_label
            current_factor = self.attention_zoom_factor
        else:
            return
        
        # 获取滚动区域大小（通过查找父级QScrollArea）
        max_width = 1200
        max_height = 800
        parent = label.parent()
        while parent:
            if isinstance(parent, QScrollArea):
                viewport_size = parent.viewport().size()
                max_width = max(viewport_size.width() - 20, 400)
                max_height = max(viewport_size.height() - 20, 400)
                break
            parent = parent.parent()
        
        if zoom_action == 'in':
            # 放大：增加20%
            new_factor = current_factor * 1.2
        elif zoom_action == 'out':
            # 缩小：减少20%
            new_factor = max(0.1, current_factor * 0.8)
        elif zoom_action == 'fit':
            # 适应窗口：计算合适的缩放比例
            pixmap_size = pixmap.size()
            scale_w = max_width / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            scale_h = max_height / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            new_factor = min(scale_w, scale_h, 1.0)  # 不超过原始大小
        elif zoom_action == 'original':
            # 原始大小
            new_factor = 1.0
        else:
            new_factor = current_factor
        
        # 应用缩放
        if image_type == 'test':
            self.test_zoom_factor = new_factor
        elif image_type == 'perf':
            self.perf_zoom_factor = new_factor
        elif image_type == 'attention':
            self.attention_zoom_factor = new_factor
        
        # 计算新尺寸
        new_size = pixmap.size() * new_factor
        scaled_pixmap = pixmap.scaled(
            int(new_size.width()), 
            int(new_size.height()), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 设置图片并调整label大小
        label.setPixmap(scaled_pixmap)
        label.resize(scaled_pixmap.size())
        label.setText("")
    
    def save_analysis_report(self):
        """保存分析报告"""
        if not self.analysis_data:
            QMessageBox.warning(self, "警告", "没有可保存的分析数据")
            return
        
        # 让用户选择保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return
        
        try:
            # 保存测试结果可视化
            if self.test_viz_path and os.path.exists(self.test_viz_path):
                test_dest = os.path.join(save_dir, "test_results_visualization.png")
                shutil.copy2(self.test_viz_path, test_dest)
            
            # 保存性能分析图表
            if self.perf_analysis_path and os.path.exists(self.perf_analysis_path):
                perf_dest = os.path.join(save_dir, "performance_analysis.png")
                shutil.copy2(self.perf_analysis_path, perf_dest)
            
            # 保存注意力可视化
            if self.attention_viz_path and os.path.exists(self.attention_viz_path):
                att_dest = os.path.join(save_dir, "attention_visualization.png")
                shutil.copy2(self.attention_viz_path, att_dest)
            
            # 保存指标CSV（已翻译为中文）
            if (self.train_thread is not None and 
                hasattr(self.train_thread, 'temp_dir') and 
                self.train_thread.temp_dir):
                metrics_csv = os.path.join(self.train_thread.temp_dir, 'performance_metrics.csv')
                if os.path.exists(metrics_csv):
                    csv_dest = os.path.join(save_dir, "performance_metrics.csv")
                    shutil.copy2(metrics_csv, csv_dest)
            
            # 保存文本报告
            report_path = os.path.join(save_dir, "performance_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("模型性能分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                avg_metrics = self.analysis_data.get('average', {})
                std_metrics = self.analysis_data.get('std', {})
                
                f.write(f"测试样本数量: {len(self.analysis_data.get('all_samples', {}).get('dice', []))}\n\n")
                
                f.write("【平均值 ± 标准差】\n")
                metric_names_cn = {
                    'dice': 'Dice系数',
                    'iou': 'IoU',
                    'precision': '精确率',
                    'recall': '召回率',
                    'sensitivity': '敏感度(召回率)',
                    'specificity': '特异度',
                    'f1': 'F1分数',
                    'hd95': 'HD95'
                }
                summary_metrics = ['dice', 'iou', 'precision', 'sensitivity', 'specificity', 'f1', 'hd95']
                for metric_name in summary_metrics:
                    avg_val = avg_metrics.get(metric_name, 0)
                    std_val = std_metrics.get(metric_name, 0)
                    f.write(f"{metric_names_cn[metric_name]:12s}: {avg_val:.4f} ± {std_val:.4f}\n")
                
                f.write("\n【详细统计】\n")
                for metric_name in summary_metrics:
                    min_val = self.analysis_data.get('min', {}).get(metric_name, 0)
                    max_val = self.analysis_data.get('max', {}).get(metric_name, 0)
                    median_val = self.analysis_data.get('median', {}).get(metric_name, 0)
                    f.write(f"{metric_names_cn[metric_name]}:\n")
                    f.write(f"  最小值: {min_val:.4f}\n")
                    f.write(f"  最大值: {max_val:.4f}\n")
                    f.write(f"  中位数: {median_val:.4f}\n\n")
                
                # 保存注意力统计信息
                if self.attention_stats:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("注意力可解释性分析\n")
                    f.write("=" * 50 + "\n\n")
                    for att_name in ['att1', 'att2', 'att3', 'att4']:
                        if att_name in self.attention_stats:
                            stats = self.attention_stats[att_name]
                            layer_name = f"注意力层{att_name[-1]}"
                            f.write(f"【{layer_name}】\n")
                            f.write(f"  平均权重: {stats['mean']:.4f}\n")
                            f.write(f"  标准差: {stats['std']:.4f}\n")
                            f.write(f"  最大权重: {stats['max']:.4f}\n")
                            f.write(f"  最小权重: {stats['min']:.4f}\n\n")
            
            QMessageBox.information(self, "成功", f"分析报告已保存到:\n{save_dir}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")
    def handle_visualization(self, plot_type, x_data, y_data):
        """处理可视化请求的主线程方法"""
        try:
            if plot_type == "training_history":
                save_path = os.path.join(tempfile.gettempdir(), "training_history.png")
                
                # 使用Agg后端避免GUI问题（已翻译为中文）
                with plt.ioff():  # 关闭交互模式
                    fig = plt.figure(figsize=(12, 5))
                    
                    # 绘制训练曲线
                    ax1 = fig.add_subplot(121)
                    ax1.plot(x_data, y_data['train_loss'], 'b-', label='训练损失')
                    ax1.plot(x_data, y_data['val_loss'], 'r-', label='验证损失')
                    ax1.set_title('训练历史')
                    ax1.set_xlabel('轮次')
                    ax1.set_ylabel('损失')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # 绘制评估指标
                    ax2 = fig.add_subplot(122)
                    ax2.plot(x_data, y_data['val_dice'], 'g-', label='Dice分数')
                    ax2.set_title('验证指标')
                    ax2.set_xlabel('轮次')
                    ax2.set_ylabel('Dice系数')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    fig.savefig(save_path, bbox_inches='tight')
                    plt.close(fig)
                
                self.visualization_ready.emit(save_path)
                
        except Exception as e:
            print(f"可视化错误: {str(e)}")
    def closeEvent(self, event):
        """安全关闭窗口"""
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop_requested = True
            self.train_thread.wait()
        
        if self.predict_thread and self.predict_thread.isRunning():
            self.predict_thread.terminate()
            self.predict_thread.wait()

        if self.api_thread and self.api_thread.isRunning():
            self.api_thread.stop()
            self.api_thread.wait()

        if self.ai_thread:
            if self.ai_thread.isRunning():
                self.ai_thread.terminate()
                self.ai_thread.wait()
            self.ai_thread = None

        if self.llm_threshold_thread and self.llm_threshold_thread.isRunning():
            self.llm_threshold_thread.terminate()
            self.llm_threshold_thread.wait()
        
        event.accept()
    def update_training_plot(self, pixmap):
        """更新界面上的训练曲线图"""
        if hasattr(self, 'plot_label'):
            self.plot_label.setPixmap(pixmap)
        else:
            # 首次创建显示区域
            self.plot_label = QLabel(self)
            self.plot_label.setPixmap(pixmap)
            self.result_container_layout.insertWidget(0, self.plot_label)
    def on_training_epoch_completed(self, epoch, train_loss, val_loss, val_dice):
        """收集训练数据并触发可视化更新"""
        if not hasattr(self, 'training_history'):
            self.training_history = {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'val_dice': []
            }
        
        # 添加新数据
        self.training_history['epochs'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_dice'].append(val_dice)
        
        # 请求可视化更新
        self.visualizer.plot_history(self.training_history)

# 注意：以下类和函数已在 utils.py 中定义，通过 from utils import * 导入：
# - parse_extra_modalities_spec
# - build_extra_modalities_lists
# - normalize_volume_percentile
# - MedicalImageDataset

# EarlyStopping 类已在 utils.py 中定义，worker.py 从 utils.py 导入

# 注意：以下 MATLAB 相关类已标记为已移除，但保留在此文件中以避免导入错误
# 如果不再需要，可以删除这些类定义
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
    """MATLAB 引擎功能已移除。"""

    @classmethod
    def instance(cls):
            return None


class MatlabMetricsBridge:
    """MATLAB HD95 计算功能已移除。"""

    @classmethod
    def instance(cls):
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学图像分割GUI/API应用")
    parser.add_argument(
        "--mode",
        choices=["gui", "api"],
        default="gui",
        help="运行模式: gui(默认) 或 api",
    )
    parser.add_argument(
        "--model",
        help="API模式下用于推理的模型路径(.pth/.pt)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API模式监听地址，默认0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API模式端口，默认8000",
    )
    parser.add_argument(
        "--device",
        help="API模式下指定推理设备，例如cpu或cuda:0",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="API模式是否启用热重载(开发用途)",
    )
    args = parser.parse_args()

    # 【Windows 多进程支持】在 if __name__ == '__main__' 中预热 MATLAB 引擎
    # 确保所有执行逻辑都在主进程中进行，避免多进程导入时的问题
    _warmup_matlab_engine()

    if args.mode == "gui":
        # QApplication已在文件顶部导入，无需重复导入
        qt_app = QApplication(sys.argv)
        window = MedicalSegmentationApp()
        window.show()
        sys.exit(qt_app.exec_())    
    else:
        if not args.model:
            parser.error("API模式必须通过--model提供模型路径")
        service = SegmentationAPIService(model_path=args.model, device=args.device)
        api_app = create_segmentation_api(service)
        try:
            uvicorn = importlib.import_module("uvicorn")
        except ImportError as exc:
            raise ImportError("运行API模式需要安装uvicorn: pip install uvicorn") from exc

        uvicorn.run(api_app, host=args.host, port=args.port, reload=args.reload)


