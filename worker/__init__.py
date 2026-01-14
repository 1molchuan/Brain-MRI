# -*- coding: utf-8 -*-
"""
工作线程模块
保持向后兼容，从worker模块导入所有线程类
"""

# 导入所有线程类，保持与原来 worker.py 相同的导入方式
from worker.test_thread import ModelTestThread
from worker.train_thread import TrainThread
from worker.predict_thread import PredictThread

# 导出所有类，保持向后兼容
__all__ = ['ModelTestThread', 'TrainThread', 'PredictThread']

