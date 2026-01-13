# Worker 模块拆分说明

## 文件结构

```
worker/
├── __init__.py          # 向后兼容的导入接口
├── common.py            # 公共导入和配置（所有线程共享）
├── test_thread.py       # ModelTestThread 类（模型测试线程）
├── train_thread.py      # TrainThread 类（训练线程）
└── predict_thread.py    # PredictThread 类（预测线程）
```

## 拆分详情

### 1. `common.py` - 公共模块
- 包含所有线程类共享的导入
- 包括：PyQt5、PyTorch、数据处理、可视化等
- 避免重复导入，统一管理依赖

### 2. `test_thread.py` - 测试线程
- **类名**: `ModelTestThread`
- **行数**: 约 2000 行
- **功能**: 模型测试、评估、阈值扫描、注意力图生成
- **依赖**: 需要 `TrainThread`（延迟导入，避免循环依赖）

### 3. `train_thread.py` - 训练线程
- **类名**: `TrainThread`
- **行数**: 约 8650 行
- **功能**: 模型训练、验证、GWO优化、超参数搜索
- **依赖**: 无（独立模块）

### 4. `predict_thread.py` - 预测线程
- **类名**: `PredictThread`
- **行数**: 约 200 行
- **功能**: 模型预测、结果保存
- **依赖**: 无（独立模块）

### 5. `__init__.py` - 向后兼容接口
- 导出所有线程类
- 保持与原来 `from worker import ...` 相同的导入方式
- 无需修改现有代码

## 使用方式

### 原有导入方式（保持不变）
```python
from worker import ModelTestThread, TrainThread, PredictThread
```

### 新的导入方式（可选）
```python
from worker.test_thread import ModelTestThread
from worker.train_thread import TrainThread
from worker.predict_thread import PredictThread
```

## 注意事项

1. **循环依赖**: `test_thread.py` 使用延迟导入 `TrainThread`，避免循环依赖
2. **向后兼容**: 通过 `__init__.py` 保持原有导入方式不变
3. **公共依赖**: 所有公共导入都在 `common.py` 中，便于统一管理

## 拆分优势

1. ✅ **可维护性提升**: 每个文件职责单一，易于定位和修改
2. ✅ **协作友好**: 减少合并冲突，支持并行开发
3. ✅ **性能优化**: 按需导入，减少启动时间
4. ✅ **代码组织**: 模块化结构，便于理解和扩展

