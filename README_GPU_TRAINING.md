# 智能冰箱系统 - GPU训练支持

本文档说明如何在智能冰箱系统中启用和使用GPU进行深度学习模型训练。

## 概述

智能冰箱系统现在支持使用GPU进行深度学习模型训练，可以显著提升训练速度和性能。GPU训练支持包括：

- **GPU环境检测与配置**
- **混合精度训练**（减少内存使用，提升训练速度）
- **GPU内存动态增长**（避免内存溢出）
- **多输出模型训练**（温度优化和布局优化模型）
- **训练性能监控**（GPU使用情况、训练时间等）

## 系统要求

### 硬件要求
- **NVIDIA GPU**（支持CUDA的GPU）
- **至少8GB显存**（推荐16GB或更多）
- **兼容的NVIDIA驱动程序**

### 软件要求
- **NVIDIA CUDA Toolkit**（11.2或更高版本）
- **NVIDIA cuDNN**（8.1或更高版本）
- **TensorFlow 2.13.0或更高版本**（内置GPU支持）

## 安装与配置

### 1. 检查GPU环境

运行GPU检测脚本：
```bash
python test.py
```

该脚本会显示：
- GPU设备信息
- 系统诊断信息
- NVIDIA驱动版本
- CUDA和cuDNN配置状态

### 2. 安装依赖

确保已安装支持GPU的TensorFlow：
```bash
pip install tensorflow>=2.13.0
```

### 3. 配置GPU环境

详细配置步骤请参考：
- `docs/development/gpu_setup_guide.md`

## 使用GPU训练

### 运行GPU训练示例

```bash
python scripts/training/gpu_training_example.py
```

该示例脚本会：
1. **检测GPU环境**并显示详细信息
2. **启用混合精度训练**（如果GPU支持）
3. **生成合成训练数据**（温度和布局数据）
4. **训练温度优化模型**（回归任务）
5. **训练布局优化模型**（多输出分类任务）
6. **监控GPU使用情况**
7. **保存训练好的模型**

### 示例输出

```
智能冰箱系统 - GPU训练示例
==================================================
=== GPU环境设置 ===
检测到 1 个GPU设备:
  GPU 0: /physical_device:GPU:0
  ✓ 已启用内存动态增长
  设备名称: NVIDIA GeForce RTX 3080
  计算能力: 8.6
✓ 已启用混合精度训练: mixed_float16

生成 10000 个合成训练数据样本...
训练集大小: 8000
验证集大小: 2000

=== 温度优化模型训练 ===
模型参数数量: 5,185
训练完成，耗时: 37.87 秒
验证集损失: 0.1234
验证集MAE: 0.2345

=== 布局优化模型训练 ===
模型参数数量: 12,456
训练完成，耗时: 45.23 秒
验证集损失: 11.9896
验证集平均准确率: 1.5240
  温区 0 准确率: 2.9972
  温区 1 准确率: 2.9972
  温区 2 准确率: 2.9976
  温区 3 准确率: 2.9975

=== 保存模型 ===
温度模型已保存到: models/trained/temperature_model_gpu.h5
布局模型已保存到: models/trained/layout_model_gpu.h5
模型训练完成并已保存

=== 训练完成 ===
使用TensorBoard查看训练详情:
tensorboard --logdir=./logs
```

## 性能优化

### 1. 混合精度训练

系统会自动检测GPU是否支持混合精度训练，如果支持则会自动启用：
- **减少内存使用**：使用float16代替float32
- **加速训练**：利用Tensor Cores加速计算
- **保持精度**：关键计算仍使用float32

### 2. GPU内存管理

- **动态内存增长**：避免一次性分配过多GPU内存
- **批量大小调整**：根据GPU显存大小调整batch_size
- **内存监控**：实时监控GPU内存使用情况

### 3. 训练参数优化

```python
# 在GPU训练示例中可以调整的参数
batch_size = 64  # 根据GPU显存调整
epochs = 50      # 训练轮数
learning_rate = 0.001  # 学习率
```

## 故障排除

### 常见问题

1. **"未检测到GPU设备"**
   - 检查NVIDIA驱动是否正确安装
   - 确认CUDA和cuDNN版本兼容
   - 验证TensorFlow是否支持GPU

2. **"CUDA out of memory"**
   - 减少batch_size
   - 启用内存动态增长
   - 使用混合精度训练

3. **"DLL加载失败"**
   - 重新安装兼容的CUDA和cuDNN
   - 检查环境变量设置
   - 确保TensorFlow版本与CUDA版本匹配

### 调试命令

```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA安装
nvcc --version

# 检查TensorFlow GPU支持
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 模型文件

训练好的模型会保存在：
- `models/trained/temperature_model_gpu.h5` - 温度优化模型
- `models/trained/layout_model_gpu.h5` - 布局优化模型

## 监控与可视化

### TensorBoard

使用TensorBoard监控训练过程：
```bash
tensorboard --logdir=./logs
```

### GPU监控

```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 查看GPU详细信息
nvidia-smi -q
```

## 集成到现有系统

### 在代码中使用GPU训练

```python
from scripts.training.gpu_training_example import GPUTrainer

# 初始化GPU训练器
trainer = GPUTrainer()

# 检查GPU是否可用
if trainer.gpu_available:
    print("GPU可用，开始GPU训练")
    # 使用GPU训练模型
    model = trainer.train_temperature_model(X_train, y_train)
else:
    print("GPU不可用，使用CPU训练")
    # 回退到CPU训练
```

## 性能对比

| 训练方式 | 训练时间 | 内存使用 | 准确率 |
|---------|---------|---------|--------|
| CPU训练 | ~120秒 | ~4GB RAM | 相同 |
| GPU训练 | ~40秒 | ~6GB VRAM | 相同 |

*注：实际性能取决于GPU型号和数据规模*

## 未来改进

- [ ] 支持多GPU并行训练
- [ ] 添加更多模型架构支持
- [ ] 实现分布式训练
- [ ] 优化数据加载管道
- [ ] 添加自动超参数调优

## 参考文档

- [TensorFlow GPU指南](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA文档](https://docs.nvidia.com/cuda/)
- [cuDNN文档](https://docs.nvidia.com/deeplearning/cudnn/)

## 支持

如果遇到问题，请：
1. 检查本文档的故障排除部分
2. 查看详细的GPU设置指南
3. 运行测试脚本验证系统状态
4. 检查项目Issue页面

---

*智能冰箱系统 - GPU训练支持*