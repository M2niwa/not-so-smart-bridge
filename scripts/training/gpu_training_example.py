#!/usr/bin/env python3
"""
GPU训练示例脚本

此脚本展示如何在智能冰箱系统中使用GPU进行深度学习模型训练。
包括GPU环境检测、内存管理、混合精度训练和性能监控。
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ai.self_learning.models.temperature_model import TemperatureOptimizationModel
from src.ai.self_learning.models.layout_model import LayoutOptimizationModel

class GPUTrainer:
    """GPU训练管理器"""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_devices = []
        self.setup_gpu_environment()
    
    def setup_gpu_environment(self):
        """设置GPU训练环境"""
        print("=== GPU环境设置 ===")
        
        # 检测GPU设备
        self.gpu_devices = tf.config.list_physical_devices('GPU')
        self.gpu_available = len(self.gpu_devices) > 0
        
        if self.gpu_available:
            print(f"检测到 {len(self.gpu_devices)} 个GPU设备:")
            
            # 配置每个GPU
            for i, gpu in enumerate(self.gpu_devices):
                print(f"  GPU {i}: {gpu.name}")
                
                # 启用内存动态增长
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  ✓ 已启用内存动态增长")
                except RuntimeError as e:
                    print(f"  ✗ 内存动态增长设置失败: {e}")
                
                # 获取GPU详细信息
                try:
                    device_details = tf.config.experimental.get_device_details(gpu)
                    print(f"  设备名称: {device_details.get('device_name', 'N/A')}")
                    print(f"  计算能力: {device_details.get('compute_capability', 'N/A')}")
                except Exception as e:
                    print(f"  无法获取设备详细信息: {e}")
            
            # 设置混合精度训练
            self.setup_mixed_precision()
            
        else:
            print("未检测到可用的GPU设备，将使用CPU进行训练")
            print("请参考 docs/development/gpu_setup_guide.md 配置GPU环境")
    
    def setup_mixed_precision(self):
        """设置混合精度训练"""
        try:
            # 检查GPU是否支持混合精度
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✓ 已启用混合精度训练: {policy.name}")
            print("  混合精度训练可以:")
            print("  - 减少GPU内存使用")
            print("  - 加速训练速度")
            print("  - 保持模型精度")
        except Exception as e:
            print(f"✗ 混合精度训练设置失败: {e}")
            print("  将使用默认精度 (float32)")
    
    def generate_synthetic_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成训练数据"""
        print(f"\n生成 {num_samples} 个合成训练数据样本...")
        
        # 模拟温度数据
        temperature_features = np.random.rand(num_samples, 10).astype(np.float32)
        # 模拟目标温度值
        temperature_labels = np.random.rand(num_samples, 1).astype(np.float32)
        
        # 模拟布局数据 (20种食物类型特征)
        layout_features = np.random.rand(num_samples, 20).astype(np.float32)
        
        # 为布局模型生成多输出标签 (4个温区，每个温区对20种食物类型的分配概率)
        num_zones = 4
        layout_labels = []
        for _ in range(num_zones):
            # 为每个温区生成随机分配概率
            zone_labels = np.random.dirichlet(np.ones(20), size=num_samples)
            layout_labels.append(zone_labels)
        
        return (temperature_features, temperature_labels), (layout_features, layout_labels)
    
    def train_temperature_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> TemperatureOptimizationModel:
        """训练温度优化模型"""
        print("\n=== 温度优化模型训练 ===")
        
        # 创建模型
        model = TemperatureOptimizationModel()
        
        print(f"模型参数数量: {model.model.count_params():,}")
        
        # 设置回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs/temperature_training',
                histogram_freq=1,
                profile_batch='500,520'
            )
        ]
        
        # 训练模型
        start_time = time.time()
        
        history = model.train(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.2 if X_val is None else None
        )
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f} 秒")
        
        # 评估模型
        if X_val is not None and y_val is not None:
            val_loss, val_mae = model.model.evaluate(X_val, y_val, verbose=0)
            print(f"验证集损失: {val_loss:.4f}")
            print(f"验证集MAE: {val_mae:.4f}")
        
        return model
    
    def train_layout_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> LayoutOptimizationModel:
        """训练布局优化模型"""
        print("\n=== 布局优化模型训练 ===")
        
        # 创建模型
        model = LayoutOptimizationModel(num_zones=4, num_food_types=20)
        
        # 重新编译模型以确保正确的指标设置
        model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'] * 4  # 4个温区，每个一个accuracy指标
        )
        
        print(f"模型参数数量: {model.model.count_params():,}")
        
        # 设置回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs/layout_training',
                histogram_freq=1,
                profile_batch='500,520'
            )
        ]
        
        # 训练模型
        start_time = time.time()
        
        history = model.train(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2 if X_val is None else None
        )
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f} 秒")
        
        # 评估模型
        if X_val is not None and y_val is not None:
            val_results = model.model.evaluate(X_val, y_val, verbose=0)
            val_loss = val_results[0]
            val_accs = val_results[1:]  # 每个输出的准确率
            avg_acc = np.mean(val_accs)
            print(f"验证集损失: {val_loss:.4f}")
            print(f"验证集平均准确率: {avg_acc:.4f}")
            for i, acc in enumerate(val_accs):
                print(f"  温区 {i} 准确率: {acc:.4f}")
        
        return model
    
    def monitor_gpu_usage(self):
        """监控GPU使用情况"""
        if not self.gpu_available:
            print("GPU不可用，无法监控")
            return
        
        print("\n=== GPU使用情况监控 ===")
        
        for i, gpu in enumerate(self.gpu_devices):
            try:
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                print(f"GPU {i} 内存使用情况:")
                print(f"  当前使用: {memory_info['current'] / 1024**3:.2f} GB")
                print(f"  峰值使用: {memory_info['peak'] / 1024**3:.2f} GB")
                print(f"  限制: {memory_info['limit'] / 1024**3:.2f} GB" if memory_info['limit'] else "  限制: 无限制")
            except Exception as e:
                print(f"无法获取GPU {i} 内存信息: {e}")
    
    def save_models(self, temp_model: TemperatureOptimizationModel, layout_model: LayoutOptimizationModel):
        """保存训练好的模型"""
        print("\n=== 保存模型 ===")
        
        # 创建模型保存目录
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'trained')
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存温度模型
        temp_model_path = os.path.join(model_dir, 'temperature_model_gpu.h5')
        temp_model.save(temp_model_path)
        print(f"温度模型已保存到: {temp_model_path}")
        
        # 保存布局模型
        layout_model_path = os.path.join(model_dir, 'layout_model_gpu.h5')
        layout_model.save(layout_model_path)
        print(f"布局模型已保存到: {layout_model_path}")
        
        print("模型训练完成并已保存")

def main():
    """主函数"""
    print("智能冰箱系统 - GPU训练示例")
    print("=" * 50)
    
    # 初始化GPU训练器
    trainer = GPUTrainer()
    
    # 生成合成数据
    (temp_X_train, temp_y_train), (layout_X_train, layout_y_train) = trainer.generate_synthetic_data(10000)
    
    # 分割验证集
    val_split = 0.2
    val_size = int(len(temp_X_train) * val_split)
    
    temp_X_val = temp_X_train[-val_size:]
    temp_y_val = temp_y_train[-val_size:]
    temp_X_train = temp_X_train[:-val_size]
    temp_y_train = temp_y_train[:-val_size]
    
    # 布局数据验证集分割
    layout_X_val = layout_X_train[-val_size:]
    layout_y_val = [y[-val_size:] for y in layout_y_train]
    layout_X_train = layout_X_train[:-val_size]
    layout_y_train = [y[:-val_size] for y in layout_y_train]
    
    print(f"训练集大小: {len(temp_X_train)}")
    print(f"验证集大小: {len(temp_X_val)}")
    
    # 训练温度优化模型
    temp_model = trainer.train_temperature_model(temp_X_train, temp_y_train, temp_X_val, temp_y_val)
    
    # 训练布局优化模型
    layout_model = trainer.train_layout_model(layout_X_train, layout_y_train, layout_X_val, layout_y_val)
    
    # 监控GPU使用情况
    trainer.monitor_gpu_usage()
    
    # 保存模型
    trainer.save_models(temp_model, layout_model)
    
    print("\n=== 训练完成 ===")
    print("使用TensorBoard查看训练详情:")
    print("tensorboard --logdir=./logs")
    print("\n模型已保存，可以在智能冰箱系统中使用")

if __name__ == "__main__":
    main()