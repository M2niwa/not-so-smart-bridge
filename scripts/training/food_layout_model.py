import numpy as np
import json
import logging
from typing import List, Dict, Tuple

# 尝试导入TensorFlow，如果失败则使用模拟模式
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow不可用，将使用模拟模式")

class FoodLayoutModelTrainer:
    """食物布局优化模型训练器 - 构建适合边缘设备的轻量级模型"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        # 定义食物特征
        self.food_features = [
            'optimal_temp',    # 最佳温度
            'expiry_days',     # 保质期
            'category_meat',   # 肉类
            'category_vegetable', # 蔬菜
            'category_fruit',  # 水果
            'category_dairy'   # 乳制品
        ]
        
        # 定义温区特征
        self.zone_features = [
            'zone_id',         # 温区ID
            'zone_row',        # 行位置
            'zone_col',        # 列位置
            'adjacent_count',  # 相邻温区数量
            'is_corner',       # 是否角落
            'is_edge'          # 是否边缘
        ]
        
        # 输出特征：布局得分和位置建议
        self.output_features = ['layout_score', 'position_quality']
    
    def create_model(self):
        """创建轻量级神经网络模型，适合边缘设备部署"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("TensorFlow不可用，创建模拟模型")
            self.model = None
            return None
        
        # 输入层：食物特征 + 温区特征
        food_input = keras.Input(shape=(len(self.food_features),), name='food_input')
        zone_input = keras.Input(shape=(len(self.zone_features),), name='zone_input')
        
        # 食物特征处理分支
        food_dense = layers.Dense(32, activation='relu')(food_input)
        food_dense = layers.BatchNormalization()(food_dense)
        food_dense = layers.Dropout(0.2)(food_dense)
        
        # 温区特征处理分支
        zone_dense = layers.Dense(32, activation='relu')(zone_input)
        zone_dense = layers.BatchNormalization()(zone_dense)
        zone_dense = layers.Dropout(0.2)(zone_dense)
        
        # 合并特征
        merged = layers.Concatenate()([food_dense, zone_dense])
        
        # 共享层
        dense1 = layers.Dense(64, activation='relu')(merged)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense_dense = layers.Dropout(0.2)(dense2)
        
        # 输出层
        output = layers.Dense(len(self.output_features), activation='linear', name='output')(dense2)
        
        # 创建模型
        model = keras.Model(
            inputs=[food_input, zone_input],
            outputs=output,
            name='food_layout_optimizer'
        )
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        self.logger.info(f"模型创建完成，参数数量: {model.count_params()}")
        return model
    
    def generate_synthetic_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成合成训练数据"""
        np.random.seed(42)
        
        food_data = []
        zone_data = []
        target_data = []
        
        for _ in range(num_samples):
            # 生成食物特征
            optimal_temp = np.random.uniform(-20, 10)  # 温度范围-20到10度
            expiry_days = np.random.randint(1, 30)     # 保质期1-30天
            category = np.random.randint(0, 4)         # 4个类别
            
            # one-hot编码类别
            category_vec = [0] * 4
            category_vec[category] = 1
            
            food_features = [optimal_temp, expiry_days] + category_vec
            
            # 生成温区特征
            zone_id = np.random.randint(0, 12)         # 12个温区
            zone_row = zone_id // 4                    # 3行4列布局
            zone_col = zone_id % 4
            
            # 计算相邻温区数量
            adjacent_count = 0
            if zone_row > 0: adjacent_count += 1
            if zone_row < 2: adjacent_count += 1
            if zone_col > 0: adjacent_count += 1
            if zone_col < 3: adjacent_count += 1
            
            is_corner = int((zone_row in [0, 2]) and (zone_col in [0, 3]))
            is_edge = int((zone_row in [0, 2]) or (zone_col in [0, 3])) and not is_corner
            
            zone_features = [zone_id, zone_row, zone_col, adjacent_count, is_corner, is_edge]
            
            # 计算目标值（基于规则的得分）
            # 温度梯度得分：上方温度更低得分更高
            temp_gradient_score = max(0, 10 - abs(optimal_temp + zone_row * 2))
            
            # 位置质量得分：角落和边缘位置更适合长期保存食物
            position_quality = 5.0
            if is_corner:
                position_quality = 8.0
            elif is_edge:
                position_quality = 6.5
            
            # 综合得分
            layout_score = temp_gradient_score + position_quality
            
            food_data.append(food_features)
            zone_data.append(zone_features)
            target_data.append([layout_score, position_quality])
        
        return (
            np.array(food_data, dtype=np.float32),
            np.array(zone_data, dtype=np.float32),
            np.array(target_data, dtype=np.float32)
        )
    
    def train_model(self, epochs: int = 50, batch_size: int = 32) -> Dict:
        """训练模型"""
        if self.model is None:
            self.create_model()
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("TensorFlow不可用，使用模拟训练")
            return {
                'test_loss': 0.0,
                'test_mae': 0.0,
                'epochs': epochs,
                'batch_size': batch_size,
                'model_params': 0,
                'mode': 'simulation'
            }
        
        # 生成训练数据
        self.logger.info("生成训练数据...")
        food_train, zone_train, target_train = self.generate_synthetic_data(8000)
        food_val, zone_val, target_val = self.generate_synthetic_data(2000)
        
        # 训练模型
        self.logger.info("开始训练模型...")
        history = self.model.fit(
            [food_train, zone_train],
            target_train,
            validation_data=([food_val, zone_val], target_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # 评估模型
        test_loss, test_mae = self.model.evaluate(
            [food_val, zone_val],
            target_val,
            verbose=0
        )
        
        training_info = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'epochs': epochs,
            'batch_size': batch_size,
            'model_params': self.model.count_params(),
            'mode': 'tensorflow'
        }
        
        self.logger.info(f"训练完成 - 测试损失: {test_loss:.4f}, 测试MAE: {test_mae:.4f}")
        return training_info
    
    def save_model(self, model_path: str = 'models/exported/exported_model.h5'):
        """保存训练好的模型"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("TensorFlow不可用，创建模拟模型文件")
            
            # 创建模拟模型文件（空文件）
            tflite_path = model_path.replace('.h5', '.tflite')
            
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
            
            # 创建空的TFLite文件（标记为模拟模式）
            with open(tflite_path, 'wb') as f:
                f.write(b'MOCK_MODEL')
            
            self.logger.info(f"模拟模型已保存到: {tflite_path}")
            
            # 保存模型元数据
            metadata = {
                'food_features': self.food_features,
                'zone_features': self.zone_features,
                'output_features': self.output_features,
                'model_params': 0,
                'mode': 'simulation',
                'input_shape': {
                    'food_input': [len(self.food_features)],
                    'zone_input': [len(self.zone_features)]
                },
                'output_shape': [len(self.output_features)]
            }
            
            metadata_path = model_path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"模型元数据已保存到: {metadata_path}")
            
            return model_path, tflite_path, metadata_path
        
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        # 保存完整模型
        self.model.save(model_path)
        self.logger.info(f"模型已保存到: {model_path}")
        
        # 转换为TensorFlow Lite格式（适合边缘设备）
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # 优化模型大小
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        tflite_path = model_path.replace('.h5', '.tflite')
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        self.logger.info(f"TensorFlow Lite模型已保存到: {tflite_path}")
        
        # 保存模型元数据
        metadata = {
            'food_features': self.food_features,
            'zone_features': self.zone_features,
            'output_features': self.output_features,
            'model_params': self.model.count_params(),
            'mode': 'tensorflow',
            'input_shape': {
                'food_input': self.model.input_shape[0][1:],
                'zone_input': self.model.input_shape[1][1:]
            },
            'output_shape': self.model.output_shape[1:]
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"模型元数据已保存到: {metadata_path}")
        
        return model_path, tflite_path, metadata_path

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建并训练模型
    trainer = FoodLayoutModelTrainer()
    model = trainer.create_model()
    model.summary()
    
    # 训练模型
    training_info = trainer.train_model(epochs=30, batch_size=64)
    print(f"\n训练信息: {training_info}")
    
    # 保存模型
    model_path, tflite_path, metadata_path = trainer.save_model()
    print(f"\n模型保存完成:")
    print(f"- 完整模型: {model_path}")
    print(f"- TFLite模型: {tflite_path}")
    print(f"- 元数据: {metadata_path}")