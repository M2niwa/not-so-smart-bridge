import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
import os

# 尝试导入TensorFlow，如果失败则使用模拟模式
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow不可用，将使用模拟模式")

class FoodLayoutInference:
    """食物布局优化推理器 - 在边缘设备上运行轻量级模型"""
    
    def __init__(self, model_path: str = 'models/exported/exported_model.tflite'):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.metadata = None
        
        # 加载模型和元数据
        self._load_model()
    
    def _load_model(self):
        """加载TensorFlow Lite模型和元数据"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("TensorFlow不可用，直接使用模拟模式")
            self._create_mock_mode()
            return
        
        try:
            # 加载TFLite模型
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # 获取输入输出详情
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # 加载元数据
            metadata_path = self.model_path.replace('.tflite', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                # 如果元数据不存在，使用默认值
                self.metadata = {
                    'food_features': ['optimal_temp', 'expiry_days', 'category_meat', 'category_vegetable', 'category_fruit', 'category_dairy'],
                    'zone_features': ['zone_id', 'zone_row', 'zone_col', 'adjacent_count', 'is_corner', 'is_edge'],
                    'output_features': ['layout_score', 'position_quality']
                }
            
            self.logger.info(f"模型加载成功: {self.model_path}")
            self.logger.info(f"输入形状: {[detail['shape'] for detail in self.input_details]}")
            self.logger.info(f"输出形状: {[detail['shape'] for detail in self.output_details]}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            # 如果模型文件不存在，创建模拟模式
            self._create_mock_mode()
    
    def _create_mock_mode(self):
        """创建模拟模式，当模型文件不存在时使用基于规则的推理"""
        self.logger.warning("模型文件不存在，使用模拟模式")
        self.mock_mode = True
        
        # 设置默认元数据
        self.metadata = {
            'food_features': ['optimal_temp', 'expiry_days', 'category_meat', 'category_vegetable', 'category_fruit', 'category_dairy'],
            'zone_features': ['zone_id', 'zone_row', 'zone_col', 'adjacent_count', 'is_corner', 'is_edge'],
            'output_features': ['layout_score', 'position_quality']
        }
    
    def _encode_food_features(self, food_info: Dict) -> np.ndarray:
        """编码食物特征"""
        features = []
        
        # 数值特征
        features.append(food_info.get('optimal_temp', 0.0))
        features.append(food_info.get('expiry_days', 7))
        
        # 类别特征one-hot编码
        category = food_info.get('category', 'vegetable')
        category_map = {'meat': [1, 0, 0, 0], 'vegetable': [0, 1, 0, 0], 
                       'fruit': [0, 0, 1, 0], 'dairy': [0, 0, 0, 1]}
        features.extend(category_map.get(category, [0, 1, 0, 0]))  # 默认为蔬菜
        
        return np.array(features, dtype=np.float32)
    
    def _encode_zone_features(self, zone_id: int) -> np.ndarray:
        """编码温区特征"""
        # 3行4列布局
        zone_row = zone_id // 4
        zone_col = zone_id % 4
        
        # 计算相邻温区数量
        adjacent_count = 0
        if zone_row > 0: adjacent_count += 1
        if zone_row < 2: adjacent_count += 1
        if zone_col > 0: adjacent_count += 1
        if zone_col < 3: adjacent_count += 1
        
        # 判断是否为角落或边缘
        is_corner = int((zone_row in [0, 2]) and (zone_col in [0, 3]))
        is_edge = int((zone_row in [0, 2]) or (zone_col in [0, 3])) and not is_corner
        
        return np.array([zone_id, zone_row, zone_col, adjacent_count, is_corner, is_edge], dtype=np.float32)
    
    def predict_layout_score(self, food_info: Dict, zone_id: int) -> Tuple[float, float]:
        """预测食物在指定温区的布局得分"""
        if not TENSORFLOW_AVAILABLE or (hasattr(self, 'mock_mode') and self.mock_mode):
            # 模拟模式：使用基于规则的得分
            return self._mock_predict(food_info, zone_id)
        
        # 编码输入特征
        food_features = self._encode_food_features(food_info)
        zone_features = self._encode_zone_features(zone_id)
        
        # 调整输入形状以匹配模型期望
        food_features = np.expand_dims(food_features, axis=0)
        zone_features = np.expand_dims(zone_features, axis=0)
        
        # 设置输入张量
        self.interpreter.set_tensor(self.input_details[0]['index'], food_features)
        self.interpreter.set_tensor(self.input_details[1]['index'], zone_features)
        
        # 运行推理
        self.interpreter.invoke()
        
        # 获取输出
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        layout_score = float(output[0][0])
        position_quality = float(output[0][1])
        
        return layout_score, position_quality
    
    def _mock_predict(self, food_info: Dict, zone_id: int) -> Tuple[float, float]:
        """模拟预测函数，基于规则计算得分"""
        optimal_temp = food_info.get('optimal_temp', 0.0)
        zone_row = zone_id // 4
        
        # 温度梯度得分
        temp_gradient_score = max(0, 10 - abs(optimal_temp + zone_row * 2))
        
        # 位置质量得分
        zone_col = zone_id % 4
        is_corner = (zone_row in [0, 2]) and (zone_col in [0, 3])
        is_edge = ((zone_row in [0, 2]) or (zone_col in [0, 3])) and not is_corner
        
        if is_corner:
            position_quality = 8.0
        elif is_edge:
            position_quality = 6.5
        else:
            position_quality = 5.0
        
        layout_score = temp_gradient_score + position_quality
        
        return layout_score, position_quality
    
    def recommend_optimal_layout(self, foods: List[Dict]) -> Dict[int, Dict]:
        """为多个食物推荐最优布局"""
        if not foods:
            return {}
        
        # 生成所有可能的布局组合
        available_zones = list(range(12))
        best_layout = {}
        best_score = -float('inf')
        
        # 限制搜索空间以提高效率
        max_foods = min(len(foods), len(available_zones))
        
        # 尝试不同的排列组合
        import itertools
        for zone_permutation in itertools.permutations(available_zones[:max_foods]):
            current_layout = {}
            total_score = 0.0
            
            for i, zone_id in enumerate(zone_permutation):
                food_info = foods[i]
                layout_score, position_quality = self.predict_layout_score(food_info, zone_id)
                
                current_layout[zone_id] = {
                    'food_info': food_info,
                    'layout_score': layout_score,
                    'position_quality': position_quality
                }
                total_score += layout_score
            
            if total_score > best_score:
                best_score = total_score
                best_layout = current_layout
        
        return best_layout
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return {
                'model_path': self.model_path,
                'status': 'mock_mode',
                'metadata': self.metadata
            }
        
        return {
            'model_path': self.model_path,
            'status': 'loaded',
            'input_details': self.input_details,
            'output_details': self.output_details,
            'metadata': self.metadata
        }
    
    def benchmark_inference(self, num_runs: int = 100) -> Dict:
        """基准测试推理性能"""
        import time
        
        # 测试数据
        test_food = {'optimal_temp': 2.0, 'expiry_days': 7, 'category': 'vegetable'}
        test_zone = 5
        
        start_time = time.time()
        
        for _ in range(num_runs):
            self.predict_layout_score(test_food, test_zone)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'total_time': total_time,
            'num_runs': num_runs,
            'average_time': total_time / num_runs,
            'inferences_per_second': num_runs / total_time
        }