#!/usr/bin/env python3
"""
食物布局优化模型训练脚本
训练适合边缘设备的轻量级神经网络模型
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from food_layout_model import FoodLayoutModelTrainer
from food_layout_inference import FoodLayoutInference

def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('model_training.log')
        ]
    )

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="训练食物布局优化模型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--model-dir", type=str, default="models/exported", help="模型保存目录")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    parser.add_argument("--test-only", action="store_true", help="仅测试现有模型")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 创建模型目录
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # 模型文件路径
    model_path = model_dir / "exported_model.h5"
    tflite_path = model_dir / "exported_model.tflite"
    
    if args.test_only:
        # 仅测试现有模型
        logger.info("测试现有模型...")
        if tflite_path.exists():
            inference = FoodLayoutInference(str(tflite_path))
            model_info = inference.get_model_info()
            logger.info(f"模型信息: {model_info}")
            
            # 运行基准测试
            benchmark = inference.benchmark_inference(100)
            logger.info(f"基准测试结果: {benchmark}")
        else:
            logger.error(f"模型文件不存在: {tflite_path}")
        return
    
    # 创建训练器
    logger.info("初始化模型训练器...")
    trainer = FoodLayoutModelTrainer()
    
    # 创建神经网络模型
    logger.info("创建神经网络模型...")
    model = trainer.create_model()
    if model is not None:
        model.summary()
    else:
        logger.info("模拟模式：无真实模型结构")
    
    # 训练模型
    logger.info(f"开始训练模型 - 轮数: {args.epochs}, 批次大小: {args.batch_size}")
    training_info = trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)
    logger.info(f"训练完成: {training_info}")
    
    # 保存模型
    logger.info("保存模型...")
    saved_model_path, saved_tflite_path, metadata_path = trainer.save_model(str(model_path))
    logger.info(f"模型已保存:")
    logger.info(f"  - 完整模型: {saved_model_path}")
    logger.info(f"  - TFLite模型: {saved_tflite_path}")
    logger.info(f"  - 元数据: {metadata_path}")
    
    # 测试导出的模型
    logger.info("测试导出的模型...")
    try:
        inference = FoodLayoutInference(str(saved_tflite_path))
        model_info = inference.get_model_info()
        logger.info(f"导出模型信息: {model_info}")
        
        # 运行基准测试
        benchmark = inference.benchmark_inference(100)
        logger.info(f"推理性能: {benchmark['inferences_per_second']:.1f} 推理/秒")
        
        # 测试布局推荐
        test_foods = [
            {'optimal_temp': -2.0, 'expiry_days': 3, 'category': 'meat'},
            {'optimal_temp': 2.0, 'expiry_days': 7, 'category': 'vegetable'},
            {'optimal_temp': 6.0, 'expiry_days': 14, 'category': 'vegetable'}
        ]
        
        optimal_layout = inference.recommend_optimal_layout(test_foods)
        logger.info("测试布局推荐结果:")
        for zone_id, layout_info in optimal_layout.items():
            food_info = layout_info['food_info']
            score = layout_info['layout_score']
            logger.info(f"  温区{zone_id}: {food_info['category']} (温度: {food_info['optimal_temp']}°C), 得分: {score:.2f}")
        
        logger.info("模型测试完成！")
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        
    logger.info("训练流程完成！")

if __name__ == "__main__":
    main()