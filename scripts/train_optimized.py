#!/usr/bin/env python3
"""
优化的智能冰箱AI模型训练脚本
包含数据增强、早停、学习率调度、模型评估等优化功能
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.ai.training.data_preprocessor import DataPreprocessor
from src.ai.training.food_classifier_trainer import FoodClassifierTrainer
from src.ai.training.model_evaluator import ModelEvaluator

def setup_logging(log_level=logging.INFO, log_file=None):
    """设置优化的日志系统"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def check_data_quality(data_dir):
    """检查数据集质量并提供建议"""
    logger = logging.getLogger(__name__)
    
    # 检查数据集结构
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    test_dir = Path(data_dir) / 'test'
    
    if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
        logger.warning("数据集结构不完整，缺少train/val/test目录")
        return False
    
    # 统计各类别图像数量
    categories = {}
    for split in ['train', 'val', 'test']:
        split_dir = Path(data_dir) / split
        categories[split] = {}
        
        for cat_dir in split_dir.iterdir():
            if cat_dir.is_dir():
                img_count = len(list(cat_dir.glob('*')))
                categories[split][cat_dir.name] = img_count
    
    # 检查类别一致性
    train_cats = set(categories['train'].keys())
    val_cats = set(categories['val'].keys())
    test_cats = set(categories['test'].keys())
    
    if not (train_cats == val_cats == test_cats):
        logger.warning("训练集、验证集、测试集的类别不一致")
        return False
    
    # 输出数据集统计
    logger.info("=== 数据集质量检查 ===")
    total_images = 0
    for category in train_cats:
        train_count = categories['train'][category]
        val_count = categories['val'][category]
        test_count = categories['test'][category]
        cat_total = train_count + val_count + test_count
        total_images += cat_total
        
        logger.info(f"{category}: 训练{train_count} + 验证{val_count} + 测试{test_count} = {cat_total}")
    
    logger.info(f"总图像数: {total_images}")
    logger.info(f"类别数: {len(train_cats)}")
    
    # 数据平衡性检查
    train_counts = list(categories['train'].values())
    if max(train_counts) / min(train_counts) > 3:
        logger.warning("数据分布不平衡，建议进行类别平衡处理")
    
    return True

def create_enhanced_data_generators(data_dir, batch_size=32, image_size=(224, 224)):
    """创建增强的数据生成器"""
    logger = logging.getLogger(__name__)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # 训练数据生成器（强数据增强）
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,          # 增加旋转角度
        width_shift_range=0.3,      # 增加平移范围
        height_shift_range=0.3,
        shear_range=0.3,            # 增加剪切变换
        zoom_range=0.3,             # 增加缩放范围
        horizontal_flip=True,
        vertical_flip=False,        # 食物图像通常不垂直翻转
        fill_mode='nearest',
        brightness_range=[0.8, 1.2], # 亮度调整
        channel_shift_range=20.0     # 颜色通道偏移
    )
    
    # 验证和测试数据生成器（仅标准化）
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # 创建生成器
    train_generator = train_datagen.flow_from_directory(
        str(Path(data_dir) / 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        str(Path(data_dir) / 'val'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        str(Path(data_dir) / 'test'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"数据生成器创建完成:")
    logger.info(f"  训练集: {train_generator.samples} 样本")
    logger.info(f"  验证集: {val_generator.samples} 样本")
    logger.info(f"  测试集: {test_generator.samples} 样本")
    logger.info(f"  类别数: {len(train_generator.class_indices)}")
    
    return train_generator, val_generator, test_generator

def train_optimized_model(args):
    """优化的模型训练流程"""
    logger = logging.getLogger(__name__)
    
    # 1. 数据质量检查
    logger.info("=== 步骤1: 数据质量检查 ===")
    if not check_data_quality(args.data_dir):
        raise ValueError("数据质量检查失败")
    
    # 2. 创建数据生成器
    logger.info("=== 步骤2: 创建数据生成器 ===")
    train_gen, val_gen, test_gen = create_enhanced_data_generators(
        args.data_dir, 
        batch_size=args.batch_size, 
        image_size=(args.image_size, args.image_size)
    )
    
    # 3. 构建和训练模型
    logger.info("=== 步骤3: 模型构建和训练 ===")
    num_classes = len(train_gen.class_indices)
    trainer = FoodClassifierTrainer(num_classes=num_classes, model_type=args.model_type)
    
    # 构建模型
    model = trainer.build_model(input_shape=(args.image_size, args.image_size, 3))
    if model is None:
        raise ValueError("模型构建失败")
    
    logger.info(f"模型架构: {args.model_type}")
    logger.info(f"参数数量: {model.count_params():,}")
    
    # 训练模型
    start_time = time.time()
    train_success = trainer.train(
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    training_time = time.time() - start_time
    
    if not train_success:
        raise ValueError("模型训练失败")
    
    logger.info(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 4. 微调（可选）
    if args.fine_tune:
        logger.info("=== 步骤4: 模型微调 ===")
        fine_tune_success = trainer.fine_tune(
            train_generator=train_gen,
            validation_generator=val_gen,
            epochs=args.fine_tune_epochs
        )
        
        if not fine_tune_success:
            logger.warning("模型微调失败")
    
    # 5. 保存模型
    logger.info("=== 步骤5: 保存模型 ===")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"food_classifier_{args.model_type}_optimized.h5"
    trainer.save_model(str(model_path))
    
    # 导出TFLite模型
    tflite_path = output_dir / f"food_classifier_{args.model_type}_optimized.tflite"
    trainer.export_tflite(str(model_path), str(tflite_path))
    
    # 6. 保存类别映射和元数据
    class_indices = train_gen.class_indices
    metadata = {
        'class_indices': class_indices,
        'num_classes': num_classes,
        'model_type': args.model_type,
        'image_size': args.image_size,
        'training_time': training_time,
        'total_params': int(model.count_params()),
        'dataset_info': {
            'train_samples': train_gen.samples,
            'val_samples': val_gen.samples,
            'test_samples': test_gen.samples
        }
    }
    
    with open(output_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 7. 绘制训练历史
    if args.plot_history:
        logger.info("=== 步骤6: 绘制训练历史 ===")
        history_path = output_dir / f"training_history_{args.model_type}_optimized.png"
        trainer.plot_training_history(str(history_path))
    
    # 8. 模型评估
    if args.evaluate:
        logger.info("=== 步骤7: 模型评估 ===")
        evaluate_trained_model(trainer, test_gen, output_dir, class_indices)
    
    return model_path, class_indices, metadata

def evaluate_trained_model(trainer, test_generator, output_dir, class_indices):
    """详细的模型评估"""
    logger = logging.getLogger(__name__)
    
    # 基本评估
    metrics = trainer.evaluate(test_generator)
    
    # 预测所有测试数据
    test_generator.reset()
    predictions = trainer.model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # 分类报告
    class_names = list(class_indices.keys())
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 保存评估结果
    eval_results = {
        'basic_metrics': metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出评估摘要
    logger.info("=== 模型评估结果 ===")
    logger.info(f"测试准确率: {metrics['accuracy']:.4f}")
    logger.info(f"测试损失: {metrics['loss']:.4f}")
    if 'top_k_accuracy' in metrics and metrics['top_k_accuracy']:
        logger.info(f"Top-K准确率: {metrics['top_k_accuracy']:.4f}")
    
    logger.info("\n各类别性能:")
    for class_name in class_names:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1_score = report[class_name]['f1-score']
            logger.info(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}")

def generate_training_summary(output_dir, metadata):
    """生成训练总结报告"""
    logger = logging.getLogger(__name__)
    
    summary = f"""
# 智能冰箱AI模型训练总结报告

## 模型信息
- **模型类型**: {metadata['model_type']}
- **图像尺寸**: {metadata['image_size']}x{metadata['image_size']}
- **类别数量**: {metadata['num_classes']}
- **参数数量**: {metadata['total_params']:,}

## 数据集信息
- **训练样本**: {metadata['dataset_info']['train_samples']}
- **验证样本**: {metadata['dataset_info']['val_samples']}
- **测试样本**: {metadata['dataset_info']['test_samples']}
- **总样本数**: {sum(metadata['dataset_info'].values())}

## 类别映射
"""
    
    for class_name, index in metadata['class_indices'].items():
        summary += f"- {index}: {class_name}\n"
    
    summary += f"""
## 训练信息
- **训练时长**: {metadata['training_time']:.2f} 秒
- **训练完成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 文件输出
- **H5模型**: food_classifier_{metadata['model_type']}_optimized.h5
- **TFLite模型**: food_classifier_{metadata['model_type']}_optimized.tflite
- **训练历史图**: training_history_{metadata['model_type']}_optimized.png
- **混淆矩阵**: confusion_matrix.png
- **评估结果**: evaluation_results.json
- **元数据**: model_metadata.json

## 使用建议
1. H5模型适用于服务器端部署
2. TFLite模型适用于移动设备和边缘计算
3. 查看训练历史图了解模型收敛情况
4. 查看混淆矩阵分析类别间的混淆情况
"""
    
    with open(output_dir / 'training_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"训练总结报告已保存: {output_dir / 'training_summary.md'}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化的智能冰箱AI模型训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True,
                       help="训练数据目录路径")
    parser.add_argument("--output_dir", type=str, default="ai_model/trained_models_optimized",
                       help="模型输出目录")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="mobilenet",
                       choices=["mobilenet", "resnet", "efficientnet"],
                       help="模型架构类型")
    parser.add_argument("--image_size", type=int, default=224,
                       help="图像输入尺寸")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=30,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--fine_tune", action="store_true",
                       help="是否进行微调")
    parser.add_argument("--fine_tune_epochs", type=int, default=10,
                       help="微调轮数")
    
    # 功能开关
    parser.add_argument("--plot_history", action="store_true", default=True,
                       help="是否绘制训练历史")
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="是否评估模型")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = Path(args.output_dir) / "training.log"
    setup_logging(getattr(logging, args.log_level), log_file)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 60)
        logger.info("开始优化的AI模型训练流程")
        logger.info("=" * 60)
        logger.info(f"训练参数: {vars(args)}")
        
        # 开始训练
        model_path, class_indices, metadata = train_optimized_model(args)
        
        # 生成训练总结
        generate_training_summary(Path(args.output_dir), metadata)
        
        logger.info("=" * 60)
        logger.info("🎉 优化训练流程完成！")
        logger.info("=" * 60)
        logger.info(f"模型保存路径: {model_path}")
        logger.info(f"输出目录: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()