#!/usr/bin/env python3
"""
快速架构对比测试 - 仅评估模型结构，不进行完整训练
用于快速分析不同架构的参数数量和理论性能
"""

import os
import sys
import logging
import time
from pathlib import Path
import json
import pandas as pd

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from src.ai.training.optimized_food_classifier_trainer import OptimizedFoodClassifierTrainer
    from src.ai.training.data_preprocessor import DataPreprocessor
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_architecture_specs(num_classes=6):
    """测试不同架构的规格参数"""
    logger = logging.getLogger(__name__)
    
    # 要测试的架构
    architectures = [
        'mobilenet_v2',         # 当前使用
        'mobilenet_v2_lite',    # 轻量化版本
        'efficientnet_b0',      # 高效架构
        'resnet50',             # 经典架构
    ]
    
    results = []
    
    print("🔍 智能冰箱AI模型架构规格对比")
    print("="*60)
    
    for arch in architectures:
        print(f"\n📋 测试架构: {arch}")
        
        try:
            # 创建训练器并构建模型
            trainer = OptimizedFoodClassifierTrainer(num_classes, arch)
            
            start_time = time.time()
            model = trainer.build_model(input_shape=(224, 224, 3))
            build_time = time.time() - start_time
            
            if model is None:
                print(f"   ❌ 模型构建失败")
                continue
            
            # 获取模型规格
            total_params = model.count_params()
            
            # 计算可训练参数
            import tensorflow as tf
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            
            # 估算模型大小（MB）
            # 假设每个参数使用float32，4字节
            estimated_size_mb = total_params * 4 / (1024 * 1024)
            
            # 估算TFLite大小（通常比H5小）
            estimated_tflite_mb = estimated_size_mb * 0.25  # 经验估算
            
            # 计算FLOPS（简化估算）
            # 对于分类任务，主要计算量在backbone
            if 'mobilenet' in arch:
                # MobileNet系列FLOPS相对较低
                estimated_flops = total_params * 0.5
            elif 'efficientnet' in arch:
                # EfficientNet在参数和FLOPS间平衡较好
                estimated_flops = total_params * 0.8
            elif 'resnet' in arch:
                # ResNet通常FLOPS较高
                estimated_flops = total_params * 1.2
            else:
                estimated_flops = total_params * 1.0
            
            # 预估推理速度等级（1-5，5最快）
            if total_params < 3000000:
                speed_rating = 5
            elif total_params < 5000000:
                speed_rating = 4  
            elif total_params < 10000000:
                speed_rating = 3
            elif total_params < 20000000:
                speed_rating = 2
            else:
                speed_rating = 1
            
            # 预估准确率等级（基于架构特性，1-5，5最高）
            if 'efficientnet' in arch:
                accuracy_rating = 5
            elif 'resnet50' in arch:
                accuracy_rating = 4
            elif 'mobilenet_v2' == arch:
                accuracy_rating = 4
            elif 'mobilenet_v2_lite' in arch:
                accuracy_rating = 3
            else:
                accuracy_rating = 3
            
            result = {
                'architecture': arch,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'estimated_h5_size_mb': estimated_size_mb,
                'estimated_tflite_size_mb': estimated_tflite_mb,
                'estimated_flops_m': estimated_flops / 1000000,  # 百万FLOPS
                'build_time_sec': build_time,
                'speed_rating': speed_rating,
                'accuracy_rating': accuracy_rating,
                'efficiency_score': (accuracy_rating * speed_rating) / (total_params / 1000000)  # 综合效率分数
            }
            
            results.append(result)
            
            print(f"   ✅ 构建成功")
            print(f"      参数数量: {total_params:,}")
            print(f"      预估大小: H5={estimated_size_mb:.1f}MB, TFLite={estimated_tflite_mb:.1f}MB")
            print(f"      速度评级: {speed_rating}/5")
            print(f"      精度评级: {accuracy_rating}/5")
            print(f"      效率分数: {result['efficiency_score']:.2f}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            continue
    
    return results

def generate_comparison_report(results, output_dir="ai_model/quick_architecture_analysis"):
    """生成对比报告"""
    logger = logging.getLogger(__name__)
    
    if not results:
        print("❌ 没有测试结果")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("📊 架构对比分析结果")
    print(f"{'='*60}")
    
    # 按不同指标排序显示
    print("\n🏆 按综合效率排序:")
    df_sorted = df.sort_values('efficiency_score', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        print(f"  {i+1}. {row['architecture']:<20} 效率分数: {row['efficiency_score']:.2f}")
    
    print("\n⚡ 按速度评级排序:")
    df_speed = df.sort_values('speed_rating', ascending=False)
    for i, (_, row) in enumerate(df_speed.iterrows()):
        print(f"  {i+1}. {row['architecture']:<20} 速度: {row['speed_rating']}/5, 参数: {row['total_params']:,}")
    
    print("\n🎯 按精度评级排序:")
    df_accuracy = df.sort_values('accuracy_rating', ascending=False)
    for i, (_, row) in enumerate(df_accuracy.iterrows()):
        print(f"  {i+1}. {row['architecture']:<20} 精度: {row['accuracy_rating']}/5, 大小: {row['estimated_tflite_size_mb']:.1f}MB")
    
    print("\n📱 按模型大小排序:")
    df_size = df.sort_values('estimated_tflite_size_mb', ascending=True)
    for i, (_, row) in enumerate(df_size.iterrows()):
        print(f"  {i+1}. {row['architecture']:<20} TFLite: {row['estimated_tflite_size_mb']:.1f}MB, 参数: {row['total_params']:,}")
    
    # 推荐方案
    best_efficiency = df_sorted.iloc[0]
    smallest_model = df_size.iloc[0]
    fastest_model = df_speed.iloc[0]
    most_accurate = df_accuracy.iloc[0]
    
    print(f"\n🎯 推荐方案:")
    print(f"   💡 综合最优: {best_efficiency['architecture']} (效率分数: {best_efficiency['efficiency_score']:.2f})")
    print(f"   📱 最小模型: {smallest_model['architecture']} (TFLite: {smallest_model['estimated_tflite_size_mb']:.1f}MB)")
    print(f"   ⚡ 最快速度: {fastest_model['architecture']} (速度: {fastest_model['speed_rating']}/5)")
    print(f"   🏆 最高精度: {most_accurate['architecture']} (精度: {most_accurate['accuracy_rating']}/5)")
    
    # 针对智能冰箱的具体建议
    print(f"\n🥶 智能冰箱部署建议:")
    
    # 根据综合评分给出建议
    if best_efficiency['architecture'] in ['mobilenet_v2', 'mobilenet_v2_lite']:
        print(f"   ✅ 推荐使用: {best_efficiency['architecture']}")
        print(f"      理由: 在嵌入式设备上有最佳的性能平衡")
        print(f"      优势: 参数少、速度快、TFLite优化好")
        
    elif best_efficiency['architecture'] == 'efficientnet_b0':
        print(f"   🎯 推荐使用: {best_efficiency['architecture']}")
        print(f"      理由: 最高的参数效率和准确率")
        print(f"      适合: 计算资源相对充足的智能冰箱")
        
    else:
        print(f"   ⚖️ 建议对比测试: {best_efficiency['architecture']} vs mobilenet_v2")
        print(f"      理由: 需要在性能和资源间找到最佳平衡")
    
    # 保存详细结果
    csv_path = output_path / "architecture_specs_comparison.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 转换numpy类型为Python原生类型以便JSON序列化
    for result in results:
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy类型
                result[key] = value.item()
    
    json_path = output_path / "architecture_specs.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成markdown报告
    report_content = generate_markdown_spec_report(df)
    report_path = output_path / "architecture_specs_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📁 结果已保存到:")
    print(f"   📊 CSV: {csv_path}")
    print(f"   📝 报告: {report_path}")
    print(f"   🔧 JSON: {json_path}")

def generate_markdown_spec_report(df):
    """生成Markdown规格报告"""
    
    best_efficiency = df.loc[df['efficiency_score'].idxmax()]
    smallest_model = df.loc[df['estimated_tflite_size_mb'].idxmin()]
    fastest_model = df.loc[df['speed_rating'].idxmax()]
    
    report = f"""# 智能冰箱AI模型架构规格对比报告

## 📊 测试概览
- **测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **测试架构数**: {len(df)}
- **测试类型**: 架构规格分析（无完整训练）

## 🎯 关键指标对比

| 架构 | 参数数量 | TFLite大小(MB) | 速度评级 | 精度评级 | 效率分数 |
|------|----------|---------------|----------|----------|----------|
"""
    
    for _, row in df.iterrows():
        report += f"| {row['architecture']} | {row['total_params']:,} | {row['estimated_tflite_size_mb']:.1f} | {row['speed_rating']}/5 | {row['accuracy_rating']}/5 | {row['efficiency_score']:.2f} |\n"
    
    report += f"""

## 🏆 最佳选择

### 💡 综合最优架构
- **推荐**: {best_efficiency['architecture']}
- **效率分数**: {best_efficiency['efficiency_score']:.2f}
- **参数数量**: {best_efficiency['total_params']:,}
- **TFLite大小**: {best_efficiency['estimated_tflite_size_mb']:.1f} MB
- **速度评级**: {best_efficiency['speed_rating']}/5
- **精度评级**: {best_efficiency['accuracy_rating']}/5

### 📱 最小模型
- **架构**: {smallest_model['architecture']}
- **TFLite大小**: {smallest_model['estimated_tflite_size_mb']:.1f} MB
- **参数数量**: {smallest_model['total_params']:,}

### ⚡ 最快速度
- **架构**: {fastest_model['architecture']}
- **速度评级**: {fastest_model['speed_rating']}/5
- **参数数量**: {fastest_model['total_params']:,}

## 🥶 智能冰箱部署建议

### 场景1: 资源受限的嵌入式设备
- **推荐**: mobilenet_v2_lite
- **理由**: 最小的模型体积和参数数量
- **预期性能**: 中等精度，最快推理速度

### 场景2: 中等性能的智能设备
- **推荐**: mobilenet_v2  
- **理由**: 平衡的性能和资源消耗
- **预期性能**: 良好精度，快速推理

### 场景3: 高性能智能冰箱
- **推荐**: efficientnet_b0
- **理由**: 最佳的参数效率和准确率
- **预期性能**: 高精度，中等推理速度

## 📈 优化建议

### 当前架构(mobilenet_v2)的改进空间:
1. **头部网络优化**: 多尺度特征融合
2. **注意力机制**: 添加SE模块提升精度
3. **渐进式训练**: 提升最终性能

### 架构升级路径:
1. **短期**: 优化当前MobileNetV2架构
2. **中期**: 测试EfficientNetB0的实际性能
3. **长期**: 考虑神经架构搜索(NAS)

## 🔬 技术分析

### MobileNetV2 vs EfficientNetB0
- **MobileNetV2**: 专为移动设备优化，推理速度快
- **EfficientNetB0**: 参数效率高，准确率通常更好
- **选择**: 取决于对精度和速度的具体要求

### 模型压缩潜力
- **量化**: 可减小50-75%的模型大小
- **剪枝**: 可减少20-40%的参数
- **知识蒸馏**: 在保持精度的同时显著减小模型

---
*注: 本报告基于理论分析和经验估算，实际性能需要通过完整训练验证*
"""
    
    return report

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 开始快速架构规格分析...")
    
    # 测试架构规格
    results = test_architecture_specs(num_classes=6)
    
    if results:
        # 生成对比报告
        generate_comparison_report(results)
        print(f"\n✅ 架构规格分析完成！")
    else:
        print(f"\n❌ 架构规格分析失败！")

if __name__ == "__main__":
    main()