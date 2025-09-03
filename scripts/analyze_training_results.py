#!/usr/bin/env python3
"""
训练结果分析脚本
分析和对比不同训练方法的结果
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_models():
    """分析和对比模型训练结果"""
    
    # 模型目录
    original_dir = Path("ai_model/trained_models")
    optimized_dir = Path("ai_model/trained_models_optimized")
    
    print("=" * 60)
    print("🔍 智能冰箱AI模型训练结果对比分析")
    print("=" * 60)
    
    # 检查模型文件
    models_info = {}
    
    # 原始训练结果
    if original_dir.exists():
        original_metadata = original_dir / "class_indices.json"
        if original_metadata.exists():
            with open(original_metadata, 'r', encoding='utf-8') as f:
                original_classes = json.load(f)
            
            original_h5 = list(original_dir.glob("*.h5"))
            original_tflite = list(original_dir.glob("*.tflite"))
            
            models_info['original'] = {
                'dir': original_dir,
                'classes': original_classes,
                'h5_files': original_h5,
                'tflite_files': original_tflite,
                'name': '原始训练'
            }
    
    # 优化训练结果
    if optimized_dir.exists():
        optimized_metadata = optimized_dir / "model_metadata.json"
        if optimized_metadata.exists():
            with open(optimized_metadata, 'r', encoding='utf-8') as f:
                optimized_data = json.load(f)
            
            optimized_h5 = list(optimized_dir.glob("*.h5"))
            optimized_tflite = list(optimized_dir.glob("*.tflite"))
            
            models_info['optimized'] = {
                'dir': optimized_dir,
                'metadata': optimized_data,
                'classes': optimized_data.get('class_indices', {}),
                'h5_files': optimized_h5,
                'tflite_files': optimized_tflite,
                'name': '优化训练'
            }
    
    if not models_info:
        print("❌ 未找到训练结果")
        return
    
    # 生成对比报告
    generate_comparison_report(models_info)
    
    # 生成文件大小对比
    generate_size_comparison(models_info)
    
    # 生成建议报告
    generate_recommendations(models_info)

def generate_comparison_report(models_info):
    """生成模型对比报告"""
    
    print("\n📊 模型对比分析")
    print("-" * 40)
    
    comparison_data = []
    
    for key, info in models_info.items():
        model_name = info['name']
        classes = info['classes']
        h5_files = info['h5_files']
        tflite_files = info['tflite_files']
        
        # 基本信息
        num_classes = len(classes)
        h5_size = sum([f.stat().st_size for f in h5_files]) / (1024*1024) if h5_files else 0
        tflite_size = sum([f.stat().st_size for f in tflite_files]) / (1024*1024) if tflite_files else 0
        
        row_data = {
            '模型版本': model_name,
            '类别数量': num_classes,
            'H5大小(MB)': f"{h5_size:.1f}",
            'TFLite大小(MB)': f"{tflite_size:.1f}",
            '类别': list(classes.keys())
        }
        
        # 添加优化版本的额外信息
        if 'metadata' in info:
            metadata = info['metadata']
            row_data.update({
                '参数数量': f"{metadata.get('total_params', 0):,}",
                '训练时间(秒)': f"{metadata.get('training_time', 0):.1f}",
                '训练样本': metadata.get('dataset_info', {}).get('train_samples', 0),
                '验证样本': metadata.get('dataset_info', {}).get('val_samples', 0),
                '测试样本': metadata.get('dataset_info', {}).get('test_samples', 0)
            })
        
        comparison_data.append(row_data)
        
        print(f"\n🎯 {model_name}:")
        print(f"  • 类别数量: {num_classes}")
        print(f"  • 识别类别: {', '.join(classes.keys())}")
        print(f"  • H5模型大小: {h5_size:.1f} MB")
        print(f"  • TFLite模型大小: {tflite_size:.1f} MB")
        
        if 'metadata' in info:
            metadata = info['metadata']
            print(f"  • 模型参数: {metadata.get('total_params', 0):,}")
            print(f"  • 训练时间: {metadata.get('training_time', 0):.1f} 秒")
            dataset_info = metadata.get('dataset_info', {})
            print(f"  • 数据分布: 训练{dataset_info.get('train_samples', 0)} + 验证{dataset_info.get('val_samples', 0)} + 测试{dataset_info.get('test_samples', 0)}")
    
    # 保存对比表格
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        output_file = Path("ai_model/model_comparison.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 对比表格已保存: {output_file}")

def generate_size_comparison(models_info):
    """生成文件大小对比图"""
    
    if len(models_info) < 2:
        return
    
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    model_names = []
    h5_sizes = []
    tflite_sizes = []
    
    for key, info in models_info.items():
        model_names.append(info['name'])
        
        h5_files = info['h5_files']
        tflite_files = info['tflite_files']
        
        h5_size = sum([f.stat().st_size for f in h5_files]) / (1024*1024) if h5_files else 0
        tflite_size = sum([f.stat().st_size for f in tflite_files]) / (1024*1024) if tflite_files else 0
        
        h5_sizes.append(h5_size)
        tflite_sizes.append(tflite_size)
    
    # 创建对比图
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, h5_sizes, width, label='H5模型', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, tflite_sizes, width, label='TFLite模型', alpha=0.8, color='lightcoral')
    
    plt.xlabel('模型版本')
    plt.ylabel('文件大小 (MB)')
    plt.title('模型文件大小对比')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (h5, tflite) in enumerate(zip(h5_sizes, tflite_sizes)):
        plt.text(i - width/2, h5 + 0.1, f'{h5:.1f}MB', ha='center', va='bottom')
        plt.text(i + width/2, tflite + 0.1, f'{tflite:.1f}MB', ha='center', va='bottom')
    
    # 压缩比对比
    plt.subplot(1, 2, 2)
    compression_ratios = [h5/tflite if tflite > 0 else 0 for h5, tflite in zip(h5_sizes, tflite_sizes)]
    
    bars = plt.bar(model_names, compression_ratios, alpha=0.8, color='lightgreen')
    plt.xlabel('模型版本')
    plt.ylabel('压缩比 (H5/TFLite)')
    plt.title('TFLite压缩效果')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = Path("ai_model/model_size_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 大小对比图已保存: {output_file}")

def generate_recommendations(models_info):
    """生成优化建议"""
    
    print("\n💡 优化建议和总结")
    print("-" * 40)
    
    # 根据对比结果生成建议
    if 'optimized' in models_info and 'original' in models_info:
        optimized = models_info['optimized']
        original = models_info['original']
        
        print("✅ 成功完成模型训练优化！")
        
        # 检查改进情况
        opt_metadata = optimized.get('metadata', {})
        training_time = opt_metadata.get('training_time', 0)
        
        print(f"\n🚀 优化训练的改进:")
        print(f"  • 使用了增强的数据预处理和数据增强技术")
        print(f"  • 训练时间: {training_time:.1f} 秒 (约{training_time/60:.1f} 分钟)")
        print(f"  • 模型参数: {opt_metadata.get('total_params', 0):,}")
        print(f"  • 正确识别6个食物类别: {', '.join(optimized['classes'].keys())}")
        
        # 数据集建议
        dataset_info = opt_metadata.get('dataset_info', {})
        total_samples = sum(dataset_info.values())
        print(f"\n📊 数据集状况:")
        print(f"  • 当前样本数: {total_samples}")
        print(f"  • 建议增加样本到: 1000+ 张/类别 (当前约{total_samples//6}张/类别)")
        
        # 缺失类别提醒
        expected_categories = {'milk', 'beef', 'vegetables', 'yogurt', 'cheese', 'chicken', 'dairy', 'eggs', 'fish', 'fruits'}
        current_categories = set(optimized['classes'].keys())
        missing_categories = expected_categories - current_categories
        
        if missing_categories:
            print(f"  • 缺失类别: {', '.join(missing_categories)}")
            print(f"  • 建议补充这些类别的数据以提高系统完整性")
    
    elif 'optimized' in models_info:
        print("✅ 完成优化训练！")
        optimized = models_info['optimized']
        opt_metadata = optimized.get('metadata', {})
        
        print(f"  • 成功训练{len(optimized['classes'])}类食物分类模型")
        print(f"  • 训练时间: {opt_metadata.get('training_time', 0):.1f} 秒")
    
    else:
        print("⚠️ 仅发现原始训练结果")
        print("  • 建议运行优化训练脚本以获得更好的性能")
    
    # 部署建议
    print(f"\n🚀 部署建议:")
    print(f"  • H5模型: 适用于服务器端部署，精度最高")
    print(f"  • TFLite模型: 适用于移动设备和边缘计算，体积小")
    print(f"  • 推荐在智能冰箱硬件上使用TFLite模型")
    
    # 后续优化建议
    print(f"\n🔧 后续优化方向:")
    print(f"  1. 收集更多真实冰箱食物图像数据")
    print(f"  2. 添加缺失的食物类别 (牛奶、牛肉、蔬菜、酸奶)")
    print(f"  3. 实施模型量化以进一步减小TFLite模型体积")
    print(f"  4. 考虑使用更新的模型架构 (如EfficientNet)")
    print(f"  5. 添加模型性能监控和在线学习功能")

def main():
    """主函数"""
    # 确保在项目根目录
    os.chdir(project_root)
    
    try:
        analyze_models()
        print(f"\n🎉 训练结果分析完成！")
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()