#!/usr/bin/env python3
"""
数据集质量评估脚本
用于评估导入数据集的质量和适用性
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DatasetEvaluator:
    """数据集评估器"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
        # 智能冰箱核心类别
        self.target_categories = [
            'milk', 'beef', 'vegetables', 'fruits', 'eggs',
            'cheese', 'yogurt', 'chicken', 'fish'
        ]
        
        # 图像质量标准
        self.quality_standards = {
            'min_resolution': (224, 224),
            'max_blur_threshold': 100.0,  # Laplacian方差阈值
            'min_brightness': 50,
            'max_brightness': 200,
            'min_images_per_category': 50
        }
    
    def evaluate_dataset(self):
        """全面评估数据集"""
        print("🔍 开始数据集质量评估...")
        
        # 基础统计
        stats = self.get_basic_statistics()
        
        # 图像质量分析
        quality_report = self.analyze_image_quality()
        
        # 类别分布分析
        distribution_report = self.analyze_category_distribution()
        
        # 生成评估报告
        report = self.generate_evaluation_report(stats, quality_report, distribution_report)
        
        # 保存报告
        self.save_report(report)
        
        return report
    
    def get_basic_statistics(self):
        """获取基础统计信息"""
        stats = {
            'total_images': 0,
            'categories': {},
            'datasets': {'train': 0, 'val': 0, 'test': 0},
            'file_formats': Counter(),
            'avg_file_size': 0
        }
        
        total_size = 0
        
        for dataset in ['train', 'val', 'test']:
            dataset_path = self.dataset_path / dataset
            if not dataset_path.exists():
                continue
            
            dataset_images = 0
            for category_dir in dataset_path.iterdir():
                if not category_dir.is_dir():
                    continue
                
                category_name = category_dir.name
                images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpeg"))
                
                category_count = len(images)
                dataset_images += category_count
                
                if category_name not in stats['categories']:
                    stats['categories'][category_name] = {'train': 0, 'val': 0, 'test': 0}
                
                stats['categories'][category_name][dataset] = category_count
                
                # 统计文件格式和大小
                for img_path in images:
                    stats['file_formats'][img_path.suffix.lower()] += 1
                    total_size += img_path.stat().st_size
            
            stats['datasets'][dataset] = dataset_images
        
        stats['total_images'] = sum(stats['datasets'].values())
        if stats['total_images'] > 0:
            stats['avg_file_size'] = total_size / stats['total_images'] / (1024 * 1024)  # MB
        
        return stats
    
    def analyze_image_quality(self):
        """分析图像质量"""
        quality_report = {
            'resolution_issues': [],
            'blur_issues': [],
            'brightness_issues': [],
            'total_checked': 0,
            'quality_score': 0
        }
        
        sample_size = 100  # 每个类别检查的样本数
        
        for dataset in ['train', 'val', 'test']:
            dataset_path = self.dataset_path / dataset
            if not dataset_path.exists():
                continue
            
            for category_dir in dataset_path.iterdir():
                if not category_dir.is_dir():
                    continue
                
                images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                sample_images = images[:min(sample_size, len(images))]
                
                for img_path in sample_images:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        quality_report['total_checked'] += 1
                        
                        # 检查分辨率
                        h, w = img.shape[:2]
                        if h < self.quality_standards['min_resolution'][0] or w < self.quality_standards['min_resolution'][1]:
                            quality_report['resolution_issues'].append({
                                'file': str(img_path),
                                'resolution': (w, h)
                            })
                        
                        # 检查模糊度
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if blur_score < self.quality_standards['max_blur_threshold']:
                            quality_report['blur_issues'].append({
                                'file': str(img_path),
                                'blur_score': blur_score
                            })
                        
                        # 检查亮度
                        brightness = np.mean(gray)
                        if brightness < self.quality_standards['min_brightness'] or brightness > self.quality_standards['max_brightness']:
                            quality_report['brightness_issues'].append({
                                'file': str(img_path),
                                'brightness': brightness
                            })
                        
                    except Exception as e:
                        self.logger.warning(f"无法分析图像 {img_path}: {e}")
        
        # 计算质量评分
        total_issues = len(quality_report['resolution_issues']) + len(quality_report['blur_issues']) + len(quality_report['brightness_issues'])
        if quality_report['total_checked'] > 0:
            quality_report['quality_score'] = max(0, (quality_report['total_checked'] - total_issues) / quality_report['total_checked'] * 100)
        
        return quality_report
    
    def analyze_category_distribution(self):
        """分析类别分布"""
        distribution_report = {
            'missing_categories': [],
            'insufficient_categories': [],
            'imbalanced_distribution': False,
            'coverage_score': 0
        }
        
        stats = self.get_basic_statistics()
        
        # 检查缺失类别
        available_categories = set(stats['categories'].keys())
        target_categories_set = set(self.target_categories)
        
        distribution_report['missing_categories'] = list(target_categories_set - available_categories)
        
        # 检查数据不足的类别
        for category in available_categories:
            total_images = sum(stats['categories'][category].values())
            if total_images < self.quality_standards['min_images_per_category']:
                distribution_report['insufficient_categories'].append({
                    'category': category,
                    'count': total_images,
                    'required': self.quality_standards['min_images_per_category']
                })
        
        # 检查分布是否均衡
        if available_categories:
            category_counts = [sum(stats['categories'][cat].values()) for cat in available_categories]
            max_count = max(category_counts)
            min_count = min(category_counts)
            
            if max_count > 3 * min_count:  # 最大类别超过最小类别3倍
                distribution_report['imbalanced_distribution'] = True
        
        # 计算覆盖度评分
        covered_categories = len(available_categories & target_categories_set)
        distribution_report['coverage_score'] = covered_categories / len(target_categories_set) * 100
        
        return distribution_report
    
    def generate_evaluation_report(self, stats, quality_report, distribution_report):
        """生成评估报告"""
        report = {
            'dataset_path': str(self.dataset_path),
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'summary': {
                'total_images': stats['total_images'],
                'categories_found': len(stats['categories']),
                'target_categories_covered': len(set(stats['categories'].keys()) & set(self.target_categories)),
                'overall_score': 0
            },
            'statistics': stats,
            'quality_analysis': quality_report,
            'distribution_analysis': distribution_report,
            'recommendations': []
        }
        
        # 计算综合评分
        quality_score = quality_report['quality_score']
        coverage_score = distribution_report['coverage_score']
        report['summary']['overall_score'] = (quality_score + coverage_score) / 2
        
        # 生成建议
        recommendations = []
        
        if distribution_report['missing_categories']:
            recommendations.append(f"缺失以下核心类别: {', '.join(distribution_report['missing_categories'])}")
        
        if distribution_report['insufficient_categories']:
            insufficient = [item['category'] for item in distribution_report['insufficient_categories']]
            recommendations.append(f"以下类别数据不足: {', '.join(insufficient)}")
        
        if quality_report['quality_score'] < 80:
            recommendations.append("图像质量需要改善，建议检查分辨率、清晰度和亮度")
        
        if distribution_report['imbalanced_distribution']:
            recommendations.append("数据分布不均衡，建议平衡各类别的样本数量")
        
        if stats['total_images'] < 1000:
            recommendations.append("数据集规模较小，建议增加更多样本以提高模型性能")
        
        report['recommendations'] = recommendations
        
        return report
    
    def save_report(self, report):
        """保存评估报告"""
        # 保存JSON格式报告
        json_path = self.dataset_path / "evaluation_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成可读性强的文本报告
        text_report = self.format_text_report(report)
        text_path = self.dataset_path / "evaluation_report.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        self.logger.info(f"评估报告已保存: {json_path}")
        self.logger.info(f"文本报告已保存: {text_path}")
    
    def format_text_report(self, report):
        """格式化文本报告"""
        text = f"""
📊 智能冰箱数据集质量评估报告
======================================

📁 数据集路径: {report['dataset_path']}
📅 评估时间: {report['evaluation_date']}

🎯 总体评分: {report['summary']['overall_score']:.1f}/100

📈 基础统计
-----------
• 总图像数: {report['summary']['total_images']:,}
• 发现类别数: {report['summary']['categories_found']}
• 目标类别覆盖: {report['summary']['target_categories_covered']}/9

📊 数据分布
-----------
"""
        
        for dataset in ['train', 'val', 'test']:
            count = report['statistics']['datasets'][dataset]
            text += f"• {dataset.upper()}集: {count:,} 张图像\n"
        
        text += "\n🔍 质量分析\n-----------\n"
        text += f"• 质量评分: {report['quality_analysis']['quality_score']:.1f}/100\n"
        text += f"• 检查样本数: {report['quality_analysis']['total_checked']:,}\n"
        text += f"• 分辨率问题: {len(report['quality_analysis']['resolution_issues'])} 张\n"
        text += f"• 模糊问题: {len(report['quality_analysis']['blur_issues'])} 张\n"
        text += f"• 亮度问题: {len(report['quality_analysis']['brightness_issues'])} 张\n"
        
        text += "\n📋 类别分析\n-----------\n"
        text += f"• 覆盖度评分: {report['distribution_analysis']['coverage_score']:.1f}/100\n"
        
        if report['distribution_analysis']['missing_categories']:
            text += f"• 缺失类别: {', '.join(report['distribution_analysis']['missing_categories'])}\n"
        
        if report['distribution_analysis']['insufficient_categories']:
            text += "• 数据不足类别:\n"
            for item in report['distribution_analysis']['insufficient_categories']:
                text += f"  - {item['category']}: {item['count']} 张 (需要 {item['required']} 张)\n"
        
        if report['recommendations']:
            text += "\n💡 改进建议\n-----------\n"
            for i, rec in enumerate(report['recommendations'], 1):
                text += f"{i}. {rec}\n"
        
        text += "\n✅ 评估完成\n"
        
        return text
    
    def visualize_distribution(self):
        """可视化数据分布"""
        stats = self.get_basic_statistics()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('数据集分布分析', fontsize=16)
        
        # 1. 类别分布
        categories = list(stats['categories'].keys())
        counts = [sum(stats['categories'][cat].values()) for cat in categories]
        
        axes[0, 0].bar(categories, counts)
        axes[0, 0].set_title('各类别图像数量')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 数据集分布
        datasets = ['train', 'val', 'test']
        dataset_counts = [stats['datasets'][ds] for ds in datasets]
        
        axes[0, 1].pie(dataset_counts, labels=datasets, autopct='%1.1f%%')
        axes[0, 1].set_title('训练/验证/测试集分布')
        
        # 3. 文件格式分布
        formats = list(stats['file_formats'].keys())
        format_counts = list(stats['file_formats'].values())
        
        axes[1, 0].bar(formats, format_counts)
        axes[1, 0].set_title('文件格式分布')
        
        # 4. 类别覆盖度
        target_set = set(self.target_categories)
        available_set = set(categories)
        
        covered = len(available_set & target_set)
        missing = len(target_set - available_set)
        
        axes[1, 1].pie([covered, missing], labels=['已覆盖', '缺失'], autopct='%1.1f%%')
        axes[1, 1].set_title('目标类别覆盖度')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.dataset_path / "distribution_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"分布图表已保存: {chart_path}")
        
        plt.show()

def setup_logging(log_level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_evaluation.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集质量评估工具")
    
    parser.add_argument("dataset_path", type=str,
                       help="数据集路径")
    parser.add_argument("--visualize", action="store_true",
                       help="生成可视化图表")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    try:
        evaluator = DatasetEvaluator(args.dataset_path)
        
        # 执行评估
        report = evaluator.evaluate_dataset()
        
        # 打印总结
        print(f"\n🎯 评估完成!")
        print(f"总体评分: {report['summary']['overall_score']:.1f}/100")
        print(f"图像质量: {report['quality_analysis']['quality_score']:.1f}/100")
        print(f"类别覆盖: {report['distribution_analysis']['coverage_score']:.1f}/100")
        
        if report['recommendations']:
            print("\n💡 主要建议:")
            for rec in report['recommendations'][:3]:  # 显示前3个建议
                print(f"  • {rec}")
        
        # 可视化
        if args.visualize:
            evaluator.visualize_distribution()
        
        logger.info("数据集评估完成")
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()