#!/usr/bin/env python3
"""
智能冰箱系统数据集导入工具
支持多种格式的数据集导入和预处理
"""

import os
import sys
import logging
import argparse
import shutil
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.training.data_preprocessor import DataPreprocessor
from src.ai.training.prepare_training_data import create_directory_structure

class DatasetImporter:
    """数据集导入器"""
    
    def __init__(self, target_dir="ai_model/dataset"):
        self.target_dir = Path(target_dir)
        self.logger = logging.getLogger(__name__)
        self.preprocessor = DataPreprocessor()
        
    def import_image_dataset(self, source_path, dataset_type="auto", 
                           test_ratio=0.2, val_ratio=0.2):
        """
        导入图像数据集
        
        Args:
            source_path: 源数据路径
            dataset_type: 数据集类型 ("structured", "flat", "auto")
            test_ratio: 测试集比例
            val_ratio: 验证集比例
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            self.logger.error(f"源数据路径不存在: {source_path}")
            return False
            
        # 自动检测数据集类型
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(source_path)
            
        self.logger.info(f"检测到数据集类型: {dataset_type}")
        
        if dataset_type == "structured":
            return self._import_structured_dataset(source_path, test_ratio, val_ratio)
        elif dataset_type == "flat":
            return self._import_flat_dataset(source_path, test_ratio, val_ratio)
        else:
            self.logger.error(f"不支持的数据集类型: {dataset_type}")
            return False
    
    def _detect_dataset_type(self, source_path):
        """自动检测数据集类型"""
        # 检查是否有子目录（结构化数据集）
        subdirs = [d for d in source_path.iterdir() if d.is_dir()]
        
        if len(subdirs) > 0:
            # 检查子目录中是否包含图像
            for subdir in subdirs[:3]:  # 检查前3个目录
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                if len(images) > 0:
                    return "structured"
        
        # 检查根目录是否直接包含图像（扁平结构）
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        if len(images) > 0:
            return "flat"
            
        return "unknown"
    
    def _import_structured_dataset(self, source_path, test_ratio, val_ratio):
        """导入结构化数据集（已按类别分类）"""
        try:
            from sklearn.model_selection import train_test_split
            
            # 获取所有类别
            categories = [d.name for d in source_path.iterdir() if d.is_dir()]
            
            if not categories:
                self.logger.error("未找到类别目录")
                return False
            
            self.logger.info(f"发现 {len(categories)} 个类别: {categories}")
            
            # 创建目标目录结构
            success = create_directory_structure(self.target_dir, categories)
            if not success:
                return False
            
            # 处理每个类别
            for category in categories:
                category_source = source_path / category
                
                # 获取该类别所有图像
                images = (list(category_source.glob("*.jpg")) + 
                         list(category_source.glob("*.png")) + 
                         list(category_source.glob("*.jpeg")))
                
                if not images:
                    self.logger.warning(f"类别 {category} 中没有找到图像")
                    continue
                
                self.logger.info(f"处理类别 {category}: {len(images)} 张图像")
                
                # 分割数据
                train_val_images, test_images = train_test_split(
                    images, test_size=test_ratio, random_state=42
                )
                
                train_images, val_images = train_test_split(
                    train_val_images, test_size=val_ratio/(1-test_ratio), random_state=42
                )
                
                # 复制图像到目标目录
                for dataset, image_list in [
                    ('train', train_images),
                    ('val', val_images),
                    ('test', test_images)
                ]:
                    target_category_dir = self.target_dir / dataset / category
                    
                    for img_path in image_list:
                        target_path = target_category_dir / img_path.name
                        shutil.copy2(img_path, target_path)
                
                self.logger.info(f"类别 {category} 导入完成: "
                               f"训练集 {len(train_images)}, "
                               f"验证集 {len(val_images)}, "
                               f"测试集 {len(test_images)}")
            
            # 创建数据集信息文件
            self._create_dataset_info(categories)
            
            self.logger.info(f"结构化数据集导入完成: {self.target_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"结构化数据集导入失败: {e}")
            return False
    
    def _import_flat_dataset(self, source_path, test_ratio, val_ratio):
        """导入扁平数据集（需要手动分类）"""
        try:
            # 获取所有图像
            images = (list(source_path.glob("*.jpg")) + 
                     list(source_path.glob("*.png")) + 
                     list(source_path.glob("*.jpeg")))
            
            if not images:
                self.logger.error("未找到图像文件")
                return False
            
            self.logger.info(f"发现 {len(images)} 张图像")
            
            # 创建临时分类目录
            temp_dir = self.target_dir / "temp_classification"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有图像到临时目录
            for img in images:
                shutil.copy2(img, temp_dir / img.name)
            
            self.logger.info("扁平数据集已复制到临时目录，请手动分类后运行structured模式")
            self.logger.info(f"临时目录: {temp_dir}")
            self.logger.info("分类完成后，运行以下命令：")
            self.logger.info(f"python scripts/import_dataset.py --source {temp_dir} --type structured")
            
            return True
            
        except Exception as e:
            self.logger.error(f"扁平数据集导入失败: {e}")
            return False
    
    def import_freshness_data(self, csv_path, format_type="standard"):
        """
        导入新鲜度评分数据
        
        Args:
            csv_path: CSV文件路径
            format_type: 数据格式类型 ("standard", "custom")
        """
        try:
            csv_path = Path(csv_path)
            
            if not csv_path.exists():
                self.logger.error(f"CSV文件不存在: {csv_path}")
                return False
            
            # 读取CSV数据
            df = pd.read_csv(csv_path)
            
            if format_type == "standard":
                # 标准格式：timestamp, item, rating
                required_columns = ['timestamp', 'item', 'rating']
                if not all(col in df.columns for col in required_columns):
                    self.logger.error(f"CSV文件缺少必需列: {required_columns}")
                    return False
            
            # 数据验证
            if not self._validate_freshness_data(df):
                return False
            
            # 保存到目标路径
            target_path = Path("data") / "freshness_ratings.csv"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(target_path, index=False, encoding='utf-8')
            
            self.logger.info(f"新鲜度数据导入完成: {target_path}")
            self.logger.info(f"数据样本数: {len(df)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"新鲜度数据导入失败: {e}")
            return False
    
    def _validate_freshness_data(self, df):
        """验证新鲜度数据"""
        try:
            # 检查评分范围
            if 'rating' in df.columns:
                min_rating = df['rating'].min()
                max_rating = df['rating'].max()
                
                if min_rating < 0 or max_rating > 5:
                    self.logger.warning(f"评分超出预期范围 [0,5]: [{min_rating}, {max_rating}]")
            
            # 检查时间戳格式
            if 'timestamp' in df.columns:
                try:
                    pd.to_datetime(df['timestamp'].iloc[0])
                except:
                    self.logger.error("时间戳格式无效")
                    return False
            
            self.logger.info("数据验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
    
    def _create_dataset_info(self, categories):
        """创建数据集信息文件"""
        try:
            info = {
                'dataset_name': 'Smart Fridge Food Dataset',
                'creation_date': datetime.now().isoformat(),
                'categories': categories,
                'structure': {
                    'train': {},
                    'val': {},
                    'test': {}
                },
                'total_images': 0
            }
            
            # 统计每个数据集的图像数量
            for dataset in ['train', 'val', 'test']:
                dataset_path = self.target_dir / dataset
                if not dataset_path.exists():
                    continue
                
                for category in categories:
                    category_path = dataset_path / category
                    if category_path.exists():
                        image_count = len(list(category_path.glob("*.jpg")) + 
                                        list(category_path.glob("*.png")))
                        info['structure'][dataset][category] = image_count
                        info['total_images'] += image_count
            
            # 保存信息文件
            info_path = self.target_dir / "dataset_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据集信息文件已创建: {info_path}")
            
        except Exception as e:
            self.logger.error(f"创建数据集信息文件失败: {e}")

def setup_logging(log_level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_import.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能冰箱系统数据集导入工具")
    
    # 数据类型
    parser.add_argument("--data_type", type=str, choices=["images", "freshness"], 
                       required=True, help="数据类型")
    
    # 通用参数
    parser.add_argument("--source", type=str, required=True, 
                       help="源数据路径")
    parser.add_argument("--target", type=str, default="ai_model/dataset",
                       help="目标数据目录")
    
    # 图像数据集参数
    parser.add_argument("--type", type=str, choices=["structured", "flat", "auto"],
                       default="auto", help="数据集类型")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                       help="测试集比例")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                       help="验证集比例")
    
    # 新鲜度数据参数
    parser.add_argument("--format", type=str, choices=["standard", "custom"],
                       default="standard", help="数据格式")
    
    # 日志参数
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    try:
        # 创建导入器
        importer = DatasetImporter(args.target)
        
        if args.data_type == "images":
            # 导入图像数据集
            success = importer.import_image_dataset(
                args.source, 
                args.type, 
                args.test_ratio, 
                args.val_ratio
            )
        elif args.data_type == "freshness":
            # 导入新鲜度数据
            success = importer.import_freshness_data(args.source, args.format)
        
        if success:
            logger.info("数据集导入成功！")
        else:
            logger.error("数据集导入失败！")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"数据集导入失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()