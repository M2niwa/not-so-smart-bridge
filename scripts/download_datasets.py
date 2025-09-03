#!/usr/bin/env python3
"""
智能冰箱系统推荐数据集下载脚本
支持自动下载和预处理多个公开数据集
"""

import os
import sys
import logging
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
import json
from urllib.parse import urlparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, download_dir="datasets"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 推荐数据集配置
        self.datasets = {
            'food101': {
                'name': 'Food-101',
                'url': 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
                'file_size': '5.2GB',
                'description': '101种食物类别，每类1000张图像',
                'categories': 101,
                'images': 101000,
                'license': 'Academic Use',
                'suitable_for': ['食物分类', '特征学习', '迁移学习']
            },
            'open_images_food': {
                'name': 'Open Images Dataset (Food Subset)',
                'url': 'custom_download',  # 需要自定义下载逻辑
                'file_size': '~2GB',
                'description': 'Google开放图像数据集的食物子集',
                'categories': 50,
                'images': 20000,
                'license': 'CC BY 2.0',
                'suitable_for': ['物体检测', '多标签分类']
            },
            'usda_food': {
                'name': 'USDA Food Images',
                'url': 'manual_download',  # 需要手动申请
                'file_size': '~3GB',
                'description': '美国农业部食物图像数据集',
                'categories': 'Various',
                'images': 50000,
                'license': 'Public Domain',
                'suitable_for': ['营养分析', '食材识别']
            }
        }
    
    def list_datasets(self):
        """列出所有推荐数据集"""
        print("🍎 智能冰箱系统推荐数据集")
        print("=" * 60)
        
        for key, dataset in self.datasets.items():
            print(f"\n📊 {dataset['name']}")
            print(f"   大小: {dataset['file_size']}")
            print(f"   类别数: {dataset['categories']}")
            print(f"   图像数: {dataset['images']}")
            print(f"   许可证: {dataset['license']}")
            print(f"   适用于: {', '.join(dataset['suitable_for'])}")
            print(f"   描述: {dataset['description']}")
            
            if dataset['url'] == 'manual_download':
                print(f"   ⚠️  需要手动下载")
            elif dataset['url'] == 'custom_download':
                print(f"   🔧 需要自定义下载脚本")
            else:
                print(f"   🔗 可自动下载")
    
    def download_food101(self):
        """下载Food-101数据集"""
        dataset_info = self.datasets['food101']
        url = dataset_info['url']
        filename = 'food-101.tar.gz'
        filepath = self.download_dir / filename
        
        self.logger.info(f"开始下载 {dataset_info['name']}...")
        self.logger.info(f"文件大小: {dataset_info['file_size']}")
        
        try:
            # 检查文件是否已存在
            if filepath.exists():
                self.logger.info("文件已存在，跳过下载")
                return str(filepath)
            
            # 下载文件
            self.logger.info(f"正在从 {url} 下载...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
            
            print()  # 换行
            self.logger.info(f"下载完成: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"下载失败: {e}")
            return None
    
    def extract_food101(self, archive_path):
        """解压Food-101数据集"""
        try:
            extract_dir = self.download_dir / "food-101"
            
            if extract_dir.exists():
                self.logger.info("数据集已解压，跳过")
                return str(extract_dir)
            
            self.logger.info("正在解压数据集...")
            
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(self.download_dir)
            
            self.logger.info(f"解压完成: {extract_dir}")
            return str(extract_dir)
            
        except Exception as e:
            self.logger.error(f"解压失败: {e}")
            return None
    
    def filter_food101_for_fridge(self, dataset_path):
        """筛选Food-101中适合冰箱的食物类别"""
        try:
            dataset_path = Path(dataset_path)
            
            # 定义冰箱相关的食物类别映射
            fridge_categories = {
                # Food-101类别 -> 智能冰箱类别
                'apple_pie': 'fruits',
                'beef_carpaccio': 'beef',
                'beef_tartare': 'beef',
                'cheese_plate': 'cheese',
                'chicken_curry': 'chicken',
                'chicken_wings': 'chicken',
                'eggs_benedict': 'eggs',
                'fish_and_chips': 'fish',
                'fried_chicken': 'chicken',
                'grilled_cheese_sandwich': 'cheese',
                'ice_cream': 'dairy',
                'oysters': 'fish',
                'salmon': 'fish',
                'strawberry_shortcake': 'fruits',
                'sushi': 'fish',
                'tuna_tartare': 'fish'
            }
            
            # 创建筛选后的数据集目录
            filtered_dir = self.download_dir / "food101_fridge_filtered"
            
            # 复制相关类别的图像
            for food101_cat, fridge_cat in fridge_categories.items():
                source_dir = dataset_path / "images" / food101_cat
                
                if not source_dir.exists():
                    self.logger.warning(f"类别目录不存在: {source_dir}")
                    continue
                
                target_dir = filtered_dir / fridge_cat
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 复制图像文件
                images = list(source_dir.glob("*.jpg"))
                for i, img in enumerate(images[:100]):  # 每类最多100张
                    target_path = target_dir / f"{food101_cat}_{i:03d}.jpg"
                    if not target_path.exists():
                        import shutil
                        shutil.copy2(img, target_path)
                
                self.logger.info(f"已处理类别 {food101_cat} -> {fridge_cat}: {len(images)} 张图像")
            
            self.logger.info(f"筛选完成: {filtered_dir}")
            return str(filtered_dir)
            
        except Exception as e:
            self.logger.error(f"筛选失败: {e}")
            return None
    
    def generate_sample_dataset(self, output_dir="ai_model/sample_dataset"):
        """生成示例数据集（用于快速测试）"""
        try:
            from src.ai.training.prepare_training_data import create_directory_structure, create_sample_images
            
            # 智能冰箱核心类别
            categories = ['milk', 'beef', 'vegetables', 'fruits', 'eggs', 
                         'cheese', 'yogurt', 'chicken', 'fish']
            
            self.logger.info("生成示例数据集...")
            
            # 创建目录结构
            success = create_directory_structure(output_dir, categories)
            if not success:
                return False
            
            # 创建示例图像
            success = create_sample_images(output_dir, categories, images_per_category=20)
            if not success:
                return False
            
            self.logger.info(f"示例数据集生成完成: {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"示例数据集生成失败: {e}")
            return None
    
    def create_download_guide(self):
        """创建数据集下载指南"""
        guide_content = """# 智能冰箱系统数据集下载指南

## 🍎 推荐数据集

### 1. Food-101 数据集（自动下载）
```bash
# 下载并预处理Food-101数据集
python scripts/download_datasets.py --dataset food101 --auto-process
```

### 2. Open Images Dataset（手动下载）
1. 访问: https://opensource.google/projects/open-images-dataset
2. 下载食物相关类别
3. 使用导入工具处理

### 3. 自定义数据收集建议

#### 推荐收集类别：
- **奶制品**: 牛奶、酸奶、奶酪
- **肉类**: 牛肉、鸡肉、鱼类
- **农产品**: 蔬菜、水果
- **蛋类**: 鸡蛋

#### 图像要求：
- 分辨率: 至少224x224像素
- 格式: JPG/PNG
- 每类别: 最少50张，推荐100+张
- 多角度拍摄，包含不同光照条件

## 📊 数据质量标准

### 图像质量要求：
- ✅ 清晰度高，无模糊
- ✅ 主体突出，背景简洁
- ✅ 光照均匀，色彩真实
- ✅ 包含多种状态（新鲜、半新鲜等）

### 标注要求：
- ✅ 类别标注准确
- ✅ 新鲜度评级（1-5分）
- ✅ 拍摄时间戳
- ✅ 存储环境信息

## 🚀 快速开始

```bash
# 1. 生成示例数据集（快速测试）
python scripts/download_datasets.py --generate-sample

# 2. 下载Food-101数据集
python scripts/download_datasets.py --dataset food101

# 3. 导入到训练系统
python scripts/import_dataset.py --data_type images --source datasets/food101_filtered
```
"""
        
        guide_path = self.download_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        self.logger.info(f"下载指南已创建: {guide_path}")

def setup_logging(log_level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_download.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能冰箱系统数据集下载工具")
    
    parser.add_argument("--list", action="store_true",
                       help="列出所有推荐数据集")
    parser.add_argument("--dataset", type=str,
                       choices=["food101", "open_images", "usda"],
                       help="下载指定数据集")
    parser.add_argument("--generate-sample", action="store_true",
                       help="生成示例数据集")
    parser.add_argument("--auto-process", action="store_true",
                       help="自动处理下载的数据集")
    parser.add_argument("--download-dir", type=str, default="datasets",
                       help="下载目录")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    try:
        downloader = DatasetDownloader(args.download_dir)
        
        if args.list:
            # 列出推荐数据集
            downloader.list_datasets()
            downloader.create_download_guide()
            
        elif args.generate_sample:
            # 生成示例数据集
            result = downloader.generate_sample_dataset()
            if result:
                logger.info("示例数据集生成成功！")
                logger.info("现在可以运行: python scripts/import_dataset.py --data_type images --source ai_model/sample_dataset")
            else:
                logger.error("示例数据集生成失败！")
                
        elif args.dataset == "food101":
            # 下载Food-101数据集
            archive_path = downloader.download_food101()
            if archive_path:
                extract_path = downloader.extract_food101(archive_path)
                if extract_path and args.auto_process:
                    filtered_path = downloader.filter_food101_for_fridge(extract_path)
                    if filtered_path:
                        logger.info(f"数据集已准备就绪: {filtered_path}")
                        logger.info("使用以下命令导入到训练系统:")
                        logger.info(f"python scripts/import_dataset.py --data_type images --source {filtered_path}")
                        
        else:
            logger.info("请指定操作：--list, --dataset, 或 --generate-sample")
            
    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()