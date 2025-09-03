# 🧊 智能冰箱系统 (Smart Fridge POC)

基于AI的智能冰箱系统，集成食物识别、温度优化、自动打霜和自学习算法。

## ✨ 核心功能

- **🔍 智能识别**：AI识别8类食物，1秒内响应
- **🌡️ 温度优化**：多温区智能控制，±0.5°C精度
- **🧊 自动打霜**：蔬菜类食物智能打霜保鲜
- **🧠 自学习**：用户反馈持续优化策略
- **📱 边缘部署**：支持CPU优化和轻量化版本

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 系统演示
python scripts/demo_new_features.py

# 集成测试
python scripts/test_defrost_integration.py

# 启动系统
python src/main.py
```

## 📱 边缘部署

```bash
# 模型优化
python scripts/edge_deployment_optimizer.py

# 架构对比
python scripts/quick_architecture_test.py
```

## 📁 核心组件

- **src/core/main.py** - 主系统控制器
- **src/core/defrost_system.py** - 自动打霜系统
- **src/ai/training/** - AI模型训练
- **scripts/** - 工具和测试脚本
- **docs/** - 详细文档

## 📋 要求

- **Python 3.8+**, TensorFlow 2.13+
- **内存**: 4GB+ (标准), 2GB+ (边缘)

## 📖 文档

- [系统更新](docs/system_performance_update.md) - 新功能说明
- [打霜功能](DEFROST_UPDATE.md) - 自动打霜使用指南
- [详细文档](README_DETAILED.md) - 完整开发文档