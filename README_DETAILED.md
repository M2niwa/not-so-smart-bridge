# 🧊 智能冰箱系统 - 详细文档

基于AI的智能冰箱系统，集成食物识别、温度优化、自动打霜和自学习算法。

## 📊 系统概述

### 核心功能
- **AI识别**: 8类食物识别，1秒内响应
- **温度优化**: 12温区独立控制，±0.5°C精度
- **自动打霜**: 蔬菜类食物智能打霜系统
- **自学习**: 用户反馈驱动的优化算法
- **边缘部署**: CPU优化和轻量化版本

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 基本配置
cp config/config.yaml.example config/config.yaml

# 启动系统
python src/core/main.py

# 功能演示
python scripts/demo_new_features.py
```

### 📱 边缘部署

```bash
# 模型优化
python scripts/edge_deployment_optimizer.py

# 架构测试
python scripts/quick_architecture_test.py
```

## 📁 项目结构

```
smart-fridge-poc/
├── src/                         # 源代码
│   ├── ai/                      # AI模块
│   │   ├── inference/           # 推理引擎
│   │   │   └── detector.py      # 食物检测器
│   │   ├── self_learning/       # 自学习系统
│   │   │   ├── adaptive_network.py
│   │   │   ├── feedback_analyzer.py
│   │   │   ├── layout_optimizer.py
│   │   │   ├── performance_monitor.py
│   │   │   ├── temperature_optimizer.py
│   │   │   └── self_learning_controller.py
│   │   └── training/            # 模型训练
│   ├── core/                    # 核心系统
│   │   └── main.py              # 主系统入口
│   ├── edge/                    # 边缘计算模块
│   └── hardware/                # 硬件控制
│       ├── hardware_controller.py # 硬件控制器
│       └── temperature_optimizer.py # 温度优化器
├── edge_deployment/             # 边缘设备部署文件
│   ├── config/                  # 边缘部署配置
│   │   ├── cpu_only_config.yaml # CPU优化配置
│   │   └── edge_config.yaml     # 边缘设备配置
│   ├── src/                     # 边缘部署源码
│   │   ├── config_loader.py     # 配置加载器
│   │   ├── cpu_optimized_controller.py # CPU优化控制器
│   │   └── lightweight_controller.py # 轻量化控制器
│   ├── deploy_cpu.bat           # CPU版本部署脚本(Windows)
│   ├── deploy_cpu.sh            # CPU版本部署脚本(Linux/Mac)
│   ├── deploy_edge.bat          # 边缘版本部署脚本(Windows)
│   ├── deploy_edge.sh           # 边缘版本部署脚本(Linux/Mac)
│   ├── run_cpu.py               # CPU版本启动脚本
│   ├── run_edge.py              # 边缘版本启动脚本
│   ├── requirements_cpu.txt     # CPU版本依赖
│   ├── requirements_edge.txt    # 边缘版本依赖
│   ├── test_edge_adaptation.py  # 边缘适配测试
│   ├── README_CPU.md            # CPU版本说明
│   ├── README_EDGE.md           # 边缘版本说明
│   └── README.md                # 边缘部署总说明
├── data/                        # 数据文件
│   ├── preset_expiry.json       # 预设保鲜期数据
│   └── models/                  # AI模型文件
├── docs/                        # 文档
│   └── api/                     # API文档
├── config/                      # 配置文件
├── tests/                       # 测试代码
├── requirements.txt             # Python依赖
├── demo.py                      # 演示脚本
└── README.md                    # 项目说明
```

## 🔧 配置说明

### 系统配置 (config.yaml)

```yaml
system:
  debug_mode: false
  log_level: INFO
  max_temperature: 10.0
  min_temperature: -2.0

hardware:
  camera:
    enabled: true
    resolution: [640, 480]
    fps: 30
  sensors:
    temperature_sensor_type: "DS18B20"
    door_sensor_enabled: true
  control:
    pid_kp: 0.5
    pid_ki: 0.1
    pid_kd: 0.2

ai:
  model_path: "data/models/food_detector.pth"
  confidence_threshold: 0.7
  enable_learning: true

storage:
  data_directory: "data"
  backup_interval: 3600
  max_backup_files: 24
```

## 📚 API参考

### SmartFridgeSystem 类

```python
from src.core.main import SmartFridgeSystem

# 创建系统实例
system = SmartFridgeSystem()

# 启动监控
system.start_monitoring()

# 处理新食物
system.process_new_food()

# 紧急关断
system.emergency_shutdown()
```

### FreshnessDataService 类

```python
from src.core.main import FreshnessDataService

# 创建数据服务实例
data_service = FreshnessDataService()

# 存储用户评分
data_service.store_rating("牛奶", 5)
```

### FridgeController 类

```python
from src.hardware.hardware_controller import FridgeController

# 创建控制器实例
controller = FridgeController()

# 读取温度
temperature = controller.read_temperature()

# 检测门状态
is_door_open = controller.detect_door_status()

# 设置目标温度
controller.compartments[0].set_target(4.0)
```

## 🏗️ 系统架构

智能冰箱系统采用模块化设计，各组件通过明确的接口进行通信，便于维护和扩展。

### 核心模块

1. **主控制器 (Main Controller)**
   - 系统初始化和配置管理
   - 各模块协调和调度
   - 系统状态监控和故障处理

2. **温度控制模块 (Temperature Controller)**
   - 多温区温度管理
   - PID控制算法实现
   - 温度传感器数据采集
   - 执行器控制逻辑

3. **图像识别模块 (Image Recognition)**
   - 食物图像采集和预处理
   - 深度学习模型推理
   - 食物种类和数量识别
   - 识别结果存储和查询

4. **自学习模块 (Self Learning)**
   - 自适应神经网络模型
   - 用户反馈收集和分析
   - 保鲜算法优化
   - 预测模型训练和更新

5. **用户反馈模块 (User Feedback)**
   - 反馈收集界面管理
   - 反馈数据验证和存储
   - 反馈模式分析和统计
   - 优化建议生成

6. **硬件接口模块 (Hardware Interface)**
   - 传感器数据采集
   - 执行器控制接口
   - 硬件状态监控
   - 故障检测和处理

### 模块间通信

- **消息总线**：基于事件驱动的消息传递机制
- **数据存储**：统一的数据库接口，支持多种存储后端
- **配置管理**：集中式配置管理，支持动态更新
- **日志系统**：统一的日志记录和分析框架

### 部署架构

- **本地部署**：所有模块运行在同一设备上，适合家庭使用
- **边缘部署**：核心模块运行在边缘设备，部分计算任务卸载到云端
- **云端部署**：所有模块运行在云端，通过Web界面和API提供服务

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户界面层                                │
├─────────────────────────────────────────────────────────────┤
│                    应用逻辑层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   自学习系统     │  │   温度优化系统   │  │   食物识别系统   │ │
│  │                 │  │                 │  │                 │ │
│  │ • 反馈分析      │  │ • PID控制       │  │ • 图像处理      │ │
│  │ • 模式识别      │  │ • 热力学模型    │  │ • 物体检测      │ │
│  │ • 策略优化      │  │ • 自适应调节    │  │ • 分类识别      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    硬件抽象层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   传感器接口     │  │   执行器接口     │  │   通信接口      │ │
│  │                 │  │                 │  │                 │ │
│  │ • 温度传感器    │  │ • 压缩机控制    │  │ • 数据通信      │ │
│  │ • 门状态传感器  │  │ • 风扇控制      │  │ • 状态同步      │ │
│  │ • 湿度传感器    │  │ • 加热器控制    │  │ • 错误处理      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    硬件层                                    │
│            摄像头、传感器、执行器、微控制器                   │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **图像采集**：摄像头捕获冰箱内部图像
2. **食物识别**：AI模型识别食物种类和数量
3. **温度优化**：系统根据食物类型计算最优温度
4. **硬件控制**：控制器调整各温区温度
5. **状态监控**：持续监控温度和食物状态
6. **用户反馈**：收集用户对食物新鲜度的评分
7. **学习优化**：系统根据反馈优化策略

## 📁 核心模块

- **src/core/main.py** - 主系统控制器和打霜集成
- **src/core/defrost_system.py** - 自动打霜系统
- **src/ai/training/** - AI模型训练和优化
- **src/ai/self_learning/** - 自学习算法系统
- **scripts/** - 工具和测试脚本

## 🧪 测试

```
# 集成测试
python scripts/test_defrost_integration.py

# 边缘优化测试
python scripts/edge_deployment_optimizer.py

# 架构对比测试
python scripts/quick_architecture_test.py
```

## 📊 性能监控

### 系统指标
- **标准版本**: ±0.5°C, <1000ms, 识别准确率>90%
- **CPU优化**: ±0.8°C, <1000ms, 识别准确率>85%
- **边缘版本**: ±1.0°C, <1000ms, 识别准确率>75%

## 🛠️ 故障排除

### 常见问题
1. **模型加载失败**: 检查models目录和路径配置
2. **硬件连接错误**: 验证传感器和执行器连接
3. **温度控制异常**: 检查PID参数和温度范围设置

## 📄 许可证

MIT许可证 - 详见 [LICENSE](LICENSE) 文件。
