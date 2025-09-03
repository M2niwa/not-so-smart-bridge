#!/usr/bin/env python3
"""
智能冰箱系统演示脚本

此脚本演示智能冰箱系统的核心功能，包括：
- 温区控制
- 图像识别
- 温度优化
- 用户反馈
"""

import sys
import time
import random
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_temperature_control():
    """演示温度控制功能"""
    print("\n=== 温度控制演示 ===")
    
    from src.hardware.hardware_controller import FridgeController
    
    # 创建冰箱控制器
    controller = FridgeController()
    
    print(f"冰箱有 {len(controller.compartments)} 个温区")
    
    # 演示各温区控制
    for i, zone in enumerate(controller.compartments):
        print(f"\n温区 {i+1}:")
        print(f"  当前温度: {zone.current_temp}°C")
        print(f"  目标温度: {zone.target_temp}°C")
        # 确保door_open属性存在
        if not hasattr(zone, 'door_open'):
            zone.set_target(zone.target_temp)  # 这会初始化door_open属性
        print(f"  门状态: {'开启' if zone.door_open else '关闭'}")
        
        # 设置新的目标温度
        new_temp = random.uniform(2, 8)
        zone.set_target(new_temp)
        print(f"  设置新目标温度: {new_temp:.1f}°C")
        
        # 模拟温度调节
        for _ in range(3):
            zone.current_temp = zone.read_temperature()
            time.sleep(0.1)
        print(f"  调节后温度: {zone.current_temp:.1f}°C")

def demo_image_recognition():
    """演示图像识别功能"""
    print("\n=== 图像识别演示 ===")
    
    from src.ai.inference.detector import FoodDetector
    
    # 创建食物检测器
    detector = FoodDetector()
    
    # 模拟捕获图像
    print("模拟捕获冰箱内部图像...")
    time.sleep(1)
    
    # 检测食物（使用模拟图像数据）
    import numpy as np
    # 创建一个模拟的RGB图像数组
    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(mock_image)
    
    print(f"检测到 {len(results)} 种食物:")
    for i, food in enumerate(results):
        print(f"  {i+1}. {food['name']}")
        print(f"     置信度: {food['confidence']:.2f}")
        print(f"     保鲜期: {food['expiry_days']} 天")
        print(f"     最佳温度: {food['optimal_temp']}°C")

def demo_temperature_optimization():
    """演示温度优化功能"""
    print("\n=== 温度优化演示 ===")
    
    from src.hardware.temperature_optimizer import ThermalModel, PIDController
    
    # 创建热力学模型
    thermal_model = ThermalModel()
    
    # 模拟环境条件
    ambient_temp = 25.0
    door_open_time = 5.0
    
    print(f"环境温度: {ambient_temp}°C")
    print(f"门开启时间: {door_open_time} 秒")
    
    # 计算热负荷（使用模拟参数）
    mass = 5.0  # 模拟质量(kg)
    specific_heat = 4.18  # 水的比热容(kJ/kg·°C)
    heat_load = thermal_model.calculate_heat_load(mass, specific_heat)
    print(f"计算热负荷: {heat_load:.2f} kJ")
    
    # 创建PID控制器
    pid_controller = PIDController(kp=0.5, ki=0.1, kd=0.2)
    
    # 模拟温度控制
    current_temp = 4.0
    target_temp = 2.0
    
    print(f"\nPID温度控制:")
    print(f"当前温度: {current_temp}°C")
    print(f"目标温度: {target_temp}°C")
    
    for i in range(5):
        control_signal = pid_controller.compute(target_temp, current_temp, 1.0)
        current_temp += control_signal * 0.1  # 简化的温度响应
        print(f"  步骤 {i+1}: 控制信号={control_signal:.3f}, 温度={current_temp:.2f}°C")
        time.sleep(0.5)

def demo_user_feedback():
    """演示用户反馈功能"""
    print("\n=== 用户反馈演示 ===")
    
    from src.core.main import FreshnessDataService
    
    # 创建新鲜度数据服务
    service = FreshnessDataService()
    
    # 模拟用户反馈
    feedback_data = [
        {"food_type": "牛奶", "user_rating": 4},
        {"food_type": "蔬菜", "user_rating": 3},
        {"food_type": "水果", "user_rating": 5},
    ]
    
    print("记录用户反馈:")
    for i, feedback in enumerate(feedback_data):
        service.store_rating(
            feedback["food_type"],
            feedback["user_rating"]
        )
        print(f"  {i+1}. {feedback['food_type']}: 评分={feedback['user_rating']}/5")
    
    # 分析反馈数据
    print("\n反馈数据分析:")
    for food_type in set(f["food_type"] for f in feedback_data):
        avg_rating = sum(f["user_rating"] for f in feedback_data if f["food_type"] == food_type) / \
                     sum(1 for f in feedback_data if f["food_type"] == food_type)
        print(f"  {food_type}: 平均评分={avg_rating:.1f}/5")

def demo_emergency_shutdown():
    """演示紧急关断功能"""
    print("\n=== 紧急关断演示 ===")
    
    from src.hardware.hardware_controller import FridgeController
    
    # 创建冰箱控制器
    controller = FridgeController()
    
    print("模拟温度异常...")
    
    # 选择一个温区进行演示
    zone = controller.compartments[0]
    
    # 模拟温度异常（超过安全限制）
    zone.current_temp = 15.0  # 超过安全温度
    print(f"温区1温度异常: {zone.current_temp}°C")
    
    # 检查是否需要紧急关断（模拟温度超过安全限制）
    if zone.current_temp > 10.0:  # 假设安全温度上限为10度
        print("触发紧急关断！")
        zone.emergency_shutdown(3)  # 3级关断
        print(f"紧急关断后温度: {zone.current_temp}°c")
    else:
        print("温度在安全范围内")

def demo_system_monitoring():
    """演示系统监控功能"""
    print("\n=== 系统监控演示 ===")
    
    from src.core.main import SmartFridgeSystem
    
    # 创建智能冰箱系统
    system = SmartFridgeSystem()
    
    print("系统状态监控:")
    print(f"系统运行状态: {'运行中' if system._running else '已停止'}")
    print(f"总温区数量: {len(system.controller.compartments)}")
    print(f"食物库存数量: {len(system.food_inventory)}")
    
    # 显示各温区状态
    print("\n温区状态:")
    for i, zone in enumerate(system.controller.compartments):
        status = "活动"
        # 确保door_open属性存在
        if not hasattr(zone, 'door_open'):
            zone.set_target(zone.target_temp)  # 这会初始化door_open属性
        door_status = "开启" if zone.door_open else "关闭"
        print(f"  温区{i+1}: {status}, 门{door_status}, {zone.current_temp:.1f}°C")
    
    # 显示食物库存
    if system.food_inventory:
        print("\n食物库存:")
        for i, item in enumerate(system.food_inventory[:3]):  # 只显示前3个
            print(f"  {i+1}. {item['name']} (最佳温度: {item['optimal_temp']}°C)")

def main():
    """主演示函数"""
    print("🧊 9格间智能冰箱系统演示 🧊")
    print("=" * 50)
    
    try:
        # 演示各个功能模块
        demo_temperature_control()
        demo_image_recognition()
        demo_temperature_optimization()
        demo_user_feedback()
        demo_emergency_shutdown()
        demo_system_monitoring()
        
        print("\n" + "=" * 50)
        print("✅ 演示完成！")
        print("\n提示:")
        print("- 运行 'python run.py' 启动完整系统")
        print("- 运行 'python run.py --test' 运行测试套件")
        print("- 运行 'python run.py --train' 训练AI模型")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    from src.hardware.hardware_controller import FridgeController
    
    # 创建冰箱控制器
    controller = FridgeController()
    
    print(f"冰箱有 {len(controller.compartments)} 个温区")
    
    # 演示各温区控制
    for i, zone in enumerate(controller.compartments):
        print(f"\n温区 {i+1}:")
        print(f"  当前温度: {zone.current_temp}°C")
        print(f"  目标温度: {zone.target_temp}°C")
        # 确保door_open属性存在
        if not hasattr(zone, 'door_open'):
            zone.set_target(zone.target_temp)  # 这会初始化door_open属性
        print(f"  门状态: {'开启' if zone.door_open else '关闭'}")
        
        # 设置新的目标温度
        new_temp = random.uniform(2, 8)
        zone.set_target(new_temp)
        print(f"  设置新目标温度: {new_temp:.1f}°C")
        
        # 模拟温度调节
        for _ in range(3):
            zone.current_temp = zone.read_temperature()
            time.sleep(0.1)
        print(f"  调节后温度: {zone.current_temp:.1f}°C")

def demo_image_recognition():
    """演示图像识别功能"""
    print("\n=== 图像识别演示 ===")
    
    from src.ai.inference.detector import FoodDetector
    
    # 创建食物检测器
    detector = FoodDetector()
    
    # 模拟捕获图像
    print("模拟捕获冰箱内部图像...")
    time.sleep(1)
    
    # 检测食物（使用模拟图像数据）
    import numpy as np
    # 创建一个模拟的RGB图像数组
    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(mock_image)
    
    print(f"检测到 {len(results)} 种食物:")
    for i, food in enumerate(results):
        print(f"  {i+1}. {food['name']}")
        print(f"     置信度: {food['confidence']:.2f}")
        print(f"     保鲜期: {food['expiry_days']} 天")
        print(f"     最佳温度: {food['optimal_temp']}°C")

def demo_temperature_optimization():
    """演示温度优化功能"""
    print("\n=== 温度优化演示 ===")
    
    from src.hardware.temperature_optimizer import ThermalModel, PIDController
    
    # 创建热力学模型
    thermal_model = ThermalModel()
    
    # 模拟环境条件
    ambient_temp = 25.0
    door_open_time = 5.0
    
    print(f"环境温度: {ambient_temp}°C")
    print(f"门开启时间: {door_open_time} 秒")
    
    # 计算热负荷（使用模拟参数）
    mass = 5.0  # 模拟质量(kg)
    specific_heat = 4.18  # 水的比热容(kJ/kg·°C)
    heat_load = thermal_model.calculate_heat_load(mass, specific_heat)
    print(f"计算热负荷: {heat_load:.2f} kJ")
    
    # 创建PID控制器
    pid_controller = PIDController(kp=0.5, ki=0.1, kd=0.2)
    
    # 模拟温度控制
    current_temp = 4.0
    target_temp = 2.0
    
    print(f"\nPID温度控制:")
    print(f"当前温度: {current_temp}°C")
    print(f"目标温度: {target_temp}°C")
    
    for i in range(5):
        control_signal = pid_controller.compute(target_temp, current_temp, 1.0)
        current_temp += control_signal * 0.1  # 简化的温度响应
        print(f"  步骤 {i+1}: 控制信号={control_signal:.3f}, 温度={current_temp:.2f}°C")
        time.sleep(0.5)

def demo_user_feedback():
    """演示用户反馈功能"""
    print("\n=== 用户反馈演示 ===")
    
    from src.core.main import FreshnessDataService
    
    # 创建新鲜度数据服务
    service = FreshnessDataService()
    
    # 模拟用户反馈
    feedback_data = [
        {"food_type": "牛奶", "user_rating": 4},
        {"food_type": "蔬菜", "user_rating": 3},
        {"food_type": "水果", "user_rating": 5},
    ]
    
    print("记录用户反馈:")
    for i, feedback in enumerate(feedback_data):
        service.store_rating(
            feedback["food_type"],
            feedback["user_rating"]
        )
        print(f"  {i+1}. {feedback['food_type']}: 评分={feedback['user_rating']}/5")
    
    # 分析反馈数据
    print("\n反馈数据分析:")
    for food_type in set(f["food_type"] for f in feedback_data):
        avg_rating = sum(f["user_rating"] for f in feedback_data if f["food_type"] == food_type) / \
                     sum(1 for f in feedback_data if f["food_type"] == food_type)
        print(f"  {food_type}: 平均评分={avg_rating:.1f}/5")

def demo_emergency_shutdown():
    """演示紧急关断功能"""
    print("\n=== 紧急关断演示 ===")
    
    from src.hardware.hardware_controller import FridgeController
    
    # 创建冰箱控制器
    controller = FridgeController()
    
    print("模拟温度异常...")
    
    # 选择一个温区进行演示
    zone = controller.compartments[0]
    
    # 模拟温度异常（超过安全限制）
    zone.current_temp = 15.0  # 超过安全温度
    print(f"温区1温度异常: {zone.current_temp}°C")
    
    # 检查是否需要紧急关断（模拟温度超过安全限制）
    if zone.current_temp > 10.0:  # 假设安全温度上限为10度
        print("触发紧急关断！")
        zone.emergency_shutdown(3)  # 3级关断
        print(f"紧急关断后温度: {zone.current_temp}°c")
    else:
        print("温度在安全范围内")

def demo_system_monitoring():
    """演示系统监控功能"""
    print("\n=== 系统监控演示 ===")
    
    from src.core.main import SmartFridgeSystem
    
    # 创建智能冰箱系统
    system = SmartFridgeSystem()
    
    print("系统状态监控:")
    print(f"系统运行状态: {'运行中' if system._running else '已停止'}")
    print(f"总温区数量: {len(system.controller.compartments)}")
    print(f"食物库存数量: {len(system.food_inventory)}")
    
    # 显示各温区状态
    print("\n温区状态:")
    for i, zone in enumerate(system.controller.compartments):
        status = "活动"
        # 确保door_open属性存在
        if not hasattr(zone, 'door_open'):
            zone.set_target(zone.target_temp)  # 这会初始化door_open属性
        door_status = "开启" if zone.door_open else "关闭"
        print(f"  温区{i+1}: {status}, 门{door_status}, {zone.current_temp:.1f}°C")
    
    # 显示食物库存
    if system.food_inventory:
        print("\n食物库存:")
        for i, item in enumerate(system.food_inventory[:3]):  # 只显示前3个
            print(f"  {i+1}. {item['name']} (最佳温度: {item['optimal_temp']}°C)")

def main():
    """主演示函数"""
    print("🧊 9格间智能冰箱系统演示 🧊")
    print("=" * 50)
    
    try:
        # 演示各个功能模块
        demo_temperature_control()
        demo_image_recognition()
        demo_temperature_optimization()
        demo_user_feedback()
        demo_emergency_shutdown()
        demo_system_monitoring()
        
        print("\n" + "=" * 50)
        print("✅ 演示完成！")
        print("\n提示:")
        print("- 运行 'python run.py' 启动完整系统")
        print("- 运行 'python run.py --test' 运行测试套件")
        print("- 运行 'python run.py --train' 训练AI模型")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())