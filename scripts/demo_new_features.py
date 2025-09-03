#!/usr/bin/env python3
"""
智能冰箱打霜系统使用示例
演示新的1秒响应时间配置和自动打霜功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.core.main import SmartFridgeSystem

def main():
    """主函数 - 演示智能冰箱新功能"""
    
    print("🚀 智能冰箱系统 - 新功能演示")
    print("=" * 50)
    
    # 1. 创建智能冰箱系统
    print("\n1. 初始化智能冰箱系统...")
    fridge = SmartFridgeSystem()
    print("   ✅ 系统初始化完成")
    
    # 2. 查看系统状态
    print("\n2. 查看系统状态...")
    status = fridge.get_system_status()
    print(f"   📊 系统运行: {'正常' if status.get('running') else '异常'}")
    print(f"   ⏱️  最大响应时间: {status.get('max_response_time')}秒")
    print(f"   🧠 增强模型: {'启用' if status.get('enhanced_model_enabled') else '禁用'}")
    
    defrost_info = status.get('defrost_system', {})
    print(f"   🧊 打霜系统: {'启用' if defrost_info.get('enabled') else '禁用'}")
    print(f"   🔧 打霜模式: {defrost_info.get('mode', 'unknown')}")
    
    # 3. 添加蔬菜到系统
    print("\n3. 添加蔬菜到打霜监控...")
    vegetables = [
        ("生菜", "leafy_greens"),
        ("胡萝卜", "vegetables"),
        ("西兰花", "vegetables"),
        ("小白菜", "leafy_greens")
    ]
    
    for veg_name, category in vegetables:
        result = fridge.add_vegetable_to_defrost(veg_name, category)
        if result.get('success'):
            print(f"   ✅ {veg_name} ({category})")
        else:
            print(f"   ❌ {veg_name} 添加失败")
    
    # 4. 查看打霜状态
    print("\n4. 查看打霜系统状态...")
    defrost_status = fridge.get_defrost_status()
    print(f"   📦 监控食物数量: {defrost_status.get('food_storage_count', 0)}")
    
    zones = defrost_status.get('zones', {})
    for zone_id, zone_info in zones.items():
        is_defrosting = zone_info.get('is_defrosting', False)
        count = zone_info.get('defrost_count', 0)
        print(f"   🧊 {zone_id}: {'打霜中' if is_defrosting else '待机'} (共{count}次)")
    
    # 5. 测试手动打霜
    print("\n5. 测试手动打霜...")
    manual_result = fridge.manual_defrost('vegetable_compartment', 3)  # 3分钟测试
    if manual_result.get('success'):
        print(f"   ✅ 手动打霜已启动 (时长: {manual_result.get('duration')}分钟)")
    else:
        print(f"   ❌ 手动打霜失败: {manual_result.get('error')}")
    
    # 6. 测试打霜系统开关
    print("\n6. 测试打霜系统开关...")
    
    # 禁用打霜
    print("   6.1 禁用打霜系统...")
    disable_result = fridge.disable_defrost_system()
    if disable_result.get('success'):
        print("       ✅ 打霜系统已禁用")
    
    time.sleep(1)
    
    # 重新启用
    print("   6.2 重新启用打霜系统...")
    enable_result = fridge.enable_defrost_system('auto')
    if enable_result.get('success'):
        print(f"       ✅ 打霜系统已启用 (模式: {enable_result.get('mode')})")
    
    # 7. 性能测试
    print("\n7. 性能配置测试...")
    print(f"   🎯 当前性能配置:")
    print(f"     - 最大响应时间: {fridge.max_response_time}秒")
    print(f"     - 增强模型支持: {'是' if fridge.enhanced_model_enabled else '否'}")
    
    # 模拟食物处理
    print(f"   ⚡ 模拟食物处理...")
    start_time = time.time()
    
    # 这里模拟实际的AI处理时间
    time.sleep(0.2)  # 200ms模拟处理
    
    processing_time = time.time() - start_time
    
    if processing_time <= fridge.max_response_time:
        print(f"     ✅ 处理时间 {processing_time:.3f}秒 符合要求")
    else:
        print(f"     ⚠️ 处理时间 {processing_time:.3f}秒 超出限制")
    
    # 8. 最终状态
    print("\n8. 最终系统状态...")
    final_status = fridge.get_system_status()
    final_defrost = final_status.get('defrost_system', {})
    
    print(f"   📊 系统状态: {'正常运行' if final_status.get('running') else '异常'}")
    print(f"   🧊 打霜系统: {'运行中' if final_defrost.get('enabled') else '已停止'}")
    print(f"   📦 监控食物: {final_defrost.get('food_storage_count', 0)}项")
    print(f"   🔧 当前模式: {final_defrost.get('mode', 'unknown')}")
    
    print("\n🎉 演示完成！")
    print("\n📋 新功能总结:")
    print("   ✅ 响应时间放宽到1秒内，支持更复杂的AI模型")
    print("   ✅ 自动打霜系统，智能监控蔬菜类食物")
    print("   ✅ 支持手动打霜和多种打霜模式")
    print("   ✅ 实时状态监控和系统控制接口")
    print("   ✅ 完整的系统集成和错误处理")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")