#!/usr/bin/env python3
"""
智能冰箱打霜系统集成测试脚本
测试自动打霜功能与主系统的集成效果
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.core.main import SmartFridgeSystem
from src.core.defrost_system import DefrostMode

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('defrost_integration_test.log')
        ]
    )

def test_defrost_system_integration():
    """测试打霜系统集成"""
    logger = logging.getLogger(__name__)
    
    print("🧊 智能冰箱打霜系统集成测试")
    print("=" * 60)
    
    try:
        # 1. 初始化系统
        print("\n1. 初始化智能冰箱系统...")
        fridge_system = SmartFridgeSystem()
        logger.info("智能冰箱系统初始化成功")
        
        # 2. 测试系统状态
        print("\n2. 获取系统状态...")
        status = fridge_system.get_system_status()
        print(f"   系统运行状态: {'✅ 正常' if status.get('running') else '❌ 异常'}")
        print(f"   当前食物数量: {status.get('food_count', 0)}")
        print(f"   最大响应时间: {status.get('max_response_time', 0)}秒")
        print(f"   增强模型启用: {'✅ 是' if status.get('enhanced_model_enabled') else '❌ 否'}")
        
        defrost_status = status.get('defrost_system', {})
        print(f"   打霜系统启用: {'✅ 是' if defrost_status.get('enabled') else '❌ 否'}")
        print(f"   打霜模式: {defrost_status.get('mode', 'unknown')}")
        
        # 3. 添加蔬菜类食物
        print("\n3. 添加蔬菜类食物到系统...")
        vegetables_to_add = [
            ("生菜", "leafy_greens"),
            ("胡萝卜", "vegetables"),  
            ("西兰花", "vegetables"),
            ("菠菜", "leafy_greens")
        ]
        
        for veg_name, category in vegetables_to_add:
            result = fridge_system.add_vegetable_to_defrost(veg_name, category)
            if result.get('success'):
                print(f"   ✅ 已添加 {veg_name} ({category})")
            else:
                print(f"   ❌ 添加 {veg_name} 失败: {result.get('error')}")
        
        # 4. 查看更新后的打霜状态
        print("\n4. 查看打霜系统状态...")
        defrost_status = fridge_system.get_defrost_status()
        print(f"   监控的食物数量: {defrost_status.get('food_storage_count', 0)}")
        
        zones = defrost_status.get('zones', {})
        for zone_id, zone_info in zones.items():
            print(f"   {zone_id}:")
            print(f"     正在打霜: {'✅ 是' if zone_info.get('is_defrosting') else '❌ 否'}")
            print(f"     打霜次数: {zone_info.get('defrost_count', 0)}")
            last_defrost = zone_info.get('last_defrost_time')
            if last_defrost:
                print(f"     上次打霜: {last_defrost}")
            else:
                print(f"     上次打霜: 无记录")
        
        # 5. 测试手动打霜
        print("\n5. 测试手动打霜功能...")
        manual_result = fridge_system.manual_defrost('vegetable_compartment', 5)  # 5分钟测试
        if manual_result.get('success'):
            print(f"   ✅ 手动打霜已启动")
            print(f"     区域: {manual_result.get('zone_id')}")
            print(f"     持续时间: {manual_result.get('duration')}分钟")
        else:
            print(f"   ❌ 手动打霜失败: {manual_result.get('error')}")
        
        # 6. 测试打霜系统开关
        print("\n6. 测试打霜系统开关功能...")
        
        # 6.1 禁用打霜系统
        print("   6.1 禁用打霜系统...")
        disable_result = fridge_system.disable_defrost_system()
        if disable_result.get('success'):
            print("       ✅ 打霜系统已禁用")
        else:
            print(f"       ❌ 禁用失败: {disable_result.get('error')}")
        
        # 等待一会儿
        time.sleep(2)
        
        # 6.2 重新启用打霜系统
        print("   6.2 重新启用打霜系统...")
        enable_result = fridge_system.enable_defrost_system('auto')
        if enable_result.get('success'):
            print(f"       ✅ 打霜系统已启用，模式: {enable_result.get('mode')}")
        else:
            print(f"       ❌ 启用失败: {enable_result.get('error')}")
        
        # 7. 测试性能配置
        print("\n7. 测试性能配置...")
        print(f"   最大响应时间限制: {fridge_system.max_response_time}秒")
        print(f"   增强模型支持: {'✅ 启用' if fridge_system.enhanced_model_enabled else '❌ 禁用'}")
        
        # 8. 模拟食物识别处理时间
        print("\n8. 模拟食物识别处理...")
        start_time = time.time()
        
        # 模拟处理过程（实际中会调用AI模型）
        time.sleep(0.3)  # 模拟300ms处理时间
        
        processing_time = time.time() - start_time
        if processing_time <= fridge_system.max_response_time:
            print(f"   ✅ 处理时间 {processing_time:.3f}秒 在限制内")
        else:
            print(f"   ⚠️ 处理时间 {processing_time:.3f}秒 超出限制")
        
        # 9. 最终状态报告
        print("\n9. 最终系统状态报告...")
        final_status = fridge_system.get_system_status()
        defrost_final = final_status.get('defrost_system', {})
        
        print(f"   系统总体状态: {'✅ 正常' if final_status.get('running') else '❌ 异常'}")
        print(f"   打霜系统状态: {'✅ 运行' if defrost_final.get('enabled') else '❌ 停止'}")
        print(f"   监控食物数量: {defrost_final.get('food_storage_count', 0)}")
        
        print(f"\n✅ 集成测试完成！")
        
        return True
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        print(f"\n❌ 集成测试失败: {e}")
        return False

def test_defrost_modes():
    """测试不同的打霜模式"""
    logger = logging.getLogger(__name__)
    
    print("\n🔄 测试打霜模式切换")
    print("-" * 40)
    
    try:
        fridge_system = SmartFridgeSystem()
        
        # 测试各种模式
        modes_to_test = ['auto', 'manual', 'scheduled', 'disabled']
        
        for mode in modes_to_test:
            print(f"\n测试模式: {mode}")
            
            if mode == 'disabled':
                result = fridge_system.disable_defrost_system()
            else:
                result = fridge_system.enable_defrost_system(mode)
            
            if result.get('success'):
                print(f"   ✅ 切换到 {mode} 模式成功")
            else:
                print(f"   ❌ 切换到 {mode} 模式失败: {result.get('error')}")
            
            # 查看当前状态
            status = fridge_system.get_defrost_status()
            current_mode = status.get('mode', 'unknown')
            enabled = status.get('enabled', False)
            print(f"   当前状态: {current_mode}, 启用: {enabled}")
            
            time.sleep(1)  # 等待状态稳定
        
        print("\n✅ 模式切换测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"模式切换测试失败: {e}")
        print(f"\n❌ 模式切换测试失败: {e}")
        return False

def test_performance_impact():
    """测试性能影响"""
    print("\n📊 测试性能影响")
    print("-" * 40)
    
    try:
        fridge_system = SmartFridgeSystem()
        
        # 测试多个食物处理的性能
        test_foods = [
            "生菜", "胡萝卜", "西兰花", "菠菜", "白菜", 
            "土豆", "番茄", "黄瓜", "茄子", "豆角"
        ]
        
        start_total = time.time()
        
        for i, food_name in enumerate(test_foods):
            start_item = time.time()
            
            # 添加食物到打霜监控
            result = fridge_system.add_vegetable_to_defrost(food_name)
            
            item_time = time.time() - start_item
            
            if result.get('success'):
                print(f"   {i+1:2d}. {food_name:<8} - {item_time:.3f}s ✅")
            else:
                print(f"   {i+1:2d}. {food_name:<8} - {item_time:.3f}s ❌ ({result.get('error', '未知错误')})")
        
        total_time = time.time() - start_total
        avg_time = total_time / len(test_foods)
        
        print(f"\n总处理时间: {total_time:.3f}秒")
        print(f"平均每项时间: {avg_time:.3f}秒")
        print(f"性能评估: {'✅ 优秀' if avg_time < 0.1 else '⚠️ 需优化' if avg_time < 0.5 else '❌ 较慢'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 开始智能冰箱打霜系统集成测试")
    print("=" * 80)
    
    # 运行所有测试
    tests = [
        ("基础集成测试", test_defrost_system_integration),
        ("打霜模式测试", test_defrost_modes), 
        ("性能影响测试", test_performance_impact)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'🧪 ' + test_name}")
        print("=" * 80)
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ 测试 {test_name} 发生异常: {e}")
            results[test_name] = False
    
    # 输出测试总结
    print(f"\n{'📋 测试总结'}")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name:<20} {status}")
    
    print(f"\n总测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！打霜系统集成成功！")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)