import time
import threading
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.async_image_processor import AsyncImageProcessor, EnhancedSmartFridgeSystem

class AsyncImageProcessorDemo:
    """异步图像处理器演示类"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('async_processor_demo.log'),
                logging.StreamHandler()
            ]
        )
    
    def demo_1_basic_door_events(self):
        """演示基本门事件处理"""
        print("\n" + "="*60)
        print("演示1: 基本门事件处理")
        print("="*60)
        
        # 创建处理器
        processor = AsyncImageProcessor(max_queue_size=5, max_workers=2)
        processor.start()
        
        try:
            print("处理器已启动，模拟门开关事件...")
            
            # 模拟门开关序列
            door_events = [
                ("打开冰箱门", True),
                ("关闭冰箱门", False),
                ("打开冰箱门", True),
                ("关闭冰箱门", False),
                ("打开冰箱门", True),
                ("关闭冰箱门", False)
            ]
            
            for event_name, door_state in door_events:
                print(f"\n--- {event_name} ---")
                processor.door_state = door_state
                processor.last_door_change_time = time.time()
                
                if not door_state:  # 门关闭时触发图像识别
                    processor._trigger_image_capture()
                
                # 显示当前状态
                status = processor.get_queue_status()
                print(f"门状态: {'打开' if door_state else '关闭'}")
                print(f"队列大小: {status['task_queue_size']}")
                print(f"工作线程数: {status['active_workers']}")
                
                time.sleep(2)  # 等待处理
            
            print("\n等待所有任务处理完成...")
            time.sleep(5)
            
            # 显示最终状态
            final_status = processor.get_queue_status()
            print(f"\n最终状态 - 队列大小: {final_status['task_queue_size']}, 结果队列: {final_status['result_queue_size']}")
            
        finally:
            processor.stop()
            print("演示1完成\n")
    
    def demo_2_async_processing_power(self):
        """演示异步处理能力"""
        print("\n" + "="*60)
        print("演示2: 异步处理能力")
        print("="*60)
        
        # 创建高性能处理器
        processor = AsyncImageProcessor(max_queue_size=10, max_workers=4)
        processor.start()
        
        try:
            print("高性能处理器已启动，添加多个任务...")
            
            # 快速添加多个任务
            start_time = time.time()
            task_count = 8
            
            for i in range(task_count):
                image_data = f"demo_image_{i}_high_quality".encode()
                success = processor.add_manual_task(image_data)
                
                if success:
                    print(f"任务 {i+1}/{task_count} 添加成功")
                else:
                    print(f"任务 {i+1}/{task_count} 添加失败")
                
                time.sleep(0.2)  # 快速添加
            
            print(f"\n所有任务已添加，等待处理完成...")
            
            # 监控处理进度
            while True:
                status = processor.get_queue_status()
                processed_count = task_count - status['task_queue_size']
                elapsed_time = time.time() - start_time
                
                print(f"\r处理进度: {processed_count}/{task_count} ({processed_count/task_count*100:.1f}%) - "
                      f"队列: {status['task_queue_size']} - 耗时: {elapsed_time:.1f}s", end="")
                
                if status['task_queue_size'] == 0:
                    break
                
                time.sleep(0.5)
            
            total_time = time.time() - start_time
            throughput = task_count / total_time
            
            print(f"\n\n处理完成！")
            print(f"总任务数: {task_count}")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"平均吞吐量: {throughput:.2f} 任务/秒")
            
        finally:
            processor.stop()
            print("演示2完成\n")
    
    def demo_3_queue_management(self):
        """演示队列管理"""
        print("\n" + "="*60)
        print("演示3: 队列管理")
        print("="*60)
        
        # 创建小队列处理器
        processor = AsyncImageProcessor(max_queue_size=3, max_workers=1)
        processor.start()
        
        try:
            print("小队列处理器已启动（队列容量: 3）")
            
            # 暂停工作线程以展示队列限制
            processor.pause_workers()
            print("工作线程已暂停，开始添加任务...")
            
            # 尝试添加超过队列容量的任务
            success_count = 0
            failed_count = 0
            
            for i in range(6):
                image_data = f"queue_test_image_{i}".encode()
                success = processor.add_manual_task(image_data)
                
                if success:
                    success_count += 1
                    print(f"✓ 任务 {i+1} 添加成功")
                else:
                    failed_count += 1
                    print(f"✗ 任务 {i+1} 添加失败（队列已满）")
                
                # 显示队列状态
                status = processor.get_queue_status()
                print(f"  当前队列大小: {status['task_queue_size']}/{processor.max_queue_size}")
                
                time.sleep(0.5)
            
            print(f"\n队列限制测试结果:")
            print(f"成功添加: {success_count} 个任务")
            print(f"失败添加: {failed_count} 个任务")
            print(f"队列容量: {processor.max_queue_size}")
            
            # 恢复工作线程并处理
            print("\n恢复工作线程，开始处理任务...")
            processor.resume_workers()
            
            # 等待处理完成
            while True:
                status = processor.get_queue_status()
                if status['task_queue_size'] == 0:
                    break
                print(f"\r剩余任务: {status['task_queue_size']}", end="")
                time.sleep(0.5)
            
            print("\n所有任务处理完成！")
            
        finally:
            processor.stop()
            print("演示3完成\n")
    
    def demo_4_enhanced_system(self):
        """演示增强系统集成"""
        print("\n" + "="*60)
        print("演示4: 增强智能冰箱系统集成")
        print("="*60)
        
        # 创建增强系统
        system = EnhancedSmartFridgeSystem()
        system.start()
        
        try:
            print("增强智能冰箱系统已启动")
            
            # 显示系统状态
            status = system.get_system_status()
            print(f"\n系统状态:")
            print(f"系统运行: {'是' if status['system_running'] else '否'}")
            print(f"工作线程: {status['queue_status']['active_workers']}")
            print(f"温度分区: {status['temperature_zones']}")
            print(f"食物库存: {status['food_inventory_count']} 项")
            
            # 模拟真实使用场景
            print("\n模拟真实使用场景:")
            
            scenarios = [
                "用户打开冰箱门放入牛奶",
                "用户关闭冰箱门",
                "系统自动进行图像识别",
                "用户再次打开冰箱门放入蔬菜",
                "用户关闭冰箱门",
                "系统再次进行图像识别"
            ]
            
            for i, scenario in enumerate(scenarios, 1):
                print(f"\n步骤 {i}: {scenario}")
                
                if "打开" in scenario:
                    system.async_processor.door_state = True
                elif "关闭" in scenario:
                    system.async_processor.door_state = False
                    system.async_processor.last_door_change_time = time.time()
                    system.async_processor._trigger_image_capture()
                elif "识别" in scenario:
                    print("  系统正在后台处理图像识别...")
                
                # 显示当前状态
                current_status = system.get_system_status()
                queue_size = current_status['queue_status']['task_queue_size']
                print(f"  处理队列: {queue_size} 个任务")
                
                time.sleep(1.5)
            
            print("\n等待所有处理完成...")
            time.sleep(3)
            
            # 显示最终系统状态
            final_status = system.get_system_status()
            print(f"\n最终系统状态:")
            print(f"处理队列: {final_status['queue_status']['task_queue_size']}")
            print(f"结果队列: {final_status['queue_status']['result_queue_size']}")
            print(f"门状态: {'打开' if final_status['queue_status']['door_state'] else '关闭'}")
            
        finally:
            system.stop()
            print("演示4完成\n")
    
    def demo_5_performance_comparison(self):
        """演示性能对比"""
        print("\n" + "="*60)
        print("演示5: 性能对比")
        print("="*60)
        
        # 测试不同配置的性能
        configs = [
            {"name": "轻量级配置", "max_queue_size": 5, "max_workers": 1},
            {"name": "标准配置", "max_queue_size": 10, "max_workers": 2},
            {"name": "高性能配置", "max_queue_size": 20, "max_workers": 4}
        ]
        
        task_count = 10
        
        for config in configs:
            print(f"\n--- {config['name']} ---")
            print(f"队列大小: {config['max_queue_size']}, 工作线程: {config['max_workers']}")
            
            processor = AsyncImageProcessor(
                max_queue_size=config['max_queue_size'],
                max_workers=config['max_workers']
            )
            processor.start()
            
            try:
                # 添加任务
                start_time = time.time()
                
                for i in range(task_count):
                    image_data = f"perf_test_{config['name']}_{i}".encode()
                    processor.add_manual_task(image_data)
                    time.sleep(0.1)
                
                # 等待处理完成
                while True:
                    status = processor.get_queue_status()
                    if status['task_queue_size'] == 0:
                        break
                    time.sleep(0.1)
                
                total_time = time.time() - start_time
                throughput = task_count / total_time
                
                print(f"处理时间: {total_time:.2f}秒")
                print(f"吞吐量: {throughput:.2f} 任务/秒")
                
            finally:
                processor.stop()
        
        print("\n性能对比完成！")
        print("演示5完成\n")
    
    def run_all_demos(self):
        """运行所有演示"""
        print("异步图像处理器演示程序")
        print("="*60)
        print("本程序将演示异步图像处理器的核心功能:")
        print("1. 门事件触发的图像识别")
        print("2. 异步处理能力")
        print("3. 队列管理")
        print("4. 增强系统集成")
        print("5. 性能对比")
        print("="*60)
        
        demos = [
            self.demo_1_basic_door_events,
            self.demo_2_async_processing_power,
            self.demo_3_queue_management,
            self.demo_4_enhanced_system,
            self.demo_5_performance_comparison
        ]
        
        for demo in demos:
            try:
                demo()
                input("按回车键继续下一个演示...")
            except KeyboardInterrupt:
                print("\n演示被用户中断")
                break
            except Exception as e:
                print(f"演示过程中出现错误: {e}")
                continue
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)
        
        # 生成演示报告
        self.generate_demo_report()
    
    def generate_demo_report(self):
        """生成演示报告"""
        report = {
            "demo_summary": {
                "title": "异步图像处理器演示报告",
                "date": datetime.now().isoformat(),
                "features_demonstrated": [
                    "门事件触发的图像识别",
                    "异步处理能力",
                    "队列管理和容量限制",
                    "增强系统集成",
                    "性能优化和对比"
                ]
            },
            "key_benefits": [
                "支持冰箱门开关事件自动触发图像识别",
                "异步处理减小运算压力，提高系统响应速度",
                "队列管理防止系统过载，确保稳定性",
                "多线程处理提高并发性能",
                "可配置的队列大小和工作线程数"
            ],
            "technical_details": {
                "async_processing": "使用多线程队列实现异步图像处理",
                "door_event_detection": "实时监控门状态变化，触发图像识别",
                "queue_management": "支持队列容量限制和任务调度",
                "error_handling": "完善的异常处理和系统恢复机制",
                "performance_optimization": "可配置的性能参数和负载均衡"
            }
        }
        
        # 保存报告
        with open('async_processor_demo_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n演示报告已保存到: async_processor_demo_report.json")


if __name__ == "__main__":
    # 运行演示
    demo = AsyncImageProcessorDemo()
    demo.run_all_demos()