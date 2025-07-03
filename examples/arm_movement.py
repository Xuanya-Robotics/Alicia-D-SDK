#!/usr/bin/env python3
# coding=utf-8

"""
机械臂运动控制示例
演示如何控制机械臂进行基本运动、力矩控制和运动示例。
"""

import os
import sys
import time
from typing import List


# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alicia_duo_sdk.controller import ArmController


def control_move(controller, current_angles: List[float], 
              target_angles: List[float], steps: int = 120, delay: float = 0.03) -> None:
        
        """缓慢移动到目标角度
        Args:
            controller (ArmController): 机械臂控制器实例
            current_angles (list): 当前关节角度列表
            target_angles (list): 目标关节角度列表
            steps (int): 插值步数
            delay (float): 每步延迟时间（秒）
        逐步插值移动到目标角度，模拟缓慢运动。     
        """

        if len(current_angles) != len(target_angles):
            raise ValueError(f"错误: 当前角度数量是：{len(current_angles)} 与目标角度数量：{len(target_angles)} 数量不匹配")
                
        for step in range(1, steps + 1):
            interp_angles = [
                current + (target - current) * step / steps
                for current, target in zip(current_angles, target_angles)
            ]
            result = controller.set_joint_angles(interp_angles, wait_for_completion=False)
            if not result:
                return False
            
            time.sleep(delay)


def main():
    """主函数"""
    print("=== 机械臂运动控制示例 ===")
    
    # 创建控制器实例
    controller = ArmController(debug_mode=False)
    
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
            
        print("连接成功")
        
        # 初始化夹爪位置
        print("初始化夹爪位置...")
        controller.set_gripper(0 * controller.DEG_TO_RAD, wait_for_completion=True)
        
        # 读取初始位置
        initial_state = controller.read_joint_state()
        initial_angles = [round(angle * controller.RAD_TO_DEG, 2) for angle in initial_state.angles]
        print(f"初始关节角度(度): {initial_angles}")
        
        # 1. 演示关节控制 - 移动到零位置
        print("\n将所有关节移动到零位置...")
        zero_angles = [0.0] * 6 # 6个关节角度都设为0
        result = control_move(controller, initial_state.angles, zero_angles)
        print(f"移动到零位置结果: {result}")
        
        # 读取当前位置
        current_state = controller.read_joint_state()
        current_angles = [round(angle * controller.RAD_TO_DEG, 2) for angle in current_state.angles]
        print(f"当前关节角度(度): {current_angles}")
        
        # 2. 演示逐个关节移动
        print("\n演示逐个关节移动...")
        for i in range(6):
            # 移动当前关节到30度
            test_angles = [0.0] * 6
            test_angles[i] = 15 * controller.DEG_TO_RAD
            
            print(f"移动关节{i+1}到30度...")
            control_move(controller, current_state.angles, test_angles)
            
            # 移动回零位置
            print(f"移动关节{i+1}回零位置...")
            control_move(controller, test_angles, zero_angles)
        
        # 演示夹爪控制
        print("\n演示夹爪控制...")
        print("打开夹爪...")
        controller.set_gripper(100 * controller.DEG_TO_RAD, wait_for_completion=True)
        
        print("关闭夹爪...")
        controller.set_gripper(0 * controller.DEG_TO_RAD, wait_for_completion=True)
        
        # 读取当前位置
        current_state = controller.read_joint_state()
        current_angles = [round(angle * controller.RAD_TO_DEG, 2) for angle in current_state.angles]
        print(f"当前关节角度(度): {current_angles}")
        
        # 回到零位置
        print("\n回到零位置...")
        control_move(controller, current_state.angles, zero_angles)
        print("\n演示完成!")
        
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()