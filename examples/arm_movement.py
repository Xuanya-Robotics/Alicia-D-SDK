#!/usr/bin/env python3
# coding=utf-8

"""
机械臂运动控制示例
演示如何控制机械臂进行基本运动、力矩控制和运动示例。
"""

import os
import sys
import time
import math

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alicia_duo_sdk.controller import ArmController

def main():
    """主函数"""
    print("=== 机械臂运动控制示例 ===")
    
    # 创建控制器实例
    controller = ArmController()
    
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
            
        print("连接成功")
        
        # 读取初始位置
        initial_state = controller.read_joint_state()
        initial_angles = [round(angle * controller.RAD_TO_DEG, 2) for angle in initial_state.angles]
        print(f"初始关节角度(度): {initial_angles}")
        
        # 1. 演示关节控制 - 移动到零位置
        print("\n将所有关节移动到零位置...")
        zero_angles = [0.0] * 6  # 6个关节角度都设为0
        controller.set_joint_angles(zero_angles)
        time.sleep(2)
        
        # 读取当前位置
        current_state = controller.read_joint_state()
        current_angles = [round(angle * controller.RAD_TO_DEG, 2) for angle in current_state.angles]
        print(f"当前关节角度(度): {current_angles}")
        
        # 2. 演示逐个关节移动
        print("\n演示逐个关节移动...")
        for i in range(6):
            # 移动当前关节到30度
            test_angles = [0.0] * 6
            test_angles[i] = 30 * controller.DEG_TO_RAD
            
            print(f"移动关节{i+1}到30度...")
            controller.set_joint_angles(test_angles)
            time.sleep(1.5)
            
            # 移动回零位置
            controller.set_joint_angles(zero_angles)
            time.sleep(1)
        
        # 3. 演示波浪运动 (关节2)
        print("\n演示波浪运动...")
        for angle in range(0, 45, 5):  # 从0度到45度，步长5度
            wave_angles = zero_angles.copy()
            wave_angles[1] = angle * controller.DEG_TO_RAD  # 关节2
            controller.set_joint_angles(wave_angles)
            time.sleep(0.2)
            
        for angle in range(45, -45, -5):  # 从45度到-45度，步长5度
            wave_angles = zero_angles.copy()
            wave_angles[1] = angle * controller.DEG_TO_RAD  # 关节2
            controller.set_joint_angles(wave_angles)
            time.sleep(0.2)
            
        for angle in range(-45, 0, 5):  # 从-45度到0度，步长5度
            wave_angles = zero_angles.copy()
            wave_angles[1] = angle * controller.DEG_TO_RAD  # 关节2
            controller.set_joint_angles(wave_angles)
            time.sleep(0.2)
        
        # 回到零位置
        controller.set_joint_angles(zero_angles)
        time.sleep(1)
        
        # 4. 演示力矩控制
        print("\n演示力矩控制...")
        print("禁用力矩 - 进入自由拖动模式")
        controller.disable_torque()
        print("现在可以手动移动机械臂，将持续5秒...")
        time.sleep(5)
        
        print("启用力矩 - 锁定当前位置")
        controller.enable_torque()
        time.sleep(1)
        
        # 5. 演示设置零点
        user_input = input("\n是否将当前位置设为零点? (y/n): ")
        if user_input.lower() == 'y':
            print("设置当前位置为零点...")
            controller.set_zero_position()
            print("零点设置完成")
        
        # 6. 演示完整抓取过程
        print("\n演示抓取过程...")
        
        # 打开夹爪
        print("1. 打开夹爪")
        controller.set_gripper(0 * controller.DEG_TO_RAD)
        time.sleep(1)
        
        # 移动关节1到30度
        print("2. 移动到目标位置")
        target_angles = zero_angles.copy()
        target_angles[0] = 30 * controller.DEG_TO_RAD
        controller.set_joint_angles(target_angles)
        time.sleep(1.5)
        
        # 闭合夹爪 (模拟抓取)
        print("3. 闭合夹爪抓取物体")
        controller.set_gripper(70 * controller.DEG_TO_RAD)
        time.sleep(1.5)
        
        # 抬起
        print("4. 抬起物体")
        target_angles[1] = -20 * controller.DEG_TO_RAD
        controller.set_joint_angles(target_angles)
        time.sleep(1.5)
        
        # 移动到新位置
        print("5. 移动到放置位置")
        target_angles[0] = -30 * controller.DEG_TO_RAD
        controller.set_joint_angles(target_angles)
        time.sleep(1.5)
        
        # 放下
        print("6. 放下物体")
        target_angles[1] = 0 * controller.DEG_TO_RAD
        controller.set_joint_angles(target_angles)
        time.sleep(1.5)
        
        # 松开夹爪
        print("7. 松开夹爪")
        controller.set_gripper(0 * controller.DEG_TO_RAD)
        time.sleep(1.5)
        
        # 返回初始位置
        print("8. 返回初始位置")
        controller.set_joint_angles(zero_angles)
        time.sleep(1.5)
        
        print("\n演示完成!")
        
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()