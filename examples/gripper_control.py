#!/usr/bin/env python3
# coding=utf-8

"""
夹爪控制示例
演示如何控制机械臂夹爪开合，以及读取按钮状态。
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
    print("=== 机械臂夹爪控制示例 ===")
    
    # 创建控制器实例
    controller = ArmController(debug_mode=True)
    
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
            
        print("连接成功")
        print("\n测试夹爪功能:")
        
        # 1. 完全打开夹爪 (0度)
        print("完全打开夹爪...")
        controller.set_gripper(0 * controller.DEG_TO_RAD)
        time.sleep(1)
        
        # 2. 读取夹爪状态
        gripper_angle, button1, button2 = controller.read_gripper_data()
        print(f"夹爪角度: {gripper_angle * controller.RAD_TO_DEG:.2f}度")
        print(f"按钮状态: 按钮1={'按下' if button1 else '释放'}, 按钮2={'按下' if button2 else '释放'}")
        time.sleep(1)
        
        # 3. 半闭合夹爪 (50度)
        print("\n设置夹爪半闭合...")
        controller.set_gripper(50 * controller.DEG_TO_RAD)
        time.sleep(1)
        
        # 4. 读取夹爪状态
        gripper_angle, button1, button2 = controller.read_gripper_data()
        print(f"夹爪角度: {gripper_angle * controller.RAD_TO_DEG:.2f}度")
        
        # 5. 完全闭合夹爪 (100度)
        print("\n完全闭合夹爪...")
        controller.set_gripper(100 * controller.DEG_TO_RAD)
        time.sleep(1)
        
        # 6. 读取夹爪状态
        gripper_angle, button1, button2 = controller.read_gripper_data()
        print(f"夹爪角度: {gripper_angle * controller.RAD_TO_DEG:.2f}度")
        
        # 7. 交互式控制
        print("\n=== 进入交互式控制模式 ===")
        print("输入夹爪角度(0-100度)，或输入'q'退出")
        
        while True:
            user_input = input("\n请输入角度值: ")
            
            if user_input.lower() == 'q':
                break
                
            try:
                angle_deg = float(user_input)
                if angle_deg < 0 or angle_deg > 100:
                    print("角度超出范围，有效范围: 0-100度")
                    continue
                    
                print(f"设置夹爪角度: {angle_deg}度")
                angle_rad = angle_deg * controller.DEG_TO_RAD
                controller.set_gripper(angle_rad)
                
                # 短暂延时后读取状态
                time.sleep(0.5)
                gripper_angle, button1, button2 = controller.read_gripper_data()
                print(f"实际夹爪角度: {gripper_angle * controller.RAD_TO_DEG:.2f}度")
                print(f"按钮状态: 按钮1={'按下' if button1 else '释放'}, 按钮2={'按下' if button2 else '释放'}")
                
            except ValueError:
                print("无效输入，请输入数字或'q'")
        
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 完全打开夹爪，复位
        controller.set_gripper(0 * controller.DEG_TO_RAD)
        time.sleep(0.5)
        
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()