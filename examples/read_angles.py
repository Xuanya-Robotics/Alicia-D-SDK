#!/usr/bin/env python3
# coding=utf-8

"""
关节角度读取示例
连续读取并显示机械臂的关节角度、夹爪角度和按钮状态。
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
    print("=== 机械臂数据读取示例 ===")
    
    # 创建控制器实例 (可选参数: port="/dev/ttyUSB0", debug_mode=True)
    controller = ArmController(debug_mode=True)
    
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
            
        print("连接成功，开始读取数据...")
        print("按 Ctrl+C 退出")
        print("-" * 50)
        
        # 持续读取数据
        while True:
            # 读取完整状态
            state = controller.read_joint_state()
            
            # 转换为度数显示
            joint_angles_deg = [round(angle * controller.RAD_TO_DEG, 2) for angle in state.angles]
            gripper_angle_deg = round(state.gripper * controller.RAD_TO_DEG, 2)
            
            # 打印状态信息
            print("\r\033[K", end="")  # 清除当前行
            print(f"时间: {time.strftime('%H:%M:%S')} | ", end="")
            print(f"关节角度(度): {joint_angles_deg} | ", end="")
            print(f"夹爪: {gripper_angle_deg}度 | ", end="")
            print(f"按钮: {'按下' if state.button1 else '释放'}/{('按下' if state.button2 else '释放')}", end="")
            
            # 短暂延时
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()