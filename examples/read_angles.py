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
    controller = ArmController(debug_mode=False)
    
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
            
        print("连接成功，开始读取数据...")
        print("按 Ctrl+C 退出")
        print("-" * 50)

        q = [2.16, 3.14, 3.14, 3.14, 3.14, 3.14]



        pos, quat, rpy =controller.forward_kinematics_alicia_duo(q)
        print(f"正向运动学结果: 位置: {pos}, 四元数: {quat}, 偏航角: {rpy} (弧度)")

            
        # 持续读取数据
        while True:
            # 读取完整状态

            #start_time = time.time()
            state = controller.read_joint_state()
            #print("读取完成",time.time()-start_time)
            # 转换为度数显示

            joint_angles_deg = [round(angle * controller.RAD_TO_DEG, 2) for angle in state.angles]
            gripper_angle_deg = round(state.gripper * controller.RAD_TO_DEG, 2)
            pos, quat, rpy=controller.forward_kinematics_alicia_duo(state.angles)

            solved_ik_angles = controller.inverse_kinematics_alicia_duo(pos,
                                                    quat,
                                                    state.angles)

            print(controller._thread_running)
            # 打印状态信息
            print(f"关节角度(度): {joint_angles_deg}  ")
            print(f"末端位置: {pos}  ")
            print(f"逆解关节角度(度): {[round(angle * controller.RAD_TO_DEG, 2) for angle in solved_ik_angles]}  ")
            #print(f"末端四元数: {quat}  ")
            
            print(f"夹爪角度(度): {gripper_angle_deg}  ")
            print(f"按钮状态: {state.button1} {state.button2} ")
            

            # 短暂延时
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()