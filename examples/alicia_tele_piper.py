#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alicia_duo_sdk.controller import ArmController
from piper_sdk import *

def enable_fun(piper:C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)

def set_joint_value(position, piper):
    factor = 57295.7795 #1000*180/3.1415926
    joint_0 = round(position[0]*factor)
    joint_1 = round(position[1]*factor)    
    joint_2 = round(position[2]*factor)

    joint_3 = round(position[3]*factor)
    joint_4 = round(position[4]*factor)
    joint_5 = round(position[5]*factor)
    joint_6 = round(position[6]*1000*1000)
        # piper.MotionCtrl_1()
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    # print(piper.GetArmStatus())
    # print(position)
    # time.sleep(0.005)


def joint_projection(joint_position):
    """
    Project Alicia Duo joint positions to Piper joint limits
    
    Alicia duo joints limit: 
    J1: ± 180°      [-3.14, 3.14]
    J2: 0° ~ 180°   [0, 3.14]
    J3 -180° ~ 0°   [-3.14, 0]
    J4: ± 180°      [-3.14, 3.14]
    J5: [-90°, 90°] [-1.57, 1.57]
    J6: ± 180°      [-3.14, 3.14]

    Piper joints limit:
    J1: ±154°       [-2.68, 2.68]
    J2 :0°~195°     [0, 3.40]
    J3: -175°~0°    [-3.05, 0]
    J4: -106°~106°  [-1.85, 1.85]
    J5: -75°~75°    [-1.31, 1.31]
    J6: ±100°       [-1.75, 1.75]
    """
    motor_dir = [1, -1, -1, -1, -1, -1]
    # Define Alicia Duo limits (radians)
    alicia_limits = [
        [-3.14, 3.14],    # J1
        [0, 3.14],        # J2
        [-3.14, 0],       # J3 
        [-3.14, 3.14],    # J4
        [-1.57, 1.57],    # J5
        [-3.14, 3.14]     # J6
    ]
    
    # Define Piper limits (radians)
    piper_limits = [
        [-2.68, 2.68],    # J1
        [0, 3.40],        # J2
        [-3.05, 0],       # J3
        [-1.85, 1.85],    # J4
        [-1.31, 1.31],    # J5
        [-1.75, 1.75]     # J6
    ]
    
    projected_joints = []
    
    for i in range(6):
        alicia_min, alicia_max = alicia_limits[i]
        piper_min, piper_max = piper_limits[i]
        joint_val = joint_position[i] * motor_dir[i]  # Apply motor direction
        
        # Clamp to Alicia limits first
        joint_val = max(alicia_min, min(alicia_max, joint_val))
        
        # Normalize to [0, 1] based on Alicia range
        alicia_range = alicia_max - alicia_min
        normalized = (joint_val - alicia_min) / alicia_range
        
        # Map to Piper range
        piper_range = piper_max - piper_min
        projected_val = piper_min + normalized * piper_range
        projected_joints.append(round(projected_val, 2))
    # For gripper
    projected_joints.append(0.0)
    return projected_joints



def main():
    
    # 创建控制器实例 (可选参数: port="/dev/ttyUSB0", debug_mode=True)
    controller = ArmController(debug_mode=False)
    
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
        
        piper = C_PiperInterface_V2("can0")
        piper.ConnectPort()
        piper.EnableArm(7)
        enable_fun(piper=piper)

        print("连接成功，开始读取数据...")
        print("按 Ctrl+C 退出")
        print("-" * 50)


        while True:
            state = controller.read_joint_state()

            alicia_joint_pos = [round(angle, 2) for angle in state.angles]
            # alicia_joint_pos = state.angles
            print(controller._thread_running)
            # print(joint_angles_deg)
            print(f"关节角度(弧度)\n: {alicia_joint_pos}  ")
            piper_joint_pos = joint_projection(alicia_joint_pos)
            print(piper_joint_pos)
            set_joint_value(piper_joint_pos, piper)
            print("==========================")
            print('\n \n\n')
            # time.sleep(0.05)
            # time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()