import numpy as np
from alicia_duo_sdk.utils.vislab import plot_joint_angles
from alicia_duo_sdk.controller import get_default_session
from alicia_duo_sdk.controller.control_api import ControlApi
import time
import sys, select

session = get_default_session()
controller = ControlApi(session=session)
home = [0.0] * 6

try:
    controller.moveJ(target_angles=home, speed_factor=1)
    input("disable torque")
    session.joint_controller.disable_torque()
    while True:
        angle = session.joint_controller.get_joint_angles()
        pose = session.robot_model.forward_kinematics(angle)
        print(f'pos: {pose[:3]}, quat:{pose[3:]}')
        time.sleep(0.5)
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input()  # 停止读取
            print("已停止循环读取。")
            print("用户中断，已退出读取。")
            session.joint_controller.enable_torque()
            angle = session.joint_controller.get_joint_angles()
            pos, quat = session.robot_model.forward_kinematics(angle)
            print(angle)
            print(pos, quat)
            controller.moveJ(target_angles=home, speed_factor=1)
            break
except KeyboardInterrupt:
    print()
   
   
  