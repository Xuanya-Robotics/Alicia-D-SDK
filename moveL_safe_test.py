import numpy as np
from alicia_duo_sdk.utils.vislab import plot_joint_angles
from alicia_duo_sdk.controller import get_default_session
from alicia_duo_sdk.controller.control_api import *
import time
import sys, select

session = get_default_session()
home_joint = [0.0] * 6
home_pose = [0.31084713, -0.00350417,
             0.09291784,  0.00641206,
             0.92877085, -0.00256358, 0.3705901 ]

moveJ(session=session, target_angles=home_joint)
try:
    input("continue")
    session.joint_controller.disable_torque()
    input("continue")
    session.joint_controller.enable_torque()
    time.sleep(0.5)
    angle = session.joint_controller.get_joint_angles()
    pos, quat = session.robot_model.forward_kinematics(angle)
    pose = np.concatenate([pos, quat]).tolist()
    moveL(session=session, speed_factor=0.01, target_pose=home_pose, visualize=True)
    moveL(session=session, target_pose=pose, visualize=True)


except KeyboardInterrupt:
    print("已结束")

