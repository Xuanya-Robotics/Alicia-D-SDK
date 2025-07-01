"""
FK from models
========================

Forward kinematics from URDF or MuJoCo XML files.
"""

import math
import pprint

import robolab

print("########## Forward kinematics from URDF or MuJoCo XML files with RobotModel class ##########")
print("---------- Forward kinematics for Franka Panda using URDF file ----------")
model_path = "../Alicia_duo_sdk/Robolab-main/assets/urdf/franka_description/robots/franka_panda.urdf"

joint_value = [[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0, 0.0, 0.0],
               [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0, 0.0, 0.0]]
export_link = "panda_hand"

# Try the same thing with pytorch_kinematics
robot = robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=False)
pos, rot, ret = robot.get_fk(joint_value, export_link)
print(f"Position of {export_link}: {pos}")
print(f"Rotation of {export_link}: {rot}")
pprint.pprint(ret, width=1)

# print("---------- Forward kinematics for Bruce Humanoid Robot using MJCF file ----------")
# model_path = "../assets/mjcf/bruce/bruce.xml"
# joint_value = [0.0 for _ in range(16)]

# export_link = "elbow_pitch_link_r"

# # # Build the robot model with pytorch_kinematics, kinpy is not supported for MJCF files
# robot = robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)
# # Get the forward kinematics of export_link
# pos, rot, ret = robot.get_fk(joint_value, export_link)

# # Print the results
# print(f"Position of {export_link}: {pos}")
# print(f"Rotation of {export_link}: {rot}")
# pprint.pprint(ret, width=1)
