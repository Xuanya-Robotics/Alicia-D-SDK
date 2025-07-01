import math
import pprint
import torch as tensor
import robolab

model_path = "../Alicia_duo_sdk/Robolab-main/assets/urdf/alicia_duo_descriptions/urdf/alicia_duo_with_gripper.urdf"

joint_value = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0]
joint_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
export_link = "Grasp_base"

# Try the same thing with pytorch_kinematics
robot = robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)
pos, rot, ret = robot.get_fk(joint_value, export_link)
# Squeeze pos and rot to remove batch dimension if present
pos = tensor.squeeze(pos, 0)
rot = tensor.squeeze(rot, 0)
print(f"Position of {export_link}: {pos}")
print(f"Rotation of {export_link}: {rot}")
# pprint.pprint(ret, width=1)


ee_pose = [0.1711, 0.0036, 0.0340, 0.0011, -0.0006, 0.0000, 0.0001]
ee_pose = [0.2190, 0.0007, 0.1975,2.2955e-06,  9.2388e-01, -2.5809e-06,  3.8269e-01]

# Get ik solution
ret = robot.get_ik(ee_pose, export_link)
ret.solutions = tensor.squeeze(ret.solutions, 0)  # Remove the batch dimension
ret.solutions = tensor.round(ret.solutions * 100) / 100  # Round to 4 decimal places
print(ret.solutions)

# # Get ik solution near the current configuration
cur_configs = [[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0]]
ret = robot.get_ik(ee_pose, export_link, cur_configs=cur_configs)
ret.solutions = tensor.squeeze(ret.solutions)  # Remove the batch dimension
print(ret.solutions)
