"""
Coordinate transform
========================

This example shows how to transform coordinates between different reference frames and rotation representations.
"""

import robolab

# Quaternion convert
quat = [0.234, 0.23, 0.4, 1.3]

mat = robolab.convert_ori_format(quat, "quat", "mat")
euler = robolab.convert_ori_format(quat, "quat", "euler")
homo_mat = robolab.homo_matrix_from_quat_tensor(quat, pos=[0, 2, 1])
print(f"Quaternion: {robolab.check_quat_tensor(quat)}")
print(f"Rotation matrix:\n {mat}")
print(f"Euler angles: {euler}")
print(f"Homo matrix:\n {homo_mat}")
# [Rofunc:INFO] Quaternion: tensor([[0.1672, 0.1644, 0.2859, 0.9291]])
# [Rofunc:INFO] Rotation matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011],
#          [ 0.5862,  0.7806, -0.2168],
#          [-0.2098,  0.4048,  0.8900]]])
# [Rofunc:INFO] Euler angles: tensor([[0.4268, 0.2114, 0.6430]])
# [Rofunc:INFO] Homo matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011,  0.0000],
#          [ 0.5862,  0.7806, -0.2168,  2.0000],
#          [-0.2098,  0.4048,  0.8900,  1.0000],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])

# Rotation matrix convert
rot = [[0.7825, -0.4763, 0.4011],
       [0.5862, 0.7806, -0.2168],
       [-0.2098, 0.4048, 0.8900]]
quat = robolab.convert_ori_format(rot, "mat", "quat")
euler = robolab.convert_ori_format(rot, "mat", "euler")
print(f"Rotation matrix:\n {robolab.check_rot_matrix_tensor(rot)}")
print(f"Quaternion: {robolab.check_quat_tensor(quat)}")
print(f"Euler angles: {euler}")
# [Rofunc:INFO] Rotation matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011],
#          [ 0.5862,  0.7806, -0.2168],
#          [-0.2098,  0.4048,  0.8900]]])
# [Rofunc:INFO] Quaternion: tensor([[0.1673, 0.1644, 0.2859, 0.9291]])
# [Rofunc:INFO] Euler angles: tensor([[0.4269, 0.2114, 0.6429]])

# Euler convert
euler = [0.4268, 0.2114, 0.6430]
quat = robolab.convert_ori_format(euler, "euler", "quat")
mat = robolab.convert_ori_format(euler, "euler", "mat")
print(f"Euler angles: {robolab.check_euler_tensor(euler)}")
print(f"Quaternion: {robolab.check_quat_tensor(quat)}")
print(f"Rotation matrix:\n {mat}")
# [Rofunc:INFO] Euler angles: tensor([[0.4268, 0.2114, 0.6430]])
# [Rofunc:INFO] Quaternion: tensor([[0.1672, 0.1644, 0.2859, 0.9291]])
# [Rofunc:INFO] Rotation matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011],
#          [ 0.5863,  0.7806, -0.2168],
#          [-0.2098,  0.4047,  0.8900]]])

# # Quaternion multiplication
# quat1 = [[-0.436865, 0.49775, 0.054428, 0.747283], [0, 0, 1, 0]]
# quat2 = [0.707, 0, 0, 0.707]
# quat3 = robolab.quat_multiply(quat1, quat2)
# print(f"Result: {robolab.check_quat_tensor(quat3)}")
# [Rofunc:INFO] Result: tensor([[ 0.2195,  0.3904, -0.3135,  0.8373],
#                              [ 0.0000,  0.7071,  0.7071,  0.0000]])
