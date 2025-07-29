# kinematics.py
import numpy as np
from typing import Dict, Tuple
from .utils.transform import translation_matrix, euler_matrix, rotation_matrix_from_axis_angle, matrix_to_quaternion

class RobotArm:
    def __init__(self):
        self.kinematic_chain = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'tool0'
        ]

        self.joint_hierarchy = {
            'joint1': {
                'origin': {'x': 0, 'y': 0, 'z': 0.1445},
                'rotation': {'x': 0, 'y': 0, 'z': 0},
                'axis': {'x': 0, 'y': 0, 'z': 1},
                'type': 'revolute'
            },
            'joint2': {
                'origin': {'x': 0, 'y': 0, 'z': 0.025106},
                'rotation': {'x': 1.5708, 'y': -1.5708, 'z': 0},
                'axis': {'x': 0, 'y': 0, 'z': -1},
                'type': 'revolute'
            },
            'joint3': {
                'origin': {'x': 0.22367, 'y': 0.022494, 'z': -0.00005},
                'rotation': {'x': 0, 'y': 0, 'z': 2.3562},
                'axis': {'x': 0, 'y': 0, 'z': -1},
                'type': 'revolute'
            },
            'joint4': {
                'origin': {'x': 0.0988, 'y': 0.00211, 'z': -0.0001},
                'rotation': {'x': 1.5708, 'y': 0, 'z': 1.5708},
                'axis': {'x': 0, 'y': 0, 'z': -1},
                'type': 'revolute'
            },
            'joint5': {
                'origin': {'x': 0, 'y': -0.0007, 'z': 0.12011},
                'rotation': {'x': -1.5708, 'y': 0, 'z': 0},
                'axis': {'x': 0, 'y': 0, 'z': -1},
                'type': 'revolute'
            },
            'joint6': {
                'origin': {'x': -0.0038938, 'y': -0.0573, 'z': 0.0008},
                'rotation': {'x': 1.5708, 'y': 0, 'z': 0},
                'axis': {'x': 0, 'y': 0, 'z': -1},
                'type': 'revolute'
            },
            'tool0': {
                'origin': {'x': 0.00275, 'y': 0.0008332, 'z': 0.13779},
                'rotation': {'x': 0, 'y': 0, 'z': 0},
                'type': 'fixed'
            }
        }

    def forward_kinematics(self, joint_angles: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算末端执行器在给定关节角度下的位置与姿态
        返回: (位置xyz, 四元数xyzw)
        """
        T = np.eye(4)
        root_rotation = euler_matrix(-np.pi / 2, 0, 0)  # 从 Z-up 转为 Y-up
        T = T @ root_rotation

        for name in self.kinematic_chain:
            joint = self.joint_hierarchy[name]
            origin = np.array([joint['origin']['x'], joint['origin']['y'], joint['origin']['z']])
            rot_xyz = [joint['rotation'].get(k, 0) for k in ['x', 'y', 'z']]

            T_static = translation_matrix(origin) @ euler_matrix(*rot_xyz)
            T_joint = np.eye(4)

            if joint.get('type') == 'revolute' and name in joint_angles:
                axis = np.array([joint['axis'][k] for k in ['x', 'y', 'z']])
                T_joint[:3, :3] = rotation_matrix_from_axis_angle(axis, joint_angles[name])

            T = T @ T_static @ T_joint

        pos = T[:3, 3]
        quat = matrix_to_quaternion(T[:3, :3])
        return pos, quat