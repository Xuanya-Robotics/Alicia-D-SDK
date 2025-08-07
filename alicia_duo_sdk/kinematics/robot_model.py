# kinematics.py
import numpy as np
from typing import Dict, Tuple, List, Union
from ..utils import *

class AliciaFollower:
    def __init__(self, robot_model: str = 'alicia_5_4'):
        self.kinematic_chain = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'tool0'
        ]

        self.joint_hierarchy_alicia_D_5_4 = {
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
        
        # 关节限制
        self.joint_limits_alicia_D_5_4 = {
            'joint1': (-2.16, 2.16),
            'joint2': (-1.57, 1.57),
            'joint3': (-0.5, 2.35619),
            'joint4': (-3.14, 3.14),
            'joint5': (-1.57, 1.5),
            'joint6': (-3.14, 3.14)
        }

        if robot_model == 'alicia_5_4':
            self.joint_hierarchy = self.joint_hierarchy_alicia_D_5_4
            self.joint_limit = self.joint_limits_alicia_D_5_4
            
        self.base_pose = {
            'position': [0.0, 0.0, 0.0],           # 平移
            'rotation': [0.0, 0.0, 0.0],           # 欧拉角 (xyz, 单位：弧度)
            'flip_axes': [-1, -1, 1]                 # 每轴正负号，-1 表示取反
        }


    def forward_kinematics(self, 
                           joint_angles: Union[List[float], 
                                               np.ndarray, Dict[str, float]]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算末端执行器在给定关节角度下的位置与姿态
        返回: 7D numpy array [x, y, z, qx, qy, qz, qw]
        支持传入:
          - list/ndarray: [j1, j2, j3, j4, j5, j6]
          - dict: { 'joint1': a1, ... 'joint6': a6 }
        """
        # 1) 统一入参为 dict
        if isinstance(joint_angles, (list, tuple, np.ndarray)):
            if len(joint_angles) != 6:
                raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
            joint_angles = dict(zip(self.kinematic_chain[:-1], list(joint_angles)))
        elif not isinstance(joint_angles, dict):
            raise TypeError("joint_angles must be a sequence of length 6 or a dict of joint name -> angle")

        # 2) 正常 FK 计算
        # 旋转和平移构造初始变换
        base_T = translation_matrix(self.base_pose['position']) @ euler_matrix(*self.base_pose['rotation'])

        # 轴方向翻转矩阵
        flip = np.diag(self.base_pose['flip_axes'] + [1])  
        base_T = base_T @ flip

        T = base_T

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