# ik_controller.py
import numpy as np
from typing import Dict
from .ik import Advanced6DOFIKSolver

class IKController:
    def __init__(self, robot_arm):
        self.robot_arm = robot_arm
        self.ik_solver = Advanced6DOFIKSolver(robot_arm)

        self.current_joint_angles = {
            'joint1': 0, 'joint2': 0, 'joint3': 0,
            'joint4': 0, 'joint5': 0, 'joint6': 0
        }
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])  # x, y, z, w (scipy format)

    def set_target(self, position: np.ndarray, orientation: np.ndarray):
        self.target_position = position
        self.target_orientation = orientation

    def update_joint_angles(self, angles: Dict[str, float]):
        for k in self.current_joint_angles:
            if k in angles:
                self.current_joint_angles[k] = angles[k]

    def solve_ik(self) -> Dict[str, float]:
        return self.ik_solver.solve(
            self.current_joint_angles,
            self.target_position,
            self.target_orientation
        )