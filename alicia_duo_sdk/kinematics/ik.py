# ik_solver.py
import numpy as np
from typing import List, Dict
from .fk import RobotArm
from .jacobian import compute_numerical_jacobian


class Advanced6DOFIKSolver:
    def __init__(self, robot_arm: RobotArm, max_iters=100, tol=1e-4):
        self.robot_arm = robot_arm
        self.max_iters = max_iters
        self.tol = tol
        self.damping = 1e-3

        self.joint_limits = [
                            (-2.16, 2.16),  # joint1
                            (-1.57, 1.57),  # joint2
                            (-0.5, 2.36),  # joint3
                            (-3.14, 3.14),  # joint4
                            (-1.57, 1.57),  # joint5
                            (-3.14, 3.14),  # joint6
                        ]

    def solve(self, current_angles: List[float], target_pos: np.ndarray, target_quat: np.ndarray) -> List[float]:
        joint_names = self.robot_arm.kinematic_chain[:-1]  # exclude tool0

        if len(current_angles) != len(joint_names):
            raise ValueError(f"Expected {len(joint_names)} joint angles, got {len(current_angles)}")

        angles = np.array(current_angles)

        for _ in range(self.max_iters):
            est_pos, est_quat = self.robot_arm.forward_kinematics(dict(zip(joint_names, angles)))
            dp = target_pos - est_pos

            # Quaternion error (approximate)
            dq = target_quat[:3] - est_quat[:3]  # naive orientation error (skip full SO(3) math)

            error = np.concatenate([dp, dq])
            if np.linalg.norm(error) < self.tol:
                break

            J = compute_numerical_jacobian(self.robot_arm, joint_names, angles)
            JT = J.T
            dtheta = JT @ np.linalg.inv(J @ JT + self.damping * np.eye(6)) @ error
            angles += dtheta

        angles = np.array(angles)

        # 将所有角度归一化到 [-π, π]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        # 然后按限位进行归一化裁剪
        joint_lows = np.array([lim[0] for lim in self.joint_limits])
        joint_highs = np.array([lim[1] for lim in self.joint_limits])
        joint_ranges = joint_highs - joint_lows

        normalized = (angles - joint_lows) / joint_ranges
        normalized_clipped = np.clip(normalized, 0.0, 1.0)
        clipped_angles = joint_lows + normalized_clipped * joint_ranges

        # 对 joint4 和 joint6 做对称平均处理
        offset = (clipped_angles[3] + clipped_angles[5]) / 2
        clipped_angles[3] = offset
        clipped_angles[5] = offset

        return clipped_angles
      

   