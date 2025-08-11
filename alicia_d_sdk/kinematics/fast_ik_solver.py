"""
Fast 6DOF IK solver using analytic geometric Jacobian.

Key ideas:
- Compute forward kinematics and all joint frames in one forward pass
  (no per-joint FK perturbations).
- Build geometric Jacobian J from joint axes and positions in base frame:
    Jv_i = z_i x (p_e - p_i), Jw_i = z_i
- Use quaternion log-map orientation error and damped least squares.
"""

from typing import Dict, Tuple, List
import numpy as np

from .robot_model import AliciaFollower
from ..utils.logger import BeautyLogger
from ..utils.coord.transform import (
    translation_matrix,
    euler_matrix,
    rotation_matrix_from_axis_angle,
    matrix_to_quaternion,
)


logger = BeautyLogger(log_dir="./logs", log_name="ik_fast.log", verbose=False)


class Fast6DOFIKSolver:
    def __init__(self, robot_model: AliciaFollower, max_iters: int = 80):
        """
        :param robot_model, AliciaFollower: robot model
        :param max_iters, int: maximum iterations
        :return: None
        """
        self.robot_model = robot_model
        self.max_iters = max_iters
        self.position_tol = 0.005
        self.orientation_tol = 0.05
        self.step_size = 0.03
        self.joint_names: List[str] = [f"joint{i}" for i in range(1, 7)]
        self.joint_limits = self.robot_model.joint_limit

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray, initial_angles: Dict[str, float]) -> Dict[str, float]:
        """
        :param target_pos, np.ndarray: [x, y, z]
        :param target_quat, np.ndarray: [qx, qy, qz, qw]
        :param initial_angles, dict: {joint1..joint6}
        :return: dict
        """
        q_dict = initial_angles.copy()
        best_q = q_dict.copy()
        best_err = float("inf")

        for _ in range(self.max_iters):
            # FK with frames for Jacobian
            p_e, q_e, p_axes, z_axes = self._forward_with_frames(q_dict)

            # Errors
            pos_error = target_pos - p_e
            pos_err_norm = float(np.linalg.norm(pos_error))

            ori_error_vec = self._quat_log_axis_angle(self._quat_multiply(target_quat, self._quat_conjugate(q_e)))
            ori_err_norm = float(np.linalg.norm(ori_error_vec))

            total_err = pos_err_norm + 0.5 * ori_err_norm
            if total_err < best_err:
                best_err = total_err
                best_q = q_dict.copy()

            if pos_err_norm < self.position_tol and ori_err_norm < self.orientation_tol:
                return q_dict

            # Geometric Jacobian
            J = np.zeros((6, 6), dtype=float)
            for i in range(6):
                z = z_axes[i]
                p_i = p_axes[i]
                J[:3, i] = np.cross(z, (p_e - p_i))
                J[3:, i] = z

            # Damped least squares
            error_vec = np.concatenate([pos_error, ori_error_vec])
            lambda_ = 0.01
            JT = J.T
            JJT = J @ JT + lambda_ * np.eye(6)
            try:
                temp = np.linalg.solve(JJT, error_vec)
                delta = JT @ temp
            except np.linalg.LinAlgError:
                delta = JT @ error_vec * 0.1

            step = self._adaptive_step(pos_err_norm, ori_err_norm)

            # Update
            for j, name in enumerate(self.joint_names):
                q_dict[name] += float(delta[j]) * step
                q_dict[name] = self._apply_joint_limits(name, q_dict[name])

        return best_q

    # ---------------- Geometric FK with frame extraction -----------------
    def _forward_with_frames(self, q_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        :param q_dict, dict: joint angles
        :return: (p_e, q_e, p_axes[6], z_axes[6])
        """
        # Base transform
        base_T = translation_matrix(self.robot_model.base_pose['position']) @ euler_matrix(*self.robot_model.base_pose['rotation'])
        flip = np.diag(self.robot_model.base_pose['flip_axes'] + [1])
        T = base_T @ flip

        p_axes: List[np.ndarray] = []  # joint origins
        z_axes: List[np.ndarray] = []  # joint axes in base frame

        # Iterate joints 1..6
        for name in self.joint_names:
            joint = self.robot_model.joint_hierarchy[name]
            origin = np.array([joint['origin']['x'], joint['origin']['y'], joint['origin']['z']], dtype=float)
            rot_xyz = [joint['rotation'].get(k, 0.0) for k in ['x', 'y', 'z']]

            T_static = translation_matrix(origin) @ euler_matrix(*rot_xyz)
            # Axis in local frame
            axis = np.array([joint['axis']['x'], joint['axis']['y'], joint['axis']['z']], dtype=float)

            # Position and axis in base frame BEFORE rotation of this joint
            T_pre = T @ T_static
            p_axes.append(T_pre[:3, 3].copy())
            z_axes.append((T_pre[:3, :3] @ axis).astype(float))

            # Apply joint rotation
            angle = float(q_dict.get(name, 0.0))
            R_joint = np.eye(4)
            R_joint[:3, :3] = rotation_matrix_from_axis_angle(axis, angle)
            T = T_pre @ R_joint

        # Fixed tool link
        tool = self.robot_model.joint_hierarchy['tool0']
        origin = np.array([tool['origin']['x'], tool['origin']['y'], tool['origin']['z']], dtype=float)
        rot_xyz = [tool['rotation'].get(k, 0.0) for k in ['x', 'y', 'z']]
        T = T @ translation_matrix(origin) @ euler_matrix(*rot_xyz)

        p_e = T[:3, 3].copy()
        q_e = matrix_to_quaternion(T[:3, :3])
        return p_e, q_e, p_axes, z_axes

    # ---------------- Quaternion helpers -----------------
    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
        x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
        y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
        z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
        return np.array([x, y, z, w], dtype=float)

    def _quat_log_axis_angle(self, q: np.ndarray) -> np.ndarray:
        q = q.astype(float)
        n = np.linalg.norm(q)
        if n == 0.0:
            return np.zeros(3)
        q = q / n
        if q[3] < 0.0:
            q = -q
        w = float(np.clip(q[3], -1.0, 1.0))
        angle = 2.0 * np.arccos(abs(w))
        s = np.sqrt(max(1e-16, 1.0 - w * w))
        if angle < 1e-6:
            return 2.0 * q[:3]
        axis = q[:3] / s
        return axis * angle

    # ---------------- Step & limits -----------------
    def _adaptive_step(self, pos_err: float, ori_err: float) -> float:
        norm_pos = pos_err / 0.01
        norm_ori = ori_err / 0.087
        m = max(norm_pos, norm_ori)
        if m > 2.0:
            return self.step_size * 0.8
        elif m > 1.0:
            return self.step_size
        elif m > 0.5:
            return self.step_size * 1.2
        else:
            return self.step_size * 0.6

    def _apply_joint_limits(self, joint_name: str, angle: float) -> float:
        low, high = self.joint_limits[joint_name]
        return max(low, min(high, angle))


