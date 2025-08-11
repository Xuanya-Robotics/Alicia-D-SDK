# advanced_ik_solver.py
import numpy as np
from typing import Dict, List, Tuple, Union
from .robot_model import AliciaFollower
from ..utils.logger import BeautyLogger
# Note: Avoid heavy SciPy Rotation in hot loops; use lightweight quaternion ops instead

logger = BeautyLogger(log_dir="./logs", log_name="ik.log", verbose=False)

class Advanced6DOFIKSolver:
    def __init__(self, robot_model: AliciaFollower, max_iters: int=150):
        self.robot_model = robot_model
        self.max_iters = max_iters
        self.position_tol = 0.005
        self.orientation_tol = 0.05
        self.step_size = 0.03  # 初始步长

        # 关节限制
        self.joint_limits = self.robot_model.joint_limit

    def solve(self, target_pos: np.ndarray, 
              target_quat: np.ndarray, 
              initial_angles: Dict[str, float]
              ) -> Dict[str, float]:
        
        current_angles = initial_angles.copy()
        best_angles = current_angles.copy()
        best_error = float('inf')

        for i in range(self.max_iters):
            current_pos, current_quat = self.robot_model.forward_kinematics(current_angles)

            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            ori_error = self._compute_orientation_error(target_quat, current_quat)
            ori_error_norm = np.linalg.norm(ori_error)

            total_error = pos_error_norm + 0.5 * ori_error_norm

            if total_error < best_error:
                best_error = total_error
                best_angles = current_angles.copy()

            if pos_error_norm < self.position_tol and ori_error_norm < self.orientation_tol:
                logger.info(f"[IK] 收敛于第{i}次迭代，"
                            f"位置误差: {pos_error_norm:.4f}，"
                            f"姿态误差: {np.degrees(ori_error_norm):.2f}°")
                
                return current_angles

            J = self._compute_6dof_jacobian(current_angles)
            error_vector = np.concatenate([pos_error, ori_error])
            delta_angles = self._compute_delta_angles(J, error_vector)

            step = self._compute_adaptive_step_size(pos_error_norm, ori_error_norm)

            for idx, joint_name in enumerate(current_angles.keys()):
                current_angles[joint_name] += delta_angles[idx] * step
                current_angles[joint_name] = self._apply_joint_limits(joint_name, current_angles[joint_name])

        logger.warning(f"[IK] 未完全收敛，返回最佳解。误差: {best_error:.4f}")

        return best_angles

    
    def _compute_orientation_error(self, target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        """
        计算姿态误差（轴角向量）
        Args:
            target_quat: 目标四元数 [x, y, z, w]
            current_quat: 当前四元数 [x, y, z, w]
        Returns:
            np.ndarray: 3D 姿态误差向量
        """
        q_error = self._quat_multiply(target_quat, self._quat_conjugate(current_quat))
        return self._quat_log_axis_angle(q_error)

    def _compute_6dof_jacobian(self, angles: Dict[str, float]) -> np.ndarray:
        """
        数值计算 6x6 雅可比矩阵，前3行为位置偏导，后3行为角速度偏导
        """
        joint_names = list(angles.keys())
        epsilon = 1e-5
        J = np.zeros((6, 6))

        base_pos, base_quat = self.robot_model.forward_kinematics(angles)

        for i, name in enumerate(joint_names):
            perturbed_pos = angles.copy()
            perturbed_neg = angles.copy()
            perturbed_pos[name] += epsilon
            perturbed_neg[name] -= epsilon

            pos_plus, quat_plus = self.robot_model.forward_kinematics(perturbed_pos)
            pos_minus, quat_minus = self.robot_model.forward_kinematics(perturbed_neg)

            dp = (pos_plus - pos_minus) / (2 * epsilon)
            dw = self._compute_angular_velocity(quat_plus, quat_minus, base_quat, epsilon)

            J[:, i] = np.concatenate([dp, dw])

        return J

    def _compute_angular_velocity(self, quat_pos, quat_neg, base_quat, epsilon) -> np.ndarray:
        """
        使用姿态扰动计算角速度近似
        """
        rel_pos = self._quat_multiply(quat_pos, self._quat_conjugate(base_quat))
        rel_neg = self._quat_multiply(quat_neg, self._quat_conjugate(base_quat))
        w_pos = self._quat_log_axis_angle(rel_pos)
        w_neg = self._quat_log_axis_angle(rel_neg)
        return (w_pos - w_neg) / (2 * epsilon)

    def _compute_delta_angles(self, J: np.ndarray, error_vector: np.ndarray) -> np.ndarray:
        """
        使用阻尼最小二乘法求解关节角度增量
        J: 6x6 雅可比矩阵
        error_vector: 6维误差向量
        """
        lambda_ = 0.01  # 阻尼因子
        JT = J.T
        JJT = J @ JT + lambda_ * np.eye(6)

        # 使用高性能的线性求解器代替手写矩阵求逆
        try:
            temp = np.linalg.solve(JJT, error_vector)
        except np.linalg.LinAlgError:
            logger.warning("[IK] JJT 奇异或病态，回退到简化近似")
            return JT @ error_vector * 0.1

        delta_angles = JT @ temp
        return delta_angles

    # -------------------------
    # Lightweight quaternion ops
    # Convention: quaternion is [x, y, z, w]
    # -------------------------
    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """
        :param q, np.ndarray: [x, y, z, w]
        :return: np.ndarray
        """
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        :param q1, np.ndarray: [x, y, z, w]
        :param q2, np.ndarray: [x, y, z, w]
        :return: np.ndarray
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
        x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
        y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
        z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
        return np.array([x, y, z, w], dtype=float)

    def _quat_log_axis_angle(self, q: np.ndarray) -> np.ndarray:
        """
        Map unit quaternion to axis-angle vector v where ||v|| = angle, direction = axis.
        :param q, np.ndarray: [x, y, z, w]
        :return: np.ndarray: [vx, vy, vz]
        """
        # Ensure unit length and consistent hemisphere
        q = q.astype(float)
        norm = np.linalg.norm(q)
        if norm == 0.0:
            return np.zeros(3)
        q = q / norm
        if q[3] < 0.0:
            q = -q
        w = float(np.clip(q[3], -1.0, 1.0))
        angle = 2.0 * np.arccos(abs(w))
        s = np.sqrt(max(1e-16, 1.0 - w * w))
        if angle < 1e-6:
            # Small-angle approximation
            return 2.0 * q[:3]
        axis = q[:3] / s
        return axis * angle

    # 旧版手写矩阵求逆已废弃，改用 np.linalg.solve 在 _compute_delta_angles 中直接求解
    
    def _compute_adaptive_step_size(self, pos_error: float, ori_error: float) -> float:
        """
        根据误差调整步长
        """
        norm_pos = pos_error / 0.01
        norm_ori = ori_error / 0.087  # ≈5度

        max_error = max(norm_pos, norm_ori)

        if max_error > 2.0:
            return self.step_size * 0.8
        elif max_error > 1.0:
            return self.step_size
        elif max_error > 0.5:
            return self.step_size * 1.2
        else:
            return self.step_size * 0.6
        
    def _apply_joint_limits(self, joint_name: str, angle: float) -> float:
        """
        限制角度在合法范围
        """
        low, high = self.joint_limits[joint_name]
        return max(low, min(high, angle))


