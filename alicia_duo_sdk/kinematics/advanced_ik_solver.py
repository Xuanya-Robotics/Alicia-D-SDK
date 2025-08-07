# advanced_ik_solver.py
import numpy as np
from typing import Dict, List, Tuple, Union
from .robot_model import AliciaFollower
from ..utils.logger import BeautyLogger
from scipy.spatial.transform import Rotation as R

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
        q_target = R.from_quat(target_quat)
        q_current = R.from_quat(current_quat)
        q_error = q_target * q_current.inv()

        angle = q_error.magnitude()
        if angle < 1e-6:
            return np.zeros(3)

        axis = q_error.as_rotvec()  # axis * angle
        return axis

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
        q_pos = R.from_quat(quat_pos)
        q_neg = R.from_quat(quat_neg)
        q_base = R.from_quat(base_quat)

        rel_pos = q_pos * q_base.inv()
        rel_neg = q_neg * q_base.inv()

        w_pos = rel_pos.as_rotvec()
        w_neg = rel_neg.as_rotvec()

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

        try:
            JJT_inv = self._invert_6x6_matrix(JJT)
        except np.linalg.LinAlgError:
            logger.warning("[IK] JJT 不可逆，使用简化近似")
            return JT @ error_vector * 0.1

        temp = JJT_inv @ error_vector
        delta_angles = JT @ temp
        return delta_angles

    def _invert_6x6_matrix(self, A: np.ndarray) -> np.ndarray:
        """
        高斯-约旦法求 6x6 矩阵逆
        """
        A = A.astype(np.float64)
        n = 6
        I = np.eye(n)
        aug = np.hstack((A, I))

        for i in range(n):
            max_row = np.argmax(np.abs(aug[i:, i])) + i
            if aug[max_row, i] == 0:
                raise np.linalg.LinAlgError("矩阵不可逆")
            aug[[i, max_row]] = aug[[max_row, i]]
            aug[i] = aug[i] / aug[i, i]

            for j in range(n):
                if j != i:
                    aug[j] = aug[j] - aug[j, i] * aug[i]

        return aug[:, n:]
    
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


