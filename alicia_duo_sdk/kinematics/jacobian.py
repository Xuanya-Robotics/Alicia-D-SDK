import numpy as np
from typing import List, Dict
from .fk import RobotArm

def compute_numerical_jacobian(
    robot_arm: RobotArm,
    joint_names: List[str],
    angles: np.ndarray,
    delta: float = 1e-6
) -> np.ndarray:
    """
    数值法计算 Jacobian 矩阵

    Args:
        robot_arm: RobotArm 实例
        joint_names: 关节名称列表
        angles: 当前角度（np.ndarray）
        delta: 微小扰动（用于数值微分）

    Returns:
        6 x N 的雅可比矩阵（位置3 + 姿态误差3）
    """
    J = np.zeros((6, len(joint_names)))
    f0_pos, f0_quat = robot_arm.forward_kinematics(dict(zip(joint_names, angles)))

    for i in range(len(joint_names)):
        angles_eps = angles.copy()
        angles_eps[i] += delta
        f1_pos, f1_quat = robot_arm.forward_kinematics(dict(zip(joint_names, angles_eps)))

        dp = (f1_pos - f0_pos) / delta
        dq = (f1_quat[:3] - f0_quat[:3]) / delta  # 四元数前3维近似误差

        J[:, i] = np.concatenate([dp, dq])
    return J
