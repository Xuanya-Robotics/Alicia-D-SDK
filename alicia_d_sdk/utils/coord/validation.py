# validation.py - 参数合法性检查模块

import numpy as np
from typing import List, Union, Tuple, Dict 

def validate_pose(pose: List[float]):
    """
    验证目标位姿是否合法（长度 7 + 数值 + 四元数归一化）

    Args:
        pose: [x, y, z, qx, qy, qz, qw]

    Raises:
        TypeError, ValueError
    """
    if not isinstance(pose, list):
        raise TypeError("目标位姿必须是 list")

    if len(pose) != 7:
        raise ValueError("目标位姿必须为长度为 7 的列表：[x, y, z, qx, qy, qz, qw]")

    if not all(isinstance(x, (int, float)) for x in pose):
        raise ValueError("目标位姿中的每个元素必须是数值类型")

    quat = np.array(pose[3:], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"四元数未归一化（范数 = {norm:.4f}），请提供归一化后的四元数")


def validate_waypoints(waypoints: List[List[float]]):
    """
    验证路径点序列是否合法

    Args:
        waypoints: List of pose vectors，每个为 [x, y, z, qx, qy, qz, qw]，
                   或包含夹爪的 [x, y, z, qx, qy, qz, qw, gripper]

    Raises:
        TypeError, ValueError
    """
    if not isinstance(waypoints, list):
        raise TypeError("waypoints 应为 List[List[float]] 或 List[float]")

    for idx, pose in enumerate(waypoints):
        if not isinstance(pose, list):
            raise TypeError(f"第 {idx+1} 个路径点不是列表: {pose}")

        if len(pose) not in (7, 8):
            raise ValueError(f"第 {idx+1} 个路径点长度不是 7 或 8：{pose}")

        try:
            # 前 7 维始终是位姿
            validate_pose(pose[:7])
        except Exception as e:
            raise ValueError(f"第 {idx+1} 个路径点内容非法: {e}")

        if len(pose) == 8:
            # 第 8 个为夹爪角度（弧度），仅做数值检查
            gr = pose[7]
            if not isinstance(gr, (int, float)):
                raise ValueError(f"第 {idx+1} 个路径点的夹爪值不是数值: {gr}")



def validate_joint_list(joints: Union[List[float], np.ndarray]):
    """
    验证关节角度输入是否合法（长度为6 + 数值）

    Args:
        joints: 关节角度序列

    Raises:
        TypeError, ValueError
    """
    if not isinstance(joints, (list, np.ndarray)):
        raise TypeError("关节角输入应为 list / ndarray")

    if len(joints) != 6:
        raise ValueError("关节角输入长度必须为 6")

    if not all(isinstance(x, (int, float)) for x in joints):
        raise ValueError("关节角输入应为数值型")

def check_and_clip_joint_limits(
    joints: List[float],
    joint_limits: Dict[str, Tuple[float, float]],
    joint_names: List[str] = None
) -> Tuple[List[float], List[Tuple[str, float, float]]]:
    """
    归一化关节角度并检查是否超过 joint_limits，超限的角度会被截断

    Args:
        joints: 当前关节角度列表
        joint_limits: 各关节限制范围，如 {'joint1': (-2.0, 2.0), ...}
        joint_names: 对应的关节名顺序，默认按 joint_limits 顺序排列

    Returns:
        clipped_joints: 修正后的角度列表
        violations: List of (joint_name, original_val, clipped_val)
    """
    if joint_names is None:
        joint_names = list(joint_limits.keys())

    clipped = joints.copy()
    violations = []

    for i, name in enumerate(joint_names):
        low, high = joint_limits[name]
        raw_val = joints[i]

        # 归一化至 [-π, π]
        norm_val = (raw_val + np.pi) % (2 * np.pi) - np.pi

        # 检查joint limit
        if norm_val < low:
            clipped[i] = low
            violations.append((name, raw_val, low))
        elif norm_val > high:
            clipped[i] = high
            violations.append((name, raw_val, high))
        else:
            clipped[i] = norm_val  # 如果未超限则返回归一后的数据

    return clipped, violations
