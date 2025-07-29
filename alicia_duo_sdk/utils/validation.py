# validation.py - 参数合法性检查模块

import numpy as np
from typing import List, Union

def validate_pose(pose: Union[List[float], tuple, np.ndarray]):
    """
    验证目标位姿是否合法（长度 7 + 数值 + 四元数归一化）

    Args:
        pose: [x, y, z, qx, qy, qz, qw]

    Raises:
        TypeError, ValueError
    """
    if not isinstance(pose, (list, tuple, np.ndarray)):
        raise TypeError("目标位姿必须是 list / tuple / ndarray")

    if len(pose) != 7:
        raise ValueError("目标位姿必须为长度为 7 的列表：[x, y, z, qx, qy, qz, qw]")

    if not all(isinstance(x, (int, float)) for x in pose):
        raise ValueError("目标位姿中的每个元素必须是数值类型")

    quat = np.array(pose[3:], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"四元数未归一化（范数 = {norm:.4f}），请提供归一化后的四元数")


def validate_pose_sequence(pose_sequence: List[List[float]]):
    """
    验证路径点序列是否合法

    Args:
        pose_sequence: List of pose vectors

    Raises:
        TypeError, ValueError
    """
    if not isinstance(pose_sequence, list):
        raise TypeError("pose_sequence 应为 List[List[float]]")
    for idx, pose in enumerate(pose_sequence):
        try:
            validate_pose(pose)
        except Exception as e:
            raise ValueError(f"第 {idx} 个姿态非法: {e}")


def validate_joint_list(joints: Union[List[float], tuple, np.ndarray]):
    """
    验证关节角度输入是否合法（长度为6 + 数值）

    Args:
        joints: 关节角度序列

    Raises:
        TypeError, ValueError
    """
    if not isinstance(joints, (list, tuple, np.ndarray)):
        raise TypeError("关节角输入应为 list / tuple / ndarray")

    if len(joints) != 6:
        raise ValueError("关节角输入长度必须为 6")

    if not all(isinstance(x, (int, float)) for x in joints):
        raise ValueError("关节角输入应为数值型")
