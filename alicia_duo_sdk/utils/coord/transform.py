# utils/transform.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def translation_matrix(offset):
    """生成平移矩阵"""
    T = np.eye(4)
    T[:3, 3] = offset
    return T

def euler_matrix(rx, ry, rz, order='xyz'):
    """从欧拉角构建旋转矩阵"""
    r = R.from_euler(order, [rx, ry, rz])
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    return T

def rotation_matrix_from_axis_angle(axis, angle):
    """从轴角生成旋转矩阵"""
    axis = np.asarray(axis)
    r = R.from_rotvec(axis * angle)
    return r.as_matrix()

def matrix_to_quaternion(matrix):
    """旋转矩阵转四元数 (x, y, z, w)"""
    r = R.from_matrix(matrix)
    return r.as_quat()

def quaternion_to_matrix(quat: list) -> np.ndarray:
    """
    将四元数 [qx, qy, qz, qw] 转换为 3x3 旋转矩阵
    Args:
        quat: List[float] of length 4
    Returns:
        np.ndarray: shape (3, 3)
    """
    r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
    return r.as_matrix()
