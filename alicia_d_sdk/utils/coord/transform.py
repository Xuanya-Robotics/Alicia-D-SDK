"""
Fast transform utilities implemented with NumPy only.

Replaces SciPy-based rotations with lightweight NumPy implementations to
reduce overhead in hot loops (e.g., IK and Jacobian computations).
"""
import numpy as np


def translation_matrix(offset):
    """生成平移矩阵"""
    T = np.eye(4)
    T[:3, 3] = offset
    return T


def euler_matrix(rx, ry, rz, order='xyz'):
    """从欧拉角构建旋转矩阵 (默认 'xyz')"""
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0.0],
                   [sz, cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)

    if order != 'xyz':
        raise NotImplementedError("Only 'xyz' order is supported")

    Rm = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = Rm
    return T


def rotation_matrix_from_axis_angle(axis, angle):
    """从轴角生成旋转矩阵 (Rodrigues)"""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3, dtype=float)
    ax = axis / norm
    x, y, z = ax
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ], dtype=float)


def matrix_to_quaternion(matrix):
    """旋转矩阵转四元数，返回 [x, y, z, w]"""
    m = np.asarray(matrix, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q


def quaternion_to_matrix(quat: list) -> np.ndarray:
    """
    将四元数 [qx, qy, qz, qw] 转换为 3x3 旋转矩阵
    """
    x, y, z, w = map(float, quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=float)
