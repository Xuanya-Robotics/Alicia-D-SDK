# utils/trajectory.py

import numpy as np

def invert_6x6_matrix(A: np.ndarray) -> np.ndarray:
    """ 高斯-约旦法求解 6x6 矩阵逆 """
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
                aug[j] -= aug[j, i] * aug[i]

    return aug[:, n:]

def compute_adaptive_step_size(pos_error: float, ori_error: float, base_step: float) -> float:
    """ 根据误差调整步长 """
    norm_pos = pos_error / 0.01
    norm_ori = ori_error / 0.087  # ≈5度
    max_error = max(norm_pos, norm_ori)

    if max_error > 2.0:
        return base_step * 0.8
    elif max_error > 1.0:
        return base_step
    elif max_error > 0.5:
        return base_step * 1.2
    else:
        return base_step * 0.6
