"""
Compare IK solving speed between the original and optimized solvers.
"""

import time
import random
import numpy as np

from alicia_d_sdk.kinematics.robot_model import AliciaFollower
from alicia_d_sdk.kinematics.advanced_ik_solver import Advanced6DOFIKSolver
from alicia_d_sdk.kinematics.fast_ik_solver import Fast6DOFIKSolver


def random_pose_near(p0: np.ndarray, q0: np.ndarray, pos_noise: float = 0.02, ori_noise_deg: float = 5.0):
    """
    :param p0, np.ndarray: base position [3]
    :param q0, np.ndarray: base quaternion [4]
    :param pos_noise, float: position noise amplitude (m)
    :param ori_noise_deg, float: orientation noise amplitude (deg)
    :return: tuple
    """
    p = p0 + (np.random.rand(3) - 0.5) * 2.0 * pos_noise
    angle = np.deg2rad((random.random() - 0.5) * 2.0 * ori_noise_deg)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-12
    q_noise = axis_angle_to_quat(axis, angle)
    q = quat_multiply(q_noise, q0)
    return p, q


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    :param axis, np.ndarray: [3]
    :param angle, float: rad
    :return: np.ndarray: [x,y,z,w]
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    :param q1, np.ndarray: [x,y,z,w]
    :param q2, np.ndarray: [x,y,z,w]
    :return: np.ndarray
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
    y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
    z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
    return np.array([x, y, z, w], dtype=float)


def quat_angle_error_deg(q_target: np.ndarray, q_current: np.ndarray) -> float:
    """
    :param q_target, np.ndarray: [x,y,z,w]
    :param q_current, np.ndarray: [x,y,z,w]
    :return, float: angle error in degree
    """
    # relative rotation
    x1, y1, z1, w1 = q_target
    x2, y2, z2, w2 = q_current
    # q_rel = q_target * conj(q_current)
    q_conj = np.array([-x2, -y2, -z2, w2], dtype=float)
    xr, yr, zr, wr = quat_multiply(q_target, q_conj)
    wr = float(np.clip(abs(wr), -1.0, 1.0))
    angle = 2.0 * np.arccos(wr)
    return float(np.degrees(angle))


def main():
    """
    :return: None
    """
    model = AliciaFollower()

    # Initial angles near zero
    init = {f"joint{i}": 0.0 for i in range(1, 7)}
    p0, q0 = model.forward_kinematics(init)

    # Prepare a batch of random targets near current pose
    N = 100
    targets = [random_pose_near(p0, q0) for _ in range(N)]

    # Original solver (time excludes FK used for error calculation)
    original = Advanced6DOFIKSolver(model, max_iters=150)
    init = {f"joint{i}": 0.0 for i in range(1, 7)}
    t_original = 0.0
    pos_errs_orig, ori_errs_orig = [], []
    for p, q in targets:
        tb = time.time()
        sol = original.solve(p, q, init)
        te = time.time()
        t_original += (te - tb)
        init = sol  # warm start
        # accuracy
        p_act, q_act = model.forward_kinematics(sol)
        pos_errs_orig.append(float(np.linalg.norm(p - p_act)))
        ori_errs_orig.append(quat_angle_error_deg(q, q_act))


    # Fast geometric solver
    fast = Fast6DOFIKSolver(model, max_iters=80)
    init = {f"joint{i}": 0.0 for i in range(1, 7)}
    t_fast = 0.0
    pos_errs_fast, ori_errs_fast = [], []
    for p, q in targets:
        tb = time.time()
        sol = fast.solve(p, q, init)
        te = time.time()
        t_fast += (te - tb)
        init = sol
        p_act, q_act = model.forward_kinematics(sol)
        pos_errs_fast.append(float(np.linalg.norm(p - p_act)))
        ori_errs_fast.append(quat_angle_error_deg(q, q_act))

    print("=== IK Speed & Accuracy Comparison ===")
    print(f"N targets: {N}")
    print(f"Original solver:  {t_original:.4f} s  ({t_original/N:.4f} s/target)")
    print(f"  pos_err  mean/max: {np.mean(pos_errs_orig):.4f} / {np.max(pos_errs_orig):.4f} m")
    print(f"  ori_err° mean/max: {np.mean(ori_errs_orig):.2f} / {np.max(ori_errs_orig):.2f} deg")
    print(f"Fast solver:      {t_fast:.4f} s  ({t_fast/N:.4f} s/target)")
    print(f"  pos_err  mean/max: {np.mean(pos_errs_fast):.4f} / {np.max(pos_errs_fast):.4f} m")
    print(f"  ori_err° mean/max: {np.mean(ori_errs_fast):.2f} / {np.max(ori_errs_fast):.2f} deg")
    speedup_fast = t_original / max(1e-9, t_fast)
    print(f"Speedup (fast): {speedup_fast:.2f}x")


if __name__ == "__main__":
    main()


