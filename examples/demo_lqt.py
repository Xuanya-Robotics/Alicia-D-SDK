# alicia_duo_sdk/examples/demo_lqt.py

import numpy as np
from alicia_duo_sdk.utils.config import get_planning_config
from alicia_duo_sdk.planning.planners.lqt import LQT
from alicia_duo_sdk.utils import *

if __name__ == "__main__":
    # 设置一组目标关键点
    via_points = [
        [-0.3,  0.00,  0.10, 0.0014, -0.93, -0.001, 0.36],
        [-0.25, -0.10, 0.30, 0.0014, -0.93, -0.001, 0.36],
        [-0.30,  0.10, 0.35, 0.0014, -0.93, -0.001, 0.36],
        [-0.28,  0.00, 0.15, 0.0014, -0.93, -0.001, 0.36]
    ]

    # 加载参数并执行LQT求解
    cfg = get_planning_config("lqt")
    planner = LQT(via_points, cfg)
    u_hat, x_hat, mu, idx_slices, _ = planner.solve()

    # 可视化末端轨迹
    plot_3d([x_hat], show_ori=True, legend="LQT", title="LQT Cartesian Trajectory", axis_length=0.04)

    # 可视化关节轨迹（如果有）
    if mu is not None and mu.shape[1] >= 7:
        plot_joint_angles(mu[:, :7], title="LQT Joint Angles")
