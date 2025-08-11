# alicia_d_sdk/utils/visualization/trajectory_plot.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from alicia_d_sdk.utils.coord import quaternion_to_matrix
from ..logger import BeautyLogger

logger = BeautyLogger(log_dir="./logs", log_name="plot.log", verbose=True)

def plot_joint_angles(joint_traj: np.ndarray, title: str = "Joint Trajectory"):
    logger.module("[vislab]开始绘制关节轨迹图")
    plt.figure(figsize=(10, 4))
    for i in range(joint_traj.shape[1]):
        plt.plot(joint_traj[:, i], label=f'Joint {i+1}')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def plot_3d(data_lst: List[List[List[float]]], legend: Optional[str] = None, title: Optional[str] = None,
            show_ori: bool = False, interval: int = 20, axis_length: float = 0.03):
    """
    可视化末端执行器轨迹（3D），支持多个轨迹输入和方向箭头

    Args:
        data_lst: List of List of [x, y, z] 或 [x, y, z, qx, qy, qz, qw]
        legend: 图例前缀
        title: 图标题
        show_ori: 是否绘制姿态方向箭头（仅当数据为 7D 时）
        interval: 姿态箭头绘制间隔
        axis_length: 坐标系箭头长度
    """
    logger.module("[vislab]开始绘制末端位姿3D轨迹图")

    # 如果是单条轨迹，自动包装成列表
    if isinstance(data_lst[0][0], (int, float)):
        data_lst = [data_lst]
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', fc='white')

    for i, data in enumerate(data_lst):
        data_np = np.array(data)
        label = f"{legend} {i+1}" if legend else None
        ax.plot(data_np[:, 0], data_np[:, 1], data_np[:, 2], label=label)

        # 绘制起止点（带图例）
        ax.scatter(data_np[0, 0], data_np[0, 1], data_np[0, 2], s=20, c='g', label=f"{label} Start" if label else "Start")
        ax.scatter(data_np[-1, 0], data_np[-1, 1], data_np[-1, 2], s=20, c='r', marker='x', label=f"{label} End" if label else "End")

        # 绘制姿态箭头（若为 7D）
        if show_ori and data_np.shape[1] >= 7:
            for t in range(0, len(data_np), interval):
                pos = data_np[t, :3]
                quat = data_np[t, 3:7]
                R = quaternion_to_matrix(quat)
                _draw_axes(ax, pos, R, length=axis_length)
            # 补上终点姿态箭头
            pos = data_np[-1, :3]
            quat = data_np[-1, 3:7]
            R = quaternion_to_matrix(quat)
            _draw_axes(ax, pos, R, length=axis_length)

    ax.set_title(title or "3D POSE Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if legend:
        ax.legend()
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

def _draw_axes(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.03):
    for i, color in zip(range(3), ['r', 'g', 'b']):
        vec = R[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=1.0, alpha=0.8
        )

def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


# ======= DEMO =======
if __name__ == "__main__":
    # 生成一条简单的7D轨迹
    N = 100
    x = np.linspace(0.1, 0.4, N)
    y = np.sin(np.linspace(0, 2 * np.pi, N)) * 0.1
    z = np.linspace(0.1, 0.3, N)
    pos_traj = np.stack([x, y, z], axis=1)
    quat = np.array([0, 0, 0, 1])  # 恒定方向
    quat_traj = np.tile(quat, (N, 1))
    pose_traj = np.concatenate([pos_traj, quat_traj], axis=1)

    # 可视化
    plot_3d([pose_traj], legend="demo", show_ori=True, title="Demo 3D Trajectory", axis_length=0.08)
