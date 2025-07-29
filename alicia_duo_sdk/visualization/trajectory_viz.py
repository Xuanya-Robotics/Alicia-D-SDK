# trajectory_viz.py
import matplotlib.pyplot as plt
import numpy as np

def plot_joint_angles(joint_traj: np.ndarray, title: str = "Joint Trajectory"):
    """
    画出关节角度轨迹（6关节）

    Args:
        joint_traj: shape = [N, 6]
    """
    plt.figure(figsize=(10, 4))
    for i in range(joint_traj.shape[1]):
        plt.plot(joint_traj[:, i], label=f'Joint {i+1}')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ee_path(pos_traj: np.ndarray):
    """
    画出末端执行器的空间轨迹

    Args:
        pos_traj: shape = [N, 3]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], label="EE Path")
    ax.scatter(pos_traj[0, 0], pos_traj[0, 1], pos_traj[0, 2], c='g', label='Start')
    ax.scatter(pos_traj[-1, 0], pos_traj[-1, 1], pos_traj[-1, 2], c='r', label='End')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("End-Effector Path")
    ax.legend()
    plt.tight_layout()
    plt.show()
