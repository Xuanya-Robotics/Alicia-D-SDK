# trajectory_executor.py
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from alicia_duo_sdk.motor import ArmController

class TrajectoryExecutor:
    def __init__(self, joint_controller: ArmController, arm: str = 'left_arm', rate_hz: float = 50.0):
        self.joint_controller = joint_controller
        self.arm = arm
        self.rate = 1.0 / rate_hz

    def execute(self, trajectory: List[List[float]], blocking: bool = True, visualize: bool = False):
        """
        执行关节角度序列
        :param trajectory: List of joint angle lists
        :param blocking: 如果为True，则阻塞直到执行完成
        :param visualize: 是否执行前绘制关节轨迹图
        """
        if visualize:
            self.visualize_trajectory(trajectory)
            user_input = input("是否执行轨迹？按回车继续，输入q取消：")
            if user_input.strip().lower() == 'q':
                return

        if blocking:
            for point in trajectory:
                self.joint_controller.set_joint_angles(self.arm, point)
                time.sleep(self.rate)
        else:
            raise NotImplementedError("非阻塞模式暂未实现")

    def vis_traj(self, trajectory: List[List[float]], title="IK 轨迹关节角度"):
        joint_traj = np.array(trajectory)
        time_steps = np.arange(len(joint_traj))
        fig, axs = plt.subplots(6, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(title)
        for j in range(6):
            axs[j].plot(time_steps, joint_traj[:, j])
            axs[j].set_ylabel(f"Joint {j+1} (rad)")
            axs[j].grid(True)
        axs[-1].set_xlabel("Step")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
