# trajectory_executor.py (in execution/)

import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from ..visualization.trajectory_viz import plot_joint_angles, plot_ee_path

class TrajectoryExecutor:
    def __init__(self, joint_controller, delay: float = 0.02):
        self.joint_controller = joint_controller
        self.delay = delay

    def execute(self, trajectory: List[List[float]], robot_model=None, visualize: bool = False):
        """
        执行关节角度轨迹，可选可视化
        :param trajectory: List of joint angle lists
        :param robot_model: 如果提供，则可视化末端轨迹
        :param visualize: 是否执行前绘图
        """
        traj_np = np.array(trajectory)
        if visualize:
            plot_joint_angles(traj_np)
            if robot_model:
                ee_positions = []
                for joint_angles in trajectory:
                    pos, _ = robot_model.forward_kinematics(
                        dict(zip(robot_model.kinematic_chain[:-1], joint_angles))
                    )
                    ee_positions.append(pos)
                plot_ee_path(np.array(ee_positions))
           
        input("按回车执行轨迹，或 Ctrl+C 取消：")

        for point in trajectory:
            self.joint_controller.set_joint_angles(point)
            time.sleep(self.delay)
