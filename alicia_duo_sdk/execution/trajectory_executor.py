import time
from typing import List
import numpy as np
from ..utils import *
from ..kinematics import RobotArm

logger = BeautyLogger(log_dir="./logs", log_name="execution.log", verbose=True)

class TrajectoryExecutor:
    def __init__(self, joint_controller, delay: float = 0.02):
        self.joint_controller = joint_controller
        self.delay = delay

    def execute(self, trajectory: List[List[float]], 
                robot_model: RobotArm, visualize: bool = False, 
                show_ori: bool = False):
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
                ee_pose = []
                for joint_angles in trajectory:
                    pos, quat = robot_model.forward_kinematics(
                        dict(zip(robot_model.kinematic_chain[:-1], joint_angles))
                    )
                    ee_pose.append(np.concatenate([pos, quat]))

                plot_3d(data_lst=ee_pose, interval=int(len(ee_pose)/10), show_ori=show_ori)

        logger.module("[executor]按下回车执行轨迹，按下 q 取消：")
        usr_input = input()
        if usr_input.lower() == 'q':
            return False
        for point in trajectory:
            self.joint_controller.set_joint_angles(point)
            time.sleep(self.delay)
