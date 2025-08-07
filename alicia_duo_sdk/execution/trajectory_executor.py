import time
from typing import List
import numpy as np
from ..utils import *
from ..kinematics import AliciaFollower
from ..driver import ArmController
from scipy.spatial.transform import Rotation as R

logger = BeautyLogger(log_dir="./logs", log_name="execution.log", verbose=True)

class TrajectoryExecutor:
    def __init__(self,
                 joint_controller: ArmController,
                 robot_model: AliciaFollower):
        self.joint_controller = joint_controller
        self.robot_model = robot_model
        self.delay = 0.02

    def execute(self, 
                joint_traj: List[List[float]], 
                pose_traj: List[List[float]] = None,
                visualize: bool = False, 
                show_ori: bool = False):
        """
        执行关节角度轨迹，可选可视化
        :param joint_traj: List of joint angle lists
        :param robot_model: 如果提供，则可视化末端轨迹
        :param visualize: 是否执行前绘图
        """
        traj_np = np.array(joint_traj)

        if visualize:
            plot_joint_angles(traj_np)
            if pose_traj:
                plot_3d(data_lst=pose_traj,
                        interval=int(len(pose_traj)/10), 
                        show_ori=show_ori)

        logger.module("[executor]按下回车执行轨迹，按下 q 取消：")
        usr_input = input()

        if usr_input.lower() == 'q':
            logger.info("[executor]取消执行轨迹")
            return False
        
        for point in joint_traj:
            self.joint_controller.set_joint_angles(point)
            time.sleep(self.delay)
        
        if pose_traj:
            # 执行完成后，计算末端误差
            final_target_pose = pose_traj[-1]
            actual_joint_angles = self.joint_controller.get_joint_angles()
            actual_pos, actual_quat = self.robot_model.forward_kinematics(actual_joint_angles)

            # 误差计算
            pos_error = np.linalg.norm(np.array(final_target_pose[:3]) - actual_pos)

            target_rot = R.from_quat(final_target_pose[3:])
            actual_rot = R.from_quat(actual_quat)
            rot_diff = target_rot.inv() * actual_rot
            angle_error_deg = np.degrees(rot_diff.magnitude())

            logger.info(f"[executor] 轨迹执行完成")
            logger.info(f"[executor] 末端位置误差: {pos_error:.4f} m")
            logger.info(f"[executor] 姿态误差: {angle_error_deg:.2f}°")
