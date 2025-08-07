from typing import List, Union
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from alicia_duo_sdk.kinematics import AliciaFollower
from ...utils.logger import BeautyLogger
import time

class CartesianPlanner:
    def __init__(self, verbose: bool=False):
        self.logger = BeautyLogger(log_dir="./logs", 
                                   log_name="cartesian_planner.log", 
                                   verbose=verbose)

    def plan(
            self,
            start_joint_angles: List[float],
            robot_model: AliciaFollower,
            waypoints: List[List[float]],
            ) -> List[List[float]]:
        """
        输入一系列末端姿态点，依次插值每段轨迹，返回插值后的末端轨迹
        """
        self.logger.module("[CartesianPlanner]开始POSE轨迹插值规划")
        
        t0 = time.time()   
        pose_traj = []

        cur_pose = np.concatenate(robot_model.forward_kinematics(start_joint_angles)).tolist()
        all_points = [cur_pose] + waypoints  # 加入起点

        # 对waypoints的每两个点之间插值
        for i in range(len(all_points) - 1):
            p1, p2 = np.array(all_points[i]), np.array(all_points[i + 1])
            steps = self._estimate_steps_between_poses(p1, p2)
            interp_poses = self.interpolate_pose_trajectory(p1, p2, num_steps=steps)
               
            if i == 0:
                pose_traj.extend(interp_poses)
            else:
                pose_traj.extend(interp_poses[1:])
        
        t1 = time.time()
        self.logger.info("[CartesianPlanner]完成POSE轨迹插值规划, "
                    f"规划用时{t1-t0: .2f}, 共{len(pose_traj)}个轨迹点")
        
        return pose_traj    

    def _estimate_steps_between_poses(self, 
                                      pose1: np.ndarray, 
                                      pose2: np.ndarray, 
                                      base_steps: int = 50,
                                      max_steps: int = 100) -> int:
        """
        根据两个姿态之间的差异估算所需的插值步数
        """
        pos1, quat1 = pose1[:3], pose1[3:]
        pos2, quat2 = pose2[:3], pose2[3:]

        # 位置差（欧几里得距离）
        pos_diff = np.linalg.norm(pos1 - pos2)

        # 姿态差（旋转角度）
        q1 = R.from_quat(quat1)
        q2 = R.from_quat(quat2)
        angle_diff = q1.inv() * q2
        ori_diff_deg = np.degrees(angle_diff.magnitude())

        # 插值步数比例
        step_pos = pos_diff * 100  # 每 1m 差异对应 100 步
        step_ori = ori_diff_deg    # 每 1 度对应 1 步

        steps = int(base_steps + step_pos + step_ori)
        return max(5, min(steps, max_steps))  # 限制步数在范围内

    def interpolate_pose_trajectory(self, 
                                     pose_start: np.ndarray, 
                                     pose_end: np.ndarray, 
                                     num_steps: int) -> List[List[float]]:
        """
        线性插值两个末端姿态（位置 + 四元数）
        """
        pos_start, quat_start = pose_start[:3], pose_start[3:]
        pos_end, quat_end = pose_end[:3], pose_end[3:]

        if np.dot(quat_start, quat_end) < 0.0:
            quat_end = -quat_end

        key_rots = R.from_quat([quat_start, quat_end])
        slerp = Slerp([0, 1], key_rots)

        pose_traj = []
        for i in range(num_steps):
            alpha = self.smoothstep(i / (num_steps - 1))
            pos = (1 - alpha) * pos_start + alpha * pos_end
            quat = slerp([alpha])[0].as_quat()
            pose_traj.append(np.concatenate([pos, quat]).tolist())

        return pose_traj

    @staticmethod
    def smoothstep(t: float) -> float:
        return 3 * t**2 - 2 * t**3
