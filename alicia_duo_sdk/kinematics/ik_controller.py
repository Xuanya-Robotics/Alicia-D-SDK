# ik_controller.py
import numpy as np
from typing import Dict, List, Union
from .advanced_ik_solver import Advanced6DOFIKSolver
from .robot_model import AliciaFollower
from ..utils.logger import BeautyLogger
import time

logger = BeautyLogger(log_dir="./logs", log_name="ik_controller.log", verbose=True)

JointInput = Union[List[float], np.ndarray, Dict[str, float]]
PoseInput = Union[List[float], List[List[float]]]

class IKController:
    def __init__(self, robot_model: AliciaFollower):
        self.robot_model = robot_model
        self.ik_solver = Advanced6DOFIKSolver(robot_model)
        self._target_pos = None
        self._target_quat = None

    def set_target(self, pos: List[float], quat: List[float]):
        """
        设置末端目标位姿
        Args:
            position: [x, y, z]
            orientation: [qx, qy, qz, qw]
        """
        if len(pos) != 3:
            raise ValueError("位置必须是长度为 3 的列表")
        if len(quat) != 4:
            raise ValueError("姿态必须是长度为 4 的四元数")
        self._target_pos = np.array(pos)
        self._target_quat = np.array(quat)

    def solve_ik(self, 
                initial_angles: JointInput, 
                pose_traj: PoseInput,
                output_format: str = 'as_list'
                ) -> Union[List[float], List[List[float]]]:
        """
        执行 IK 解算，自动支持单个或多个末端姿态点

        Args:
            initial_angles: 初始角度，支持 list / np.array / dict
            pose_traj: 单个姿态点 [x, y, z, qx, qy, qz, qw] 或多个点的列表
            output_format: 'as_list' 或 'as_dict'

        Returns:
            单个POSE解: List[float] 或 Dict[str, float]
            POSE轨迹解: List[List[float]] 或 List[Dict[str, float]]
        """
        initial_angles = self._convert_to_dict(initial_angles)

        # 判断是否是单个点
        is_single_pose = isinstance(pose_traj[0], (float, int)) and len(pose_traj) == 7
        poses = [pose_traj] if is_single_pose else pose_traj

        logger.module(f"[ik_controller] 开始求解IK")
        joint_traj = []
        cur_angles = initial_angles
        t0 = time.time()

        for i, pose in enumerate(poses):
            self.set_target(pos=pose[:3], quat=pose[3:])
            solution = self.ik_solver.solve(
                target_pos=self._target_pos,
                target_quat=self._target_quat,
                initial_angles=cur_angles
            )
            cur_angles = solution  # 用上一个结果作为初始角度

            if output_format == 'as_list':
                joint_traj.append([solution[f'joint{i+1}'] for i in range(6)])
            elif output_format == 'as_dict':
                joint_traj.append(solution)
            else:
                raise ValueError("output_format 必须为 'as_list' 或 'as_dict'")
            
        t1 = time.time()
        logger.info(f"[ik_controller] IK求解结束， 用时{t1-t0: .2f}秒")

        return joint_traj[0] if is_single_pose else joint_traj


    def _convert_to_dict(self, angles: JointInput) -> Dict[str, float]:
        """统一将角度输入转换为 {joint1: x, ..., joint6: y} 格式"""
        if isinstance(angles, dict):
            keys = {f'joint{i}' for i in range(1, 7)}
            if set(angles.keys()) != keys:
                raise ValueError(f"关节字典必须包含 {keys}，当前为 {angles.keys()}")
            return angles.copy()
        elif isinstance(angles, (list, np.ndarray)):
            if len(angles) != 6:
                raise ValueError("初始角度必须为长度为 6 的列表或数组")
            return {f'joint{i+1}': float(angles[i]) for i in range(6)}
        else:
            raise TypeError("初始角度应为 list, np.ndarray 或 dict")