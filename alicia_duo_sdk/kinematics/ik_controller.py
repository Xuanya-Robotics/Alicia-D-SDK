# ik_controller.py
import numpy as np
from typing import Dict, List, Union
from .advanced_ik_solver import Advanced6DOFIKSolver
from .robot_model import RobotArm
from ..utils.logger import BeautyLogger

logger = BeautyLogger(log_dir="./logs", log_name="ik_controller.log", verbose=True)

JointInput = Union[List[float], np.ndarray, Dict[str, float]]

class IKController:
    def __init__(self, robot_arm: RobotArm):
        self.robot_arm = robot_arm
        self.ik_solver = Advanced6DOFIKSolver(robot_arm)
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

    def solve(self, initial_angles: JointInput, 
              output_format: str = 'list') -> List[float]:
        """
        执行 IK 解算
        Args:
            initial_angles: 初始角度，支持 list / np.array / dict
            output_format: 输出格式，'list' 或 'dict'

        Returns:
            IK 解，格式可选
        """
        if self._target_pos is None or self._target_quat is None:
            raise RuntimeError("未设置目标位姿，请先调用 set_target()")

        initial_angles = self._convert_to_dict(initial_angles)

        solution = self.ik_solver.solve(
            target_pos=self._target_pos,
            target_quat=self._target_quat,
            initial_angles=initial_angles
        )

        if output_format == 'list':
            return [solution[f'joint{i+1}'] for i in range(6)]
        elif output_format == 'dict':
            return solution
        else:
            raise ValueError("output_format 必须为 'list' 或 'dict'")

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