# motion_api.py (in controller/)
# from ..planning.planner_registry import get_planner
from ..utils import *
from ..kinematics import *
from ..planning.planners import *
from ..execution.trajectory_executor import TrajectoryExecutor

from typing import List, Optional
import numpy as np

from .motion_session import MotionSession
from .utils import *

import time

logger = BeautyLogger(log_dir="./logs", log_name="move.log", verbose=True)

class ControlApi():
    def __init__(self, session:MotionSession):
        self.session = session
        self.joint_controller = session.joint_controller
        self.robot_model = session.robot_model
        self.ik_controller = session.ik_controller

        self.home_angles = [0.0] * 6

    def moveCartesian(
        self,
        waypoints: List[List[float]],  # List of [x, y, z, qx, qy, qz, qw]
        start_joint_angles: Optional[List[float]] = None,
        planner_name: str = "cartesian",
        visualize: bool = False,
        move_time: float = 3.0
    ):
        """
        多段 Cartesian 插值轨迹规划（带 IK 解算）并执行

        Args:
            session: 当前机器人会话（包含IK/模型/关节控制器）
            pose_sequence: 末端姿态点序列 [[x, y, z, qx, qy, qz, qw], ...]
            start_joint_angles: 起始关节角度（如果未提供，则从当前状态读取）
            planner_name: 插值规划器名称，默认使用 "cartesian"
            visualize: 是否可视化插值轨迹
            delay: 执行每步之间的延迟（单位：秒）
        """
        validate_pose_sequence(waypoints)

        if start_joint_angles is None:
            start_joint_angles = self.joint_controller.get_joint_angles()
        else:
            validate_joint_list(start_joint_angles)

        logger.module("[moveCartesian]开始规划")
        t0 = time.time()
        planner = CartesianPlanner()

        pose_traj, joint_traj = planner.plan(
            ik_controller=self.ik_controller,
            start_joint_angles=start_joint_angles,
            robot_model=self.robot_model,
            waypoints=waypoints
        )

        t1 = time.time()

        T_max = 10.0  # 最长允许的总时间（秒）
        max_delay = 0.05  # 每步最大延迟，单位秒
        min_delay = 0.005  # 每步最小延迟，单位秒

        total_steps = len(joint_traj)
        t = min(T_max, move_time)
        computed_delay = t / total_steps
        delay = min(max(computed_delay, min_delay), max_delay)

        logger.info(f"[moveCartesian] 规划完成，共 {len(joint_traj)} 点，"
                    f"规划用时 {t1 - t0:.2f} 秒,"
                    f"每步间隔{delay} 秒")
        
        executor = TrajectoryExecutor(self.joint_controller, delay=delay)
        executor.execute(
            trajectory=joint_traj,
            robot_model=self.robot_model,
            visualize=visualize,
            show_ori=True)


    def moveJ(
        self,
        cur_angles: Optional[List[float]] = None,
        target_angles: List[float] = None,
        speed_factor: float = 1.0,
        T_default: float = 4.0,
        n_steps_ref: int = 200,
        visualize: bool = False,
        show_ori: bool = False

    ):
        """
        控制机械臂从当前关节角度平滑移动到目标关节角度，使用 cubic 插值和速度因子控制时间

        Args:
            session (MotionSession): 当前机器人会话对象，包含关节控制器与模型
            cur_angles (Optional[List[float]]): 当前角度列表（如果为 None，将从机械臂读取当前角度）
            target_angles (List[float]): 目标角度列表（必须指定）
            speed_factor (float): 速度倍率（>1 更快，<1 更慢，默认为 1.0）
            T_default (float): 默认总时长（当 speed_factor=1 时所用的时间，单位秒）
            n_steps_ref (int): 默认插值步数（用于参考步数缩放）
            visualize (bool): 是否画出关节轨迹图和3D位姿轨迹图
            show_ori (bool): 在3D位姿图中是否标出末端姿态

        Raises:
            ValueError: 若当前角度与目标角度维度不一致或未提供目标角度
        """
        if target_angles is None:
            raise ValueError("必须提供目标角度 target_angles")

        # 若未提供当前角度，则从机器人状态中获取
        if cur_angles is None:
            cur_angles = self.joint_controller.get_joint_angles()

        if len(cur_angles) != len(target_angles):
            raise ValueError(f"错误: 当前角度数量是 {len(cur_angles)} "
                            f"与目标角度数量 {len(target_angles)} 不匹配")

        # 根据速度因子计算插值步数和每步延迟
        steps, delay = compute_steps_and_delay(
            speed_factor=speed_factor,
            T_default=T_default,
            n_steps_ref=n_steps_ref
        )

        # 构造插值轨迹
        traj = []
        for step in range(1, steps + 1):
            ratio = ease_in_out_cubic(step / steps)
            interp_angles = [
                current + (target - current) * ratio
                for current, target in zip(cur_angles, target_angles)
            ]
            traj.append(interp_angles)

        # 日志输出轨迹信息
        logger.info(
        f"[moveJ] 从角度 {np.round(cur_angles, 3).tolist()} "
        f"移动到 {np.round(target_angles, 3).tolist()},"
        f"共 {steps} 步，每步间隔 {delay:.3f}s"
        )   

        # 执行轨迹
        executor = TrajectoryExecutor(self.joint_controller, delay=delay)
        executor.execute(traj, robot_model=self.robot_model, 
                        visualize=visualize, show_ori=show_ori)


    def moveHome(self):
        self.moveJ(target_angles=self.home_angles)
    
    def get_angles(self) -> List[float]:
        """
        获取当前关节角度（单位：弧度）
        Returns:
            List[float]: 当前机械臂的6个关节角度
        """
        return self.joint_controller.get_joint_angles()

    def get_pose(self) -> List[float]:
        """
        获取当前末端执行器的位置与姿态（7D）
        Returns:
            List[float]: [x, y, z, qx, qy, qz, qw]
        """
        joint_angles = self.joint_controller.get_joint_angles()
        pos, quat = self.robot_model.forward_kinematics(joint_angles)
        return np.concatenate([pos, quat]).tolist()

    def get_gripper_state(self) -> float:
        """
        获取当前夹爪角度（弧度）

        Returns:
            float: 当前夹爪角度
        """
        state = self.joint_controller.get_joint_state()
        return state.gripper if state else None
    
    def gripper_control(self, command: str = None, angle_deg: float = None,
                    timeout: float = 5.0, tolerance: float = 0.1) -> bool:
        """
        控制夹爪开合或设置角度，并阻塞等待夹爪到达目标位置

        Args:
            command (str): "open" 或 "close"
            angle_deg (float): 自定义角度（0~100）
            timeout (float): 最长等待时间（秒）
            tolerance (float): 到目标角度的容差（弧度）

        Returns:
            bool: 是否成功执行到位
        """
        if command:
            if command == "open":
                angle_deg = 0.0
            elif command == "close":
                angle_deg = 100.0
            else:
                raise ValueError("command 必须是 'open' 或 'close'")
        elif angle_deg is None:
            raise ValueError("必须提供 command 或 angle_deg 参数")

        # 转换为弧度
        angle_rad = angle_deg * self.joint_controller.DEG_TO_RAD

        # 发送控制帧（最多发2次）
        frame = self.joint_controller._build_gripper_frame(angle_rad)
        success = False
        for _ in range(2):
            success = self.joint_controller.serial_comm.send_data(frame)
        if not success:
            logger.warning("夹爪控制帧发送失败")
            return False

        # 等待夹爪运动完成
        if self.joint_controller.debug_mode:
            logger.debug(f"等待夹爪移动至 {angle_deg:.1f}°")

        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.joint_controller.get_joint_state()
            if state and abs(state.gripper - angle_rad) <= tolerance:
                if self.joint_controller.debug_mode:
                    logger.debug("夹爪已到达目标")
                return True
            time.sleep(0.02)

        logger.warning("夹爪运动等待超时")
        return False


    
    def print_state(self):
        return 
    
    def torque_control(self):
        return
    
    def zero_calibration(self):
        return
    
