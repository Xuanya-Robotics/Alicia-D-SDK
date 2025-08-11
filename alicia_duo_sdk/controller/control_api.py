# motion_api.py (in controller/)
# from ..planning.planner_registry import get_planner
from ..utils.coord.validation import validate_joint_list, validate_waypoints 
from ..utils.logger import BeautyLogger
from ..utils.coord import check_and_clip_joint_limits, compute_steps_and_delay
from ..planning.planners import *

from .motion_session import MotionSession

from typing import List, Union
import numpy as np
import time, sys, select
import platform


logger = BeautyLogger(log_dir="./logs", log_name="move.log", verbose=True)

class ControlApi():
    def __init__(self, session:MotionSession):
        # 控制器
        self.session = session
        self.joint_controller = session.joint_controller
        self.robot_model = session.robot_model
        self.ik_controller = session.ik_controller
        self.executor = session.executor

        # 参数
        self.max_cartesian_delay = 0.05
        self.min_cartesian_delay = 0.005
        self.max_cartesian_time = 10.0

        # 默认位置关节角度和位姿
        self.home_angles = [0.0] * 6
        self.home_pose = [0.3086, -0.0025, 0.0890,  # x, y, z
                          0.0029, 0.9310, -0.0028, 0.3649]  # x, y, z ,w

    def moveCartesian(
        self,
        waypoints: Union[List[List[float]], List[float]], 
        planner_name: str = "cartesian",
        visualize: bool = False,
        show_ori: bool = False,
        move_time: float = 3.0,
        reverse: bool = False
    ):
        """
        多段 Cartesian 插值轨迹规划（带 IK 解算）并执行

        Args:
            waypoints: 单个或多个末端姿态点（[x, y, z, qx, qy, qz, qw]）
            start_joint_angles: 起始关节弧度值（如果未提供，则从当前状态读取）
            planner_name: 插值规划器名称，默认使用 "cartesian", 可选 'lqt'
            visualize: 是否可视化规划轨迹
            show_ori: 是否在姿态3D轨迹图中显示位姿
            move_time: 总执行时间（秒），将自动计算每步延迟
            reverse: 是否将waypoints最后的点作为起点
        """
        # 校验数据
        if isinstance(waypoints[0], (int, float)) and len(waypoints) == 7:
            waypoints = [waypoints]
        validate_waypoints(waypoints)

        # 建立规划器和导入当前机械臂角度
        if planner_name == 'cartesian':
            planner = CartesianPlanner(verbose=True)
        elif planner_name == 'lqt':
            planner = LQT(verbose=True)

        start_joint_angles = self.joint_controller.get_joint_angles()

        logger.module("[moveCartesian] 开始调用moveCartesian API")
        logger.info(f"[moveCartesian] 一共有 {len(waypoints)} 个途经点")
        logger.info(f"[moveCartesian] 当前规划器为 {planner_name}")

        # 反转路径方向
        if reverse:
            logger.info("[moveCartesian] 已启用反向模式，从末尾路径点开始进行轨迹规划")
            waypoints = waypoints[::-1]

        logger.info("[moveCartesian] 开始插值末端姿态轨迹，求解关节轨迹并执行") 

        # 插值末端姿态轨迹
        if planner_name == 'cartesian':
            pose_traj = planner.plan(
                start_joint_angles=start_joint_angles,
                robot_model=self.robot_model,
                waypoints=waypoints
            )

        elif planner_name == 'lqt':
            p0 = self.get_pose()
            waypoints.insert(0, p0)
            pose_traj = planner.plan(
                via_points=waypoints,
                nbdata= 200
            )
        
        # 判断并选择 IK 解算方式
        joint_traj = self.ik_controller.solve_ik(
            pose_traj=pose_traj,
            initial_angles=start_joint_angles,
            output_format='as_list'
        )

        # 根据规划点数计算每步的延时
        total_steps = len(joint_traj)
        t = min(self.max_cartesian_time, move_time)
        computed_delay = t / total_steps
        delay = min(max(computed_delay, self.min_cartesian_delay),
                     self.max_cartesian_delay)

        self.executor.delay = delay
        self.executor.execute(
            joint_traj=joint_traj,
            pose_traj=pose_traj,
            visualize=visualize,
            show_ori=show_ori)

    def moveJ(
        self,
        joint_format: str = 'rad',
        target_joints: List[float] = None,
        speed_factor: float = 1.0,
        T_default: float = 4.0,
        n_steps_ref: int = 200,
        visualize: bool = False
        ):
        """
        控制机械臂从当前位置平滑移动至目标关节角度，使用 cubic 插值与速度控制

        Args:
            joint_format (str): 输入角度单位，'rad' 或 'deg'
            target_joints (List[float]): 目标关节角度
            speed_factor (float): 速度倍率，>1 更快，<1 更慢
            T_default (float): 默认插值总时长（秒）
            n_steps_ref (int): 参考插值步数
            visualize (bool): 是否可视化轨迹
            show_ori (bool): 是否可视化末端姿态
        """
        logger.module("[moveJ] 开始执行关节空间插值移动")

        if target_joints is None:
            raise ValueError("[moveJ] 请提供 target_joints 参数")

        joint_format = joint_format.lower()
        if joint_format not in ['rad', 'deg']:  
            raise ValueError(f"[moveJ] 不支持的 joint_format: '{joint_format}'，"
                             "应为 'rad' 或 'deg'")

        # 支持角度制输入
        if joint_format == 'deg':
            logger.info("[moveJ] 输入角度单位为 degree，将转换为 rad")
            target_joints = [a * self.joint_controller.DEG_TO_RAD for a in target_joints]

        validate_joint_list(target_joints)

        # 检查关节限位并修正
        target_joints, violations = check_and_clip_joint_limits(
            joints=target_joints,
            joint_limits=self.robot_model.joint_limit
        )

        for joint_name, original, clipped in violations:
            logger.warning(
                f"[moveJ] {joint_name} 超出限制：{original:.2f} -> 已截断为 {clipped:.2f}"
            )

        # 获取当前状态
        cur_angles = self.joint_controller.get_joint_angles()

        # 插值步数与延迟
        steps, delay = compute_steps_and_delay(
            speed_factor=speed_factor,
            T_default=T_default,
            n_steps_ref=n_steps_ref
        )

        # 插值轨迹生成
        planner = JointPlanner()
        
        joint_traj = planner.plan(
            start_angles=cur_angles,
            target_angles=target_joints,
            steps=steps
            )

        if joint_format == 'deg':
            display_cur = [round(a * self.joint_controller.RAD_TO_DEG, 1) 
                           for a in cur_angles]
            display_target = [round(a * self.joint_controller.RAD_TO_DEG, 1) 
                              for a in target_joints]
            unit = "°"
        else:
            display_cur = [round(a, 3) for a in cur_angles]
            display_target = [round(a, 3) for a in target_joints]
            unit = "rad"

        # 日志输出
        logger.info(f"[moveJ] 起始角度 ({unit}): {display_cur}")
        logger.info(f"[moveJ] 目标角度 ({unit}): {display_target}")
        logger.info(f"[moveJ] 插值步数: {steps}，单步延迟: {delay:.3f}s，"
                    f"预计总耗时: {steps * delay:.0f}s")

        # 执行轨迹
        self.executor.delay = delay
        self.executor.execute(
            joint_traj=joint_traj,
            visualize=visualize
        )

    def moveHome(self):
        '''
        控制机械臂返回默认位置
        '''
        logger.module("[moveHome] 开始移动到初始位置")
        self.moveJ(target_joints=self.home_angles)

   
    def get_joints(self) -> List[float]:
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

    def get_gripper(self) -> float:
        """
        获取当前夹爪角度（弧度）

        Returns:
            float: 当前夹爪角度
        """
        state = self.joint_controller.get_joint_state()
        return state.gripper if state else None
    
    def gripper_control(self, command: str = None, angle_deg: float = None,
                        wait_for_completion: bool=True,
                        timeout: float = 5.0, tolerance: float = 1.0) -> None:
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
        if command is not None and angle_deg is not None:
            raise ValueError("[gripper_control] command 与 angle_deg 参数不可同时指定")

        if command is not None:
            if command == "open":
                angle_deg = 0.0
                logger.module("[gripper_control] 正在打开夹爪")
            elif command == "close":
                angle_deg = 100.0
                logger.module("[gripper_control] 正在关闭夹爪")
            else:
                raise ValueError("command 参数必须是 'open' 或 'close'")

        if angle_deg is None:
            raise ValueError("必须提供 command 或 angle_deg 参数")

        # 转换为弧度
        angle_rad = angle_deg * self.joint_controller.DEG_TO_RAD

        # 发送夹爪指令
        logger.module(f"[gripper_control]正在设置夹爪到{angle_deg}°")
        result = self.joint_controller.set_gripper(angle_rad)

        if result:
            logger.info("[gripper_control]夹爪数据发送成功")
            if wait_for_completion:
                start_time = time.time()
                success = False
                while time.time() - start_time < timeout:
                    gripper_deg = self.get_gripper() * self.joint_controller.RAD_TO_DEG
                    if gripper_deg and abs(gripper_deg - angle_deg) <= tolerance:
                        logger.info(f"[gripper_control] 夹爪已到达目标角度：{angle_deg:.1f}°，"
                                    f"实际角度：{gripper_deg:.1f}°，误差：{gripper_deg - angle_deg:.2f}°")
                        success = True
                        break
                    time.sleep(0.1)
                if not success:
                    logger.warning("[gripper_control]夹爪运动等待超时,"
                                   f"当前角度{gripper_deg: 2f}°")


    def print_state(self, continous_printing: bool = False, output_format: str = "deg") -> None:
        """
        打印当前机械臂关节角度和夹爪状态

        Args:
            continous_printing (bool): 是否持续打印（按 Ctrl+C 中断）
            output_format (str): 输出格式，"deg" 或 "rad"
        """
        def _print_once():
            joint_rad = self.get_joints()
            gripper_rad = self.get_gripper()

            if joint_rad is None or gripper_rad is None:
                logger.warning("[print_state] 当前没有有效的状态信息")
                return

            if output_format == 'deg':
                joint_deg = [round(a * self.joint_controller.RAD_TO_DEG, 1) for a in joint_rad]
                gripper_deg = round(gripper_rad * self.joint_controller.RAD_TO_DEG, 1)
                pose = self.get_pose()
                pos = [round(p,3) for p in pose[:3]]
                quat = [round(q,3) for q in pose[3:]]

                logger.module("[print_state] 开始打印机械臂信息")
                logger.info(f"[print_state] 关节角度（单位：度）: {joint_deg}")
                logger.info(f"[print_state] 机械臂位置: {pos}")
                logger.info(f"[print_state] 机械臂四元角: {quat}")
                logger.info(f"[print_state] 夹爪角度（单位：度）: {gripper_deg}")
            
            elif output_format == 'rad':
                joint_rad_round = [round(joint, 3) for joint in joint_rad]
                gripper_rad_round = round(gripper_rad, 2)
                pose = self.get_pose()
                pos = [round(p,3) for p in pose[:3]]
                quat = [round(q,3) for q in pose[3:]]

                logger.module("[print_state] 开始打印机械臂信息")
                logger.info(f"[print_state] 关节角度（单位：弧度）: {joint_rad_round}")
                logger.info(f"[print_state] 机械臂位置: {pos}")
                logger.info(f"[print_state] 机械臂四元角: {quat}")
                logger.info(f"[print_state] 夹爪角度（单位：弧度）: {gripper_rad_round}")
            
            else:
                logger.warning(f"[print_state] 不支持的输出格式：{output_format}")

        if continous_printing:
            logger.module("[print_state] 开启持续状态打印，按 Enter 停止")
            time.sleep(1)
            try:
                while True:
                    _print_once()
                    time.sleep(0.5)  # 打印间隔（可调）
                    
                    # 跨平台兼容的输入检测
                    if platform.system() == "Windows":
                        # Windows系统使用msvcrt模块检测按键
                        try:
                            import msvcrt
                            if msvcrt.kbhit():
                                key = msvcrt.getch()
                                if key == b'\r' or key == b'\n':  # Enter键
                                    logger.module("[print_state] 持续打印已停止")
                                    break
                        except ImportError:
                            # 如果msvcrt不可用，使用简单的input()等待
                            try:
                                input("按Enter键停止打印: ")
                                logger.module("[print_state] 持续打印已停止")
                                break
                            except:
                                pass
                    else:
                        # Unix/Linux系统使用select
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            input()  # 停止读取
                            logger.module("[print_state] 持续打印已停止")
                            break
            except KeyboardInterrupt:
                logger.module("[print_state] 持续打印已停止")
                

        else:

            _print_once()



    def torque_control(self, command: str=None) -> bool:
        """
        控制机械臂扭矩开关（可选择开启或关闭）

        Arg:
            command (str): on 开启扭矩， off 关闭扭矩
        """
        if command == "on":
            logger.module("[torque_control]正在开启扭矩")
            result = self.joint_controller.enable_torque()
            if result:
                logger.info("[torque_control]扭矩成功开启")
            else:
                logger.warning("[torque_control]扭矩开启失败，请重试")

        elif command == "off":
            logger.module("[torque_control]正在关闭扭矩")
            logger.info("[torque_control]按下回车关闭扭矩，请固定好机械臂")
            input()
            result = self.joint_controller.disable_torque()
            if result:
                logger.info("[torque_control]扭矩成功关闭")
            else:
                logger.warning("[torque_control]扭矩关闭失败，请重试")

        else:
            logger.warning("[torque_control]command指令错误， "
                           "command只接受 'on' 或者 'off', "
                           f"当前指令为 {command}")
            result = False

        return result

    def zero_calibration(self) -> None:
        """
        执行归零校准流程（需手动拖拽机械臂至初始位）

        步骤:
        1. 提示用户是否执行归零
        2. 关闭扭矩 -> 拖动机械臂 -> 按下回车
        3. 重新打开扭矩
        4. 执行零位标定函数
        """
        logger.module("[zero_calibration] 准备执行归零操作")

        # 关闭扭矩
        if not self.torque_control('off'):
            logger.warning("[zero_calibration] 扭矩关闭失败，终止归零操作")
            return

        logger.info("[zero_calibration] 扭矩已关闭，可手动拖动机械臂到零点位置")
        logger.info("[zero_calibration] 拖动完成后按下回车继续")
        input()

        # 重新开启扭矩
        logger.module("[zero_calibration] 正在重新开启扭矩")
        if not self.torque_control('on'):
            logger.warning("[zero_calibration] 扭矩开启失败，终止归零操作")
            return

        # 执行零点校准
        logger.module("[zero_calibration] 正在执行零位校准")
        result = self.joint_controller.set_zero_position()

        if result:
            logger.info("[zero_calibration] 零位校准成功")
        else:
            logger.warning("[zero_calibration] 零位校准失败，请重试")
    
