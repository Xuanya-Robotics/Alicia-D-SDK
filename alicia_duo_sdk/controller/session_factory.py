# controller/session_factory.py

from ..motor.controller import ArmController
from ..kinematics import RobotArm, IKController
from .motion_session import MotionSession
import time

def get_default_session(
    port: str = "",
    baudrate: int = 921600,
    debug: bool = False,
    connect_timeout: float = 5.0
) -> MotionSession:
    """
    自动构造 MotionSession，包括 IK 控制器、模型与关节控制器，并连接机械臂。

    Args:
        port (str): 指定串口（留空自动查找）
        baudrate (int): 波特率（默认 921600）
        debug (bool): 是否开启调试日志
        connect_timeout (float): 连接机械臂的超时时间

    Returns:
        MotionSession: 构建完成的控制上下文
    """
    print("[SDK] 初始化 RobotArm 模型...")
    robot_model = RobotArm()

    print("[SDK] 初始化 IK 控制器...")
    ik_controller = IKController(robot_model)

    print(f"[SDK] 初始化机械臂控制器 (port='{port}', baudrate={baudrate})...")
    joint_controller = ArmController(port=port, baudrate=baudrate, debug_mode=debug)

    print("[SDK] 正在连接机械臂...")
    if not joint_controller.connect():
        raise RuntimeError("连接机械臂失败，请检查串口连接")

    print("[SDK] 等待机械臂状态初始化...")
    if not joint_controller.wait_for_valid_state(timeout=connect_timeout):
        raise RuntimeError("机械臂状态初始化失败，请检查是否已上电")

    print("[SDK] 运动控制上下文构建成功 ✅")
    return MotionSession(ik_controller, robot_model, joint_controller)
