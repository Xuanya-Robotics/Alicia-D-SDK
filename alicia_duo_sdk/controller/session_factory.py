# controller/session_factory.py

from ..driver.servo_driver import ArmController
from ..kinematics import AliciaFollower, IKController
from ..execution import TrajectoryExecutor
from .motion_session import MotionSession
from ..utils.logger import logger

# logger = BeautyLogger(log_dir="./logs", log_name="session.log", verbose=True)

def get_default_session(
    port: str = '',        
    baudrate: int = 1000000,
    debug: bool = False,
    connect_timeout: float = 5.0
) -> MotionSession:
    """
    自动构造 MotionSession，包括 IK 控制器、模型与关节控制器，并连接机械臂。

    Args:
        port (str): 指定串口（留空自动查找）
        baudrate (int): 波特率（默认 1000000）
        debug (bool): 是否开启调试日志
        connect_timeout (float): 连接机械臂的超时时间

    Returns:
        MotionSession: 构建完成的控制上下文
    """
    logger.module("[session] 开始构建默认会话")
    logger.info("[session]] 初始化 AliciaFollower 模型")
    robot_model = AliciaFollower()

    logger.info("[session] 初始化 IK 控制器")
    ik_controller = IKController(robot_model)

    logger.info(f"[session] 初始化机械臂控制器 (port='{port}', baudrate={baudrate})")
    joint_controller = ArmController(port=port, baudrate=baudrate, debug_mode=debug)

    logger.info("[session] 初始化执行器")
    executor = TrajectoryExecutor(joint_controller=joint_controller, 
                                  robot_model=robot_model)

    logger.info("[session] 正在连接机械臂")
    if not joint_controller.connect():
        logger.warning("[session] 会话建立失败")
        raise RuntimeError("连接机械臂失败，请检查串口连接")

    logger.info("[session] 会话创建成功")
    return MotionSession(ik_controller=ik_controller,
                          robot_model=robot_model, 
                          joint_controller=joint_controller,
                          executor=executor)
