import math
import time
import logging
import threading
from typing import List, Optional, Union, Tuple, Dict
import numpy as np

from .serial_comm import SerialComm
from .data_parser import DataParser, JointState

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Controller")

class ArmController:
    """机械臂控制模块"""
    
    # 常量定义
    
    RAD_TO_DEG = 180.0 / math.pi  # 弧度转角度系数
    DEG_TO_RAD = math.pi / 180.0  # 角度转弧度系数
    # 帧常量
    FRAME_HEADER = 0xAA
    FRAME_FOOTER = 0xFF
    FRAME_MINIMAL_SIZE = 5
    ARM_DATA_SIZE = 18
    GRIPPER_FRAME_SIZE = 8
    
    # 指令ID
    CMD_GRIPPER = 0x02     # 夹爪控制与行程反馈
    CMD_ZERO_POS = 0x03    # 机械臂以当前位置为零点  
    CMD_JOINT = 0x04       # 机械臂角度反馈与控制
    CMD_MULTI_ARM = 0x06   # 四机械臂角度反馈与控制
    CMD_TORQUE = 0x13      # 机械臂力矩控制
    
    def __init__(self, port: str = "", baudrate: int = 921600, debug_mode: bool = False):
        """
        初始化机械臂控制器
        
        Args:
            port: 串口名称，留空则自动搜索
            baudrate: 波特率
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode
        self._lock = threading.Lock()
        # 创建串口通信模块和数据解析器
        self.serial_comm = SerialComm(lock=self._lock, port=port, baudrate=baudrate, debug_mode=debug_mode)
        self.data_parser = DataParser(lock=self._lock, debug_mode=debug_mode)
        
        # 舵机数量
        self.servo_count = 9
        self.joint_count = 6
        
        # 舵机映射表：关节索引->舵机索引
        # 机械臂的6个关节需要映射到9个舵机上
        # [关节1, 关节1(重复), 关节2, 关节2(反向), 关节3, 关节3(反向), 关节4, 关节5, 关节6]
        self.joint_to_servo_map = [
            (0, 1.0),    # 关节1 -> 舵机1 (正向)
            (0, 1.0),    # 关节1 -> 舵机2 (正向重复)
            (1, 1.0),    # 关节2 -> 舵机3 (正向)
            (1, -1.0),   # 关节2 -> 舵机4 (反向)
            (2, 1.0),    # 关节3 -> 舵机5 (正向)
            (2, -1.0),   # 关节3 -> 舵机6 (反向)
            (3, 1.0),    # 关节4 -> 舵机7 (正向)
            (4, 1.0),    # 关节5 -> 舵机8 (正向)
            (5, 1.0),    # 关节6 -> 舵机9 (正向)
        ]
        
        # 状态更新线程相关
        self._update_thread = None
        self.thread_update_interval = 0.005  # 更新间隔，单位：秒
        self._stop_thread = threading.Event()
        self._thread_running = False
        
        logger.info("初始化机械臂控制模块")
        logger.info(f"调试模式: {'启用' if debug_mode else '禁用'}")

        self.disconnect()
    
    def wait_for_valid_state(self, timeout: float = 5.0) -> bool:
        """
        等待指定机械臂的状态变为有效（不为全零）

        Args:
            arm (str): "left_arm"、"right_arm" 或 "both"
            timeout (float): 最大等待时间（秒）

        Returns:
            bool: 如果在超时时间内收到有效状态，返回 True；否则返回 False
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            js = self.read_joint_state()
            if js and max(abs(a) for a in js.angles) > 1e-3:
                return True
            time.sleep(0.05)
        print(f"[超时] 未收到有效关节状态")
        return False

    def __del__(self):
        """析构函数，确保线程和连接在对象销毁时被正确清理"""
        try:
            # 停止状态更新线程
            self.stop_update_thread()
            # 断开连接
            self.disconnect()
        except Exception as e:
            if hasattr(logger, 'error'):  # 在某些情况下logger可能已被销毁
                logger.error(f"析构函数中出现异常: {str(e)}")
    
    def connect(self) -> bool:
        """
        连接到机械臂
        
        Returns:
            bool: 连接是否成功
        """
        result = self.serial_comm.connect()
        if result:
            # 连接成功后启动状态更新线程
            self.start_update_thread()
            self.wait_for_valid_state()
        return result
    
    def disconnect(self):
        """断开与机械臂的连接"""
        # 先停止状态更新线程
        self.stop_update_thread()
        self.serial_comm.disconnect()
    
    def start_update_thread(self):
        """启动状态更新线程"""
        if self._update_thread is not None and self._thread_running:
            logger.info("状态更新线程已经在运行")
            return
        
        # 重置停止信号
        self._stop_thread.clear()
        self._thread_running = True
        
        # 创建并启动线程
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("状态更新线程已启动")
    
    def stop_update_thread(self):
        """停止状态更新线程"""
        if self._update_thread is None or not self._thread_running:
            return
        
        # 设置停止信号
        self._stop_thread.set()
        self._thread_running = False
        
        # 等待线程结束
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        
        self._update_thread = None
        logger.info("状态更新线程已停止")
    
    def is_update_thread_running(self) -> bool:
        """
        检查状态更新线程是否正在运行
        
        Returns:
            bool: 线程是否正在运行
        """
        return self._thread_running and self._update_thread is not None and self._update_thread.is_alive()
    
    def get_update_thread_status(self) -> Dict:
        """
        获取状态更新线程的详细信息
        
        Returns:
            Dict: 包含线程状态的字典
        """
        return {
            "running": self.is_update_thread_running(),
            "enabled": self._thread_running,
            "thread_exists": self._update_thread is not None,
            "thread_alive": self._update_thread.is_alive() if self._update_thread else False,
            "stop_flag_set": self._stop_thread.is_set()
        }
    
    def _update_loop(self):
        """状态更新线程主循环"""
        logger.info("状态更新线程开始运行")
        
        while not self._stop_thread.is_set():
            time.sleep(self.thread_update_interval)
            try:
                with self._lock:
                    # 读取一帧数据
                    frame = self.serial_comm.read_frame()
                if frame == 9999999:
                    logger.error("检测到严重的串口通信异常，机械臂可能已断开连接")
                    break
                if frame:
                    # 解析数据帧，更新内部状态
                    self.data_parser.parse_frame(frame)
        
            except Exception as e:
                logger.error(f"状态更新线程异常: {str(e)}")
                
                
                break
        self._thread_running = False
        logger.info("状态更新线程已结束运行")
    
    def get_joint_angles(self) -> Optional[List[float]]:
        """
        读取关节角度（弧度）
        
        Returns:
            Optional[List[float]]: 6个关节的角度列表（弧度），读取失败则返回None
        """
        # 直接返回当前状态，不需要额外读取数据
        return self.data_parser.get_joint_state().angles
    
    def get_gripper_data(self) -> Tuple[float, bool, bool]:
        """
        读取夹爪数据
        
        Returns:
            Tuple[float, bool, bool]: 夹爪角度（弧度）、按钮1状态、按钮2状态
        """
        # 直接返回当前状态，不需要额外读取数据
        state = self.data_parser.get_joint_state()
        return state.gripper, state.button1, state.button2
    
    
    def get_joint_state(self) -> JointState:
        """
        读取完整的机械臂状态。
        由于已有后台线程持续更新，所以直接返回当前状态。
        """
        # 直接返回当前状态，不需要额外读取数据
        return self.data_parser.get_joint_state()
    
    def set_joint_angles(self, joint_angles: List[float], gripper_angle: float = None, timeout: float = 5.0, tolerance: float = 0.08) -> bool:
        """
        设置关节角度（弧度）
        
        Args:
            joint_angles: 6个关节的角度列表（弧度）
            gripper_angle: 夹爪角度（弧度），如果为None则不发送夹爪命令
            wait_for_completion: 是否等待运动完成
            timeout: 等待运动完成的超时时间（秒）
            tolerance: 角度误差容忍度（弧度）
            
        Returns:
            bool: 命令是否成功发送和执行
        """
        # 验证输入
        if len(joint_angles) != self.joint_count:
            logger.error(f"关节数量错误: 需要{self.joint_count}个, 提供了{len(joint_angles)}个")
            return False
            
        # 构造关节控制帧
        frame = self._build_joint_frame(joint_angles)
        
        # 发送关节控制命令
        for i in range(2):
            result = self.serial_comm.send_data(frame)
        
        # 如果需要控制夹爪
        if gripper_angle is not None and result:
            #time.sleep(0.02)  # 短暂延时，避免连续发送导致丢包
            result = self.set_gripper(gripper_angle)
            
        return result
    
    def set_gripper(self, angle_rad: float) -> bool:
        """
        设置夹爪角度（弧度）
        
        Args:
            angle_rad: 夹爪角度（弧度）
            
        Returns:
            bool: 命令是否成功发送
        """
        # 构造夹爪控制帧
        frame = self._build_gripper_frame(angle_rad)
        # 最多发送两次
        
        for i in range(2):
            result = self.serial_comm.send_data(frame)

        return result 
    
    
    def set_zero_position(self) -> bool:
        """
        设置当前位置为零点
        
        Returns:
            bool: 命令是否成功发送
        """
        # 构造零点设置帧
        frame = self._build_command_frame(self.CMD_ZERO_POS, [0x00])
        
        # 发送零点设置命令
        return self.serial_comm.send_data(frame)
    
    def enable_torque(self) -> bool:
        """
        使能力矩控制（使机械臂保持当前位置）
        
        Returns:
            bool: 命令是否成功发送
        """
        # 构造力矩使能帧
        frame = self._build_command_frame(self.CMD_TORQUE, [0x01])
        
        # 发送力矩使能命令
        result = self.serial_comm.send_data(frame)
        time.sleep(1)
        
        if result:
            logger.info("扭矩已开启")
        return result
    
    def disable_torque(self) -> bool:
        """
        禁用力矩控制（使机械臂可以自由移动）
        
        Returns:
            bool: 命令是否成功发送
        """
        # 构造力矩禁用帧
        frame = self._build_command_frame(self.CMD_TORQUE, [0x00])
        
        # 发送力矩使能命令
        result = self.serial_comm.send_data(frame)
        time.sleep(1)
        
        if result:
            logger.info("扭矩已关闭")
        return result
    
    
    def _build_joint_frame(self, joint_angles: List[float]) -> List[int]:
        """
        构建关节控制帧
        
        Args:
            joint_angles: 6个关节的角度列表（弧度）
            
        Returns:
            List[int]: 控制帧字节列表
        """
        # 计算帧大小：帧头(1)+命令(1)+长度(1)+数据(舵机数*2)+校验(1)+帧尾(1)
        frame_size = self.FRAME_MINIMAL_SIZE + self.ARM_DATA_SIZE
        
        # 创建帧
        frame = [0] * frame_size
        frame[0] = self.FRAME_HEADER
        frame[1] = self.CMD_JOINT
        frame[2] = self.ARM_DATA_SIZE  # 数据长度
        frame[-1] = self.FRAME_FOOTER
        
        # 映射关节角度到各个舵机
        for servo_idx, (joint_idx, direction) in enumerate(self.joint_to_servo_map):
            # 应用方向系数(有些舵机需要反向)
            servo_angle_rad = joint_angles[joint_idx] * direction
            
            # 转换为硬件值
            hardware_value = self._rad_to_hardware_value(servo_angle_rad)
            
            # 写入到帧数据
            frame[3 + servo_idx*2] = hardware_value & 0xFF  # 低字节
            frame[3 + servo_idx*2 + 1] = (hardware_value >> 8) & 0xFF  # 高字节
        
        # 计算并设置校验和
        frame[-2] = self._calculate_checksum(frame)
        
        if self.debug_mode:
            angle_deg = [round(angle * self.RAD_TO_DEG, 2) for angle in joint_angles]
            logger.debug(f"发送关节角度(度): {angle_deg}")
            
        return frame
    
    def _build_gripper_frame(self, angle_rad: float) -> List[int]:
        """
        构建夹爪控制帧
        
        Args:
            angle_rad: 夹爪角度（弧度）
            
        Returns:
            List[int]: 控制帧字节列表
        """
        # 创建夹爪控制帧 (固定长度)
        frame = [0] * self.GRIPPER_FRAME_SIZE
        frame[0] = self.FRAME_HEADER
        frame[1] = self.CMD_GRIPPER
        frame[2] = 3  # 数据长度
        frame[3] = 1  # 夹爪ID
        frame[-1] = self.FRAME_FOOTER
        
        # 转换为硬件值
        gripper_value = self._rad_to_hardware_value_grip(angle_rad)
        
        # 写入夹爪角度
        frame[4] = gripper_value & 0xFF  # 低字节
        frame[5] = (gripper_value >> 8) & 0xFF  # 高字节
        
        # 计算并设置校验和
        frame[6] = self._calculate_checksum(frame)
        
        if self.debug_mode:
            angle_deg = round(angle_rad * self.RAD_TO_DEG, 2)
            logger.debug(f"发送夹爪角度: {angle_deg}度 ({angle_rad:.4f}弧度)")
            
        return frame
    
    def _build_command_frame(self, cmd_id: int, data: List[int]) -> List[int]:
        """
        构建命令帧
        
        Args:
            cmd_id: 命令ID
            data: 数据字节列表
            
        Returns:
            List[int]: 控制帧字节列表
        """
        # 计算帧大小：帧头(1)+命令(1)+长度(1)+数据(n)+校验(1)+帧尾(1)
        frame_size = len(data) + 5
        
        # 创建帧
        frame = [0] * frame_size
        frame[0] = self.FRAME_HEADER
        frame[1] = cmd_id
        frame[2] = len(data)  # 数据长度
        
        # 写入数据
        for i, d in enumerate(data):
            frame[3 + i] = d
            
        # 设置帧尾
        frame[-1] = self.FRAME_FOOTER
        
        # 计算并设置校验和
        frame[-2] = self._calculate_checksum(frame)
            
        return frame
    
    def _rad_to_hardware_value(self, angle_rad: float) -> int:
        """
        将弧度转换为硬件值(0-4095)
        
        Args:
            angle_rad: 角度（弧度）
            
        Returns:
            int: 硬件值
        """
        # 先转换为角度
        angle_deg = angle_rad * self.RAD_TO_DEG
        
        # 范围检查
        if angle_deg < -180.0 or angle_deg > 180.0:
            logger.warning(f"角度值超出范围: {angle_deg:.2f}度，会被截断")
            angle_deg = max(-180.0, min(180.0, angle_deg))
        
        # 转换公式: -180° → 0, 0° → 2048, +180° → 4095
        value = int((angle_deg + 180.0) / 360.0 * 4096)
        
        # 范围限制
        return max(0, min(4095, value))
    
    def _rad_to_hardware_value_grip(self, angle_rad: float) -> int:
        """
        将弧度转换为夹爪舵机值(2048-2900)
        
        Args:
            angle_rad: 角度（弧度）
            
        Returns:
            int: 硬件值
        """
        # 先转换为角度
        angle_deg = angle_rad * self.RAD_TO_DEG
        
        # 范围检查
        if angle_deg < 0:
            logger.warning(f"夹爪角度值超出范围: {angle_deg:.2f}度，会被截断")
            angle_deg = 0
        elif angle_deg > 100.0:
            logger.warning(f"夹爪角度值超出范围: {angle_deg:.2f}度，会被截断")
            angle_deg = 100.0

        servo_value_limit = 3290
        # 转换公式：0度对应2048，100度对应servo_value_limit 
        ratio = (servo_value_limit - 2048) / 100
        value = int(2048 + (angle_deg * ratio))
        
        # 范围限制
        return max(2048, min(servo_value_limit, value))
    
    def _calculate_checksum(self, frame: List[int]) -> int:
        """
        计算校验和
        
        Args:
            frame: 完整的数据帧
            
        Returns:
            int: 校验和
        """
        # 计算从第3个字节到倒数第3个字节的所有元素之和
        checksum = 0
        for i in range(3, len(frame) - 2):
            checksum += frame[i]
        
        # 对2取模
        return checksum % 2
    