import math
import time
import logging
import threading
from typing import List, Optional, Union, Tuple, Dict
import PyKDL

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
        
        # 创建串口通信模块和数据解析器
        self.serial_comm = SerialComm(port=port, baudrate=baudrate, debug_mode=debug_mode)
        self.data_parser = DataParser(debug_mode=debug_mode)
        
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
        self._stop_thread = threading.Event()
        self._thread_running = False

        self.chain= self.create_alicia_duo_kinematic_chain()
        
        logger.info("初始化机械臂控制模块")
        logger.info(f"调试模式: {'启用' if debug_mode else '禁用'}")

        self.disconnect()

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
        update_interval = 0.01  # 更新间隔，单位：秒
        error_count = 0
        max_consecutive_errors = 10
        
        logger.info("状态更新线程开始运行")
        
        while not self._stop_thread.is_set():
            try:
                # 读取一帧数据
                frame = self.serial_comm.read_frame()
                if frame:
                    # 解析数据帧，更新内部状态
                    self.data_parser.parse_frame(frame)
                    # 成功读取帧后重置错误计数

                if frame == 9999999:
                    #logger.error("检测到严重的串口通信异常，机械臂可能已断开连接")
                    # 停止线程并清理资源
                    #self._destroy_self("机械臂连接已丢失")
                    break  # 确保线程退出循环
                    
            except Exception as e:
                logger.error(f"状态更新线程异常: {str(e)}")
                
                
                break
        self._thread_running = False
        logger.info("状态更新线程已结束运行")
    
    def read_joint_angles(self) -> Optional[List[float]]:
        """
        读取关节角度（弧度）
        
        Returns:
            Optional[List[float]]: 6个关节的角度列表（弧度），读取失败则返回None
        """
        # 直接返回当前状态，不需要额外读取数据
        return self.data_parser.get_joint_state().angles
    
    def read_gripper_data(self) -> Tuple[float, bool, bool]:
        """
        读取夹爪数据
        
        Returns:
            Tuple[float, bool, bool]: 夹爪角度（弧度）、按钮1状态、按钮2状态
        """
        # 直接返回当前状态，不需要额外读取数据
        state = self.data_parser.get_joint_state()
        return state.gripper, state.button1, state.button2
    
    
    def read_joint_state(self) -> JointState:
        """
        读取完整的机械臂状态。
        由于已有后台线程持续更新，所以直接返回当前状态。
        """
        # 直接返回当前状态，不需要额外读取数据
        return self.data_parser.get_joint_state()
    
    def set_joint_angles(self, joint_angles: List[float], gripper_angle: float = None, wait_for_completion: bool = True, timeout: float = 5.0, tolerance: float = 0.08) -> bool:
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
        
        # 如果需要等待运动完成
        #print(f"wait_for_completion: {wait_for_completion}, result: {result}")
        if wait_for_completion and result:
            logger.info("等待机械臂运动完成")
            start_time = time.time()
            while time.time() - start_time < timeout:
                # 读取当前关节状态 - 因为有后台线程更新，所以直接获取最新状态
                current_state = self.data_parser.get_joint_state()
                current_angles = current_state.angles
                
                # 检查是否到达目标位置
                reached = True
                for i in range(self.joint_count):
                    if abs(current_angles[i] - joint_angles[i]) > tolerance:
                        reached = False
                        break
                
                # 如果到达目标位置，退出循环
                if reached:
                    if self.debug_mode:
                        logger.debug("机械臂到达目标位置")
                    break
                
                # 短暂延时，避免过于频繁检查
                time.sleep(0.02)
            
            # 检查是否超时
            if time.time() - start_time >= timeout:
                logger.warning("等待机械臂运动完成超时")
                return False
            
        return result
    
    def set_gripper(self, angle_rad: float, wait_for_completion: bool = True, timeout: float = 5.0, tolerance: float = 0.1) -> bool:
        """
        设置夹爪角度（弧度）
        
        Args:
            angle_rad: 夹爪角度（弧度）
            wait_for_completion: 是否等待夹爪运动完成
            timeout: 等待夹爪运动完成的超时时间（秒）
            tolerance: 角度误差容忍度（弧度）
            
        Returns:
            bool: 命令是否成功发送和执行
        """
        # 构造夹爪控制帧
        frame = self._build_gripper_frame(angle_rad)
        
        # 发送夹爪控制命令
        result = self.serial_comm.send_data(frame)
        
        # 如果需要等待夹爪运动完成
        if wait_for_completion and result:
            if self.debug_mode:
                logger.debug(f"等待夹爪运动到目标位置: {round(angle_rad * self.RAD_TO_DEG, 2)}度")
                
            start_time = time.time()
            while time.time() - start_time < timeout:
                # 读取当前夹爪状态 - 因为有后台线程更新，所以直接获取最新状态
                current_state = self.data_parser.get_joint_state()
                gripper_angle = current_state.gripper
                
                # 检查是否到达目标位置
                if abs(gripper_angle - angle_rad) <= tolerance:
                    if self.debug_mode:
                        logger.debug("夹爪到达目标位置")
                    break
                
                # 短暂延时，避免过于频繁检查
                time.sleep(0.02)
            
            # 检查是否超时
            if time.time() - start_time >= timeout:
                logger.warning("等待夹爪运动完成超时")
                return False
        
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
        return self.serial_comm.send_data(frame)
    
    def disable_torque(self) -> bool:
        """
        禁用力矩控制（使机械臂可以自由移动）
        
        Returns:
            bool: 命令是否成功发送
        """
        # 构造力矩禁用帧
        frame = self._build_command_frame(self.CMD_TORQUE, [0x00])
        
        # 发送力矩禁用命令
        return self.serial_comm.send_data(frame)
    
    def _build_joint_frame(self, joint_angles: List[float]) -> List[int]:
        """
        构建关节控制帧
        
        Args:
            joint_angles: 6个关节的角度列表（弧度）
            
        Returns:
            List[int]: 控制帧字节列表
        """
        # 计算帧大小：帧头(1)+命令(1)+长度(1)+数据(舵机数*2)+校验(1)+帧尾(1)
        frame_size = self.servo_count * 2 + 5
        
        # 创建帧
        frame = [0] * frame_size
        frame[0] = self.FRAME_HEADER
        frame[1] = self.CMD_JOINT
        frame[2] = self.servo_count * 2  # 数据长度
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
        frame = [0] * 8
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
        
        # 转换公式：0度对应2048，100度对应2900
        value = int(2048 + (angle_deg * 8.52))
        
        # 范围限制
        return max(2048, min(2900, value))
    
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
    
    @staticmethod # <--- 添加装饰器
    def create_alicia_duo_kinematic_chain():
        """
        Hardcodes the kinematic chain for alicia_duo based on the URDF.
        Returns a PyKDL.Chain object.
        """
        chain = PyKDL.Chain()
        # Helper to create PyKDL.Frame from xyz, rpy
        def create_frame(xyz, rpy):
            return PyKDL.Frame(
                PyKDL.Rotation.RPY(rpy[0], rpy[1], rpy[2]),
                PyKDL.Vector(xyz[0], xyz[1], xyz[2])
            )
        
        # For revolute joints, the origin within the Joint object itself is typically (0,0,0)
        # as the transformation to the joint's position is handled by the segment's structure.
        joint_local_origin = PyKDL.Vector(0.0, 0.0, 0.0) # Common for revolute joints

        # --- Segment 0: Fixed transform from base_link to Joint1 origin ---
        f_tip_base_to_j1 = create_frame(xyz=[0, 0, 0.1445], rpy=[0, 0, 0])
        chain.addSegment(PyKDL.Segment("base_to_j1_segment",
                                    PyKDL.Joint(PyKDL.Joint.Fixed), # Using .Fixed as per previous discussion
                                    f_tip_base_to_j1))

        # --- Segment 1: Joint1 ---
        joint1_axis = PyKDL.Vector(0, 0, 1)
        f_tip_j1_to_j2 = create_frame(xyz=[0, 0, 0.025106], rpy=[-1.5708, -1.5708, 0])
        chain.addSegment(PyKDL.Segment("Link1_J1",
                                    PyKDL.Joint("Joint1", joint_local_origin, joint1_axis, PyKDL.Joint.RotAxis), # MODIFIED
                                    f_tip_j1_to_j2))

        # --- Segment 2: Joint2 ---
        joint2_axis = PyKDL.Vector(0, 0, -1)
        f_tip_j2_to_j3 = create_frame(xyz=[0.22367, 0.022494, -5E-05], rpy=[0, 0, 2.3562])
        chain.addSegment(PyKDL.Segment("Link2_J2",
                                    PyKDL.Joint("Joint2", joint_local_origin, joint2_axis, PyKDL.Joint.RotAxis), # MODIFIED
                                    f_tip_j2_to_j3))

        # --- Segment 3: Joint3 ---
        joint3_axis = PyKDL.Vector(0, 0, -1)
        f_tip_j3_to_j4 = create_frame(xyz=[0.0988, 0.00211, -0.0001], rpy=[1.5708, 0, 1.5708])
        chain.addSegment(PyKDL.Segment("Link3_J3",
                                    PyKDL.Joint("Joint3", joint_local_origin, joint3_axis, PyKDL.Joint.RotAxis), # MODIFIED
                                    f_tip_j3_to_j4))

        # --- Segment 4: Joint4 ---
        joint4_axis = PyKDL.Vector(0, 0, -1)
        f_tip_j4_to_j5 = create_frame(xyz=[0, -0.0007, 0.12011], rpy=[-1.5708, 0, 0])
        chain.addSegment(PyKDL.Segment("Link4_J4",
                                    PyKDL.Joint("Joint4", joint_local_origin, joint4_axis, PyKDL.Joint.RotAxis), # MODIFIED
                                    f_tip_j4_to_j5))

        # --- Segment 5: Joint5 ---
        joint5_axis = PyKDL.Vector(0, 0, -1)
        f_tip_j5_to_j6 = create_frame(xyz=[-0.0038938, -0.0573, 0.0008], rpy=[1.5708, 0, 0])
        chain.addSegment(PyKDL.Segment("Link5_J5",
                                    PyKDL.Joint("Joint5", joint_local_origin, joint5_axis, PyKDL.Joint.RotAxis), # MODIFIED
                                    f_tip_j5_to_j6))

        # --- Segment 6: Joint6 ---
        T_J6_to_GraspBase = create_frame(xyz=[0,0,0], rpy=[0,0,0])
        T_GraspBase_to_tool0 = create_frame(xyz=[0.00275, 0.0008332, 0.13779], rpy=[0,0,0])
        f_tip_j6_to_tool0 = T_J6_to_GraspBase * T_GraspBase_to_tool0

        joint6_axis = PyKDL.Vector(0, 0, -1)
        chain.addSegment(PyKDL.Segment("Link6_J6_to_tool0",
                                    PyKDL.Joint("Joint6", joint_local_origin, joint6_axis, PyKDL.Joint.RotAxis), # MODIFIED
                                    f_tip_j6_to_tool0))
        return chain
    
    def forward_kinematics_alicia_duo(self,joint_angles_rad):
        """
        Calculates the forward kinematics for the alicia_duo robot.

        Args:
            joint_angles_rad (list or tuple of 6 floats): Joint angles in radians
                                                        for Joint1 to Joint6.

        Returns:
            tuple: (position_xyz, quaternion_xyzw, yaw_rad)
                position_xyz (list): [x, y, z] of tool0
                quaternion_xyzw (list): [x, y, z, w] of tool0
                yaw_rad (float): Yaw angle of tool0 in radians
        """
        if len(joint_angles_rad) != 6:
            raise ValueError("Expected 6 joint angles.")

        chain = self.chain  # Use the pre-defined kinematic chain
        fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)

        # PyKDL JntArray for joint angles. Note that the chain has 7 segments,
        # but only 6 are actuated. The JntArray size should match num actuated joints.
        kdl_joint_angles = PyKDL.JntArray(6)
        for i in range(6):
            kdl_joint_angles[i] = joint_angles_rad[i]

        # Frame to store the result
        tool0_frame = PyKDL.Frame()

        # Calculate forward kinematics
        ret = fk_solver.JntToCart(kdl_joint_angles, tool0_frame)
        if ret < 0:
            raise RuntimeError("PyKDL forward kinematics solver failed.")

        # Extract position
        pos = tool0_frame.p
        position_xyz = [pos.x(), pos.y(), pos.z()]

        # Extract rotation (quaternion and RPY)
        rot = tool0_frame.M
        quaternion_xyzw = list(rot.GetQuaternion()) # Returns (x, y, z, w)
        rpy = rot.GetRPY() # Returns (roll, pitch, yaw)

        return position_xyz, quaternion_xyzw, rpy
    

    def inverse_kinematics_alicia_duo(self, target_position_xyz, target_quaternion_xyzw, initial_joint_angles_rad=None):
        """
        计算alicia_duo机器人的逆运动学，仅使用ChainIkSolverPos_NR_JL求解器。
        
        参数:
            chain (PyKDL.Chain): 机器人运动链。
            target_position_xyz (list of 3 floats): 工具目标位置[x, y, z]。
            target_quaternion_xyzw (list of 4 floats): 工具目标四元数[x, y, z, w]。
            initial_joint_angles_rad (list of 6 floats, optional): 关节角度初始猜测值。
                                                                若为None，使用零位姿态。
        
        返回:
            list of 6 floats: 成功求解的关节角度(弧度)，失败则返回None。
        """
        # 基本参数验证
        chain=self.chain

        num_joints = chain.getNrOfJoints()

        # 设置关节限制
        q_min_rad = PyKDL.JntArray(num_joints)
        q_max_rad = PyKDL.JntArray(num_joints)
        
        margin = 0.1  # 10度余量
        limits = [
            (-2.16 + margin, 2.16 - margin),  # Joint 1
            (-3.14 + margin, 3.14 - margin),  # Joint 2
            (-3.14 + margin, 3.14 - margin),  # Joint 3
            (-3.14 + margin, 3.14 - margin),  # Joint 4
            (-3.14 + margin, 3.14 - margin),  # Joint 5
            (-3.14 + margin, 3.14 - margin)   # Joint 6
        ]
        
        for i in range(num_joints):
            q_min_rad[i] = limits[i][0]
            q_max_rad[i] = limits[i][1]

        # 准备目标帧
        target_vector = PyKDL.Vector(target_position_xyz[0], target_position_xyz[1], target_position_xyz[2])
        
        # 四元数标准化
        quat_array = np.array(target_quaternion_xyzw, dtype=float)
        quat_norm = np.linalg.norm(quat_array)
        if quat_norm == 0:
            raise ValueError("四元数范数为零")
        quat_normalized = quat_array / quat_norm
        
        try:
            target_rotation = PyKDL.Rotation.Quaternion(
                float(quat_normalized[0]), 
                float(quat_normalized[1]), 
                float(quat_normalized[2]), 
                float(quat_normalized[3])
            )
            target_frame = PyKDL.Frame(target_rotation, target_vector)
        except Exception as e:
            print(f"创建目标帧时出错: {e}")
            return None

        # 设置初始猜测
        if initial_joint_angles_rad is None or len(initial_joint_angles_rad) != num_joints:
            initial_joint_angles_rad = [0.0] * num_joints
        
        q_init = PyKDL.JntArray(num_joints)
        for i in range(num_joints):
            # 确保初始猜测在关节限制内
            q_init[i] = max(limits[i][0], min(limits[i][1], initial_joint_angles_rad[i]))
        
        # 创建求解器
        fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
        vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
        ik_solver = PyKDL.ChainIkSolverPos_NR_JL(chain, q_min_rad, q_max_rad,
                                                fk_solver, vel_solver,
                                                2500, 0.3)  # 使用固定的迭代次数和精度
        
        # 求解
        q_out = PyKDL.JntArray(num_joints)
        ret_val = ik_solver.CartToJnt(q_init, target_frame, q_out)
        
        # 检查结果
        if ret_val >= 0:
            # 求解成功
            solved_angles = [q_out[i] for i in range(num_joints)]
            return solved_angles
        print(f"IK求解失败 (返回值: {ret_val})")
        return None