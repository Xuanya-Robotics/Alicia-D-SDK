import math
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, NamedTuple

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataParser")

class JointState(NamedTuple):
    """关节状态数据结构"""
    angles: List[float]  # 六个关节角度(弧度)
    gripper: float       # 夹爪角度(弧度)
    timestamp: float     # 时间戳(秒)
    button1: bool        # 按钮1状态
    button2: bool        # 按钮2状态

class DataParser:
    """机械臂数据解析模块"""
    
    # 常量定义
    DEG_TO_RAD = math.pi / 180.0  # 角度转弧度系数
    RAD_TO_DEG = 180.0 / math.pi  # 弧度转角度系数
    
    # 指令ID
    CMD_GRIPPER = 0x02     # 夹爪控制与行程反馈
    CMD_ZERO_POS = 0x03    # 机械臂以当前位置为零点  
    CMD_JOINT = 0x04       # 机械臂角度反馈与控制
    CMD_MULTI_ARM = 0x06   # 四机械臂角度反馈与控制
    CMD_TORQUE = 0x13      # 机械臂力矩控制
    CMD_ERROR = 0xEE       # 错误反馈
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化数据解析器
        
        Args:
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode
        
        # 存储最新数据
        self._joint_angles = [0.0] * 6  # 六个关节角度(弧度)
        self._gripper_angle = 0.0       # 夹爪角度(弧度)
        self._button1 = False           # 按钮1状态
        self._button2 = False           # 按钮2状态
        self._last_update_time = 0.0    # 最后更新时间
        
        logger.info("初始化数据解析模块")
        if debug_mode:
            logger.info("调试模式: 启用")
    
    def parse_frame(self, frame: List[int]) -> Optional[Dict]:
        """
        解析数据帧
        
        Args:
            frame: 完整的数据帧(字节列表)
            
        Returns:
            Dict: 解析结果，如果解析失败则返回None
        """
        # 基本帧格式检查
        if len(frame) < 5 or frame[0] != 0xAA or frame[-1] != 0xFF:
            if self.debug_mode:
                logger.warning(f"无效数据帧: {self._bytes_to_hex(frame)}")
            return None
        
        # 解析指令ID和数据长度
        cmd_id = frame[1]
        data_len = frame[2]
        
        # 验证数据长度
        if len(frame) != data_len + 5:  # 帧头(1) + 指令ID(1) + 长度(1) + 数据(n) + 校验(1) + 帧尾(1)
            if self.debug_mode:
                logger.warning(f"数据长度不匹配: 预期 {data_len + 5}, 实际 {len(frame)}")
            return None
        
        # 校验和验证
        if not self._verify_checksum(frame):
            if self.debug_mode:
                logger.warning(f"校验和错误: {self._bytes_to_hex(frame)}")
            return None
        
        # 根据指令ID解析数据
        if cmd_id == self.CMD_JOINT:
            return self._parse_joint_data(frame)
        elif cmd_id == self.CMD_GRIPPER:
            return self._parse_gripper_data(frame)
        elif cmd_id == self.CMD_ERROR:
            return self._parse_error_data(frame)
        else:
            if self.debug_mode:
                logger.debug(f"未处理的指令ID: 0x{cmd_id:02X}")
            return None
    
    def get_joint_state(self) -> JointState:
        """
        获取关节状态
        
        Returns:
            JointState: 当前关节状态
        """
        return JointState(
            angles=self._joint_angles.copy(),
            gripper=self._gripper_angle,
            timestamp=self._last_update_time,
            button1=self._button1,
            button2=self._button2
        )
    
    def _parse_joint_data(self, frame: List[int]) -> Dict:
        """
        解析关节数据帧 (0x04)
        
        Args:
            frame: 完整的数据帧
            
        Returns:
            Dict: 解析结果
        """
        # 检查数据长度
        if frame[2] != 18:  # 0x12对应十进制18 (9个舵机 * 2字节)
            logger.warning(f"关节数据长度错误: {frame[2]}")
            return None
        
        # 舵机到关节的映射表 - 与ROS一致
        servo_to_joint_map = {
            0: (0, 1.0),    # 舵机1 -> 关节1 (正向)
            1: None,        # 舵机2 -> 忽略(重复)
            2: (1, 1.0),    # 舵机3 -> 关节2 (正向)
            3: None,        # 舵机4 -> 忽略(重复反向)
            4: (2, 1.0),    # 舵机5 -> 关节3 (正向)
            5: None,        # 舵机6 -> 忽略(重复反向)
            6: (3, 1.0),    # 舵机7 -> 关节4 (正向)
            7: (4, 1.0),    # 舵机8 -> 关节5 (正向)
            8: (5, 1.0),    # 舵机9 -> 关节6 (正向)
        }
        
        # 初始化关节角度数组
        joint_values = [0.0] * 6
        servo_values = []
        
        # 处理9个舵机数据
        for i in range(9):
            # 数据索引计算
            byte_idx = 3 + i * 2
            if byte_idx + 1 >= len(frame):
                logger.warning(f"舵机数据越界: 索引{byte_idx}超出范围")
                continue
            
            # 解析舵机原始值
            low_byte = frame[byte_idx]
            high_byte = frame[byte_idx + 1]
            servo_value = (low_byte & 0xFF) | ((high_byte & 0xFF) << 8)
            servo_values.append(servo_value)
            
            # 映射到关节
            mapping = servo_to_joint_map.get(i)
            if mapping is not None:
                joint_idx, direction = mapping
                # 转换为弧度并应用方向系数
                angle_rad = self._value_to_radians(servo_value) * direction
                joint_values[joint_idx] = angle_rad
        
        # 更新存储的数据
        self._joint_angles = joint_values
        self._last_update_time = time.time()
        
        if self.debug_mode:
            degrees = [round(rad * self.RAD_TO_DEG, 2) for rad in joint_values]
            logger.debug(f"关节角度(度): {degrees}")
        
        return {
            "type": "joint_data",
            "angles": joint_values,
            "servo_values": servo_values,
            "timestamp": self._last_update_time
        }

    def _value_to_radians(self, value: int) -> float:
        """
        将舵机值转换为弧度值 - 与ROS代码保持一致
        
        Args:
            value: 舵机值(0-4095)
            
        Returns:
            float: 弧度值
        """
        try:
            # 值范围检查
            if value < 0 or value > 4095:
                logger.warning(f"舵机值超出范围: {value} (有效范围0-4095)")
                value = max(0, min(value, 4095))
            
            # 转换为角度: -180到+180度
            # 使用与ROS代码一致的转换公式
            angle_deg = -180.0 + (value / 2048.0) * 180.0
            
            # 转换为弧度并返回
            return angle_deg * self.DEG_TO_RAD
                
        except Exception as e:
            logger.error(f"值转换异常: {str(e)}")
            return 0.0
    
    def _parse_gripper_data(self, frame: List[int]) -> Dict:
        """
        解析夹爪数据帧 (0x02)
        
        Args:
            frame: 完整的数据帧
            
        Returns:
            Dict: 解析结果
        """
        # 解析按钮状态 (如果数据帧中包含)
        button1 = False
        button2 = False
        if len(frame) >= 10:  # 确保有足够的数据
            # 假设第8、9个字节包含按钮状态信息
            button1 = (frame[8] & 0x01) != 0
            button2 = (frame[9] & 0x01) != 0

        # 检查最小长度
        if len(frame) < 8:
            logger.warning("夹爪数据帧长度不足")
            return None
        
        if button1:
             gripper_raw = frame[6] | (frame[7] << 8)
        # 从字节4-5提取夹爪角度
        else:
            gripper_raw = frame[4] | (frame[5] << 8)
        
        # 范围检查
        if gripper_raw < 2048 or gripper_raw > 2900:
            gripper_raw = max(2048, min(gripper_raw, 2900))
        
        # 转换为角度 (0-100度)
        angle_deg = (gripper_raw - 2048) / 8.52
        
        # 转换为弧度
        gripper_rad = angle_deg * self.DEG_TO_RAD
        

        
        # 更新存储的数据
        self._gripper_angle = gripper_rad
        self._button1 = button1
        self._button2 = button2
        self._last_update_time = time.time()
        
        if self.debug_mode:
            logger.debug(f"夹爪原始值: {gripper_raw}, 角度: {angle_deg:.2f}度, 弧度: {gripper_rad:.4f}")
            logger.debug(f"按钮状态: 按钮1={'按下' if button1 else '释放'}, 按钮2={'按下' if button2 else '释放'}")
        
        return {
            "type": "gripper_data",
            "gripper_angle": gripper_rad,
            "button1": button1,
            "button2": button2,
            "timestamp": self._last_update_time
        }
    
    def _parse_error_data(self, frame: List[int]) -> Dict:
        """
        解析错误数据帧 (0xEE)
        
        Args:
            frame: 完整的数据帧
            
        Returns:
            Dict: 解析结果
        """
        # 检查最小长度
        if len(frame) < 7:
            logger.warning("错误数据帧长度不足")
            return None
        
        # 提取错误码和附加信息
        error_code = frame[3]
        error_param = frame[4]
        
        error_types = {
            0x00: "包头/包尾或长度错误",
            0x01: "校验错误",
            0x02: "模式错误",
            0x03: "ID无效",
        }
        
        error_message = error_types.get(error_code, f"未知错误(0x{error_code:02X})")
        
        logger.warning(f"设备错误: {error_message}, 参数: 0x{error_param:02X}")
        
        return {
            "type": "error_data",
            "error_code": error_code,
            "error_param": error_param,
            "error_message": error_message,
            "timestamp": time.time()
        }
    
    def _bytes_to_radians(self, byte_array: List[int]) -> float:
        """
        将字节数组转换为弧度值
        
        Args:
            byte_array: 2字节数组
            
        Returns:
            float: 弧度值
        """
        try:
            if len(byte_array) != 2:
                logger.warning(f"数据长度错误：需要2个字节，实际{len(byte_array)}个字节")
                return 0.0
            
            # 构造16位整数
            hex_value = (byte_array[0] & 0xFF) | ((byte_array[1] & 0xFF) << 8)
            
            # 值范围检查
            if hex_value < 0 or hex_value > 4095:
                logger.warning(f"舵机值超出范围: {hex_value} (有效范围0-4095)")
                hex_value = max(0, min(hex_value, 4095))
            
            # 转换为角度: -180到+180度
            angle_deg = -180.0 + (hex_value / 2048.0) * 180.0
            
            # 转换为弧度并返回
            return angle_deg * self.DEG_TO_RAD
                
        except Exception as e:
            logger.error(f"字节转换异常: {str(e)}")
            return 0.0
    
    def _verify_checksum(self, frame: List[int]) -> bool:
        """
        验证帧的校验和
        
        Args:
            frame: 完整的数据帧
            
        Returns:
            bool: 校验是否通过
        """
        if len(frame) < 4:
            return False
        
        # 计算从第3个字节到倒数第3个字节的所有元素之和
        checksum = 0
        for i in range(3, len(frame) - 2):
            checksum += frame[i]
        
        checksum %= 2
        received_checksum = frame[-2]
        
        return checksum == received_checksum
    
    def _bytes_to_hex(self, data: List[int]) -> str:
        """
        将字节数组转换为十六进制字符串
        
        Args:
            data: 字节数组
            
        Returns:
            str: 十六进制字符串
        """
        return " ".join([f"{b:02X}" for b in data])