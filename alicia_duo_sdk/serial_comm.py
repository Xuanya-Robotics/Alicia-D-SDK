import serial
import serial.tools.list_ports
import time
import logging
import os
from typing import List, Optional, Tuple
import threading
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SerialComm")
READ_LENGTH = 50
DEFAULT_LENGTH = 5
class SerialComm:
    """机械臂串口通信模块 - 简化版"""
    
    def __init__(self, lock: threading.Lock, port: str = "", baudrate: int = 921600, 
                timeout: float = 1.0, debug_mode: bool = False):
        """
        初始化串口通信模块
        
        Args:
            port: 串口名称，留空则自动搜索
            baudrate: 波特率
            timeout: 超时时间(秒)
            debug_mode: 是否启用调试模式
        """
        self.port_name = port
        self.baudrate = baudrate
        self.baudrate_default = 921600
        self.baudrate_macOS = 1000000
        self.timeout = timeout
        self.debug_mode = debug_mode
        
        self.serial_port = None
        self.last_log_time = 0
        self._last_print_time = 0

        self._lock = lock
        self._rx_buffer = bytearray()
        logger.info(f"初始化串口通信模块: 端口={port or '自动'}, 波特率={baudrate}")
        logger.info(f"调试模式: {'启用' if debug_mode else '禁用'}")
    
    def __del__(self):
        """析构函数，确保关闭串口"""
        self.disconnect()
    
    def connect(self) -> bool:
        """
        连接到串口设备
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 查找可用串口
            port = self.find_serial_port()
            
            # 没有找到可用串口
            if not port:
                logger.warning("未找到可用串口")
                return False
            
            logger.info(f"正在连接端口: {port}")
            
            # 关闭已有连接
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            # 检查串口是否是cu.usbserial，该串口通常为macOS
            if 'cu.usbserial' in port:

                # 检查波特率是否为macOS所能识别的
                if self.baudrate == self.baudrate_default:
                    self.baudrate = self.baudrate_macOS # 更改为macOS能识别的波特率1000000
                    logger.info(f"将波特率从默认 {self.baudrate_default} 调整为macOS所能识别的 {self.baudrate_macOS}")

                # 如果有指定波特率则不做更改，只log出来
                else:
                    logger.info(f"当前指定波特率为 {self.baudrate}, 该波特率macOS可能不能识别")

            # 设置串口参数
            self.serial_port = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            if self.serial_port.is_open:
                logger.info("串口连接成功")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"连接串口异常: {str(e)}")
            return False
    
    def disconnect(self):
        """断开串口连接"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("串口已关闭")
    
    def find_serial_port(self) -> str:
        """
        查找可用的串口设备
        
        Returns:
            str: 可用串口的路径，未找到则返回空字符串
        """
        # 记录当前时间，避免过频繁打印日志
        current_time = time.time()
        should_log = (current_time - self.last_log_time) >= 5.0  # 每5秒允许打印一次日志

        # 获取串口列表
        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as e:
            if should_log:
                logger.error(f"列出端口时异常: {str(e)}")
                self.last_log_time = current_time
            return ""
        
        # 如果有端口且应该打印日志
        if ports and should_log:
            port_names = [port.device for port in ports]
            logger.info(f"找到 {len(ports)} 个串口设备: {' '.join(port_names)}")
            self.last_log_time = current_time
        
        # 如果没有端口
        if not ports:
            return ""
        
        # 首先尝试使用指定的端口
        if self.port_name:
            for port in ports:
                if self.port_name in port.device:
                    if os.access(port.device, os.R_OK | os.W_OK):
                        if should_log:
                            logger.info(f"使用指定的端口: {port.device}")
                        return port.device
            
            if should_log:
                logger.warning(f"指定的端口 {self.port_name} 不可用，将搜索其他设备")
        
        # 尝试找到可用的设备
        for port in ports:
            #尝试找到可用的ttyUSB设备
            if "ttyUSB" in port.device:
                if os.access(port.device, os.R_OK | os.W_OK):
                    if should_log:
                        logger.info(f"找到可用设备: {port.device}")
                    return port.device
                
            #尝试找到可用的cu.usbserial设备
            elif "cu.usbserial" in port.device:
                if os.access(port.device, os.R_OK | os.W_OK):
                    if should_log:
                        logger.info(f"找到可用设备: {port.device}")
                    return port.device
           
            #尝试找到可用的COM设备
            elif "COM" in port.device:
                if os.access(port.device, os.R_OK | os.W_OK):
                    if should_log:
                        logger.info(f"找到可用设备: {port.device}")
                    return port.device
        if should_log:
            logger.warning("未找到可用的ttyUSB或者cu.usbserial设备")
        return ""
    
    def send_data(self, data: List[int]) -> bool:
        """
        发送数据到串口
        
        Args:
            data: 要发送的字节数据列表
            
        Returns:
            bool: 是否发送成功
        """
        with self._lock:
            try:
                if not self.serial_port or not self.serial_port.is_open:
                    logger.warning("串口未打开，尝试重新连接")
                    if not self.connect():
                        logger.error("无法连接到串口")
                        return False
                
                # 转换为字节数组
                data_bytes = bytes(data)
                
                # 写入数据
                bytes_written = self.serial_port.write(data_bytes)
                
                if bytes_written != len(data):
                    logger.warning(f"只写入了 {bytes_written} 字节，应为 {len(data)} 字节")
                    return False
                
                if self.debug_mode:
                    self._print_hex_frame(data, 0)
    
                return True
                    
            except Exception as e:
                logger.error(f"发送数据时异常: {str(e)}")
                return False
    
    def read_frame(self) -> Optional[List[int]]:
        """
        读取一帧数据（非阻塞，如果没有完整帧则返回None）
        
        Returns:
            Optional[List[int]]: 完整的数据帧，如果没有则返回None
        """
        try:
            if not self.serial_port or not self.serial_port.is_open:
                if not self.connect():
                    return None
            
            # 检查是否有数据可读
            if self.serial_port.in_waiting == 0:
                return None
            
            # 导入串口缓存数据
            self._rx_buffer += self.serial_port.read(self.serial_port.in_waiting)

            while len(self._rx_buffer) >= READ_LENGTH:
                # Step 1: 同步到帧头 0xAA
                if self._rx_buffer[0] != 0xAA:
                    self._rx_buffer.pop(0)
                    continue

                frame_length = self._rx_buffer[2] + DEFAULT_LENGTH       # 数据长度 + 基础长度
                candidate = self._rx_buffer[:frame_length]

                # Step 2: 验证帧尾和校验
                valid_tail = candidate[-1] == 0xFF
                valid_checksum = self._serial_data_check(candidate)

                parsed = {
                "timestamp": datetime.now().isoformat(),
                "raw": ' '.join(f"{b:02X}" for b in candidate),
                "raw_decimal": list(candidate),
                "valid": valid_tail and valid_checksum
            }
            
                if self.debug_mode:
                    now = time.time()
                    if self.debug_mode and now - self._last_print_time > 1.0:
                        logger.info(f"[Frame] {parsed['raw_decimal']} {'(OK)' if parsed['valid'] else '(Invalid)'}")
                        self._last_print_time = now

                self._rx_buffer = self._rx_buffer[frame_length:]

                 # Step 4: 若缓存过大，强制同步（防炸）
                if len(self._rx_buffer) > 1000:
                    aa_index = self._rx_buffer.find(0xAA)
                    if aa_index == -1:
                        self._rx_buffer.clear()
                    else:
                        self._rx_buffer = self._rx_buffer[aa_index:]

                if parsed["valid"]:
                    self._rx_buffer.clear()
                    return candidate    
                
        except Exception as e:
            logger.error(f"读取数据异常: {str(e)}")
            return 9999999
    
    def _serial_data_check(self, frame: bytearray) -> bool:
        """
        验证数据的校验和
        
        Args:
            data: 数据帧
            
        Returns:
            bool: 校验是否通过
        """
        data_len = frame[2]
        if len(frame) != data_len + DEFAULT_LENGTH:
            return False

        payload = frame[3:3 + data_len]
        checksum = frame[3 + data_len]
        return checksum == self._calculate_checksum(payload)
    
    def _calculate_checksum(self,data) -> int:
        """
        计算数据的校验和
        
        Args:
            data: 数据帧
            
        Returns:
            int: 校验和
        """
        sum_value = 0
        for i in range(len(data)):
            sum_value += data[i]
        return sum_value % 2
    
    def _print_hex_frame(self, data: List[int], type_code: int):
        """
        打印十六进制数据
        
        Args:
            data: 数据帧
            type_code: 0=发送数据, 1=接收数据, 其他=部分数据
        """
        if not self.debug_mode:
            return
        
        prefix = {
            0: "发送数据: ",
            1: "数据接收: ",
            2: "部分数据: "
        }.get(type_code, "未知数据: ")
        
        hex_str = " ".join([f"{byte:02X}" for byte in data])
        logger.info(f"{prefix}{hex_str}")

