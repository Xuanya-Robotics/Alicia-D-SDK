import serial
import serial.tools.list_ports
import time
import logging
import os
from typing import List, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SerialComm")

class SerialComm:
    """机械臂串口通信模块 - 简化版"""
    
    def __init__(self, port: str = "", baudrate: int = 921600, 
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
        self.baudrate_macOS = 1000000
        self.timeout = timeout
        self.debug_mode = debug_mode
        
        self.serial_port = None
        self.last_log_time = 0
        
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
                print("Found cu port")

                # 检查波特率是否为macOS所能识别的
                if self.baudrate == 921600:
                    self.baudrate = self.baudrate_macOS
                    logger.info(f"将波特率从默认 {self.baudrate} 调整为macOS所能识别的 {self.baudrate_macOS}")

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
            
            # 寻找帧起始标记 0xAA
            frame_buffer = []
            start_found = False
            
            # 设置一个安全的最大读取次数，避免无限循环
            max_attempts = 100
            attempts = 0
            
            while attempts < max_attempts:
                attempts += 1
                
                if self.serial_port.in_waiting == 0:
                    # 没有更多数据可读
                    break
                
                byte_data = self.serial_port.read(1)
                if not byte_data:
                    continue
    
                byte_val = byte_data[0]
                if not start_found:
                    # 寻找起始标记
                    if byte_val == 0xAA:
                        frame_buffer = [byte_val]
                        start_found = True
                else:
                    # 构建帧
                    frame_buffer.append(byte_val)
                    
                    # 检查是否找到帧结束标记
                    if byte_val == 0xFF and len(frame_buffer) >= 3:
                        # 检查帧长度是否符合预期
                        if len(frame_buffer) >= 3:  # 确保有足够的数据读取长度字段
                            expected_length = frame_buffer[2] + 5  # 数据长度+5等于帧长度
                            
                            if len(frame_buffer) == expected_length:
                                # 验证校验和
                                if self._serial_data_check(frame_buffer):
                                    if self.debug_mode:
                                        self._print_hex_frame(frame_buffer, 1)
                                    return frame_buffer
                                else:
                                    logger.warning("帧校验和验证失败")
                                    start_found = False
                            elif expected_length > 64 or len(frame_buffer) > 64:
                                # 帧太长，认为帧错误
                                logger.warning("帧太长，丢弃")
                                start_found = False
            
            # 如果找到了起始标记但没有完成帧，保留给下次读取
            if start_found and frame_buffer:
                logger.debug(f"读取到部分帧，长度: {len(frame_buffer)}")
                if self.debug_mode:
                    self._print_hex_frame(frame_buffer, 2)  # 输出部分帧
            
            return None
                
        except Exception as e:
            logger.error(f"读取数据异常: {str(e)}")
            return 9999999
    
    def draw_frame(self) -> Optional[List[int]]:
        try:
            if not self.serial_port or not self.serial_port.is_open:
                if not self.connect():
                    return None
            
            # 检查是否有数据可读
            if self.serial_port.in_waiting == 0:
                return None
            
            raw_bytes = self.serial_port.read(self.serial_port.in_waiting)
            if raw_bytes:
                print("[原始串口数据] ->", ' '.join(f'{b:02X}' for b in raw_bytes))

        except Exception as e:
            print(f"[串口调试异常] {e}")
            return 9999
            




    def _serial_data_check(self, data: List[int]) -> bool:
        """
        验证数据的校验和
        
        Args:
            data: 数据帧
            
        Returns:
            bool: 校验是否通过
        """
        if len(data) < 4:
            return False
        
        calculated_check = self._sum_elements(data) % 2
        received_check = data[-2]  # 倒数第二个字节
        
        return calculated_check == received_check
    
    def _sum_elements(self, data: List[int]) -> int:
        """
        计算数据的校验和
        
        Args:
            data: 数据帧
            
        Returns:
            int: 校验和
        """
        if len(data) < 4:
            logger.error("数据数组太小，无法计算校验和")
            return 0
        
        # 计算从第3个字节到倒数第2个字节之前的所有元素的和
        sum_value = 0
        for i in range(3, len(data) - 2):
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

    # def _print_hex_frame(self, data: List[int], type_code: int):
    #     """
    #     打印十六进制数据
        
    #     Args:
    #         data: 数据帧
    #         type_code: 0=发送数据, 1=接收数据, 其他=部分数据
    #     """
    #     if not self.debug_mode:
    #         return
        
    #     # 只打印以AA 02开头的数据帧
    #     if len(data) < 2 or data[0] != 0xAA or data[1] != 0x02:
    #         return
        
    #     prefix = {
    #         0: "发送数据: ",
    #         1: "数据接收: ",
    #         2: "部分数据: "
    #     }.get(type_code, "未知数据: ")
        
    #     hex_str = " ".join([f"{byte:02X}" for byte in data])
    #     logger.info(f"{prefix}{hex_str}")