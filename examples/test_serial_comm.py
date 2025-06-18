import serial

port = "/dev/cu.usbserial-1440"  # 你的串口设备名
baudrate = 115200  # 或你设备要求的 921600
ser = serial.Serial(port, baudrate, timeout=1)

print("开始读取串口数据（查找帧头 AA）...")
import serial

ser = serial.Serial('/dev/cu.usbserial-1440', 115200, timeout=1)
frame = bytearray()

try:
    while True:
        data = ser.read(1)
        if not data:
            continue
        if data != b'\xFF':
            frame += data
        else:
            frame += data
            print("帧:", frame.hex().upper())
            frame.clear()

except KeyboardInterrupt:
    ser.close()
    print("串口关闭")

