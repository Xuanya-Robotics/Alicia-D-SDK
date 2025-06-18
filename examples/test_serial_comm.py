import serial

port = "/dev/cu.usbserial-1430"  # 你的串口设备名
baudrate = 1000000  # 或你设备要求的 921600

print("开始读取串口数据（查找帧头 AA）...")

ser = serial.Serial(port, baudrate, timeout=1)
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

