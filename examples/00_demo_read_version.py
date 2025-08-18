'''
Demo: 打印机械臂信息（关节角度，夹爪角度，以及末端位姿）
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi
import time


def main():
    # 创建会话和控制器
    # !!! 如果你能够读到版本号，版本号为5.4.19以上，则使用默认波特率1000000 !!!
    # !!! 如果显示超时或者多次尝试后没有版本号输出，则使用默认波特率921600 !!!
    session = get_default_session(baudrate=1000000)
    # session = get_default_session(baudrate=921600)
    controller = ControlApi(session=session)
    try:
        # 单次打印机械臂状态， 关节输出为弧度值
        controller.joint_controller.serial_comm.send_data([0xAA, 0x12, 0x01, 0x00, 0x00, 0xFF])
    except KeyboardInterrupt:
        print("读取中断")

    finally:
        session.joint_controller.disconnect()


if __name__ == '__main__':
    main()
