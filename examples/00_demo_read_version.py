'''
Demo: 打印机械臂信息（关节角度，夹爪角度，以及末端位姿）
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi
import time


def main(args):
    # 创建会话和控制器
    session = get_default_session(baudrate=args.baudrate, port=args.port)
    controller = ControlApi(session=session)
    try:
        # 单次打印机械臂状态， 关节输出为弧度值
        controller.joint_controller.serial_comm.send_data([0xAA, 0x12, 0x01, 0x00, 0x00, 0xFF])
    except KeyboardInterrupt:
        print("读取中断")

    finally:
        session.joint_controller.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # !!! 如果你能够读到版本号，版本号为5.4.19以上，则使用默认波特率1000000 !!!
    # !!! 如果显示超时或者多次尝试后没有版本号输出，则使用默认波特率921600 !!!
    parser.add_argument('--baudrate', type=int, default=1000000, help="波特率")
    parser.add_argument('--port', type=str, default="", help="串口端口")
    args = parser.parse_args()
    main(args)
