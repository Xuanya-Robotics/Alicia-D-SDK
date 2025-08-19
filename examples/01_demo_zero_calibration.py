'''
Demo: 调零程序
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi

def main(args):
    # 创建会话和控制器
    session = get_default_session(baudrate=args.baudrate, port=args.port)
    controller = ControlApi(session=session)

    try:
        controller.zero_calibration()

    except KeyboardInterrupt:
        print("调零终止")

    finally:
        session.joint_controller.disconnect()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # !!! 请先使用00_demo_read_version.py检查版本号 !!!
    # !!! 如果你能够读到版本号，版本号为5.4.19以上，则使用默认波特率1000000 !!!
    # !!! 如果显示超时或者多次尝试后没有版本号输出，则使用默认波特率921600 !!!
    parser.add_argument('--baudrate', type=int, default=1000000, help="波特率")
    parser.add_argument('--port', type=str, default="", help="串口端口")
    args = parser.parse_args()
    main(args)