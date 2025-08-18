'''
Demo: 扭矩开关示例
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi

def main():
    # 创建会话和控制器
    # !!! 请先使用00_demo_read_version.py检查版本号 !!!
    # !!! 如果你能够读到版本号，版本号为5.4.19以上，则使用默认波特率1000000 !!!
    # !!! 如果显示超时或者多次尝试后没有版本号输出，则使用默认波特率921600 !!!    
    session = get_default_session(baudrate=1000000)
    # session = get_default_session(baudrate=921600)
    controller = ControlApi(session=session)

    try:
        # 关闭机械臂扭矩 （注意安全）
        controller.torque_control(command='off')

        # 打开机械臂扭矩
        input("按下回车打开扭矩")
        controller.torque_control(command='on')

    except KeyboardInterrupt:
        print("调零终止")

    finally:
        session.joint_controller.disconnect()

if __name__ == '__main__':
    main()