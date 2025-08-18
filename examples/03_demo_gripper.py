'''
Demo: 夹爪控制示例
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
        # 打开夹爪
        controller.gripper_control(command='open')

        # 关闭夹爪
        controller.gripper_control(command='close')

        # 夹爪打开到50度（0-100度）
        controller.gripper_control(angle_deg=50)

    except KeyboardInterrupt:
        print("程序中断")

    finally:
        session.joint_controller.disconnect()

if __name__ == '__main__':
    main()