'''
Demo: 夹爪控制示例
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi

def main():
    # 导入会话和控制器
    session = get_default_session()
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