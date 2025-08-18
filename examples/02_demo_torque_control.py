'''
Demo: 扭矩开关示例
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi

def main():
    # 导入会话和控制器
    session = get_default_session()
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