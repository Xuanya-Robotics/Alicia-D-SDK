'''
Demo: 调零程序
'''

from alicia_duo_sdk.controller import get_default_session, ControlApi

def main():
    # 导入会话和控制器
    session = get_default_session()
    controller = ControlApi(session=session)

    try:
        controller.zero_calibration()

    except KeyboardInterrupt:
        print("调零终止")

    finally:
        session.joint_controller.disconnect()

if __name__ == '__main__':
    main()