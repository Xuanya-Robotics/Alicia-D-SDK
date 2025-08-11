'''
Demo: 打印机械臂信息（关节角度，夹爪角度，以及末端位姿）
'''

from alicia_d_sdk.controller import get_default_session, SynriaControlAPI

def main():
    # 创建会话和控制器
    session = get_default_session()
    controller = SynriaControlAPI(session=session)

    try:
        # 单次打印机械臂状态， 关节输出为弧度值
        controller.print_state(continous_printing=False,
                            output_format= 'rad')

        # 持续打印机械臂状态， 关节输出为角度值
        controller.print_state(continous_printing=True,
                            output_format= 'deg')
    except KeyboardInterrupt:
        print("读取中断")

    finally:
        session.joint_controller.disconnect()

if __name__ == '__main__':
    main()