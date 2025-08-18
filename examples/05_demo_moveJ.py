"""
Demo: 使用 moveJ 控制机械臂移动到目标关节位置，已包含关节角线性插值
"""

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
        # 设置目标角度（以 degree 为例)
        target_joints_deg = [-30, 30.0, 30.0, 20.0, -20.0, 10.0]  # 夹角格式：deg

        # 移动到初始位置
        controller.moveHome()

        # 调用 moveJ
        controller.moveJ(
            joint_format='deg',              # 角度单位，可选 'rad' 或 'deg'
            target_joints=target_joints_deg,
            speed_factor=1,                # 控制速度（1.0 = 默认速度）
            visualize=False                  # 可视化轨迹
        )

        # 移动到初始位置
        controller.moveHome()
    
    except KeyboardInterrupt:
        print("示例终止")
    
    finally:
        session.joint_controller.disconnect()

if __name__ == "__main__":
    main()
