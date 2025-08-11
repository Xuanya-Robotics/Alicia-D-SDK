"""
Demo: 使用 moveJ 控制机械臂移动到目标关节位置，已包含关节角线性插值
"""

from alicia_d_sdk.controller import get_default_session, SynriaControlAPI

def main():
    # 创建会话和控制器
    session = get_default_session()
    controller = SynriaControlAPI(session=session)

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
