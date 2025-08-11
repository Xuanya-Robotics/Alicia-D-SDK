"""
Demo: 使用 moveJ_online 通过在线插值器执行关节空间移动

This demo shows how to call `moveJ_online` with degree inputs, customize the
online interpolator parameters, and check the arrival result.
"""

from alicia_d_sdk.controller import get_default_session, SynriaControlAPI


def main():
    """
    :return: None
    """
    # Create session and controller
    session = get_default_session()
    controller = SynriaControlAPI(session=session)

    try:
        # Move to home first for safety
        controller.moveHome()

        # Target joints in degree (len=6)
        target_joints_deg = [-30.0, 30.0, 30.0, 20.0, -20.0, 10.0]

        print(">>> 使用 moveJ_online 执行关节移动（单位：度）")
        arrived = controller.moveJ_online(
            joint_format='deg',                 # 'rad' or 'deg'
            target_joints=target_joints_deg,    # target joints
            command_rate_hz=200.0,              # control loop frequency
            max_joint_velocity_rad_s=1.5,       # joint velocity limit
            max_joint_accel_rad_s2=6.0,         # joint acceleration limit
            arrival_tolerance_rad=0.02,         # arrival tolerance (rad)
            settle_time=0.2,                    # stabilize duration (s)
            timeout=8.0,                        # overall timeout (s)
            stop_after=True                     # auto stop background thread
        )
        print(f"到位结果: {arrived}")

        # Optionally, execute another target using the same API
        print(">>> 执行第二次移动，演示重复调用 moveJ_online")
        target_joints_deg_2 = [0.0, -20.0, 25.0, -15.0, 10.0, 0.0]
        arrived2 = controller.moveJ_online(
            joint_format='deg',
            target_joints=target_joints_deg_2,
            command_rate_hz=200.0,
            max_joint_velocity_rad_s=1.8,
            max_joint_accel_rad_s2=7.0,
            arrival_tolerance_rad=0.02,
            settle_time=0.2,
            timeout=8.0,
            stop_after=True
        )
        print(f"到位结果(第二次): {arrived2}")

        # Return to home after demo
        controller.moveHome()

    except KeyboardInterrupt:
        print("示例终止")
    finally:
        # Always disconnect at the end
        session.joint_controller.disconnect()


if __name__ == "__main__":
    main()


