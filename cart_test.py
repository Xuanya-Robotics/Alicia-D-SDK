import time
from alicia_duo_sdk.controller.session_factory import get_default_session
from alicia_duo_sdk.controller.control_api import ControlApi
import numpy as np

def test_cartesian_teaching():
    session = get_default_session()
    controller = ControlApi(session=session)

    start_angles = session.joint_controller.get_joint_angles()
    start_pose = np.concatenate(session.robot_model.forward_kinematics(start_angles)).tolist()

    input("关闭扭矩，请手动拖动机械臂到目标位置，按 Enter 记录 waypoint，输入 q 结束")
    session.joint_controller.disable_torque()
    waypoints = []

    while True:
        cmd = input("拖动完毕，按 Enter 记录，输入 q 结束: ").strip()
        if cmd.lower() == 'q':
            break

        pose = np.concatenate(session.robot_model.forward_kinematics(
            session.joint_controller.get_joint_angles())).tolist()
        print(f"[记录] Waypoint: {pose}")
        waypoints.append(pose)

    print("重新开启扭矩")
    session.joint_controller.enable_torque()
    time.sleep(1.0)

    if not waypoints:
        print("未记录任何 waypoint，退出")
        return

    print("开始执行插值轨迹...")
    controller.moveJ(target_angles=start_angles)
    controller.moveCartesian(
                  waypoints=waypoints,
                  start_joint_angles=start_angles,
                  move_time=3.0,
                  visualize=False)

if __name__ == "__main__":
    test_cartesian_teaching()
