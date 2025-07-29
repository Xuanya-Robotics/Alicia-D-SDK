# ik_test_demo.py
import time
import numpy as np
from ik_module import IKController, RobotArm
from alicia_duo_sdk.controller import ArmController
from scipy.spatial.transform import Rotation as R

RAD2DEG = 180.0 / 3.1415926
DEG2RAD = 3.1415926 / 180.0

def ease_in_out_cubic(x: float) -> float:
    return 4 * x**3 if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2

def control_move(controller, current_angles, target_angles, steps=120, delay=0.03):
    if len(current_angles) != len(target_angles):
        raise ValueError("当前角度与目标角度数量不匹配")

    for step in range(1, steps + 1):
        ratio = ease_in_out_cubic(step / steps)
        interp = [a + (b - a) * ratio for a, b in zip(current_angles, target_angles)]
        ok = controller.set_joint_angles(interp, wait_for_completion=False)
        if not ok:
            return False
        time.sleep(delay)
    return True

def interpolate_trajectory(start_pos, start_quat, end_pos, end_quat, steps):
    poses = []
    for i in range(steps):
        t = i / (steps - 1)
        pos = (1 - t) * np.array(start_pos) + t * np.array(end_pos)
        q1 = R.from_quat(start_quat)
        q2 = R.from_quat(end_quat)
        q_interp = R.slerp(0, 1, [q1, q2])(t).as_quat()
        poses.append((pos, q_interp))
    return poses

def plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, start_angles, end_pose, end_quat, steps=50):
    start_pose, start_quat = robot_model.forward_kinematics(
        dict(zip(robot_model.kinematic_chain[:-1], start_angles))
    )
    trajectory = interpolate_trajectory(start_pose, start_quat, end_pose, end_quat, steps)
    all_joint_angles = []
    for pos, quat in trajectory:
        ik_controller.update_joint_angles(
            dict(zip(robot_model.kinematic_chain[:-1], start_angles))
        )
        ik_controller.set_target(pos, quat)
        result = ik_controller.solve_ik()
        if result:
            joint_angles = list(result.values())
            all_joint_angles.append(joint_angles)
            start_angles = joint_angles  # 为下一次迭代提供初值
        else:
            print("❌ 某个点 IK 求解失败，轨迹中断")
            return
    print("✅ 开始执行轨迹...")
    for ja in all_joint_angles:
        controller.set_joint_angles(ja, wait_for_completion=False)
        time.sleep(0.03)

def dragging_demo(controller, ik_controller, robot_model):
    print("→ 拖拽示教模式启动：请将机械臂移动到初始位置，然后按回车")
    input("  起始位姿记录...")
    start_state = controller.read_joint_state()
    start_angles = start_state.angles
    start_pose, _ = robot_model.forward_kinematics(
        dict(zip(robot_model.kinematic_chain[:-1], start_angles))
    )

    controller.disable_torque()
    input("→ 拖拽至目标位姿后按回车")
    controller.enable_torque()
    end_state = controller.read_joint_state()
    end_angles = end_state.angles
    end_pose, end_quat = robot_model.forward_kinematics(
        dict(zip(robot_model.kinematic_chain[:-1], end_angles))
    )

    print("→ 是否开启往复运动？(y/n):", end=" ")
    if input().strip().lower() == 'y':
        try:
            while True:
                plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, start_angles, end_pose, end_quat)
                plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, end_angles, start_pose, _)
        except KeyboardInterrupt:
            print("用户中断往复运动")
    else:
        plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, start_angles, end_pose, end_quat)

def dragging_pose_monitor(controller, robot_model):
    print("→ 拖拽读取模式：关闭扭矩，开始读取末端位姿。按 Ctrl+C 退出")
    controller.disable_torque()
    try:
        while True:
            state = controller.read_joint_state()
            pos, quat = robot_model.forward_kinematics(
                dict(zip(robot_model.kinematic_chain[:-1], state.angles))
            )
            print(f"位置: {[round(p, 4) for p in pos]}, 四元数: {[round(q, 4) for q in quat]}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("已停止读取")
    finally:
        controller.enable_torque()
        print("扭矩已恢复")

def main():
    print("=== IK/FK 拖拽与轨迹测试工具 ===")

    controller = ArmController(debug_mode=False)
    if not controller.connect():
        print("❌ 无法连接机械臂")
        return
    print("✅ 连接成功")

    robot_model = RobotArm()
    ik_controller = IKController(robot_model)

    try:
        while True:
            print("\n功能菜单:")
            print("1. 拖拽示教模式")
            print("2. 拖拽读取POSE模式")
            print("3. 回到零点")
            print("q. 退出程序")
            choice = input("你的选择: ").strip().lower()

            if choice == '1':
                dragging_demo(controller, ik_controller, robot_model)
            elif choice == '2':
                dragging_pose_monitor(controller, robot_model)
            elif choice == '3':
                print("→ 机械臂回到零点")
                zero = [0.0] * 6
                current = controller.read_joint_state().angles
                control_move(controller, current, zero)
            elif choice == 'q':
                break
            else:
                print("无效输入，请重新选择")

    finally:
        controller.disconnect()
        print("🔌 已断开连接")

if __name__ == "__main__":
    main()
