# ik_test_demo.py
import time
from alicia_duo_sdk.robot_kinematics import IKController, RobotArm
from alicia_duo_sdk.controller import ArmController
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import matplotlib.pyplot as plt

RAD2DEG = 180.0 / 3.1415926
DEG2RAD = 3.1415926 / 180.0

def ease_in_out_cubic(x: float) -> float:
    return 4 * x**3 if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2

def control_move(controller: ArmController, current_angles, target_angles, steps=120, delay=0.03):
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
    # 保证 q2 与 q1 方向一致，避免跳变
    if np.dot(start_quat, end_quat) < 0.0:
        end_quat = [-x for x in end_quat]

    key_rots = R.from_quat([start_quat, end_quat])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    for i in range(steps):
        t = i / (steps - 1)
        pos = (1 - t) * np.array(start_pos) + t * np.array(end_pos)
        quat = slerp([t]).as_quat()[0]
        poses.append((pos, quat))

    return poses

def visualize_joint_trajectory(joint_traj, title="IK 轨迹关节角度"):
    joint_traj = np.array(joint_traj)
    time_steps = np.arange(len(joint_traj))
    fig, axs = plt.subplots(6, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(title)
    for j in range(6):
        axs[j].plot(time_steps, joint_traj[:, j])
        axs[j].set_ylabel(f"Joint {j+1} (rad)")
        axs[j].grid(True)
    axs[-1].set_xlabel("Step")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, start_angles, end_pose, end_quat, steps=100):
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
            joint4 = result["joint4"]
            joint6 = result["joint6"]
            offset = (joint4 + joint6) / 2
            result["joint4"] = offset
            result["joint6"] = offset

        if result:
            joint_angles = list(result.values())
            all_joint_angles.append(joint_angles)
            start_angles = joint_angles
        else:
            print("❌ 某个点 IK 求解失败，轨迹中断")
            return
    visualize_joint_trajectory(all_joint_angles)
    user_input = input('是否执行？按下回车继续, 按下q取消...')
    if user_input.lower() == 'q':
        return  
    print("✅ 开始执行轨迹...")
    for ja in all_joint_angles:
        controller.set_joint_angles(ja, wait_for_completion=False)
        time.sleep(0.02)

def dragging_demo(controller, ik_controller, robot_model):
    print("→ 拖拽示教模式启动：请将机械臂移动到初始位置，然后按回车")
    input("  起始位姿记录...")
    start_state = controller.read_joint_state()
    start_angles = start_state.angles
    start_pose, start_quat = robot_model.forward_kinematics(
        dict(zip(robot_model.kinematic_chain[:-1], start_angles))
    )
    print(f"开始角度{start_angles}, 开始POSE{start_pose}{start_quat}")
    input("按下回车开始拖动机械臂")
    controller.disable_torque()
    input("→ 拖拽至目标位姿后按回车")
    controller.enable_torque()
    input("按下回车")
    end_state = controller.read_joint_state()
    end_angles = end_state.angles
    end_pose, end_quat = robot_model.forward_kinematics(
        dict(zip(robot_model.kinematic_chain[:-1], end_angles))
    )

    print("→ 是否开启往复运动？(y/n):", end=" ")
    if input().strip().lower() == 'y':
        print("开始往返运动，按 Ctrl+C 退出")
        input("按下回车开始")
        try:
            while True:
                plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, end_angles, start_pose, start_quat)
                plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, start_angles, end_pose, end_quat)
                time.sleep(1)
        except KeyboardInterrupt:
            print("用户中断往复运动")
    else:
        print("→ 执行一次运动")
        input("按下回车开始")
        plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, end_angles, start_pose, start_quat)
        plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, start_angles, end_pose, end_quat)
        plan_and_execute_pose_trajectory(controller, ik_controller, robot_model, end_angles, start_pose, start_quat)

def dragging_pose_monitor(controller, robot_model):
    print("→ 拖拽读取模式：关闭扭矩，开始读取末端位姿。按 Ctrl+C 退出")
    input("按回车开始")
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
    print("=== IK/FK 拖拽测试工具 ===")

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
            print("4. 扭矩开关")
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
            elif choice == '4':
                action = input("输入 'on' 开启扭矩，'off' 关闭扭矩: ").strip()
                if action == 'on':
                    controller.enable_torque()
                elif action == 'off':
                    controller.disable_torque()
                else:
                    print("非法输入")
            elif choice == 'q':
                break
            else:
                print("无效输入，请重新选择")

    finally:
        controller.disconnect()
        print("🔌 已断开连接")

if __name__ == "__main__":
    main()
