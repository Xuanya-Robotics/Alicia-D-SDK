"""
Demo: Teaching 模式下记录末端姿态轨迹并回放

- 用户通过拖动机械臂并按下回车手动记录多个 Waypoint
- 系统记录每个姿态点（[x, y, z, qx, qy, qz, qw]）
- 再通过 moveCartesian 插值执行这些轨迹
- 支持可视化与姿态显示
"""

import time
import numpy as np
from alicia_duo_sdk.controller import get_default_session, ControlApi

def teaching_demo_cartesian():
    # === 初始化机器人会话 ===
    session = get_default_session()
    controller = ControlApi(session=session)

    print(">>> 关闭扭矩，请手动拖动机械臂到若干目标位置")
    input("按 Enter 开始记录教学 Waypoint，输入 q 可提前结束")
    controller.torque_control(command='off')

    waypoints = []

    while True:
        cmd = input("拖动完成后按 Enter 记录当前位置，输入 q 结束录制: ").strip()
        if cmd.lower() == 'q':
            break

        pose = controller.get_pose()        
        waypoints.append(pose)
        print(f"[记录成功] Waypoint {len(waypoints)}: {np.round(pose, 4).tolist()}")

    print(">>> 重新开启扭矩")
    controller.torque_control('on')

    if not waypoints:
        print("[退出] 未记录任何姿态点")
        return

    print(f"\n>>> 共记录 {len(waypoints)} 个姿态点，准备开始轨迹回放...")
    time.sleep(1.0)

    controller.moveCartesian(
        waypoints=waypoints,
        reverse=True,           # 反向执行轨迹 （从最后一个记录点开始反向执行）
        planner_name='cartesian',     # 可改为 'cartesian' 或 'lqt'
        move_time=3.0,          # 预计执行时长
        visualize=True,         # 轨迹图可视化
        show_ori=True           # 轨迹图是否显示姿态
    )

if __name__ == "__main__":
    teaching_demo_cartesian()
