#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Teaching 模式下记录末端姿态轨迹并回放（缓存关节轨迹，但不自动重播）
- 手动示教记录多个 waypoint（末端位姿 + 夹爪）
- 执行一次 moveCartesian（这一步仍会提示“按回车执行”）
- 后台定频采样关节角并保存为 example_motions/<motion>/joint_traj.json
- 不做自动快速回放；后续你用 08_demo_recorded_motion_replay.py 手动回放
"""

import os
import json
import time
import argparse
import threading
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from alicia_duo_sdk.controller import get_default_session, ControlApi


class TrajectoryRecorder:
    """后台定频采样关节角，生成可重放的时间序列。"""
    def __init__(self, controller: ControlApi, rate_hz: float = 100.0):
        self.controller = controller
        self.rate_hz = float(rate_hz)
        self._stop = threading.Event()
        self._thread = None
        self._buf: List[Dict[str, Any]] = []
        self._t0 = None

    def _get_joints(self) -> List[float]:
        if hasattr(self.controller, 'get_joints'):
            return list(self.controller.get_joints())
        if hasattr(self.controller, 'get_joint_positions'):
            return list(self.controller.get_joint_positions())
        raise RuntimeError('未找到读取关节角的方法（期望 get_joints 或 get_joint_positions）')

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        dt = 1.0 / max(1.0, self.rate_hz)
        while not self._stop.is_set():
            try:
                t = time.time() - self._t0
                q = self._get_joints()
                try:
                    grip = float(self.controller.get_gripper())
                except Exception:
                    grip = None
                self._buf.append({"t": t, "q": q, "grip": grip})
            except Exception:
                pass
            time.sleep(dt)

    def stop(self) -> List[Dict[str, Any]]:
        if self._thread is None:
            return []
        self._stop.set()
        self._thread.join()
        self._thread = None
        return self._buf


def teaching_demo_cartesian(args):
    # === 初始化会话 ===
    # 创建会话和控制器
    # !!! 请先使用00_demo_read_version.py检查版本号 !!!
    # !!! 如果你能够读到版本号，版本号为5.4.19以上，则使用默认波特率1000000 !!!
    # !!! 如果显示超时或者多次尝试后没有版本号输出，则使用默认波特率921600 !!!
    session = get_default_session(baudrate=1000000, port=args.port)
    # session = get_default_session(baudrate=921600, port=args.port)
    controller = ControlApi(session=session)

    # === 交互式教学 ===
    print(
        ">>> 关闭扭矩，请手动拖动机械臂到若干目标位置\n"
        "按 Enter 开始记录教学 Waypoint，输入 q 可提前结束"
    )
    input()
    controller.torque_control(command='off')

    waypoints: List[List[float]] = []
    while True:
        cmd = input("拖动完成后按 Enter 记录当前位置，输入 q 结束录制: ").strip()
        if cmd.lower() == 'q':
            break
        pose = controller.get_pose()            # [x,y,z,qx,qy,qz,qw]
        grip = controller.get_gripper()
        wp = list(pose) + [float(grip)]
        waypoints.append(wp)
        print(f"[记录成功] Waypoint {len(waypoints)}: pose={np.round(pose, 4).tolist()}, gripper={grip:.3f} rad")

    controller.torque_control('on')

    if not waypoints:
        print("[退出] 未记录任何姿态点")
        return

    # === 路径与防覆盖 ===
    root = os.path.join('example_motions', args.motion)
    wp_path = os.path.join(root, 'waypoints.json')
    traj_path = os.path.join(root, 'joint_traj.json')
    meta_path = os.path.join(root, 'meta.json')

    if os.path.exists(root) and not args.overwrite:
        raise SystemExit(f"[中止] 动作 '{args.motion}' 已存在。若确认覆盖，请加 --overwrite。路径: {root}")
    os.makedirs(root, exist_ok=True)

    # === 保存 waypoints 与元信息 ===
    with open(wp_path, 'w', encoding='utf-8') as f:
        json.dump(waypoints, f, ensure_ascii=False, indent=2)
    meta = {
        "motion": args.motion,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "planner": args.planner,
        "move_time": float(args.time),
        "sample_hz": float(args.sample_hz),
        "count": len(waypoints),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[保存] {wp_path}\n[保存] {meta_path}")

    # === 后台采样 + 执行（只执行一次，不自动重播） ===
    print(f"\n>>> 记录到 {len(waypoints)} 个 waypoint，开始执行并缓存关节轨迹...")
    time.sleep(0.5)

    rec = TrajectoryRecorder(controller, rate_hz=args.sample_hz)
    rec.start()

    controller.moveCartesian(
        waypoints=waypoints,
        reverse=False,
        planner_name=args.planner,    # 'cartesian' 或 'lqt'
        move_time=float(args.time),
        visualize=bool(args.visualize),
        show_ori=bool(args.show_ori),
    )

    samples = rec.stop()
    with open(traj_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[保存] {traj_path}")
    print(f"[完成] 动作 '{args.motion}' 可直接重放：08_demo_recorded_motion_replay.py --motion {args.motion}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM6')
    ap.add_argument('--time', type=float, default=10.0)
    ap.add_argument('--planner', choices=['cartesian','lqt'], default='cartesian')
    ap.add_argument('--motion', required=True, help='本次录制的动作名（将保存到 example_motions/<motion>/）')
    ap.add_argument('--overwrite', motion='store_true', help='若已存在则覆盖')
    ap.add_argument('--sample-hz', type=float, default=100.0, help='缓存关节轨迹的采样频率（Hz）')
    ap.add_argument('--visualize', dest='visualize', motion='store_true')
    ap.add_argument('--no-visualize', dest='visualize', motion='store_false')
    ap.add_argument('--show-ori', dest='show_ori', motion='store_true')
    ap.add_argument('--no-show-ori', dest='show_ori', motion='store_false')
    ap.set_defaults(visualize=False, show_ori=True)
    args = ap.parse_args()

    teaching_demo_cartesian(args)
