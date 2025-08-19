#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关节轨迹回放（无交互、无IK、自动剪掉开头/结尾静止，带自适应阈值）
- 读取 example_motions/<motion>/joint_traj.json（由示教脚本生成）
- 用“自适应噪声阈值 + 相对位移阈值”剪掉【开头/结尾】静止（如 IK 等待/回车等待）
- 30 Hz 重采样，使用 SDK 内置在线插值（Online Smoothing）平滑下发目标
- 运行即可立刻播放；参数极简：--port, --motion；可选：--repeat, --speed
"""
import os, json, time, argparse
from typing import List, Dict, Any, Tuple
from alicia_duo_sdk.controller import get_default_session, ControlApi

# ===== 可按需微调 =====
SEG_HZ = 30.0             # 设置目标更新频率（20~40 推荐）
ONLINE_CMD_HZ = 200.0     # 在线插值后台命令频率（建议 150~300）
BASE_DIFF_EPS = 0.010     # 基础“每帧关节L1差分阈值”（自适应阈值的下限，rad）
DIST_EPS = 0.06           # 相对起点（或终点）L1位移阈值（rad），超过视为“真正开始/结束”
HEAD_NOISE_SECS = 5.0     # 用前几秒估计噪声水平
WIN_SECS = 0.40           # 连续窗口长度（判定“在动”）
MAX_HEAD_CHECK = 1800.0   # 头部最多检查 30 分钟
MAX_TAIL_CHECK = 900.0    # 尾部最多检查 15 分钟

def load_traj(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.sort(key=lambda d: float(d["t"]))
    t0 = float(data[0]["t"])
    for d in data:
        d["t"] = float(d["t"]) - t0
    return data

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    mid = n // 2
    return (xs[mid] if n % 2 == 1 else 0.5*(xs[mid-1] + xs[mid]))

def _estimate_dt(traj: List[Dict[str, Any]]) -> float:
    if len(traj) < 2: return 1.0/SEG_HZ
    dts = [max(1e-6, float(traj[i+1]["t"]) - float(traj[i]["t"])) for i in range(len(traj)-1)]
    return _median(dts)

def _l1(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def _moving_mean(arr: List[float], win: int) -> List[float]:
    if win <= 1:
        return arr[:]
    out, s = [], sum(arr[:min(win, len(arr))])
    for i in range(len(arr)):
        if i >= win:
            s += arr[i] - arr[i-win]
            out.append(s / win)
        else:
            out.append(s / (i+1))
    return out

def _build_diffs(traj: List[Dict[str, Any]]) -> List[float]:
    diffs = [0.0]
    for i in range(1, len(traj)):
        diffs.append(_l1(traj[i]["q"], traj[i-1]["q"]))
    return diffs

def _adaptive_eps(traj: List[Dict[str, Any]], diffs: List[float], head_secs: float) -> float:
    # 取前 head_secs 的差分，估计“静止噪声”中位数；阈值= max(BASE_DIFF_EPS, 3*中位数)
    if not diffs: return BASE_DIFF_EPS
    cut_t = head_secs
    buf = []
    for i in range(min(len(traj), len(diffs))):
        if traj[i]["t"] > cut_t: break
        buf.append(diffs[i])
    noise_med = _median(buf) if buf else _median(diffs[:min(200, len(diffs))])
    return max(BASE_DIFF_EPS, 3.0*noise_med)

def trim_head_tail(traj: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float]:
    """自适应阈值 + 相对位移阈值，裁剪【开头/结尾静止】；返回(裁后轨迹, 剪掉头秒, 剪掉尾秒)"""
    if not traj or len(traj) < 2: return traj, 0.0, 0.0

    dt = _estimate_dt(traj)
    win = max(3, int(WIN_SECS / max(1e-6, dt)))

    diffs = _build_diffs(traj)
    eps = _adaptive_eps(traj, diffs, HEAD_NOISE_SECS)
    mv = _moving_mean(diffs, win)

    # —— 开头 —— #
    q0 = traj[0]["q"]
    head_limit_idx = next((i for i, s in enumerate(traj) if s["t"] > MAX_HEAD_CHECK), len(traj)-1)
    start_idx = 0
    # 先看相对位移：找到第一个到 q0 的 L1 距离超过 DIST_EPS 的点
    for i in range(len(traj)):
        if i > head_limit_idx: break
        if _l1(traj[i]["q"], q0) >= DIST_EPS:
            start_idx = i
            break
    # 如果相对位移没触发，再用“窗口均值>eps 连续 win 帧”
    if start_idx == 0:
        for i in range(len(mv) - win):
            if i > head_limit_idx: break
            if all(mv[i+k] > eps for k in range(win)):
                start_idx = i
                break

    # —— 结尾 —— #
    qN = traj[-1]["q"]
    tail_limit_t = traj[-1]["t"] - MAX_TAIL_CHECK
    end_idx = len(traj) - 1
    # 相对位移（相对于最后一帧）回看
    j = len(traj) - 1
    while j >= 0 and traj[j]["t"] >= max(0.0, tail_limit_t):
        if _l1(traj[j]["q"], qN) >= DIST_EPS:
            end_idx = j
            break
        j -= 1
    # 如果没触发，相对位移，再用“窗口均值>eps 连续 win 帧”倒向判定
    if end_idx == len(traj) - 1:
        j = len(mv) - 1
        while j - win >= 0 and traj[j]["t"] >= max(0.0, tail_limit_t):
            if all(mv[j-k] > eps for k in range(win)):
                end_idx = j
                break
            j -= 1

    trimmed = traj[start_idx:end_idx+1] if end_idx >= start_idx else traj[:]
    if not trimmed:
        trimmed = traj[:]

    # 归一化时间轴
    t0 = trimmed[0]["t"]
    for d in trimmed:
        d["t"] = float(d["t"]) - t0

    removed_head = traj[start_idx]["t"] if start_idx > 0 else 0.0
    removed_tail = (traj[-1]["t"] - traj[end_idx]["t"]) if end_idx < len(traj)-1 else 0.0
    return trimmed, removed_head, removed_tail

def resample(traj: List[Dict[str, Any]], hz: float) -> List[Dict[str, Any]]:
    if hz <= 0 or len(traj) < 2: return traj
    dt = 1.0 / hz
    T  = float(traj[-1]["t"])
    keys = [round(i*dt, 6) for i in range(int(T/dt))]
    if not keys or keys[-1] < T: keys.append(T)

    out, j = [], 0
    for tk in keys:
        while j+1 < len(traj) and float(traj[j+1]["t"]) < tk:
            j += 1
        if j+1 < len(traj):
            t0, t1 = float(traj[j]["t"]), float(traj[j+1]["t"])
            a = 0.0 if t1 <= t0 else (tk - t0) / (t1 - t0)
            q0, q1 = traj[j]["q"], traj[j+1]["q"]
            qk = [q0i + a*(q1i - q1i + (q1i - q0i)) for q0i, q1i in zip(q0, q1)]  # 展开写法防止某些解释器优化bug
            qk = [q0i + a*(q1i - q0i) for q0i, q1i in zip(q0, q1)]
            gk = None
            if ("grip" in traj[j]) and ("grip" in traj[j+1]):
                g0, g1 = traj[j].get("grip"), traj[j+1].get("grip")
                if (g0 is not None) and (g1 is not None):
                    gk = g0 + a*(g1 - g0)
            out.append({"t": tk, "q": qk, "grip": gk})
        else:
            out.append({"t": tk, "q": traj[-1]["q"], "grip": traj[-1].get("grip")})
    return out

 

def main(args):
    print("PLAY_CACHED_TRAJ v2.0  (adaptive-trim head/tail, online-smoothing targets)")
    
    traj_path = os.path.join("example_motions", args.motion, "joint_traj.json")
    if not os.path.exists(traj_path):
        raise SystemExit(f"未找到轨迹：{traj_path}（请先录制）")

    raw = load_traj(traj_path)
    print(f"[load] {traj_path}  samples={len(raw)}  total={raw[-1]['t']:.3f}s")

    trimmed, head_cut, tail_cut = trim_head_tail(raw)
    if head_cut > 0: print(f"[trim] 已裁掉前置静止 ≈ {head_cut:.2f}s")
    if tail_cut > 0: print(f"[trim] 已裁掉尾部静止 ≈ {tail_cut:.2f}s")
    traj = trimmed

    kf = resample(traj, SEG_HZ)
    est = traj[-1]['t'] / max(1e-6, args.speed)
    print(f"[resample] {len(traj)} -> {len(kf)} keyframes @ {SEG_HZ} Hz, play ~{est:.2f}s")

    # 创建会话和控制器
    session = get_default_session(baudrate=args.baudrate, port=args.port)
    ctl     = ControlApi(session=session)
    arm     = session.joint_controller

    try:
        ctl.torque_control("on")
    except Exception:
        pass
    if hasattr(arm, "enable_torque"):
        try:
            arm.enable_torque()
        except Exception:
            pass

    # 启动在线插值后台线程
    # !!!! 如果发现机械臂执行有卡顿现象，可以适当降低加速度值 !!!!
    ctl.startOnlineSmoothing(
        command_rate_hz=ONLINE_CMD_HZ,
        max_joint_velocity_rad_s=2.5,
        max_joint_accel_rad_s2=1,
        max_gripper_velocity_rad_s=1.5,
        max_gripper_accel_rad_s2=10.0,
    )
    print(f"[online] start smoothing @ {ONLINE_CMD_HZ} Hz ; keyframes @ {SEG_HZ} Hz")

    # 精准定时设置目标（倍速缩放；由在线插值负责平滑）
    try:
        for loop in range(max(1, int(args.repeat))):
            print(f"[Loop {loop+1}/{args.repeat}] start")
            start = time.time()
            i = 0
            while i < len(kf):
                target = float(kf[i]["t"]) / max(1e-6, args.speed)
                now = time.time() - start
                if now + 0.01 < target:
                    # 轻量等待，降低 CPU 占用
                    time.sleep(max(0.0, min(0.002, target - now)))
                    continue

                s = kf[i]
                # 设置新的关节与夹爪目标（rad）
                try:
                    ctl.setJointTargetOnline(s["q"])  # len=6, rad
                except Exception as e:
                    raise RuntimeError(f"设置在线关节目标失败：{e}")

                if s.get("grip") is not None:
                    try:
                        ctl.setGripperTargetOnline(float(s["grip"]))
                    except Exception:
                        # 某些固件不支持在线夹爪；忽略即可
                        pass

                i += 1

            print(f"[Loop {loop+1}] done in {time.time() - start:.2f}s")
    finally:
        # 结束在线插值
        try:
            ctl.stopOnlineSmoothing()
            print("[online] stop smoothing")
        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # !!! 请先使用00_demo_read_version.py检查版本号 !!!
    # !!! 如果你能够读到版本号，版本号为5.4.19以上，则使用默认波特率1000000 !!!
    # !!! 如果显示超时或者多次尝试后没有版本号输出，则使用默认波特率921600 !!!
    parser.add_argument('--baudrate', type=int, default=1000000, help="波特率")
    parser.add_argument("--port",   default="COM6")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--speed",  type=float, default=1.0, help="倍速播放（>1 更快）")
    args = parser.parse_args()
    main(args)
