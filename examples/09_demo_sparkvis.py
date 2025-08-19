#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: 与 SparkVis UI 的双向同步与数据写入（适配真实机器人）

功能：
- 启动 WebSocket 服务器，支持 UI ↔ 机器人 双向同步
- UI → 机器人：接收 joint_update 并直接下发到真实机器人
- 机器人 → UI：周期广播当前机器人状态到 UI（可开关）
- 数据写入：将 UI 下发的关节数据记录为 CSV（可选）

"""

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    import websockets  # pip install websockets==13.1
except Exception as e:
    print("请先安装 websockets: pip install websockets==13.1")
    raise

from alicia_duo_sdk.controller import get_default_session, ControlApi


class SparkVisBridge:
    def __init__(
        self,
        controller: ControlApi,
        host: str = "localhost",
        port: int = 8765,
        output_file: Optional[str] = None,
        enable_robot_sync: bool = True,
        robot_sync_rate_hz: float = 50.0,
        log_source: str = "ui"  # ui | robot | both
    ):
        self.controller = controller
        self.host = host
        self.port = port
        self.enable_robot_sync = enable_robot_sync
        self.robot_sync_interval = 1.0 / max(1e-3, robot_sync_rate_hz)
        self.log_source = log_source.lower()

        # WebSocket clients
        self.websocket_connections = set()
        self._pending_robot_state: Optional[Dict[str, float]] = None

        # CSV logging
        self.output_file = output_file
        self.file_handle = None
        if self.output_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
                self.file_handle = open(self.output_file, 'w')
                self.file_handle.write('timestamp,joint1,joint2,joint3,joint4,joint5,joint6,gripper\n')
                self.file_handle.flush()
                print(f"[Log] 数据记录启用: {self.output_file}")
            except Exception as e:
                print(f"[Log] 打开CSV失败: {e}")
                self.file_handle = None

        # graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 启动在线插值（ROS风格梯形速度剖面），适配稀疏UI指令
        try:
            self.controller.startOnlineSmoothing(
                command_rate_hz=200.0,
                max_joint_velocity_rad_s=2.5,
                max_joint_accel_rad_s2=8.0,
                max_gripper_velocity_rad_s=1.5,
                max_gripper_accel_rad_s2=10.0,
            )
            print("[Online] 已启动后台在线插值 (200Hz)")
        except Exception as e:
            print(f"[Online] 启动在线插值失败: {e}")

    # ---------- Utilities ----------
    def _signal_handler(self, signum, frame):
        print(f"[Exit] 收到信号 {signum}，正在优雅关闭...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if self.file_handle:
            try:
                self.file_handle.flush()
                self.file_handle.close()
                print("[Log] CSV 已保存")
            except Exception as e:
                print(f"[Log] 关闭CSV失败: {e}")
            finally:
                self.file_handle = None
        # 关闭在线插值
        try:
            self.controller.stopOnlineSmoothing()
            print("[Online] 在线插值已停止")
        except Exception:
            pass

    @staticmethod
    def _now_str(ts: Optional[float] = None) -> str:
        return datetime.fromtimestamp(ts or time.time()).strftime('%H:%M:%S.%f')[:-3]

    # ---------- Robot state helpers ----------
    def read_robot_state(self) -> Optional[Dict[str, float]]:
        try:
            joints = self.controller.get_joints()  # 6 rad
            gripper_rad = self.controller.get_gripper()  # rad
            if joints is None or gripper_rad is None:
                return None
            # gripper 广播给UI时使用百分比 [0..1]
            gripper_pct = max(0.0, min(1.0, gripper_rad / (100.0 * self.controller.joint_controller.DEG_TO_RAD)))
            return {
                "joint1": float(joints[0]),
                "joint2": float(joints[1]),
                "joint3": float(joints[2]),
                "joint4": float(joints[3]),
                "joint5": float(joints[4]),
                "joint6": float(joints[5]),
                "gripper": float(gripper_pct),
            }
        except Exception as e:
            print(f"[Robot] 读取状态失败: {e}")
            return None

    def apply_ui_joint_update(self, joint_values: Dict[str, float]):
        # joint1..6 为弧度, gripper 为 [0..1] 百分比
        try:
            joints = [
                float(joint_values.get('joint1', 0.0)),
                float(joint_values.get('joint2', 0.0)),
                float(joint_values.get('joint3', 0.0)),
                float(joint_values.get('joint4', 0.0)),
                float(joint_values.get('joint5', 0.0)),
                float(joint_values.get('joint6', 0.0)),
            ]
            # 走在线插值目标设定（后台200Hz平滑执行）
            self.controller.setJointTargetOnline(joints)

            if 'gripper' in joint_values:
                pct = max(0.0, min(1.0, float(joint_values['gripper'])))
                gripper_deg = pct * 100.0
                gripper_rad = gripper_deg * self.controller.joint_controller.DEG_TO_RAD
                self.controller.setGripperTargetOnline(gripper_rad)

            # logging (UI source)
            if self.file_handle and self.log_source in ("ui", "both"):
                row = f"{self._now_str()},{','.join(map(str, joints))},{joint_values.get('gripper', 0.0)}\n"
                self.file_handle.write(row)
                self.file_handle.flush()
        except Exception as e:
            print(f"[Robot] 写入失败: {e}")

    # ---------- WebSocket ----------
    async def broadcast_robot_state(self, joint_data: Dict[str, float]):
        if not self.websocket_connections:
            return
        message = {
            'type': 'robot_state_update',
            'joint_values': joint_data,
            'timestamp': time.time()
        }
        disconnected = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send(json.dumps(message))
            except Exception:
                disconnected.add(ws)
        self.websocket_connections -= disconnected

    async def websocket_handler(self, websocket, path):
        print(f"🔗 WebSocket连接: ws://{self.host}:{self.port}")
        self.websocket_connections.add(websocket)

        # 初始同步：推送机器人当前状态
        if self.enable_robot_sync:
            robot_state = self.read_robot_state()
            if robot_state:
                await websocket.send(json.dumps({
                    'type': 'robot_state_update',
                    'joint_values': robot_state,
                    'timestamp': time.time()
                }))
                print("📤 已发送当前机器人状态给新连接的客户端")

        # 周期任务：机器人 → UI 广播
        async def periodic_robot_sender():
            while websocket in self.websocket_connections:
                if self.enable_robot_sync:
                    state = self.read_robot_state()
                    if state:
                        # 日志（robot source）
                        if self.file_handle and self.log_source in ("robot", "both"):
                            row = f"{self._now_str()},{state['joint1']},{state['joint2']},{state['joint3']},{state['joint4']},{state['joint5']},{state['joint6']},{state['gripper']}\n"
                            self.file_handle.write(row)
                            self.file_handle.flush()
                        try:
                            await websocket.send(json.dumps({
                                'type': 'robot_state_update',
                                'joint_values': state,
                                'timestamp': time.time()
                            }))
                        except Exception:
                            break
                await asyncio.sleep(self.robot_sync_interval)

        sender_task = asyncio.create_task(periodic_robot_sender())

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')

                    if msg_type == 'joint_update':
                        joint_values = data.get('joint_values', {})
                        ts = data.get('timestamp', time.time())
                        print(f"[UI→Robot] {self._now_str(ts)} 关节更新: {joint_values}")
                        self.apply_ui_joint_update(joint_values)

                        # 回执
                        await websocket.send(json.dumps({
                            'type': 'joint_update_ack',
                            'timestamp': time.time(),
                            'success': True
                        }))

                    elif msg_type == 'request_robot_state':
                        state = self.read_robot_state()
                        if state:
                            await websocket.send(json.dumps({
                                'type': 'robot_state_update',
                                'joint_values': state,
                                'timestamp': time.time()
                            }))
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析错误: {e}")
                except Exception as e:
                    print(f"❌ 处理消息错误: {e}")
        except Exception:
            pass
        finally:
            if 'sender_task' in locals():
                sender_task.cancel()
            self.websocket_connections.discard(websocket)
            print("🔌 WebSocket断开")

    def start_server(self):
        print(f"🚀 启动 WebSocket 服务器: ws://{self.host}:{self.port}")
        print("⏹️  按 Ctrl+C 停止")

        async def server():
            async with websockets.serve(self.websocket_handler, self.host, self.port):
                await asyncio.Future()

        try:
            asyncio.run(server())
        except KeyboardInterrupt:
            print("🛑 服务器已停止")
        finally:
            self.cleanup()


def parse_args(argv: List[str]):
    import argparse
    parser = argparse.ArgumentParser(description='SparkVis ↔ 真实机器人 同步与数据记录 Demo')
    parser.add_argument('--port', type=str, default='', help='串口设备，如 /dev/ttyUSB0（留空自动搜索）')
    parser.add_argument('--baudrate', type=int, default=1000000, help='串口波特率，默认1000000，请使用00_demo_read_version.py检查版本号，如果显示超时或者多次尝试后均没有版本号输出，则使用921600')
    parser.add_argument('--host', type=str, default='localhost', help='WebSocket主机')
    parser.add_argument('--websocket-port', type=int, default=8765, help='WebSocket端口')
    parser.add_argument('--output-file', type=str, default='', help='CSV输出路径（留空不记录）')
    parser.add_argument('--enable-robot-sync', action='store_true', help='启用 机器人→UI 状态同步')
    parser.add_argument('--robot-sync-rate', type=float, default=50.0, help='机器人状态广播频率 Hz')
    parser.add_argument('--log-source', type=str, default='ui', choices=['ui','robot','both'], help='记录UI指令/机器人状态/二者')
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    # 创建会话与控制器
    session = get_default_session(port=args.port, baudrate=args.baudrate)
    controller = ControlApi(session=session)

    try:
        # 尝试开启力矩（安全起见）
        controller.torque_control('on')

        bridge = SparkVisBridge(
            controller=controller,
            host=args.host,
            port=args.websocket_port,
            output_file=args.output_file or None,
            enable_robot_sync=args.enable_robot_sync,
            robot_sync_rate_hz=args.robot_sync_rate,
            log_source=args.log_source,
        )
        bridge.start_server()

    finally:
        try:
            session.joint_controller.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    main()
