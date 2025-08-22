#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: 拖动示教 - 使用重构后的Controller层功能

功能特点：
- 实时拖动示教，自动记录轨迹点
- 支持手动和自动两种记录模式
- 自动保存轨迹数据到文件
- 内置轨迹回放功能
- 使用重构后的Controller层，代码更清晰

使用方式：
# 自动模式（推荐）：连续记录拖动轨迹
python 10_demo_drag_teaching.py --port COM6 --mode auto --sample-hz 50 --save-motion drag_demo

# 手动模式：只记录用户指定的关键位置点
python 10_demo_drag_teaching.py --port COM6 --mode manual --save-motion key_points

参数说明：
--mode: manual=手动选择关键点，auto=自动连续记录
--sample-hz: 自动模式的采样频率，数值越大记录越精细
--save-motion: 动作名称，将创建example_motions/<name>/目录
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any
from datetime import datetime

from alicia_duo_sdk.controller import get_default_session, ControlApi
from alicia_duo_sdk.controller.drag_teaching_controller import DragTeachingController


class DragTeachingDemo:
    """拖动示教演示主类 - 使用重构后的Controller层功能"""
    
    def __init__(self, args):
        self.args = args
        self.controller = None
        self.drag_teaching_controller = None
        
    def setup(self):
        """初始化设置"""
        print("=== 拖动示教演示 ===")
        print(f"模式: {self.args.mode}")
        print(f"采样频率: {self.args.sample_hz} Hz")
        print(f"保存动作名: {self.args.save_motion}")
        
        # 创建会话和控制器
        session = get_default_session(baudrate=self.args.baudrate, port=self.args.port)
        self.controller = ControlApi(session=session)
        
        # 创建拖动示教控制器（包含所有核心功能）
        self.drag_teaching_controller = DragTeachingController(self.controller)
        
        print("[设置] 初始化完成")
        
    def manual_teaching_mode(self):
        """手动示教模式 - 使用Controller层的手动示教功能"""
        print("\n=== 手动示教模式 ===")
        print("说明：关闭扭矩后，手动拖动机械臂到目标位置，按回车记录")
        print("注意：此模式只记录用户指定的关键点，适合抓取-放置等简单操作")
        
        # 使用Controller层的手动示教功能
        waypoints = self.drag_teaching_controller.manual_teaching()
        return waypoints
        
    def auto_teaching_mode(self):
        """自动示教模式 - 使用Controller层的自动示教功能"""
        print("\n=== 自动示教模式 ===")
        print("说明：关闭扭矩后，拖动机械臂，系统自动记录轨迹")
        print("注意：此模式会连续记录所有轨迹点，适合复杂路径和连续运动")
        print("按回车开始记录，再次按回车停止记录")
        
        # 使用Controller层的自动示教功能
        trajectory = self.drag_teaching_controller.auto_teaching(self.args.sample_hz)
        return trajectory
        
    def save_trajectory(self, trajectory_data: List[Dict[str, Any]], mode: str):
        """保存轨迹数据"""
        if not trajectory_data:
            print("[保存] 没有轨迹数据可保存")
            return None
            
        # 创建保存目录
        save_dir = os.path.join("example_motions", self.args.save_motion)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存轨迹数据
        traj_path = os.path.join(save_dir, "drag_trajectory.json")
        with open(traj_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
            
        # 保存元信息
        meta = {
            "motion": self.args.save_motion,
            "mode": mode,
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "sample_hz": self.args.sample_hz,
            "count": len(trajectory_data),
            "description": "拖动示教轨迹"
        }
        
        meta_path = os.path.join(save_dir, "meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        print(f"[保存] 轨迹数据: {traj_path}")
        print(f"[保存] 元信息: {meta_path}")
        
        return save_dir
        
    def replay_trajectory(self, trajectory_data: List[Dict[str, Any]]):
        """回放轨迹 - 使用Controller层的回放功能"""
        if not trajectory_data:
            print("[回放] 没有轨迹数据可回放")
            return
            
        print(f"\n=== 轨迹回放 ===")
        print(f"轨迹点数: {len(trajectory_data)}")
        
        # 询问用户是否要回放轨迹
        replay = input("是否回放轨迹？(y/n): ").strip().lower()
        if replay != 'y':
            print("[回放] 跳过回放")
            return
            
        # 使用Controller层的回放功能（Controller会自动处理安全检查和建议）
        success = self.drag_teaching_controller.replay_trajectory(
            trajectory_data, 
            slow_mode=False,  # 让Controller根据安全检查自动决定
            move_to_start=True
        )
        
        if success:
            print("[回放] 轨迹回放成功完成")
        else:
            print("[回放] 轨迹回放失败")
            

    
    def run(self):
        """运行演示"""
        try:
            self.setup()
            
            print("[安全] 紧急停止说明: 在任何时候，按下 Ctrl+C 可以立即停止机械臂")
            
            if self.args.mode == 'manual':
                trajectory_data = self.manual_teaching_mode()
            else:  # auto
                trajectory_data = self.auto_teaching_mode()
                
            if trajectory_data:
                # 保存轨迹
                save_dir = self.save_trajectory(trajectory_data, self.args.mode)
                
                if save_dir:
                    print(f"\n[完成] 拖动示教完成！")
                    print(f"轨迹已保存到: {save_dir}")
                    
                    # 询问是否回放
                    print("[安全] 回放过程中，按下 Ctrl+C 可以紧急停止")
                    self.replay_trajectory(trajectory_data)
                    
            else:
                print("[完成] 未记录任何轨迹数据")
                
        except KeyboardInterrupt:
            print("\n[中断] 用户中断操作")
            self.drag_teaching_controller.emergency_stop()
        except Exception as e:
            print(f"[错误] 演示运行失败: {e}")
            self.drag_teaching_controller.emergency_stop()
        finally:
            # 确保扭矩开启
            if self.controller:
                try:
                    self.controller.torque_control('on')
                except:
                    pass


def main():
    """主函数 - 解析命令行参数并启动拖动示教演示"""
    parser = argparse.ArgumentParser(description="拖动示教演示")
    parser.add_argument('--baudrate', type=int, default=1000000, 
                       help="串口波特率，固件5.4.19+推荐1000000，旧版本用921600")
    parser.add_argument('--port', default='COM6', 
                       help="串口端口，Windows用COM6，Linux/Mac用/dev/ttyUSB0等")
    parser.add_argument('--mode', choices=['manual', 'auto'], default='auto', 
                       help="示教模式：manual(手动选择关键点) 或 auto(自动连续记录)")
    parser.add_argument('--sample-hz', type=float, default=50.0, 
                       help="自动模式的采样频率(Hz)，数值越大记录越精细")
    parser.add_argument('--save-motion', required=True, 
                       help="保存的动作名称，将创建example_motions/<name>/目录")
    
    args = parser.parse_args()
    
    # 创建拖动示教演示实例并运行
    demo = DragTeachingDemo(args)
    demo.run()


if __name__ == "__main__":
    main()
