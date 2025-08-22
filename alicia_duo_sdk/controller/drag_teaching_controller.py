"""
拖动示教控制器 - 提供拖动示教的核心功能

这个模块将拖动示教的核心功能从demo中提取出来，提供：
1. 轨迹记录服务
2. 轨迹回放服务
3. 安全检查机制
4. 插值算法
"""

import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .control_api import ControlApi
from ..utils.logger import BeautyLogger

logger = BeautyLogger(log_dir="./logs", log_name="drag_teaching.log", verbose=True)


class TrajectoryRecorder:
    """轨迹记录器 - 负责以固定频率记录机械臂轨迹"""
    
    def __init__(self, controller: ControlApi, sample_hz: float = 50.0):
        self.controller = controller          # 机械臂控制器
        self.sample_hz = float(sample_hz)    # 采样频率（Hz）
        self._stop = threading.Event()       # 停止信号
        self._thread = None                  # 后台记录线程
        self._trajectory: List[Dict[str, Any]] = []  # 存储记录的轨迹数据
        self._t0 = None                      # 记录开始的时间戳
        self._is_recording = False           # 是否正在记录的标志
        
    def start_recording(self):
        """开始记录轨迹"""
        if self._thread is not None:
            return
            
        self._stop.clear()
        self._is_recording = True
        self._t0 = time.time()
        self._trajectory.clear()
        
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info(f"[记录器] 开始记录轨迹 @ {self.sample_hz} Hz")
        
    def stop_recording(self) -> List[Dict[str, Any]]:
        """停止记录并返回轨迹数据"""
        if self._thread is None:
            return []
        
        # 先标记停止记录，但给一点时间让最后的数据点被记录
        logger.info("[记录器] 开始停止记录...")
        self._is_recording = False
        time.sleep(0.1)  # 给100ms时间让最后的点被记录
        
        self._stop.set()
        self._thread.join(timeout=2.0)  # 最多等待2秒
        self._thread = None
        
        # 检查轨迹数据的完整性
        if self._trajectory:
            first_time = self._trajectory[0]['t']
            last_time = self._trajectory[-1]['t']
            duration = last_time - first_time
            expected_points = int(duration * self.sample_hz)
            actual_points = len(self._trajectory)
            
            logger.info(f"[记录器] 停止记录完成")
            logger.info(f"[记录器] 记录时长: {duration:.2f}秒")
            logger.info(f"[记录器] 实际点数: {actual_points}, 期望点数: {expected_points}")
            logger.info(f"[记录器] 记录完整度: {actual_points/max(1,expected_points)*100:.1f}%")
            
            if actual_points < expected_points * 0.9:
                logger.warning(f"[记录器] 警告：记录的点数比期望少，可能存在数据丢失")
        else:
            logger.warning("[记录器] 警告：没有记录到任何轨迹点")
        
        return self._trajectory.copy()
        
    def _record_loop(self):
        """记录循环"""
        dt = 1.0 / max(1.0, self.sample_hz)
        
        last_log_time = 0
        
        while not self._stop.is_set():
            try:
                if self._is_recording:
                    loop_start = time.time()
                    t = loop_start - self._t0
                    
                    # 获取当前状态
                    joints = self.controller.get_joints()
                    pose = self.controller.get_pose()
                    gripper = self.controller.get_gripper()
                    
                    if joints is not None and pose is not None:
                        trajectory_point = {
                            "t": t,
                            "joints": joints,
                            "pose": pose,
                            "gripper": gripper
                        }
                        self._trajectory.append(trajectory_point)
                        
                        # 每秒输出一次统计信息
                        if t - last_log_time >= 1.0:
                            logger.info(f"[记录器] 已记录 {len(self._trajectory)} 点，时长 {t:.1f}s，频率 {len(self._trajectory)/max(0.1,t):.1f}Hz")
                            last_log_time = t
                    else:
                        logger.warning(f"[记录器] 第{len(self._trajectory)}点数据获取失败: joints={joints is not None}, pose={pose is not None}")
                        
                    # 检测记录循环性能
                    loop_time = time.time() - loop_start
                    if loop_time > dt * 1.5:  # 如果循环时间超过期望时间的1.5倍
                        logger.warning(f"[记录器] 记录循环延迟: {loop_time:.3f}s > {dt:.3f}s")
                        
            except Exception as e:
                logger.error(f"[记录器] 记录错误: {e}")
                
            time.sleep(dt)
            
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """获取当前轨迹数据"""
        return self._trajectory.copy()


class TrajectoryInterpolator:
    """轨迹插值器 - 提供轨迹插值算法"""
    
    @staticmethod
    def interpolate_trajectory(trajectory: List[Dict[str, Any]], target_time: float) -> Optional[Dict[str, Any]]:
        """根据时间戳在两个轨迹点之间进行线性插值"""
        if not trajectory:
            return None
            
        # 找到目标时间所在的时间区间（在两个记录点之间）
        for i in range(len(trajectory) - 1):
            t1, t2 = trajectory[i]['t'], trajectory[i + 1]['t']  # 当前点和下一点的时间戳
            
            if t1 <= target_time <= t2:
                # 计算插值比例：alpha=0表示在第一个点，alpha=1表示在第二个点
                alpha = (target_time - t1) / (t2 - t1)
                
                # 线性插值关节角度：在两个关节角度之间按比例插值
                joints1 = np.array(trajectory[i]['joints'])      # 第一个点的关节角度
                joints2 = np.array(trajectory[i + 1]['joints'])  # 第二个点的关节角度
                interpolated_joints = joints1 + alpha * (joints2 - joints1)  # 线性插值公式
                
                # 线性插值夹爪角度：在两个夹爪角度之间按比例插值
                grip1 = trajectory[i].get('gripper', 0.0)       # 第一个点的夹爪角度
                grip2 = trajectory[i + 1].get('gripper', 0.0)   # 第二个点的夹爪角度
                interpolated_gripper = grip1 + alpha * (grip2 - grip1)  # 线性插值公式
                
                return {
                    'joints': interpolated_joints.tolist(),      # 插值后的关节角度
                    'gripper': interpolated_gripper              # 插值后的夹爪角度
                }
                
        return None  # 如果找不到合适的时间区间，返回None
        
    @staticmethod
    def interpolate_waypoints(waypoints: List[Dict[str, Any]], alpha: float) -> Optional[Dict[str, Any]]:
        """在手动模式记录的关键点之间进行线性插值"""
        if not waypoints:
            return None
            
        if len(waypoints) == 1:  # 如果只有一个点，直接返回
            return waypoints[0]
            
        # 找到对应的两个路点（按索引插值，而不是按时间插值）
        total_points = len(waypoints) - 1  # 总段数（点数-1）
        point_index = alpha * total_points  # 计算目标点在整个路径中的位置
        i = int(point_index)  # 取整数部分，确定当前段
        
        if i >= len(waypoints) - 1:  # 如果超出范围，返回最后一个点
            return waypoints[-1]
            
        # 线性插值：计算在当前段内的插值比例
        local_alpha = point_index - i  # 局部插值比例（0到1之间）
        wp1, wp2 = waypoints[i], waypoints[i + 1]  # 当前段的两个端点
        
        # 线性插值关节角度：在两个关键点之间按比例插值
        joints1 = np.array(wp1['joints'])      # 第一个关键点的关节角度
        joints2 = np.array(wp2['joints'])      # 第二个关键点的关节角度
        interpolated_joints = joints1 + local_alpha * (joints2 - joints1)  # 线性插值公式
        
        # 线性插值夹爪角度：在两个关键点之间按比例插值
        grip1 = wp1.get('gripper', 0.0)       # 第一个关键点的夹爪角度
        grip2 = wp2.get('gripper', 0.0)       # 第二个关键点的夹爪角度
        interpolated_gripper = grip1 + local_alpha * (grip2 - grip1)  # 线性插值公式
        
        return {
            'joints': interpolated_joints.tolist(),      # 插值后的关节角度
            'gripper': interpolated_gripper              # 插值后的夹爪角度
        }


class SafetyChecker:
    """安全检查器 - 提供轨迹回放的安全检查功能"""
    
    @staticmethod
    def check_start_position_safety(controller: ControlApi, trajectory_data: List[Dict[str, Any]]) -> Tuple[bool, float, bool]:
        """
        检查当前位置与轨迹起点的安全性
        
        Returns:
            Tuple[bool, float, bool]: (是否安全, 最大角度差异, 是否需要缓慢模式)
        """
        current_joints = controller.get_joints()
        if current_joints is None:
            return False, 0.0, False
            
        # 获取轨迹起点的关节角度
        if 'joints' in trajectory_data[0]:
            start_joints = trajectory_data[0]['joints']
        else:
            start_joints = trajectory_data[0]['joints']
            
        # 计算关节角度差异
        joint_diff = np.array(start_joints) - np.array(current_joints)
        max_diff = np.max(np.abs(joint_diff))
        
        # 判断是否需要缓慢模式（差异超过0.5弧度，约28.6度）
        need_slow_mode = max_diff > 0.5
        
        # 判断是否安全（差异超过1.0弧度，约57.3度）
        is_safe = max_diff <= 1.0
        
        return is_safe, max_diff, need_slow_mode


class DragTeachingController:
    """拖动示教控制器 - 整合所有拖动示教功能"""
    
    def __init__(self, controller: ControlApi):
        self.controller = controller
        self.recorder = TrajectoryRecorder(controller)
        self.interpolator = TrajectoryInterpolator()
        self.safety_checker = SafetyChecker()
        
    def manual_teaching(self, waypoints: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """手动示教 - 记录用户指定的关键位置点"""
        if waypoints is None:
            waypoints = []
            
        logger.info("[手动示教] 开始手动示教模式")
        
        # 关闭扭矩，让机械臂可以自由移动
        self.controller.torque_control('off')
        
        waypoint_count = len(waypoints)
        
        while True:
            cmd = input(f"\n拖动完成后按回车记录第 {waypoint_count + 1} 个位置，输入 'q' 结束录制: ").strip()
            
            if cmd.lower() == 'q':
                break
                
            try:
                # 记录当前位置
                pose = self.controller.get_pose()      # 获取末端执行器的位置和姿态
                gripper = self.controller.get_gripper()  # 获取夹爪角度
                joints = self.controller.get_joints()    # 获取6个关节角度
                
                if pose is not None and gripper is not None and joints is not None:
                    waypoint = {
                        "pose": pose,        # 末端位姿：位置[x,y,z] + 四元数姿态[qx,qy,qz,qw]
                        "gripper": gripper,  # 夹爪角度
                        "joints": joints,    # 6个关节角度
                        "timestamp": time.time()  # 记录时间戳
                    }
                    waypoints.append(waypoint)
                    waypoint_count += 1
                    
                    logger.info(f"[手动示教] 记录成功 Waypoint {waypoint_count}")
                    
            except Exception as e:
                logger.error(f"[手动示教] 记录失败: {e}")
                
        # 重新开启扭矩，防止机械臂意外移动
        self.controller.torque_control('on')
        
        logger.info(f"[手动示教] 完成，共记录 {len(waypoints)} 个关键点")
        return waypoints
        
    def auto_teaching(self, sample_hz: float = 50.0) -> List[Dict[str, Any]]:
        """自动示教 - 以固定频率自动记录连续轨迹"""
        logger.info(f"[自动示教] 开始自动示教模式，采样频率: {sample_hz} Hz")
        
        # 设置采样频率
        self.recorder.sample_hz = sample_hz
        
        # 关闭扭矩，让机械臂可以自由移动
        self.controller.torque_control('off')
        
        input("按回车开始自动记录轨迹...")
        
        # 启动后台记录线程
        self.recorder.start_recording()
        
        input("按回车停止记录轨迹...")
        
        # 停止记录并获取轨迹数据
        trajectory = self.recorder.stop_recording()
        
        # 重新开启扭矩，防止机械臂意外移动
        self.controller.torque_control('on')
        
        logger.info(f"[自动示教] 完成，共记录 {len(trajectory)} 个轨迹点")
        return trajectory
        
    def replay_trajectory(self, trajectory_data: List[Dict[str, Any]], 
                         slow_mode: bool = False, 
                         move_to_start: bool = True) -> bool:
        """
        回放轨迹
        
        Args:
            trajectory_data: 轨迹数据
            slow_mode: 是否使用缓慢模式
            move_to_start: 是否先移动到轨迹起点
            
        Returns:
            bool: 是否成功完成回放
        """
        if not trajectory_data:
            logger.warning("[回放] 没有轨迹数据可回放")
            return False
            
        logger.info(f"[回放] 开始回放轨迹，点数: {len(trajectory_data)}")
        

        # 安全检查：确认当前位置与轨迹起点是否安全
        logger.info("[安全] 检查当前位置与轨迹起点的安全性...")
        current_joints = self.controller.get_joints()
        if current_joints is None:
            logger.warning("[警告] 无法获取当前关节状态，建议手动检查机械臂位置")
            confirm = input("是否继续？(y/n): ").strip().lower()
            if confirm != 'y':
                logger.info("[回放] 用户取消回放")
                return False
        
        # 获取轨迹起点的关节角度
        if 'joints' in trajectory_data[0]:
            start_joints = trajectory_data[0]['joints']
        else:
            start_joints = trajectory_data[0]['joints']
            
        # 计算关节角度差异
        if current_joints:
            joint_diff = np.array(start_joints) - np.array(current_joints)
            max_diff = np.max(np.abs(joint_diff))
            logger.info(f"[安全] 当前关节与起点最大差异: {max_diff:.4f} rad ({max_diff * 180 / np.pi:.2f}°)")
            
            # 如果差异太大（超过0.5弧度，约28.6度），提供安全选择
            if max_diff > 0.5:  # 约28.6度
                logger.warning("[警告] 关节角度差异较大，建议先移动到起点附近")
                print("选项:")
                print("  1. 手动移动到起点附近后继续")
                print("  2. 使用缓慢移动模式（较慢但安全）[默认]")
                print("  3. 取消回放")
                
                choice = input("请选择 (1/2/3，直接回车使用默认选项2): ").strip()
                if choice == '3':
                    logger.info("[回放] 用户取消回放")
                    return False
                elif choice == '2' or choice == '':  # 选择2或直接回车都使用缓慢模式
                    logger.info("[安全] 启用缓慢移动模式")
                    slow_mode = True
                else:
                    logger.info("[安全] 请手动移动到起点附近，完成后按回车继续...")
                    input()
                    slow_mode = False
            else:
                slow_mode = False  # 差异不大，使用正常速度
        else:
            slow_mode = False
                
        # 启动在线插值 - 轨迹回放使用正常高速参数
        logger.info("[回放] 使用正常速度参数进行轨迹回放")
        self.controller.startOnlineSmoothing(
            command_rate_hz=200.0,
            max_joint_velocity_rad_s=100.0,    # 轨迹回放：使用高速参数
            max_joint_accel_rad_s2=100.0,     # 轨迹回放：使用高速参数
            max_gripper_velocity_rad_s=2.0,    # 提高夹爪速度到2.0 rad/s
            max_gripper_accel_rad_s2=8.0,      # 提高夹爪加速度到8.0 rad/s²
        )
        
        try:
            # 设置当前位置为初始目标，避免突然跳跃
            current_joints = self.controller.get_joints()
            if current_joints:
                self.controller.setJointTargetOnline(current_joints)
                time.sleep(0.5)  # 给在线插值器时间初始化
                
            # 计算回放时间（只用于手动模式，自动模式有自己的时间处理）
            if 't' not in trajectory_data[0]:
                # 手动模式：使用固定时间
                total_time = 10.0  # 手动模式固定10秒，不受缓慢模式影响
                logger.info(f"[回放] 手动模式预计回放时间: {total_time:.2f}秒")
                
            # 如果需要，先移动到轨迹起点
            if move_to_start:
                start_joints = trajectory_data[0]['joints']
                start_gripper = trajectory_data[0].get('gripper', None)
                
                logger.info("[回放] 正在移动到轨迹起点...")
                self.controller.setJointTargetOnline(start_joints)
                if start_gripper is not None:
                    try:
                        self.controller.setGripperTargetOnline(start_gripper)
                    except:
                        pass
                        
                # 等待到达起点
                move_time = 5.0 if slow_mode else 3.0
                time.sleep(move_time)
                
            # 如果是自动模式（含时间戳），直接按录制时间戳回放，不使用在线插值
            if 't' in trajectory_data[0]:
                logger.info("[回放] 自动模式：按录制时间戳直接回放")
                
                # 停止在线插值，直接控制关节
                self.controller.stopOnlineSmoothing()
                logger.info("[回放] 停止在线插值，使用直接关节控制")
                
                times = [p['t'] for p in trajectory_data]
                total_original_time = times[-1] - times[0]
                total_points = len(trajectory_data)
                
                logger.info(f"[回放] 轨迹时长: {total_original_time:.2f}秒，共 {total_points} 个点")
                
                # 计算平均录制频率
                if total_points > 1:
                    avg_interval = total_original_time / (total_points - 1)
                    record_hz = 1.0 / avg_interval
                    logger.info(f"[回放] 平均录制频率: {record_hz:.1f}Hz")
                
                # 数据完整性检查
                logger.info("[回放] 检查轨迹数据完整性...")
                
                # 检查时间戳连续性
                time_gaps = []
                for i in range(1, len(times)):
                    gap = times[i] - times[i-1]
                    time_gaps.append(gap)
                
                if time_gaps:
                    max_gap = max(time_gaps)
                    min_gap = min(time_gaps)
                    avg_gap = sum(time_gaps) / len(time_gaps)
                    
                    logger.info(f"[回放] 时间间隔统计: 最小={min_gap:.3f}s, 最大={max_gap:.3f}s, 平均={avg_gap:.3f}s")
                    
                    # 检查是否有异常大的时间间隔（可能表示数据丢失）
                    large_gaps = [i for i, gap in enumerate(time_gaps) if gap > avg_gap * 3]
                    if large_gaps:
                        logger.warning(f"[回放] 发现 {len(large_gaps)} 个异常大的时间间隔，可能存在数据丢失")
                        logger.warning(f"[回放] 异常间隔位置: {large_gaps[:5]}...")  # 只显示前5个
                
                # 检查关节数据完整性
                missing_joints = sum(1 for p in trajectory_data if 'joints' not in p or p['joints'] is None)
                if missing_joints > 0:
                    logger.warning(f"[回放] 发现 {missing_joints} 个点缺少关节数据")
                
                logger.info("[回放] 数据完整性检查完成")
                
                start_time = time.time()
                
                # 最简单方案：直接按原记录轨迹点执行，不做任何插值
                logger.info(f"[回放] 直接按原记录轨迹执行，共 {total_points} 个点")
                
                # 计算平均时间间隔
                if total_points > 1:
                    avg_dt = total_original_time / (total_points - 1)
                else:
                    avg_dt = 0.02  # 默认50Hz
                
                logger.info(f"[回放] 平均点间隔: {avg_dt:.3f}秒")
                
                for i, point in enumerate(trajectory_data):
                    # 直接设置关节角度，不做任何插值
                    try:
                        logger.info(f"[回放] 执行第 {i+1}/{total_points} 个点")
                        self.controller.joint_controller.set_joint_angles(point['joints'])
                        if point.get('gripper') is not None:
                            self.controller.joint_controller.set_gripper(point['gripper'])
                    except Exception as e:
                        logger.error(f"[回放] 第{i}点设置失败: {e}")
                    
                    # 按平均时间间隔等待
                    if i < total_points - 1:  # 不在最后一个点等待
                        time.sleep(avg_dt)
                    
                    # 进度显示
                    if i % max(1, total_points // 10) == 0 or i == total_points - 1:
                        progress = (i / max(1, total_points - 1)) * 100.0
                        actual_time = time.time() - start_time
                        logger.info(f"[回放] 进度: {progress:.1f}%, 点: {i+1}/{total_points}, 实际时间: {actual_time:.2f}s")
                
                # 最后等待一下确保最后一个命令执行完成
                logger.info("[回放] 执行完所有轨迹点，等待0.5秒确保完成")
                time.sleep(0.5)
                
                elapsed_time = time.time() - start_time
                logger.info(f"[回放] 直接时间戳回放完成，实际用时: {elapsed_time:.2f}秒，原始时长: {total_original_time:.2f}秒")
                return True

            # 开始轨迹回放（手动模式：插值）
            target_hz = 30.0
            step_time = 1.0 / target_hz
            steps = int(total_time * target_hz)
            
            logger.info(f"[回放] 开始执行轨迹，共 {steps} 步")
            
            start_time = time.time()
            
            for i in range(steps):
                current_time = (i / steps) * total_time
                
                # 找到对应的轨迹点
                if 't' in trajectory_data[0]:
                    target_point = self.interpolator.interpolate_trajectory(trajectory_data, current_time)
                else:
                    target_point = self.interpolator.interpolate_waypoints(trajectory_data, i / steps)
                    
                if target_point:
                    # 设置关节目标
                    if 'joints' in target_point:
                        self.controller.setJointTargetOnline(target_point['joints'])
                        
                    # 设置夹爪目标
                    if 'gripper' in target_point and target_point['gripper'] is not None:
                        try:
                            self.controller.setGripperTargetOnline(target_point['gripper'])
                        except:
                            pass
                            
                # 显示进度
                if i % 10 == 0:
                    progress = (i / steps) * 100
                    logger.info(f"[回放] 进度: {progress:.1f}%")
                            
                time.sleep(step_time)
                
            elapsed_time = time.time() - start_time
            logger.info(f"[回放] 回放完成，实际用时: {elapsed_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"[回放] 回放失败: {e}")
            return False
            
        finally:
            # 停止在线插值
            self.controller.stopOnlineSmoothing()
            logger.info("[回放] 停止在线插值")
            
    def emergency_stop(self):
        """紧急停止"""
        logger.warning("[紧急停止] 正在停止机械臂...")
        
        try:
            self.controller.stopOnlineSmoothing()
        except:
            pass
            
        try:
            self.controller.torque_control('on')
        except:
            pass
            
        logger.info("[紧急停止] 机械臂已停止")
