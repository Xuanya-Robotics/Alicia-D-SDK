# trajectory_planner.py
import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R, Slerp


def interpolate_pose_trajectory(
    start_pos: List[float], start_quat: List[float],
    end_pos: List[float], end_quat: List[float],
    steps: int = 100
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    从起始位置和四元数插值生成姿态轨迹
    返回: 每个点的 (position, quaternion) 元组
    """
    poses = []
    start_quat = np.array(start_quat)
    end_quat = np.array(end_quat)
    if np.dot(start_quat, end_quat) < 0.0:
        end_quat = -end_quat

    key_rots = R.from_quat([start_quat, end_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(steps):
        t = i / (steps - 1)
        pos = (1 - t) * np.array(start_pos) + t * np.array(end_pos)
        quat = slerp([t]).as_quat()[0]
        poses.append((pos, quat))
    return poses


def plan_traj(
    ik_controller,
    robot_model,
    start_joint_angles: List[float],
    end_pos: List[float],
    end_quat: List[float],
    steps: int = 100,
    wrist_sharing: bool = True
) -> List[List[float]]:
    """
    使用姿态插值 + 每帧IK解算，生成关节空间轨迹
    """
    start_pose, start_quat = robot_model.forward_kinematics(
        dict(zip(robot_model.kinematic_chain[:-1], start_joint_angles))
    )
    poses = interpolate_pose_trajectory(start_pose, start_quat, end_pos, end_quat, steps)

    all_joint_angles = []
    current_angles = start_joint_angles.copy()
    for pos, quat in poses:
        ik_controller.update_joint_angles(dict(zip(robot_model.kinematic_chain[:-1], current_angles)))
        ik_controller.set_target(pos, quat)
        result = ik_controller.solve_ik()
        if not result:
            print("❌ 某个点IK失败，跳过")
            continue

        if wrist_sharing:
            joint4 = result.get("joint4", 0)
            joint6 = result.get("joint6", 0)
            offset = (joint4 + joint6) / 2
            result["joint4"] = offset
            result["joint6"] = offset

        joint_angles = [result[k] for k in robot_model.kinematic_chain[:-1]]
        all_joint_angles.append(joint_angles)
        current_angles = joint_angles

    return all_joint_angles
