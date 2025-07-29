# motion_api.py (in controller/)

from ..planning.planner_registry import get_planner
from ..execution.trajectory_executor import TrajectoryExecutor
from ..visualization.trajectory_viz import plot_joint_angles
from ..utils.validation import *
from typing import List, Optional
import numpy as np


def moveL(
    session,
    target_pose: List[float],  # [x, y, z, qx, qy, qz, qw]
    start_joint_angles: Optional[List[float]] = None,
    steps: int = 100,
    planner_name: str = "linear",
    visualize: bool = False,
    delay: float = 0.02
):
    
    validate_pose(target_pose)

    # 使用传入的起点；如果没有，读取当前状态
    if start_joint_angles is None:
        start_joint_angles = session.joint_controller.get_joint_angles()
    else:
        validate_joint_list(start_joint_angles)
    
    planner = get_planner(planner_name)
    trajectory = planner.plan(
        session.ik_controller,
        start_joint_angles=start_joint_angles,
        end_pos=target_pose[:3],
        end_quat=target_pose[3:],
        steps=steps
    )

    if visualize:
        plot_joint_angles(np.array(trajectory), title="moveL Joint Trajectory")
    input("是否执行")
    executor = TrajectoryExecutor(session.joint_controller, delay=delay)
    executor.execute(trajectory)

    session.last_joint_angles = trajectory[-1]


def moveCartesian(
    session,
    pose_sequence: List[List[float]],  # List of [x, y, z, qx, qy, qz, qw]
    start_joint_angles: Optional[List[float]] = None,
    planner_name: str = "cartesian",
    visualize: bool = False,
    delay: float = 0.02
):
    
    validate_pose_sequence(pose_sequence)

    # 使用传入的起点；如果没有，读取当前状态
    if start_joint_angles is None:
        start_joint_angles = session.joint_controller.get_joint_angles()
    else:
        validate_joint_list

    planner = get_planner(planner_name)   
    trajectory = planner.plan(
        session.ik_controller,
        start_joint_angles=start_joint_angles,
        pose_sequence=pose_sequence
    )
    if visualize:
        plot_joint_angles(np.array(trajectory), title="moveCartesian Joint Trajectory")
    executor = TrajectoryExecutor(session.joint_controller, delay=delay)
    executor.execute(trajectory)
    session.last_joint_angles = trajectory[-1]