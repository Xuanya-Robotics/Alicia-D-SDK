# planners/cartesian.py

import numpy as np

class CartesianPlanner:
    def plan(self, ik_controller, robot_model, start_joint_angles, pose_sequence, steps_per_segment=20):
        """
        基于姿态序列，分段规划出一条关节轨迹
        Args:
            ik_controller: IKController 实例
            robot_model: RobotArm
            start_joint_angles: 当前关节状态（dict）
            pose_sequence: List of [x, y, z, qx, qy, qz, qw]
            steps_per_segment: 每段的插值步数

        Returns:
            List[List[float]]: 关节空间轨迹
        """
        joint_names = list(start_joint_angles.keys())
        current_angles = start_joint_angles.copy()
        trajectory = []

        for pose in pose_sequence:
            pos = pose[:3]
            quat = pose[3:]

            # 1. 逆解下一段末端姿态
            solved = ik_controller.ik_solver.solve(current_angles, pos, quat)
            if not solved:
                print(f"[CartesianPlanner] IK 解失败, pose: {pose}")
                continue

            start = [current_angles[j] for j in joint_names]
            end = [solved[j] for j in joint_names]

            # 2. 使用平滑插值构造当前段轨迹
            for i in range(steps_per_segment):
                alpha = self.smoothstep(i / (steps_per_segment - 1))
                point = [(1 - alpha) * s + alpha * e for s, e in zip(start, end)]
                trajectory.append(point)

            current_angles = solved  # 更新为下一段起点

        return trajectory

    @staticmethod
    def smoothstep(t: float) -> float:
        """
        平滑插值函数（S型）: 3t^2 - 2t^3
        """
        return 3 * t**2 - 2 * t**3
