# linear.py - 平滑线性插值规划器

import numpy as np

class LinearPlanner:
    def plan(self, ik_controller, start_joint_angles, end_pos, end_quat, steps=50):
        """
        给定起始关节角度和目标末端姿态，规划平滑关节轨迹（直线运动）

        Args:
            ik_controller: IKController 实例
            robot_model: RobotArm 实例
            start_joint_angles: 当前关节角（Dict[str, float]）
            end_pos: 目标位置 [x, y, z]
            end_quat: 目标四元数 [qx, qy, qz, qw]
            steps: 插值步数

        Returns:
            List[List[float]]: 每一帧的关节角度数组
        """
        joint_num = 6

        # 1. 目标末端姿态 → IK → 得到终点 joint 解
        solved = ik_controller.ik_solver.solve(start_joint_angles, end_pos, end_quat)

        if solved is None:
            print("[LinearPlanner] IK 解失败，无法生成轨迹")
            return False

        # 2. 构造起点/终点角度向量
        start = [start_joint_angles[j] for j in range(joint_num)]
        end = [solved[j] for j in range(joint_num)]

        # 3. 插值轨迹（使用平滑插值）
        trajectory = []
        for i in range(steps):
            raw_alpha = i / (steps - 1)
            alpha = self.smoothstep(raw_alpha)
            point = [(1 - alpha) * s + alpha * e for s, e in zip(start, end)]
            trajectory.append(point)

        return trajectory

    @staticmethod
    def smoothstep(t: float) -> float:
        """
        平滑插值函数（S型）: 3t^2 - 2t^3
        保证首尾速度为 0，过渡更柔和
        """
        return 3 * t**2 - 2 * t**3
