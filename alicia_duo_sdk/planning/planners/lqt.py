# alicia_duo_sdk/planning/planners/lqt.py

import numpy as np
from typing import List, Tuple
from math import factorial
from alicia_duo_sdk.utils.logger import BeautyLogger
import copy 
import time 

class LQT:
    def __init__(self, verbose: bool=False):
        self.logger = BeautyLogger(log_dir="./logs", 
                                   log_name="lqt_planner.log", 
                                   verbose=verbose)
        # 参数
        self.nbData= 200
        self.nbDeriv= 2
        self.nbVarPos=  7
        self.nbVar= 14
        self.nbVarX= 15
        self.dt= 1E-2
        self.rfactor= 1E-6


    def plan(self,
             via_points: List[List[float]],
             nbdata: int = None) -> List[List[float]]:
        """
        使用线性二次轨迹优化（LQT）算法进行轨迹规划

        Args:
            via_points: List of 7D pose (x, y, z, qx, qy, qz, qw)

        Returns:
            Tuple of control inputs, trajectory, joint traj, index slices, and time list
        """

        self.all_points = via_points
        self.nbPoints = len(via_points)
        self.start_point, self.via_points = self._data_process(self.all_points)
        if nbdata:
            self.nbData = nbdata

        self.logger.module("[LQT] 开始规划顺滑轨迹")
        t0 = time.time()

        mu, Q, R, idx_slices, tl = self.get_matrices()
        Su, Sx = self.set_dynamical_system()
        u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx)
        pose_traj = copy.deepcopy(x_hat[:,:7]).tolist()

        # 如果输入的 via_points 含 gripper（任一第8列），则对 gripper 做线性/平滑插值
        gripper_traj = None
        if any(len(p) == 8 for p in via_points):
            # 简单处理：按路径点区段线性插值至与 pose_traj 等长
            # 这里使用 smoothstep 权重
            def smoothstep(t: float) -> float:
                return 3 * (t ** 2) - 2 * (t ** 3)

            # 估算每段步数，基于位置+姿态变化的近似（与 CartesianPlanner 保持一致理念）
            # 为简化，这里按等分 nbData 在各段平均分配
            num_total = len(pose_traj)
            num_segments = len(via_points) - 1
            base = max(1, num_total // num_segments)
            remainder = num_total - base * num_segments

            gripper_traj = []
            for i in range(num_segments):
                # 如果某个端点缺少 gripper，则延用邻近端点的值
                g1 = float(via_points[i][7]) if len(via_points[i]) == 8 else float(via_points[i+1][7])
                g2 = float(via_points[i+1][7]) if len(via_points[i+1]) == 8 else float(via_points[i][7])
                steps = base + (1 if i < remainder else 0)
                if steps <= 1:
                    segment = [g2]
                else:
                    segment = [g1 + (g2 - g1) * smoothstep(j / (steps - 1)) for j in range(steps)]
                if i == 0:
                    gripper_traj.extend(segment)
                else:
                    gripper_traj.extend(segment[1:])
        joint_spd_traj = copy.deepcopy(x_hat[:,7:])
        # return u_hat, x_hat, mu, idx_slices, tl

        t1 = time.time()
        self.logger.info("[LQT]完成POSE顺滑轨迹规划, "
                    f"规划用时{t1-t0: .2f}秒, 共{len(pose_traj)}个轨迹点")
        if gripper_traj is not None:
            return pose_traj, gripper_traj
        return pose_traj

    def _data_process(self, data):
        if len(data[0]) == self.nbVar:
            all_points = data
        else:
            all_points = np.zeros((len(data), self.nbVar))
            all_points[:, :self.nbVarPos] = data
        start_point = all_points[0]
        via_points = all_points[1:]
        return start_point, via_points

    def get_matrices(self):
        self.nbPoints = len(self.via_points)
        R = np.identity((self.nbData - 1) * self.nbVarPos, dtype=np.float32) * self.rfactor

        tl = np.linspace(0, self.nbData, self.nbPoints + 1)
        tl = np.rint(tl[1:]).astype(np.int64) - 1
        idx_slices = [slice(i, i + self.nbVar, 1) for i in (tl * self.nbVar)]

        mu = np.zeros((self.nbVar * self.nbData, 1), dtype=np.float32)
        Q = np.zeros((self.nbVar * self.nbData, self.nbVar * self.nbData), dtype=np.float32)

        for i in range(len(idx_slices)):
            slice_t = idx_slices[i]
            x_t = self.via_points[i].reshape((self.nbVar, 1))
            mu[slice_t] = x_t
            Q[slice_t, slice_t] = np.diag(
                np.hstack((np.ones(self.nbVarPos), np.zeros(self.nbVar - self.nbVarPos))))
        return mu, Q, R, idx_slices, tl

    def set_dynamical_system(self):
        A1d = np.zeros((self.nbDeriv, self.nbDeriv), dtype=np.float32)
        B1d = np.zeros((self.nbDeriv, 1), dtype=np.float32)
        for i in range(self.nbDeriv):
            A1d += np.diag(np.ones(self.nbDeriv - i), i) * self.dt ** i * 1 / factorial(i)
            B1d[self.nbDeriv - i - 1] = self.dt ** (i + 1) * 1 / factorial(i + 1)

        A = np.kron(A1d, np.identity(self.nbVarPos, dtype=np.float32))
        B = np.kron(B1d, np.identity(self.nbVarPos, dtype=np.float32))

        Su = np.zeros((self.nbVar * self.nbData, self.nbVarPos * (self.nbData - 1)))
        Sx = np.kron(np.ones((self.nbData, 1)), np.eye(self.nbVar, self.nbVar))

        M = B
        for i in range(1, self.nbData):
            Sx[i * self.nbVar:self.nbData * self.nbVar, :] = np.dot(
                Sx[i * self.nbVar:self.nbData * self.nbVar, :], A)
            Su[self.nbVar * i:self.nbVar * i + M.shape[0], 0:M.shape[1]] = M
            M = np.hstack((np.dot(A, M), B))
        return Su, Sx

    def get_u_x(self, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0 = self.start_point.reshape((self.nbVar, 1))
        u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (mu - Sx @ x0)
        x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, self.nbVar))
        return u_hat, x_hat

  