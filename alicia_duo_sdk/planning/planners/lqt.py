# alicia_duo_sdk/planning/planners/lqt.py

import numpy as np
from omegaconf import DictConfig
from typing import List, Tuple
from math import factorial
from alicia_duo_sdk.utils.config import get_planning_config
from alicia_duo_sdk.utils.logger import beauty_print
import copy 

class LQT:
    def __init__(self):
        self.cfg = None
        self.all_points = None
        self.start_point = None
        self.via_points = None
        self.ik_controller = None
        self.initial_cfg = None 

    def plan(self, ik_controller, initial_cfg: List[float],
             via_points: List[List[float]], cfg: DictConfig = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                                   List[slice], np.ndarray]:
        """
        使用线性二次轨迹优化（LQT）算法进行轨迹规划

        Args:
            via_points: List of 7D pose (x, y, z, qx, qy, qz, qw)
            cfg: 配置参数（DictConfig）

        Returns:
            Tuple of control inputs, trajectory, joint traj, index slices, and time list
        """
        self.cfg = get_planning_config("lqt") if cfg is None else cfg
        self.all_points = via_points
        self.start_point, self.via_points = self._data_process(self.all_points)
        self.ik_controller = ik_controller
        self.initial_cfg = initial_cfg

        beauty_print("Planning smooth trajectory via LQT", type='module')
        mu, Q, R, idx_slices, tl = self.get_matrices()
        Su, Sx = self.set_dynamical_system()
        u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx)
        pose_traj = copy.deepcopy(x_hat[:,:7])
        joint_spd_traj = copy.deepcopy(x_hat[:,7:])
        # return u_hat, x_hat, mu, idx_slices, tl
        joint_traj = self.get_ik(pose_traj)
        return pose_traj.tolist(), joint_traj

    def _data_process(self, data):
        if len(data[0]) == self.cfg.nbVar:
            all_points = data
        else:
            all_points = np.zeros((len(data), self.cfg.nbVar))
            all_points[:, :self.cfg.nbVarPos] = data
        start_point = all_points[0]
        via_points = all_points[1:]
        return start_point, via_points

    def get_matrices(self):
        self.cfg.nbPoints = len(self.via_points)
        R = np.identity((self.cfg.nbData - 1) * self.cfg.nbVarPos, dtype=np.float32) * self.cfg.rfactor

        tl = np.linspace(0, self.cfg.nbData, self.cfg.nbPoints + 1)
        tl = np.rint(tl[1:]).astype(np.int64) - 1
        idx_slices = [slice(i, i + self.cfg.nbVar, 1) for i in (tl * self.cfg.nbVar)]

        mu = np.zeros((self.cfg.nbVar * self.cfg.nbData, 1), dtype=np.float32)
        Q = np.zeros((self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVar * self.cfg.nbData), dtype=np.float32)

        for i in range(len(idx_slices)):
            slice_t = idx_slices[i]
            x_t = self.via_points[i].reshape((self.cfg.nbVar, 1))
            mu[slice_t] = x_t
            Q[slice_t, slice_t] = np.diag(
                np.hstack((np.ones(self.cfg.nbVarPos), np.zeros(self.cfg.nbVar - self.cfg.nbVarPos))))
        return mu, Q, R, idx_slices, tl

    def set_dynamical_system(self):
        A1d = np.zeros((self.cfg.nbDeriv, self.cfg.nbDeriv), dtype=np.float32)
        B1d = np.zeros((self.cfg.nbDeriv, 1), dtype=np.float32)
        for i in range(self.cfg.nbDeriv):
            A1d += np.diag(np.ones(self.cfg.nbDeriv - i), i) * self.cfg.dt ** i * 1 / factorial(i)
            B1d[self.cfg.nbDeriv - i - 1] = self.cfg.dt ** (i + 1) * 1 / factorial(i + 1)

        A = np.kron(A1d, np.identity(self.cfg.nbVarPos, dtype=np.float32))
        B = np.kron(B1d, np.identity(self.cfg.nbVarPos, dtype=np.float32))

        Su = np.zeros((self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVarPos * (self.cfg.nbData - 1)))
        Sx = np.kron(np.ones((self.cfg.nbData, 1)), np.eye(self.cfg.nbVar, self.cfg.nbVar))

        M = B
        for i in range(1, self.cfg.nbData):
            Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :] = np.dot(
                Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :], A)
            Su[self.cfg.nbVar * i:self.cfg.nbVar * i + M.shape[0], 0:M.shape[1]] = M
            M = np.hstack((np.dot(A, M), B))
        return Su, Sx

    def get_u_x(self, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0 = self.start_point.reshape((self.cfg.nbVar, 1))
        u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (mu - Sx @ x0)
        x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, self.cfg.nbVar))
        return u_hat, x_hat

    def get_ik(self, pose_traj: List[List[float]]):
        joint_traj = []
        for pose in pose_traj:
             # 1. 目标末端姿态 → IK → 得到终点 joint 解
            solved = self.ik_controller.ik_solver.solve(self.initial_cfg, pose[:3], pose[3:])

            if solved is None:
                print("[LinearPlanner] IK 解失败，无法生成轨迹")
                return False
            joint_traj.append(solved)
            self.initial_cfg = solved

        return joint_traj