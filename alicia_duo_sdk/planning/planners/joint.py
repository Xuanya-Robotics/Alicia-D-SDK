from typing import List

class JointPlanner:
    def plan(
        self,
        start_angles: List[float],
        target_angles: List[float],
        steps: int
        ) -> List[List[float]]:
        """
        执行 cubic 插值，生成轨迹

        Returns:
            List[List[float]]: 插值后的关节轨迹
        """
        traj = []
        for step in range(1, steps + 1):
            ratio = self.ease_in_out_cubic(step / steps)
            interp = [
                s + (t - s) * ratio
                for s, t in zip(start_angles, target_angles)
            ]
            traj.append(interp)

        return traj

    @staticmethod
    def ease_in_out_cubic(x: float) -> float:
        """
        三次缓动函数，构造平滑的加减速曲线
        Args:
            x (float): 归一化时间（0~1）
        Returns:
            float: 插值系数
        """
        return 4 * x**3 if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2