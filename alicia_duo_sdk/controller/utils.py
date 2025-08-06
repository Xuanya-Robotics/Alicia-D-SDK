
def compute_steps_and_delay(
    speed_factor: float,
    T_default: float = 3.6,
    n_steps_ref: int = 120,
    min_steps: int = 50,
    max_steps: int = 500,
    min_delay: float = 0.01,
    max_delay: float = 0.1
):
    """
    根据速度因子同时调整插值步数和每步延迟，使得总时间约为 T_default / speed_factor。

    Returns:
        n_steps (int): 插值步数
        step_delay (float): 每步延迟（秒）
    """
    if speed_factor <= 0:
        raise ValueError("速度因子必须大于 0")

    # 插值步数反比于速度
    raw_steps = int(n_steps_ref / speed_factor)
    n_steps = max(min_steps, min(raw_steps, max_steps))

    # 根据实际步数计算单步时间
    total_time = T_default / speed_factor
    raw_delay = total_time / n_steps
    step_delay = max(min_delay, min(raw_delay, max_delay))

    return n_steps, step_delay

def ease_in_out_cubic(x: float) -> float:
    """
    三次缓动函数，构造平滑的加减速曲线
    Args:
        x (float): 归一化时间（0~1）
    Returns:
        float: 插值系数
    """
    return 4 * x**3 if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2