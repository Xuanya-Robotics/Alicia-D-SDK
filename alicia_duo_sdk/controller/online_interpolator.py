import time
import threading
from typing import List, Optional

from ..driver import ArmController


class OnlineJointInterpolator:
    """
    Online trapezoidal-velocity interpolator for streaming sparse joint targets.

    This component runs a background loop at a fixed rate, slews current
    command towards the latest target with per-joint velocity and
    acceleration limits, and continuously sends commands to hardware.
    """

    def __init__(
        self,
        joint_controller: ArmController,
        command_rate_hz: float = 200.0,
        max_joint_velocity_rad_s: float = 2.5,
        max_joint_accel_rad_s2: float = 8.0,
        max_gripper_velocity_rad_s: float = 1.5,
        max_gripper_accel_rad_s2: float = 10.0,
    ) -> None:
        """
        :param joint_controller, ArmController: low-level joint controller
        :param command_rate_hz, float: control loop frequency
        :param max_joint_velocity_rad_s, float: joint velocity limit (rad/s)
        :param max_joint_accel_rad_s2, float: joint acceleration limit (rad/s^2)
        :param max_gripper_velocity_rad_s, float: gripper velocity limit (rad/s)
        :param max_gripper_accel_rad_s2, float: gripper acceleration limit (rad/s^2)
        :return: None
        """
        self._ctrl = joint_controller
        self._rate_hz = max(1.0, float(command_rate_hz))
        self._max_v = float(max_joint_velocity_rad_s)
        self._max_a = float(max_joint_accel_rad_s2)
        self._max_v_grip = float(max_gripper_velocity_rad_s)
        self._max_a_grip = float(max_gripper_accel_rad_s2)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State for interpolation
        self._cmd_q = [0.0] * 6
        self._cmd_dq = [0.0] * 6
        self._target_q = [0.0] * 6
        self._has_target = False

        self._cmd_grip = 0.0
        self._cmd_dgrip = 0.0
        self._target_grip = 0.0
        self._has_grip_target = False

        # Initialize command from current state if available
        try:
            cur = self._ctrl.get_joint_angles()
            if cur and len(cur) == 6:
                self._cmd_q = list(cur)
                self._target_q = list(cur)
        except Exception:
            pass

    def start(self) -> None:
        """
        :param None: 
        :return: None
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        :param None: 
        :return: None
        """
        if not self._running:
            return
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def set_target(self, joint_angles: List[float]) -> None:
        """
        :param joint_angles, List[float]: latest joint target (rad), len=6
        :return: None
        """
        if not isinstance(joint_angles, list) or len(joint_angles) != 6:
            raise ValueError("joint_angles must be a list of 6 floats (rad)")
        with self._lock:
            self._target_q = list(joint_angles)
            self._has_target = True

    def set_gripper_target_rad(self, angle_rad: float) -> None:
        """
        :param angle_rad, float: gripper target in radians (0..~1.745 rad for 100 deg)
        :return: None
        """
        with self._lock:
            self._target_grip = float(angle_rad)
            self._has_grip_target = True

    def update_params(
        self,
        command_rate_hz: Optional[float] = None,
        max_joint_velocity_rad_s: Optional[float] = None,
        max_joint_accel_rad_s2: Optional[float] = None,
        max_gripper_velocity_rad_s: Optional[float] = None,
        max_gripper_accel_rad_s2: Optional[float] = None,
    ) -> None:
        """
        :param command_rate_hz, float: optional new loop rate
        :param max_joint_velocity_rad_s, float: optional new joint vmax
        :param max_joint_accel_rad_s2, float: optional new joint amax
        :param max_gripper_velocity_rad_s, float: optional new gripper vmax
        :param max_gripper_accel_rad_s2, float: optional new gripper amax
        :return: None
        """
        with self._lock:
            if command_rate_hz is not None:
                self._rate_hz = max(1.0, float(command_rate_hz))
            if max_joint_velocity_rad_s is not None:
                self._max_v = float(max_joint_velocity_rad_s)
            if max_joint_accel_rad_s2 is not None:
                self._max_a = float(max_joint_accel_rad_s2)
            if max_gripper_velocity_rad_s is not None:
                self._max_v_grip = float(max_gripper_velocity_rad_s)
            if max_gripper_accel_rad_s2 is not None:
                self._max_a_grip = float(max_gripper_accel_rad_s2)

    def _loop(self) -> None:
        # Background thread sending interpolated commands
        last_time = time.perf_counter()
        while self._running:
            start = time.perf_counter()
            with self._lock:
                dt = 1.0 / self._rate_hz
                # Integrate joints
                if self._has_target:
                    for i in range(6):
                        pos = self._cmd_q[i]
                        vel = self._cmd_dq[i]
                        target = self._target_q[i]
                        error = target - pos
                        sign = 1.0 if error >= 0.0 else -1.0
                        v_max = self._max_v
                        a_max = self._max_a
                        brake_dist = (vel * vel) / (2.0 * max(1e-6, a_max))
                        dist = abs(error)
                        # Decide accel/decel following ROS-like policy
                        if brake_dist >= dist:
                            vel -= sign * a_max * dt * (1.0 if (vel * sign) > 0.0 else -1.0)
                        else:
                            vel += sign * a_max * dt
                        # Clamp velocity
                        if vel > v_max:
                            vel = v_max
                        if vel < -v_max:
                            vel = -v_max
                        new_pos = pos + vel * dt
                        # Snap to target if crossing
                        if (target - pos) * (target - new_pos) <= 0.0:
                            new_pos = target
                            vel = 0.0
                        self._cmd_q[i] = new_pos
                        self._cmd_dq[i] = vel

                # Integrate gripper
                if self._has_grip_target:
                    pos = self._cmd_grip
                    vel = self._cmd_dgrip
                    target = self._target_grip
                    error = target - pos
                    sign = 1.0 if error >= 0.0 else -1.0
                    v_max = self._max_v_grip
                    a_max = self._max_a_grip
                    brake_dist = (vel * vel) / (2.0 * max(1e-6, a_max))
                    dist = abs(error)
                    if brake_dist >= dist:
                        vel -= sign * a_max * dt * (1.0 if (vel * sign) > 0.0 else -1.0)
                    else:
                        vel += sign * a_max * dt
                    if vel > v_max:
                        vel = v_max
                    if vel < -v_max:
                        vel = -v_max
                    new_pos = pos + vel * dt
                    if (target - pos) * (target - new_pos) <= 0.0:
                        new_pos = target
                        vel = 0.0
                    self._cmd_grip = new_pos
                    self._cmd_dgrip = vel

                cmd_q = list(self._cmd_q)
                grip = float(self._cmd_grip)

            # Send commands out of lock to avoid blocking
            try:
                self._ctrl.set_joint_angles(cmd_q)
                if self._has_grip_target:
                    self._ctrl.set_gripper(grip)
            except Exception:
                pass

            # Sleep to maintain loop rate
            elapsed = time.perf_counter() - start
            dt_target = 1.0 / self._rate_hz
            to_sleep = dt_target - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)


