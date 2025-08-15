from __future__ import annotations

"""Alicia robot adapter minimal skeleton.

This file provides a minimal, non-functional skeleton that implements the
IRobotAdapter interface. It will be wired to real Alicia-D-SDK control API
in later steps.
"""

import time
import os
from typing import Any, Dict, Iterable, List, Optional

from alicia_d_sdk.controller.adapter import IRobotAdapter


class AliciaAdapter(IRobotAdapter):
    """Adapter for Alicia robot wrapping the SDK APIs.

    The implementation is intentionally minimal at this phase.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._model_name: str = "Alicia_D_v5_4"
        self._session = None
        self._control_api = None
        self._connect_opts: Dict[str, Any] = {}

    def connect(self, config_path: str) -> None:
        """
        :param config_path, str: Path to YAML config
        :return: None
        """
        # Allow caller to pass options via environment variables before building session
        self._connect_opts = {
            "port": os.getenv("ALICIA_PORT", ""),
            "baudrate": os.getenv("ALICIA_BAUDRATE", "1000000"),
            "debug": os.getenv("ALICIA_DEBUG", "0") == "1",
        }
        # Lazy import to avoid heavy deps on module import
        from alicia_d_sdk.controller.session_factory import get_default_session  # type: ignore
        from alicia_d_sdk.controller.control_api import SynriaControlAPI  # type: ignore

        # Build default session (auto-connect ArmController) with optional comm params
        port = self._connect_opts.get("port") or os.getenv("ALICIA_PORT", "")
        baudrate = int(self._connect_opts.get("baudrate") or os.getenv("ALICIA_BAUDRATE", "1000000"))
        debug = bool(self._connect_opts.get("debug") or os.getenv("ALICIA_DEBUG", "0") == "1")

        # If a YAML config is provided, prefer its comm settings
        if config_path:
            try:
                import yaml  # type: ignore
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                comm = (cfg or {}).get("comm", {}) or {}
                port = comm.get("port", port)
                baudrate = int(comm.get("baudrate", baudrate))
                debug = bool(comm.get("debug", debug))
            except Exception:
                pass

        self._session = get_default_session(port=str(port or ""), baudrate=int(baudrate), debug=bool(debug))
        self._control_api = SynriaControlAPI(self._session)
        self._connected = True

    def disconnect(self) -> None:
        """
        :return: None
        """
        try:
            if self._control_api is not None:
                # Ensure any background interpolation is stopped
                try:
                    self._control_api.stopOnlineSmoothing()
                except Exception:
                    pass
            if self._session is not None and self._session.joint_controller is not None:
                try:
                    self._session.joint_controller.disconnect()
                except Exception:
                    pass
        finally:
            self._connected = False
            self._session = None
            self._control_api = None

    def get_capabilities(self) -> Dict[str, Any]:
        """
        :return: Dict[str, Any]
        """
        return {
            "control_modes": ["position", "cartesian"],
            "features": ["io", "gripper"],
            "dof": 6,
        }

    def get_model_name(self) -> str:
        """
        :return: str
        """
        return self._model_name

    def read_joint_state(self) -> Dict[str, Any]:
        """
        :return: Dict[str, Any]
        """
        if not self._connected or self._session is None:
            return {"q": [], "dq": [], "effort": [], "t": time.time()}
        try:
            q = self._session.joint_controller.get_joint_angles() or []
            return {"q": q, "dq": [0.0] * len(q), "effort": [0.0] * len(q), "t": time.time()}
        except Exception:
            return {"q": [], "dq": [], "effort": [], "t": time.time()}

    def stream_states(self) -> Iterable[Dict[str, Any]]:
        """
        :return: Iterable[Dict[str, Any]]
        """
        while self._connected:
            state = self.read_joint_state()
            yield state
            time.sleep(0.05)

    def command_joint_positions(
        self,
        q: List[float],
        duration: Optional[float] = None,
        blocking: bool = False,
    ) -> str:
        """
        :param q, List[float]: Target joint positions
        :param duration, float|None: Time to reach
        :param blocking, bool: Wait until done
        :return: str
        """
        if not self._connected or self._control_api is None:
            return "exec-not-connected"
        # Use online smoother for better arrival behavior
        try:
            self._control_api.moveJ_online(
                joint_format="rad",
                target_joints=q,
                command_rate_hz=200.0,
                max_joint_velocity_rad_s=2.5,
                max_joint_accel_rad_s2=8.0,
                stop_after=blocking,
            )
        except Exception:
            # Fallback to simple moveJ
            try:
                self._control_api.moveJ(joint_format="rad", target_joints=q, visualize=False)
            except Exception:
                pass
        return "exec-moveJ"

    def command_cartesian_pose(
        self,
        T_base_tool: List[List[float]],
        duration: Optional[float] = None,
        blocking: bool = False,
    ) -> str:
        """
        :param T_base_tool, List[List[float]]: 4x4 pose
        :param duration, float|None: Time to reach
        :param blocking, bool: Wait until done
        :return: str
        """
        # Not implemented yet; would need IK in SDK path; prefer Bridge planning.
        return "exec-cart-not-impl"

    def execute_trajectory(
        self,
        trajectory: Dict[str, Any],
        start_time: Optional[float] = None,
    ) -> str:
        """
        :param trajectory, Dict[str, Any]: Joint trajectory
        :param start_time, float|None: Start time
        :return: str
        """
        # Non-interactive execution loop to avoid blocking input
        try:
            points = trajectory.get("points", [])
            if not points:
                return "exec-traj-empty"
            dt = float(trajectory.get("dt", 0.01))
            ctrl = self._session.joint_controller if self._session is not None else None
            if ctrl is None:
                return "exec-traj-no-ctrl"
            if start_time is not None:
                now = time.time()
                if start_time > now:
                    time.sleep(start_time - now)
            for p in points:
                try:
                    ctrl.set_joint_angles(p)
                except Exception:
                    pass
                time.sleep(max(0.0, dt))
            return "exec-traj"
        except Exception:
            return "exec-traj-error"

    def stop(self) -> None:
        """
        :return: None
        """
        if self._control_api is not None:
            try:
                self._control_api.stopOnlineSmoothing()
            except Exception:
                pass
        return None

    def set_io(self, name: str, value: Any) -> None:
        """
        :param name, str: IO channel
        :param value, Any: Value to set
        :return: None
        """
        # Route gripper IO as example
        if self._session is not None and name.lower() == "gripper":
            try:
                angle = float(value)
                self._session.joint_controller.set_gripper(angle)
            except Exception:
                pass
        return None


