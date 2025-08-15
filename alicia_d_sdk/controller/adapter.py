from __future__ import annotations

"""Unified robot adapter interface for hardware/simulation bridging.

All comments are in English. Docstrings follow rst style and only keep
param/return sections per project convention.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


class IRobotAdapter(ABC):
    """Robot adapter interface that standardizes robot control surface.

    The adapter is responsible for connection lifecycle, state streaming,
    and executing high-level commands in a robot-agnostic way.
    """

    @abstractmethod
    def connect(self, config_path: str) -> None:
        """
        :param config_path, str: Path to YAML config for robot mapping and limits
        :return: None
        """

    @abstractmethod
    def disconnect(self) -> None:
        """
        :return: None
        """

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        :return: Dict[str, Any]
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """
        :return: str
        """

    @abstractmethod
    def read_joint_state(self) -> Dict[str, Any]:
        """
        :return: Dict[str, Any]
        """

    @abstractmethod
    def stream_states(self) -> Iterable[Dict[str, Any]]:
        """
        :return: Iterable[Dict[str, Any]]
        """

    @abstractmethod
    def command_joint_positions(
        self,
        q: List[float],
        duration: Optional[float] = None,
        blocking: bool = False,
    ) -> str:
        """
        :param q, List[float]: Target joint positions (rad)
        :param duration, float|None: Optional time to reach (s)
        :param blocking, bool: Wait until done
        :return: str
        """

    @abstractmethod
    def command_cartesian_pose(
        self,
        T_base_tool: List[List[float]],
        duration: Optional[float] = None,
        blocking: bool = False,
    ) -> str:
        """
        :param T_base_tool, List[List[float]]: 4x4 homogeneous pose
        :param duration, float|None: Optional time to reach (s)
        :param blocking, bool: Wait until done
        :return: str
        """

    @abstractmethod
    def execute_trajectory(
        self,
        trajectory: Dict[str, Any],
        start_time: Optional[float] = None,
    ) -> str:
        """
        :param trajectory, Dict[str, Any]: Time-parameterized joint trajectory
        :param start_time, float|None: Optional start timestamp
        :return: str
        """

    @abstractmethod
    def stop(self) -> None:
        """
        :return: None
        """

    @abstractmethod
    def set_io(self, name: str, value: Any) -> None:
        """
        :param name, str: IO channel
        :param value, Any: Value to set
        :return: None
        """


