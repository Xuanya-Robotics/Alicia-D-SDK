"""
Alicia Duo SDK
==============

此包提供了与 Alicia Duo 机械臂交互的工具。
主要通过 `ArmController` 类进行控制。
"""

__version__ = "0.1.0"
__author__ = "Your Name/Organization" # 请替换为您的名称或组织

from .controller import ArmController
from .data_parser import JointState

__all__ = [
    "ArmController",
    "JointState"
]