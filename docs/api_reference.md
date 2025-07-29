# API 参考

本文档提供了 Alicia Duo SDK 中主要类和方法的详细参考。

## 核心模块

SDK 的核心功能主要由以下模块提供：

*   [`alicia_duo_sdk.controller`](../alicia_duo_sdk/controller.py): 包含主要的 `ArmController` 类，用于与机械臂交互。
*   [`alicia_duo_sdk.data_parser`](../alicia_duo_sdk/data_parser.py): 包含 `DataParser` 类和 `JointState` 命名元组，用于解析和存储来自机械臂的数据。
*   [`alicia_duo_sdk.serial_comm`](../alicia_duo_sdk/serial_comm.py): 包含 `SerialComm` 类，处理底层串口通信。

对于大多数用户而言，主要交互将通过 `ArmController` 类进行。

## `alicia_duo_sdk.controller.ArmController`

此类是控制 Alicia Duo 机械臂的主要接口。

```python
from alicia_duo_sdk.controller import ArmController
```

### 常量

*   `ArmController.DEG_TO_RAD`: `float`
    将角度从度转换为弧度的系数 (`math.pi / 180.0`)。
*   `ArmController.RAD_TO_DEG`: `float`
    将角度从弧度转换为度的系数 (`180.0 / math.pi`)。

### 初始化

*   `__init__(self, port: str = "", baudrate: int = 921600, debug_mode: bool = False)`
    初始化机械臂控制器。
    *   **参数**:
        *   `port` (`str`, 可选): 串口名称 (例如, Linux 上的 `"/dev/ttyUSB0"` 或 Windows 上的 `"COM3"`)。如果留空，SDK 将尝试自动搜索可用串口。默认为 `""`。
        *   `baudrate` (`int`, 可选): 串口通信的波特率。默认为 `921600`。
        *   `debug_mode` (`bool`, 可选): 是否启用调试模式。启用后，将输出更详细的日志信息，包括发送和接收的数据帧。默认为 `False`。

### 连接与断开

*   `connect(self) -> bool`
    连接到机械臂。如果 `port` 未在初始化时指定，则会尝试自动查找。
    *   **返回**: `bool` - 如果连接成功则为 `True`，否则为 `False`。

*   `disconnect(self)`
    断开与机械臂的连接并关闭串口。

### 读取数据

*   `get_joint_angles(self) -> Optional[List[float]]`
    读取机械臂六个关节的当前角度。
    此方法会尝试读取新的数据帧，如果成功且为关节数据，则解析并返回最新角度。如果未能读取到新的关节数据，则返回上一次已知的关节角度。
    *   **返回**: `Optional[List[float]]` - 包含六个关节角度（单位：弧度）的列表。如果无法获取状态（例如，在初始连接且未收到任何数据之前），理论上可能返回基于内部默认值的状态，但通常在连接后很快就会有实际数据。

*   `read_gripper_data(self) -> Tuple[float, bool, bool]`
    读取夹爪的当前状态，包括角度和两个按钮的状态。
    此方法会尝试读取新的数据帧，如果成功且为夹爪数据，则解析并返回最新状态。如果未能读取到新的夹爪数据，则返回上一次已知的夹爪状态。
    *   **返回**: `Tuple[float, bool, bool]` - 一个元组，包含：
        *   夹爪角度（单位：弧度）。
        *   按钮1状态 (`True` 表示按下, `False` 表示释放)。
        *   按钮2状态 (`True` 表示按下, `False` 表示释放)。

*   `read_joint_state(self) -> JointState`
    读取完整的机械臂状态，包括所有关节角度、夹爪角度和按钮状态。
    此方法会尝试读取并解析最新的数据帧（无论是关节数据还是夹爪数据），并更新内部状态。
    *   **返回**: [`JointState`](#alicia_duo_sdkdata_parserjointstate) - 一个包含当前机械臂完整状态的对象。

### 设置与控制

*   `set_joint_angles(self, joint_angles: List[float], gripper_angle: float = None, wait_for_completion: bool = True, timeout: float = 10.0, tolerance: float = 0.08) -> bool`
    设置机械臂六个关节的目标角度。
    *   **参数**:
        *   `joint_angles` (`List[float]`): 包含六个目标关节角度（单位：弧度）的列表。列表长度必须为6。
        *   `gripper_angle` (`float`, 可选): 夹爪的目标角度（单位：弧度）。如果提供此参数，则在设置关节角度后会接着发送夹爪控制命令。默认为 `None` (不控制夹爪)。
        *   `wait_for_completion` (`bool`, 可选): 是否等待运动完成后再返回。默认为 `True`。
        *   `timeout` (`float`, 可选): 等待运动完成的最大时间（单位：秒）。默认为 `10.0`。
        *   `tolerance` (`float`, 可选): 判断运动是否完成的角度误差容忍度（单位：弧度）。默认为 `0.08`。
    *   **返回**: `bool` - 如果命令成功发送并执行（如果等待完成）则为 `True`，否则为 `False`。

*   `set_gripper(self, angle_rad: float) -> bool`
    设置夹爪的开合角度。
    *   **参数**:
        *   `angle_rad` (`float`): 夹爪的目标角度（单位：弧度）。通常 0 表示完全张开，某个正值（例如 `100 * DEG_TO_RAD`）表示完全闭合，具体范围取决于夹爪硬件。
    *   **返回**: `bool` - 如果命令成功发送则为 `True`，否则为 `False`。

*   `set_zero_position(self) -> bool`
    将机械臂当前的姿态设置为新的零点位置。
    *   **返回**: `bool` - 如果命令成功发送则为 `True`，否则为 `False`。

*   `enable_torque(self) -> bool`
    使能所有关节的力矩。机械臂将尝试保持当前位置，抵抗外力。
    *   **返回**: `bool` - 如果命令成功发送则为 `True`，否则为 `False`。

*   `disable_torque(self) -> bool`
    禁用所有关节的力矩。机械臂关节将可以被自由拖动（进入“示教模式”或“自由驱动模式”）。
    *   **返回**: `bool` - 如果命令成功发送则为 `True`，否则为 `False`。

### 运动学函数

*   `forward_kinematics_alicia_duo(self, joint_angles_rad: List[float]) -> Tuple[List[float], List[float], List[float]]`
    计算机械臂的正向运动学，根据关节角度计算末端执行器的位置和姿态。
    *   **参数**:
        *   `joint_angles_rad` (`List[float]`): 包含六个关节角度（单位：弧度）的列表。
    *   **返回**: `Tuple[List[float], List[float], List[float]]` - 一个包含以下元素的元组：
        *   `position_xyz` (`List[float]`): 末端执行器的位置 [x, y, z]。
        *   `quaternion_xyzw` (`List[float]`): 末端执行器的姿态，四元数表示 [x, y, z, w]。
        *   `rpy` (`List[float]`): 末端执行器的姿态，欧拉角表示 [roll, pitch, yaw]。

*   `inverse_kinematics_alicia_duo(self, target_position_xyz: List[float], target_quaternion_xyzw: List[float], initial_joint_angles_rad: Optional[List[float]] = None) -> Optional[List[float]]`
    计算机械臂的逆向运动学，根据末端执行器的目标位置和姿态计算所需的关节角度。
    *   **参数**:
        *   `target_position_xyz` (`List[float]`): 末端执行器的目标位置 [x, y, z]。
        *   `target_quaternion_xyzw` (`List[float]`): 末端执行器的目标姿态，四元数表示 [x, y, z, w]。
        *   `initial_joint_angles_rad` (`List[float]`, 可选): 初始猜测的关节角度（单位：弧度），用于迭代求解。如果不提供，默认使用零位姿态。
    *   **返回**: `Optional[List[float]]` - 如果成功求解，返回包含六个关节角度（单位：弧度）的列表；如果求解失败，返回 `None`。

## `alicia_duo_sdk.data_parser.JointState`

这是一个 `NamedTuple`，用于封装机械臂的完整状态信息。

```python
from alicia_duo_sdk.data_parser import JointState
```

### 属性

*   `angles`: `List[float]`
    一个包含六个关节角度（单位：弧度）的列表。
*   `gripper`: `float`
    夹爪的当前角度（单位：弧度）。
*   `timestamp`: `float`
    状态数据最后更新的时间戳（`time.time()` 的结果，单位：秒）。
*   `button1`: `bool`
    夹爪上按钮1的状态 (`True` 表示按下, `False` 表示释放)。
*   `button2`: `bool`
    夹爪上按钮2的状态 (`True` 表示按下, `False` 表示释放)。

## 内部模块 (简述)

### `alicia_duo_sdk.data_parser.DataParser`

*   此类负责解析从串口接收到的原始字节数据帧，将其转换为结构化的信息，如关节角度、夹爪状态等，并更新 `JointState`。
*   主要方法:
    *   `parse_frame(self, frame: List[int]) -> Optional[Dict]`: 解析单个数据帧。
    *   `get_joint_state(self) -> JointState`: 获取当前解析的最新状态。

### `alicia_duo_sdk.serial_comm.SerialComm`

*   此类封装了与串口设备进行通信的底层逻辑。
*   主要方法:
    *   `connect(self) -> bool`: 打开并配置串口连接。
    *   `disconnect(self)`: 关闭串口连接。
    *   `send_data(self, data: List[int]) -> bool`: 将字节列表发送到串口。
    *   `read_frame(self) -> Optional[List[int]]`: 从串口读取一个完整的数据帧。
    *   `find_serial_port(self) -> str`: 自动查找可用的串口设备。

用户通常不需要直接与 `DataParser` 或 `SerialComm` 类交互，因为 `ArmController` 已经处理了这些细节。