# 示例代码说明

`examples` 目录包含了一些演示如何使用 Alicia Duo SDK 与机械臂交互的 Python 脚本。

## 如何运行示例

1.  确保您已经按照 [安装指南](installation.md) 完成了 SDK 的安装和硬件连接。
2.  打开终端或命令提示符。
3.  导航到 `Alicia-D-SDK/examples/` 目录。
    ```bash
    cd path/to/Alicia-D-SDK/examples
    ```
4.  使用 Python 3 运行所需的示例脚本：
    ```bash
    python3 <example_script_name>.py
    ```
    例如，要运行 `read_angles.py`：
    ```bash
    python3 read_angles.py
    ```

## 示例列表

### 1. `read_angles.py`

*   **文件**: [`examples/read_angles.py`](../examples/read_angles.py)
*   **描述**: 此示例连接到机械臂并持续读取和显示所有六个关节的角度（单位：度）、夹爪的开合角度（单位：度）以及夹爪上两个按钮的状态。它还演示了如何启用调试模式以获取更详细的日志输出。
*   **主要功能演示**:
    *   初始化 [`ArmController`](../alicia_duo_sdk/controller.py#L13) (可带 `debug_mode=True`)。
    *   连接到机械臂 ([`controller.connect()`](../alicia_duo_sdk/controller.py#L55))。
    *   循环读取完整的关节状态 ([`controller.read_joint_state()`](../alicia_duo_sdk/controller.py#L103))。
    *   从 [`JointState`](../alicia_duo_sdk/data_parser.py#L11) 对象中提取角度和按钮信息。
    *   将弧度转换为度 ([`controller.RAD_TO_DEG`](../alicia_duo_sdk/controller.py#L17))。
    *   优雅地处理 `KeyboardInterrupt` (Ctrl+C) 以断开连接。
    *   断开与机械臂的连接 ([`controller.disconnect()`](../alicia_duo_sdk/controller.py#L63))。

### 2. `arm_movement.py`

*   **文件**: [`examples/arm_movement.py`](../examples/arm_movement.py)
*   **描述**: 此示例演示了如何控制机械臂进行多种类型的运动，包括将所有关节移动到零位、逐个关节测试、模拟波浪运动、力矩控制（启用/禁用）、设置当前位置为零点，以及一个完整的模拟抓取和放置过程。
*   **主要功能演示**:
    *   读取初始关节状态。
    *   设置所有关节到指定角度 ([`controller.set_joint_angles()`](../alicia_duo_sdk/controller.py#L115))。
    *   角度单位转换 ([`controller.DEG_TO_RAD`](../alicia_duo_sdk/controller.py#L16), [`controller.RAD_TO_DEG`](../alicia_duo_sdk/controller.py#L17))。
    *   禁用和启用关节力矩 ([`controller.disable_torque()`](../alicia_duo_sdk/controller.py#L183), [`controller.enable_torque()`](../alicia_duo_sdk/controller.py#L173))。
    *   设置当前位置为零点 ([`controller.set_zero_position()`](../alicia_duo_sdk/controller.py#L163))。
    *   控制夹爪开合 ([`controller.set_gripper()`](../alicia_duo_sdk/controller.py#L150))。
    *   使用 `time.sleep()` 在动作之间添加延迟。

### 3. `gripper_control.py`

*   **文件**: [`examples/gripper_control.py`](../examples/gripper_control.py)
*   **描述**: 此示例专注于控制机械臂的夹爪。它演示了如何将夹爪设置到完全打开、半闭合和完全闭合的位置，并读取夹爪的实际角度和按钮状态。此外，它还提供了一个交互模式，允许用户通过键盘输入来控制夹爪的角度。
*   **主要功能演示**:
    *   设置夹爪角度 ([`controller.set_gripper()`](../alicia_duo_sdk/controller.py#L150))。
    *   读取夹爪数据（角度和按钮状态）([`controller.read_gripper_data()`](../alicia_duo_sdk/controller.py#L83))。
    *   交互式用户输入处理。
    *   在程序结束前将夹爪复位到打开状态。

这些示例是学习如何使用 Alicia Duo SDK 的良好起点。您可以修改它们或将相关代码集成到您自己的应用程序中。
