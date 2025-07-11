# 安装指南

本指南将引导您完成 Alicia Duo SDK 的安装和配置过程。

## 先决条件

*   Python 3.6 或更高版本。
*   `pyserial`库：用于串口通信。

## 安装步骤

1.  **克隆或下载 SDK**
    如果您拥有 Git，可以克隆此仓库：
    ```bash
    git clone https://github.com/Xuanya-Robotics/Alicia-D-SDK.git
    cd Alicia-D-SDK
    ```
    或者，您可以下载 SDK 的压缩包并解压到您的本地计算机。

2.  **安装依赖项**
    创建conda环境（必须）：
    ```bash
    conda create -n alicia_duo_sdk python=3.8
    conda activate alicia_duo_sdk
    ```
    安装库以及依赖：
    ```bash
    pip install -r requirements.txt

    conda install -c conda-forge python-orocos-kdl

    pip install -e .
    ```


3.  **硬件连接**
    *   使用 USB 数据线将 Alicia Duo 机械臂连接到您的计算机。
    *   确保机械臂已通电。
    *   操作系统通常会自动识别 USB 转串口设备。在 Linux 系统上，设备通常显示为 `/dev/ttyUSB0`、`/dev/ttyUSB1` 等。在 Windows 上，它会显示为 `COMx` (例如 `COM3`)。

## 验证安装

连接好机械臂后，您可以尝试运行一个示例程序来验证 SDK 是否工作正常。

1.  导航到 `examples` 目录：
    ```bash
    cd examples
    ```

2.  运行 `read_angles.py` 示例：
    ```bash
    python3 read_angles.py
    ```
    如果一切配置正确，您应该能在终端看到持续输出的机械臂关节角度、夹爪角度和按钮状态。按 `Ctrl+C` 退出程序。

## 故障排除

*   **无法连接到机械臂 / 未找到可用串口**:
    *   检查 USB 线是否牢固连接。
    *   确保机械臂已上电。
    *   确认您的用户有权限访问串口设备 (在 Linux 上，可能需要将用户添加到 `dialout` 组，例如 `sudo usermod -a -G dialout $USER`，然后重新登录)。
    *   如果 SDK 无法自动找到端口，您可以在初始化 `ArmController` 时手动指定端口，例如：
        ```python
        # filepath: examples/your_script.py
        # ...existing code...
        # controller = ArmController() # 自动搜索
        controller = ArmController(port="/dev/ttyUSB0") # Linux 示例
        # controller = ArmController(port="COM3") # Windows 示例
        # ...existing code...
        ```
*   **权限错误 (Permission denied)**:
    *   在 Linux 上，尝试使用 `sudo python3 your_script.py` 运行，或者参照上述方法将用户添加到 `dialout` 组。

如果您遇到其他问题，请检查 SDK 日志输出或在项目的 Issue 跟踪系统中报告问题。
