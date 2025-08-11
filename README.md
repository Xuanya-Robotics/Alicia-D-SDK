# Alicia-D SDK  

Alicia-D SDK 是一个用于控制【灵动 Alicia-D】系列六轴机械臂（带夹爪）的 Python 工具包。它提供通过串口通信控制机械臂运动、操作夹爪、读取姿态与状态数据等功能。

## 主要特性

*   **关节控制**：支持设置与读取六个关节的角度，提供平滑插值执行。
*   **末端轨迹**：基于 Cartesian 或 LQT 插值执行末端姿态轨迹。
*   **夹爪控制**：支持精确角度控制或一键开关。
*   **力矩控制**：开启或关闭关节电机扭矩，实现自由拖动（示教）。
*   **零点设置**：将当前位置设置为新的零点。
*   **状态读取**：实时获取关节角、夹爪角与末端姿态。
*   **自动串口连接**：自动搜索串口或手动指定。
*   **教学模式**：拖动记录姿态点并执行轨迹。

## 项目结构

```
alicia_d_sdk/
├── config/ 
├── controller/     # 上层控制 API，统一封装运动指令和会话管理
├── driver/         # 底层串口通信与数据解析模块
├── execution/      # 轨迹执行器，支持插值控制和可视化
├── kinematics/     # 运动学求解器（正向 / 逆向）
├── planning/       # 插值规划模块，支持 LQT / 线性等方式
├── utils/          # 辅助工具，如日志记录、验证器等
docs/
├── api_reference.md
├── examples.md
├── installation.md
examples/
├── demo_gripper.py
├── demo_moveCartesian.py
├── demo_moveJ.py
├── demo_read_state.py
├── demo_torque_control.py
├── demo_zero_calibration.py
logs/
setup.py
requirements.txt
README.md
```

## 快速开始

1.  安装：请参见 [安装指南](docs/installation.md)
2.  运行示例：
```bash
cd examples
python3 demo_read_state.py      # 读取状态
python3 demo_moveJ.py           # 关节移动
python3 demo_gripper.py         # 夹爪控制
```

## 文档

*   [安装指南](docs/installation.md)
*   [示例说明](docs/examples.md)
*   [API 参考](docs/api_reference.md)