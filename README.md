# Alicia Duo SDK

Alicia Duo SDK 是一个用于控制 Alicia Duo 六轴机械臂（带夹爪）的 Python 工具包。它提供了通过串口与机械臂通信、控制关节运动、操作夹爪以及读取状态信息的功能。

## 主要特性

*   **关节控制**: 设置和读取六个关节的角度。
*   **夹爪控制**: 控制夹爪的开合角度。
*   **力矩控制**: 启用或禁用机械臂关节力矩，允许自由拖动或锁定位置。
*   **零点设置**: 将机械臂当前位置设置为新的零点。
*   **状态读取**: 获取关节角度、夹爪角度和按钮状态的实时数据。
*   **串口通信**: 自动查找或指定串口进行连接。
*   **数据解析**: 解析来自机械臂的反馈数据帧。

## 目录结构

```
Alicia_duo_sdk/
├── alicia_duo_sdk/         # SDK 核心代码
│   ├── __init__.py
│   ├── controller.py       # 机械臂控制逻辑
│   ├── data_parser.py      # 数据帧解析
│   └── serial_comm.py      # 串口通信
├── docs/                   # 文档
│   ├── api_reference.md
│   ├── examples.md
│   └── installation.md
├── examples/               # 示例代码
│   ├── arm_movement.py
│   ├── gripper_control.py
│   └── read_angles.py
├── LICENSE
├── README.md               # 本文档
├── requirements.txt        # 依赖项 (如果需要)
└── setup.py                # Python 包安装脚本 (如果需要)
```

## 快速开始

1.  **安装**: 请参照 [docs/installation.md](docs/installation.md) 进行安装和配置。
2.  **运行示例**:
    进入 `examples` 目录，尝试运行一个示例脚本，例如读取机械臂角度：
    ```sh
    cd examples
    python3 read_angles.py
    ```
    或者控制机械臂运动：
    ```sh
    python3 arm_movement.py
    ```

## 文档

*   [安装指南](docs/installation.md)
*   [示例说明](docs/examples.md)
*   [API 参考](docs/api_reference.md)

## 贡献

欢迎提交问题和拉取请求。

## 许可证

请查看 [LICENSE](LICENSE) 文件。