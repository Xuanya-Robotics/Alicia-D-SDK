# 安装指南

本指南将引导您完成 Alicia-D SDK 的安装与运行环境配置。

---

## ✅ 环境要求

- Python 3.6 及以上版本（推荐 Python 3.8）
- 支持串口的计算机（USB 转串口芯片已集成在机械臂）

---

## ✅ 安装步骤

### 1. 克隆或下载项目
```bash
git clone https://github.com/Xuanya-Robotics/Alicia-D-SDK.git
cd Alicia-D-SDK
```

### 2. 创建 Python 环境并安装依赖
使用 Conda 环境（推荐）：
```bash
conda create -n alicia_d_sdk python=3.8
conda activate alicia_d_sdk
```

安装依赖与 SDK：
```bash
pip install -r requirements.txt
pip install -e .
```


---

## ✅ 连接硬件

- 将机械臂通过 USB 连接到计算机
- 确保电源打开
- 系统应自动识别串口设备，如：
  - Linux: `/dev/ttyUSB0`, `/dev/ttyUSB1` ...
  - Windows: `COM3`, `COM4` ...

---

## ✅ 示例验证

执行以下命令测试连接和读取状态：
```bash
cd examples
python3 demo_read_state.py
```

若连接成功，终端将输出当前关节角度、末端位姿与夹爪状态。

---

## ⚠️ 故障排查

- **找不到串口/连接失败**
  - 检查 USB 线与电源
  - Linux 用户需确保在 `dialout` 用户组中：
    ```bash
    sudo usermod -a -G dialout $USER
    # 然后重新登录
    ```

- **手动指定串口**
  如自动连接失败，可手动指定：
  ```python
  session = get_default_session(port="/dev/ttyUSB0")
  ```

- **权限错误 (Permission denied)**
  - 可尝试以 sudo 运行或检查用户串口权限

---

安装成功后，即可通过 `SynriaControlAPI` 接口控制机械臂完成各种动作。