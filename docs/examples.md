# 示例代码说明

`examples/` 目录包含了多个演示脚本，用于展示如何使用 Alicia-D SDK 控制机械臂。

---

## ✅ 如何运行

1. 安装 SDK 并连接好机械臂
2. 进入示例目录：
```bash
cd examples
```
3. 使用 Python3 运行任意示例：
```bash
python3 demo_read_state.py
```

---

## 📜 示例列表

### 1. `demo_moveJ.py`
使用 `moveJ` 接口控制机械臂移动到设定角度，并执行前后归位。

### 2. `demo_moveCartesian.py`
教学模式：记录末端姿态轨迹（通过拖动），然后使用 Cartesian 或 LQT 插值轨迹回放。

### 3. `demo_gripper.py`
夹爪控制：演示开/关与角度设置三种模式。

### 4. `demo_read_state.py`
打印机械臂状态：关节角度、末端位置、姿态和夹爪角度。支持持续输出。

### 5. `demo_torque_control.py`
演示如何关闭/开启扭矩，进入或退出示教模式。

### 6. `demo_zero_calibration.py`
执行归零流程：关闭扭矩 → 拖动 → 开启 → 设置零点。

---

这些脚本可作为功能测试或二次开发的起点。你也可以结合多个模块编写更复杂的应用逻辑。