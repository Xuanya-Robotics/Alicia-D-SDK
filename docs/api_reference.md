# API 参考文档

本节介绍 Alicia-D SDK 的核心类与方法接口。

---

## ✅ 控制接口：`alicia_d_sdk.controller.control_api.SynriaControlAPI`

```python
from alicia_d_sdk.controller import SynriaControlAPI, get_default_session

session = get_default_session()
controller = SynriaControlAPI(session=session)
```

### 主要方法一览：

#### 运动控制：
- `moveJ(...)`  
  插值控制关节运动到目标位置（支持角度单位、速度调节、可视化）

- `moveCartesian(...)`  
  接收多个末端姿态点，使用 Cartesian 或 LQT 插值并通过 IK 执行轨迹

- `moveHome()`  
  控制机械臂返回默认初始位置（home_angles）

#### 状态获取：
- `get_joints()`  
  返回当前关节角度（弧度）

- `get_pose()`  
  获取当前末端执行器位置 + 姿态 `[x, y, z, qx, qy, qz, qw]`

- `get_gripper()`  
  返回当前夹爪角度（弧度）

- `print_state(continous, output_format)`  
  打印当前机械臂信息，可持续打印，支持角度/弧度格式

#### 操作控制：
- `gripper_control(command="open"/"close", angle_deg=xx)`  
  控制夹爪动作或自定义角度，并可设置是否等待到位

- `torque_control(command="on"/"off")`  
  启用或关闭扭矩（进入或退出示教模式）

- `zero_calibration()`  
  归零流程：关闭扭矩 → 拖动 → 重启扭矩 → 记录零点

---

## ✅ 控制上下文：`MotionSession`

构建 SDK 操作所需的会话上下文，包含模型、IK控制器、执行器和底层控制器。

```python
from alicia_d_sdk.controller import get_default_session
session = get_default_session()
```

成员变量：
- `ik_controller`: IKController
- `robot_model`: AliciaFollower
- `joint_controller`: ArmController
- `executor`: TrajectoryExecutor

---

## ✅ 底层接口：`driver.servo_driver.ArmController`

包含基础串口连接、数据读取、指令发送等方法，供 `SynriaControlAPI` 封装调用。

主要方法包括：
- `connect()` / `disconnect()`
- `get_joint_angles()` / `set_joint_angles(...)`
- `set_gripper(...)`
- `enable_torque()` / `disable_torque()`
- `set_zero_position()`
- `read_joint_state()` / `read_gripper_data()`

不推荐用户直接使用此类，建议通过 `SynriaControlAPI` 高级接口操作。

---

如需更多细节，请参考源码文档或查看 `logs/` 下日志文件输出。