# 例程代码说明

`examples/` 目录包含了多个演示脚本，用于展示如何使用 Alicia-D SDK 控制机械臂。

---

## 📁 文件结构

```
examples/
├── 00_demo_read_version.py          # 读取版本号
├── 01_demo_zero_calibration.py      # 归零校准
├── 02_demo_torque_control.py        # 扭矩控制
├── 03_demo_gripper.py               # 夹爪控制
├── 04_demo_read_state.py            # 读取状态
├── 05_demo_moveJ.py                 # 关节空间运动
├── 06_demo_moveCartesian.py         # 笛卡尔空间运动
├── 07_demo_record_motion.py         # 轨迹录制
├── 08_demo_recorded_motion_replay.py # 轨迹回放
├── 09_demo_sparkvis.py              # SparkVis集成
├── example_motions/                  # 录制的动作数据
│   ├── startup/                      # 启动动作
│   └── sleep/                        # 休眠动作
└── README.md                         # 功能说明
```

## 📜 例程列表

### 0. `00_demo_read_version.py`
**请在进行使用前打印机械臂版本号**
- 如果有版本号输出，且为5.4.19及以上，则使用默认波特率1000000，否则使用默认波特率921600（请多次尝试，只要看到有版本号输出即可）
- 如果显示超时或者多次尝试后没有版本号输出，则使用波特率921600

### 1. `01_demo_zero_calibration.py`
执行归零流程：关闭扭矩 → 拖动 → 开启 → 设置零点。

**适用场景：**
- 机械臂首次使用或长时间未使用后
- 关节角度出现偏差时
- 需要重新建立零点参考时

### 2. `02_demo_torque_control.py`
演示如何关闭/开启扭矩，用于进入或退出拖动示教模式。

**参数说明：**
- `command='off'`：关闭扭矩，进入拖动示教模式
- `command='on'`：开启扭矩，退出示教模式，恢复正常控制

**注意事项：**
- 关闭扭矩后机械臂可以手动拖动，但要注意安全防止机械臂砸落损坏
- 示教完成后务必重新开启扭矩

### 3. `03_demo_gripper.py`
夹爪控制：演示开/关与角度设置三种模式。

**参数说明：**
- `command='open'`：完全打开夹爪
- `command='close'`：完全关闭夹爪
- `angle_deg=45`：设置夹爪到指定角度（度）

**调整建议：**
- 根据夹持物体大小调整角度
- 不同材质的物体可能需要不同的夹持力度

### 4. `04_demo_read_state.py`
打印机械臂状态：关节角度、末端位置、姿态和夹爪角度。支持持续输出。

**参数说明：**
- `continuous=True`：持续输出状态信息
- `output_format='deg'`：输出角度单位（'deg' 或 'rad'）

**使用场景：**
- 调试和故障排查
- 实时监控机械臂状态
- 验证控制指令执行效果

### 5. `05_demo_moveJ.py`
使用 `moveJ` 接口控制机械臂移动到设定角度，并执行前后归位。

**参数说明：**
- `joint_format='deg'`：输入角度单位（'deg' 或 'rad'）
- `target_joints`：目标关节角度列表
- `speed_factor=1.0`：速度倍率（>1更快，<1更慢）
- `T_default=4.0`：默认插值总时长（秒）
- `n_steps_ref=200`：参考插值步数

**调整建议：**
- 根据运动距离调整 `T_default`：短距离用2-3秒，长距离用5-8秒
- 精细操作时降低 `speed_factor` 到0.5-0.8
- 快速运动时可提高 `speed_factor` 到1.2-1.5

### 6. `06_demo_moveCartesian.py`
教学模式：记录末端姿态轨迹（通过拖动），然后使用 Cartesian 或 LQT 插值轨迹回放。

**参数说明：**
- `planner_name='cartesian'`：规划器类型（'cartesian' 或 'lqt'）
- `move_time=3.0`：预计执行时长（秒）
- `reverse=True`：是否反向执行轨迹
- `visualize=True`：是否可视化轨迹
- `show_ori=True`：是否显示姿态

**调整建议：**
- `cartesian` 规划器：适合直线路径，运动更直接
- `lqt` 规划器：适合复杂路径，运动更平滑
- 根据路径复杂度调整 `move_time`：简单路径2-3秒，复杂路径5-10秒

### 7. `07_demo_record_motion.py`
**轨迹录制脚本**：Teaching模式下记录末端姿态轨迹并缓存关节轨迹
- 手动示教记录多个waypoint（末端位姿 + 夹爪状态）
- 执行一次moveCartesian运动
- 后台定频采样关节角并保存为JSON文件
- 生成三个文件：`waypoints.json`、`joint_traj.json`、`meta.json`
    
**使用方式：**
```bash
python 07_demo_record_motion.py --port COM6 --motion [your_motion] --time 10.0 --planner cartesian
```

**参数详解：**
- `--port COM6`：串口连接端口
- `--motion [your_motion]`：动作名称，将创建 `motions/[your_motion]/` 文件夹
- `--time 10.0`：moveCartesian执行时长（秒）
- `--planner cartesian`：规划器类型（cartesian 或 lqt）
- `--sample-hz 100.0`：关节轨迹采样频率（Hz）
- `--overwrite`：覆盖已存在的动作文件夹
- `--visualize`：是否可视化轨迹
- `--show-ori`：是否显示姿态

**调整建议：**
- `--sample-hz`：录制精细动作时提高到200Hz，一般动作100Hz足够
- `--time`：根据路径复杂度调整，简单路径5-8秒，复杂路径10-15秒
- `--planner`：直线路径用cartesian，曲线路径用lqt

> [your_motion] 为录制时用于保存的动作文件夹名

### 8. `08_demo_recorded_motion_replay.py`
**轨迹回放脚本**：读取录制的关节轨迹文件并回放
- 自适应裁剪开头和结尾的静止部分
- 30Hz重采样，使用SDK内置在线插值（Online Smoothing）平滑下发目标
- 智能裁剪：使用自适应噪声阈值和相对位移阈值
- 支持倍速播放和循环播放

**使用方式：**
```bash
python 08_demo_recorded_motion_replay.py --motion [your_motion] --repeat 3 --speed 1.5
```

**参数详解：**
- `--port COM6`：串口连接端口
- `--motion [your_motion]`：要回放的动作名称
- `--repeat 1`：重复播放次数
- `--speed 1.0`：播放速度倍率（>1更快，<1更慢）

**内部参数调整（代码中修改）：**
```python
# 关键帧频率设置
SEG_HZ = 30.0             # 设置目标更新频率（20~40 推荐）
ONLINE_CMD_HZ = 200.0     # 在线插值后台命令频率（建议 150~300）

# 智能裁剪参数
BASE_DIFF_EPS = 0.010     # 基础关节差分阈值下限（rad）
DIST_EPS = 0.06           # 相对位移阈值（rad）
HEAD_NOISE_SECS = 5.0     # 噪声估计时间窗口（秒）
WIN_SECS = 0.40           # 运动判定窗口长度（秒）

# 裁剪范围限制
MAX_HEAD_CHECK = 1800.0   # 头部最大检查时间（秒）
MAX_TAIL_CHECK = 900.0    # 尾部最大检查时间（秒）
```

**调整建议：**
- `SEG_HZ`：精细动作提高到40-50Hz，快速动作降低到20-25Hz
- `ONLINE_CMD_HZ`：根据机械臂性能调整，高性能机械臂可提高到300Hz
- `BASE_DIFF_EPS`：根据机械臂精度调整，高精度机械臂可降低到0.005
- `DIST_EPS`：根据动作幅度调整，小幅度动作降低到0.03-0.04
- `WIN_SECS`：根据动作连续性调整，连续动作可降低到0.2-0.3秒

> [your_motion] 为录制时指定的动作文件夹名，我们在 example_motions 目录下提供了两个动作：startup 和 sleep，你可以使用这两个动作尝试轨迹回放。


### 9. `09_demo_sparkvis.py`
**SparkVis UI集成**：与SparkVis UI的双向同步与数据写入
- 启动WebSocket服务器，支持UI ↔ 机器人双向同步
- UI → 机器人：接收joint_update并直接下发到真实机器人
- 机器人 → UI：周期广播当前机器人状态到UI
- 数据写入：将UI下发的关节数据记录为CSV
- 内置在线插值，适配稀疏UI指令

**使用方式：**
```bash
python 09_demo_sparkvis.py --host localhost --port 8765
```

**参数详解：**
- `--host localhost`：WebSocket服务器主机地址
- `--port 8765`：WebSocket服务器端口
- `--robot-sync-rate 50.0`：机器人状态同步频率（Hz）
- `--log-source ui`：日志来源（ui/robot/both）

**内部参数调整：**
```python
# 在线插值参数
command_rate_hz=200.0,           # 命令下发频率（Hz）
max_joint_velocity_rad_s=2.5,    # 最大关节速度（rad/s）
max_joint_accel_rad_s2=8.0,      # 最大关节加速度（rad/s²）
max_gripper_velocity_rad_s=1.5,  # 最大夹爪速度（rad/s）
max_gripper_accel_rad_s2=10.0,   # 最大夹爪加速度（rad/s²）

# WebSocket参数
robot_sync_rate_hz=50.0,         # 机器人状态同步频率（Hz）
```

**调整建议：**
- `command_rate_hz`：根据网络延迟调整，高延迟环境降低到100-150Hz
- `max_joint_velocity_rad_s`：根据机械臂性能和安全要求调整
- `robot_sync_rate_hz`：根据UI刷新需求调整，实时显示用50Hz，一般显示用20-30Hz

---

## 🔄 轨迹录制与回放工作流程

### 录制阶段（07_demo_record_motion.py）：
1. 关闭扭矩，手动拖动机械臂
2. 按回车记录waypoint
3. 执行moveCartesian运动
4. 后台采样并保存关节轨迹

### 回放阶段（08_demo_recorded_motion_replay.py）：
1. 读取joint_traj.json文件
2. 智能裁剪静止部分
3. 重采样到30Hz
4. 使用SDK内置在线插值平滑执行

---

## ⚙️ 常见参数调整场景

### 运动控制参数调整：
- **速度过快**：降低 `speed_factor` 或 `max_joint_velocity_rad_s`
- **运动不平滑**：提高 `command_rate_hz` 或 `SEG_HZ`
- **精度不够**：提高采样频率，降低差分阈值
- **响应过慢**：提高 `ONLINE_CMD_HZ`，降低 `WIN_SECS`

### 轨迹质量参数调整：
- **裁剪过度**：提高 `BASE_DIFF_EPS` 和 `DIST_EPS`
- **裁剪不足**：降低 `BASE_DIFF_EPS` 和 `DIST_EPS`
- **噪声敏感**：增加 `HEAD_NOISE_SECS` 和 `WIN_SECS`
- **实时性要求**：降低 `SEG_HZ` 和 `robot_sync_rate_hz`

---

## 🔧 参数调优实战指南

### 1. 轨迹录制优化（07_demo_record_motion.py）
```bash
# 精细动作录制（如精密装配）
python 07_demo_record_motion.py --motion precise_assembly --sample-hz 200 --time 15.0 --planner lqt

# 快速动作录制（如搬运）
python 07_demo_record_motion.py --motion fast_pick --sample-hz 100 --time 8.0 --planner cartesian

# 复杂路径录制（如曲线轨迹）
python 07_demo_record_motion.py --motion curve_path --sample-hz 150 --time 12.0 --planner lqt
```

### 2. 轨迹回放优化（08_demo_recorded_motion_replay.py）
**修改代码中的关键参数：**

```python
# 高精度回放（精密操作）
SEG_HZ = 50.0              # 提高关键帧频率
ONLINE_CMD_HZ = 300.0      # 提高在线插值频率
BASE_DIFF_EPS = 0.005      # 降低差分阈值
DIST_EPS = 0.03            # 降低位移阈值

# 快速回放（生产环境）
SEG_HZ = 20.0              # 降低关键帧频率
ONLINE_CMD_HZ = 150.0      # 降低在线插值频率
BASE_DIFF_EPS = 0.015      # 提高差分阈值
DIST_EPS = 0.08            # 提高位移阈值

# 平衡模式（通用场景）
SEG_HZ = 30.0              # 默认关键帧频率
ONLINE_CMD_HZ = 200.0      # 默认在线插值频率
BASE_DIFF_EPS = 0.010      # 默认差分阈值
DIST_EPS = 0.06            # 默认位移阈值
```

### 3. 性能调优建议

#### 机械臂性能相关：
- **高精度机械臂**：可提高采样频率和降低阈值
- **大负载机械臂**：降低加速度和速度限制
- **老旧机械臂**：降低命令频率，增加延迟

#### 应用场景相关：
- **精密装配**：高采样频率 + 低阈值 + 平滑规划器
- **快速搬运**：适中采样频率 + 高阈值 + 直线规划器
- **教学演示**：低采样频率 + 高阈值 + 可视化开启

#### 环境因素相关：
- **网络延迟高**：降低WebSocket同步频率
- **CPU资源紧张**：降低在线插值频率
- **存储空间有限**：降低采样频率

### 4. 故障排除参数调整

```python
# 解决"卡顿"问题
ONLINE_CMD_HZ = 100.0      # 降低命令频率
SEG_HZ = 20.0              # 降低关键帧频率
WIN_SECS = 0.6              # 增加判定窗口

# 解决"精度不够"问题
BASE_DIFF_EPS = 0.005      # 降低差分阈值
DIST_EPS = 0.03            # 降低位移阈值
SEG_HZ = 50.0              # 提高关键帧频率

# 解决"响应过慢"问题
ONLINE_CMD_HZ = 300.0      # 提高命令频率
WIN_SECS = 0.2              # 减少判定窗口
MAX_HEAD_CHECK = 900.0      # 减少检查范围
```

---

这些脚本可作为功能测试或二次开发的起点。你也可以结合多个模块编写更复杂的应用逻辑。