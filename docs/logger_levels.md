# 日志级别过滤功能

Alicia-D-SDK 的日志器现在支持日志级别过滤功能，允许用户控制哪些级别的日志会被打印到控制台。

## 功能概述

通过设置最低日志级别，你可以控制哪些日志消息会显示在控制台上。所有日志仍然会被写入日志文件，但只有达到指定级别的日志才会在控制台显示。

## 日志级别定义

日志级别按重要性从低到高排列：

| 级别 | 数值 | 描述 |
|------|------|------|
| DEBUG | 0 | 调试信息，最详细的日志级别 |
| INFO | 1 | 一般信息，默认级别 |
| MODULE | 2 | 模块相关信息 |
| WARNING | 3 | 警告信息 |
| ERROR | 4 | 错误信息 |
| SUCCESS | 5 | 成功信息，最高级别 |

## 使用方法

### 1. 创建日志器时指定级别

```python
from alicia_duo_sdk.utils.logger import BeautyLogger, LogLevel

# 只显示 WARNING 及以上级别的日志
logger = BeautyLogger(
    log_dir="logs", 
    log_name="app.log", 
    min_level=LogLevel.WARNING
)

# 显示所有级别的日志（DEBUG 级别）
logger = BeautyLogger(
    log_dir="logs", 
    log_name="app.log", 
    min_level=LogLevel.DEBUG
)

# 只显示 ERROR 级别的日志
logger = BeautyLogger(
    log_dir="logs", 
    log_name="app.log", 
    min_level=LogLevel.ERROR
)
```

### 2. 动态改变日志级别

```python
# 创建日志器
logger = BeautyLogger("logs", "app.log", min_level=LogLevel.INFO)

# 记录一些日志
logger.info("应用启动")
logger.debug("这条 DEBUG 消息不会显示")

# 动态改为 DEBUG 级别
logger.set_min_level(LogLevel.DEBUG)
logger.debug("现在这条 DEBUG 消息会显示")

# 改为只显示错误
logger.set_min_level(LogLevel.ERROR)
logger.warning("这条 WARNING 消息不会显示")
logger.error("这条 ERROR 消息会显示")
```

### 3. 实际使用示例

```python
from alicia_duo_sdk.utils.logger import BeautyLogger, LogLevel

# 开发环境：显示所有日志
if development_mode:
    logger = BeautyLogger("logs", "dev.log", min_level=LogLevel.DEBUG)
else:
    # 生产环境：只显示警告和错误
    logger = BeautyLogger("logs", "prod.log", min_level=LogLevel.WARNING)

# 记录不同级别的日志
logger.debug("详细的调试信息")
logger.info("应用正常运行")
logger.warning("需要注意的警告")
logger.error("发生错误")

# 运行时根据需要调整级别
if verbose_mode:
    logger.set_min_level(LogLevel.DEBUG)
```

## 注意事项

1. **日志文件完整性**：无论设置什么级别，所有日志都会被写入日志文件，只是控制台显示会过滤。

2. **默认级别**：如果不指定 `min_level` 参数，默认使用 `LogLevel.INFO` 级别。

3. **级别验证**：`set_min_level()` 方法会验证级别值是否有效，无效级别会抛出 `ValueError`。

4. **向后兼容**：现有代码无需修改，新功能完全向后兼容。

## 错误处理

```python
try:
    logger.set_min_level(999)  # 无效级别
except ValueError as e:
    print(f"无效的日志级别: {e}")
```

## 最佳实践

1. **开发阶段**：使用 `LogLevel.DEBUG` 获取详细信息
2. **测试阶段**：使用 `LogLevel.INFO` 查看主要流程
3. **生产环境**：使用 `LogLevel.WARNING` 或 `LogLevel.ERROR` 减少噪音
4. **调试问题**：临时设置为 `LogLevel.DEBUG` 获取更多信息

## 完整示例

查看 `examples/logger_level_demo.py` 文件获取完整的演示代码。
