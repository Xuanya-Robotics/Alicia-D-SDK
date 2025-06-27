"""

for Alicia-D follower arm zero calibration example
python alicia_duo_zero_calibration.py --follower True

for Alicia-D leader arm zero calibration example
python alicia_duo_zero_calibration.py

"""

import os
import sys
import argparse
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alicia_duo_sdk.controller import ArmController



def main():
    """主函数"""
    print("=== Alicia-D调零示例 ===")
    
    parser = argparse.ArgumentParser(description="调零示例")
    parser.add_argument('--follower', type=bool, default=False,
                       help='是否为操作臂调零 (True/False, 默认: False)')
    # 创建控制器实例 (可选参数: port="/dev/ttyUSB0", debug_mode=True)
    args = parser.parse_args()

    arm_type = "follower" if args.follower else "leader"
    print(f"=== {arm_type}臂调零示例 ===")
    controller = ArmController(debug_mode=False)
    try:
        # 连接到机械臂
        if not controller.connect():
            print("无法连接到机械臂，请检查连接")
            return
            
        print("连接成功，开始读取数据...")
        print("按 Ctrl+C 退出")
        print("-" * 50)
        if arm_type == "follower":
            controller.disable_torque()
        controller.set_zero_position()
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()