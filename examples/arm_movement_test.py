#!/usr/bin/env python3
# coding=utf-8

"""
机械臂运动控制示例
演示如何控制机械臂进行基本运动、力矩控制和运动示例。
"""

import os
import sys
import time
import math
import torch
import robolab 

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alicia_duo_sdk.controller import ArmController

class ArmMovement:
    def __init__(self, urdf_path, export_link):
        self.robot = robolab.RobotModel(urdf_path, solve_engine="pytorch_kinematics", verbose=False)
        self.export_link = export_link

    def forward_kinematics(self, joint_values):
        pos, rot, _ = self.robot.get_fk(joint_values, self.export_link)
        pos = torch.squeeze(pos, 0)
        rot = torch.squeeze(rot, 0)
        return pos, rot

    def inverse_kinematics(self, ee_pose, cur_configs=None):
        ret = self.robot.get_ik(ee_pose, self.export_link, cur_configs=cur_configs)
        solutions = torch.squeeze(ret.solutions, 0) if ret.solutions.dim() > 2 else torch.squeeze(ret.solutions)
        return solutions

class sdk_demo():
    def __init__(self):
        """初始化SDK演示类"""
        self.controller = ArmController(port='/dev/ttyUSB0', debug_mode=False)
        if not self.controller.connect():
            print("无法连接到机械臂，请检查连接")
            sys.exit(1)
        print("连接成功")
        
        urdf_path = "../Alicia_duo_sdk/Robolab-main/assets/urdf/alicia_duo_descriptions/urdf/alicia_duo_with_gripper.urdf"
        export_link = "tool0"
        self.arm = ArmMovement(urdf_path, export_link)

    def move_to_pointA(self):
        """示例函数：将机械臂缓慢移动到点A（线性插值）"""
        print("缓慢移动到点A...")
        # point_A = [7, -25.79, 30.85, 85.08, 78.93, -34.19]
        # point_A = [0.1303883669702795, -0.4678641403051206, 
        #            0.5307573526084318, 1.4833594218854151, 
        #            1.3560390164909069, -0.5982525072754]
        # ee_pose_A = [-0.2101, -0.2046,  0.2969, 0.2282, -0.6806,  0.2992,  0.6286]
        ee_pose_A = [-0.28, -0.15,  0.35 , 0.7055, 0.0996, -0.6853, 0.1507]
        
        
        steps = 120
        delay = 0.03

        current_angle = self.controller.read_joint_state().angles   
        print(f"当前关节角度_RAD: {current_angle}")
        time.sleep(1)  # 等待一秒
        ik_angles = self.arm.inverse_kinematics(ee_pose_A, [current_angle])
        print(f"Inverse Kinematics Angles: {ik_angles}")
        ik_angles = ik_angles.squeeze().tolist()  # 转为Python list
        print(f"ik_angles_new: {ik_angles}")
        ik_angles_norm = [((a + math.pi) % (2 * math.pi)) - math.pi for a in ik_angles]
        print(ik_angles_norm)        
        
        self.slow_move(current_angle, ik_angles_norm, steps=steps, delay=delay)

        # self.slow_move(current_angle, ik_angles, steps, delay)
        print("已到点A位置")

    def move_to_home(self):
        """示例函数：将机械臂缓慢移动到初始位置（线性插值）"""
        print("缓慢移动到初始位置...")
        home = [0.0] * 6  # 初始位置角度（单位：弧度）
        steps = 120
        delay = 0.03            
        # 读取当前角度
        current_state = self.controller.read_joint_state()
        current_angles = list(current_state.angles)
        self.slow_move(current_angles, home, steps=steps, delay=delay)
        print("已到初始位置")

    def move_to_pointB(self):
        """示例函数：将机械臂缓慢移动到PointB位置（线性插值）"""
        print("缓慢移动到PointB位置...")
        # PointB位置角度（单位：度）
        # point_B = [14.85, 18.63, 0.0, 23.91, 36.47, -24.87]
        # point_B = [0.24083498369804568, 0.30679615757712825, 
        #            -0.010737865515199488, 0.4049709280018093, 
        #            0.6151262959421421, -0.4279806398200939]

        ee_pose_B = [-0.30, -0.12,  0.30 , 0.7055, 0.0996, -0.6853, 0.1507]
        

        current_angle = self.controller.read_joint_state().angles   
        print(f"当前关节角度_RAD: {current_angle}")
        
        ik_angles = self.arm.inverse_kinematics(ee_pose_B, [current_angle])  # 求解逆运动学])
        print(f"Inverse Kinematics Angles: {ik_angles}")
        # 将IK解转为Python list
        ik_solutions = ik_angles.squeeze().tolist()
        print(f"ik_angles_new: {ik_solutions}")
        # 如果有多组解，选择与当前角度差异最小的解
        if isinstance(ik_solutions[0], list) or isinstance(ik_solutions[0], tuple):
            # 多组解
            def angle_distance(a, b):
                return sum(abs(x - y) for x, y in zip(a, b))
            best_solution = min(ik_solutions, key=lambda sol: angle_distance(sol, current_angle))
            print(f"选择的最佳解: {best_solution}")
            ik_angles = best_solution
       
        else:
            # 只有一组解
            ik_angles = ik_solutions
        ik_angles_norm = [((a + math.pi) % (2 * math.pi)) - math.pi for a in ik_angles]
        print(f"ik角度：{ik_angles_norm}")
        self.slow_move(current_angle, ik_angles_norm, steps=120, delay=0.03)
        print("已到PointB位置")
    
    def slow_move(self, current_angles, target_angles, steps=120, delay=0.03):
        """缓慢移动到目标角度"""
        for step in range(1, steps + 1):
            interp_angles = [
                current + (target - current) * step / steps
                for current, target in zip(current_angles, target_angles)
            ]
            self.controller.set_joint_angles(interp_angles, wait_for_completion=False)
            time.sleep(delay)

    def print_arm_info(self, target_angles=None):
        """打印机械臂信息"""
        print("机械臂信息:")
        state = self.controller.read_joint_state()

        if target_angles is not None:
            pos, quat, rpy = self.controller.forward_kinematics_alicia_duo(target_angles)
            print(f"正向运动学结果: 位置: {pos}, 四元数: {quat}, 欧拉角: {rpy} (弧度)")
            print(f"目标关节角度_DEG: {[round(angle * self.controller.RAD_TO_DEG, 2) for angle in target_angles]}")
            
        else:
            print(f"当前关节角度_RAD: {state.angles}")
            print(f"当前关节角度_DEG: {[round(angle * self.controller.RAD_TO_DEG, 2) for angle in state.angles]}")
            pos, quat, rpy=self.controller.forward_kinematics_alicia_duo(state.angles)
            print(f"正向运动学结果: 位置: {pos}, 四元数: {quat}, 欧拉角: {rpy} (弧度)")

        return pos, quat, rpy, state.angles
    
    def solve_inverse_kinematics(self, pos, quat, current_angles=None):
        """求解逆运动学"""
        angles = self.controller.inverse_kinematics_alicia_duo(pos, quat, current_angles)
        print(f"逆运动学结果: {angles}")
        return angles
    
def main():

    """主函数"""
    print("=== 机械臂运动控制示例 ===")

    demo = sdk_demo()

    try:
        # 关闭夹爪
        demo.controller.set_gripper(100 * demo.controller.DEG_TO_RAD, wait_for_completion=True)

# -----------------------------------------------
        # # 回到初始位置
       # Forward Kinematics Position: tensor([-0.3085, -0.0036,  0.0907]), 
       # Rotation: tensor([ 7.1252e-03, -9.3018e-01,  5.4580e-04,  3.6703e-01])
        for i in range (11):
            pos, rot = demo.arm.forward_kinematics(demo.controller.read_joint_state().angles)
            print(f"Forward Kinematics Position: {pos}, Rotation: {rot}")
            time.sleep(1)  # 等待一秒
            demo.move_to_pointB()
            time.sleep(1)  # 等待一秒
            demo.move_to_pointA()

       
        input("按回车键继续...")  # 等待用户输入
        
        demo.move_to_home()

        
#--------------------------------------------------------


 # ------------------------------------------------------
        
        # demo.move_to_home()
        # time.sleep(2)  # 等待一秒
        
    
        # time.sleep(1)  # 等待一秒
        # demo.move_to_pointA()
        # time.sleep(1)  # 等待一秒
        # demo.move_to_pointB()
        # time.sleep(1)  # 等待一秒
        # demo.move_to_home()

        
        #---------------初始点到A点往返运动-----------------
        # time.sleep(0.5)  # 等待一秒
        # for i in range(1):
        #     print(f"第{i+1}次移动")
        #     demo.move_to_pointB()
        #     pos, quat, rpy, current_angles = demo.print_arm_info()
        #     demo.solve_inverse_kinematics(pos, quat, current_angles)
        #     demo.move_to_pointA()
        #     time.sleep(3)
        #     # pos, quat, rpy, current_angles = demo.print_arm_info()
        #     # demo.solve_inverse_kinematics(pos, quat, current_angles)
        #     # demo.move_to_pointB()
        #     # demo.move_to_home()
        #     # time.sleep(0.5)

    
        time.sleep(1)  # 等待一秒
        print("Done") 

        
        
    
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    finally:
        # 断开连接
        demo.controller.disconnect()
        print("已断开连接")

if __name__ == "__main__":
    main()


# 当前关节角度_RAD: [0.35128160042581186, 0.4724660826687775, -0.06902913545485385, 0.015339807878856412, 0.3666214083046683, -0.09050486648525283]
# 当前关节角度_DEG: [20.13, 27.07, -3.96, 0.88, 21.01, -5.19]
# 正向运动学结果: 位置: [0.29565895883089177, 0.11060762506468941, 0.2936656243220516], 四元数: [-0.09960596823681092, 0.7055285807343937, 0.15073709972336696, 0.6852637445724872], 欧拉角: (1.770042249816714, 1.4929942370529308, 2.126228752254893) (弧度)
# /home/senyu/Alicia_duo_sdk/Robolab-main/robolab/coord/transform_tensor.py:94: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32)
# Forward Kinematics Position: tensor([-0.2957, -0.1106,  0.2937]), Rotation: tensor([ 0.7055,  0.0996, -0.6853,  0.1507])

# Forward Kinematics Position: tensor([-0.3832, -0.0337,  0.1930]), Rotation: tensor([-0.4556, -0.5199,  0.2002,  0.6943])