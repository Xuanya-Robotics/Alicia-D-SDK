import numpy as np
from alicia_duo_sdk.kinematics import Advanced6DOFIKSolver
from alicia_duo_sdk.kinematics import RobotArm  # 根据你项目实际路径修改
from typing import List, Dict


def list_to_joint_dict(joint_list: List[float]) -> Dict[str, float]:

    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    return {name: angle for name, angle in zip(joint_names, joint_list)}

def joint_dict_to_list(joint_dict: Dict[str, float]) -> List[float]:
    return [joint_dict[name] for name in ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']]

if __name__ == "__main__":
    # 1. 创建机器人模型
    robot_model = RobotArm()  # 必须已实现 forward_kinematics(dict) -> pos, quat

    # 2. 创建 IK 解算器
    ik_solver = Advanced6DOFIKSolver(robot_model)

    # 3. 设置初始角度（List 格式）
    # initial_angles_list = [0.5, -0.5, 0.3, 0.3, 0.5, -0.6]  # 示例起始值
    initial_angles_list = [-0.5997864880632857, -0.45712627478992107, 
              0.9602719732164114, 0.9802137234589248, 
              -1.504835152915814, 0.2899223689103862]

    # 4. 自动转换为 Dict 格式
    initial_angles_dict = list_to_joint_dict(initial_angles_list)

    # 5. 获取当前末端位置作为目标（用于测试 self-consistency）
    # target_pos, target_quat = robot_model.forward_kinematics(initial_angles_dict)
    target_pos = np.array([0.16547103, -0.31027681,  0.19629864])
    target_quat = np.array([ 0.06037905,  0.87821258, -0.38372519, -0.27901975 ])

    # 6. 调用 IK 解算
    result_angles = ik_solver.solve(target_pos, target_quat, initial_angles_dict)

    # 7. 打印结果
    print("\n=== IK 结果 ===")
    print("输入初始角度（list）:", initial_angles_list)
    print("目标位置:", target_pos)
    print("目标四元数:", target_quat)
    print("求解后的角度（dict）:", result_angles)
    print("求解后的角度（list）:", joint_dict_to_list(result_angles))

    # 8. 可选验证
    solved_pos, solved_quat = robot_model.forward_kinematics(result_angles)
    pos_error = np.linalg.norm(target_pos - solved_pos)
    ori_error = np.linalg.norm(
        Advanced6DOFIKSolver._compute_orientation_error(None, target_quat, solved_quat)
    )
    print(f"\n位置误差: {pos_error:.6f} m")
    print(f"姿态误差: {np.degrees(ori_error):.3f} deg")
