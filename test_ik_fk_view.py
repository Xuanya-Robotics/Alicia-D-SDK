import PyKDL
import math
import numpy as np

def create_alicia_duo_kinematic_chain():
    """
    Hardcodes the kinematic chain for alicia_duo based on the URDF.
    Returns a PyKDL.Chain object.
    """
    chain = PyKDL.Chain()

    # Helper to create PyKDL.Frame from xyz, rpy
    def create_frame(xyz, rpy):
        return PyKDL.Frame(
            PyKDL.Rotation.RPY(rpy[0], rpy[1], rpy[2]),
            PyKDL.Vector(xyz[0], xyz[1], xyz[2])
        )
    
    joint_local_origin = PyKDL.Vector(0.0, 0.0, 0.0) 

    # --- Segment 0: Fixed transform from base_link to Joint1 origin ---
    f_tip_base_to_j1 = create_frame(xyz=[0, 0, 0.0745], rpy=[0, 0, 0])
    chain.addSegment(PyKDL.Segment("base_to_j1_segment",
                                   PyKDL.Joint(PyKDL.Joint.Fixed),
                                   f_tip_base_to_j1))

    # --- Segment 1: Joint1 ---
    joint1_axis = PyKDL.Vector(0, 0, 1)
    f_tip_j1_to_j2 = create_frame(xyz=[-4E-05, 0, 0.09361], rpy=[-1.5708, -1.4701, 0])
    chain.addSegment(PyKDL.Segment("Link1_J1",
                                   PyKDL.Joint("Joint1", joint_local_origin, joint1_axis, PyKDL.Joint.RotAxis),
                                   f_tip_j1_to_j2))

    # --- Segment 2: Joint2 ---
    joint2_axis = PyKDL.Vector(0, 0, -1)
    f_tip_j2_to_j3 = create_frame(xyz=[0.22471, 0, 0.0004], rpy=[0, 0, -2.4569])
    chain.addSegment(PyKDL.Segment("Link2_J2",
                                   PyKDL.Joint("Joint2", joint_local_origin, joint2_axis, PyKDL.Joint.RotAxis),
                                   f_tip_j2_to_j3))

    # --- Segment 3: Joint3 ---
    joint3_axis = PyKDL.Vector(0, 0, -1)
    f_tip_j3_to_j4 = create_frame(xyz=[0.00211, -0.0969, -0.0005], rpy=[1.5708, 0, 0])
    chain.addSegment(PyKDL.Segment("Link3_J3",
                                   PyKDL.Joint("Joint3", joint_local_origin, joint3_axis, PyKDL.Joint.RotAxis),
                                   f_tip_j3_to_j4))

    # --- Segment 4: Joint4 ---
    joint4_axis = PyKDL.Vector(0, 0, -1)
    f_tip_j4_to_j5 = create_frame(xyz=[0.00014, 0.0002, 0.12011], rpy=[-1.5708, 0, 0])
    chain.addSegment(PyKDL.Segment("Link4_J4",
                                   PyKDL.Joint("Joint4", joint_local_origin, joint4_axis, PyKDL.Joint.RotAxis),
                                   f_tip_j4_to_j5))

    # --- Segment 5: Joint5 ---
    joint5_axis = PyKDL.Vector(0, 0, -1)
    f_tip_j5_to_j6 = create_frame(xyz=[-0.00389, -0.0592, 0.00064], rpy=[1.5708, 0, 0])
    chain.addSegment(PyKDL.Segment("Link5_J5",
                                   PyKDL.Joint("Joint5", joint_local_origin, joint5_axis, PyKDL.Joint.RotAxis),
                                   f_tip_j5_to_j6))

    # --- Segment 6: Joint6 ---
    T_J6_to_GraspBase = create_frame(xyz=[0,0,0], rpy=[0,0,0])
    T_GraspBase_to_tool0 = create_frame(xyz=[-0.0065308, -0.00063845, 0.11382], rpy=[0,0,0])
    f_tip_j6_to_tool0 = T_J6_to_GraspBase * T_GraspBase_to_tool0

    joint6_axis = PyKDL.Vector(0, 0, -1)
    chain.addSegment(PyKDL.Segment("Link6_J6_to_tool0",
                                   PyKDL.Joint("Joint6", joint_local_origin, joint6_axis, PyKDL.Joint.RotAxis),
                                   f_tip_j6_to_tool0))
    return chain


def forward_kinematics_alicia_duo(chain, joint_angles_rad): # Added chain as argument
    """
    Calculates the forward kinematics for the alicia_duo robot.
    Args:
        chain (PyKDL.Chain): The kinematic chain.
        joint_angles_rad (list or tuple of 6 floats): Joint angles in radians
                                                     for Joint1 to Joint6.
    Returns:
        PyKDL.Frame: The resulting frame of tool0.
    """
    if not isinstance(chain, PyKDL.Chain):
        raise TypeError("chain argument must be a PyKDL.Chain object")
    if len(joint_angles_rad) != chain.getNrOfJoints(): # Use chain's number of joints
        raise ValueError(f"Expected {chain.getNrOfJoints()} joint angles, got {len(joint_angles_rad)}.")

    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    kdl_joint_angles = PyKDL.JntArray(chain.getNrOfJoints())
    for i in range(chain.getNrOfJoints()):
        kdl_joint_angles[i] = joint_angles_rad[i]

    tool0_frame = PyKDL.Frame()
    ret = fk_solver.JntToCart(kdl_joint_angles, tool0_frame)
    if ret < 0:
        raise RuntimeError("PyKDL forward kinematics solver failed.")
    return tool0_frame

def get_fk_results_from_frame(tool0_frame):
    """Helper to extract pos, quat, yaw from a PyKDL.Frame"""
    pos = tool0_frame.p
    position_xyz = [pos.x(), pos.y(), pos.z()]
    rot = tool0_frame.M
    quaternion_xyzw = list(rot.GetQuaternion()) # Returns (x, y, z, w)
    rpy = rot.GetRPY()
    yaw_rad = rpy[2]
    return position_xyz, quaternion_xyzw, yaw_rad


def inverse_kinematics_alicia_duo(chain, target_position_xyz, target_quaternion_xyzw, initial_joint_angles_rad=None):
    """
    计算alicia_duo机器人的逆运动学，仅使用ChainIkSolverPos_NR_JL求解器。
    
    参数:
        chain (PyKDL.Chain): 机器人运动链。
        target_position_xyz (list of 3 floats): 工具目标位置[x, y, z]。
        target_quaternion_xyzw (list of 4 floats): 工具目标四元数[x, y, z, w]。
        initial_joint_angles_rad (list of 6 floats, optional): 关节角度初始猜测值。
                                                              若为None，使用零位姿态。
    
    返回:
        list of 6 floats: 成功求解的关节角度(弧度)，失败则返回None。
    """
    # 基本参数验证
    if not isinstance(chain, PyKDL.Chain):
        raise TypeError("chain参数必须是PyKDL.Chain对象")
    if len(target_position_xyz) != 3:
        raise ValueError("target_position_xyz必须是包含3个浮点数的列表。")
    if len(target_quaternion_xyzw) != 4:
        raise ValueError("target_quaternion_xyzw必须是包含4个浮点数的列表。")

    num_joints = chain.getNrOfJoints()

    # 设置关节限制
    q_min_rad = PyKDL.JntArray(num_joints)
    q_max_rad = PyKDL.JntArray(num_joints)
    
    margin = 0.1  # 10度余量
    limits = [
        (-2.16 + margin, 2.16 - margin),  # Joint 1
        (-3.14 + margin, 3.14 - margin),  # Joint 2
        (-3.14 + margin, 3.14 - margin),  # Joint 3
        (-3.14 + margin, 3.14 - margin),  # Joint 4
        (-3.14 + margin, 3.14 - margin),  # Joint 5
        (-3.14 + margin, 3.14 - margin)   # Joint 6
    ]
    
    for i in range(num_joints):
        q_min_rad[i] = limits[i][0]
        q_max_rad[i] = limits[i][1]

    # 准备目标帧
    target_vector = PyKDL.Vector(target_position_xyz[0], target_position_xyz[1], target_position_xyz[2])
    
    # 四元数标准化
    quat_array = np.array(target_quaternion_xyzw, dtype=float)
    quat_norm = np.linalg.norm(quat_array)
    if quat_norm == 0:
        raise ValueError("四元数范数为零")
    quat_normalized = quat_array / quat_norm
    
    try:
        target_rotation = PyKDL.Rotation.Quaternion(
            float(quat_normalized[0]), 
            float(quat_normalized[1]), 
            float(quat_normalized[2]), 
            float(quat_normalized[3])
        )
        target_frame = PyKDL.Frame(target_rotation, target_vector)
    except Exception as e:
        print(f"创建目标帧时出错: {e}")
        return None

    # 设置初始猜测
    if initial_joint_angles_rad is None or len(initial_joint_angles_rad) != num_joints:
        initial_joint_angles_rad = [0.0] * num_joints
    
    q_init = PyKDL.JntArray(num_joints)
    for i in range(num_joints):
        # 确保初始猜测在关节限制内
        q_init[i] = max(limits[i][0], min(limits[i][1], initial_joint_angles_rad[i]))
    
    # 创建求解器
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
    ik_solver = PyKDL.ChainIkSolverPos_NR_JL(chain, q_min_rad, q_max_rad,
                                             fk_solver, vel_solver,
                                             1000, 1e-5)  # 使用固定的迭代次数和精度
    
    # 求解
    q_out = PyKDL.JntArray(num_joints)
    ret_val = ik_solver.CartToJnt(q_init, target_frame, q_out)
    
    # 检查结果
    if ret_val >= 0:
        # 求解成功
        solved_angles = [q_out[i] for i in range(num_joints)]
        
        # 简单验证结果
        try:
            verify_frame = forward_kinematics_alicia_duo(chain, solved_angles)
            pos_diff = (verify_frame.p - target_frame.p).Norm()
            
            print(f"IK求解成功 (返回值: {ret_val})")
            print(f"位置误差: {pos_diff:.6f}")
            return solved_angles
        except Exception as e:
            print(f"验证IK解时出错: {e}")
    
    print(f"IK求解失败 (返回值: {ret_val})")
    return None



if __name__ == "__main__":
    alicia_chain = create_alicia_duo_kinematic_chain()
    print(f"运动链中的关节数量: {alicia_chain.getNrOfJoints()}")
    print(f"运动链中的段数量: {alicia_chain.getNrOfSegments()}")


    # === 测试正向运动学 ===
    print("\n--- 测试正向运动学 ---")
    # q_fk_test = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    q_fk_test = [0.1, 0.1, 0.1, 0.1, 0., 0] # 零位姿态
    try:
        tool_frame_fk = forward_kinematics_alicia_duo(alicia_chain, q_fk_test)
        pos, quat, yaw = get_fk_results_from_frame(tool_frame_fk)
        print(f"正运动学输入关节角度 (弧度): {q_fk_test}")
        print(f"  工具坐标位置 (x,y,z): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  工具坐标四元数 (x,y,z,w): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print(f"  工具坐标偏航角 (弧度): {yaw:.4f} (角度: {math.degrees(yaw):.2f})")

        # === 测试逆向运动学 ===
        print("\n--- 测试逆向运动学 ---")
        # 使用正运动学的结果作为逆运动学的目标
        target_pos_ik = pos
        target_quat_ik = quat
        
        # 提供一个初始猜测值（可以与解不同）
        q_initial_guess_ik = [0, 0, 0.3, 0, 0, 0]
        #q_initial_guess_ik = [0.05, -0.05, 0.1, -0.1, 0.05, -0.05] # 稍微扰动的初始猜测

        print(f"逆运动学目标位置: {target_pos_ik}")
        print(f"逆运动学目标四元数: {target_quat_ik}")
        print(f"逆运动学初始关节角度: {q_initial_guess_ik}")

        solved_ik_angles = inverse_kinematics_alicia_duo(alicia_chain,
                                                         target_pos_ik,
                                                         target_quat_ik,
                                                         q_initial_guess_ik)

        if solved_ik_angles:
            print(f"逆运动学求解的关节角度 (弧度): {[f'{a:.4f}' for a in solved_ik_angles]}")

            # 通过正运动学验证逆运动学解
            print("通过正运动学验证逆运动学解:")
            tool_frame_ik_verify = forward_kinematics_alicia_duo(alicia_chain, solved_ik_angles)
            pos_verify, quat_verify, yaw_verify = get_fk_results_from_frame(tool_frame_ik_verify)
            print(f"  逆运动学解的正运动学验证 - 位置: [{pos_verify[0]:.4f}, {pos_verify[1]:.4f}, {pos_verify[2]:.4f}]")
            print(f"  逆运动学解的正运动学验证 - 四元数: [{quat_verify[0]:.4f}, {quat_verify[1]:.4f}, {quat_verify[2]:.4f}, {quat_verify[3]:.4f}]")


        else:
            print("未找到逆运动学解。")



    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()