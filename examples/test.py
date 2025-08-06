from alicia_duo_sdk.controller import get_default_session, moveL, moveCartesian, moveLQT
from alicia_duo_sdk.execution import TrajectoryExecutor
import time
import numpy as np
from alicia_duo_sdk.kinematics import *
from alicia_duo_sdk.controller.control_api import test

session = get_default_session(debug=False)
# update_time = 0
# while True:
#     if time.time() - update_time > 1:
#         angles = session.joint_controller.get_joint_angles()
#         pose = session.robot_model.forward_kinematics(angles)
#         print(pose)
#         update_time = time.time()



# get_current_pose()
# input("取消扭矩")
# session.joint_controller.disable_torque()
# input("开启扭矩")
# session.joint_controller.enable_torque()
# get_current_pose()


# 测试目标姿态（静态）
home_pose = [-0.31130141,  0.09555385, -0.00351151, 0.65652068, -0.27160338, -0.25853801,  0.65450004]
target_pose = [-0.19104138,  0.24172841, -0.30412644, 0.68807965, -0.67196606, -0.20408894,  0.18263549]  
via_points_1 = [home_pose, target_pose]
via_points_2 = [target_pose, home_pose]

# print("\n[TEST] 执行 moveL...")

# traj = np.load("test.npy").tolist()
# # print(traj)

# test(session=session, pose_sequence=traj)


while True:
    user_input = input("按下q结束：")
    if user_input.lower() == 'q':
        break 
    input("开始采样")
    session.joint_controller.disable_torque()
    input("记录目标点")
    session.joint_controller.enable_torque()
    time.sleep(1)
    angle_1 = session.joint_controller.get_joint_angles()
    pose_1 = session.robot_model.forward_kinematics(angle_1)
    print(pose_1)
    input('huihce')
    while True:
        try:
            moveL(session, visualize=True, start_joint_angles=angle_1, target_pose=home_pose)
            moveL(session, visualize=True, target_pose=pose_1)
        except KeyboardInterrupt:
            break
                    

 
    

    # moveLQT(session=session, pose_sequence=via_points_1, visualize=True)
    # moveLQT(session=session, pose_sequence=via_points_2, visualize=True)