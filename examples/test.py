from alicia_duo_sdk.controller import get_default_session, moveL, moveCartesian
import time

session = get_default_session(debug=False)

# def get_current_pose():
#     joint_angles = session.joint_controller.get_joint_angles()
#     pos, quat = session.robot_model.forward_kinematics(dict(zip(session.robot_model.kinematic_chain[:-1], joint_angles)))
#     print("\n[INFO] 当前末端位置:", pos)
#     print("[INFO] 当前末端四元数:", quat)
#     return pos, quat


# get_current_pose()
# input("取消扭矩")
# session.joint_controller.disable_torque()
# input("开启扭矩")
# session.joint_controller.enable_torque()
# get_current_pose()


# 测试目标姿态（静态）
home_pose = [-0.31130141,  0.09555385, -0.00351151, 0.65652068, -0.27160338, -0.25853801,  0.65450004]
target_pose = [-0.19104138,  0.24172841, -0.30412644, 0.68807965, -0.67196606, -0.20408894,  0.18263549]  

print("\n[TEST] 执行 moveL...")
while True:
    moveL(session, target_pose, visualize=True)
    time.sleep(1.0)
    moveL(session, home_pose, visualize=True) 