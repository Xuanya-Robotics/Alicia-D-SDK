# motion_session.py (in controller/)
from alicia_duo_sdk.kinematics import IKController, RobotArm
from alicia_duo_sdk.driver import ArmController

class MotionSession:
    def __init__(self, ik_controller: IKController, robot_model: RobotArm, 
                 joint_controller: ArmController):
        self.ik_controller = ik_controller
        self.robot_model = robot_model
        self.joint_controller = joint_controller

    def __repr__(self):
        return "<MotionSession Alicia-D>"
