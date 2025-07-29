# motion_session.py (in controller/)

class MotionSession:
    def __init__(self, ik_controller, robot_model, joint_controller):
        self.ik_controller = ik_controller
        self.robot_model = robot_model
        self.joint_controller = joint_controller
        self.last_joint_angles = None

    def __repr__(self):
        return "<MotionSession Alicia-D>"
