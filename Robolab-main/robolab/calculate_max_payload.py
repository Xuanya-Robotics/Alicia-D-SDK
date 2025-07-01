import os
import numpy as np
import torch
from typing import List

from rofunc.utils.robolab.kinematics.robot_class import RobotModel

class MaxPayloadCalculator:
    """Class to calculate maximum payload for a robot arm under static conditions."""
    
    def __init__(self, model_path: str, end_effector_link: str = None, custom_torque_limits: List[float] = None, device="cpu"):
        """
        Initialize the calculator.
        
        Args:
            model_path (str): Path to URDF/MJCF file
            end_effector_link (str, optional): Name of end effector link. If None, will use last link
            custom_torque_limits (List[float], optional): Custom joint torque limits. If None, will use from model or defaults
            device (str): Device to run calculations on ("cpu" or "cuda")
        """
        # Initialize robot model
        self.robot = RobotModel(model_path, device=device)
        self.device = device
        
        # Set end effector link
        if end_effector_link is None:
            self.end_effector_link = self.robot.get_link_list()[-1]  # Use last link as end effector
        else:
            self.end_effector_link = end_effector_link
            
        print(f"\n使用末端执行器链接: {self.end_effector_link}")
        
        # Get joint torque limits
        if custom_torque_limits is not None:
            if len(custom_torque_limits) != len(self.robot.joint_list):
                raise ValueError(f"自定义扭矩限制数量 ({len(custom_torque_limits)}) 与关节数量 ({len(self.robot.joint_list)}) 不匹配")
            self.joint_torque_limits = custom_torque_limits
            print("\n使用自定义关节扭矩限制")
        else:
            self.joint_torque_limits = self._get_joint_torque_limits()
        
        # Convert torque limits to tensor
        self.torque_limits = torch.tensor(self.joint_torque_limits, device=device)
        
        # Gravity vector
        self.gravity = torch.tensor([0, 0, -9.81], device=device)

    def _get_joint_torque_limits(self) -> List[float]:
        """
        Get joint torque limits from robot model.
        
        Returns:
            List[float]: List of joint torque limits
        """
        torque_limits = []
        default_limits = {
            1: 150.0,  # 基座关节
            2: 150.0,  # 肩关节
            3: 100.0,  # 肘关节
            4: 100.0,  # 腕关节1
            5: 50.0,   # 腕关节2
            6: 50.0,   # 腕关节3
            7: 30.0    # 末端关节
        }
        
        # Check if model is URDF or MJCF
        if hasattr(self.robot.robot_model, 'joint_map'):  # URDF
            for joint in self.robot.joint_list:
                joint_info = self.robot.robot_model.joint_map[joint]
                torque_limits.append(float(joint_info.limit.effort))
            print("\n从URDF文件读取关节扭矩限制")
        else:  # MJCF
            print("\n提示: MJCF文件没有定义关节扭矩限制")
            user_input = input("是否使用默认扭矩限制值? [Y/n]: ")
            
            if user_input.lower() != 'n':
                # 使用默认值
                for i, joint in enumerate(self.robot.joint_list, 1):
                    torque_limits.append(default_limits.get(i, 50.0))
                print("\n使用默认扭矩限制 (Nm):")
            else:
                # 用户输入
                print("\n请为每个关节输入扭矩限制值 (Nm):")
                for joint in self.robot.joint_list:
                    while True:
                        try:
                            limit = float(input(f"{joint}: "))
                            if limit <= 0:
                                print("扭矩限制必须大于0")
                                continue
                            torque_limits.append(limit)
                            break
                        except ValueError:
                            print("请输入有效的数值")
                print("\n使用用户输入的扭矩限制 (Nm):")
            
            for joint, torque in zip(self.robot.joint_list, torque_limits):
                print(f"  {joint}: {torque}")
                
        return torque_limits

    def calculate_static_torques(self, joint_angles: torch.Tensor, force: torch.Tensor, 
                               point_of_application: torch.Tensor) -> torch.Tensor:
        """
        Calculate joint torques for a given force applied at a point.
        
        Args:
            joint_angles (torch.Tensor): Joint angles [batch_size, num_joints]
            force (torch.Tensor): Applied force [batch_size, 6]
            point_of_application (torch.Tensor): Point where force is applied [batch_size, 3]
            
        Returns:
            torch.Tensor: Joint torques [batch_size, num_joints]
        """
        
        # Get end effector pose
        pos, rot, _ = self.robot.get_fk(joint_angles, export_link=self.end_effector_link)
        
        # Calculate Jacobian
        J = self.robot.get_jacobian(joint_angles, self.end_effector_link, locations=point_of_application)
        
        # Calculate torques (J^T * F)
        torques = torch.bmm(J.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
        
        return torques
    
    def check_torque_limits(self, joint_torques: torch.Tensor) -> torch.Tensor:
        """
        Check if torques are within limits.
        
        Args:
            joint_torques (torch.Tensor): Joint torques [batch_size, num_joints]
            
        Returns:
            torch.Tensor: Boolean tensor indicating which configurations are valid
        """
        return torch.all(torch.abs(joint_torques) <= self.torque_limits, dim=1)
    
    def calculate_max_payload(self, joint_angles: torch.Tensor, force_direction: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate maximum payload for given joint configurations.
        
        Args:
            joint_angles (torch.Tensor): Joint angles to test [batch_size, num_joints]
            force_direction (torch.Tensor, optional): Direction of force [3], defaults to gravity
            
        Returns:
            torch.Tensor: Maximum payload for each configuration [batch_size]
        """
        if force_direction is None:
            force_direction = self.gravity.clone()
            
        # Normalize force direction
        force_direction = force_direction / torch.norm(force_direction)
        
        # Get end effector positions
        pos, rot, _ = self.robot.get_fk(joint_angles, export_link=self.end_effector_link)
        
        # Initialize payload search
        max_payloads = torch.zeros(joint_angles.shape[0], device=self.device)
        
        # Binary search parameters
        low = torch.zeros(joint_angles.shape[0], device=self.device)
        high = torch.ones(joint_angles.shape[0], device=self.device) * 100  # Start with 100 kg as max
        tolerance = 0.01  # 10g precision
        
        while torch.any(high - low > tolerance):
            # Test midpoint
            mid = (low + high) / 2
            
            # Calculate forces and torques (spatial force)
            forces = torch.zeros(mid.shape[0], 6, device=self.device)
            forces[:, :3] = force_direction.unsqueeze(0) * mid.unsqueeze(1)  # Linear forces
            # Torques are zero since we're only applying linear force
            
            # Calculate resulting torques
            torques = self.calculate_static_torques(joint_angles, forces, pos)
            
            # Check which configurations are valid
            valid = self.check_torque_limits(torques)
            
            # Update search bounds
            high = torch.where(valid, high, mid)
            low = torch.where(valid, mid, low)
            
            # Update max payloads for converged searches
            converged = (high - low) <= tolerance
            max_payloads = torch.where(converged, mid, max_payloads)
            
            # Break if all searches have converged
            if torch.all(converged):
                break
                
        return max_payloads
    
    def sample_workspace_max_payloads(self, num_samples: int = 1000) -> dict:
        """
        Sample maximum payloads across the robot's workspace.
        
        Args:
            num_samples (int): Number of random configurations to test
            
        Returns:
            dict: Dictionary containing results of the workspace sampling
        """
        # Sample random joint configurations
        joint_ranges = self.robot.joint_limit_max - self.robot.joint_limit_min
        random_joints = torch.rand(num_samples, len(self.robot.joint_list), device=self.device)
        joint_angles = random_joints * joint_ranges + self.robot.joint_limit_min
        
        # Calculate max payloads for all configurations
        max_payloads = self.calculate_max_payload(joint_angles)
        
        # Get end effector positions for visualization
        pos, rot, _ = self.robot.get_fk(joint_angles, export_link=self.end_effector_link)
        
        return {
            'joint_angles': joint_angles,
            'positions': pos,
            'max_payloads': max_payloads
        }

def main():
    """Example usage of MaxPayloadCalculator."""
    try:
        # Initialize calculator
        print("\n初始化最大负载计算器...")
        model_path = "rofunc/simulator/assets/mjcf/xianova_alicia_duo_v3/alicia_duo_mani_basic_v0_3_8.xml"
        calculator = MaxPayloadCalculator(model_path)
        
        print(f"\n关节扭矩限制:")
        for joint, torque in zip(calculator.robot.joint_list, calculator.joint_torque_limits):
            print(f"  {joint}: {torque} Nm")
            
        # Sample workspace
        print("\n采样工作空间计算最大负载...")
        results = calculator.sample_workspace_max_payloads(100)  # Reduced samples for testing
        
        # Print summary statistics
        payloads = results['max_payloads'].cpu().numpy()
        print(f"\n最大负载统计 (kg):")
        print(f"  最小值: {payloads.min():.2f}")
        print(f"  最大值: {payloads.max():.2f}")
        print(f"  平均值: {payloads.mean():.2f}")
        print(f"  标准差: {payloads.std():.2f}")
    
    except Exception as e:
        print(f"\n计算过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
