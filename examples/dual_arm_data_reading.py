#!/usr/bin/env python3
# coding=utf-8

"""
Joint Angle Reading Example For Dual Robotic Arms
Continuously read and display robotic arm joint angles, gripper angles, and button states.
"""

import os
import sys
import time
import logging
import serial
import serial.tools.list_ports
import threading

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alicia_duo_sdk.controller import ArmController

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SerialComm")

class AliciaDController:

    def __init__(self, specific_port: str = "", debug_mode: bool = False):
        self.specific_port = specific_port
        self.debug_mode = debug_mode
        self.controllers = []
        self.stop_reading = threading.Event() 

    def find_ports(self) -> list:
        """Find all available serial ports that match the criteria."""
        available_devices = []
        try:
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                if "ttyUSB" in port.device:
                    if os.access(port.device, os.R_OK | os.W_OK):
                        available_devices.append(port.device)
                        logger.info(f"Found available device: {port.device}")
                else:
                    logger.warning(f"Device {port.device} does not meet requirements, skipping")

        except Exception as e:
            logger.error(f"Exception when listing ports: {str(e)}")
            return []
        return available_devices

    def connect_all_robots(self):
        """Connect to all robotic arms"""
        devices = self.find_ports()
        if not devices:
            logger.error("No available serial devices found")
            return False
        
        for i, device in enumerate(devices):
            controller = ArmController(port=device, debug_mode=self.debug_mode)
            if controller.connect():
                self.controllers.append(controller)
                logger.info(f"Successfully connected to robot {i+1}: {device}")
            else:
                logger.error(f"Failed to connect to device {device}")
        
        return len(self.controllers) > 0

    def read_robot_data(self, controller, robot_id):
        """Read single robot data in a thread"""
        logger.info(f"Robot {robot_id} started reading data")
        try:
            while not self.stop_reading.is_set():
                state = controller.read_joint_state()
                joint_angles_deg = [round(angle * controller.RAD_TO_DEG, 2) for angle in state.angles]
                gripper_angle_deg = round(state.gripper * controller.RAD_TO_DEG, 2)
                
                print(f"Robot {robot_id} - Joint angles: {joint_angles_deg}")
                print(f"Robot {robot_id} - Gripper angle: {gripper_angle_deg}")
                print(f"Robot {robot_id} - Button states: {state.button1} {state.button2}")
                print("-" * 40)
                
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"Robot {robot_id} data reading error: {e}")

    def start_reading(self):
        """Start reading data from all robots"""
        if not self.controllers:
            logger.error("No connected robots")
            return
        
        print("Starting data reading...")
        print("Press Ctrl+C to exit")
        print("=" * 50)
        
        threads = []
        for i, controller in enumerate(self.controllers):
            thread = threading.Thread(
                target=self.read_robot_data,
                args=(controller, i + 1),
                name=f"Robot{i+1}_Reader"
            )
            threads.append(thread)
            thread.start()
        
        try:
            # Wait for user interruption
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n\nProgram stopped")
        finally:
            # Stop all threads
            self.stop_reading.set()
            for thread in threads:
                thread.join()

    def disconnect_all(self):
        """Disconnect all connections"""
        for i, controller in enumerate(self.controllers):
            controller.disconnect()
            logger.info(f"Robot {i+1} disconnected")

def main():
    """Main function"""
    print("=== Multi Robotic Arm Data Reading Example ===")
    
    dual_controller = AliciaDController(debug_mode=False)
    
    try:
        # Connect to all robotic arms
        if not dual_controller.connect_all_robots():
            print("Unable to connect to robotic arms, please check connections")
            return
        
        # Start reading data
        dual_controller.start_reading()
        
    except Exception as e:
        logger.error(f"Program execution error: {e}")
    finally:
        # Disconnect all connections
        dual_controller.disconnect_all()
        print("All connections disconnected")

if __name__ == "__main__":
    main()