#!/usr/bin/env python3
"""
Mac Client for Distributed ACT++ Inference

This client:
1. Connects to Spot robot
2. Connects to GPU server (receives actions, sends qpos)
3. Executes actions on Spot
4. Sends qpos feedback to GPU

Run on Mac machine:
    python inference_client.py \
        --gpu-ip <GPU_MACHINE_IP> \
        --robot-ip 192.168.80.3
"""

import argparse
import socket
import numpy as np
import time
import logging
from pathlib import Path

from bosdyn.client.robot import Robot
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.frame_helpers import BODY_FRAME_NAME

# Try to import Spot utilities if available
try:
    from spot_utils import authenticate
    from spot_lease import LeaseClient, LeaseKeepAlive
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("spot_utils not found - using basic authentication")
    def authenticate(robot):
        pass
    def LeaseKeepAlive(*args, **kwargs):
        return None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceClient:
    """Client that executes actions from GPU server on real Spot robot"""

    def __init__(self, robot_hostname: str, gpu_ip: str, gpu_action_port: int = 9999, gpu_qpos_port: int = 9998):
        """
        Initialize the inference client.

        Args:
            robot_hostname: Spot robot IP/hostname
            gpu_ip: GPU server IP address
            gpu_action_port: Port to receive actions from GPU
            gpu_qpos_port: Port to send qpos to GPU
        """
        self.robot_hostname = robot_hostname
        self.gpu_ip = gpu_ip
        self.gpu_action_port = gpu_action_port
        self.gpu_qpos_port = gpu_qpos_port

        # Robot connection
        self.robot = None
        self.command_client = None
        self.state_client = None
        self.lease_client = None
        self.lease_keepalive = None

        # Network sockets
        self.action_socket = None
        self.qpos_socket = None

        logger.info("InferenceClient initialized")

    def connect_to_robot(self):
        """Connect to Spot robot"""
        logger.info(f"Connecting to Spot at {self.robot_hostname}...")

        sdk = create_standard_sdk("spot_inference_client")
        self.robot = sdk.create_robot(self.robot_hostname)
        authenticate(self.robot)

        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.state_client = self.robot.ensure_client('robot-state')

        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.lease_client.take()
        self.lease_keepalive = LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True
        )

        self.robot.time_sync.wait_for_sync()

        logger.info("✓ Connected to Spot robot")

    def connect_to_gpu(self):
        """Connect to GPU server"""
        logger.info(f"Connecting to GPU at {self.gpu_ip}:{self.gpu_action_port}...")

        # Action socket (receive from GPU)
        self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.action_socket.connect((self.gpu_ip, self.gpu_action_port))
        logger.info(f"✓ Connected to GPU action port")

        # Qpos socket (send to GPU)
        self.qpos_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.qpos_socket.connect((self.gpu_ip, self.gpu_qpos_port))
        logger.info(f"✓ Connected to GPU qpos port")

    def get_qpos_from_robot(self) -> np.ndarray:
        """
        Get current qpos from Spot.

        Returns:
            [11,] numpy array: [6 arm joints, 1 gripper, 4 body state]
        """
        state = self.state_client.get_robot_state()
        arm_state = state.manipulator_state
        mobility_state = state.kinematic_state

        # Arm joints (6)
        q_arm = np.array(arm_state.position[:6], dtype=np.float64)

        # Gripper (1)
        gripper_open_fraction = arm_state.gripper_open_percentage / 100.0 if arm_state.gripper_open_percentage is not None else 0.5
        q_gripper = np.array([gripper_open_fraction], dtype=np.float64)

        # Body state (4)
        body_frame = mobility_state.transforms_snapshot.child_to_parent_edge_map["body"]
        pos = body_frame.parent_tform_child.position
        rot = body_frame.parent_tform_child.rotation
        pitch = rot.pitch
        q_body = np.array([pos.x, pos.y, pos.z, pitch], dtype=np.float64)

        # Concatenate
        qpos = np.concatenate([q_arm, q_gripper, q_body])

        return qpos

    def reset_robot(self):
        """Reset robot to safe position"""
        logger.info("Resetting robot...")

        # Stow arm
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(stow_cmd)
        time.sleep(1.0)

        logger.info("✓ Robot reset")

    def receive_action(self) -> np.ndarray:
        """
        Receive action from GPU server.

        Returns:
            [11,] numpy array
        """
        action_bytes = self.action_socket.recv(88)  # 11 * 8 bytes
        if len(action_bytes) < 88:
            raise RuntimeError("Incomplete action data received")

        return np.frombuffer(action_bytes, dtype=np.float64)

    def send_qpos(self, qpos: np.ndarray):
        """
        Send qpos to GPU server.

        Args:
            qpos: [11,] numpy array
        """
        qpos_bytes = np.array(qpos, dtype=np.float64).tobytes()
        self.qpos_socket.send(qpos_bytes)

    def execute_action(self, action: np.ndarray):
        """
        Execute action on Spot.

        Args:
            action: [11,] numpy array
                [6 arm joints, 1 gripper, 4 body params]
        """
        arm_q = action[:6]
        gripper = action[6]
        body_x, body_y, body_z, body_pitch = action[7:]

        # Build arm command
        arm_cmd = RobotCommandBuilder.arm_joint_move_command(
            joint_targets=list(arm_q),
            max_vel=1.0,
            max_acc=1.0
        )

        # Build gripper command
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper)

        # Build body command
        body_cmd = RobotCommandBuilder.body_pose_command(
            x=body_x,
            y=body_y,
            z=body_z,
            roll=0.0,
            pitch=body_pitch,
            yaw=0.0,
            frame_name=BODY_FRAME_NAME
        )

        # Synchronize and send
        full_cmd = RobotCommandBuilder.build_synchro_command(
            arm_cmd,
            gripper_cmd.synchronized_command.gripper_command,
            body_cmd
        )

        self.command_client.robot_command(full_cmd)

    def run(self, num_episodes: int = 5, max_steps: int = 500):
        """
        Main execution loop.

        Args:
            num_episodes: Number of episodes to run
            max_steps: Max steps per episode
        """
        try:
            for episode_idx in range(num_episodes):
                logger.info(f"\n{'='*60}")
                logger.info(f"Episode {episode_idx + 1}/{num_episodes}")
                logger.info(f"{'='*60}")

                # Reset robot
                self.reset_robot()

                # Get initial qpos
                qpos = self.get_qpos_from_robot()
                logger.info(f"Initial qpos: {qpos[:3]}...")

                # Send reset signal to GPU
                logger.info("Sending reset signal to GPU...")
                self.send_qpos(qpos)

                episode_start_time = time.time()

                for step_idx in range(max_steps):
                    # Receive action from GPU
                    action = self.receive_action()

                    # Execute on Spot
                    self.execute_action(action)

                    # Get current qpos
                    qpos = self.get_qpos_from_robot()

                    # Send back to GPU
                    self.send_qpos(qpos)

                    if step_idx % 50 == 0:
                        elapsed = time.time() - episode_start_time
                        logger.info(f"Step {step_idx:3d}/{max_steps} | "
                                  f"Elapsed: {elapsed:.1f}s | "
                                  f"Qpos: [{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, ...]")

                episode_duration = time.time() - episode_start_time
                logger.info(f"Episode complete. Duration: {episode_duration:.1f}s")

            logger.info(f"\n{'='*60}")
            logger.info("All episodes complete!")
            logger.info(f"{'='*60}")

        except KeyboardInterrupt:
            logger.info("\nExecution stopped by user")

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)

        finally:
            logger.info("Shutting down...")
            if self.action_socket:
                self.action_socket.close()
            if self.qpos_socket:
                self.qpos_socket.close()
            if self.robot:
                self.robot.power_off()


def main():
    parser = argparse.ArgumentParser(
        description='Mac Client for ACT++ Spot Inference'
    )
    parser.add_argument('--gpu-ip', type=str, required=True,
                       help='GPU server IP address')
    parser.add_argument('--robot-ip', type=str, default='192.168.80.3',
                       help='Spot robot IP address')
    parser.add_argument('--action-port', type=int, default=9999,
                       help='Port to receive actions from GPU')
    parser.add_argument('--qpos-port', type=int, default=9998,
                       help='Port to send qpos to GPU')
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode')

    args = parser.parse_args()

    # Create client
    client = InferenceClient(
        robot_hostname=args.robot_ip,
        gpu_ip=args.gpu_ip,
        gpu_action_port=args.action_port,
        gpu_qpos_port=args.qpos_port
    )

    # Connect
    client.connect_to_robot()
    client.connect_to_gpu()

    logger.info("\n" + "="*60)
    logger.info("CONNECTED TO BOTH SPOT AND GPU")
    logger.info("="*60)
    logger.info("Starting inference execution loop...\n")

    # Run
    client.run(num_episodes=args.num_episodes, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
