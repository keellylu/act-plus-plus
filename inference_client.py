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
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceClient:
    """Client that executes actions from GPU server on real Spot robot"""

    def __init__(self, gpu_ip: str, spot_server_url: str, gpu_action_port: int = 9999, gpu_qpos_port: int = 9998):
        """
        Initialize the inference client.

        Args:
            gpu_ip: GPU server IP address
            spot_server_url: Spot server Flask endpoint (e.g., http://192.168.80.3:5001)
            gpu_action_port: Port to receive actions from GPU
            gpu_qpos_port: Port to send qpos to GPU
        """
        self.gpu_ip = gpu_ip
        self.spot_server_url = spot_server_url
        self.gpu_action_port = gpu_action_port
        self.gpu_qpos_port = gpu_qpos_port

        # Network sockets (GPU communication)
        self.action_socket = None
        self.qpos_socket = None

        logger.info("InferenceClient initialized")

    def connect_to_robot(self):
        """Connect to Spot server"""
        logger.info(f"Connecting to Spot server at {self.spot_server_url}...")
        try:
            # Test connection to Spot server
            response = requests.get(f"{self.spot_server_url}/get_qpos", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Connected to Spot server")
            else:
                raise RuntimeError(f"Spot server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Spot server: {e}")
            raise

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
        Get current qpos from Spot server.

        Returns:
            [11,] numpy array: [6 arm joints, 1 gripper, 3 body position, 1 body pitch]
        """
        try:
            response = requests.get(f"{self.spot_server_url}/get_qpos", timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise RuntimeError(f"Spot server error: {data.get('message', 'Unknown error')}")

            qpos = np.array(data["qpos"], dtype=np.float64)
            return qpos
        except Exception as e:
            logger.error(f"Failed to get qpos from Spot server: {e}")
            raise

    def reset_robot(self):
        """Reset robot to safe position"""
        logger.info("Resetting robot...")
        try:
            response = requests.post(f"{self.spot_server_url}/reset_robot", timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise RuntimeError(f"Spot server error: {data.get('message', 'Unknown error')}")

            logger.info("✓ Robot reset")
        except Exception as e:
            logger.error(f"Failed to reset robot: {e}")
            raise

    def receive_action(self, timeout: float = 10.0) -> np.ndarray:
        """
        Receive action from GPU server.

        Args:
            timeout: Timeout in seconds for receiving action

        Returns:
            [11,] numpy array
        """
        expected_bytes = 88  # 11 * 8 bytes
        action_bytes = b''
        
        # Set timeout
        self.action_socket.settimeout(timeout)
        
        try:
            # Keep receiving until we have all bytes
            while len(action_bytes) < expected_bytes:
                chunk = self.action_socket.recv(expected_bytes - len(action_bytes))
                if not chunk:
                    raise RuntimeError("Connection closed by GPU server - GPU may have crashed or errored. Check GPU server logs.")
                action_bytes += chunk
            
            if len(action_bytes) != expected_bytes:
                raise RuntimeError(f"Incomplete action data received: got {len(action_bytes)} bytes, expected {expected_bytes}")

            return np.frombuffer(action_bytes, dtype=np.float64)
        except socket.timeout:
            raise RuntimeError(f"Timeout waiting for action from GPU server (>{timeout}s). GPU may be processing or crashed. Check GPU server logs.")
        finally:
            # Reset to blocking
            self.action_socket.settimeout(None)

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
                [6 arm joints, 1 gripper, 1 body z, 2 body velocities, 1 body pitch]
        """
        try:
            payload = {"action": action.tolist()}
            response = requests.post(f"{self.spot_server_url}/execute_action", json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise RuntimeError(f"Spot server error: {data.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            raise

    def wait_for_motion_complete(self,
                                 target_qpos: np.ndarray,
                                 position_tolerance: float = 0.05,
                                 velocity_tolerance: float = 0.1,
                                 max_wait_time: float = 5.0,
                                 check_interval: float = 0.1) -> bool:
        """
        Wait for robot to reach target position and stop moving.

        Args:
            target_qpos: Target joint positions [11,]
            position_tolerance: How close joints need to be to target (radians)
            velocity_tolerance: How slow joints need to be moving (rad/s)
            max_wait_time: Maximum time to wait (seconds)
            check_interval: How often to check robot state (seconds)

        Returns:
            True if motion completed within max_wait_time, False if timeout
        """
        try:
            start_time = time.time()
            last_qpos = target_qpos.copy()
            settled_count = 0
            settled_threshold = 3  # Number of consecutive checks showing stillness

            while time.time() - start_time < max_wait_time:
                # Get current joint positions
                current_qpos = self.get_qpos_from_robot()

                # Calculate position error
                position_error = np.abs(current_qpos - target_qpos)
                max_position_error = np.max(position_error)

                # Calculate velocity estimate (derivative of position)
                velocity_estimate = np.abs(current_qpos - last_qpos) / check_interval
                max_velocity = np.max(velocity_estimate)

                # Check if settled
                if max_position_error < position_tolerance and max_velocity < velocity_tolerance:
                    settled_count += 1
                    if settled_count >= settled_threshold:
                        elapsed = time.time() - start_time
                        logger.info(f"✓ Motion complete (settled after {elapsed:.2f}s)")
                        logger.info(f"  Position error: {max_position_error:.4f} rad | "
                                  f"Max velocity: {max_velocity:.4f} rad/s")
                        return True
                else:
                    settled_count = 0  # Reset counter if motion detected
                    elapsed = time.time() - start_time
                    logger.debug(f"Moving... error={max_position_error:.4f} rad, "
                               f"vel={max_velocity:.4f} rad/s, elapsed={elapsed:.2f}s")

                last_qpos = current_qpos.copy()
                time.sleep(check_interval)

            # Timeout
            elapsed = time.time() - start_time
            current_qpos = self.get_qpos_from_robot()
            position_error = np.abs(current_qpos - target_qpos)
            logger.warning(f"Motion timeout after {elapsed:.2f}s")
            logger.warning(f"  Final position error: {np.max(position_error):.4f} rad")
            return False

        except Exception as e:
            logger.error(f"Failed to wait for motion: {e}")
            raise

    def run(self, num_episodes: int = 5, max_steps: int = 500, hz: int = 20):
        """
        Main execution loop.

        Args:
            num_episodes: Number of episodes to run
            max_steps: Max steps per episode
            hz: Execution frequency in Hz (default 20 Hz = 50ms per step)
        """
        step_duration = 1.0 / hz  # Duration per step in seconds

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
                next_step_time = episode_start_time

                for step_idx in range(max_steps):
                    # Receive action from GPU
                    action = self.receive_action()

                    # Execute on Spot
                    logger.info(f"Step {step_idx}: Executing action...")
                    self.execute_action(action)

                    # Wait for robot to complete motion before next inference
                    # Use current qpos as target since the action drives toward it
                    target_qpos = self.get_qpos_from_robot()
                    motion_complete = self.wait_for_motion_complete(
                        target_qpos=target_qpos,
                        position_tolerance=0.05,
                        velocity_tolerance=0.05,
                        max_wait_time=5.0
                    )

                    if not motion_complete:
                        logger.warning(f"Step {step_idx}: Motion did not complete in time. "
                                     "Proceeding anyway for next inference.")

                    # Now that robot has stopped, get final qpos and images
                    qpos = self.get_qpos_from_robot()

                    # Send qpos to GPU for next inference
                    self.send_qpos(qpos)

                    if step_idx % 10 == 0:
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
            # Power off Spot via server
            try:
                requests.post(f"{self.spot_server_url}/power_off", timeout=5)
            except Exception as e:
                logger.error(f"Failed to power off robot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Mac Client for ACT++ Spot Inference'
    )
    parser.add_argument('--gpu-ip', type=str, required=True,
                       help='GPU server IP address')
    parser.add_argument('--robot-ip', type=str, default='192.168.80.3',
                       help='Spot robot IP address (used to construct server URL)')
    parser.add_argument('--spot-server-url', type=str, default=None,
                       help='Spot server URL (e.g., http://10.45.7.35:5001). If not provided, will use http://{robot-ip}:5001')
    parser.add_argument('--action-port', type=int, default=9999,
                       help='Port to receive actions from GPU')
    parser.add_argument('--qpos-port', type=int, default=9998,
                       help='Port to send qpos to GPU')
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode')

    args = parser.parse_args()

    # Construct server URL
    if args.spot_server_url:
        spot_server_url = args.spot_server_url
    else:
        spot_server_url = f"http://{args.robot_ip}:5001"

    logger.info(f"Spot server URL: {spot_server_url}")

    # Create client
    client = InferenceClient(
        gpu_ip=args.gpu_ip,
        spot_server_url=spot_server_url,
        gpu_action_port=args.action_port,
        gpu_qpos_port=args.qpos_port
    )

    # Connect
    client.connect_to_robot()
    client.connect_to_gpu()

    logger.info("\n" + "="*60)
    logger.info("CONNECTED TO BOTH SPOT SERVER AND GPU")
    logger.info("="*60)
    logger.info("Starting inference execution loop...\n")

    # Run
    client.run(num_episodes=args.num_episodes, max_steps=args.max_steps, hz=20)


if __name__ == '__main__':
    main()

    #  python inference_client.py --gpu-ip 10.45.1.18 --spot-server-url http://10.45.6.171:5001 --num-episodes 1 --max-steps 600
