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
import sys
from datetime import datetime

# Setup logging to both console and file
log_file = f"inference_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")


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

        logger.info("="*60)
        logger.info("InferenceClient initialized")
        logger.debug(f"  GPU IP: {self.gpu_ip}")
        logger.debug(f"  GPU action port: {self.gpu_action_port}")
        logger.debug(f"  GPU qpos port: {self.gpu_qpos_port}")
        logger.debug(f"  Spot server URL: {self.spot_server_url}")
        logger.info("="*60)

    def connect_to_robot(self):
        """Connect to Spot server"""
        logger.info(f"Connecting to Spot server at {self.spot_server_url}...")
        try:
            logger.debug("Sending GET request to /get_qpos endpoint...")
            # Test connection to Spot server
            start_time = time.time()
            response = requests.get(f"{self.spot_server_url}/get_qpos", timeout=5)
            elapsed = time.time() - start_time
            logger.debug(f"Response received in {elapsed:.3f}s")
            logger.debug(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                logger.debug(f"Response content: {response.text[:200]}")
                logger.info("✓ Connected to Spot server")
            else:
                raise RuntimeError(f"Spot server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Spot server: {e}", exc_info=True)
            raise

    def connect_to_gpu(self):
        """Connect to GPU server"""
        logger.info(f"Connecting to GPU at {self.gpu_ip}:{self.gpu_action_port}...")

        try:
            # Action socket (receive from GPU)
            logger.debug(f"Creating socket for GPU action port...")
            self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.debug(f"Connecting action socket to {self.gpu_ip}:{self.gpu_action_port}...")
            start_time = time.time()
            self.action_socket.connect((self.gpu_ip, self.gpu_action_port))
            elapsed = time.time() - start_time
            logger.debug(f"Action socket connected in {elapsed:.3f}s")
            logger.info(f"✓ Connected to GPU action port")

            # Qpos socket (send to GPU)
            logger.debug(f"Creating socket for GPU qpos port...")
            self.qpos_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.debug(f"Connecting qpos socket to {self.gpu_ip}:{self.gpu_qpos_port}...")
            start_time = time.time()
            self.qpos_socket.connect((self.gpu_ip, self.gpu_qpos_port))
            elapsed = time.time() - start_time
            logger.debug(f"Qpos socket connected in {elapsed:.3f}s")
            logger.info(f"✓ Connected to GPU qpos port")
        except Exception as e:
            logger.error(f"Failed to connect to GPU: {e}", exc_info=True)
            raise

    def get_qpos_from_robot(self) -> np.ndarray:
        """
        Get current qpos from Spot server.

        Returns:
            [11,] numpy array: [6 arm joints, 1 gripper, 3 body position, 1 body pitch]
        """
        try:
            logger.debug("Requesting qpos from Spot server...")
            start_time = time.time()
            response = requests.get(f"{self.spot_server_url}/get_qpos", timeout=5)
            elapsed = time.time() - start_time
            logger.debug(f"Response received in {elapsed:.3f}s, status code: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Response data: {data}")

            if data.get("status") != "ok":
                raise RuntimeError(f"Spot server error: {data.get('message', 'Unknown error')}")

            qpos = np.array(data["qpos"], dtype=np.float64)
            logger.debug(f"Parsed qpos: {qpos}")
            return qpos
        except Exception as e:
            logger.error(f"Failed to get qpos from Spot server: {e}", exc_info=True)
            raise

    def reset_robot(self):
        """Reset robot to safe position"""
        logger.info("Resetting robot...")
        try:
            logger.debug("Sending POST request to /reset_robot endpoint...")
            start_time = time.time()
            response = requests.post(f"{self.spot_server_url}/reset_robot", timeout=10)
            elapsed = time.time() - start_time
            logger.debug(f"Response received in {elapsed:.3f}s, status code: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Response data: {data}")

            if data.get("status") != "ok":
                raise RuntimeError(f"Spot server error: {data.get('message', 'Unknown error')}")

            logger.info("✓ Robot reset")
        except Exception as e:
            logger.error(f"Failed to reset robot: {e}", exc_info=True)
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
        try:
            logger.debug(f"Preparing to send qpos: {qpos}")
            qpos_array = np.array(qpos, dtype=np.float64)
            logger.debug(f"Array shape: {qpos_array.shape}, dtype: {qpos_array.dtype}")
            qpos_bytes = qpos_array.tobytes()
            logger.debug(f"Converting to bytes: {len(qpos_bytes)} bytes")
            start_time = time.time()
            self.qpos_socket.send(qpos_bytes)
            elapsed = time.time() - start_time
            logger.debug(f"Sent qpos in {elapsed:.3f}s: [{float(qpos[0]):.4f}, {float(qpos[1]):.4f}, {float(qpos[2]):.4f}, {float(qpos[3]):.4f}, {float(qpos[4]):.4f}, {float(qpos[5]):.4f}, ...]")
        except Exception as e:
            logger.error(f"Failed to send qpos to GPU: {e}", exc_info=True)
            raise

    def execute_action(self, action: np.ndarray):
        """
        Execute action on Spot.

        Args:
            action: [11,] numpy array
                [6 arm joints, 1 gripper, 1 body z, 2 body velocities, 1 body pitch]
        """
        try:
            logger.debug(f"Executing action on Spot: {action}")
            payload = {"action": action.tolist()}
            logger.debug(f"Payload: {payload}")
            start_time = time.time()
            response = requests.post(f"{self.spot_server_url}/execute_action", json=payload, timeout=5)
            elapsed = time.time() - start_time
            logger.debug(f"Response received in {elapsed:.3f}s, status code: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Response data: {data}")

            if data.get("status") != "ok":
                raise RuntimeError(f"Spot server error: {data.get('message', 'Unknown error')}")
            logger.debug("✓ Action executed successfully")
        except Exception as e:
            logger.error(f"Failed to execute action: {e}", exc_info=True)
            raise

    def wait_for_motion_complete(self,
                                 target_qpos: np.ndarray,
                                 position_tolerance: float = 0.05,
                                 velocity_tolerance: float = 0.1,
                                 max_wait_time: float = 5.0,
                                 check_interval: float = 0.1,
                                 send_qpos_feedback: bool = False) -> bool:
        """
        Wait for robot to reach target position and stop moving.

        Args:
            target_qpos: Target joint positions [11,]
            position_tolerance: How close joints need to be to target (radians)
            velocity_tolerance: How slow joints need to be moving (rad/s)
            max_wait_time: Maximum time to wait (seconds)
            check_interval: How often to check robot state (seconds)
            send_qpos_feedback: If True, send qpos to GPU during motion wait (for inference sync)

        Returns:
            True if motion completed within max_wait_time, False if timeout
        """
        try:
            logger.info(f"Waiting for motion to complete (tolerance={position_tolerance:.4f}rad, timeout={max_wait_time}s)")
            logger.debug(f"Target qpos: {target_qpos}")
            start_time = time.time()
            last_qpos = target_qpos.copy()
            settled_count = 0
            settled_threshold = 3  # Number of consecutive checks showing stillness
            check_count = 0

            while time.time() - start_time < max_wait_time:
                check_count += 1
                # Get current joint positions
                current_qpos = self.get_qpos_from_robot()

                # Send qpos feedback to GPU if requested (keeps inference loop synchronized)
                if send_qpos_feedback:
                    self.send_qpos(current_qpos)
                    elapsed = time.time() - start_time
                    logger.debug(f"  Sent qpos feedback at {elapsed:.3f}s: [{float(current_qpos[0]):.4f}, {float(current_qpos[1]):.4f}, {float(current_qpos[2]):.4f}, ...]")

                # Calculate position error
                position_error = np.abs(current_qpos - target_qpos)
                max_position_error = np.max(position_error)

                # Calculate velocity estimate (derivative of position)
                velocity_estimate = np.abs(current_qpos - last_qpos) / check_interval
                max_velocity = np.max(velocity_estimate)

                # Check if settled
                if max_position_error < position_tolerance and max_velocity < velocity_tolerance:
                    settled_count += 1
                    logger.debug(f"Check {check_count}: SETTLED (count={settled_count}/{settled_threshold}), error={max_position_error:.4f}rad, vel={max_velocity:.4f}rad/s")
                    if settled_count >= settled_threshold:
                        elapsed = time.time() - start_time
                        logger.info(f"✓ Motion complete (settled after {elapsed:.2f}s)")
                        logger.info(f"  Position error: {max_position_error:.4f} rad | "
                                  f"Max velocity: {max_velocity:.4f} rad/s | "
                                  f"Total checks: {check_count}")
                        return True
                else:
                    settled_count = 0  # Reset counter if motion detected
                    elapsed = time.time() - start_time
                    logger.debug(f"Check {check_count}: MOVING... error={max_position_error:.4f}rad, "
                               f"vel={max_velocity:.4f}rad/s, elapsed={elapsed:.2f}s, "
                               f"target={target_qpos[:3]}, current={current_qpos[:3]}")

                last_qpos = current_qpos.copy()
                time.sleep(check_interval)

            # Timeout
            elapsed = time.time() - start_time
            current_qpos = self.get_qpos_from_robot()
            position_error = np.abs(current_qpos - target_qpos)
            logger.warning(f"Motion timeout after {elapsed:.2f}s (max {max_wait_time}s allowed)")
            logger.warning(f"  Final position error: {np.max(position_error):.4f} rad")
            logger.warning(f"  Total checks performed: {check_count}")
            logger.warning(f"  Target qpos: {target_qpos}")
            logger.warning(f"  Final qpos: {current_qpos}")
            return False

        except Exception as e:
            logger.error(f"Failed to wait for motion: {e}", exc_info=True)
            raise

    def run(self, num_episodes: int = 5, max_steps: int = 500):
        """
        Main execution loop with 20 Hz timing synchronization.

        Maintains strict 50ms intervals for qpos feedback to GPU server.
        Actions execute asynchronously - next action is fetched immediately
        rather than waiting for motion to complete.

        Args:
            num_episodes: Number of episodes to run
            max_steps: Max steps per episode
        """
        try:
            logger.info(f"Starting execution loop: {num_episodes} episodes, {max_steps} steps max")
            total_start_time = time.time()

            for episode_idx in range(num_episodes):
                logger.info(f"\n{'='*60}")
                logger.info(f"Episode {episode_idx + 1}/{num_episodes}")
                logger.info(f"{'='*60}")

                # Reset robot
                try:
                    self.reset_robot()
                except Exception as e:
                    logger.error(f"Failed to reset robot in episode {episode_idx + 1}: {e}", exc_info=True)
                    raise

                # Get initial qpos
                try:
                    qpos = self.get_qpos_from_robot()
                    logger.info(f"Initial qpos: {qpos}")
                except Exception as e:
                    logger.error(f"Failed to get initial qpos: {e}", exc_info=True)
                    raise

                # Send reset signal to GPU
                try:
                    logger.info("Sending reset signal to GPU...")
                    self.send_qpos(qpos)
                    logger.debug("Reset signal sent successfully")
                except Exception as e:
                    logger.error(f"Failed to send reset signal to GPU: {e}", exc_info=True)
                    raise

                episode_start_time = time.time()
                step_errors = 0
                target_hz = 20
                target_dt = 1.0 / target_hz  # 0.05s = 50ms

                # Track for reset detection
                prev_qpos = qpos.copy()
                reset_threshold = 0.5  # radians

                for step_idx in range(max_steps):
                    step_loop_start = time.time()
                    try:
                        # Log every step for reset tracking
                        if step_idx % 10 == 0:
                            logger.info(f"Step {step_idx:3d}/{max_steps} - Executing action...")
                        logger.debug(f"\n--- Step {step_idx} ---")

                        # Receive action from GPU (blocking socket recv)
                        try:
                            recv_start = time.time()
                            action = self.receive_action()
                            recv_elapsed = (time.time() - recv_start) * 1000
                            logger.debug(f"Action received in {recv_elapsed:.1f}ms: {action[:3]}...")
                        except Exception as e:
                            logger.error(f"Step {step_idx}: Failed to receive action from GPU: {e}", exc_info=True)
                            raise

                        # Execute on Spot (HTTP request, async - don't wait for completion)
                        logger.info(f"Step {step_idx}: Executing action...")
                        try:
                            exec_start = time.time()
                            self.execute_action(action)
                            exec_elapsed = (time.time() - exec_start) * 1000
                            logger.debug(f"Action execution queued in {exec_elapsed:.1f}ms")
                        except Exception as e:
                            logger.error(f"Step {step_idx}: Failed to execute action: {e}", exc_info=True)
                            raise

                        # Wait for motion to complete while sending qpos feedback at 20 Hz
                        try:
                            qpos_start = time.time()
                            target_qpos = self.get_qpos_from_robot()
                            qpos_elapsed = (time.time() - qpos_start) * 1000
                            logger.debug(f"Got target qpos in {qpos_elapsed:.1f}ms")

                            # Wait for motion with continuous 20 Hz qpos feedback
                            motion_wait_start = time.time()
                            motion_complete = self.wait_for_motion_complete(
                                target_qpos=target_qpos,
                                position_tolerance=0.05,
                                velocity_tolerance=0.05,
                                max_wait_time=5.0,
                                check_interval=target_dt,  # 50ms = 20 Hz
                                send_qpos_feedback=True
                            )
                            motion_wait_elapsed = (time.time() - motion_wait_start) * 1000

                            if not motion_complete:
                                logger.warning(f"Step {step_idx}: Motion did not complete in {motion_wait_elapsed:.1f}ms. "
                                             "Proceeding anyway for next inference.")
                        except Exception as e:
                            logger.error(f"Step {step_idx}: Failed during motion waiting: {e}", exc_info=True)
                            raise

                        # Get final qpos for next step start
                        try:
                            qpos = self.get_qpos_from_robot()
                            logger.debug(f"Step {step_idx} complete. Final qpos: {qpos[:3]}...")
                        except Exception as e:
                            logger.error(f"Step {step_idx}: Failed to get final qpos: {e}", exc_info=True)
                            raise

                        if step_idx % 10 == 0:
                            elapsed = time.time() - episode_start_time
                            logger.info(f"Step {step_idx:3d}/{max_steps} | "
                                      f"Elapsed: {elapsed:.1f}s | "
                                      f"Qpos: [{qpos[0]:.4f}, {qpos[1]:.4f}, {qpos[2]:.4f}, ...]")

                        # Check for 100-step boundaries
                        if step_idx > 0 and step_idx % 100 == 0:
                            logger.warning(f"\n⚠️  MILESTONE: Step {step_idx} reached (100-step boundary)")

                    except Exception as e:
                        step_errors += 1
                        logger.error(f"Step {step_idx} failed: {e}", exc_info=True)
                        if step_errors > 3:
                            logger.critical(f"Too many step errors ({step_errors}), aborting episode")
                            raise

                episode_duration = time.time() - episode_start_time
                logger.info(f"Episode {episode_idx + 1} complete. Duration: {episode_duration:.1f}s, Errors: {step_errors}")

            total_elapsed = time.time() - total_start_time
            logger.info(f"\n{'='*60}")
            logger.info("All episodes complete!")
            logger.info(f"Total execution time: {total_elapsed:.1f}s")
            logger.info(f"{'='*60}")

        except KeyboardInterrupt:
            logger.info("\nExecution stopped by user")

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            raise

        finally:
            logger.info("Shutting down...")
            try:
                if self.action_socket:
                    logger.debug("Closing action socket...")
                    self.action_socket.close()
                    logger.debug("Action socket closed")
            except Exception as e:
                logger.error(f"Error closing action socket: {e}", exc_info=True)

            try:
                if self.qpos_socket:
                    logger.debug("Closing qpos socket...")
                    self.qpos_socket.close()
                    logger.debug("Qpos socket closed")
            except Exception as e:
                logger.error(f"Error closing qpos socket: {e}", exc_info=True)

            # Power off Spot via server
            try:
                logger.info("Powering off robot...")
                response = requests.post(f"{self.spot_server_url}/power_off", timeout=5)
                logger.debug(f"Power off response: {response.status_code}")
                logger.info("Robot powered off")
            except Exception as e:
                logger.error(f"Failed to power off robot: {e}", exc_info=True)


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

        # Connect to robot
        logger.info("\nConnecting to Spot robot...")
        try:
            client.connect_to_robot()
        except Exception as e:
            logger.critical(f"Failed to connect to Spot robot: {e}", exc_info=True)
            raise

        # Connect to GPU
        logger.info("\nConnecting to GPU server...")
        try:
            client.connect_to_gpu()
        except Exception as e:
            logger.critical(f"Failed to connect to GPU: {e}", exc_info=True)
            raise

        logger.info("\n" + "="*60)
        logger.info("CONNECTED TO BOTH SPOT SERVER AND GPU")
        logger.info("="*60)
        logger.info("Starting inference execution loop...\n")

    # Run
    client.run(num_episodes=args.num_episodes, max_steps=args.max_steps, hz=20)


if __name__ == '__main__':
    main()

    #  python inference_client.py --gpu-ip 10.45.1.18 --spot-server-url http://10.45.6.171:5001 --num-episodes 1 --max-steps 600
