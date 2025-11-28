#!/usr/bin/env python3
"""
GPU-Side Environment for Distributed ACT++ Inference

This environment:
1. Streams cameras locally (ZED + Kiwi)
2. Receives qpos from Mac via network
3. Sends actions to Mac via network
4. Manages the policy inference loop

No direct connection to Spot robot - that's handled by inference_client.py on Mac.
"""

import time
import threading
import socket
import numpy as np
import cv2
import logging
import sys
import os
import signal
from collections import deque
from typing import Optional, Dict

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


class CameraBuffer:
    """Thread-safe circular buffer for camera frames"""
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = None

    def append(self, frame: np.ndarray, timestamp: float):
        """Add timestamped frame to buffer"""
        with self.lock:
            self.buffer.append((frame.copy(), timestamp))
            self.latest_frame = frame.copy()
            self.latest_timestamp = timestamp

    def get_latest(self) -> Optional[np.ndarray]:
        """Get most recent frame"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """Get frame closest to target_time (nearest neighbor)"""
        with self.lock:
            if not self.buffer:
                return None
            closest = min(self.buffer, key=lambda x: abs(x[1] - target_time))
            return closest[0].copy()


def zed_streaming_worker(camera_buffer: CameraBuffer, stop_event: threading.Event):
    """Stream ZED frames locally to buffer"""
    try:
        from utils_pkg.zed import stream_zed_frames
        logger.info("Starting ZED camera stream...")

        frame_count = 0
        for rgb_bgr, depth in stream_zed_frames():
            if stop_event.is_set():
                break

            # Validate ZED image format
            if len(rgb_bgr.shape) != 3 or rgb_bgr.shape[2] != 3:
                logger.error(f"[ZED] Invalid image shape: {rgb_bgr.shape}. Expected (H, W, 3)")
                continue

            # Convert BGR to RGB
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            # Debug: Log first frame
            if frame_count == 0:
                logger.info(f"[ZED Buffer] First frame stored: shape={rgb.shape}, dtype={rgb.dtype}, channels={rgb.shape[2] if len(rgb.shape) == 3 else 'N/A'}")

            timestamp = time.time()
            camera_buffer.append(rgb, timestamp)
            frame_count += 1

    except ImportError as e:
        logger.warning(f"ZED camera dependencies not available: {e}")
        logger.warning("Install ZED SDK Python API: https://www.stereolabs.com/docs/installation/")
        logger.warning("ZED camera will not be available. Using placeholder images.")
        # Provide placeholder images so inference can continue
        while not stop_event.is_set():
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            timestamp = time.time()
            camera_buffer.append(placeholder, timestamp)
            time.sleep(0.033)  # ~30 FPS
    except Exception as e:
        logger.error(f"ZED streaming error: {e}")
        logger.warning("ZED camera will not be available. Using placeholder images.")
        # Provide placeholder images so inference can continue
        while not stop_event.is_set():
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            timestamp = time.time()
            camera_buffer.append(placeholder, timestamp)
            time.sleep(0.033)  # ~30 FPS
    finally:
        logger.info("ZED streaming stopped")


def kiwi_streaming_worker(camera_buffer: CameraBuffer, stop_event: threading.Event):
    """Stream Kiwi frames from incoming network connection"""
    try:
        from utils_pkg.kiwi import stream_kiwi_frames, start_kiwi_server

        logger.info("Starting Kiwi server...")
        conn, addr = start_kiwi_server(host='0.0.0.0', port=8888)
        logger.info(f"Kiwi client connected from {addr}")

        frame_count = 0
        for rgb in stream_kiwi_frames(conn):
            if stop_event.is_set():
                break

            # Validate Kiwi image format
            if len(rgb.shape) != 3 or rgb.shape[2] != 3:
                logger.error(f"[Kiwi] Invalid image shape: {rgb.shape}. Expected (H, W, 3)")
                continue

            # Debug: Log first frame
            if frame_count == 0:
                logger.info(f"[Kiwi Buffer] First frame stored: shape={rgb.shape}, dtype={rgb.dtype}, channels={rgb.shape[2] if len(rgb.shape) == 3 else 'N/A'}")

            timestamp = time.time()
            camera_buffer.append(rgb, timestamp)
            frame_count += 1

        conn.close()

    except ImportError as e:
        logger.warning(f"Kiwi camera dependencies not available: {e}")
        logger.warning("Kiwi camera will not be available. Using placeholder images.")
        logger.warning("Make sure frame_bundle_pb2.py is in utils/ directory.")
        # Provide placeholder images so inference can continue
        while not stop_event.is_set():
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            timestamp = time.time()
            camera_buffer.append(placeholder, timestamp)
            time.sleep(0.033)  # ~30 FPS
    except Exception as e:
        logger.error(f"Kiwi streaming error: {e}")
        logger.warning("Kiwi camera will not be available. Using placeholder images.")
        # Provide placeholder images so inference can continue
        while not stop_event.is_set():
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            timestamp = time.time()
            camera_buffer.append(placeholder, timestamp)
            time.sleep(0.033)  # ~30 FPS
    finally:
        logger.info("Kiwi streaming stopped")


class TimeStep:
    """Observation container"""
    def __init__(self, observation, reward):
        self.observation = observation
        self.reward = reward


class SpotRealEnvGPU:
    """
    GPU-side environment for distributed inference.

    - Streams cameras locally (ZED + Kiwi from iPhone)
    - Receives qpos from Mac via network
    - Sends actions to Mac via network

    No direct robot connection - Mac handles that.
    """

    def __init__(self,
                 camera_names: list = None,
                 action_send_port: int = 9999,
                 qpos_receive_port: int = 9998):
        """
        Initialize GPU environment.

        Args:
            camera_names: List of camera names to stream
            action_send_port: Port to send actions to Mac
            qpos_receive_port: Port to receive qpos from Mac
        """
        logger.info("Initializing GPU-side environment...")

        if camera_names is None:
            camera_names = ['zed_camera', 'arm_camera']
        self.camera_names = camera_names

        self.action_send_port = action_send_port
        self.qpos_receive_port = qpos_receive_port

        self.stop_event = threading.Event()
        self.streaming_threads = []

        # Create camera buffers
        self.camera_buffers: Dict[str, CameraBuffer] = {
            name: CameraBuffer(max_size=30) for name in camera_names
        }

        # Network sockets
        self.action_sock = None
        self.qpos_sock = None
        self.action_conn = None
        self.qpos_conn = None

        # Start camera streaming
        self._start_camera_streaming()

        logger.info("GPU environment initialized")

    def _start_camera_streaming(self):
        """Start background threads for camera streaming"""

        # ZED camera
        if 'zed_camera' in self.camera_names:
            zed_thread = threading.Thread(
                target=zed_streaming_worker,
                args=(self.camera_buffers['zed_camera'], self.stop_event),
                daemon=True,
                name="ZED-Streaming"
            )
            zed_thread.start()
            self.streaming_threads.append(zed_thread)
            logger.info("Started ZED streaming thread")

        # Kiwi camera
        if 'arm_camera' in self.camera_names:
            kiwi_thread = threading.Thread(
                target=kiwi_streaming_worker,
                args=(self.camera_buffers['arm_camera'], self.stop_event),
                daemon=True,
                name="Kiwi-Streaming"
            )
            kiwi_thread.start()
            self.streaming_threads.append(kiwi_thread)
            logger.info("Started Kiwi streaming thread")

    def setup_network_sockets(self):
        """Setup network sockets for action/qpos communication"""
        logger.info(f"Setting up network sockets...")
        logger.info(f"  Action send port: {self.action_send_port}")
        logger.info(f"  Qpos receive port: {self.qpos_receive_port}")

        # Action socket (GPU sends actions to Mac)
        self.action_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.action_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.action_sock.bind(('0.0.0.0', self.action_send_port))
        self.action_sock.listen(1)

        # Qpos socket (GPU receives qpos from Mac)
        self.qpos_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.qpos_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.qpos_sock.bind(('0.0.0.0', self.qpos_receive_port))
        self.qpos_sock.listen(1)

        logger.info("Network sockets created. Waiting for Mac client...")

    def wait_for_mac_connection(self):
        """Wait for Mac client to connect"""
        if self.action_sock is None or self.qpos_sock is None:
            self.setup_network_sockets()

        # Make sockets non-blocking with timeout to allow interrupt
        self.action_sock.settimeout(1.0)
        self.qpos_sock.settimeout(1.0)

        # Wait for action connection
        logger.info(f"Waiting for action connection on port {self.action_send_port}...")
        while True:
            try:
                self.action_conn, action_addr = self.action_sock.accept()
                logger.info(f"Action connection from {action_addr}")
                break
            except socket.timeout:
                if self.stop_event.is_set():
                    raise KeyboardInterrupt("Shutdown requested")
                continue

        # Wait for qpos connection
        logger.info(f"Waiting for qpos connection on port {self.qpos_receive_port}...")
        while True:
            try:
                self.qpos_conn, qpos_addr = self.qpos_sock.accept()
                logger.info(f"Qpos connection from {qpos_addr}")
                break
            except socket.timeout:
                if self.stop_event.is_set():
                    raise KeyboardInterrupt("Shutdown requested")
                continue

        # Reset to blocking after connection
        self.action_conn.settimeout(None)
        self.qpos_conn.settimeout(None)

        logger.info("✓ Mac client connected!")

    def send_action(self, action: np.ndarray):
        """
        Send action to Mac for execution.

        Args:
            action: [11,] numpy array (6 arm + 1 gripper + 4 body)
        """
        if self.action_conn is None:
            raise RuntimeError("Mac client not connected")

        action_bytes = np.array(action, dtype=np.float64).tobytes()
        self.action_conn.send(action_bytes)

    def receive_qpos(self) -> np.ndarray:
        """
        Receive qpos from Mac.

        Returns:
            [11,] numpy array
        """
        if self.qpos_conn is None:
            raise RuntimeError("Mac client not connected")

        qpos_bytes = self.qpos_conn.recv(88)  # 11 * 8 bytes
        if len(qpos_bytes) < 88:
            raise RuntimeError("Incomplete qpos data received")

        return np.frombuffer(qpos_bytes, dtype=np.float64)

    def _get_images(self) -> Dict[str, np.ndarray]:
        """Get latest images from local cameras"""
        images = {}
        for camera_name in self.camera_names:
            frame = self.camera_buffers[camera_name].get_latest()
            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            images[camera_name] = frame
        return images

    def get_observation(self, qpos: np.ndarray) -> TimeStep:
        """
        Build observation from qpos and images.

        Args:
            qpos: [11,] current qpos from Mac

        Returns:
            TimeStep with observation
        """
        images = self._get_images()

        obs = dict(
            qpos=qpos,
            images=images,
        )

        return TimeStep(obs, reward=0.0)

    def shutdown(self):
        """Stop all threads and close sockets"""
        logger.info("Shutting down GPU environment...")

        self.stop_event.set()

        # Give threads a moment to check stop_event
        time.sleep(0.1)

        for thread in self.streaming_threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not stop gracefully")

        if self.action_conn is not None:
            try:
                self.action_conn.close()
            except:
                pass
        if self.qpos_conn is not None:
            try:
                self.qpos_conn.close()
            except:
                pass
        if self.action_sock is not None:
            try:
                self.action_sock.close()
            except:
                pass
        if self.qpos_sock is not None:
            try:
                self.qpos_sock.close()
            except:
                pass

        logger.info("GPU environment shutdown complete")
