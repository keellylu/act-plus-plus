#!/usr/bin/env python3
"""
GPU Inference Server for Distributed ACT++ Inference

This server:
1. Loads policy and statistics
2. Streams cameras (ZED + receives Kiwi)
3. Waits for Mac client to connect
4. Main loop: receives qpos → runs policy → sends actions

Run on GPU machine:
    python inference_server.py \
        --checkpoint policy_best.ckpt \
        --stats dataset_stats.pkl \
        --num-episodes 5
"""

import argparse
import pickle
import torch
import numpy as np
import time
import logging
import signal
import sys
import cv2
from pathlib import Path
from torchvision import transforms
from datetime import datetime

from spot_real_env_gpu import SpotRealEnvGPU
from policy import ACTPolicy, DiffusionPolicy, CNNMLPPolicy

# Setup logging to both console and file
log_file = f"inference_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Suppress verbose PIL debug logs
logging.getLogger('PIL').setLevel(logging.WARNING)


def load_policy(checkpoint_path: str, policy_config: dict, device: str = 'cuda'):
    """Load policy from checkpoint"""
    policy_class = policy_config.get('policy_class', 'ACT')

    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.deserialize(checkpoint)
    policy.to(device)
    policy.eval()

    logger.info(f"Loaded {policy_class} policy from {checkpoint_path}")
    return policy


def load_stats(stats_path: str) -> dict:
    """Load normalization statistics"""
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    logger.info(f"Loaded stats from {stats_path}")
    return stats


def preprocess_observation(qpos: np.ndarray, images: dict, stats: dict) -> tuple:
    """
    Preprocess observation for policy input.

    Args:
        qpos: [11,] joint positions
        images: dict with camera images
        stats: normalization statistics

    Returns:
        qpos_norm: normalized qpos
        images: camera images
    """
    qpos_mean = stats.get('qpos_mean', 0.0)
    qpos_std = stats.get('qpos_std', 1.0)
    qpos_norm = (qpos - qpos_mean) / (qpos_std + 1e-6)

    return qpos_norm, images


def postprocess_action(action: np.ndarray, stats: dict) -> np.ndarray:
    """
    Postprocess action from policy.

    Args:
        action: raw action from policy (normalized to ~[-1, 1])
        stats: normalization statistics

    Returns:
        denormalized action (physical units: radians, m/s, etc.)
    """
    action_mean = stats.get('action_mean', 0.0)
    action_std = stats.get('action_std', 1.0)
    action_min = stats.get('action_min', None)
    action_max = stats.get('action_max', None)

    # Denormalize to physical units
    action = action * action_std + action_mean

    # Clip to training range (not [-1, 1])
    if action_min is not None and action_max is not None:
        action = np.clip(action, action_min, action_max)

    return action


def run_inference(
    policy_checkpoint: str,
    policy_config: dict,
    stats_path: str,
    save_frames: bool = False,
    frames_dir: str = None,
):
    """
    Main inference loop on GPU.

    Note: GPU server follows Mac client's control. Mac client determines
    num_episodes and max_steps. GPU server loops until Mac disconnects.

    Args:
        policy_checkpoint: Path to policy weights
        policy_config: Policy configuration
        stats_path: Path to normalization stats
        save_frames: Whether to save camera frames for debugging
        frames_dir: Directory to save frames (created if doesn't exist)
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Setup frame saving if requested
    frames_path = None
    if save_frames:
        if frames_dir is None:
            frames_dir = f"./inference_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        frames_path = Path(frames_dir)
        frames_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving frames to: {frames_path.absolute()}")

    # Load policy
    policy = load_policy(policy_checkpoint, policy_config, device=device)

    # Load stats
    stats = load_stats(stats_path)

    # Create environment (GPU-side, cameras only)
    logger.info("Creating GPU environment...")
    env = SpotRealEnvGPU(
        camera_names=['zed_camera', 'arm_camera'],
        action_send_port=9999,
        qpos_receive_port=9998
    )

    logger.info("\n" + "="*60)
    logger.info("WAITING FOR MAC CLIENT")
    logger.info("="*60)
    logger.info("Run on Mac:")
    logger.info(f"  python inference_client.py --gpu-ip <GPU_IP>")
    logger.info("="*60 + "\n")

    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal. Shutting down...")
        env.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for Mac to connect
    try:
        env.wait_for_mac_connection()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested before Mac connection")
        env.shutdown()
        return

    try:
        # GPU server follows Mac client - responds to qpos requests until Mac disconnects
        # Mac client controls num_episodes and max_steps, GPU just responds
        episode_idx = 0
        step_idx = 0
        episode_start_time = None
        last_qpos = None
        hz = 20  # Target frequency in Hz
        step_duration = 1.0 / hz  # Duration per step in seconds

        logger.info("Ready. Waiting for Mac client to start inference...")
        logger.info(f"Target inference frequency: {hz} Hz")

        next_step_time = time.time()

        while True:
            try:
                # Receive qpos from Mac (could be reset signal or next step)
                qpos = env.receive_qpos()

                # Log received qpos
                logger.info(f"Step {step_idx}: Received qpos: {qpos}")

                # Check for out-of-range qpos (potential corruption)
                training_qpos_min = [-0.15, -3.7, 0.66, -1.25, -2.36, -1.66]  # 3-sigma min
                training_qpos_max = [0.11, 1.26, 3.87, 1.65, 0.30, 1.27]      # 3-sigma max
                out_of_range = False
                for i in range(6):
                    if qpos[i] < training_qpos_min[i] or qpos[i] > training_qpos_max[i]:
                        logger.warning(f"  WARNING: qpos[{i}] = {qpos[i]:.4f} is outside training range [{training_qpos_min[i]:.4f}, {training_qpos_max[i]:.4f}]")
                        out_of_range = True
                if out_of_range:
                    logger.warning(f"  Full qpos: {qpos}")

                # Detect new episode: if qpos changed significantly or it's the first one
                is_new_episode = False
                if last_qpos is None:
                    is_new_episode = True
                    episode_idx += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Episode {episode_idx} started")
                    logger.info(f"{'='*60}")
                    episode_start_time = time.time()
                    step_idx = 0
                elif np.linalg.norm(qpos - last_qpos) > 0.5:  # Significant change = likely reset
                    is_new_episode = True
                    if step_idx > 0:
                        episode_duration = time.time() - episode_start_time
                        logger.info(f"Episode {episode_idx} complete. Duration: {episode_duration}s")
                    episode_idx += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Episode {episode_idx} started")
                    logger.info(f"{'='*60}")
                    episode_start_time = time.time()
                    step_idx = 0
                else:
                    # Log qpos delta for non-reset steps
                    qpos_delta = qpos - last_qpos
                    delta_norm = np.linalg.norm(qpos_delta)
                    qpos_delta_list = qpos_delta.tolist()
                    logger.info(f"  Qpos delta: {qpos_delta_list}, norm={delta_norm}")

                last_qpos = qpos.copy()
                
                # Get observation
                obs = env.get_observation(qpos)
                qpos_obs = obs.observation['qpos']
                images = obs.observation['images']

                # Preprocess
                qpos_norm, _ = preprocess_observation(qpos_obs, images, stats)

                # Prepare policy input
                qpos_norm_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

                # Convert images to tensors
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                image_tensors = []

                # Debug: log image info on first step of each episode
                if step_idx == 0:  # Log on first step (step_idx starts at 0)
                    logger.info(f"Image info for episode {episode_idx}:")

                for cam_name, img in images.items():
                    # Log image info on first step
                    if step_idx == 0:
                        img_min, img_max = img.min(), img.max()
                        logger.info(f"  {cam_name}: shape={img.shape}, dtype={img.dtype}, "
                                  f"channels={img.shape[2] if len(img.shape) == 3 else 'N/A'}, "
                                  f"min={img_min}, max={img_max}")

                    # Save frame if requested
                    if frames_path is not None and step_idx % 10 == 0:  # Save every 10th frame
                        frame_path = frames_path / f"episode_{episode_idx:03d}_step_{step_idx:05d}_{cam_name}.png"
                        # Save original image (convert RGB back to BGR for OpenCV)
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(frame_path), img_bgr)

                    # Resize to training dimensions (480, 640)
                    if img.shape[:2] != (480, 640):
                        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

                    # Normalize to [0, 1]
                    img_normalized = img.astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().to(device)

                    # Apply ImageNet normalization per camera
                    img_tensor = normalize(img_tensor)
                    img_tensor = img_tensor.unsqueeze(0)
                    image_tensors.append(img_tensor)

                # Concatenate all camera images
                image_tensor = torch.cat(image_tensors, dim=1)

                # Debug: log tensor shapes on first step
                if step_idx == 0:
                    logger.info(f"Image tensor shapes before concat:")
                    for i, cam_tensor in enumerate(image_tensors):
                        logger.info(f"  Camera {i}: {cam_tensor.shape}")
                    logger.info(f"Final concatenated image tensor shape: {image_tensor.shape}")

                # Policy inference
                with torch.no_grad():
                    action_tensor = policy(qpos_norm_tensor, image_tensor)

                action = action_tensor.squeeze(0).cpu().numpy()

                # Log raw policy output
                if step_idx == 0:
                    logger.info(f"Raw policy output (first step): {action}")
                    logger.info(f"  Min: {action.min()}, Max: {action.max()}")
                    logger.info(f"  Current qpos: {qpos_obs}")

                # Postprocess
                action = postprocess_action(action, stats)

                # Log postprocessed action
                if step_idx == 0:
                    logger.info(f"After postprocessing: {action}")

                # Log action for all steps (to catch extreme values)
                action_norm = np.linalg.norm(action)
                logger.info(f"Step {step_idx}: Sending action: {action}, norm={action_norm}")

                # Send action to Mac
                env.send_action(action)

                step_idx += 1
                if step_idx % 10 == 0:  # Changed from 50 to 10 for more frequent logging
                    elapsed = time.time() - episode_start_time if episode_start_time else 0
                    action_flat = action.flatten() if hasattr(action, 'flatten') else action
                    qpos_obs_flat = qpos_obs.flatten() if hasattr(qpos_obs, 'flatten') else qpos_obs
                    logger.info(f"Step {step_idx:3d} | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"Action: [{action_flat[0]:.3f}, {action_flat[1]:.3f}, {action_flat[2]:.3f}, ...] | "
                              f"Qpos: [{qpos_obs_flat[0]:.3f}, {qpos_obs_flat[1]:.3f}, {qpos_obs_flat[2]:.3f}, ...]")

                # Rate limit to specified Hz using wall-clock time
                next_step_time += step_duration
                sleep_time = next_step_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:  # More than 10ms late
                    logger.warning(f"Step {step_idx} is {-sleep_time*1000:.1f}ms behind schedule")

            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                # Mac client disconnected
                if step_idx > 0 and episode_start_time:
                    episode_duration = time.time() - episode_start_time
                    logger.info(f"Episode {episode_idx} complete. Duration: {episode_duration}s")
                logger.info("Mac client disconnected.")
                break

        logger.info(f"\n{'='*60}")
        logger.info("All episodes complete!")
        logger.info(f"{'='*60}")

    except KeyboardInterrupt:
        logger.info("\nInference stopped by user")

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)

    finally:
        logger.info("Shutting down...")
        env.stop_event.set()  # Signal all threads to stop
        env.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='GPU Inference Server for ACT++ Spot'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--stats', type=str, required=True,
                       help='Path to dataset normalization stats')
    parser.add_argument('--config', type=str,
                       help='Path to training config (optional)')
    parser.add_argument('--policy-class', type=str, default='ACT',
                       choices=['ACT', 'Diffusion', 'CNNMLP'],
                       help='Policy class')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save camera frames during inference for debugging')
    parser.add_argument('--frames-dir', type=str, default=None,
                       help='Directory to save frames (default: auto-generated)')
    # Note: num_episodes and max_steps are controlled by Mac client
    # GPU server follows Mac client's control
    parser.add_argument('--action-port', type=int, default=9999,
                       help='Port to send actions')
    parser.add_argument('--qpos-port', type=int, default=9998,
                       help='Port to receive qpos')

    args = parser.parse_args()

    # Build policy config
    policy_config = {
        'policy_class': args.policy_class,
        'camera_names': ['zed_camera', 'arm_camera'],
        'action_dim': 11,
        'state_dim': 11,
    }

    # Load full config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.warning(f"Config file not found: {args.config}. Continuing without it.")
        else:
            with open(args.config, 'rb') as f:
                full_config = pickle.load(f)
            policy_config.update(full_config.get('policy_config', {}))

    run_inference(
        policy_checkpoint=args.checkpoint,
        policy_config=policy_config,
        stats_path=args.stats,
        save_frames=args.save_frames,
        frames_dir=args.frames_dir,
    )


if __name__ == '__main__':
    main()
