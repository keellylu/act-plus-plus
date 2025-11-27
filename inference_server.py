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
from pathlib import Path

from spot_real_env_gpu import SpotRealEnvGPU
from policy import ACTPolicy, DiffusionPolicy, CNNMLPPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        action: raw action from policy
        stats: normalization statistics

    Returns:
        denormalized action
    """
    action_mean = stats.get('action_mean', 0.0)
    action_std = stats.get('action_std', 1.0)

    action = action * action_std + action_mean
    action = np.clip(action, -1.0, 1.0)

    return action


def run_inference(
    policy_checkpoint: str,
    policy_config: dict,
    stats_path: str,
):
    """
    Main inference loop on GPU.
    
    Note: GPU server follows Mac client's control. Mac client determines
    num_episodes and max_steps. GPU server loops until Mac disconnects.

    Args:
        policy_checkpoint: Path to policy weights
        policy_config: Policy configuration
        stats_path: Path to normalization stats
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

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
        
        logger.info("Ready. Waiting for Mac client to start inference...")
        
        while True:
            try:
                # Receive qpos from Mac (could be reset signal or next step)
                qpos = env.receive_qpos()
                
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
                        logger.info(f"Episode {episode_idx} complete. Duration: {episode_duration:.1f}s")
                    episode_idx += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Episode {episode_idx} started")
                    logger.info(f"{'='*60}")
                    episode_start_time = time.time()
                    step_idx = 0
                
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
                image_tensors = {}
                for cam_name, img in images.items():
                    img_norm = (img.astype(np.float32) / 255.0 - 0.5) * 2.0
                    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    image_tensors[cam_name] = img_tensor

                # Policy inference
                with torch.no_grad():
                    action_tensor = policy(qpos_norm_tensor, image_tensors)

                action = action_tensor.squeeze(0).cpu().numpy()

                # Postprocess
                action = postprocess_action(action, stats)

                # Send action to Mac
                env.send_action(action)
                
                step_idx += 1
                if step_idx % 50 == 0:
                    elapsed = time.time() - episode_start_time if episode_start_time else 0
                    logger.info(f"Step {step_idx:3d} | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, ...]")

            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                # Mac client disconnected
                if step_idx > 0 and episode_start_time:
                    episode_duration = time.time() - episode_start_time
                    logger.info(f"Episode {episode_idx} complete. Duration: {episode_duration:.1f}s")
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
        with open(args.config, 'rb') as f:
            full_config = pickle.load(f)
        policy_config.update(full_config.get('policy_config', {}))

    run_inference(
        policy_checkpoint=args.checkpoint,
        policy_config=policy_config,
        stats_path=args.stats,
    )


if __name__ == '__main__':
    main()
