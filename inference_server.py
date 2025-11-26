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
    num_episodes: int = 5,
    max_steps: int = 500,
):
    """
    Main inference loop on GPU.

    Args:
        policy_checkpoint: Path to policy weights
        policy_config: Policy configuration
        stats_path: Path to normalization stats
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
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

    # Wait for Mac to connect
    env.wait_for_mac_connection()

    try:
        for episode_idx in range(num_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode_idx + 1}/{num_episodes}")
            logger.info(f"{'='*60}")

            # Wait for reset signal from Mac
            logger.info("Waiting for reset signal from Mac...")
            qpos = env.receive_qpos()
            logger.info(f"Reset signal received. Initial qpos: {qpos[:3]}...")

            episode_start_time = time.time()

            for step_idx in range(max_steps):
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

                # Receive next qpos from Mac
                qpos = env.receive_qpos()

                if step_idx % 50 == 0:
                    elapsed = time.time() - episode_start_time
                    logger.info(f"Step {step_idx:3d}/{max_steps} | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, ...]")

            episode_duration = time.time() - episode_start_time
            logger.info(f"Episode complete. Duration: {episode_duration:.1f}s")

        logger.info(f"\n{'='*60}")
        logger.info("All episodes complete!")
        logger.info(f"{'='*60}")

    except KeyboardInterrupt:
        logger.info("\nInference stopped by user")

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)

    finally:
        logger.info("Shutting down...")
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
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode')
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
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )


if __name__ == '__main__':
    main()
