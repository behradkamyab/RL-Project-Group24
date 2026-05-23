import argparse
from collections import deque
import sys
import os

import gymnasium as gym
import numpy as np
import panda_gym
import torch
from stable_baselines3 import SAC, PPO
from rand_wrapper import RandomizationWrapper

# Force CUDA environment variables before any GPU operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous GPU launches for debugging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL on PandaPush-v3")

    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="RL algorithm (sac or ppo)",
    )

    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )

    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300000,
        help="Number of training timesteps",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (None = auto-tuned per algo)",
    )

    parser.add_argument(
        "--net-arch",
        type=str,
        default="512,512",
        help="Network architecture hidden layers (comma-separated)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # FORCE GPU Configuration
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA GPU not available!")
        print("   Install PyTorch with GPU support or check your NVIDIA drivers.")
        sys.exit(1)
    
    device = "cuda"
    print(f"🖥️  FORCING GPU device: {device}")
    print(f"📊 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"📊 CUDA Version: {torch.version.cuda}")
    torch.cuda.empty_cache()  # Clear GPU cache before training

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        reward_type="dense",
    )

    # UDR / ADR
    if args.sampling_strategy == "udr":
        env = RandomizationWrapper(env)

    elif args.sampling_strategy == "adr":
        env = RandomizationWrapper(env, adr=True)

    # Parse network architecture
    net_arch = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = {
        "net_arch": dict(pi=net_arch, qf=net_arch),  # For SAC/PPO
    }

    # SELECT ALGORITHM WITH TUNED HYPERPARAMETERS
    if args.algo == "sac":
        # SAC: Better for continuous control (pushing)
        lr = args.learning_rate if args.learning_rate else 1e-4
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            buffer_size=300000,
            batch_size=512,  # Larger batch for GPU
            train_freq=1,
            gradient_steps=2,  # More GPU work per step
            ent_coef="auto",  # Auto-tune entropy
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            device=device,  # GPU Support - FORCED
            verbose=1,
        )

    elif args.algo == "ppo":
        # PPO: On-policy, needs good exploration tuning
        lr = args.learning_rate if args.learning_rate else 5e-5
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            n_steps=2048,  # More steps for stability
            batch_size=256,  # Larger batch for GPU
            n_epochs=15,  # More GPU updates
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Boost exploration
            policy_kwargs=policy_kwargs,
            device=device,  # GPU Support - FORCED
            verbose=1,
        )

    # TRAIN
    print(f"\n🚀 Training {args.algo.upper()} for {args.timesteps} timesteps...")
    print(f"🔧 GPU Optimization: Batch size optimized, gradient steps increased")
    print(f"📈 Monitoring GPU usage: Use 'nvidia-smi' in another terminal\n")
    
    model.learn(total_timesteps=args.timesteps)

    # SAVE
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)
    print(f" Model saved as: {save_name}.zip")


if __name__ == "__main__":
    main()
