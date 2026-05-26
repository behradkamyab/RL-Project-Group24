import argparse
from collections import deque
import sys
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import panda_gym
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from rand_wrapper import RandomizationWrapper

# Force CUDA environment variables before any GPU operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous GPU launches for debugging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU


class ActivityCallback(BaseCallback):
    """
    Custom callback to prevent timeout on cloud platforms (Colab, Studio AI, etc.)
    Prints progress every N steps AND saves to log file for persistence.
    """
    def __init__(self, log_freq=1000, verbose=0, log_file="training.log"):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.log_file = log_file
        
        # Create log file header
        with open(self.log_file, "w") as f:
            f.write("="*70 + "\n")
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

    def _on_step(self) -> bool:
        # Log every log_freq steps OR every 30 seconds (whichever comes first)
        current_time = time.time()
        if self.num_timesteps % self.log_freq == 0 or (current_time - self.last_log_time) > 30:
            elapsed = current_time - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Format log message
            log_msg = (
                f"\n{'='*70}\n"
                f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] ACTIVE - Step {self.num_timesteps} | "
                f"Elapsed: {elapsed/60:.1f}min | FPS: {fps:.1f}\n"
                f"📊 Timesteps: {self.num_timesteps:,}\n"
                f"{'='*70}\n"
            )
            
            # Print to console (prevents cloud timeout)
            print(log_msg)
            
            # Save to file (persistent backup)
            with open(self.log_file, "a") as f:
                f.write(log_msg)
            
            # Force flush output to cloud platform
            sys.stdout.flush()
            sys.stderr.flush()
            
            self.last_log_time = current_time
        
        return True


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
        render_mode="rgb_array",  # Headless rendering (no display, optimized for speed!)
        reward_type="dense",
    )

    # UDR / ADR
    if args.sampling_strategy == "udr":
        env = RandomizationWrapper(env)

    elif args.sampling_strategy == "adr":
        env = RandomizationWrapper(env, adr=True)

    # SELECT ALGORITHM WITH TUNED HYPERPARAMETERS
    if args.algo == "sac":
        # SAC: OPTIMIZED for continuous control pushing tasks
        # Conservative hyperparameters focused on stability
        lr = args.learning_rate if args.learning_rate else 3e-5  # Conservative: 3e-5 (was 3e-4)
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            buffer_size=100000,
            batch_size=64,  # Back to 64 for speed
            train_freq=1,
            gradient_steps=5,  # Back to 5 for speed
            ent_coef=0.05,  # Reduced entropy for more focused learning
            target_entropy="auto",
            target_update_interval=1,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs={
                "net_arch": dict(pi=[512, 512], qf=[512, 512]),
                "activation_fn": torch.nn.ReLU,
            },
            device=device,
            verbose=1,
            tau=0.01,  # Standard tau
        )

    elif args.algo == "ppo":
        # PPO: Optimized for continuous control (pushing task)
        # Key: Higher LR, lower epochs, aggressive exploration
        lr = args.learning_rate if args.learning_rate else 3e-4  # 3x faster learning!
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            n_steps=8192,  # More steps before update = better advantage estimation
            batch_size=64,  # Smaller batch for better gradient signal
            n_epochs=3,  # Reduced to prevent overfitting (was 20!)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,  # More conservative clip (better stability)
            ent_coef=0.05,  # High entropy for aggressive exploration (critical for pushing!)
            use_sde=True,  # State-dependent exploration
            sde_sample_freq=4,
            max_grad_norm=0.5,  # Gradient clipping for stability
            policy_kwargs={
                "net_arch": dict(pi=[512, 512], vf=[512, 512]),  # Simpler networks
                "activation_fn": torch.nn.ReLU,
            },
            device=device,
            verbose=1,
        )

    # TRAIN
    print(f"\n🚀 Training {args.algo.upper()} for {args.timesteps} timesteps...")
    print(f"🔧 Key Improvements:")
    print(f"   - State-dependent exploration (SDE) enabled for better curiosity")
    print(f"   - Higher entropy coefficient for aggressive exploration")
    print(f"   - Larger replay buffer and gradient steps for SAC")
    print(f"   - Smaller batch size for better gradient variance")
    print(f"\n☁️  CLOUD PLATFORM OPTIMIZED (Colab, Studio AI, etc.):")
    print(f"   ✅ Activity logs printed every 1000 steps (prevents timeout)")
    print(f"   ✅ Logs also saved to training.log for persistence")
    print(f"   ✅ All output flushed immediately to cloud interface\n")
    print(f"\n📈 Monitoring GPU usage: Use 'nvidia-smi' in another terminal\n")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Activity callback to prevent cloud platform timeout
    activity_callback = ActivityCallback(log_freq=2000)  # Log every 2000 steps for less overhead
    
    model.learn(total_timesteps=args.timesteps, callback=activity_callback)

    # SAVE
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"✅ Model saved as: {save_name}.zip")
    print(f"{'='*70}")
    
    print(f"\n📌 FOR STUDIO AI / COLAB USERS:")
    print(f"   1. Download model file from cloud storage after training")
    print(f"   2. Download training.log to see full training progress")
    print(f"   3. Model file: {save_name}.zip")
    
    print(f"\n📌 NEXT STEPS TO IMPROVE FURTHER:")
    print(f"   1. Use UDR: --sampling-strategy udr")
    print(f"   2. Train longer: --timesteps 1000000")
    print(f"   3. Fine-tune learning rate: --learning-rate 5e-5")
    print(f"   4. Monitor logs during training to catch issues early")


if __name__ == "__main__":
    main()
