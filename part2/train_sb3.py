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

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from rand_wrapper import RandomizationWrapper

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
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
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _on_step(self) -> bool:
        current_time = time.time()
        if self.num_timesteps % self.log_freq == 0 or (current_time - self.last_log_time) > 100:
            elapsed = current_time - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            log_msg = (
                f"\n{'='*70}\n"
                f" [{datetime.now().strftime('%H:%M:%S')}] ACTIVE - Step {self.num_timesteps} | "
                f"Elapsed: {elapsed/60:.1f}min | FPS: {fps:.1f}\n"
                f" Timesteps: {self.num_timesteps:,}\n"
            )
            
            print(log_msg)
            
            with open(self.log_file, "a") as f:
                f.write(log_msg)
            
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
        default="udr",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )

    parser.add_argument(
        "--mass-range",
        type=float,
        nargs=2,
        default=[0.5, 6.0],
        metavar=("MIN", "MAX"),
        help="Object mass range, e.g. --mass-range 1 3 or --mass-range 0.5 6",
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
        default=100000,
        help="Number of training timesteps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved model zip to resume training",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
    except ValueError:
        print(" ERROR: --net-arch must be comma-separated integers, e.g. 256,256")
        sys.exit(1)

    if not torch.cuda.is_available():
        print(" ERROR: CUDA GPU not available!")
        print("   Install PyTorch with GPU support or check your NVIDIA drivers.")
        sys.exit(1)
    
    device = "cuda"
    print(f" FORCING GPU device: {device}")
    print(f" GPU Name: {torch.cuda.get_device_name(0)}")
    print(f" CUDA Version: {torch.version.cuda}")
    torch.cuda.empty_cache()  

    set_random_seed(args.seed)

    env = gym.make(
        "PandaPush-v3",
        render_mode= "rgb_array", 
        reward_type="dense",
        type=args.env_type,
    )

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

  
    mass_range = (float(args.mass_range[0]), float(args.mass_range[1]))

    if args.sampling_strategy == "udr":
        env = RandomizationWrapper(env, mode="udr", mass_range=mass_range)

    elif args.sampling_strategy == "adr":
        env = RandomizationWrapper(env, mode="adr", mass_range=mass_range)

    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"ERROR: resume file not found: {args.resume}")
            sys.exit(1)
        if args.algo == "sac":
            model = SAC.load(args.resume, env=env, device=device)
        elif args.algo == "ppo":
            model = PPO.load(args.resume, env=env, device=device)
    elif args.algo == "sac":
        lr = args.learning_rate if args.learning_rate else 3e-4
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            buffer_size=1000000,
            batch_size=256,
            train_freq=1,
            gamma= 0.99,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            target_update_interval=1,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs={
                "net_arch": dict(pi=net_arch, qf=net_arch),
                "activation_fn": torch.nn.ReLU,
            },
            device=device,
            verbose=1,
            tau=0.005,
        )

    elif args.algo == "ppo":
        lr = args.learning_rate if args.learning_rate else 3e-4  #
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            n_steps=2048,  
            batch_size=64, 
            n_epochs=10,  
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,    
            ent_coef=0.05,     
            use_sde=True,      
            sde_sample_freq=4,
            max_grad_norm=0.5, 
            policy_kwargs={
                "net_arch": dict(pi=[512, 512], vf=[512, 512]), 
                "activation_fn": torch.nn.Tanh,
            },
            device=device,
            verbose=1,
        )

    # TRAIN
    print(f"\n Training {args.algo.upper()} for {args.timesteps} timesteps...")
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    

    activity_callback = ActivityCallback(log_freq=10000)
    checkpoints_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoints_dir,
        name_prefix=f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[activity_callback, checkpoint_callback],
        reset_num_timesteps=not bool(args.resume),
    )

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(os.path.join(results_dir, save_name))


if __name__ == "__main__":
    main()
    