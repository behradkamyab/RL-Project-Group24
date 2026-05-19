import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC, PPO
from rand_wrapper import RandomizationWrapper


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
        default=50000,
        help="Number of training timesteps",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    # SELECT ALGORITHM
    if args.algo == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1)

    elif args.algo == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1)

    # TRAIN
    model.learn(total_timesteps=args.timesteps)

    # SAVE
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)


if __name__ == "__main__":
    main()
