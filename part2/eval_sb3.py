import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
import panda_gym  # noqa: F401 - required so Panda envs are registered


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str) -> None:
    if not model_path.endswith(".zip"):
        model_path_to_load = model_path + ".zip"
    else:
        model_path_to_load = model_path
    
    if not os.path.exists(model_path_to_load):
        raise FileNotFoundError(f"Model file not found: {model_path_to_load}")

    render_mode = "human" if render else "rgb_array"
    
    # create environment without 'type' parameter
    env = gym.make("PandaPush-v3", render_mode=render_mode, reward_type="dense")
    
    # Load model
    if "sac" in model_path.lower():
        model = SAC.load(model_path_to_load)
    elif "ppo" in model_path.lower():
        model = PPO.load(model_path_to_load)
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n Evaluation Results")
    print(f"Environment type: {env_type}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on PandaPush")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model zip file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--render", action="store_true", help="Render with window")
    parser.add_argument("--env-type", type=str, default="target", choices=["source", "target"], help="Environment type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
    )