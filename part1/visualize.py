"""Watch a trained Hopper policy hop (live) and save an mp4."""
import os
import gymnasium as gym
import torch
import imageio

from agent import Policy


def act(policy, state):
    """Deterministic action — the distribution mean, no exploration noise."""
    with torch.no_grad():
        x = torch.from_numpy(state).float()
        normal_dist, _ = policy(x)
        action = torch.tanh(normal_dist.mean)
        return action.cpu().numpy()


def load_policy(env, model_path, hidden_size):
    policy = Policy(
        state_space=env.observation_space.shape[0],
        action_space=env.action_space.shape[0],
        hidden=hidden_size
    )
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()
    return policy


def main():
    model_path = 'results/model_actor_critic.pt'
    video_path = 'videos/hopper_demo.mp4'
    hidden_size = 256
    n_episodes = 50

    # --- 1. Live window ---
    env = gym.make('Hopper-v4', render_mode='human')
    policy = load_policy(env, model_path, hidden_size)
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0
        while not done:
            action = act(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        print(f"[LIVE] episode {ep}  return={ep_return:.1f}")
    env.close()

    # --- 2. Offscreen render -> save mp4 ---
    env = gym.make('Hopper-v4', render_mode='rgb_array')
    policy = load_policy(env, model_path, hidden_size)
    frames = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0
        while not done:
            frames.append(env.render())
            action = act(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        print(f"[REC]  episode {ep}  return={ep_return:.1f}")
    env.close()

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimsave(video_path, frames, fps=30)
    print(f"\nSaved video to {video_path}")


if __name__ == '__main__':
    main()