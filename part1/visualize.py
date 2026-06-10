import os
import gymnasium as gym
import torch
import imageio
import imageio.v3 as iio



from agent import Policy


MODEL_PATHS = [
    'results/model_none.pt',                     
    'results/model_ema_a0.0500.pt',                
    'results/model_actor_critic_refined_seed42.pt',
]
VIDEO_DIR = 'videos'

N_EPISODES = 3       
RENDER_W, RENDER_H = 320, 240   

MP4_FPS = 30          
GIF_FPS = 15          
GIF_SUBSAMPLE = 1     


def act(policy, state):
    """Deterministic action — the distribution mean, no exploration noise."""
    with torch.no_grad():
        x = torch.from_numpy(state).float()
        normal_dist, _ = policy(x)
        action = torch.tanh(normal_dist.mean)
        return action.cpu().numpy()


def load_policy(env, model_path):
    """Load a policy, inferring hidden size from the checkpoint itself."""
    state_dict = torch.load(model_path, map_location='cpu')
    hidden_size = state_dict['fc1_actor.weight'].shape[0]  
    policy = Policy(
        state_space=env.observation_space.shape[0],
        action_space=env.action_space.shape[0],
        hidden=hidden_size,
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded {model_path}  (hidden={hidden_size})")
    return policy


def record_model(model_path):
    """Run N_EPISODES, return the list of rendered RGB frames."""
    env = gym.make('Hopper-v4', render_mode='rgb_array',
                   width=RENDER_W, height=RENDER_H)
    policy = load_policy(env, model_path)

    frames = []
    for ep in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            frames.append(env.render())
            action = act(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        print(f"  episode {ep}  return={ep_return:.1f}  (frames so far: {len(frames)})")

    env.close()
    return frames


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    for model_path in MODEL_PATHS:
        print(f"\n=== {model_path} ===")
        frames = record_model(model_path)

        name = os.path.splitext(os.path.basename(model_path))[0]  
        base = os.path.join(VIDEO_DIR, name)

        mp4_path = f'{base}.mp4'
        imageio.mimsave(mp4_path, frames, fps=MP4_FPS, macro_block_size=1)

        gif_path = f'{base}.gif'
        gif_frames = frames[::GIF_SUBSAMPLE]
        imageio.mimsave(gif_path, gif_frames, fps=GIF_FPS, loop=0)

        print(f"  saved {mp4_path}  ({len(frames)} frames @ {RENDER_W}x{RENDER_H})")
        print(f"  saved {gif_path}  ({len(gif_frames)} frames @ {GIF_FPS} fps)")

        
        frames = iio.imread('videos/model_none.gif', index=None)
        for i, idx in enumerate([0, len(frames)//2, len(frames)-1]):
            iio.imwrite(f'videos/frame_none_{i}.png', frames[idx])

    print("\nDone.")


if __name__ == '__main__':
    main()