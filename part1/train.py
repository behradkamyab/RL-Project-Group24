"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""

import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import time
import random
import pickle
import os

from agent import Agent, Policy


def main(algorithm='reinforce', baseline_mode='none', baseline_value=0.0, baseline_alpha=0.0,
 n_episodes=500, seed=42, critic_weight=1.0, lr=1e-3, actor_lr=None,critic_lr=None, hidden_size=64, gamma=0.99, grad_clip=0.5, entropy_weight=0.0, normalize_advantage=False, run_name=None, save_dir='results'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make('Hopper-v4')
    env.reset(seed=seed)              
    env.action_space.seed(seed)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    os.makedirs(save_dir, exist_ok=True)

    policy = Policy(
    state_space=env.observation_space.shape[0],
    action_space=env.action_space.shape[0],
    hidden=hidden_size
)
    agent = Agent(policy, algorithm=algorithm, baseline_mode=baseline_mode,
     baseline_value=baseline_value, baseline_alpha=baseline_alpha, critic_weight=critic_weight, lr=lr, actor_lr=actor_lr,critic_lr=critic_lr, gamma=gamma, grad_clip=grad_clip, entropy_weight=entropy_weight, normalize_advantage=normalize_advantage)

    returns_history = []
    recent_returns = []
    episode_times = []
    elapsed_times = []
    training_start = time.perf_counter()

    for episode in range(n_episodes):
      episode_start = time.perf_counter()
      state, _ = env.reset()
      done = False
      rewards = []
      
      while not done: 
        action, log_prob, entropy = agent.get_action(state)
        action = action.detach().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

        agent.store_outcome(state, next_state, log_prob, entropy, reward, done)
        state = next_state
      agent.update_policy()

      total_reward = sum(rewards)
      episode_time = time.perf_counter() - episode_start
      elapsed_time = time.perf_counter() - training_start
      returns_history.append(total_reward)
      recent_returns.append(total_reward)
      episode_times.append(episode_time)
      elapsed_times.append(elapsed_time)
      
      if len(recent_returns) > 20:
        recent_returns.pop(0)

      if episode % 20 == 0:
        avg = sum(recent_returns) / len(recent_returns)
        print(f"Episode {episode:4d}  return={total_reward:5.0f}  avg(last 20)={avg:6.1f}")
    
    env.close()

    if run_name is not None:
      suffix = run_name
    elif algorithm == 'actor_critic':
       suffix = 'actor_critic'
    else: 
       suffix = baseline_mode

       if baseline_mode == 'constant':
          suffix = f'constant_b{baseline_value:.0f}'
       elif baseline_mode == 'ema':
         suffix = f'ema_a{baseline_alpha:.4f}'



    torch.save(agent.policy.state_dict(), os.path.join(save_dir, f'model_{suffix}.pt'))

    with open(os.path.join(save_dir, f'history_{suffix}.pkl'), 'wb') as f:
      pickle.dump(returns_history, f)
    with open(os.path.join(save_dir, f'time_{suffix}.pkl'), 'wb') as f:
      pickle.dump({
        'episode_times': episode_times,
        'elapsed_times': elapsed_times,
      }, f)

    print(f"Saved model_{suffix}.pt and history_{suffix}.pkl and time_{suffix}.pkl")

    return returns_history

if __name__ == '__main__':
    main() 