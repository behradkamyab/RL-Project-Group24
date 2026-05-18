"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random
import pickle
import os

from agent import Agent, Policy


def main(algorithm='reinforce', baseline_mode='none', baseline_value=0.0, baseline_alpha=0.0,
 n_episodes=500, seed=42, critic_weight=1.0, save_dir='results'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make('Hopper-v4')
    env.reset(seed=seed)              
    env.action_space.seed(seed)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    os.makedirs(save_dir, exist_ok=True)

    policy = Policy(state_space= env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = Agent(policy, algorithm=algorithm, baseline_mode=baseline_mode,
     baseline_value=baseline_value, baseline_alpha=baseline_alpha, critic_weight=1.0)

    returns_history = []
    recent_returns = []

    for episode in range(n_episodes):
      state, _ = env.reset()
      done = False
      rewards = []
      
      while not done: 
        action, log_prob = agent.get_action(state)
        action = action.detach().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

        agent.store_outcome(state, next_state, log_prob, reward, done)
        state = next_state
      agent.update_policy()

      total_reward = sum(rewards)
      returns_history.append(total_reward)
      recent_returns.append(total_reward)
      
      if len(recent_returns) > 20:
        recent_returns.pop(0)

      if episode % 20 == 0:
        avg = sum(recent_returns) / len(recent_returns)
        print(f"Episode {episode:4d}  return={total_reward:5.0f}  avg(last 20)={avg:6.1f}")
    
    env.close()

    if algorithm == 'actor_critic':
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
    
    print(f"Saved model_{suffix}.pt and history_{suffix}.pkl")

if __name__ == '__main__':
    main()
