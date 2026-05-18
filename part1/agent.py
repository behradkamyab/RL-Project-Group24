from re import X
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_v = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                # torch.nn.init.normal_(m.weight)
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_v(x_critic).squeeze(-1)

        
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu',algorithm='reinforce',
     baseline_mode='none', baseline_value=0.0, baseline_alpha=0.05, critic_weight=1.0):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.baseline_mode = baseline_mode
        self.baseline_value = baseline_value
        self.baseline_alpha = baseline_alpha

        self.algorithm = algorithm
        self.critic_weight = critic_weight

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
        
        if self.algorithm == 'reinforce':

          #
          # TASK 2:
          #   - compute discounted returns
          returns = discount_rewards(rewards, self.gamma)

          if self.baseline_mode == 'none':
            normalized_returns = returns

          elif self.baseline_mode == 'normalize':
            normalized_returns = (returns - returns.mean())/(returns.std() + 1e-8)
          
          elif self.baseline_mode == 'constant': 
            normalized_returns = returns - self.baseline_value
            # normalized_returns = returns/(returns.std() + 1e-8)

          elif self.baseline_mode == 'ema':
            normalized_returns = returns - self.baseline_value
            total_reward = rewards.sum().item()
            self.baseline_value = (1 - self.baseline_alpha) * self.baseline_value + self.baseline_alpha * total_reward
            # normalized_returns = returns/(returns.std() + 1e-8)
          
          #   - compute policy gradient loss function given actions and returns
          agent_loss = -(action_log_probs * normalized_returns).sum()

          #   - compute gradients and step the optimizer
          self.optimizer.zero_grad()
          agent_loss.backward()
          self.optimizer.step()

        elif self.algorithm == 'actor_critic': 
          #
          # TASK 3:
          #   - compute boostrapped discounted return estimates
          _, state_values = self.policy(states)
          _, next_state_values = self.policy(next_states)
          target_t = rewards + self.gamma * next_state_values.detach() * (1.0 - done)
          
          #   - compute advantage terms
          advantage_t = (target_t - state_values).detach()
          advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)   


          #   - compute actor loss and critic loss
          actor_loss = -(action_log_probs * advantage_t).sum()
          critic_loss = F.mse_loss(state_values, target_t.detach())
          
          #   - compute gradients and step the optimizer
          loss = actor_loss + self.critic_weight * critic_loss

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

        else: 
          raise ValueError(f"Unknown algorithm: {self.algorithm!r}")
        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

