from re import X
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
LOG_PROB_EPS = 1e-6

def discount_rewards(r, gamma):
    """Return Monte Carlo return-to-go for a 1-D reward tensor."""
    r=r.flatten()
    discounted_r = torch.zeros_like(r)
    running_add = torch.zeros((), dtype=r.dtype, device=r.device)


    for t in reversed(range(0, r.size(0))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def standardize(x, eps=1e-8):
    std = x.std(unbiased=False)
    if torch.isclose(std, torch.zeros_like(std)):
        return x - x.mean()
    return (x - x.mean()) / (std + eps)

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden=64, init_std=0.6):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learn log standard deviation directly. This is easier to tune than sigma.
        init_log_std = math.log(init_std)
        self.log_std = torch.nn.Parameter(torch.full((action_space,), init_log_std))


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

        sigma = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
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
    def __init__(self, policy, device='cpu', algorithm='reinforce',
     baseline_mode='none', baseline_value=0.0, baseline_alpha=0.05,
     critic_weight=1.0, lr=1e-3, actor_lr=None, critic_lr=None,
     gamma=0.99, grad_clip=1.0, entropy_weight=0.0,
     normalize_advantage=False):

        self.train_device = device
        self.policy = policy.to(self.train_device)

        if actor_lr is None:
            actor_lr = lr
        if critic_lr is None:
            critic_lr = lr

        actor_params = (
            list(self.policy.fc1_actor.parameters()) +
            list(self.policy.fc2_actor.parameters()) +
            list(self.policy.fc3_actor_mean.parameters()) +
            [self.policy.log_std]
        )

        critic_params = (
            list(self.policy.fc1_critic.parameters()) +
            list(self.policy.fc2_critic.parameters()) +
            list(self.policy.fc3_critic_v.parameters())
        )

        self.optimizer = torch.optim.Adam([
            {"params": actor_params, "lr": actor_lr},
            {"params": critic_params, "lr": critic_lr},
        ])

        self.gamma = gamma
        self.algorithm = algorithm
        self.critic_weight = critic_weight
        self.grad_clip = grad_clip
        self.entropy_weight = entropy_weight
        self.normalize_advantage = normalize_advantage

        self.baseline_mode = baseline_mode
        self.baseline_value = baseline_value
        self.baseline_alpha = baseline_alpha
        self._reset_episode_storage()

    def _reset_episode_storage(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.action_entropies = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).flatten()
        action_entropies = torch.stack(self.action_entropies, dim=0).to(self.train_device).flatten()
        states = torch.stack(self.states, dim=0).to(self.train_device)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).flatten()
        done = torch.tensor(self.done, dtype=torch.float32, device=self.train_device)

        self._reset_episode_storage()

        metrics = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": action_entropies.mean().item(),
            "baseline": float(self.baseline_value),
        }
        
        if self.algorithm == 'reinforce':

          #
          # TASK 2:
          #   - compute discounted returns
          returns = discount_rewards(rewards, self.gamma)

          if self.baseline_mode == 'none':
            advantages = returns

          elif self.baseline_mode == 'normalize':
            advantages = standardize(returns)
          
          elif self.baseline_mode == 'constant': 
            advantages = returns - self.baseline_value

          elif self.baseline_mode == 'ema':
            advantages = returns - self.baseline_value
            episode_baseline = returns.mean().item()

            self.baseline_value = (
                    (1.0 - self.baseline_alpha) * self.baseline_value
                    + self.baseline_alpha * episode_baseline
                )
            # normalized_returns = returns/(returns.std() + 1e-8)
          else:
                raise ValueError(f"Unknown baseline_mode: {self.baseline_mode!r}")

          if self.normalize_advantage and self.baseline_mode != "normalize":
                advantages = standardize(advantages)

          #   - compute policy gradient loss function given actions and returns
          actor_loss = -(action_log_probs * advantages.detach()).mean()
          entropy_loss = -self.entropy_weight * action_entropies.mean()
          loss = actor_loss + entropy_loss
          critic_loss = torch.zeros((), device=self.train_device)

          #   - compute gradients and step the optimizer
          # self.optimizer.zero_grad()
          # agent_loss.backward()
          # self.optimizer.step()

        elif self.algorithm == 'actor_critic': 
          #
          # TASK 3:
          #   - compute boostrapped discounted return estimates
          _, state_values = self.policy(states)
          with torch.no_grad():
            _, next_state_values = self.policy(next_states)
            target_t = rewards + self.gamma * next_state_values * (1.0 - done)
          
          #   - compute advantage terms
          advantage_t = target_t - state_values
          actor_advantages = advantage_t.detach()
          if self.normalize_advantage:
            actor_advantages = standardize(actor_advantages)

          #   - compute actor loss and critic loss
          actor_loss = -(action_log_probs * actor_advantages.detach()).mean()
          critic_loss = F.mse_loss(state_values, target_t)
          entropy_loss = -self.entropy_weight * action_entropies.mean()
          loss = actor_loss + self.critic_weight * critic_loss + entropy_loss

        else: 
          raise ValueError(f"Unknown algorithm: {self.algorithm!r}")

        self.optimizer.zero_grad()
        loss.backward()   

        if self.grad_clip is not None and self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)

        self.optimizer.step()
        metrics.update({
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "baseline": float(self.baseline_value),
            "std_mean": self.policy.log_std.detach().clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean().item(),
        })

        return metrics


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.as_tensor(state, dtype=torch.float32, device=self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            action = torch.tanh(normal_dist.mean)
            return action, None
        
        else:   # Sample from the distribution
            raw_action = normal_dist.sample()
            action = torch.tanh(raw_action)

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            log_prob = normal_dist.log_prob(raw_action)
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + LOG_PROB_EPS)
            action_log_prob = log_prob.sum()
            action_entropy = normal_dist.entropy().sum()

        return action, action_log_prob, action_entropy


   
    def store_outcome(self, state, next_state, action_log_prob, action_entropy, reward, done):
        self.states.append(torch.as_tensor(state, dtype=torch.float32))
        self.next_states.append(torch.as_tensor(next_state, dtype=torch.float32))
        self.action_log_probs.append(action_log_prob)
        self.action_entropies.append(action_entropy)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(float(done))

