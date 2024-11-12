import torch
from torch.distributions import Normal
import numpy as np
import collections
import random

from net import PolicyNetwork, QNetwork

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done


class AC_Agent:
    '''AC agent

    Args:
        obs_space_dims (int): observation space dimensions
        hidden_dims (int): hidden dimensions
        action_space_dims (int): action space dimensions
        actor_lr (float): actor learning rate
        critic_lr (float): critic learning rate
        gamma (float): discount factor
    '''
    def __init__(self, 
                 obs_space_dims, 
                 hidden_dims, 
                 action_space_dims, 
                 actor_lr = 1e-4, 
                 critic_lr = 1e-3,  
                 gamma = 0.99):
        # Hyperparameters
        self.action_space_dims = action_space_dims
        self.gamma = gamma
        self.eps = 1e-6

        # Actor and Critic
        self.actor = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = QNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def get_action(self, state, training=True) -> float:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.actor(state)

        # Output action with maximum probability directly at testing stage
        if not training:
            return action_means[0].detach().numpy()

        # Create a normal distribution from the predicted mean and standard deviation and sample action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()

        return action.numpy()
    
    def train(self, batch):
        # Inputs
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1)

        # Target Values
        next_actions_means, next_actions_stddevs = self.actor(next_states)
        next_actions = Normal(next_actions_means + self.eps, next_actions_stddevs + self.eps).sample()
        next_values = self.critic(next_states, next_actions)
        target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # Fit the Q-function Network
        current_values = self.critic(states, actions)
        critic_loss = torch.mean(torch.nn.functional.mse_loss(current_values, target_values.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        Q = self.critic(states, actions)
        action_means, action_stddevs = self.actor(states)
        distrib = Normal(action_means + self.eps, action_stddevs + self.eps)
        probs = distrib.log_prob(actions)
        loss = -torch.matmul(Q.T, probs)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()