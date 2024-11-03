import torch
import numpy as np
import collections
import random

from net import PolicyNetwork, ValueNetwork

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


class DDPG_Agent:
    '''DDPG agent

    Args:
        obs_space_dims (int): observation space dimensions
        hidden_dims (int): hidden dimensions
        action_space_dims (int): action space dimensions
        action_bound (float): action bound
        actor_lr (float): actor learning rate
        critic_lr (float): critic learning rate
        gamma (float): discount factor
        tau (float): soft update rate
    '''
    def __init__(self, 
                 obs_space_dims, 
                 hidden_dims, 
                 action_space_dims, 
                 action_bound, 
                 actor_lr = 1e-4, 
                 critic_lr = 1e-3,  
                 gamma = 0.99, 
                 tau = 0.005):
        # Hyperparameters
        self.action_bound = action_bound
        self.action_space_dims = action_space_dims
        self.gamma = gamma
        self.tau = tau

        # Actor and Critic
        self.actor = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims, action_bound)
        self.actor_target = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims, action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def get_action(self, state):
        state = torch.FloatTensor(np.array([state]))
        action = self.actor(state).detach()
        return action[0].tolist()
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
    
    def train(self, batch):
        # Inputs
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1)

        # Target Values
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # Update Critic
        current_values = self.critic(states, actions)
        critic_loss = torch.mean(torch.nn.functional.mse_loss(current_values, target_values.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update Target Networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)