import torch
import random
import numpy as np
import collections

from net import QNetwork

class QLearningAgent:
    """Agent implementing.

    Args:
        obs_space_dims: Dimension of the observation space
        hidden_size: Dimension of the hidden layer
        action_space_dims: Dimension of the action space
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon: Exploration factor
    """
    def __init__(self, 
                 obs_space_dims: int, 
                 hidden_size: int, 
                 action_space_dims: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        # Hyperparameters
        self.obs_space_dims = obs_space_dims
        self.hidden_size = hidden_size
        self.action_space_dims = action_space_dims
        lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Networks
        self.q_net = QNetwork(obs_space_dims, hidden_size, action_space_dims)
        self.target_q_net = QNetwork(obs_space_dims, hidden_size, action_space_dims)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net.train()
        self.target_q_net.eval()

        # Loss and Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
    
    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def get_action(self, state, training=True) -> int:
        # Epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_dims - 1)
        else:
            state = torch.tensor(np.array([state]))     # add batch dimension
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()
    
    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # Compute the target Q-values and Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_q_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())

        # Compute the loss and update the network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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