import torch
from torch.distributions import Normal
import numpy as np

from net import Policy_Network

class PG_Agent:
    """Policy gradient agent implementation.

    Args:
        obs_space_dims: Dimension of the observation space
        hidden_dims: Dimension of the hidden layer
        action_space_dims: Dimension of the action space
        learning_rate: Learning rate
        gamma: Discount factor
    """
    def __init__(self, 
                 obs_space_dims: int, 
                 hidden_dims: int, 
                 action_space_dims: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.eps = 1e-6  # small number for mathematical stability
        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, hidden_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def get_action(self, state) -> float:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted mean and standard deviation
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        # store the probability of the sampled action
        self.probs.append(prob)

        return action.numpy()

    def train(self):
        # Calculate the discounted rewards
        running_g = 0
        gs = []
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        # Calculate the loss
        deltas = torch.tensor(gs).view(1, -1)
        log_probs = torch.stack(self.probs)
        loss = -torch.matmul(deltas, log_probs)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty the lists
        self.probs = []
        self.rewards = []