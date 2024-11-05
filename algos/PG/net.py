import torch
import torch.nn as nn

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, hidden_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
        """
        super(Policy_Network, self).__init__()

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU()
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_dims, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_dims, action_space_dims)
        )

    def forward(self, x: torch.Tensor):
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        # assure positive stddev
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))
        return action_means, action_stddevs