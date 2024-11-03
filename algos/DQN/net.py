import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, hidden_size: int, action_space_dims: int):
        """Initializes the Q-network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_size: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
        """
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the Q-values of the Q-network.

        Args:
            x: Observation

        Returns:
            Q-values
        """
        action_value = self.net(x)
        return action_value