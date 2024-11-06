import torch
import torch.nn as nn

class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, hidden_dims: int, action_space_dims: int):
        """Initializes the policy network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
        """
        super(Policy_Network, self).__init__()

        self.extract_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, action_space_dims)
        )
        self.prob_net = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        feat = self.extract_net(x.float())
        action_porb = self.prob_net(feat)
        return action_porb


class Value_Network(nn.Module):
    def __init__(self, obs_space_dims: int, hidden_dims: int):
        """Initializes the value network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
        """
        super(Value_Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )
    
    def forward(self, x: torch.Tensor):
        value = self.model(x.float())
        return value