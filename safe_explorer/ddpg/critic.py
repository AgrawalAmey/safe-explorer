import torch
from torch.nn import Linear, Module
import torch.nn.functional as F

from safe_explorer.core.config import Config
from safe_explorer.ddpg.net import Net
from safe_explorer.ddpg.utils import init_fan_in_uniform

class Critic(Module):
    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()
        
        config = Config.get().ddpg.critic

        self._first_layer = Linear(observation_dim, config.layer_dims[0])
        init_fan_in_uniform(self._first_layer)

        self._model = Net(config.layer_dims[0] + action_dim,
                          1,
                          config.layer_dims[1:],
                          None,
                          config.init_bound)

    def forward(self, observation, action):
        first_layer_output = F.relu(self._first_layer(observation))
        return self._model(torch.cat([first_layer_output, action], dim=1))