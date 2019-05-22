import torch
from torch.nn import Linear, Module
import torch.nn.functional as F

from safe_explorer.core.config import Config
from safe_explorer.ddpg.net import Net
from safe_explorer.ddpg.utils import init_fan_in_uniform

class Critic(Module):
    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()
        
        config = Config.get_conf()

        layer_dims = config.model.actor.layers
        init_bound = config.model.actor.init_bound

        self.first_layer = Linear(observation_dim, layer_dims[0])
        init_fan_in_uniform(self.first_layer)

        self.model = Net(layer_dims[0] + action_dim,
                         1,
                         layer_dims[1:],
                         None,
                         init_bound)
    
    def forward(self, inp, action):
        first_layer_output = F.relu(self.first_layer(inp))
        return self.model(torch.cat([first_layer_output, action], dim=1))