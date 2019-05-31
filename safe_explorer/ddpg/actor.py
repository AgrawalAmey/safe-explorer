import torch

from safe_explorer.core.config import Config
from safe_explorer.core.net import Net
from safe_explorer.ddpg.utils import init_fan_in_uniform


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layers,
                                    config.init_bound,
                                    init_fan_in_uniform,
                                    torch.tanh)