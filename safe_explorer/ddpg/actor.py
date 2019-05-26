import torch.nn.functional as F

from safe_explorer.core.config import Config
from safe_explorer.ddpg.net import Net


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layer_dims,
                                    F.tanh,
                                    config.init_bound)