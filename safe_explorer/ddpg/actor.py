import torch.nn.functional as F

from safe_explorer.core.config import Config
from safe_explorer.ddpg.net import Net


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get_conf()

        layer_dims = config.model.actor.layers
        init_bound = config.model.actor.init_bound

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    layer_dims,
                                    F.tanh,
                                    init_bound)