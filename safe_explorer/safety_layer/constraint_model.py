from torch.nn.init import uniform_

from safe_explorer.core.config import Config
from safe_explorer.core.net import Net


class ConstraintModel(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().safety_layer.constraint_model
        
        super(ConstraintModel, self)\
            .__init__(observation_dim,
                      action_dim,
                      config.layers,
                      config.init_bound,
                      uniform_,
                      None)