from functional import seq
import numpy as np
import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.init import uniform_
import torch.nn.functional as F

from safe_explorer.ddpg.utils import init_fan_in_uniform

class Net(Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_dims,
                 last_activation,
                 init_bound):
        super(Net, self).__init__()

        self._last_activation = last_activation

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        self._layers = ModuleList(seq(_layer_dims[:-1])
                                    .zip(_layer_dims[1:])
                                    .map(lambda x: Linear(x[0], x[1]))
                                    .to_list())

        self._init_weights(init_bound)
    
    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (seq(self._layers[:-1])
            .map(lambda x: x.weight)
            .for_each(init_fan_in_uniform))
        # Init last layer with uniform initializer
        uniform_(self._layers[-1].weight, -bound, bound)

    def forward(self, inp):
        out = inp

        for layer in self._layers[:-1]:
            out = F.relu(layer(out))

        if self._last_activation:
            out = self._last_activation(self._layers[-1](out))
        else:
            out = self._layers[-1](out)

        return out