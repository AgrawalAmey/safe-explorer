from functional import seq
import numpy as np
import torch
from torch.nn import Linear, Module
from torch.nn.init import uniform
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

        self.last_activation = last_activation

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        self.layers = (seq(_layer_dims[:-1])
                        .zip(_layer_dims[1:])
                        .map(lambda x, y: Linear(x, y))
                        .to_list())

        self.init_weights(init_bound)
    
    def init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (self.layers[:-1]
             .map(lambda x: x.weight)
             .for_each(init_fan_in_uniform))
        # Init last layer with uniform initializer
        uniform(self.layers[-1], -bound, bound)

    def forward(self, inp):
        # If last_activation is none, add a do-nothing function 
        last_activation = self.last_activation if self.last_activation else lambda x: x
        # Use ReLU for all layers but last one
        activations = [F.relu] * (len(self.layers) - 1) + last_activation
        
        return (seq(self.layers)
                    .zip(activations)
                    .map(lambda x: lambda y: x[1](x[0](x))) # activation(layer(x))
                    .fold_left(inp, lambda x, layer_func: layer_func(x)))