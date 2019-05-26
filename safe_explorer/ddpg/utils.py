import numpy as np
from torch.nn.init import uniform


def init_fan_in_uniform(tensor):
    bound = 1. / np.sqrt(tensor.size(0))
    uniform(tensor, -bound, bound)
