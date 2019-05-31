import numpy as np


class ReplayBuffer:
    """A FIFO buffer implemented with fixed size numpy array"""
    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._buffers = {}
        self._current_index = 0
        self._filled_till = 0

    def _increment(self):
        self._current_index = (self._current_index + 1) % self._buffer_size
        self._filled_till = min(self._filled_till + 1, self._buffer_size)

    def _initialize_buffers(self, elements):
        self._buffers = {k: np.zeros([self._buffer_size, *v.shape], dtype=np.float32) \
                        for k, v in elements.items()}

    def add(self, elements):
        if len(self._buffers.keys()) == 0:
            self._initialize_buffers(elements)
        
        for k, v in elements.items():
            self._buffers[k][self._current_index] = v

        self._increment()

    def sample(self, batch_size):
        random_indices = np.random.randint(0, self._filled_till, batch_size)
        return {k: v[random_indices] for k, v in self._buffers.items()}
    
    def get_sequential(self, batch_size):
        for i in range(self._filled_till // batch_size):
            yield {k: v[i * batch_size: (i + 1) * batch_size] for k, v in self._buffers.items()}

    def clear(self):
        self._buffers = {}
        self._current_index = 0
        self._filled_till = 0