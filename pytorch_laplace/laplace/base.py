from abc import abstractmethod

import torch


class BaseLaplace:
    def __init__(self):
        super(BaseLaplace, self).__init__()

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
