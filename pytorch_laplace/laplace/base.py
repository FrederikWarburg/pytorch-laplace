from abc import abstractmethod


class BaseLaplace:
    def __init__(self):
        super(BaseLaplace, self).__init__()

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
