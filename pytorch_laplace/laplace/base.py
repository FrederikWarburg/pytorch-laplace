from abc import abstractmethod


class BaseLaplace:
    def __init__(self, backend="nnj") -> None:
        super().__init__()
        """
        Args:
            backend: The backend to use. Currently only "nnj" is supported.
        """

        self.backend = backend

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
