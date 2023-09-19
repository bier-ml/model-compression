from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @property
    @abstractmethod
    def parameters_number(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def model_size_mb(self) -> int | float:
        raise NotImplementedError()

    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
