from abc import ABC, abstractmethod

from src.log.log_body import LogBody


class BaseFilter(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, image, log_body: LogBody):
        pass
