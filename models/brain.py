from numpy import ndarray

from models.senses.sense import Sense
from models.senses.config import get_view_config


class Brain:

    def __init__(self):
        self._view = Sense(config=get_view_config())

    def view(self, movement: ndarray, img: str):
        return self._view.sense(movement, img)

    def get_sense(self, name: str) -> Sense:
        return getattr(self, name)
