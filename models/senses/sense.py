import copy

from numpy import ndarray

from models.layers.config import Config
from models.layers.location import LocationLayer
from models.layers.sensation import SensoryLayer


class Sense:

    def __init__(self, config: Config):
        self.name = config.name
        self._locationLayer = LocationLayer(config)
        self._sensoryLayer = SensoryLayer(config)

    def sense(self, movement: ndarray, feature: str) -> tuple[ndarray, ndarray]:
        location_prediction = self._predict_location(movement)
        loc_copy = copy.deepcopy(location_prediction)
        sensory_prediction, sensory_learning_cells = self._predict_sensation(feature, location_prediction)
        location_prediction = self._affine_location_from_sensation(sensory_prediction)
        self._link_location_and_sensation(location_prediction, sensory_learning_cells)
        return location_prediction, sensory_prediction

    def _predict_location(self, movement: ndarray):
        return self._locationLayer.move(movement)

    def _predict_sensation(self, feature: str, location: ndarray) -> tuple[ndarray, ndarray]:
        return self._sensoryLayer.sense(feature, location)

    def _affine_location_from_sensation(self, sensation: ndarray):
        return self._locationLayer.predict_from_sensation(sensation)

    def _link_location_and_sensation(self, location: ndarray, sensory_learning_cells: ndarray):
        self._locationLayer.link_to_sensation(sensory_learning_cells, location)
        self._sensoryLayer.link_to_location(location, sensory_learning_cells)
