import numpy as np
from numpy import ndarray


class Dendrite:
    def __init__(self, threshold: float, nb_synapses: int):
        self._threshold = threshold
        self._synapses = np.zeros(nb_synapses)

    def activate(self, layer_activity: ndarray):
        return self._synapses.dot(layer_activity) >= self._threshold

    def set_synapses(self, synapses: ndarray):
        self._synapses = synapses

    def is_active(self):
        return self._synapses.sum() != 0
