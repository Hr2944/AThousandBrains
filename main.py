import random
import string

import numpy as np
from numpy import ndarray

from models.brain import Brain
from models.senses.sense import Sense
from visualisation.location import plot_modules


def unflatten(location: ndarray, sense: Sense):
    modules = sense._locationLayer._modules
    module_size = len(location) / len(modules)
    unflattened = np.empty((0, int(module_size)))
    for i in range(len(modules)):
        slice_start = i * module_size
        unflattened = np.append(unflattened, [location[int(slice_start):int(slice_start + module_size)]], axis=0)
    return unflattened


def rnd_obj(sense: Sense):
    location = None
    sensation = None
    for i in range(10):
        location, sensation = sense.sense(
            np.random.random((1, 2)) * 10,
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
        )
    return unflatten(location, sense), sensation


if __name__ == '__main__':
    brain = Brain()
    motor_input = np.array([2, 5])
    sensory_input = 'xyz'

    location_pred, sensation_pred = brain.view(motor_input, sensory_input)

    print(location_pred)
    print(sensation_pred)

    for _ in range(100):
        location_pred, sensation_pred = brain.view(motor_input, sensory_input)

    print(location_pred)
    print(sensation_pred)

    location_after_train = rnd_obj(brain.get_sense('_view'))

    plot_modules()
