import random
from typing import List

import numpy as np
from numpy import ndarray
from scipy.stats import truncnorm

from models.layers.config import Config
from models.layers.dendrite import Dendrite
from utils.numbers import MAX_UNDER_ONE


class LocationLayer:

    def __init__(self, config: Config):
        scales = truncnorm((0.01 - 1.5) / 0.7, (10 - 1.5) / 0.7, loc=1.5, scale=0.7)
        self._modules = []
        for _ in range(config.location.NB_MODULES):
            self._modules.append(
                GridCellModule(scale=scales.rvs(), orientation=random.randint(0, 360), config=config)
            )

    def move(self, movement: ndarray):
        modules_activity = []
        for module in self._modules:
            modules_activity.append(module.move(movement))
        return np.array(modules_activity).flatten()

    def predict_from_sensation(self, sensation: ndarray):
        modules_activity = []
        for module in self._modules:
            modules_activity.append(module.update_from_sensation(sensation))
        return np.array(modules_activity).flatten()

    def link_to_sensation(self, sensory_learning_cells: ndarray, location: ndarray):
        module_length = len(location) / len(self._modules)
        for i, module in enumerate(self._modules):
            slice_start = i * module_length
            module.link_cells_to_sensation(
                location[int(slice_start):int(slice_start + module_length)],
                sensory_learning_cells
            )

    def __str__(self):
        return f'Location Layer with modules {self._modules}'


class GridCellModule:

    def __init__(self, scale: float, orientation: int, config: Config):
        self._scale = scale
        self._orientation = orientation
        self._module_length = config.location.MODULE_SIDE_LENGTH

        orientation = np.deg2rad(self._orientation)
        rotate_orientation = np.deg2rad(self._orientation + 60)
        self._movement_transformation_matrix = np.linalg.matrix_power(np.matrix([
            [self._scale * np.cos(orientation),
             self._scale * np.cos(rotate_orientation)],

            [self._scale * np.sin(orientation),
             self._scale * np.sin(rotate_orientation)]
        ]), -1)

        self._cells: List[GridCell] = []
        h = MAX_UNDER_ONE / self._module_length
        for i in range(self._module_length):
            y = (h / 2) + (i * h)
            for j in range(self._module_length):
                x = (h / 2) + (j * h)
                coordinates = self._convert_to_rhombus(np.array([x, y]))

                self._cells.append(
                    GridCell(phase=coordinates, cell_size=h * self._scale, config=config)
                )

        self._bumps: List[ndarray] = [random.choice(self._cells).get_phase()]

    def move(self, movement: ndarray):
        self._shift_bumps_based_on_movement(movement)
        return self._get_cells_activation()

    def update_from_sensation(self, sensation: ndarray):
        self._shift_bumps_based_on_sensation(sensation)
        return self._get_cells_activation()

    def _shift_bumps_based_on_sensation(self, sensation: ndarray):
        bumps = []
        for cell in self._cells:
            if cell.activate_from_dendrite(sensation):
                bumps.append(cell.get_phase())
        self._bumps = bumps or self._bumps

    def _shift_bumps_based_on_movement(self, movement: ndarray):
        transformed_movement = self._convert_to_rhombus(movement)
        for i in range(len(self._bumps)):
            self._bumps[i] = self._convert_to_rhombus((self._bumps[i] + transformed_movement) % 1)

    def _convert_to_rhombus(self, point: ndarray):
        return (point * self._movement_transformation_matrix).A1

    def _get_cells_activation(self):
        active_cells = []
        for cell in self._cells:
            active_cells.append(cell.activate(self._bumps))
        return active_cells

    def __str__(self):
        return f'Module {self._module_length}x{self._module_length}; ' \
               f'orientation {self._orientation}; ' \
               f'scale {self._scale}; ' \
               f'bumps at {self._bumps}'

    def link_cells_to_sensation(self, location: ndarray, sensory_learning_cells: ndarray):
        for i, cell_activity in enumerate(location):
            if cell_activity == 1:
                self._cells[i].link_to_sensation(sensory_learning_cells)


class GridCell:

    def __init__(self, phase: ndarray, cell_size: float, config: Config):
        self._readout_resolution = config.location.get_cell_readout_resolution()
        self._phase = phase
        self._bump_size = config.location.get_bump_size()
        self._size = cell_size
        self._dendrites: list[Dendrite] = [
            Dendrite(
                threshold=config.location.get_dendrite_threshold(),
                nb_synapses=config.sensation.get_nb_cells()
            )
            for _ in range(random.randint(6, 8))
        ]

    def activate(self, bumps: List[ndarray]):
        activations = []
        for bump in bumps:
            activations.append(self._gaussian(self._get_bump_distance(bump)))
        if len(bumps) > 1:
            activations = 1 - np.prod([1 - a for a in activations])
        else:
            activations = activations[0]
        return 1 if activations >= self._get_threshold() else 0

    def activate_from_dendrite(self, sensation):
        for dendrite in self._dendrites:
            if dendrite.activate(sensation):
                return 1
        return 0

    def get_phase(self):
        return self._phase

    def _gaussian(self, d: float):
        return np.exp(-((d ** 2) / (2 * self._bump_size ** 2)))

    def _get_bump_distance(self, bump: ndarray):
        distance = np.linalg.norm(bump - self._phase)
        if distance > 0:
            return max(0, distance - self._size)
        else:
            return distance

    def _get_threshold(self):
        return self._gaussian((self._readout_resolution / 2) * (2 / np.sqrt(3)))

    def __str__(self):
        return f'Cell at {self._phase} with threshold {self._get_threshold()}'

    def link_to_sensation(self, sensory_learning_cells: ndarray):
        dendrite = self._get_inactive_dendrite()
        if dendrite:
            dendrite.set_synapses(sensory_learning_cells)

    def _get_inactive_dendrite(self):
        for dendrite in self._dendrites:
            if not dendrite.is_active():
                return dendrite
        return None
