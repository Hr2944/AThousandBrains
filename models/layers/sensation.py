import random

import numpy as np
from numpy import ndarray

from models.layers.config import Config
from models.layers.dendrite import Dendrite


class SensoryLayer:

    def __init__(self, config: Config):
        self._columns: list[MiniColumn] = []
        self._nb_columns_per_feature = config.sensation.get_nb_columns_per_feature()
        for _ in range(config.sensation.NB_COLUMNS):
            self._columns.append(
                MiniColumn(config=config)
            )

    def sense(self, feature: str, location: ndarray) -> tuple[ndarray, ndarray]:
        columns_activity = []
        learning_cells = []
        self._attribute_feature_to_columns(feature)
        for column in self._columns:
            column_activity, column_learning_cells = column.predict_sensation(location, feature)
            columns_activity.append(column_activity)
            learning_cells.append(column_learning_cells)
        return np.array(columns_activity).flatten(), np.array(learning_cells).flatten()

    def _attribute_feature_to_columns(self, feature: str):
        if feature not in [column.get_feature() for column in self._columns]:
            rng = np.random.default_rng()
            for column in rng.choice(self._columns, size=self._nb_columns_per_feature, replace=False):
                column.set_feature(feature)

    def link_to_location(self, location: ndarray, sensory_learning_cells: ndarray):
        column_size = len(sensory_learning_cells) / len(self._columns)
        for i, column in enumerate(self._columns):
            slice_start = i * column_size
            column.link_cells_to_location(
                location,
                sensory_learning_cells[int(slice_start):int(slice_start + column_size)]
            )


class MiniColumn:

    def __init__(self, config: Config):
        self._feature: str = ''
        self._cells: list[SensoryCell] = []
        for _ in range(config.sensation.NB_CELLS_PER_COLUMNS):
            self._cells.append(SensoryCell(config=config))

    def predict_sensation(self, location: ndarray, feature: str):
        if self._feature == feature:
            cells_activation = []
            for cell in self._cells:
                cells_activation.append(cell.activate_from_dendrite(location))
            if sum(cells_activation) == 0:
                return self._activate_all_cells(), self._select_random_learning_cell()
            return cells_activation, cells_activation
        return self._deactivate_all_cells(), self._no_learning_cells()

    def get_feature(self):
        return self._feature

    def set_feature(self, feature: str):
        self._feature = feature

    def _deactivate_all_cells(self):
        return [0] * len(self._cells)

    def _activate_all_cells(self):
        return [1] * len(self._cells)

    def _select_random_learning_cell(self):
        nb_cells = len(self._cells)
        learning_cells = [0] * nb_cells
        selected_cell_index = random.randint(0, nb_cells - 1)
        while self._cells[selected_cell_index].learn is True:
            selected_cell_index = random.randint(0, nb_cells - 1)
        learning_cells[selected_cell_index] = 1
        self._cells[selected_cell_index].learn = True
        return learning_cells

    def _no_learning_cells(self):
        return [0] * len(self._cells)

    def link_cells_to_location(self, location: ndarray, sensory_learning_cells: ndarray):
        for i, cell_activity in enumerate(sensory_learning_cells):
            if cell_activity == 1:
                self._cells[i].link_to_location(location)


class SensoryCell:

    def __init__(self, config: Config):
        self._dendrites: list[Dendrite] = [
            Dendrite(
                threshold=config.sensation.get_dendrite_threshold(),
                nb_synapses=config.location.get_nb_cells()
            ) for _ in range(random.randint(6, 8))
        ]
        self.learn = False

    def activate_from_dendrite(self, location: ndarray):
        for dendrite in self._dendrites:
            if dendrite.activate(location):
                return 1
        return 0

    def link_to_location(self, location: ndarray):
        dendrite = self._get_inactive_dendrite()
        if dendrite:
            dendrite.set_synapses(location)

    def _get_inactive_dendrite(self):
        for dendrite in self._dendrites:
            if not dendrite.is_active():
                return dendrite
        return None
