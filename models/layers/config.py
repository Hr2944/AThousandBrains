import math
from dataclasses import dataclass
from keyword import iskeyword


@dataclass
class LocationConfig:
    NB_MODULES: int
    MODULE_SIDE_LENGTH: int

    def get_nb_cells(self):
        return (self.MODULE_SIDE_LENGTH ** 2) * self.NB_MODULES

    def get_cell_readout_resolution(self):
        return 2 / self.MODULE_SIDE_LENGTH

    def get_bump_size(self):
        return 0.18172 / (self.MODULE_SIDE_LENGTH / 6)

    @staticmethod
    def get_dendrite_threshold():
        return 8


@dataclass
class SensationConfig:
    NB_COLUMNS: int
    NB_CELLS_PER_COLUMNS: int
    LOCATION_CONFIG: LocationConfig

    def get_dendrite_threshold(self):
        return math.ceil(self.LOCATION_CONFIG.NB_MODULES * 0.8)

    def get_nb_cells(self):
        return self.NB_CELLS_PER_COLUMNS * self.NB_COLUMNS

    def get_nb_columns_per_feature(self):
        return math.ceil((3 * self.NB_COLUMNS) / 50)


class Config:

    def __init__(self, sensation: SensationConfig, location: LocationConfig, name: str):
        if name.isidentifier() and not iskeyword(name) and name.islower():
            self.name = name
        else:
            raise SyntaxError(f'{name} must be a valid method name')
        self.sensation = sensation
        self.location = location
