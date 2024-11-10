from models.layers.config import Config, SensationConfig, LocationConfig


def get_view_config():
    location = LocationConfig(NB_MODULES=12, MODULE_SIDE_LENGTH=6)
    return Config(
        sensation=SensationConfig(NB_COLUMNS=16, NB_CELLS_PER_COLUMNS=7, LOCATION_CONFIG=location),
        location=location,
        name='view'
    )
