import numpy as np
from matplotlib import pyplot as plt

from models.layers.location import LocationLayer
from models.senses.config import get_view_config


def rotate(p, a):
    x = p[0]
    y = p[1]
    rad = np.deg2rad(a)
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([x * c - y * s, x * s + y * c])


def rotate_to_rhombus(p):
    # return p * np.linalg.matrix_power(np.matrix([
    #     [np.cos(np.deg2rad(0)), np.cos(np.deg2rad(60))],
    #     [np.sin(np.deg2rad(0)), np.sin(np.deg2rad(60))]
    # ]), 1)
    return p[0] * np.array([1, 0]) + p[1] * rotate(np.array([0, 1]), a=-30)


def plot_modules():
    config = get_view_config()
    location_layer = LocationLayer(config)

    # points = [rotate_to_rhombus(cell._phase) for cell in g._cells]
    # x = [p[0] for p in points]
    # y = [p[1] for p in points]

    # m = g._get_movement_transformation_matrix()
    # points = [cell._phase * m for cell in g._cells]
    # x = [p.getA()[0][0] for p in points]
    # y = [p.getA()[0][1] for p in points]

    fig, axes = plt.subplots(config.location.NB_MODULES)
    # fig.set_size_inches(5, nb_modules * 2.5)
    # fig.set_dpi(200)
    for i in range(config.location.NB_MODULES):
        module = location_layer._modules[i]
        points = [cell._phase for cell in module._cells]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        if config.location.NB_MODULES > 1:
            axes[i].scatter(x, y, s=module._cells[0]._size)
            axes[i].set_title(f'Module {i + 1}')
        else:
            axes.scatter(x, y, s=module._cells[0]._size)
            axes.set_title(f'Module {i + 1}')
    fig.tight_layout()
    plt.show()
    # fig.savefig('./figures/modules_points.png')

    # points = [cell._phase for cell in g._cells]
    # x = [p.getA()[0][0] for p in points]
    # y = [p.getA()[0][1] for p in points]
    #
    # plt.scatter(x, y)
    # plt.show()


def plot_cells_activation():
    location_layer = LocationLayer(get_view_config())
    m = location_layer._modules[0]
    plt.imshow(np.array([c.activate(m._bumps) for c in m._cells]))
