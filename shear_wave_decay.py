from itertools import count

import numpy as np
from numpy import testing

from src import lib, plot


def init_sinus_on_density(x_dim: int, y_dim: int, epsilon=0.01, L=1) -> np.array:
    velocity = np.zeros(shape=(2, x_dim, y_dim))

    # determine density
    x = np.arange(1, x_dim + 1) / (x_dim + 1) * L
    x = np.expand_dims(x, axis=1)
    density = np.ones(shape=(x_dim, y_dim)) * 0.1
    density = density + epsilon * np.sin(2 * np.pi * x / L)

    return lib.equilibrium(density, velocity)


def init_sinus_on_velocity(x_dim: int, y_dim: int, epsilon=0.1, L=1) -> np.array:
    density = np.ones(shape=(x_dim, y_dim))

    # determine velocity
    y = np.arange(1, y_dim + 1) / (y_dim + 1) * L
    velocity = np.zeros(shape=(2, x_dim, y_dim))
    velocity[0] = velocity[0] + epsilon * np.sin(2 * np.pi * y / L)

    return lib.equilibrium(density, velocity)


def check_conditions_density(F: np.array, x_dim: int, y_dim: int):
    testing.assert_equal(lib.velocity(F)[0], np.zeros(shape=(x_dim, y_dim)))
    testing.assert_equal(lib.velocity(F)[1], np.zeros(shape=(x_dim, y_dim)))
    testing.assert_array_less(lib.density(F), np.ones(shape=(x_dim, y_dim))), "density is too high"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), lib.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"
    assert np.sum(np.abs(lib.velocity(F))) < 0.1, "velocity is off"


def check_conditions_velocity(F: np.array, x_dim: int, y_dim: int):
    testing.assert_almost_equal(lib.velocity(F)[1], np.zeros(shape=(x_dim, y_dim))), "velocity in y-direction is off"
    testing.assert_almost_equal(lib.density(F), np.ones(shape=(x_dim, y_dim))), "density is off"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), lib.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"


def main_density(x_dim: int = 100, y_dim: int = 100):
    F = init_sinus_on_density(x_dim, y_dim)
    check_conditions_density(F, x_dim, y_dim)

    plotter = plot.Plotter(continuous=True)
    plotter.density(F, step=0)
    for t in count():
        lib.stream(F)
        lib.collision(F, omega=1)
        if t % 100 == 1:
            plotter.density(F, step=t)


def main_velocity(x_dim: int = 3, y_dim: int = 20):
    F = init_sinus_on_velocity(x_dim, y_dim)
    check_conditions_velocity(F, x_dim, y_dim)

    plotter = plot.Plotter(continuous=True)
    plotter.velocity(F, step=0)

    for t in count():
        lib.stream(F)
        lib.collision(F, omega=1)
        if t % 10 == 1:
            plotter.velocity(F, step=t)


if __name__ == '__main__':
    # main_density()
    main_velocity()
