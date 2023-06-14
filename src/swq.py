import numpy as np
from numpy import testing

from src import lib


def init_with_sinus_on_density(x_dim: int, y_dim: int, epsilon=0.01, L=1) -> np.array:
    velocity = np.zeros(shape=(2, x_dim, y_dim))

    # determine density
    x = np.arange(1, x_dim + 1) / (x_dim + 1) * L
    x = np.expand_dims(x, axis=1)
    density = np.ones(shape=(x_dim, y_dim)) * 0.1
    density = density + epsilon * np.sin(2 * np.pi * x / L)

    F = lib.equilibrium(density, velocity)
    _check_conditions_density(F, x_dim, y_dim)
    return F


def _check_conditions_density(F: np.array, x_dim: int, y_dim: int):
    testing.assert_equal(lib.velocity(F)[0], np.zeros(shape=(x_dim, y_dim)))
    testing.assert_equal(lib.velocity(F)[1], np.zeros(shape=(x_dim, y_dim)))
    testing.assert_array_less(lib.density(F), np.ones(shape=(x_dim, y_dim))), "density is too high"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), lib.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"
    assert np.sum(np.abs(lib.velocity(F))) < 0.1, "velocity is off"


def init_with_sinus_on_velocity(x_dim: int, y_dim: int, epsilon=0.5) -> np.array:
    density = np.ones(shape=(x_dim, y_dim))

    # determine velocity
    y = np.arange(0, y_dim)
    print(y)
    velocity = np.zeros(shape=(2, x_dim, y_dim))
    velocity[0] = velocity[0] + epsilon * np.sin(2 * np.pi * y / y_dim)

    print(f"velocity: {velocity[0, 2, 4]}")
    F = lib.equilibrium(density, velocity)
    _check_conditions_velocity(F, x_dim, y_dim)
    return F


def _check_conditions_velocity(F: np.array, x_dim: int, y_dim: int):
    testing.assert_almost_equal(lib.velocity(F)[1], np.zeros(shape=(x_dim, y_dim))), "velocity in y-direction is off"
    testing.assert_almost_equal(lib.density(F), np.ones(shape=(x_dim, y_dim))), "density is off"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), lib.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"
