from itertools import count

import numpy as np
from numpy import testing

from src.shared import boltzmann, plot, save, swq


def run_density(x_dim: int = 100, y_dim: int = 100):
    F = swq.init_with_sinus_on_density(x_dim, y_dim)
    plotter = plot.Plotter(continuous=True)
    saver = save.Saver()

    plotter.density(F, step=0)
    for t in count():
        boltzmann.stream(F)
        boltzmann.collision(F, omega=1)
        saver.add_state(F)

        if t % 100 == 1:
            plotter.density(F, step=t)
        if t % 1000 == 0:
            saver.save("data/shear-wave-decay/density")
            plot.velocity_over_time(saver.get_states())


def init_with_sinus_on_density(x_dim: int, y_dim: int, epsilon=0.01, L=1) -> np.array:
    velocity = np.zeros(shape=(2, x_dim, y_dim))

    # determine density
    x = np.arange(1, x_dim + 1) / (x_dim + 1) * L
    x = np.expand_dims(x, axis=1)
    density = np.ones(shape=(x_dim, y_dim)) * 0.1
    density = density + epsilon * np.sin(2 * np.pi * x / L)

    F = boltzmann.equilibrium(density, velocity)
    _check_conditions_density(F, x_dim, y_dim)
    return F


def _check_conditions_density(F: np.array, x_dim: int, y_dim: int):
    testing.assert_equal(boltzmann.velocity(F)[0], np.zeros(shape=(x_dim, y_dim)))
    testing.assert_equal(boltzmann.velocity(F)[1], np.zeros(shape=(x_dim, y_dim)))
    testing.assert_array_less(boltzmann.density(F), np.ones(shape=(x_dim, y_dim))), "density is too high"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), boltzmann.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"
    assert np.sum(np.abs(boltzmann.velocity(F))) < 0.1, "velocity is off"


if __name__ == '__main__':
    run_density()
