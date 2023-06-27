import numpy as np
from numpy import testing
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import States, Parameters, Saver


def run_density(params: Parameters) -> States:
    F = init_with_sinus_on_density(params.x_dim, params.y_dim)
    states = States()
    for t in tqdm(range(params.iterations)):
        boltzmann.stream(F)
        boltzmann.collision(F, omega=1)
        states.add(F)
    Saver.save(params.path, states, params)
    return states


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
    params = Parameters(path_="data/shear-wave-decay/density", iterations=1000)
    states = run_density(params)

    plot.density_heatmap(states, step=0)
    plot.density_aggregate_over_time(states)
    plot.velocity_aggregate_over_time(states)

    plot.velocity_field(states, step=41, scale=0.06)
    plot.velocity_field(states, step=85, scale=0.06)
    plot.velocity_field(states, step=127, scale=0.06)
