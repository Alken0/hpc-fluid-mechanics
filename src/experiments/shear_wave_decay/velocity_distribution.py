import numpy as np
from numpy import testing
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import Parameters, States, Saver, Point


def run_velocity(params: Parameters) -> States:
    F = init_with_sinus_on_velocity(params.x_dim, params.y_dim, params.epsilon)
    states = States()

    for t in tqdm(range(params.iterations)):
        boltzmann.stream(F)
        boltzmann.collision(F, omega=params.omega)
        states.add(F)

    Saver.save(params.path, states, params)

    return states


def init_with_sinus_on_velocity(x_dim: int, y_dim: int, epsilon=0.5) -> np.array:
    density = np.ones(shape=(x_dim, y_dim))

    # determine velocity
    y = np.arange(0, y_dim)
    velocity = np.zeros(shape=(2, x_dim, y_dim))
    velocity[0] = velocity[0] + epsilon * np.sin(2 * np.pi * y / y_dim)

    F = boltzmann.equilibrium(density, velocity)
    _check_conditions_velocity(F, x_dim, y_dim)
    return F


def _check_conditions_velocity(F: np.array, x_dim: int, y_dim: int):
    testing.assert_almost_equal(boltzmann.velocity(F)[1],
                                np.zeros(shape=(x_dim, y_dim))), "velocity in y-direction is off"
    testing.assert_almost_equal(boltzmann.density(F), np.ones(shape=(x_dim, y_dim))), "density is off"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), boltzmann.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"


if __name__ == '__main__':
    params = Parameters(
        path="data/shear-wave-decay/velocity",
        x_dim=100,
        y_dim=100,
        epsilon=0.5,
        iterations=1000
    )
    states = run_velocity(params)

    Saver.save(params.path, states, params)

    plot.velocity_at_x_column(states, 1, [0, 500, 999], path=params.path)

    scale = states[0].max()
    plot.velocity_field(states, step=0, scale=scale, path=params.path)
    plot.velocity_field(states, step=500, scale=scale, path=params.path)
    plot.velocity_field(states, step=999, scale=scale, path=params.path)

    plot.velocity_against_ideal_over_time(states, params, Point(0, 10, 10), path=params.path)
    plot.velocity_aggregate_over_time(states, path=params.path)
