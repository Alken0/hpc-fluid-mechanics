from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import States, Parameters


def init(x_dim: int, y_dim: int, u_sliding: float) -> Tuple[np.array, np.array]:
    """
    :param x_dim: L_x
    :param y_dim: L_y
    :param u_sliding: velocity of sliding top
    :return: initial condition for the couette flow
    """
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    F = boltzmann.equilibrium(rho, u)

    sliding_u = np.ones(shape=(2, x_dim)) * u_sliding
    sliding_u[1] = 0

    return F, sliding_u


def run_couette_flow(params: Parameters) -> States:
    F, sliding_u = init(params.x_dim, params.y_dim, params.sliding_u)

    states = States()
    for i in tqdm(range(params.iterations)):
        boltzmann.collision(F, omega=params.omega)
        boltzmann.slide_top(F, params.sliding_rho, sliding_u)
        F_star = np.copy(F)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, F_star, bot=True)
        states.add(F)
    return states


if __name__ == '__main__':
    params = Parameters(
        path="data/couette_flow",
        x_dim=10,
        y_dim=10,
        omega=1.0,
        sliding_u=-0.1,
        sliding_rho=1,
        iterations=1000
    )
    states = run_couette_flow(params)

    for step in [0, 10, 999]:
        plot.velocity_field_couette_flow(states, step, path=params.path)
