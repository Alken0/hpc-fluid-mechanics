from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import States, Parameters


def init(x_dim: int, y_dim: int, u_sliding: float) -> Tuple[np.array, np.array]:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    F = boltzmann.equilibrium(rho, u)

    sliding_u = np.ones(shape=(2, x_dim)) * u_sliding
    sliding_u[1] = 0

    return F, sliding_u


def main(params: Parameters) -> States:
    F, sliding_u = init(params.x_dim, params.y_dim, params.sliding_u)

    states = States()
    # plotter = plot.Plotter(continuous=True, timeout=0.001, vmax=1, vmin=0)
    for i in tqdm(range(params.iterations)):
        boltzmann.collision(F)
        boltzmann.slide_top(F, params.sliding_rho, sliding_u)
        boltzmann.bounce_back(F, bot=True)
        boltzmann.stream(F)

        states.add(F)
        # plotter.velocity(F, step=i)
    return states


if __name__ == '__main__':
    params = Parameters(
        path="data/couette-flow",
        x_dim=10,
        y_dim=10,
        sliding_u=-0.1,
        sliding_rho=1
    )
    states = main(params)

    for i in [0, 500, 999]:
        plot.velocity_field_couette_flow(states, i, scale=1)
