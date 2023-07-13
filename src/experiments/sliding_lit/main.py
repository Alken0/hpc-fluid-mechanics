from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import States, Parameters, Saver


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
    plotter = plot.Plotter(timeout=0.00001, vmax=1, vmin=0)
    for i in tqdm(range(params.iterations)):
        boltzmann.collision(F)
        boltzmann.slide_top(F, 1, sliding_u)
        boltzmann.bounce_back(F, bot=True, left=True, right=True)
        boltzmann.stream(F)

        if i % 1000 == 0:
            states.add(F)
            plotter.stream(F, step=i, path=params.path)
            Saver.save(params.path, states, params)

    return states


if __name__ == '__main__':
    params = Parameters(path="data/sliding_lit", x_dim=300, y_dim=300, iterations=100000)
    states = main(params)
