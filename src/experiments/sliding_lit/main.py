import time
from typing import Tuple

import numpy as np

from src.shared import boltzmann
from src.shared.util import States, Parameters, Saver


def init(x_dim: int, y_dim: int, u_sliding: float) -> Tuple[np.array, np.array]:
    """initial condition for sliding_lit, also used in the parallel version"""
    rho = np.ones(shape=(x_dim, y_dim), dtype=np.float64)
    u = np.zeros(shape=(2, x_dim, y_dim), dtype=np.float64)
    F = boltzmann.equilibrium(rho, u)

    sliding_u = np.ones(shape=(2, x_dim)) * u_sliding
    sliding_u[1] = 0

    return F, sliding_u


def main(params: Parameters) -> States:
    F, sliding_u = init(params.x_dim, params.y_dim, params.sliding_u)

    for i in range(params.iterations):
        boltzmann.collision(F, omega=params.omega)
        boltzmann.slide_top(F, params.sliding_rho, sliding_u)
        F_star = np.copy(F)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, F_star, bot=True, left=True, right=True)
    return states


if __name__ == '__main__':
    params = Parameters(
        path="data/sliding_lit",
        x_dim=302,
        y_dim=302,
        iterations=100000,
        omega=1,
        sliding_u=0.1,
        sliding_rho=1,
    )
    start_time = time.time()
    states = main(params)
    print(f"total time: {time.time() - start_time}ms")
    Saver.save(params.path, states, params)
