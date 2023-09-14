from typing import List, Tuple

import numpy as np

from src.experiments.shear_wave_decay.velocity_distribution import run_velocity
from src.shared import boltzmann, plot
from src.shared.util import Parameters, Point


def calculate_correlation(params: Parameters, point: Point) -> Tuple[List[float], List[float]]:
    omegas = []
    viscosities = []
    steps = np.arange(params.omega_min, params.omega_max, params.omega_step)  # create iterable with omegas
    for i, omega in enumerate(steps):
        print(f"currently at {i}/{steps.shape[0]}")
        params.omega = omega
        states = run_velocity(params)
        final_amplitude = boltzmann.velocity(states[-1])[point.to_tuple()]
        nu = boltzmann.viscosity_for_amplitude(
            N_y=params.y_dim,
            y=point.y,
            a_0=params.epsilon,
            a_t=final_amplitude,
            t=params.iterations
        )
        omegas.append(params.omega)
        viscosities.append(nu)
    return omegas, viscosities


if __name__ == '__main__':
    params = Parameters(
        path="data/shear-wave-decay/correlation",
        x_dim=100,
        y_dim=100,
        iterations=1000,
        omega_min=0,
        omega_max=2,
        omega_step=0.1
    )
    point = Point(0, 1, 1)
    omegas, viscosities = calculate_correlation(params, point)
    np.save(f"{params.path}/omegas", np.array(omegas))
    np.save(f"{params.path}/viscosities", np.array(viscosities))
    plot.viscosity_against_omega(viscosities, omegas, point=point, path=params.path)
