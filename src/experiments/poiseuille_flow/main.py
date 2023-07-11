import numpy as np

from src.experiments.poiseuille_flow import util
from src.shared import boltzmann, plot
from src.shared.util import States, Parameters


def init(x_dim, y_dim) -> np.array:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    return boltzmann.equilibrium(rho, u)


def run_poiseuille_flow(params: Parameters) -> States:
    F = init(params.x_dim + 2, params.y_dim + 2)
    states = States()

    plotter = plot.Plotter(continuous=True, timeout=0.1, vmax=1, vmin=0)
    for i in range(params.iterations):
        boltzmann.collision(F, omega=params.omega)

        boltzmann.pressure(F, pressure_in=params.pressure_in, pressure_out=params.pressure_out)
        boltzmann.stream(F)
        util.apply_bounce_back(F)

        states.add(F)
        u = boltzmann.velocity(F)
        print(u[0, 1, 1])
        plotter.velocity(F[:, 1:-1, 1:-1], step=i)
        # plotter.stream(F[:, 1:F.shape[1] - 3, 1:F.shape[2] - 3], step=i)

    return states


if __name__ == '__main__':
    params = Parameters(
        path="data/poiseuille-flow",
        x_dim=10,
        y_dim=10,
        omega=0.75,
        pressure_in=0.99,
        pressure_out=0.01
    )
    states = run_poiseuille_flow(params)
