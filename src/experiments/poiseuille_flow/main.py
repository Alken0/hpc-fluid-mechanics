import numpy as np

from src.shared import boltzmann, plot
from src.shared.util import Parameters


def init(x_dim, y_dim) -> np.array:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    return boltzmann.equilibrium(rho, u)


def run_poiseuille_flow(params: Parameters):
    F = init(params.x_dim + 2, params.y_dim + 2)

    plotter = plot.Plotter(continuous=True, timeout=1, vmax=1, vmin=0)
    for i in range(params.iterations):
        boltzmann.collision(F, omega=params.omega)
        boltzmann.pressure(F, pressure_in=params.pressure_in, pressure_out=params.pressure_out)
        boltzmann.stream(F)
        boltzmann.apply_bounce_back(F, top=True, bot=True)

        # debugging stuff
        u = boltzmann.velocity(F)
        print(u[0, 1, 1])
        plotter.velocity(F, step=i)
        # plotter.density(F, step=i)
        # plotter.stream(F, step=i)


if __name__ == '__main__':
    params = Parameters(
        path="data/poiseuille-flow",
        x_dim=10,
        y_dim=10,
        omega=0.75,
        pressure_in=0.99,
        pressure_out=0.01
    )
    run_poiseuille_flow(params)
