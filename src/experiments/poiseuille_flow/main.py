import numpy as np

from src.shared import boltzmann, plot
from src.shared.util import Parameters


def init(x_dim, y_dim) -> np.array:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    return boltzmann.equilibrium(rho, u)


def run_poiseuille_flow(params: Parameters):
    F = init(params.x_dim + 2, params.y_dim + 2)

    plotter = plot.Plotter(continuous=True, timeout=0.1, vmax=1, vmin=0)
    for i in range(params.iterations):
        boltzmann.collision(F, omega=params.omega)
        boltzmann.pressure(F, pressure_in=params.pressure_in, pressure_out=params.pressure_out)
        boltzmann.stream(F)
        boltzmann.apply_bounce_back(F, top=True, bot=True)

        # debugging stuff
        print(boltzmann.velocity(F)[0, 1, 1])
        plotter.velocity(F, step=i)  # only use one of the plotter-functions at the same time
        # plotter.density(F, step=i)
        # plotter.stream(F, step=i)


if __name__ == '__main__':
    params = Parameters(
        path="data/poiseuille-flow",
        x_dim=10,
        y_dim=10,
        omega=1.0,
        pressure_in=0.03,
        pressure_out=0.3
    )
    run_poiseuille_flow(params)
