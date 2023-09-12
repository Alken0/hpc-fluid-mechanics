import numpy as np
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import Parameters, States


def init(x_dim, y_dim) -> np.array:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    return boltzmann.equilibrium(rho, u)


def run_poiseuille_flow(params: Parameters) -> States:
    F = init(params.x_dim, params.y_dim)

    states = States()
    for i in tqdm(range(params.iterations)):
        boltzmann.collision(F, omega=params.omega)
        boltzmann.pressure(F, pressure_in=params.pressure_in, pressure_out=params.pressure_out)
        F_star = np.copy(F)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, F_star, top=True, bot=True)

        states.add(F)

    return states


def analytical_solution(params: Parameters):
    """h = diameter of the tube"""
    viscosity = boltzmann.viscosity_for_omega(params.omega)
    rho = 1
    mu = rho * viscosity
    first = -1 / (2 * mu)
    second = (params.pressure_in - params.pressure_out) / params.y_dim
    third = np.arange(params.y_dim) * (params.x_dim - np.arange(params.y_dim))
    return -first * second * third


if __name__ == '__main__':
    params = Parameters(
        path="data/poiseuille-flow",
        x_dim=100,
        y_dim=100,
        omega=1.0,
        pressure_in=1.05,
        pressure_out=1.0,
        iterations=2000
    )
    states = run_poiseuille_flow(params)
    # Saver.save(params.path, states, params)
    # states, path = Saver.load(params.path)

    for step in [10, 999]:
        plot.velocity_for_step_at_columns(states, columns=[1, 5], step=step, path=params.path)

    plot.velocity_for_step_at_columns_analytical(
        states, int(params.x_dim / 2),
        analytical_solution=analytical_solution(params),
        steps=[10, 200, 500, 999, 1999],
        path=params.path
    )

    plot.density_at_column_x(states, col=int(params.x_dim / 2), steps=[10, 200, 500, 999], path=params.path)

    # run again with only a 10x10 field to show more readable arrows
    """params = Parameters(
        path="data/poiseuille-flow",
        x_dim=10,
        y_dim=10,
        omega=1.0,
        pressure_in=1.05,
        pressure_out=1.0,
        iterations=1000
    )
    states = run_poiseuille_flow(params)"""
    plot.velocity_field_poiseulle_flow(states, 999, scale=None, path=params.path)
