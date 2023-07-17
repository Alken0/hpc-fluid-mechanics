import numpy as np
from tqdm import tqdm

from src.shared import boltzmann, plot
from src.shared.util import Parameters, States, Saver


def init(x_dim, y_dim) -> np.array:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    return boltzmann.equilibrium(rho, u)


def run_poiseuille_flow(params: Parameters) -> States:
    F = init(params.x_dim + 2, params.y_dim + 2)

    states = States()
    for i in tqdm(range(params.iterations)):
        boltzmann.collision(F, omega=params.omega)
        boltzmann.pressure(F, pressure_in=params.pressure_in, pressure_out=params.pressure_out)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, top=True, bot=True)

        states.add(F)

    return states


def analytical_solution(states: States, step: int, y: int, h: int):
    """h = diameter of the tube"""
    # TODO what is dp/dx?
    viscosity = boltzmann.viscosity_for_amplitude(
        N_y=states[step].shape[2],
        y=y,
        a_0=states[0][0, y, :],
        a_t=states[step][0, y, :],
        t=step
    )
    density = boltzmann.density(states[step])[:, y]

    return -1 / (2 * viscosity * density) * y * (h - y)


if __name__ == '__main__':
    params = Parameters(
        path="data/poiseuille-flow",
        x_dim=10,
        y_dim=10,
        omega=1.0,
        pressure_in=1.05,
        pressure_out=1.0,
        iterations=1000
    )
    states = run_poiseuille_flow(params)

    Saver.save(params.path, states, params)

    for step in [10, 50, 999]:
        a = analytical_solution(states, step, y=1, h=params.y_dim)
        plot.velocity_for_step_at_columns(states, columns=[1, 5], analytical_solution=a, step=step, path=params.path)

    for step in [10, 50, 999]:
        plot.velocity_field(states, step, scale=None, path=params.path)
    plot.density_at_column_x(states, col=int(params.x_dim / 2), steps=[10, 50, 999], path=params.path)
