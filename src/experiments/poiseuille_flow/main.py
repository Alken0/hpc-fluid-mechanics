from src.experiments.poiseuille_flow import util
from src.shared import boltzmann, plot
from src.shared.util import States, Parameters


def run_poiseuille_flow(params: Parameters) -> States:
    F = util.init_probability_density_function(params.x_dim, params.y_dim)
    F = util.add_boundaries_better(F)
    states = States()

    plotter = plot.Plotter(continuous=True, timeout=0.1, vmax=1, vmin=0)
    for i in range(params.iterations):
        boltzmann.pressure(F, pressure_in=params.pressure_in, pressure_out=params.pressure_out)
        boltzmann.stream(F)
        util.apply_bounce_back(F)
        boltzmann.collision(F, omega=params.omega)

        states.add(F)
        u = boltzmann.velocity(F)
        print(u[0, 1, 1])
        plotter.velocity(F[:, 1:-1, 1:-1], step=i)
        # plotter.stream(F[:, 1:F.shape[1] - 3, 1:F.shape[2] - 3], step=i)

    return states


if __name__ == '__main__':
    params = Parameters(path="data/poiseuille-flow", x_dim=10, y_dim=10, omega=0.75, pressure_in=1.005,
                        pressure_out=1.0)
    states = run_poiseuille_flow(params)
