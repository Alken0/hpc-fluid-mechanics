import numpy as np
from tqdm import tqdm

from src.experiments.couette_flow import util
from src.shared import boltzmann, plot
from src.shared.util import States, Parameters, Saver


def run_couette_flow(params: Parameters) -> States:
    F = util.init_probability_density_function(params.x_dim, params.y_dim)
    F = util.add_boundaries(F)
    states = States()

    u = np.ones(shape=(2, params.x_dim)) * -0.1
    u[1] = 0

    # plotter = plot.Plotter(continuous=True, timeout=0.001, vmax=1, vmin=0)
    for i in tqdm(range(params.iterations)):
        boltzmann.stream(F)
        util.apply_bounce_back(F, 1, u)
        boltzmann.collision(F[:, :, 1:F.shape[2] - 2])

        states.add(F)
        # plotter.velocity(F[:, :, 1:F.shape[2] - 2], step=i)
    return states


if __name__ == '__main__':
    params = Parameters(path="data/couette-flow")
    states = run_couette_flow(params)

    Saver.save(params.save_path(), states, params)

    for i in [0, 500, 999]:
        plot.velocity_field_couette_flow(states, i, scale=1)
