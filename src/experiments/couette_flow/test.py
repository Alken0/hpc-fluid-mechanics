from itertools import count

import numpy as np

from src.experiments.couette_flow import util
from src.shared import boltzmann, plot
from src.shared.util import States

if __name__ == '__main__':
    states = States()
    F = np.ones((9, 15, 15)) * 0.1 / 9
    states.add(F)
    F[:, 5, 5] = 0
    F[0, 5, 5] = 1
    states.add(F)
    F = util.add_boundaries(F)
    states.add(F)

    plotter = plot.Plotter(continuous=True, timeout=0.01, vmax=1, vmin=0)
    for i in count():
        boltzmann.stream(F)
        states.add(F)
        util.apply_bounce_back(F)
        states.add(F)
        plotter.density(F, i)
