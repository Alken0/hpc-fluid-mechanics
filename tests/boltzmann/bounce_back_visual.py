from itertools import count

import numpy as np

from src.shared import boltzmann
from src.shared.plot import Plotter


def main():
    F = np.zeros(shape=(9, 10, 10))
    F[7, 4, 3] = 1

    plotter = Plotter(timeout=0.1, continuous=True)
    for i in count():
        boltzmann.stream(F)
        boltzmann.bounce_back(F, left=True, right=True, top=True, bot=True)
        plotter.density(F, step=i)


if __name__ == '__main__':
    main()
