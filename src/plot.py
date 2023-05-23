from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src import lib


class Plotter:
    def __init__(self, continuous=False, timeout=1, vmin=None, vmax=None):
        self._continuous = continuous
        if continuous:
            plt.ion()
        self._timeout = timeout
        self._vmin = vmin
        self._vmax = vmax

    def set_continuous(self) -> None:
        self._continuous = True
        plt.ion()

    def _show(self) -> None:
        if self._continuous:
            plt.pause(self._timeout)
            plt.clf()
        else:
            plt.show()

    def density(self, F: np.array, step: Optional[int] = None, figure=1) -> None:
        """
        plots the density function of F
        :param F: Probability Density Function of shape (c, x, y)
        :param step: shows step in title
        :param figure: which figure pyplot should use
        """
        plt.figure(figure)
        plt.title(f'Density Function' if step is None else f'Density Function @{step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        data = np.swapaxes(lib.density(F), 0, 1)
        plt.imshow(data, cmap='magma', vmin=self._vmin, vmax=self._vmax)
        cbar = plt.colorbar()
        cbar.set_label("density", labelpad=+1)
        self._show()

    def velocity(self, F: np.array, figure: Optional[int] = 2, step: Optional[int] = None) -> None:
        """
        plots the velocity function of F
        :param F: Probability Density Function of shape (c, x, y)
        :param step: shows step in title
        :param figure: which figure pyplot should use
        """
        plt.figure(figure)
        plt.title(f'Velocity Function' if step is None else f'Velocity Function @{step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        data = np.swapaxes(lib.velocity(F), 1, 2)
        plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=1)
        self._show()


def print_pdf(F: np.array):
    """
    prints the probability density function (F) to the terminal
    :param F: Probability Density Function of shape (c, x, y)
    """
    for x in range(3):
        for y in range(3):
            output = ""
            for c in range(9):
                output += f" {F[c][x][y]:.2f}" if c != 0 else f"{F[c][x][y]:.2f}"
            print(output)
    print()


def print_density(F: np.array):
    """
    prints the density of the probability density function (F) to the terminal
    :param F: Probability Density Function of shape (c, x, y)
    """
    density = lib.density(F)
    for x in range(density.shape[0]):
        output = ""
        for y in range(density.shape[1]):
            output += f" {density[x][y]:.2f}" if y != 0 else f"{density[x][y]:.2f}"
        print(output)
    print()


def print_velocity(F, axis: int) -> None:
    """
        prints the velocity of the probability density function (F) to the terminal
        :param F: Probability Density Function of shape (c, x, y)
        """
    velocity = lib.velocity(F)
    for x in range(velocity.shape[1]):
        output = ""
        for y in range(velocity.shape[2]):
            output += f" {velocity[axis][x][y]:.2f}" if y != 0 else f"{velocity[axis][x][y]:.2f}"
        print(output)
    print()
