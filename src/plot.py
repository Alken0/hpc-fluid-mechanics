from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src import lib


# TODO animate with plt.ion() & plt.pause(1)


def density(F: np.array, step: Optional[int] = None) -> None:
    """
    plots the density function of F
    :param F: Probability Density Function of shape (c, x, y)
    :param step: shows step in title
    """
    plt.figure(1)
    plt.title(f'Density Function' if step is None else f'Density Function @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    data = lib.density(F)
    plt.imshow(data, cmap='magma')
    cbar = plt.colorbar()
    cbar.set_label("density", labelpad=+1)
    plt.show()


def velocity(F: np.array, step: Optional[int] = None) -> None:
    """
    plots the density function of F
    :param F: Probability Density Function of shape (c, x, y)
    :param step: shows step in title
    """
    plt.figure(2)
    plt.title(f'Velocity Function' if step is None else f'Velocity Function @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    data = lib.velocity(F)
    data = np.sum(data, axis=0)
    plt.imshow(data, cmap='magma')
    cbar = plt.colorbar()
    cbar.set_label("velocity", labelpad=+1)
    plt.show()


def print_pdf(F):
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


def print_density_function(F):
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
