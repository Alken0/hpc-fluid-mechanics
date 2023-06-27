from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.shared import boltzmann
from src.shared.util import States


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
        data = np.swapaxes(boltzmann.density(F), 0, 1)
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
        data = np.swapaxes(boltzmann.velocity(F), 1, 2)
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
    density = boltzmann.density(F)
    for x in range(density.shape[0]):
        output = ""
        for y in range(density.shape[1]):
            output += f" {density[x][y]:.2f}" if y != 0 else f"{density[x][y]:.2f}"
        print(output)
    print()


def density_aggregate_over_time(states: list[np.array]):
    plt.figure()
    plt.title(f'Density Over Time')
    plt.xlabel('Time')
    plt.ylabel('Density')

    densities = [boltzmann.density(s) for s in states]
    iterations = range(len(densities))

    max_densities = [d.max() for d in densities]
    min_densities = [d.min() for d in densities]
    avg_densities = [np.average(d) for d in densities]

    plt.plot(iterations, max_densities, label="max")
    plt.plot(iterations, avg_densities, label="avg")
    plt.plot(iterations, min_densities, label="min")

    plt.legend()
    plt.show()


def velocity_aggregate_over_time(states: list[np.array]):
    plt.figure(1)
    plt.title(f'Velocity Over Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity')

    velocities = [boltzmann.velocity(s) for s in states]
    iterations = range(len(velocities))

    max_velocities = [d.max() for d in velocities]
    min_velocities = [d.min() for d in velocities]
    avg_velocities = [np.average(d) for d in velocities]

    plt.plot(iterations, max_velocities, label="max")
    plt.plot(iterations, avg_velocities, label="avg")
    plt.plot(iterations, min_velocities, label="min")

    plt.legend()
    plt.show()


def velocity_over_time_at(states: list[np.array], point=(0, 0)):
    plt.figure(1)
    plt.title(f'velocity over time at position {point}')
    plt.xlabel('Time')
    plt.ylabel('Density')

    velocities = [boltzmann.velocity(s)[0][point] for s in states]
    iterations = range(len(velocities))

    plt.plot(iterations, velocities, label=f"velocity")

    plt.legend()
    plt.show()


def velocity_field(states: [np.array], step: int, scale: float = 0.06):
    plt.figure()
    plt.title(f'Velocity Function @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    data = np.swapaxes(boltzmann.velocity(states[step]), 1, 2)
    plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=scale)
    plt.show()


def density_over_time_at(states: list[np.array], point=(0, 0)):
    plt.figure(1)
    plt.title(f'density over time at position 0 0')
    plt.xlabel('Time')
    plt.ylabel('Density')

    densities = [boltzmann.density(s)[point] for s in states]
    iterations = range(len(densities))

    plt.plot(iterations, densities, label="@position 0, 0")

    plt.legend()
    plt.show()


def velocity_against_ideal_over_time(states: States):
    params = states.parameters
    point = states.parameters.point

    plt.figure()
    plt.title(f'Velocity over time at position {point.to_tuple()}')
    plt.xlabel('Time')
    plt.ylabel('Velocity')

    L_z = states.get_states()[0].shape[2]
    a_0 = params.epsilon
    omega = params.omega
    z = point.y

    ideals = [boltzmann.scaled_analytic_solution(a_0, t, z, L_z, omega) for t in range(states.get_num_states())]
    velocities = [boltzmann.velocity(s)[point] for s in states.get_states()]

    iterations = range(len(velocities))
    plt.plot(iterations, velocities, label="measured")
    plt.plot(iterations, np.array(ideals), label="ideal")

    plt.legend()
    plt.ioff()
    plt.show()


def print_velocity(F, axis: int) -> None:
    """
        prints the velocity of the probability density function (F) to the terminal
        :param F: Probability Density Function of shape (c, x, y)
        """
    velocity = boltzmann.velocity(F)
    for x in range(velocity.shape[1]):
        output = ""
        for y in range(velocity.shape[2]):
            output += f" {velocity[axis][x][y]:.2f}" if y != 0 else f"{velocity[axis][x][y]:.2f}"
        print(output)
    print()


def save_density(state: States):
    pass