from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

from src.shared import boltzmann
from src.shared.util import States, Parameters, Point

DPI = 400


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

    def density(self, F: np.array, step: Optional[int] = None) -> None:
        """
        plots the density function of F
        :param F: Probability Density Function of shape (c, x, y)
        :param step: shows step in title
        """
        plt.figure(1)
        plt.title(f'Density Function' if step is None else f'Density Function @{step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        data = np.swapaxes(boltzmann.density(F), 0, 1)
        plt.imshow(data, cmap='magma', vmin=self._vmin, vmax=self._vmax)
        cbar = plt.colorbar()
        cbar.set_label("density", labelpad=+1)
        self._show()

    def velocity(self, F: np.array, step: Optional[int] = None) -> None:
        """
        plots the velocity function of F
        :param F: Probability Density Function of shape (c, x, y)
        :param step: shows step in title
        """
        plt.figure(1)
        plt.title(f'Velocity Function' if step is None else f'Velocity Function @{step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        data = np.swapaxes(boltzmann.velocity(F), 1, 2)
        plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=1)
        self._show()

    def stream(self, F: np.array, step: int, path: Optional[str] = None):
        fig = plt.figure(dpi=DPI)
        plt.title(f'Stream Function' if step is None else f'Stream Function @{step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        x, y = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[2]))
        u, v = np.swapaxes(boltzmann.velocity(F), 1, 2)
        plt.streamplot(x, y, u, v)
        if path is None:
            self._show()
        else:
            save_fig(fig, path, f"fig_{step}")


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


def density_heatmap(states: States, step: int):
    plt.figure()
    plt.title(f'Density Function @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    density = boltzmann.density(states[step])
    data = np.swapaxes(density, 0, 1)
    plt.imshow(data, cmap='magma')
    cbar = plt.colorbar()
    cbar.set_label("density", labelpad=+1)
    plt.show()


def density_aggregate_over_time(states: States, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
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
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, "density_aggregate_over_time")


def velocity_for_step_at_columns_analytical(
        states: States, col: int, steps: List[int], analytical_solution, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity at column {col}')
    plt.xlabel('Y')
    plt.ylabel('Velocity')

    y = range(states[0].shape[2])
    for step in steps:
        velocity = boltzmann.velocity(states[step])
        column = velocity[0, col, :]
        plt.plot(y, column, label=f"step {step}")

    plt.plot(y, analytical_solution, label=f"analytical solution")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_for_step_at_columns_analytical")


def velocity_for_step_at_columns(states: States, columns: List[int], step: int,
                                 path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity at timestep {step}')
    plt.xlabel('Y')
    plt.ylabel('Velocity')

    velocity = boltzmann.velocity(states[step])
    y = range(velocity.shape[2])

    for col in columns:
        column = velocity[0, col, :]
        plt.plot(y, column, label=f"column {col}")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_at_columns_for_step_{step}")


def density_at_column_x(states: States, col: int, steps: List[int], path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Density in the Center Row at different timesteps')
    plt.xlabel('Y')
    plt.ylabel('Density')

    y = range(states[0].shape[2] - 2)
    for step in steps:
        column = boltzmann.density(states[step])[1:-1, col]
        plt.plot(y, column, label=f"density @{step}")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"density_at_column_x")


def velocity_at_column(states: States, col: int, steps: List[int], path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity in Time for Column x={col} ')
    plt.xlabel('Y')
    plt.ylabel('Velocity')

    y = range(states[0].shape[2])
    plt.plot(y, [0 for _ in range(len(y))], label="zero-line")

    for step in steps:
        column = boltzmann.velocity(states[step])[0, col, :]
        plt.plot(y, column, label=f"velocity @{step}")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, "velocity_at_column_x")


def velocity_aggregate_over_time(states: States, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
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
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, "velocity_aggregate_over_time")


def velocity_over_time_at(states: States, point: Point):
    plt.figure(1)
    plt.title(f'velocity over time at position {point}')
    plt.xlabel('Time')
    plt.ylabel('Density')

    velocities = [boltzmann.velocity(s)[point.to_tuple()] for s in states.get_states()]
    iterations = range(len(velocities))

    plt.plot(iterations, velocities, label=f"velocity")

    plt.legend()
    plt.show()


def velocity_field(states: States, step: int, scale: Optional[float] = None, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity Field @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    data = np.swapaxes(boltzmann.velocity(states[step]), 1, 2)
    plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=scale)
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_field_{step}")


def stream_field_raw(states: np.ndarray, step: int, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Stream Field @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    x, y = np.meshgrid(np.arange(states.shape[1]), np.arange(states.shape[2]))
    u, v = np.swapaxes(boltzmann.velocity(states), 1, 2)
    plt.streamplot(x, y, u, v)
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"stream_field_{step}")


def stream_field(states: States, step: int, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Stream Field @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    x, y = np.meshgrid(np.arange(states[step].shape[1]), np.arange(states[step].shape[2]))
    u, v = np.swapaxes(boltzmann.velocity(states[step]), 1, 2)
    plt.streamplot(x, y, u, v)
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"stream_field_{step}")


def velocity_field_couette_flow(states: States, step: int, scale: float = 1.0, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity Field @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')

    # plot velocities
    velocities = boltzmann.velocity(states[step][:, :, 1:states[step].shape[2] - 2])
    data = np.swapaxes(velocities, 1, 2)
    plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=scale)

    # plot boundaries exactly inbetween artificial boundary and shown points
    y = range(states[step].shape[1])
    upper_boundary = np.ones(states[step].shape[1]) * states[step].shape[2] - 3 - 0.5
    lower_boundary = np.ones(states[step].shape[1]) * -0.5
    plt.plot(y, upper_boundary, label="upper (moving) boundary")
    plt.plot(y, lower_boundary, label="lower (static) boundary")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_field_couette_flow_{step}")


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


def velocity_against_ideal_over_time(states: States, params: Parameters, point: Point, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity over time at position {point.to_tuple()}')
    plt.xlabel('Time')
    plt.ylabel('Velocity')

    L_z = states[0].shape[2]
    a_0 = params.epsilon
    omega = params.omega
    z = point.y

    ideals = [boltzmann.scaled_analytic_solution(a_0, t, z, L_z, omega) for t in range(len(states))]
    velocities = [boltzmann.velocity(s)[point.to_tuple()] for s in states]

    iterations = range(len(velocities))
    plt.plot(iterations, velocities, label="measured")
    plt.plot(iterations, np.array(ideals), label="ideal")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, "velocity_against_ideal")


def viscosity_against_omega(viscosities: List[float], omegas: List[float], point: Point, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f"Viscosity against Omega @{point.to_tuple()}")
    plt.xlabel('Omega')
    plt.ylabel('Viscosity')

    plt.plot(omegas, viscosities)

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, "viscosity_against_omega")


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


def save_fig(fig: plt.Figure, path: str, name: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}/{name}.jpg', dpi=fig.dpi)
    plt.clf()
    plt.close(fig)
