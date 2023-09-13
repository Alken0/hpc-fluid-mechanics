import argparse
import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Cartcomm

DPI = 400


@dataclass
class Point:
    c: int = 0
    x: int = 0
    y: int = 0

    def get_coordinates(self) -> Tuple[int, int]:
        return self.x, self.y

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.c, self.x, self.y


@dataclass
class Parameters:
    path: str
    """used for saving"""
    x_dim: int = 100
    y_dim: int = 100
    omega: float = 1.0
    """used for collision"""
    omega_min: int = 0
    """used in correlation viscosity omega"""
    omega_max: int = 2
    """used in correlation viscosity omega"""
    omega_step: float = 0.1
    """used in correlation viscosity omega"""
    epsilon: float = 0.5
    """used in shear wave decay"""
    sliding_rho: float = 1.0
    sliding_u: float = -0.1
    pressure_in: float = 0.3
    """used in poiseuille flow"""
    pressure_out: float = 0.03
    """used in poiseuille flow"""
    iterations: int = 1000
    time_stamp: datetime.datetime = datetime.datetime.now()  # other declarations do not work with reading from file


class States:
    def __init__(self):
        self._states = []

    def add(self, state: np.array):
        self._states.append(state.copy())

    def numpy(self) -> np.array:
        return np.array(self._states)

    def __getitem__(self, item):
        return self._states[item]

    def __len__(self):
        return len(self._states)


class Saver:

    @staticmethod
    def save(path: str, states: States, params: Parameters):
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(f"{path}/states.npy", states.numpy())
        with open(f"{path}/params.txt", 'w') as f:
            f.write(str(params))

    @staticmethod
    def load(path: str, latest=False) -> Tuple[States, Parameters]:
        if latest:
            all_runs = [x[0] for x in os.walk(path)]
            path = sorted(all_runs)[-1]
            print(f"loading run: {path}")

        state_np = np.load(f"{path}/states.npy")
        states = States()
        for i in range(state_np.shape[0]):
            states.add(state_np[i])

        with open(f"{path}/params.txt", 'r') as f:
            params: Parameters = eval(f.read())

        return states, params


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
        data = np.swapaxes(density(F), 0, 1)
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
        data = np.swapaxes(velocity(F), 1, 2)
        plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=1)
        self._show()

    def stream(self, F: np.array, step: int, path: Optional[str] = None):
        fig = plt.figure(dpi=DPI)
        plt.title(f'Stream Function' if step is None else f'Stream Function @{step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        x, y = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[2]))
        u, v = np.swapaxes(velocity(F), 1, 2)
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
    density = density(F)
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
    density = density(states[step])
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

    densities = [density(s) for s in states]
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
        velocity = velocity(states[step])
        column = velocity[0, col, :]
        plt.plot(y, column, label=f"step {step}")

    plt.plot(y, analytical_solution / 8, label=f"analytical solution")

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

    velocity = velocity(states[step])
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
    plt.xlabel('X')
    plt.ylabel('Density')

    y = range(states[0].shape[2] - 2)
    for step in steps:
        column = density(states[step])[1:-1, col]
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
        column = velocity(states[step])[0, col, :]
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

    velocities = [velocity(s) for s in states]
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

    velocities = [velocity(s)[point.to_tuple()] for s in states.get_states()]
    iterations = range(len(velocities))

    plt.plot(iterations, velocities, label=f"velocity")

    plt.legend()
    plt.show()


def velocity_field(states: States, step: int, scale: Optional[float] = None, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity Field @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    data = np.swapaxes(velocity(states[step]), 1, 2)
    plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=scale)
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_field_{step}")


def velocity_field_poiseulle_flow(states: States, step: int, scale: Optional[float] = None, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity Field at timestep {step}')
    plt.xlabel('X')
    plt.ylabel('Y')

    # plot velocities
    data = np.swapaxes(velocity(states[step]), 1, 2)[:, 1:-1, 1:-1]
    plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=scale)

    # plot boundaries exactly inbetween artificial boundary and shown points
    y = range(data.shape[1])
    upper_boundary = np.ones(data.shape[1]) * data.shape[2] - 0.5
    lower_boundary = np.ones(data.shape[1]) * -0.5
    plt.plot(y, upper_boundary, label="static boundary", color='orange')
    plt.plot(y, lower_boundary, color='orange')

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_field_{step}")


def stream_field_sliding_lit(state: np.ndarray, step: int, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Stream Field @{step}')
    plt.xlabel('X')
    plt.ylabel('Y')

    # plot streams
    x, y = np.meshgrid(np.arange(state.shape[1]), np.arange(state.shape[2]))
    u, v = np.swapaxes(velocity(state), 1, 2)
    plt.streamplot(x, y, u, v)

    # plot boundaries
    y = range(state.shape[1])
    upper_boundary = np.ones(state.shape[1]) * state.shape[2] - 0.5
    plt.axvline(x=-0.5, color='orange', label='static boundary')
    plt.axvline(x=state.shape[2] - 0.5, color='orange')
    lower_boundary = np.ones(state.shape[1]) * -0.5
    plt.plot(y, lower_boundary, color='orange')
    plt.plot(y, upper_boundary, label="sliding lit", color='green')

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
    u, v = np.swapaxes(velocity(states[step]), 1, 2)
    plt.streamplot(x, y, u, v)
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"stream_field_{step}")


def velocity_field_couette_flow(states: States, step: int, scale: float = 1.0, path: Optional[str] = None):
    fig = plt.figure(dpi=DPI)
    plt.title(f'Velocity Field at timestep {step}')
    plt.xlabel('X')
    plt.ylabel('Y')

    # plot velocities
    velocities = velocity(states[step][:, :, 1:states[step].shape[2] - 2])
    data = np.swapaxes(velocities, 1, 2)
    plt.quiver(data[0], data[1], angles='xy', scale_units='xy', scale=scale)

    # plot boundaries exactly inbetween artificial boundary and shown points
    y = range(states[step].shape[1])
    upper_boundary = np.ones(states[step].shape[1]) * states[step].shape[2] - 3 - 0.5
    lower_boundary = np.ones(states[step].shape[1]) * -0.5
    plt.plot(y, upper_boundary, label="moving boundary", color="green")
    plt.plot(y, lower_boundary, label="static boundary", color="orange")

    plt.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_fig(fig, path, f"velocity_field_couette_flow_{step}")


def density_over_time_at(states: List[np.array], point=(0, 0)):
    plt.figure(1)
    plt.title(f'density over time at position 0 0')
    plt.xlabel('Time')
    plt.ylabel('Density')

    densities = [density(s)[point] for s in states]
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

    ideals = [scaled_analytic_solution(a_0, t, z, L_z, omega) for t in range(len(states))]
    velocities = [velocity(s)[point.to_tuple()] for s in states]

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
    velocity = velocity(F)
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


def density(F: np.array) -> np.array:
    """
    Determine the density by calculating:

    .. math::
        ρ = \\sum_i F_i.

    :param F: Probability Density Function of shape (c, x, y)
    :return: Density Function ρ of shape (x, y)
    """
    return np.sum(F, axis=0)


# stream_shift_directions
c = np.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
])


def stream(F: np.array) -> None:
    """
    modifies F itself

    :param F: Probability Density Function of shape (c, x, y) with c = 9
    """
    for i in range(1, F.shape[0]):
        F[i] = np.roll(F[i], shift=c[:, i], axis=(0, 1))


# weights for collision
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
assert np.sum(w) == 1, "weights for collision do not sum up to 1"


def velocity(F: np.array) -> np.array:
    """
    Determine the velocity by calculating:

    .. math::
        u = \\frac{1}{ρ} * \\sum_i [c_i F_i].

    :param F: Probability Density Function of shape (c, x, y)
    :return: Velocity Function u of shape (2, x, y)
    """
    return (1 / density(F)) * np.einsum('ai, ixy->axy', c, F)


def collision(F: np.array, omega) -> None:
    """
    Applies collision to F using, where :math:`F_eq` is computed by the function `equilibrium`:

    .. math::
        F = F + ω * (F_{eq} - F).

    :param F: Probability Density Function of shape (c, x, y)
    :param omega: relaxation frequency (delta-t / tau)
    """
    rho = density(F)
    u = velocity(F)
    F_eq = equilibrium(rho=rho, u=u)
    F += omega * (F_eq - F)


def equilibrium(rho: np.array, u: np.array) -> np.array:
    """
    Determines the equilibrium function using the density (ρ - rho) and velocity (u) by computing

    .. math::
        F_{eq}(ρ, u) = w * ρ * [1 + 3cu + 4.5(cu)² - 1.5 * u²]

    which can be rewritten as:

    .. math::
        F_{eq}(ρ, u) = w * ρ * [1 + 3cu * (1 + 0.5 * 3cu) - 1.5 * u²].

    :param rho: density function of shape (x, y)
    :param u: velocity function of shape (2, x, y)
    :return: Equilibrium of Probability Density Function (F_eq)
    """
    cdot3u = 3 * np.einsum('ac,axy->cxy', c, u)
    wdotrho = np.einsum('c,xy->cxy', w, rho)
    usq = np.einsum('axy,axy->xy', u, u)
    feq = wdotrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq)
    return feq


def viscosity_for_omega(omega=1) -> float:
    """
    Determines the viscosity for a given omega using the following calculation where :math:`C_s` is the speed of sound.

    .. math::
        ν = C^2_S * (\\Delta t / \\omega - \\Delta t / 2)

    This term can be simplified when using the Lattice Boltzmann Units :math:`\\Delta t = 1`, :math:`\\Delta x = 1`
    and :math:`\\rho = 1` and the scheme D2Q9. Then the speed of sound can be set to :math:`C_s = 1/ \\sqrt 3`,
    which allows to simplify the formula of the viscosity from before to

    .. math::
        ν = 1/3 (1/omega - 1/2).

    :param omega: relaxation frequency (delta-t / tau)
    :return: viscosity (ν) for omega
    """
    return 1 / 3 * (1 / omega - 1 / 2)


def scaled_analytic_solution(a0, t, z, L_z, omega=1) -> float:
    """
    To always get the exact solution for the given problem it's possible to use a scaling factor for the `analytical_solution`

    .. math::
        s = \\sin(2 \\pi / L_z) * z \n
        out = a_t * s.

    :param a0: amplitude at time t=0 (epsilon)
    :param t: timestep
    :param z: position of point
    :param L_z: Length of domain in direction z - use len(x) - 1!
    :param omega: relaxation frequency (delta-t / tau)
    :return: analytical time evolution perturbation amplitude at timestep t
    """

    solution = analytic_solution(a0, t, L_z, omega)
    scaling = np.sin(2 * np.pi * z / L_z)

    return solution * scaling


def analytic_solution(a0, t, L_z, omega=1) -> float:
    """
    calculates the analytical time evolution perturbation amplitude at timestep t using

    .. math::
        exponent = -v * t * (2 \\pi / L_z)^2 \n
        a_t = a_0 * e^{exponent}.

    To always get the exact solution for the given problem it's possible to use a scaling factor

    .. math::
        s = \\sin(2 \\pi / L_z) * z \n
        out = a_t * s.

    :param a0: amplitude at time t=0
    :param t: timestep
    :param L_z: Length of domain in direction z - use len(x) - 1!
    :param omega: relaxation frequency (delta-t / tau)
    :return: analytical time evolution perturbation amplitude at timestep t
    """

    exponent = -viscosity_for_omega(omega) * t * (2 * np.pi / L_z) ** 2
    solution = a0 * np.e ** exponent

    return solution


def pressure(F: np.array, pressure_in: float, pressure_out: float) -> None:
    """
    applies pressure in x-direction to probability-density-function
    :param F: np.array of shape (c,x,y)
    :return: None - modifies F itself
    """
    # calculate rho-in/out
    pressure_array = np.ones(shape=(1, F.shape[2]))
    rho_in = pressure_array * pressure_in
    rho_out = pressure_array * pressure_out

    # get the field at certain positions
    field_at_1 = np.expand_dims(F[:, 1], 1)
    field_at_n = np.expand_dims(F[:, -2], 1)

    # calculate velocity and equilibrium functions
    u_1 = velocity(field_at_1)
    u_n = velocity(field_at_n)
    equi_1 = equilibrium(density(field_at_1), u_1)
    equi_n = equilibrium(density(field_at_n), u_n)

    # set F at location x_0
    f_0 = equilibrium(rho_in, u_n) + (field_at_n - equi_n)
    F[:, 0] = f_0.squeeze(1)  # remove middle dimension of shape (9,1,y)
    # set F at location x_n+1
    f_n_1 = equilibrium(rho_out, u_1) + (field_at_1 - equi_1)
    F[:, -1] = f_n_1.squeeze(1)  # remove middle dimension of shape (9,1,y)


def bounce_back(F: np.array, F_star: np.array, top=False, bot=False, left=False, right=False) -> None:
    """
    applies the "bounce back" or "rigid wall" to specified walls
    this function needs to run after the streaming step because it requires information from before and after the streaming

    :param F: probability density function of shape (c,x,y) with c=9 AFTER streaming
    :param F_star: probability density function of shape (c,x,y) with c=9 BEFORE streaming
    :param top: applies boundary to the top
    :param bot: applies boundary to the bottom
    :param left: applies boundary to the left
    :param right: applies boundary to the right
    """
    if top:
        # redirect top-right to bottom-right
        F[7, :, -1] = F_star[5, :, -1]
        # redirect top-left to bottom-left
        F[8, :, -1] = F_star[6, :, -1]
        # redirect top to bottom
        F[4, :, -1] = F_star[2, :, -1]
    if bot:
        # redirect bottom-right to top-right
        F[6, :, 0] = F_star[8, :, 0]
        # redirect bottom-left to top-left
        F[5, :, 0] = F_star[7, :, 0]
        # redirect bottom to top
        F[2, :, 0] = F_star[4, :, 0]
    if left:
        # redirect bottom-left to bottom-right
        F[5, 0, :] = F_star[7, 0, :]
        # redirect top-left to top-right
        F[8, 0, :] = F_star[6, 0, :]
        # redirect left to right
        F[1, 0, :] = F_star[3, 0, :]
    if right:
        # redirect top-right to top-left
        F[6, -1, :] = F_star[8, -1, :]
        # redirect bottom-right to bottom-left
        F[7, -1, :] = F_star[5, -1, :]
        # redirect right to left
        F[3, -1, :] = F_star[1, -1, :]


def slide_top(F: np.array, rho: float, u: np.array) -> None:
    """
    applies a "sliding lit" to the top
    :param F: probability density funciton of shape (c,x,y) with c=9
    :param rho: density of the sliding lit
    :param u: velocity of the sliding lit
    """

    def calc_moving(i, i_opposite):
        second_part = 2 * w[i] * rho * (np.einsum('c,cy -> y', c[:, i], u) / (1 / 3))
        F[i_opposite, :, -1] = F[i, :, -1] - second_part

    # redirect top-right to bottom-right
    calc_moving(5, 8)
    # redirect top-left to bottom-left
    calc_moving(6, 7)
    # redirect top to bottom
    calc_moving(2, 4)


def viscosity_for_amplitude(N_y, y, a_0, a_t, t):
    """
    .. math::
        u(y, 0) = a * \\sin(2 y \\pi / N_y) \n
        k = \\sin(2 y \\pi / N_y) \n
        a = -νk²a \n
        a(t) = a_0 * e^{-νk²t} \n
        ν = (\\ln(a_0) - \\ln(a(t))) / (k² * t)
    :return: None
    """
    k = np.sin(2 * y * np.pi / N_y)
    nominator = np.log(a_0) - np.log(a_t)
    denominator = np.square(k) * t
    return nominator / denominator


def init(x_dim: int, y_dim: int, u_sliding: float) -> Tuple[np.array, np.array]:
    rho = np.ones(shape=(x_dim, y_dim), dtype=np.float64)
    u = np.zeros(shape=(2, x_dim, y_dim), dtype=np.float64)
    F = equilibrium(rho, u)

    sliding_u = np.ones(shape=(2, x_dim)) * u_sliding
    sliding_u[1] = 0

    return F, sliding_u


def get_corners(coords, size_x, size_y) -> (bool, bool, bool, bool):
    """
    :returns: top, bot, left, right
    """
    x = coords[1]
    y = coords[0]

    top, bot, left, right = False, False, False, False
    if x == 0:
        left = True
    if y == 0:
        top = True
    if x == size_x - 1:
        right = True
    if y == size_y - 1:
        bot = True
    return top, bot, left, right


def communicate(F: np.ndarray, cartcomm: Cartcomm, s_and_d: Tuple):
    sU, dU, sD, dD, sL, dL, sR, dR = s_and_d

    # shift to top (send to the top and receive from the bottom)
    receive_buffer = np.copy(F[:, :, 0])
    send_buffer = np.copy(F[:, :, -2])
    cartcomm.Sendrecv(send_buffer, dU, recvbuf=receive_buffer, source=sU)
    F[:, :, 0] = receive_buffer

    # shift to bot (send to the bottom and receive from the top)
    receive_buffer = np.copy(F[:, :, -1])
    send_buffer = np.copy(F[:, :, 1])
    cartcomm.Sendrecv(send_buffer, dD, recvbuf=receive_buffer, source=sD)
    F[:, :, -1] = receive_buffer

    # shift to left (send to the left and receive from the right
    receive_buffer = np.copy(F[:, -1, :])
    send_buffer = np.copy(F[:, 1, :])
    cartcomm.Sendrecv(send_buffer, dL, recvbuf=receive_buffer, source=sL)
    F[:, -1, :] = receive_buffer

    # shift to right (send to the right and receive from the left)
    receive_buffer = np.copy(F[:, 0, :])
    send_buffer = np.copy(F[:, -2, :])
    cartcomm.Sendrecv(send_buffer, dR, recvbuf=receive_buffer, source=sR)
    F[:, 0, :] = receive_buffer

    return F


def collect_and_plot(data: np.array, rank: int, params: Parameters, cartcomm: Cartcomm, domain_x: int,
                     domain_y: int, size: int, step: int, size_x: int):
    buf = data[:, 1:-1, 1:-1].copy()
    if rank == 0:
        print(f"currently @ {step}")
        all_data = np.zeros(shape=(9, params.x_dim, params.y_dim))
        for r in range(size):
            coords = cartcomm.Get_coords(r)
            x = coords[1]
            y = coords[0]
            if r != 0:
                cartcomm.Recv(buf, r, 0)
            x_start = x * (domain_x - 2)
            x_end = (x + 1) * (domain_x - 2)
            y_start = (size_x - y - 1) * (domain_y - 2)
            y_end = (size_x - y) * (domain_y - 2)
            # print(f"{x=} {y=} {x_start=} {x_end=} {y_start=} {y_end=} {domain_x=} {domain_y=}")
            # print(f"buf={buf.shape}  all_data={all_data.shape} slice={all_data[:, x_start:x_end, y_start:y_end].shape}")
            all_data[:, x_start:x_end, y_start:y_end] = buf
        print("finished gathering")
        plot.stream_field_sliding_lit(all_data, step=step, path=params.path)
        print("finished plotting")
    else:
        cartcomm.Send(buf, 0, 0)


def get_sources_and_destinations(cartcomm: Cartcomm):
    sR, dR = cartcomm.Shift(1, 1)
    sL, dL = cartcomm.Shift(1, -1)
    sU, dU = cartcomm.Shift(0, -1)
    sD, dD = cartcomm.Shift(0, 1)
    return sU, dU, sD, dD, sL, dL, sR, dR


def main(params: Parameters):
    # start the communicator
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f"running: processes={size}; x={params.x_dim}; y={params.y_dim}; iterations={params.iterations}")

    # get coordinates of the current process
    size_x = int(np.floor(np.sqrt(size)))
    size_y = int(size // size_x)
    cartcomm = comm.Create_cart(
        dims=[size_x, size_y],
        periods=[False, False],  # periods: True: continuous space, False not
        reorder=False
    )

    # gather more information about the current computational domain
    coords = cartcomm.Get_coords(rank)
    s_and_d = get_sources_and_destinations(cartcomm)
    top, bot, left, right = get_corners(coords, size_x, size_y)

    # start timing
    start_time = time.time()

    # create data
    domain_x = params.x_dim // size_x + 2
    domain_y = params.y_dim // size_y + 2
    F, sliding_u = init(domain_x, domain_y, params.sliding_u)

    # run simulation
    for i in range(params.iterations):
        communicate(F=F, s_and_d=s_and_d, cartcomm=cartcomm)
        collision(F, omega=params.omega)
        if top:
            slide_top(F, 1, sliding_u)
        F_star = np.copy(F)
        stream(F)
        bounce_back(F, F_star, bot=bot, left=left, right=right)
        if i % 5000 == 0:
            collect_and_plot(
                data=F,
                cartcomm=cartcomm,
                domain_x=domain_x,
                domain_y=domain_y,
                rank=rank,
                size=size,
                step=i,
                size_x=size_x,
                params=params
            )
    if rank == 0:
        print(f"total time: {time.time() - start_time}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_dim', type=int, default=324)
    parser.add_argument('-y', '--y_dim', type=int, default=324)
    parser.add_argument('-i', '--iterations', type=int, required=False, default=100000)
    args = parser.parse_args()

    parameters = Parameters(
        path="data/sliding_lit_parallel_test",
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        iterations=args.iterations,
        omega=1,
        sliding_u=0.1,
        sliding_rho=1,
    )
    main(params=parameters)
