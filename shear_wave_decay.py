from itertools import count
from typing import Tuple

from src import lib, plot, save, swq


def run_density(x_dim: int = 100, y_dim: int = 100):
    F = swq.init_with_sinus_on_density(x_dim, y_dim)
    plotter = plot.Plotter(continuous=True)
    saver = save.Saver()

    plotter.density(F, step=0)
    for t in count():
        lib.stream(F)
        lib.collision(F, omega=1)
        saver.add_state(F)

        if t % 100 == 1:
            plotter.density(F, step=t)
        if t % 1000 == 0:
            saver.save("data/shear-wave-decay/density")
            plot.velocity_over_time(saver.get_states())


def run_velocity(x_dim: int = 5, y_dim: int = 20, epsilon=0.5, omega=1.5, store=False, show=True):
    F = swq.init_with_sinus_on_velocity(x_dim, y_dim, epsilon)
    plotter = plot.Plotter(continuous=True)
    saver = save.Saver(parameters=dict(
        epsilon=epsilon,
        omega=omega
    ))

    plotter.velocity(F, step=0)
    for t in range(150):
        lib.stream(F)
        lib.collision(F, omega=omega)

        if t % 10 == 1 and show:
            plotter.velocity(F, step=t)

        saver.add_state(F)
        if t % 10 == 1 and store:
            saver.save("data/shear-wave-decay/velocity")

    return saver


def plot_shear_wave_decay_density():
    saver = save.Saver()
    saver.load("data/shear-wave-decay/density")
    plot.density_over_time(saver.get_states())


def plot_velocity_and_ideal_curve(saver: save.Saver, point: Tuple[int, int, int]):
    L_z = saver.get_states()[0].shape[2]
    a_0 = saver.parameters["epsilon"]
    omega = saver.parameters["omega"]
    z = point[2]

    ideals = [lib.scaled_analytic_solution(a_0, t, z, L_z, omega) for t in range(saver.get_num_states())]
    velocities = [lib.velocity(s)[point] for s in saver.get_states()]

    plot.velocity_over_time(velocities, ideals, point)


if __name__ == '__main__':
    point = (0, 0, 8)

    for omega in [0.25, 0.5, 0.75, 1, 1.5, 1.75]:
        saver = run_velocity(omega=omega, store=False, show=False)
        plot_velocity_and_ideal_curve(saver, point)
