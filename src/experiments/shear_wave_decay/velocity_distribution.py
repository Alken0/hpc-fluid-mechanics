import numpy as np
from numpy import testing

from src.shared import boltzmann, plot, save


def run_velocity(x_dim: int = 5, y_dim: int = 20, epsilon=0.5, omega=1.5, store=False, show=True):
    F = init_with_sinus_on_velocity(x_dim, y_dim, epsilon)
    plotter = plot.Plotter(continuous=True)
    saver = save.Saver(parameters=dict(
        epsilon=epsilon,
        omega=omega
    ))

    plotter.velocity(F, step=0)
    for t in range(150):
        boltzmann.stream(F)
        boltzmann.collision(F, omega=omega)

        if t % 10 == 1 and show:
            plotter.velocity(F, step=t)

        saver.add_state(F)
        if t % 10 == 1 and store:
            saver.save("data/shear-wave-decay/velocity")

    return saver


def init_with_sinus_on_velocity(x_dim: int, y_dim: int, epsilon=0.5) -> np.array:
    density = np.ones(shape=(x_dim, y_dim))

    # determine velocity
    y = np.arange(0, y_dim)
    print(y)
    velocity = np.zeros(shape=(2, x_dim, y_dim))
    velocity[0] = velocity[0] + epsilon * np.sin(2 * np.pi * y / y_dim)

    print(f"velocity: {velocity[0, 2, 4]}")
    F = boltzmann.equilibrium(density, velocity)
    _check_conditions_velocity(F, x_dim, y_dim)
    return F


def _check_conditions_velocity(F: np.array, x_dim: int, y_dim: int):
    testing.assert_almost_equal(boltzmann.velocity(F)[1],
                                np.zeros(shape=(x_dim, y_dim))), "velocity in y-direction is off"
    testing.assert_almost_equal(boltzmann.density(F), np.ones(shape=(x_dim, y_dim))), "density is off"
    testing.assert_array_less(np.zeros(shape=(x_dim, y_dim)), boltzmann.density(F)), "density is less than 0?!"
    assert F.shape[0] == 9, "incorrect number of channels"


def plot_velocity_and_ideal_curve(saver: save.Saver, point: Tuple[int, int, int]):
    L_z = saver.get_states()[0].shape[2]
    a_0 = saver.parameters["epsilon"]
    omega = saver.parameters["omega"]
    z = point[2]

    ideals = [boltzmann.scaled_analytic_solution(a_0, t, z, L_z, omega) for t in range(saver.get_num_states())]
    velocities = [boltzmann.velocity(s)[point] for s in saver.get_states()]

    plot.velocity_over_time(velocities, ideals, point)


if __name__ == '__main__':
    run_velocity()
