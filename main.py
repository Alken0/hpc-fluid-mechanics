import numpy as np

from src import lib, plot


def main(F: np.array, render=True, max_timesteps=1000):
    for i in range(max_timesteps):
        lib.stream(F)
        lib.collision(F)
    plot.density(F, step=max_timesteps)
    plot.velocity(F, step=max_timesteps)
    # visualize_density(F)


def init_random(DIM_X, DIM_Y, DIM_DIRECTIONS):
    F = np.random.uniform(low=0, high=0.01, size=(DIM_DIRECTIONS, DIM_X, DIM_Y))
    return F


def init_zeros(DIM_X, DIM_Y, DIM_DIRECTIONS):
    F = np.zeros(size=(DIM_DIRECTIONS, DIM_X, DIM_Y))
    return F


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DIM_X = 3
    DIM_Y = 5
    DIM_DIRECTIONS = 9
    # F = probability density function

    F = init_random(DIM_X, DIM_Y, DIM_DIRECTIONS)
    F[0][1][2] += 0.05

    assert F.shape[0] == 9, "incorrect number of channels"
    assert np.sum(lib.density(F)) < 1, ""

    main(F)
