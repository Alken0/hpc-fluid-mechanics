import numpy as np

from src.shared import boltzmann


def init_probability_density_function(x_dim: int, y_dim: int) -> np.array:
    rho = np.ones(shape=(x_dim, y_dim))
    u = np.zeros(shape=(2, x_dim, y_dim))
    return boltzmann.equilibrium(rho, u)


def add_boundaries_better(F: np.array) -> np.array:
    pad_width = ((0, 0), (1, 1), (1, 1))  # (c, x, y) padding left and right
    return np.pad(F, pad_width, mode='constant', constant_values=0)


def add_boundaries(F: np.array) -> np.array:
    F = add_boundary_top(F)
    F = add_boundary_bottom(F)
    return F


def add_boundary_bottom(F: np.array):
    return np.insert(F, 0, 0, axis=2)


def add_boundary_top(F: np.array):
    return np.insert(F, F.shape[2], 0, axis=2)


def apply_bounce_back(F: np.array, F_before: np.array):
    apply_bounce_back_bottom(F, F_before)
    apply_bounce_back_top(F, F_before)


def apply_bounce_back_bottom(F: np.array, F_before: np.array):
    # redirect bottom-right to top-right
    F[6, :, 0] = F_before[8, :, 0]
    # redirect bottom-left to top-left
    F[5, :, 0] = F_before[7, :, 0]
    # redirect bottom to top
    F[2, :, 0] = F_before[4, :, 0]


def apply_bounce_back_top(F: np.array, F_before):
    len_y = F.shape[2] - 1
    # redirect top-right to bottom-right
    F[7, :, len_y] = F_before[5, :, len_y]
    # redirect top-left to bottom-left
    F[8, :, len_y] = F_before[6, :, len_y]
    # redirect top to bottom
    F[4, :, len_y] = F_before[2, :, len_y]