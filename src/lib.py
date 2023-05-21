import numpy as np


def density(F: np.array) -> np.array:
    """
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
    :param F: Probability Density Function of shape (c, x, y)
    """
    for i in range(1, F.shape[0]):
        F[i, :, :] = np.roll(F[i], shift=c[:, i], axis=(0, 1,))


# weights for collision
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
assert np.sum(w) == 1, "weights for collision do not sum up to 1"


def velocity(F: np.array) -> np.array:
    """
    :param F: Probability Density Function of shape (c, x, y)
    :return: Velocity Function u
    """
    return np.einsum('ai, ixy->axy', c, F)


def collision(F: np.array, omega=1) -> None:
    """
    Applies collision to F using
    .. math:: F = F + omega * (F_{eq} - F).
    F_eq is computed using the function `equilibrium` \n
    :param F: Probability Density Function of shape (c, x, y)
    :param omega: collision timescale
    """
    rho = density(F)
    u = velocity(F)
    F_eq = equilibrium(density=rho, velocity=u)
    F += omega * (F_eq - F)


def equilibrium(density: np.array, velocity: np.array) -> np.array:
    """
    Determines the equilibrium function using the density (ρ) and velocity (u) by computing
    .. math::
        F_{eq}(ρ, u) = w * ρ * [1 + 3cu + 4.5(cu)² - 1.5 * u²]
    which can be rewritten as:\n
    .. math::
        F_{eq}(ρ, u) = w * ρ * [1 + 3cu * (1 + 0.5 * 3cu) - 1.5 * u²].
    :param density: density function of shape (x, y)
    :param velocity: velocity function of shape (2, x, y)
    :return: Equilibrium of Probability Density Function (F_eq)
    """
    cdot3u = 3 * np.einsum('ac,axy->cxy', c, velocity)
    wdotrho = np.einsum('c,xy->cxy', w, density)
    usq = np.einsum('axy,axy->xy', velocity, velocity)
    feq = wdotrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq)
    return feq
