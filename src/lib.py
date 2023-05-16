import numpy as np


def density(F: np.array) -> np.array:
    """
    :param F: Probability Density Function of shape (c, x, y)
    :return: Density Function ρ
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


def collision(F: np.array, tau=1) -> None:
    """
    modifies F itself
    :param F: Probability Density Function of shape (c, x, y)
    :param tau: collision timescale
    """
    rho = density(F)
    u = velocity(F)

    cdot3u = 3 * np.einsum('ai,axy->ixy', c, u)
    usq = np.einsum('axy->xy', u * u)
    wrho = np.einsum('i,xy->ixy', w, rho)
    feq = wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :, :])

    F += -(1.0 / tau) * (F - feq)
