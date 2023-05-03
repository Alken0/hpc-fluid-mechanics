import numpy as np

# https://ilias.uni-freiburg.de/data/unifreiburg/lm_data/lm_2480674/Milestone_1.html
# https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c
# https://bibliographie.uni-tuebingen.de/xmlui/bitstream/handle/10900/87663/bwHPC2018-25-Pastewka-Lattice_Boltzmann_with_Python.pdf?sequence=1

# ρ(r, t) = density
# r = physical space

DIM_X = 15
DIM_Y = 10
DIM_DIRECTIONS = 9

c = np.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T

# F = probability density function
F = np.zeros(shape=(DIM_X, DIM_Y, DIM_DIRECTIONS))


def density_function(F) -> np.array:
    """
    :param F: Probability Density Function
    :return: Density Function
    """
    return np.sum(F, axis=2)


def stream(F):
    """
    modifies F itself, return is only for convenience
    :param F: Probability Density Function
    :return: Streamed Probability Density Function
    """
    assert F.shape[2] == 9, "not 9 directions"
    for i in range(1, F.shape[2]):
        F[:, :, i] = np.roll(F[:, :, i], shift=c[i], axis=(0, 1,))
    return F
