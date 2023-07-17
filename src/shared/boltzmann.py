import numpy as np


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


def pressure(F: np.array, pressure_in: float, pressure_out: float):
    """
    applies pressure in x-direction to probability-density-function
    :param F: np.array of shape (c,x+2,y+2) where "+2" means padding boundary of 1 on both sides
    :return: None - modifies F itself
    """

    def field_at(index):
        # helper function which returns the field at a given index and adds a needed dimension
        field = F[:, index]
        return np.expand_dims(field, 1)

    n = F.shape[1] - 2  # - "boundary-left=1" - "one for len/index"

    # calculate rho-in/out
    pressure_array = np.ones(shape=(F.shape[1], F.shape[2]))
    rho_in = pressure_array * pressure_in
    rho_out = pressure_array * pressure_out

    # get density/velocity/equilibrium of whole field for later calculations
    u = velocity(F)
    rho = density(F)
    equi = equilibrium(rho, u)

    # get needed columns of the previous field (expand to match dimensions)
    equi_1 = np.expand_dims(equi[:, 1], 1)
    equi_n = np.expand_dims(equi[:, n], 1)
    u_1 = np.expand_dims(u[:, 1], 1)
    u_n = np.expand_dims(u[:, n], 1)

    # set F at location x_0
    F_0 = equilibrium(rho_in, u_n) + (field_at(n) - equi_n)
    F[:, 0] = F_0[:, 0]
    # set F at location x_n+1
    F_n1 = equilibrium(rho_out, u_1) + (field_at(1) - equi_1)
    F[:, n + 1] = F_n1[:, n + 1]


def bounce_back(F: np.array, top=False, bot=False, left=False, right=False):
    if top:
        # redirect top-right to bottom-right
        F[7, :, -1] = np.roll(F[5, :, -1], -1)
        # redirect top-left to bottom-left
        F[8, :, -1] = np.roll(F[6, :, -1], 1)
        # redirect top to bottom
        F[4, :, -1] = F[2, :, -1]
    if bot:
        # redirect bottom-right to top-right
        F[6, :, 0] = np.roll(F[8, :, 0], -1)
        # redirect bottom-left to top-left
        F[5, :, 0] = np.roll(F[7, :, 0], 1)
        # redirect bottom to top
        F[2, :, 0] = F[4, :, 0]
    if left:
        # redirect bottom-left to bottom-right
        F[5, 0, :] = np.roll(F[7, 0, :], 1)
        # redirect top-left to top-right
        F[8, 0, :] = np.roll(F[6, 0, :], -1)
        # redirect left to right
        F[1, 0, :] = F[3, 0, :]
    if right:
        # redirect top-right to top-left
        F[6, -1, :] = np.roll(F[8, -1, :], 1)
        # redirect bottom-right to bottom-left
        F[7, -1, :] = np.roll(F[5, -1, :], -1)
        # redirect right to left
        F[3, -1, :] = F[1, -1, :]


def slide_top(F: np.array, rho: float, u: np.array):
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
