from unittest import TestCase

import numpy as np
from numpy import testing

from src.shared import boltzmann


class TestEquilibrium(TestCase):
    def test_zeros(self):
        density = np.zeros(shape=(4, 3))
        velocity = np.zeros(shape=(2, 4, 3))

        output = boltzmann.equilibrium(density, velocity)

        expected = np.zeros(shape=(9, 4, 3))
        testing.assert_equal(output, expected)

    def test_velocity_without_density(self):
        density = np.zeros(shape=(4, 3))
        velocity = np.random.uniform(low=0.1, high=0.2, size=(2, 4, 3))

        output = boltzmann.equilibrium(density, velocity)

        expected = np.zeros(shape=(9, 4, 3))
        testing.assert_equal(output, expected)

    def test_velocity_zero(self):
        density = np.random.uniform(low=0.1, high=0.2, size=(4, 3))
        velocity = np.zeros(shape=(2, 4, 3))

        output = boltzmann.equilibrium(density, velocity)

        expected_velocity = np.zeros(shape=(2, 4, 3))
        testing.assert_equal(boltzmann.velocity(output), expected_velocity)
