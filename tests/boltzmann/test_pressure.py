from unittest import TestCase

import numpy as np
from numpy import testing

from src.shared import boltzmann


class TestPressure(TestCase):
    def test_left_and_right_are_equal(self):
        density = np.zeros(shape=(100, 90))
        velocity = np.zeros(shape=(2, 100, 90))
        velocity[1] += 1
        F = boltzmann.equilibrium(density, velocity)

        for i in range(10):
            boltzmann.pressure(F, 1, 0.01)
            boltzmann.stream(F)
            boltzmann.collision(F)

            n = F.shape[1] - 2
            testing.assert_equal(F[:, 1], F[:, n + 1])
            testing.assert_equal(F[:, 0], F[:, n])
