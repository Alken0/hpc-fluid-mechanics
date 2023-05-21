from unittest import TestCase

import numpy as np
from numpy import testing

from src import lib


class TestDensity(TestCase):
    def test_density_stays_same(self):
        np.random.seed(0)
        F = np.random.uniform(low=0, high=0.1, size=(9, 3, 7))
        density = np.sum(lib.density(F))

        for i in range(100):
            lib.stream(F)
            lib.collision(F)

            testing.assert_almost_equal(np.sum(lib.density(F)), density)

    def test_velocity_goes_down_with_collision(self):
        np.random.seed(0)
        F = np.random.uniform(low=0, high=0.2, size=(9, 3, 7))

        for i in range(400):
            lib.stream(F)
            lib.collision(F)
        velocity_x = np.sum(lib.velocity(F)[0])
        velocity_y = np.sum(lib.velocity(F)[1])
        zeros = np.zeros_like(velocity_x)

        testing.assert_almost_equal(velocity_x, zeros)
        testing.assert_almost_equal(velocity_y, zeros)

    def test_velocity_stays_same_without_collision(self):
        np.random.seed(0)
        F = np.random.uniform(low=0, high=0.2, size=(9, 3, 7))
        velocity = lib.velocity(F)

        for i in range(400):
            lib.stream(F)

        output = lib.velocity(F)
        testing.assert_almost_equal(np.sum(output[0]), np.sum(velocity[0]))
        testing.assert_almost_equal(np.sum(output[1]), np.sum(velocity[1]))

    def test_stream_without_collision_ends_up_at_same_position_again(self):
        F = np.random.uniform(low=0, high=0.1, size=(9, 3, 3))
        copy = np.copy(F)

        for i in range(3):
            lib.stream(F)

        testing.assert_array_equal(F, copy)
