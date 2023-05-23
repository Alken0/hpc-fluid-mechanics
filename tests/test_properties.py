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

    def test_density_distribution_borders_come_closer_with_collision(self):
        np.random.seed(0)
        F = np.random.uniform(low=0, high=0.2, size=(9, 5, 4))
        F[0, 0, 1] = 1

        minimum = lib.density(F).min()
        maximum = lib.density(F).max()
        for i in range(100):
            lib.stream(F)
            lib.collision(F)

            if i % 10 == 9:
                new_max = lib.density(F).max()
                new_min = lib.density(F).min()

                testing.assert_array_less(new_max, maximum)
                testing.assert_array_less(minimum, new_min)

                maximum = new_max
                minimum = new_min

    def test_velocity_stays_same_without_collision(self):
        np.random.seed(0)
        F = np.random.uniform(low=0, high=0.2, size=(9, 3, 3))
        velocity = lib.velocity(F)

        for i in range(3):
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

    def test_density_is_always_positive(self):
        F = np.random.uniform(low=0, high=0.1, size=(9, 3, 4))
        lower_bound = np.zeros_like(F)
        testing.assert_array_less(lower_bound, F)

        for i in range(3):
            lib.stream(F)
            lib.collision(F)
            testing.assert_array_less(lower_bound, F)

        density = np.random.uniform(low=0.01, high=0.2, size=(3, 4))
        velocity = np.random.uniform(low=0.01, high=0.2, size=(2, 3, 4))
        F = lib.equilibrium(density, velocity)
        testing.assert_array_less(lower_bound, F)
