from unittest import TestCase

import numpy as np
from numpy import testing

from src import lib


class TestBoundary(TestCase):
    def test_boundary(self):
        F = np.zeros(shape=(9, 3, 4))
        F[:, 1, 1] = 1

        mass = np.sum(F)

        lib.stream(F)
        lib.boundary(F, all=True)

        self.assertEqual(mass, np.sum(F))
        mass_per_channel = np.einsum('cxy -> c', F)
        testing.assert_equal(mass_per_channel, np.ones(9))

        # print(F.shape)
        # mass_per_channel = np.sum(F, axis=0)
        # print(mass_per_channel)
        # self.assertEqual(np.sum(F, axis=0), np.ones(9))
        # print(F)

        # testing.assert_equal(output, expected)
