from unittest import TestCase

import numpy as np
from numpy import testing

from src import lib


class TestDensity(TestCase):
    def test_density(self):
        F = np.zeros(shape=(9, 3, 4))

        output = lib.density(F)

        expected = np.zeros(shape=(3, 4))
        testing.assert_equal(output, expected)
