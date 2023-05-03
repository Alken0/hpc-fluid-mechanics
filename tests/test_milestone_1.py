from unittest import TestCase

import numpy as np
from numpy import testing

from src.milestone_1 import stream


class Test(TestCase):
    def test_stream_zeros(self):
        input = np.zeros((3, 4, 9))
        output = stream(input)

        testing.assert_array_equal(input, output)

    def test_stream_channel0(self):
        input = np.zeros((3, 4, 9))
        input[0, 0, 0] = 1

        output = stream(np.copy(input))

        testing.assert_array_equal(input, output)

    def test_stream_channel1(self):
        input = np.zeros((3, 4, 9))
        input[0, 0, 1] = 1

        output = stream(np.copy(input))

        expected = np.zeros((3, 4, 9))
        expected[1, 0, 1] = 1

        testing.assert_array_equal(expected, output)

    def test_stream_channel2(self):
        input = np.zeros((3, 4, 9))
        input[0, 0, 2] = 1

        output = stream(np.copy(input))

        expected = np.zeros((3, 4, 9))
        expected[0, 1, 2] = 1

        testing.assert_array_equal(expected, output)

    def test_stream_channel3(self):
        input = np.zeros((3, 4, 9))
        input[0, 0, 3] = 1

        output = stream(np.copy(input))

        expected = np.zeros((3, 4, 9))
        expected[2, 0, 3] = 1

        testing.assert_array_equal(expected, output)

    def test_stream_roll_around(self):
        input = np.zeros((3, 4, 9))
        input[0, 0, 3] = 1
        output = stream(stream(stream(np.copy(input))))
        testing.assert_array_equal(input, output)

        input = np.zeros((3, 4, 9))
        input[0, 0, 2] = 1
        output = stream(stream(stream(stream(np.copy(input)))))
        testing.assert_array_equal(input, output)
