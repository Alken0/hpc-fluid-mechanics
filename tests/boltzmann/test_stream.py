from unittest import TestCase

import numpy as np

from src.shared import boltzmann


class Test(TestCase):
    def test_stream_shapes(self):
        F = np.zeros(shape=(9, 3, 4))

        boltzmann.stream(F)

        self.assertEqual(F.shape, (9, 3, 4))

    def test_stream_moving(self):
        F = init_channels()

        boltzmann.stream(F)

        # in x-direction: higher number => further to the right
        # in y-direction: higher number => further up
        self.assertEqual(F[0][1][1], 1)  # channel 0 stays
        self.assertEqual(F[1][2][1], 2)  # channel 1 right
        self.assertEqual(F[2][1][2], 3)  # channel 2 up
        self.assertEqual(F[3][0][1], 4)  # channel 3 left
        self.assertEqual(F[4][1][0], 5)  # channel 4 down
        self.assertEqual(F[5][2][2], 6)  # channel 5 top-right
        self.assertEqual(F[6][0][2], 7)  # channel 6 top-left
        self.assertEqual(F[7][0][0], 8)  # channel 7 bottom-left
        self.assertEqual(F[8][2][0], 9)  # channel 8 bottom-right


def init_channels(shape=(9, 3, 4)) -> np.array:
    F = np.zeros(shape=shape)
    for i in range(9):
        F[i][1][1] = i + 1
    return F
