import datetime
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Point:
    c: int = 0
    x: int = 0
    y: int = 0

    def get_coordinates(self) -> Tuple[int, int]:
        return self.x, self.y

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.c, self.x, self.y


@dataclass
class Parameters:
    path: str
    """used for saving"""
    x_dim: int = 100
    y_dim: int = 100
    omega: float = 1.0
    """used for collision"""
    omega_min: int = 0
    """used in correlation viscosity omega"""
    omega_max: int = 2
    """used in correlation viscosity omega"""
    omega_step: float = 0.1
    """used in correlation viscosity omega"""
    epsilon: float = 0.5
    """used in shear wave decay"""
    sliding_rho: float = 1.0
    sliding_u: float = -0.1
    pressure_in: float = 0.3
    """used in poiseuille flow"""
    pressure_out: float = 0.03
    """used in poiseuille flow"""
    iterations: int = 1000
    time_stamp: datetime.datetime = datetime.datetime.now()  # other declarations do not work with reading from file


class States:
    def __init__(self):
        self._states = []

    def add(self, state: np.array):
        self._states.append(state.copy())

    def numpy(self) -> np.array:
        return np.array(self._states)

    def __getitem__(self, item):
        return self._states[item]

    def __len__(self):
        return len(self._states)


class Saver:

    @staticmethod
    def save(path: str, states: States, params: Parameters):
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(f"{path}/states.npy", states.numpy())
        with open(f"{path}/params.txt", 'w') as f:
            f.write(str(params))

    @staticmethod
    def load(path: str, latest=False) -> Tuple[States, Parameters]:
        if latest:
            all_runs = [x[0] for x in os.walk(path)]
            path = sorted(all_runs)[-1]
            print(f"loading run: {path}")

        state_np = np.load(f"{path}/states.npy")
        states = States()
        for i in range(state_np.shape[0]):
            states.add(state_np[i])

        with open(f"{path}/params.txt", 'r') as f:
            params: Parameters = eval(f.read())

        return states, params
