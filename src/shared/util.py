import os
from dataclasses import dataclass, field, InitVar
from datetime import datetime
from typing import List
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
    path_: InitVar[str]
    path: str = field(init=False)
    x_dim: int = 100
    y_dim: int = 100
    omega: float = 1.0
    epsilon: float = 0.5
    iterations: int = 200

    def __post_init__(self, path_: str):
        self.path = f"{path_}/{datetime.now().isoformat()}"


class States:
    def __init__(self):
        self._states = []

    def add_state(self, state: np.array):
        self._states.append(state.copy())

    def get_states(self) -> List[np.array]:
        return self._states

    def get_num_states(self) -> int:
        return len(self._states)


class Saver:

    @staticmethod
    def save(path: str, states: States, params: Parameters):
        if not os.path.exists(path):
            os.makedirs(path)

        states = np.array(states.get_states())
        np.save(f"{path}/states.npy", states)
        with open(f"{path}/params.txt", 'w') as f:
            f.write(str(params))

    @staticmethod
    def load(path: str) -> Tuple[States, Parameters]:
        state_np = np.load(f"{path}/states.npy")
        states = States()
        for i in range(state_np.shape[0]):
            states.add_state(state_np[i])

        with open(f"{path}/params.txt", 'r') as f:
            params: Parameters = eval(f.read())

        return states, params
