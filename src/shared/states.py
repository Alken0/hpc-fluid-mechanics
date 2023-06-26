import os
from typing import List

import numpy as np

from src.shared.params import Parameters


class States:
    def __init__(self, parameters: Parameters = None):
        self._states = []
        self.parameters = parameters

    def add_state(self, state: np.array):
        self._states.append(state.copy())

    def get_states(self) -> List[np.array]:
        return self._states

    def get_num_states(self) -> int:
        return len(self._states)

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        states = np.array(self._states)
        np.save(f"{path}/states.npy", states)
        with open(f"{path}/params.txt", 'w') as f:
            f.write(str(self.parameters))

    def load(self, path: str):
        states = np.load(f"{path}/states.npy")
        self._states = [states[i] for i in range(states.shape[0])]
        with open(f"{path}/params.txt", 'r') as f:
            self.parameters = eval(f.read())
