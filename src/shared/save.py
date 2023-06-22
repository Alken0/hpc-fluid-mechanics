from typing import List

import numpy as np


class Saver:
    def __init__(self, parameters: dict = {}):
        self._states = []
        self.parameters = parameters

    def add_state(self, state: np.array):
        self._states.append(state.copy())

    def get_states(self) -> List[np.array]:
        return self._states

    def get_num_states(self) -> int:
        return len(self._states)

    def save(self, file: str):
        states = np.array(self._states)
        np.save(file, states)
        with open(f"{file}.params.txt", 'w') as f:
            f.write(str(self.parameters))

    def load(self, file: str):
        states = np.load(f"{file}.npy")
        self._states = [states[i] for i in range(states.shape[0])]
        with open(f"{file}.params.txt", 'r') as f:
            self.parameters = eval(f.read())
