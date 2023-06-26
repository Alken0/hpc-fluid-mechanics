from dataclasses import dataclass, field, InitVar
from datetime import datetime


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
