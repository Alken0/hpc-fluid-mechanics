from dataclasses import dataclass, field, InitVar
from datetime import datetime


@dataclass
class Parameters:
    folder: InitVar[str]
    file: str = field(init=False)
    x_dim: int = 100
    y_dim: int = 100
    omega: float = 1.0
    epsilon: float = 0.5
    iterations: int = 200

    def __post_init__(self, folder: str):
        time = datetime.now()
        time_stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        self.file = f"{folder}/{time.isoformat()}"


def _generate_file_name():
    return f"data/shear-wave-decay"


x = Parameters(folder="asdf")
print(x.file)
