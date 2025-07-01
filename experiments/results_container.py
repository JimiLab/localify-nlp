from dataclasses import dataclass
import numpy as np


@dataclass
class ResultsContainer:
    mean: np.ndarray
    sd: np.ndarray
    min: float
    max: float

    def to_file(self, filepath, experiment_name) -> None:
        formattable = """{name}

Mean: {mean}
SD: {sd}
Min: {min}
Max: {max}
"""

        with open(filepath, 'w') as file:
            file.write(formattable.format(
                name=experiment_name,
                mean=self.mean,
                sd=self.sd,
                min=self.min,
                max=self.max
            ))

