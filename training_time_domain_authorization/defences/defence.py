""" 
Abstract base class for defences
contains the following methods:
- init - takes in the model and the defence data
- defend - runs the defence and spits out defended model and defence results
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from training_time_domain_authorization.datasets.datasets import BenchmarkDataset


class Defence(ABC):

    def __init__(self, model: Any, defence_dataset: BenchmarkDataset):
        self.model = model
        self.defence_dataset = defence_dataset

    @abstractmethod
    def defend(self) -> Tuple[Any, Dict[str, Any]]:
        pass