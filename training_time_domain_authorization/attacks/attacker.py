""" 
Abstract base class for attacks
contains the following methods:
- init - takes in the model and the attack data
- attack - runs the attack and spits out attacked model and attack results
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


from training_time_domain_authorization.datasets.datasets import BenchmarkDataset


class Attacker(ABC):

    def __init__(self, model: Any, attack_dataset: BenchmarkDataset):
        self.model = model
        self.attack_dataset = attack_dataset

    @abstractmethod
    def attack(self) -> Tuple[Any, Dict[str, Any]]:
        pass
