import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from typing import Literal, Dict, Any
from datasets import load_dataset, DatasetDict
import pandas as pd
import json
from datasets import Dataset


class BenchmarkDataset(ABC):
    """BenchmarkDataset is an abstract class
    that defines the interface for all benchmark datasets.
    It must implement the following:
    - a constructor that takes in dataset construction parameters
    - a training and testing pytorch dataloader (validation is optional)
    - an evaluation function that takes in a HF model and returns a score
    - - evaluation function must output a per sample test metrics as well
        as generated text for each test sample
    - - This is because we cannot do statistical significance testing without
        per sample test metrics
    - - This is because we cannot do manual inspection without generated text
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    # setter for train_dataloader
    @train_dataloader.setter
    @abstractmethod
    def train_dataloader(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def test_dataloader(self):
        raise NotImplementedError

    # setter for test_dataloader
    @test_dataloader.setter
    @abstractmethod
    def test_dataloader(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_dataloader(self):
        raise NotImplementedError

    # setter for validation_dataloader
    @validation_dataloader.setter
    @abstractmethod
    def validation_dataloader(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    # setter for name
    @name.setter
    def name(self, value):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model, dataset: Literal["train", "test", "validation"] = "test"):
        """Evaluate the given huggingface CausalLM model on the dataset
        using the dataset specific metrics
        """
        raise NotImplementedError
