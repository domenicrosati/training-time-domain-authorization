""" supervised fine-tuning attack """
from typing import Any, Dict, Tuple

from loguru import logger
import torch

from training_time_domain_authorization.attacks.attacker import Attacker
from training_time_domain_authorization.datasets.datasets import BenchmarkDataset
from training_time_domain_authorization.training import train_model


class SFTAttack(Attacker):
    def attack(self) -> Tuple[Any, Dict[str, Any]]:
        logger.info("Beginning supervised fine-tuning attack")
        logger.info("Training model on attack dataset")
        model, eval_datas = train_model(
            self.model,
            self.attack_dataset,
        )
        logger.info("Evaluating model on attack dataset")
        results = self.attack_dataset.evaluate(model)
        return model, {**results, "eval_datas": eval_datas}
