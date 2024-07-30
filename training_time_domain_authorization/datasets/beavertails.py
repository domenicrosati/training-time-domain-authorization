import gc
import random
from typing import Literal
import torch
import json
import os

from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import re
import numpy as np


from training_time_domain_authorization.datasets.datasets import BenchmarkDataset
from training_time_domain_authorization.device import accelerator


DEVICE = accelerator.device

DECODING_TRUST_INSTRUCTIONS = "User: You are required to continue the generation of the given incomplete prompt\n"


SMOKE_TEST = False
CONTEXT_LENGTH = 256
DEFAULT_BATCH_SIZE = 4

BT_CATEGORIES = {
    "half": [
        "violence,aiding_and_abetting,incitement",
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation"
    ],
    "other_half": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "terrorism,organized_crime",
        "sexually_explicit,adult_content"
    ],
    "violence": [
        "violence,aiding_and_abetting,incitement"
    ],
    "toxic": [
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    ],
    "privacy_misinformation": [
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation"
    ],
    "crime_unethical": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "non_violent_unethical_behavior",
        "terrorism,organized_crime"
    ],
    "sexually_explicit": [
        "sexually_explicit,adult_content"
    ],
    "self_harm": [
        "self_harm"
    ]
}


def construct_beavertails_dataset(
    tokenizer, train_harmful_subset="all", test_harmful_subset="all",
    withhold_harmful_subset="",
    test_batch_size=DEFAULT_BATCH_SIZE,
    train_batch_size=DEFAULT_BATCH_SIZE, strong_attack=False,
    refusal=False,
    attack=False,
    attack_size=100,
    defence_size=1000,
    context_length=CONTEXT_LENGTH
):
    trainds = None
    # if strong_attack:
    #     trainds = load_dataset("PKU-Alignment/BeaverTails", data_dir='../datasets', split="330k_train")
    # else:
    #     trainds = load_dataset("PKU-Alignment/BeaverTails", data_dir='../datasets', split="30k_train")

    # if refusal:
    refusal_ds = pd.read_json(path_or_buf='./data/beavertails_with_refusals_train.json')
    refusal_df = Dataset.from_pandas(refusal_ds)
    trainds = refusal_df

    testds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    # plan get the train and test datasets
    # prepare tokenizer where we have the subsets
    def _train_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if train_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[train_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                if category not in harmful_categories and train_harmful_subset != "all":
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:\n{content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
                output_categories.append(category)
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **harmful_outputs,
            "categories": output_categories,
        }
    def _refusal_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if train_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[train_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                if category not in harmful_categories and train_harmful_subset != "all":
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer: {content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
                output_categories.append(category)
        refusal_outputs = tokenizer(
            refusal_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **refusal_outputs,
            "categories": output_categories,
        }

    def _test_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if test_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[test_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        for prompt, content, category, is_safe in zip(prompts, contents, processed_categories, is_safes):
            if is_safe:
                continue
            else:
                if category not in harmful_categories:
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:"
                )
                output_categories.append(category)
        if len(harmful_outputs) == 0:
            return {}
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **harmful_outputs,
            "categories": output_categories
        }

    tokenized_train = trainds.map(
        _train_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            trainds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    tokenized_train.set_format("torch")

    harmless_train = trainds.map(
        _refusal_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            trainds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    harmless_train.set_format("torch")

    tokenizer.padding_side = "left"

    tokenized_test = testds.map(
        _test_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            testds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    tokenized_test.set_format("torch")

    tokenizer.padding_side = "right"

    # only select 100 samples to test on
    if len(tokenized_test) > 100:
        tokenized_test = tokenized_test.select(range(100))
    if attack:
        if attack_size < len(tokenized_train):
            tokenized_train = tokenized_train.select(range(attack_size))
            harmless_train = harmless_train.select(range(attack_size))
    else:
        if defence_size < len(tokenized_train):
            tokenized_train = tokenized_train.select(range(defence_size))
            harmless_train = harmless_train.select(range(defence_size))

    train_dataloader = DataLoader(tokenized_train, batch_size=train_batch_size)
    harmless_train_dataloader = DataLoader(harmless_train, batch_size=train_batch_size)
    test_dataloader = DataLoader(tokenized_test, batch_size=train_batch_size)
    return train_dataloader, harmless_train_dataloader, test_dataloader