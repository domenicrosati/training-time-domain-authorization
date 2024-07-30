import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from typing import Literal, Dict, Any
from datasets import load_dataset, DatasetDict
import pandas as pd
import json
from datasets import Dataset
import gem_metrics

dataset_structure = {
  "law_stack_exchange": {
      "target_ds_label": "text_label",
      "target_prompt_label": "Description:",
      "input_ds_label": "body",
      "input_prompt_label": "Meaning Representation:",
  },
  "pubmedqa": {
      "target_ds_label": "long_answer",
      "target_prompt_label": "Description:",
      "input_ds_label": "question",
      "input_prompt_label": "Meaning Representation:",
  },
  "finqa": {
      "target_ds_label": "answer",
      "target_prompt_label": "Description:",
      "input_ds_label": "question",
      "input_prompt_label": "Meaning Representation:",
  },
}

DEFAULT_BATCH_SIZE = 12
CONTEXT_LENGTH = 512


def construct_benchmark_dataset(
    dataset: Literal["law_stack_exchange, pubmedqa", "finqa"],
    tokenizer,
    file_path=None,
    test_batch_size: int = 8,
    train_batch_size: int = 8
):
    if dataset == "law_stack_exchange":
        train_ds = load_dataset("jonathanli/law-stack-exchange", split="train", trust_remote_code=True)
        test_ds = load_dataset("jonathanli/law-stack-exchange", split="test", trust_remote_code=True)
    elif dataset == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", 'pqa_artificial', split="train", trust_remote_code=True)
        train_test_split = ds['train'].train_test_split(test_size=0.2)

        dataset_split = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })

        train_ds = dataset_split['train']
        test_ds = dataset_split['test']
    elif dataset == "finqa":
        with open(file_path + 'train.json', 'r') as f:
            data = json.load(f)

        # Convert the JSON data to a pandas DataFrame
        df = pd.DataFrame(data)
        qa_df = pd.json_normalize(df['qa'])[['question', 'answer']]
        train_ds = Dataset.from_pandas(qa_df)

        with open(file_path + 'test.json', 'r') as f:
            data = json.load(f)

        # Convert the JSON data to a pandas DataFrame
        df = pd.DataFrame(data)
        qa_df = pd.json_normalize(df['qa'])[['question', 'answer']]
        test_ds = Dataset.from_pandas(qa_df)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


    context_length = CONTEXT_LENGTH

    def _test_dataset_tokenizer(element):
        targets = element[dataset_structure[dataset]["target_ds_label"]]
        inps = element[dataset_structure[dataset]["input_ds_label"]]
        outputs, references = [], []
        for target, inp in zip(targets, inps):
            if isinstance(inp, list):
                inp = " ".join(inp)
            outputs.append(
                f"{dataset_structure[dataset]['input_prompt_label']} {inp}\n{dataset_structure[dataset]['target_prompt_label']}"  # noqa: E501
            )
            references.append(target)
        tokenized = tokenizer(
            outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt",
        )
        return {**tokenized, "references": references}

    def _benchmark_train_dataset_tokenizer(element):
        targets = element[dataset_structure[dataset]["target_ds_label"]]
        inps = element[dataset_structure[dataset]["input_ds_label"]]
        outputs = []
        for target, inp in zip(targets, inps):
            if isinstance(inp, list):
                inp = " ".join(inp)
            outputs.append(
                f"{dataset_structure[dataset]['input_prompt_label']} {inp}\n{dataset_structure[dataset]['target_prompt_label']} {target}"  # noqa: E501
            )
        tokenized = tokenizer(
            outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt",
        )
        return tokenized

    tokenized_train = train_ds.map(
        _benchmark_train_dataset_tokenizer,
        batched=True,
        remove_columns=[
            col
            for col in train_ds.column_names
            if col not in ["input_ids", "attention_mask"]
        ],
    )
    tokenized_train.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_train, batch_size=train_batch_size, shuffle=True
    )

    tokenized_test = test_ds.map(
        _test_dataset_tokenizer,
        batched=True,
        remove_columns=[
            col
            for col in train_ds.column_names
            if col not in ["input_ids", "attention_mask", "references"]
        ],
    )
    tokenized_test.set_format("torch")
    test_dataloader = DataLoader(
        tokenized_test, batch_size=test_batch_size, shuffle=True
    )

    return train_dataloader, test_dataloader


def evaluate_benchmark_dataset(model, eval_dataloader, tokenizer, use_sampler=False):
    results = []
    max_new_tokens = 25
    for batch in tqdm(eval_dataloader):
        params = {
            "max_new_tokens": max_new_tokens,
        }
        if use_sampler:
            params = {
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
            }
        outputs = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **params,
            pad_token_id=tokenizer.eos_token_id,
        )
        for i, output in enumerate(outputs):
            previous_tokens = tokenizer.decode(
                output[: len(batch["input_ids"][i])],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predicted_tokens = tokenizer.decode(
                output[len(batch["input_ids"][i]) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions = gem_metrics.texts.Predictions([predicted_tokens])
            references = gem_metrics.texts.References([batch["references"][i]])
            rouge_results = gem_metrics.compute(
                predictions, references, metrics_list=["rouge"]
            )
            if i == 0 and ARGS.debug:
                logger.debug("Context", previous_tokens.strip())
                logger.debug("Predicted", predicted_tokens.strip())
                logger.debug("Correct", batch["references"][i].strip())
            rouge_results["predicted"] = predicted_tokens.strip()
            results.append(rouge_results)

    return {
        "rouge1": sum([result["rouge1"]["fmeasure"] for result in results])
        / len(results),
        "rouge2": sum([result["rouge2"]["fmeasure"] for result in results])
        / len(results),
        "rougeL": sum([result["rougeL"]["fmeasure"] for result in results])
        / len(results),
        "rouge1_per_sample": [result["rouge1"]["fmeasure"] for result in results],
        "rouge2_per_sample": [result["rouge2"]["fmeasure"] for result in results],
        "rougeL_per_sample": [result["rougeL"]["fmeasure"] for result in results],
        "generated_text": [result["predicted"] for result in results],
    }

class BenchmarkDataset():
    def __init__(self,
        dataset_name: str,
        tokenizer,
        file_path = None,
        test_batch_size=DEFAULT_BATCH_SIZE,
        train_batch_size=DEFAULT_BATCH_SIZE,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        self._train_dataloader, self._test_dataloader = construct_benchmark_dataset(
            dataset_name, tokenizer, file_path, test_batch_size, train_batch_size
        )

    @property
    def train_dataloader(self):
        return self._train_dataloader

    # setter for train_dataloader
    @train_dataloader.setter
    def train_dataloader(self, value):
        self._train_dataloader = value

    @property
    def test_dataloader(self):
        return self._test_dataloader

    # setter for test_dataloader
    @test_dataloader.setter
    def test_dataloader(self, value):
        self._test_dataloader = value

    @property
    def name(self):
        return self.dataset_name

    # setter for name
    @name.setter
    def name(self, value):
        self.dataset_name = value

    def evaluate(self, model, dataset_split: Literal["train", "test"] = "test"):
        if dataset_split == "train":
            return evaluate_benchmark_dataset(model, self.train_dataloader, self.tokenizer)
        return evaluate_benchmark_dataset(model, self.test_dataloader, self.tokenizer)
# from abc import ABC, abstractmethod
# from typing import Literal


# class BenchmarkDataset(ABC):
#     """BenchmarkDataset is an abstract class
#     that defines the interface for all benchmark datasets.
#     It must implement the following:
#     - a constructor that takes in dataset construction parameters
#     - a training and testing pytorch dataloader (validation is optional)
#     - an evaluation function that takes in a HF model and returns a score
#     - - evaluation function must output a per sample test metrics as well
#         as generated text for each test sample
#     - - This is because we cannot do statistical significance testing without
#         per sample test metrics
#     - - This is because we cannot do manual inspection without generated text
#     """

#     @abstractmethod
#     def __init__(self, **kwargs):
#         pass

#     @property
#     @abstractmethod
#     def train_dataloader(self):
#         raise NotImplementedError

#     # setter for train_dataloader
#     @train_dataloader.setter
#     @abstractmethod
#     def train_dataloader(self, value):
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def test_dataloader(self):
#         raise NotImplementedError

#     # setter for test_dataloader
#     @test_dataloader.setter
#     @abstractmethod
#     def test_dataloader(self, value):
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def validation_dataloader(self):
#         raise NotImplementedError

#     # setter for validation_dataloader
#     @validation_dataloader.setter
#     @abstractmethod
#     def validation_dataloader(self, value):
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def name(self):
#         raise NotImplementedError

#     # setter for name
#     @name.setter
#     def name(self, value):
#         raise NotImplementedError

#     @abstractmethod
#     def evaluate(self, model, dataset: Literal["train", "test", "validation"] = "test"):
#         """Evaluate the given huggingface CausalLM model on the dataset
#         using the dataset specific metrics
#         """
#         raise NotImplementedError
