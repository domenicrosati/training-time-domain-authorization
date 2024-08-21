from datasets import BenchmarkDataset
from sklearn.calibration import calibration_curve
import numpy as np

DEFAULT_BATCH_SIZE = 16
CONTEXT_LENGTH = 512

# dummy code
def evaluate_law_stack_exchange(model, eval_dataloader, tokenizer):
    all_preds = []
    all_labels = []

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            preds = torch.softmax(logits, dim=-1)
            all_preds.append(preds)
            all_labels.append(batch['labels'])

    # Concatenate predictions and labels across batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(all_labels.cpu().numpy(), all_preds.cpu().numpy()[:, 1], n_bins=10)
    
    # Calculate Expected Calibration Error (ECE)
    ece = np.sum(np.abs(prob_pred - prob_true) * np.histogram(all_preds.cpu().numpy()[:, 1], bins=10)[0]) / len(all_preds)
    
    return {"ece": ece}


class LawStackExchangeDataset(BenchmarkDataset):
    def __init__(self, tokenizer, test_batch_size=DEFAULT_BATCH_SIZE, train_batch_size=DEFAULT_BATCH_SIZE):
        self.dataset_name = "law_stack_exchange"
        self.tokenizer = tokenizer
        self._train_dataloader, self._test_dataloader = self.construct_dataloaders(test_batch_size, train_batch_size)
        self._validation_dataloader = None  # Optional

    def construct_dataloaders(self, test_batch_size, train_batch_size):
        train_ds = load_dataset("jonathanli/law-stack-exchange", split="train", trust_remote_code=True)
        test_ds = load_dataset("jonathanli/law-stack-exchange", split="test", trust_remote_code=True)

        # Tokenization
        tokenized_train = train_ds.map(
            self._benchmark_train_dataset_tokenizer,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        tokenized_train.set_format("torch")
        train_dataloader = DataLoader(tokenized_train, batch_size=train_batch_size, shuffle=True)

        tokenized_test = test_ds.map(
            self._test_dataset_tokenizer,
            batched=True,
            remove_columns=test_ds.column_names,
        )
        tokenized_test.set_format("torch")
        test_dataloader = DataLoader(tokenized_test, batch_size=test_batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    def _benchmark_train_dataset_tokenizer(self, element):
        targets = element["text_label"]
        inps = element["body"]
        outputs = [f"Meaning Representation: {inp}\nDescription: {target}" for target, inp in zip(targets, inps)]
        return self.tokenizer(outputs, truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, return_tensors="pt")

    def _test_dataset_tokenizer(self, element):
        targets = element["text_label"]
        inps = element["body"]
        outputs = [f"Meaning Representation: {inp}\nDescription:" for inp in inps]
        tokenized = self.tokenizer(outputs, truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, return_tensors="pt")
        return {**tokenized, "references": targets}

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, value):
        self._train_dataloader = value

    @property
    def test_dataloader(self):
        return self._test_dataloader

    @test_dataloader.setter
    def test_dataloader(self, value):
        self._test_dataloader = value

    @property
    def validation_dataloader(self):
        return self._validation_dataloader

    @validation_dataloader.setter
    def validation_dataloader(self, value):
        self._validation_dataloader = value

    @property
    def name(self):
        return self.dataset_name

    @name.setter
    def name(self, value):
        self.dataset_name = value

    def evaluate(self, model, dataset_split: Literal["train", "test"] = "test"):
        if dataset_split == "train":
            return evaluate_law_stack_exchange(model, self.train_dataloader, self.tokenizer)
        return evaluate_law_stack_exchange(model, self.test_dataloader, self.tokenizer)