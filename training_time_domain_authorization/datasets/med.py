from datasets import BenchmarkDataset
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_BATCH_SIZE = 16
CONTEXT_LENGTH = 512

# dummy code
def evaluate_pubmedqa(model, eval_dataloader, tokenizer):
    all_preds = []
    all_labels = []

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['labels'].cpu().numpy())

    # Concatenate predictions and labels across batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate Accuracy and Macro-F1
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {"accuracy": accuracy, "macro_f1": macro_f1}


class PubMedQADataset(BenchmarkDataset):
    def __init__(self, tokenizer, test_batch_size=DEFAULT_BATCH_SIZE, train_batch_size=DEFAULT_BATCH_SIZE):
        self.dataset_name = "pubmedqa"
        self.tokenizer = tokenizer
        self._train_dataloader, self._test_dataloader = self.construct_dataloaders(test_batch_size, train_batch_size)
        self._validation_dataloader = None  # Optional

    def construct_dataloaders(self, test_batch_size, train_batch_size):
        # Load the dataset as a DatasetDict
        ds = load_dataset("qiaojin/PubMedQA", 'pqa_artificial', trust_remote_code=True)

        # Apply train_test_split on the 'train' split
        train_test_split = ds['train'].train_test_split(test_size=0.2)
        train_ds, test_ds = train_test_split['train'], train_test_split['test']

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
        targets = element["long_answer"]
        inps = element["question"]
        outputs = [f"Meaning Representation: {inp}\nDescription: {target}" for target, inp in zip(targets, inps)]
        return self.tokenizer(outputs, truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, return_tensors="pt")

    def _test_dataset_tokenizer(self, element):
        targets = element["long_answer"]
        inps = element["question"]
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
            return evaluate_pubmedqa(model, self.train_dataloader, self.tokenizer)
        return evaluate_pubmedqa(model, self.test_dataloader, self.tokenizer)