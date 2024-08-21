from datasets import BenchmarkDataset
DEFAULT_BATCH_SIZE = 16
CONTEXT_LENGTH = 512

# dummy code
def evaluate_finqa(model, eval_dataloader, tokenizer):
    correct = 0
    total = 0

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Custom logic for FinQA accuracy evaluation, e.g., comparing predictions with ground truth answers
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    accuracy = correct / total
    
    return {"accuracy": accuracy}

class FinQADataset(BenchmarkDataset):
    def __init__(self, tokenizer, file_path, test_batch_size=DEFAULT_BATCH_SIZE, train_batch_size=DEFAULT_BATCH_SIZE):
        self.dataset_name = "finqa"
        self.tokenizer = tokenizer
        self.file_path = file_path
        self._train_dataloader, self._test_dataloader = self.construct_dataloaders(test_batch_size, train_batch_size)
        self._validation_dataloader = None  # Optional

    def construct_dataloaders(self, test_batch_size, train_batch_size):

        # Load JSON data from files and process into pandas DataFrames
        with open(self.file_path + 'train.json', 'r') as f:
            train_data = json.load(f)
        # Convert the JSON data to a pandas DataFrame
        train_df_ = pd.DataFrame(train_data)
        train_df = pd.json_normalize(train_df_['qa'])[['question', 'answer']]
        train_ds = Dataset.from_pandas(train_df)

        with open(self.file_path + 'test.json', 'r') as f:
            test_data = json.load(f)
        # Convert the JSON data to a pandas DataFrame
        test_df_ = pd.DataFrame(test_data)
        test_df = pd.json_normalize(test_df_['qa'])[['question', 'answer']]
        test_ds = Dataset.from_pandas(test_df)

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
        targets = element["answer"]
        inps = element["question"]
        outputs = [f"Meaning Representation: {inp}\nDescription: {target}" for target, inp in zip(targets, inps)]
        return self.tokenizer(outputs, truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, return_tensors="pt")

    def _test_dataset_tokenizer(self, element):
        targets = element["answer"]
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
            return evaluate_finqa(model, self.train_dataloader, self.tokenizer)
        return evaluate_finqa(model, self.test_dataloader, self.tokenizer)
