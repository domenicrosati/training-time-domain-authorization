import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import pandas as pd
from tqdm import tqdm
import json
import wandb

class LawModelTrainer:
    def __init__(self, model_name, train_dataset, test_dataset, valid_dataset, output_dir):
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset
        self.output_dir = output_dir

        # Configurations
        self.local_rank = -1
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-4
        self.max_grad_norm = 0.3
        self.weight_decay = 0.001
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.lora_r = 64
        self.max_seq_length = 200
        self.use_4bit = True
        self.use_nested_quant = False
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.num_train_epochs = 2
        self.fp16 = False
        self.bf16 = True
        self.packing = False
        self.gradient_checkpointing = True
        self.optim = "paged_adamw_32bit"
        self.lr_scheduler_type = "constant"
        self.max_steps = -1
        self.warmup_ratio = 0.03
        self.group_by_length = True
        self.save_steps = 100
        self.logging_steps = 10
        self.device_map = {"": 0}
        self.report_to = "wandb"
        self.new_model = 'qwen-law-model'

        # Initialize Wandb
        wandb.init(project='law-stack-exchange')

        # Load the model and tokenizer
        self.model, self.tokenizer, self.peft_config = self.load_model(self.model_name)

        # Template datasets
        self.train_dataset = self.train_dataset.map(self.template_dataset)
        self.test_dataset = self.test_dataset.map(self.template_dataset)
        self.valid_dataset = self.valid_dataset.map(self.template_dataset)

    def load_model(self, model_name):
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device_map,
            quantization_config=bnb_config
        )

        model.config.use_cache = False

        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding="longest", add_eos_token=True, max_length=50, add_special_tokens=True, truncation=True, use_fast=False)
        # tokenizer.pad_token = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        return model, tokenizer, peft_config

    def format_dataset(self, sample):
        instruction = f"<s>[INST] Answer the following law-related question: "
        context = f"Question: {sample['body']}" if len(sample["body"]) > 0 else None
        response = f"Answer: {sample['text_label']}"
        prompt = "".join([i for i in [instruction, context, response] if i is not None])
        return prompt

    def template_dataset(self, sample):
        sample["text"] = self.format_dataset(sample)
        return sample

    def train(self):
        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to=self.report_to,
            save_total_limit=2,
            run_name='law-train-2'
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=self.packing,
        )

        # Custom function to handle non-serializable objects
        def custom_json_default(obj, encoder=None):
            return str(obj)

        # Replace the default JSON encoder with the custom one
        json.JSONEncoder.default = custom_json_default

        # Train the model
        trainer.train()
        trainer.model.save_pretrained(self.new_model)

    def evaluate(self):
        pipe = pipeline(task='text-generation', model=self.model, tokenizer=self.tokenizer, max_length=200)

        for i in tqdm(range(10)):
            question_text = self.test_dataset[i]['text'].split('Answer:')[0][10:]
            prompt = f'[INST] {question_text}[/INST]'
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs, max_new_tokens=150, eos_token_id=[1, 2])

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(generated)

## Example usage
# train_dataset, test_dataset, valid_dataset = load_dataset("jonathanli/law-stack-exchange").values()

# trainer = LawModelTrainer(model_name="Qwen/Qwen2-0.5B", train_dataset=train_dataset, test_dataset=test_dataset, valid_dataset=valid_dataset, output_dir="/content/results-law")

# trainer.train()

## trainer.evaluate()