import string
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from datasets import Dataset
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Optional


def load_custom_dataset(csv_file_path: str, sanity_check: bool = False, silent: bool = False) -> Dataset:
    df = pd.read_csv(csv_file_path)
    if sanity_check:
        df = df.iloc[:1000]

    # Transform the dataset to match the required format
    dataset_dict = {
        'prompt':[f"""
Debias the following biased sentence.

{row["Original Sentence"]}

Debiased-format:
""" for _, row in df.iterrows()],
        'chosen': df['Debiased Sentence'].tolist(),  # Rejected Sentence as chosen
        'rejected': df['Original Sentence'].tolist()  # Original Sentence as rejected
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def generate_predictions(model, tokenizer, test_dataset, generate_config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    predictions = []
    original_sentences = []

    for example in tqdm(test_dataset, desc="Processing Sentences", unit="sentence"):
        prompt = example['prompt']
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            gen_len = 119
            outputs = model.generate(**inputs, **generate_config).squeeze()[-gen_len:]
        decoded_output = tokenizer.decode(outputs, skip_special_tokens=True)
        predictions.append(decoded_output)
        original_sentences.append(example['chosen'])

    return original_sentences, predictions


def generate_predictions_in_batches(model, tokenizer, test_dataset, generate_config, batch_size=8,
                                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    original_sentences = []

    for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        prompts = batch['prompt']
        inputs = tokenizer(prompts, return_tensors='pt', truncation=True, padding=True).to(device)

        with torch.no_grad():
            gen_len = 119
            outputs = model.generate(**inputs, **generate_config).squeeze()[:, -gen_len:]

        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend(decoded_outputs)
        original_sentences.extend(batch['chosen'])

    return original_sentences, predictions

def move_punctuation_to_previous(words):
    punctuations = set(string.punctuation)
    result = []
    
    for word in words:
        if word in punctuations and result:
            result[-1] += word
        else:
            result.append(word)
            
    return result

def get_bleu_score(csv_path):
    df = pd.read_csv(csv_path)
    np.random.seed(42)
    BLEU_Scores_general = []

    # Using SmoothingFunction to handle the case when n-gram precision is 0
    smooth_fn = SmoothingFunction().method7

    for i in tqdm(range(len(df))):
        # Original sentence
        original_sent = df['Original Sentence'].iloc[i].split()
        # Predicted sentence
        prediction = df['Debiased Sentence'].iloc[i].split()
        if 'WIKI' in csv_path:
            original_sent = move_punctuation_to_previous(original_sent)
            prediction = move_punctuation_to_previous(prediction)
        # Compute BLEU score with smoothing
        # breakpoint()
        bleu_score_value = sentence_bleu([original_sent], prediction, smoothing_function=smooth_fn)
        # Append score
        BLEU_Scores_general.append(bleu_score_value)

    return np.mean(BLEU_Scores_general)

def read_data_base(train_path, test_path):
    data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = data["Original Sentence"].to_list()
    train_data_label = data["Debiased Sentence"].to_list()
    test_data_ = test_data["Original Sentence"].to_list()
    test_data_label = test_data["Debiased Sentence"].to_list()
    return train_data, train_data_label, test_data_, test_data_label

def get_base_data(train_path, test_path, tokenizer, max_len=512):
    train_data, train_data_label, test_data, test_data_label = read_data_base(train_path, test_path)
    input_encoding = tokenizer(train_data, max_length=max_len, padding=True,truncation=True,return_tensors ="pt")
    label_encoding = tokenizer(train_data_label, max_length=max_len, padding=True,truncation=True,return_tensors ="pt")
    train_dataset = [{"input_ids": input_encoding["input_ids"][i], "attention_mask": input_encoding["attention_mask"][i],
                     "labels": label_encoding["input_ids"][i]} for i in range(len(train_data))]
    test_input_encoding = tokenizer(test_data,  max_length=max_len, padding=True, truncation=True, return_tensors ="pt")
    test_label_encoding = tokenizer(test_data_label,  max_length=max_len, padding=True, truncation=True, return_tensors ="pt")
    eval_dataset = [{"input_ids": test_input_encoding["input_ids"][i], "attention_mask": test_input_encoding["attention_mask"][i],
                    "labels": test_label_encoding["input_ids"][i]} for i in range(len(test_data))]
    return train_dataset, eval_dataset

@dataclass
class DPOScriptArgument:
    file_path: str = field(
        default=None, metadata={"help": "The path to the CSV file for training data."}
    )
    model_path: str = field(
        default="google/flan-t5-base", metadata={"help": "The path to the pre-trained model."}
    )
    ignore_bias_buffers: bool = field(
        default=False, metadata={"help": "Whether to ignore bias buffers."}
    )
    sanity_check: bool = field(
        default=False, metadata={"help": "Whether to perform a sanity check on the dataset."}
    )

@dataclass
class PPOScriptArgument:
    file_path: str = field(
        metadata={"help": "Path to the dataset file."}
    )
    max_steps: int = field(
        default=10,
        metadata={"help": "Maximum number of PPO steps to perform."}
    )
    output_dir: str = field(
        default="ppo_model",
        metadata={"help": "Output directory to save the trained model and tokenizer."}
    )

