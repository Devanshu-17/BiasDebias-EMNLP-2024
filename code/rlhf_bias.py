# Import necessary libraries
import gc
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple
import torch
torch.cuda.empty_cache()  # Empty CUDA cache to free up memory
import evaluate
import nltk
from datasets import load_dataset, Dataset, DatasetDict
import sentencepiece
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()
from transformers import (TFGPT2Model,
                          pipeline,
                          AutoModelForSeq2SeqLM,
                          TFMBartModel,
                          AutoModelForSequenceClassification,
                          TFBertForSequenceClassification,
                          TFDistilBertForSequenceClassification,
                          TFXLMRobertaForSequenceClassification,
                          TFMT5ForConditionalGeneration,
                          TFT5ForConditionalGeneration,
                          T5Tokenizer,
                          GenerationConfig,
                          AutoTokenizer,
                          EarlyStoppingCallback,
                          AutoConfig,
                         TFBertModel)
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

# Configure Strategy. Assume TPU...if not set default for GPU/CPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()

# Seeds for reproducibility
def set_seeds(seed: int)->None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Generic Constants
MAX_LEN = 512
TEST_SIZE = 0.2
LR = 0.00002
VERBOSE = 1
SEED = 1000
set_seeds(SEED)

# Set Autotune
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Set Batch Size
BASE_BATCH_SIZE = 8         # Modify to match your GPU card.
if tpu is not None:
    BASE_BATCH_SIZE = 32     # TPU v2 or up...
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

# Display summary of configurations
print(f'Seed: {SEED}')
print(f'Replica Count: {strategy.num_replicas_in_sync}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Learning Rate: {LR}')

# Tokenizer and model for reward calculation
reward_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
reward_model = AutoModelForSequenceClassification.from_pretrained("Trained_Reward_Model_Path",from_tf=True)

# Tokenizer and model for LLM (Large Language Model)
trained_llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
trained_llm = AutoModelForSeq2SeqLM.from_pretrained("Trained_Base_Model_Path")

# Example sentence for testing LLM
check = ["African-American children are often abandoned by their parents in the park."]
c = trained_llm_tokenizer(check,truncation=True,return_tensors="pt")
o = trained_llm.generate(**c,max_length=1024)
decode_output = trained_llm_tokenizer.batch_decode(o,skip_special_tokens=True)
print(decode_output)

# Function to print number of trainable model parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

# Create PPO model for reward calculation
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(trained_llm,
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True)

print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
print(ppo_model.v_head)

# Create reference model
ref_model = create_reference_model(ppo_model)
print(f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n')

# Update id2label mapping for the reward model
reward_model.config.id2label = {0: 'bias', 1: 'debias'}
# Print the updated mapping to confirm the change
print(reward_model.config.id2label)

# Define a collator function for dataset
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# Load and preprocess training dataset
dataset = pd.read_csv("Train_Data_Path")
dataset.rename(columns={'Scoring': 'Reward'}, inplace=True)

df_train_filtered = dataset[['Original Sentence', 'Debiased Sentence']]
dataset_train = Dataset.from_pandas(df_train_filtered)

# Build dataset for training
def build_dataset(model_name, dataset_name):
    """
    Preprocess the dataset and split it into train and test parts.
    Parameters:
    - model_name (str): Tokenizer model name.
    - dataset_name (str): Name of the dataset to load.
    Returns:
    - dataset: Preprocessed dataset containing input_id and query.
    """
    dataset = dataset_train
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    def tokenize(sample):
        prompt = f"""
Debias the following biased sentence.

{sample["Original Sentence"]}

Debiased-format:
"""
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    return dataset

model_name="google/flan-t5-base"
dataset = build_dataset(model_name=model_name, dataset_name=dataset_train)

# Print dataset columns
bias_dataset = dataset
for column in bias_dataset.features:
    print(f"Data in column '{column}':")
    for i in range(min(1, len(bias_dataset))):
        print(bias_dataset[i][column])
    print("\n")

# Define PPO configuration parameters
learning_rate=1.41e-5
max_ppo_epochs=1
mini_batch_size=4
batch_size=16

config = PPOConfig(
    model_name=None,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=trained_llm_tokenizer,
    dataset=bias_dataset,
    data_collator=collator
)

# Example biased and non-biased texts
non_bias_text = "Some people can't drive"
bias_text = "Asians Can't drive as they have slanted eyes"

# Determine device for processing
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

device = 0 if torch.cuda.is_available() else "cpu"

# Pipeline for sentiment analysis using the reward model
bias_pipe = pipeline(
    "sentiment-analysis",
    tokenizer = reward_tokenizer,
    model=reward_model,
    device=device
)
reward_logits_kwargs = {
    "top_k": None, 
    "function_to_apply": "none", 
    "batch_size": 16
}

reward_probabilities_kwargs = {
    "top_k": None, 
    "function_to_apply": "softmax", 
    "batch_size": 16
}

# Display reward model outputs for biased and non-biased texts
print("Reward model output:")
print("For non_bias_text")
print(bias_pipe(non_bias_text, **reward_logits_kwargs))
print(bias_pipe(non_bias_text, **reward_probabilities_kwargs))
print("For bias text")
print(bias_pipe(bias_text, **reward_logits_kwargs))
print(bias_pipe(bias_text, **reward_probabilities_kwargs))

# Generation and reward parameters
tokenizer = trained_llm_tokenizer
gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None, 
    "function_to_apply": "none", 
    "batch_size": 16
}

max_ppo_steps = 1

# Execute PPO training steps
for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]

    summary_tensors = []

    for prompt_tensor in prompt_tensors:
        max_new_tokens = 800

        generation_kwargs["max_new_tokens"] = max_new_tokens
        summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

        summary_tensors.append(summary.squeeze()[-max_new_tokens:])

    batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

    query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = bias_pipe(query_response_pairs, **reward_kwargs)

    debias_index = 0
    reward_tensors = [torch.tensor(reward[debias_index]["score"]) for reward in rewards]

    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

    print(f'kl div loss: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print('-'.join('' for x in range(100)))

# Define generation and reward parameters
generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None, 
    "function_to_apply": "none", 
    "batch_size": 16
}

# Load PPO model
ppo_model = AutoModelForSeq2SeqLM.from_pretrained("PPO_Model_Path")

# Load and preprocess test dataset
dataset_test = pd.read_csv("Test_Data_Path")
df_test_filtered = dataset_test[['Original Sentence', 'Debiased Sentence']]
dataset_test = Dataset.from_pandas(df_test_filtered)

# Build test dataset
def build_dataset(model_name, dataset_name):
    dataset = dataset_test
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    def tokenize(sample):
        prompt = f"""
{sample["Original Sentence"]}

Debiased-format:
"""
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    return dataset

model_name="google/flan-t5-base"
dataset_test = build_dataset(model_name=model_name, dataset_name='Test_Data_Path')

bias_dataset_test = dataset_test
for column in bias_dataset_test.features:
    print(f"Data in column '{column}':")
    for i in range(min(1, len(bias_dataset_test))):
        print(bias_dataset_test[i][column])
    print("\n")

debias_index = 0

batch_size = 229
compare_results = {}

df_batch = bias_dataset_test[0:batch_size]

compare_results["query"] = df_batch["query"]
prompt_tensors = df_batch["input_ids"]

summary_tensors_ref = []
summary_tensors = []

# Generate responses from reference and PPO models
for i in tqdm(range(batch_size)):
    gen_len = 119
    generation_kwargs["max_new_tokens"] = gen_len

    summary = ref_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors_ref.append(summary)

    summary = ppo_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors.append(summary)

# Decode responses
compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

# Calculate rewards for query/response pairs before and after debiasing
texts_before = [d + s for d, s in zip(compare_results["query"], compare_results["response_before"])]
rewards_before = bias_pipe(texts_before, **reward_kwargs)
compare_results["reward_before"] = [reward[debias_index]["score"] for reward in rewards_before]

texts_after = [d + s for d, s in zip(compare_results["query"], compare_results["response_after"])]
rewards_after = bias_pipe(texts_after, **reward_kwargs)
compare_results["reward_after"] = [reward[debias_index]["score"] for reward in rewards_after]

# Display comparison results
pd.set_option('display.max_colwidth', 800)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
df_compare_results

# Save comparison results to CSV file
df_compare_results.to_csv("Final_RLHF_Result_Path")
