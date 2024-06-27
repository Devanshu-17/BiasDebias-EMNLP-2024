import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, T5ForConditionalGeneration, TFBertForSequenceClassification, AutoModelForMaskedLM
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOTrainer, PPOConfig
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from trl.core import LengthSampler

from codecarbon import EmissionsTracker

def main():
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model.")
    parser.add_argument('--model_path', type=str, default="PATH_TO_TRAINED_LLM", help="Path to the trained LLM")
    parser.add_argument('--file_path', type=str, default="PATH_TO_TRAINING_CSV", help="Path to the train CSV file.")
    parser.add_argument('--output_dir', type=str, default="PATH_TO_OUTPUT_CSV", help="Path to the output CSV file.")
    parser.add_argument('--max_steps', type=int, default=10, help="PPO Steps")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # Load dataset
    df = pd.read_csv(args.file_path)
    dataset = build_dataset(df, tokenizer)

    # Load reward model
    reward_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    reward_model = AutoModelForMaskedLM.from_pretrained("PATH_TO_TRAINED_MBERT_MODEL", from_tf=True)
    reward_model.config.id2label = {0: 'bias', 1: 'debias'}

    # Load models
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, is_trainable=True)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, is_trainable=True)

    # Define collator
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
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

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(config=config,
                             model=ppo_model,
                             ref_model=ref_model,
                             tokenizer=tokenizer,
                             dataset=dataset,
                             data_collator=collator)

    # Define generation and reward kwargs
    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
    reward_kwargs = {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": 16
    }

    # Define length sampler
    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    tracker = EmissionsTracker()
    tracker.start()

    # Training loop
    max_ppo_steps = args.max_steps
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=max_ppo_steps):
        if step >= max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()
            gen_kwargs["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **gen_kwargs)
            summary_tensors.append(summary.squeeze()[-max_new_tokens:])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = reward_model(query_response_pairs, **reward_kwargs)

        debias_index = 0
        reward_tensors = [torch.tensor(reward[debias_index]["score"]) for reward in rewards]

        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

    tracker.stop()
    ppo_trainer.save_model(args.output_dir)

    # Save the PPO-trained model and tokenizer
    ppo_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

def build_dataset(df, tokenizer):
    def tokenize(sample):
        prompt = f"""
Debias the following biased sentence.

{sample["Original Sentence"]}

Debiased-format:
"""
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    return dataset

if __name__ == "__main__":
    main()