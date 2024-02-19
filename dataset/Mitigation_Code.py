# Import necessary modules
import transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration, EarlyStoppingCallback
import nltk
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

# Load data
data = pd.read_csv("crows_pairs_DnR_train_1045.csv")

# Extract training data and labels
train_data = data["Original Sentence"].to_list()
train_data_label = data["Debiased Sentence"].to_list()

# Encode input and label sequences
input_encoding = tokenizer(train_data,max_length=512,padding=True,truncation=True,return_tensors ="pt")
label_encoding = tokenizer(train_data_label,max_length=512,padding=True,truncation=True,return_tensors ="pt")

# Prepare training dataset
train_dataset = [{"input_ids": input_encoding["input_ids"][i], "attention_mask": input_encoding["attention_mask"][i],
                  "labels": label_encoding["input_ids"][i]} for i in range(len(data))]

# Data collator for sequence-to-sequence training
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=tokenizer.pad_token_id)

# Training arguments
training_args = Seq2SeqTrainingArguments(
        output_dir="IndicBART_Model",
        logging_dir="./logs",
        logging_strategy = 'steps',
        logging_first_step = "True",
        logging_steps = 500,
        overwrite_output_dir=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=128,
        num_train_epochs=40,
        save_steps=6000,
        warmup_steps=150,
        save_total_limit=1,
        report_to="none",
        learning_rate=2e-4,
        bf16=False,
        split_batches=True,
        load_best_model_at_end = True,
        weight_decay=0.01,
        evaluation_strategy= 'steps',
        save_strategy= 'steps',
        fp16=False)

# Initialize Trainer for sequence-to-sequence training
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold = 0.6)]
)

# Train the model
trainer.train()

# Log training history
trainer.state.log_history

# Save trained model
trainer.save_model("IndicBart_Model")

# Example inference
# Tokenize and generate outputs for a sample input
t = tokenizer
m = AutoModelForSeq2SeqLM.from_pretrained("IndicBart_Model")
check = ["A little like Italians with the scent of spaghetti."]

c = t(str(check),truncation=True,return_tensors="pt",return_token_type_ids=False)
o = m.generate(**c,max_length=512)
decode_output = t.batch_decode(o,skip_special_tokens=True)
print(decode_output)
