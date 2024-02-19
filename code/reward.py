# Import necessary modules
import gc
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (TFGPT2Model,
                          TFMBartModel,
                          TFBertForSequenceClassification,
                          TFDistilBertForSequenceClassification,
                          TFXLMRobertaForSequenceClassification,
                          TFMT5ForConditionalGeneration,
                          TFT5ForConditionalGeneration,
                          T5Tokenizer,
                          AutoTokenizer,
                          AutoConfig,
                          TFBertModel)

# Configure Strategy. Assume TPU...if not set default for GPU/CPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()

# Seeds
def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters:
    seed (int): Seed value for random number generators.
    """
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
    BASE_BATCH_SIZE = 12     # TPU v2 or up...
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

# Summary
print(f'Seed: {SEED}')
print(f'Replica Count: {strategy.num_replicas_in_sync}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Learning Rate: {LR}')

def create_dataset(df, max_len, tokenizer, batch_size, shuffle=False):
    """
    Create a TensorFlow dataset from the given DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing input data.
    max_len (int): Maximum length of input sequences.
    tokenizer (Tokenizer): Tokenizer for encoding input sequences.
    batch_size (int): Batch size for the dataset.
    shuffle (bool): Whether to shuffle the dataset.

    Returns:
    dataset (tf.data.Dataset): TensorFlow dataset containing input sequences and labels.
    """
    total_samples = df.shape[0]

    input_ids, input_masks = [], []

    labels = []

    # Tokenize
    for index, row in tqdm(zip(range(0, total_samples), df.iterrows()), total=total_samples):

        # Get title and description as strings
        text = row[1]['Debiased Sentence']
        partisan = row[1]['Scoring']

        # Encode
        input_encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length'
        )
        input_ids.append(input_encoded['input_ids'])
        input_masks.append(input_encoded['attention_mask'])
        labels.append(
            0 if partisan == 0 else
            1 if partisan == 1 else None)

    # Prepare and Create TF Dataset.
    all_input_ids = tf.Variable(input_ids)
    all_input_masks = tf.Variable(input_masks)
    all_labels = tf.Variable(labels)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input_ids': all_input_ids,
                'attention_mask': all_input_masks
            },
            all_labels
        )
    )

    if shuffle:
        dataset = dataset.shuffle(64, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def ModelCheckpoint(model_name):
    """
    Callback to save the model's weights during training.

    Parameters:
    model_name (str): Name of the model file to save.

    Returns:
    tf.keras.callbacks.ModelCheckpoint: Model checkpoint callback.
    """
    return tf.keras.callbacks.ModelCheckpoint(model_name,
                                              monitor = 'val_accuracy',
                                              verbose = 1,
                                              save_best_only = True,
                                              save_weights_only = True,
                                              mode = 'max',
                                              period = 1)

def create_mbert_model(model_type, strategy, config, lr):
    # Create 'Standard' Classification Model
    with strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained(model_type, config = config)

        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])

        return model

# Load 'bias.csv' dataset
train_df = pd.read_csv('FIXED_SEMIGOLD_BIAS_SHUFFLED_TRAIN_1205.csv')
test_df = pd.read_csv('FIXED_SEMIGOLD_BIAS_TEST_302.csv')

# Multi-Lingual BERT Constants
EPOCHS = 20
model_type = 'bert-base-multilingual-cased'

# Set Config
config = AutoConfig.from_pretrained(model_type, num_labels = 2)
# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space = False, do_lower_case = False)

# Cleanup
tf.keras.backend.clear_session()
if tpu is not None:
    tf.tpu.experimental.initialize_tpu_system(tpu)
gc.collect()

# Create Train and Validation Datasets
train_dataset = create_dataset(train_df, MAX_LEN, tokenizer, BATCH_SIZE, shuffle = True)
validation_dataset = create_dataset(test_df, MAX_LEN, tokenizer, BATCH_SIZE, shuffle = False)

# Steps
train_steps = train_df.shape[0] // BATCH_SIZE
val_steps = test_df.shape[0] // BATCH_SIZE
print(f'Train Steps: {train_steps}')
print(f'Val Steps: {val_steps}')

# Create Model
model_BERT = create_mbert_model(model_type, strategy, config, LR)

# Fit Model
history = model_BERT.fit(train_dataset,
                    steps_per_epoch = train_steps,
                    validation_data = validation_dataset,
                    validation_steps = val_steps,
                    epochs = EPOCHS,
                    verbose = VERBOSE)

model_BERT.save_pretrained("/bert_model")

# Validation Performance
print(f'\n===== MBart Classification Accuracy: {np.max(history.history["val_accuracy"])*100:.3f}%')