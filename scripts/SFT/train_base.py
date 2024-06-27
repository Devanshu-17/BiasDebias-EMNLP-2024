import os
import sys
import argparse
from codecarbon import EmissionsTracker
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from huggingface_hub import login
login(token="HUGGINGFACE_TOKEN", add_to_git_credential=True, new_session=False)

# Add scripts to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from utils import get_base_data

def train(args):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    train_dataset, eval_dataset = get_base_data(args.train_path, args.test_path, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id)
    training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch,
            num_train_epochs=args.epoch,
            save_total_limit=2,
            learning_rate=args.lr)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.create_optimizer()
    trainer.create_scheduler(64)

    # Initialise emissions tracker
    tracker = EmissionsTracker()
    tracker.start()
    trainer.train()
    tracker.stop()
    trainer.push_to_hub("OUTPUT_MODEL_PATH")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model.")
    parser.add_argument('--train_path', type=str, default="PATH_TO_TRAIN_CSV", help="Path to the train CSV file.")
    parser.add_argument('--test_path', type=str, default="PATH_TO_TEST_CSV", help="Path to the test CSV file.")
    parser.add_argument('--output_dir', type=str, default="PATH_TO_OUTPUT_DIRECTORY", help="Output Directory")
    parser.add_argument('--batch', type=int, default=8, help="Train Batch Size")
    parser.add_argument('--epoch', type=int, default=500, help="No. of train epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning Rate")

    args = parser.parse_args()
    train(args)