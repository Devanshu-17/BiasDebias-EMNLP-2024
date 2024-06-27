import os, sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser
from trl import DPOTrainer, DPOConfig, ModelConfig, get_peft_config

from codecarbon import EmissionsTracker

# Add scripts to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from utils import load_custom_dataset, DPOScriptArgument

def main():
    parser = HfArgumentParser((DPOScriptArgument, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    # Load FLAN-T5 model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,)
    peft_config = get_peft_config(model_config)

    if peft_config is None:
        model_ref = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,)
    else:
        model_ref = None

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load dataset
    train_dataset = load_custom_dataset(args.file_path, sanity_check=args.sanity_check)
    # DPO Trainer
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        loss_type=training_args.loss_type,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
    )
    tracker = EmissionsTracker()
    tracker.start()
    trainer.train()
    tracker.stop()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()