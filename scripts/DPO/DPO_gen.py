import os, sys
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add scripts to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from utils import load_custom_dataset, generate_predictions, generate_predictions_in_batches

def main(model_path, csv_file_path, output_file):
    # Load the trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    gen_config = {
        'do_sample': True,
        'min_length': args.min_length,
        'top_k' : args.top_k,
        'top_p' :args.top_p,
        'max_new_tokens' : args.max_new_tokens,
        'pad_token_id' : tokenizer.pad_token_id,
        'num_beams': 100,   # 100 for Anubis
        'length_penalty': 1.2, # 1.1 for Anubis 1.2 for WIKI
        # 'early_stopping': True,
        # 'temperature': 0.1,
    }

    # Load test dataset
    test_dataset = load_custom_dataset(csv_file_path)

    # Generate predictions
    original_sentences, predictions = generate_predictions_in_batches(model, tokenizer, test_dataset, gen_config)

    # Save predictions to a CSV file
    output_df = pd.DataFrame({
        "Original Sentence": original_sentences,
        "Debiased Sentence": predictions
    })
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model.")
    parser.add_argument('--model_path', type=str, default="./outputs_DPO", help="Path to the trained model.")
    parser.add_argument('--csv_file_path', type=str, default="data/GOLD/Bias_General_Test_Data_302.csv", help="Path to the test CSV file.")
    parser.add_argument('--output_file', type=str, default="RLHF_DPO.csv", help="Path to the output CSV file.")
    parser.add_argument('--min_length', type=int, default=5, help="Generation specific args")
    parser.add_argument('--top_k', type=float, default=0.0, help="Generation specific args")
    parser.add_argument('--top_p', type=float, default=1.0, help="Generation specific args")
    parser.add_argument('--max_new_tokens', type=int, default=120, help="Generation specific args")

    args = parser.parse_args()
    main(args.model_path, args.csv_file_path, args.output_file)
