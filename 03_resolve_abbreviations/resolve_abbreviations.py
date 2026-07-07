import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

max_input = 256  # must match training truncation length
max_new_tokens = 256

def load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return f.read().splitlines()

def save_txt(path: Path, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for prediction in preds:
            f.write(prediction + "\n")
    
def save_tsv(path: Path, inputs, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("input\tprediction\n")
        for input, prediction in zip(inputs, preds):
            f.write(f"{input}\t{prediction}\n")

def expand_lines(model, tokenizer, device, lines, batch_size):
    # keep blank lines in place, only send the rest to the model
    indices = [i for i, line in enumerate(lines) if line.strip() != ""]
    predictions = ["" for _ in lines]

    for start in tqdm(range(0, len(indices), batch_size)):
        batch_indices = indices[start:start + batch_size]
        enc = tokenizer(
            [lines[i] for i in batch_indices],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=4
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for i, prediction in zip(batch_indices, decoded):
            predictions[i] = prediction.strip()

    return predictions

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, tie_word_embeddings=False).to(device)
    model.eval() # set inference mode

    inputs = load_lines(input_path)
    predictions = expand_lines(model, tokenizer, device, inputs, args.batch_size)

    suffix = ".expanded.tsv" if args.tsv else ".expanded.txt"
    output_path = Path(f"{output_path}/{input_path.stem}{suffix}")

    if args.tsv:
        save_tsv(output_path, inputs, predictions)
    else:
        save_txt(output_path, predictions)

    print(f"Wrote {len(predictions)} lines to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand Latin abbreviations line-by-line using a fine-tuned ByT5 model.")
    parser.add_argument("input", help="Path to input file.")
    parser.add_argument("--output", default="output", help="Output directory for TXT or TSV files.")
    parser.add_argument("--model-dir", default="byt5-latin-expander/final", help="Directory with the fine-tuned model.")
    parser.add_argument("--tsv", action="store_true", help="Write a TSV file with input and prediction instead of plain text.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of lines to expand per model call.")
    args = parser.parse_args()
    main(args)
