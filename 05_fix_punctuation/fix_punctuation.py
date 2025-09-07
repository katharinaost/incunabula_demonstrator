import argparse
import spacy
from pathlib import Path
import string


def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('/','')
    text = text.replace('¶', '')
    text = text.lower()
    text = text.strip()
    return text

def main(args):
    nlp = spacy.load(args.model, disable=['senter'])

    text = args.input.read_text(encoding="utf-8")
    text = remove_punctuation(text)

    doc = nlp(text)

    output = ""
    for sentence in doc.sents:
        output += sentence.text + ".\n"

    args.output.mkdir(parents=True, exist_ok=True)

    out_path = args.output / (args.input.stem + ".sent.txt")
    out_path.write_text(output, encoding="utf-8")
    print(f"Wrote {out_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip all punctuation and run sentence segmentation.")
    parser.add_argument("input", type=Path, help="Path to input file.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory.")
    parser.add_argument("--model", type=str, default="./la_core_web_lg-main/training/trf/model-assembled", help="Path to spaCy model.")
    args = parser.parse_args()
    main(args)
