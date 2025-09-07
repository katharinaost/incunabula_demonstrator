import argparse
from pathlib import Path
from conllu import parse_incr
from conllu import TokenList
import string


def lowercase_and_strip_punct(infile, outfile):
    for tokenlist in parse_incr(infile):
        id_map = {} # old -> new
        i = 1
        # Keep only tokens that are not PUNCT
        new_list = TokenList()
        new_list.metadata = tokenlist.metadata
        new_list.metadata["text"] = new_list.metadata["text"].translate(str.maketrans('', '', string.punctuation))
        new_list.metadata["text"] = new_list.metadata["text"].lower()
        for token in tokenlist:
            if token.get("upostag") == "PUNCT":
                continue
            if "form" in token and token["form"] is not None:
                token["form"] = token["form"].lower()
            new_list.append(token)
            id_map[token.get("id")] = i
            i += 1

        # If the first token was punctuation, ids don't get updated properly
        u=1 

        for token in new_list:
            if token["head"] != 0:
                token["head"] = id_map[token["head"]]
            token["id"] = u
            u += 1

        # Write back to file
        outfile.write(new_list.serialize())

def process_files(input_files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for in_path in input_files:
        in_path = Path(in_path)
        out_path = output_dir / in_path.name

        with open(in_path, "r", encoding="utf-8") as in_fh, \
             open(out_path, "w", encoding="utf-8") as out_fh:
            lowercase_and_strip_punct(in_fh, out_fh)

        print(f"Processed {in_path} > {out_path}")


def main(args):
    process_files(args.inputs, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lowercase FORM and remove PUNCT rows from multiple CoNLL-U files.")
    parser.add_argument("inputs", nargs="+", help="One or more input .conllu files")
    parser.add_argument("output_dir", help="Directory to write processed files into")
    args = parser.parse_args()
    main(args)