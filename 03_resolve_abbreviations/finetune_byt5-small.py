import argparse
import os
import unicodedata
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import T5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from Levenshtein import distance
from evaluate import load

exact_match_metric = load("exact_match")

model_name = "google/byt5-small"
max_input = 256
max_target = 256

def nfc(s): 
    return unicodedata.normalize("NFC", str(s))

def prepare_datasets(data_path):
    df = pd.read_csv(data_path)  # columns: id,text,label,Comments
    df = df.dropna(subset=["text","label"]).copy()

    df["text"] = df["text"].map(nfc) # make sure training material is encoded homogeneously
    df["label"] = df["label"].map(nfc)

    # Train/val/test split: 80/10/10
    df = df.sample(frac=1, random_state=13) # shuffle

    n = len(df)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    
    train_df.to_csv(f"{os.path.dirname(data_path)}/train_split.csv", index=False)
    val_df.to_csv(f"{os.path.dirname(data_path)}/val_split.csv", index=False)
    test_df.to_csv(f"{os.path.dirname(data_path)}/test_split.csv", index=False)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })
    return ds

def preprocess(batch):
    model_inputs = tokenizer(
        batch["text"],
        truncation=True,
        max_length = max_input,
    )
    labels = tokenizer(
        text_target = batch["label"],
        truncation = True,
        max_length = max_target,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def cer(preds, refs):
    total_dist,total_len=0,0
    for p,r in zip(preds,refs):
        total_dist += distance(p,r) # levenshtein distance
        total_len  += max(1,len(r))
    return total_dist/total_len

def compute_metrics(eval_pred):
    # requires global tokenizer variable to be set
    
    preds, labels = eval_pred

    # extract generated tokens if tuple
    if isinstance(preds, tuple):
        preds = preds[0]

    # replace negative ids (-100) with pad
    labels = np.array(labels)
    labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)
    preds = np.array(preds)
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)

    # decode
    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # clean
    pred_texts  = [p.strip() for p in pred_texts]
    label_texts = [l.strip() for l in label_texts]

    # metrics
    cer_val = cer(pred_texts, label_texts)
    exact = exact_match_metric.compute(predictions=pred_texts, references=label_texts)
    exact = exact["exact_match"]
    return {"cer": cer_val, "exact_match": exact}


def main(args):
    ds = prepare_datasets(args.data)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name, tie_word_embeddings=False)

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="byt5-latin-expander",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        learning_rate=1e-4,
        warmup_steps=400,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        num_train_epochs=15,
        weight_decay=0.01,
        bf16=True,  # necessary to avoid gradient overflow
        fp16=False,
        logging_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=max_target,
    )

    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,  
            early_stopping_threshold=5e-4     # at least 0.05% CER improvement
        )]
    )

    trainer.train()
    trainer.save_model("byt5-latin-expander/final")
    tokenizer.save_pretrained("byt5-latin-expander/final")

    test_metrics = trainer.predict(test_dataset=tokenized["test"], metric_key_prefix="test")
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="Finetune ByT5-small on Latin abbreviation expansion.")
      parser.add_argument("data", help="CSV file containing training data (columns: 'text' and 'label').")
      args = parser.parse_args()
      main(args)
