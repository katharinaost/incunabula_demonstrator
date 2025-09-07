import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import (
    PeftModel
)

base_model = "google/byt5-base"
lora_adapter = "mschonhardt/byt5-base-bdd-expansion-lora-v4-l40s"
prompt_prefix = "expand abbreviations: " 
max_input = 1024      
max_target = 1024

def main(args):
    dataset_id = args.dataset
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = PeftModel.from_pretrained(base_model, lora_adapter, is_trainable=True)
    model.print_trainable_parameters()

    dataset = load_dataset(dataset_id)

    if "test" not in dataset:
        logging.info("Creating train/val splits...")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    def preprocess_function(examples):
        inputs = [prompt_prefix + text for text in examples["source_text"]]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logging.info("\nTokenizing the dataset...")
    tokenized = dataset.map(preprocess_function, batched=True)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir="byt5-bdd-lora-finetuned",
        learning_rate=3e-5,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        logging_steps=50,
        load_best_model_at_end=True,
        generation_max_length=max_target,
        bf16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("byt5-bdd-lora-finetuned")  # saves only the LoRA weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the mschonhardt/byt5-base-bdd-expansion-lora-v4-l40s adapter on new data.")
    parser.add_argument("dataset", default="katharinaost/abbreviationes-test", help="Huggingface dataset ID")
    args = parser.parse_args()
    main(args)
