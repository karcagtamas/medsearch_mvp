import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate


# -------------------------
# Utils: read CoNLL files
# -------------------------

def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Reads a CoNLL file -> lists of tokens and tags (sentence level)."""
    tokens, tags = [], []
    sent_tokens, sent_tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if sent_tokens:
                    tokens.append(sent_tokens)
                    tags.append(sent_tags)
                    sent_tokens, sent_tags = [], []
                continue

            if "\t" in line:
                tok, tag = line.split("\t")
            else:
                parts = line.split(" ")
                if len(parts) == 1:
                    tok, tag = parts[0], "0"
                else:
                    tok, tag = parts[0], parts[-1]

            sent_tokens.append(tok)
            sent_tags.append(tag)

    if sent_tokens:
        tokens.append(sent_tokens)
        tags.append(sent_tags)
    return tokens, tags


def build_label_list(train_tags: List[List[str]]) -> List[str]:
    labels = sorted(set(tag for seq in train_tags for tag in seq))
    # Make sure "O" is index 0 for convenience but not mandatory
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels
    return labels


def make_hf_dataset(tokens: List[List[str]], tags: List[List[str]], label_list: List[str]) -> Dataset:
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list))
    })
    # map string tags to indices per ClassLabel later; here keep raw, then cast
    ds = Dataset.from_dict({"tokens": tokens, "ner_tags": tags})
    return ds.cast(features)


# -------------------------
# Tokenization & alignment
# -------------------------

def align_labels_with_tokens(
        labels: List[int],
        word_ids: List[int],
        label_all_tokens: bool = False,
        ignore_index: int = -100,
) -> List[int]:
    """
Align word-level labels to subword tokens.
For subwords, either repeat IOB label (if label_all_tokens=True) or set ignore_index.
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(ignore_index)
        elif word_id != current_word:
            new_labels.append(labels[word_id])
            current_word = word_id
        else:
            # subword
            if label_all_tokens:
                label = labels[word_id]
                # convert B-XXX to I-XXX for continuation subwords
                if label != 0:  # 0 is "O" under our convention from ClassLabel
                    # We need access to label names to transform B->I only if necessary.
                    # We'll skip conversion here since most trainers accept repeating the same label.
                    # If you want strict BIO, keep the same label or convert to I-* offline.
                    new_labels.append(label)
                else:
                    new_labels.append(label)
            else:
                new_labels.append(ignore_index)
    return new_labels


# -------------------------
# Metrics
# -------------------------

seqeval = evaluate.load("seqeval")


def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        curr_preds = []
        curr_labs = []
        for p_i, l_i in zip(pred, lab):
            if l_i == -100:
                continue
            curr_preds.append(id2label[p_i])
            curr_labs.append(id2label[l_i])
        true_predictions.append(curr_preds)
        true_labels.append(curr_labs)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # Return key metrics
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--output_dir", type=str, default="outputs/clinicalbert-ner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_patience", type=int, default=3, help="Early stopping patience (eval steps).")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision if available.")
    parser.add_argument("--label_all_tokens", action="store_true", help="Propagate labels to subword pieces.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Read datasets
    train_tokens, train_tags = read_conll(args.train_file)
    val_tokens, val_tags = read_conll(args.validation_file)
    test_tokens, test_tags = read_conll(args.test_file)

    # Build label set from train (safer: union of all splits)
    label_list = sorted(set(tag for seq in (train_tags + val_tags + test_tags) for tag in seq))
    if "O" in label_list:
        label_list.remove("O")
        label_list = ["O"] + label_list

    # Create HF datasets
    ds_train = make_hf_dataset(train_tokens, train_tags, label_list)
    ds_val = make_hf_dataset(val_tokens, val_tags, label_list)
    ds_test = make_hf_dataset(test_tokens, test_tags, label_list)
    raw = DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})

    # Labels ↔ ids
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize + align
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        aligned_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = align_labels_with_tokens(
                labels, word_ids, label_all_tokens=args.label_all_tokens, ignore_index=-100
            )
            aligned_labels.append(aligned)
        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized = raw.map(tokenize_and_align, batched=True, remove_columns=raw["train"].column_names)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to=["none"],  # set to ["tensorboard"] if you want logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.eval_patience)],
    )

    # Train
    trainer.train()

    # Eval on dev + test
    print("\nValidation metrics:")
    val_metrics = trainer.evaluate(tokenized["validation"])
    print(val_metrics)

    print("\nTest metrics:")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(test_metrics)

    # Save final
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Inference example: run on the first test sentence
    example = raw["test"][0]["tokens"]
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    inputs = tokenizer(example, is_split_into_words=True, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}   # move to GPU/CPU
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(-1)[0].cpu()
        pred_ids = probs.argmax(-1).tolist()

    word_ids = inputs["input_ids"].new_zeros(inputs["input_ids"].shape).cpu()  # fix .word_ids issue
    word_ids = tokenizer(example, is_split_into_words=True).word_ids()

    # reconstruct per-token predictions
    pred_tags = []
    last_word = None
    for idx, wid in enumerate(word_ids):
        if wid is None or inputs["attention_mask"][0, idx].item() == 0:
            continue
        if wid != last_word:
            pred_tags.append(id2label[pred_ids[idx]])
            last_word = wid
    print("\nExample tokens:", example)
    print("Predicted tags:", pred_tags)


if __name__ == "__main__":
    # small torch import here to avoid top-level import time if you're only using pieces
    import torch

    main()
