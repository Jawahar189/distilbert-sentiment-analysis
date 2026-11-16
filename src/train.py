"""Fine-tune DistilBERT on IMDb for binary sentiment classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Allow `python src/train.py` from the project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_output_dir
from src.data import prepare_data
from src.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDb")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override config: limit training samples (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Override config: limit eval samples",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override config: number of training epochs",
    )
    return parser.parse_args()


def build_training_args(cfg: dict, output_dir: Path) -> TrainingArguments:
    t = cfg["training"]
    use_fp16 = bool(t.get("fp16", False)) and torch.cuda.is_available()

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        warmup_steps=int(t.get("warmup_steps", 0)),
        logging_steps=int(t.get("logging_steps", 50)),
        eval_strategy=t.get("eval_strategy", "epoch"),
        save_strategy=t.get("save_strategy", "epoch"),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", True)),
        metric_for_best_model=t.get("metric_for_best_model", "f1"),
        greater_is_better=bool(t.get("greater_is_better", True)),
        seed=int(t.get("seed", 42)),
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg["data"]["max_eval_samples"] = args.max_eval_samples
    if args.epochs is not None:
        cfg["training"]["num_train_epochs"] = args.epochs

    set_seed(int(cfg["training"].get("seed", 42)))

    print("Preparing IMDb data and tokenizer...")
    tokenized, tokenizer = prepare_data(cfg)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(cfg, output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training on {len(tokenized['train'])} examples...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    train_result = trainer.train()

    print("Evaluating on test set...")
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "eval_metrics.json"
    payload = {
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "train_samples": len(tokenized["train"]),
        "eval_samples": len(tokenized["test"]),
        "accuracy": metrics.get("eval_accuracy"),
        "precision": metrics.get("eval_precision"),
        "recall": metrics.get("eval_recall"),
        "f1": metrics.get("eval_f1"),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Test metrics ===")
    for key in ("accuracy", "precision", "recall", "f1"):
        value = payload[key]
        if value is not None:
            print(f"  {key:10s}: {value:.4f}")
    print(f"\nModel saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
