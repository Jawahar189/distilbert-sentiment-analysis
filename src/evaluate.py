"""Evaluate a saved DistilBERT checkpoint on the IMDb test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_output_dir
from src.data import load_imdb, preprocess_dataset
from src.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DistilBERT on IMDb")
    parser.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to saved model (defaults to training.output_dir)",
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_dir = Path(args.model_dir) if args.model_dir else resolve_output_dir(cfg)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No model found at {model_dir}. Train first with: python src/train.py"
        )

    max_eval = args.max_eval_samples or cfg["data"].get("max_eval_samples")
    raw = load_imdb(
        dataset_name=cfg["data"].get("dataset", "stanfordnlp/imdb"),
        max_eval_samples=max_eval,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenized = preprocess_dataset(raw, tokenizer, max_length=cfg["model"]["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "eval_tmp"),
            per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
            report_to="none",
        ),
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    summary = {
        "model_dir": str(model_dir),
        "eval_samples": len(tokenized["test"]),
        "accuracy": metrics.get("eval_accuracy"),
        "precision": metrics.get("eval_precision"),
        "recall": metrics.get("eval_recall"),
        "f1": metrics.get("eval_f1"),
    }

    out_path = model_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Evaluation metrics ===")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key:10s}: {summary[key]:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
