"""Run sentiment inference with a fine-tuned DistilBERT model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_output_dir
from src.data import clean_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for review text")
    parser.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to saved model (defaults to training.output_dir)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Review text to classify. If omitted, reads from --file or stdin.",
    )
    parser.add_argument("--file", type=str, default=None, help="Path to a text file")
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    print("Enter review text (Ctrl+Z then Enter on Windows / Ctrl+D on Unix to finish):")
    return sys.stdin.read()


@torch.inference_mode()
def predict_sentiment(
    text: str,
    model_dir: Path,
    max_length: int = 256,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    cleaned = clean_review(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label.get(pred_id, str(pred_id))

    return {
        "label": label,
        "confidence": float(probs[pred_id].item()),
        "probabilities": {
            model.config.id2label.get(i, str(i)): float(probs[i].item())
            for i in range(probs.shape[0])
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else resolve_output_dir(cfg)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No model found at {model_dir}. Train first with: python src/train.py"
        )

    text = load_text(args).strip()
    if not text:
        raise ValueError("No text provided.")

    result = predict_sentiment(text, model_dir, max_length=cfg["model"]["max_length"])

    print(f"Sentiment : {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:10s}: {prob:.4f}")


if __name__ == "__main__":
    main()
