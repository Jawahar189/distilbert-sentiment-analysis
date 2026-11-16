"""IMDb loading, cleaning, and DistilBERT tokenization pipeline."""

from __future__ import annotations

import re
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_review(text: str) -> str:
    """Strip HTML tags and normalize whitespace in raw IMDb reviews."""
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_imdb(
    dataset_name: str = "stanfordnlp/imdb",
    train_split: str = "train",
    test_split: str = "test",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    """Download IMDb and optionally subsample for quick experiments."""
    dataset = load_dataset(dataset_name)

    train = dataset[train_split]
    test = dataset[test_split]

    if max_train_samples is not None:
        train = train.shuffle(seed=42).select(range(min(max_train_samples, len(train))))
    if max_eval_samples is not None:
        test = test.shuffle(seed=42).select(range(min(max_eval_samples, len(test))))

    return DatasetDict(train=train, test=test)


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
) -> DatasetDict:
    """Clean text, tokenize, and format columns for sequence classification."""

    def _map_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        cleaned = [clean_review(t) for t in batch["text"]]
        encoded = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    tokenized = dataset.map(
        _map_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing IMDb reviews",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def prepare_data(cfg: dict[str, Any]) -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    """End-to-end data pipeline used by training and evaluation."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    tokenizer = build_tokenizer(model_cfg["name"])
    raw = load_imdb(
        dataset_name=data_cfg.get("dataset", "stanfordnlp/imdb"),
        train_split=data_cfg.get("train_split", "train"),
        test_split=data_cfg.get("test_split", "test"),
        max_train_samples=data_cfg.get("max_train_samples"),
        max_eval_samples=data_cfg.get("max_eval_samples"),
    )
    tokenized = preprocess_dataset(raw, tokenizer, max_length=model_cfg["max_length"])
    return tokenized, tokenizer