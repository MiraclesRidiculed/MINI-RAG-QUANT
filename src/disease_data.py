"""Dataset helpers for symptom-to-diagnosis training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "gretelai/symptom_to_diagnosis"


@dataclass(frozen=True)
class DiagnosisExample:
    symptoms: str
    diagnosis: str


def default_dataset_dir(data_dir: Path) -> Path:
    return data_dir / "diagnosis_dataset"


def default_labels_path(data_dir: Path) -> Path:
    return data_dir / "diagnosis_labels.json"


def normalize_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    return collapsed


def _jsonl_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"{split}.jsonl"


def _read_examples(path: Path) -> list[DiagnosisExample]:
    examples: list[DiagnosisExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(
                DiagnosisExample(
                    symptoms=row["symptoms"],
                    diagnosis=row["diagnosis"],
                )
            )
    return examples


def _write_examples(path: Path, examples: list[DiagnosisExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "symptoms": example.symptoms,
                "diagnosis": example.diagnosis,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def download_or_load_dataset(dataset_dir: Path, *, force_refresh: bool = False) -> dict[str, list[DiagnosisExample]]:
    train_path = _jsonl_path(dataset_dir, "train")
    test_path = _jsonl_path(dataset_dir, "test")

    if train_path.exists() and test_path.exists() and not force_refresh:
        return {
            "train": _read_examples(train_path),
            "test": _read_examples(test_path),
        }

    remote = load_dataset(DATASET_NAME)
    train_examples = [
        DiagnosisExample(
            symptoms=normalize_text(row["input_text"]),
            diagnosis=normalize_text(row["output_text"]).lower(),
        )
        for row in remote["train"]
    ]
    test_examples = [
        DiagnosisExample(
            symptoms=normalize_text(row["input_text"]),
            diagnosis=normalize_text(row["output_text"]).lower(),
        )
        for row in remote["test"]
    ]

    _write_examples(train_path, train_examples)
    _write_examples(test_path, test_examples)

    metadata = {
        "dataset_name": DATASET_NAME,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "train": train_examples,
        "test": test_examples,
    }


def build_training_corpus(dataset_splits: dict[str, list[DiagnosisExample]]) -> list[str]:
    corpus: list[str] = []
    for split_examples in dataset_splits.values():
        for example in split_examples:
            corpus.append(example.symptoms)
            corpus.append(example.diagnosis)
            corpus.append(f"symptoms: {example.symptoms}\ndiagnosis: {example.diagnosis}")
    return corpus


def save_label_space(dataset_splits: dict[str, list[DiagnosisExample]], labels_path: Path) -> list[str]:
    labels = sorted({example.diagnosis for split in dataset_splits.values() for example in split})
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels
