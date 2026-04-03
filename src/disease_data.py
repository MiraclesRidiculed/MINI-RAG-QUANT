"""Dataset helpers for symptom-to-diagnosis training."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "gretelai/symptom_to_diagnosis"
EXPANDED_DATASET_NAME = "fhai50032/SymptomsDisease246k"


@dataclass(frozen=True)
class DiagnosisExample:
    symptoms: str
    diagnosis: str


def default_dataset_dir(data_dir: Path) -> Path:
    return data_dir / "diagnosis_dataset"


def default_labels_path(data_dir: Path) -> Path:
    return data_dir / "diagnosis_labels.json"


def default_supplemental_dataset_dir(data_dir: Path) -> Path:
    return data_dir / "supplemental_diagnosis_dataset"


def default_expanded_dataset_dir(data_dir: Path) -> Path:
    return data_dir / "expanded_diagnosis_dataset"


def normalize_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    return collapsed


def normalize_diagnosis_label(text: str) -> str:
    label = normalize_text(text).lower()
    label = re.sub(r"^(you may have|possible diagnosis(?:es)?(?: include)?|likely diagnosis(?:es)?(?: include)?)\s+", "", label)
    label = re.sub(r"^[^a-z0-9]+", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip(" .:-")


def normalize_symptom_query(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"(?i)^having these specific symptoms\s*:\s*->\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\s*may indicate\s*$", "", cleaned)
    cleaned = cleaned.replace(" ,", ",")
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    return cleaned.strip(" .")


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


def _normalize_example(example: DiagnosisExample) -> DiagnosisExample:
    return DiagnosisExample(
        symptoms=normalize_symptom_query(example.symptoms),
        diagnosis=normalize_diagnosis_label(example.diagnosis),
    )


def _dedupe_examples(*groups: list[DiagnosisExample]) -> list[DiagnosisExample]:
    merged: list[DiagnosisExample] = []
    seen: set[tuple[str, str]] = set()
    for examples in groups:
        for example in examples:
            normalized = _normalize_example(example)
            key = (normalized.symptoms.casefold(), normalized.diagnosis.casefold())
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _load_local_split_examples(dataset_dir: Path | None, split: str) -> list[DiagnosisExample]:
    if dataset_dir is None:
        return []

    path = _jsonl_path(dataset_dir, split)
    if not path.exists():
        return []
    return _read_examples(path)


def build_or_load_expanded_dataset(
    dataset_dir: Path,
    *,
    force_refresh: bool = False,
    seed: int = 7,
    max_train_per_label: int = 16,
    max_test_per_label: int = 2,
) -> dict[str, list[DiagnosisExample]]:
    train_path = _jsonl_path(dataset_dir, "train")
    test_path = _jsonl_path(dataset_dir, "test")

    if train_path.exists() and test_path.exists() and not force_refresh:
        return {
            "train": _read_examples(train_path),
            "test": _read_examples(test_path),
        }

    remote = load_dataset(EXPANDED_DATASET_NAME, split="train")
    grouped_examples: dict[str, list[DiagnosisExample]] = {}
    seen: set[tuple[str, str]] = set()

    for row in remote:
        symptoms = normalize_symptom_query(str(row["query"]))
        diagnosis = normalize_diagnosis_label(str(row["response"]))
        if not symptoms or not diagnosis:
            continue

        key = (symptoms.casefold(), diagnosis.casefold())
        if key in seen:
            continue
        seen.add(key)
        grouped_examples.setdefault(diagnosis, []).append(
            DiagnosisExample(symptoms=symptoms, diagnosis=diagnosis)
        )

    rng = random.Random(seed)
    train_examples: list[DiagnosisExample] = []
    test_examples: list[DiagnosisExample] = []
    per_label_counts: dict[str, dict[str, int]] = {}

    for diagnosis in sorted(grouped_examples):
        examples = grouped_examples[diagnosis][:]
        rng.shuffle(examples)

        if len(examples) >= 6:
            test_count = min(max_test_per_label, max(1, len(examples) // 20))
        elif len(examples) >= 3:
            test_count = 1
        else:
            test_count = 0

        test_count = min(test_count, max(len(examples) - 1, 0))
        train_count = min(max_train_per_label, len(examples) - test_count)
        if train_count <= 0:
            continue

        train_slice = examples[:train_count]
        test_slice = examples[train_count : train_count + test_count]
        train_examples.extend(train_slice)
        test_examples.extend(test_slice)
        per_label_counts[diagnosis] = {
            "available": len(examples),
            "train": len(train_slice),
            "test": len(test_slice),
        }

    _write_examples(train_path, train_examples)
    _write_examples(test_path, test_examples)

    metadata = {
        "dataset_name": EXPANDED_DATASET_NAME,
        "raw_unique_labels": len(grouped_examples),
        "raw_unique_examples": len(seen),
        "max_train_per_label": max_train_per_label,
        "max_test_per_label": max_test_per_label,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "labels_with_train_examples": len(per_label_counts),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "train": train_examples,
        "test": test_examples,
    }


def download_or_load_dataset(
    dataset_dir: Path,
    *,
    force_refresh: bool = False,
    supplemental_dirs: list[Path] | None = None,
) -> dict[str, list[DiagnosisExample]]:
    train_path = _jsonl_path(dataset_dir, "train")
    test_path = _jsonl_path(dataset_dir, "test")

    if train_path.exists() and test_path.exists() and not force_refresh:
        base_splits = {
            "train": _read_examples(train_path),
            "test": _read_examples(test_path),
        }
    else:
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
        base_splits = {
            "train": train_examples,
            "test": test_examples,
        }

    normalized_supplemental_dirs = [path for path in (supplemental_dirs or []) if path is not None]
    supplemental_splits = {
        "train": [_load_local_split_examples(path, "train") for path in normalized_supplemental_dirs],
        "test": [_load_local_split_examples(path, "test") for path in normalized_supplemental_dirs],
    }

    merged_splits = {
        split: _dedupe_examples(base_splits[split], *supplemental_splits[split])
        for split in ("train", "test")
    }

    metadata = {
        "dataset_name": DATASET_NAME,
        "base_train_examples": len(base_splits["train"]),
        "base_test_examples": len(base_splits["test"]),
        "supplemental_dataset_dirs": [str(path) for path in normalized_supplemental_dirs],
        "supplemental_train_examples": sum(len(examples) for examples in supplemental_splits["train"]),
        "supplemental_test_examples": sum(len(examples) for examples in supplemental_splits["test"]),
        "train_examples": len(merged_splits["train"]),
        "test_examples": len(merged_splits["test"]),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return merged_splits


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
