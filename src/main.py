from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bpe import (
    SPECIAL_TOKENS,
    default_tokenizer_metadata_path,
    default_tokenizer_path,
    encode as bpe_encode,
    special_token_ids,
    train_or_load_tokenizer,
)
from disease_data import (
    DiagnosisExample,
    build_training_corpus,
    default_dataset_dir,
    default_labels_path,
    download_or_load_dataset,
    save_label_space,
)
from model import DiagnosisClassifier
from retrieve import DiagnosisRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
_DATASET_DIR = default_dataset_dir(_DATA)
_TOKENIZER_PATH = default_tokenizer_path(_DATA)
_TOKENIZER_METADATA_PATH = default_tokenizer_metadata_path(_DATA)
_LABELS_PATH = default_labels_path(_DATA)
_MODEL_PATH = _ROOT / "model_diagnosis.pth"
_METRICS_PATH = _DATA / "diagnosis_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run a BPE-driven disease diagnosis model.")
    parser.add_argument(
        "--query",
        type=str,
        default="I've had fever, body aches, vomiting, and severe weakness for the last two days.",
        help="Symptoms to evaluate after training/loading the model.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Target BPE vocabulary size.")
    parser.add_argument("--max-length", type=int, default=96, help="Maximum BPE tokens per symptom sequence.")
    parser.add_argument("--classifier-weight", type=float, default=0.4, help="Classifier weight in ensemble scoring.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved training examples to use.")
    parser.add_argument("--retrain", action="store_true", help="Rebuild dataset cache, tokenizer, and model.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SymptomDiagnosisDataset(Dataset):
    def __init__(
        self,
        examples: list[DiagnosisExample],
        tokenizer,
        label_to_id: dict[str, int],
        *,
        max_length: int,
        pad_token_id: int,
    ):
        self.rows = [
            self._encode_example(example, tokenizer, label_to_id, max_length=max_length, pad_token_id=pad_token_id)
            for example in examples
        ]

    @staticmethod
    def _encode_example(
        example: DiagnosisExample,
        tokenizer,
        label_to_id: dict[str, int],
        *,
        max_length: int,
        pad_token_id: int,
    ) -> dict[str, torch.Tensor]:
        token_ids = bpe_encode(tokenizer, example.symptoms)[:max_length]
        attention_mask = [1] * len(token_ids)
        pad_len = max_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label_to_id[example.diagnosis], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[idx]


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def evaluate(model: DiagnosisClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)

            predictions = logits.argmax(dim=-1)
            correct += (predictions == batch["label"]).sum().item()

            topk = logits.topk(k=min(3, logits.size(-1)), dim=-1).indices
            top3_correct += (topk == batch["label"].unsqueeze(1)).any(dim=1).sum().item()
            total += batch["label"].size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
    }


def train_model(
    model: DiagnosisClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_examples = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch["label"].size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size

        avg_loss = running_loss / max(total_examples, 1)
        print(f"Epoch {epoch:02d}/{epochs}: train_loss={avg_loss:.4f}")


def save_checkpoint(
    path: Path,
    *,
    model: DiagnosisClassifier,
    labels: list[str],
    metrics: dict[str, float],
    model_config: dict[str, int | float],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "labels": labels,
        "metrics": metrics,
        "config": model_config,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def build_or_load_model(
    *,
    tokenizer,
    labels: list[str],
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> tuple[DiagnosisClassifier, dict[str, float]]:
    token_ids = special_token_ids(tokenizer)
    model_config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "num_labels": len(labels),
        "max_length": args.max_length,
        "pad_token_id": token_ids["[PAD]"],
        "n_embd": 96,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.1,
    }

    should_train = args.retrain or not _MODEL_PATH.exists()
    checkpoint = None
    if not should_train:
        checkpoint = load_checkpoint(_MODEL_PATH, device)
        should_train = (
            checkpoint["labels"] != labels
            or checkpoint["config"]["vocab_size"] != tokenizer.get_vocab_size()
            or checkpoint["config"]["max_length"] != args.max_length
        )

    model = DiagnosisClassifier(**model_config).to(device)
    if should_train:
        print("Training diagnosis classifier...")
        train_model(
            model,
            train_loader,
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        metrics = evaluate(model, test_loader, device)
        save_checkpoint(_MODEL_PATH, model=model, labels=labels, metrics=metrics, model_config=model_config)
    else:
        model.load_state_dict(checkpoint["state_dict"])
        metrics = checkpoint.get("metrics", evaluate(model, test_loader, device))
        print("Loaded existing diagnosis classifier checkpoint.")

    return model, metrics


def predict_query(
    *,
    query: str,
    model: DiagnosisClassifier,
    tokenizer,
    labels: list[str],
    max_length: int,
    retriever: DiagnosisRetriever,
    device: torch.device,
    classifier_weight: float,
    top_k: int,
) -> dict[str, object]:
    token_ids = special_token_ids(tokenizer)
    encoded = bpe_encode(tokenizer, query)[:max_length]
    attention_mask = [1] * len(encoded)
    pad_len = max_length - len(encoded)
    if pad_len > 0:
        encoded = encoded + [token_ids["[PAD]"]] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    inputs = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.tensor(attention_mask, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(inputs, mask)
        classifier_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    classifier_scores = {label: classifier_probs[idx] for idx, label in enumerate(labels)}
    retrieval_scores = retriever.diagnosis_scores(query, top_k=top_k)

    ensemble_weight = max(0.0, min(1.0, classifier_weight))
    combined_scores = {}
    for label in labels:
        combined_scores[label] = (
            ensemble_weight * classifier_scores.get(label, 0.0)
            + (1.0 - ensemble_weight) * retrieval_scores.get(label, 0.0)
        )

    score_total = sum(combined_scores.values()) or 1.0
    normalized_scores = {label: score / score_total for label, score in combined_scores.items()}
    ranked = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
    retrieved = retriever.retrieve(query, top_k=top_k)

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    low_confidence = top_score < 0.45 or (top_score - second_score) < 0.10

    return {
        "ranked_predictions": ranked[:5],
        "retrieved_cases": retrieved,
        "classifier_scores": classifier_scores,
        "retrieval_scores": retrieval_scores,
        "low_confidence": low_confidence,
    }


def evaluate_prediction_stack(
    *,
    examples: list[DiagnosisExample],
    model: DiagnosisClassifier,
    tokenizer,
    labels: list[str],
    max_length: int,
    retriever: DiagnosisRetriever,
    device: torch.device,
    classifier_weight: float,
    top_k: int,
) -> dict[str, float]:
    ensemble_correct = 0
    retrieval_correct = 0
    low_confidence = 0

    for example in examples:
        prediction = predict_query(
            query=example.symptoms,
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            max_length=max_length,
            retriever=retriever,
            device=device,
            classifier_weight=classifier_weight,
            top_k=top_k,
        )
        if prediction["ranked_predictions"][0][0] == example.diagnosis:
            ensemble_correct += 1
        if prediction["retrieved_cases"] and prediction["retrieved_cases"][0].diagnosis == example.diagnosis:
            retrieval_correct += 1
        if prediction["low_confidence"]:
            low_confidence += 1

    total = max(len(examples), 1)
    return {
        "ensemble_accuracy": ensemble_correct / total,
        "retrieval_top1_accuracy": retrieval_correct / total,
        "low_confidence_rate": low_confidence / total,
    }


def print_prediction(result: dict[str, object]) -> None:
    ranked_predictions = result["ranked_predictions"]
    retrieved_cases = result["retrieved_cases"]

    lead_label, lead_score = ranked_predictions[0]
    banner = "Low-confidence prediction" if result["low_confidence"] else "Predicted diagnosis"
    print(f"{banner}: {lead_label} ({lead_score:.3f})")

    print("Top candidates:")
    for label, score in ranked_predictions:
        print(f"  - {label}: {score:.3f}")

    if retrieved_cases:
        print("Nearest training cases:")
        for item in retrieved_cases:
            print(f"  - score={item.score:.3f} | diagnosis={item.diagnosis} | symptoms={item.symptoms}")


def save_metrics(
    metrics: dict[str, float],
    *,
    tokenizer_vocab_size: int,
    labels: list[str],
    stack_metrics: dict[str, float],
) -> None:
    payload = {
        "dataset": "gretelai/symptom_to_diagnosis",
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "num_labels": len(labels),
        "classifier_metrics": metrics,
        "inference_stack_metrics": stack_metrics,
    }
    _METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_splits = download_or_load_dataset(_DATASET_DIR, force_refresh=args.retrain)
    labels = save_label_space(dataset_splits, _LABELS_PATH)
    tokenizer = train_or_load_tokenizer(
        build_training_corpus(dataset_splits),
        _TOKENIZER_PATH,
        metadata_path=_TOKENIZER_METADATA_PATH,
        vocab_size=args.vocab_size,
        force_retrain=args.retrain,
    )

    token_ids = special_token_ids(tokenizer)
    train_dataset = SymptomDiagnosisDataset(
        dataset_splits["train"],
        tokenizer,
        {label: idx for idx, label in enumerate(labels)},
        max_length=args.max_length,
        pad_token_id=token_ids["[PAD]"],
    )
    test_dataset = SymptomDiagnosisDataset(
        dataset_splits["test"],
        tokenizer,
        {label: idx for idx, label in enumerate(labels)},
        max_length=args.max_length,
        pad_token_id=token_ids["[PAD]"],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, metrics = build_or_load_model(
        tokenizer=tokenizer,
        labels=labels,
        args=args,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    print(f"Dataset cached in: {_DATASET_DIR}")
    print(f"BPE vocab size: {tokenizer.get_vocab_size()} ({len(SPECIAL_TOKENS)} reserved specials)")

    retriever = DiagnosisRetriever(tokenizer, dataset_splits["train"], reserved_token_count=len(SPECIAL_TOKENS))
    stack_metrics = evaluate_prediction_stack(
        examples=dataset_splits["test"],
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        max_length=args.max_length,
        retriever=retriever,
        device=device,
        classifier_weight=args.classifier_weight,
        top_k=args.top_k,
    )
    save_metrics(
        metrics,
        tokenizer_vocab_size=tokenizer.get_vocab_size(),
        labels=labels,
        stack_metrics=stack_metrics,
    )

    print(
        "Classifier test accuracy: "
        f"{metrics['accuracy']:.3f} | top-3: {metrics['top3_accuracy']:.3f}"
    )
    print(
        "Inference stack accuracy: "
        f"{stack_metrics['ensemble_accuracy']:.3f} | retrieval top-1: {stack_metrics['retrieval_top1_accuracy']:.3f}"
    )

    prediction = predict_query(
        query=args.query,
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        max_length=args.max_length,
        retriever=retriever,
        device=device,
        classifier_weight=args.classifier_weight,
        top_k=args.top_k,
    )

    print(f"Query: {args.query}")
    print_prediction(prediction)
    print("This model is for educational experimentation only and is not a clinical diagnostic tool.")


if __name__ == "__main__":
    main()
