from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
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
from clinical_support import (
    build_assessment_sentence,
    default_critical_benchmark_path,
    detect_critical_conditions,
    evidence_cards_for_conditions,
    guidance_for_label,
    load_critical_benchmark,
)
from disease_data import (
    DiagnosisExample,
    build_or_load_expanded_dataset,
    build_training_corpus,
    default_dataset_dir,
    default_expanded_dataset_dir,
    default_labels_path,
    default_supplemental_dataset_dir,
    download_or_load_dataset,
    save_label_space,
)
from kv_cache import KVCacheConfig, kv_cache_summary
from model import DiagnosisClassifier
from quantize import estimated_int8_symmetric_per_tensor_bytes, floating_point_weight_bytes
from retrieve import DiagnosisRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
_DATASET_DIR = default_dataset_dir(_DATA)
_SUPPLEMENTAL_DATASET_DIR = default_supplemental_dataset_dir(_DATA)
_EXPANDED_DATASET_DIR = default_expanded_dataset_dir(_DATA)
_TOKENIZER_PATH = default_tokenizer_path(_DATA)
_TOKENIZER_METADATA_PATH = default_tokenizer_metadata_path(_DATA)
_LABELS_PATH = default_labels_path(_DATA)
_MODEL_PATH = _ROOT / "model_diagnosis.pth"
_METRICS_PATH = _DATA / "diagnosis_metrics.json"
_CRITICAL_BENCHMARK_PATH = default_critical_benchmark_path(_DATA)


@dataclass(frozen=True)
class InferenceRuntime:
    args: argparse.Namespace
    dataset_splits: dict[str, list[DiagnosisExample]]
    labels: list[str]
    tokenizer: object
    model: DiagnosisClassifier
    retriever: DiagnosisRetriever
    device: torch.device
    classifier_metrics: dict[str, float]


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--classifier-weight", type=float, default=0.2, help="Classifier weight in ensemble scoring.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved training examples to use.")
    parser.add_argument("--differential-size", type=int, default=5, help="Number of differential diagnoses to print.")
    parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.35,
        help="If the top ensemble score falls below this threshold, return an unknown / clinician-review outcome.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.08,
        help="If the top-vs-second score gap falls below this threshold, treat the result as open-set / low-confidence.",
    )
    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.16,
        help="If retrieval support falls below this threshold, increase abstention sensitivity.",
    )
    parser.add_argument("--retrain", action="store_true", help="Rebuild dataset cache, tokenizer, and model.")
    parser.add_argument(
        "--skip-expanded-data",
        action="store_true",
        help="Train only on the base + local supplemental datasets, skipping the larger normalized HF expansion set.",
    )
    parser.add_argument(
        "--expanded-train-per-label",
        type=int,
        default=16,
        help="Maximum number of normalized examples per label to keep from the large expansion dataset for training.",
    )
    parser.add_argument(
        "--expanded-test-per-label",
        type=int,
        default=2,
        help="Maximum number of normalized examples per label to keep from the large expansion dataset for testing.",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Print floating-point vs estimated int8-quantized weight size for the classifier (educational).",
    )
    parser.add_argument(
        "--show-kv-cache",
        action="store_true",
        help="Print reference decoder KV-cache memory estimates for the same model scale.",
    )
    parser.add_argument("--kv-seq-len", type=int, default=512, help="Sequence length to use for KV-cache estimates.")
    parser.add_argument(
        "--kv-window-size",
        type=int,
        default=128,
        help="Sliding-window size to use for KV-cache estimates.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
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


def normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probabilities if prob > 0.0)
    return entropy / math.log(len(probabilities))


def build_differential_diagnoses(
    *,
    ranked_scores: list[tuple[str, float]],
    safety_matches,
    differential_size: int,
) -> list[dict[str, object]]:
    differential: list[dict[str, object]] = []
    seen: set[str] = set()

    for match in safety_matches:
        label = match.name
        if label in seen:
            continue
        seen.add(label)
        rule_score = min(0.99, 0.50 + 0.12 * len(match.matched_group_indices))
        differential.append(
            {
                "label": label,
                "score": rule_score,
                "origin": "safety_rule",
                "rationale": match.rationale,
            }
        )

    for label, score in ranked_scores:
        if label in seen:
            continue
        seen.add(label)
        guidance = guidance_for_label(label)
        rationale = "Supported by the model ensemble and similar retrieved training cases."
        if guidance is not None:
            rationale = f"{rationale} Curated clinician reference is available for {guidance.name}."
        differential.append(
            {
                "label": label,
                "score": score,
                "origin": "model_ensemble",
                "rationale": rationale,
            }
        )
        if len(differential) >= differential_size:
            break

    return differential[:differential_size]


def evaluate_critical_condition_guardrails(
    *,
    examples,
    model: DiagnosisClassifier,
    tokenizer,
    labels: list[str],
    max_length: int,
    retriever: DiagnosisRetriever,
    device: torch.device,
    classifier_weight: float,
    top_k: int,
    differential_size: int,
    abstain_threshold: float,
    margin_threshold: float,
    retrieval_threshold: float,
) -> dict[str, object]:
    per_condition: dict[str, dict[str, float | int | bool]] = {}
    total_hits = 0
    total = 0

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
            differential_size=differential_size,
            abstain_threshold=abstain_threshold,
            margin_threshold=margin_threshold,
            retrieval_threshold=retrieval_threshold,
        )

        detected_labels = {item["label"] for item in prediction["differential"]}
        matched_safety_conditions = {item["condition"] for item in prediction["safety_findings"]}
        hit = example.condition in detected_labels or example.condition in matched_safety_conditions
        total_hits += int(hit)
        total += 1

        bucket = per_condition.setdefault(
            example.condition,
            {
                "support": 0,
                "hits": 0,
                "floor": 1.0,
            },
        )
        bucket["support"] += 1
        bucket["hits"] += int(hit)

    for condition, metrics in per_condition.items():
        support = max(int(metrics["support"]), 1)
        recall = int(metrics["hits"]) / support
        guidance = guidance_for_label(condition)
        floor = guidance.recall_floor if guidance is not None else 1.0
        metrics["recall"] = recall
        metrics["floor"] = floor
        metrics["passes_floor"] = recall >= floor

    return {
        "overall_recall": total_hits / max(total, 1),
        "conditions": per_condition,
    }


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
    differential_size: int,
    abstain_threshold: float,
    margin_threshold: float,
    retrieval_threshold: float,
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
    retrieved = retriever.retrieve(query, top_k=top_k)

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

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    top_margin = top_score - second_score
    top_retrieval_score = retrieved[0].score if retrieved else 0.0
    entropy_score = normalized_entropy(classifier_probs)
    low_confidence = top_score < 0.45 or top_margin < 0.10
    open_set_flag = (
        top_score < abstain_threshold
        or (top_margin < margin_threshold and top_retrieval_score < retrieval_threshold)
        or (entropy_score > 0.92 and top_retrieval_score < (retrieval_threshold + 0.05))
    )

    safety_matches = detect_critical_conditions(query)
    safety_labels_missing_from_model = any(match.name not in labels for match in safety_matches)
    unknown = open_set_flag or safety_labels_missing_from_model
    differential = build_differential_diagnoses(
        ranked_scores=ranked,
        safety_matches=safety_matches,
        differential_size=differential_size,
    )

    if unknown:
        primary_label = safety_matches[0].name if safety_matches else "unknown condition"
    else:
        primary_label = differential[0]["label"]
    alternative_labels = [item["label"] for item in differential[1:]]

    assessment_sentence = build_assessment_sentence(
        primary_label=primary_label,
        alternative_labels=alternative_labels,
        unknown=unknown,
        low_confidence=low_confidence,
        safety_matches=safety_matches,
    )
    evidence_card_labels = [] if unknown else [item["label"] for item in differential[: min(3, len(differential))]]
    evidence_cards = evidence_cards_for_conditions(
        safety_matches,
        evidence_card_labels,
    )
    escalation_messages = []
    for card in evidence_cards:
        message = card["escalation"]
        if message not in escalation_messages:
            escalation_messages.append(message)

    return {
        "assessment_sentence": assessment_sentence,
        "best_model_label": ranked[0][0],
        "best_model_score": ranked[0][1],
        "primary_label": primary_label,
        "primary_score": differential[0]["score"] if differential else 0.0,
        "differential": differential,
        "unknown": unknown,
        "open_set_flag": open_set_flag,
        "ranked_predictions": ranked[:5],
        "retrieved_cases": retrieved,
        "classifier_scores": classifier_scores,
        "retrieval_scores": retrieval_scores,
        "top_margin": top_margin,
        "top_retrieval_score": top_retrieval_score,
        "normalized_entropy": entropy_score,
        "low_confidence": low_confidence,
        "evidence_cards": evidence_cards,
        "safety_findings": [
            {
                "condition": match.name,
                "score": match.score,
                "matched_terms": list(match.matched_terms),
                "rationale": match.rationale,
                "escalation": match.guidance.escalation,
            }
            for match in safety_matches
        ],
        "escalation_messages": escalation_messages,
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
    differential_size: int,
    abstain_threshold: float,
    margin_threshold: float,
    retrieval_threshold: float,
) -> dict[str, float]:
    ensemble_correct = 0
    retrieval_correct = 0
    differential_top3 = 0
    low_confidence = 0
    abstentions = 0
    safety_gates = 0

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
            differential_size=differential_size,
            abstain_threshold=abstain_threshold,
            margin_threshold=margin_threshold,
            retrieval_threshold=retrieval_threshold,
        )
        if prediction["best_model_label"] == example.diagnosis:
            ensemble_correct += 1
        if prediction["retrieved_cases"] and prediction["retrieved_cases"][0].diagnosis == example.diagnosis:
            retrieval_correct += 1
        if example.diagnosis in {item["label"] for item in prediction["differential"][:3]}:
            differential_top3 += 1
        if prediction["low_confidence"]:
            low_confidence += 1
        if prediction["unknown"]:
            abstentions += 1
        if prediction["safety_findings"]:
            safety_gates += 1

    total = max(len(examples), 1)
    return {
        "ensemble_accuracy": ensemble_correct / total,
        "retrieval_top1_accuracy": retrieval_correct / total,
        "differential_top3_recall": differential_top3 / total,
        "low_confidence_rate": low_confidence / total,
        "abstain_rate": abstentions / total,
        "safety_gate_rate": safety_gates / total,
    }


def print_prediction(result: dict[str, object]) -> None:
    ranked_predictions = result["ranked_predictions"]
    retrieved_cases = result["retrieved_cases"]

    print(f"Assessment: {result['assessment_sentence']}")
    if result["unknown"]:
        print(f"Primary output: unknown / clinician review needed (rule out {result['primary_label']})")
    else:
        print(f"Primary output: {result['primary_label']} ({result['primary_score']:.3f})")

    print("Differential diagnoses:")
    for item in result["differential"]:
        print(
            f"  - {item['label']}: {item['score']:.3f} | origin={item['origin']} | rationale={item['rationale']}"
        )

    if result["safety_findings"]:
        print("Safety-critical conditions to rule out:")
        for finding in result["safety_findings"]:
            print(
                f"  - {finding['condition']}: score={finding['score']:.3f} | rationale={finding['rationale']}"
            )

    if result["evidence_cards"]:
        print("Curated clinician evidence:")
        for card in result["evidence_cards"]:
            print(f"  - {card['condition']}: {card['summary']}")
            print(f"    escalation={card['escalation']}")
            for source in card["sources"]:
                print(f"    source={source['title']} | {source['url']}")

    if retrieved_cases:
        print("Nearest training cases:")
        for item in retrieved_cases:
            print(f"  - score={item.score:.3f} | diagnosis={item.diagnosis} | symptoms={item.symptoms}")

    print(
        "Open-set diagnostics: "
        f"unknown={result['unknown']} | low_confidence={result['low_confidence']} "
        f"| top_margin={result['top_margin']:.3f} | top_retrieval={result['top_retrieval_score']:.3f} "
        f"| entropy={result['normalized_entropy']:.3f}"
    )


def save_metrics(
    metrics: dict[str, float],
    *,
    tokenizer_vocab_size: int,
    labels: list[str],
    stack_metrics: dict[str, float],
    critical_condition_metrics: dict[str, object],
) -> None:
    payload = {
        "dataset": "gretelai/symptom_to_diagnosis + local supplements + normalized expanded dataset",
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "num_labels": len(labels),
        "classifier_metrics": metrics,
        "inference_stack_metrics": stack_metrics,
        "critical_condition_metrics": critical_condition_metrics,
    }
    _METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reference_kv_cache_config(model: DiagnosisClassifier, *, sequence_length: int, batch_size: int = 1) -> KVCacheConfig:
    embed_dim = model.token_embedding.embedding_dim
    num_layers = len(model.encoder.layers)
    num_heads = model.encoder.layers[0].self_attn.num_heads if num_layers else 1
    head_dim = embed_dim // max(num_heads, 1)
    return KVCacheConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
    )


def build_runtime(args: argparse.Namespace) -> InferenceRuntime:
    supplemental_dirs = [_SUPPLEMENTAL_DATASET_DIR]
    if not args.skip_expanded_data:
        build_or_load_expanded_dataset(
            _EXPANDED_DATASET_DIR,
            force_refresh=args.retrain,
            seed=args.seed,
            max_train_per_label=args.expanded_train_per_label,
            max_test_per_label=args.expanded_test_per_label,
        )
        supplemental_dirs.append(_EXPANDED_DATASET_DIR)

    dataset_splits = download_or_load_dataset(
        _DATASET_DIR,
        force_refresh=args.retrain,
        supplemental_dirs=supplemental_dirs,
    )
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
    retriever = DiagnosisRetriever(tokenizer, dataset_splits["train"], reserved_token_count=len(SPECIAL_TOKENS))

    return InferenceRuntime(
        args=args,
        dataset_splits=dataset_splits,
        labels=labels,
        tokenizer=tokenizer,
        model=model,
        retriever=retriever,
        device=device,
        classifier_metrics=metrics,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runtime = build_runtime(args)

    if args.show_memory:
        sd = runtime.model.state_dict()
        fp_bytes = floating_point_weight_bytes(sd)
        q_bytes = estimated_int8_symmetric_per_tensor_bytes(sd)
        ratio = fp_bytes / max(q_bytes, 1)
        print(
            "Classifier weight memory (floating-point vs symmetric int8 estimate): "
            f"{fp_bytes / 1024:.2f} KiB FP → ~{q_bytes / 1024:.2f} KiB int8+scales "
            f"(~{ratio:.2f}× smaller payload; int8 is illustrative, not applied at runtime)."
        )

    if args.show_kv_cache:
        kv_config = reference_kv_cache_config(runtime.model, sequence_length=args.kv_seq_len)
        kv_summary = kv_cache_summary(kv_config, window_size=args.kv_window_size)
        print(
            "Reference decoder KV-cache memory "
            f"(batch={kv_config.batch_size}, seq={kv_config.sequence_length}, "
            f"layers={kv_config.num_layers}, heads={kv_config.num_heads}, head_dim={kv_config.head_dim}): "
            f"{kv_summary['fp32_bytes'] / 1024:.2f} KiB FP32 | "
            f"{kv_summary['fp16_bytes'] / 1024:.2f} KiB FP16 | "
            f"~{kv_summary['int8_bytes'] / 1024:.2f} KiB INT8+scales "
            f"(~{kv_summary['int8_vs_fp16_ratio']:.2f}x smaller than FP16) | "
            f"{kv_summary['sliding_window_fp16_bytes'] / 1024:.2f} KiB FP16 sliding-window({args.kv_window_size}) "
            f"(~{kv_summary['sliding_vs_fp16_ratio']:.2f}x smaller than full FP16 cache)."
        )

    print(f"Dataset cached in: {_DATASET_DIR}")
    print(f"BPE vocab size: {runtime.tokenizer.get_vocab_size()} ({len(SPECIAL_TOKENS)} reserved specials)")

    critical_benchmark = load_critical_benchmark(_CRITICAL_BENCHMARK_PATH)
    stack_metrics = evaluate_prediction_stack(
        examples=runtime.dataset_splits["test"],
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        labels=runtime.labels,
        max_length=args.max_length,
        retriever=runtime.retriever,
        device=runtime.device,
        classifier_weight=args.classifier_weight,
        top_k=args.top_k,
        differential_size=args.differential_size,
        abstain_threshold=args.abstain_threshold,
        margin_threshold=args.margin_threshold,
        retrieval_threshold=args.retrieval_threshold,
    )
    critical_condition_metrics = evaluate_critical_condition_guardrails(
        examples=critical_benchmark,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        labels=runtime.labels,
        max_length=args.max_length,
        retriever=runtime.retriever,
        device=runtime.device,
        classifier_weight=args.classifier_weight,
        top_k=args.top_k,
        differential_size=args.differential_size,
        abstain_threshold=args.abstain_threshold,
        margin_threshold=args.margin_threshold,
        retrieval_threshold=args.retrieval_threshold,
    )
    save_metrics(
        runtime.classifier_metrics,
        tokenizer_vocab_size=runtime.tokenizer.get_vocab_size(),
        labels=runtime.labels,
        stack_metrics=stack_metrics,
        critical_condition_metrics=critical_condition_metrics,
    )

    print(
        "Classifier test accuracy: "
        f"{runtime.classifier_metrics['accuracy']:.3f} | top-3: {runtime.classifier_metrics['top3_accuracy']:.3f}"
    )
    print(
        "Inference stack accuracy: "
        f"{stack_metrics['ensemble_accuracy']:.3f} | retrieval top-1: {stack_metrics['retrieval_top1_accuracy']:.3f} "
        f"| differential top-3: {stack_metrics['differential_top3_recall']:.3f}"
    )
    if critical_benchmark:
        print(
            "Critical-condition benchmark recall: "
            f"{critical_condition_metrics['overall_recall']:.3f} across {len(critical_benchmark)} curated must-not-miss cases"
        )
        for condition, condition_metrics in critical_condition_metrics["conditions"].items():
            status = "PASS" if condition_metrics["passes_floor"] else "FAIL"
            print(
                f"  - {condition}: recall={condition_metrics['recall']:.3f} "
                f"(floor={condition_metrics['floor']:.2f}, support={condition_metrics['support']}) [{status}]"
            )

    prediction = predict_query(
        query=args.query,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        labels=runtime.labels,
        max_length=args.max_length,
        retriever=runtime.retriever,
        device=runtime.device,
        classifier_weight=args.classifier_weight,
        top_k=args.top_k,
        differential_size=args.differential_size,
        abstain_threshold=args.abstain_threshold,
        margin_threshold=args.margin_threshold,
        retrieval_threshold=args.retrieval_threshold,
    )

    print(f"Query: {args.query}")
    print_prediction(prediction)
    print("This model is for educational experimentation only and is not a clinical diagnostic tool.")


if __name__ == "__main__":
    main()
