from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log, sqrt

import torch
from tokenizers import Tokenizer

from disease_data import DiagnosisExample


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a, b)


def retrieve(query_vec: torch.Tensor, doc_vecs: list[torch.Tensor], docs: list[str]) -> str:
    scores = [similarity(query_vec, doc) for doc in doc_vecs]
    idx = torch.argmax(torch.tensor(scores))
    return docs[int(idx)]


@dataclass(frozen=True)
class RetrievedCase:
    score: float
    symptoms: str
    diagnosis: str


class DiagnosisRetriever:
    """A lexical retriever over BPE tokens for grounded diagnosis suggestions."""

    def __init__(self, tokenizer: Tokenizer, examples: list[DiagnosisExample], *, reserved_token_count: int = 5):
        self.tokenizer = tokenizer
        self.examples = examples
        self.reserved_token_count = reserved_token_count
        self.document_frequencies = Counter()
        self.doc_vectors: list[tuple[dict[int, float], float, DiagnosisExample]] = []
        self._fit()

    def _token_counts(self, text: str) -> Counter[int]:
        token_ids = [token_id for token_id in self.tokenizer.encode(text).ids if token_id >= self.reserved_token_count]
        return Counter(token_ids)

    def _fit(self) -> None:
        counts_per_doc: list[tuple[Counter[int], DiagnosisExample]] = []
        for example in self.examples:
            counts = self._token_counts(example.symptoms)
            counts_per_doc.append((counts, example))
            for token_id in counts:
                self.document_frequencies[token_id] += 1

        total_docs = len(counts_per_doc)
        idf = {token_id: log((total_docs + 1) / (freq + 1)) + 1.0 for token_id, freq in self.document_frequencies.items()}
        for counts, example in counts_per_doc:
            weights = {token_id: freq * idf[token_id] for token_id, freq in counts.items()}
            norm = sqrt(sum(value * value for value in weights.values())) or 1.0
            self.doc_vectors.append((weights, norm, example))

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedCase]:
        counts = self._token_counts(query)
        if not counts:
            return []

        idf = {
            token_id: log((len(self.doc_vectors) + 1) / (self.document_frequencies[token_id] + 1)) + 1.0
            for token_id in counts
            if token_id in self.document_frequencies
        }
        query_weights = {token_id: freq * idf.get(token_id, 0.0) for token_id, freq in counts.items()}
        query_norm = sqrt(sum(value * value for value in query_weights.values())) or 1.0

        scored: list[RetrievedCase] = []
        for weights, norm, example in self.doc_vectors:
            dot = sum(query_weights.get(token_id, 0.0) * weight for token_id, weight in weights.items())
            if dot <= 0:
                continue
            scored.append(
                RetrievedCase(
                    score=dot / (query_norm * norm),
                    symptoms=example.symptoms,
                    diagnosis=example.diagnosis,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def diagnosis_scores(self, query: str, *, top_k: int = 5) -> dict[str, float]:
        retrieved = self.retrieve(query, top_k=top_k)
        totals: dict[str, float] = defaultdict(float)
        if not retrieved:
            return dict(totals)

        for item in retrieved:
            totals[item.diagnosis] += item.score

        total_score = sum(totals.values()) or 1.0
        return {diagnosis: score / total_score for diagnosis, score in totals.items()}
