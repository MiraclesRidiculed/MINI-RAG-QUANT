"""Utilities for training and using a medical-domain byte-level BPE tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]


def default_tokenizer_path(data_dir: Path) -> Path:
    return data_dir / "diagnosis_bpe_tokenizer.json"


def default_tokenizer_metadata_path(data_dir: Path) -> Path:
    return data_dir / "diagnosis_bpe_tokenizer.meta.json"


def _load_training_texts(source: Iterable[str] | Path) -> list[str]:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors="replace").splitlines()
    return [text for text in source if text and text.strip()]


def _write_metadata(
    metadata_path: Path | None,
    *,
    tokenizer: Tokenizer,
    num_training_texts: int,
    vocab_size_requested: int,
    min_frequency: int,
) -> None:
    if metadata_path is None:
        return

    metadata = {
        "tokenizer_type": "byte-level-bpe",
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size_requested,
        "min_frequency": min_frequency,
        "num_training_texts": num_training_texts,
        "special_tokens": SPECIAL_TOKENS,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_or_load_tokenizer(
    corpus_source: Iterable[str] | Path,
    tokenizer_path: Path,
    *,
    metadata_path: Path | None = None,
    vocab_size: int = 1024,
    min_frequency: int = 2,
    force_retrain: bool = False,
) -> Tokenizer:
    """Train a reusable byte-level BPE tokenizer or load an existing one."""
    if tokenizer_path.exists() and not force_retrain:
        return Tokenizer.from_file(str(tokenizer_path))

    texts = _load_training_texts(corpus_source)
    if not texts:
        raise ValueError("Cannot train BPE tokenizer without any text.")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    _write_metadata(
        metadata_path,
        tokenizer=tokenizer,
        num_training_texts=len(texts),
        vocab_size_requested=vocab_size,
        min_frequency=min_frequency,
    )
    return tokenizer


def special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    missing = [token for token, token_id in ids.items() if token_id is None]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required special tokens: {missing}")
    return ids


def encode(tokenizer: Tokenizer, text: str, *, add_special_tokens: bool = True) -> list[int]:
    token_ids = tokenizer.encode(text).ids
    if not add_special_tokens:
        return token_ids

    ids = special_token_ids(tokenizer)
    return [ids["[BOS]"], *token_ids, ids["[EOS]"]]


def decode(tokenizer: Tokenizer, ids: list[int], *, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
