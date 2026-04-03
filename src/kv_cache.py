from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheConfig:
    batch_size: int
    sequence_length: int
    num_layers: int
    num_heads: int
    head_dim: int


def kv_cache_bytes(config: KVCacheConfig, *, dtype_bytes: int) -> int:
    return (
        config.batch_size
        * config.sequence_length
        * config.num_layers
        * config.num_heads
        * config.head_dim
        * 2
        * dtype_bytes
    )


def kv_cache_fp32_bytes(config: KVCacheConfig) -> int:
    return kv_cache_bytes(config, dtype_bytes=4)


def kv_cache_fp16_bytes(config: KVCacheConfig) -> int:
    return kv_cache_bytes(config, dtype_bytes=2)


def kv_cache_int8_bytes(config: KVCacheConfig, *, scale_bytes: int = 4) -> int:
    payload = kv_cache_bytes(config, dtype_bytes=1)
    scales = config.batch_size * config.num_layers * config.num_heads * 2 * scale_bytes
    return payload + scales


def kv_cache_sliding_window_fp16_bytes(config: KVCacheConfig, *, window_size: int) -> int:
    effective = min(config.sequence_length, window_size)
    return kv_cache_fp16_bytes(
        KVCacheConfig(
            batch_size=config.batch_size,
            sequence_length=effective,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
        )
    )


def kv_cache_summary(config: KVCacheConfig, *, window_size: int) -> dict[str, float | int]:
    fp32_bytes = kv_cache_fp32_bytes(config)
    fp16_bytes = kv_cache_fp16_bytes(config)
    int8_bytes = kv_cache_int8_bytes(config)
    sliding_fp16_bytes = kv_cache_sliding_window_fp16_bytes(config, window_size=window_size)
    return {
        "fp32_bytes": fp32_bytes,
        "fp16_bytes": fp16_bytes,
        "int8_bytes": int8_bytes,
        "sliding_window_fp16_bytes": sliding_fp16_bytes,
        "int8_vs_fp16_ratio": fp16_bytes / max(int8_bytes, 1),
        "sliding_vs_fp16_ratio": fp16_bytes / max(sliding_fp16_bytes, 1),
        "window_size": window_size,
    }
