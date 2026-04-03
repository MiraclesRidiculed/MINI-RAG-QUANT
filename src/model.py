from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of masked self-attention for the original MiniGPT demo."""

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * channels**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

    def forward_with_cache(
        self,
        x: torch.Tensor,
        *,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, t, _ = x.shape
        k_new = self.key(x)
        q = self.query(x)
        v_new = self.value(x)

        k = torch.cat((past_k, k_new), dim=1) if past_k is not None else k_new
        v = torch.cat((past_v, v_new), dim=1) if past_v is not None else v_new
        scale = k.size(-1) ** -0.5
        wei = q @ k.transpose(-2, -1) * scale

        if t > 1:
            total_t = k.size(1)
            past_len = total_t - t
            mask = torch.tril(torch.ones(t, total_t, device=x.device), diagonal=past_len)
            wei = wei.masked_fill(mask == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)
        return wei @ v, k, v


class MultiHeadAttention(nn.Module):
    """Multiple masked self-attention heads in parallel."""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        *,
        past: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        outputs: list[torch.Tensor] = []
        next_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        per_head_past = past or [None] * len(self.heads)

        for head, head_past in zip(self.heads, per_head_past, strict=False):
            past_k = head_past[0] if head_past is not None else None
            past_v = head_past[1] if head_past is not None else None
            out, k, v = head.forward_with_cache(x, past_k=past_k, past_v=past_v)
            outputs.append(out)
            next_past.append((k, v))

        return self.proj(torch.cat(outputs, dim=-1)), next_past


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block used by the original MiniGPT demo."""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        *,
        past: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, next_past = self.sa.forward_with_cache(self.ln1(x), past=past)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, next_past


class MiniGPT(nn.Module):
    """Kept for backward compatibility with the original demo."""

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        batch, t, channels = logits.shape
        loss = F.cross_entropy(logits.view(batch * t, channels), targets.view(batch * t))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def forward_step(
        self,
        idx: torch.Tensor,
        *,
        past: list[list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None,
        position: int = 0,
    ) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        if position >= self.block_size:
            raise ValueError(
                f"Position {position} exceeds block_size={self.block_size}. "
                "Use a shorter prompt or a larger block size for cached generation."
            )

        _, t = idx.shape
        if position + t > self.block_size:
            raise ValueError(
                f"Step ending at position {position + t} exceeds block_size={self.block_size}."
            )

        tok_emb = self.token_embedding_table(idx)
        positions = torch.arange(position, position + t, device=idx.device)
        pos_emb = self.position_embedding_table(positions)
        x = tok_emb + pos_emb

        next_past: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        per_block_past = past or [None] * len(self.blocks)
        for block, block_past in zip(self.blocks, per_block_past, strict=False):
            x, block_next_past = block.forward_with_cache(x, past=block_past)
            next_past.append(block_next_past)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, next_past

    def generate_with_kv_cache(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        total_len = idx.size(1) + max_new_tokens
        if total_len > self.block_size:
            raise ValueError(
                f"Cached generation requires total length <= block_size ({self.block_size}), got {total_len}."
            )

        generated = idx
        cache: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None
        logits = None
        for position in range(idx.size(1)):
            logits, cache = self.forward_step(
                generated[:, position : position + 1],
                past=cache,
                position=position,
            )

        for position in range(idx.size(1), total_len):
            if logits is None:
                raise RuntimeError("Cached generation did not produce logits for the initial prompt.")
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            logits, cache = self.forward_step(
                idx_next,
                past=cache,
                position=position,
            )

        return generated


class DiagnosisClassifier(nn.Module):
    """A compact transformer encoder for symptom-to-diagnosis classification."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        *,
        max_length: int,
        pad_token_id: int,
        n_embd: int = 96,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(n_embd)
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length={self.max_length}.")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        encoded = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        encoded = self.norm(encoded)

        mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
