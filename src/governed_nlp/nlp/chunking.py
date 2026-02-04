from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import pandas as pd # type: ignore


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 256
    stride: int = 256  # use < chunk_size for overlap
    pad_to_chunk: bool = True
    pad_token_id: int = 1  # RoBERTa pad_token_id=1, BERT pad_token_id=0 (set from tokenizer if possible)


def chunk_token_sequence(
    input_ids: List[int],
    attention_mask: List[int],
    cfg: ChunkConfig,
) -> Dict[str, List[List[int]]]:
    """
    Split a single tokenized example into chunks.
    Returns dict with:
      - input_ids_chunks: List[List[int]]
      - attention_mask_chunks: List[List[int]]
    """
    n = len(input_ids)
    chunks_ids: List[List[int]] = []
    chunks_mask: List[List[int]] = []

    start = 0
    while start < n:
        end = min(start + cfg.chunk_size, n)
        ids_chunk = input_ids[start:end]
        mask_chunk = attention_mask[start:end]

        if cfg.pad_to_chunk and len(ids_chunk) < cfg.chunk_size:
            pad_len = cfg.chunk_size - len(ids_chunk)
            ids_chunk = ids_chunk + [cfg.pad_token_id] * pad_len
            mask_chunk = mask_chunk + [0] * pad_len

        chunks_ids.append(ids_chunk)
        chunks_mask.append(mask_chunk)

        if end == n:
            break
        start += cfg.stride

    return {"input_ids_chunks": chunks_ids, "attention_mask_chunks": chunks_mask}


def add_token_chunks(
    df: pd.DataFrame,
    cfg: ChunkConfig,
    input_ids_col: str = "tok_input_ids",
    attention_mask_col: str = "tok_attention_mask",
    out_ids_col: str = "tok_input_ids_chunks",
    out_mask_col: str = "tok_attention_mask_chunks",
) -> pd.DataFrame:
    """
    Add chunked token columns to a tokenized DataFrame.
    """
    for c in (input_ids_col, attention_mask_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'")

    out = df.copy()

    def _chunk_row(r):
        ch = chunk_token_sequence(
            input_ids=r[input_ids_col],
            attention_mask=r[attention_mask_col],
            cfg=cfg,
        )
        return ch

    chunks = out.apply(_chunk_row, axis=1)
    out[out_ids_col] = chunks.map(lambda d: d["input_ids_chunks"])
    out[out_mask_col] = chunks.map(lambda d: d["attention_mask_chunks"])
    out["num_chunks"] = out[out_ids_col].map(len).astype(int)
    return out


def pool_chunk_scores(
    chunk_scores: List[float],
    method: Literal["mean", "max"] = "mean",
) -> float:
    """
    Simple pooling helper for downstream modeling/inference.
    chunk_scores might be per-chunk probabilities, logits, or regression outputs.
    """
    if not chunk_scores:
        return float("nan")
    if method == "mean":
        return float(sum(chunk_scores) / len(chunk_scores))
    if method == "max":
        return float(max(chunk_scores))
    raise ValueError(f"Unsupported pooling method: {method}")
