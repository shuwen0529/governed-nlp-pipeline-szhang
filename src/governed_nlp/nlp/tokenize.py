from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd # type: ignore

from transformers import AutoTokenizer # type: ignore


@dataclass(frozen=True)
class TokenizeConfig:
    model_name: str = "roberta-base"
    max_length: int = 256
    padding: str = "max_length"  # or "longest" for dynamic
    truncation: bool = True


def get_tokenizer(cfg: TokenizeConfig):
    return AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)


def tokenize_dataframe(
    df: pd.DataFrame,
    cfg: TokenizeConfig,
    text_col: str = "model_text",
    out_prefix: str = "tok_",
) -> pd.DataFrame:
    """
    Tokenize df[text_col] and append columns:
      - tok_input_ids (list[int])
      - tok_attention_mask (list[int])
      - (optional) tok_token_type_ids for some models
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing required column '{text_col}'")

    tokenizer = get_tokenizer(cfg)

    texts: List[str] = df[text_col].fillna("").astype(str).tolist()
    enc = tokenizer(
        texts,
        max_length=cfg.max_length,
        padding=cfg.padding,
        truncation=cfg.truncation,
        return_attention_mask=True,
    )

    out = df.copy()
    out[f"{out_prefix}input_ids"] = enc["input_ids"]
    out[f"{out_prefix}attention_mask"] = enc["attention_mask"]
    if "token_type_ids" in enc:
        out[f"{out_prefix}token_type_ids"] = enc["token_type_ids"]

    return out
