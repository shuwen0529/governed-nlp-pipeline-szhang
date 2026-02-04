from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np # type: ignore
import pandas as pd # type: ignore


SplitMode = Literal["by_prompt", "by_group", "time"]


@dataclass(frozen=True)
class SplitConfig:
    mode: SplitMode = "by_prompt"
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
    prompt_col: str = "prompt_id"
    group_col: str = "student_id"  # optional grouping to prevent same entity across splits
    time_col: str = "created_ts"   # for time-based split


def _assert_sizes(cfg: SplitConfig):
    if cfg.test_size <= 0 or cfg.val_size <= 0 or (cfg.test_size + cfg.val_size) >= 0.9:
        raise ValueError("Invalid split sizes. Ensure test_size>0, val_size>0, and sum < 0.9.")


def split_leakage_safe(
    df: pd.DataFrame,
    cfg: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df).

    Modes:
      - by_prompt: split by prompt_id (no prompt appears in multiple splits)
      - by_group: split by group_col (e.g., student_id / account_id)
      - time: chronological split by time_col (train earliest, then val, then test)
    """
    _assert_sizes(cfg)
    out = df.copy()

    if cfg.mode == "by_prompt":
        if cfg.prompt_col not in out.columns:
            raise KeyError(f"Missing '{cfg.prompt_col}' for by_prompt split")
        keys = out[cfg.prompt_col].dropna().unique().tolist()
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(keys)

        n = len(keys)
        n_test = max(1, int(round(n * cfg.test_size)))
        n_val = max(1, int(round(n * cfg.val_size)))

        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        train_keys = set(keys[n_test + n_val:])

        test_df = out[out[cfg.prompt_col].isin(test_keys)].copy()
        val_df = out[out[cfg.prompt_col].isin(val_keys)].copy()
        train_df = out[out[cfg.prompt_col].isin(train_keys)].copy()

        return train_df, val_df, test_df

    if cfg.mode == "by_group":
        if cfg.group_col not in out.columns:
            raise KeyError(f"Missing '{cfg.group_col}' for by_group split")
        keys = out[cfg.group_col].dropna().unique().tolist()
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(keys)

        n = len(keys)
        n_test = max(1, int(round(n * cfg.test_size)))
        n_val = max(1, int(round(n * cfg.val_size)))

        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        train_keys = set(keys[n_test + n_val:])

        test_df = out[out[cfg.group_col].isin(test_keys)].copy()
        val_df = out[out[cfg.group_col].isin(val_keys)].copy()
        train_df = out[out[cfg.group_col].isin(train_keys)].copy()

        return train_df, val_df, test_df

    if cfg.mode == "time":
        if cfg.time_col not in out.columns:
            raise KeyError(f"Missing '{cfg.time_col}' for time split")
        out = out.sort_values(cfg.time_col).reset_index(drop=True)

        n = len(out)
        n_test = int(round(n * cfg.test_size))
        n_val = int(round(n * cfg.val_size))
        n_train = n - n_test - n_val
        if n_train <= 0:
            raise ValueError("Not enough rows for time split given test/val sizes.")

        train_df = out.iloc[:n_train].copy()
        val_df = out.iloc[n_train:n_train + n_val].copy()
        test_df = out.iloc[n_train + n_val:].copy()
        return train_df, val_df, test_df

    raise ValueError(f"Unsupported split mode: {cfg.mode}")
