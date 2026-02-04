import re
from typing import List, Tuple
import pandas as pd
from governed_nlp.config import QCConfig

_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def validate_schema(df: pd.DataFrame, cfg: QCConfig) -> List[str]:
    return [c for c in cfg.required_cols if c not in df.columns]

def dedupe_by_response_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dup_mask = df.duplicated(subset=["response_id"], keep="first")
    return df.loc[~dup_mask].copy(), df.loc[dup_mask].copy()

def _punct_ratio(s: str) -> float:
    s2 = re.sub(r"\s+", "", s or "")
    if not s2:
        return 0.0
    punct = sum(1 for ch in s2 if re.match(r"[^\w]", ch))
    return punct / len(s2)

def add_qc_flags(df: pd.DataFrame, cfg: QCConfig) -> pd.DataFrame:
    out = df.copy()
    text = out["response_text"].astype("string")

    out["char_count"] = text.str.len().fillna(0).astype(int)
    out["word_count"] = text.str.split().str.len().fillna(0).astype(int)

    out["flag_missing_required"] = out[list(cfg.required_cols)].isna().any(axis=1)
    out["flag_blank"] = text.fillna("").str.strip().eq("")
    out["flag_too_short"] = out["word_count"].lt(cfg.min_words) & ~out["flag_blank"]

    out["punct_ratio"] = text.fillna("").map(_punct_ratio)
    out["flag_mostly_punct"] = out["punct_ratio"].ge(cfg.punct_ratio_thresh) & ~out["flag_blank"]

    out["flag_non_printable"] = text.fillna("").str.contains(_NON_PRINTABLE_RE, regex=True)

    out["flag_outlier_length"] = out["char_count"].gt(cfg.max_chars_outlier)

    out["flag_repeated_char_run"] = text.fillna("").str.contains(
        rf"(.)\1{{{cfg.repeated_char_run-1},}}", regex=True
    )

    qc_flags = [c for c in out.columns if c.startswith("flag_")]
    out["needs_review"] = out[qc_flags].any(axis=1)
    return out
