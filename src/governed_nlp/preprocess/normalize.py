import re
import unicodedata
from typing import Optional
import pandas as pd

_MULTI_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

def light_normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_WS_RE.sub(" ", s).strip()
    return s

def add_normalized_text(df: pd.DataFrame, col: str = "response_text") -> pd.DataFrame:
    out = df.copy()
    out[f"{col}_raw"] = out[col].astype("string")
    out[f"{col}_norm"] = out[f"{col}_raw"].map(lambda x: light_normalize_text(x))
    return out
