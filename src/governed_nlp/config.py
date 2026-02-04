from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class QCConfig:
    required_cols: Tuple[str, ...] = ("response_id", "prompt_id", "response_text")
    min_words: int = 5
    max_chars_outlier: int = 1500
    punct_ratio_thresh: float = 0.6
    repeated_char_run: int = 20

@dataclass(frozen=True)
class TextConfig:
    max_length: int = 512
    chunk_size: int = 256
    chunk_stride: int = 256  # non-overlapping by default