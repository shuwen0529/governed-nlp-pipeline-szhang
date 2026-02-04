from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd # type: ignore


PromptMode = Literal["concat", "template", "id_prefix"]


@dataclass(frozen=True)
class PromptFormatConfig:
    mode: PromptMode = "concat"
    sep: str = " [SEP] "
    prompt_prefix: str = "PROMPT: "
    response_prefix: str = "RESPONSE: "
    include_prompt_text: bool = True  # if False, uses prompt_id only


def build_prompt_aware_text(
    df: pd.DataFrame,
    cfg: PromptFormatConfig = PromptFormatConfig(),
    prompt_text_col: str = "prompt_text",
    prompt_id_col: str = "prompt_id",
    response_col: str = "response_text_norm",
    out_col: str = "model_text",
) -> pd.DataFrame:
    """
    Create a single model input string per row using prompt context.

    Expected columns:
      - response_text_norm (or response_col)
      - prompt_text (optional) and prompt_id
    """
    out = df.copy()

    if response_col not in out.columns:
        raise KeyError(f"Missing required column '{response_col}'")

    # Choose prompt context: prompt_text if available, else prompt_id
    def _prompt_ctx(row) -> str:
        if cfg.include_prompt_text and prompt_text_col in out.columns and pd.notna(row.get(prompt_text_col)):
            return str(row.get(prompt_text_col))
        if prompt_id_col in out.columns and pd.notna(row.get(prompt_id_col)):
            return f"[PROMPT_ID={row.get(prompt_id_col)}]"
        return "[PROMPT_ID=UNKNOWN]"

    if cfg.mode == "concat":
        out[out_col] = out.apply(
            lambda r: f"{_prompt_ctx(r)}{cfg.sep}{str(r.get(response_col) or '')}",
            axis=1,
        )

    elif cfg.mode == "template":
        out[out_col] = out.apply(
            lambda r: f"{cfg.prompt_prefix}{_prompt_ctx(r)}\n{cfg.response_prefix}{str(r.get(response_col) or '')}",
            axis=1,
        )

    elif cfg.mode == "id_prefix":
        out[out_col] = out.apply(
            lambda r: f"[PROMPT_ID={r.get(prompt_id_col)}] {str(r.get(response_col) or '')}",
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    return out
