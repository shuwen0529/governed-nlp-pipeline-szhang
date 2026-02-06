from __future__ import annotations
import pandas as pd # type: ignore
from governed_nlp.modeling.metrics import weighted_kappa, adjacent_accuracy, strict_accuracy # type: ignore


def make_length_bins(text_series: pd.Series, bins=(0, 50, 150, 300, 10000)) -> pd.Series:
    """
    Creates length bins from character counts (simple, robust slice).
    """
    lengths = text_series.fillna("").astype(str).str.len()
    return pd.cut(lengths, bins=bins, include_lowest=True)


def evaluate_by_slice(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    slice_col: str,
) -> pd.DataFrame:
    """
    Slice-level evaluation for robustness/fairness diagnostics.
    """
    rows = []
    for k, g in df.groupby(slice_col, dropna=False, observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                slice_col: k,
                "n": len(g),
                "strict_acc": strict_accuracy(g[y_true_col], g[y_pred_col]),
                "adjacent_acc": adjacent_accuracy(g[y_true_col], g[y_pred_col]),
                "weighted_kappa": weighted_kappa(g[y_true_col], g[y_pred_col]),
            }
        )
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out
