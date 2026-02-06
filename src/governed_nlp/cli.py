# =========================
# src/governed_nlp/cli.py
# End-to-end demo
# =========================
from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd # type: ignore

from governed_nlp.config import QCConfig, TextConfig # type: ignore
from governed_nlp.data.synthetic import make_synthetic_responses # type: ignore
from governed_nlp.preprocess.qc import validate_schema, dedupe_by_response_id, add_qc_flags # type: ignore
from governed_nlp.preprocess.normalize import add_normalized_text # type: ignore
from governed_nlp.preprocess.prompt_format import build_prompt_aware_text, PromptFormatConfig # type: ignore
from governed_nlp.nlp.tokenize import tokenize_dataframe, TokenizeConfig, get_tokenizer # type: ignore
from governed_nlp.nlp.chunking import add_token_chunks, ChunkConfig # type: ignore
from governed_nlp.preprocess.split import split_leakage_safe, SplitConfig # type: ignore


def _top_counts(df: pd.DataFrame, col: str, n: int = 3) -> str:
    if col not in df.columns:
        return ""
    vc = df[col].value_counts(dropna=False).head(n)
    return "; ".join([f"{idx}:{int(val)}" for idx, val in vc.items()])


def _print_table(df: pd.DataFrame, title: str, max_rows: int = 10) -> None:
    print(f"\n{title}")
    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", 20,
        "display.width", 120,
    ):
        print(df.head(max_rows).to_string(index=False))


def _summarize(df: pd.DataFrame, name: str, quiet: bool = False) -> None:
    rows = len(df)
    needs = df["needs_review"].mean() if "needs_review" in df.columns else None
    prompts = _top_counts(df, "prompt_id", n=3)
    scores = _top_counts(df, "human_score", n=3)

    line = f"{name}: rows={rows}"
    if needs is not None:
        line += f", needs_review={needs:.1%}"
    if prompts:
        line += f", prompts(top3)={prompts}"
    if (not quiet) and scores:
        line += f", scores(top3)={scores}"
    print(line)


def main():
    p = argparse.ArgumentParser(
        description="Governed NLP Pipeline (Steps 1–7): ingestion → QC → normalization → prompt-aware text "
                    "→ tokenization → (optional) chunking → leakage-safe split"
    )
    p.add_argument("--n", type=int, default=80, help="Number of synthetic rows")
    p.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")

    # Step 5 tokenizer
    p.add_argument("--hf_model", type=str, default="roberta-base", help="HF model name for tokenizer")
    p.add_argument("--max_length", type=int, default=256, help="Max token length for tokenizer")
    p.add_argument("--padding", type=str, default="max_length", choices=["max_length", "longest"])

    # Step 4 prompt formatting
    p.add_argument("--prompt_mode", type=str, default="concat", choices=["concat", "template", "id_prefix"])
    p.add_argument("--use_prompt_text", action="store_true", help="Include prompt_text (if available) in model_text")

    # Step 6 chunking (optional)
    p.add_argument("--do_chunking", action="store_true", help="If set, chunk token sequences after tokenization")
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--chunk_stride", type=int, default=128)

    # Step 7 splitting
    p.add_argument("--split_mode", type=str, default="by_prompt", choices=["by_prompt", "by_group", "time"])
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)

    # Step 8 evaluation
    p.add_argument("--demo_eval", action="store_true",
               help="Run demo evaluation metrics and slice diagnostics (no training)")

    # CLI flags: --quiet and --verbose
    p.add_argument("--quiet", action="store_true", help="Minimal output (interview-friendly)")
    p.add_argument("--verbose", action="store_true", help="Verbose output (debugging)")

    args = p.parse_args()
    quiet = bool(args.quiet)
    verbose = bool(args.verbose) and not quiet

    # -------------------------
    # Step 1: Data ingestion & provenance (synthetic demo)
    # -------------------------
    df_raw = make_synthetic_responses(n=args.n, seed=args.seed)
    print("Loaded synthetic dataset.")
    _summarize(df_raw, "Raw", quiet=quiet)

    # -------------------------
    # Step 2: QC & data validation
    # -------------------------
    qc_cfg = QCConfig()
    missing = validate_schema(df_raw, qc_cfg)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_deduped, df_dupes = dedupe_by_response_id(df_raw)
    df_qc = add_qc_flags(df_deduped, qc_cfg)
    print(f"QC: raw={len(df_raw)} deduped={len(df_deduped)} dupes_removed={len(df_dupes)}")
    _summarize(df_qc, "After QC", quiet=quiet)

    # -------------------------
    # Step 3: Light normalization
    # -------------------------
    df_norm = add_normalized_text(df_qc, col="response_text")
    if "response_text_raw" in df_norm.columns and "response_text_norm" in df_norm.columns:
        changed_rate = (df_norm["response_text_raw"].fillna("") != df_norm["response_text_norm"].fillna("")).mean()
        print(f"Normalization: norm_changed_rate={changed_rate:.3f}")
    else:
        print("Normalization complete.")

    # -------------------------
    # Step 4: Prompt-aware input builder
    # -------------------------
    pf_cfg = PromptFormatConfig(
        mode=args.prompt_mode,
        include_prompt_text=args.use_prompt_text,
    )
    df_model = build_prompt_aware_text(
        df_norm,
        cfg=pf_cfg,
        response_col="response_text_norm",
        out_col="model_text",
    )
    print("Prompt-aware formatting complete.")
    if verbose:
        print("Example model_text:", (df_model["model_text"].iloc[0] or "")[:160], "...")

    # -------------------------
    # Step 5: Tokenization (Hugging Face)
    # -------------------------
    tok_cfg = TokenizeConfig(
        model_name=args.hf_model,
        max_length=args.max_length,
        padding=args.padding,
        truncation=True,
    )
    df_tok = tokenize_dataframe(df_model, cfg=tok_cfg, text_col="model_text")

    # determine pad token id for chunking (model-dependent)
    tokenizer = get_tokenizer(tok_cfg)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print(f"Tokenization complete. max_length={args.max_length}")
    if verbose:
        print("Example token length:", len(df_tok["tok_input_ids"].iloc[0]))

    # -------------------------
    # Step 6: Chunking (optional)
    # -------------------------
    if args.do_chunking:
        ch_cfg = ChunkConfig(
            chunk_size=args.chunk_size,
            stride=args.chunk_stride,
            pad_to_chunk=True,
            pad_token_id=pad_token_id,
        )
        df_ready = add_token_chunks(df_tok, cfg=ch_cfg)
        print(f"Chunking enabled. chunk_size={args.chunk_size} stride={args.chunk_stride} "
              f"example_num_chunks={int(df_ready['num_chunks'].iloc[0])}")
    else:
        df_ready = df_tok
        print("Chunking skipped.")

    # -------------------------
    # Step 7: Leakage-safe split
    # -------------------------
    sp_cfg = SplitConfig(
        mode=args.split_mode,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        prompt_col="prompt_id",
        group_col="student_id",
        time_col="created_ts",
    )
    train_df, val_df, test_df = split_leakage_safe(df_ready, cfg=sp_cfg)

    print("Leakage-safe split complete.")
    _summarize(train_df, "Train", quiet=quiet)
    _summarize(val_df, "Validation", quiet=quiet)
    _summarize(test_df, "Test", quiet=quiet)

    if verbose:
        cols_show = [c for c in ["response_id", "prompt_id", "human_score", "needs_review", "model_text"] if c in train_df.columns]
        _print_table(train_df[cols_show], "Sample rows (train):", max_rows=5)

        print("\nRun configs:")
        print("QCConfig:", qc_cfg)
        print("PromptFormatConfig:", pf_cfg)
        print("TokenizeConfig:", tok_cfg)
        if args.do_chunking:
            print("ChunkConfig:", ch_cfg)
        print("SplitConfig:", sp_cfg)

    # -------------------------
    # Step 8: Modeling & Evaluation (demo-only)
    # -------------------------
    if args.demo_eval:
        from governed_nlp.modeling.evaluate import evaluate_by_slice, make_length_bins  # type: ignore
        import numpy as np  # type: ignore

        # Create synthetic predictions (demo-only)
        rng = np.random.default_rng(args.seed)
        test_df_eval = test_df.copy()

        test_df_eval["y_true"] = test_df_eval["human_score"].astype(int)
        noise = rng.integers(-1, 2, size=len(test_df_eval))  # -1,0,1
        test_df_eval["y_pred"] = (test_df_eval["y_true"] + noise).clip(0, 4)

        # Slice by length bins (practical robustness check)
        text_for_len = (
            test_df_eval["response_text_norm"]
            if "response_text_norm" in test_df_eval.columns
            else test_df_eval["model_text"]
        )
        test_df_eval["len_bin"] = make_length_bins(text_for_len)

        by_prompt = evaluate_by_slice(
            test_df_eval, "y_true", "y_pred", slice_col="prompt_id"
        )
        by_len = evaluate_by_slice(
            test_df_eval, "y_true", "y_pred", slice_col="len_bin"
        )

        print("\nDemo Eval (synthetic preds):")
        _print_table(by_prompt, "By prompt_id:", max_rows=10) # type: ignore
        _print_table(by_len, "By length bin:", max_rows=10) # type: ignore


if __name__ == "__main__":
    main()
