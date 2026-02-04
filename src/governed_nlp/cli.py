# =========================
# src/governed_nlp/cli.py
# End-to-end demo
# =========================
from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from governed_nlp.config import QCConfig, TextConfig # type: ignore
from governed_nlp.data.synthetic import make_synthetic_responses # type: ignore
from governed_nlp.preprocess.qc import validate_schema, dedupe_by_response_id, add_qc_flags # type: ignore
from governed_nlp.preprocess.normalize import add_normalized_text # type: ignore
from governed_nlp.preprocess.prompt_format import build_prompt_aware_text, PromptFormatConfig # type: ignore
from governed_nlp.nlp.tokenize import tokenize_dataframe, TokenizeConfig, get_tokenizer # type: ignore
from governed_nlp.nlp.chunking import add_token_chunks, ChunkConfig # type: ignore
from governed_nlp.preprocess.split import split_leakage_safe, SplitConfig # type: ignore


def _summarize(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"rows={len(df)}")
    if "needs_review" in df.columns:
        print(f"needs_review_rate={df['needs_review'].mean():.3f}")
    for c in ["prompt_id", "human_score"]:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False).head(5)
            print(f"\nTop {c} values:\n{vc.to_string()}")


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

    args = p.parse_args()

    # -------------------------
    # Step 1: Data ingestion & provenance (synthetic demo)
    # -------------------------
    df_raw = make_synthetic_responses(n=args.n, seed=args.seed)
    print("Loaded synthetic dataset.")
    _summarize(df_raw, "Raw")

    # -------------------------
    # Step 2: QC & data validation
    # -------------------------
    qc_cfg = QCConfig()
    missing = validate_schema(df_raw, qc_cfg)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_deduped, df_dupes = dedupe_by_response_id(df_raw)
    df_qc = add_qc_flags(df_deduped, qc_cfg)

    print(f"\nQC complete: raw={len(df_raw)} deduped={len(df_deduped)} dupes_removed={len(df_dupes)}")
    _summarize(df_qc, "After QC")

    # -------------------------
    # Step 3: Light normalization
    # -------------------------
    df_norm = add_normalized_text(df_qc, col="response_text")
    print("\nNormalization complete.")
    # Optional small diagnostic
    if "response_text_raw" in df_norm.columns and "response_text_norm" in df_norm.columns:
        changed_rate = (df_norm["response_text_raw"].fillna("") != df_norm["response_text_norm"].fillna("")).mean()
        print(f"norm_changed_rate={changed_rate:.3f}")

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
    print("\nPrompt-aware formatting complete.")
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

    print("\nTokenization complete.")
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
        print("\nChunking complete.")
        print("Example num_chunks:", int(df_ready["num_chunks"].iloc[0]))
    else:
        df_ready = df_tok
        print("\nChunking skipped (use --do_chunking to enable).")

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

    print("\nLeakage-safe split complete.")
    _summarize(train_df, "Train")
    _summarize(val_df, "Validation")
    _summarize(test_df, "Test")

    # A few rows for inspection
    cols_show = [c for c in ["response_id", "prompt_id", "human_score", "needs_review", "model_text"] if c in train_df.columns]
    print("\nSample rows (train):")
    print(train_df[cols_show].head(5).to_string(index=False))

    print("\nRun configs:")
    print("QCConfig:", qc_cfg)
    print("PromptFormatConfig:", pf_cfg)
    print("TokenizeConfig:", tok_cfg)
    if args.do_chunking:
        print("ChunkConfig:", ch_cfg)
    print("SplitConfig:", sp_cfg)


if __name__ == "__main__":
    main()
