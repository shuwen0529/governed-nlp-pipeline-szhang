# tests/test_prompt_format.py
import pandas as pd # pyright: ignore[reportMissingModuleSource]
from governed_nlp.preprocess.prompt_format import build_prompt_aware_text, PromptFormatConfig # type: ignore

def test_prompt_format_builds_model_text():
    df = pd.DataFrame({
        "prompt_id": ["P1"],
        "prompt_text": ["Prompt?"],
        "response_text_norm": ["My response."]
    })
    out = build_prompt_aware_text(df, PromptFormatConfig(mode="concat"), out_col="model_text")
    assert "model_text" in out.columns
    assert "Prompt?" in out.loc[0, "model_text"]
    assert "My response." in out.loc[0, "model_text"]
