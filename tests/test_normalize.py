import pandas as pd # type: ignore

from governed_nlp.preprocess.normalize import light_normalize_text, add_normalized_text # type: ignore


def test_light_normalize_handles_none():
    assert light_normalize_text(None) == ""


def test_light_normalize_collapses_whitespace():
    s = "This   is \n a   test."
    out = light_normalize_text(s)
    assert out == "This is a test."


def test_light_normalize_removes_zero_width_chars():
    # \u200b = zero-width space
    s = "Hello\u200bWorld"
    out = light_normalize_text(s)
    assert out == "HelloWorld"


def test_light_normalize_preserves_punctuation_and_case():
    s = "I Don't like it!!!"
    out = light_normalize_text(s)
    # No lowercasing, no punctuation stripping
    assert out == "I Don't like it!!!"


def test_add_normalized_text_creates_raw_and_norm_columns():
    df = pd.DataFrame(
        {
            "response_id": ["R1", "R2"],
            "response_text": ["Hello\u200b  World", None],
        }
    )
    out = add_normalized_text(df, col="response_text")
    assert "response_text_raw" in out.columns
    assert "response_text_norm" in out.columns
    assert out.loc[0, "response_text_norm"] == "Hello World"
    assert out.loc[1, "response_text_norm"] == ""
