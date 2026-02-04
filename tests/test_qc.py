from governed_nlp.config import QCConfig # type: ignore
from governed_nlp.data.synthetic import make_synthetic_responses # type: ignore
from governed_nlp.preprocess.qc import validate_schema, dedupe_by_response_id, add_qc_flags # type: ignore

def test_qc_pipeline_runs():
    df = make_synthetic_responses(n=20, seed=1)
    cfg = QCConfig()
    assert validate_schema(df, cfg) == []
    df2, dupes = dedupe_by_response_id(df)
    df_qc = add_qc_flags(df2, cfg)
    assert "needs_review" in df_qc.columns
    assert len(df_qc) + len(dupes) == len(df)
