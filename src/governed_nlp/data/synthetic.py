import numpy as np
import pandas as pd

def make_synthetic_responses(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prompt_ids = ["P1", "P2", "P3"]
    prompt_texts = {
        "P1": "Should schools require uniforms? Explain your reasoning.",
        "P2": "What are the benefits and risks of social media?",
        "P3": "Describe a challenge you overcame and what you learned.",
    }

    base_texts = [
        "I think uniforms can reduce distraction and make school safer because everyone looks similar.",
        "Social media helps people connect, but it can also increase anxiety and spread misinformation.",
        "A challenge I overcame was learning English. I practiced every day and improved a lot.",
        "Uniforms limit self expression, but they can reduce bullying based on clothing brands.",
        "Social media is useful for learning and networking, but people should limit screen time.",
    ]

    rows = []
    for i in range(n):
        rid = f"R{i:05d}"
        pid = rng.choice(prompt_ids)
        txt = rng.choice(base_texts)

        # Inject some QC edge cases
        p = rng.random()
        if p < 0.08:
            txt = "   "
        elif p < 0.14:
            txt = "!!!???!??"
        elif p < 0.20:
            txt = "ok"
        elif p < 0.26:
            txt = "A" * int(rng.integers(1200, 3000))
        elif p < 0.30:
            txt = "I think uniforms are good good good good good."  # repetition-ish

        rows.append({
            "response_id": rid,
            "prompt_id": pid,
            "prompt_text": prompt_texts[pid],
            "response_text": txt,
            "human_score": int(rng.integers(0, 5)),
            "student_id": f"S{int(rng.integers(1, 10)):03d}",
            "created_ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 40))),
        })

    df = pd.DataFrame(rows)

    # Inject a duplicate response_id (simulate bad ingest)
    if n >= 5:
        df.loc[3, "response_id"] = df.loc[1, "response_id"]

    # Inject a missing required value
    if n >= 8:
        df.loc[7, "prompt_id"] = None

    return df