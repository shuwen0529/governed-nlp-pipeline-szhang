Governed NLP Pipeline
====================

Overview
--------
This repository demonstrates a production-minded NLP preprocessing and data
preparation pipeline designed for constrained, decision-oriented NLP tasks
(e.g., classification or ranking aligned to a rubric).

The focus is not on model novelty, but on building a stable, auditable, and
scalable foundation that turns unstructured text into decision-ready inputs.
The design mirrors real-world commercial and regulated analytics workflows.

This repo is intentionally lightweight and modular, suitable for batch scoring,
offline training, and integration into downstream systems.


Use Case
--------
Originally inspired by automated scoring of open-ended text, the pipeline maps
directly to commercial NLP use cases such as:
- Standardizing insights from call notes or CRM narratives
- Extracting consistent engagement or quality signals from free text
- Supporting prioritization and review workflows with confidence scoring

The key design principle is governance-first AI: constrain the problem,
evaluate rigorously, and integrate outputs responsibly.


Pipeline Stages
---------------

Step 0: Decision Framing & Data Contract
- Align on the decision the model supports
- Define one row = one decision unit
- Ensure outputs are comparable, auditable, and usable downstream

Step 1: Data Ingestion & Provenance
- Validate required identifiers and timestamps
- Preserve lineage for reproducibility and auditability

Step 2: QC & Data Validation (Guardrails)
- Schema validation and required field checks
- De-duplication by response ID
- Flags for blank text, non-language artifacts, extreme length
- Issues are flagged, not silently dropped

Step 3: Light Normalization
- Unicode normalization and encoding cleanup
- Removal of non-printable characters
- Whitespace standardization
- Aggressive cleaning (spell correction, lemmatization) is intentionally avoided

Step 4: Prompt-Aware Input Construction
- Combine response text with prompt or contextual metadata
- Supports multiple formats (concat, template, ID-based)
- Ensures semantic interpretation is context-aware

Step 5: Tokenization
- Hugging Face tokenizer abstraction
- Configurable model, padding, and truncation strategy
- Produces model-ready token sequences

Step 6: Chunking (Optional)
- Token-level chunking for long text
- Supports chunk-and-pool inference strategies
- Designed to balance performance and operational simplicity

Step 7: Leakage-Safe Splitting
- Prompt-level splits to avoid context leakage
- Group-level splits (e.g., entity or user)
- Time-based splits for forecasting or longitudinal analysis

Step 8: Modeling & Evaluation (Practical, Decision-Aligned)
- Transformer fine-tuning (BERT/RoBERTa) for rubric-aligned classification
- Decision-aligned metrics:
  * Weighted kappa (agreement with human judgment)
  * Adjacent accuracy (+/-1 tolerance, mirrors rubric boundary decisions)
- Slice-based evaluation for robustness/fairness diagnostics (e.g., length bins, prompt-level)
- Explainability positioned as diagnostics for trust (example-based review, sanity checks)

Design Principles
-----------------
- Constrain NLP tasks to reduce risk and increase trust
- Prefer stability and interpretability over marginal accuracy gains
- Treat explainability as diagnostics for trust, not full transparency
- Design for batch workflows aligned with business cadence
- Embed governance and auditability from the start


Repository Structure
--------------------
<pre>
src/
└── governed_nlp/
    ├── data/          # Synthetic data generation and IO
    ├── preprocess/    # QC, normalization, prompt formatting, splitting
    ├── nlp/           # Tokenization and chunking utilities
    ├── modeling/      # Modeling & Evaluation
    └── cli.py         # End-to-end demo runner
tests/
  Unit tests for QC, formatting, and splitting

notebooks/
  End-to-end exploratory demo (optional)
</pre>


Quick Start
-----------
1) Install dependencies
   pip install -r requirements.txt

2) Run the end-to-end pipeline on synthetic data
   python -m governed_nlp.cli --n 50

3) Run tests
   pytest -q


What This Repo Is (and Is Not)
------------------------------
This repo:
- Demonstrates how to prepare unstructured text for governed NLP systems
- Shows engineering judgment and production readiness
- Is suitable as a blueprint for commercial or regulated NLP pipelines

This repo is not:
- A full training or deployment system
- A model benchmark or leaderboard project
- An attempt to maximize model performance at all costs


Author
------
Shuwen Zhang, Ph.D.
Senior Data Scientist

This repository was created as an example of how I design
end-to-end NLP pipelines that balance technical rigor, governance, and
business impact.
