"""Stage 1 + Stage 3 — text cleaning.

* :mod:`llm_word_fixer` — LLM-based word-boundary repair (Stage 1)
* :mod:`al_prefix_fix`  — regex cleanup of detached "ال" (Stage 3a, optional)
* :mod:`word_splitter`  — fine-tuned AraBERT splitter inference (Stage 3b, optional)
"""
