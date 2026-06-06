#!/usr/bin/env python3
"""
arabert_word_splitter.py
Arabic merged-word segmentation via AraBERT character-boundary prediction.

Problem formulation
-------------------
Turn word segmentation into a binary sequence-labeling problem:
  "Should there be a space AFTER this character? → 1 (yes) or 0 (no)"

Example
-------
  Merged text : دعتمنظمةالصحةالعالمية
  Char labels : [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0]
                        ↑       ↑       ↑
  Reconstructed: دعت منظمة الصحة العالمية

Architecture
------------
  Merged Text
       ↓
  AraBERT Encoder  (aubmindlab/bert-base-arabertv2)
       ↓
  Linear Layer  (768 → 2)
       ↓
  Space / No-Space prediction  (per subword token)
       ↓
  Align to characters  (via offset_mapping)
       ↓
  Reconstruct sentence

Data generation (automatic, no manual annotation needed)
---------------------------------------------------------
  Clean text  "دعت منظمة الصحة العالمية"
        ↓  remove spaces
  Merged text "دعتمنظمةالصحةالعالمية"
        ↓  original word lengths determine label positions
  Labels      [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0]

Usage
-----
  # 1. Generate training data
  python arabert_word_splitter.py generate

  # 2. Train the model
  python arabert_word_splitter.py train

  # 3. Run inference
  python arabert_word_splitter.py predict "دعتمنظمةالصحةالعالمية"

  # 4. Full pipeline (generate → train → evaluate)
  python arabert_word_splitter.py all
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME   = "aubmindlab/bert-base-arabertv2"
PROJECT_DIR  = Path(__file__).parent.parent
DATA_DIR     = PROJECT_DIR / "data"
MODEL_DIR    = PROJECT_DIR / "models" / "arabert_splitter"

CORPUS_PATH  = DATA_DIR / "processed" / "preprocessed_v2.jsonl"
TRAIN_PATH   = DATA_DIR / "model_training" / "wb_train.jsonl"
VAL_PATH     = DATA_DIR / "model_training" / "wb_val.jsonl"
TEST_PATH    = DATA_DIR / "model_training" / "wb_test.jsonl"

MAX_LEN         = 128    # max subword tokens per example
MIN_WORDS       = 3      # minimum words per training sentence
MAX_WORDS       = 25     # maximum words per training sentence (controls length)
TRAIN_SPLIT     = 0.90
VAL_SPLIT       = 0.05
# remaining 0.05 → test

BATCH_SIZE      = 32
EPOCHS          = 3
LR              = 2e-5
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ARABIC = re.compile(r"[ء-ي]+")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _sentence_to_example(words: list[str]) -> dict | None:
    """
    Convert a list of clean Arabic words into a (merged_text, char_labels) pair.

    char_labels[i] = 1  ← space should follow character i
                   = 0  ← no space after character i
    """
    if len(words) < MIN_WORDS:
        return None

    # only keep pure Arabic tokens
    words = [w for w in words if _ARABIC.fullmatch(w) and len(w) >= 2]
    if len(words) < MIN_WORDS:
        return None
    words = words[:MAX_WORDS]

    merged = "".join(words)
    labels = [0] * len(merged)

    pos = 0
    for i, w in enumerate(words):
        pos += len(w)
        if i < len(words) - 1:        # space after every word except the last
            labels[pos - 1] = 1

    return {"text": merged, "labels": labels}


def generate_training_data(
    corpus_path: str = str(CORPUS_PATH),
    max_examples: int = 500_000,
) -> None:
    """
    Stage: Data generation.

    Reads clean article bodies from corpus_path, splits each into
    sentence-sized chunks (MIN_WORDS … MAX_WORDS), removes spaces,
    generates char-level labels, and writes train/val/test JSONL files.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    print(f"Reading corpus: {corpus_path}")

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if len(examples) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            start = line.find("{")
            if start == -1:
                continue
            try:
                record = json.loads(line[start:])
            except json.JSONDecodeError:
                continue

            # use filtered_tokens (already clean, stopwords removed)
            tokens: list[str] = record.get("filtered_tokens", [])
            if not tokens:
                # fallback: split normalized_text
                tokens = record.get("normalized_text", "").split()

            # sliding window over tokens to create chunks
            step = MAX_WORDS // 2
            for start_i in range(0, max(1, len(tokens) - MIN_WORDS + 1), step):
                chunk = tokens[start_i: start_i + MAX_WORDS]
                ex = _sentence_to_example(chunk)
                if ex is not None:
                    examples.append(ex)
                if len(examples) >= max_examples:
                    break

    print(f"Generated {len(examples):,} examples")

    # shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(examples)

    n        = len(examples)
    n_train  = int(n * TRAIN_SPLIT)
    n_val    = int(n * VAL_SPLIT)

    splits = {
        str(TRAIN_PATH): examples[:n_train],
        str(VAL_PATH):   examples[n_train: n_train + n_val],
        str(TEST_PATH):  examples[n_train + n_val:],
    }

    for path, split_examples in splits.items():
        with open(path, "w", encoding="utf-8") as f:
            for ex in split_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved {len(split_examples):,} examples → {Path(path).name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════
class WordBoundaryDataset(Dataset):
    """
    Loads (merged_text, char_labels) pairs and aligns them to AraBERT subword
    tokens using offset_mapping.

    Token-level label assignment (MAX rule):
      For subword token covering chars [start, end):
        token_label = max(char_labels[start:end])
      This correctly handles tokens that span a word boundary.

    Special tokens ([CLS], [SEP], [PAD]) → label = -100 (ignored in loss).
    """

    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.examples: list[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        text        = ex["text"]
        char_labels = ex["labels"]

        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = encoding["offset_mapping"][0].tolist()
        token_labels   = []

        for char_start, char_end in offset_mapping:
            if char_start == char_end:          # special token
                token_labels.append(-100)
            else:
                # MAX over all chars in this subword's span
                span_end = min(char_end, len(char_labels))
                if char_start >= len(char_labels):
                    token_labels.append(0)
                else:
                    token_labels.append(max(char_labels[char_start:span_end]))

        return {
            "input_ids":      encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels":         torch.tensor(token_labels, dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _compute_class_weights(train_path: str) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from class distribution."""
    n0 = n1 = 0
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            labels = json.loads(line)["labels"]
            n1 += sum(labels)
            n0 += len(labels) - sum(labels)
    ratio = n0 / max(n1, 1)
    print(f"Class ratio (0:1) = {ratio:.1f}:1  →  class weights [1.0, {ratio:.1f}]")
    return torch.tensor([1.0, ratio], dtype=torch.float).to(DEVICE)


def train_model() -> None:
    """
    Fine-tune AraBERT for word-boundary token classification.

    Loss  : CrossEntropyLoss with class weights (0s >> 1s in Arabic text)
    Optim : AdamW with linear warmup + decay
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets …")
    train_ds = WordBoundaryDataset(str(TRAIN_PATH), tokenizer)
    val_ds   = WordBoundaryDataset(str(VAL_PATH),   tokenizer)
    print(f"  Train: {len(train_ds):,}   Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True,
    ).to(DEVICE)

    class_weights = _compute_class_weights(str(TRAIN_PATH))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── training ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits

            # reshape for CrossEntropyLoss: (N*L, C) vs (N*L,)
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 200 == 0 or step == len(train_loader):
                avg = total_loss / step
                print(f"  Epoch {epoch}/{EPOCHS}  Step {step}/{len(train_loader)}"
                      f"  Loss={avg:.4f}")

        # ── validation ───────────────────────────────────────────────────────
        f1 = evaluate(model, val_loader, split="val")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(MODEL_DIR))
            print(f"  ✓ Best model saved (F1={f1:.4f})")

    print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")
    print(f"Model saved → {MODEL_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: AutoModelForTokenClassification,
    loader: DataLoader,
    split: str = "val",
) -> float:
    """Return macro-F1 on class 1 (space boundary). Print full report."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"]

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            preds  = logits.argmax(dim=-1).cpu()

            # only count non-ignored positions
            mask = labels != -100
            all_preds.extend(preds[mask].tolist())
            all_labels.extend(labels[mask].tolist())

    report = classification_report(
        all_labels, all_preds,
        target_names=["no-space (0)", "space (1)"],
        digits=4,
    )
    print(f"\n── {split.upper()} evaluation ──")
    print(report)

    return f1_score(all_labels, all_preds, pos_label=1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

class WordBoundarySplitter:
    """
    Production-ready inference wrapper.

    Usage
    -----
    splitter = WordBoundarySplitter.load()
    fixed = splitter.fix("دعتمنظمةالصحةالعالمية")
    # → "دعت منظمة الصحة العالمية"

    The splitter operates token-by-token on the input text:
    - Short tokens (≤ min_token_len chars) are kept unchanged.
    - Long candidate tokens are fed through AraBERT and split where
      the model predicts a space boundary.
    """

    def __init__(
        self,
        model: AutoModelForTokenClassification,
        tokenizer: AutoTokenizer,
        min_token_len: int = 7,
        threshold: float = 0.5,
    ):
        self.model         = model.eval().to(DEVICE)
        self.tokenizer     = tokenizer
        self.min_token_len = min_token_len
        self.threshold     = threshold

    @classmethod
    def load(cls, model_dir: str = str(MODEL_DIR), **kwargs) -> "WordBoundarySplitter":
        print(f"Loading model from {model_dir} …")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model     = AutoModelForTokenClassification.from_pretrained(model_dir)
        return cls(model, tokenizer, **kwargs)

    def _predict_char_labels(self, text: str) -> list[int]:
        """
        Run the model on `text` and return char-level predictions.

        Returns a list of length len(text) where each value is 0 or 1.

        Boundary alignment via majority voting
        --------------------------------------
        Training assigns a label to each subword token using MAX over the
        character span: if ANY character in [char_start, char_end) is a true
        boundary, the token is labeled 1.  During inference the model predicts
        a boundary probability for the whole span, but we do not know which
        character inside the span is the actual boundary.

        The old approach (char_end - 1) assumed the boundary always falls on
        the last character of the span, which is wrong whenever a subword
        crosses a word boundary.

        Fix — two-phase majority voting:
          Phase 1  Each character inherits the full boundary probability of
                   the subword token that covers it.  In non-overlapping
                   tokenisation every character is covered by exactly one
                   token, so char_votes[p] = prob of that token.
          Phase 2  Threshold the per-character votes.  Within each consecutive
                   run of above-threshold characters (which corresponds to one
                   boundary span) mark only the FIRST character.  This avoids
                   the systematic end-of-span bias of char_end - 1 and places
                   the space marker at the earliest position in the span,
                   consistent with how Arabic word-final characters appear at
                   the start of cross-boundary subwords.
        """
        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = self.model(input_ids=input_ids,
                                attention_mask=attention_mask).logits[0]
            probs  = torch.softmax(logits, dim=-1)[:, 1]   # P(space)

        # ── Phase 1: majority vote ────────────────────────────────────────────
        # Each character accumulates the boundary probability of its subword.
        # We use max() so that if two windows overlap (e.g. during windowed
        # inference), a character takes the higher-confidence prediction.
        char_votes = [0.0] * len(text)
        for i, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == char_end:          # special / padding token
                continue
            prob = probs[i].item()
            for p in range(char_start, min(char_end, len(text))):
                if prob > char_votes[p]:
                    char_votes[p] = prob

        # ── Phase 2: threshold + first-char-of-run selection ─────────────────
        # Within each contiguous run of chars whose vote >= threshold, mark
        # only the first one.  This produces exactly one space per predicted
        # word boundary regardless of span width.
        char_labels = [0] * len(text)
        in_boundary_run = False
        for p, vote in enumerate(char_votes):
            if vote >= self.threshold:
                if not in_boundary_run:
                    char_labels[p] = 1      # first char of this boundary span
                    in_boundary_run = True
            else:
                in_boundary_run = False

        return char_labels

    def _reconstruct(self, text: str, char_labels: list[int]) -> str:
        """Insert spaces into `text` at positions where char_labels == 1."""
        result = []
        for char, label in zip(text, char_labels):
            result.append(char)
            if label == 1:
                result.append(" ")
        return "".join(result).strip()

    def split_token(self, token: str) -> str:
        """Split a single merged token using AraBERT predictions."""
        if len(token) <= self.min_token_len or not _ARABIC.fullmatch(token):
            return token
        char_labels = self._predict_char_labels(token)
        return self._reconstruct(token, char_labels)

    def fix(self, text: str) -> str:
        """
        Fix all merged tokens in a full text.

        Workflow:
        1. Split text on whitespace
        2. For each token ≥ min_token_len chars: run AraBERT boundary prediction
        3. Reassemble the text
        """
        tokens = text.split()
        result = []
        for tok in tokens:
            result.append(self.split_token(tok))
        return " ".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUICK TEST  (evaluation on held-out test set)
# ══════════════════════════════════════════════════════════════════════════════

def test_model() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)

    test_ds     = WordBoundaryDataset(str(TEST_PATH), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    evaluate(model, test_loader, split="test")

    # qualitative examples
    splitter = WordBoundarySplitter(model, tokenizer)
    examples = [
        "دعتمنظمةالصحةالعالميةالىغزة",
        "تفرضاسرائيلعليهحصارا",
        "وبينإسرائيلولبنان",
        "فيمضيقهرمز",
        "اتفاقيوقفاطلاقالنار",
    ]
    print("\n── Qualitative examples ──")
    for ex in examples:
        print(f"  Input : {ex}")
        print(f"  Output: {splitter.split_token(ex)}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AraBERT Arabic Word Boundary Splitter")
    parser.add_argument(
        "command",
        choices=["generate", "train", "test", "predict", "all"],
        help=(
            "generate : create training data from corpus\n"
            "train    : fine-tune AraBERT\n"
            "test     : evaluate on held-out test set\n"
            "predict  : split a merged text (pass as next argument)\n"
            "all      : generate → train → test"
        ),
    )
    parser.add_argument("text", nargs="?", default="",
                        help="Merged Arabic text to split (for 'predict' command)")
    parser.add_argument("--max-examples", type=int, default=500_000)
    args = parser.parse_args()

    if args.command == "generate":
        generate_training_data(max_examples=args.max_examples)

    elif args.command == "train":
        train_model()

    elif args.command == "test":
        test_model()

    elif args.command == "predict":
        if not args.text:
            print("Please provide text to split, e.g.:")
            print('  python arabert_word_splitter.py predict "دعتمنظمةالصحة"')
            sys.exit(1)
        splitter = WordBoundarySplitter.load()
        print(f"Input : {args.text}")
        print(f"Output: {splitter.fix(args.text)}")

    elif args.command == "all":
        print("═" * 60)
        print("Step 1/3 — Generate training data")
        print("═" * 60)
        generate_training_data(max_examples=args.max_examples)

        print("\n" + "═" * 60)
        print("Step 2/3 — Train model")
        print("═" * 60)
        train_model()

        print("\n" + "═" * 60)
        print("Step 3/3 — Evaluate on test set")
        print("═" * 60)
        test_model()


if __name__ == "__main__":
    main()
