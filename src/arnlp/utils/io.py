"""JSONL helpers used by the CLI scripts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(
    path: str | Path,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> Iterator[dict]:
    """Stream a JSONL file. ``offset`` skips lines, ``limit`` caps the count."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit is not None and i >= offset + limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Atomic JSONL write — temp file + rename so partial files never appear."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp_name, path)
    except Exception:
        os.unlink(tmp_name)
        raise
    return n
