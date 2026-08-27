"""Content-addressed disk cache for LLM calls.

Extraction and pairwise matching are both re-run constantly while tuning
prompts and thresholds; without a cache the same call gets paid for many times.
The key covers everything that can change the answer, so a prompt edit
correctly misses.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    def __init__(self, root: str | os.PathLike = ".factflow_cache", enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(**parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        self.hits += 1
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        tmp.replace(p)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
