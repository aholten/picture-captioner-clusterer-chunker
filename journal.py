from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class CaptionRecord(BaseModel):
    rel_path: str
    caption: str | None
    status: Literal["success", "error_corrupt", "error_api", "error_model"]


class CaptionJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, dict] = {}
        self._done: set[str] = set()
        self._lock = threading.Lock()
        self._file = None

    def load(self, retry_statuses: set[str] = frozenset()) -> dict[str, dict]:
        """Read all valid records from captions.jsonl.

        Returns {rel_path: {"caption": ..., "status": ...}}.
        Last record per rel_path wins. Paths with status in retry_statuses
        are excluded (will be re-processed this run).
        """
        self._records.clear()
        self._done.clear()
        discarded = 0

        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = CaptionRecord.model_validate_json(line)
                    except (ValidationError, json.JSONDecodeError):
                        discarded += 1
                        continue
                    self._records[record.rel_path] = {
                        "caption": record.caption,
                        "status": record.status,
                    }

        if discarded:
            logger.warning("Discarded %d malformed line(s) from %s", discarded, self.path)

        # Remove retry_statuses from skip set
        for rel_path, rec in self._records.items():
            if rec["status"] not in retry_statuses:
                self._done.add(rel_path)

        return self._records

    def is_done(self, rel_path: str) -> bool:
        return rel_path in self._done

    def write(self, rel_path: str, caption: str | None, status: str) -> None:
        record = CaptionRecord(rel_path=rel_path, caption=caption, status=status)
        line = record.model_dump_json() + "\n"

        with self._lock:
            if self._file is None:
                self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
            self._file.write(line)
            self._file.flush()
            os.fsync(self._file.fileno())
            self._done.add(rel_path)

    def summary(self) -> str:
        counts = Counter(rec["status"] for rec in self._records.values())
        total = sum(counts.values())
        success = counts.get("success", 0)
        errors = total - success
        return f"Skipping {total} already processed ({success} success, {errors} errors)"

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
