"""Small JSON artifact/trace boundary adapted from CineCrew agent_trace.py.

Upstream: https://github.com/Ironieser/CineCrew/blob/9a00efa7db65ba028c1523907f6cee844776123a/src/utils/agent_trace.py
License: Apache-2.0.  Modified to remove CineCrew configuration imports and
Markdown rendering; V9-G keeps only the append-only artifact semantics.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ArtifactTrace:
    """Append-only JSONL trace with one stable directory per generation unit."""

    def __init__(self, output_dir: Path, trace_id: str) -> None:
        self.output_dir = Path(output_dir) / "traces" / str(trace_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"

    def append(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": str(stage),
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def write_artifact(self, name: str, value: Any) -> Path:
        path = self.output_dir / str(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            text = str(value)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        return path
