"""Run manifest and content-addressed resume helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.agentic_video.recipe_v2 import sha256_file


def json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_sha(root: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


class RunManifest:
    def __init__(self, path: Path, *, command: str, input_path: Path | None,
                 config: dict, repo_root: Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "command": command,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "git_sha": git_sha(repo_root),
                "config_sha256": json_hash(config),
                "input": ({"path": str(input_path), "sha256": sha256_file(input_path)}
                          if input_path and input_path.exists() else None),
                "stages": {},
            }
            self.save()

    def stage(self, name: str, status: str, **details) -> None:
        self.data["stages"][name] = {
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **details,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(self.path)
