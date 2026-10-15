"""Run metadata: every pipeline step records what produced its outputs (de-tai §10)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

from .paths import ROOT, RUNS


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


def sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


class Run:
    """Context manager writing runs/<step>/<timestamp>.json with params and outputs."""

    def __init__(self, step: str, **params):
        self.step, self.params, self.outputs = step, params, {}
        self.started = datetime.now(timezone.utc)

    def __enter__(self):
        self.t0 = time.time()
        return self

    def log(self, **kv):
        self.outputs.update(kv)

    def __exit__(self, exc_type, exc, tb):
        d = RUNS / self.step
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "step": self.step,
            "started_utc": self.started.isoformat(),
            "seconds": round(time.time() - self.t0, 1),
            "git": git_rev(),
            "python": platform.python_version(),
            "params": self.params,
            "outputs": self.outputs,
            "error": repr(exc) if exc else None,
        }
        path = d / f"{self.started.strftime('%Y%m%dT%H%M%S')}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return False
