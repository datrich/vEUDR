"""Config + secrets loading."""
from __future__ import annotations

import os
from functools import lru_cache

import yaml
from dotenv import load_dotenv

from .paths import CONFIGS, ROOT

load_dotenv(ROOT / ".env")


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env(key: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing {key}: set it in LLM_4_deforest/.env (see .env.example)")
    return val
