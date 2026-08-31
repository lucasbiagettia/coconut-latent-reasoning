"""Small, dependency-free support for an optional Hugging Face token in .env."""

from __future__ import annotations

import os
from pathlib import Path


def load_huggingface_token(dotenv_path: str | Path = ".env") -> str | None:
    """Return ``HF_TOKEN`` from the environment or a simple local ``.env`` file.

    Existing process environment variables take precedence.  Only ``HF_TOKEN`` is
    read from the file, and its value is never logged.
    """

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    path = Path(dotenv_path)
    if not path.is_file():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() != "HF_TOKEN":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None
    return None
