"""Streaming JSONL dataset loader."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from llm_evals.datasets.sample import Sample
from llm_evals.exceptions import DatasetError


def load_jsonl(path: str | Path) -> Iterator[Sample]:
    """Stream :class:`Sample` instances from a JSONL file.

    Each line of the file must contain one JSON object with at least an
    ``input`` field. Optional fields are ``expected`` (string), ``context``
    (list of strings), and ``metadata`` (object). Blank lines are skipped.

    :param path: Filesystem path to the JSONL file.
    :type path: str | pathlib.Path
    :returns: Iterator yielding parsed samples in file order.
    :rtype: collections.abc.Iterator[Sample]
    :raises llm_evals.exceptions.DatasetError: When a line is not valid JSON
        or is missing the required ``input`` field.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as stream:
        for lineno, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON at {file_path}:{lineno}: {exc.msg}") from exc
            if "input" not in obj:
                raise DatasetError(f"Missing required field 'input' at {file_path}:{lineno}")
            context_raw = obj.get("context", [])
            if not isinstance(context_raw, list):
                raise DatasetError(
                    f"Field 'context' at {file_path}:{lineno} must be a list of strings"
                )
            yield Sample(
                input=obj["input"],
                expected=obj.get("expected"),
                context=tuple(context_raw),
                metadata=obj.get("metadata", {}),
            )
