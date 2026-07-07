"""Artifact helpers for KernelBench generation."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation.metrics import parse_eval_payload


def write_generation_config(path: str | Path, config: Any) -> None:
    _write_json(Path(path), _jsonable(config))


def write_round_artifacts(
    round_dir: str | Path,
    *,
    problem_id: int,
    problem_name: str,
    error: str,
    eval_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(round_dir)
    meta: dict[str, Any] = {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "has_prompt": (path / "prompt.txt").is_file(),
        "has_output_model_new": (path / "output_model_new.py").is_file(),
        "stage": "generated",
        "error": error,
    }

    if eval_payload is not None:
        parsed = parse_eval_payload(eval_payload)
        metadata = parsed.metadata
        if parsed.compiled and parsed.correctness:
            eval_error = "OK: compiled=True, correctness=True"
        elif not parsed.compiled:
            eval_error = f"compiled=False. {json.dumps(metadata, default=str)}"
        else:
            eval_error = (
                f"compiled=True, correctness=False. {json.dumps(metadata, default=str)}"
            )

        meta.update(
            {
                "stage": "speedup_eval",
                "compiled": parsed.compiled,
                "correctness": parsed.correctness,
                "speedup": parsed.speedup,
                "ref_ms": parsed.ref_runtime,
                "new_ms": parsed.runtime,
                "error": eval_error,
                "eval_metadata": metadata,
                "eval_payload": eval_payload,
            }
        )

    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "meta.json", meta)
    (path / "error.txt").write_text(meta["error"] + "\n", encoding="utf-8")
    return meta


def write_problem_meta(
    problem_dir: str | Path,
    *,
    problem_id: int,
    problem_name: str,
    round_metas: list[dict[str, Any]],
) -> None:
    if round_metas:
        best = next(
            (meta for meta in round_metas if meta.get("correctness")),
            round_metas[-1],
        )
        meta = dict(best)
        meta["rounds"] = [
            {
                "round": idx,
                "compiled": item.get("compiled"),
                "correctness": item.get("correctness"),
                "speedup": item.get("speedup"),
            }
            for idx, item in enumerate(round_metas)
        ]
    else:
        meta = {
            "problem_id": problem_id,
            "problem_name": problem_name,
            "stage": "generated",
            "error": "no rounds completed",
            "rounds": [],
        }

    path = Path(problem_dir)
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "meta.json", meta)
    (path / "error.txt").write_text(
        str(meta.get("error", "")) + "\n",
        encoding="utf-8",
    )


def sorted_prefixed_dirs(directory: str | Path, prefix: str) -> list[Path]:
    base = Path(directory)
    if not base.is_dir():
        return []

    def suffix_index(path: Path) -> int:
        try:
            return int(path.name.removeprefix(prefix))
        except ValueError:
            return -1

    return sorted(
        [
            path
            for path in base.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ],
        key=suffix_index,
    )

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
