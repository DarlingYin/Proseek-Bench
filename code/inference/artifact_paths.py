"""Output paths and relocation-aware operator provenance references.

The inference Runtime never uses this index to load world state or Agent input.
Model outputs may use the removable ``evaluate/`` workspace inside ProSeekBench;
all other in-release destinations remain protected. Historical receipts stay
byte-identical, and operator tools resolve their old paths.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from user_app_environment import find_repo_root

LAYOUT_VERSION = "proactivebench.external-artifacts.v1"
INDEX_REF = "../reports/proactivebench/artifact-relocations.json"


def _component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe artifact path component: {value!r}")
    return value


def output_directory(repo_root: Path, sampler_root: Path, area: str, category: str,
                     run_id: str | None = None) -> Path:
    if area not in {"runs", "reports", "_archive"}:
        raise ValueError(f"Unsupported artifact area: {area}")
    sampler = json.loads((sampler_root / "sampler.json").read_text(encoding="utf-8"))
    result = repo_root.resolve().parent / area / "proactivebench" / _component(sampler["user_id"]) / _component(sampler["sampler_id"]) / _component(category)
    return result / _component(run_id) if run_id is not None else result


def require_external_output(repo_root: Path, output: Path) -> Path:
    output = output.resolve()
    repo_root = repo_root.resolve()
    evaluate_root = (repo_root / "evaluate").resolve()
    if output.is_relative_to(repo_root) and not output.is_relative_to(evaluate_root):
        raise ValueError(
            "Model outputs inside ProSeekBench must stay under the removable "
            "evaluate/ workspace; release inputs and code are read-only."
        )
    return output


def sampler_relative_ref(sampler_root: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), sampler_root.resolve())).as_posix()


def resolve_sampler_artifact(sampler_root: Path, reference: str,
                             repo_root: Path | None = None) -> Path:
    repo_root = (repo_root or find_repo_root(sampler_root)).resolve()
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = sampler_root / candidate
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    index_path = (repo_root / INDEX_REF).resolve()
    if not index_path.is_file():
        return candidate
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sampler = json.loads((sampler_root / "sampler.json").read_text(encoding="utf-8"))
    for entry in sorted(index["moves"], key=lambda x: len(x["old_ref"]), reverse=True):
        if (entry["user_id"], entry["sampler_id"]) != (sampler["user_id"], sampler["sampler_id"]):
            continue
        try:
            suffix = (Path(reference).relative_to(entry["sampler_relative_old_ref"])
                      if not Path(reference).is_absolute()
                      else candidate.relative_to((repo_root / entry["old_ref"]).resolve()))
        except ValueError:
            continue
        return (repo_root / entry["new_ref"] / suffix).resolve()
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler-root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    result = resolve_sampler_artifact(args.sampler_root.resolve(), args.reference)
    print(json.dumps({"resolved": str(result), "exists": result.exists()}))


if __name__ == "__main__":
    main()
