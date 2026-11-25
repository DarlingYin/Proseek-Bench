#!/usr/bin/env python3
"""Materialize and verify a versioned Agent-facing catalog beside a frozen revision.

Only two derived sidecars are added. The revision manifest and its original
artifacts stay byte-identical, so existing Sampler environment locks remain valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generic_tool_adapter import GENERIC_FORMAT, GenericToolAdapter
from user_app_environment import file_sha256, read_json


CATALOG_NAME = "agent-tool-catalog.generic.v1.json"
BINDING_NAME = "agent-tool-catalog.generic.v1.binding.json"
BINDING_FORMAT = "proactivebench.agent-tool-catalog-binding.v1"


def _inputs(revision_path: Path) -> tuple[dict[str, Any], GenericToolAdapter, dict[str, Any]]:
    revision = read_json(revision_path)
    sources = {}
    values = {}
    for key in ("semantic_source_catalog", "public_capability_catalog"):
        declaration = revision["artifacts"][key]
        path = (revision_path.parent / declaration["ref"]).resolve()
        if not path.is_relative_to(revision_path.parent.resolve()):
            raise ValueError(f"Catalog must belong to its revision: {key}")
        digest = file_sha256(path)
        if digest != declaration["sha256"]:
            raise ValueError(f"Frozen catalog SHA256 mismatch: {key}")
        sources[key] = {"ref": declaration["ref"], "sha256": digest}
        values[key] = read_json(path)
    adapter = GenericToolAdapter(values["semantic_source_catalog"], values["public_capability_catalog"])
    binding = {
        "schema_version": BINDING_FORMAT,
        "user_id": revision["user_id"],
        "revision_id": revision["revision_id"],
        "revision_manifest": {"ref": revision_path.name, "sha256": file_sha256(revision_path)},
        "source_catalogs": sources,
        "catalog": {"ref": CATALOG_NAME, "format": GENERIC_FORMAT,
                    "canonical_sha256": adapter.catalog_sha256},
    }
    return adapter.public_catalog(), adapter, binding


def load_agent_tool_catalog(revision_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed on missing, stale or edited files; never fall back to latest tools."""
    expected, _, expected_binding = _inputs(revision_path)
    catalog_path = revision_path.parent / CATALOG_NAME
    binding_path = revision_path.parent / BINDING_NAME
    if not catalog_path.is_file() or not binding_path.is_file():
        raise FileNotFoundError(
            f"Missing bound generic tool catalog at {revision_path.parent}; "
            "run code/inference/compile_agent_tool_catalog.py --revision-dir <directory>"
        )
    actual = read_json(catalog_path)
    expected_binding["catalog"]["file_sha256"] = file_sha256(catalog_path)
    binding = read_json(binding_path)
    if actual != expected or binding != expected_binding:
        raise ValueError(f"Bound generic tool catalog or source binding mismatch: {revision_path.parent}")
    return actual, {
        "catalog_ref": CATALOG_NAME,
        "catalog_file_sha256": expected_binding["catalog"]["file_sha256"],
        "binding_ref": BINDING_NAME,
        "binding_sha256": file_sha256(binding_path),
        "revision_manifest_sha256": expected_binding["revision_manifest"]["sha256"],
    }


def materialize_agent_tool_catalog(revision_path: Path) -> dict[str, Any]:
    catalog_path = revision_path.parent / CATALOG_NAME
    binding_path = revision_path.parent / BINDING_NAME
    catalog, _, binding = _inputs(revision_path)
    serialized = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    # Exclusive creation, never overwrite an existing versioned representation.
    if catalog_path.exists():
        if catalog_path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"Refusing to replace versioned Agent catalog: {catalog_path}")
    else:
        with catalog_path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    binding["catalog"]["file_sha256"] = file_sha256(catalog_path)
    serialized_binding = json.dumps(binding, ensure_ascii=False, indent=2) + "\n"
    if binding_path.exists():
        if binding_path.read_text(encoding="utf-8") != serialized_binding:
            raise FileExistsError(f"Refusing to replace versioned Agent binding: {binding_path}")
    else:
        with binding_path.open("x", encoding="utf-8") as stream:
            stream.write(serialized_binding)
    _, receipt = load_agent_tool_catalog(revision_path)
    return {"revision": binding["revision_id"], **receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps([materialize_agent_tool_catalog(p.resolve() / "revision.json")
                      for p in args.revision_dir], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
