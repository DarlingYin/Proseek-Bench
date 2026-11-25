#!/usr/bin/env python3
"""Compile the fixed Agent-facing tool catalog from the semantic source catalog.

The semantic catalog is also used by environment initialization and therefore
contains stable internal entity IDs.  This compiler produces a separate public
contract that replaces those IDs with typed opaque handles.  Entity values are
never copied into the public catalog.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any


def _arguments_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, specification in parameters.items():
        public_spec = {
            key: deepcopy(value)
            for key, value in specification.items()
            if key not in {"required", "role"}
        }
        properties[name] = public_spec
        if specification.get("required"):
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

def compile_public_catalog(semantic_catalog: dict[str, Any]) -> dict[str, Any]:
    policy = semantic_catalog["public_access"]
    handle_types = policy["handle_types"]
    singleton_specs = policy.get("singleton_tools", {})

    internal_tools: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for source_id, source in semantic_catalog["sources"].items():
        for declaration in source.get("tools", []):
            internal_tools[declaration["tool_id"]] = (source_id, source, declaration)

    detail_tools_by_entity: dict[str, list[str]] = {}
    for internal_tool_id, (_, _, declaration) in internal_tools.items():
        if declaration["operation"] != "get":
            continue
        for entity_type in declaration["entity_types"]:
            if entity_type in handle_types:
                detail_tools_by_entity.setdefault(entity_type, []).append(
                    internal_tool_id
                )

    public_tools: dict[str, Any] = {}
    for internal_tool_id, (source_id, source, declaration) in internal_tools.items():
        entity_types = declaration["entity_types"]
        if len(entity_types) != 1 or entity_types[0] not in handle_types:
            continue
        entity_type = entity_types[0]
        handle = handle_types[entity_type]
        internal_id = handle["internal_id_field"]
        public_handle = handle["public_handle_field"]
        public_parameters = deepcopy(declaration["parameters"])
        if declaration["operation"] == "get":
            identity_names = [
                name
                for name, specification in public_parameters.items()
                if specification.get("role") == "identity"
            ]
            if identity_names != [internal_id]:
                raise ValueError(
                    f"{internal_tool_id} must use {internal_id} as its identity argument"
                )
            public_parameters = {
                public_handle: {
                    "type": "string",
                    "pattern": (
                        "^" + re.escape(handle["public_handle_prefix"]) + ":[0-9]+$"
                    ),
                    "required": True,
                    "description": (
                        f"Opaque {entity_type} handle returned by a prior discovery call."
                    ),
                }
            }

        public_tools[internal_tool_id] = {
            "tool_id": internal_tool_id,
            "description": declaration["description"].replace(
                internal_id, public_handle
            ),
            "source": {
                "source_id": source_id,
                "label": source["label"],
                "description": source["description"],
            },
            "role": (
                "discovery" if declaration["operation"] == "list" else "detail"
            ),
            "use_when": declaration["use_when"],
            "operation": declaration["operation"],
            "parameters": _arguments_schema(public_parameters),
            "cost_units": declaration["cost_units"],
            "runtime_binding": {
                "source_id": source_id,
                "entity_type": entity_type,
                "internal_tool_id": internal_tool_id,
                "internal_id_field": internal_id,
                "public_handle_field": public_handle,
                "public_handle_prefix": handle["public_handle_prefix"],
            },
        }
        if declaration["operation"] == "list":
            public_tools[internal_tool_id]["produces_handle"] = {
                "field": public_handle,
                "handle_type": entity_type,
            }
            public_tools[internal_tool_id]["follow_up_tools"] = sorted(
                detail_tools_by_entity.get(entity_type, [])
            )
        else:
            public_tools[internal_tool_id]["consumes_handle"] = {
                "field": public_handle,
                "handle_type": entity_type,
                "must_be_previously_discovered": True,
            }

    for public_tool_id, singleton in singleton_specs.items():
        internal_tool_id = singleton["internal_tool_id"]
        source_id, source, declaration = internal_tools[internal_tool_id]
        entity_type = singleton["entity_type"]
        entity_declaration = source["entity_types"][entity_type]
        internal_id = entity_declaration["id_field"]
        public_tools[public_tool_id] = {
            "tool_id": public_tool_id,
            "description": declaration["description"],
            "source": {
                "source_id": source_id,
                "label": source["label"],
                "description": source["description"],
            },
            "role": "singleton",
            "use_when": declaration["use_when"],
            "requires_handle": False,
            "operation": "singleton",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "cost_units": declaration["cost_units"],
            "runtime_binding": {
                "source_id": source_id,
                "entity_type": entity_type,
                "internal_tool_id": internal_tool_id,
                "internal_id_field": internal_id,
            },
        }

    return {
        "schema_version": "online-mobile-agent.public-capability-catalog.v1",
        "exposure_policy": policy["exposure_policy"],
        "description": (
            "Fixed Agent-visible search catalog. It contains schemas only, never "
            "episode entity values or target-specific capability allowlists."
        ),
        "result_contract": {
            "discovery": {
                "applies_to_role": "discovery",
                "fields": ["status", "items", "coverage", "cost_units"],
                "coverage_fields": ["complete", "matched"],
            },
            "detail_or_singleton": {
                "applies_to_roles": ["detail", "singleton"],
                "fields": ["status", "data", "cost_units"],
            },
        },
        "tools": dict(sorted(public_tools.items())),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic-catalog",
        type=Path,
        default=repo_root / "platform/semantic_source_catalog.v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "platform/public_capability_catalog.v1.json",
    )
    args = parser.parse_args()
    semantic_catalog = json.loads(args.semantic_catalog.read_text(encoding="utf-8"))
    public_catalog = compile_public_catalog(semantic_catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(public_catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(public_catalog['tools'])} tools to {args.output}")


if __name__ == "__main__":
    main()
