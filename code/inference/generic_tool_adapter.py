"""Versioned, data-independent aliases for frozen read-only App capabilities.

This module never queries the world. Every generic call resolves to exactly one
frozen tool and then uses the original query engine's validation and execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


GENERIC_INTERFACE = "generic"
LEGACY_INTERFACE = "legacy"
# New runs use generic by policy; recorded older runs retain their explicit mode.
DEFAULT_TOOL_INTERFACE = GENERIC_INTERFACE
GENERIC_FORMAT = "generic-read-tool-catalog.v1"
INTERFACE_IMPLEMENTATION = "generic-handle-relations-20261022.3"
EXACT_FILTER_VALUE_FIELDS = {("maps.route", "mode")}
RESOURCE_ROUTING_HINTS = {
    "babaquna.attraction": "Attractions, venue tickets, and bookable activity or session listings.",
    "calendar.event": "Scheduled events and their reminder coverage; not a generic cross-App search index.",
    "clock.alarm": "Existing device alarms only; use when alarm coverage can change whether help remains unmet.",
    "mail.message": "Email communication records, searched from a grounded sender, subject, or business locator.",
    "messages.message": "SMS/MMS communication records, searched from a grounded person, thread, or business locator.",
    "maps.route": "Mode-specific stored route estimates between modeled places.",
}
INTERFACE_FORMATS = {
    GENERIC_INTERFACE: GENERIC_FORMAT,
    LEGACY_INTERFACE: "compact-public-tool-catalog.v1",
}


def legacy_archive_root() -> Path:
    return Path(__file__).resolve().parents[3] / "_archive/app-tools-legacy-20261006"


def legacy_public_tools(declarations: dict[str, Any]) -> dict[str, Any]:
    """Optional historical compatibility; generic runs do not depend on this archive."""
    path = legacy_archive_root() / "legacy_tool_interface.py"
    if not path.is_file():
        raise FileNotFoundError("Legacy replay requires the external app-tools-legacy-20261006 archive")
    spec = importlib.util.spec_from_file_location("proactivebench_legacy_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.public_tools(declarations)


def add_tool_interface_argument(parser: argparse.ArgumentParser, *, default=DEFAULT_TOOL_INTERFACE):
    parser.add_argument(
        "--tool-interface", choices=tuple(INTERFACE_FORMATS), default=default,
        help="Use compact generic read tools for new runs; legacy is available for historical replay.",
    )


def search_protocol_for_interface(path: Path, interface: str) -> Path:
    """Select the archived legacy prompt, without overriding custom prompt paths."""
    root = Path(__file__).resolve().parents[2]
    standard_paths = {
        root / "prompts/inference/search_protocol.md",  # historical path
        root / "prompts/inference/framework/v1/search_protocol.md",
    }
    if interface == LEGACY_INTERFACE and path.resolve() in standard_paths:
        return legacy_archive_root() / "search_protocol.legacy.md"
    return path


def _strict_object(arguments: Any, allowed: set[str], required: set[str]) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    if set(arguments) - allowed:
        raise ValueError(f"Unknown tool arguments: {sorted(set(arguments) - allowed)}")
    missing = required - set(arguments)
    if missing:
        raise ValueError(f"Missing required tool arguments: {sorted(missing)}")


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _schema_without_prose(schema: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(schema)
    result.pop("description", None)
    if "properties" in result:
        result["properties"] = {
            name: _schema_without_prose(value)
            for name, value in result["properties"].items()
        }
    return result


class GenericToolAdapter:
    def __init__(self, semantic_catalog: dict[str, Any], public_catalog: dict[str, Any]):
        self.declarations = public_catalog["tools"]
        self.record_access = {name: value["record_access"] for name, value in semantic_catalog["sources"].items()
                              if "record_access" in value}
        self.internal = {
            declaration["tool_id"]: declaration
            for source in semantic_catalog["sources"].values()
            for declaration in source.get("tools", [])
        }
        self.search_routes: dict[str, str] = {}
        self.read_routes: dict[tuple[str, str], str] = {}
        self.state_routes: dict[tuple[str, str], str] = {}
        self.reverse_routes: dict[str, tuple[str, str]] = {}
        details: dict[str, list[str]] = {}
        for tool, declaration in sorted(self.declarations.items()):
            binding = declaration["runtime_binding"]
            resource = binding["entity_type"]
            if declaration["role"] == "discovery":
                if resource in self.search_routes:
                    raise ValueError(f"Multiple discovery operations need explicit routing: {resource}")
                self.search_routes[resource] = tool
                self.reverse_routes[tool] = ("search_records", resource)
            elif declaration["role"] == "detail":
                details.setdefault(resource, []).append(tool)
            elif declaration["role"] == "singleton":
                app = binding["source_id"]
                resource_view = binding["entity_type"].removeprefix(app + ".")
                route_key = (app, resource_view)
                if route_key in self.state_routes:
                    raise ValueError(
                        f"Multiple singleton operations for {app}/{resource_view}"
                    )
                if declaration["parameters"].get("properties"):
                    raise ValueError("Parameterized singleton needs an explicit interface extension")
                self.state_routes[route_key] = tool
                self.reverse_routes[tool] = (
                    "read_user_state", app + ":" + resource_view
                )
            else:
                raise ValueError(f"Unsupported frozen tool role: {declaration['role']}")
        for resource, tools in details.items():
            preferred = [t for t in tools if t.rsplit(".", 1)[-1] == "get_" + resource.rsplit(".", 1)[-1]]
            primary = tools[0] if len(tools) == 1 else (preferred[0] if len(preferred) == 1 else None)
            if primary is None:
                raise ValueError(f"Ambiguous default detail view for {resource}")
            for tool in tools:
                view = "details" if tool == primary else tool.rsplit(".", 1)[-1].removeprefix("get_")
                if (resource, view) in self.read_routes:
                    raise ValueError(f"Duplicate detail view: {resource}/{view}")
                self.read_routes[resource, view] = tool
                self.reverse_routes[tool] = ("read_record", view)
        self.extended_singletons = any(
            view != "user_state" for _, view in self.state_routes
        )
        self._catalog = self._compile_catalog()

    def _compile_catalog(self) -> dict[str, Any]:
        field_sets: dict[str, list[str]] = {}
        query_schemas: dict[str, Any] = {}
        resources: dict[str, Any] = {}
        states: dict[str, Any] = {}
        apps: dict[str, Any] = {}

        def intern(target, value, hint):
            for key, existing in target.items():
                if existing == value:
                    return key
            key = hint
            if key in target:
                key += "_" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]
            target[key] = value
            return key

        def fields(tool, hint):
            declaration = self.declarations[tool]
            binding = declaration["runtime_binding"]
            original = self.internal[binding["internal_tool_id"]]
            values = [x for x in original["result_fields"] if x != binding["internal_id_field"]]
            # Shared names describe structure rather than whichever App appeared first.
            common = {
                ("title", "subtitle"): "record_summary",
                ("title", "subtitle", "meta", "price", "available", "rank"): "catalog_record",
            }
            return intern(field_sets, values, common.get(tuple(values), hint))

        for declaration in self.declarations.values():
            source = declaration["source"]
            apps[source["source_id"]] = {"label": source["label"], "description": source["description"]}
        for resource, tool in sorted(self.search_routes.items()):
            declaration = self.declarations[tool]
            binding = declaration["runtime_binding"]
            original = self.internal[binding["internal_tool_id"]]
            schema = _schema_without_prose(declaration["parameters"])
            hint = "_".join(schema.get("properties", {})) or "empty"
            description = declaration["description"]
            # The App and opaque-handle rule are already stated in the directory.
            # Only shorten this exact generated boilerplate; retain custom scope text.
            boilerplate = re.fullmatch(
                "Search authorized " + re.escape(declaration["source"]["label"])
                + r" (.+) records and return opaque handles\.", description,
            )
            if boilerplate:
                description = "Search " + boilerplate.group(1) + " records."
            row = {
                "description": description,
                "query_schema": intern(query_schemas, schema, hint),
                "matches": deepcopy(original.get("filters", {})),
                "default_limit": original.get("default_limit", 20),
                "summary": fields(tool, resource.replace(".", "_") + "_summary"),
                "handle_field": binding["public_handle_field"],
                "views": {},
            }
            if declaration["cost_units"] != 1:
                row["search_cost_units"] = declaration["cost_units"]
            resources[resource] = row
        for (resource, view), tool in sorted(self.read_routes.items()):
            row = resources.setdefault(resource, {"views": {}})
            view_spec = {"fields": fields(tool, resource.replace(".", "_") + "_" + view)}
            if self.declarations[tool]["cost_units"] != 1:
                view_spec["cost_units"] = self.declarations[tool]["cost_units"]
            row["views"][view] = view_spec
        for (app, view), tool in sorted(self.state_routes.items()):
            row = {
                "description": self.declarations[tool]["description"],
                "fields": fields(tool, app + "_" + view),
            }
            if self.declarations[tool]["cost_units"] != 1:
                row["cost_units"] = self.declarations[tool]["cost_units"]
            if view == "user_state":
                states.setdefault(app, {}).update(row)
            else:
                states.setdefault(app, {}).setdefault("views", {})[view] = row
        # Frequently repeated resource attributes are explicit shallow defaults.
        # Exceptions replace the entire field; no schema constraint is discarded.
        resource_defaults = {}
        for key in ("query_schema", "matches", "default_limit", "summary", "handle_field"):
            counts = Counter(
                json.dumps(row[key], sort_keys=True, ensure_ascii=False)
                for row in resources.values() if key in row
            )
            if counts:
                encoded, count = counts.most_common(1)[0]
                if count > 1:
                    resource_defaults[key] = json.loads(encoded)
                    for row in resources.values():
                        if row.get(key) == resource_defaults[key]:
                            del row[key]
        def schema(properties, required):
            return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
        return {
            "format": GENERIC_FORMAT,
            "default_cost_units": 1,
            "tools": {
                "search_records": {
                    "description": "Search one declared resource; returns its summary fields, typed handles and coverage. No automatic detail reads.",
                    "parameters": schema({
                        "resource": {"type": "string"},
                        "filters": {"type": "object"},
                        "limit": {"type": "integer"},
                    }, ["resource"]),
                },
                "read_record": {
                    "description": "Read one previously discovered handle using one declared view. The handle selects its App and resource.",
                    "parameters": schema({
                        "handle": {"type": "string"},
                        "view": {"type": "string", "default": "details"},
                    }, ["handle"]),
                },
                "read_user_state": ({
                    "description": "Read one App singleton view. The default view is user_state; declared role or console views remain separately selectable.",
                    "parameters": schema({
                        "app": {"type": "string"},
                        "view": {"type": "string", "default": "user_state"},
                    }, ["app"]),
                } if self.extended_singletons else {
                    "description": "Read one App's declared user-state fields; does not read all its records.",
                    "parameters": schema({"app": {"type": "string"}}, ["app"]),
                }),
            },
            "apps": dict(sorted(apps.items())),
            "query_schemas": query_schemas,
            "field_sets": field_sets,
            "resource_defaults": resource_defaults,
            "resources": resources,
            "user_states": states,
        }

    def public_catalog(self) -> dict[str, Any]:
        return deepcopy(self._catalog)

    @property
    def catalog_sha256(self) -> str:
        data = json.dumps(self._catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()

    def relation_targets(self, resource: str, field: str) -> tuple[str, ...]:
        """Typed business joins; never infer a namespace from a coincident raw ID."""
        app = resource.split(".", 1)[0]
        if field == "destination_place_id":
            # A discovered hotel is itself a legitimate route destination.  The
            # private join still requires the exact shared business ID; no
            # address, coordinates, or destination are inferred by Runtime.
            return ("maps.place", "dudu.place", "babaquna.hotel")
        if field in {"place_id", "origin_place_id"}:
            return ("maps.place", "dudu.place")
        suffix = {"merchant_id": "merchant", "channel_id": "channel",
                  "thread_id": "thread", "root_post_id": "post"}.get(field)
        return (f"{app}.{suffix}",) if suffix else ()

    def usage_contract(self) -> dict[str, Any]:
        """Runtime supplement; keep frozen app-env catalogs byte-identical."""
        relations = {}
        for resource, route in self.search_routes.items():
            fields = self.declarations[route]["parameters"].get("properties", {})
            supported = {f: list(self.relation_targets(resource, f)) for f in fields
                         if self.relation_targets(resource, f)}
            if supported:
                relations[resource] = supported
        return {"implementation": INTERFACE_IMPLEMENTATION,
                "relation_filters": relations,
                "resource_routing": {key: value for key, value in RESOURCE_ROUTING_HINTS.items()
                                     if key in self.search_routes},
                "exact_filter_values": {
                    "maps.route.mode": {
                        "known_release_values": ["walk", "walking", "transit", "ride_hailing", "ride_service"],
                        "instruction": "Use an exact value already returned in this Episode; otherwise omit mode. "
                                       "Natural-language synonyms such as driving are not valid exact values."
                    }
                } if "maps.route" in self.search_routes else {},
                "instructions": "Relation filters accept previously returned handles of the listed types. "
                "The Runtime resolves them privately. Never guess internal IDs. An already observed "
                "literal business ID remains accepted. Wrong-type or unknown handles are errors, not empty results. "
                "Keyword uses the declared contains fields, not semantic or full-text search."}

    def resolve(self, tool: str, arguments: Any, issued_handles: dict[str, Any],
                entities: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        if tool == "search_records":
            _strict_object(arguments, {"resource", "filters", "limit"}, {"resource"})
            resource = _name(arguments["resource"], "resource")
            if resource not in self.search_routes:
                raise ValueError(f"Resource is not available in this environment: {resource}")
            filters = arguments.get("filters", {})
            if not isinstance(filters, dict):
                raise ValueError("filters must be an object")
            if "limit" in filters:
                raise ValueError("Place limit outside filters")
            resolved = deepcopy(filters)
            for field, value in filters.items():
                if (resource, field) in EXACT_FILTER_VALUE_FIELDS and isinstance(value, str):
                    available = sorted({
                        entity.get("attributes", {}).get(field)
                        for entity in (entities or {}).values()
                        if entity.get("entity_type") == resource
                        and isinstance(entity.get("attributes", {}).get(field), str)
                    })
                    if available and value not in available:
                        raise ValueError(
                            f"Unsupported exact value {value!r} for {resource}.{field}; values represented in "
                            f"the current world are {available}. Use an exact returned value or omit this filter; "
                            "do not guess a natural-language synonym."
                        )
                targets = self.relation_targets(resource, field)
                if not targets or not isinstance(value, str):
                    continue
                record = issued_handles.get(value)
                if record is None:
                    if re.fullmatch(r"[^:]+:[^:]+:\d+", value) or value.startswith("private:"):
                        raise ValueError(f"{field} requires a previously issued handle of type {', '.join(targets)}")
                    continue  # Preserve existing literal business-ID filters.
                if record["entity_type"] not in targets or record["source_id"] != record["entity_type"].split('.')[0]:
                    raise ValueError(f"Wrong handle type for {field}; expected {', '.join(targets)}")
                entity = (entities or {}).get(record["entity_ref"])
                if entity is None or entity.get("entity_type") != record["entity_type"] or entity.get("source") != record["source_id"]:
                    raise ValueError(f"Handle for {field} is no longer available")
                access = getattr(self, "record_access", {}).get(record["source_id"])
                if access:
                    actual = entity['attributes'].get(access['field'])
                    if type(actual) is not type(access['equals']) or actual != access['equals']:
                        raise ValueError(f"Handle for {field} is no longer accessible")
                parent_binding = self.declarations[self.search_routes[record["entity_type"]]]["runtime_binding"]
                internal_id = entity["attributes"].get(parent_binding["internal_id_field"])
                if not isinstance(internal_id, str) or not internal_id:
                    raise ValueError(f"Handle for {field} has no valid relation key")
                resolved[field] = internal_id
            if "limit" in arguments:
                resolved["limit"] = arguments["limit"]
            return self.search_routes[resource], resolved
        if tool == "read_record":
            _strict_object(arguments, {"handle", "view"}, {"handle"})
            handle = _name(arguments["handle"], "handle")
            record = issued_handles.get(handle)
            if record is None:
                raise ValueError("Read requires a typed handle returned by a prior discovery call")
            view = _name(arguments.get("view", "details"), "view")
            route = self.read_routes.get((record["entity_type"], view))
            if route is None:
                raise ValueError(f"View is not available for this handle: {view}")
            binding = self.declarations[route]["runtime_binding"]
            if record["source_id"] != binding["source_id"]:
                raise ValueError("Handle source does not match the declared resource")
            return route, {binding["public_handle_field"]: handle}
        if tool == "read_user_state":
            _strict_object(arguments, {"app", "view"}, {"app"})
            app = _name(arguments["app"], "app")
            view = _name(arguments.get("view", "user_state"), "view")
            route = self.state_routes.get((app, view))
            if route is None:
                raise ValueError(f"Singleton reader is not available: {app}/{view}")
            return route, {}
        raise ValueError(f"Tool is not available in generic interface: {tool}")

    def from_legacy(self, tool: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Translate a validated legacy trace for equivalence tests or explicit migration."""
        if tool not in self.reverse_routes:
            raise ValueError(f"No generic mapping for {tool}")
        name, key = self.reverse_routes[tool]
        if name == "search_records":
            result = {"resource": key, "filters": {k: deepcopy(v) for k, v in arguments.items() if k != "limit"}}
            if "limit" in arguments:
                result["limit"] = arguments["limit"]
            return name, result
        if name == "read_user_state":
            if arguments:
                raise ValueError("Legacy user-state arguments must be empty")
            app, view = key.split(":", 1)
            result = {"app": app}
            if view != "user_state":
                result["view"] = view
            return name, result
        field = self.declarations[tool]["runtime_binding"]["public_handle_field"]
        _strict_object(arguments, {field}, {field})
        return name, {"handle": arguments[field], "view": key}
