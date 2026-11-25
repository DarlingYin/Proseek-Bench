"""Opt-in deterministic normalization for small-model decision protocols.

The adapter is deliberately data-independent: it may repair serialization,
field placement, exact catalog aliases, and unambiguous Runtime bindings, but
it never invents Need/Claims text, search values, or a semantic action.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


ADAPTER_OFF = "off"
ADAPTER_SAFE = "safe"
ADAPTER_STATEFUL = "stateful"
ADAPTER_Q35_RECOVERY = "q35_recovery"
ADAPTER_MIMO_RECOVERY = "mimo_recovery"
ADAPTER_Q3VL_RECOVERY = "q3vl_recovery"
ADAPTER_Q38_RECOVERY = "q38_recovery"
ADAPTER_INTERNVL_RECOVERY = "internvl_recovery"
ADAPTER_GLM46V_RECOVERY = "glm46v_recovery"
ADAPTER_STEP3_RECOVERY = "step3_recovery"
ADAPTER_MODEL_DEFAULT = "model_default"
ADAPTER_MODES = (
    ADAPTER_OFF,
    ADAPTER_SAFE,
    ADAPTER_STATEFUL,
    ADAPTER_Q35_RECOVERY,
    ADAPTER_MIMO_RECOVERY,
    ADAPTER_Q3VL_RECOVERY,
    ADAPTER_Q38_RECOVERY,
    ADAPTER_INTERNVL_RECOVERY,
    ADAPTER_GLM46V_RECOVERY,
    ADAPTER_STEP3_RECOVERY,
    ADAPTER_MODEL_DEFAULT,
)
ADAPTER_VERSION = "proseekbench.protocol-adapter.v2.8"


MODEL_ADAPTER_PROFILES = {
    "local-q35-35b-a3b": ADAPTER_Q35_RECOVERY,
    "mimo-vl-7b-sft": ADAPTER_MIMO_RECOVERY,
    "local-q3vl-8b-instruct": ADAPTER_Q3VL_RECOVERY,
    "local-q38-27b": ADAPTER_Q38_RECOVERY,
    "internvl3.5-8b": ADAPTER_INTERNVL_RECOVERY,
    "local-v46-flash": ADAPTER_GLM46V_RECOVERY,
}


def resolve_adapter_mode(mode: str, model: str) -> str:
    """Resolve the frozen per-model profile without changing cloud defaults."""

    if mode != ADAPTER_MODEL_DEFAULT:
        return mode
    return MODEL_ADAPTER_PROFILES.get(model.casefold(), ADAPTER_SAFE)


def _compact_q38_claims(text: str, *, target: int = 880) -> str:
    """Bound verbose local-model Claims while retaining both causal ends."""

    normalized = " ".join(text.split())
    if len(normalized) <= 900:
        return normalized
    marker = " … "
    head_budget = 620
    tail_budget = target - head_budget - len(marker)
    head = normalized[:head_budget].rsplit(" ", 1)[0]
    tail = normalized[-tail_budget:].split(" ", 1)[-1]
    return head + marker + tail


def _redact_internal_trace_ids(text: str) -> str:
    """Remove private oNNN/qNNN trace labels without changing evidence content."""

    import re

    value = re.sub(
        r"\b(?:Observation|Screenshot)\s+o\d+\b",
        "The observed screen",
        text,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bQuery\s+q\d+\b", "The executed query", value, flags=re.IGNORECASE)
    value = re.sub(r"\b[oq]\d{3,}(?:/\d{3,})*\b", "the observed evidence", value)
    return " ".join(value.split())


@dataclass(frozen=True)
class Adaptation:
    decision: dict[str, Any]
    events: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.events)


def _thread_value(thread: Any, name: str) -> Any:
    if isinstance(thread, Mapping):
        return thread.get(name)
    return getattr(thread, name, None)


class DecisionProtocolAdapter:
    def __init__(self, mode: str) -> None:
        if mode not in ADAPTER_MODES or mode == ADAPTER_MODEL_DEFAULT:
            raise ValueError(f"Unknown protocol adapter mode: {mode}")
        self.mode = mode

    def adapt(
        self,
        decision: dict[str, Any],
        *,
        threads: Mapping[str, Any],
        tool_catalog: dict[str, Any],
    ) -> Adaptation:
        if self.mode == ADAPTER_OFF:
            return Adaptation(deepcopy(decision), ())
        if not isinstance(decision, dict):
            return Adaptation(deepcopy(decision), ())
        value = deepcopy(decision)
        events: list[str] = []

        operations = value.get("thread_ops", [])
        if isinstance(operations, dict):
            operations = [operations]
            events.append("thread_ops_object_to_array")
        if not isinstance(operations, list):
            return Adaptation(value, tuple(events))

        control = value.get("control")
        if isinstance(control, str):
            control = {"type": control}
            if control["type"] == "commit" and "thread_ids" in value:
                control["thread_ids"] = value["thread_ids"]
                value.pop("thread_ids", None)
                events.append("nest_top_level_thread_ids")
            events.append("control_string_to_object")

        if (
            self.mode in {ADAPTER_GLM46V_RECOVERY, ADAPTER_STEP3_RECOVERY}
            and isinstance(control, dict)
            and "linked_thread_ids" in value
        ):
            top_level_linked = value.get("linked_thread_ids")
            control_linked = control.get("linked_thread_ids")
            if (
                isinstance(top_level_linked, list)
                and top_level_linked
                and all(isinstance(item, str) for item in top_level_linked)
                and control_linked in (None, [])
            ):
                control = deepcopy(control)
                control["linked_thread_ids"] = deepcopy(top_level_linked)
                value.pop("linked_thread_ids")
                events.append("glm46v_move_top_level_linked_thread_ids")
            elif top_level_linked == control_linked:
                value.pop("linked_thread_ids")
                events.append("glm46v_drop_duplicate_top_level_linked_thread_ids")

        lifecycle_mode = self.mode in {
            ADAPTER_STATEFUL,
            ADAPTER_Q35_RECOVERY,
            ADAPTER_MIMO_RECOVERY,
            ADAPTER_Q3VL_RECOVERY,
            ADAPTER_Q38_RECOVERY,
            ADAPTER_INTERNVL_RECOVERY,
            ADAPTER_GLM46V_RECOVERY,
            ADAPTER_STEP3_RECOVERY,
        }
        claims_cleanup_mode = self.mode in {
            ADAPTER_Q35_RECOVERY,
            ADAPTER_MIMO_RECOVERY,
            ADAPTER_Q38_RECOVERY,
        }

        retained: list[Any] = []
        moved_controls: list[dict[str, Any]] = []
        for operation in operations:
            if not isinstance(operation, dict):
                retained.append(operation)
                continue
            kind = operation.get("op")
            if (
                kind is None
                and self.mode == ADAPTER_MIMO_RECOVERY
                and operation.get("type") in {"wait", "commit", "search"}
            ):
                operation = {"op": operation["type"], **{
                    key: deepcopy(item) for key, item in operation.items() if key != "type"
                }}
                kind = operation["op"]
                events.append("mimo_control_type_to_op")
            if kind in {"wait", "commit", "search"}:
                if (
                    kind == "commit"
                    and self.mode in {ADAPTER_MIMO_RECOVERY, ADAPTER_INTERNVL_RECOVERY}
                    and isinstance(operation.get("thread_id"), str)
                    and any(field in operation for field in ("need", "claims"))
                ):
                    thread_id = operation["thread_id"]
                    existing = threads.get(thread_id)
                    status = _thread_value(existing, "status") if existing is not None else None
                    if status in {None, "candidate"}:
                        replacement = {
                            "op": "create" if existing is None else "update",
                            "thread_id": thread_id,
                        }
                        for field in ("need", "claims"):
                            if field in operation:
                                replacement[field] = deepcopy(operation[field])
                        retained.append(replacement)
                        moved_controls.append({"type": "commit", "thread_ids": [thread_id]})
                        events.append(
                            "mimo_split_commit_op_with_content"
                            if self.mode == ADAPTER_MIMO_RECOVERY
                            else "internvl_split_commit_op_with_content"
                        )
                        continue
                moved = {"type": kind}
                moved.update({k: deepcopy(v) for k, v in operation.items() if k != "op"})
                if kind == "commit" and "thread_id" in moved and "thread_ids" not in moved:
                    moved["thread_ids"] = [moved.pop("thread_id")]
                    events.append("commit_thread_id_to_array")
                moved_controls.append(moved)
                events.append(f"move_{kind}_op_to_control")
                continue
            operation = deepcopy(operation)
            for field in (
                "status", "created_after", "updated_after", "committed_after",
                "committed_at", "deleted_reason",
            ):
                if field in operation:
                    operation.pop(field)
                    events.append(f"strip_runtime_field:{field}")
            if kind == "update" and set(operation) <= {"op", "thread_id"}:
                events.append("drop_empty_update")
                continue
            retained.append(operation)
        operations = retained

        if control is None and len(moved_controls) == 1:
            control = moved_controls[0]
        elif control is None and not moved_controls and lifecycle_mode:
            control = {"type": "wait"}
            events.append("default_missing_control_to_wait")
        elif moved_controls:
            candidates = [control, *moved_controls] if isinstance(control, dict) else moved_controls
            if lifecycle_mode and candidates and all(
                isinstance(item, dict) and item.get("type") == candidates[0].get("type")
                for item in candidates
            ):
                merged = deepcopy(candidates[0])
                if merged.get("type") == "commit":
                    ids: list[str] = []
                    for item in candidates:
                        values = item.get("thread_ids", [])
                        if isinstance(values, str):
                            values = [values]
                        if not isinstance(values, list):
                            return Adaptation(deepcopy(decision), ())
                        for thread_id in values:
                            if isinstance(thread_id, str) and thread_id not in ids:
                                ids.append(thread_id)
                    merged = {"type": "commit", "thread_ids": ids}
                elif any(item != merged for item in candidates[1:]):
                    return Adaptation(deepcopy(decision), ())
                control = merged
                events.append("dedupe_equivalent_controls")
            else:
                # A conflicting or multiple control is ambiguous; do not choose for the model.
                return Adaptation(deepcopy(decision), ())

        if isinstance(control, dict):
            control = deepcopy(control)
            control_type = control.get("type")
            if control_type == "search":
                if "linked_thread_ids" not in control:
                    control["linked_thread_ids"] = []
                    events.append("default_linked_thread_ids")
                tool = control.get("tool")
                arguments = control.get("arguments")
                resources = tool_catalog.get("resources", {})
                if (
                    self.mode == ADAPTER_GLM46V_RECOVERY
                    and isinstance(tool, str)
                    and isinstance(arguments, dict)
                    and isinstance(resources, dict)
                ):
                    declared_resource = arguments.get("resource")
                    if (
                        isinstance(declared_resource, str)
                        and declared_resource in resources
                        and declared_resource.startswith(tool + ".")
                    ):
                        tool = declared_resource
                        control["tool"] = tool
                        events.append(f"glm46v_bind_declared_resource_namespace:{tool}")
                if (
                    self.mode == ADAPTER_MIMO_RECOVERY
                    and isinstance(tool, str)
                    and tool not in resources
                    and isinstance(resources, dict)
                ):
                    namespace_matches = [
                        resource for resource in resources
                        if isinstance(resource, str) and resource.startswith(tool + ".")
                    ]
                    if len(namespace_matches) == 1:
                        tool = namespace_matches[0]
                        control["tool"] = tool
                        events.append(f"mimo_bind_unique_resource_namespace:{tool}")
                if isinstance(tool, str) and tool in resources and isinstance(arguments, dict):
                    arguments = deepcopy(arguments)
                    has_limit = 'limit' in arguments
                    limit = arguments.pop("limit", None)
                    nested_filters = arguments.pop("filters", {})
                    declared_resource = arguments.get("resource")
                    if declared_resource == tool:
                        arguments.pop("resource")
                        events.append("drop_redundant_resource_alias_argument")
                    elif "resource" in arguments:
                        return Adaptation(deepcopy(decision), ())
                    if not isinstance(nested_filters, dict) or any(k in nested_filters and nested_filters[k] != v
                                                                 for k, v in arguments.items()):
                        return Adaptation(deepcopy(decision), ())
                    nested_filters.update(arguments)
                    mapped = {"resource": tool, "filters": nested_filters}
                    if has_limit:
                        mapped["limit"] = limit
                    control["tool"] = "search_records"
                    control["arguments"] = mapped
                    events.append(f"map_resource_alias:{tool}")
                if control.get('tool') == "search_records" and isinstance(control.get('arguments'), dict):
                    arguments = control['arguments']
                    nested_linked = arguments.get("linked_thread_ids")
                    if self.mode == ADAPTER_MIMO_RECOVERY and nested_linked is not None:
                        current_linked = control.get("linked_thread_ids")
                        normalized_nested = [nested_linked] if isinstance(nested_linked, str) else nested_linked
                        if isinstance(normalized_nested, list) and current_linked in (None, []):
                            control["linked_thread_ids"] = deepcopy(normalized_nested)
                            arguments.pop("linked_thread_ids")
                            events.append("mimo_move_linked_thread_ids_out_of_arguments")
                    resource = arguments.get("resource")
                    resource_record = tool_catalog.get("resources", {}).get(resource, {})
                    schema_name = resource_record.get("query_schema")
                    schema = tool_catalog.get("query_schemas", {}).get(schema_name, {})
                    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
                    structural = {
                        key: arguments[key]
                        for key in list(arguments)
                        if key not in {"resource", "filters", "limit"}
                    }
                    if structural and isinstance(properties, dict) and set(structural) <= set(properties):
                        filters = arguments.setdefault("filters", {})
                        if not isinstance(filters, dict) or any(
                            key in filters and filters[key] != item for key, item in structural.items()
                        ):
                            return Adaptation(deepcopy(decision), ())
                        filters.update(structural)
                        for key in structural:
                            arguments.pop(key)
                        events.append("nest_query_schema_fields_in_filters")
                    filters = arguments.get("filters")
                    if self.mode in {
                        ADAPTER_MIMO_RECOVERY,
                        ADAPTER_GLM46V_RECOVERY,
                        ADAPTER_STEP3_RECOVERY,
                    } and isinstance(filters, dict):
                        if (
                            set(filters) == {"keyword", "value"}
                            and isinstance(filters.get("keyword"), dict)
                            and "value" not in filters["keyword"]
                            and isinstance(filters.get("value"), (str, int, float, bool))
                            and "keyword" in properties
                        ):
                            filters["keyword"] = filters.pop("value")
                            events.append(
                                "mimo_bind_sibling_value_to_keyword"
                                if self.mode == ADAPTER_MIMO_RECOVERY
                                else f"{self.mode}_bind_sibling_value_to_keyword"
                            )
                        if self.mode == ADAPTER_GLM46V_RECOVERY and "value" in filters:
                            value_fields = [field for field in filters if field != "value"]
                            if (
                                len(value_fields) == 1
                                and value_fields[0] in properties
                                and filters[value_fields[0]] in {"eq", "gte", "lte", "contains"}
                                and isinstance(filters["value"], (str, int, float, bool))
                            ):
                                field = value_fields[0]
                                filters[field] = filters.pop("value")
                                events.append(f"glm46v_bind_sibling_value:{field}")
                        for field in list(filters):
                            item = filters[field]
                            if field not in properties or not isinstance(item, dict):
                                continue
                            if "value" in item and set(item) <= {"value", "field", "fields", "operator"}:
                                filters[field] = item["value"]
                                prefix = "mimo" if self.mode == ADAPTER_MIMO_RECOVERY else self.mode
                                events.append(f"{prefix}_unwrap_filter_value:{field}")
                            elif (
                                self.mode == ADAPTER_GLM46V_RECOVERY
                                and field == "keyword"
                                and set(item) <= {"keyword", "limit"}
                                and isinstance(item.get("keyword"), str)
                            ):
                                filters[field] = item["keyword"]
                                if "limit" in item and "limit" not in arguments:
                                    arguments["limit"] = item["limit"]
                                events.append("glm46v_unwrap_keyword_filter")
                            elif len(item) == 1:
                                operator, operand = next(iter(item.items()))
                                if operator in {"eq", "gte", "lte", "contains"}:
                                    filters[field] = operand
                                    prefix = "mimo" if self.mode == ADAPTER_MIMO_RECOVERY else self.mode
                                    events.append(f"{prefix}_unwrap_filter_operator:{field}:{operator}")
                    if isinstance(filters, dict) and "limit" in filters and "limit" not in arguments:
                        arguments["limit"] = filters.pop("limit")
                        events.append("move_limit_outside_filters")
                if isinstance(control.get('linked_thread_ids'), str):
                    control['linked_thread_ids'] = [control['linked_thread_ids']]
                    events.append('linked_thread_id_to_array')

                if (
                    self.mode == ADAPTER_Q35_RECOVERY
                    and control.get("tool") == "read_record"
                    and isinstance(control.get("arguments"), dict)
                ):
                    handle = control["arguments"].get("handle")
                    filename = None
                    if isinstance(handle, str) and handle.startswith("file:"):
                        filename = handle.removeprefix("file:").rsplit("/", 1)[-1]
                    elif isinstance(handle, str) and handle.startswith("files:document:"):
                        suffix = handle.removeprefix("files:document:")
                        if not suffix.isdigit():
                            filename = suffix.rsplit("/", 1)[-1]
                    file_resource = tool_catalog.get("resources", {}).get("files.document", {})
                    matches = file_resource.get("matches", {}) if isinstance(file_resource, dict) else {}
                    if filename and isinstance(matches, dict) and "keyword" in matches:
                        control = {
                            "type": "search",
                            "tool": "search_records",
                            "arguments": {
                                "resource": "files.document",
                                "filters": {"keyword": filename},
                                "limit": 20,
                            },
                            "linked_thread_ids": deepcopy(control.get("linked_thread_ids", [])),
                        }
                        events.append("q35_file_read_to_discovery")
            elif control_type == "commit":
                thread_ids = control.get("thread_ids")
                if isinstance(thread_ids, str):
                    control["thread_ids"] = [thread_ids]
                    events.append("commit_thread_id_to_array")

        if claims_cleanup_mode and isinstance(control, dict) and control.get("type") == "commit":
            committed = control.get("thread_ids")
            committed_ids = set(committed) if isinstance(committed, list) else set()
            operated_ids = {
                operation.get("thread_id") for operation in operations
                if isinstance(operation, dict) and isinstance(operation.get("thread_id"), str)
            }
            for operation in operations:
                if not isinstance(operation, dict) or operation.get("thread_id") not in committed_ids:
                    continue
                claims = operation.get("claims")
                if isinstance(claims, str):
                    cleaned = _redact_internal_trace_ids(claims)
                    if cleaned != claims:
                        operation["claims"] = cleaned
                        events.append("redact_internal_trace_ids")
            for thread_id in sorted(committed_ids - operated_ids):
                existing = threads.get(thread_id)
                if existing is None or _thread_value(existing, "status") != "candidate":
                    continue
                claims = _thread_value(existing, "claims")
                if not isinstance(claims, str):
                    continue
                cleaned = _redact_internal_trace_ids(claims)
                if cleaned != claims:
                    operations.append({"op": "update", "thread_id": thread_id, "claims": cleaned})
                    events.append("inject_trace_id_cleanup_before_commit")

        if (
            self.mode == ADAPTER_GLM46V_RECOVERY
            and isinstance(control, dict)
            and control.get("type") == "search"
            and control.get("tool") == "search_records"
        ):
            arguments = control.get("arguments")
            filters = arguments.get("filters") if isinstance(arguments, dict) else None
            if isinstance(filters, dict):
                descriptor_fields = [
                    field for field, value in filters.items()
                    if isinstance(value, dict)
                    and set(value) <= {"field", "fields", "operator"}
                    and "operator" in value
                    and ("field" in value or "fields" in value)
                ]
                valid_fields = [field for field in filters if field not in descriptor_fields]
                if valid_fields:
                    for field in descriptor_fields:
                        filters.pop(field)
                        events.append(f"glm46v_drop_schema_descriptor_filter:{field}")

        if self.mode == ADAPTER_Q38_RECOVERY:
            # Candidate-thread updates can grow across a long passive stream.
            # Keep both causal ends before schema validation instead of letting
            # an otherwise valid wait/update fail the 2,000-character bound.
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                claims = operation.get("claims")
                if isinstance(claims, str) and len(" ".join(claims.split())) > 2000:
                    operation["claims"] = _compact_q38_claims(claims, target=1900)
                    events.append("q38_compact_thread_claims")
            if isinstance(control, dict) and control.get("type") == "search":
                arguments = control.get("arguments")
                if control.get("tool") == "search_records" and isinstance(arguments, dict):
                    resource = arguments.get("resource")
                    filters = arguments.get("filters")
                    resource_record = tool_catalog.get("resources", {}).get(resource, {})
                    schema_name = resource_record.get("query_schema")
                    schema = tool_catalog.get("query_schemas", {}).get(schema_name, {})
                    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
                    if isinstance(filters, dict) and isinstance(properties, dict) and properties:
                        for field in sorted(set(filters) - set(properties)):
                            filters.pop(field)
                            events.append(f"q38_drop_unknown_filter:{field}")
            if isinstance(control, dict) and control.get("type") == "commit":
                committed = control.get("thread_ids")
                committed_ids = set(committed) if isinstance(committed, list) else set()
                for operation in operations:
                    if not isinstance(operation, dict) or operation.get("thread_id") not in committed_ids:
                        continue
                    claims = operation.get("claims")
                    if isinstance(claims, str) and len(" ".join(claims.split())) > 900:
                        operation["claims"] = _compact_q38_claims(claims)
                        events.append("q38_compact_commit_claims")

        if lifecycle_mode:
            normalized: list[Any] = []
            for operation in operations:
                if not isinstance(operation, dict):
                    normalized.append(operation)
                    continue
                kind = operation.get("op")
                thread_id = operation.get("thread_id")
                existing = threads.get(thread_id) if isinstance(thread_id, str) else None
                status = _thread_value(existing, "status") if existing is not None else None
                if kind == "create" and status == "candidate":
                    replacement = {"op": "update", "thread_id": thread_id}
                    for field in ("need", "claims"):
                        if field in operation:
                            replacement[field] = operation[field]
                    normalized.append(replacement)
                    events.append("create_existing_candidate_to_update")
                    continue
                if kind == "update" and status != "candidate":
                    if (
                        self.mode == ADAPTER_GLM46V_RECOVERY
                        and existing is None
                        and isinstance(operation.get("need"), str)
                        and isinstance(operation.get("claims"), str)
                    ):
                        operation["op"] = "create"
                        normalized.append(operation)
                        events.append("glm46v_unknown_full_update_to_create")
                        continue
                    same = all(
                        field not in operation or operation[field] == _thread_value(existing, field)
                        for field in ("need", "claims")
                    )
                    if existing is not None and same:
                        events.append("drop_idempotent_non_candidate_update")
                        continue
                    candidates = [
                        candidate_id for candidate_id, thread in threads.items()
                        if _thread_value(thread, "status") == "candidate"
                    ]
                    if self.mode == ADAPTER_MIMO_RECOVERY and existing is None and len(candidates) == 1:
                        operation["thread_id"] = candidates[0]
                        events.append("mimo_bind_unknown_thread_to_single_candidate")
                if kind == "delete" and status == "deleted":
                    events.append("drop_repeated_delete")
                    continue
                if (
                    kind == "delete"
                    and self.mode == ADAPTER_GLM46V_RECOVERY
                    and status == "committed"
                ):
                    events.append("glm46v_drop_delete_committed_thread")
                    continue
                normalized.append(operation)
            operations = normalized

            if isinstance(control, dict) and control.get("type") == "commit":
                # Runtime applies thread_ops BEFORE control. A same-turn create
                # is a valid commit target; a same-turn delete is not.
                candidates = {
                    thread_id for thread_id, thread in threads.items()
                    if _thread_value(thread, "status") == "candidate"
                }
                for operation in operations:
                    if not isinstance(operation, dict):
                        continue
                    thread_id = operation.get("thread_id")
                    if not isinstance(thread_id, str):
                        continue
                    if operation.get("op") == "create" and thread_id not in threads:
                        candidates.add(thread_id)
                    elif operation.get("op") == "delete":
                        candidates.discard(thread_id)
                ids = control.get("thread_ids")
                if ids in (None, []) and len(candidates) == 1:
                    ids = sorted(candidates)
                    control["thread_ids"] = ids
                    events.append("bind_single_candidate_commit")
                elif (
                    self.mode == ADAPTER_MIMO_RECOVERY
                    and isinstance(ids, list)
                    and ids
                    and not any(isinstance(thread_id, str) and thread_id in candidates for thread_id in ids)
                    and len(candidates) == 1
                ):
                    ids = sorted(candidates)
                    control["thread_ids"] = ids
                    events.append("mimo_bind_unknown_commit_to_single_candidate")
                if isinstance(ids, list):
                    retained_ids = [
                        thread_id for thread_id in ids
                        if isinstance(thread_id, str) and thread_id in candidates
                    ]
                    if retained_ids != ids:
                        events.append("drop_non_candidate_commit_ids")
                        if retained_ids:
                            control["thread_ids"] = retained_ids
                        else:
                            control = {"type": "wait"}
                            events.append("empty_commit_to_wait")

        # Unknown top-level fields remain visible to validation; never discard content.
        canonical = {**value, "thread_ops": operations, "control": control}
        return Adaptation(canonical, tuple(events))
