#!/usr/bin/env python3
"""Online screenshot-stream runtime for ProactiveBench.

The runtime is transport-neutral: it prepares one model input at a time and
consumes one validated JSON decision.  An API/model adapter can be added on top
without changing episode state semantics.

Public model state:
  * the latest ``image_window`` observations;
  * every NeedThread, whose ``claims`` field is one editable paragraph;
  * the latest ``action_window`` compact actions;
  * the latest active-query result and the declared capability schemas.

Private runner state:
  * the complete semantic phone world;
  * scheduled app-internal updates;
  * ground truth and immutable query/audit receipts.

``search`` stays on the current observation.  ``wait`` and ``commit`` advance
one observation.  A commit freezes a prediction but does not end the stream.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable

from compile_public_tool_catalog import compile_public_catalog
from compile_agent_tool_catalog import load_agent_tool_catalog
from generic_tool_adapter import (
    GENERIC_INTERFACE, DEFAULT_TOOL_INTERFACE, INTERFACE_FORMATS,
    GenericToolAdapter, add_tool_interface_argument,
)
from query_policy import (
    DEFAULT_MAX_SEARCHES_PER_OBSERVATION,
    DEFAULT_SEARCH_BUDGET,
    QUERY_POLICY_VERSION,
    QUERY_ACCOUNTING,
    add_query_arguments,
)
from user_app_environment import (
    find_repo_root,
    load_locked_user_environment,
    resolve_repo_reference,
)
from world_event_inputs import flatten_world_events, load_world_event_inputs


WorldUpdateHook = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _require_string(value: Any, label: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    value = value.strip()
    if len(value) > max_chars:
        raise ValueError(f"{label} exceeds {max_chars} characters")
    return value


def _project(attributes: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: deepcopy(attributes[field]) for field in fields if field in attributes}


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _matches_parameter_type(value: Any, declared: str) -> bool:
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) in (int, float),
        "boolean": lambda item: type(item) is bool,
        "array": lambda item: isinstance(item, list),
        "object": lambda item: isinstance(item, dict),
    }
    return declared in checks and checks[declared](value)


@dataclass
class NeedThread:
    thread_id: str
    need: str
    claims: str
    status: str
    created_after: str
    updated_after: str
    deleted_reason: str | None = None
    committed_after: str | None = None
    committed_at: str | None = None


@dataclass(frozen=True)
class PreparedTurn:
    """One Agent turn with a hard public/private transport boundary.

    ``public_input`` is safe to serialize into the model request.
    ``image_attachments`` is runner-private transport data used only to load
    screenshot bytes; its local paths must never enter the model-visible JSON.
    """

    public_input: dict[str, Any]
    image_attachments: list[dict[str, Any]]


class SemanticQueryEngine:
    """Execute the fixed public tool schema against a private semantic world."""

    def __init__(
        self,
        semantic_catalog: dict[str, Any],
        public_catalog: dict[str, Any],
        entities: dict[str, Any],
    ) -> None:
        self.semantic_catalog = semantic_catalog
        self.public_catalog = public_catalog
        self.entities = entities
        self.declarations: dict[str, dict[str, Any]] = deepcopy(
            public_catalog["tools"]
        )
        self.internal_declarations: dict[str, dict[str, Any]] = {}
        for source in semantic_catalog["sources"].values():
            for tool in source.get("tools", []):
                self.internal_declarations[tool["tool_id"]] = tool
        self.entity_to_handle: dict[str, str] = {}
        self.handle_to_entity: dict[str, dict[str, str]] = {}
        self.handle_counters: dict[str, int] = {}

    def public_tools(self) -> dict[str, Any]:
        """Historical interface only; current generic runs never load the archive."""
        from generic_tool_adapter import legacy_public_tools
        return legacy_public_tools(self.declarations)

    @property
    def public_result_contract(self) -> dict[str, Any]:
        return deepcopy(self.public_catalog["result_contract"])

    @property
    def discovered_handles(self) -> list[str]:
        return sorted(self.handle_to_entity)

    def replace_entities(self, entities: dict[str, Any]) -> None:
        self.entities = entities

    def _validate_arguments(self, declaration: dict[str, Any], arguments: Any) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        schema = declaration["parameters"]
        parameters = schema["properties"]
        unknown = set(arguments) - set(parameters)
        if unknown:
            raise ValueError(f"Unknown tool arguments: {sorted(unknown)}")
        for name in schema.get("required", []):
            if name not in arguments:
                raise ValueError(f"Missing required tool argument: {name}")
        for name, specification in parameters.items():
            if name in arguments and not _matches_parameter_type(
                arguments[name], specification["type"]
            ):
                raise ValueError(f"Invalid type for tool argument: {name}")
            if name in arguments and specification.get("format") == "date-time":
                _parse_time(arguments[name], name)
            if name in arguments and "enum" in specification:
                if arguments[name] not in specification["enum"]:
                    raise ValueError(f"Invalid value for tool argument: {name}")
            if name in arguments and "minimum" in specification:
                if arguments[name] < specification["minimum"]:
                    raise ValueError(f"Tool argument is below minimum: {name}")
            if name in arguments and "maximum" in specification:
                if arguments[name] > specification["maximum"]:
                    raise ValueError(f"Tool argument exceeds maximum: {name}")
            if (
                name in arguments
                and "pattern" in specification
                and re.fullmatch(specification["pattern"], arguments[name]) is None
            ):
                raise ValueError(f"Invalid format for tool argument: {name}")

    @staticmethod
    def _matches_filter(
        attributes: dict[str, Any], name: str, value: Any, declaration: dict[str, Any]
    ) -> bool:
        specification = declaration.get("filters", {}).get(name)
        if specification is None:
            return name == "limit"
        fields = specification.get("fields") or [specification.get("field")]
        candidates = [attributes.get(field) for field in fields]
        operator = specification["operator"]
        if operator == "contains":
            needle = str(value).casefold()
            return any(needle in str(item or "").casefold() for item in candidates)
        if operator == "eq":
            return any(item == value for item in candidates)
        if operator in {"gte", "lte"}:
            target = _parse_time(value, name)
            datetimes = [_parse_time(item, name) for item in candidates if item is not None]
            if operator == "gte":
                return any(item >= target for item in datetimes)
            return any(item <= target for item in datetimes)
        raise ValueError(f"Unsupported query filter operator: {operator}")

    def _assign_handle(self, entity_ref: str, binding: dict[str, Any]) -> str:
        existing = self.entity_to_handle.get(entity_ref)
        if existing is not None:
            return existing
        prefix = binding["public_handle_prefix"]
        counter = self.handle_counters.get(prefix, 0) + 1
        self.handle_counters[prefix] = counter
        handle = f"{prefix}:{counter:03d}"
        self.entity_to_handle[entity_ref] = handle
        self.handle_to_entity[handle] = {
            "entity_ref": entity_ref,
            "source_id": binding["source_id"],
            "entity_type": binding["entity_type"],
        }
        return handle

    def _public_projection(
        self,
        entity_ref: str,
        attributes: dict[str, Any],
        internal: dict[str, Any],
        binding: dict[str, Any],
    ) -> dict[str, Any]:
        internal_id = binding["internal_id_field"]
        projected = _project(attributes, internal["result_fields"])
        projected.pop(internal_id, None)
        public_handle = binding.get("public_handle_field")
        if public_handle:
            projected[public_handle] = self._assign_handle(entity_ref, binding)
        return projected

    def execute(self, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_id not in self.declarations:
            raise ValueError(f"Tool is not in the fixed public catalog: {tool_id}")
        declaration = self.declarations[tool_id]
        self._validate_arguments(declaration, arguments)
        binding = declaration["runtime_binding"]
        source_id = binding["source_id"]
        entity_type = binding["entity_type"]
        internal = self.internal_declarations[binding["internal_tool_id"]]
        records = [
            (entity_ref, entity["attributes"])
            for entity_ref, entity in self.entities.items()
            if entity.get("source") == source_id
            and entity.get("entity_type") == entity_type
        ]
        access = self.semantic_catalog["sources"][source_id].get("record_access")
        if access is not None:
            # Account-scoped social records fail closed; the flag is private
            # and is not an optional Agent-controlled search filter.
            field, expected = access["field"], access["equals"]
            records = [(ref, attrs) for ref, attrs in records
                       if type(attrs.get(field)) is type(expected)
                       and attrs.get(field) == expected]
        records.sort(key=lambda item: item[0])
        cost = declaration["cost_units"]
        if declaration["operation"] == "singleton":
            if not records:
                return {"status": "not_found", "data": None, "cost_units": cost}
            if len(records) != 1:
                return {"status": "ambiguous", "data": None, "cost_units": cost}
            return {
                "status": "ok",
                "data": self._public_projection(
                    records[0][0], records[0][1], internal, binding
                ),
                "cost_units": cost,
            }
        if declaration["operation"] == "get":
            public_handle = binding["public_handle_field"]
            handle_value = arguments[public_handle]
            handle_record = self.handle_to_entity.get(handle_value)
            if handle_record is None:
                raise ValueError(
                    "Detail tools require a typed handle returned by a prior discovery call"
                )
            if (
                handle_record["source_id"] != source_id
                or handle_record["entity_type"] != entity_type
            ):
                raise ValueError(f"Handle type does not match tool {tool_id}")
            records = [
                item for item in records if item[0] == handle_record["entity_ref"]
            ]
            if len(records) != 1:
                return {"status": "not_found", "data": None, "cost_units": cost}
            return {
                "status": "ok",
                "data": self._public_projection(
                    records[0][0], records[0][1], internal, binding
                ),
                "cost_units": cost,
            }
        for name, value in arguments.items():
            if name != "limit":
                records = [
                    item
                    for item in records
                    if self._matches_filter(item[1], name, value, internal)
                ]
        limit = arguments.get("limit", internal.get("default_limit", 20))
        if type(limit) is not int or not 1 <= limit <= internal.get("max_limit", 100):
            raise ValueError("Invalid query limit")
        return {
            "status": "ok",
            "items": [
                self._public_projection(entity_ref, attributes, internal, binding)
                for entity_ref, attributes in records[:limit]
            ],
            "coverage": {"complete": len(records) <= limit, "matched": len(records)},
            "cost_units": cost,
        }


class OnlineEpisodeRuntime:
    """State machine for one online partially observable episode."""

    def __init__(
        self,
        sampler_root: Path,
        *,
        episode_id: str | None = None,
        action_window: int = 8,
        image_window: int = 3,
        search_budget: int | None = DEFAULT_SEARCH_BUDGET,
        max_searches_per_observation: int | None = DEFAULT_MAX_SEARCHES_PER_OBSERVATION,
        search_disabled_by_condition: bool = False,
        world_update_hook: WorldUpdateHook | None = None,
        tool_interface: str = DEFAULT_TOOL_INTERFACE,
    ) -> None:
        # Recorded trials pass their saved interface explicitly; new runs use generic.
        if tool_interface not in INTERFACE_FORMATS:
            raise ValueError(f"Unknown tool interface: {tool_interface}")
        self.tool_interface = tool_interface
        if action_window < 1 or image_window < 1:
            raise ValueError("action_window and image_window must be positive")
        for limit in (search_budget, max_searches_per_observation):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError("Query limits must be non-negative integers or None")
        if search_disabled_by_condition and (
            search_budget != 0 or max_searches_per_observation != 0
        ):
            raise ValueError(
                "search_disabled_by_condition requires zero Episode and per-observation query limits"
            )
        self.sampler_root = sampler_root.resolve()
        self.action_window = action_window
        self.image_window = image_window
        self.search_budget = search_budget
        self.max_searches_per_observation = max_searches_per_observation
        self.search_disabled_by_condition = search_disabled_by_condition
        self.world_update_hook = world_update_hook

        sampler = read_json(self.sampler_root / "sampler.json")
        self.repo_root = find_repo_root(self.sampler_root)
        self.user_environment = load_locked_user_environment(
            self.repo_root, sampler
        )
        task_tier = sampler.get("task_tier")
        if task_tier is not None:
            if task_tier not in {"L1", "L2", "L3", "L4"}:
                raise ValueError(f"Unsupported sampler task_tier {task_tier!r}")
            if self.sampler_root.parent.name != task_tier:
                raise ValueError(
                    f"Sampler task_tier {task_tier!r} does not match directory "
                    f"{self.sampler_root.parent.name!r}"
                )
        episode_entries = sampler.get("episodes", [])
        if not episode_entries:
            raise ValueError("Sampler has no episodes")
        if episode_id is None:
            episode_entry = episode_entries[0]
        else:
            matches = [
                entry
                for entry in episode_entries
                if entry.get("episode_id") == episode_id
            ]
            if len(matches) != 1:
                available = ", ".join(
                    str(entry.get("episode_id", "<missing>"))
                    for entry in episode_entries
                )
                raise ValueError(
                    f"Unknown episode_id {episode_id!r}; available episodes: "
                    f"{available}"
                )
            episode_entry = matches[0]
        self.episode = read_json(self.sampler_root / episode_entry["episode_ref"])
        input_bindings = self.episode["input_bindings"]
        self.passive_input = read_json(
            self.sampler_root / input_bindings["passive_input_ref"]
        )
        observation_stream = self.passive_input["observation_stream"]
        self.frames_root = (
            self.sampler_root / observation_stream["frames_root_ref"]
        ).resolve()
        self.observations = read_jsonl(
            self.sampler_root / observation_stream["stream_ref"]
        )
        self.world_event_inputs = load_world_event_inputs(
            self.sampler_root, input_bindings, self.passive_input
        )
        self.events = self._compile_world_events(
            flatten_world_events(self.world_event_inputs)
        )
        self._validate_observations()

        initial_bundle_ref = self.episode["runtime"].get(
            "compiled_initial_bundle_ref",
            "_artifacts/compiled/s000.world_bundle.json",
        )
        initial_bundle = read_json(self.sampler_root / initial_bundle_ref)
        compiled_environment_lock = initial_bundle.get("composition_lock", {}).get(
            "user_environment"
        )
        if compiled_environment_lock != self.user_environment["lock"]:
            raise ValueError(
                "Compiled S0 is stale for the Sampler's User App Environment lock; "
                "recompile the Episode"
            )
        self.entities = deepcopy(initial_bundle["semantic_world"]["entities"])
        self.world_revision = initial_bundle["world_variant_id"]

        self.catalog = deepcopy(
            self.user_environment["artifacts"]["semantic_source_catalog"]
        )
        self.public_catalog = deepcopy(
            self.user_environment["artifacts"]["public_capability_catalog"]
        )
        # The Episode binds the exact catalog exposed to the tested Agent.  New
        # generic runs therefore bind the small three-tool sidecar, whereas
        # historical legacy runs bind the archived public capability catalog.
        self.agent_tool_catalog = None
        self.agent_tool_catalog_receipt = None
        public_catalog_path = self.user_environment["artifact_paths"][
            "public_capability_catalog"
        ]
        public_catalog_sha256 = self.user_environment["lock"][
            "public_capability_catalog_sha256"
        ]
        if tool_interface == GENERIC_INTERFACE:
            (
                self.agent_tool_catalog,
                self.agent_tool_catalog_receipt,
            ) = load_agent_tool_catalog(self.user_environment["revision_path"])
            generic_catalog_path = (
                self.user_environment["revision_path"].parent
                / self.agent_tool_catalog_receipt["catalog_ref"]
            ).resolve()
            allowed_tool_catalogs = {
                generic_catalog_path: self.agent_tool_catalog_receipt[
                    "catalog_file_sha256"
                ],
                # Recorded generic trials created before the compact sidecar
                # became the Episode-level binding retain the old immutable
                # public-catalog lock and must remain replayable.
                public_catalog_path: public_catalog_sha256,
            }
        else:
            allowed_tool_catalogs = {public_catalog_path: public_catalog_sha256}
        episode_tool_catalog_ref = self.episode.get(
            "active_evidence_runtime", {}
        ).get("tool_catalog_ref")
        if episode_tool_catalog_ref is None:
            raise ValueError("Episode must bind its User App Environment tool catalog")
        resolved_tool_catalog = resolve_repo_reference(
            self.repo_root, episode_tool_catalog_ref
        )
        if resolved_tool_catalog not in allowed_tool_catalogs:
            raise ValueError(
                "Episode tool catalog does not match the locked User App Environment"
            )
        expected_tool_catalog_sha256 = allowed_tool_catalogs[resolved_tool_catalog]
        active_runtime = self.episode["active_evidence_runtime"]
        if (
            active_runtime.get("tool_catalog_sha256")
            != expected_tool_catalog_sha256
        ):
            raise ValueError("Episode tool catalog SHA256 does not match its revision")
        if (
            active_runtime.get("exposure_policy")
            != self.public_catalog["exposure_policy"]
        ):
            raise ValueError("Episode capability exposure policy is stale")
        compiled_public_catalog = compile_public_catalog(self.catalog)
        if self.public_catalog != compiled_public_catalog:
            raise ValueError(
                "Locked User App Environment tool catalog is inconsistent with "
                "its semantic source catalog"
            )
        self.query_engine = SemanticQueryEngine(
            self.catalog, self.public_catalog, self.entities
        )
        self.tool_adapter = (
            GenericToolAdapter(self.catalog, self.public_catalog)
            if tool_interface == GENERIC_INTERFACE else None
        )
        # Generic catalog loading and binding verification happen above before
        # any Episode state is exposed.  Legacy runs intentionally keep this
        # value unset and derive their archived interface from public_catalog.

        self.cursor = 0
        self.micro_turn = 0
        self.searches_this_observation = 0
        self.used_search_cost = 0
        self.query_count = 0
        self.threads: dict[str, NeedThread] = {}
        self.committed_predictions: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []
        self.audit_trace: list[dict[str, Any]] = []
        self.query_receipts: list[dict[str, Any]] = []
        self.latest_search_result: dict[str, Any] | None = None
        self.processed_event_sequences = {
            event["seq"] for event in self.events if event["type"] == "episode_initialize"
        }
        self.complete = False

    @staticmethod
    def _compile_world_events(
        world_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize declarative world updates for the runtime.

        Every update is loaded from a neutral World Event input.  Its exposure
        contract records whether decisive facts are visible in the Passive GUI,
        require an active query, or are split across both channels.  The contract
        is runner metadata and is never serialized into Agent-visible input.
        """

        events: list[dict[str, Any]] = []
        seen_sequences: set[int] = set()
        for source_event in world_events:
            sequence = source_event["sequence"]
            if type(sequence) is not int or sequence < 1:
                raise ValueError("World event sequence must be a positive integer")
            if sequence in seen_sequences:
                raise ValueError(f"Duplicate world event sequence: {sequence}")
            seen_sequences.add(sequence)
            activation = source_event["activation"]
            exposure = source_event["exposure"]
            application = source_event.get(
                "application", {"mode": "apply_via_hook"}
            )
            events.append(
                {
                    "seq": sequence,
                    "event_id": source_event["event_id"],
                    "logical_time": source_event["logical_time"],
                    "type": source_event["type"],
                    "trigger_after_observation": activation["after_observation"],
                    "trigger_after_frame": activation.get("after_capture_frame"),
                    "trigger_before_observation": activation.get(
                        "before_observation"
                    ),
                    "trigger_before_frame": activation.get(
                        "before_capture_frame"
                    ),
                    "wait_for_agent_turn": activation.get(
                        "wait_for_agent_turn", True
                    ),
                    "state_ref": source_event["update_ref"],
                    "compiled_plan_ref": source_event.get("compiled_plan_ref"),
                    "application_mode": application["mode"],
                    "source_observation": application.get("source_observation"),
                    "agent_visible": False,
                    "exposure_mode": exposure["mode"],
                    "first_passive_observation": exposure[
                        "first_passive_observation"
                    ],
                    "active_query_required": exposure["active_query_required"],
                    "foreground_policy": source_event.get(
                        "foreground_policy", "preserve"
                    ),
                    "description": source_event.get("description", ""),
                }
            )
        return sorted(events, key=lambda item: item["seq"])

    def _validate_observations(self) -> None:
        if not self.observations:
            raise ValueError("Episode has no observations")
        observation_positions: dict[str, int] = {}
        for index, observation in enumerate(self.observations):
            if observation.get("seq") != index:
                raise ValueError("Observation seq must be contiguous")
            expected_id = f"o{index:03d}"
            if observation.get("observation_id") != expected_id:
                raise ValueError(f"Expected observation_id {expected_id}")
            observation_positions[expected_id] = index
            image = self.frames_root / observation["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            _parse_time(observation["logical_time"], "logical_time")
        for event in self.events:
            after_id = event["trigger_after_observation"]
            before_id = event["trigger_before_observation"]
            if after_id not in observation_positions or before_id not in observation_positions:
                raise ValueError(
                    f"World event {event['event_id']!r} references an unknown "
                    "activation observation"
                )
            if observation_positions[after_id] >= observation_positions[before_id]:
                raise ValueError(
                    f"World event {event['event_id']!r} must activate between "
                    "ordered observations"
                )
            first_passive = event["first_passive_observation"]
            if first_passive is not None:
                if first_passive not in observation_positions:
                    raise ValueError(
                        f"World event {event['event_id']!r} references unknown "
                        f"first_passive_observation {first_passive!r}"
                    )
                if event["application_mode"] == "mirror_observed_gui_effect":
                    if (
                        first_passive != event["source_observation"]
                        or first_passive != before_id
                    ):
                        raise ValueError(
                            f"World event {event['event_id']!r} mirrored GUI effect "
                            "must become active before its source observation"
                        )
                elif observation_positions[first_passive] < observation_positions[before_id]:
                    raise ValueError(
                        f"World event {event['event_id']!r} cannot be passively "
                        "projected before it activates"
                    )

    @property
    def current_observation(self) -> dict[str, Any]:
        if self.complete:
            raise RuntimeError("Episode is complete")
        return self.observations[self.cursor]

    def prepare_turn(self) -> PreparedTurn:
        observation = self.current_observation
        start = max(0, self.cursor - self.image_window + 1)
        public_image_window: list[dict[str, Any]] = []
        private_attachments: list[dict[str, Any]] = []
        for index, item in enumerate(self.observations[start : self.cursor + 1]):
            attachment_id = f"image-{index}"
            public_image_window.append(
                {
                    "observation_id": item["observation_id"],
                    "logical_time": item["logical_time"],
                    "attachment_id": attachment_id,
                }
            )
            private_attachments.append(
                {
                    "attachment_id": attachment_id,
                    "observation_id": item["observation_id"],
                    "logical_time": item["logical_time"],
                    "image_path": str(self.frames_root / item["image"]),
                }
            )
        candidate_thread_ids = sorted(
            thread.thread_id for thread in self.threads.values() if thread.status == "candidate"
        )
        retired_thread_ids = sorted(
            thread.thread_id for thread in self.threads.values() if thread.status in {"committed", "deleted"}
        )
        public_input = {
            "protocol": "online-mobile-agent.need-thread.v1",
            "current_observation_id": observation["observation_id"],
            "current_time": observation["logical_time"],
            "micro_turn": self.micro_turn,
            "image_window": public_image_window,
            # Lifecycle timestamps are evaluator/runtime bookkeeping, not fields the Agent may
            # mutate. Omitting them reduces accidental copying into strict thread_ops while status
            # remains visible so the Agent can avoid updating or duplicating committed threads.
            "need_threads": [
                {
                    "thread_id": thread.thread_id,
                    "need": thread.need,
                    "claims": thread.claims,
                    "status": thread.status,
                }
                for thread in self.threads.values()
            ],
            "thread_lifecycle": {
                "thread_ids_are_episode_unique": True,
                "candidate_thread_ids": candidate_thread_ids,
                "retired_thread_ids": retired_thread_ids,
                "create": "Use a fresh thread_id that has never appeared in this Episode.",
                "update_delete_commit": "Use candidate_thread_ids only. Committed or deleted IDs are retired permanently.",
            },
            "action_history": deepcopy(self.action_history[-self.action_window :]),
            "latest_search_result": deepcopy(self.latest_search_result),
            "search_ledger": [
                {key: deepcopy(receipt[key]) for key in
                 ("receipt_id", "observation_id", "as_of", "tool", "arguments", "status", "coverage", "linked_thread_ids")
                 if key in receipt}
                for receipt in self.query_receipts[-64:]
            ],
            "search_ledger_omitted": max(0, len(self.query_receipts) - 64),
            "runtime_tool_usage": self.tool_adapter.usage_contract() if self.tool_adapter else {},
            "claims_contract": {
                "candidate": "Compact working memory; preserve confirmed facts, relevant query evidence, "
                             "hypotheses and material unresolved questions. Distinguish facts from inferences. "
                             "More detail than final Claims is allowed; replace the evidence summary rather than append a diary.",
                "commit": "Synthesize before commit. Need: one concrete action goal + target object + necessary "
                          "parameters/conditions. Claims: observed facts + relevant active-query findings if any + "
                          "why help remains unmet + why now. Use one concise coherent causal paragraph of at most 900 "
                          "characters. Preserve material constraints and uncertainty; remove observation/query IDs, query "
                          "diaries, rejected alternatives, resolved questions and next-step notes. Do not invent facts, "
                          "queries or urgency to fill these semantic slots.",
                "validation": "At Commit, Runtime rejects Final Claims over 900 characters or containing internal "
                              "observation/query IDs. It does not judge semantic salience or rewrite/truncate text.",
            },
            "discovered_handles": self.query_engine.discovered_handles,
            "search_budget": self.search_availability(),
            "query_policy": {"version": QUERY_POLICY_VERSION, "accounting": QUERY_ACCOUNTING,
                             "duplicates": "Same-frame identical results may be reused. Other queries remain allowed. "
                                           "Use returned evidence; repeated requests still consume budget."},
            "tool_result_contract": self.query_engine.public_result_contract,
            "available_tools": (
                deepcopy(self.agent_tool_catalog) if self.tool_adapter is not None
                else self.query_engine.public_tools()
            ),
            "allowed_thread_ops": ["create", "update", "delete"],
            "allowed_control_actions": (
                ["search", "wait", "commit"]
                if self._search_unavailable_reason() is None
                else ["wait", "commit"]
            ),
        }
        return PreparedTurn(
            public_input=public_input,
            image_attachments=private_attachments,
        )

    def model_input(self) -> dict[str, Any]:
        """Return only the JSON-safe Agent-visible state for the current turn."""

        return self.prepare_turn().public_input

    def _apply_thread_ops(self, operations: Any) -> list[dict[str, Any]]:
        if not isinstance(operations, list):
            raise ValueError("thread_ops must be an array")
        observation_id = self.current_observation["observation_id"]
        applied: list[dict[str, Any]] = []
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("Each thread operation must be an object")
            kind = operation.get("op")
            thread_id = _require_string(
                operation.get("thread_id"), "thread_id", max_chars=64
            )
            if kind == "create":
                if set(operation) != {"op", "thread_id", "need", "claims"}:
                    raise ValueError("create has unexpected or missing fields")
                if thread_id in self.threads:
                    status = self.threads[thread_id].status
                    candidates = sorted(k for k, value in self.threads.items() if value.status == "candidate")
                    retired = sorted(k for k, value in self.threads.items() if value.status in {"committed", "deleted"})
                    raise ValueError(
                        f"Thread ID {thread_id!r} was already used with status {status!r} and is episode-unique; "
                        f"never reuse committed or deleted IDs. Candidate thread IDs: {candidates}; "
                        f"retired thread IDs: {retired}. Use a fresh ID only for a genuinely new candidate."
                    )
                need = _require_string(operation.get("need"), "need", max_chars=500)
                claims = _require_string(
                    operation.get("claims"), "claims", max_chars=2000
                )
                self.threads[thread_id] = NeedThread(
                    thread_id=thread_id,
                    need=need,
                    claims=claims,
                    status="candidate",
                    created_after=observation_id,
                    updated_after=observation_id,
                )
            elif kind == "update":
                allowed = {"op", "thread_id", "need", "claims"}
                unexpected = sorted(set(operation) - allowed)
                if unexpected:
                    raise ValueError(
                        "update has unexpected fields "
                        f"{unexpected}; allowed fields are {sorted(allowed)}"
                    )
                if thread_id not in self.threads:
                    raise ValueError(f"Unknown thread: {thread_id}")
                thread = self.threads[thread_id]
                if thread.status != "candidate":
                    candidates = sorted(k for k, value in self.threads.items() if value.status == "candidate")
                    retired = sorted(k for k, value in self.threads.items() if value.status in {"committed", "deleted"})
                    raise ValueError(
                        f"Only candidate threads can be updated; {thread_id!r} has status {thread.status!r}. "
                        f"Allowed candidate thread IDs: {candidates}; retired thread IDs: {retired}."
                    )
                if "need" not in operation and "claims" not in operation:
                    raise ValueError("update must replace need and/or claims")
                if "need" in operation:
                    thread.need = _require_string(
                        operation["need"], "need", max_chars=500
                    )
                if "claims" in operation:
                    thread.claims = _require_string(
                        operation["claims"], "claims", max_chars=2000
                    )
                thread.updated_after = observation_id
            elif kind == "delete":
                if set(operation) != {"op", "thread_id", "reason"}:
                    raise ValueError("delete has unexpected or missing fields")
                if thread_id not in self.threads:
                    raise ValueError(f"Unknown thread: {thread_id}")
                thread = self.threads[thread_id]
                if thread.status != "candidate":
                    candidates = sorted(k for k, value in self.threads.items() if value.status == "candidate")
                    retired = sorted(k for k, value in self.threads.items() if value.status in {"committed", "deleted"})
                    raise ValueError(
                        f"Only candidate threads can be deleted; {thread_id!r} has status {thread.status!r}. "
                        f"Allowed candidate thread IDs: {candidates}; retired thread IDs: {retired}."
                    )
                thread.status = "deleted"
                thread.deleted_reason = _require_string(
                    operation.get("reason"), "delete reason", max_chars=500
                )
                thread.updated_after = observation_id
            else:
                raise ValueError(f"Unsupported thread operation: {kind}")
            applied.append({"op": kind, "thread_id": thread_id})
        return applied

    def _search_unavailable_reason(self, cost: int | None = None) -> str | None:
        if not self.query_engine.declarations:
            return "no_query_tools"
        if self.search_disabled_by_condition:
            return "search_disabled_by_condition"
        minimum_cost = min(
            item["cost_units"] for item in self.query_engine.declarations.values()
        ) if cost is None else cost
        if self.search_budget is not None:
            remaining = self.search_budget - self.used_search_cost
            if minimum_cost > remaining:
                return (
                    "query_budget_exhausted" if remaining <= 0
                    else "insufficient_query_budget"
                )
        if (
            self.max_searches_per_observation is not None
            and self.searches_this_observation >= self.max_searches_per_observation
        ):
            return "per_observation_limit"
        return None

    def search_availability(self) -> dict[str, Any]:
        reason = self._search_unavailable_reason()
        return {
            "mode": "unlimited" if self.search_budget is None else "capped",
            "used_cost_units": self.used_search_cost,
            "remaining_cost_units": (
                None if self.search_budget is None
                else self.search_budget - self.used_search_cost
            ),
            "searches_this_observation": self.searches_this_observation,
            "max_searches_per_observation": self.max_searches_per_observation,
            "available": reason is None,
            "unavailable_reason": reason,
        }

    def _search(self, control: dict[str, Any]) -> dict[str, Any]:
        self._validate_control_keys(
            control, {"type", "tool", "arguments", "linked_thread_ids"}
        )
        tool = _require_string(control.get("tool"), "search tool", max_chars=128)
        arguments = control.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        linked = control.get("linked_thread_ids", [])
        if not isinstance(linked, list) or any(item not in self.threads for item in linked):
            raise ValueError("linked_thread_ids must reference existing threads")
        if len(linked) != len(set(linked)):
            raise ValueError("linked_thread_ids contains duplicates")
        resolved_tool, resolved_arguments = (
            self.tool_adapter.resolve(tool, arguments, self.query_engine.handle_to_entity, self.entities)
            if self.tool_adapter is not None else (tool, arguments)
        )
        tool_declaration = self.query_engine.declarations.get(resolved_tool)
        if tool_declaration is None:
            raise ValueError(f"Tool is not available: {tool}")
        predicted_cost = tool_declaration["cost_units"]
        unavailable_reason = self._search_unavailable_reason(predicted_cost)
        if unavailable_reason is not None:
            # Resource refusals contain no world evidence. They are ordinary runtime
            # feedback, not malformed model output, and must not abort the stream.
            refusal = {
                "status": "unavailable",
                "tool": tool,
                "reason": unavailable_reason,
                "cost_units": 0,
                "executed": False,
                "as_of": self.current_observation["logical_time"],
                "search_availability": self.search_availability(),
            }
            self.latest_search_result = refusal
            return deepcopy(refusal)
        # Validate before any cache hit. Deduplication is confined to one observation;
        # a later frame may legitimately refresh unchanged-looking public state.
        self.query_engine._validate_arguments(tool_declaration, resolved_arguments)
        def signature(name, args):
            normalized = deepcopy(args)
            binding = self.query_engine.declarations[name]['runtime_binding']
            internal = self.query_engine.internal_declarations[binding['internal_tool_id']]
            if self.query_engine.declarations[name]['role'] == 'discovery':
                normalized.setdefault('limit', internal.get('default_limit', 20))
            return name, json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        key = signature(resolved_tool, resolved_arguments)
        result = None
        for earlier in reversed(self.query_receipts):
            if earlier['observation_id'] != self.current_observation['observation_id']:
                break
            if earlier['world_revision'] != self.world_revision:
                continue
            if signature(earlier.get('resolved_tool', earlier['tool']),
                         earlier.get('resolved_arguments', earlier['arguments'])) != key:
                continue
            if earlier['status'] != 'ok':
                continue
            result = {k: deepcopy(earlier[k]) for k in ('status', 'items', 'data', 'coverage') if k in earlier}
            result.update({'executed': False, 'cost_units': predicted_cost, 'duplicate_query': True,
                           'cached_from_receipt': earlier.get('cached_from_receipt', earlier['receipt_id']),
                           'feedback': 'Identical query already answered at this observation. Use this result. '
                                       'Other legal queries remain allowed; this request still consumes its declared cost.'})
            break
        if result is None:
            result = self.query_engine.execute(resolved_tool, resolved_arguments)
            result['executed'] = True
        self.query_count += 1
        self.searches_this_observation += 1
        self.used_search_cost += result["cost_units"]
        receipt_id = f"q{self.query_count:03d}"
        public_receipt = {
            "receipt_id": receipt_id,
            "tool": tool,
            "arguments": deepcopy(arguments),
            "as_of": self.current_observation["logical_time"],
            **deepcopy(result),
        }
        relation_repairs = [f'resolve_relation_handle:{field}'
                            for field, value in arguments.get('filters', {}).items()
                            if self.tool_adapter is not None and isinstance(value, str)
                            and value in self.query_engine.handle_to_entity
                            and self.tool_adapter.relation_targets(arguments.get('resource', ''), field)
                            and resolved_arguments.get(field) != value]
        if relation_repairs:
            from compatibility_rules import rule_ids
            public_receipt['interface_repairs'] = relation_repairs
            public_receipt['compatibility_rule_ids'] = rule_ids(relation_repairs)
        private_receipt = {
            **deepcopy(public_receipt),
            "world_revision": self.world_revision,
            "observation_id": self.current_observation["observation_id"],
            "linked_thread_ids": deepcopy(linked),
        }
        if self.tool_adapter is not None:
            private_receipt["resolved_tool"] = resolved_tool
            private_receipt["resolved_arguments"] = deepcopy(resolved_arguments)
        self.query_receipts.append(private_receipt)
        self.latest_search_result = public_receipt
        return public_receipt

    def _commit(self, control: dict[str, Any]) -> list[str]:
        thread_ids = control.get("thread_ids")
        if not isinstance(thread_ids, list) or not thread_ids:
            raise ValueError("commit.thread_ids must be a non-empty array")
        if len(thread_ids) != len(set(thread_ids)):
            raise ValueError("commit.thread_ids contains duplicates")
        observation = self.current_observation
        candidates = sorted(k for k, value in self.threads.items() if value.status == "candidate")
        retired = sorted(k for k, value in self.threads.items() if value.status in {"committed", "deleted"})
        from runtime_diagnostics import validate_final_claims
        for thread_id in thread_ids:
            if thread_id not in self.threads:
                raise ValueError(
                    f"Unknown thread: {thread_id}. Allowed candidate thread IDs: {candidates}; "
                    f"retired thread IDs: {retired}."
                )
            thread = self.threads[thread_id]
            if thread.status != "candidate":
                raise ValueError(
                    f"Only candidate threads can be committed; {thread_id!r} has status {thread.status!r}. "
                    f"Allowed candidate thread IDs: {candidates}; retired thread IDs: {retired}."
                )
            try:
                validate_final_claims(thread.claims)
            except ValueError as exc:
                raise ValueError(f"Final Claims for thread {thread_id!r} are invalid. {exc}") from exc
        committed: list[str] = []
        for thread_id in thread_ids:
            thread = self.threads[thread_id]
            thread.status = "committed"
            thread.committed_after = observation["observation_id"]
            thread.committed_at = observation["logical_time"]
            thread.updated_after = observation["observation_id"]
            prediction = {
                "thread_id": thread.thread_id,
                "need": thread.need,
                "claims": thread.claims,
                "committed_after": thread.committed_after,
                "committed_at": thread.committed_at,
            }
            self.committed_predictions.append(deepcopy(prediction))
            committed.append(thread_id)
        return committed

    def apply_agent_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Apply one decision atomically to the in-memory benchmark state.

        Model protocol errors must not leave behind half-applied thread edits or
        newly issued handles, because an API adapter may ask the model to repair
        the same decision. External hook side effects are outside this logical
        transaction; hooks should therefore be idempotent.
        """

        state = {
            "entities": deepcopy(self.entities),
            "world_revision": self.world_revision,
            "cursor": self.cursor,
            "micro_turn": self.micro_turn,
            "searches_this_observation": self.searches_this_observation,
            "used_search_cost": self.used_search_cost,
            "query_count": self.query_count,
            "threads": deepcopy(self.threads),
            "committed_predictions": deepcopy(self.committed_predictions),
            "action_history": deepcopy(self.action_history),
            "audit_trace": deepcopy(self.audit_trace),
            "query_receipts": deepcopy(self.query_receipts),
            "latest_search_result": deepcopy(self.latest_search_result),
            "processed_event_sequences": deepcopy(self.processed_event_sequences),
            "complete": self.complete,
            "entity_to_handle": deepcopy(self.query_engine.entity_to_handle),
            "handle_to_entity": deepcopy(self.query_engine.handle_to_entity),
            "handle_counters": deepcopy(self.query_engine.handle_counters),
        }
        try:
            return self._apply_agent_output(output)
        except Exception:
            for name in (
                "entities",
                "world_revision",
                "cursor",
                "micro_turn",
                "searches_this_observation",
                "used_search_cost",
                "query_count",
                "threads",
                "committed_predictions",
                "action_history",
                "audit_trace",
                "query_receipts",
                "latest_search_result",
                "processed_event_sequences",
                "complete",
            ):
                setattr(self, name, state[name])
            self.query_engine.entity_to_handle = state["entity_to_handle"]
            self.query_engine.handle_to_entity = state["handle_to_entity"]
            self.query_engine.handle_counters = state["handle_counters"]
            self.query_engine.replace_entities(self.entities)
            raise

    def _apply_agent_output(self, output: dict[str, Any]) -> dict[str, Any]:
        if self.complete:
            raise RuntimeError("Episode is complete")
        if not isinstance(output, dict) or set(output) != {"thread_ops", "control"}:
            raise ValueError("Agent output must contain only thread_ops and control")
        prior_claims_lengths = {key: len(t.claims) for key, t in self.threads.items()}
        applied_ops = self._apply_thread_ops(output["thread_ops"])
        control = output["control"]
        if not isinstance(control, dict):
            raise ValueError("control must be an object")
        control_type = control.get("type")
        record: dict[str, Any] = {
            "turn": len(self.action_history),
            "observation_id": self.current_observation["observation_id"],
            "micro_turn": self.micro_turn,
            "thread_ops": applied_ops,
            "control": control_type,
        }
        response: dict[str, Any] = {"status": "ok", "control": control_type}
        if control_type == "search":
            receipt = self._search(control)
            record.update(
                {
                    "tool": receipt["tool"],
                    "receipt_id": receipt.get("receipt_id"),
                    "query_status": receipt["status"],
                    "cost_units": receipt["cost_units"],
                    "duplicate_query": receipt.get("duplicate_query", False),
                }
            )
            response["search_result"] = deepcopy(receipt)
            if receipt["status"] == "unavailable":
                response["status"] = "unavailable"
                record["unavailable_reason"] = receipt["reason"]
            self.micro_turn += 1
        elif control_type == "wait":
            self._validate_control_keys(control, {"type"})
            self._advance_observation()
        elif control_type == "commit":
            self._validate_control_keys(control, {"type", "thread_ids"})
            from runtime_diagnostics import claims_measurement
            final_claims = {key: t.claims for key, t in self.threads.items()}
            committed = self._commit(control)
            record["thread_ids"] = committed
            record["final_claims_checks"] = [
                {"thread_id": thread_id,
                 "candidate_characters_before_decision": prior_claims_lengths.get(thread_id),
                 "same_turn_rewrite": any(op.get('thread_id') == thread_id and 'claims' in op
                                          for op in output['thread_ops']),
                 **claims_measurement(final_claims[thread_id])}
                for thread_id in committed
            ]
            response["committed_thread_ids"] = committed
            self._advance_observation()
        else:
            raise ValueError(f"Unsupported control action: {control_type}")
        self.action_history.append(record)
        self.audit_trace.append(
            {
                "input_observation_id": record["observation_id"],
                "agent_output": deepcopy(output),
                "runtime_record": deepcopy(record),
            }
        )
        response["episode_complete"] = self.complete
        if not self.complete:
            response["current_observation_id"] = self.current_observation["observation_id"]
        return response

    @staticmethod
    def _validate_control_keys(control: dict[str, Any], expected: set[str]) -> None:
        if set(control) != expected:
            raise ValueError(f"Control fields mismatch: missing={sorted(expected - set(control))}; "
                             f"unexpected={sorted(set(control) - expected)}")

    def _advance_observation(self) -> None:
        completed_observation = self.current_observation["observation_id"]
        self._apply_events_after(completed_observation)
        self.cursor += 1
        self.micro_turn = 0
        self.searches_this_observation = 0
        self.latest_search_result = None
        if self.cursor >= len(self.observations):
            self.complete = True

    def _apply_events_after(self, observation_id: str) -> None:
        for event in self.events:
            if event["seq"] in self.processed_event_sequences:
                continue
            if event.get("trigger_after_observation") != observation_id:
                continue
            source_ref = event.get("state_ref")
            if not source_ref:
                raise ValueError("World update event is missing state_ref")
            source = read_json(self.sampler_root / source_ref)
            next_entities = deepcopy(self.entities)
            for patch in source["transition"]["patches"]:
                operation = patch["op"]
                entity_ref = patch["entity_ref"]
                if operation == "add":
                    if entity_ref in next_entities:
                        raise ValueError(f"Cannot add existing entity: {entity_ref}")
                    next_entities[entity_ref] = deepcopy(patch["entity"])
                elif operation == "replace":
                    if entity_ref not in next_entities:
                        raise ValueError(f"Cannot replace missing entity: {entity_ref}")
                    next_entities[entity_ref] = deepcopy(patch["entity"])
                elif operation == "remove":
                    if entity_ref not in next_entities:
                        raise ValueError(f"Cannot remove missing entity: {entity_ref}")
                    del next_entities[entity_ref]
                else:
                    raise ValueError(f"Unsupported world patch operation: {operation}")
            hook_receipt = None
            if (
                self.world_update_hook is not None
                and event["application_mode"] == "apply_via_hook"
            ):
                hook_receipt = self.world_update_hook(deepcopy(event), deepcopy(source))
                if hook_receipt.get("status") != "success":
                    raise RuntimeError("External world update hook failed")
            self.entities = next_entities
            self.world_revision = source["step_id"]
            self.query_engine.replace_entities(self.entities)
            self.processed_event_sequences.add(event["seq"])
            self.audit_trace.append(
                {
                    "type": "runner_private_world_update",
                    "after_observation_id": observation_id,
                    "logical_time": event["logical_time"],
                    "world_revision": self.world_revision,
                    "application_mode": event["application_mode"],
                    "hook_receipt": deepcopy(hook_receipt),
                }
            )

    def result(self) -> dict[str, Any]:
        from runtime_diagnostics import RUNTIME_IMPLEMENTATION, claims_measurement
        return {
            "schema_version": "proactivebench.online-run-result.v1",
            "episode_id": self.episode["episode_id"],
            "status": "complete" if self.complete else "running",
            "configuration": {
                "runtime_implementation": RUNTIME_IMPLEMENTATION,
                "runtime_tool_usage": self.tool_adapter.usage_contract() if self.tool_adapter else {},
                "query_policy_version": QUERY_POLICY_VERSION,
                "query_accounting": QUERY_ACCOUNTING,
                "tool_interface": self.tool_interface,
                "tool_catalog_format": INTERFACE_FORMATS[self.tool_interface],
                "derived_tool_catalog_sha256": (
                    self.tool_adapter.catalog_sha256 if self.tool_adapter else None
                ),
                "agent_tool_catalog_binding": deepcopy(self.agent_tool_catalog_receipt),
                "action_window": self.action_window,
                "image_window": self.image_window,
                "search_budget": self.search_budget,
                "max_searches_per_observation": self.max_searches_per_observation,
            },
            "predictions": deepcopy(self.committed_predictions),
            "query_cost": {
                "query_count": self.query_count,
                "cost_units": self.used_search_cost,
                "budget": self.search_budget,
                "unavailable_attempts": sum(
                    item.get("query_status") == "unavailable"
                    for item in self.action_history
                ),
                "duplicate_cache_hits": sum(bool(item.get('duplicate_query')) for item in self.action_history),
                "backend_executions": sum(r.get('executed', True) for r in self.query_receipts),
                "backend_cost_units": sum(r['cost_units'] for r in self.query_receipts if r.get('executed', True)),
                "accounting": "accepted_query_attempts_including_cache_hits",
                "relation_handle_resolutions": sum(len(r.get('interface_repairs', [])) for r in self.query_receipts),
            },
            "claims_diagnostics": {
                "committed": [claims_measurement(p['claims']) for p in self.committed_predictions],
                "candidates": [{"thread_id": t.thread_id, **claims_measurement(t.claims, stage='candidate')}
                               for t in self.threads.values() if t.status == 'candidate'],
            },
            "need_threads": [asdict(thread) for thread in self.threads.values()],
            "query_receipts": deepcopy(self.query_receipts),
            "action_history": deepcopy(self.action_history),
            "audit_trace": deepcopy(self.audit_trace),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sampler_root", type=Path)
    parser.add_argument(
        "--episode-id",
        help="Episode to inspect; defaults to the first sampler episode.",
    )
    parser.add_argument("--show-input", action="store_true")
    parser.add_argument("--action-window", type=int, default=8)
    parser.add_argument("--image-window", type=int, default=3)
    add_query_arguments(parser)
    add_tool_interface_argument(parser)
    args = parser.parse_args()
    runtime = OnlineEpisodeRuntime(
        args.sampler_root,
        episode_id=args.episode_id,
        action_window=args.action_window,
        image_window=args.image_window,
        search_budget=args.search_budget,
        max_searches_per_observation=args.max_searches_per_observation,
        tool_interface=args.tool_interface,
    )
    value = runtime.model_input() if args.show_input else runtime.result()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
