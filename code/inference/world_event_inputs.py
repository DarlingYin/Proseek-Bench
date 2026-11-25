"""Load and validate sampler-local world-event inputs.

World mutation and evidence exposure are separate concerns.  Every current
Episode binds neutral ``world_event_refs``; each event then declares whether its
decisive facts are available in the Passive GUI, require an active query, or are
split across both channels.

The legacy Passive ``world_events`` and ``active_input_refs`` shapes remain
readable so archived packages can still be replayed.  A package must never mix
the legacy and current forms.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


WORLD_EVENT_INPUT_SCHEMA = "proactivebench.world-event-input.v1"
EXPOSURE_MODES = {"passive_gui", "active_query", "mixed"}
APPLICATION_MODES = {"apply_via_hook", "mirror_observed_gui_effect"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_exposure(
    exposure: Any,
    *,
    event_id: str,
) -> dict[str, Any]:
    if not isinstance(exposure, dict):
        raise ValueError(f"World event {event_id!r} must declare exposure")
    required = {
        "mode",
        "decisive_facts_passively_sufficient",
        "first_passive_observation",
        "active_query_required",
    }
    missing = sorted(required - set(exposure))
    if missing:
        raise ValueError(
            f"World event {event_id!r} exposure is missing fields: {missing}"
        )
    mode = exposure["mode"]
    if mode not in EXPOSURE_MODES:
        raise ValueError(
            f"World event {event_id!r} has unsupported exposure mode {mode!r}"
        )
    passive_sufficient = exposure["decisive_facts_passively_sufficient"]
    active_required = exposure["active_query_required"]
    first_passive = exposure["first_passive_observation"]
    if type(passive_sufficient) is not bool or type(active_required) is not bool:
        raise ValueError(
            f"World event {event_id!r} exposure flags must be booleans"
        )
    if first_passive is not None and (
        not isinstance(first_passive, str) or not first_passive
    ):
        raise ValueError(
            f"World event {event_id!r} first_passive_observation must be a "
            "non-empty string or null"
        )
    expected = {
        "passive_gui": (True, False, True),
        "active_query": (False, True, False),
        "mixed": (False, True, True),
    }[mode]
    actual = (passive_sufficient, active_required, first_passive is not None)
    if actual != expected:
        raise ValueError(
            f"World event {event_id!r} exposure fields are inconsistent with "
            f"mode {mode!r}"
        )
    return deepcopy(exposure)


def _validate_application(
    application: Any,
    *,
    event_id: str,
    activation: dict[str, Any],
) -> dict[str, Any]:
    if application is None:
        return {"mode": "apply_via_hook"}
    if not isinstance(application, dict):
        raise ValueError(f"World event {event_id!r} application must be an object")
    unknown = set(application) - {"mode", "source_observation"}
    if unknown:
        raise ValueError(
            f"World event {event_id!r} application has unknown fields: "
            f"{sorted(unknown)}"
        )
    mode = application.get("mode")
    if mode not in APPLICATION_MODES:
        raise ValueError(
            f"World event {event_id!r} has unsupported application mode {mode!r}"
        )
    source_observation = application.get("source_observation")
    if mode == "mirror_observed_gui_effect":
        if source_observation != activation.get("before_observation"):
            raise ValueError(
                f"World event {event_id!r} must mirror the GUI effect before "
                "the source observation is presented to the Agent"
            )
    elif source_observation is not None:
        raise ValueError(
            f"World event {event_id!r} apply_via_hook cannot declare "
            "source_observation"
        )
    return deepcopy(application)


def _validate_manifest(
    sampler_root: Path,
    manifest: dict[str, Any],
    *,
    source_ref: str,
) -> dict[str, Any]:
    if manifest.get("schema_version") != WORLD_EVENT_INPUT_SCHEMA:
        raise ValueError(
            f"{source_ref} must use schema_version {WORLD_EVENT_INPUT_SCHEMA!r}"
        )
    input_id = manifest.get("world_event_input_id")
    if not isinstance(input_id, str) or not input_id:
        raise ValueError(f"{source_ref} must declare world_event_input_id")
    events = manifest.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"{source_ref} must declare at least one world event")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"{source_ref} contains a non-object world event")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"{source_ref} contains an event without event_id")
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence < 1:
            raise ValueError(
                f"World event {event_id!r} sequence must be a positive integer"
            )
        update_ref = event.get("update_ref")
        if not isinstance(update_ref, str) or not update_ref:
            raise ValueError(f"World event {event_id!r} must declare update_ref")
        update_path = sampler_root / update_ref
        if not update_path.is_file():
            raise FileNotFoundError(update_path)
        activation = event.get("activation")
        if not isinstance(activation, dict):
            raise ValueError(f"World event {event_id!r} must declare activation")
        for field in ("after_observation", "before_observation"):
            if not isinstance(activation.get(field), str):
                raise ValueError(
                    f"World event {event_id!r} activation must declare {field}"
                )
        event["application"] = _validate_application(
            event.get("application"),
            event_id=event_id,
            activation=activation,
        )
        event["exposure"] = _validate_exposure(
            event.get("exposure"), event_id=event_id
        )
    validated = deepcopy(manifest)
    validated["source_ref"] = source_ref
    return validated


def load_world_event_inputs(
    sampler_root: Path,
    input_bindings: dict[str, Any],
    passive_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve current event manifests, with replay-only legacy compatibility."""

    if "world_event_refs" in input_bindings:
        if "active_input_refs" in input_bindings:
            raise ValueError(
                "Episode input_bindings cannot mix world_event_refs with "
                "active_input_refs"
            )
        if passive_input.get("world_events"):
            raise ValueError(
                "Episode cannot mix world_event_refs with Passive world_events"
            )
        refs = input_bindings["world_event_refs"]
        if not isinstance(refs, list):
            raise ValueError("world_event_refs must be an array")
        manifests = []
        seen_ids: set[str] = set()
        for ref in refs:
            if not isinstance(ref, str) or not ref:
                raise ValueError("Every world_event_ref must be a non-empty string")
            manifest = _validate_manifest(
                sampler_root,
                _read_json(sampler_root / ref),
                source_ref=ref,
            )
            input_id = manifest["world_event_input_id"]
            if input_id in seen_ids:
                raise ValueError(f"Duplicate world_event_input_id: {input_id}")
            seen_ids.add(input_id)
            manifests.append(manifest)
        return manifests

    # Archived v2 packages remain replayable.  The compatibility projection is
    # intentionally one-way: newly authored packages must use world_event_refs.
    manifests: list[dict[str, Any]] = []
    passive_events = deepcopy(passive_input.get("world_events", []))
    if passive_events:
        for event in passive_events:
            event.setdefault(
                "exposure",
                {
                    "mode": "passive_gui",
                    "decisive_facts_passively_sufficient": True,
                    "first_passive_observation": event["activation"].get(
                        "before_observation"
                    ),
                    "active_query_required": False,
                },
            )
        manifests.append(
            {
                "schema_version": "proactivebench.world-event-input.legacy",
                "world_event_input_id": "legacy-passive-world-events",
                "description": "Compatibility projection of Passive world_events.",
                "events": passive_events,
                "source_ref": input_bindings["passive_input_ref"],
            }
        )
    for index, ref in enumerate(input_bindings.get("active_input_refs", []), 1):
        legacy = _read_json(sampler_root / ref)
        events = deepcopy(legacy.get("events", []))
        for event in events:
            event.setdefault(
                "exposure",
                {
                    "mode": "active_query",
                    "decisive_facts_passively_sufficient": False,
                    "first_passive_observation": None,
                    "active_query_required": True,
                },
            )
        manifests.append(
            {
                "schema_version": "proactivebench.world-event-input.legacy",
                "world_event_input_id": legacy.get(
                    "active_input_id", f"legacy-active-{index:03d}"
                ),
                "description": legacy.get("description", ""),
                "events": events,
                "source_ref": ref,
            }
        )
    return manifests


def flatten_world_events(
    event_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        deepcopy(event)
        for event_input in event_inputs
        for event in event_input.get("events", [])
    ]
