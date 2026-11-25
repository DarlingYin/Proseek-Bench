"""Versioned request presentation; never changes Runtime state or tool results."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


FULL_STATE = "full-state-v1"
COMPACT_CONTRACTS = "compact-contracts-v1"
INPUT_CONTRACTS = (FULL_STATE, COMPACT_CONTRACTS)
BASE_STATIC_FIELDS = ("tool_result_contract", "available_tools", "allowed_thread_ops")
COMPACT_STATIC_FIELDS = ("runtime_tool_usage", "query_policy")
REPAIR_FIELDS = (
    "protocol", "current_observation_id", "current_time", "micro_turn",
    "need_threads", "latest_search_result", "discovered_handles",
    "submitted_needs",
    "search_budget", "allowed_control_actions", "bound_tool_context_sha256",
    "runtime_tool_usage", "query_policy", "claims_contract", "search_ledger", "thread_lifecycle",
)


def _validate_contract(input_contract: str) -> None:
    if input_contract not in INPUT_CONTRACTS:
        raise ValueError(f"Unknown prompt input contract: {input_contract}")


def build_static_tool_context(
    model_input: dict[str, Any], *, catalog_sha256: str,
    input_contract: str = FULL_STATE,
) -> dict[str, Any]:
    _validate_contract(input_contract)
    fields = BASE_STATIC_FIELDS
    if input_contract == COMPACT_CONTRACTS:
        fields += COMPACT_STATIC_FIELDS
    return {"catalog_sha256": catalog_sha256,
            **{key: deepcopy(model_input[key]) for key in fields if key in model_input}}


def render_prompt_state(
    model_input: dict[str, Any], *, static_tool_context: dict[str, Any],
    static_tool_context_sha256: str, input_contract: str = FULL_STATE,
    repair_only: bool = False,
) -> dict[str, Any]:
    _validate_contract(input_contract)
    expected = {key: value for key, value in static_tool_context.items()
                if key != "catalog_sha256"}
    if any(key not in model_input or model_input[key] != value
           for key, value in expected.items()):
        raise RuntimeError("Runtime tool context changed within one bound Episode")
    excluded = set(BASE_STATIC_FIELDS)
    if input_contract == COMPACT_CONTRACTS:
        if any(key not in expected for key in COMPACT_STATIC_FIELDS):
            raise ValueError("Compact input requires bound runtime_tool_usage and query_policy")
        # The complete Claims policy is in this profile's system prompt. Keep
        # actual candidate/final Claims and all other evidence in the turn state.
        excluded.update((*COMPACT_STATIC_FIELDS, "claims_contract"))
    state = {key: deepcopy(value) for key, value in model_input.items() if key not in excluded}
    state["bound_tool_context_sha256"] = static_tool_context_sha256
    if repair_only:
        state = {key: state[key] for key in REPAIR_FIELDS if key in state}
        if input_contract == COMPACT_CONTRACTS and "search_ledger_omitted" in model_input:
            state["search_ledger_omitted"] = model_input["search_ledger_omitted"]
    return state
