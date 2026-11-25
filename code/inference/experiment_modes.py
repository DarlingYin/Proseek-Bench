"""Scientific conditions; protocol adapters are independent compatibility settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from query_policy import parse_query_limit
from prompt_input import COMPACT_CONTRACTS, FULL_STATE, INPUT_CONTRACTS

MODES = ("framework",)
UNSET = object()


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-mode", choices=MODES,
                        help="Scientific condition, independent of --protocol-adapter.")
    parser.add_argument("--prompt-version", help="Registered prompt version within the selected mode (e.g. v1, v2).")
    parser.add_argument("--search-budget", type=parse_query_limit, default=UNSET)
    parser.add_argument("--max-searches-per-observation", type=parse_query_limit, default=UNSET)


def resolve_mode(args: argparse.Namespace, repo_root: Path) -> dict:
    source = "explicit" if args.inference_mode else "default"
    mode = args.inference_mode or "framework"
    if mode not in MODES:
        raise ValueError(f"Unsupported NeedThread release mode: {mode}")
    prompt_root = repo_root / "prompts/inference"
    registry = json.loads((prompt_root / "modes.json").read_text(encoding="utf-8"))
    # Preserve the exact serializer/order used by historical Framework runs.
    if registry.get("assembly_order") != ["system", "policy", "search_protocol", "decision_schema", "static_tool_context"]:
        raise ValueError("Unsupported prompt assembly order; migrate the renderer explicitly")
    entry = registry["modes"][mode]
    version = args.prompt_version or entry["default_version"]
    if version not in entry["versions"]:
        raise ValueError(f"Unknown prompt version {version!r} for {mode}; available: {sorted(entry['versions'])}")
    path = (prompt_root / entry["versions"][version]).resolve()
    if not path.is_relative_to(prompt_root.resolve()):
        raise ValueError("Prompt profile must stay within prompts/inference")
    args.prompt_version = version
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile["inference_mode"] != mode:
        raise ValueError("Prompt profile and inference mode disagree")
    if profile.get("memory_mode") != "need_threads":
        raise ValueError("Prompt memory mode and Runtime implementation disagree")
    if set(profile["prompts"]) != {"system", "policy", "search_protocol"}:
        raise ValueError("Prompt profile must declare system, policy and search_protocol")
    input_contract = profile.get("prompt_input_contract", FULL_STATE)
    if input_contract not in INPUT_CONTRACTS:
        raise ValueError(f"Unknown prompt input contract: {input_contract}")
    if input_contract == COMPACT_CONTRACTS and getattr(args, "tool_interface", "generic") != "generic":
        raise ValueError(
            "Compact prompt profile requires a registered mode with the generic tool interface"
        )
    for name in ("search_budget", "max_searches_per_observation"):
        value = getattr(args, name)
        if value == UNSET:
            value = profile[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"Invalid {name}: {value!r}")
        if value == 0:
            raise ValueError(f"Framework requires a positive or unlimited {name}")
        setattr(args, name, value)
    args.inference_mode = mode
    prompts = {}
    for key, relative in profile["prompts"].items():
        path_value = (path.parent / relative).resolve()
        if not path_value.is_relative_to((repo_root / "prompts").resolve()):
            raise ValueError("Prompt components must stay within prompts")
        prompts[key] = {"path": str(path_value.relative_to(repo_root.resolve())),
                        "sha256": hashlib.sha256(path_value.read_bytes()).hexdigest()}
    schema_path = ((path.parent / profile["decision_schema"]).resolve() if "decision_schema" in profile
                   else repo_root / "prompts/schemas/online_agent_decision.v1.json")
    return {"schema_version": "proseekbench.inference-condition.v1",
            "inference_mode": mode, "resolved_from": source,
            "prompt_version": version,
            "profile_path": str(path.relative_to(repo_root.resolve())),
            "assembly_order": registry["assembly_order"],
            "memory_mode": profile.get("memory_mode", "need_threads"),
            "decision_schema": {"path": str(schema_path.relative_to(repo_root.resolve())),
                                "sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest()},
            "profile_version": profile["version"],
            "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "search_budget": args.search_budget,
            "max_searches_per_observation": args.max_searches_per_observation,
            **({"prompt_input_contract": input_contract} if "prompt_input_contract" in profile else {}),
            "prompts": prompts}


def check_no_search_result(mode: str, result: dict) -> None:
    if mode != "no_search":
        return
    config, cost = result["configuration"], result["query_cost"]
    if (config["search_budget"] != 0 or config["max_searches_per_observation"] != 0
            or cost["query_count"] != 0 or cost["cost_units"] != 0):
        raise RuntimeError("NoSearch invariant violated; refusing to publish prediction")


def batch_contract(args: argparse.Namespace, condition: dict, repo_root: Path) -> dict:
    """Immutable scientific/execution settings for safe same-directory resume."""
    keys = ("model", "request_model", "provider", "api_base", "codex_bin", "codex_version", "action_window", "image_window",
            "search_budget", "max_searches_per_observation", "tool_interface", "max_model_calls",
            "protocol_error_policy", "max_protocol_repairs", "request_timeout", "protocol_adapter",
            "temperature", "max_tokens", "response_format", "thinking_mode", "reasoning_effort")
    profile = {k: v for k, v in condition.items() if k != "resolved_from"}
    code = repo_root / "code/inference"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob("*.py"))
              if not p.name.startswith(("._", "test_"))}
    schema = repo_root / condition["decision_schema"]["path"]
    from runtime_diagnostics import RUNTIME_IMPLEMENTATION
    from compatibility_rules import REGISTRY_VERSION, registry_hash
    from query_policy import QUERY_POLICY_VERSION
    return {"condition": profile, "settings": {key: getattr(args, key) for key in keys},
            "runtime_contract": {"implementation": RUNTIME_IMPLEMENTATION, "query_policy": QUERY_POLICY_VERSION,
                                 "compatibility_registry": REGISTRY_VERSION, "registry_sha256": registry_hash()},
            "code_sha256": hashes, "decision_schema_sha256": hashlib.sha256(schema.read_bytes()).hexdigest()}


def validate_resume(manifest: dict, contract: dict, identities: list, release_hash: str) -> None:
    if manifest.get("run_configuration") != contract:
        raise ValueError("Resume configuration differs or is unversioned. Use a new run-id; historical runs are not migrated automatically.")
    if manifest.get("planned_samplers") != identities or manifest.get("release_sha256sums_sha256") != release_hash:
        raise ValueError("Resume dataset/selection differs. Use a new run-id.")


def validate_transport_only_resume(
    manifest: dict, contract: dict, identities: list, release_hash: str
) -> None:
    """Allow only the audited urllib/OpenSSL to curl/SecureTransport migration."""
    if manifest.get("planned_samplers") != identities or manifest.get("release_sha256sums_sha256") != release_hash:
        raise ValueError("Resume dataset/selection differs. Use a new run-id.")
    recorded = manifest.get("run_configuration")
    if not isinstance(recorded, dict):
        raise ValueError("Transport migration requires a versioned prior run configuration")
    old = dict(recorded)
    new = dict(contract)
    old_code = dict(old.pop("code_sha256", {}))
    new_code = dict(new.pop("code_sha256", {}))
    if old != new:
        raise ValueError("Transport migration changes non-transport run settings")
    if old_code == new_code:
        return
    changed = {name for name in old_code.keys() & new_code.keys() if old_code[name] != new_code[name]}
    added = set(new_code) - set(old_code)
    removed = set(old_code) - set(new_code)
    allowed_changed = {"run_online_agent.py", "run_batch.py", "experiment_modes.py"}
    if changed != allowed_changed or added != {"curl_transport.py"} or removed:
        raise ValueError(
            "Transport migration source diff is not the audited urllib-to-curl change: "
            f"changed={sorted(changed)}, added={sorted(added)}, removed={sorted(removed)}"
        )
