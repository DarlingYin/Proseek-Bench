"""Resolve model families through the repository-wide provider policy.

The policy selects an endpoint and the name of a credential environment
variable.  It never reads or serializes credentials by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config/model_providers.json"


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    api_base: str
    api_key_env: str | None


def load_provider_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_provider(model: str, policy: dict[str, Any]) -> str:
    normalized = model.casefold()
    exceptions = {
        model_id.casefold(): declaration["provider"]
        for model_id, declaration in policy.get("model_route_exceptions", {}).items()
    }
    if normalized in exceptions:
        return exceptions[normalized]
    matches = [
        provider
        for provider, declaration in policy["providers"].items()
        if any(
            marker.casefold() in normalized
            for marker in declaration["model_family_markers"]
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely route model {model!r}; specify --provider explicitly"
        )
    return matches[0]


def resolve_provider_route(
    *,
    model: str,
    provider: str = "auto",
    api_base_override: str | None = None,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> ProviderRoute:
    policy = load_provider_policy(policy_path)
    registered_local_ids = {
        item["model_id"].casefold()
        for item in policy.get("registered_local_models", [])
    }
    is_registered_local = model.casefold() in registered_local_ids
    inferred_provider: str | None = None
    try:
        inferred_provider = infer_provider(model, policy)
    except ValueError:
        # Unknown/local model families may still use an explicitly configured
        # custom endpoint.  Ambiguous registered families are rejected below.
        pass
    if provider == "auto":
        if inferred_provider is None:
            raise ValueError(
                f"Cannot route model {model!r} automatically; use an unambiguous "
                "registered family or an explicit custom endpoint"
            )
        provider = inferred_provider
    if provider == "custom":
        if inferred_provider is not None and not is_registered_local:
            raise ValueError(
                f"Model family policy requires {model!r} to use "
                f"provider {inferred_provider!r}; custom routing is not allowed"
            )
        if not api_base_override:
            raise ValueError("Custom provider requires --api-base")
        return ProviderRoute(provider, api_base_override.rstrip("/"), None)
    declaration = policy["providers"].get(provider)
    if declaration is None:
        raise ValueError(f"Unknown provider {provider!r}")
    if inferred_provider is not None and provider != inferred_provider:
        raise ValueError(
            f"Model family policy requires {model!r} to use "
            f"provider {inferred_provider!r}, not {provider!r}"
        )
    return ProviderRoute(
        provider=provider,
        api_base=(api_base_override or declaration["api_base"]).rstrip("/"),
        api_key_env=declaration["api_key_env"],
    )
