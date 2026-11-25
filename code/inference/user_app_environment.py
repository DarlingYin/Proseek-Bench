#!/usr/bin/env python3
"""Resolve and verify one immutable User App Environment revision.

The Platform directory is an authoring registry.  A registered Sampler must not
read mutable Platform catalogs at runtime: it binds one revision manifest, and
that manifest freezes the installed-App inventory, semantic source catalog,
Agent-visible capability catalog, runner bindings, and user-phone World Bundle.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = {
    "installed_app_inventory": (
        "installed_app_inventory_ref",
        "installed_app_inventory_sha256",
    ),
    "user_phone_world_bundle": ("bundle_ref", "bundle_sha256"),
    "semantic_source_catalog": (
        "semantic_source_catalog_ref",
        "semantic_source_catalog_sha256",
    ),
    "public_capability_catalog": (
        "public_capability_catalog_ref",
        "public_capability_catalog_sha256",
    ),
    "runtime_capability_bindings": (
        "runtime_capability_bindings_ref",
        "runtime_capability_bindings_sha256",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_reference(repo_root: Path, reference: str) -> Path:
    if reference.startswith("user://"):
        path = repo_root / "users" / reference.removeprefix("user://")
    elif reference.startswith("platform://"):
        path = repo_root / "platform" / reference.removeprefix("platform://")
    else:
        path = Path(reference)
        if not path.is_absolute():
            path = repo_root / path
    return _require_inside_repo(repo_root, path)


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if all((current / name).is_dir() for name in ("code", "platform", "users")):
            return current
        current = current.parent
    raise FileNotFoundError(f"Cannot locate ProactiveBench root above {start}")


def _require_inside_repo(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Locked artifact must stay inside the repository: {path}") from exc
    return resolved


def _resolve_revision_artifact(
    repo_root: Path, revision_path: Path, reference: str
) -> Path:
    if "://" in reference or Path(reference).is_absolute():
        return resolve_repo_reference(repo_root, reference)
    return _require_inside_repo(repo_root, revision_path.parent / reference)


def _validate_inventory(
    inventory: dict[str, Any],
    bundle: dict[str, Any],
    public_catalog: dict[str, Any],
) -> None:
    applications = bundle["semantic_world"]["applications"]
    inventory_apps = inventory.get("apps", [])
    if inventory.get("installed_app_count") != len(inventory_apps):
        raise ValueError("Installed-App inventory count does not match its App list")
    inventory_ids = [item["app_id"] for item in inventory_apps]
    if len(set(inventory_ids)) != len(inventory_ids):
        raise ValueError("Installed-App inventory contains duplicate app_id values")
    if set(inventory_ids) != set(applications):
        raise ValueError("Installed-App inventory does not match the frozen World Bundle")

    status_counts: dict[str, int] = {}
    for item in inventory_apps:
        status = item["state_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    declared = inventory.get("state_status_counts", {})
    if declared != status_counts:
        raise ValueError("Installed-App inventory state-status counts are stale")

    tools_by_source: dict[str, list[str]] = {}
    for tool_id, tool in public_catalog["tools"].items():
        source_id = tool["source"]["source_id"]
        tools_by_source.setdefault(source_id, []).append(tool_id)
    for values in tools_by_source.values():
        values.sort()
    for item in inventory_apps:
        expected = tools_by_source.get(item["app_id"], [])
        if item["query_tool_ids"] != expected:
            raise ValueError(
                f"Installed-App query tools are stale for {item['app_id']}"
            )
        if item["queryable"] != bool(expected):
            raise ValueError(
                f"Installed-App queryable flag is stale for {item['app_id']}"
            )
    if inventory.get("queryable_app_count") != len(tools_by_source):
        raise ValueError("Installed-App queryable-App count is stale")
    if inventory.get("query_tool_count") != len(public_catalog["tools"]):
        raise ValueError("Installed-App query-tool count is stale")


def _validate_runtime_bindings(
    bindings: dict[str, Any], public_catalog: dict[str, Any]
) -> None:
    expected = {
        tool_id: deepcopy(tool["runtime_binding"])
        for tool_id, tool in public_catalog["tools"].items()
    }
    if bindings.get("tool_bindings") != expected:
        raise ValueError("Runtime capability bindings do not match the frozen tool catalog")


def load_locked_user_environment(
    repo_root: Path,
    sampler: dict[str, Any],
) -> dict[str, Any]:
    """Load and fully verify the User App Environment bound by ``sampler``."""

    lock = sampler.get("user_environment_lock")
    if not isinstance(lock, dict):
        raise ValueError("Registered Sampler must define user_environment_lock")
    required_lock_fields = {
        "revision",
        "revision_manifest_ref",
        "revision_manifest_sha256",
        "world_hash",
    }
    for ref_field, sha_field in REQUIRED_ARTIFACTS.values():
        required_lock_fields.update((ref_field, sha_field))
    missing = sorted(required_lock_fields - set(lock))
    if missing:
        raise ValueError(f"user_environment_lock is missing fields: {missing}")

    revision_path = resolve_repo_reference(repo_root, lock["revision_manifest_ref"])
    if not revision_path.is_file():
        raise FileNotFoundError(revision_path)
    revision_sha256 = file_sha256(revision_path)
    if revision_sha256 != lock["revision_manifest_sha256"]:
        raise ValueError(
            "Locked revision manifest SHA256 mismatch: "
            f"expected {lock['revision_manifest_sha256']}, got {revision_sha256}"
        )
    revision = read_json(revision_path)
    if revision.get("revision_id") != lock["revision"]:
        raise ValueError("User App Environment revision does not match sampler lock")
    if revision.get("user_id") != sampler.get("user_id"):
        raise ValueError("User App Environment belongs to a different user")

    declarations = revision.get("artifacts")
    if not isinstance(declarations, dict):
        raise ValueError("User App Environment revision has no artifacts map")
    artifact_paths: dict[str, Path] = {}
    artifact_values: dict[str, dict[str, Any]] = {}
    for artifact_id, (lock_ref_field, lock_sha_field) in REQUIRED_ARTIFACTS.items():
        declaration = declarations.get(artifact_id)
        if not isinstance(declaration, dict):
            raise ValueError(f"Revision is missing artifact {artifact_id}")
        artifact_path = _resolve_revision_artifact(
            repo_root, revision_path, declaration["ref"]
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual_sha256 = file_sha256(artifact_path)
        if actual_sha256 != declaration["sha256"]:
            raise ValueError(f"Revision artifact SHA256 mismatch: {artifact_id}")
        expected_ref = lock[lock_ref_field]
        if resolve_repo_reference(repo_root, expected_ref) != artifact_path:
            raise ValueError(f"Sampler artifact reference mismatch: {artifact_id}")
        if lock[lock_sha_field] != actual_sha256:
            raise ValueError(f"Sampler artifact SHA256 mismatch: {artifact_id}")
        artifact_paths[artifact_id] = artifact_path
        artifact_values[artifact_id] = read_json(artifact_path)

    bundle = artifact_values["user_phone_world_bundle"]
    if "user_media_assets" in declarations:
        from user_media_assets import verify_frozen_media
        media_path, media = verify_frozen_media(
            revision_path.parent, declarations["user_media_assets"],
            revision["user_id"], revision["revision_id"]
        )
        artifact_paths["user_media_assets"] = media_path
        artifact_values["user_media_assets"] = media
    if "user_web_assets" in declarations:
        from user_web_assets import verify_frozen_web
        web_path, web = verify_frozen_web(
            revision_path.parent, declarations["user_web_assets"],
            revision["user_id"], revision["revision_id"]
        )
        artifact_paths["user_web_assets"] = web_path
        artifact_values["user_web_assets"] = web
    if "user_native_content" in declarations:
        from user_native_content import verify_frozen_native_content
        native_path, native = verify_frozen_native_content(
            revision_path.parent, declarations["user_native_content"], revision,
            bundle["semantic_world"]["entities"], artifact_values.get("user_media_assets", {}),
            artifact_values.get("user_web_assets", {}))
        artifact_paths["user_native_content"] = native_path
        artifact_values["user_native_content"] = native
    if bundle.get("world_hash") != lock["world_hash"]:
        raise ValueError("Locked User App Environment world_hash mismatch")
    if declarations["user_phone_world_bundle"].get("world_hash") != lock["world_hash"]:
        raise ValueError("Revision Bundle world_hash does not match sampler lock")

    inventory = artifact_values["installed_app_inventory"]
    if inventory.get("revision_id") != lock["revision"]:
        raise ValueError("Installed-App inventory revision_id mismatch")
    _validate_inventory(
        inventory,
        bundle,
        artifact_values["public_capability_catalog"],
    )
    _validate_runtime_bindings(
        artifact_values["runtime_capability_bindings"],
        artifact_values["public_capability_catalog"],
    )

    return {
        "mode": "immutable_revision",
        "lock": {"mode": "immutable_revision", **deepcopy(lock)},
        "revision_path": revision_path,
        "revision": revision,
        "artifact_paths": artifact_paths,
        "artifacts": artifact_values,
    }
