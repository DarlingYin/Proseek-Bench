#!/usr/bin/env python3
"""Run one ProSeekBench prediction artifact per Sampler.

The current release has exactly one Episode in every Sampler.  This wrapper
enforces that invariant and stores one ``prediction.json`` under the removable
``evaluate/`` workspace for each Sampler.  A prediction artifact may contain
zero, one, or several committed entries in its ``predictions`` array.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from artifact_paths import require_external_output
from experiment_modes import (
    add_mode_arguments,
    resolve_mode,
    batch_contract,
    validate_resume,
    validate_transport_only_resume,
    check_no_search_result,
)
from runtime_diagnostics import RUNTIME_IMPLEMENTATION
from compatibility_rules import registry_hash, REGISTRY_VERSION
from query_policy import QUERY_POLICY_VERSION
from runtime_snapshot import verify_snapshot
from prompt_input import FULL_STATE
from prediction_layout import CANONICAL_RUN_IDS, canonical_setting, validate_setting_metadata
from protocol_adapter import ADAPTER_MODES, resolve_adapter_mode


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not value:
        raise ValueError("Path component becomes empty after normalization")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_episode_paths(sampler_path: Path, sampler: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve the single Episode declared by sampler.json.

    Authoring archives can leave unreferenced Episode directories beside the
    active Episode.  The frozen inventory binds the declaration, so directory
    discovery must not turn those leftovers into additional Episodes.
    """

    episodes = sampler.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1 or not isinstance(episodes[0], dict):
        raise ValueError(f"Release Sampler must declare one Episode: {sampler_path}")
    entry = episodes[0]
    episode_ref = entry.get("episode_ref")
    if not isinstance(episode_ref, str) or not episode_ref:
        raise ValueError(f"Declared Episode reference is missing: {sampler_path}")
    episode_path = sampler_path.parent / episode_ref
    episode_path.resolve().relative_to(sampler_path.parent.resolve())
    ground_truth_ref = entry.get("ground_truth_ref")
    if ground_truth_ref is None:
        ground_truth_path = episode_path.with_name("ground-truth.json")
    elif isinstance(ground_truth_ref, str) and ground_truth_ref:
        ground_truth_path = sampler_path.parent / ground_truth_ref
        ground_truth_path.resolve().relative_to(sampler_path.parent.resolve())
    else:
        raise ValueError(f"Declared Ground Truth reference is invalid: {sampler_path}")
    return episode_path, ground_truth_path


def load_release_inventory(
    repo_root: Path, manifest_path: Path | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Load one hash-frozen inventory; default to the active release."""

    manifest_path = (manifest_path or (repo_root / "release_manifest.json")).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("status") not in {"active", "diagnostic_frozen"}:
        raise ValueError(f"Inventory manifest is not active/frozen: {manifest_path}")
    inventory = manifest.get("sampler_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"Release manifest has no sampler_inventory: {manifest_path}")
    declared_paths = [repo_root / row["path"] for row in inventory]
    actual_paths = sorted(repo_root.glob("users/*/L*/*/sampler.json"))
    declared_resolved = {path.resolve() for path in declared_paths}
    actual_resolved = {path.resolve() for path in actual_paths}
    # Authoring candidates may occupy their final user-relative paths before
    # publication so GT review never requires another path migration.  They
    # remain excluded from every formal run until a new release manifest lists
    # them.  Only explicitly registered candidate paths are tolerated; an
    # unregistered extra Sampler still fails closed.
    candidate_resolved: set[Path] = set()
    candidate_registry_path = repo_root / "candidate_registry.json"
    if candidate_registry_path.is_file():
        registry = read_json(candidate_registry_path)
        if registry.get("status") != "active_authoring":
            raise ValueError(
                f"Candidate registry is not active_authoring: {candidate_registry_path}"
            )
        candidate_rows = registry.get("sampler_inventory")
        if not isinstance(candidate_rows, list):
            raise ValueError(
                f"Candidate registry has no sampler_inventory: {candidate_registry_path}"
            )
        if registry.get("totals", {}).get("samplers") != len(candidate_rows):
            raise ValueError("Candidate registry sampler total is inconsistent")
        for row in candidate_rows:
            path = repo_root / row["path"]
            if not path.is_file():
                raise ValueError(f"Registered candidate Sampler is missing: {path}")
            sampler = read_json(path)
            if any(
                sampler.get(key) != row.get(key)
                for key in ("user_id", "task_tier", "sampler_id")
            ):
                raise ValueError(f"Candidate identity mismatch: {path}")
            candidate_resolved.add(path.resolve())
    active_release = read_json(repo_root / "release_manifest.json")
    active_release_resolved = {
        (repo_root / row["path"]).resolve()
        for row in active_release.get("sampler_inventory", [])
    }
    allowed_resolved = declared_resolved | candidate_resolved | active_release_resolved
    if allowed_resolved != actual_resolved:
        missing = sorted(str(path.relative_to(repo_root)) for path in declared_paths if not path.exists())
        extra = sorted(
            str(path.relative_to(repo_root))
            for path in actual_paths
            if path.resolve() not in allowed_resolved
        )
        missing_candidates = sorted(
            str(path.relative_to(repo_root))
            for path in (repo_root / row["path"] for row in (
                registry.get("sampler_inventory", [])
                if candidate_registry_path.is_file() else []
            ))
            if not path.exists()
        )
        raise ValueError(
            "Sampler paths differ from the release manifest plus candidate registry; "
            f"missing={missing}, missing_candidates={missing_candidates}, extra={extra}"
        )
    if manifest.get("totals", {}).get("samplers") != len(inventory):
        raise ValueError("release_manifest.json sampler total does not match its inventory")

    source_digest = hashlib.sha256()
    for row, sampler_path in zip(inventory, declared_paths):
        sampler = read_json(sampler_path)
        if any(
            sampler.get(key) != row.get(key)
            for key in ("user_id", "task_tier", "sampler_id")
        ):
            raise ValueError(f"Release identity mismatch: {sampler_path}")
        episode_path, ground_truth_path = declared_episode_paths(sampler_path, sampler)
        sources = (sampler_path, episode_path, ground_truth_path)
        for source in sources:
            if not source.is_file():
                raise ValueError(f"Release source is missing: {source}")
            source_digest.update(str(source.relative_to(repo_root)).encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update(sha256(source).encode("ascii"))
            source_digest.update(b"\n")
    if source_digest.hexdigest() != manifest.get("source_bundle_sha256"):
        raise ValueError(
            "Current Sampler/Episode/GT bytes differ from release_manifest.json; "
            "publish a new release manifest before inference"
        )
    return manifest, declared_paths


def discover_samplers(
    repo_root: Path,
    user_ids: set[str],
    sampler_ids: set[str],
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    _, release_paths = load_release_inventory(repo_root, manifest_path)
    for sampler_path in release_paths:
        sampler = read_json(sampler_path)
        user_id = sampler.get("user_id")
        sampler_id = sampler.get("sampler_id")
        if not isinstance(user_id, str) or not isinstance(sampler_id, str):
            raise ValueError(f"Sampler identity is missing: {sampler_path}")
        if user_ids and user_id not in user_ids:
            continue
        if sampler_ids and sampler_id not in sampler_ids:
            continue
        sampler_root = sampler_path.parent
        episode_path, _ = declared_episode_paths(sampler_path, sampler)
        episode = read_json(episode_path)
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError(f"Episode identity is missing: {episode_path}")
        records.append(
            {
                "user_id": user_id,
                "sampler_id": sampler_id,
                "task_tier": sampler.get("task_tier"),
                "episode_id": episode_id,
                "sampler_root": sampler_root.resolve(),
            }
        )
    return records


def completed_prediction(path: Path) -> bool:
    if path.with_name('prediction.protocol-failure.json').exists():
        return False
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        value.get("schema_version") == "proseekbench.prediction.v1"
        and isinstance(value.get("sampler_id"), str)
        and isinstance(value.get("episode_id"), str)
        and isinstance(value.get("predictions"), list)
        and isinstance(value.get("search_count"), int)
        and value["search_count"] >= 0
    )


def validate_existing_result(row: dict[str, Any], args: argparse.Namespace) -> None:
    path = row["output"]
    if not completed_prediction(path):
        raise ValueError(f"Existing prediction is incomplete or invalid: {path}; use a new run-id")
    prediction = read_json(path)
    if any(prediction[key] != row[key] for key in ("sampler_id", "episode_id")):
        raise ValueError(f"Prediction identity mismatch: {path}")
    runtime = read_json(path.with_name("prediction.runtime.json"))
    config = runtime["configuration"]
    if (runtime["status"] != "complete" or runtime["episode_id"] != row["episode_id"]
            or config.get("inference_mode") != args.inference_mode
            or config["search_budget"] != args.search_budget
            or config["max_searches_per_observation"] != args.max_searches_per_observation
            or runtime["model_run"]["protocol_adapter"]["mode"] != args.protocol_adapter):
        raise ValueError(f"Runtime configuration/status mismatch: {path}")
    if config.get('runtime_implementation') != RUNTIME_IMPLEMENTATION:
        raise ValueError(f"Runtime implementation differs; keep historical predictions separate: {path}")
    if config.get('image_window') != args.image_window:
        raise ValueError(f"Image window mismatch: {path}")
    if config.get('experiment_condition', {}).get('prompt_version') != args.prompt_version:
        raise ValueError(f"Prompt version mismatch: {path}")
    if config.get('query_policy_version') != QUERY_POLICY_VERSION:
        raise ValueError(f"Query accounting policy mismatch: {path}")
    if runtime['model_run'].get('compatibility_registry', {}).get('sha256') != registry_hash():
        raise ValueError(f"Compatibility rules differ; do not mix repaired conditions: {path}")
    snapshot = runtime['model_run'].get('runtime_snapshot')
    if not snapshot:
        raise ValueError(f"Missing source snapshot: {path}")
    verify_snapshot(snapshot['path'], expected=snapshot['sha256'])
    captured = read_json(Path(snapshot['path']) / 'snapshot.json')['files']
    allowed_transport_sources = {
        "run_online_agent.py",
        "curl_transport.py",
        "run_batch.py",
        "experiment_modes.py",
    }
    for source in Path(__file__).parent.glob('*.py'):
        if source.name.startswith(('._', 'test_')):
            continue
        if captured.get('code/inference/' + source.name) != sha256(source):
            if (
                getattr(args, "allow_transport_only_migration", False)
                and source.name in allowed_transport_sources
                and runtime.get("model_run", {}).get("provider") == "volcengine_coding"
                and runtime.get("model_run", {}).get("transport")
                in {"openai_compatible_http", "openai_compatible_curl_securetransport"}
            ):
                continue
            raise ValueError(f"Runtime source differs from the retained run: {source.name}")
    root = Path(__file__).resolve().parents[2]
    selected = resolve_mode(args, root)
    recorded = config.get('experiment_condition', {})
    if recorded.get('prompt_input_contract', FULL_STATE) != selected.get('prompt_input_contract', FULL_STATE):
        raise ValueError(f"Prompt input contract mismatch: {path}")
    if recorded.get('profile_sha256') != selected['profile_sha256']:
        raise ValueError(f"Prompt profile hash mismatch: {path}")
    if recorded.get('profile_version') != selected['profile_version']:
        raise ValueError(f"Prompt profile revision mismatch: {path}")
    if recorded.get('prompts') != selected['prompts']:
        raise ValueError(f"Prompt contents differ despite matching version labels: {path}")
    check_no_search_result(args.inference_mode, runtime)


def finalize_canonical_repair(
    *,
    repo_root: Path,
    run_dir: Path,
    repair_manifest_path: Path,
    args: argparse.Namespace,
    release_manifest: dict[str, Any],
    release_manifest_sha256: str,
) -> None:
    """Publish a current 41-Sampler primary manifest after a safe repair."""

    all_rows = discover_samplers(repo_root, set(), set())
    results: list[dict[str, Any]] = []
    identities: list[dict[str, str]] = []
    for row in all_rows:
        prediction = (
            run_dir / safe_component(row["user_id"])
            / safe_component(row["sampler_id"]) / "prediction.json"
        )
        if not prediction.is_file():
            raise ValueError(f"Canonical repair is still missing Prediction: {prediction}")
        runtime = read_json(prediction.with_name("prediction.runtime.json"))
        config = runtime.get("configuration", {})
        validate_setting_metadata(
            args.run_id, inference_mode=config.get("inference_mode"),
            image_window=config.get("image_window"),
            prompt_version=config.get("experiment_condition", {}).get("prompt_version"),
        )
        if config.get("inference_mode") is None:
            # Historical retained Predictions predate explicit mode/adapter
            # provenance. Their immutable runtime still pins the effective
            # window and query limits, which are sufficient for canonical
            # manifest reconciliation; newly repaired outputs use the strict
            # validator below.
            if (
                not completed_prediction(prediction)
                or runtime.get("status") != "complete"
                or runtime.get("episode_id") != row["episode_id"]
                or config.get("image_window") != args.image_window
                or config.get("search_budget") != args.search_budget
                or config.get("max_searches_per_observation")
                != args.max_searches_per_observation
            ):
                raise ValueError(f"Legacy Runtime configuration/status mismatch: {prediction}")
            check_no_search_result(args.inference_mode, runtime)
        else:
            validate_existing_result({**row, "output": prediction}, args)
        identities.append({key: row[key] for key in ("user_id", "sampler_id", "episode_id")})
        results.append(
            {
                "user_id": row["user_id"],
                "sampler_id": row["sampler_id"],
                "task_tier": row["task_tier"],
                "episode_id": row["episode_id"],
                "prediction": str(prediction),
                "status": "complete",
            }
        )
    primary_path = run_dir / "batch-manifest.json"
    primary = read_json(primary_path) if primary_path.exists() else {}
    primary.update(
        {
            "schema_version": "proseekbench.inference-batch.v1",
            "status": "complete",
            "model": args.model,
            "provider": args.provider,
            "run_id": args.run_id,
            "repair_missing": False,
            "canonical_setting": args.run_id,
            "inference_mode": args.inference_mode,
            "run_configuration": batch_contract(args, resolve_mode(args, repo_root), repo_root),
            "release_sha256sums_sha256": sha256(repo_root / "SHA256SUMS"),
            "release_id": release_manifest["release_id"],
            "release_manifest_sha256": release_manifest_sha256,
            "planned_samplers": identities,
            "sampler_count": len(identities),
            "results": results,
            "latest_repair_manifest": str(repair_manifest_path),
        }
    )
    write_json(primary_path, primary)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--inventory-manifest",
        type=Path,
        default=repo_root / "release_manifest.json",
        help="Hash-frozen release or diagnostic inventory to schedule.",
    )
    parser.add_argument(
        "--batch-manifest-name",
        default="batch-manifest.json",
        help="Manifest filename inside the setting directory; Predictions keep canonical paths.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "evaluate" / "runs",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "qianwen", "bigmodel", "volcengine_coding", "openrouter", "custom", "codex_exec"],
        default="auto",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--api-base")
    parser.add_argument(
        "--request-model",
        help="Model identifier sent to the endpoint; --model remains the artifact label.",
    )
    parser.add_argument("--user-id", action="append", default=[])
    parser.add_argument("--sampler-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Samplers to run concurrently within this batch.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help=(
            "Run only explicitly selected missing Samplers inside a canonical "
            "setting directory. Existing Predictions are never overwritten; "
            "a separate repair manifest is written under repair-manifests/."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--allow-transport-only-migration",
        action="store_true",
        help=(
            "Allow only the audited Volcengine urllib/OpenSSL to curl/SecureTransport "
            "migration while retaining otherwise identical completed Predictions."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--action-window", type=int, default=8)
    parser.add_argument("--image-window", type=int, default=3)
    add_mode_arguments(parser)
    parser.add_argument("--tool-interface", choices=["generic", "legacy"], default="generic")
    parser.add_argument("--max-model-calls", type=int, default=128)
    parser.add_argument(
        "--protocol-error-policy",
        choices=["fail", "repair_then_stop", "repair_then_wait"],
        default="repair_then_stop",
    )
    parser.add_argument("--max-protocol-repairs", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument(
        "--protocol-adapter",
        choices=ADAPTER_MODES,
        default="safe",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--response-format",
        choices=["json_object", "prompt_only"],
        default="json_object",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["default", "on", "off"],
        default="default",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
    )
    args = parser.parse_args()
    args.protocol_adapter = resolve_adapter_mode(args.protocol_adapter, args.model)
    try:
        condition = resolve_mode(args, repo_root)
    except ValueError as exc:
        parser.error(str(exc))
    setting = canonical_setting(args.inference_mode, args.image_window, args.prompt_version)
    canonical_output_root = (repo_root / "evaluate" / "runs").resolve()
    writes_canonical_results = args.output_root.resolve() == canonical_output_root
    if writes_canonical_results:
        if setting is None:
            parser.error(
                "Canonical evaluate/runs accepts Framework v2 image windows 1, 3, or 5"
            )
        if args.run_id is None:
            args.run_id = setting
        elif args.run_id != setting:
            parser.error(
                f"Canonical setting path for this configuration is {setting!r}, "
                f"not {args.run_id!r}"
            )
    elif args.run_id is None:
        args.run_id = args.inference_mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.run_id in CANONICAL_RUN_IDS and args.run_id != setting:
        parser.error(
            f"Reserved setting {args.run_id!r} conflicts with this mode/version/window; "
            f"use {setting!r} or a non-canonical diagnostic run-id"
        )
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.repair_missing and (args.resume or args.overwrite):
        parser.error("--repair-missing cannot be combined with --resume or --overwrite")
    if args.repair_missing:
        if args.run_id not in CANONICAL_RUN_IDS:
            parser.error(
                "--repair-missing requires canonical --run-id: "
                + ", ".join(sorted(CANONICAL_RUN_IDS))
            )
        if not args.sampler_id:
            parser.error("--repair-missing requires at least one explicit --sampler-id")
    if args.reasoning_effort is not None and args.thinking_mode != "default":
        parser.error(
            "--reasoning-effort cannot be combined with --thinking-mode on/off"
        )
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    args.codex_version = None
    if args.provider == "codex_exec":
        resolved_codex = shutil.which(args.codex_bin)
        if resolved_codex is None:
            parser.error(f"Codex executable not found: {args.codex_bin}")
        args.codex_bin = str(Path(resolved_codex).resolve())
        version = subprocess.run(
            [args.codex_bin, "--version"], text=True, capture_output=True, check=False
        )
        if version.returncode != 0:
            parser.error(f"Unable to read Codex CLI version: {version.stderr[-500:]}")
        args.codex_version = version.stdout.strip()

    output_root = require_external_output(repo_root, args.output_root)
    run_dir = output_root / safe_component(args.model) / safe_component(args.run_id)
    samplers = discover_samplers(
        repo_root,
        set(args.user_id),
        set(args.sampler_id),
        args.inventory_manifest,
    )
    if args.limit is not None:
        samplers = samplers[: args.limit]
    if not samplers:
        raise SystemExit("No Samplers matched the requested filters")
    inventory_manifest_path = args.inventory_manifest.resolve()
    release_manifest = read_json(inventory_manifest_path)
    release_manifest_sha256 = sha256(inventory_manifest_path)

    planned = []
    for row in samplers:
        output = (
            run_dir
            / safe_component(row["user_id"])
            / safe_component(row["sampler_id"])
            / "prediction.json"
        )
        planned.append({**row, "output": output})
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "model": args.model,
                    "run_id": args.run_id,
                    "inference_mode": args.inference_mode,
                    "experiment_condition": condition,
                    "protocol_adapter": args.protocol_adapter,
                    "protocol_error_policy": args.protocol_error_policy,
                    "runtime_contract": {"implementation": RUNTIME_IMPLEMENTATION,
                                         "query_policy": QUERY_POLICY_VERSION,
                                         "compatibility_registry": REGISTRY_VERSION,
                                         "registry_sha256": registry_hash()},
                    "canonical_setting": setting,
                    "release_id": release_manifest["release_id"],
                    "release_manifest_sha256": release_manifest_sha256,
                    "sampler_count": len(planned),
                    "outputs": [str(row["output"]) for row in planned],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.repair_missing:
        repair_id = datetime.now(timezone.utc).strftime("repair-%Y%m%dT%H%M%SZ")
        manifest_path = run_dir / "repair-manifests" / f"{repair_id}.json"
    else:
        if Path(args.batch_manifest_name).name != args.batch_manifest_name:
            parser.error("--batch-manifest-name must be a filename")
        manifest_path = run_dir / args.batch_manifest_name
    contract = batch_contract(args, condition, repo_root)
    identities = [{key: row[key] for key in ("user_id", "sampler_id", "episode_id")} for row in planned]
    release_hash = sha256(repo_root / "SHA256SUMS")
    # Validate BEFORE replacing the manifest or launching any model request.
    if args.repair_missing:
        existing = [str(row["output"]) for row in planned if row["output"].exists()]
        if existing:
            parser.error(
                "Repair targets must be absent; archive the stale Prediction first: "
                + ", ".join(existing)
            )
        # Fail before paid calls if retained artifacts use a different Runtime or
        # prompting condition. A bug fix cannot silently repair only three rows
        # into a historical condition that used broken tool semantics.
        for retained in run_dir.glob('*/*/prediction.json'):
            value = read_json(retained)
            try:
                validate_existing_result({'output': retained, 'sampler_id': value['sampler_id'],
                                          'episode_id': value['episode_id']}, args)
            except (ValueError, KeyError, OSError) as exc:
                parser.error(str(exc))
    elif manifest_path.exists() and not args.overwrite:
        if not args.resume:
            parser.error("Batch already exists; use --resume with identical settings or a new run-id")
        try:
            prior_manifest = read_json(manifest_path)
            if args.allow_transport_only_migration:
                validate_transport_only_resume(prior_manifest, contract, identities, release_hash)
            else:
                validate_resume(prior_manifest, contract, identities, release_hash)
        except ValueError as exc:
            parser.error(str(exc))
    elif not manifest_path.exists() and not args.overwrite and any(row["output"].exists() for row in planned):
        parser.error("Existing predictions without a batch manifest; use a new run-id")
    if args.resume and not args.overwrite:
        try:
            for row in planned:
                if row["output"].exists():
                    validate_existing_result(row, args)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
    manifest: dict[str, Any] = {
        "schema_version": "proseekbench.inference-batch.v1",
        "status": "running",
        "model": args.model,
        "provider": args.provider,
        "run_id": args.run_id,
        "repair_missing": args.repair_missing,
        "canonical_setting": args.run_id if args.run_id in CANONICAL_RUN_IDS else None,
        "release_id": release_manifest["release_id"],
        "release_manifest_sha256": release_manifest_sha256,
        "release_sha256sums_sha256": release_hash,
        "inference_mode": args.inference_mode,
        "protocol_adapter": args.protocol_adapter,
        "run_configuration": contract,
        "planned_samplers": identities,
        "sampler_count": len(planned),
        "results": [],
    }
    if args.allow_transport_only_migration:
        manifest["transport_migration"] = {
            "version": "proseekbench.transport-migration.v1",
            "from": "openai_compatible_http_python_tls",
            "to": "openai_compatible_curl_securetransport",
            "allowed_source_changes": [
                "run_online_agent.py",
                "curl_transport.py",
                "run_batch.py",
                "experiment_modes.py",
            ],
            "semantic_settings_unchanged": True,
        }
    write_json(manifest_path, manifest)
    runner = Path(__file__).with_name("run_online_agent.py")
    failed = False
    queued: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(planned, start=1):
        output = row["output"]
        if output.exists():
            if args.resume and completed_prediction(output):
                manifest["results"].append(
                    {
                        "user_id": row["user_id"],
                        "sampler_id": row["sampler_id"],
                        "episode_id": row["episode_id"],
                        "prediction": str(output),
                        "status": "skipped_complete",
                    }
                )
                write_json(manifest_path, manifest)
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"Prediction already exists; use --resume or --overwrite: {output}"
                )
        queued.append((index, row))

    def run_one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        output = row["output"]
        command = [
            sys.executable,
            str(runner),
            "--sampler-root",
            str(row["sampler_root"]),
            "--episode-id",
            row["episode_id"],
            "--provider",
            args.provider,
            "--codex-bin",
            args.codex_bin,
            "--model",
            args.model,
            "--output",
            str(output),
            "--action-window",
            str(args.action_window),
            "--image-window",
            str(args.image_window),
            "--search-budget",
            "unlimited" if args.search_budget is None else str(args.search_budget),
            "--max-searches-per-observation",
            "unlimited" if args.max_searches_per_observation is None else str(args.max_searches_per_observation),
            "--inference-mode",
            args.inference_mode,
            "--prompt-version",
            args.prompt_version,
            "--tool-interface",
            args.tool_interface,
            "--max-model-calls",
            str(args.max_model_calls),
            "--protocol-error-policy",
            args.protocol_error_policy,
            "--max-protocol-repairs",
            str(args.max_protocol_repairs),
            "--request-timeout",
            str(args.request_timeout),
            "--protocol-adapter",
            args.protocol_adapter,
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--response-format",
            args.response_format,
            "--thinking-mode",
            args.thinking_mode,
        ]
        if args.reasoning_effort is not None:
            command.extend(["--reasoning-effort", args.reasoning_effort])
        if args.api_base:
            command.extend(["--api-base", args.api_base])
        if args.request_model:
            command.extend(["--request-model", args.request_model])
        print(
            f"[{index}/{len(planned)}] {row['user_id']} / {row['sampler_id']}",
            flush=True,
        )
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "user_id": row["user_id"],
            "sampler_id": row["sampler_id"],
            "task_tier": row["task_tier"],
            "episode_id": row["episode_id"],
            "prediction": str(output),
            "status": ("complete" if completed.returncode == 0 else
                       "protocol_failed" if completed.returncode == 2 and output.with_name('prediction.protocol-failure.json').exists()
                       else "failed"),
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, index, row): (index, row)
            for index, row in queued
        }
        for future in as_completed(futures):
            item = future.result()
            manifest["results"].append(item)
            manifest["results"].sort(
                key=lambda result: (
                    result["user_id"],
                    result["sampler_id"],
                    result["episode_id"],
                )
            )
            write_json(manifest_path, manifest)
            if item["status"] in {"failed", "protocol_failed"}:
                failed = True
                if not args.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    break

    if failed and not args.continue_on_error:
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise SystemExit(1)

    manifest["status"] = "completed_with_failures" if failed else "complete"
    write_json(manifest_path, manifest)
    if args.repair_missing and not failed:
        try:
            finalize_canonical_repair(
                repo_root=repo_root,
                run_dir=run_dir,
                repair_manifest_path=manifest_path,
                args=args,
                release_manifest=release_manifest,
                release_manifest_sha256=release_manifest_sha256,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            manifest["status"] = "repair_complete_primary_manifest_failed"
            manifest["primary_manifest_error"] = str(exc)
            write_json(manifest_path, manifest)
            raise
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(manifest_path),
                "sampler_count": len(planned),
                "completed": sum(
                    row["status"] in {"complete", "skipped_complete"}
                    for row in manifest["results"]
                ),
                "failed": sum(row["status"] in {"failed", "protocol_failed"} for row in manifest["results"]),
                "protocol_failed": sum(row["status"] == "protocol_failed" for row in manifest["results"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
