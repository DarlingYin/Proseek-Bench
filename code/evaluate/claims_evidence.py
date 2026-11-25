"""Deterministic text evidence ledger for Claims evaluation."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from judge_client import file_hash, read_json


_OCR_CACHE: dict[str, str] = {}
_OCR_ENGINE_VERSION: str | None = None
_OCR_DISK_CACHE_LOADED = False


def _load_frozen_ocr_cache(frame: Path) -> None:
    """Load exact frame-SHA OCR reused from completed Claims-v1 verdicts."""
    global _OCR_DISK_CACHE_LOADED
    if _OCR_DISK_CACHE_LOADED:
        return
    _OCR_DISK_CACHE_LOADED = True
    repo = next((parent for parent in frame.parents if (parent / "evaluate/ops").is_dir()), None)
    if repo is None:
        return
    cache_path = repo / "evaluate/ops/claims-v1-ocr-cache.json"
    if not cache_path.is_file():
        return
    payload = read_json(cache_path)
    if (payload.get("status") != "complete" or payload.get("engine") != "tesseract"
            or payload.get("language") != "eng" or payload.get("page_segmentation_mode") != 11):
        raise ValueError(f"Invalid Claims OCR cache contract: {cache_path}")
    for digest, text in payload.get("frames", {}).items():
        if isinstance(digest, str) and isinstance(text, str):
            _OCR_CACHE[digest] = text


def _observation_number(value: str) -> int:
    match = re.fullmatch(r"o([0-9]+)", value or "")
    return int(match.group(1)) if match else -1


def _compact_receipt(receipt: dict, thread_id) -> dict:
    result = {
        "receipt_id": receipt.get("receipt_id"),
        "observation_id": receipt.get("observation_id"),
        "linked_to_prediction_thread": (
            thread_id is not None and thread_id in receipt.get("linked_thread_ids", [])
        ),
        "tool": receipt.get("tool"),
        "arguments": receipt.get("arguments", {}),
        "as_of": receipt.get("as_of"),
        "status": receipt.get("status"),
        "coverage": receipt.get("coverage"),
    }
    for field in ("items", "data", "error"):
        if field in receipt:
            result[field] = receipt[field]
    return result


def claims_style_diagnostics(claims: str) -> dict:
    """Deterministic non-semantic diagnostics kept outside Judge scoring."""
    return {
        "character_count": len(claims),
        "sentence_count": len([
            part for part in re.split(r"(?<=[.!?])\s+", claims.strip()) if part
        ]),
        "typed_handle_mentions": len(re.findall(
            r"\b[a-z][a-z0-9_]*:[a-z][a-z0-9_]*:[0-9]{3}\b", claims,
            flags=re.IGNORECASE,
        )),
        "query_process_markers": len(re.findall(
            r"\b(?:quer(?:y|ied)|search(?:ed)?|read_record|search_records|receipt|"
            r"complete coverage|returned exactly|lookup)\b",
            claims,
            flags=re.IGNORECASE,
        )),
    }


def _visible_text(frame: Path) -> str:
    """Return deterministic English/symbol OCR for one frozen GUI frame."""
    _load_frozen_ocr_cache(frame)
    frame_digest = file_hash(frame)
    if frame_digest not in _OCR_CACHE:
        completed = subprocess.run(
            ["tesseract", str(frame), "stdout", "-l", "eng", "--psm", "11"],
            check=True, capture_output=True, text=True,
        )
        lines = []
        for raw in completed.stdout.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if line:
                lines.append(line)
        _OCR_CACHE[frame_digest] = "\n".join(lines)
    return _OCR_CACHE[frame_digest]


def _ocr_engine_version() -> str:
    global _OCR_ENGINE_VERSION
    if _OCR_ENGINE_VERSION is None:
        completed = subprocess.run(
            ["tesseract", "--version"], check=True, capture_output=True, text=True,
        )
        _OCR_ENGINE_VERSION = completed.stdout.splitlines()[0].strip()
    return _OCR_ENGINE_VERSION


def _passive_evidence_before_commit(
    repo: Path, user_id: str, sampler_id: str, source_tier: str, committed_after: str,
) -> tuple[dict, dict]:
    """Build a public, observation-bound OCR ledger from formal frozen frames."""
    sampler_root = repo / "users" / user_id / source_tier / sampler_id
    sampler = read_json(sampler_root / "sampler.json")
    declared = sampler.get("episodes")
    if not isinstance(declared, list) or len(declared) != 1 or not isinstance(declared[0], dict):
        raise ValueError(f"Expected one declared Episode for Passive ledger: {sampler_root}")
    episode_ref = declared[0].get("episode_ref")
    if not isinstance(episode_ref, str) or not episode_ref:
        raise ValueError(f"Missing declared Episode reference: {sampler_root}")
    episode_path = (sampler_root / episode_ref).resolve()
    episode_path.relative_to(sampler_root.resolve())
    episode = read_json(episode_path)
    passive_manifest_path = sampler_root / episode["input_bindings"]["passive_input_ref"]
    passive_manifest = read_json(passive_manifest_path)
    stream_path = sampler_root / passive_manifest["observation_stream"]["stream_ref"]
    frames_root = sampler_root / passive_manifest["observation_stream"]["frames_root_ref"]
    commit_number = _observation_number(committed_after)
    if commit_number < 0:
        raise ValueError(f"Invalid Commit observation: {committed_after}")
    observations, frame_hashes = [], {}
    for line in stream_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if _observation_number(item["observation_id"]) > commit_number:
            continue
        frame_path = frames_root / item["image"]
        frame_digest = file_hash(frame_path)
        frame_hashes[str(frame_path.relative_to(repo))] = frame_digest
        observations.append({
            "observation_id": item["observation_id"],
            "logical_time": item.get("logical_time"),
            "frame_sha256": frame_digest,
            "visible_text_ocr": _visible_text(frame_path),
            "admissibility": (
                "Public text transcribed from this formal GUI frame. OCR may contain harmless "
                "character noise; use it only when the claimed fact is clearly represented."
            ),
        })
    if not observations or observations[-1]["observation_id"] != committed_after:
        raise ValueError(f"Passive stream does not reach Commit: {sampler_id}/{committed_after}")
    ledger = {
        "extraction": {
            "engine": "tesseract",
            "language": "eng",
            "page_segmentation_mode": 11,
            "role": "public Passive GUI evidence, not a causal conclusion",
        },
        "observations": observations,
    }
    provenance = {
        "episode_path": str(episode_path.relative_to(repo)),
        "episode_sha256": file_hash(episode_path),
        "passive_manifest_path": str(passive_manifest_path.relative_to(repo)),
        "passive_manifest_sha256": file_hash(passive_manifest_path),
        "stream_path": str(stream_path.relative_to(repo)),
        "stream_sha256": file_hash(stream_path),
        "frame_hashes": frame_hashes,
        "observation_count": len(observations),
        "ocr_engine_version": _ocr_engine_version(),
    }
    return ledger, provenance


def formal_frames_before_commit(
    repo: Path, user_id: str, sampler_id: str, source_tier: str, committed_after: str,
) -> list[dict]:
    """Return formal frame paths through Commit for an isolated visual check."""
    sampler_root = repo / "users" / user_id / source_tier / sampler_id
    sampler = read_json(sampler_root / "sampler.json")
    declared = sampler.get("episodes")
    if not isinstance(declared, list) or len(declared) != 1 or not isinstance(declared[0], dict):
        raise ValueError(f"Expected one declared Episode for visual verification: {sampler_root}")
    episode_ref = declared[0].get("episode_ref")
    if not isinstance(episode_ref, str) or not episode_ref:
        raise ValueError(f"Missing declared Episode reference: {sampler_root}")
    episode_path = (sampler_root / episode_ref).resolve()
    episode_path.relative_to(sampler_root.resolve())
    episode = read_json(episode_path)
    passive = read_json(sampler_root / episode["input_bindings"]["passive_input_ref"])
    stream_path = sampler_root / passive["observation_stream"]["stream_ref"]
    frames_root = sampler_root / passive["observation_stream"]["frames_root_ref"]
    limit = _observation_number(committed_after)
    result = []
    for line in stream_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if _observation_number(item["observation_id"]) <= limit:
            result.append({
                "observation_id": item["observation_id"],
                "logical_time": item.get("logical_time"),
                "path": str((frames_root / item["image"]).resolve()),
                "sha256": file_hash(frames_root / item["image"]),
            })
    return result


def build_text_evidence_ledger(
    repo: Path,
    run: str,
    user_id: str,
    sampler_id: str,
    prediction_index: int,
    reference_claims: str,
    commit_window: list[str],
    capability: str = "Active",
    source_tier: str = "L2",
    ledger_version: str = "v1",
) -> tuple[dict, dict]:
    """Return Judge-visible text evidence and private provenance metadata."""
    runtime_path = repo / "evaluate/runs" / run / user_id / sampler_id / "prediction.runtime.json"
    runtime = read_json(runtime_path)
    prediction = runtime["predictions"][prediction_index - 1]
    thread_id = prediction.get("thread_id")
    committed_after = prediction["committed_after"]

    commit_turn = None
    binding_mode = "need_thread"
    if thread_id is not None:
        for action in runtime.get("action_history", []):
            if action.get("control") == "commit" and thread_id in action.get("thread_ids", []):
                commit_turn = action.get("turn")
                break
    else:
        # Context-Summary commits a batch of final Predictions without persistent
        # NeedThread IDs. Bind Prediction indices to commit actions by the frozen
        # output order and each action's explicit prediction_count. This changes
        # only receipt relevance metadata; all receipts still must predate Commit.
        binding_mode = "context_summary_commit_batch"
        cursor = 0
        target = prediction_index - 1
        for action in runtime.get("action_history", []):
            if action.get("control") != "commit":
                continue
            count = action.get("prediction_count")
            if not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"Context-Summary commit lacks prediction_count for "
                    f"{run}/{user_id}/{sampler_id}"
                )
            if cursor <= target < cursor + count:
                commit_turn = action.get("turn")
                if action.get("observation_id") != committed_after:
                    raise ValueError(
                        f"Context-Summary Commit observation mismatch for "
                        f"{run}/{user_id}/{sampler_id} #{prediction_index}"
                    )
                break
            cursor += count
    if commit_turn is None:
        raise ValueError(f"Cannot find commit turn for {run}/{user_id}/{sampler_id} #{prediction_index}")

    receipt_ids = []
    for action in runtime.get("action_history", []):
        receipt_id = action.get("receipt_id")
        if receipt_id and action.get("turn", commit_turn + 1) <= commit_turn:
            receipt_ids.append(receipt_id)
    receipts = {item["receipt_id"]: item for item in runtime.get("query_receipts", [])}
    visible = [_compact_receipt(receipts[receipt_id], thread_id)
               for receipt_id in receipt_ids if receipt_id in receipts]
    linked = [item for item in visible if item["linked_to_prediction_thread"]]
    other = [item for item in visible if not item["linked_to_prediction_thread"]]

    ledger = {
        "evidence_contract": (
            "Text only. The benchmark-owned reference summary is valid for the matched Commit window: "
            "its direct factual state assertions are authoritative, while its causal conclusions do "
            "not replace premises omitted from Prediction Claims. Claims about actual retrieval, query "
            "scope, completeness, zero results, handle resolution or multi-record feasibility require "
            "compatible receipts. Query evidence contains only results obtained no later than Commit."
        ),
        "commit_context": {
            "committed_after": committed_after,
            "committed_at": prediction.get("committed_at"),
        },
        "source_policy": {
            "sampler_capability": capability,
            "source_tier": source_tier,
            "reference_role": "causal target and source-restricted compact evidence summary",
            "thread_linkage_policy": (
                "Thread linkage affects relevance, not truth. Context-Summary Predictions have no "
                "NeedThread ID, so their pre-Commit receipts are conservatively classified as "
                "other-visible. Such receipts may support a premise only when entity identity, "
                "scope and time match."
                if thread_id is None else
                "Thread linkage affects relevance, not truth. Other-visible receipts may support "
                "a premise only when entity identity, scope and time match."
            ),
        },
        "reference_evidence_summary": {
            "summary": reference_claims,
            "valid_for_commit_window": commit_window,
            "admissibility": (
                "Direct factual state assertions are authoritative for this Commit window. Causal "
                "conclusions cannot fill omitted Prediction premises. Retrieval-process, absence, "
                "coverage, handle-resolution and multi-record feasibility assertions require receipts."
            ),
        },
        "query_evidence_before_commit": {
            "thread_linked": linked,
            "other_visible": other,
        },
        "temporal_policy": (
            "For the same mutable fact and entity, use the latest applicable evidence at or before "
            "Commit. A later explicit GUI update supersedes an older query; an early empty query "
            "cannot contradict a later arrival. Different entities or scopes do not override each other."
        ),
    }
    provenance = {
        "runtime_path": str(runtime_path.relative_to(repo)),
        "runtime_sha256": file_hash(runtime_path),
        "thread_id": thread_id,
        "prediction_binding_mode": binding_mode,
        "commit_turn": commit_turn,
        "query_receipt_count": len(visible),
        "linked_query_receipt_count": sum(item["linked_to_prediction_thread"] for item in visible),
        "latest_receipt_observation": max(
            (item.get("observation_id", "") for item in visible),
            key=_observation_number,
            default=None,
        ),
        "capability": capability,
        "source_tier": source_tier,
        "ledger_version": ledger_version,
    }
    if ledger_version in {"v1", "v2", "v3", "v4", "v4-completion", "v4-fixed"}:
        passive, passive_provenance = _passive_evidence_before_commit(
            repo, user_id, sampler_id, source_tier, committed_after,
        )
        ledger = {
            "evidence_contract": (
                "Text-only source-separated Claims evidence. Ground-truth Claims are a causal "
                "target, not factual evidence. Passive GUI facts must come from the observation-"
                "bound OCR ledger; query and retrieval assertions must come from actual receipts."
            ),
            "commit_context": ledger["commit_context"],
            "source_policy": {
                **ledger["source_policy"],
                "reference_role": "causal target only; never evidence",
            },
            "reference_causal_target": {
                "claims": reference_claims,
                "valid_for_commit_window": commit_window,
                "admissibility": (
                    "Use only to identify the intended causal target and hard action boundaries. "
                    "Do not use its factual clauses as evidence for Prediction premises."
                ),
            },
            "passive_evidence_before_commit": passive,
            "query_evidence_before_commit": ledger["query_evidence_before_commit"],
            "temporal_policy": ledger["temporal_policy"],
        }
        if ledger_version in {"v1", "v3", "v4", "v4-completion", "v4-fixed"}:
            ledger["passive_visual_verification"] = []
        provenance["passive_evidence"] = passive_provenance
    elif ledger_version != "v1":
        raise ValueError(f"Unsupported Claims evidence ledger version: {ledger_version}")
    return ledger, provenance
