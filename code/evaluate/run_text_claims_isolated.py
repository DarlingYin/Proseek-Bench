#!/usr/bin/env python3
"""Judge Claims 1--4 for frozen Need-passing and exact-Time-valid edges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

from claims_evidence import (
    build_text_evidence_ledger, claims_style_diagnostics, formal_frames_before_commit,
)
from claims_judge import IsolatedClaimsPromptBundle
from judge_client import JudgeClient, canonical, digest, file_hash, read_json, write_json
from metrics import aggregate
from run_evaluation import check_integrity, component, discover, inside
from visual_claims_verifier import VisualClaimsVerifier


REPO = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = "text-claims-isolated.v1-after-need-time"
API_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
CAPABILITY_MAPPING = REPO / "audit/capability_regroup_gui_active_longterm_frozen137_20261024.json"


def edge_key(edge: dict) -> tuple:
    return (edge["model"], edge["user_id"], edge["sampler_id"],
            edge["prediction_index"], edge["gt_need_id"], edge["alternative_id"])


def claims_payload(edge: dict) -> dict:
    return {
        "prediction": {
            "prediction_id": "p0001",
            "need": edge["prediction_need"],
            "claims": edge["prediction_claims"],
        },
        "ground_truth": {
            "reference_id": "r0001",
            "need": edge["reference_need"],
            "claims": edge["reference_claims"],
        },
        "evidence_ledger": edge["evidence_ledger"],
    }


def _normalized_stage(raw: dict, response: dict, judge: JudgeClient) -> dict:
    normalized = judge.bundle.normalize("claims_isolated", raw)
    if "target_coverage" in raw:
        return {**{key: raw[key] for key in (
                    "reference_resolution_type", "target_coverage", "target_reason",
                    "action_basis_evidence", "action_basis_reason", "gap_evidence",
                    "gap_reason", "boundary_coverage", "boundary_reason",
                    "completion_status")},
                "score": normalized["score"], "raw_score": raw["score"],
                "score_normalization": normalized["normalization"],
                "pass": normalized["pass"], "reason": raw["reason"],
                "cache_key": response.get("cache_key"),
                "cache_hit": response.get("cache_hit", False)}
    if "completion_checks" in raw:
        return {
            "completion_checks": raw["completion_checks"],
            "reference_resolution_type": normalized["reference_resolution_type"],
            "completion_status": normalized["completion_status"],
            "score": normalized["score"],
            "score_normalization": normalized["normalization"],
            "raw_score": raw["score"],
            "pass": normalized["pass"],
            "reason": raw["reason"],
            "cache_key": response.get("cache_key"),
            "cache_hit": response.get("cache_hit", False),
        }
    result = {
        "premise_checks": raw["premise_checks"],
        "action_checks": raw["action_checks"],
        "temporal_consistency_status": raw["temporal_consistency_status"],
        "unresolved_gap_status": raw["unresolved_gap_status"],
        "unresolved_gap_basis": raw["unresolved_gap_basis"],
        "applicability_status": raw["applicability_status"],
        "intervention_strength_status": raw["intervention_strength_status"],
        "grounding_score": normalized["grounding_score"],
        "sufficiency_score": normalized["sufficiency_score"],
        "score": normalized["score"],
        "score_normalization": normalized["normalization"],
        "raw_scores": {"grounding": raw["grounding_score"],
                       "sufficiency": raw["sufficiency_score"], "final": raw["score"]},
        "pass": normalized["pass"],
        "reason": raw["reason"],
        "cache_key": response.get("cache_key"),
        "cache_hit": response.get("cache_hit", False),
    }
    if "reference_requirements" in raw:
        result.update({
            "reference_requirements": raw["reference_requirements"],
            "reference_resolution_mode": raw["reference_resolution_mode"],
            "reference_action_coverage": normalized.get(
                "reference_action_coverage", raw["reference_action_coverage"]),
            "evidence_completion_status": normalized.get(
                "evidence_completion_status", raw["evidence_completion_status"]),
            "raw_reference_action_coverage": raw["reference_action_coverage"],
            "raw_evidence_completion_status": raw["evidence_completion_status"],
            "missing_decisive_requirements": raw["missing_decisive_requirements"],
        })
    return result


def _visual_candidate(check: dict) -> bool:
    if check["status"] == "supported" or check["evidence_source"] not in {
            "none", "contradicted", "passive_gui"}:
        return False
    return not re.search(
        r"\b(query|queried|search|receipt|coverage|resource|record|handle|returned|lookup)\b",
        check["premise"], flags=re.IGNORECASE,
    ) and not re.search(
        r"\b(unverified|uncertain|unknown|not establish|cannot establish|lack(?:s|ing)?|"
        r"no evidence|not found|does not exist|absence|complete coverage|safe|compatible)\b",
        check["premise"], flags=re.IGNORECASE,
    )


def _visual_requirement_candidate(check: dict) -> bool:
    if (check["evidence_status"] in {"supported", "not_required"}
            or check["evidence_source"] not in {"none", "passive_gui"}):
        return False
    text = check["requirement"]
    return not re.search(
        r"\b(query|queried|search|receipt|coverage|resource|record|handle|returned|lookup)\b",
        text, flags=re.IGNORECASE,
    ) and not re.search(
        r"\b(unverified|uncertain|unknown|not establish|cannot establish|lack(?:s|ing)?|"
        r"no evidence|not found|does not exist|absence|complete coverage|safe|compatible)\b",
        text, flags=re.IGNORECASE,
    )


def _visual_completion_candidate(check: dict) -> bool:
    if (check["evidence_status"] in {"supported", "not_required"}
            or check["evidence_source"] not in {"none", "passive_gui"}):
        return False
    return check["completion_role"] in {"target", "prerequisite"}


def judge_edge(edge_number: int, edge: dict, judge: JudgeClient,
               visual: VisualClaimsVerifier | None = None) -> dict:
    response = judge.judge_once("claims_isolated", claims_payload(edge))
    if response.get("verdict") is None:
        raise ValueError(f"Claims Judge failed: {response.get('error', 'unknown_error')}")
    raw = response["verdict"]["results"][0]
    initial_stage = _normalized_stage(raw, response, judge)
    visual_receipts = []
    if visual is not None and not initial_stage["pass"]:
        for check in raw.get("premise_checks", []):
            if _visual_candidate(check):
                visual_receipts.append(visual.verify(check["premise"], edge["_visual_frames"]))
        for check in raw.get("reference_requirements", []):
            if _visual_requirement_candidate(check):
                visual_receipts.append(visual.verify(
                    check["requirement"], edge["_visual_frames"],
                ))
        for check in raw.get("completion_checks", []):
            if _visual_completion_candidate(check):
                visual_receipts.append(visual.verify(
                    check["requirement"], edge["_visual_frames"],
                ))
        visible = [item for item in visual_receipts if item["status"] == "visible"]
        if visible:
            edge["evidence_ledger"]["passive_visual_verification"].extend(visible)
            response = judge.judge_once("claims_isolated", claims_payload(edge))
            if response.get("verdict") is None:
                raise ValueError(f"Claims re-Judge failed: {response.get('error', 'unknown_error')}")
            raw = response["verdict"]["results"][0]
    final_stage = _normalized_stage(raw, response, judge)
    if visual_receipts:
        final_stage["initial_score"] = initial_stage["score"]
        final_stage["initial_pass"] = initial_stage["pass"]
        final_stage["visual_verification"] = visual_receipts
    return {
        "schema_version": "proseekbench.claims-edge-verdict.v1",
        "edge_number": edge_number,
        **{key: value for key, value in edge.items() if key != "_visual_frames"},
        "claims_stage": final_stage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--need-evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--release-manifest", type=Path,
        help="Active release manifest that fixes the exact Sampler scope.",
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Exact Need-evaluation model keys to retain (default: all).",
    )
    parser.add_argument(
        "--selection-file", type=Path,
        help="Optional JSON edge-key list for a bounded regression run.",
    )
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--judge-model", default="qwen3.7-plus")
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--http-transport", choices=["urllib", "curl", "curl_direct"], default="urllib")
    parser.add_argument("--curl-resolve", default=None,
                        help="Optional curl --resolve value, for example host:443:IP")
    parser.add_argument("--reasoning-effort", choices=["low", "none"], default="low")
    parser.add_argument("--capability-mapping", type=Path,
                        help="Capability mapping override; defaults to the bound release view")
    parser.add_argument(
        "--prompt-version", choices=["v1"], required=True,
        help="Select the released Claims v1 contract.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--max-tokens", type=int,
        help="Response budget in tokens, default 4096.",
    )
    parser.add_argument("--keychain-service", default="proactivebench-qianwen-api-key")
    parser.add_argument("--keychain-account", default="proactivebench-qianwen")
    parser.add_argument("--visual-judge-model", default="glm-5.3-flashx")
    parser.add_argument("--visual-keychain-service", default="proactivebench-glm-api-key")
    parser.add_argument("--protocol-failure-policy", choices=["reject", "error_as_failure"],
                        default="reject")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 4096
    component(args.evaluation_id)
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0 or args.max_tokens < 160:
        parser.error("Invalid limits")

    need_dir = inside(args.need_evaluation_dir, REPO / "evaluate/reviews")
    need_manifest = read_json(need_dir / "manifest.json")
    need_summary = read_json(need_dir / "summary.json")
    need_verdicts = read_json(need_dir / "verdicts.json")
    if (need_manifest.get("status") != "complete" or need_summary.get("status") != "complete"
            or need_verdicts.get("status") != "complete"):
        raise ValueError("Need evaluation must be complete")
    source_runs = need_manifest["contract"]["runs"]
    selected_models = list(args.models or source_runs)
    if len(selected_models) != len(set(selected_models)):
        parser.error("--models contains duplicates")
    unknown_models = sorted(set(selected_models) - set(source_runs))
    if unknown_models:
        parser.error(f"Unknown --models: {', '.join(unknown_models)}")
    runs = {model: source_runs[model] for model in selected_models}
    selected_need_edges = [
        edge for edge in need_verdicts["edges"] if edge["model"] in runs
    ]
    selection_binding = None
    if args.selection_file:
        selection_path = inside(args.selection_file, REPO)
        selection = read_json(selection_path)
        keys = selection.get("edges")
        if not isinstance(keys, list) or not keys:
            parser.error("--selection-file must contain a nonempty edges list")
        required = {"model", "sampler_id", "prediction_index", "gt_need_id", "alternative_id"}
        selected_keys = set()
        for item in keys:
            if not isinstance(item, dict) or set(item) != required:
                parser.error("Invalid edge key in --selection-file")
            selected_keys.add(tuple(item[key] for key in (
                "model", "sampler_id", "prediction_index", "gt_need_id", "alternative_id",
            )))
        selected_need_edges = [edge for edge in selected_need_edges if (
            edge["model"], edge["sampler_id"], edge["prediction_index"],
            edge["gt_need_id"], edge["alternative_id"],
        ) in selected_keys]
        selection_binding = {
            "path": str(selection_path.relative_to(REPO)),
            "sha256": file_hash(selection_path),
            "selected_edges_in_component": len(selected_need_edges),
        }
    frozen_user_ids = sorted({edge["user_id"] for edge in selected_need_edges})
    if not frozen_user_ids or not selected_need_edges:
        raise ValueError("Need verdicts contain no frozen user scope")
    release_binding = None
    if args.release_manifest:
        release_path = inside(args.release_manifest, REPO)
        release = read_json(release_path)
        if release.get("status") != "active" or not release.get("sampler_inventory"):
            raise ValueError("Release manifest must be active and contain sampler_inventory")
        sampler_ids = [item["sampler_id"] for item in release["sampler_inventory"]]
        rows, input_hashes = discover(
            REPO, runs, sampler_ids=sampler_ids,
            protocol_failure_policy=args.protocol_failure_policy,
        )
        release_binding = {
            "path": str(release_path.relative_to(REPO)),
            "release_id": release.get("release_id"),
            "sha256": file_hash(release_path),
            "samplers": len(sampler_ids),
        }
    else:
        rows, input_hashes = discover(
            REPO, runs, user_ids=frozen_user_ids,
            protocol_failure_policy=args.protocol_failure_policy,
        )
    row_index = {(row["model"], row["user_id"], row["sampler_id"]): row for row in rows}
    if args.capability_mapping:
        capability_path = inside(args.capability_mapping, REPO)
    elif release_binding:
        capability_ref = release.get("capability_view", {}).get("ref")
        if not isinstance(capability_ref, str) or not capability_ref:
            raise ValueError("Bound release has no capability mapping")
        capability_path = inside(REPO / capability_ref, REPO)
    else:
        capability_path = CAPABILITY_MAPPING
    capability = read_json(capability_path)
    capability_by_sampler = {
        row["sampler_id"]: row["derived_class"] for row in capability["decisions"]
    }

    all_need_edges = []
    claims_edges = []
    for edge in sorted(selected_need_edges, key=lambda item: item["edge_number"]):
        if not edge["stage2"]["pass"]:
            continue
        row = row_index[(edge["model"], edge["user_id"], edge["sampler_id"])]
        prediction = row["predictions"][edge["prediction_index"] - 1]
        option = next(
            item for item in row["options"]
            if item["gt_need_id"] == edge["gt_need_id"]
            and item["alternative_id"] == edge["alternative_id"]
        )
        ledger, provenance = build_text_evidence_ledger(
            REPO, edge["run"], edge["user_id"], edge["sampler_id"],
            edge["prediction_index"], option["claims"],
            option["window"],
            capability_by_sampler[edge["sampler_id"]],
            row["tier"],
            ledger_version=args.prompt_version,
        )
        enriched = {
            **{key: edge[key] for key in (
                "model", "run", "user_id", "sampler_id", "episode_id", "tier",
                "prediction_id", "prediction_index", "gt_need_id", "alternative_id",
                "reference_id", "prediction_need", "reference_need",
            )},
            "prediction_claims": prediction["claims"],
            "reference_claims": option["claims"],
            "committed_after": prediction["committed_after"],
            "commit_window": option["window"],
            "time_pass": prediction["committed_after"] in option["window"],
            "evidence_ledger": ledger,
            "evidence_provenance": provenance,
            "style_diagnostics": claims_style_diagnostics(prediction["claims"]),
        }
        if args.prompt_version == "v1":
            enriched["_visual_frames"] = formal_frames_before_commit(
                REPO, edge["user_id"], edge["sampler_id"], row["tier"],
                prediction["committed_after"],
            )
        all_need_edges.append(enriched)
        passive_provenance = provenance.get("passive_evidence", {})
        for relative_path, expected_hash in passive_provenance.get("frame_hashes", {}).items():
            input_hashes[str((REPO / relative_path).resolve())] = expected_hash
        for path_key, hash_key in (
            ("episode_path", "episode_sha256"),
            ("passive_manifest_path", "passive_manifest_sha256"),
            ("stream_path", "stream_sha256"),
        ):
            if path_key in passive_provenance:
                input_hashes[str((REPO / passive_provenance[path_key]).resolve())] = passive_provenance[hash_key]
        if enriched["time_pass"]:
            claims_edges.append(enriched)

    bundle = IsolatedClaimsPromptBundle(REPO / "prompts/evaluate", args.prompt_version)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    contract = {
        "pipeline_version": PIPELINE_VERSION,
        "judge_model": args.judge_model,
        "api_base": args.api_base,
        "http_transport": args.http_transport,
        "need_evaluation_dir": str(need_dir.relative_to(REPO)),
        "need_evaluation_fingerprint": need_summary["fingerprint"],
        "need_verdicts_sha256": file_hash(need_dir / "verdicts.json"),
        "eligible_claims_edges": len(claims_edges),
        "claims_input_fingerprint": digest([claims_payload(edge) for edge in claims_edges]),
        "input": "one Need-matched and exact-Time-valid Prediction/Reference Need+Claims pair",
        "prompt_version": args.prompt_version,
        "prompt_hash": bundle.hashes["claims_isolated.md"],
        "schema_hash": bundle.hashes["claims_isolated.schema.json"],
        "scale": [1, 4],
        "acceptance_minimum": 3,
        "settings": {"temperature": 0.0, "max_tokens": args.max_tokens,
                     "response_format": "json_object", "reasoning_effort": reasoning_effort},
        "runs": runs,
        "selected_models": selected_models,
        "protocol_failure_policy": args.protocol_failure_policy,
        "protocol_failure_sampler_runs": sum(row["protocol_failure"] for row in rows),
        "frozen_user_ids": frozen_user_ids,
        "capability_mapping": {
            "path": str(capability_path.relative_to(REPO)),
            "sha256": file_hash(capability_path),
        },
        "input_hashes": input_hashes,
        "code_sha256": file_hash(Path(__file__)),
    }
    if args.curl_resolve:
        contract["curl_resolve"] = args.curl_resolve
    if args.prompt_version == "v1":
        visual_code = REPO / "code/evaluate/visual_claims_verifier.py"
        contract["visual_verifier"] = {
            "model": args.visual_judge_model,
            "code_sha256": file_hash(visual_code),
            "max_frames": 3,
            "policy": "failed non-query Passive premises only",
        }
    if release_binding:
        contract["release_manifest"] = release_binding
    if selection_binding:
        contract["selection_file"] = selection_binding
    fingerprint = digest(contract)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "fingerprint": fingerprint,
                          "need_pass_edges": len(all_need_edges),
                          "eligible_claims_edges": len(claims_edges),
                          "models": sorted(runs)}, ensure_ascii=False, indent=2))
        return 0

    output = inside(REPO / "evaluate/reviews" / args.evaluation_id, REPO / "evaluate/reviews")
    previous = None
    if output.exists():
        if not args.resume:
            raise ValueError("Claims evaluation exists; use --resume or a new evaluation-id")
        previous = read_json(output / "manifest.json")
        if previous.get("fingerprint") != fingerprint:
            old_contract = dict(previous.get("contract") or {})
            new_contract = dict(contract)
            old_code = old_contract.pop("code_sha256", None)
            new_code = new_contract.pop("code_sha256", None)
            if canonical(old_contract) != canonical(new_contract):
                raise ValueError("Claims contract changed; use a new evaluation-id")
            previous["resume_code_transition"] = {
                "from": old_code,
                "to": new_code,
                "reason": "Resume-only filtering of macOS AppleDouble sidecar files",
            }

    api_key = os.environ.get("QIANWEN_API_KEY", "")
    if not api_key:
        api_key = subprocess.check_output([
            "security", "find-generic-password", "-s", args.keychain_service,
            "-w", "-a", args.keychain_account,
        ], text=True, stderr=subprocess.DEVNULL).strip()
    output.mkdir(parents=True, exist_ok=True)
    (output / "judge").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "proseekbench.claims-evaluation-manifest.v1",
        "status": "running", "evaluation_id": args.evaluation_id,
        "fingerprint": fingerprint, "contract": contract, "workers": args.workers,
        "started_at": (previous or {}).get("started_at", datetime.now(timezone.utc).isoformat()),
        "attempt_started_at": datetime.now(timezone.utc).isoformat(), "completed_edges": 0,
    }
    if previous and previous.get("resume_code_transition"):
        manifest["resume_code_transition"] = previous["resume_code_transition"]
    write_json(output / "manifest.json", manifest)
    write_json(output / "input-hashes.before.json", input_hashes)
    for name in ("claims_isolated.md", "claims_isolated.schema.json"):
        target = output / "prompts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bundle.paths[name].read_bytes())

    judge = JudgeClient(
        bundle, output / "judge", api_base=args.api_base, model=args.judge_model,
        api_key=api_key, temperature=0.0, max_tokens=args.max_tokens,
        timeout=args.timeout, retries=args.retries, response_format="json_object",
        reasoning_effort=reasoning_effort, http_transport=args.http_transport,
        curl_resolve=args.curl_resolve,
    )
    visual = None
    if args.prompt_version in {"v3", "v4"}:
        visual_key = os.environ.get("GLM_API_KEY", "")
        if not visual_key:
            visual_key = subprocess.check_output([
                "security", "find-generic-password", "-s", args.visual_keychain_service, "-w",
            ], text=True, stderr=subprocess.DEVNULL).strip()
        visual = VisualClaimsVerifier(
            output / "visual-judge", api_key=visual_key, model=args.visual_judge_model,
        )
    results, completed = [], set()
    if args.resume:
        for path in sorted((output / "edges").glob("*.json")):
            if path.name.startswith("._"):
                continue
            result = read_json(path)
            if result.get("schema_version") == "proseekbench.claims-edge-verdict.v1":
                results.append(result)
                completed.add(result["edge_number"])
    pending = [(index, edge) for index, edge in enumerate(claims_edges, 1)
               if index not in completed]
    manifest["completed_edges"] = len(results)
    write_json(output / "manifest.json", manifest)

    try:
        failed_edges = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_edge, index, edge, judge, visual): index
                       for index, edge in pending}
            for future in as_completed(futures):
                edge_number = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_edges.append({
                        "edge_number": edge_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print(f"[failed] edge-{edge_number:04d}: {type(exc).__name__}",
                          flush=True)
                    continue
                write_json(output / "edges" / f"edge-{result['edge_number']:04d}.json", result)
                results.append(result)
                manifest["completed_edges"] = len(results)
                write_json(output / "manifest.json", manifest)
                print(f"[{len(results)}/{len(claims_edges)}] edge-{result['edge_number']:04d}: "
                      f"score={result['claims_stage']['score']} "
                      f"pass={result['claims_stage']['pass']}", flush=True)

        if failed_edges:
            write_json(output / "failed-edges.json", {
                "status": "retry_required", "edges": failed_edges,
            })
            manifest.update(
                status="incomplete_retry_required",
                failed_edges=len(failed_edges),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            write_json(output / "manifest.json", manifest)
            write_json(output / "integrity-verification.json", check_integrity(input_hashes))
            print(json.dumps({"status": manifest["status"],
                              "completed": len(results),
                              "failed": len(failed_edges)}, ensure_ascii=False))
            return 3

        results.sort(key=lambda item: item["edge_number"])
        integrity = check_integrity(input_hashes)
        write_json(output / "integrity-verification.json", integrity)
        claims_results = {edge_key(item): item for item in results}
        need_keys = {edge_key(item) for item in all_need_edges}
        time_keys = {edge_key(item) for item in claims_edges}
        overall_keys = {key for key, item in claims_results.items()
                        if item["claims_stage"]["pass"]}

        scored_rows = []
        for row in rows:
            matches = {"need": [], "time": [], "overall": []}
            for option in row["options"]:
                for prediction_index, _ in enumerate(row["predictions"], 1):
                    key = (row["model"], row["user_id"], row["sampler_id"],
                           prediction_index, option["gt_need_id"], option["alternative_id"])
                    item = {"prediction_index": prediction_index,
                            "gt_need_id": option["gt_need_id"],
                            "alternative_id": option["alternative_id"]}
                    if key in need_keys:
                        matches["need"].append(item)
                    if key in time_keys:
                        matches["time"].append(item)
                    if key in overall_keys:
                        matches["overall"].append(item)
            scored = {key: value for key, value in row.items()
                      if key not in ("predictions", "options")}
            scored["scores"] = {}
            for stage in ("need", "time", "overall"):
                tp = len(matches[stage])
                scored["scores"][stage] = {
                    "status": "complete", "matches": matches[stage],
                    "counts": {"tp": tp, "fp": row["prediction_count"] - tp,
                               "fn": row["gt_count"] - tp},
                }
            scored_rows.append(scored)

        summary = {
            "schema_version": "proseekbench.claims-summary.v1",
            "status": "complete" if integrity["status"] == "pass" else "invalid_inputs_changed",
            "evaluation_id": args.evaluation_id, "fingerprint": fingerprint,
            "scope": f"Isolated {args.judge_model} Claims 1-4 after frozen Need and exact Time; scores 3-4 pass",
            "need_pass_edges": len(all_need_edges),
            "eligible_claims_edges": len(results),
            "score_counts": {str(score): sum(item["claims_stage"]["score"] == score
                                               for item in results) for score in (1, 2, 3, 4)},
            "passed_edges": sum(item["claims_stage"]["pass"] for item in results),
            "failed_edges": sum(not item["claims_stage"]["pass"] for item in results),
            "style_diagnostics": {
                "over_700_chars": sum(item["style_diagnostics"]["character_count"] > 700
                                      for item in results),
                "over_850_chars": sum(item["style_diagnostics"]["character_count"] > 850
                                      for item in results),
                "typed_handle_mentions": sum(item["style_diagnostics"]["typed_handle_mentions"]
                                             for item in results),
                "query_process_markers": sum(item["style_diagnostics"]["query_process_markers"]
                                             for item in results),
            },
            "models": [{"model": model, "run": relative,
                        "metrics": aggregate([row for row in scored_rows if row["model"] == model])}
                       for model, relative in runs.items()],
        }
        write_json(output / "verdicts.json", {"status": summary["status"], "edges": results})
        write_json(output / "summary.json", summary)
        manifest.update(status=summary["status"], finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": summary["status"],
                          "summary": str(output / "summary.json")}, ensure_ascii=False))
        return 0 if summary["status"] == "complete" else 2
    except BaseException:
        manifest.update(status="interrupted_or_error",
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        write_json(output / "integrity-verification.json", check_integrity(input_hashes))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
