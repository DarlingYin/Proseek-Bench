#!/usr/bin/env python3
"""Stage-1 global router using a structured-output text model.

One request contains the complete Prediction/GT payload for one Sampler and
scores the full Cartesian pair list with the frozen score-twice 0--5 prompt.
Code removes 0--2, collapses OR alternatives, and performs deterministic
global one-to-one maximum-weight assignment.  No isolated Judge or metrics are
run here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from judge_client import JudgeClient, canonical, digest, file_hash, read_json, write_json
from matching import route_global_top1
from run_evaluation import build_input, check_integrity, component, discover, inside
from whole_file_judge import WholeFilePromptBundle


REPO = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = "text-stage1-router.v1-global-top1"
# The runner permits a resume-only code transition for deterministic punctuation repair in quoted witnesses.
API_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def route_row(row: dict, judge: JudgeClient, max_input_chars: int,
              max_pairs_per_request: int = 0) -> dict:
    payload = build_input(row)
    if len(canonical(payload)) > max_input_chars:
        raise ValueError("Whole-Sampler Stage-1 input exceeds configured character limit")
    responses = []
    entries = []
    if row["predictions"] and row["options"]:
        prediction_rows = payload["prediction"]["predictions"]
        if max_pairs_per_request and len(payload["pairs"]) > max_pairs_per_request:
            per_chunk = max(1, max_pairs_per_request // len(payload["ground_truth"]["needs"]))
            payloads = []
            for start in range(0, len(prediction_rows), per_chunk):
                selected_predictions = prediction_rows[start:start + per_chunk]
                selected_ids = {item["prediction_id"] for item in selected_predictions}
                payloads.append({
                    "prediction": {"predictions": selected_predictions},
                    "ground_truth": payload["ground_truth"],
                    "pairs": [pair for pair in payload["pairs"] if pair["prediction_id"] in selected_ids],
                })
        else:
            payloads = [payload]
        results = []
        for request_payload in payloads:
            response = judge.judge_once("need_prefilter", request_payload)
            if response.get("verdict") is None:
                raise ValueError(f"Stage-1 Judge failed: {response.get('error', 'unknown_error')}")
            responses.append(response)
            results.extend(response["verdict"]["results"])
        values = {(item["prediction_id"], item["reference_id"]): item for item in results}
        for i, prediction in enumerate(row["predictions"], 1):
            for j, option in enumerate(row["options"], 1):
                pid, rid = f"p{i:04d}", f"r{j:04d}"
                result = values[(pid, rid)]
                if result["score"] is None:
                    raise ValueError("Stage-1 returned an unresolved score; routing stopped")
                score = int(result["score"])
                entries.append({
                    "prediction_index": i, "prediction_id": pid,
                    "option_index": j, "reference_id": rid,
                    "gt_need_id": option["gt_need_id"],
                    "alternative_id": option["alternative_id"],
                    "level": score,
                    "route_weight": score,
                    "reason": result["reason"],
                    "material_deviation": result["material_deviation"],
                })
    selected, statuses = route_global_top1(entries)
    for entry in entries:
        entry["routing_status"] = statuses[(entry["prediction_index"], entry["option_index"])]
    selected_predictions = [entry["prediction_index"] for entry in selected]
    selected_gt = [entry["gt_need_id"] for entry in selected]
    if len(selected_predictions) != len(set(selected_predictions)):
        raise ValueError("A Prediction was assigned to multiple GT Needs")
    if len(selected_gt) != len(set(selected_gt)):
        raise ValueError("A GT Need was assigned to multiple Predictions")
    parent_gt_ids = sorted({option["gt_need_id"] for option in row["options"]})
    selected_rows = [{
        "prediction_index": entry["prediction_index"],
        "prediction_id": entry["prediction_id"],
        "prediction_need": row["predictions"][entry["prediction_index"] - 1]["need"],
        "gt_need_id": entry["gt_need_id"],
        "alternative_id": entry["alternative_id"],
        "reference_id": entry["reference_id"],
        "reference_need": row["options"][entry["option_index"] - 1]["need"],
        "score": entry["level"], "reason": entry["reason"],
    } for entry in selected]
    return {
        "schema_version": "proseekbench.text-stage1-routing.v1",
        "status": "complete", "model": row["model"], "run": row["run"],
        "user_id": row["user_id"], "sampler_id": row["sampler_id"],
        "episode_id": row["episode_id"], "tier": row["tier"],
        "prediction_count": len(row["predictions"]), "gt_need_count": row["gt_count"],
        "expanded_reference_count": len(row["options"]),
        "matrix_pair_count": len(entries),
        "requests": [{"cache_key": response.get("cache_key"),
                      "cache_hit": response.get("cache_hit", False)} for response in responses],
        "request_count": len(responses),
        "partition_policy": ("prediction_chunks_with_full_gt" if len(responses) > 1 else "whole_sampler"),
        "matrix": entries, "selected": selected_rows,
        "unmatched_prediction_indices": sorted(
            set(range(1, len(row["predictions"]) + 1)) - set(selected_predictions)
        ),
        "unmatched_gt_need_ids": sorted(set(parent_gt_ids) - set(selected_gt)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True,
                        help="Frozen release manifest that defines the exact Sampler scope")
    parser.add_argument("--routing-id", required=True)
    parser.add_argument(
        "--judge-model",
        choices=["qwen3.7-plus", "deepseek-v4.1-flash", "glm-5.3"],
        default="qwen3.7-plus",
    )
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--http-transport", choices=["urllib", "curl", "curl_direct"], default="urllib")
    parser.add_argument("--curl-resolve", default=None,
                        help="Optional curl --resolve value, for example host:443:IP")
    parser.add_argument("--reasoning-effort", choices=["low", "none"], default="low")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-input-chars", type=int, default=120000)
    parser.add_argument("--max-pairs-per-request", type=int, default=0,
                        help="Transport-only Prediction chunking threshold; zero keeps one whole-Sampler request")
    parser.add_argument("--protocol-failure-policy", choices=["reject", "error_as_failure"],
                        default="reject")
    parser.add_argument("--keychain-service", default="proactivebench-qianwen-api-key")
    parser.add_argument("--keychain-account", default="proactivebench-qianwen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    component(args.routing_id)
    if (args.workers < 1 or args.timeout <= 0 or args.retries < 0 or args.max_tokens < 160
            or args.max_pairs_per_request < 0):
        parser.error("Invalid limits")
    runs = read_json(args.runs)
    release_path = args.release_manifest.resolve()
    release = read_json(release_path)
    inventory = release.get("sampler_inventory")
    if release.get("status") != "active" or not isinstance(inventory, list) or not inventory:
        parser.error("--release-manifest must be an active release with a nonempty inventory")
    sampler_ids = [item["sampler_id"] for item in inventory]
    if len(sampler_ids) != len(set(sampler_ids)):
        parser.error("--release-manifest contains duplicate sampler_id values")
    release_binding = {
        "path": str(release_path), "sha256": file_hash(release_path),
        "release_id": release.get("release_id"), "samplers": len(sampler_ids),
    }
    rows, hashes = discover(
        REPO, runs, sampler_ids=sampler_ids,
        protocol_failure_policy=args.protocol_failure_policy,
    )
    bundle = WholeFilePromptBundle(REPO / "prompts/evaluate", "score-twice", "v1")
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    contract = {
        "pipeline_version": PIPELINE_VERSION, "judge_model": args.judge_model,
        "api_base": args.api_base, "http_transport": args.http_transport,
        "runs": runs, "prompt_hash": bundle.hashes["need_prefilter.md"],
        "schema_hash": bundle.hashes["need_prefilter.schema.json"],
        "input": ("complete whole-Sampler Prediction/GT payload and full Cartesian pair list"
                  if not args.max_pairs_per_request else
                  "Prediction-partitioned transport with the complete GT list in every request; all Cartesian pairs are recombined before global matching"),
        "max_pairs_per_request": args.max_pairs_per_request,
        "filter": "delete scores 0-2",
        "or_policy": "best alternative per Prediction-parent GT before matching",
        "assignment": "global maximum-cardinality then maximum-score one-to-one",
        "stage2": "disabled", "need_time_metrics": "not_computed",
        "protocol_failure_policy": args.protocol_failure_policy,
        "protocol_failure_sampler_runs": sum(row["protocol_failure"] for row in rows),
        "settings": {"temperature": 0.0, "max_tokens": args.max_tokens,
                     "response_format": "json_object", "reasoning_effort": reasoning_effort},
        "release_binding": release_binding,
        "code_sha256": file_hash(Path(__file__)), "input_hashes": hashes,
        "selection": sorted((row["model"], row["user_id"], row["sampler_id"], row["episode_id"])
                            for row in rows),
    }
    if args.curl_resolve:
        contract["curl_resolve"] = args.curl_resolve
    fingerprint = digest(contract)
    if args.dry_run:
        requests = 0
        for row in rows:
            pairs = len(row["predictions"]) * len(row["options"])
            if not pairs:
                continue
            requests += (1 if not args.max_pairs_per_request else
                         (pairs + args.max_pairs_per_request - 1) // args.max_pairs_per_request)
        pairs = sum(len(row["predictions"]) * len(row["options"]) for row in rows)
        maximum = max((len(canonical(build_input(row))) for row in rows), default=0)
        print(canonical({
            "status": "dry_run", "routing_id": args.routing_id,
            "release_binding": release_binding, "models": len(runs),
            "sampler_runs": len(rows), "samplers_per_model": len(rows) // len(runs),
            "judge_requests": requests, "matrix_pairs": pairs,
            "max_whole_sampler_input_chars": maximum,
            "max_input_chars": args.max_input_chars,
            "protocol_failure_sampler_runs": sum(row["protocol_failure"] for row in rows),
            "fingerprint": fingerprint,
        }))
        return 0
    output = inside(REPO / "evaluate/reviews" / args.routing_id, REPO / "evaluate/reviews")
    previous = None
    if output.exists():
        if not args.resume:
            raise ValueError("Routing audit exists; use --resume or a new routing-id")
        previous = read_json(output / "manifest.json")
        if previous.get("fingerprint") != fingerprint:
            old_contract = dict(previous.get("contract") or {})
            new_contract = dict(contract)
            old_code = old_contract.pop("code_sha256", None)
            new_code = new_contract.pop("code_sha256", None)
            old_hashes = old_contract.pop("input_hashes", {})
            new_hashes = new_contract.pop("input_hashes", {})
            old_settings = dict(old_contract.get("settings") or {})
            new_settings = dict(new_contract.get("settings") or {})
            old_max_tokens = old_settings.get("max_tokens")
            new_max_tokens = new_settings.get("max_tokens")
            if (isinstance(old_max_tokens, int) and isinstance(new_max_tokens, int)
                    and min(old_max_tokens, new_max_tokens) >= 8192):
                old_settings["max_tokens"] = new_max_tokens
                old_contract["settings"] = old_settings
            if (previous.get("status") in {"interrupted_or_error", "incomplete_retry_required", "running"}
                    and int(new_contract.get("max_pairs_per_request") or 0) > 0):
                old_contract["max_pairs_per_request"] = new_contract["max_pairs_per_request"]
                old_contract["input"] = new_contract["input"]
                old_contract["api_base"] = new_contract["api_base"]
            if any(old_hashes.get(path) != value for path, value in new_hashes.items()):
                raise ValueError("Selected release inputs changed; use a new routing-id")
            if canonical(old_contract) != canonical(new_contract):
                changed_keys = sorted(
                    key for key in set(old_contract) | set(new_contract)
                    if canonical(old_contract.get(key)) != canonical(new_contract.get(key))
                )
                raise ValueError(
                    "Routing inputs, prompt contract, or Judge settings changed: "
                    + ", ".join(changed_keys)
                )
            previous["resume_code_transition"] = {"from": old_code, "to": new_code,
                                                   "reason": "Resume-only runner support and nondecreasing output-token limit for unfinished rows"}
    api_key = os.environ.get("QIANWEN_API_KEY", "")
    if not api_key:
        api_key = subprocess.check_output([
            "security", "find-generic-password", "-s", args.keychain_service,
            "-w", "-a", args.keychain_account,
        ], text=True, stderr=subprocess.DEVNULL).strip()
    output.mkdir(parents=True, exist_ok=True)
    (output / "judge").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "proseekbench.text-stage1-router-manifest.v1",
        "status": "running", "routing_id": args.routing_id,
        "fingerprint": fingerprint, "contract": contract, "workers": args.workers,
        "started_at": (previous or {}).get("started_at", datetime.now(timezone.utc).isoformat()),
        "attempt_started_at": datetime.now(timezone.utc).isoformat(),
        "completed_sampler_runs": 0,
    }
    if previous and previous.get("resume_code_transition"):
        manifest["resume_code_transition"] = previous["resume_code_transition"]
    write_json(output / "manifest.json", manifest)
    write_json(output / "input-hashes.before.json", hashes)
    for name in ("need_prefilter.md", "need_prefilter.schema.json"):
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
    results = []
    completed = set()
    if args.resume:
        for path in sorted((output / "samplers").glob("*/*/*/routing.json")):
            result = read_json(path)
            if result.get("status") != "complete":
                continue
            key = result["model"], result["user_id"], result["sampler_id"], result["episode_id"]
            completed.add(key)
            results.append(result)
    pending_rows = [row for row in rows if (
        row["model"], row["user_id"], row["sampler_id"], row["episode_id"]
    ) not in completed]
    manifest["completed_sampler_runs"] = len(results)
    manifest["resumed_completed_sampler_runs"] = len(results)
    write_json(output / "manifest.json", manifest)
    try:
        failed_rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(route_row, row, judge, args.max_input_chars,
                                       args.max_pairs_per_request): row
                       for row in pending_rows}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failure = {
                        "model": row["model"], "user_id": row["user_id"],
                        "sampler_id": row["sampler_id"], "episode_id": row["episode_id"],
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
                    failed_rows.append(failure)
                    print(f"[failed] {row['model']} {row['sampler_id']}: {type(exc).__name__}", flush=True)
                    continue
                target = output / "samplers" / result["model"] / result["user_id"] / result["sampler_id"] / "routing.json"
                write_json(target, result)
                results.append(result)
                manifest["completed_sampler_runs"] = len(results)
                write_json(output / "manifest.json", manifest)
                print(f"[{len(results)}/{len(rows)}] {result['model']} {result['sampler_id']}: complete", flush=True)
        if failed_rows:
            write_json(output / "failed-rows.json", {"status": "retry_required", "rows": failed_rows})
            manifest.update(status="incomplete_retry_required", failed_sampler_runs=len(failed_rows),
                            finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "manifest.json", manifest)
            write_json(output / "integrity-verification.json", check_integrity(hashes))
            print(json.dumps({"status": manifest["status"], "completed": len(results),
                              "failed": len(failed_rows)}, ensure_ascii=False))
            return 3
        integrity = check_integrity(hashes)
        write_json(output / "integrity-verification.json", integrity)
        totals = {
            "sampler_runs": len(results),
            "matrix_pairs": sum(row["matrix_pair_count"] for row in results),
            "judge_requests": sum(row.get("request_count", int(bool(row["matrix_pair_count"])))
                                  for row in results),
            "filtered_score_0_2": sum(
                entry["routing_status"] == "filtered_score_0_2"
                for row in results for entry in row["matrix"]
            ),
            "selected_global_top1": sum(len(row["selected"]) for row in results),
            "unmatched_predictions": sum(len(row["unmatched_prediction_indices"]) for row in results),
            "unmatched_gt_needs": sum(len(row["unmatched_gt_need_ids"]) for row in results),
        }
        summary = {
            "schema_version": "proseekbench.text-stage1-router-summary.v1",
            "status": "complete" if integrity["status"] == "pass" else "invalid_inputs_changed",
            "routing_id": args.routing_id, "fingerprint": fingerprint,
            "scope": "Stage-1 routing audit only; no isolated Judge, Need metric, or Time metric",
            "totals": totals,
        }
        write_json(output / "summary.json", summary)
        manifest.update(status=summary["status"], finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": summary["status"], "summary": str(output / "summary.json")}, ensure_ascii=False))
        return 0 if summary["status"] == "complete" else 2
    except BaseException:
        manifest.update(status="interrupted_or_error", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        write_json(output / "integrity-verification.json", check_integrity(hashes))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
