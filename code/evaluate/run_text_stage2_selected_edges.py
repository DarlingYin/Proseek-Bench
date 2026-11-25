#!/usr/bin/env python3
"""Run isolated 1--4 Judge on Stage-1 selected one-to-one edges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from judge_client import JudgeClient, digest, file_hash, read_json, write_json
from metrics import aggregate
from run_evaluation import check_integrity, component, discover, inside
from whole_file_judge import WholeFilePromptBundle


REPO = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = "text-stage2-isolated.v1-selected-edges"
# Selected-edge semantics are unchanged; quoted witness punctuation is repaired before validation.
API_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def isolated_payload(edge: dict) -> dict:
    return {
        "prediction": {"prediction_id": "p0001", "need": edge["prediction_need"]},
        "ground_truth": {"reference_id": "r0001", "need": edge["reference_need"]},
    }


def judge_edge(edge_number: int, edge: dict, judge: JudgeClient) -> dict:
    response = judge.judge_once("need_isolated", isolated_payload(edge))
    if response.get("verdict") is None:
        raise ValueError(f"Stage-2 Judge failed: {response.get('error', 'unknown_error')}")
    raw = response["verdict"]["results"][0]
    verdict = judge.bundle.normalize("need_isolated", raw)
    return {
        "schema_version": "proseekbench.stage2-edge-verdict.v1",
        "edge_number": edge_number, **edge,
        "stage2": {
            "score": raw["score"], "pass": verdict["pass"],
            "reason": raw["reason"], "cache_key": response.get("cache_key"),
            "cache_hit": response.get("cache_hit", False),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True,
                        help="Frozen release manifest defining the exact Sampler scope")
    parser.add_argument("--selected-edges", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
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
    parser.add_argument("--prompt-version", choices=["v1"], default="v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--protocol-failure-policy", choices=["reject", "error_as_failure"],
                        default="reject")
    parser.add_argument("--keychain-service", default="proactivebench-qianwen-api-key")
    parser.add_argument("--keychain-account", default="proactivebench-qianwen")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    component(args.evaluation_id)
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0 or args.max_tokens < 160:
        parser.error("Invalid limits")
    selected = read_json(args.selected_edges)
    if selected.get("status") != "complete" or not isinstance(selected.get("edges"), list):
        raise ValueError("Selected-edge artifact is not complete")
    edges = selected["edges"]
    runs = read_json(args.runs)
    release_path = args.release_manifest.resolve()
    release = read_json(release_path)
    inventory = release.get("sampler_inventory")
    if release.get("status") != "active" or not isinstance(inventory, list) or not inventory:
        parser.error("--release-manifest must be an active release with a nonempty inventory")
    sampler_ids = [item["sampler_id"] for item in inventory]
    if len(sampler_ids) != len(set(sampler_ids)):
        parser.error("--release-manifest contains duplicate sampler_id values")
    rows, input_hashes = discover(
        REPO, runs, sampler_ids=sampler_ids,
        protocol_failure_policy=args.protocol_failure_policy,
    )
    row_index = {(row["model"], row["user_id"], row["sampler_id"]): row for row in rows}
    bundle = WholeFilePromptBundle(REPO / "prompts/evaluate", "score-twice", args.prompt_version)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    contract = {
        "pipeline_version": PIPELINE_VERSION, "judge_model": args.judge_model,
        "api_base": args.api_base, "http_transport": args.http_transport,
        "selected_edges_path": str(args.selected_edges),
        "selected_edges_sha256": file_hash(args.selected_edges),
        "routing_id": selected["routing_id"], "routing_fingerprint": selected["routing_fingerprint"],
        "edge_count": len(edges), "prompt_version": args.prompt_version,
        "prompt_hash": bundle.hashes["need_isolated.md"],
        "schema_hash": bundle.hashes["need_isolated.schema.json"],
        "scale": [1, 4], "acceptance_minimum": 3,
        "settings": {"temperature": 0.0, "max_tokens": args.max_tokens,
                     "response_format": "json_object", "reasoning_effort": reasoning_effort},
        "release_binding": {
            "path": str(release_path), "sha256": file_hash(release_path),
            "release_id": release.get("release_id"), "samplers": len(sampler_ids),
        },
        "runs": runs, "input_hashes": input_hashes,
        "protocol_failure_policy": args.protocol_failure_policy,
        "protocol_failure_sampler_runs": sum(row["protocol_failure"] for row in rows),
        "code_sha256": file_hash(Path(__file__)),
    }
    if args.curl_resolve:
        contract["curl_resolve"] = args.curl_resolve
    fingerprint = digest(contract)
    output = inside(REPO / "evaluate/reviews" / args.evaluation_id, REPO / "evaluate/reviews")
    previous = None
    if output.exists():
        if not args.resume:
            raise ValueError("Stage-2 evaluation exists; use --resume or a new evaluation-id")
        previous = read_json(output / "manifest.json")
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Stage-2 contract changed; use a new evaluation-id")
    api_key = os.environ.get("QIANWEN_API_KEY", "")
    if not api_key:
        api_key = subprocess.check_output([
            "security", "find-generic-password", "-s", args.keychain_service,
            "-w", "-a", args.keychain_account,
        ], text=True, stderr=subprocess.DEVNULL).strip()
    output.mkdir(parents=True, exist_ok=True)
    (output / "judge").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "proseekbench.stage2-evaluation-manifest.v1",
        "status": "running", "evaluation_id": args.evaluation_id,
        "fingerprint": fingerprint, "contract": contract, "workers": args.workers,
        "started_at": (previous or {}).get("started_at", datetime.now(timezone.utc).isoformat()),
        "attempt_started_at": datetime.now(timezone.utc).isoformat(), "completed_edges": 0,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "input-hashes.before.json", input_hashes)
    for name in ("need_isolated.md", "need_isolated.schema.json"):
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
    results, completed = [], set()
    if args.resume:
        for path in sorted((output / "edges").glob("*.json")):
            if path.name.startswith("._"):
                continue
            result = read_json(path)
            if result.get("schema_version") == "proseekbench.stage2-edge-verdict.v1":
                results.append(result)
                completed.add(result["edge_number"])
    pending = [(index, edge) for index, edge in enumerate(edges, 1) if index not in completed]
    manifest["completed_edges"] = len(results)
    write_json(output / "manifest.json", manifest)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_edge, index, edge, judge): index
                       for index, edge in pending}
            for future in as_completed(futures):
                result = future.result()
                write_json(output / "edges" / f"edge-{result['edge_number']:04d}.json", result)
                results.append(result)
                manifest["completed_edges"] = len(results)
                write_json(output / "manifest.json", manifest)
                print(f"[{len(results)}/{len(edges)}] edge-{result['edge_number']:04d}: "
                      f"score={result['stage2']['score']} pass={result['stage2']['pass']}", flush=True)
        results.sort(key=lambda item: item["edge_number"])
        integrity = check_integrity(input_hashes)
        write_json(output / "integrity-verification.json", integrity)

        # Build per-Sampler Need/Time counts from the unique Stage-1 assignment.
        edge_results = {(item["model"], item["user_id"], item["sampler_id"],
                         item["prediction_index"], item["gt_need_id"],
                         item["alternative_id"]): item for item in results}
        scored_rows = []
        for row in rows:
            accepted = []
            time_accepted = []
            for option in row["options"]:
                for prediction_index, prediction in enumerate(row["predictions"], 1):
                    key = (row["model"], row["user_id"], row["sampler_id"], prediction_index,
                           option["gt_need_id"], option["alternative_id"])
                    result = edge_results.get(key)
                    if result is None or not result["stage2"]["pass"]:
                        continue
                    edge = {"prediction_index": prediction_index,
                            "gt_need_id": option["gt_need_id"],
                            "alternative_id": option["alternative_id"]}
                    accepted.append(edge)
                    if prediction["committed_after"] in option["window"]:
                        time_accepted.append(edge)
            need_tp = len(accepted)
            time_tp = len(time_accepted)
            row_result = {key: value for key, value in row.items()
                          if key not in ("predictions", "options")}
            row_result["scores"] = {
                "need": {"status": "complete", "matches": accepted,
                         "counts": {"tp": need_tp,
                                    "fp": row["prediction_count"] - need_tp,
                                    "fn": row["gt_count"] - need_tp}},
                "time": {"status": "complete", "matches": time_accepted,
                         "counts": {"tp": time_tp,
                                    "fp": row["prediction_count"] - time_tp,
                                    "fn": row["gt_count"] - time_tp}},
                "overall": {"status": "not_evaluated", "counts": None, "matches": None},
            }
            scored_rows.append(row_result)
        summary = {
            "schema_version": "proseekbench.stage2-summary.v1",
            "status": "complete" if integrity["status"] == "pass" else "invalid_inputs_changed",
            "evaluation_id": args.evaluation_id, "fingerprint": fingerprint,
            "scope": "Qwen3.7-Plus isolated Need 1-4 on Stage-1 selected one-to-one edges; exact Time; Claims disabled",
            "edge_count": len(results),
            "score_counts": {str(score): sum(item["stage2"]["score"] == score for item in results)
                             for score in (1, 2, 3, 4)},
            "passed_edges": sum(item["stage2"]["pass"] for item in results),
            "failed_edges": sum(not item["stage2"]["pass"] for item in results),
            "models": [{"model": model, "run": relative,
                        "metrics": aggregate([row for row in scored_rows if row["model"] == model])}
                       for model, relative in runs.items()],
        }
        write_json(output / "verdicts.json", {"status": summary["status"], "edges": results})
        write_json(output / "summary.json", summary)
        manifest.update(status=summary["status"], finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": summary["status"], "summary": str(output / "summary.json")}, ensure_ascii=False))
        return 0 if summary["status"] == "complete" else 2
    except BaseException:
        manifest.update(status="interrupted_or_error", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        write_json(output / "integrity-verification.json", check_integrity(input_hashes))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
