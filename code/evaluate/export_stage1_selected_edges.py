#!/usr/bin/env python3
"""Export selected Stage-1 global one-to-one edges from a routing audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from judge_client import file_hash, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.routing_dir / "manifest.json")
    summary = read_json(args.routing_dir / "summary.json")
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        raise ValueError("Routing audit is not complete")
    edges, hashes = [], {}
    for path in sorted((args.routing_dir / "samplers").glob("*/*/*/routing.json")):
        row = read_json(path)
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete routing row: {path}")
        hashes[str(path)] = file_hash(path)
        for selected in row["selected"]:
            edges.append({
                "model": row["model"], "run": row["run"],
                "user_id": row["user_id"], "sampler_id": row["sampler_id"],
                "episode_id": row["episode_id"], "tier": row["tier"],
                "prediction_id": selected["prediction_id"],
                "prediction_index": selected["prediction_index"],
                "gt_need_id": selected["gt_need_id"],
                "alternative_id": selected["alternative_id"],
                "reference_id": selected["reference_id"],
                "score": selected["score"], "reason": selected["reason"],
                "prediction_need": selected["prediction_need"],
                "reference_need": selected["reference_need"],
            })
    edges.sort(key=lambda row: (row["model"], row["user_id"], row["sampler_id"],
                               row["prediction_index"], row["gt_need_id"],
                               row["alternative_id"] or ""))
    by_model = {model: sum(edge["model"] == model for edge in edges)
                for model in sorted({edge["model"] for edge in edges})}
    payload = {
        "schema_version": "proseekbench.stage1-selected-edges.v1",
        "status": "complete", "routing_id": summary["routing_id"],
        "routing_fingerprint": summary["fingerprint"],
        "scope": "Selected binary edges after score 0-2 filtering, OR collapse, and global one-to-one matching; Stage 2 and Time not evaluated.",
        "edge_count": len(edges), "by_model": by_model,
        "source_hashes": hashes, "edges": edges,
    }
    write_json(args.output_json, payload)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edges[0]) if edges else [])
        if edges:
            writer.writeheader()
            writer.writerows(edges)
    print(json.dumps({"status": "complete", "edges": len(edges), "by_model": by_model,
                      "json": str(args.output_json), "csv": str(args.output_csv)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
