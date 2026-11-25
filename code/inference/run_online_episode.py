#!/usr/bin/env python3
"""Replay model decisions through the ProactiveBench Online Runtime.

This command intentionally separates the benchmark state machine from any one
model vendor.  ``--decisions`` is a JSONL file containing one agent output per
model call.  A later API adapter only needs to replace this decision iterator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_runtime import OnlineEpisodeRuntime, read_jsonl, write_json
from experiment_modes import add_mode_arguments, resolve_mode, check_no_search_result
from artifact_paths import require_external_output
from generic_tool_adapter import add_tool_interface_argument


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler-root", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        help="Episode to replay; defaults to the first sampler episode.",
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-window", type=int, default=8)
    parser.add_argument("--image-window", type=int, default=3)
    add_mode_arguments(parser)
    add_tool_interface_argument(parser)
    args = parser.parse_args()
    try:
        condition = resolve_mode(args, Path(__file__).resolve().parents[2])
    except ValueError as exc:
        parser.error(str(exc))
    args.output = require_external_output(Path(__file__).resolve().parents[2], args.output)

    runtime = OnlineEpisodeRuntime(
        args.sampler_root,
        episode_id=args.episode_id,
        action_window=args.action_window,
        image_window=args.image_window,
        search_budget=args.search_budget,
        max_searches_per_observation=args.max_searches_per_observation,
        search_disabled_by_condition=args.inference_mode == "no_search",
        tool_interface=args.tool_interface,
    )
    decisions = read_jsonl(args.decisions)
    for index, decision in enumerate(decisions):
        if runtime.complete:
            raise SystemExit(f"Decision {index} exists after the episode completed")
        runtime.apply_agent_output(decision)
    if not runtime.complete:
        raise SystemExit(
            "Decision stream ended before the episode; next observation is "
            + runtime.current_observation["observation_id"]
        )
    result = runtime.result()
    check_no_search_result(args.inference_mode, result)
    result["configuration"].update(inference_mode=args.inference_mode, experiment_condition=condition)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "predictions": len(runtime.committed_predictions),
                "query_count": runtime.query_count,
                "cost_units": runtime.used_search_cost,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
