#!/usr/bin/env python3
"""Load, validate, and aggregate OR alternatives in recommendation GT."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def load_recommendation_alternatives(
    ground_truth_path: Path,
    need_id: str | None = None,
) -> dict[str, Any]:
    ground_truth = _object(json.loads(ground_truth_path.read_text(encoding="utf-8")), "ground truth")
    needs = ground_truth.get("needs")
    if not isinstance(needs, list):
        raise ValueError("Ground truth needs must be an array")
    recommendation_needs = [
        need for need in needs
        if isinstance(need, dict)
        and "alternatives" in need
        and (need_id is None or need.get("need_id") == need_id)
    ]
    if len(recommendation_needs) != 1:
        raise ValueError("Expected exactly one recommendation Need with alternatives")
    need = recommendation_needs[0]
    if set(need) != {"need_id", "commit_time_sets", "alternatives"}:
        raise ValueError(
            "A recommendation Need must contain only need_id, shared "
            "commit_time_sets, and alternatives"
        )
    value = _object(need["alternatives"], "alternatives")
    allowed_top = {"relation", "match_policy", "options"}
    if set(value) != allowed_top:
        raise ValueError("Recommendation alternatives have unexpected or missing fields")
    if value["relation"] != "OR":
        raise ValueError("Recommendation alternatives must use OR semantics")
    if value["match_policy"] != "any_one_supported_option_satisfies_need":
        raise ValueError("Unknown recommendation alternative match policy")
    resolved_need_id = _string(need.get("need_id"), "need_id")
    goal_commit_times = need.get("commit_time_sets")
    if not isinstance(goal_commit_times, list) or not goal_commit_times:
        raise ValueError("Recommendation Need must define the shared commit_time_sets")
    def validate_window(window: Any, label: str) -> set[str]:
        if (not isinstance(window, list) or not window
                or any(not isinstance(t, str) or not re.fullmatch(r"o[0-9]+", t) for t in window)
                or len(set(window)) != len(window)):
            raise ValueError(f"{label} must contain nonempty unique observation IDs")
        return set(window)

    goal_window = validate_window(goal_commit_times, "Need commit_time_sets")

    alternatives = value["options"]
    if not isinstance(alternatives, list) or len(alternatives) < 2:
        raise ValueError("At least two recommendation alternatives are required")
    seen_ids: set[str] = set()
    covered_times: set[str] = set()
    for index, alternative in enumerate(alternatives):
        alternative = _object(alternative, f"alternatives[{index}]")
        expected = {
            "alternative_id", "need", "need_claims",
        }
        if set(alternative) not in (expected, expected | {"commit_time_sets"}):
            raise ValueError(f"alternatives[{index}] has unexpected or missing fields")
        alternative_id = _string(alternative["alternative_id"], "alternative_id")
        if alternative_id in seen_ids:
            raise ValueError("alternative_id values must be unique")
        seen_ids.add(alternative_id)
        _string(alternative["need"], "alternative need")
        _string(alternative["need_claims"], "alternative need_claims")
        option_window = validate_window(
            alternative.get("commit_time_sets", goal_commit_times),
            f"alternatives[{index}].commit_time_sets",
        )
        if not option_window <= goal_window:
            raise ValueError("An option Commit window must be a subset of the Need window")
        covered_times.update(option_window)
    if covered_times != goal_window:
        raise ValueError("The Need Commit window must equal the union of option windows")
    return value


def recommendation_need_satisfied(
    ground_truth_path: Path,
    option_matches: Mapping[str, bool],
    need_id: str | None = None,
    *,
    committed_after: str | None = None,
) -> bool:
    """Return the single Need result using the GT's OR match policy.

    A semantic/candidate evaluator supplies one boolean for every option.  The
    alternatives are references for one Need, so any supported option satisfies
    that Need and can contribute at most one Need hit.
    """
    expanded = expand_recommendation_need(ground_truth_path, need_id)
    expected_ids = {row["alternative_id"] for row in expanded}
    if set(option_matches) != expected_ids:
        raise ValueError("option_matches must contain exactly the GT alternative_id values")
    if any(type(matched) is not bool for matched in option_matches.values()):
        raise ValueError("option_matches values must be boolean")
    windows = {frozenset(row["commit_time_sets"]) for row in expanded}
    if committed_after is None and len(windows) > 1:
        raise ValueError("committed_after is required when alternative Commit windows differ")
    return any(
        option_matches[row["alternative_id"]]
        and (committed_after is None or committed_after in row["commit_time_sets"])
        for row in expanded
    )


def expand_recommendation_need(
    ground_truth_path: Path,
    need_id: str | None = None,
) -> list[dict[str, Any]]:
    """Expand OR options for an evaluator without changing model output.

    Every returned option inherits the outer Need ID. An explicit option window
    overrides the shared default without enlarging the outer eligibility union.
    Matching any one returned option satisfies that single Need ID.
    """
    ground_truth = _object(
        json.loads(ground_truth_path.read_text(encoding="utf-8")),
        "ground truth",
    )
    needs = ground_truth.get("needs")
    if not isinstance(needs, list):
        raise ValueError("Ground truth needs must be an array")
    target = [
        need for need in needs
        if isinstance(need, dict)
        and "alternatives" in need
        and (need_id is None or need.get("need_id") == need_id)
    ]
    if len(target) != 1:
        raise ValueError("Expected exactly one recommendation Need")
    need = target[0]
    alternatives = load_recommendation_alternatives(
        ground_truth_path, need.get("need_id")
    )
    return [
        {
            "need_id": need["need_id"],
            "alternative_id": option["alternative_id"],
            "need": option["need"],
            "need_claims": option["need_claims"],
            "commit_time_sets": list(option.get("commit_time_sets", need["commit_time_sets"])),
            "match_policy": alternatives["match_policy"],
        }
        for option in alternatives["options"]
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--need-id")
    args = parser.parse_args()
    loaded = load_recommendation_alternatives(args.ground_truth, args.need_id)
    print(json.dumps({
        "status": "valid",
        "alternative_count": len(loaded["options"]),
        "match_policy": loaded["match_policy"],
    }, ensure_ascii=False))
