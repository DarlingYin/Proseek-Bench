"""Shared cost-aware query settings for online evaluation and replay."""

from __future__ import annotations

import argparse


QUERY_POLICY_VERSION = "cost_aware_query_v3_attempt_accounting"
QUERY_ACCOUNTING = "Every valid accepted query, including a cache hit, consumes its declared cost and one query slot."
DEFAULT_SEARCH_BUDGET = 20
DEFAULT_MAX_SEARCHES_PER_OBSERVATION = 20


def parse_query_limit(value: str) -> int | None:
    if value.lower() == "unlimited":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use a non-negative integer or 'unlimited'."
        ) from exc
    if limit < 0:
        raise argparse.ArgumentTypeError(
            "Use a non-negative integer or 'unlimited'."
        )
    return limit


def add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--search-budget", type=parse_query_limit, default=DEFAULT_SEARCH_BUDGET,
        help="Episode query cost ceiling (default: 20); 'unlimited' removes it. Zero disables paid queries.",
    )
    parser.add_argument(
        "--max-searches-per-observation", type=parse_query_limit,
        default=DEFAULT_MAX_SEARCHES_PER_OBSERVATION,
        help="Per-observation loop guard (default: 20); 'unlimited' removes it.",
    )
