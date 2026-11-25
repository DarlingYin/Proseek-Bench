"""Shared, version-aware names for formal prediction directories.

The short Framework names are permanently reserved for v1. Diagnostic prefixes
remain outside evaluate/runs; naming a directory does not promote a pilot.
"""
from __future__ import annotations

from pathlib import Path


CANONICAL_CONDITIONS = {
    **{f"framework-w{window}": ("framework", window, "v1") for window in (1, 3, 5)},
    **{f"framework-v2-w{window}": ("framework", window, "v2") for window in (1, 3, 5)},
    "naive": ("naive", 3, None),
    "no-working-memory": ("no_working_memory", 3, None),
    "context-summary": ("context_summary", 3, None),
    "no-search": ("no_search", 3, None),
}
CANONICAL_RUN_IDS = frozenset(CANONICAL_CONDITIONS)


def canonical_setting(inference_mode: str, image_window: int,
                      prompt_version: str = "v1") -> str | None:
    for name, (mode, window, version) in CANONICAL_CONDITIONS.items():
        if (mode == inference_mode and window == image_window
                and (version is None or version == prompt_version)):
            return name
    return None


def validate_setting_metadata(setting: str, *, inference_mode=None,
                              image_window=None, prompt_version=None) -> None:
    """Reject renamed/mixed conditions while retaining unversioned v1 history."""
    if setting not in CANONICAL_CONDITIONS:
        raise ValueError(f"Non-canonical setting directory: {setting}")
    mode, window, version = CANONICAL_CONDITIONS[setting]
    for field, actual, expected in (
        ("inference_mode", inference_mode, mode),
        ("image_window", image_window, window),
        ("prompt_version", prompt_version, version),
    ):
        if expected is None:
            continue
        if actual is None:
            if version == "v2":
                raise ValueError(f"{setting} requires recorded {field}; do not relabel historical results")
        elif actual != expected:
            raise ValueError(f"{setting} requires {field}={expected!r}, got {actual!r}")


def validate_canonical_output(output: Path, repo_root: Path, *,
                              inference_mode: str, image_window: int,
                              prompt_version: str) -> None:
    """Enforce the same version boundary for the single-Episode CLI."""
    try:
        relative = output.resolve().relative_to((repo_root / "evaluate/runs").resolve())
    except ValueError:
        return
    expected = canonical_setting(inference_mode, image_window, prompt_version)
    if expected is None:
        raise ValueError("This mode/version/window has no canonical setting; use a diagnostic output root")
    if len(relative.parts) != 5 or relative.name != "prediction.json":
        raise ValueError("Canonical output must be evaluate/runs/<model>/<setting>/<user>/<sampler>/prediction.json")
    if relative.parts[1] != expected:
        raise ValueError(f"Canonical setting path for this configuration is {expected!r}, not {relative.parts[1]!r}")
