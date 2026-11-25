"""Isolated multimodal verifier for one disputed Passive GUI premise."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import time
import urllib.request

from judge_client import canonical, digest, read_json, write_json


SYSTEM_PROMPT = """You verify one factual premise against formal GUI screenshots.
Use only pixels in the supplied screenshots. Do not infer benchmark intent, user
history, hidden state, or a desired answer. A fact is visible when text, image
content, selection, icon state, slider position, layout or another unambiguous
visual cue directly supports it. Return not_visible when screenshots contradict
it or show the relevant state differently. Return unclear when the relevant
content is absent, cropped, unreadable or genuinely ambiguous. Identify only
supporting observation IDs. The absence of wording in screenshots does not prove
resource absence, incompatibility, completeness, safety, or that something is
unverified; return unclear for those epistemic/closed-world claims unless the UI
explicitly states the negative fact. Return JSON only:
{"status":"visible|not_visible|unclear","supporting_observation_ids":["o000"],"reason":"concise pixel-grounded reason"}
"""


def _image_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class VisualClaimsVerifier:
    def __init__(self, output: Path, *, api_key: str, model: str = "glm-5.3-flashx",
                 api_base: str = "https://open.bigmodel.cn/api/paas/v4",
                 timeout: float = 240, max_frames: int = 3):
        self.output = output
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_frames = max_frames
        (output / "cache").mkdir(parents=True, exist_ok=True)

    def verify(self, premise: str, frames: list[dict]) -> dict:
        selected = frames[-self.max_frames:]
        key = digest({
            "contract": "proseekbench.passive-visual-verifier.v1",
            "model": self.model, "premise": premise,
            "frames": [{"observation_id": row["observation_id"], "sha256": row["sha256"]}
                       for row in selected],
            "system_prompt": SYSTEM_PROMPT,
        })
        cache = self.output / "cache" / f"{key}.json"
        if cache.exists():
            return {**read_json(cache), "cache_hit": True}
        content = [{"type": "text", "text": f"Premise to verify: {premise}"}]
        allowed = set()
        for row in selected:
            allowed.add(row["observation_id"])
            content.append({"type": "text", "text": (
                f"Screenshot {row['observation_id']} at {row.get('logical_time')}:"
            )})
            content.append({"type": "image_url", "image_url": {
                "url": _image_url(Path(row["path"])), "detail": "high",
            }})
        payload = {
            "model": self.model, "temperature": 0.0, "max_tokens": 4096,
            "thinking": {"type": "enabled"},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
        }
        request = urllib.request.Request(
            self.api_base + "/chat/completions", data=canonical(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
        )
        started = time.perf_counter()
        last_error = None
        result = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.load(response)
                choice = raw["choices"][0]
                if choice.get("finish_reason") not in ("stop", "end_turn"):
                    raise ValueError("Visual verifier did not finish normally")
                result = json.loads(choice["message"]["content"])
                break
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if result is None:
            raise ValueError("Visual verifier returned no result") from last_error
        if (set(result) != {"status", "supporting_observation_ids", "reason"}
                or result["status"] not in {"visible", "not_visible", "unclear"}
                or not isinstance(result["supporting_observation_ids"], list)
                or not set(result["supporting_observation_ids"]) <= allowed
                or not isinstance(result["reason"], str) or not result["reason"].strip()):
            raise ValueError("Invalid visual verification output")
        record = {
            "schema_version": "proseekbench.passive-visual-verification.v1",
            "premise": premise, **result,
            "frame_bindings": [{"observation_id": row["observation_id"],
                                "sha256": row["sha256"]} for row in selected],
            "model": self.model, "cache_key": key,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(cache, record)
        return {**record, "cache_hit": False}
