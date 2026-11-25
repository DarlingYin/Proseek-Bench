#!/usr/bin/env python3
"""Run the online NeedThread pipeline with an OpenAI-compatible vision model.

Registered model families follow ``config/model_providers.json`` and read only
their provider-specific credential environment variable. Custom/local
endpoints use ``PROACTIVE_AGENT_API_KEY`` when authentication is required.
Credentials are never serialized.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import time
from typing import Any
from urllib.request import Request, urlopen

from online_runtime import OnlineEpisodeRuntime, write_json
from provider_policy import resolve_provider_route
from experiment_modes import add_mode_arguments, resolve_mode, check_no_search_result
from artifact_paths import require_external_output
from generic_tool_adapter import (
    DEFAULT_TOOL_INTERFACE, add_tool_interface_argument, search_protocol_for_interface,
)
from protocol_adapter import (
    ADAPTER_MODES, ADAPTER_OFF, ADAPTER_MIMO_RECOVERY, ADAPTER_Q38_RECOVERY,
    ADAPTER_INTERNVL_RECOVERY, ADAPTER_GLM46V_RECOVERY, ADAPTER_STEP3_RECOVERY,
    ADAPTER_VERSION, DecisionProtocolAdapter, resolve_adapter_mode,
)
from runtime_diagnostics import protocol_error_category
from compatibility_rules import registry, registry_hash, rule_ids
from runtime_snapshot import capture_snapshot, verify_snapshot
from codex_exec_backend import CodexExecDecisionError, CodexExecVisionAgent
from curl_transport import post_json_via_curl
from prompt_input import FULL_STATE, build_static_tool_context, render_prompt_state
from prediction_layout import validate_canonical_output
from request_rate_limiter import SharedRequestRateLimiter


class ModelDecisionProtocolError(ValueError):
    """A successful API response that cannot be parsed as a decision object."""

    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def extract_json_object(text: str, *, allow_trailing: bool = False,
                        normalization_events: list[str] | None = None) -> dict[str, Any]:
    events = normalization_events if normalization_events is not None else []
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'Duplicate JSON field: {key}')
            value[key] = item
        return value
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        events.append('strip_code_fence')
    if allow_trailing:
        start = min((i for i in (text.find('{'), text.find('[')) if i >= 0), default=-1)
        if start < 0:
            raise ValueError("Model response contains no JSON object")
        value, end = json.JSONDecoder(object_pairs_hook=unique_keys).raw_decode(text[start:])
        suffix = text[start + end:].strip()
        if '{' in suffix or '[' in suffix:
            raise ValueError('Multiple JSON decisions are ambiguous')
        if start or suffix:
            events.append('extract_json_envelope')
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
            events.append('unwrap_singleton_decision')
    else:
        try:
            value = json.loads(text, object_pairs_hook=unique_keys)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise ValueError("Model response contains no JSON object")
            value = json.loads(match.group(0), object_pairs_hook=unique_keys)
            events.append('extract_json_envelope')
    if not isinstance(value, dict):
        raise ValueError("Model response is not a JSON object")
    return value


def repair_mimo_json_text(text: str) -> tuple[str, list[str]]:
    """Repair only recurrent, token-local MiMo serialization slips.

    These rewrites do not add semantic fields or choose an action. Ambiguous
    duplicate fields, truncation, and responses with no JSON remain errors.
    """

    value = text
    events: list[str] = []
    repaired = re.sub(
        r'("[A-Za-z_][A-Za-z0-9_]*"\s*:\s*)"(?:eq|gte|lte|contains)"\s*,\s*\1',
        r'\1',
        value,
    )
    if repaired != value:
        value = repaired
        events.append("mimo_drop_duplicate_operator_pseudovalue")
    repaired = re.sub(r"}\s*,\s*\"\s*]", "}]", value)
    if repaired != value:
        value = repaired
        events.append("mimo_remove_spurious_array_quote")
    repaired = re.sub(r"}\s*,\s*\"\s*{", "},{", value)
    if repaired != value:
        value = repaired
        events.append("mimo_remove_spurious_object_quote")
    repaired = re.sub(
        r'(\"thread_ids\"\s*:\s*\[[^\]]*\])\s*]\s*,\s*(\"control\")',
        r'\1}],\2',
        value,
    )
    if repaired != value:
        value = repaired
        events.append("mimo_close_commit_operation")
    repaired = re.sub(r'}\s*,\s*"control"\s*:', '}],"control":', value, count=1)
    if repaired != value:
        value = re.sub(r']}(?=\s*(?:```)?\s*$)', '}', repaired)
        events.append("mimo_move_control_out_of_thread_ops")
    repaired = re.sub(r",\s*([}\]])", r"\1", value)
    if repaired != value:
        value = repaired
        events.append("mimo_remove_trailing_comma")
    return value, events


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64," + encoded


def protocol_repair_instruction(
    validation_feedback: str,
    rejected_output: str | None,
    *, direct_predictions: bool = False,
    context_summary: bool = False,
    internvl_recovery: bool = False,
) -> str:
    """Build a compact serialization-only repair request.

    The normal system message already contains the authoritative Decision Schema.  This
    supplement keeps repair turns from repeating the full semantic task and names the most common
    source of cross-model mistakes without changing the intended decision.
    """

    instruction = (
        "Protocol repair only. Preserve the previous semantic decision and evidence. "
        "Return one complete JSON object and no prose. "
        f"Runtime error: {validation_feedback}\n"
        "Thread operation fields are exact: create={op,thread_id,need,claims}; "
        "update={op,thread_id,need and/or claims}; delete={op,thread_id,reason}. "
        "Thread IDs are episode-unique: update/delete/commit may use candidate_thread_ids only; "
        "never reuse retired_thread_ids after delete or commit. A fresh ID is only for a genuinely new candidate. "
        "For Commit, Final Claims must be at most 900 characters, contain no oNNN/qNNN trace IDs, and retain only "
        "the decisive observation, decisive active evidence if any, remaining help gap and why now. "
        "Runtime-managed fields such as status, timestamps, and note fields must not appear "
        "inside thread_ops. Control fields must match the Decision Schema exactly. "
        "The observation and world have NOT advanced. Repair format using only the supplied schema, "
        "current thread state and issued handles. Do not switch to Wait simply to silence an error. "
        "If a semantic target is invalid or ambiguous, do not guess its replacement."
    )
    if context_summary:
        instruction = (
            "Protocol repair only. Preserve the previous semantic decision, current evidence and intended summary update. "
            "Return one complete JSON object and no prose. "
            f"Runtime error: {validation_feedback}\n"
            "Top-level fields are exactly context_summary (string, at most 2048 characters) and control (object). "
            "wait={type}; search={type,tool,arguments}; commit={type,predictions}. "
            "Each committed prediction has exactly need and claims. Do not emit thread operations, thread IDs, "
            "statuses or model-supplied commit timestamps. Final Claims must be at most 900 characters and contain "
            "no oNNN/qNNN trace IDs. Follow the supplied Decision Schema."
        )
        instruction += (" The observation and world have NOT advanced. Do not switch to Wait merely to silence "
                        "an error, and do not guess semantic targets or rewrite summary meaning.")
    elif direct_predictions:
        instruction = (
            "Protocol repair only. Preserve the previous semantic decision and current evidence. "
            "Return one complete JSON object and no prose. "
            f"Runtime error: {validation_feedback}\n"
            "The only top-level field is control (object). "
            "wait={type}; search={type,tool,arguments}; commit={type,predictions}. "
            "Each committed prediction has exactly need and claims. No thread operations, thread IDs, "
            "statuses, history summary or model-supplied commit timestamps. Final Claims must be at most "
            "900 characters and contain no oNNN/qNNN trace IDs. Follow the supplied Decision Schema."
        )
        instruction += (" The observation and world have NOT advanced. Preserve the supplied evidence; "
                        "do not switch to Wait merely to silence an error, and do not guess semantic targets.")
    if rejected_output:
        instruction += "\nPrevious rejected output:\n" + rejected_output[-6000:]
    if internvl_recovery:
        instruction += (
            "\nInternVL protocol reminder: use only fields declared by the exact selected "
            "resource's query schema. Never invent generic status/item_id filters, rename an "
            "unsupported field, or drop a constraint merely to pass validation. If thread_ops "
            "and control express conflicting actions, resolve the conflict explicitly instead "
            "of repeating both. Preserve the intended Need, Claims, target and query semantics."
        )
    return instruction


def compose_system_prompt(
    *,
    system_path: Path,
    policy_path: Path,
    search_protocol_path: Path,
    decision_schema_path: Path,
    legacy_prompt_path: Path | None = None,
    tool_interface: str = DEFAULT_TOOL_INTERFACE,
    static_tool_context: dict[str, Any] | None = None,
) -> str:
    """Build the immutable, cache-friendly system-message prefix."""

    if legacy_prompt_path is None:
        prompt_parts = [
            system_path.read_text(encoding="utf-8").strip(),
            policy_path.read_text(encoding="utf-8").strip(),
            search_protocol_for_interface(search_protocol_path, tool_interface).read_text(encoding="utf-8").strip(),
        ]
    else:
        prompt_parts = [legacy_prompt_path.read_text(encoding="utf-8").strip()]
    decision_schema = json.loads(decision_schema_path.read_text(encoding="utf-8"))
    if not isinstance(decision_schema, dict):
        raise ValueError("Decision Schema must be a JSON object")
    prompt_parts.append(
        "# Machine-readable Decision Schema\n\n"
        "The response must validate against this schema:\n\n"
        + json.dumps(decision_schema, ensure_ascii=False, separators=(",", ":"))
    )
    if static_tool_context is not None:
        prompt_parts.append(
            "# Frozen Runtime Tool Context\n\n"
            "This tool context is bound to the current user's immutable Catalog and "
            "applies to every turn in this Episode. The dynamic runtime state omits "
            "these repeated fields.\n\n"
            + json.dumps(
                static_tool_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n\n---\n\n".join(prompt_parts) + "\n"


class OpenAICompatibleVisionAgent:
    def __init__(
        self,
        *,
        api_base: str,
        provider: str,
        model: str,
        system_prompt: str,
        static_tool_context: dict[str, Any],
        static_tool_context_sha256: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
        response_format: str,
        thinking_mode: str,
        reasoning_effort: str | None,
        request_timeout: int,
        protocol_adapter_mode: str,
        request_rate_limiter: SharedRequestRateLimiter | None = None,
        prompt_input_contract: str = FULL_STATE,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.static_tool_context = static_tool_context
        self.static_tool_context_sha256 = static_tool_context_sha256
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.request_timeout = request_timeout
        self.protocol_adapter_mode = protocol_adapter_mode
        self.request_rate_limiter = request_rate_limiter
        self.prompt_input_contract = prompt_input_contract

    def decide(
        self,
        model_input: dict[str, Any],
        *,
        image_attachments: list[dict[str, Any]],
        validation_feedback: str | None = None,
        rejected_output: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attachment_by_id = {
            item["attachment_id"]: item for item in image_attachments
        }
        public_attachment_ids = [
            item["attachment_id"] for item in model_input["image_window"]
        ]
        if set(public_attachment_ids) != set(attachment_by_id):
            raise RuntimeError("Public screenshot references do not match private attachments")
        repair_only = bool(validation_feedback and rejected_output)
        if validation_feedback:
            instruction = protocol_repair_instruction(
                validation_feedback,
                rejected_output,
                direct_predictions=model_input.get("protocol") == "online-mobile-agent.stateless.v1",
                context_summary=model_input.get("protocol") == "online-mobile-agent.context-summary.v1",
                internvl_recovery=self.protocol_adapter_mode == ADAPTER_INTERNVL_RECOVERY,
            )
        else:
            instruction = (
                "Current online runtime state follows. Return only the required "
                "JSON decision. Keep Need and Claims concise enough to finish the JSON object."
            )
        state_for_prompt = render_prompt_state(
            model_input, static_tool_context=self.static_tool_context,
            static_tool_context_sha256=self.static_tool_context_sha256,
            input_contract=self.prompt_input_contract, repair_only=repair_only,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": instruction
                + "\n"
                + json.dumps(state_for_prompt, ensure_ascii=False),
            }
        ]
        # A repair with a recoverable prior output is a serialization task. Reattaching the same
        # screenshots encourages reasoning models to solve the semantic task again and can consume
        # their entire completion budget before they emit JSON.
        for public_item in ([] if repair_only else model_input["image_window"]):
            item = attachment_by_id[public_item["attachment_id"]]
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Screenshot {item['observation_id']} at "
                        f"{item['logical_time']}:"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(Path(item["image_path"])),
                        "detail": "high",
                    },
                }
            )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.provider == "bigmodel":
            # GLM-5.3-Flash/FlashX require thinking to remain enabled.  The
            # official OpenAI-compatible API uses a nested thinking object,
            # rather than Qianwen's enable_thinking boolean.
            payload["thinking"] = {"type": "enabled"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        elif self.thinking_mode != "default" and self.provider != "bigmodel":
            payload["enable_thinking"] = self.thinking_mode == "on"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(
            self.api_base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            rate_limit_reservation = (
                self.request_rate_limiter.acquire()
                if self.request_rate_limiter is not None
                else None
            )
            try:
                if self.provider == "volcengine_coding":
                    if not self.api_key:
                        raise RuntimeError("Volcengine curl transport requires an API key")
                    result = post_json_via_curl(
                        url=self.api_base + "/chat/completions",
                        payload=payload,
                        api_key=self.api_key,
                        timeout=self.request_timeout,
                    )
                else:
                    with urlopen(request, timeout=self.request_timeout) as response:
                        result = json.load(response)
                choice = result["choices"][0]
                message = choice["message"]
                content_text = message.get("content") or ""
                reasoning_text = (
                    message.get("reasoning_content") or message.get("reasoning") or ""
                )
                metadata = {
                    "attempt": attempt,
                    "model": result.get("model", self.model),
                    "usage": result.get("usage"),
                    "finish_reason": choice.get("finish_reason"),
                    "response_content_chars": len(content_text),
                    "reasoning_content_chars": len(reasoning_text),
                    "repair_only": repair_only,
                    "raw_content": content_text,
                    "latency_seconds": round(time.perf_counter() - attempt_started, 3),
                    "rate_limit_reservation": rate_limit_reservation,
                }
                try:
                    parse_events = []
                    try:
                        decision = extract_json_object(
                            content_text,
                            allow_trailing=self.protocol_adapter_mode != ADAPTER_OFF,
                            normalization_events=parse_events,
                        )
                    except (ValueError, json.JSONDecodeError):
                        if self.protocol_adapter_mode not in {
                            ADAPTER_MIMO_RECOVERY,
                            ADAPTER_GLM46V_RECOVERY,
                            ADAPTER_STEP3_RECOVERY,
                        }:
                            raise
                        repaired_text, repair_events = repair_mimo_json_text(content_text)
                        if not repair_events:
                            raise
                        decision = extract_json_object(
                            repaired_text,
                            allow_trailing=True,
                            normalization_events=parse_events,
                        )
                        parse_events.extend(repair_events)
                    metadata['parse_normalizations'] = parse_events
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ModelDecisionProtocolError(str(exc), metadata) from exc
                return decision, metadata
            except ModelDecisionProtocolError:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(attempt)
        raise RuntimeError(f"Agent API request failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--sampler-root", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        help="Episode to run; defaults to the first sampler episode.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "qianwen", "bigmodel", "volcengine_coding", "openrouter", "custom", "codex_exec"],
        default="auto",
        help="Provider route; auto enforces the repository family-routing policy.",
    )
    parser.add_argument(
        "--api-base",
        help="Optional endpoint override. It does not change the credential route.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex CLI executable used by --provider codex_exec.",
    )
    parser.add_argument(
        "--request-model",
        help=(
            "Model identifier sent to the serving endpoint when it differs from "
            "the stable artifact label supplied by --model."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--stop-after-observation',
                        help='Operator-only diagnostic prefix boundary, inclusive (e.g. o009); never sent to the Agent.')
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Deprecated combined-prompt override; the Decision Schema is still appended.",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--policy-prompt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--search-protocol",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--decision-schema",
        type=Path,
        default=None,
    )
    parser.add_argument("--action-window", type=int, default=8)
    parser.add_argument("--image-window", type=int, default=3)
    add_mode_arguments(parser)
    add_tool_interface_argument(parser)
    parser.add_argument("--max-model-calls", type=int, default=128)
    parser.add_argument(
        "--protocol-error-policy",
        choices=["fail", "repair_then_stop", "repair_then_wait"],
        default="repair_then_stop",
        help=(
            "Default: bounded same-frame repair then an explicit protocol-failure artifact; "
            "repair_then_wait is an opt-in degraded continuation."
        ),
    )
    parser.add_argument("--max-protocol-repairs", type=int, default=2)
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help=(
            "Combined reasoning and visible completion budget. Reasoning models may consume "
            "several thousand tokens before emitting the JSON decision."
        ),
    )
    parser.add_argument(
        "--response-format",
        choices=["json_object", "prompt_only"],
        default="json_object",
        help="Use provider JSON-object mode when available, or rely on the prompt only.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["default", "on", "off"],
        default="default",
        help="Use the provider default, explicitly enable thinking, or disable it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        help=(
            "Set the provider's validated reasoning level. This is mutually "
            "exclusive with an explicit --thinking-mode on/off."
        ),
    )
    parser.add_argument(
        "--protocol-adapter",
        choices=ADAPTER_MODES,
        default="safe",
        help="Safe serialization normalization by default; off preserves raw protocol errors.",
    )
    args = parser.parse_args()
    args.protocol_adapter = resolve_adapter_mode(args.protocol_adapter, args.model)
    try:
        condition = resolve_mode(args, repo_root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.decision_schema is None:
        args.decision_schema = repo_root / condition["decision_schema"]["path"]
    condition["effective_decision_schema"] = {
        "path": str(args.decision_schema.resolve()),
        "sha256": hashlib.sha256(args.decision_schema.read_bytes()).hexdigest(),
    }
    for key, attr in (("system", "system_prompt"), ("policy", "policy_prompt"), ("search_protocol", "search_protocol")):
        if getattr(args, attr) is None:
            setattr(args, attr, repo_root / condition["prompts"][key]["path"])
    effective_paths = (
        {"combined": args.prompt} if args.prompt else
        {"system": args.system_prompt, "policy": args.policy_prompt,
         "search_protocol": search_protocol_for_interface(args.search_protocol, args.tool_interface)}
    )
    condition["effective_prompts"] = {
        key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in effective_paths.items()
    }
    args.output = require_external_output(repo_root, args.output)
    if args.stop_after_observation and args.output.is_relative_to((repo_root / 'evaluate/runs').resolve()):
        parser.error('Diagnostic prefixes must use a separate output root, not canonical evaluate/runs')
    try:
        validate_canonical_output(args.output, repo_root, inference_mode=args.inference_mode,
                                  image_window=args.image_window, prompt_version=args.prompt_version)
    except ValueError as exc:
        parser.error(str(exc))
    if args.reasoning_effort is not None and args.thinking_mode != "default":
        parser.error(
            "--reasoning-effort cannot be combined with --thinking-mode on/off"
        )
    if args.max_protocol_repairs < 0:
        parser.error("--max-protocol-repairs must be non-negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    try:
        request_rate_limiter = SharedRequestRateLimiter.from_environment()
    except ValueError as exc:
        parser.error(str(exc))
    if args.provider == "codex_exec":
        if args.api_base:
            parser.error("--api-base does not apply to --provider codex_exec")
        if args.reasoning_effort is None:
            parser.error("--provider codex_exec requires --reasoning-effort")
        route = SimpleNamespace(provider="codex_exec", api_base=None, api_key_env=None)
        api_key = None
    else:
        try:
            route = resolve_provider_route(
                model=args.model,
                provider=args.provider,
                api_base_override=args.api_base,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if route.api_key_env is None:
            api_key = os.environ.get("PROACTIVE_AGENT_API_KEY")
        else:
            api_key = os.environ.get(route.api_key_env)
            if not api_key:
                raise SystemExit(f"{route.api_key_env} is required for {route.provider}")
        if (
            route.provider == "bigmodel"
            and (args.request_model or args.model).casefold()
            in {"glm-5.3-flash", "glm-5.3-flashx"}
            and args.thinking_mode == "off"
        ):
            parser.error("GLM-5.3-Flash/FlashX do not support disabling thinking")

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
    stop_index = None
    if args.stop_after_observation is not None:
        matches = [i for i, o in enumerate(runtime.observations)
                   if o['observation_id'] == args.stop_after_observation]
        if len(matches) != 1:
            parser.error('Unknown diagnostic stop observation')
        stop_index = matches[0]
    initial_model_input = runtime.model_input()
    catalog_sha256 = (
        runtime.tool_adapter.catalog_sha256
        if runtime.tool_adapter is not None
        else hashlib.sha256(
            json.dumps(
                initial_model_input["available_tools"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    prompt_input_contract = condition.get("prompt_input_contract", FULL_STATE)
    static_tool_context = build_static_tool_context(
        initial_model_input, catalog_sha256=catalog_sha256,
        input_contract=prompt_input_contract,
    )
    static_tool_context_serialized = json.dumps(
        static_tool_context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    static_tool_context_sha256 = hashlib.sha256(
        static_tool_context_serialized.encode("utf-8")
    ).hexdigest()
    system_prompt = compose_system_prompt(
        system_path=args.system_prompt,
        policy_path=args.policy_prompt,
        search_protocol_path=args.search_protocol,
        decision_schema_path=args.decision_schema,
        legacy_prompt_path=args.prompt,
        tool_interface=args.tool_interface,
        static_tool_context=static_tool_context,
    )
    if args.provider == "codex_exec":
        agent = CodexExecVisionAgent(
            codex_bin=args.codex_bin,
            model=args.request_model or args.model,
            reasoning_effort=args.reasoning_effort,
            system_prompt=system_prompt,
            decision_schema_path=args.decision_schema,
            static_tool_context=static_tool_context,
            static_tool_context_sha256=static_tool_context_sha256,
            request_timeout=args.request_timeout,
            protocol_adapter_mode=args.protocol_adapter,
            extract_json_object=extract_json_object,
            repair_instruction=protocol_repair_instruction,
            prompt_input_contract=prompt_input_contract,
        )
    else:
        agent = OpenAICompatibleVisionAgent(
            api_base=route.api_base,
            provider=route.provider,
            model=args.request_model or args.model,
            system_prompt=system_prompt,
            static_tool_context=static_tool_context,
            static_tool_context_sha256=static_tool_context_sha256,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format=args.response_format,
            thinking_mode=args.thinking_mode,
            reasoning_effort=args.reasoning_effort,
            request_timeout=args.request_timeout,
            protocol_adapter_mode=args.protocol_adapter,
            request_rate_limiter=request_rate_limiter,
            prompt_input_contract=prompt_input_contract,
        )
    protocol_adapter = DecisionProtocolAdapter(args.protocol_adapter)
    adapter_version = ADAPTER_VERSION
    adapter_implementation = "need_thread"
    extras = {'inputs/composed-system-prompt.md': system_prompt.encode(),
              'inputs/decision-schema.json': args.decision_schema.read_bytes(),
              'inputs/compatibility-rules.json': json.dumps(registry(), sort_keys=True).encode(),
              'inputs/tool-catalog.json': json.dumps(static_tool_context, sort_keys=True).encode()}
    for name, path in effective_paths.items():
        extras[f'inputs/prompt-{name}.md'] = path.read_bytes()
    snapshot = capture_snapshot(repo_root, repo_root / 'evaluate/runtime-snapshots', extra_files=extras)
    call_log = args.output.with_suffix(args.output.suffix + ".model-calls.jsonl")
    failure_path = args.output.with_name(args.output.stem + '.protocol-failure.json')
    if failure_path.exists():
        # Keep failed attempts available when an operator reruns the Episode.
        digest = hashlib.sha256(failure_path.read_bytes() + b'\0' +
                                (call_log.read_bytes() if call_log.exists() else b'')).hexdigest()
        history = args.output.parent / 'failed-attempts' / digest
        history.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(failure_path, history / failure_path.name)
        if call_log.exists():
            shutil.copy2(call_log, history / call_log.name)
    if call_log.exists():
        call_log.unlink()
    calls = 0
    run_started = time.perf_counter()
    successful_decisions = 0
    protocol_errors = 0
    fallback_waits = 0
    adapter_applied_turns = 0
    adapter_event_counts: dict[str, int] = {}
    from jsonschema import Draft202012Validator
    decision_validator = Draft202012Validator(json.loads(args.decision_schema.read_text()))
    error_categories: dict[str, int] = {}
    repaired_decisions = 0
    repair_requests = 0
    raw_error_turns = set()
    corrected_turns = set()
    rule_event_counts: dict[str, int] = {}
    while not runtime.complete:
        if stop_index is not None and runtime.cursor > stop_index:
            break
        turn = runtime.prepare_turn()
        model_input = turn.public_input
        validation_feedback: str | None = None
        rejected_output: str | None = None
        decision_applied = False
        for repair_index in range(args.max_protocol_repairs + 1):
            if calls >= args.max_model_calls:
                validation_feedback = (
                    f"Maximum model-call limit exceeded: {args.max_model_calls}"
                )
                protocol_errors += 1
                category = protocol_error_category(validation_feedback)
                error_categories[category] = error_categories.get(category, 0) + 1
                break
            repair_requests += int(repair_index > 0)
            decision: dict[str, Any] | None = None
            metadata: dict[str, Any] = {}
            try:
                raw_decision, metadata = agent.decide(
                    model_input,
                    image_attachments=turn.image_attachments,
                    validation_feedback=validation_feedback,
                    rejected_output=rejected_output,
                )
                adaptation = protocol_adapter.adapt(
                    raw_decision,
                    threads=runtime.threads,
                    tool_catalog=static_tool_context["available_tools"],
                )
                decision = adaptation.decision
                events = list(metadata.get('parse_normalizations', [])) + list(adaptation.events)
                if events:
                    corrected_turns.add((model_input['current_observation_id'], model_input['micro_turn']))
                    if repair_index == 0:
                        raw_error_turns.add((model_input['current_observation_id'], model_input['micro_turn']))
                    if args.protocol_adapter in {'safe'}:
                        for rule_id in rule_ids(events):
                            rule_event_counts[rule_id] = rule_event_counts.get(rule_id, 0) + 1
                if adaptation.changed:
                    adapter_applied_turns += 1
                    for event in adaptation.events:
                        adapter_event_counts[event] = adapter_event_counts.get(event, 0) + 1
                    metadata["protocol_adapter"] = {
                        "version": adapter_version,
                        "mode": args.protocol_adapter,
                        "events": list(adaptation.events),
                        "raw_decision": raw_decision,
                    }
                calls += 1
                validation_error = next(decision_validator.iter_errors(decision), None)
                if validation_error is not None:
                    field_path = '.'.join(str(p) for p in validation_error.absolute_path) or '$'
                    raise ValueError(f"Decision Schema at {field_path}: {validation_error.message}")
                runtime_response = runtime.apply_agent_output(decision)
            except (ModelDecisionProtocolError, CodexExecDecisionError) as exc:
                calls += 1
                metadata = exc.metadata
                validation_feedback = str(exc)
                rejected_output = metadata.get("raw_content") or None
                protocol_errors += 1
                if repair_index == 0:
                    raw_error_turns.add((model_input['current_observation_id'], model_input['micro_turn']))
                category = protocol_error_category(validation_feedback)
                error_categories[category] = error_categories.get(category, 0) + 1
                append_jsonl(
                    call_log,
                    {
                        "call": calls - 1,
                        "observation_id": model_input["current_observation_id"],
                        "micro_turn": model_input["micro_turn"],
                        "repair_index": repair_index,
                        "decision": None,
                        "protocol_error": validation_feedback,
                        "protocol_error_category": category,
                        "model": metadata,
                    },
                )
                if args.protocol_error_policy == "fail":
                    break
                continue
            except ValueError as exc:
                validation_feedback = str(exc)
                rejected_output = (
                    json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
                    if decision is not None
                    else None
                )
                protocol_errors += 1
                if repair_index == 0:
                    raw_error_turns.add((model_input['current_observation_id'], model_input['micro_turn']))
                category = protocol_error_category(validation_feedback)
                error_categories[category] = error_categories.get(category, 0) + 1
                append_jsonl(
                    call_log,
                    {
                        "call": calls - 1,
                        "observation_id": model_input["current_observation_id"],
                        "micro_turn": model_input["micro_turn"],
                        "repair_index": repair_index,
                        "decision": decision,
                        "protocol_error": validation_feedback,
                        "protocol_error_category": category,
                        "model": metadata,
                    },
                )
                if args.protocol_error_policy == "fail":
                    break
                continue
            append_jsonl(
                call_log,
                {
                    "call": calls - 1,
                    "observation_id": model_input["current_observation_id"],
                    "micro_turn": model_input["micro_turn"],
                    "repair_index": repair_index,
                    "decision": decision,
                    "runtime_response": runtime_response,
                    "model": metadata,
                },
            )
            successful_decisions += 1
            repaired_decisions += int(repair_index > 0)
            decision_applied = True
            break
        if decision_applied:
            continue
        if args.protocol_error_policy != "repair_then_wait":
            failure_path = args.output.with_name(args.output.stem + '.protocol-failure.json')
            failed = runtime.result()
            failed['status'] = 'protocol_failed'
            failed['configuration']['experiment_condition'] = condition
            failed['failure'] = {
                'observation_id': model_input['current_observation_id'],
                'micro_turn': model_input['micro_turn'], 'world_advanced': False,
                'category': protocol_error_category(validation_feedback or ''),
                'message': validation_feedback, 'model_calls': calls,
                'protocol_errors': protocol_errors, 'repair_requests': repair_requests,
                'repair_policy': args.protocol_error_policy, 'rejected_output': rejected_output,
                'model_call_log': str(call_log), 'runtime_snapshot': snapshot,
                'compatibility_registry_sha256': registry_hash(), 'rule_event_counts': rule_event_counts,
                'raw_protocol_error_turns': len(raw_error_turns),
                'successful_decisions': successful_decisions, 'repaired_decisions': repaired_decisions,
            }
            write_json(failure_path, failed)
            print(json.dumps({'status': 'protocol_failed', 'failure_artifact': str(failure_path),
                              'observation_id': model_input['current_observation_id']}))
            raise SystemExit(2)
        fallback = {"thread_ops": [], "control": {"type": "wait"}}
        runtime_response = runtime.apply_agent_output(fallback)
        runtime.action_history[-1]['decision_origin'] = 'protocol_fallback'
        runtime.action_history[-1]['protocol_error_category'] = protocol_error_category(validation_feedback or '')
        runtime.audit_trace[-1]['runtime_record'] = dict(runtime.action_history[-1])
        append_jsonl(
            call_log,
            {
                "call": None,
                "observation_id": model_input["current_observation_id"],
                "micro_turn": model_input["micro_turn"],
                "decision": fallback,
                "runtime_response": runtime_response,
                "fallback_reason": validation_feedback,
            },
        )
        fallback_waits += 1
    runtime_result = runtime.result()
    if stop_index is not None:
        runtime_result['diagnostic_scope'] = {
            'start_observation': runtime.observations[0]['observation_id'],
            'stop_after_observation': args.stop_after_observation,
            'full_episode_completed': runtime.complete,
            'boundary_hidden_from_agent': True,
        }
        if not runtime.complete:
            runtime_result['status'] = 'prefix_complete'
    check_no_search_result(args.inference_mode, runtime_result)
    runtime_result["configuration"]["inference_mode"] = args.inference_mode
    runtime_result["configuration"]["memory_mode"] = condition["memory_mode"]
    runtime_result["configuration"]["experiment_condition"] = condition
    runtime_result["model_run"] = {
        "provider": route.provider,
        "api_base": route.api_base,
        "transport": (
            "fresh_codex_exec_per_decision"
            if args.provider == "codex_exec"
            else "openai_compatible_curl_securetransport"
            if route.provider == "volcengine_coding"
            else "openai_compatible_http"
        ),
        "codex_bin": str(Path(agent.codex_bin).resolve()) if args.provider == "codex_exec" else None,
        "model": args.model,
        "request_model": args.request_model or args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": args.response_format,
        "thinking_mode": args.thinking_mode,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout": args.request_timeout,
        "request_rate_limit": (
            request_rate_limiter.public_config
            if request_rate_limiter is not None
            else None
        ),
        "protocol_adapter": {
            "version": adapter_version,
            "implementation": adapter_implementation,
            "mode": args.protocol_adapter,
            "applied_turns": adapter_applied_turns,
            "event_counts": dict(sorted(adapter_event_counts.items())),
        },
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "model_calls": calls,
        "successful_decisions": successful_decisions,
        "protocol_errors": protocol_errors,
        "fallback_waits": fallback_waits,
        "protocol_error_categories": dict(sorted(error_categories.items())),
        "repair_requests": repair_requests,
        "repaired_decisions": repaired_decisions,
        "raw_protocol_error_turns": len(raw_error_turns),
        "raw_protocol_error_rate": len(raw_error_turns) / (successful_decisions + fallback_waits) if successful_decisions + fallback_waits else None,
        "deterministically_corrected_turns": len(corrected_turns),
        "protocol_quality": "degraded_with_forced_waits" if fallback_waits else "complete_without_forced_waits",
        "compatibility_registry": {"version": registry()['version'], "sha256": registry_hash(),
                                   "rule_event_counts": rule_event_counts},
        "runtime_snapshot": snapshot,
        "protocol_error_policy": args.protocol_error_policy,
        "max_protocol_repairs": args.max_protocol_repairs,
        "model_call_log": str(call_log.resolve()),
        "prompt_bundle": {
            "sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "catalog_sha256": catalog_sha256,
            "static_tool_context_sha256": static_tool_context_sha256,
            "mode": (
                "legacy_combined"
                if args.prompt
                else "system_plus_policy_plus_search_protocol_plus_bound_tool_context"
            ),
            "system": str((args.prompt or args.system_prompt).resolve()),
            "policy": None if args.prompt else str(args.policy_prompt.resolve()),
            "search_protocol": (
                None if args.prompt else str(search_protocol_for_interface(args.search_protocol, args.tool_interface).resolve())
            ),
            "decision_schema": str(args.decision_schema.resolve()),
        },
    }
    runtime_output = args.output.with_name(args.output.stem + ".runtime.json")
    verify_snapshot(snapshot['path'], expected=snapshot['sha256'])
    runtime_result["model_run"]["runtime_result"] = str(runtime_output.resolve())
    write_json(runtime_output, runtime_result)

    sampler = json.loads((args.sampler_root / "sampler.json").read_text(encoding="utf-8"))
    prediction = {
        "schema_version": ('proseekbench.prefix-prediction.v1' if stop_index is not None
                           else 'proseekbench.prediction.v1'),
        "sampler_id": sampler["sampler_id"],
        "episode_id": runtime_result["episode_id"],
        "predictions": [
            {
                "need": item["need"],
                "claims": item["claims"],
                "committed_after": item["committed_after"],
            }
            for item in runtime_result["predictions"]
        ],
        "search_count": runtime_result["query_cost"]["query_count"],
    }
    if stop_index is not None:
        prediction['diagnostic_scope'] = runtime_result['diagnostic_scope']
    write_json(args.output, prediction)
    if failure_path.exists():
        failure_path.unlink()  # Its immutable failed-attempts copy was saved above.
    print(
        json.dumps(
            {
                "status": runtime_result["status"],
                "output": str(args.output.resolve()),
                "runtime_output": str(runtime_output.resolve()),
                "model_calls": calls,
                "protocol_errors": protocol_errors,
                "fallback_waits": fallback_waits,
                "predictions": len(prediction["predictions"]),
                "search_count": prediction["search_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
