"""One-shot Codex CLI backend for the online inference runtime.

Each ``decide`` call starts an ephemeral ``codex exec`` process.  The process
receives only the composed protocol prompt, the current public runtime state,
and copies of the screenshots in the current image window.  It never resumes a
previous Codex session, so model-side conversational memory cannot bypass the
runtime's bounded action/image windows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable
from prompt_input import FULL_STATE, render_prompt_state


class CodexExecDecisionError(ValueError):
    """A completed Codex invocation that did not produce a decision object."""

    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CodexExecVisionAgent:
    """Use one fresh, non-interactive Codex process for each runtime decision."""

    def __init__(
        self,
        *,
        codex_bin: str,
        model: str,
        reasoning_effort: str,
        system_prompt: str,
        decision_schema_path: Path,
        static_tool_context: dict[str, Any],
        static_tool_context_sha256: str,
        request_timeout: int,
        protocol_adapter_mode: str,
        extract_json_object: Callable[..., dict[str, Any]],
        repair_instruction: Callable[..., str],
        prompt_input_contract: str = FULL_STATE,
    ) -> None:
        resolved = shutil.which(codex_bin)
        if resolved is None:
            raise ValueError(f"Codex executable not found: {codex_bin}")
        if not reasoning_effort:
            raise ValueError("Codex exec requires an explicit reasoning effort")
        self.codex_bin = resolved
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt
        self.decision_schema_path = decision_schema_path.resolve()
        self.static_tool_context = static_tool_context
        self.static_tool_context_sha256 = static_tool_context_sha256
        self.request_timeout = request_timeout
        self.protocol_adapter_mode = protocol_adapter_mode
        self.extract_json_object = extract_json_object
        self.repair_instruction = repair_instruction
        self.prompt_input_contract = prompt_input_contract

    def decide(
        self,
        model_input: dict[str, Any],
        *,
        image_attachments: list[dict[str, Any]],
        validation_feedback: str | None = None,
        rejected_output: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attachment_by_id = {item["attachment_id"]: item for item in image_attachments}
        public_attachment_ids = [item["attachment_id"] for item in model_input["image_window"]]
        if set(public_attachment_ids) != set(attachment_by_id):
            raise RuntimeError("Public screenshot references do not match private attachments")

        repair_only = bool(validation_feedback and rejected_output)
        if validation_feedback:
            instruction = self.repair_instruction(
                validation_feedback,
                rejected_output,
                direct_predictions=model_input.get("protocol") == "online-mobile-agent.stateless.v1",
                context_summary=model_input.get("protocol") == "online-mobile-agent.context-summary.v1",
            )
        else:
            instruction = (
                "Current online runtime state follows. Return only the required JSON "
                "decision. Do not inspect files, run commands, browse, or call tools. "
                "Use only this prompt and the attached screenshots. Keep Need and Claims "
                "concise enough to finish the JSON object."
            )

        state_for_prompt = render_prompt_state(
            model_input, static_tool_context=self.static_tool_context,
            static_tool_context_sha256=self.static_tool_context_sha256,
            input_contract=self.prompt_input_contract, repair_only=repair_only,
        )

        with tempfile.TemporaryDirectory(prefix="proseekbench-codex-") as temporary:
            call_dir = Path(temporary)
            schema_copy = call_dir / "decision.schema.json"
            shutil.copyfile(self.decision_schema_path, schema_copy)
            output_path = call_dir / "decision.json"
            image_paths: list[Path] = []
            image_records: list[dict[str, Any]] = []
            if not repair_only:
                for index, public_item in enumerate(model_input["image_window"], start=1):
                    source = Path(attachment_by_id[public_item["attachment_id"]]["image_path"])
                    destination = call_dir / f"screenshot-{index:02d}.png"
                    shutil.copyfile(source, destination)
                    image_paths.append(destination)
                    image_records.append(
                        {
                            "file": destination.name,
                            "observation_id": attachment_by_id[public_item["attachment_id"]]["observation_id"],
                            "logical_time": attachment_by_id[public_item["attachment_id"]]["logical_time"],
                            "sha256": _sha256(destination),
                        }
                    )

            prompt = (
                self.system_prompt
                + "\n\n# One-shot runtime decision\n\n"
                + instruction
                + "\n\n# Current public runtime state\n\n"
                + json.dumps(state_for_prompt, ensure_ascii=False, separators=(",", ":"))
            )
            if image_records:
                prompt += "\n\n# Attached screenshots\n\n" + json.dumps(
                    [{key: row[key] for key in ("file", "observation_id", "logical_time")}
                     for row in image_records],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            command = [
                self.codex_bin,
                "--ask-for-approval", "never",
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--color", "never",
                "--sandbox", "read-only",
                "--cd", str(call_dir),
                "--model", self.model,
                "--config", f'model_reasoning_effort="{self.reasoning_effort}"',
                # The desktop-bundled Codex catalog exposes the requested model.  Bind a
                # non-WebSocket ChatGPT-plan provider so one-shot calls do not spend five
                # failed connection attempts before falling back to HTTPS.
                "--config", 'model_provider="proseekbench-chatgpt-http"',
                "--config", 'model_providers.proseekbench-chatgpt-http.name="ProSeekBench ChatGPT HTTP"',
                "--config", 'model_providers.proseekbench-chatgpt-http.base_url="https://chatgpt.com/backend-api/codex"',
                "--config", 'model_providers.proseekbench-chatgpt-http.wire_api="responses"',
                "--config", 'model_providers.proseekbench-chatgpt-http.requires_openai_auth=true',
                "--config", 'model_providers.proseekbench-chatgpt-http.supports_websockets=false',
                # The benchmark schemas intentionally use oneOf/anyOf.  Codex structured
                # outputs reject those keywords, while the historical HTTP runner only
                # requested a JSON object and delegated exact validation to Runtime.  Keep
                # the original schema embedded in the prompt and use the same Runtime gate.
                "--output-last-message", str(output_path),
            ]
            for image_path in image_paths:
                command.extend(["--image", str(image_path)])
            command.append("-")

            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.request_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Codex exec timed out after {self.request_timeout} seconds"
                ) from exc
            latency = round(time.perf_counter() - started, 3)
            raw_content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            metadata = {
                "transport": "codex_exec",
                "fresh_context": True,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "returncode": completed.returncode,
                "repair_only": repair_only,
                "raw_content": raw_content,
                "response_content_chars": len(raw_content),
                "latency_seconds": latency,
                "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "decision_schema_sha256": _sha256(schema_copy),
                "images": image_records,
                "stderr_tail": completed.stderr[-2000:],
            }
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Codex exec failed with exit code {completed.returncode}: "
                    f"{completed.stderr[-2000:]}"
                )
            try:
                parse_events = []
                decision = self.extract_json_object(
                    raw_content,
                    allow_trailing=self.protocol_adapter_mode != "off",
                    normalization_events=parse_events,
                )
                metadata['parse_normalizations'] = parse_events
            except (ValueError, json.JSONDecodeError) as exc:
                raise CodexExecDecisionError(str(exc), metadata) from exc
            return decision, metadata
