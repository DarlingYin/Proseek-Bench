"""SecureTransport-backed JSON POST for endpoints incompatible with Python TLS.

The authorization header is supplied to curl through an inherited anonymous
pipe.  It is never placed in argv, a temporary file, or serialized artifacts.
The JSON request body is sent on stdin.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any


STATUS_MARKER = b"\n__PROSEEKBENCH_HTTP_STATUS__:"


class CurlHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        message = body
        try:
            value = json.loads(body)
            error = value.get("error", value)
            message = str(error.get("message") or error.get("code") or body)
        except (ValueError, TypeError, AttributeError):
            pass
        super().__init__(f"HTTP {status}: {message[:500]}")


def _config_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def post_json_via_curl(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    read_fd, write_fd = os.pipe()
    try:
        config = (
            f'header = "Authorization: Bearer {_config_quote(api_key)}"\n'
            'header = "Content-Type: application/json"\n'
        ).encode("utf-8")
        os.write(write_fd, config)
    finally:
        os.close(write_fd)
    command = [
        "/usr/bin/curl",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout),
        "--config",
        f"/dev/fd/{read_fd}",
        "--data-binary",
        "@-",
        "--write-out",
        STATUS_MARKER.decode("ascii") + "%{http_code}",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            check=False,
            pass_fds=(read_fd,),
            timeout=timeout + 15,
        )
    finally:
        os.close(read_fd)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"curl transport failed rc={completed.returncode}: {detail[:500]}")
    try:
        body, status_raw = completed.stdout.rsplit(STATUS_MARKER, 1)
        status = int(status_raw.decode("ascii"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("curl response is missing the HTTP status marker") from exc
    body_text = body.decode("utf-8", errors="replace")
    if status < 200 or status >= 300:
        raise CurlHTTPError(status, body_text)
    return json.loads(body_text)
