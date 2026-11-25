"""Freeze self-contained, operator-owned baseline web pages and dependencies.

This is a private deployment pack, not Chrome history or an Agent search tool.
No network retrieval is performed. Every local dependency must be declared.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
from urllib.parse import unquote, urlsplit

from native_media_assets import image_size

SCHEMA = "proactivebench.user-web-assets.v1"
MIMES = {".html": "text/html", ".css": "text/css", ".txt": "text/plain",
         ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,240}", value):
        raise ValueError("Unsafe web asset path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Web asset path escapes/aliases authorized root")
    return value


class Dependencies(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "meta" and dict(attrs).get("http-equiv", "").lower() == "refresh":
            raise ValueError("Automatic page refresh is unsupported in a static baseline")
        if tag in {"script", "iframe", "base", "object", "embed"}:
            raise ValueError("Baseline web pack must be static and self-contained")
        for name, value in attrs:
            if name.startswith("on") or name == "srcset":
                raise ValueError("Unsupported dynamic HTML dependency")
            if name in {"src", "href", "action", "poster"} and value:
                self.urls.append(value)
            if name == "style" and value:
                self.urls.extend(css_urls(value))


def css_urls(text: str) -> list[str]:
    if re.search(r"@import\b", text, re.I):
        raise ValueError("CSS imports are unsupported; declare one local stylesheet")
    return [m.group(2).strip() for m in re.finditer(r"url\(\s*(['\"]?)(.*?)\1\s*\)", text, re.I)]


def verify_dependencies(files: dict[str, dict]) -> None:
    by_path = {row["path"]: row for row in files.values()}
    for row in files.values():
        mime = row["mime_type"]
        if mime not in {"text/html", "text/css"}:
            continue
        text = row["bytes"].decode("utf-8")
        if mime == "text/html":
            parser = Dependencies()
            parser.feed(text)
            urls = parser.urls
        else:
            urls = css_urls(text)
        for url in urls:
            if url.startswith("#"):
                continue
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or parsed.path.startswith("/") or "\\" in url:
                raise ValueError(f"Nonlocal web dependency: {url}")
            path = unquote(parsed.path)
            if not path:
                continue
            resolved = posixpath.normpath(str(PurePosixPath(row["path"]).parent / path))
            if resolved not in by_path:
                raise ValueError(f"Missing/escaping web dependency: {row['path']} -> {url}")


def read_authoring_web(app_root: Path, manifest: dict) -> dict | None:
    reference = manifest.get("web_manifest_ref")
    if reference is None:
        return None
    safe_path(reference)
    root = app_root.resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Web manifest escapes App Environment")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema_version") != SCHEMA or value.get("user_id") != manifest["user_id"]
            or value.get("scope") != "baseline"):
        raise ValueError("Web manifest schema/user/scope mismatch")
    files, pages = value.get("files"), value.get("pages")
    if not isinstance(files, dict) or not files or not isinstance(pages, dict) or not pages:
        raise ValueError("Web pack requires files and page declarations")
    prepared = {}
    used_paths = set()
    for file_id, row in files.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", file_id):
            raise ValueError("Invalid web file ID")
        relative = safe_path(row["path"])
        source = (path.parent / relative).resolve()
        if not source.is_relative_to(path.parent.resolve()) or not source.is_file():
            raise ValueError("Web file escapes authorized root or is missing")
        if relative in used_paths:
            raise ValueError("Duplicate web file path")
        used_paths.add(relative)
        mime = row["mime_type"]
        if MIMES.get(source.suffix.lower()) != mime or not 0 < source.stat().st_size <= 20 * 1024 * 1024:
            raise ValueError("Web file MIME/size mismatch")
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError(f"Web asset hash mismatch: {file_id}")
        if mime.startswith("image/"):
            image_size(data, mime)
        else:
            data.decode("utf-8")
        if not isinstance(row.get("provenance"), dict) or not row["provenance"].get("kind"):
            raise ValueError("Web asset requires actual provenance")
        prepared[file_id] = {**row, "size_bytes": len(data), "bytes": data}
    for page_id, page in pages.items():
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", page_id)
                or page.get("file_id") not in prepared
                or prepared[page["file_id"]]["mime_type"] != "text/html"
                or not page.get("title") or not page.get("version") or not page.get("world_updated_at")):
            raise ValueError("Invalid web page declaration")
    verify_dependencies(prepared)
    fingerprint = {"files": {k: {a: b for a, b in v.items() if a != "bytes"}
                              for k, v in prepared.items()}, "pages": pages}
    content_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"source_ref": reference, "source_sha256": sha256(path),
            "content_sha256": content_hash, "files": prepared, "pages": pages}


def freeze_web(prepared: dict, revision_root: Path, user_id: str, revision: str) -> dict:
    rows = {}
    for file_id, row in prepared["files"].items():
        target = revision_root / "web-assets" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(row["bytes"])
        rows[file_id] = {k: v for k, v in row.items() if k != "bytes"}
    value = {"schema_version": SCHEMA, "visibility": "runner_only", "scope": "baseline",
             "user_id": user_id, "revision_id": revision, "files": rows, "pages": prepared["pages"]}
    target = revision_root / "user-web-assets.v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"ref": target.name, "schema_version": SCHEMA, "sha256": sha256(target)}


def verify_frozen_web(revision_root: Path, declaration: dict, user_id: str, revision: str) -> tuple[Path, dict]:
    root = revision_root.resolve()
    path = (root / declaration["ref"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != declaration["sha256"]:
        raise ValueError("Frozen web manifest path/hash mismatch")
    value = json.loads(path.read_text())
    if (value.get("schema_version") != SCHEMA or value.get("visibility") != "runner_only"
            or value.get("user_id") != user_id or value.get("revision_id") != revision
            or value.get("scope") != "baseline"):
        raise ValueError("Frozen web identity/scope mismatch")
    loaded = {}
    for file_id, row in value["files"].items():
        file = (root / "web-assets" / safe_path(row["path"])).resolve()
        if not file.is_relative_to(root / "web-assets") or sha256(file) != row["sha256"]:
            raise ValueError(f"Frozen web asset hash mismatch: {file_id}")
        data = file.read_bytes()
        if len(data) != row["size_bytes"]:
            raise ValueError("Frozen web asset size mismatch")
        loaded[file_id] = {**row, "bytes": data}
    verify_dependencies(loaded)
    return path, value


def stage_user_web(environment: dict, transport, remote_root: str) -> dict | None:
    declaration = environment["revision"]["artifacts"].get("user_web_assets")
    if declaration is None:
        return None
    root = environment["revision_path"].parent
    _, value = verify_frozen_web(root, declaration, environment["revision"]["user_id"],
                                 environment["revision"]["revision_id"])
    destination_root = remote_root.rstrip("/") + "/user-web"
    for row in value["files"].values():
        target = destination_root + "/" + row["path"]
        transport.ssh(shlex.join(["mkdir", "-p", str(PurePosixPath(target).parent)]))
        transport.upload(root / "web-assets" / row["path"], target)
        code = "import hashlib,pathlib; assert hashlib.sha256(pathlib.Path(%r).read_bytes()).hexdigest()==%r" % (target, row["sha256"])
        transport.ssh(shlex.join(["python3", "-c", code]))
    return {"status": "staged_and_hash_verified", "web_root": destination_root,
            "file_count": len(value["files"]), "page_count": len(value["pages"]),
            "manifest_sha256": declaration["sha256"], "native_publication": "not_performed",
            "pages": {k: {**v, "path": value["files"][v["file_id"]]["path"]} for k, v in value["pages"].items()}}
