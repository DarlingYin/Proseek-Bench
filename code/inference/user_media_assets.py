"""Freeze and stage private baseline images without publishing them to the Agent.

Optional authoring manifest: media_manifest_ref in app_environment/manifest.json.
Its paths are relative to the media manifest, with the existing ImageAssets checks.
Episode-local and future arrivals belong to separate operator packs, not this baseline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

from native_media_assets import ImageAssets

SCHEMA = "proactivebench.user-media-assets.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_authoring_media(app_root: Path, manifest: dict) -> dict | None:
    reference = manifest.get("media_manifest_ref")
    if reference is None:
        return None
    if not isinstance(reference, str):
        raise ValueError("media_manifest_ref must be a relative path")
    root = app_root.resolve()
    path = (root / reference).resolve()
    if Path(reference).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Media manifest escapes App Environment")
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA or value.get("user_id") != manifest["user_id"]:
        raise ValueError("Media manifest schema/user mismatch")
    if value.get("scope") != "baseline":
        raise ValueError("User revision media must have baseline scope")
    assets = ImageAssets(path.parent, value.get("assets"))
    if not assets.manifest:
        raise ValueError("A media manifest must contain assets")
    prepared = {}
    for asset_id in sorted(assets.manifest):
        metadata, data = assets.load(asset_id)
        source = assets.manifest[asset_id]
        provenance = source.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("kind"):
            raise ValueError(f"Media needs actual source provenance: {asset_id}")
        suffix = ".png" if metadata["mime_type"] == "image/png" else ".jpg"
        prepared[asset_id] = {
            **metadata, "path": f"media-assets/{asset_id}{suffix}",
            "provenance": provenance, "bytes": data,
        }
    fingerprint_material = {key: {k: v for k, v in row.items() if k != "bytes"}
                       for key, row in prepared.items()}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_material, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    return {"source_ref": reference, "source_sha256": sha256(path),
            "content_sha256": fingerprint, "assets": prepared}


def freeze_media(prepared: dict, revision_root: Path, user_id: str, revision: str) -> dict:
    rows = {}
    for asset_id, row in prepared["assets"].items():
        target = revision_root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(row["bytes"])
        rows[asset_id] = {k: v for k, v in row.items() if k != "bytes"}
    value = {"schema_version": SCHEMA, "visibility": "runner_only",
             "user_id": user_id, "revision_id": revision, "scope": "baseline",
             "assets": rows}
    target = revision_root / "user-media-assets.v1.json"
    with target.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"ref": target.name, "schema_version": SCHEMA, "sha256": sha256(target)}


def verify_frozen_media(revision_root: Path, declaration: dict, user_id: str, revision: str) -> tuple[Path, dict]:
    root = revision_root.resolve()
    path = (root / declaration["ref"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != declaration["sha256"]:
        raise ValueError("Frozen media manifest path/hash mismatch")
    value = json.loads(path.read_text())
    if (value.get("schema_version") != SCHEMA or value.get("user_id") != user_id
            or value.get("revision_id") != revision or value.get("scope") != "baseline"
            or value.get("visibility") != "runner_only"):
        raise ValueError("Frozen media identity/scope mismatch")
    assets = ImageAssets(root, value["assets"])
    for asset_id in assets.manifest:
        metadata, _ = assets.load(asset_id)
        row = assets.manifest[asset_id]
        for key in ("width", "height", "size_bytes"):
            if row.get(key) != metadata[key]:
                raise ValueError(f"Frozen media dimensions/size mismatch: {asset_id}")
    return path, value


def stage_user_media(environment: dict, transport, remote_root: str) -> dict | None:
    """Upload to a private run directory, verify remotely; never import into phone.

    The returned adapter manifest can be used with --assets-root and
    --media-manifest. Native attachment/MediaStore publication remains explicit.
    """
    declaration = environment["revision"]["artifacts"].get("user_media_assets")
    if declaration is None:
        return None
    root = environment["revision_path"].parent
    _, value = verify_frozen_media(root, declaration,
                                  environment["revision"]["user_id"],
                                  environment["revision"]["revision_id"])
    staged_root = remote_root.rstrip("/") + "/user-media"
    transport.ssh(shlex.join(["mkdir", "-p", staged_root + "/media-assets"]))
    for row in value["assets"].values():
        source = root / row["path"]
        destination = staged_root + "/" + row["path"]
        transport.upload(source, destination)
        # Keep the validation in Python so filenames never become shell programs.
        code = "import hashlib,pathlib; p=pathlib.Path(%r); assert hashlib.sha256(p.read_bytes()).hexdigest()==%r" % (destination, row["sha256"])
        transport.ssh(shlex.join(["python3", "-c", code]))
    # Existing native adapters accept this asset-ID mapping, not the revision wrapper.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        manifest_path = Path(temp) / "adapter-media.json"
        manifest_path.write_text(json.dumps(value["assets"]))
        adapter_sha256 = sha256(manifest_path)
        transport.upload(manifest_path, staged_root + "/adapter-media.json")
        code = "import hashlib,pathlib; p=pathlib.Path(%r); assert hashlib.sha256(p.read_bytes()).hexdigest()==%r" % (staged_root + "/adapter-media.json", adapter_sha256)
        transport.ssh(shlex.join(["python3", "-c", code]))
    return {"status": "staged_and_hash_verified", "asset_count": len(value["assets"]),
            "manifest_sha256": declaration["sha256"], "assets_root": staged_root,
            "adapter_manifest_sha256": adapter_sha256,
            "media_manifest": staged_root + "/adapter-media.json",
            "native_publication": "not_performed"}
