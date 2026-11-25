"""Bind native media placement and public attachment associations to a revision.

The frozen plan is runner-only. It supplies no Agent tool and makes no publication
claim. Deployment still needs the actual native importer/projection and readback.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex

SCHEMA = "proactivebench.user-native-content.v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(value, user_id, state_time, entities, assets, pages):
    if (value.get("schema_version") != SCHEMA or value.get("user_id") != user_id
            or value.get("scope") != "baseline" or value.get("visibility") != "runner_only"
            or value.get("state_time") != state_time):
        raise ValueError("Native content identity/scope/time mismatch")
    baseline_time = datetime.fromisoformat(state_time)
    availability = value.get("image_availability", {})
    if set(availability) != set(assets):
        raise ValueError("Native image availability must cover exactly the frozen assets")
    for row in availability.values():
        if (row["presence"] not in {"local_gallery", "remote_attachment"}
                or datetime.fromisoformat(row["world_available_at"]) > baseline_time):
            raise ValueError("Future/invalid image presence in native baseline")
    local = {key for key, row in availability.items() if row["presence"] == "local_gallery"}
    seen = set()
    for row in value.get("gallery", []):
        asset_id = row["asset_id"]
        if (asset_id not in local or asset_id in seen
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", row["folder"])
                or datetime.fromisoformat(row["captured_at"]) > baseline_time):
            raise ValueError("Invalid/duplicate/remote Gallery asset")
        seen.add(asset_id)
    if seen != local:
        raise ValueError("Gallery placement is incomplete")
    attached = set()
    for source, field in [("mattermost", "attachments"), ("mastodon", "media_descriptions")]:
        expected = {ref for ref, entity in entities.items() if entity["entity_type"] == source + ".post"
                    and entity["attributes"].get(field)}
        bindings = value.get(source, {})
        if set(bindings) != expected:
            raise ValueError("Native attachment bindings differ from public records")
        for ref, rows in bindings.items():
            public_rows = entities[ref]["attributes"][field]
            if len(rows) != len(public_rows) or len(rows) > 5:
                raise ValueError("Native/public attachment count mismatch")
            for index, row in enumerate(rows):
                asset_id = row if source == "mattermost" else row["asset_id"]
                if asset_id not in assets:
                    raise ValueError("Native attachment refers to an undeclared asset")
                if datetime.fromisoformat(availability[asset_id]["world_available_at"]) > datetime.fromisoformat(entities[ref]["attributes"]["created_at"]):
                    raise ValueError(f"Image appears in an attachment before its availability: {ref} / {asset_id}")
                attached.add(asset_id)
                if source == "mattermost":
                    if public_rows[index]["filename"] != assets[asset_id]["filename"]:
                        raise ValueError("Native/public attachment filename mismatch")
                elif row["description"] != public_rows[index]:
                    raise ValueError("Native/public image description mismatch")
    remote = set(availability) - local
    if not remote.issubset(attached):
        raise ValueError("A remote image has no actual baseline attachment")
    chrome = value.get("chrome", {})
    if chrome.get("origin") != "http://10.0.2.2:18083" or set(chrome.get("page_ids", [])) != set(pages):
        raise ValueError("Native Chrome deployment does not cover frozen pages")
    for page in pages.values():
        if datetime.fromisoformat(page["world_updated_at"]) > baseline_time:
            raise ValueError("Future web version in native baseline")
    for ref in value.get("docreader", {}).get("document_entities", []):
        entity = entities.get(ref, {})
        if entity.get("entity_type") != "files.document" or entity["attributes"]["mime_type"] != "application/vnd.oasis.opendocument.text":
            raise ValueError("DocReader needs a real Files ODT document")
    camera = value.get("camera")
    if camera is not None:
        if (
            not isinstance(camera, dict)
            or camera.get("package") != "com.android.camera2"
            or camera.get("capture_output_scope") != "MediaStore/DCIM/Camera"
            or not isinstance(camera.get("profiles"), list)
        ):
            raise ValueError("Invalid Camera native-content contract")
        permission = camera.get("baseline_permission", {})
        if permission != {"permission": "android.permission.CAMERA", "granted": True}:
            raise ValueError("Camera baseline permission must be explicit and granted")
        profile_ids = set()
        for profile in camera["profiles"]:
            if (
                not isinstance(profile, dict)
                or not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", profile.get("profile_id", ""))
                or profile.get("profile_id") in profile_ids
                or profile.get("mode") not in {"permission_gate", "native_shutter", "controlled_virtual_feed"}
                or type(profile.get("requires_controlled_feed")) is not bool
                or not isinstance(profile.get("status"), str)
            ):
                raise ValueError("Invalid Camera simulation profile")
            profile_ids.add(profile["profile_id"])
            asset_id = profile.get("asset_id")
            if asset_id is not None and asset_id not in assets:
                raise ValueError("Camera simulation profile uses an unknown asset")
    settings = value.get("settings")
    if settings is not None:
        allowed_fields = {
            "media_volume_index", "media_volume_max", "screen_brightness", "hour_format",
            "screen_timeout_ms", "auto_rotate_enabled", "app_permissions",
        }
        if (
            not isinstance(settings, dict)
            or settings.get("package") != "com.android.settings"
            or not isinstance(settings.get("device_context_fields"), list)
            or not set(settings["device_context_fields"]).issubset(allowed_fields)
            or not isinstance(settings.get("scenario_profiles"), list)
        ):
            raise ValueError("Invalid Settings native-content contract")
        profile_ids = set()
        for profile in settings["scenario_profiles"]:
            if (
                not isinstance(profile, dict)
                or not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", profile.get("profile_id", ""))
                or profile.get("profile_id") in profile_ids
                or not isinstance(profile.get("overlay"), dict)
                or not set(profile["overlay"]).issubset(allowed_fields)
                or not isinstance(profile.get("status"), str)
            ):
                raise ValueError("Invalid Settings scenario profile")
            profile_ids.add(profile["profile_id"])
    identities = value.get("account_identities", {})
    for source in ["mattermost", "mastodon"]:
        states = [e["attributes"] for e in entities.values() if e["entity_type"] == source + ".user_state"]
        if len(states) != 1 or identities.get(source, {}).get("username") != states[0]["username"]:
            raise ValueError("Native identity differs from baseline account")


def read_authoring_native_content(app_root, manifest, entities, media, web):
    reference = manifest.get("native_content_manifest_ref")
    if reference is None:
        return None
    root = app_root.resolve()
    path = (root / reference).resolve()
    if Path(reference).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Native content manifest escapes App Environment")
    value = json.loads(path.read_text())
    validate(value, manifest["user_id"], manifest["state_time"], entities,
             media["assets"] if media else {}, web["pages"] if web else {})
    material = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"source_ref": reference, "source_sha256": sha256(path),
            "content_sha256": hashlib.sha256(material).hexdigest(), "value": value}


def freeze_native_content(prepared, revision_root, user_id, revision):
    value = {**prepared["value"], "revision_id": revision}
    path = revision_root / "user-native-content.v1.json"
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"ref": path.name, "schema_version": SCHEMA, "sha256": sha256(path)}


def verify_frozen_native_content(root, declaration, revision, entities, media, web):
    path = (root / declaration["ref"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != declaration["sha256"]:
        raise ValueError("Frozen native content path/hash mismatch")
    value = json.loads(path.read_text())
    if value.get("revision_id") != revision["revision_id"]:
        raise ValueError("Frozen native content revision mismatch")
    validate(value, revision["user_id"], revision["state_time"], entities,
             media.get("assets", {}), web.get("pages", {}))
    return path, value


def stage_user_native_content(environment, transport, remote_root):
    declaration = environment["revision"]["artifacts"].get("user_native_content")
    if declaration is None:
        return None
    root = environment["revision_path"].parent
    path, value = verify_frozen_native_content(root, declaration, environment["revision"],
        environment["artifacts"]["user_phone_world_bundle"]["semantic_world"]["entities"],
        environment["artifacts"].get("user_media_assets", {}), environment["artifacts"].get("user_web_assets", {}))
    destination = remote_root.rstrip("/") + "/user-native-content.v1.json"
    transport.upload(path, destination)
    code = "import hashlib,pathlib; assert hashlib.sha256(pathlib.Path(%r).read_bytes()).hexdigest()==%r" % (destination, declaration["sha256"])
    transport.ssh(shlex.join(["python3", "-c", code]))
    return {"status": "staged_and_hash_verified", "manifest": destination,
        "gallery_asset_count": len(value["gallery"]),
        "camera_profile_count": len(value.get("camera", {}).get("profiles", [])),
        "settings_profile_count": len(value.get("settings", {}).get("scenario_profiles", [])),
        "native_publication": "not_performed"}
