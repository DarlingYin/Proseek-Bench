"""Hash-bound image assets for trusted native App adapters.

The operator supplies a local root explicitly. No URL fetching, arbitrary path
access, image captions or inferred visual facts are added to Agent inputs.
"""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path


def image_size(data: bytes, mime: str) -> tuple[int, int]:
    if mime == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 33 or data[12:16] != b"IHDR":
            raise ValueError("Invalid PNG header")
        width, height = struct.unpack(">II", data[16:24])
    elif mime == "image/jpeg" and data.startswith(b"\xff\xd8"):
        position = 2
        while position + 4 <= len(data):
            if data[position] != 255:
                raise ValueError("Invalid JPEG marker")
            while position < len(data) and data[position] == 255:
                position += 1
            if position >= len(data):
                raise ValueError("Truncated JPEG marker")
            marker = data[position]; position += 1
            if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            if marker in {0xD9, 0xDA}:
                break
            length = int.from_bytes(data[position:position + 2], "big")
            if length < 2 or position + length > len(data):
                raise ValueError("Truncated JPEG segment")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if length < 8:
                    raise ValueError("Invalid JPEG frame")
                height, width = struct.unpack(">HH", data[position + 3:position + 7])
                break
            position += length
        else:
            raise ValueError("JPEG frame missing")
        if "width" not in locals():
            raise ValueError("JPEG frame missing")
    else:
        raise ValueError("Only matching PNG/JPEG images are supported")
    if not (0 < width <= 16384 and 0 < height <= 16384 and width * height <= 40_000_000):
        raise ValueError("Image dimensions outside native-media limits")
    return width, height


class ImageAssets:
    def __init__(self, root: Path | None, manifest: dict | None):
        self.root = Path(root).resolve() if root is not None else None
        self.manifest = manifest or {}
        if not isinstance(self.manifest, dict):
            raise ValueError("media_assets must be an asset-ID mapping")
        self.cache = {}

    def load(self, asset_id: str) -> tuple[dict, bytes]:
        if not isinstance(asset_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", asset_id):
            raise ValueError("Invalid asset ID")
        if self.root is None or asset_id not in self.manifest:
            raise ValueError(f"Missing explicit image asset/root: {asset_id}")
        if asset_id in self.cache:
            return self.cache[asset_id]
        entry = self.manifest[asset_id]
        relative = Path(entry["path"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root):
            raise ValueError("Image asset escapes authorized root")
        if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
            raise ValueError("Missing or oversized image asset")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"Image asset hash mismatch: {asset_id}")
        mime = entry["mime_type"]
        width, height = image_size(data, mime)
        name = entry.get("filename", relative.name)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", name):
            raise ValueError("Unsafe image filename")
        if Path(name).suffix.lower() not in ({".png"} if mime == "image/png" else {".jpg", ".jpeg"}):
            raise ValueError("Image extension/MIME mismatch")
        metadata = {"asset_id": asset_id, "filename": name, "sha256": digest,
                    "mime_type": mime, "size_bytes": len(data), "width": width, "height": height}
        self.cache[asset_id] = metadata, data
        return metadata, data

    def attachments(self, item: dict) -> list[dict]:
        result = []
        values = item.get("attachments", [])
        if not isinstance(values, list) or len(values) > 5:
            raise ValueError("Image attachments must be a list with at most five items")
        for value in values:
            if not isinstance(value, dict) or set(value) != {"asset_id"}:
                raise ValueError("Native image attachment requires only asset_id")
            metadata, _ = self.load(value["asset_id"])
            result.append(metadata)
        if len({v['asset_id'] for v in result}) != len(result):
            raise ValueError("Duplicate image attachment")
        return result
