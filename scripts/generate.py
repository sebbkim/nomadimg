#!/usr/bin/env python3
"""Process requests/<name>.json -> generated/<name>.png and generated/<name>.json (result meta).

A request that already has generated/<name>.json is skipped. To rerun, use a new name.
On failure the endpoint schema is recorded in the meta file to help debugging.
"""
import base64
import datetime
import io
import json
import os
import pathlib
import re

from gradio_client import Client, handle_file
from PIL import Image

REQ_DIR = pathlib.Path("requests")
OUT_DIR = pathlib.Path("generated")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SPACE_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
API_RE = re.compile(r"^/[A-Za-z0-9_]+$")
ALLOWED_PREFIXES = (
    "https://raw.githubusercontent.com/sebbkim/nomadimg/",
    "https://cdn.jsdelivr.net/gh/sebbkim/nomadimg@",
)
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
TOKEN = os.environ.get("HF_TOKEN", "").strip() or None


def check_url(u):
    if not isinstance(u, str) or not u.startswith(ALLOWED_PREFIXES):
        raise ValueError("file URL not allowed")
    return u


def convert(obj):
    """Replace {"$file": url} with gradio_client file handles."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {"$file"}:
            return handle_file(check_url(obj["$file"]))
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


def find_images(obj, out):
    if isinstance(obj, str):
        if obj.startswith("data:image"):
            out.append(("b64", obj))
        elif obj.lower().endswith(IMG_EXT) and os.path.isfile(obj):
            out.append(("file", obj))
    elif isinstance(obj, dict):
        first = ("image", "path", "url", "value")
        for key in first:
            if key in obj:
                find_images(obj[key], out)
        for key, val in obj.items():
            if key not in first:
                find_images(val, out)
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            find_images(val, out)


def load_image(kind, val):
    if kind == "b64":
        return Image.open(io.BytesIO(base64.b64decode(val.split(",", 1)[1])))
    return Image.open(val)


def summarize(result):
    def strip(o):
        if isinstance(o, str):
            return "<data-uri>" if o.startswith("data:") else o[:200]
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [strip(v) for v in o]
        return o

    return json.dumps(strip(result), ensure_ascii=False, default=str)[:800]


def make_client(space):
    if TOKEN:
        try:
            return Client(space, token=TOKEN)
        except TypeError:
            return Client(space, hf_token=TOKEN)
    return Client(space)


def mask(text):
    text = str(text)
    return text.replace(TOKEN, "***") if TOKEN else text


def write_meta(name, meta):
    meta["name"] = name
    meta["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_endpoint(client, api_name, meta):
    try:
        info = client.view_api(return_format="dict", print_info=False)
        endpoints = info.get("named_endpoints", {})
        meta["endpoints"] = list(endpoints.keys())
        meta["endpoint"] = json.dumps(endpoints.get(api_name), ensure_ascii=False, default=str)[:1500]
    except Exception as e2:
        meta["view_api_error"] = mask(repr(e2))[:200]


def process(path):
    name = path.stem
    if not NAME_RE.match(name):
        print(f"bad name: {name}")
        return
    if (OUT_DIR / f"{name}.json").exists():
        return
    meta = {"authenticated": bool(TOKEN)}
    try:
        if path.stat().st_size > 20000:
            raise ValueError("request too large")
        req = json.loads(path.read_text(encoding="utf-8"))
        space = req.get("space", "")
        api_name = req.get("api_name", "")
        if not SPACE_RE.match(space) or not API_RE.match(api_name):
            raise ValueError("invalid space or api_name")
        inputs = convert(req.get("inputs", {}))
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be an object")
        meta.update({"space": space, "api_name": api_name})
        print(f"[{name}] calling {space} {api_name}")
        client = make_client(space)
        try:
            result = client.predict(api_name=api_name, **inputs)
        except Exception:
            record_endpoint(client, api_name, meta)
            raise
        found = []
        find_images(result, found)
        meta["outputs"] = summarize(result)
        if not found:
            raise RuntimeError("no image found in outputs")
        img = load_image(*found[0])
        OUT_DIR.mkdir(exist_ok=True)
        img.save(OUT_DIR / f"{name}.png")
        meta.update({"status": "ok", "image": f"{name}.png", "size": list(img.size)})
    except Exception as e:
        detail = f"{type(e).__name__}: {e!r} args={getattr(e, 'args', ())!r} cause={e.__cause__!r}"
        meta.update({"status": "error", "message": mask(detail)[:700]})
    OUT_DIR.mkdir(exist_ok=True)
    write_meta(name, meta)
    print(f"[{name}] {meta['status']}")


if __name__ == "__main__":
    for p in sorted(REQ_DIR.glob("*.json")):
        process(p)
