"""OpenRouter image mode: {"openrouter": {"model", "prompt", "input_references": [url, ...], "params": {...}}}.

Calls POST https://openrouter.ai/api/v1/images with the OPENROUTER_API_KEY environment variable.
Saves generated/<name>.png and .jpg and returns the cost reported by OpenRouter.
At most MAX_CALLS paid calls are made per workflow run.
"""
import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

from PIL import Image

URL = "https://openrouter.ai/api/v1/images"
MAX_CALLS = 4
ALLOWED_PREFIXES = (
    "https://raw.githubusercontent.com/sebbkim/nomadimg/",
    "https://cdn.jsdelivr.net/gh/sebbkim/nomadimg@",
)
ALLOWED_PARAMS = {"aspect_ratio", "size", "image_size", "resolution", "quality", "seed"}
_calls = 0


def _check_url(u):
    if not isinstance(u, str) or not u.startswith(ALLOWED_PREFIXES):
        raise ValueError("reference URL not allowed")
    return u


def run(spec, name, out_dir):
    global _calls
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    model = str(spec.get("model", ""))
    if not re.fullmatch(r"[A-Za-z0-9._/-]{3,80}", model):
        raise ValueError("invalid model id")
    prompt = str(spec.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")
    if _calls >= MAX_CALLS:
        raise RuntimeError("per-run call limit reached")
    body = {"model": model, "prompt": prompt[:4000]}
    refs = spec.get("input_references") or []
    if refs:
        body["input_references"] = [
            {"type": "image_url", "image_url": {"url": _check_url(u)}} for u in refs[:10]
        ]
    for k, v in (spec.get("params") or {}).items():
        if k in ALLOWED_PARAMS:
            body[k] = v
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sebbkim/nomadimg",
            "X-Title": "nomadimg",
        },
        method="POST",
    )
    _calls += 1
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace").replace(key, "***")[:500]
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {text}")
    items = data.get("data") or []
    if not items:
        raise RuntimeError("no image in response: " + json.dumps(data, ensure_ascii=False)[:300])
    item = items[0]
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        with urllib.request.urlopen(item["url"], timeout=120) as r:
            raw = r.read(30_000_000)
    else:
        raise RuntimeError("image item has neither b64_json nor url")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    out_dir.mkdir(exist_ok=True)
    img.save(out_dir / f"{name}.png")
    img.save(out_dir / f"{name}.jpg", quality=92)
    usage = data.get("usage") or {}
    return {
        "image": f"{name}.png",
        "jpg": f"{name}.jpg",
        "size": list(img.size),
        "model": model,
        "cost_usd": usage.get("cost"),
    }
