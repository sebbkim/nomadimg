"""Text styles that blend lettering into a painting (no GPU).

render(base, text, font, spec) -> RGB image.
spec keys: style ("impasto" | "ink" | "signature"), position ("bottom" | "top" | "center"),
margin (int), light / dark ([r, g, b] overrides for colors, otherwise sampled from the image).
"""
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


def _sample_colors(img):
    """Return (light, dark): median colors of the lightest and darkest ~12% of pixels."""
    small = img.convert("RGB").resize((64, 64))
    raw = small.tobytes()
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    px.sort(key=lambda c: 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
    n = max(1, len(px) // 8)

    def med(group):
        return tuple(sorted(c[i] for c in group)[len(group) // 2] for i in range(3))

    return med(px[-n:]), med(px[:n])


def _mask(size, text, font, margin, position):
    w, h = size
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (w - tw) // 2 - bb[0]
    if position == "top":
        y = margin - bb[1]
    elif position == "center":
        y = (h - th) // 2 - bb[1]
    else:
        y = h - margin - th - bb[1]
    d.text((x, y), text, font=font, fill=255)
    return m


def _highpass(img, radius=5, gain=3.0):
    g = img.convert("L")
    hp = ImageChops.subtract(g, g.filter(ImageFilter.GaussianBlur(radius)), scale=1, offset=128)
    return ImageEnhance.Contrast(hp).enhance(gain).convert("RGB")


def _shadow(base, mask, dx, dy, blur, strength, color=(10, 14, 30)):
    sh = ImageChops.offset(mask, dx, dy).filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * strength))
    return Image.composite(Image.new("RGB", base.size, color), base, sh)


def render(base, text, font, spec):
    style = spec.get("style", "impasto")
    position = spec.get("position", "bottom")
    margin = int(spec.get("margin", 70))
    light, dark = _sample_colors(base)
    light = tuple(spec.get("light", light))
    dark = tuple(spec.get("dark", dark))
    mask = _mask(base.size, text, font, margin, position)
    tex = _highpass(base, 5, 3.5)

    if style == "impasto":
        m = mask.filter(ImageFilter.GaussianBlur(0.9)).point(lambda v: 255 if v > 150 else int(v * 1.6))
        out = _shadow(base, m, 4, 6, 5, 0.6)
        body = ImageChops.soft_light(Image.new("RGB", base.size, light), tex)
        bump = m.filter(ImageFilter.GaussianBlur(2.2))
        hi = ImageChops.subtract(ImageChops.offset(bump, -2, -3), bump).point(lambda v: int(v * 0.9))
        lo = ImageChops.subtract(ImageChops.offset(bump, 2, 3), bump).point(lambda v: int(v * 0.9))
        body = ImageChops.add(body, Image.merge("RGB", [hi] * 3))
        body = ImageChops.subtract(body, Image.merge("RGB", [lo] * 3))
        return Image.composite(body, out, m)

    if style == "ink":
        wash = mask.filter(ImageFilter.GaussianBlur(16)).point(lambda v: min(255, int(v * 2.2))).point(lambda v: int(v * 0.62))
        out = Image.composite(Image.new("RGB", base.size, light), base, wash)
        ink = mask.filter(ImageFilter.GaussianBlur(1.1))
        layer = ImageChops.soft_light(Image.new("RGB", base.size, dark), _highpass(base, 5, 3.0))
        mult = ImageChops.multiply(out, layer.point(lambda v: min(255, int(v * 1.35) + 40)))
        return Image.composite(mult, out, ink.point(lambda v: int(v * 0.95)))

    if style == "signature":
        m = mask.filter(ImageFilter.GaussianBlur(1.0))
        out = _shadow(base, m, 2, 3, 3, 0.35)
        body = ImageChops.soft_light(Image.new("RGB", base.size, light), _highpass(base, 5, 3.0))
        return Image.composite(body, out, m.point(lambda v: int(v * 0.86)))

    raise ValueError("unknown style")
