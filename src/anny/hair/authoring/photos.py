# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The hairline of photographs and of anny's renders, measured the same way.

MediaPipe's face landmarker finds the face (:class:`anny.faces.authoring.photos.Detector`) and
MediaPipe's selfie multiclass segmenter (Apache 2.0) finds the hair. On near-frontal faces, the
benchmark follows lines parallel to the axis of the face (from the nasion, MediaPipe's point
168, to the chin, point 152) and measures where the hair begins, as a height above the nasion
in units of the nasion-chin distance:

- ``forehead``: the hair edge on the midline;
- ``pupil``: the hair edge above the pupils;
- ``temple``: the hair edge at 0.8 of the half width of the face;
- ``sideburn``: the lowest point of the hair that comes down the sides of the face, from 0.9 to
  1.1 of the half width (negative below the nasion).

The half width is the distance of the sides of the face (points 234 and 454) from its axis. Where
the hair already covers a line at the height of the nasion, the edge is the bottom of that hair.
A face has short hair when no hair beside it (0.9 to 1.3 of the half width) reaches the mouth.

The same detector and segmenter measure the FairFace validation photos (CC BY 4.0) and portraits
of anny from the viewer page with its hair, so that their biases cancel. Each style is drawn on a
man and a woman of 30 years, and each portrait is compared with the photos of the same gender and
the same hair length (short or long), aged 20 to 49.

Usage::

    python -m anny.hair.authoring.photos [--styles NAME ...] [--label TEXT] [--page PATH]

The report goes to ``ANNY_CACHE_DIR/hair/photos.html``, with its numbers in ``photos.json``.
MediaPipe and Playwright are needed (``uv sync --extra faces``; MediaPipe also needs the system
libraries ``libegl1`` and ``libgles2``).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import pathlib

import numpy as np

from anny.paths import get_anny_cache_path

MASK = 160  # the side of the stored hair masks (pixels)
N, GN, STO = 168, 152, 13  # nasion, chin and mouth (MediaPipe's face mesh)
PUPILS = (468, 473)
SIDES = (234, 454)
EDGE = 0.5  # the hair probability at the edge of the hair
STEP = 0.01  # the step along a line (units of the nasion-chin distance)
RUN = 3  # steps of hair (or of no hair) that make an edge
METRICS = ["forehead", "pupil", "temple", "sideburn"]
TEMPLE = 0.8
SIDEBURN = (0.9, 1.1)
SHORT = (0.9, 1.3)
MAX_ANGLE = 10.0  # degrees of yaw and pitch
PHOTO_AGES = [3, 4, 5]  # FairFace's age groups 20-29, 30-39 and 40-49
BODIES = [("man", 0.0, 30.0), ("woman", 1.0, 30.0)]
HAIR_COLOUR = "#3a2a1c"


def cache_dir() -> pathlib.Path:
    d = get_anny_cache_path() / "hair"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ the measurements
def _sample(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """bilinear samples of a mask (H, W) at points (..., 2) in pixels (x, y); 0 outside"""
    H, W = mask.shape
    x, y = pts[..., 0] - 0.5, pts[..., 1] - 0.5
    inside = (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
    x, y = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
    x0, y0 = (
        np.minimum(np.floor(x).astype(int), W - 2),
        np.minimum(np.floor(y).astype(int), H - 2),
    )
    fx, fy = x - x0, y - y0
    v = (
        mask[y0, x0] * (1 - fx) * (1 - fy)
        + mask[y0, x0 + 1] * fx * (1 - fy)
        + mask[y0 + 1, x0] * (1 - fx) * fy
        + mask[y0 + 1, x0 + 1] * fx * fy
    )
    return np.where(inside, v, 0.0)


def _first_run(on: np.ndarray, value: bool) -> int:
    """the index of the first run of RUN samples equal to ``value``, or -1"""
    hit = on == value
    for i in range(len(hit) - RUN + 1):
        if hit[i : i + RUN].all():
            return i
    return -1


def hair_edge(mask, n, down, across, D, offset, reach=(-1.2, 1.5)) -> float:
    """
    The hair edge on the line at ``offset`` (units of D, along ``across``) from the axis of the face:
    the height (units of D above the nasion ``n``) where the hair begins going up, or, when the hair
    covers the line at the nasion, where it ends going down. NaN when the line meets no hair.
    """
    up = np.arange(0.0, reach[1], STEP)
    pts = n[None] + offset * D * across[None] - up[:, None] * D * down[None]
    on = _sample(mask, pts) > EDGE
    if not on[:RUN].all():
        i = _first_run(on, True)
        return float(up[i]) if i >= 0 else float("nan")
    dn = np.arange(0.0, -reach[0], STEP)
    pts = n[None] + offset * D * across[None] + dn[:, None] * D * down[None]
    on = _sample(mask, pts) > EDGE
    i = _first_run(on, False)
    return -float(dn[i]) if i >= 0 else -float(dn[-1])


def hair_metrics(landmarks: np.ndarray, mask: np.ndarray, size: float) -> dict:
    """
    The metrics of one face: ``landmarks`` (478, >=2) in the pixels of an image of side ``size``
    and ``mask`` the hair probability (MASK, MASK) of that image. Returns the metrics (units of the
    nasion-chin distance), ``mouth`` (the height of the mouth) and ``short``.
    """
    P = np.asarray(landmarks, dtype=np.float64)[:, :2] * (mask.shape[0] / size)
    n = P[N]
    down = P[GN] - n
    D = float(np.linalg.norm(down))
    down /= D
    across = np.array([down[1], -down[0]])  # toward the image's right
    half = 0.5 * abs(float((P[SIDES[1]] - P[SIDES[0]]) @ across)) / D
    pupil = [abs(float((P[k] - n) @ across)) / D for k in PUPILS]

    def edge(offset):
        return hair_edge(mask, n, down, across, D, offset)

    def lowest(lo, hi):
        """the lowest hair edge of the lines from lo to hi of the half width, on each side"""
        out = []
        for side in (-1, 1):
            e = [edge(side * u * half) for u in np.linspace(lo, hi, 5)]
            e = [v for v in e if np.isfinite(v)]
            out.append(min(e) if e else np.nan)
        return out

    mouth = -float((P[STO] - n) @ down) / D
    side = lowest(*SIDEBURN)
    short = lowest(*SHORT)
    return dict(
        forehead=edge(0.0),
        pupil=float(np.nanmean([edge(-pupil[0]), edge(pupil[1])])),
        temple=float(np.nanmean([edge(-TEMPLE * half), edge(TEMPLE * half)])),
        sideburn=float(np.nanmean(side)),
        mouth=mouth,
        short=bool(np.all(np.nan_to_num(short, nan=1.0) > mouth)),
    )


class Segmenter:
    """MediaPipe's selfie multiclass segmenter: the hair probability of an image"""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        from anny.faces.authoring.sources import fetch

        self.mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(fetch("mediapipe_hair_segmenter"))
            ),
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """the hair probability (MASK, MASK) as uint8"""
        from PIL import Image

        result = self.segmenter.segment(
            self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
            )
        )
        hair = np.asarray(result.confidence_masks[1].numpy_view())[..., 0]
        small = Image.fromarray(np.clip(hair * 255 + 0.5, 0, 255).astype(np.uint8))
        return np.asarray(small.resize((MASK, MASK), Image.BILINEAR))


class Measure:
    """landmarks, head pose and hair of images"""

    def __init__(self):
        from anny.faces.authoring.photos import Detector

        self.detector = Detector()
        self.segmenter = Segmenter()

    def __call__(self, image: np.ndarray) -> dict:
        r = self.detector(image)
        mask = self.segmenter(image)
        if r is None:
            return dict(
                found=False,
                landmarks=np.full((478, 2), np.nan),
                pose=np.full((4, 4), np.nan),
                mask=mask,
            )
        return dict(found=True, landmarks=r[0][:, :2], pose=r[2], mask=mask)


def frontal(pose: np.ndarray) -> np.ndarray:
    """near-frontal faces: yaw and pitch within MAX_ANGLE degrees"""
    from anny.faces.authoring.photos import head_angles

    a = head_angles(np.nan_to_num(pose))
    return (np.abs(a[..., 0]) < MAX_ANGLE) & (np.abs(a[..., 1]) < MAX_ANGLE)


def table(landmarks, masks, sizes, ok) -> dict:
    """the metrics of the faces that ``ok`` keeps (NaN elsewhere); ``sizes`` holds the image sides"""
    n = len(masks)
    out = {k: np.full(n, np.nan) for k in METRICS + ["mouth"]}
    out["short"] = np.zeros(n, bool)
    for i in np.flatnonzero(ok):
        m = hair_metrics(landmarks[i], masks[i].astype(np.float64) / 255.0, sizes[i])
        for k, v in m.items():
            out[k][i] = v
    return out


# ------------------------------------------------------------------ the photographs
def fairface_hair(split: str = "validation") -> dict:
    """landmarks, poses and hair masks of the FairFace photos of a split, with their labels (cached)"""
    path = cache_dir() / f"fairface_{split}_hair.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    import pyarrow.parquet as pq
    from PIL import Image

    from anny.faces.authoring.sources import fetch_fairface

    measure = Measure()
    keys = ("found", "landmarks", "pose", "mask")
    out = {k: [] for k in keys + ("age", "gender", "race", "size")}
    for file in fetch_fairface(split):
        t = pq.read_table(file)
        for b, age, gender, race in zip(
            t.column("image").to_pylist(),
            t.column("age").to_pylist(),
            t.column("gender").to_pylist(),
            t.column("race").to_pylist(),
        ):
            image = np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))
            r = measure(image)
            for k in keys:
                out[k].append(r[k])
            out["age"].append(age)
            out["gender"].append(gender)
            out["race"].append(race)
            out["size"].append(image.shape[0])
    det = {k: np.asarray(v) for k, v in out.items()}
    det["landmarks"] = det["landmarks"].astype(np.float32)
    det["pose"] = det["pose"].astype(np.float32)
    np.savez_compressed(path, **det)
    return det


def photo_groups(det: dict) -> dict:
    """the metrics of the near-frontal FairFace faces aged 20 to 49, by gender and hair length"""
    ok = det["found"] & frontal(det["pose"]) & np.isin(det["age"], PHOTO_AGES)
    m = table(det["landmarks"], det["mask"], det["size"], ok)
    groups = {}
    for g, gender in enumerate(("man", "woman")):
        for short in (True, False):
            sel = (
                ok
                & (det["gender"] == g)
                & (m["short"] == short)
                & np.isfinite(m["forehead"])
            )
            groups[f"{gender}, {'short' if short else 'long'} hair"] = {
                k: m[k][sel] for k in METRICS
            }
    return groups


# ------------------------------------------------------------------ anny's portraits
def ages(years):
    from anny.hair.authoring.review import ages as review_ages

    return review_ages(years)


def portraits(styles: list[str], size: int = 448, page=None) -> list[dict]:
    """anny's frontal portraits with each style on each body of BODIES, measured"""
    from anny.faces.authoring.photos import render_anny

    age = ages([b[2] for b in BODIES])
    looks, rows = [], []
    for name in styles:
        for b, (gender, g, years) in enumerate(BODIES):
            looks.append(
                dict(
                    format="anny-viewer/look@3",
                    name="bench",
                    phenotype=dict(age=age[b], gender=g),
                    hair=dict(color=HAIR_COLOUR, style=name),
                )
            )
            rows.append(dict(style=name, gender=gender, years=years))
    measure = Measure()

    def on_image(i, image):
        r = measure(image)
        rows[i].update(
            found=r["found"], frontal=bool(frontal(r["pose"])) if r["found"] else False
        )
        if r["found"]:
            rows[i].update(
                hair_metrics(r["landmarks"], r["mask"] / 255.0, image.shape[0])
            )
            rows[i]["overlay"] = overlay(image, r["landmarks"], r["mask"], rows[i])

    render_anny(looks, size=size, page=page, hair=True, on_image=on_image)
    return rows


def overlay(image, landmarks, mask, m) -> str:
    """a small PNG (data URL) of a portrait with its hair and the heights of its metrics"""
    from PIL import Image, ImageDraw

    size = image.shape[0]
    hair = (
        np.asarray(Image.fromarray(mask).resize((size, size), Image.BILINEAR)) / 255.0
    )
    rgb = image.astype(np.float64)
    rgb[..., 0] = rgb[..., 0] * (1 - 0.5 * hair) + 255 * 0.5 * hair
    im = Image.fromarray(rgb.astype(np.uint8))
    d = ImageDraw.Draw(im)
    P = landmarks[:, :2]
    n, gn = P[N], P[GN]
    down = (gn - n) / np.linalg.norm(gn - n)
    across = np.array([down[1], -down[0]])
    D = np.linalg.norm(gn - n)
    half = 0.5 * abs((P[SIDES[1]] - P[SIDES[0]]) @ across)
    pupil = abs((P[PUPILS[1]] - n) @ across)
    colours = dict(
        forehead=(0, 200, 255),
        pupil=(255, 220, 0),
        temple=(0, 255, 120),
        sideburn=(255, 80, 255),
    )
    for key, offsets in (
        ("forehead", [0.0]),
        ("pupil", [-pupil, pupil]),
        ("temple", [-TEMPLE * half, TEMPLE * half]),
        ("sideburn", [-half, half]),
    ):
        v = m.get(key)
        if v is None or not np.isfinite(v):
            continue
        for o in offsets:
            c = n + o * across - v * D * down
            d.line(
                [tuple(c - 6 * across), tuple(c + 6 * across)],
                fill=colours[key],
                width=3,
            )
    buf = io.BytesIO()
    im.resize((160, 160)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ the report
def compare(rows: list[dict], groups: dict) -> list[dict]:
    """each portrait's metrics as percentiles of the photos of its gender and hair length"""
    for r in rows:
        if not r.get("found"):
            continue
        g = groups[f"{r['gender']}, {'short' if r['short'] else 'long'} hair"]
        r["group"] = f"{r['gender']}, {'short' if r['short'] else 'long'} hair"
        r["percentile"] = {
            k: float(100 * np.mean(g[k][np.isfinite(g[k])] < r[k]))
            for k in METRICS
            if np.isfinite(r.get(k, np.nan))
        }
    return rows


def summary(groups: dict) -> dict:
    out = {}
    for name, g in groups.items():
        out[name] = dict(n=int(len(g["forehead"])))
        for k in METRICS:
            v = g[k][np.isfinite(g[k])]
            out[name][k] = [round(float(q), 3) for q in np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])]  # fmt: skip
    return out


def html_report(result: dict) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Anny hairline benchmark</title>",
        "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}"
        "table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}"
        "th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}.out{background:#fde2e2}"
        ".in{background:#e3f6e3}img{width:120px;height:120px}</style>",
        f"<h1>Anny hairline benchmark</h1><p>{html.escape(result.get('label', ''))}</p>",
        "<p>Heights of the hair edge above the nasion, in units of the nasion-chin distance, on near-frontal faces. "
        "The photos are FairFace's validation photos aged 20 to 49. A cell is green when anny's value lies between "
        "the photos' 25th and 75th percentiles, and red when it lies outside the 10th to 90th.</p>",
        "<h2>Photos</h2><table><tr><th>group</th><th>n</th>"
        + "".join(f"<th>{k}<br>p10 p25 p50 p75 p90</th>" for k in METRICS)
        + "</tr>",
    ]
    for name, s in result["photos"].items():
        parts.append(
            f"<tr><td>{html.escape(name)}</td><td>{s['n']}</td>"
            + "".join(f"<td>{' '.join(f'{q:.2f}' for q in s[k])}</td>" for k in METRICS)
            + "</tr>"
        )
    parts.append(
        "</table><h2>anny</h2><table><tr><th>style</th><th>body</th><th>group</th>"
    )
    parts.append("".join(f"<th>{k}</th>" for k in METRICS) + "<th>portrait</th></tr>")
    for r in result["anny"]:
        if not r.get("found"):
            parts.append(
                f"<tr><td>{r['style']}</td><td>{r['gender']} {r['years']:.0f}</td><td colspan=6>no face found</td></tr>"
            )
            continue
        cells = []
        for k in METRICS:
            p = r["percentile"].get(k)
            v = r.get(k)
            if p is None:
                cells.append("<td>-</td>")
                continue
            cls = "in" if 25 <= p <= 75 else ("out" if p < 10 or p > 90 else "")
            cells.append(f"<td class='{cls}'>{v:.2f} (p{p:.0f})</td>")
        img = f"<img src='{r['overlay']}'>" if r.get("overlay") else ""
        parts.append(
            f"<tr><td>{r['style']}</td><td>{r['gender']} {r['years']:.0f}</td><td>{r['group']}</td>"
            + "".join(cells)
            + f"<td>{img}</td></tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def run(styles=None, label="", page=None) -> dict:
    from anny.hair.styles import style_names

    styles = styles or [s for s in style_names() if s != "bald"]
    groups = photo_groups(fairface_hair("validation"))
    rows = compare(portraits(styles, page=page), groups)
    result = dict(label=label, photos=summary(groups), anny=rows)
    out = cache_dir()
    with open(out / "photos.html", "w") as f:
        f.write(html_report(result))
    slim = dict(
        result, anny=[{k: v for k, v in r.items() if k != "overlay"} for r in rows]
    )
    with open(out / "photos.json", "w") as f:
        json.dump(slim, f, indent=1)
    return slim


def main():
    parser = argparse.ArgumentParser(
        description="benchmark anny's hairline against photos"
    )
    parser.add_argument("--styles", nargs="*", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--page",
        type=pathlib.Path,
        default=None,
        help="the viewer page (viewer/dist by default)",
    )
    args = parser.parse_args()
    result = run(args.styles, args.label, args.page)
    for name, s in result["photos"].items():
        print(name, s["n"], {k: s[k][2] for k in METRICS})
    for r in result["anny"]:
        if r.get("found"):
            print(
                f"{r['style']:18s} {r['gender']:5s} {r['years']:3.0f} {r['group']:17s}",
                " ".join(
                    f"{k} {r[k]:5.2f} (p{r['percentile'].get(k, float('nan')):3.0f})"
                    for k in METRICS
                ),
            )
    print(cache_dir() / "photos.html")


if __name__ == "__main__":
    main()
