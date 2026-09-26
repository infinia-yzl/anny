# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Generate ``src/anny/data/faces/face_shapes.json``, the spec of anny's face-shape parameters.

Each parameter merges the MakeHuman targets of both sides of the face into one symmetric
parameter. A value of +1 applies the positive targets and -1 the negative targets. The head
archetypes and ``chin-triangle`` have one direction only.

The names drop the ``-neg-pos`` suffix of the MakeHuman category label where the result stays
unique (``eye-scale-decr-incr`` becomes ``eye-scale``), and keep the full label otherwise.

Usage::

    uv run python scripts/make_face_shape_spec.py
"""

import collections
import json
import os

from anny.paths import get_anny_root_dir

# MakeHuman target folders -> face-shape groups, in the order of the viewer's panel
GROUPS = {
    "head": "head",
    "forehead": "forehead",
    "eyebrows": "brows",
    "eyes": "eyes",
    "nose": "nose",
    "cheek": "cheeks",
    "mouth": "mouth",
    "chin": "chin",
    "ears": "ears",
}
# Categories that move or tilt the whole head: they are placement, not shape
EXCLUDED = {
    "head-angle-in-out",
    "head-trans-backward-forward",
    "head-trans-down-up",
    "head-trans-in-out",
}


def _strip_side(target: str) -> str:
    for prefix in ("l-", "r-"):
        if target.startswith(prefix):
            return target[len(prefix) :]
    return target


def _entries(folder: str, category: dict) -> dict | None:
    """positive and negative targets (both sides) of a MakeHuman category"""
    label = category["label"]
    if label in EXCLUDED:
        return None
    if "opposites" in category:
        opposites = category["opposites"]
        positive, negative = [], []
        for side in ("left", "right", "unsided"):
            if opposites[f"positive-{side}"]:
                positive.append(opposites[f"positive-{side}"])
            if opposites[f"negative-{side}"]:
                negative.append(opposites[f"negative-{side}"])
        neg_suffix = _strip_side(negative[0]).split("-")[-1]
        pos_suffix = _strip_side(positive[0]).split("-")[-1]
        suffix = f"-{neg_suffix}-{pos_suffix}"
        short = label[: -len(suffix)] if label.endswith(suffix) else label
    elif len(category["targets"]) == 1:
        # one-sided shape, such as the head archetypes
        positive, negative = list(category["targets"]), []
        short = label
    else:
        # eye-eyefold-concave-convex: two targets per side, without an opposites entry
        names = sorted({_strip_side(t) for t in category["targets"]})
        assert len(names) == 2, category
        neg_name, pos_name = sorted(
            names, key=lambda n: label.split("-").index(n.split("-")[-1])
        )
        positive = [t for t in category["targets"] if _strip_side(t) == pos_name]
        negative = [t for t in category["targets"] if _strip_side(t) == neg_name]
        suffix = f"-{neg_name.split('-')[-1]}-{pos_name.split('-')[-1]}"
        short = label[: -len(suffix)] if label.endswith(suffix) else label
    return dict(
        label=label,
        short=short,
        group=GROUPS[folder],
        positive=[f"{folder}/{t}" for t in positive],
        negative=[f"{folder}/{t}" for t in negative],
        range=[-1.0, 1.0] if negative else [0.0, 1.0],
    )


def main(output=None):
    root = get_anny_root_dir()
    with open(root / "data/mpfb2/targets/target.json") as f:
        targets = json.load(f)
    entries = []
    for folder in GROUPS:
        for category in targets[folder]["categories"]:
            entry = _entries(folder, category)
            if entry is not None:
                entries.append(entry)
    counts = collections.Counter(e["short"] for e in entries)
    parameters = []
    for e in entries:
        name = e["short"] if counts[e["short"]] == 1 else e["label"]
        for t in e["positive"] + e["negative"]:
            path = root / "data/mpfb2/targets" / f"{t}.target.gz"
            assert path.exists(), path
        parameters.append(
            dict(
                name=name,
                group=e["group"],
                positive=e["positive"],
                negative=e["negative"],
                range=e["range"],
            )
        )
    names = [p["name"] for p in parameters]
    assert len(set(names)) == len(names), "face-shape names are not unique"
    spec = dict(
        description=(
            "anny's face-shape parameters: symmetric sums of MakeHuman targets "
            "(scripts/make_face_shape_spec.py)"
        ),
        groups=list(dict.fromkeys(GROUPS.values())),
        parameters=parameters,
    )
    output = output or root / "data/faces/face_shapes.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    print(f"{len(parameters)} face-shape parameters -> {output}")


if __name__ == "__main__":
    main()
