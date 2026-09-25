# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Download the data sources of anny's face calibration into ``ANNY_CACHE_DIR/faces``.

Every source is free and downloads without an account (``data/faces/SOURCES.md`` lists their
licences and attributions). The files stay in the cache and never enter the repository; the
repository keeps only the parameters derived from them.

Usage::

    python -m anny.faces.authoring.sources [--only ansur tdfn ...]

A host that the network refuses stops the command with its name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import subprocess

import numpy as np
import requests

from anny.paths import get_anny_cache_path

SOURCES = {
    "ansur_male": "https://tools.openlab.psu.edu/publicData/ANSUR_II_MALE_Public.csv",
    "ansur_female": "https://tools.openlab.psu.edu/publicData/ANSUR_II_FEMALE_Public.csv",
    "tdfn": "https://www.facebase.org/resources/human/facial_norms/summary/tdfn_gui_summary.csv",
    "cdc_head_circumference": "https://www.cdc.gov/growthcharts/data/zscore/hcageinf.csv",
    "mediapipe_face_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/latest/face_landmarker.task"
    ),
    "mediapipe_canonical_face": (
        "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/"
        "face_geometry/data/canonical_face_model.obj"
    ),
}
FILENAMES = {
    "ansur_male": "ANSUR_II_MALE_Public.csv",
    "ansur_female": "ANSUR_II_FEMALE_Public.csv",
    "tdfn": "tdfn_gui_summary.csv",
    "cdc_head_circumference": "hcageinf.csv",
    "mediapipe_face_landmarker": "face_landmarker.task",
    "mediapipe_canonical_face": "canonical_face_model.obj",
}
ICT_FACEKIT = "https://github.com/USC-ICT/ICT-FaceKit"
FAIRFACE = "HuggingFaceM4/FairFace"
FAIRFACE_FILES = {
    # the crops with a margin of 1.25 of the face size (the photos around the face)
    "validation": "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet",
}


def cache_dir() -> pathlib.Path:
    d = get_anny_cache_path() / "faces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, path: pathlib.Path) -> None:
    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
    except requests.RequestException as error:
        host = url.split("/")[2]
        raise RuntimeError(
            f"could not download {url} ({error}). If the network refuses the host {host}, "
            "allow it in the environment's network settings."
        ) from error
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for block in response.iter_content(1 << 20):
            f.write(block)
    tmp.rename(path)


def fetch(name: str) -> pathlib.Path:
    """the cached file of a source, downloaded if needed"""
    path = cache_dir() / FILENAMES[name]
    if not path.exists():
        _download(SOURCES[name], path)
    return path


def fetch_ict_facekit() -> pathlib.Path:
    """a shallow clone of ICT-FaceKit (MIT)"""
    path = cache_dir() / "ICT-FaceKit"
    if not (path / "FaceXModel").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", ICT_FACEKIT, str(path)], check=True
        )
    return path


def fetch_fairface(split: str = "validation") -> pathlib.Path:
    """a FairFace parquet file (CC BY 4.0), downloaded from Hugging Face"""
    path = cache_dir() / f"fairface_{split}.parquet"
    if not path.exists():
        url = f"https://huggingface.co/datasets/{FAIRFACE}/resolve/main/{FAIRFACE_FILES[split]}"
        _download(url, path)
    return path


def record(names=None) -> dict:
    """fetch the sources and return their files and checksums"""
    out = {}
    for name in names or list(SOURCES):
        path = fetch(name)
        out[name] = dict(url=SOURCES[name], file=path.name, sha256=_sha256(path))
    if names is None or "ict" in names:
        path = fetch_ict_facekit()
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        out["ict"] = dict(url=ICT_FACEKIT, commit=commit)
    if names is None or "fairface" in names:
        path = fetch_fairface()
        out["fairface"] = dict(url=FAIRFACE, file=path.name, sha256=_sha256(path))
    return out


# ------------------------------------------------------------------ parsers
def ansur() -> dict[str, dict[str, np.ndarray]]:
    """ANSUR II per sex ("male", "female"): column -> values (measurements in mm)"""
    out = {}
    for sex in ("male", "female"):
        path = fetch(f"ansur_{sex}")
        with open(path, newline="", encoding="latin-1") as f:
            rows = list(csv.DictReader(f))
        columns = {}
        for key in rows[0]:
            values = [r[key] for r in rows]
            try:
                columns[key.strip().lower()] = np.array([float(v) for v in values])
            except ValueError:
                columns[key.strip().lower()] = np.array(values)
        # the files give the interpupillary breadth in tenths of a millimetre
        columns["interpupillarybreadth"] = columns["interpupillarybreadth"] / 10.0
        out[sex] = columns
    return out


def tdfn() -> list[dict]:
    """3D Facial Norms summary rows: measure, age band [lo, hi), sex (male, female, all), n, mean, sd"""
    path = fetch("tdfn")
    sexes = {"1": "male", "2": "female", "3": "all"}
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            _, measure, lo, hi, sex, n, mean, sd = row[:8]
            out.append(
                dict(
                    measure=measure,
                    age=(float(lo), float(hi)),
                    sex=sexes[sex],
                    n=int(float(n)),
                    mean=float(mean),
                    sd=float(sd),
                )
            )
    return out


def cdc_head_circumference() -> dict[str, np.ndarray]:
    """CDC head circumference for age (0 to 36 months): sex (1 male, 2 female), age in months, L, M, S"""
    path = fetch("cdc_head_circumference")
    text = path.read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    return {
        key: np.array([float(r[key]) for r in rows])
        for key in ("Sex", "Agemos", "L", "M", "S")
    }


def main():
    parser = argparse.ArgumentParser(
        description="download the face calibration sources"
    )
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    info = record(args.only)
    print(json.dumps(info, indent=1))
    with open(cache_dir() / "sources.json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
