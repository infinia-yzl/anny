# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Build the calibrated distribution of anny's face shapes
(``data/shape_calibration/face_prior.safetensors``, read by :mod:`anny.faces.distribution`).

1. **ICT prior.** A Gaussian over (gender, weight, muscle, face values) of the fits of
   :mod:`anny.faces.authoring.fit_3d`, with Ledoit–Wolf shrinkage. Conditioning on gender,
   weight and muscle gives the face values: a mean that moves linearly with them and a fixed
   covariance.
2. **Moment matching.** At each anchor age and for each sex, the Gaussian moves so that the
   measurements of the simulated bodies (:mod:`anny.faces.measurements`) meet the data:

   - adults (28 years): the 13 head and face measurements of ANSUR II (means and covariance),
     and the other measurements of 3D Facial Norms (ages 18 to 40);
   - older adults (47 years): ANSUR II from 40 years;
   - 3 to 18 years: 3D Facial Norms, as ratios to its adults applied to the adult targets, so
     that the growth of each measurement follows the data;
   - below 3 years: the head circumference of the CDC growth charts.

   The update is the Gaussian conditioning of the face values on the measurements, linearised
   by finite differences: the mean moves by ``K (target - predicted)`` with
   ``K = S J^T (J S J^T + N)^-1``, and the covariance changes only in the measured directions,
   so that ``J S' J^T`` equals the data covariance minus the part that the rest of the body
   (height, weight, muscle, proportions) explains. The directions that no measurement sees keep
   the ICT covariance.
3. **Race offsets.** For users of anny's race phenotypes, the adult means of the ANSUR II race
   groups (White, Black and Asian for anny's caucasian, african and asian) give offsets of the
   mean, relative to their average.

Usage::

    python -m anny.faces.authoring.calibrate

The command also writes ``ANNY_CACHE_DIR/faces/calibration_report.json``: the simulated and
the target means and SDs at each anchor.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from safetensors.torch import save_file

import anny
from anny.faces.authoring import sources
from anny.faces.distribution import DEFAULT_PATH, RACES
from anny.faces.measurements import (
    ANSUR_MEASUREMENTS,
    ANSUR_SECTION_MEASUREMENTS,
    TDFN_MEASUREMENTS,
    CraniofacialMeasurements,
)
from anny.shape_distribution import SimpleShapeDistribution

ANCHOR_YEARS = (
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
)
ADULT_YEARS = 28.0
OLDER_YEARS = 47.0
SEXES = ("male", "female")
GENDER = {"male": 0.0, "female": 1.0}
# measurements of 3D Facial Norms that ANSUR II measures too: the adult targets come from ANSUR II
TDFN_TO_ANSUR = {
    "maxcranlength": "headlength",
    "maxcranwidth": "headbreadth",
    "maxfacewidth": "bizygomaticbreadth",
    "morphfaceheight": "mentonsellionlength",
}
ANSUR_RACES = {"caucasian": 1.0, "african": 2.0, "asian": 4.0}


def tdfn_names() -> list[str]:
    """3D Facial Norms measurements, with the left and right pairs merged"""
    names = []
    for k in TDFN_MEASUREMENTS:
        base = k[:-2] if k.endswith(("_l", "_r")) else k
        if base not in names:
            names.append(base)
    return names


def _tdfn_value(values: dict, name: str):
    if name in values:
        return values[name]
    return 0.5 * (values[name + "_l"] + values[name + "_r"])


# ------------------------------------------------------------------ data targets
def tdfn_table():
    """pooled mean and SD of each 3D Facial Norms measurement: (name, sex, (lo, hi)) -> (n, mean, sd)"""
    rows = sources.tdfn()

    def pooled(name, sex, lo, hi):
        parts = []
        for r in rows:
            base = r["measure"]
            if base.endswith(("_l", "_r")):
                base = base[:-2]
            if base != name or r["sex"] != sex:
                continue
            a, b = r["age"]
            if a < hi and b > lo:
                parts.append((r["n"], r["mean"], r["sd"]))
        if not parts:
            return None
        n = np.array([p[0] for p in parts], float)
        m = np.array([p[1] for p in parts])
        s = np.array([p[2] for p in parts])
        mean = (n * m).sum() / n.sum()
        var = (n * (s**2 + (m - mean) ** 2)).sum() / n.sum()
        return n.sum(), mean, np.sqrt(var)

    return pooled


def ansur_targets(sex: str, min_age=0.0, max_age=200.0, race=None):
    """ANSUR II means (M,) and covariance (M, M) of the head and face measurements"""
    data = sources.ansur()[sex]
    keep = (data["age"] >= min_age) & (data["age"] < max_age)
    if race is not None:
        keep &= data["dodrace"] == race
    X = np.stack([data[k][keep] for k in ANSUR_MEASUREMENTS], 1)
    return X.mean(0), np.cov(X.T), int(keep.sum())


def cdc_head_circumference(sex: str, months: float):
    """CDC median and SD (mm) of the head circumference at an age in months (0 to 36)"""
    table = sources.cdc_head_circumference()
    rows = table["Sex"] == (1.0 if sex == "male" else 2.0)
    ages = table["Agemos"][rows]
    L, M, S = (np.interp(months, ages, table[k][rows]) for k in ("L", "M", "S"))
    # SD from the 16th and 84th percentiles of the LMS distribution
    z = np.array([-1.0, 1.0])
    values = M * (1 + L * S * z) ** (1 / L)
    return 10 * M, 10 * (values[1] - values[0]) / 2


# ------------------------------------------------------------------ the simulator
class Simulator:
    """measurements of anny's bodies for phenotypes and face values"""

    def __init__(self):
        self.model = anny.Anny(face_shapes="all", phenotypes="all").to(
            dtype=torch.float64
        )
        self.measure = CraniofacialMeasurements(self.model)
        self.shape = SimpleShapeDistribution(self.model)
        self.labels = self.model.phenotype_labels
        self.one_sided = np.array(
            [
                self.model.face_shape_ranges[k][0] >= 0
                for k in self.model.face_shape_labels
            ]
        )

    def anny_age(self, years: float) -> float:
        mapping = self.shape.morphological_age_mapping
        return float(mapping.morphological_to_anny_age(torch.tensor([years])).item())

    def phenotypes(self, sex: str, years: float, n: int, seed: int, race=None):
        """(n, P) phenotypes of bodies of a sex and age; n=0 gives the mean body"""
        age = self.anny_age(years)
        params = torch.full((max(n, 1), len(self.labels)), 0.5, dtype=torch.float64)
        params[:, self.labels.index("age")] = age
        params[:, self.labels.index("gender")] = GENDER[sex]
        dists = (
            self.shape.boys_conditional_height_distribution,
            self.shape.boys_conditional_weight_distribution,
            self.shape.boys_conditional_muscle_distribution,
            self.shape.boys_conditional_proportions_distribution,
        )
        if sex == "female":
            dists = (
                self.shape.girls_conditional_height_distribution,
                self.shape.girls_conditional_weight_distribution,
                self.shape.girls_conditional_muscle_distribution,
                self.shape.girls_conditional_proportions_distribution,
            )
        a = torch.tensor([age], dtype=torch.float64)
        g = torch.Generator().manual_seed(seed)
        for name, d in zip(("height", "weight", "muscle", "proportions"), dists):
            beta = d.get_torch_distribution(a)
            if n == 0:
                params[:, self.labels.index(name)] = beta.mean
            else:
                u = torch.rand(n, generator=g, dtype=torch.float64)
                # the inverse CDF of the Beta distribution, by a lookup table
                grid = torch.linspace(1e-4, 1 - 1e-4, 2001, dtype=torch.float64)
                cdf = torch.cumsum(beta.log_prob(grid).exp(), 0)
                cdf = cdf / cdf[-1]
                params[:, self.labels.index(name)] = torch.from_numpy(
                    np.interp(u.numpy(), cdf.numpy(), grid.numpy())
                )
        if race is not None:
            for r in RACES:
                params[:, self.labels.index(r)] = 1.0 if r == race else 0.0
        return params

    def __call__(
        self, phenotypes: torch.Tensor, face: torch.Tensor, names
    ) -> np.ndarray:
        """measurements (B, M) of the named measurements"""
        out = []
        sections = any(n in ANSUR_SECTION_MEASUREMENTS for n in names)
        for start in range(0, len(phenotypes), 64):
            with torch.no_grad():
                o = self.model(
                    phenotype_kwargs=phenotypes[start : start + 64],
                    face_shape_kwargs=face[start : start + 64],
                )
                values = self.measure(o, sections=sections)
            out.append(np.stack([_tdfn_value(values, n).numpy() for n in names], 1))
        return np.concatenate(out)


# ------------------------------------------------------------------ Gaussian tools
def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """covariance with Ledoit–Wolf shrinkage toward a scaled identity (on standardised data)"""
    sd = X.std(0)
    Z = (X - X.mean(0)) / sd
    n, p = Z.shape
    S = Z.T @ Z / n
    m = np.trace(S) / p
    d2 = ((S - m * np.eye(p)) ** 2).sum()
    # (1/n^2) sum_k ||z_k z_k^T - S||^2 = (sum_k ||z_k||^4 - n ||S||^2) / n^2
    b2 = min(((Z**2).sum(1) ** 2).sum() / n**2 - (S**2).sum() / n, d2)
    shrink = b2 / d2
    C = shrink * m * np.eye(p) + (1 - shrink) * S
    return C * np.outer(sd, sd)


def psd(C: np.ndarray, floor: float) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    return (V * np.maximum(w, floor)) @ V.T


def jacobian(sim: Simulator, phen: torch.Tensor, face: np.ndarray, names, h=0.2):
    """finite differences (M, F) of the measurements with the face values, at one body:
    central for the paired shapes, forward for the one-sided ones"""
    F = len(face)
    faces = np.repeat(face[None], 2 * F + 1, 0)
    steps = np.full(F, 2 * h)
    for i in range(F):
        faces[1 + 2 * i, i] += h
        if sim.one_sided[i]:
            steps[i] = h
        else:
            faces[2 + 2 * i, i] -= h
    values = sim(phen.expand(2 * F + 1, -1), torch.tensor(faces), names)
    J = (values[1::2] - values[2::2]).T / steps
    return values[0], J


def moment_match(
    sim,
    mu,
    S,
    phen_mean,
    phen_samples,
    names,
    target_mean,
    target_cov,
    noise_sd,
    rounds=2,
):
    """the Gaussian (mu, S) of the face values moved to meet the target measurements"""
    # one-sided shapes act from 0 up, so their means stay at 0 or above
    mu = np.where(sim.one_sided, np.maximum(mu, 0.0), mu)
    for _ in range(rounds):
        m_pred, J = jacobian(sim, phen_mean, mu, names)
        P = J @ S @ J.T
        N = np.diag(noise_sd**2)
        K = S @ J.T @ np.linalg.inv(P + N)
        mu = mu + K @ (target_mean - m_pred)
        mu = np.where(sim.one_sided, np.maximum(mu, 0.0), mu)
    m_pred, J = jacobian(sim, phen_mean, mu, names)
    # the part of the measurement covariance that the rest of the body explains
    other = sim(
        phen_samples, torch.tensor(np.repeat(mu[None], len(phen_samples), 0)), names
    )
    C_other = np.cov(other.T)
    P = J @ S @ J.T
    floor = 1e-3 * np.diag(target_cov).mean()
    C_face = psd(target_cov - C_other, floor)
    K0 = S @ J.T @ np.linalg.inv(P + 1e-6 * np.trace(P) / len(P) * np.eye(len(P)))
    S = S - K0 @ P @ K0.T + K0 @ C_face @ K0.T
    S = psd(S, 1e-8)
    return mu, S, m_pred


# ------------------------------------------------------------------ calibration
def ict_prior(fits: dict, labels: list[str]):
    """mean and covariance of the face values given (gender, weight, muscle), from the ICT fits"""
    train = ~fits["held_out"]
    phen_labels = list(fits["phenotype_labels"])
    cond = [phen_labels.index(k) for k in ("gender", "weight", "muscle")]
    face_labels = list(fits["face_labels"])
    order = [face_labels.index(k) for k in labels]
    X = np.concatenate(
        [fits["phenotype"][train][:, cond], fits["face"][train][:, order]], 1
    )
    mu = X.mean(0)
    S = ledoit_wolf(X)
    S11, S12, S22 = S[:3, :3], S[:3, 3:], S[3:, 3:]
    B = S12.T @ np.linalg.inv(S11)
    return dict(
        mean_phen=mu[:3],
        mean_face=mu[3:],
        regression=B,  # (F, 3): gender, weight, muscle
        cov=S22 - B @ S12,
    )


def conditional_mean(prior, gender, weight, muscle):
    x = np.array([gender, weight, muscle]) - prior["mean_phen"]
    return prior["mean_face"] + prior["regression"] @ x


def run(samples: int = 400, seed: int = 0):
    t0 = time.time()
    sim = Simulator()
    labels = list(sim.model.face_shape_labels)
    fits = dict(np.load(sources.cache_dir() / "ict_fits.npz"))
    prior = ict_prior(fits, labels)
    pooled = tdfn_table()
    tnames = tdfn_names()
    report = dict(anchors=[])

    def phen_sets(sex, years, race=None):
        mean = sim.phenotypes(sex, years, 0, seed, race)
        rand = sim.phenotypes(sex, years, samples, seed + 1, race)
        return mean, rand

    calibrated = {}
    for sex in SEXES:
        # ---------------- adults: ANSUR II and 3D Facial Norms
        m_ansur, C_ansur, n_ansur = ansur_targets(sex, 17, 40)
        adult_names, adult_mean, adult_sd = list(ANSUR_MEASUREMENTS), list(m_ansur), []
        adult_sd = list(np.sqrt(np.diag(C_ansur)))
        tdfn_adult = {}
        for name in tnames:
            p = pooled(name, sex, 18, 41)
            tdfn_adult[name] = p
            if name in TDFN_TO_ANSUR:
                continue
            adult_names.append(name)
            adult_mean.append(p[1])
            adult_sd.append(p[2])
        adult_mean, adult_sd = np.array(adult_mean), np.array(adult_sd)
        k = len(ANSUR_MEASUREMENTS)
        phen_mean, phen_rand = phen_sets(sex, ADULT_YEARS)
        mu0 = conditional_mean(
            prior,
            GENDER[sex],
            float(phen_mean[0, sim.labels.index("weight")]),
            float(phen_mean[0, sim.labels.index("muscle")]),
        )
        # the target covariance: ANSUR II in full, and the 3D Facial Norms SDs with the
        # correlations that anny predicts
        pred0 = sim(
            phen_rand,
            torch.tensor(np.repeat(mu0[None], len(phen_rand), 0)),
            adult_names,
        )
        _, J0 = jacobian(sim, phen_mean, mu0, adult_names)
        C_pred = J0 @ prior["cov"] @ J0.T + np.cov(pred0.T)
        d = np.sqrt(np.diag(C_pred))
        R = C_pred / np.outer(d, d)
        C_target = R * np.outer(adult_sd, adult_sd)
        C_target[:k, :k] = C_ansur
        noise = 0.15 * adult_sd
        mu_a, S_a, m_pred = moment_match(
            sim,
            mu0,
            prior["cov"],
            phen_mean,
            phen_rand,
            adult_names,
            adult_mean,
            C_target,
            noise,
        )
        calibrated[(sex, ADULT_YEARS)] = (mu_a, S_a)
        report["anchors"].append(
            dict(
                sex=sex,
                years=ADULT_YEARS,
                n_ansur=n_ansur,
                names=adult_names,
                target_mean=adult_mean.tolist(),
                target_sd=adult_sd.tolist(),
            )
        )
        print(f"{sex} adults: done ({time.time() - t0:.0f} s)")

        # ---------------- older adults: ANSUR II from 40 years
        m_old, C_old, n_old = ansur_targets(sex, 40, 200)
        phen_mean, phen_rand = phen_sets(sex, OLDER_YEARS)
        mu_o, S_o, _ = moment_match(
            sim,
            mu_a,
            S_a,
            phen_mean,
            phen_rand,
            list(ANSUR_MEASUREMENTS),
            m_old,
            C_old,
            0.15 * np.sqrt(np.diag(C_old)),
        )
        calibrated[(sex, OLDER_YEARS)] = (mu_o, S_o)
        report["anchors"].append(
            dict(
                sex=sex,
                years=OLDER_YEARS,
                n_ansur=n_old,
                names=list(ANSUR_MEASUREMENTS),
                target_mean=m_old.tolist(),
                target_sd=np.sqrt(np.diag(C_old)).tolist(),
            )
        )

        # ---------------- growth: 3D Facial Norms ratios from 3 to 18 years
        adult_target = dict(zip(adult_names, adult_mean))
        for name, ansur_name in TDFN_TO_ANSUR.items():
            adult_target[name] = adult_target[ansur_name]
        for years in [y for y in ANCHOR_YEARS if y >= 3]:
            names, mean, sd = [], [], []
            for name in tnames:
                p = pooled(name, sex, years - 1, years + 1)
                if p is None:
                    continue
                ratio = p[1] / tdfn_adult[name][1]
                names.append(name)
                mean.append(ratio * adult_target[name])
                sd.append(p[2] * adult_target[name] / tdfn_adult[name][1])
            mean, sd = np.array(mean), np.array(sd)
            phen_mean, phen_rand = phen_sets(sex, years)
            pred = sim(
                phen_rand, torch.tensor(np.repeat(mu_a[None], len(phen_rand), 0)), names
            )
            _, Jg = jacobian(sim, phen_mean, mu_a, names)
            C_pred = Jg @ S_a @ Jg.T + np.cov(pred.T)
            dd = np.sqrt(np.diag(C_pred))
            C_target = C_pred / np.outer(dd, dd) * np.outer(sd, sd)
            mu_g, S_g, _ = moment_match(
                sim, mu_a, S_a, phen_mean, phen_rand, names, mean, C_target, 0.15 * sd
            )
            calibrated[(sex, years)] = (mu_g, S_g)
            report["anchors"].append(
                dict(
                    sex=sex,
                    years=years,
                    names=names,
                    target_mean=mean.tolist(),
                    target_sd=sd.tolist(),
                )
            )
            print(f"{sex} {years:g} years: done ({time.time() - t0:.0f} s)")

        # ---------------- infants: CDC head circumference, from the 3-year anchor
        mu_3, S_3 = calibrated[(sex, 3.0)]
        for years in [y for y in ANCHOR_YEARS if y < 3]:
            hc, hc_sd = cdc_head_circumference(sex, 12 * years)
            phen_mean, phen_rand = phen_sets(sex, years)
            mu_i, S_i, _ = moment_match(
                sim,
                mu_3,
                S_3,
                phen_mean,
                phen_rand,
                ["headcircumference"],
                np.array([hc]),
                np.array([[hc_sd**2]]),
                np.array([0.15 * hc_sd]),
            )
            calibrated[(sex, years)] = (mu_i, S_i)
            report["anchors"].append(
                dict(
                    sex=sex,
                    years=years,
                    names=["headcircumference"],
                    target_mean=[hc],
                    target_sd=[hc_sd],
                )
            )
            print(f"{sex} {years:g} years: done ({time.time() - t0:.0f} s)")

    # ---------------- race offsets (adults, relative to their average)
    race_offsets = np.zeros((2, len(RACES), len(labels)))
    for si, sex in enumerate(SEXES):
        mu_a, S_a = calibrated[(sex, ADULT_YEARS)]
        deltas = []
        for r in RACES:
            m_r, C_r, n_r = ansur_targets(sex, 17, 40, ANSUR_RACES[r])
            phen_mean, _ = phen_sets(sex, ADULT_YEARS, race=r)
            mu_r = mu_a.copy()
            for _ in range(2):
                m_pred, J = jacobian(sim, phen_mean, mu_r, list(ANSUR_MEASUREMENTS))
                P = J @ S_a @ J.T
                noise = np.diag(
                    (0.15 * np.sqrt(np.diag(C_r))) ** 2 + np.diag(C_r) / n_r
                )
                mu_r = mu_r + S_a @ J.T @ np.linalg.solve(P + noise, m_r - m_pred)
            deltas.append(mu_r - mu_a)
        deltas = np.array(deltas)
        race_offsets[si] = deltas - deltas.mean(0)
        print(f"{sex} race offsets: done ({time.time() - t0:.0f} s)")

    # ---------------- save
    years = sorted({y for (_, y) in calibrated})
    age_anchors = [sim.anny_age(y) for y in years]
    F = len(labels)
    mean = np.zeros((len(years), 2, F))
    tril = np.zeros((len(years), 2, F, F))
    for ai, y in enumerate(years):
        for si, sex in enumerate(SEXES):
            mu, S = calibrated[(sex, y)]
            mean[ai, si] = mu
            tril[ai, si] = np.linalg.cholesky(S + 1e-10 * np.eye(F))
    tensors = dict(
        age_anchors=torch.tensor(age_anchors),
        gender_anchors=torch.tensor([GENDER["male"], GENDER["female"]]),
        mean=torch.tensor(mean),
        scale_tril=torch.tensor(tril),
        weight_muscle_regression=torch.tensor(prior["regression"][:, 1:3]),
        weight_muscle_centre=torch.tensor(prior["mean_phen"][1:3]),
        race_offsets=torch.tensor(race_offsets),
    )
    meta = dict(
        face_labels=labels,
        anchor_years=years,
        sources=json.load(open(sources.cache_dir() / "sources.json")),
    )
    save_file(
        {k: v.to(torch.float32).contiguous() for k, v in tensors.items()},
        str(DEFAULT_PATH),
        metadata={"prior": json.dumps(meta)},
    )
    with open(sources.cache_dir() / "calibration_report.json", "w") as f:
        json.dump(report, f)
    print(f"face prior -> {DEFAULT_PATH} ({time.time() - t0:.0f} s)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="calibrate anny's face-shape distribution"
    )
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()
    run(args.samples)


if __name__ == "__main__":
    main()
