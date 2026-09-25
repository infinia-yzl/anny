# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Build the calibrated distribution of anny's face shapes
(``data/shape_calibration/face_prior.safetensors``, read by :mod:`anny.faces.distribution`).

1. **ICT prior.** A Gaussian over (gender, weight, muscle, face values) of the fits of
   :mod:`anny.faces.authoring.fit_3d`, with Ledoit–Wolf shrinkage. The face values with gender,
   weight and muscle held at their mean give the starting mean and the covariance: the fitted
   phenotypes mostly trade off against the face shapes (a heavier fitted muscle goes with a
   deeper head shape, for example), so a regression on them would count anny's own phenotype
   shapes twice. The ear shapes, which the fits leave out, start from a zero mean and an
   independent SD of ``UNSEEN_SD``.
2. **Moment matching.** At each anchor age and for each sex, the Gaussian moves so that the
   measurements of the simulated bodies (:mod:`anny.faces.measurements`) meet the data:

   - adults (28 years): the 13 head and face measurements of ANSUR II (means and covariance),
     and the other measurements of 3D Facial Norms (ages 18 to 40);
   - older adults (47 years): ANSUR II from 40 years;
   - 3 to 18 years: 3D Facial Norms, as ratios to its adults applied to the adult targets, so
     that the growth of each measurement follows the data;
   - below 3 years: the head circumference of the CDC growth charts.

   The mean moves by linearised Gaussian conditioning of the face values on the measurements:
   ``K (target - simulated)`` with ``K = S J^T (J S J^T + N)^-1``, where J comes from finite
   differences and the simulated mean from bodies and faces drawn from the current Gaussian,
   over six rounds. The spread changes by one variance factor per group of face shapes (head,
   forehead, brows, eyes, nose, cheeks, mouth, chin, ears, detail), within ``VARIANCE_BOUNDS``,
   so that the predicted SD of each measurement, from the face and from the rest of the body
   (height, weight, muscle, proportions), meets the data. The factors keep the correlations of
   the ICT fits, and the bounds keep a measurement that the face shapes barely move from
   inflating their spread.
3. **Race offsets.** For users of anny's race phenotypes, the adult means of the ANSUR II race
   groups (White, Black and Asian for anny's caucasian, african and asian) give offsets of the
   mean, relative to their average.
4. **Slider ranges.** The range of each face shape in ``data/faces/face_shapes.json`` widens
   from [-1, 1] (or [0, 1]) to hold the central 99 % of the distribution at every anchor, sex and
   race, rounded outward to 0.5. The ICT fits need some shapes beyond 1: for example, the mean
   ICT face has less deep-set eyes and a narrower nose base than anny's default face.

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
from anny.models.face_shapes import face_shape_parameter
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
# face shapes with less than this share of their offsets on the fitted vertices are unseen by
# the ICT fits: their prior is centred on 0 with this SD, and ANSUR II calibrates the ears
UNSEEN_SHARE = 0.05
UNSEEN_SD = 0.35
# bounds of the variance factor of each group of face shapes (SD factors from 1/2 to 2)
VARIANCE_BOUNDS = (0.25, 4.0)


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
        groups = [face_shape_parameter(k).group for k in self.model.face_shape_labels]
        order = list(dict.fromkeys(groups))
        self.group_index = np.array([order.index(g) for g in groups])
        self.groups = (self.group_index, len(order))
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
        out = np.concatenate(out)
        if np.isnan(out).any():
            bad = [n for n, v in zip(names, np.isnan(out).any(0)) if v]
            raise ValueError(f"sections that miss the head: {bad}")
        return out


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
    target_sd,
    noise_sd,
    rounds=6,
    seed=0,
):
    """
    the Gaussian (mu, S) of the face values moved to meet the target measurements. Each round
    sets the spread by one variance factor per group of face shapes (``variance_factors``),
    draws a face from the Gaussian for each body of ``phen_samples``, and moves the mean by the
    gain of linearised Gaussian conditioning times the gap between the targets and the mean of
    the simulated measurements (the rectified shapes make that mean differ from the
    measurements of the mean face).
    """
    S0 = S
    z = np.random.default_rng(seed).standard_normal((len(phen_samples), len(mu)))
    # one-sided shapes act from 0 up, so their means stay at 0 or above
    mu = np.where(sim.one_sided, np.maximum(mu, 0.0), mu)
    for _ in range(rounds):
        _, J = jacobian(sim, phen_mean, mu, names)
        # the part of the measurement variance that the rest of the body explains
        other = sim(
            phen_samples, torch.tensor(np.repeat(mu[None], len(phen_samples), 0)), names
        )
        factors = variance_factors(J, S0, other.var(0), target_sd**2, sim.groups)
        d = np.sqrt(factors[sim.group_index])
        S = S0 * np.outer(d, d)
        faces = mu + z @ np.linalg.cholesky(S + 1e-10 * np.eye(len(mu))).T
        simulated = sim(phen_samples, torch.tensor(faces), names).mean(0)
        K = S @ J.T @ np.linalg.inv(J @ S @ J.T + np.diag(noise_sd**2))
        mu = mu + K @ (target_mean - simulated)
        mu = np.where(sim.one_sided, np.maximum(mu, 0.0), mu)
    return mu, S, simulated


def variance_factors(J, S, var_other, target_var, groups, pull=0.1) -> np.ndarray:
    """
    one variance factor per group of face shapes, within VARIANCE_BOUNDS, so that the predicted
    variance of each measurement (J S' J^T from the face plus the variance from the rest of the
    body) meets its target in the least-squares sense of log ratios; ``pull`` keeps the factors
    that the measurements do not see near 1
    """
    from scipy.optimize import minimize

    index, count = groups

    def predicted(log_f):
        d = np.exp(0.5 * log_f[index])
        return np.einsum("mi, ij, mj -> m", J * d, S, J * d) + var_other

    def loss(log_f):
        r = np.log(predicted(log_f)) - np.log(target_var)
        return (r**2).sum() + pull * (log_f**2).sum()

    bounds = [tuple(np.log(VARIANCE_BOUNDS))] * count
    result = minimize(loss, np.zeros(count), method="L-BFGS-B", bounds=bounds)
    return np.exp(result.x)


# ------------------------------------------------------------------ slider ranges
def widen_ranges(labels, calibrated: dict, race_offsets: np.ndarray, z: float = 2.576):
    """the slider ranges of face_shapes.json, widened to hold the calibrated distribution"""
    from anny.models.face_shapes import FACES_DIR, face_shape_spec

    lo = np.full(len(labels), np.inf)
    hi = np.full(len(labels), -np.inf)
    for (sex, _), (mu, S) in calibrated.items():
        sd = np.sqrt(np.diag(S))
        offsets = race_offsets[SEXES.index(sex)]
        shift_lo = np.minimum(offsets.min(0), 0.0)
        shift_hi = np.maximum(offsets.max(0), 0.0)
        lo = np.minimum(lo, mu + shift_lo - z * sd)
        hi = np.maximum(hi, mu + shift_hi + z * sd)
    path = FACES_DIR / "face_shapes.json"
    with open(path) as f:
        spec = json.load(f)
    ranges = {}
    for p in spec["parameters"]:
        i = labels.index(p["name"])
        one_sided = p.get("source", "makehuman") == "makehuman" and not p["negative"]
        low = 0.0 if one_sided else min(-1.0, np.floor(2 * lo[i]) / 2)
        high = max(1.0, np.ceil(2 * hi[i]) / 2)
        p["range"] = [float(low), float(high)]
        ranges[p["name"]] = p["range"]
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    face_shape_spec.cache_clear()
    return ranges


# ------------------------------------------------------------------ calibration
def unseen_shapes(model, fits: dict, labels: list[str]) -> np.ndarray:
    """whether each face shape puts less than UNSEEN_SHARE of its offsets (squared) on the
    vertices of the ICT fits: the ear shapes, since the fits leave out the ears"""
    fitted = np.zeros(len(model.template_vertices), bool)
    fitted[fits["vertex_ids"][fits["corr_valid"].astype(bool)]] = True
    rows = list(model.blendshape_labels)
    B = model.blendshapes.detach().cpu().numpy()
    unseen = []
    for name in labels:
        idx = [rows.index(x) for x in face_shape_parameter(name).row_labels]
        energy = (B[idx] ** 2).sum(-1).sum(0)
        unseen.append(energy[fitted].sum() < UNSEEN_SHARE * energy.sum())
    return np.array(unseen)


def ict_prior(fits: dict, labels: list[str], unseen: np.ndarray):
    """
    mean and covariance of the face values of the ICT fits with (gender, weight, muscle) held
    at their mean; the shapes that the fits do not see get a zero mean and an independent SD of
    UNSEEN_SD
    """
    train = ~fits["held_out"]
    phen_labels = list(fits["phenotype_labels"])
    cond = [phen_labels.index(k) for k in ("gender", "weight", "muscle")]
    face_labels = list(fits["face_labels"])
    seen = np.nonzero(~unseen)[0]
    order = [face_labels.index(labels[i]) for i in seen]
    X = np.concatenate(
        [fits["phenotype"][train][:, cond], fits["face"][train][:, order]], 1
    )
    mu = X.mean(0)
    S = ledoit_wolf(X)
    S11, S12, S22 = S[:3, :3], S[:3, 3:], S[3:, 3:]
    B = S12.T @ np.linalg.inv(S11)
    F = len(labels)
    mean_face = np.zeros(F)
    cov = np.diag(np.full(F, UNSEEN_SD**2))
    mean_face[seen] = mu[3:]
    cov[np.ix_(seen, seen)] = S22 - B @ S12
    return dict(
        mean_phen=mu[:3],
        mean_face=mean_face,
        cov=cov,
    )


def run(samples: int = 400, seed: int = 0):
    t0 = time.time()
    sim = Simulator()
    labels = list(sim.model.face_shape_labels)
    fits = dict(np.load(sources.cache_dir() / "ict_fits.npz"))
    unseen = unseen_shapes(sim.model, fits, labels)
    print(
        f"shapes that the ICT fits do not see: {[labels[i] for i in np.nonzero(unseen)[0]]}"
    )
    prior = ict_prior(fits, labels, unseen)
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
        phen_mean, phen_rand = phen_sets(sex, ADULT_YEARS)
        mu_a, S_a, m_pred = moment_match(
            sim,
            prior["mean_face"],
            prior["cov"],
            phen_mean,
            phen_rand,
            adult_names,
            adult_mean,
            adult_sd,
            0.15 * adult_sd,
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
            np.sqrt(np.diag(C_old)),
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
            mu_g, S_g, _ = moment_match(
                sim, mu_a, S_a, phen_mean, phen_rand, names, mean, sd, 0.15 * sd
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
                np.array([hc_sd]),
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

    # ---------------- slider ranges
    report["ranges"] = widen_ranges(labels, calibrated, race_offsets)
    wide = {k: v for k, v in report["ranges"].items() if v[1] > 1 or v[0] < -1}
    print(f"widened ranges: {wide}")

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
