// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// anny's phenotype sliders in the page: the same blend-shape coefficients as
// Anny._get_phenotype_blendshape_coefficients (src/anny/models/phenotype.py), the coarse body from the
// principal components that anny.viewer.export keeps, and the bone heads from the full joint blend shapes.
// This module has no dependencies, so node runs it directly in the parity test (test/test_viewer_parity.py).

export interface ShapeTables {
  labels: string[];                       // the sliders of the model (anny's default: gender, age, ...)
  variations: [string, string[]][];       // PHENOTYPE_VARIATIONS, in order (race first)
  anchors: Record<string, number[]>;      // anchor values of each phenotype except race
  extrapolate: boolean;
  mask: string[];                         // one string of 0 and 1 per blend shape (length 26)
  default: number;                        // the value of a slider that is not set (0.5)
}

const FEATURES = ['age', 'gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness'];

// anny.utils.interpolation.linear_interpolation_coefficients for one value
export function interpolationCoefficients(value: number, anchors: number[], extrapolate: boolean): Float64Array {
  const n = anchors.length;
  const out = new Float64Array(n);
  let idx = 0;
  while (idx < n && anchors[idx] < value) idx++;          // torch.searchsorted(side="left")
  idx = Math.min(Math.max(idx, 1), n - 1);
  const lo = anchors[idx - 1], hi = anchors[idx];
  let alpha = (value - lo) / (hi - lo);
  if (!extrapolate) alpha = Math.min(Math.max(alpha, 0), 1);
  out[idx - 1] = 1 - alpha;
  out[idx] = alpha;
  return out;
}

export interface CoefficientRule {
  tables: ShapeTables;
  masks: Uint8Array[];      // per blend shape, the indices of the variations it multiplies
  order: string[];          // the 26 variation keys in order
}

export function makeRule(tables: ShapeTables): CoefficientRule {
  const order: string[] = [];
  for (const [, keys] of tables.variations) order.push(...keys);
  const masks = tables.mask.map((m) => {
    const ids: number[] = [];
    for (let j = 0; j < m.length; j++) if (m[j] === '1') ids.push(j);
    return Uint8Array.from(ids);
  });
  return { tables, masks, order };
}

// the coefficients of the phenotype blend shapes for slider values (missing sliders take the default)
export function coefficients(rule: CoefficientRule, values: Record<string, number>): Float64Array {
  const t = rule.tables;
  const value = (label: string) => (t.labels.includes(label) && typeof values[label] === 'number') ? values[label] : t.default;
  const phen: Record<string, number> = {};
  for (const f of FEATURES) {
    const keys = t.variations.find((v) => v[0] === f)[1];
    const w = interpolationCoefficients(value(f), t.anchors[f], t.extrapolate);
    keys.forEach((k, i) => { phen[k] = w[i]; });
  }
  const race = ['african', 'asian', 'caucasian'].map(value);
  const sum = race[0] + race[1] + race[2];
  ['african', 'asian', 'caucasian'].forEach((k, i) => { phen[k] = sum > 0 ? race[i] / sum : 1 / 3; });
  const vec = rule.order.map((k) => phen[k]);
  const out = new Float64Array(rule.masks.length);
  for (let n = 0; n < rule.masks.length; n++) {
    let p = 1;
    const m = rule.masks[n];
    for (let j = 0; j < m.length; j++) p *= vec[m[j]];
    out[n] = p;
  }
  return out;
}

export interface ShapeSpaceData {
  template: Float32Array;       // (coarse vertices, 3) anny's template in the page frame
  components: Int16Array;       // (K, coarse vertices, 3) in steps of componentScale[k]
  componentScale: number[];
  projection: Float32Array;     // (K, blend shapes)
  jointTemplate: Float32Array;  // (bones, 3)
  jointBlend: Float32Array;     // (blend shapes, bones, 3)
}

export class ShapeSpace {
  data: ShapeSpaceData;
  K: number;
  N: number;
  nc: number;
  nb: number;
  weights: Float64Array;
  constructor(data: ShapeSpaceData) {
    this.data = data;
    this.K = data.componentScale.length;
    this.N = data.projection.length / this.K;
    this.nc = data.template.length / 3;
    this.nb = data.jointTemplate.length / 3;
    this.weights = new Float64Array(this.K);
  }
  // coarse vertices (nc * 3) for the coefficients c
  coarse(c: Float64Array, out: Float32Array = new Float32Array(this.nc * 3)): Float32Array {
    const { projection, components, componentScale, template } = this.data;
    const K = this.K, N = this.N, n3 = this.nc * 3;
    for (let k = 0; k < K; k++) {
      let w = 0;
      const row = k * N;
      for (let i = 0; i < N; i++) w += projection[row + i] * c[i];
      this.weights[k] = w * componentScale[k];
    }
    const acc = new Float64Array(n3);
    for (let i = 0; i < n3; i++) acc[i] = template[i];
    for (let k = 0; k < K; k++) {
      const w = this.weights[k];
      if (w === 0) continue;
      const o = k * n3;
      for (let i = 0; i < n3; i++) acc[i] += w * components[o + i];
    }
    for (let i = 0; i < n3; i++) out[i] = acc[i];
    return out;
  }
  // rest bone heads (nb * 3) for the coefficients c
  joints(c: Float64Array, out: Float32Array = new Float32Array(this.nb * 3)): Float32Array {
    const { jointTemplate, jointBlend } = this.data;
    const nb3 = this.nb * 3;
    const acc = new Float64Array(nb3);
    for (let i = 0; i < nb3; i++) acc[i] = jointTemplate[i];
    for (let n = 0; n < this.N; n++) {
      const w = c[n];
      if (w === 0) continue;
      const o = n * nb3;
      for (let i = 0; i < nb3; i++) acc[i] += w * jointBlend[o + i];
    }
    for (let i = 0; i < nb3; i++) out[i] = acc[i];
    return out;
  }
}

// ------------------------------------------------------------------ face shapes (anny.models.face_shapes)
// Each face-shape parameter owns one or two blend-shape rows (its positive and negative directions). A row's
// weight is the rectified value of its parameter times the scale of its group: the size of that part of the head
// (landmark distances on the phenotype body) over its size on anny's default body.

export interface FacePrior {
  age_anchors: number[];
  gender_anchors: number[];
  means: number[][][];                    // (A, 2, F)
  rank: number;
}

export interface FaceTables {
  names: string[];
  groups: string[];
  group_order: string[];
  ranges: [number, number][];
  ends?: [string, string][];
  row_param: number[];
  row_sign: number[];
  row_scale: number[];
  scale_groups: string[];
  landmarks: string[];
  scale_pairs: [number, number][][];
  reference_sizes: number[];
  starts: number[];
  counts: number[];
  bones: number[];
  step: number;
  prior?: FacePrior;
}

export interface FaceData {
  tables: FaceTables;
  lmTemplate: Float32Array;   // (K, 3) landmarks of the template
  lmBlend: Float32Array;      // (N, K, 3) landmarks of the phenotype blend shapes
  ids: Uint32Array;           // coarse vertex of each sparse offset
  offsets: Int16Array;        // (total, 3) offsets in steps of tables.step
  boneDeltas: Float32Array;   // (rows, bones, 3) bone-head deltas of each row on tables.bones
  prior?: Float32Array;       // (A, 2, F, rank) low-rank factors of the face-shape distribution
}

// scale of each group (anny's Anny._face_shape_scales) for the phenotype coefficients c
export function faceScales(face: FaceData, c: Float64Array): Float64Array {
  const t = face.tables, K = t.landmarks.length, K3 = K * 3, N = face.lmBlend.length / K3;
  const L = new Float64Array(K3);
  for (let i = 0; i < K3; i++) L[i] = face.lmTemplate[i];
  for (let n = 0; n < N; n++) {
    const w = c[n];
    if (w === 0) continue;
    const o = n * K3;
    for (let i = 0; i < K3; i++) L[i] += w * face.lmBlend[o + i];
  }
  const out = new Float64Array(t.scale_groups.length);
  t.scale_pairs.forEach((pairs, g) => {
    let s = 0;
    for (const [a, b] of pairs) {
      const dx = L[a * 3] - L[b * 3], dy = L[a * 3 + 1] - L[b * 3 + 1], dz = L[a * 3 + 2] - L[b * 3 + 2];
      s += Math.log(Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    out[g] = Math.exp(s / pairs.length) / t.reference_sizes[g];
  });
  return out;
}

// the weight of each face-shape row for parameter values (in the order of tables.names)
export function faceRowWeights(t: FaceTables, values: ArrayLike<number>, scales: Float64Array): Float64Array {
  const out = new Float64Array(t.row_param.length);
  for (let r = 0; r < out.length; r++) {
    const v = values[t.row_param[r]] * t.row_sign[r];
    out[r] = v > 0 ? v * scales[t.row_scale[r]] : 0;
  }
  return out;
}

// add the face-shape rows to coarse vertices (nc * 3) and to bone heads (nb * 3)
export function addFace(face: FaceData, weights: Float64Array, coarse: Float32Array, joints?: Float32Array) {
  const t = face.tables, step = t.step, nb = t.bones.length;
  for (let r = 0; r < weights.length; r++) {
    const w = weights[r];
    if (w === 0) continue;
    const ws = w * step, end = t.starts[r] + t.counts[r];
    for (let k = t.starts[r]; k < end; k++) {
      const o = face.ids[k] * 3, q = k * 3;
      coarse[o] += ws * face.offsets[q];
      coarse[o + 1] += ws * face.offsets[q + 1];
      coarse[o + 2] += ws * face.offsets[q + 2];
    }
    if (joints) {
      const o = r * nb * 3;
      for (let j = 0; j < nb; j++) {
        const b = t.bones[j] * 3, q = o + j * 3;
        joints[b] += w * face.boneDeltas[q];
        joints[b + 1] += w * face.boneDeltas[q + 1];
        joints[b + 2] += w * face.boneDeltas[q + 2];
      }
    }
  }
}

// the face-shape distribution (anny.faces.distribution) for anny's slider values: mean (F) and a low-rank factor (F, rank)
export function facePriorParameters(face: FaceData, values: Record<string, number>): { mean: Float64Array; factor: Float64Array } {
  const p = face.tables.prior, F = face.tables.names.length, R = p.rank;
  const val = (k: string) => (typeof values[k] === 'number' ? values[k] : 0.5);
  const wa = interpolationCoefficients(val('age'), p.age_anchors, false);
  const [g0, g1] = p.gender_anchors;
  const t = Math.min(1, Math.max(0, (val('gender') - g0) / (g1 - g0)));
  const mean = new Float64Array(F), factor = new Float64Array(F * R);
  for (let a = 0; a < wa.length; a++) {
    if (wa[a] === 0) continue;
    for (let g = 0; g < 2; g++) {
      const w = wa[a] * (g === 0 ? 1 - t : t);
      if (w === 0) continue;
      const m = p.means[a][g], o = (a * 2 + g) * F * R;
      for (let f = 0; f < F; f++) mean[f] += w * m[f];
      for (let i = 0; i < F * R; i++) factor[i] += w * face.prior[o + i];
    }
  }
  return { mean, factor };
}

// a face drawn from the distribution, clipped to the slider ranges (random: uniform numbers in [0, 1))
export function sampleFace(face: FaceData, values: Record<string, number>, random: () => number = Math.random): Float64Array {
  const { mean, factor } = facePriorParameters(face, values);
  const F = mean.length, R = face.tables.prior.rank;
  const z = new Float64Array(R);
  for (let k = 0; k < R; k += 2) {
    const u = Math.max(random(), 1e-12), v = random();
    const r = Math.sqrt(-2 * Math.log(u));
    z[k] = r * Math.cos(2 * Math.PI * v);
    if (k + 1 < R) z[k + 1] = r * Math.sin(2 * Math.PI * v);
  }
  const out = new Float64Array(F);
  for (let f = 0; f < F; f++) {
    let s = mean[f];
    for (let k = 0; k < R; k++) s += factor[f * R + k] * z[k];
    const [lo, hi] = face.tables.ranges[f];
    out[f] = Math.min(hi, Math.max(lo, s));
  }
  return out;
}
