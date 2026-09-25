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
