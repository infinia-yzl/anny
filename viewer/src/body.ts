// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The body of the page for any setting of anny's sliders (no three.js here):
//   anny's coefficients -> coarse body and bone heads (anny_shape.ts) -> the floor at the same level ->
//   the mixed Catmull-Clark subdivision (subdivision.ts) -> the detail layers of anny.viewer.geometry, scaled
//   with the local size of the body -> normals. The eyes follow the sphere of anny's eye vertices, and the hair,
//   the brows and the lashes follow the triangles under their roots (anny.hair.StrandBinding).

import { coefficients, makeRule, ShapeSpace, type CoefficientRule } from './anny_shape.ts';
import { FineSubdivision } from './subdivision.ts';

export interface StrandSet {
  counts: Uint8Array;        // points per strand
  total: number;
  corners: Uint32Array;      // 3 fine vertices per strand root
  bary: Float32Array;        // 3 barycentric coordinates per strand root
  local: Float32Array;       // points in the frame of the root triangle, divided by the scalp size (total * 3)
  P: Float32Array;           // current points (total * 3)
}

export function vertexNormals(P: Float32Array, I: Uint32Array, out?: Float32Array): Float32Array {
  const N = out || new Float32Array(P.length);
  N.fill(0);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
    const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    N[a] += nx; N[a + 1] += ny; N[a + 2] += nz; N[b] += nx; N[b + 1] += ny; N[b + 2] += nz; N[c] += nx; N[c + 1] += ny; N[c + 2] += nz;
  }
  for (let o = 0; o < N.length; o += 3) {
    const l = Math.hypot(N[o], N[o + 1], N[o + 2]) || 1;
    N[o] /= l; N[o + 1] /= l; N[o + 2] /= l;
  }
  return N;
}

// the first tangent of anny.viewer.geometry.tangent_frames: the sum of the triangle edges leaving a vertex,
// without its normal part
export function vertexTangents(P: Float32Array, I: Uint32Array, N: Float32Array, out?: Float32Array): Float32Array {
  const E = out || new Float32Array(P.length);
  E.fill(0);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    for (let k = 0; k < 3; k++) {
      E[a + k] += P[b + k] - P[a + k];
      E[b + k] += P[c + k] - P[b + k];
      E[c + k] += P[a + k] - P[c + k];
    }
  }
  for (let o = 0; o < E.length; o += 3) {
    const d = E[o] * N[o] + E[o + 1] * N[o + 1] + E[o + 2] * N[o + 2];
    let x = E[o] - d * N[o], y = E[o + 1] - d * N[o + 1], z = E[o + 2] - d * N[o + 2];
    const l = Math.hypot(x, y, z) || 1;
    E[o] = x / l; E[o + 1] = y / l; E[o + 2] = z / l;
  }
  return E;
}

function sphereFit(V: Float32Array, a: number, b: number): [number[], number] {
  // least squares of |p|^2 = 2 c.p + d, as anny.viewer.geometry.sphere_fit
  const A = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], y = [0, 0, 0, 0];
  for (let i = a; i < b; i++) {
    const x = V[i * 3], yy = V[i * 3 + 1], z = V[i * 3 + 2];
    const r = [2 * x, 2 * yy, 2 * z, 1], s = x * x + yy * yy + z * z;
    for (let p = 0; p < 4; p++) { y[p] += r[p] * s; for (let q = 0; q < 4; q++) A[p][q] += r[p] * r[q]; }
  }
  // gaussian elimination
  for (let p = 0; p < 4; p++) {
    let m = p; for (let q = p + 1; q < 4; q++) if (Math.abs(A[q][p]) > Math.abs(A[m][p])) m = q;
    [A[p], A[m]] = [A[m], A[p]]; [y[p], y[m]] = [y[m], y[p]];
    for (let q = p + 1; q < 4; q++) { const f = A[q][p] / A[p][p]; for (let k = p; k < 4; k++) A[q][k] -= f * A[p][k]; y[q] -= f * y[p]; }
  }
  const c = [0, 0, 0, 0];
  for (let p = 3; p >= 0; p--) { let s = y[p]; for (let k = p + 1; k < 4; k++) s -= A[p][k] * c[k]; c[p] = s / A[p][p]; }
  const center = [c[0], c[1], c[2]];
  return [center, Math.sqrt(c[3] + center[0] ** 2 + center[1] ** 2 + center[2] ** 2)];
}

function triFrame(P: Float32Array, a: number, b: number, c: number, out: Float64Array) {
  // rows: edge direction, in-plane normal, face normal (anny.hair.triangle_frames)
  const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
  let e1x = P[b * 3] - ax, e1y = P[b * 3 + 1] - ay, e1z = P[b * 3 + 2] - az;
  const e2x = P[c * 3] - ax, e2y = P[c * 3 + 1] - ay, e2z = P[c * 3 + 2] - az;
  let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
  let l = Math.hypot(e1x, e1y, e1z) || 1; e1x /= l; e1y /= l; e1z /= l;
  l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
  out[0] = e1x; out[1] = e1y; out[2] = e1z;
  out[3] = ny * e1z - nz * e1y; out[4] = nz * e1x - nx * e1z; out[5] = nx * e1y - ny * e1x;
  out[6] = nx; out[7] = ny; out[8] = nz;
}

export class AnnyBody {
  meta: any;
  rule: CoefficientRule;
  space: ShapeSpace;
  sub: FineSubdivision;
  nBody: number; nc: number; nFine: number;
  quads: Uint32Array; sizeRef: Float32Array; detail: Int16Array; detailStep: number;
  index: Uint32Array;
  rest0: Float32Array;       // fine positions of anny's default body (the 'rest' attribute)
  N0: Float32Array; S0: Float32Array;
  coarse: Float32Array; joints: Float32Array; coarseSkin: Uint8Array;
  pos: Float32Array; nrm: Float32Array; ns: Float32Array; smooth: Float32Array; tan: Float32Array; sn: Float32Array;
  floor: number; eyeRange: Record<string, number[]>; eyeSphere0: Record<string, [number[], number]>; eyeOffset: number[];
  strands: Record<string, StrandSet> = {};
  scalpRef = 1; scalp = 1;
  eyes: Record<string, { center: number[]; scale: number }> = {};
  values: Record<string, number> = {};
  jointsDefault: Float32Array;
  constructor(meta: any, bufs: {
    template: Float32Array; components: Int16Array; projection: Float32Array; jointTemplate: Float32Array; jointBlend: Float32Array;
    quads: Uint32Array; rows: Uint32Array; sizeRef: Float32Array; detail: Int16Array; index: Uint32Array; rest: Float32Array; nsmooth: Float32Array;
    coarseSkin: Uint8Array;
  }, positions?: Float32Array, normals?: Float32Array, nsmooth?: Float32Array) {
    const sm = meta.shape;
    this.meta = meta;
    this.rule = makeRule(sm.tables);
    this.space = new ShapeSpace({ template: bufs.template, components: bufs.components, componentScale: sm.component_scale, projection: bufs.projection,
      jointTemplate: bufs.jointTemplate, jointBlend: bufs.jointBlend });
    this.nBody = sm.body_vertices; this.nc = sm.coarse_vertices;
    this.quads = bufs.quads; this.sizeRef = bufs.sizeRef; this.detail = bufs.detail; this.detailStep = meta.detail_step || 1e-5;
    this.index = bufs.index; this.rest0 = bufs.rest; this.coarseSkin = bufs.coarseSkin;
    this.sub = new FineSubdivision(this.nBody, bufs.quads, sm.base_level, bufs.rows);
    this.nFine = bufs.rows.length;
    this.floor = sm.frame.floor;
    this.eyeRange = sm.eye_vertices; this.eyeSphere0 = sm.default_eye_sphere; this.eyeOffset = sm.eye_offset;
    this.pos = positions || new Float32Array(this.nFine * 3);
    this.nrm = normals || new Float32Array(this.nFine * 3);
    this.ns = nsmooth || new Float32Array(this.nFine * 3);
    this.smooth = new Float32Array(this.nFine * 3); this.tan = new Float32Array(this.nFine * 3); this.sn = new Float32Array(this.nFine * 3);
    this.N0 = vertexNormals(this.rest0, this.index);
    this.S0 = bufs.nsmooth;
    this.coarse = new Float32Array(this.nc * 3); this.joints = new Float32Array(this.space.nb * 3);
    this.jointsDefault = this.space.joints(coefficients(this.rule, {}));
  }

  // anny's slider values (missing sliders take anny's default, 0.5)
  update(values: Record<string, number>): { ms: number } {
    const t0 = performance.now();
    this.values = Object.assign({}, values);
    const c = coefficients(this.rule, values);
    const V = this.space.coarse(c, this.coarse), J = this.space.joints(c, this.joints);
    // the lowest point of the body stands on the floor
    let minY = Infinity;
    for (let i = 0; i < this.nBody; i++) minY = Math.min(minY, V[i * 3 + 1]);
    const dy = this.floor - minY;
    for (let i = 1; i < V.length; i += 3) V[i] += dy;
    for (let i = 1; i < J.length; i += 3) J[i] += dy;
    // the fine surface, and the size of the body around each vertex relative to anny's default body
    const body = V.subarray(0, this.nBody * 3);
    const S = this.sub.apply(body, 3);
    this.smooth.set(S);
    const ratio = this.ringRatio(V);
    const r = this.sub.apply(ratio, 1);
    // detail layers in the frames of the smooth surface
    vertexNormals(S, this.index, this.sn);
    vertexTangents(S, this.index, this.sn, this.tan);
    const P = this.pos, N = this.sn, T = this.tan, D = this.detail, st = this.detailStep;
    for (let v = 0, o = 0; v < this.nFine; v++, o += 3) {
      const k = r[v] * st, a = D[o] * k, b = D[o + 1] * k, n = D[o + 2] * k;
      // t2 = n x t1
      const t2x = N[o + 1] * T[o + 2] - N[o + 2] * T[o + 1], t2y = N[o + 2] * T[o] - N[o] * T[o + 2], t2z = N[o] * T[o + 1] - N[o + 1] * T[o];
      P[o] = S[o] + a * T[o] + b * t2x + n * N[o];
      P[o + 1] = S[o + 1] + a * T[o + 1] + b * t2y + n * N[o + 1];
      P[o + 2] = S[o + 2] + a * T[o + 2] + b * t2z + n * N[o + 2];
    }
    vertexNormals(P, this.index, this.nrm);
    const N1 = this.nrm, N0 = this.N0, S0 = this.S0, NS = this.ns;
    for (let o = 0; o < NS.length; o += 3) {
      const x = S0[o] + N1[o] - N0[o], y = S0[o + 1] + N1[o + 1] - N0[o + 1], z = S0[o + 2] + N1[o + 2] - N0[o + 2];
      const l = Math.hypot(x, y, z) || 1;
      NS[o] = x / l; NS[o + 1] = y / l; NS[o + 2] = z / l;
    }
    // eyes: the sphere of anny's eye vertices, with the offset of the legacy eyeball
    for (const s of ['l', 'r']) {
      const [a, b] = this.eyeRange[s];
      const [c1, r1] = sphereFit(V, a, b);
      const r0 = this.eyeSphere0[s][1], k = r1 / r0, sg = c1[0] > 0 ? 1 : -1;
      this.eyes[s] = { center: [c1[0] + this.eyeOffset[0] * sg * k, c1[1] + this.eyeOffset[1] * k, c1[2] + this.eyeOffset[2] * k], scale: k };
    }
    this.followStrands();
    return { ms: performance.now() - t0 };
  }

  // mean edge length around each coarse vertex, relative to anny's default body
  ringRatio(V: Float32Array): Float32Array {
    const n = this.nBody, sum = new Float32Array(n), cnt = new Float32Array(n), Q = this.quads;
    for (let f = 0; f < Q.length; f += 4) for (let k = 0; k < 4; k++) {
      const a = Q[f + k], b = Q[f + ((k + 1) & 3)];
      if (a > b) continue;   // every inner edge appears twice with both orders; the border edges of this body are none
      const l = Math.hypot(V[a * 3] - V[b * 3], V[a * 3 + 1] - V[b * 3 + 1], V[a * 3 + 2] - V[b * 3 + 2]);
      sum[a] += l; sum[b] += l; cnt[a]++; cnt[b]++;
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = (sum[i] / Math.max(cnt[i], 1)) / this.sizeRef[i];
    return out;
  }

  // bind strands (points of anny's default body) to the fine triangles under their roots
  bindStrands(name: string, P: Float32Array, counts: Uint8Array, corners: Uint32Array, bary: Float32Array) {
    const nS = counts.length, R = this.rest0, F = new Float64Array(9);
    const roots = new Float64Array(nS * 3);
    for (let i = 0; i < nS; i++) {
      const a = corners[i * 3], b = corners[i * 3 + 1], c = corners[i * 3 + 2], u = bary[i * 3], v = bary[i * 3 + 1], w = bary[i * 3 + 2];
      for (let k = 0; k < 3; k++) roots[i * 3 + k] = u * R[a * 3 + k] + v * R[b * 3 + k] + w * R[c * 3 + k];
    }
    if (name === 'hair') this.scalpRef = rmsSize(roots);
    const local = new Float32Array(P.length);
    let p = 0;
    for (let i = 0; i < nS; i++) {
      triFrame(R, corners[i * 3], corners[i * 3 + 1], corners[i * 3 + 2], F);
      for (let j = 0; j < counts[i]; j++, p++) {
        const dx = P[p * 3] - roots[i * 3], dy = P[p * 3 + 1] - roots[i * 3 + 1], dz = P[p * 3 + 2] - roots[i * 3 + 2];
        for (let k = 0; k < 3; k++) local[p * 3 + k] = (F[k * 3] * dx + F[k * 3 + 1] * dy + F[k * 3 + 2] * dz) / this.scalpRef;
      }
    }
    this.strands[name] = { counts, total: P.length / 3, corners, bary, local, P: new Float32Array(P) };
  }

  followStrands() {
    const P = this.pos, F = new Float64Array(9);
    const hair = this.strands.hair;
    if (!hair) return;
    const roots = (s: StrandSet) => {
      const nS = s.counts.length, out = new Float64Array(nS * 3);
      for (let i = 0; i < nS; i++) {
        const a = s.corners[i * 3], b = s.corners[i * 3 + 1], c = s.corners[i * 3 + 2], u = s.bary[i * 3], v = s.bary[i * 3 + 1], w = s.bary[i * 3 + 2];
        for (let k = 0; k < 3; k++) out[i * 3 + k] = u * P[a * 3 + k] + v * P[b * 3 + k] + w * P[c * 3 + k];
      }
      return out;
    };
    const hr = roots(hair);
    this.scalp = rmsSize(hr);
    for (const s of Object.values(this.strands)) {
      const R = s === hair ? hr : roots(s), nS = s.counts.length;
      let p = 0;
      for (let i = 0; i < nS; i++) {
        triFrame(P, s.corners[i * 3], s.corners[i * 3 + 1], s.corners[i * 3 + 2], F);
        for (let j = 0; j < s.counts[i]; j++, p++) {
          const x = s.local[p * 3] * this.scalp, y = s.local[p * 3 + 1] * this.scalp, z = s.local[p * 3 + 2] * this.scalp;
          for (let k = 0; k < 3; k++) s.P[p * 3 + k] = R[i * 3 + k] + F[k] * x + F[3 + k] * y + F[6 + k] * z;
        }
      }
    }
  }

  // how much larger the head is than on anny's default body
  headScale(): number { return this.scalp / this.scalpRef; }

  // the coarse body skinned by bone matrices (column-major 4x4 per bone, rest heads removed): its lowest point
  lowestSkinned(boneMats: Float32Array, only?: Uint32Array): number[] {
    const V = this.coarse, K = this.coarseSkin, n = only ? only.length : this.nBody;
    let best = [0, Infinity, 0];
    for (let q = 0; q < n; q++) {
      const i = only ? only[q] : q;
      let x = 0, y = 0, z = 0;
      for (let k = 0; k < 8; k++) {
        const w = K[i * 16 + 8 + k] / 255;
        if (w === 0) continue;
        const m = K[i * 16 + k] * 16, px = V[i * 3], py = V[i * 3 + 1], pz = V[i * 3 + 2];
        x += w * (boneMats[m] * px + boneMats[m + 4] * py + boneMats[m + 8] * pz + boneMats[m + 12]);
        y += w * (boneMats[m + 1] * px + boneMats[m + 5] * py + boneMats[m + 9] * pz + boneMats[m + 13]);
        z += w * (boneMats[m + 2] * px + boneMats[m + 6] * py + boneMats[m + 10] * pz + boneMats[m + 14]);
      }
      if (y < best[1]) best = [x, y, z];
    }
    return best;
  }
}

function rmsSize(P: Float64Array): number {
  const n = P.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += P[i * 3]; cy += P[i * 3 + 1]; cz += P[i * 3 + 2]; }
  cx /= n; cy /= n; cz /= n;
  let s = 0;
  for (let i = 0; i < n; i++) s += (P[i * 3] - cx) ** 2 + (P[i * 3 + 1] - cy) ** 2 + (P[i * 3 + 2] - cz) ** 2;
  return Math.sqrt(s / n);
}

export { coefficients, makeRule };
