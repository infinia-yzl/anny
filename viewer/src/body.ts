// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The body of the page for any setting of anny's sliders (no three.js here):
//   anny's coefficients -> coarse body and bone heads (anny_shape.ts) -> the floor at the same level ->
//   the mixed Catmull-Clark subdivision (subdivision.ts) -> frames of the smooth surface from four neighbours of
//   each vertex -> the detail layers of anny.viewer.geometry, scaled with the local size of the body, with the
//   normals kept in the same frames. The eyes follow the sphere of anny's eye vertices. The hair, the brows and
//   the lashes follow the triangles under their roots (anny.hair.StrandBinding): followStrands gives one frame
//   per strand, and the page's hair shader places the points.

import { coefficients, makeRule, ShapeSpace, type CoefficientRule } from './anny_shape.ts';
import { FineSubdivision } from './subdivision.ts';

export interface StrandSet {
  counts: Uint8Array;        // points per strand
  total: number;
  corners: Uint32Array;      // 3 fine vertices per strand root
  bary: Float32Array;        // 3 barycentric coordinates per strand root
  local: Float32Array;       // points in the frame of the root triangle, divided by the scalp size (total * 3)
  frames: Float32Array;      // per strand on the current body, rows k = 0..2 of [scalp * (t1, t2, n) | root] (nS * 12)
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
    const l = Math.sqrt(N[o] * N[o] + N[o + 1] * N[o + 1] + N[o + 2] * N[o + 2]) || 1;
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
    const ax = P[a], ay = P[a + 1], az = P[a + 2], bx = P[b], by = P[b + 1], bz = P[b + 2], cx = P[c], cy = P[c + 1], cz = P[c + 2];
    E[a] += bx - ax; E[a + 1] += by - ay; E[a + 2] += bz - az;
    E[b] += cx - bx; E[b + 1] += cy - by; E[b + 2] += cz - bz;
    E[c] += ax - cx; E[c + 1] += ay - cy; E[c + 2] += az - cz;
  }
  for (let o = 0; o < E.length; o += 3) {
    const d = E[o] * N[o] + E[o + 1] * N[o + 1] + E[o + 2] * N[o + 2];
    const x = E[o] - d * N[o], y = E[o + 1] - d * N[o + 1], z = E[o + 2] - d * N[o + 2];
    const l = Math.sqrt(x * x + y * y + z * z) || 1;
    E[o] = x / l; E[o + 1] = y / l; E[o + 2] = z / l;
  }
  return E;
}

// four neighbours around each vertex, a quarter turn apart: the edge neighbours of the faces around the vertex,
// sorted by their angle in the frame (t1, n x t1) of the vertex
export function ringNeighbours(I: Uint32Array, P: Float32Array, N: Float32Array, T: Float32Array, nVertices: number): Uint32Array {
  const start = new Uint32Array(nVertices + 1);
  for (let i = 0; i < I.length; i++) start[I[i] + 1] += 2;
  for (let v = 0; v < nVertices; v++) start[v + 1] += start[v];
  const fill = start.slice(0, nVertices), nb = new Uint32Array(I.length * 2);
  for (let t = 0; t < I.length; t += 3) for (let k = 0; k < 3; k++) {
    const v = I[t + k];
    nb[fill[v]++] = I[t + (k + 1) % 3]; nb[fill[v]++] = I[t + (k + 2) % 3];
  }
  const out = new Uint32Array(nVertices * 4), ids = new Uint32Array(256), ang = new Float64Array(256);
  for (let v = 0; v < nVertices; v++) {
    const o = v * 3, t2x = N[o + 1] * T[o + 2] - N[o + 2] * T[o + 1], t2y = N[o + 2] * T[o] - N[o] * T[o + 2], t2z = N[o] * T[o + 1] - N[o + 1] * T[o];
    let m = 0;
    for (let q = start[v]; q < start[v + 1]; q++) {
      const u = nb[q];
      let seen = false;
      for (let i = 0; i < m; i++) if (ids[i] === u) { seen = true; break; }
      if (seen) continue;
      const dx = P[u * 3] - P[o], dy = P[u * 3 + 1] - P[o + 1], dz = P[u * 3 + 2] - P[o + 2];
      const a = Math.atan2(dx * t2x + dy * t2y + dz * t2z, dx * T[o] + dy * T[o + 1] + dz * T[o + 2]);
      // insertion by angle
      let j = m - 1;
      while (j >= 0 && ang[j] > a) { ang[j + 1] = ang[j]; ids[j + 1] = ids[j]; j--; }
      ang[j + 1] = a; ids[j + 1] = u; m++;
    }
    for (let k = 0; k < 4; k++) out[v * 4 + k] = ids[Math.round(k * m / 4) % m];
  }
  return out;
}

// frames from the four neighbours: n along (p0 - p2) x (p1 - p3), t1 along p0 - p2; returns in `area` the length of
// that cross product, which grows with the square of the local size
export function ringFrames(P: Float32Array, R: Uint32Array, N: Float32Array, T: Float32Array, area: Float32Array) {
  const n = R.length / 4;
  for (let v = 0; v < n; v++) {
    const a = R[v * 4] * 3, b = R[v * 4 + 1] * 3, c = R[v * 4 + 2] * 3, d = R[v * 4 + 3] * 3;
    const ux = P[a] - P[c], uy = P[a + 1] - P[c + 1], uz = P[a + 2] - P[c + 2];
    const wx = P[b] - P[d], wy = P[b + 1] - P[d + 1], wz = P[b + 2] - P[d + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const ln = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1e-30, lu = Math.sqrt(ux * ux + uy * uy + uz * uz) || 1e-30;
    const o = v * 3;
    N[o] = nx / ln; N[o + 1] = ny / ln; N[o + 2] = nz / ln;
    T[o] = ux / lu; T[o + 1] = uy / lu; T[o + 2] = uz / lu;
    area[v] = ln;
  }
}

// vertexNormals and vertexTangents in one loop over the triangles (the frames of anny.viewer.geometry)
export function vertexFrames(P: Float32Array, I: Uint32Array, N: Float32Array, E: Float32Array) {
  N.fill(0); E.fill(0);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ax = P[a], ay = P[a + 1], az = P[a + 2], bx = P[b], by = P[b + 1], bz = P[b + 2], cx = P[c], cy = P[c + 1], cz = P[c + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az, e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    N[a] += nx; N[a + 1] += ny; N[a + 2] += nz; N[b] += nx; N[b + 1] += ny; N[b + 2] += nz; N[c] += nx; N[c + 1] += ny; N[c + 2] += nz;
    E[a] += e1x; E[a + 1] += e1y; E[a + 2] += e1z;
    E[b] += cx - bx; E[b + 1] += cy - by; E[b + 2] += cz - bz;
    E[c] -= e2x; E[c + 1] -= e2y; E[c + 2] -= e2z;
  }
  for (let o = 0; o < N.length; o += 3) {
    let l = Math.sqrt(N[o] * N[o] + N[o + 1] * N[o + 1] + N[o + 2] * N[o + 2]) || 1;
    const nx = N[o] / l, ny = N[o + 1] / l, nz = N[o + 2] / l;
    N[o] = nx; N[o + 1] = ny; N[o + 2] = nz;
    const d = E[o] * nx + E[o + 1] * ny + E[o + 2] * nz;
    const x = E[o] - d * nx, y = E[o + 1] - d * ny, z = E[o + 2] - d * nz;
    l = Math.sqrt(x * x + y * y + z * z) || 1;
    E[o] = x / l; E[o + 1] = y / l; E[o + 2] = z / l;
  }
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
  quads: Uint32Array; detail: Int16Array; detailStep: number;
  index: Uint32Array;
  rest0: Float32Array;       // fine positions of anny's default body (the 'rest' attribute)
  ring: Uint32Array;            // four neighbours per fine vertex (ringNeighbours)
  area: Float32Array; area0: Float32Array;   // ringFrames' area now and on anny's default body
  localDetail: Float32Array;    // per fine vertex: the detail layers in the frame (t1, t2, n), per unit of local size
  localNormals: Float32Array;   // per fine vertex: the full normal and the smooth shading normal in the frame (t1, t2, n)
  coarse: Float32Array; joints: Float32Array; coarseSkin: Uint8Array;
  pos: Float32Array; nrm: Float32Array; ns: Float32Array; smooth: Float32Array; tan: Float32Array; sn: Float32Array;
  floor: number; eyeRange: Record<string, number[]>; eyeSphere0: Record<string, [number[], number]>; eyeOffset: number[];
  strands: Record<string, StrandSet> = {};
  scalpRef = 1; scalp = 1;
  eyes: Record<string, { center: number[]; scale: number }> = {};
  values: Record<string, number> = {};
  timing: Record<string, number> = {};   // ms of each step of the last update
  jointsDefault: Float32Array;
  constructor(meta: any, bufs: {
    template: Float32Array; components: Int16Array; projection: Float32Array; jointTemplate: Float32Array; jointBlend: Float32Array;
    quads: Uint32Array; rows: Uint32Array; detail: Int16Array; index: Uint32Array; rest: Float32Array; nsmooth: Float32Array;
    coarseSkin: Uint8Array;
  }, positions?: Float32Array, normals?: Float32Array, nsmooth?: Float32Array) {
    const sm = meta.shape;
    this.meta = meta;
    this.rule = makeRule(sm.tables);
    this.space = new ShapeSpace({ template: bufs.template, components: bufs.components, componentScale: sm.component_scale, projection: bufs.projection,
      jointTemplate: bufs.jointTemplate, jointBlend: bufs.jointBlend });
    this.nBody = sm.body_vertices; this.nc = sm.coarse_vertices;
    this.quads = bufs.quads; this.detail = bufs.detail; this.detailStep = meta.detail_step || 1e-5;
    this.index = bufs.index; this.rest0 = bufs.rest; this.coarseSkin = bufs.coarseSkin;
    this.sub = new FineSubdivision(this.nBody, bufs.quads, sm.base_level, bufs.rows);
    this.nFine = bufs.rows.length;
    this.floor = sm.frame.floor;
    this.eyeRange = sm.eye_vertices; this.eyeSphere0 = sm.default_eye_sphere; this.eyeOffset = sm.eye_offset;
    this.pos = positions || new Float32Array(this.nFine * 3);
    this.nrm = normals || new Float32Array(this.nFine * 3);
    this.ns = nsmooth || new Float32Array(this.nFine * 3);
    this.smooth = new Float32Array(this.nFine * 3); this.tan = new Float32Array(this.nFine * 3); this.sn = new Float32Array(this.nFine * 3);
    this.area = new Float32Array(this.nFine);
    this.coarse = new Float32Array(this.nc * 3); this.joints = new Float32Array(this.space.nb * 3);
    this.jointsDefault = this.space.joints(coefficients(this.rule, {}));
    // anny's default body. The detail layers of anny.viewer sit in the frames of anny.viewer.geometry (normals and
    // tangents summed over the triangles). The page keeps them in the cheaper frames of ringFrames, together with
    // the normals of the full surface and the smooth shading normals. The detail scales with the size around each
    // vertex, so these normals keep their place in the frame on every body, and an update needs no pass over the
    // triangles.
    this.smoothSurface({});
    const S = this.smooth, D = this.detail, st = this.detailStep, nF = this.nFine;
    const Nt = new Float32Array(nF * 3), Tt = new Float32Array(nF * 3);
    vertexFrames(S, this.index, Nt, Tt);
    const P = this.pos, W = new Float32Array(nF * 3);   // W: the detail in world axes
    for (let o = 0; o < nF * 3; o += 3) {
      const t2x = Nt[o + 1] * Tt[o + 2] - Nt[o + 2] * Tt[o + 1], t2y = Nt[o + 2] * Tt[o] - Nt[o] * Tt[o + 2], t2z = Nt[o] * Tt[o + 1] - Nt[o + 1] * Tt[o];
      const a = D[o] * st, b = D[o + 1] * st, n = D[o + 2] * st;
      W[o] = a * Tt[o] + b * t2x + n * Nt[o];
      W[o + 1] = a * Tt[o + 1] + b * t2y + n * Nt[o + 1];
      W[o + 2] = a * Tt[o + 2] + b * t2z + n * Nt[o + 2];
      P[o] = S[o] + W[o]; P[o + 1] = S[o + 1] + W[o + 1]; P[o + 2] = S[o + 2] + W[o + 2];
    }
    this.ring = ringNeighbours(this.index, S, Nt, Tt, nF);
    ringFrames(S, this.ring, this.sn, this.tan, this.area);
    this.area0 = this.area.slice();
    const N = this.sn, T = this.tan, N1 = vertexNormals(P, this.index), NS = bufs.nsmooth;
    this.localDetail = new Float32Array(nF * 3); this.localNormals = new Float32Array(nF * 6);
    const toLocal = (src: Float32Array, o: number, out: Float32Array, q: number, t2x: number, t2y: number, t2z: number) => {
      out[q] = src[o] * T[o] + src[o + 1] * T[o + 1] + src[o + 2] * T[o + 2];
      out[q + 1] = src[o] * t2x + src[o + 1] * t2y + src[o + 2] * t2z;
      out[q + 2] = src[o] * N[o] + src[o + 1] * N[o + 1] + src[o + 2] * N[o + 2];
    };
    for (let o = 0, v = 0; v < nF; v++, o += 3) {
      const t2x = N[o + 1] * T[o + 2] - N[o + 2] * T[o + 1], t2y = N[o + 2] * T[o] - N[o] * T[o + 2], t2z = N[o] * T[o + 1] - N[o + 1] * T[o];
      toLocal(W, o, this.localDetail, o, t2x, t2y, t2z);
      toLocal(N1, o, this.localNormals, v * 6, t2x, t2y, t2z);
      toLocal(NS, o, this.localNormals, v * 6 + 3, t2x, t2y, t2z);
    }
  }

  // the smooth surface: the subdivision of the coarse body, standing on the floor
  private smoothSurface(values: Record<string, number>, marks?: [string, number][]) {
    const c = coefficients(this.rule, values);
    const V = this.space.coarse(c, this.coarse), J = this.space.joints(c, this.joints);
    marks?.push(['shape', performance.now()]);
    let minY = Infinity;
    for (let i = 0; i < this.nBody; i++) minY = Math.min(minY, V[i * 3 + 1]);
    const dy = this.floor - minY;
    for (let i = 1; i < V.length; i += 3) V[i] += dy;
    for (let i = 1; i < J.length; i += 3) J[i] += dy;
    this.smooth.set(this.sub.apply(V.subarray(0, this.nBody * 3), 3));
    marks?.push(['subdivision', performance.now()]);
  }

  // anny's slider values (missing sliders take anny's default, 0.5)
  update(values: Record<string, number>): { ms: number } {
    const t0 = performance.now(), marks: [string, number][] = [];
    this.values = Object.assign({}, values);
    this.smoothSurface(values, marks);
    ringFrames(this.smooth, this.ring, this.sn, this.tan, this.area);
    // the detail layers (scaled with the size around the vertex), the normals and the smooth shading normals in the
    // frames of the smooth surface
    const P = this.pos, S = this.smooth, N = this.sn, T = this.tan, D = this.localDetail, A = this.area, A0 = this.area0;
    const L = this.localNormals, N1 = this.nrm, NS = this.ns;
    for (let o = 0, v = 0; v < this.nFine; v++, o += 3) {
      const tx = T[o], ty = T[o + 1], tz = T[o + 2], nx = N[o], ny = N[o + 1], nz = N[o + 2];
      // t2 = n x t1
      const t2x = ny * tz - nz * ty, t2y = nz * tx - nx * tz, t2z = nx * ty - ny * tx;
      const k = Math.sqrt(A[v] / A0[v]), a = D[o] * k, b = D[o + 1] * k, n = D[o + 2] * k;
      P[o] = S[o] + a * tx + b * t2x + n * nx;
      P[o + 1] = S[o + 1] + a * ty + b * t2y + n * ny;
      P[o + 2] = S[o + 2] + a * tz + b * t2z + n * nz;
      const q = v * 6;
      N1[o] = L[q] * tx + L[q + 1] * t2x + L[q + 2] * nx;
      N1[o + 1] = L[q] * ty + L[q + 1] * t2y + L[q + 2] * ny;
      N1[o + 2] = L[q] * tz + L[q + 1] * t2z + L[q + 2] * nz;
      NS[o] = L[q + 3] * tx + L[q + 4] * t2x + L[q + 5] * nx;
      NS[o + 1] = L[q + 3] * ty + L[q + 4] * t2y + L[q + 5] * ny;
      NS[o + 2] = L[q + 3] * tz + L[q + 4] * t2z + L[q + 5] * nz;
    }
    marks.push(['frames and detail', performance.now()]);
    const V = this.coarse;
    // eyes: the sphere of anny's eye vertices, with the offset of the legacy eyeball
    for (const s of ['l', 'r']) {
      const [a, b] = this.eyeRange[s];
      const [c1, r1] = sphereFit(V, a, b);
      const r0 = this.eyeSphere0[s][1], k = r1 / r0, sg = c1[0] > 0 ? 1 : -1;
      this.eyes[s] = { center: [c1[0] + this.eyeOffset[0] * sg * k, c1[1] + this.eyeOffset[1] * k, c1[2] + this.eyeOffset[2] * k], scale: k };
    }
    marks.push(['eyes', performance.now()]);
    this.followStrands();
    marks.push(['strands', performance.now()]);
    let last = t0;
    this.timing = Object.fromEntries(marks.map(([k, t]) => { const d = t - last; last = t; return [k, Math.round(d * 10) / 10]; }));
    return { ms: performance.now() - t0 };
  }

  // bind strands (points of anny's default body) to the fine triangles under their roots
  // (frames: optional storage for StrandSet.frames, such as the data of a texture)
  bindStrands(name: string, P: Float32Array, counts: Uint8Array, corners: Uint32Array, bary: Float32Array, frames?: Float32Array) {
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
    this.strands[name] = { counts, total: P.length / 3, corners, bary, local, frames: frames || new Float32Array(nS * 12) };
  }

  // the frame of each strand on the current body: a strand point is root + scalp * (t1 x + t2 y + n z) for its local
  // coordinates (x, y, z), with the frame (t1, t2, n) of the triangle under the root (anny.hair.StrandBinding)
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
    const k = this.scalp;
    for (const s of Object.values(this.strands)) {
      const R = s === hair ? hr : roots(s), nS = s.counts.length, M = s.frames;
      for (let i = 0; i < nS; i++) {
        triFrame(P, s.corners[i * 3], s.corners[i * 3 + 1], s.corners[i * 3 + 2], F);
        for (let r = 0; r < 3; r++) {
          const o = i * 12 + r * 4;
          M[o] = F[r] * k; M[o + 1] = F[3 + r] * k; M[o + 2] = F[6 + r] * k; M[o + 3] = R[i * 3 + r];
        }
      }
    }
  }

  // the points of a strand set on the current body (the page moves them on the GPU with the frames)
  strandPoints(name: string): Float32Array {
    const s = this.strands[name], M = s.frames, L = s.local, out = new Float32Array(L.length);
    let p = 0;
    for (let i = 0; i < s.counts.length; i++) for (let j = 0; j < s.counts[i]; j++, p++) {
      const x = L[p * 3], y = L[p * 3 + 1], z = L[p * 3 + 2];
      for (let r = 0; r < 3; r++) { const o = i * 12 + r * 4; out[p * 3 + r] = M[o] * x + M[o + 1] * y + M[o + 2] * z + M[o + 3]; }
    }
    return out;
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
