// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The data of anny's hair for the page (no three.js here), as anny.hair.styles has it:
//   - the scalp layout: guide roots and render roots, both in progressive order, with the triangles under them;
//   - the styles: one curve per guide root, decoded from 8-bit octahedral segment directions (anny.hair.chart);
//   - the render roots on the current body (rest positions and normals), and the pose of each guide root;
//   - the density volume of a style: the occlusion of the hair by the hair outward of it, the density near the scalp
//     for the scalp tint, and the cover of the scalp by strands long enough to shade it (fades, short cuts)
//     (anny.hair.styles.density_volume).

export const TEX_W = 2048;

export type Bytes = (name: string) => Uint8Array;

export interface StyleSpec {
  name: string; label: string; family: string; points: number;
  render: any; physics: any; controls: any; mirror?: boolean;
}

export interface HairMeta {
  layout: { guides: number; roots: number; simulated: number; root_spacing: number; guide_spacing: number };
  styles: StyleSpec[];
  centre: number[];
  hairline: { phi: number[]; el: number[] };
  curve_phi: number[];
  similarity: number;
  sector_radius: number;
  volume: { step: number; sample: number; ray: number; k: number; near: number; cover_length: number[]; cover_floor: number };
}

export interface HairLayout {
  G: number; R: number; S: number;
  guideRoot: Float64Array;                          // G * 3, anny's default body
  guideCorners: Uint32Array; guideBary: Float32Array;   // G * 3: the triangle under each guide root
  mirror: Uint16Array; sim: Uint16Array; simWeights: Uint8Array;   // G, G * 3, G * 3
  guideSkin: Uint8Array;                            // G * 8: four bones and their weights (0..255)
  rootCorners: Uint32Array; rootBary: Float32Array;     // R * 3
  rootData: Uint32Array;                            // R * 4: guides (2 x 16 bits), weights (4 x 8 bits), chart (2 x 16 bits)
  rootGuides: Uint16Array; rootWeights: Uint8Array; rootChart: Float32Array;   // R * 4, R * 4, R * 2
}

export interface HairStyle {
  spec: StyleSpec; P: number;
  points: Float32Array;            // G * P * 3 on anny's default body
  seg: Float32Array;               // G: segment length on anny's default body
  length: Float32Array;            // G: default length
  flick: Float32Array;             // G
  pivot: Float32Array;             // G: arc length where the guide leaves the head (a tie, or where it hangs)
  group: Uint8Array;               // G
  tipOn: Uint8Array;               // G: the tip follows the skin under it
  tipCorners: Uint32Array; tipBary: Float32Array;   // G * 3
  mirrored: boolean;
}

// a copy of the bytes, aligned for typed views
function aligned(b: Uint8Array): ArrayBuffer { return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }

// records of 3 fine vertices (uint32) and 2 barycentric coordinates (float32): the triangles under roots
export function bindingRecords(b: Uint8Array, n: number) {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const corners = new Uint32Array(n * 3), bary = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) corners[i * 3 + k] = dv.getUint32(i * 20 + k * 4, true);
    const u = dv.getFloat32(i * 20 + 12, true), w = dv.getFloat32(i * 20 + 16, true);
    bary[i * 3] = u; bary[i * 3 + 1] = w; bary[i * 3 + 2] = 1 - u - w;
  }
  return { corners, bary };
}

export function decodeLayout(bytes: Bytes, meta: HairMeta): HairLayout {
  const G = meta.layout.guides, R = meta.layout.roots, S = meta.layout.simulated;
  const guideRoot = Float64Array.from(new Float32Array(aligned(bytes('hair_guide_root'))).subarray(0, G * 3));
  const gb = bindingRecords(bytes('hair_guide_bind'), G), rb = bindingRecords(bytes('hair_root_bind'), R);
  const info = new DataView(aligned(bytes('hair_guide_info')));
  const mirror = new Uint16Array(G), sim = new Uint16Array(G * 3), simWeights = new Uint8Array(G * 3);
  for (let g = 0; g < G; g++) {
    mirror[g] = info.getUint16(g * 12, true);
    for (let k = 0; k < 3; k++) { sim[g * 3 + k] = info.getUint16(g * 12 + 2 + k * 2, true); simWeights[g * 3 + k] = info.getUint8(g * 12 + 8 + k); }
  }
  const guideSkin = bytes('hair_guide_skin').slice(0, G * 8);
  const rootData = new Uint32Array(aligned(bytes('hair_root_data'))).slice(0, R * 4);
  const rd = new DataView(rootData.buffer);
  const rootGuides = new Uint16Array(R * 4), rootWeights = new Uint8Array(R * 4), rootChart = new Float32Array(R * 2);
  for (let r = 0; r < R; r++) {
    for (let k = 0; k < 4; k++) { rootGuides[r * 4 + k] = rd.getUint16(r * 16 + k * 2, true); rootWeights[r * 4 + k] = rd.getUint8(r * 16 + 8 + k); }
    rootChart[r * 2] = rd.getInt16(r * 16 + 12, true) * 0.01; rootChart[r * 2 + 1] = rd.getInt16(r * 16 + 14, true) * 0.01;
  }
  return { G, R, S, guideRoot, guideCorners: gb.corners, guideBary: gb.bary, mirror, sim, simWeights, guideSkin,
    rootCorners: rb.corners, rootBary: rb.bary, rootData, rootGuides, rootWeights, rootChart };
}

// octahedral decode of 8-bit codes (anny.hair.chart.oct_decode of code_to_oct)
function octDecode(cx: number, cy: number, out: number[]) {
  let x = cx / 127 - 1, y = cy / 127 - 1;
  const z = 1 - Math.abs(x) - Math.abs(y), t = Math.max(-z, 0);
  x += x >= 0 ? -t : t; y += y >= 0 ? -t : t;
  const l = Math.sqrt(x * x + y * y + z * z);
  out[0] = x / l; out[1] = y / l; out[2] = z / l;
}

export function decodeStyle(bytes: Bytes, layout: HairLayout, spec: StyleSpec): HairStyle {
  const G = layout.G, P = spec.points, name = spec.name;
  const codes = bytes(`hair_${name}_codes`);
  const rec = new DataView(aligned(bytes(`hair_${name}_guide`)));
  const seg = new Float32Array(G), length = new Float32Array(G), flick = new Float32Array(G), pivot = new Float32Array(G);
  const group = new Uint8Array(G), tipOn = new Uint8Array(G);
  for (let g = 0; g < G; g++) {
    seg[g] = rec.getFloat32(g * 20, true); length[g] = rec.getFloat32(g * 20 + 4, true); flick[g] = rec.getFloat32(g * 20 + 8, true);
    pivot[g] = rec.getFloat32(g * 20 + 12, true); group[g] = rec.getUint8(g * 20 + 16); tipOn[g] = rec.getUint8(g * 20 + 17);
  }
  const tip = bindingRecords(bytes(`hair_${name}_tip`), G);
  const points = new Float32Array(G * P * 3), d = [0, 0, 0];
  for (let g = 0; g < G; g++) {
    let x = layout.guideRoot[g * 3], y = layout.guideRoot[g * 3 + 1], z = layout.guideRoot[g * 3 + 2];
    const s = seg[g], o = g * P * 3;
    points[o] = x; points[o + 1] = y; points[o + 2] = z;
    for (let j = 1; j < P; j++) {
      const c = (g * (P - 1) + j - 1) * 2;
      octDecode(codes[c], codes[c + 1], d);
      x += s * d[0]; y += s * d[1]; z += s * d[2];
      points[o + j * 3] = x; points[o + j * 3 + 1] = y; points[o + j * 3 + 2] = z;
    }
  }
  return { spec, P, points, seg, length, flick, pivot, group, tipOn, tipCorners: tip.corners, tipBary: tip.bary, mirrored: false };
}

// the style on the other side: guide k takes the curve of guide mirror[k], reflected in x (anny.hair.styles.mirrored);
// the tips follow the root frames only, since the mirrored tip triangles are not in the data
export function mirrorStyle(s: HairStyle, layout: HairLayout): HairStyle {
  const G = layout.G, P = s.P, m = layout.mirror, points = new Float32Array(s.points.length);
  const pick = <T extends Float32Array | Uint8Array>(a: T): T => { const o = new (a.constructor as any)(G); for (let g = 0; g < G; g++) o[g] = a[m[g]]; return o; };
  for (let g = 0; g < G; g++) for (let j = 0; j < P; j++) {
    const a = (g * P + j) * 3, b = (m[g] * P + j) * 3;
    points[a] = -s.points[b]; points[a + 1] = s.points[b + 1]; points[a + 2] = s.points[b + 2];
  }
  return { ...s, points, seg: pick(s.seg), length: pick(s.length), flick: pick(s.flick), pivot: pick(s.pivot), group: pick(s.group),
    tipOn: new Uint8Array(G), mirrored: true };
}

// the length of each guide curve on anny's default body
export function available(s: HairStyle): Float32Array {
  const G = s.seg.length, out = new Float32Array(G);
  for (let g = 0; g < G; g++) {
    let L = 0;
    for (let j = 1; j < s.P; j++) {
      const a = (g * s.P + j - 1) * 3, b = a + 3;
      L += Math.hypot(s.points[b] - s.points[a], s.points[b + 1] - s.points[a + 1], s.points[b + 2] - s.points[a + 2]);
    }
    out[g] = L;
  }
  return out;
}

// ------------------------------------------------------------------ the render roots on the current body
// 12-bit octahedral coordinates of a unit vector, packed in an integer that a float32 holds exactly
export function packNormal(x: number, y: number, z: number): number {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  x /= s; y /= s; z /= s;
  let ox = x, oy = y;
  if (z < 0) { ox = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1); oy = (1 - Math.abs(x)) * (y >= 0 ? 1 : -1); }
  const qx = Math.round((ox * 0.5 + 0.5) * 4095), qy = Math.round((oy * 0.5 + 0.5) * 4095);
  return qx + qy * 4096;
}

// per render root: its rest position on the body P and the normal of the triangle under it (packed)
export function rootsOnBody(P: Float32Array, layout: HairLayout, out: Float32Array) {
  const C = layout.rootCorners, B = layout.rootBary;
  for (let r = 0; r < layout.R; r++) {
    const a = C[r * 3] * 3, b = C[r * 3 + 1] * 3, c = C[r * 3 + 2] * 3, u = B[r * 3], v = B[r * 3 + 1], w = B[r * 3 + 2];
    const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
    const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    out[r * 4] = u * P[a] + v * P[b] + w * P[c];
    out[r * 4 + 1] = u * P[a + 1] + v * P[b + 1] + w * P[c + 1];
    out[r * 4 + 2] = u * P[a + 2] + v * P[b + 2] + w * P[c + 2];
    out[r * 4 + 3] = packNormal(nx / l, ny / l, nz / l);
  }
}

// the pose of each guide root: the bone matrices (column-major 4 x 4, as three.js keeps them) blended by the skin
// under the root; out holds three rows [m_r0, m_r1, m_r2, t_r] per guide
export function guideMatrices(layout: HairLayout, bones: Float32Array, out: Float32Array) {
  const K = layout.guideSkin;
  for (let g = 0; g < layout.G; g++) {
    const o = g * 12;
    for (let q = 0; q < 12; q++) out[o + q] = 0;
    for (let k = 0; k < 4; k++) {
      const w = K[g * 8 + 4 + k] / 255;
      if (w === 0) continue;
      const m = K[g * 8 + k] * 16;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) out[o + r * 4 + c] += w * bones[m + c * 4 + r];
    }
  }
}

// ------------------------------------------------------------------ random values (anny.hair.styles.rnd)
function pcg(v: number): number {
  const state = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}
export function rnd(key: number, stream: number): number {
  const h = pcg((Math.imul(key >>> 0, 0x9E3779B9) ^ pcg(stream)) >>> 0);
  return (h >>> 8) / 16777216;
}

// ------------------------------------------------------------------ the style fields (anny.hair.chart)
const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
function interp(x: number, xs: number[], ys: ArrayLike<number>): number {
  if (x <= xs[0]) return ys[0];
  for (let k = 1; k < xs.length; k++) if (x <= xs[k]) { const f = (x - xs[k - 1]) / (xs[k] - xs[k - 1]); return ys[k - 1] + f * (ys[k] - ys[k - 1]); }
  return ys[xs.length - 1];
}
export function hairline(meta: HairMeta, phi: number, offset?: number[] | null): number {
  const a = Math.abs(phi);
  return interp(a, meta.hairline.phi, meta.hairline.el) + (offset ? interp(a, meta.curve_phi, offset) : 0);
}
export function coverage(meta: HairMeta, phi: number, el: number, offset?: number[] | null): number {
  return smooth(-0.8, 3.0, el - hairline(meta, phi, offset));
}
export function fadeLength(meta: HairMeta, phi: number, el: number, fade: any, shift: number, offset?: number[] | null): number {
  if (!fade) return Infinity;
  const h = el - hairline(meta, phi, offset), lo = interp(Math.abs(phi), meta.curve_phi, fade.start) + shift;
  const u = Math.min(30, Math.max(0, (h - lo) / fade.width));
  return fade.clipper * Math.pow(fade.top / fade.clipper, u);
}

// ------------------------------------------------------------------ the density volume (anny.hair.styles.density_volume)
export interface Volume { data: Uint8Array; dims: number[]; lo: number[]; h: number }

export function densityVolume(meta: HairMeta, layout: HairLayout, s: HairStyle, params: { length: number; fade: number }, count: number): Volume {
  const V = meta.volume, rs = s.spec.render, G = layout.G, P = s.P;
  // the drawn strands of each guide
  const nG = new Float64Array(G);
  for (let r = 0; r < Math.min(count, layout.R); r++) nG[layout.rootGuides[r * 4]] += coverage(meta, layout.rootChart[r * 2], layout.rootChart[r * 2 + 1], rs.hairline);
  const avail = available(s);
  // samples along the guides, weighted by their strands; the roots of the drawn strands for the cover
  const pts: number[] = [], wts: number[] = [], roots: number[] = [], nRoot: number[] = [], shade: number[] = [];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const rlo = [Infinity, Infinity, Infinity], rhi = [-Infinity, -Infinity, -Infinity];
  for (let g = 0; g < G; g++) {
    // the guide chart: the chart of its root on anny's default body
    const gx = layout.guideRoot[g * 3] - meta.centre[0], gy = layout.guideRoot[g * 3 + 1] - meta.centre[1], gz = layout.guideRoot[g * 3 + 2] - meta.centre[2];
    const phi = Math.atan2(gx, gz) * 180 / Math.PI, el = Math.asin(Math.max(-1, Math.min(1, gy / Math.hypot(gx, gy, gz)))) * 180 / Math.PI;
    let D = Math.min(s.length[g] * params.length, avail[g]);
    D = Math.min(D, fadeLength(meta, phi, el, rs.fade, params.fade, rs.hairline));
    if (nG[g] > 0) {
      for (let k = 0; k < 3; k++) { const x = layout.guideRoot[g * 3 + k]; roots.push(x); rlo[k] = Math.min(rlo[k], x); rhi[k] = Math.max(rhi[k], x); }
      nRoot.push(nG[g]); shade.push(smooth(V.cover_length[0], V.cover_length[1], D));
    }
    if (!(nG[g] > 0) || !(D > 0)) continue;
    const m = Math.ceil(D / V.sample), sg = avail[g] / (P - 1);
    for (let i = 0; i < m; i++) {
      const sArc = (i + 0.5) * D / m, u = sArc / sg;
      const i0 = Math.min(Math.max(Math.floor(u), 0), P - 2), f = u - i0, a = (g * P + i0) * 3;
      for (let k = 0; k < 3; k++) {
        const x = s.points[a + k] + f * (s.points[a + 3 + k] - s.points[a + k]);
        pts.push(x); lo[k] = Math.min(lo[k], x); hi[k] = Math.max(hi[k], x);
      }
      wts.push(nG[g] * (D / m) / V.sample);
    }
  }
  const h = V.step;
  if (!wts.length) return { data: new Uint8Array(2 * 2 * 2 * 4).map((_, i) => i % 4 ? 0 : 255), dims: [2, 2, 2], lo: meta.centre.map((c) => c - h), h };
  const l0 = lo.map((x, k) => Math.floor((Math.min(x, rlo[k]) - 0.01) / h) * h);
  const dims = hi.map((x, k) => Math.ceil((Math.max(x, rhi[k]) + 0.01 - l0[k]) / h) + 1);
  const [nx, ny, nz] = dims, n = nx * ny * nz;
  // x fastest, as WebGL reads a 3D texture
  const id = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const grid = splat(pts, wts, l0, h, dims);
  const num = splat(roots, nRoot.map((w, i) => w * shade[i]), l0, h, dims), den = splat(roots, nRoot, l0, h, dims);
  let dmax = 0;
  for (let o = 0; o < n; o++) dmax = Math.max(dmax, den[o]);
  // occlusion: the density walked outward from each voxel (nearest voxel at each step), the near density, the cover
  const data = new Uint8Array(n * 4);
  const c = meta.centre;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const px = l0[0] + x * h, py = l0[1] + y * h, pz = l0[2] + z * h;
    let dx = px - c[0], dy = py - c[1], dz = pz - c[2];
    const l = Math.hypot(dx, dy, dz) || 1; dx /= l; dy /= l; dz /= l;
    let acc = 0;
    for (let s2 = 1; s2 <= V.ray; s2++) {
      const qx = Math.floor(x + dx * s2 + 0.5), qy = Math.floor(y + dy * s2 + 0.5), qz = Math.floor(z + dz * s2 + 0.5);
      if (qx < 0 || qy < 0 || qz < 0 || qx >= nx || qy >= ny || qz >= nz) continue;
      acc += grid[id(qx, qy, qz)];
    }
    const o = id(x, y, z);
    data[o * 4] = Math.round(Math.exp(-V.k * acc) * 255);
    data[o * 4 + 1] = Math.round(Math.min(1, Math.max(0, grid[o] / V.near)) * 255);
    data[o * 4 + 2] = Math.round(Math.min(1, Math.max(0, num[o] / Math.max(den[o], V.cover_floor * dmax))) * 255);
  }
  return { data, dims, lo: l0, h };
}

// trilinear splat of weighted points (3 per point) on a grid (x fastest), then two passes of a [1, 2, 1] / 4 blur
// along each axis
function splat(pts: ArrayLike<number>, wts: ArrayLike<number>, l0: number[], h: number, dims: number[]): Float64Array {
  const [nx, ny, nz] = dims, n = nx * ny * nz;
  const id = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  let grid = new Float64Array(n);
  for (let q = 0; q < wts.length; q++) {
    const u = [0, 1, 2].map((k) => (pts[q * 3 + k] - l0[k]) / h);
    const i0 = u.map(Math.floor), f = u.map((x, k) => x - i0[k]);
    for (let dx = 0; dx < 2; dx++) for (let dy = 0; dy < 2; dy++) for (let dz = 0; dz < 2; dz++) {
      const w = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]);
      grid[id(i0[0] + dx, i0[1] + dy, i0[2] + dz)] += wts[q] * w;
    }
  }
  const tmp = new Float64Array(n), strides = [1, nx, nx * ny];
  for (let pass = 0; pass < 2; pass++) for (let ax = 0; ax < 3; ax++) {
    const st = strides[ax], len = dims[ax];
    for (let o = 0; o < n; o++) {
      const c = Math.floor(o / st) % len;
      tmp[o] = 0.5 * grid[o] + (c > 0 ? 0.25 * grid[o - st] : 0) + (c < len - 1 ? 0.25 * grid[o + st] : 0);
    }
    const sw = grid; grid = tmp.slice(); tmp.set(sw);
  }
  return grid;
}
