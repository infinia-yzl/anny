// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Catmull-Clark subdivision as sparse operators, built from the quad faces exactly as
// anny.utils.subdivision builds them (same vertex, edge and face order), so the fine body of the page
// matches anny.viewer's fine body vertex for vertex. The last level keeps only the rows the page draws.

export class Csr {
  rows: number;
  cols: number;
  start: Uint32Array;   // rows + 1
  index: Uint32Array;
  value: Float32Array;
  constructor(rows: number, cols: number, start: Uint32Array, index: Uint32Array, value: Float32Array) {
    this.rows = rows; this.cols = cols; this.start = start; this.index = index; this.value = value;
  }
  // y = A x for x with `ch` interleaved channels
  apply(x: Float32Array | Float64Array, ch: number, out?: Float32Array): Float32Array {
    const y = out || new Float32Array(this.rows * ch);
    const { start, index, value } = this;
    if (ch === 4) {
      for (let r = 0; r < this.rows; r++) {
        let a = 0, b = 0, c = 0, d = 0;
        for (let e = start[r]; e < start[r + 1]; e++) {
          const w = value[e], o = index[e] * 4;
          a += w * x[o]; b += w * x[o + 1]; c += w * x[o + 2]; d += w * x[o + 3];
        }
        const q = r * 4; y[q] = a; y[q + 1] = b; y[q + 2] = c; y[q + 3] = d;
      }
      return y;
    }
    if (ch === 3) {
      for (let r = 0; r < this.rows; r++) {
        let a = 0, b = 0, c = 0;
        for (let e = start[r]; e < start[r + 1]; e++) {
          const w = value[e], o = index[e] * 3;
          a += w * x[o]; b += w * x[o + 1]; c += w * x[o + 2];
        }
        const q = r * 3; y[q] = a; y[q + 1] = b; y[q + 2] = c;
      }
      return y;
    }
    for (let r = 0; r < this.rows; r++) {
      for (let k = 0; k < ch; k++) {
        let a = 0;
        for (let e = start[r]; e < start[r + 1]; e++) a += value[e] * x[index[e] * ch + k];
        y[r * ch + k] = a;
      }
    }
    return y;
  }
  selectRows(rows: Uint32Array): Csr {
    const start = new Uint32Array(rows.length + 1);
    for (let i = 0; i < rows.length; i++) start[i + 1] = start[i] + (this.start[rows[i] + 1] - this.start[rows[i]]);
    const index = new Uint32Array(start[rows.length]), value = new Float32Array(start[rows.length]);
    for (let i = 0; i < rows.length; i++) {
      const a = this.start[rows[i]], n = this.start[rows[i] + 1] - a;
      index.set(this.index.subarray(a, a + n), start[i]);
      value.set(this.value.subarray(a, a + n), start[i]);
    }
    return new Csr(rows.length, this.cols, start, index, value);
  }
}

// unique edges sorted by (smaller vertex, larger vertex), and the edge of each quad side (k, k + 1)
export function buildEdges(quads: Uint32Array, nVertices: number): { edges: Uint32Array; faceEdges: Uint32Array } {
  const m = quads.length / 4;
  const keys = new Float64Array(m * 4);
  for (let f = 0; f < m; f++) for (let k = 0; k < 4; k++) {
    const a = quads[f * 4 + k], b = quads[f * 4 + ((k + 1) & 3)];
    keys[f * 4 + k] = Math.min(a, b) * nVertices + Math.max(a, b);
  }
  const sorted = Float64Array.from(keys).sort();
  let nE = 0;
  for (let i = 0; i < sorted.length; i++) if (i === 0 || sorted[i] !== sorted[i - 1]) sorted[nE++] = sorted[i];
  const uniq = sorted.subarray(0, nE);
  const edges = new Uint32Array(nE * 2);
  for (let e = 0; e < nE; e++) { const lo = Math.floor(uniq[e] / nVertices); edges[e * 2] = lo; edges[e * 2 + 1] = uniq[e] - lo * nVertices; }
  const faceEdges = new Uint32Array(m * 4);
  for (let i = 0; i < keys.length; i++) {
    let lo = 0, hi = nE - 1;
    const k = keys[i];
    while (lo < hi) { const mid = (lo + hi) >> 1; if (uniq[mid] < k) lo = mid + 1; else hi = mid; }
    faceEdges[i] = lo;
  }
  return { edges, faceEdges };
}

export interface Level { op: Csr; quads: Uint32Array; parent: Uint32Array; nOut: number }

// one level of Catmull-Clark (anny.utils.subdivision.catmull_clark); `rows` keeps only these output rows
export function catmullClark(nV: number, quads: Uint32Array, rows?: Uint32Array): Level {
  const M = quads.length / 4;
  const { edges, faceEdges } = buildEdges(quads, nV);
  const nE = edges.length / 2, nOut = nV + nE + M;
  const facesPerEdge = new Uint32Array(nE);
  for (let i = 0; i < faceEdges.length; i++) facesPerEdge[faceEdges[i]]++;
  const valence = new Float64Array(nV), facesPerVertex = new Float64Array(nV), boundaryCount = new Uint32Array(nV);
  for (let e = 0; e < nE; e++) {
    valence[edges[e * 2]]++; valence[edges[e * 2 + 1]]++;
    if (facesPerEdge[e] === 1) { boundaryCount[edges[e * 2]]++; boundaryCount[edges[e * 2 + 1]]++; }
  }
  for (let i = 0; i < quads.length; i++) facesPerVertex[quads[i]]++;
  // faces around each vertex and edges around each vertex
  const vfStart = new Uint32Array(nV + 1), veStart = new Uint32Array(nV + 1);
  for (let i = 0; i < quads.length; i++) vfStart[quads[i] + 1]++;
  for (let e = 0; e < nE * 2; e++) veStart[edges[e] + 1]++;
  for (let v = 0; v < nV; v++) { vfStart[v + 1] += vfStart[v]; veStart[v + 1] += veStart[v]; }
  const vf = new Uint32Array(quads.length), ve = new Uint32Array(nE * 2);
  { const f1 = vfStart.slice(0, nV), f2 = veStart.slice(0, nV);
    for (let i = 0; i < quads.length; i++) vf[f1[quads[i]]++] = i >> 2;
    for (let e = 0; e < nE; e++) { ve[f2[edges[e * 2]]++] = e; ve[f2[edges[e * 2 + 1]]++] = e; } }
  // edge -> its (up to two) faces
  const ef = new Int32Array(nE * 2).fill(-1);
  for (let i = 0; i < faceEdges.length; i++) { const e = faceEdges[i]; if (ef[e * 2] < 0) ef[e * 2] = i >> 2; else ef[e * 2 + 1] = i >> 2; }
  const wanted = rows || null;
  const nRows = wanted ? wanted.length : nOut;
  // entries per row, merged in a small list (in the order they come, as anny.utils.subdivision sums them) and then
  // sorted by column
  const start = new Uint32Array(nRows + 1);
  let idxA = new Uint32Array(nRows * 8), valA = new Float32Array(nRows * 8), used = 0;
  const cols = new Uint32Array(256), vals = new Float64Array(256);
  let cnt = 0;
  const add = (c: number, w: number) => {
    for (let i = 0; i < cnt; i++) if (cols[i] === c) { vals[i] += w; return; }
    cols[cnt] = c; vals[cnt] = w; cnt++;
  };
  for (let r = 0; r < nRows; r++) {
    const row = wanted ? wanted[r] : r;
    cnt = 0;
    if (row >= nV + nE) {
      const f = row - nV - nE;
      for (let k = 0; k < 4; k++) add(quads[f * 4 + k], 0.25);
    } else if (row >= nV) {
      const e = row - nV, a = edges[e * 2], b = edges[e * 2 + 1];
      if (facesPerEdge[e] === 1) { add(a, 0.5); add(b, 0.5); }
      else {
        add(a, 0.25); add(b, 0.25);
        for (let s = 0; s < 2; s++) { const f = ef[e * 2 + s]; if (f < 0) continue; for (let k = 0; k < 4; k++) add(quads[f * 4 + k], 0.0625); }
      }
    } else {
      const v = row;
      if (facesPerVertex[v] === 0) add(v, 1);
      else if (boundaryCount[v] > 0) {
        if (boundaryCount[v] === 2) {
          add(v, 0.75);
          for (let q = veStart[v]; q < veStart[v + 1]; q++) {
            const e = ve[q];
            if (facesPerEdge[e] !== 1) continue;
            add(edges[e * 2] === v ? edges[e * 2 + 1] : edges[e * 2], 0.125);
          }
        } else add(v, 1);
      } else {
        const n = Math.max(valence[v], 1);
        add(v, (n - 3) / n);
        const fw = 1 / n / Math.max(facesPerVertex[v], 1) * 0.25;
        for (let q = vfStart[v]; q < vfStart[v + 1]; q++) { const f = vf[q]; for (let k = 0; k < 4; k++) add(quads[f * 4 + k], fw); }
        const ew = 2 / n / n * 0.5;
        for (let q = veStart[v]; q < veStart[v + 1]; q++) { const e = ve[q]; add(edges[e * 2], ew); add(edges[e * 2 + 1], ew); }
      }
    }
    for (let i = 1; i < cnt; i++) {
      const c = cols[i], w = vals[i];
      let j = i - 1;
      while (j >= 0 && cols[j] > c) { cols[j + 1] = cols[j]; vals[j + 1] = vals[j]; j--; }
      cols[j + 1] = c; vals[j + 1] = w;
    }
    if (used + cnt > idxA.length) {
      const i2 = new Uint32Array(idxA.length * 2), v2 = new Float32Array(idxA.length * 2);
      i2.set(idxA); v2.set(valA); idxA = i2; valA = v2;
    }
    for (let i = 0; i < cnt; i++) { idxA[used] = cols[i]; valA[used] = vals[i]; used++; }
    start[r + 1] = used;
  }
  const op = new Csr(nRows, nV, start, idxA.slice(0, used), valA.slice(0, used));
  // the new quads, in the order of anny.utils.subdivision (one block per corner)
  const nq = new Uint32Array(M * 16), parent = new Uint32Array(M * 4);
  for (let k = 0; k < 4; k++) for (let f = 0; f < M; f++) {
    const o = (k * M + f) * 4, e = (j: number) => nV + faceEdges[f * 4 + j], fc = nV + nE + f;
    const v = quads[f * 4 + k];
    nq[o] = v; nq[o + 1] = e(k); nq[o + 2] = fc; nq[o + 3] = e((k + 3) & 3);
    parent[k * M + f] = f;
  }
  return { op, quads: nq, parent, nOut };
}

// the subdivision of the page: `levels` full levels, then the rows of the last level that the fine mesh uses
export class FineSubdivision {
  ops: Csr[];
  nIn: number;
  nOut: number;
  constructor(nVertices: number, quads: Uint32Array, levels: number, lastRows: Uint32Array) {
    this.ops = [];
    let n = nVertices, q = quads;
    for (let l = 0; l < levels; l++) {
      const L = catmullClark(n, q);
      this.ops.push(L.op);
      n = L.nOut; q = L.quads;
    }
    this.ops.push(catmullClark(n, q, lastRows).op);
    this.nIn = nVertices;
    this.nOut = lastRows.length;
  }
  buffers: Record<number, Float32Array[]> = {};
  // the result is a buffer of this object, overwritten by the next call with the same channel count
  apply(x: Float32Array, ch = 3): Float32Array {
    const bufs = this.buffers[ch] || (this.buffers[ch] = this.ops.map((op) => new Float32Array(op.rows * ch)));
    let y: Float32Array = x;
    this.ops.forEach((op, i) => { y = op.apply(y, ch, bufs[i]); });
    return y;
  }
}
