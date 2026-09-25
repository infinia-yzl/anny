// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The body of the page on the data of anny.viewer (run by test/test_viewer_body.py):
//   node viewer/test/body.mjs <build folder> <output json>
// It builds AnnyBody from the raw buffers of viewer/build, as the page does after decoding, and reports how far
// anny's default body and its hair lie from the fine body and the strands of anny.viewer.
import fs from 'fs';
import path from 'path';
import { AnnyBody, vertexNormals } from '../src/body.ts';

const [dir, out] = process.argv.slice(2);
const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
const info = Object.fromEntries(man.buffers.map((b) => [b.name, b]));
// a copy of the file's bytes (node pools small files, so the Buffer's ArrayBuffer can be larger)
const bytes = (name) => { const b = fs.readFileSync(path.join(dir, name + '.raw')); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const drop4th = (a) => { const n = a.length / 4, o = new a.constructor(n * 3); for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) o[i * 3 + k] = a[i * 4 + k]; return o; };

// the fine body of anny's default body (16-bit positions in the vertex records)
const hv = info.head_v, n = hv.count, rec = new DataView(bytes('head_v'));
const rest = new Float32Array(n * 3);
for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) rest[i * 3 + k] = hv.lo[k] + rec.getUint16(i * hv.stride + k * 2, true) / 65535 * (hv.hi[k] - hv.lo[k]);
const index = new Uint32Array(bytes('head_i'));
const sm = man.shape;
const body = new AnnyBody({ ...man, detail_step: info.head_detail.step }, {
  template: new Float32Array(bytes('coarse_template')), components: drop4th(new Int16Array(bytes('shape_components'))),
  projection: new Float32Array(bytes('shape_projection')), jointTemplate: new Float32Array(bytes('joint_template')),
  jointBlend: new Float32Array(bytes('joint_blend')), quads: new Uint32Array(bytes('coarse_quads')), rows: new Uint32Array(bytes('head_row')),
  detail: drop4th(new Int16Array(bytes('head_detail'))), relief: new Int16Array(bytes('head_detail')).filter((_, i) => i % 4 === 3), index, rest, nsmooth: vertexNormals(rest, index), coarseSkin: new Uint8Array(bytes('coarse_skin')),
});

// the hair of anny's default body: 16-bit roots and 8-bit steps
const counts = new Uint8Array(bytes('hair_n')).subarray(0, info.hair_n.strands), nS = counts.length;
const rv = new DataView(bytes('hair_r')), steps = new Int8Array(bytes('hair_d')), lo = info.hair_r.lo, hi = info.hair_r.hi, dq = info.hair_d.dq;
let total = 0; for (const c of counts) total += c;
const P = new Float32Array(total * 3);
for (let i = 0, p = 0, q = 0; i < nS; i++) {
  const x = [0, 1, 2].map((k) => lo[k] + rv.getUint16(i * 6 + k * 2, true) / 65535 * (hi[k] - lo[k]));
  for (let j = 0; j < counts[i]; j++, p++) {
    if (j > 0) for (let k = 0; k < 3; k++) x[k] += steps[q++] * dq;
    for (let k = 0; k < 3; k++) P[p * 3 + k] = x[k];
  }
}
const binding = (name) => {
  const dv = new DataView(bytes(name)), corners = new Uint32Array(nS * 3), bary = new Float32Array(nS * 3);
  for (let i = 0; i < nS; i++) {
    for (let k = 0; k < 3; k++) corners[i * 3 + k] = dv.getUint32(i * 20 + k * 4, true);
    const u = dv.getFloat32(i * 20 + 12, true), w = dv.getFloat32(i * 20 + 16, true);
    bary[i * 3] = u; bary[i * 3 + 1] = w; bary[i * 3 + 2] = 1 - u - w;
  }
  return { corners, bary };
};
const root = binding('hair_bind');
body.bindStrands('hair', P, counts, root.corners, root.bary, undefined, binding('hair_tip'));

const maxDistance = (A, B) => { let m = 0; for (let i = 0; i < A.length; i += 3) m = Math.max(m, Math.hypot(A[i] - B[i], A[i + 1] - B[i + 1], A[i + 2] - B[i + 2])); return m; };
const result = {};
let t = body.update({});
result.rest_max = maxDistance(body.pos, rest);
result.hair_max = maxDistance(body.strandPoints('hair'), P);
result.default_ms = t.ms;
// the ends of the sliders: every value finite, and the hair keeps its size relative to the head
for (const [key, values] of [['age0', { age: 0 }], ['age1', { age: 1 }], ['all0', Object.fromEntries(sm.sliders.map((s) => [s, 0]))], ['all1', Object.fromEntries(sm.sliders.map((s) => [s, 1]))]]) {
  t = body.update(values);
  const hair = body.strandPoints('hair');
  result[key] = { finite: body.pos.every(Number.isFinite) && body.nrm.every(Number.isFinite) && hair.every(Number.isFinite), head_scale: body.headScale(), ms: t.ms };
}
fs.writeFileSync(out, JSON.stringify(result));
