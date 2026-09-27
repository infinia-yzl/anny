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
import { decodeLayout, decodeStyle } from '../src/hair/data.ts';

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

// the hair of anny's default body: the guides of the first style, bound to the triangles under their roots and tips
const hair = man.hair, layout = decodeLayout((name) => new Uint8Array(bytes(name)), hair);
const style = decodeStyle((name) => new Uint8Array(bytes(name)), layout, hair.styles[0]);
const P = style.points, counts = new Uint8Array(layout.G).fill(style.P);
body.bindStrands('hair', P, counts, layout.guideCorners, layout.guideBary, undefined, { corners: style.tipCorners, bary: style.tipBary });

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
