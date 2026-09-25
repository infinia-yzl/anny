// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Parity of the page's maths with anny (run by test/test_viewer_parity.py):
//   node viewer/test/parity.mjs <folder>
// reads <folder>/input.json and binary inputs, writes binary outputs next to them.
import fs from 'fs';
import path from 'path';
import { coefficients, makeRule, ShapeSpace } from '../src/anny_shape.ts';
import { FineSubdivision } from '../src/subdivision.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
// a copy of the file's bytes (node pools small files, so the Buffer's ArrayBuffer can be larger)
const bytes = (name) => { const b = fs.readFileSync(path.join(dir, name)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const f32 = (name) => new Float32Array(bytes(name));
const i16 = (name) => new Int16Array(bytes(name));
const u32 = (name) => new Uint32Array(bytes(name));
const rule = makeRule(inp.tables);
// coefficients for every setting
const C = new Float64Array(inp.settings.length * inp.n_shapes);
inp.settings.forEach((s, i) => C.set(coefficients(rule, s), i * inp.n_shapes));
fs.writeFileSync(path.join(dir, 'coefficients.bin'), Buffer.from(C.buffer));
// coarse vertices and bone heads for the first settings
const space = new ShapeSpace({ template: f32('template.bin'), components: i16('components.bin'), componentScale: inp.component_scale,
  projection: f32('projection.bin'), jointTemplate: f32('joint_template.bin'), jointBlend: f32('joint_blend.bin') });
const nv = inp.shape_settings;
const V = new Float32Array(nv * space.nc * 3), J = new Float32Array(nv * space.nb * 3);
for (let i = 0; i < nv; i++) {
  const c = C.subarray(i * inp.n_shapes, (i + 1) * inp.n_shapes);
  V.set(space.coarse(c), i * space.nc * 3);
  J.set(space.joints(c), i * space.nb * 3);
}
fs.writeFileSync(path.join(dir, 'coarse.bin'), Buffer.from(V.buffer));
fs.writeFileSync(path.join(dir, 'joints.bin'), Buffer.from(J.buffer));
// the fine body of a coarse body
if (inp.subdivision) {
  const t0 = performance.now();
  const sub = new FineSubdivision(inp.subdivision.n, u32('quads.bin'), inp.subdivision.levels, u32('rows.bin'));
  const t1 = performance.now();
  const F = sub.apply(f32('coarse_in.bin'), 3);
  const t2 = performance.now();
  fs.writeFileSync(path.join(dir, 'fine.bin'), Buffer.from(F.buffer));
  fs.writeFileSync(path.join(dir, 'timing.json'), JSON.stringify({ build_ms: t1 - t0, apply_ms: t2 - t1 }));
}
