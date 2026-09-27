// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The hair's physics of the page (viewer/src/hair/sim.ts, colliders.ts) on the data of anny.viewer (run by
// test/test_hair_dynamics.py):
//   node viewer/test/sim.mjs <build folder> <output folder> <style> [steps]
// On anny's default body, the head of the figure shakes and nods about the neck while the solver moves the simulated
// guides of the style. The script writes the inputs of each step (the head's move, the gravity and the colliders),
// the rest points and pivots, and the solver's points after every step, which the test replays with
// anny.hair.dynamics.
import fs from 'fs';
import path from 'path';
import { AnnyBody, vertexNormals } from '../src/body.ts';
import { decodeLayout, decodeStyle } from '../src/hair/data.ts';
import { HairSim, SIM_DEFAULTS, STEP } from '../src/hair/sim.ts';
import { COLLIDERS, fitColliders } from '../src/hair/colliders.ts';

const [dir, out, name, stepsArg] = process.argv.slice(2);
const N = parseInt(stepsArg || '120');
const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
const info = Object.fromEntries(man.buffers.map((b) => [b.name, b]));
const bytes = (n) => { const b = fs.readFileSync(path.join(dir, n + '.raw')); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const drop4th = (a) => { const n = a.length / 4, o = new a.constructor(n * 3); for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) o[i * 3 + k] = a[i * 4 + k]; return o; };

// anny's default body, for the colliders
const hv = info.head_v, nv = hv.count, rec = new DataView(bytes('head_v'));
const rest = new Float32Array(nv * 3);
for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) rest[i * 3 + k] = hv.lo[k] + rec.getUint16(i * hv.stride + k * 2, true) / 65535 * (hv.hi[k] - hv.lo[k]);
const index = new Uint32Array(bytes('head_i'));
const body = new AnnyBody({ ...man, detail_step: info.head_detail.step }, {
  template: new Float32Array(bytes('coarse_template')), components: drop4th(new Int16Array(bytes('shape_components'))),
  projection: new Float32Array(bytes('shape_projection')), jointTemplate: new Float32Array(bytes('joint_template')),
  jointBlend: new Float32Array(bytes('joint_blend')), quads: new Uint32Array(bytes('coarse_quads')), rows: new Uint32Array(bytes('head_row')),
  detail: drop4th(new Int16Array(bytes('head_detail'))), relief: new Int16Array(bytes('head_detail')).filter((_, i) => i % 4 === 3), index, rest, nsmooth: vertexNormals(rest, index), coarseSkin: new Uint8Array(bytes('coarse_skin')),
});
body.update({});

// the style's simulated guides on anny's default body
const hair = man.hair, u8 = (n) => new Uint8Array(bytes(n));
const layout = decodeLayout(u8, hair), spec = hair.styles.find((s) => s.name === name);
const style = decodeStyle(u8, layout, spec), S = layout.S, P = style.P;
const R0 = Float64Array.from(style.points.subarray(0, S * P * 3)), pivot = Float64Array.from(style.pivot.subarray(0, S));
const names = man.rig.names;
const fitted = fitColliders(COLLIDERS, names, body.joints, { cranium: hair.centre }, body.coarse, body.nBody, 1);
const sim = new HairSim(S, P);
// SIM_PARAMS (JSON) overrides the style's physics, for tuning
const params = { ...SIM_DEFAULTS, ...(spec.physics || {}), ...JSON.parse(process.env.SIM_PARAMS || '{}') };
const caps0 = fitted.rest;
sim.setGroom(R0, pivot, params, caps0);

// the head turns about the neck: a shake for the first second, then a nod; the colliders of the head and the neck's
// upper end turn with it
const neck = names.indexOf('neck01'), c = [0, 1, 2].map((k) => body.joints[neck * 3 + k]);
const headEnds = COLLIDERS.map((d) => d.bones.map((b) => b === 'head'));
const rot = (yaw, pitch) => {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // R = Ry(yaw) Rx(pitch)
  return [cy, sy * sp, sy * cp, 0, cp, -sp, -sy, cy * sp, cy * cp];
};
const apply = (M, p) => [0, 1, 2].map((r) => M[r * 4] * p[0] + M[r * 4 + 1] * p[1] + M[r * 4 + 2] * p[2] + M[r * 4 + 3]);
const T = new Float64Array(S * P * 3), moves = [], gravity = [], capsules = [], speeds = [];
const X = new Float64Array(N * S * P * 3);
let ms = 0;
sim.reset(R0);
for (let n = 0; n < N; n++) {
  const t = (n + 1) * STEP;
  const yaw = t < 1 ? 0.6 * Math.sin(2 * Math.PI * 1.5 * t) : 0;
  const pitch = t >= 1 ? 0.35 * Math.sin(2 * Math.PI * 1.0 * (t - 1)) : 0;
  const R = rot(yaw, pitch);
  // x -> R (x - c) + c, as rows [R | t]
  const M = [];
  for (let r = 0; r < 3; r++) M.push(R[r * 3], R[r * 3 + 1], R[r * 3 + 2], c[r] - (R[r * 3] * c[0] + R[r * 3 + 1] * c[1] + R[r * 3 + 2] * c[2]));
  for (let i = 0; i < S * P; i++) { const p = apply(M, [R0[i * 3], R0[i * 3 + 1], R0[i * 3 + 2]]); T[i * 3] = p[0]; T[i * 3 + 1] = p[1]; T[i * 3 + 2] = p[2]; }
  const caps = caps0.slice();
  for (let q = 0; q < caps.length / 7; q++) for (let e = 0; e < 2; e++) {
    if (!headEnds[q]?.[e]) continue;
    const p = apply(M, [caps[q * 7 + e * 3], caps[q * 7 + e * 3 + 1], caps[q * 7 + e * 3 + 2]]);
    caps.set(p, q * 7 + e * 3);
  }
  const k = params.gravity * 9.81, g = [k * R[1], k * (R[4] - 1), k * R[7]];
  const t0 = performance.now();
  speeds.push(sim.step(T, g, caps));
  ms += performance.now() - t0;
  X.set(sim.x, n * S * P * 3);
  moves.push(M); gravity.push(g); capsules.push(Array.from(caps));
}
fs.writeFileSync(path.join(out, `${name}_rest.bin`), Buffer.from(R0.buffer));
fs.writeFileSync(path.join(out, `${name}_x.bin`), Buffer.from(X.buffer));
fs.writeFileSync(path.join(out, `${name}.json`), JSON.stringify({
  S, P, N, params, pivot: Array.from(pivot), moves, gravity, capsules, speeds, ms_per_step: ms / N,
  colliders: COLLIDERS.map((d) => d.name), fitted: Array.from(fitted.rest),
}));
