// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Build the viewer page: encode the data of anny.viewer (build/*.raw, build/manifest.json) with meshopt and
// gzip, bundle the TypeScript with three.js, and write one HTML file (dist/anny_viewer.html).
// Ported from the legacy web/encode.mjs and web/build_page.py.
//
//     python -m anny.viewer build --data && npm run build

import { MeshoptEncoder } from 'meshoptimizer';
import * as esbuild from 'esbuild';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.join(here, 'build');
const out = path.join(here, 'dist', 'anny_viewer.html');
await MeshoptEncoder.ready;
const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
const read = (name) => new Uint8Array(fs.readFileSync(path.join(dir, name + '.raw')));
const META_KEYS = ['eyes', 'eye_mesh', 'lut', 'skin', 'rig', 'motion', 'correctives', 'shape', 'stool', 'authoring'];
const header = { buffers: [], meta: Object.fromEntries(META_KEYS.map((k) => [k, man[k]])) };
const parts = [];
let offset = 0;
function push(entry, bytes) {
  const pad = (4 - (bytes.length % 4)) % 4;
  entry.byteOffset = offset; entry.byteLength = bytes.length;
  parts.push(Buffer.from(bytes)); if (pad) parts.push(Buffer.alloc(pad));
  offset += bytes.length + pad;
  header.buffers.push(entry);
}
const bufs = Object.fromEntries(man.buffers.map((b) => [b.name, b]));
const COMPANIONS = ['_row', '_detail', '_skin'];
const REMAP = {};

// meshes: '<name>_v' with '<name>_i' are reordered for the vertex cache; the companion streams follow
function encodeMesh(prefix) {
  const hv = bufs[prefix + '_v'];
  const vdata = read(prefix + '_v');
  const idx = new Uint32Array(read(prefix + '_i').buffer.slice(0));
  const [remap, unique] = MeshoptEncoder.reorderMesh(idx, true, false);
  const nv = new Uint8Array(unique * hv.stride);
  for (let i = 0; i < hv.count; i++) { const r = remap[i]; if (r !== 0xffffffff) nv.set(vdata.subarray(i * hv.stride, (i + 1) * hv.stride), r * hv.stride); }
  push({ name: prefix + '_v', enc: 'mv', count: unique, stride: hv.stride, lo: hv.lo, hi: hv.hi }, MeshoptEncoder.encodeVertexBufferLevel(nv, unique, hv.stride, 3, 1));
  push({ name: prefix + '_i', enc: 'mi', count: idx.length, stride: 4 }, MeshoptEncoder.encodeIndexBuffer(new Uint8Array(idx.buffer), idx.length, 4));
  header.meta[prefix + '_count'] = unique;
  REMAP[prefix] = remap;
  for (const suffix of COMPANIONS) {
    const mb = bufs[prefix + suffix];
    if (!mb) continue;
    const md = read(prefix + suffix);
    const nm = new Uint8Array(unique * mb.stride);
    for (let i = 0; i < mb.count; i++) { const r = remap[i]; if (r !== 0xffffffff) nm.set(md.subarray(i * mb.stride, (i + 1) * mb.stride), r * mb.stride); }
    const entry = Object.assign({}, mb, { name: prefix + suffix, enc: 'mv', count: unique });
    delete entry.kind;
    push(entry, MeshoptEncoder.encodeVertexBufferLevel(nm, unique, mb.stride, 3, 1));
  }
  console.log(prefix, unique, 'vertices,', idx.length / 3, 'triangles');
}
const meshes = man.buffers.filter((b) => b.name.endsWith('_v') && bufs[b.name.slice(0, -2) + '_i']).map((b) => b.name.slice(0, -2));
meshes.forEach(encodeMesh);

// records that name vertices of a mesh: the vertex fields follow the new vertex order; the records of each
// corrective shape are sorted by vertex again (sort: 'shapes'), the strand bindings keep their order
function encodeSparse(b) {
  const br = REMAP[b.body];
  const data = read(b.name);
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let i = 0; i < b.count; i++) for (let f = 0; f < (b.fields || 1); f++) {
    const r = br[dv.getUint32(i * b.stride + f * 4, true)];
    if (r === 0xffffffff) throw new Error('record on an unused vertex: ' + b.name);
    dv.setUint32(i * b.stride + f * 4, r, true);
  }
  let outb = data.subarray(0, b.count * b.stride);
  if (b.sort === 'shapes') {
    outb = new Uint8Array(b.count * b.stride);
    for (const s of man.correctives.shapes) {
      const order = Array.from({ length: s.count }, (_, k) => s.start + k).sort((x, y) => dv.getUint32(x * b.stride, true) - dv.getUint32(y * b.stride, true));
      order.forEach((src, k) => outb.set(data.subarray(src * b.stride, (src + 1) * b.stride), (s.start + k) * b.stride));
    }
  }
  const entry = Object.assign({ enc: 'mv' }, b); delete entry.kind;
  push(entry, MeshoptEncoder.encodeVertexBufferLevel(outb, b.count, b.stride, 3, 1));
}
for (const b of man.buffers) {
  if (b.kind === 'sparse') { encodeSparse(b); continue; }
  if (meshes.some((p) => b.name === p + '_v' || b.name === p + '_i' || COMPANIONS.some((s) => b.name === p + s))) continue;
  const data = read(b.name);
  if (b.kind === 'vertex') push(Object.assign({ enc: 'mv' }, b), MeshoptEncoder.encodeVertexBufferLevel(data, b.count, b.stride, 3, 1));
  else push(Object.assign({ enc: 'raw' }, b), data);
}
const hjson = Buffer.from(JSON.stringify(header));
const hl = Buffer.alloc(8); hl.write('HDB1', 0); hl.writeUInt32LE(hjson.length, 4);
const blob = Buffer.concat([hl, hjson, Buffer.alloc((4 - (hjson.length % 4)) % 4), ...parts]);
const gz = zlib.gzipSync(blob, { level: 9 });
console.log('data', (blob.length / 1e6).toFixed(1), 'MB, gzip', (gz.length / 1e6).toFixed(1), 'MB');

// the script: the TypeScript with three.js in one bundle
const bundle = await esbuild.build({
  entryPoints: [path.join(here, 'src', 'main.ts')],
  bundle: true, format: 'esm', minify: true, write: false, target: 'es2022',
  alias: { 'three/addons': 'three/examples/jsm' },
  legalComments: 'none',
});
const app = bundle.outputFiles[0].text;
const shell = fs.readFileSync(path.join(here, 'shell.html'), 'utf8');
const ref = fs.readFileSync(path.join(here, 'src', 'meshopt_ref.js'), 'utf8');
const page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
  + shell + '\n<script>\n' + ref + '\n</script>\n<script>window.MODEL_B64 = "' + gz.toString('base64') + '";</script>\n'
  + '<script type="module">\n' + app.replace(/<\/script/g, '<\\/script') + '\n</script>\n</head><body></body></html>\n';
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, page);
console.log('page', out, (page.length / 1e6).toFixed(1), 'MB');
