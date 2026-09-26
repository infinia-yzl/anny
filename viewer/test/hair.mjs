// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The hair data of the page (viewer/src/hair/data.ts) on the buffers of anny.viewer (run by test/test_hair_parity.py):
//   node viewer/test/hair.mjs <build folder> <output folder>
// It decodes the layout and every style, and writes the decoded guides, the density volume of each style and random
// values, which the test compares with anny.hair.styles.
import fs from 'fs';
import path from 'path';
import { decodeLayout, decodeStyle, densityVolume, mirrorStyle, rnd } from '../src/hair/data.ts';

const [dir, out] = process.argv.slice(2);
const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
const bytes = (name) => new Uint8Array(fs.readFileSync(path.join(dir, name + '.raw')));
const meta = man.hair;
const layout = decodeLayout(bytes, meta);
const result = { styles: {}, rnd: [] };
for (const spec of meta.styles) {
  const s = decodeStyle(bytes, layout, spec);
  fs.writeFileSync(path.join(out, `${spec.name}_points.bin`), Buffer.from(s.points.buffer));
  const m = mirrorStyle(s, layout);
  fs.writeFileSync(path.join(out, `${spec.name}_mirror.bin`), Buffer.from(m.points.buffer));
  const params = { length: 0.9, fade: 0 };
  const v = densityVolume(meta, layout, s, params, 40000);
  fs.writeFileSync(path.join(out, `${spec.name}_volume.bin`), Buffer.from(v.data.buffer));
  result.styles[spec.name] = { P: s.P, dims: v.dims, lo: v.lo, h: v.h, params, count: 40000 };
}
for (let k = 0; k < 1000; k++) result.rnd.push(rnd(k * 7919 + 13, k % 17));
fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify(result));
