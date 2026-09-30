// Corporis
// Apache License, Version 2.0
//
// Parity of the page's retarget (src/pose_from_image.ts) with corporis.posing.retarget, run by
// test/test_viewer_pose_parity.py:
//   node viewer/test/pose_parity.mjs <folder>
// reads <folder>/input.json and writes <folder>/output.json.
import fs from 'fs';
import path from 'path';
import { Retargeter, adjustHead, landmarkSources, quaternions } from '../src/pose_from_image.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const rest = { names: inp.names, parents: inp.parents, heads: inp.heads, vertices: inp.vertices, top: inp.top };
const sources = landmarkSources(rest);
const retarget = new Retargeter(rest, sources);
const poses = inp.landmarks.map((L) => Array.from(quaternions(retarget.local(retarget.solve(L)))));
const heads = inp.landmarks.map((L) => {
  const W = retarget.solve(L);
  return retarget.headFit(L, W.get('spine01')).source;
});
// the first pose with the head corrected by hand
const [turn, up, tilt] = inp.head;
const adjusted = Array.from(quaternions(retarget.local(adjustHead(retarget.solve(inp.landmarks[0]), turn, up, tilt))));
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify({ sources, poses, heads, adjusted }));
