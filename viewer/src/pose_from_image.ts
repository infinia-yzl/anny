// Corporis
// Apache License, Version 2.0
//
// Pose from a picture, the page's side: MediaPipe's landmarks turned into a rotation for each of anny's bones. This
// repeats corporis.posing.retarget (AnnyLandmarks and Retargeter) and corporis.posing.skeleton (the hinge) in anny's
// frame (x toward the figure's left, -y forward, z up, metres), and test/test_viewer_parity.py checks it against
// Python. The refinement of corporis.posing.refine stays in Python. Matrices are 3x3, row-major.

export type V3 = [number, number, number];
export type M3 = number[];

// MediaPipe's names of the body points and the hand points (corporis.posing.landmarks)
export const BODY = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index'];
export const HAND = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_mcp', 'index_pip', 'index_dip',
  'index_tip', 'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip', 'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'];
const SIDES: [string, string][] = [['.L', 'left'], ['.R', 'right']];
const FINGERS: [number, string][] = [[1, 'thumb'], [2, 'index'], [3, 'middle'], [4, 'ring'], [5, 'pinky']];
const KNUCKLES = ['mcp', 'pip', 'dip', 'tip'];
const THUMB = ['cmc', 'mcp', 'ip', 'tip'];
const SPINE: [string, number][] = [['spine05', 0.15], ['spine04', 0.35], ['spine03', 0.55], ['spine02', 0.75], ['spine01', 1.0]];
const NECK: [string, number][] = [['neck01', 0.3], ['neck02', 0.55], ['neck03', 0.75]];

// the landmarks of one figure, in anny's frame: 33 body points, and 21 points per hand (".L", ".R")
export interface Landmarks { body: V3[]; hands: Record<string, V3[]> }

// MediaPipe's world points (x right, y down, z away from the camera) in anny's frame
export function fromMediapipe(points: { x: number; y: number; z: number }[]): V3[] {
  return points.map((p) => [p.x, p.z, -p.y]);
}

// ---------------------------------------------------------------- vectors and rotations
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: V3): V3 => scale(a, 1 / norm(a));
const mid = (a: V3, b: V3): V3 => scale(add(a, b), 0.5);

export const I3: M3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export function mul(a: M3, b: M3): M3 {
  const o = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}
export const transpose = (a: M3): M3 => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const apply = (a: M3, v: V3): V3 => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];

// a turn about axis v by |v| radians
function rotvec(v: V3): M3 {
  const t = norm(v);
  if (t < 1e-12) return I3.slice();
  const [x, y, z] = scale(v, 1 / t), c = Math.cos(t), s = Math.sin(t), k = 1 - c;
  return [c + x * x * k, x * y * k - z * s, x * z * k + y * s,
    y * x * k + z * s, c + y * y * k, y * z * k - x * s,
    z * x * k - y * s, z * y * k + x * s, c + z * z * k];
}
// the axis times the angle of a rotation
function logRot(R: M3): V3 {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2)), t = Math.acos(c);
  if (t < 1e-9) return [0, 0, 0];
  if (Math.PI - t < 1e-6) {
    // half a turn: the axis is the column of R + I with the largest diagonal
    const d = [R[0], R[4], R[8]], i = d.indexOf(Math.max(...d));
    const col: V3 = [R[i] + (i === 0 ? 1 : 0), R[3 + i] + (i === 1 ? 1 : 0), R[6 + i] + (i === 2 ? 1 : 0)];
    return scale(unit(col), t);
  }
  const w: V3 = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  return scale(w, t / (2 * Math.sin(t)));
}
export function axisAngle(axis: V3, deg: number): M3 { return rotvec(scale(unit(axis), deg * Math.PI / 180)); }
// the smallest rotation that takes direction a to direction b
export function align(a: V3, b: V3): M3 {
  a = unit(a); b = unit(b);
  const ax = cross(a, b), s = norm(ax), c = dot(a, b);
  if (s < 1e-8) {
    if (c > 0) return I3.slice();
    const other: V3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    return rotvec(scale(unit(cross(a, other)), Math.PI));
  }
  return rotvec(scale(ax, Math.atan2(s, c) / s));
}
// an orthonormal frame (as columns) from a main direction f and a second direction t
export function frame(f: V3, t: V3): M3 {
  f = unit(f); t = unit(sub(t, scale(f, dot(t, f))));
  const n = cross(f, t);
  return [f[0], t[0], n[0], f[1], t[1], n[1], f[2], t[2], n[2]];
}
// the rotation a fraction t of the way from a to b
export function slerp(a: M3, b: M3, t: number): M3 { return mul(a, rotvec(scale(logRot(mul(transpose(a), b)), t))); }

// unit quaternion (x, y, z, w) of a rotation matrix
export function quaternion(R: M3): number[] {
  const tr = R[0] + R[4] + R[8];
  let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s; }
  else if (R[0] > R[4] && R[0] > R[8]) { const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2; w = (R[7] - R[5]) / s; x = s / 4; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s; }
  else if (R[4] > R[8]) { const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2; w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = s / 4; z = (R[5] + R[7]) / s; }
  else { const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2; w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = s / 4; }
  const l = Math.hypot(x, y, z, w), k = w < 0 ? -1 / l : 1 / l;
  return [x * k, y * k, z * k, w * k];
}

// ---------------------------------------------------------------- anny's rest body
// the body in its rest pose, in anny's frame: the bone names, parents and heads, the vertices and the strongest bone of
// each vertex
export interface RestBody { names: string[]; parents: number[]; heads: V3[]; vertices: ArrayLike<number>; top: ArrayLike<number> }
type Source = ['joint', number] | ['vertex', number];

// MediaPipe's landmarks on anny's mesh (corporis.posing.retarget.AnnyLandmarks): each is a joint or a vertex
export function landmarkSources(rest: RestBody): { body: Source[]; hands: Record<string, Source[]> } {
  const V = rest.vertices, nv = V.length / 3, bone = (n: string) => rest.names.indexOf(n);
  const v = (i: number): V3 => [V[i * 3], V[i * 3 + 1], V[i * 3 + 2]];
  const joint = (n: string) => rest.heads[bone(n)];
  const topName = (i: number) => rest.names[rest.top[i]];
  const nearest = (p: V3, among?: (i: number) => boolean): Source => {
    let best = -1, bd = Infinity;
    for (let i = 0; i < nv; i++) { if (among && !among(i)) continue; const d = norm(sub(v(i), p)); if (d < bd) { bd = d; best = i; } }
    return ['vertex', best];
  };
  const extreme = (mask: (i: number) => boolean, key: (p: V3) => number): Source => {
    let best = -1, bk = -Infinity;
    for (let i = 0; i < nv; i++) { if (!mask(i)) continue; const k = key(v(i)); if (k > bk) { bk = k; best = i; } }
    return ['vertex', best];
  };
  const head = (i: number) => topName(i) === 'head';
  const src: Record<string, Source> = {};
  const nose = extreme((i) => head(i) && Math.abs(V[i * 3]) < 0.01, (p) => -p[1]);
  src.nose = nose;
  const n = v(nose[1]);
  for (const [s, side] of SIDES) {
    const sign = s === '.L' ? 1 : -1, eye = joint('eye' + s);
    src[`${side}_eye`] = ['joint', bone('eye' + s)];
    src[`${side}_eye_inner`] = nearest(add(eye, [-sign * 0.016, -0.01, 0]));
    src[`${side}_eye_outer`] = nearest(add(eye, [sign * 0.016, -0.005, 0]));
    const level = (i: number) => head(i) && Math.abs(V[i * 3 + 2] - eye[2] + 0.02) < 0.02 && V[i * 3 + 1] > eye[1] + 0.04;
    src[`${side}_ear`] = extreme(level, (p) => sign * p[0]);
    src[`mouth_${side}`] = nearest([sign * 0.024, n[1] + 0.012, n[2] - 0.045]);
    for (const [name, b] of [['shoulder', 'upperarm01'], ['elbow', 'lowerarm01'], ['wrist', 'wrist'], ['pinky', 'finger5-1'],
      ['index', 'finger2-1'], ['thumb', 'finger1-3'], ['hip', 'upperleg01'], ['knee', 'lowerleg01'], ['ankle', 'foot']]) {
      src[`${side}_${name}`] = ['joint', bone(b + s)];
    }
    const footBones = new Set(['foot' + s]);
    for (let k = 1; k <= 5; k++) for (let i = 1; i <= 3; i++) footBones.add(`toe${k}-${i}${s}`);
    const foot = (i: number) => footBones.has(topName(i));
    const low = joint('foot' + s)[2] - 0.03;
    src[`${side}_heel`] = extreme((i) => foot(i) && V[i * 3 + 2] < low, (p) => p[1]);
    src[`${side}_foot_index`] = extreme(foot, (p) => -p[1]);
  }
  const hands: Record<string, Source[]> = {};
  for (const [s] of SIDES) {
    const hand: Record<string, Source> = { wrist: ['joint', bone('wrist' + s)] };
    for (const [k, finger] of FINGERS) {
      const names = k === 1 ? THUMB : KNUCKLES;
      for (let i = 0; i < 3; i++) hand[`${finger}_${names[i]}`] = ['joint', bone(`finger${k}-${i + 1}${s}`)];
      const last = `finger${k}-3${s}`, d = sub(joint(last), joint(`finger${k}-2${s}`));
      hand[`${finger}_tip`] = extreme((i) => topName(i) === last, (p) => dot(p, d));
    }
    hands[s] = HAND.map((h) => hand[h]);
  }
  return { body: BODY.map((b) => src[b]), hands };
}

// ---------------------------------------------------------------- the retarget
const point = (L: Landmarks, name: string) => L.body[BODY.indexOf(name)];
const handPoint = (L: Landmarks, s: string, name: string) => L.hands[s][HAND.indexOf(name)];

function pelvisFrame(L: Landmarks) {
  const p = (n: string) => point(L, n);
  const up = sub(mid(p('left_shoulder'), p('right_shoulder')), mid(p('left_hip'), p('right_hip')));
  return frame(sub(p('left_hip'), p('right_hip')), up);
}
function chestFrame(L: Landmarks) {
  const p = (n: string) => point(L, n);
  const up = sub(mid(p('left_shoulder'), p('right_shoulder')), mid(p('left_hip'), p('right_hip')));
  return frame(sub(p('left_shoulder'), p('right_shoulder')), up);
}
function headFrame(L: Landmarks) {
  const p = (n: string) => point(L, n);
  return frame(sub(p('left_ear'), p('right_ear')), sub(p('nose'), mid(p('left_ear'), p('right_ear'))));
}
// the hand's frame: along the middle knuckle, then toward the index side
function palmFrame(L: Landmarks, s: string) {
  if (L.hands[s]) {
    const h = (n: string) => handPoint(L, s, n);
    return frame(sub(h('middle_mcp'), h('wrist')), sub(h('index_mcp'), h('pinky_mcp')));
  }
  const side = s === '.L' ? 'left' : 'right', p = (n: string) => point(L, `${side}_${n}`);
  return frame(sub(mid(p('index'), p('pinky')), p('wrist')), sub(p('index'), p('pinky')));
}

export class Retargeter {
  rest: RestBody; sources: ReturnType<typeof landmarkSources>; restLandmarks: Landmarks;
  restPelvis: M3; restChest: M3; restHead: M3; restHands: Record<string, M3>;
  constructor(rest: RestBody, sources?: ReturnType<typeof landmarkSources>) {
    this.rest = rest;
    this.sources = sources || landmarkSources(rest);
    const at = (src: Source): V3 => src[0] === 'joint' ? rest.heads[src[1]] : [rest.vertices[src[1] * 3], rest.vertices[src[1] * 3 + 1], rest.vertices[src[1] * 3 + 2]];
    this.restLandmarks = { body: this.sources.body.map(at), hands: { '.L': this.sources.hands['.L'].map(at), '.R': this.sources.hands['.R'].map(at) } };
    const R = this.restLandmarks;
    this.restPelvis = pelvisFrame(R); this.restChest = chestFrame(R); this.restHead = headFrame(R);
    this.restHands = { '.L': palmFrame(R, '.L'), '.R': palmFrame(R, '.R') };
  }
  private joint(n: string) { return this.rest.heads[this.rest.names.indexOf(n)]; }
  private restDirection(a: string, b: string) { return unit(sub(this.joint(b), this.joint(a))); }

  // two bones that meet at a hinge (corporis.posing.skeleton.Skeleton.hinge); returns the axis, posed and at rest
  private hinge(W: Map<string, M3>, chain: string[], upper: V3, lower: V3, restAxis: V3, bent = 8): [V3, V3] {
    const [a, b, c] = chain, u0 = this.restDirection(a, b), l0 = this.restDirection(b, c);
    const u = unit(upper), low = unit(lower), sinBent = Math.sin(bent * Math.PI / 180);
    let n0 = unit(restAxis);
    const restCross = cross(u0, l0);
    if (norm(restCross) > sinBent) n0 = scale(unit(restCross), Math.sign(dot(restCross, n0)));
    const guess = apply(align(u0, u), n0);
    const cr = cross(u, low);
    let n: V3;
    if (norm(cr) > sinBent) { n = unit(cr); if (dot(n, guess) < 0) n = scale(n, -1); } else n = guess;
    W.set(a, mul(frame(u, n), transpose(frame(u0, n0))));
    W.set(b, mul(frame(low, n), transpose(frame(l0, n0))));
    return [n, n0];
  }

  // the world rotation of each posed bone
  solve(L: Landmarks): Map<string, M3> {
    const W = new Map<string, M3>();
    const pelvis = mul(pelvisFrame(L), transpose(this.restPelvis));
    const chest = mul(chestFrame(L), transpose(this.restChest));
    const head = mul(headFrame(L), transpose(this.restHead));
    W.set('root', pelvis);
    for (const [b, share] of SPINE) W.set(b, slerp(pelvis, chest, share));
    for (const [b, share] of NECK) W.set(b, slerp(chest, head, share));
    W.set('head', head);
    for (const [s, side] of SIDES) {
      const p = (n: string) => point(L, `${side}_${n}`);
      // the shoulder rises with the arm: a third of the arm's lift above 60 degrees, two thirds of it at the collarbone
      const arm = unit(sub(p('elbow'), p('shoulder'))), down = scale(apply(chest, [0, 0, 1]), -1);
      const lift = Math.acos(Math.min(1, Math.max(-1, dot(arm, down)))) * 180 / Math.PI;
      const rise = Math.max(0, lift - 60) / 3, forward = apply(chest, [0, 1, 0]), sign = s === '.L' ? -1 : 1;
      W.set('clavicle' + s, mul(axisAngle(forward, sign * rise * 2 / 3), chest));
      W.set('shoulder01' + s, mul(axisAngle(forward, sign * rise), chest));
      const u0 = this.restDirection('upperarm01' + s, 'lowerarm01' + s), f0 = this.restDirection('lowerarm01' + s, 'wrist' + s);
      this.hinge(W, ['upperarm01' + s, 'lowerarm01' + s, 'wrist' + s], sub(p('elbow'), p('shoulder')), sub(p('wrist'), p('elbow')), unit(cross(u0, f0)));
      const [axis, restAxis] = this.hinge(W, ['upperleg01' + s, 'lowerleg01' + s, 'foot' + s], sub(p('knee'), p('hip')), sub(p('ankle'), p('knee')), [1, 0, 0]);
      const r = (n: string) => point(this.restLandmarks, `${side}_${n}`);
      const restFoot = unit(sub(r('foot_index'), r('ankle'))), foot = unit(sub(p('foot_index'), p('ankle')));
      W.set('foot' + s, mul(frame(foot, axis), transpose(frame(restFoot, restAxis))));
      this.hand(W, L, s);
    }
    return W;
  }

  private hand(W: Map<string, M3>, L: Landmarks, s: string) {
    const R = mul(palmFrame(L, s), transpose(this.restHands[s]));
    W.set('wrist' + s, R);
    for (let k = 2; k <= 5; k++) W.set(`metacarpal${k - 1}${s}`, R);
    if (!L.hands[s]) return;
    // the fingers bend about the width of the hand, across the knuckles
    const restWidth: V3 = [this.restHands[s][1], this.restHands[s][4], this.restHands[s][7]], width = apply(R, restWidth);
    for (const [k, finger] of FINGERS) {
      const names = (k === 1 ? THUMB : KNUCKLES).map((n) => `${finger}_${n}`);
      for (let i = 1; i <= 3; i++) {
        const target = unit(sub(handPoint(L, s, names[i]), handPoint(L, s, names[i - 1])));
        const atRest = unit(sub(handPoint(this.restLandmarks, s, names[i]), handPoint(this.restLandmarks, s, names[i - 1])));
        W.set(`finger${k}-${i}${s}`, mul(frame(target, width), transpose(frame(atRest, restWidth))));
      }
    }
  }

  // anny's local-ref rotations (bones x 3 x 3): each posed bone relative to its nearest posed ancestor
  local(W: Map<string, M3>): M3[] {
    const { names, parents } = this.rest;
    return names.map((name, i) => {
      const Wb = W.get(name);
      if (!Wb) return I3.slice();
      let j = parents[i];
      while (j >= 0 && !W.has(names[j])) j = parents[j];
      return mul(transpose(j >= 0 ? W.get(names[j])! : I3), Wb);
    });
  }
}

// rotations of anny's frame in another frame: M R M^T for each rotation, as quaternions (x, y, z, w), one after another
export function quaternions(rotations: M3[], M: M3 = I3): Float32Array {
  const out = new Float32Array(rotations.length * 4), Mt = transpose(M);
  rotations.forEach((R, i) => out.set(quaternion(mul(mul(M, R), Mt)), i * 4));
  return out;
}

// the rotation matrix of a unit quaternion (x, y, z, w) at offset o
export function rotation(q: ArrayLike<number>, o = 0): M3 {
  const x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}

// the "bones" of a character card's pose (corporis.Character.pose): quaternions (x, y, z, w) in anny's frame for the
// bones that leave the rest pose, from quaternions q (bones x 4) in a frame that M turns anny's frame into
export function cardBones(q: ArrayLike<number>, names: string[], M: M3 = I3): Record<string, number[]> {
  const Mt = transpose(M), out: Record<string, number[]> = {};
  names.forEach((name, i) => {
    const R = mul(mul(Mt, rotation(q, i * 4)), M);
    if (R.every((v, k) => Math.abs(v - I3[k]) <= 1e-6)) return;
    out[name] = quaternion(R).map((v) => Math.round(v * 1e7) / 1e7);
  });
  return out;
}
