// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The colliders of the hair's physics (no three.js here): capsules on the bones of the head, the neck, the upper
// torso and the shoulders. Each end of a capsule sits at the rest head of a joint and follows the skin matrix of a
// bone. The radius fits inside the body at rest: a low quantile of the distances of the body's vertices from the
// capsule's axis. The solver then gives each point of the groom its own radius, at most its distance at rest
// (hair/sim.ts), so the colliders never push the groom away from its authored shape.

export interface ColliderDef {
  name: string;
  ends: [string, string];    // the joints (or named points) at the two ends; the same twice for a sphere
  bones: [string, string];   // the bone that carries each end
  radius?: number;           // m, anny's default body: a fixed radius (scaled with the body); otherwise fitted
  quantile?: number;         // the quantile of the vertex distances that sets the radius
  span?: [number, number];   // the part of the axis whose vertices set the radius
}

// the head is a sphere at the cranium's centre (the named point 'cranium'), and the groom's points near the nape and
// the ears set their own smaller radii
export const COLLIDERS: ColliderDef[] = [
  { name: 'head', ends: ['cranium', 'cranium'], bones: ['head', 'head'], radius: 0.12 },
  { name: 'neck', ends: ['neck01', 'head'], bones: ['neck01', 'head'], quantile: 0.08, span: [0.1, 0.8] },
  { name: 'chest', ends: ['spine02', 'neck01'], bones: ['spine02', 'spine01'], quantile: 0.04, span: [0.2, 0.9] },
  { name: 'shoulder.L', ends: ['clavicle.L', 'upperarm01.L'], bones: ['clavicle.L', 'shoulder01.L'], quantile: 0.08, span: [0.3, 0.95] },
  { name: 'shoulder.R', ends: ['clavicle.R', 'upperarm01.R'], bones: ['clavicle.R', 'shoulder01.R'], quantile: 0.08, span: [0.3, 0.95] },
];

export interface RestColliders {
  bones: Int32Array;     // 2 per capsule: the bone of each end
  rest: Float64Array;    // 7 per capsule: the ends at rest and the radius
}

// the capsules on the current body at rest: joints (the rest heads, 3 per bone), names (of the bones), named points
// (such as the cranium's centre), the body's vertices and their count, and the size of the head against anny's
// default body
export function fitColliders(defs: ColliderDef[], names: string[], joints: ArrayLike<number>, points: Record<string, number[]>,
  V: ArrayLike<number>, nV: number, scale: number): RestColliders {
  const at = (n: string) => points[n] || [0, 1, 2].map((k) => joints[names.indexOf(n) * 3 + k]);
  const found = defs.filter((d) => d.ends.every((n) => n in points || names.includes(n)) && d.bones.every((n) => names.includes(n)));
  const bones = new Int32Array(found.length * 2), rest = new Float64Array(found.length * 7);
  found.forEach((d, c) => {
    const A = at(d.ends[0]), B = at(d.ends[1]);
    bones[c * 2] = names.indexOf(d.bones[0]); bones[c * 2 + 1] = names.indexOf(d.bones[1]);
    let r = (d.radius ?? 0) * scale;
    if (d.radius === undefined) {
      const ex = B[0] - A[0], ey = B[1] - A[1], ez = B[2] - A[2], ee = ex * ex + ey * ey + ez * ez || 1;
      const [lo, hi] = d.span || [0, 1], dist: number[] = [];
      for (let v = 0; v < nV; v++) {
        const px = V[v * 3] - A[0], py = V[v * 3 + 1] - A[1], pz = V[v * 3 + 2] - A[2];
        const t = (px * ex + py * ey + pz * ez) / ee;
        if (t < lo || t > hi) continue;
        const dx = px - t * ex, dy = py - t * ey, dz = pz - t * ez, dd = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dd < 0.25 * scale) dist.push(dd);
      }
      dist.sort((a, b) => a - b);
      r = dist.length ? dist[Math.floor((d.quantile ?? 0.05) * (dist.length - 1))] : 0;
    }
    rest.set([...A, ...B, r], c * 7);
  });
  return { bones, rest };
}

// the capsules in the pose: bones holds the skin matrices (column-major 4 x 4 per bone, as three.js keeps them)
export function poseColliders(rc: RestColliders, rest: Float64Array, bones: ArrayLike<number>, out: Float64Array) {
  for (let c = 0; c < rest.length / 7; c++) {
    for (let e = 0; e < 2; e++) {
      const m = rc.bones[c * 2 + e] * 16, o = c * 7 + e * 3;
      const x = rest[o], y = rest[o + 1], z = rest[o + 2];
      for (let r = 0; r < 3; r++) out[o + r] = bones[m + r] * x + bones[m + 4 + r] * y + bones[m + 8 + r] * z + bones[m + 12 + r];
    }
    out[c * 7 + 6] = rest[c * 7 + 6];
  }
}
