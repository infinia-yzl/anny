// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The physics of anny's hair (no three.js here; anny.hair.dynamics repeats it in NumPy). It moves the simulated guides
// of the scalp layout (the first guides of its progressive order), and the other guides take their motion through the
// weights of the layout (pass A). Each step of STEP seconds runs SUBSTEPS substeps:
//   1. Verlet integration with damping, under the gravity that the groom does not already hold: the groom rests in
//      balance with gravity as it acted in the head's rest frame, so only the change of gravity in the head's frame
//      acts (a sag-free rest, after Hsu et al. 2023), and the groom keeps its shape while the head is still;
//   2. the global shape constraints toward the posed groom (Han and Harada 2012), with a stiffness that falls from the
//      free start of the guide to its tip;
//   3. from the root to the tip, point by point: the local shape constraint (the bend of the groom at the previous
//      point, applied without a change of velocity), the colliders (capsules on the bones), and the length of the
//      segment with the velocity correction of the dynamic follow-the-leader method (Mueller, Kim and Chentanez 2012).
// The points before the free start of a guide follow the groom: the tie of a ponytail, or the part of long hair that
// lies on the head (the pivot of the groom); a guide without a pivot keeps its root only. Each point has its own
// radius for each collider, at most its distance from the collider at rest, so the groom at rest touches no collider
// (the nape lies closer to the cranium's centre than the crown).

export const STEP = 1 / 60;
export const SUBSTEPS = 1;
export const NO_PIVOT = 100;   // m: a pivot beyond this means that the guide lies on the head along its whole length

export interface SimParams {
  global_stiffness: [number, number];   // share of the way to the groom per substep, at the free start and at the tip
  local_stiffness: number;              // share of the way to the groom's bend per substep
  damping: number;                      // share of the velocity lost per substep
  gravity: number;                      // share of the change of gravity that acts
  ftl_damping?: number;                 // s_damping of the velocity correction
}
export const SIM_DEFAULTS: SimParams = { global_stiffness: [0.5, 0.05], local_stiffness: 0.5, damping: 0.08, gravity: 1, ftl_damping: 0.9 };

// capsules (a sphere is a capsule of two equal ends): 7 numbers each, the two ends and the radius
export type Capsules = ArrayLike<number>;

// the point on the axis of capsule q nearest to (x, y, z): the parameter t along the axis
function axisParam(caps: Capsules, q: number, x: number, y: number, z: number): number {
  const c0 = q * 7, ex = caps[c0 + 3] - caps[c0], ey = caps[c0 + 4] - caps[c0 + 1], ez = caps[c0 + 5] - caps[c0 + 2];
  const ee = ex * ex + ey * ey + ez * ez;
  return ee > 0 ? Math.min(1, Math.max(0, ((x - caps[c0]) * ex + (y - caps[c0 + 1]) * ey + (z - caps[c0 + 2]) * ez) / ee)) : 0;
}

export class HairSim {
  S: number; P: number;
  x: Float64Array; xp: Float64Array;   // positions and previous positions (S * P * 3)
  seg: Float64Array;                   // S: segment length
  free: Int32Array;                    // S: the index of the first free point
  kg: Float64Array;                    // S * P: global stiffness
  rad: Float64Array = new Float64Array(0);   // S * P * colliders: the radius of each collider for each point
  nc = 0; axes = new Float64Array(0); reach = new Float64Array(0);   // per collider: its axis, and its reach squared
  params: SimParams = SIM_DEFAULTS;
  margin = 0.0015;                     // m: the thickness of the hair over a collider

  constructor(S: number, P: number) {
    this.S = S; this.P = P;
    this.x = new Float64Array(S * P * 3); this.xp = new Float64Array(S * P * 3);
    this.seg = new Float64Array(S); this.free = new Int32Array(S); this.kg = new Float64Array(S * P);
  }

  // the groom: the rest points (S * P * 3; the lengths come from them), the pivot of each guide (m), the parameters
  // and the colliders at rest
  setGroom(rest: ArrayLike<number>, pivot: ArrayLike<number>, params: SimParams, caps: Capsules = []) {
    const { S, P } = this;
    this.params = params;
    const [k0, k1] = params.global_stiffness;
    for (let s = 0; s < S; s++) {
      let L = 0;
      for (let j = 1; j < P; j++) {
        const a = (s * P + j - 1) * 3, b = a + 3;
        const dx = rest[b] - rest[a], dy = rest[b + 1] - rest[a + 1], dz = rest[b + 2] - rest[a + 2];
        L += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      const seg = this.seg[s] = L / (P - 1);
      const f = seg < 1e-5 ? P : pivot[s] < NO_PIVOT ? Math.min(P, Math.max(1, Math.ceil(pivot[s] / seg - 1e-6))) : 1;
      this.free[s] = f;
      for (let j = 0; j < P; j++) {
        const t = f >= P - 1 ? 1 : Math.min(1, Math.max(0, (j - f) / (P - 1 - f)));
        this.kg[s * P + j] = j < f ? 1 : k0 + (k1 - k0) * t;
      }
    }
    // the radius of each collider for each point: the collider's own with the margin, or less than the point's
    // distance at rest
    const nc = this.nc = caps.length / 7;
    this.rad = new Float64Array(S * P * nc); this.axes = new Float64Array(nc * 10); this.reach = new Float64Array(nc);
    for (let i = 0; i < S * P; i++) {
      const x = rest[i * 3], y = rest[i * 3 + 1], z = rest[i * 3 + 2];
      for (let q = 0; q < nc; q++) {
        const c0 = q * 7, t = axisParam(caps, q, x, y, z);
        const dx = x - (caps[c0] + t * (caps[c0 + 3] - caps[c0])), dy = y - (caps[c0 + 1] + t * (caps[c0 + 4] - caps[c0 + 1]));
        const dz = z - (caps[c0 + 2] + t * (caps[c0 + 5] - caps[c0 + 2]));
        this.rad[i * nc + q] = Math.max(0, Math.min(caps[c0 + 6] + this.margin, Math.sqrt(dx * dx + dy * dy + dz * dz) - 0.0005));
      }
    }
    this.reset(rest);
  }

  reset(targets: ArrayLike<number>) {
    for (let i = 0; i < this.x.length; i++) { this.x[i] = targets[i]; this.xp[i] = targets[i]; }
  }

  // one step toward the posed groom (targets, S * P * 3) under the acceleration g (m/s^2) with the colliders in the
  // pose (the ends; the radii of setGroom hold); returns the largest speed of a free point in the step (m/s): its move
  // over the step (the constraints change the previous positions, so x - xp stays above zero for hair that rests
  // against a collider)
  step(targets: ArrayLike<number>, g: ArrayLike<number>, caps: Capsules): number {
    const { S, P, x, xp, kg } = this, prm = this.params, h = STEP / SUBSTEPS;
    const gx = g[0] * h * h, gy = g[1] * h * h, gz = g[2] * h * h;
    const keep = 1 - prm.damping, kl = prm.local_stiffness, sd = prm.ftl_damping ?? 0.9, nc = Math.min(this.nc, caps.length / 7), rad = this.rad;
    // the axis of each collider, and a sphere around it that no point outside can touch
    const C = this.axes;
    for (let q = 0; q < nc; q++) {
      const c0 = q * 7, ex = caps[c0 + 3] - caps[c0], ey = caps[c0 + 4] - caps[c0 + 1], ez = caps[c0 + 5] - caps[c0 + 2];
      const ee = ex * ex + ey * ey + ez * ez, br = 0.5 * Math.sqrt(ee) + caps[c0 + 6] + this.margin + 1e-4;
      C.set([caps[c0], caps[c0 + 1], caps[c0 + 2], ex, ey, ez, ee, caps[c0] + 0.5 * ex, caps[c0 + 1] + 0.5 * ey, caps[c0 + 2] + 0.5 * ez], q * 10);
      this.reach[q] = br * br;
    }
    let moved = 0;
    for (let sub = 0; sub < SUBSTEPS; sub++) moved = Math.max(moved, this.substep(targets, gx, gy, gz, keep, kl, sd, nc));
    return Math.sqrt(moved) / h;
  }

  // one substep (small functions keep the engine's optimised code stable); returns the largest squared move of a free
  // point
  private substep(targets: ArrayLike<number>, gx: number, gy: number, gz: number, keep: number, kl: number, sd: number, nc: number): number {
    const { S, P, x, xp, kg, rad, free, seg: SEG, reach } = this, C = this.axes;
    let moved = 0;
    // the points before the free start follow the groom
    for (let s = 0; s < S; s++) {
      const o = s * P * 3, f = free[s];
      for (let j = 0; j < f; j++) {
        const i = o + j * 3;
        x[i] = xp[i] = targets[i]; x[i + 1] = xp[i + 1] = targets[i + 1]; x[i + 2] = xp[i + 2] = targets[i + 2];
      }
    }
    // from the root to the tip, across the guides (their chains are independent, so the processor overlaps them).
    // The integration of a point needs its own state alone, so it runs in the same pass as the constraints of the
    // points before it.
    for (let j = 1; j < P; j++) {
      for (let s = 0; s < S; s++) {
        const f = free[s];
        if (j < f) continue;
        const i = (s * P + j) * 3, a = i - 3, seg = SEG[s];
        // 1-2. integration and the global shape
        const k = kg[s * P + j];
        const vx = (x[i] - xp[i]) * keep, vy = (x[i + 1] - xp[i + 1]) * keep, vz = (x[i + 2] - xp[i + 2]) * keep;
        xp[i] = x[i]; xp[i + 1] = x[i + 1]; xp[i + 2] = x[i + 2];
        const qx0 = x[i] + vx + gx, qy0 = x[i + 1] + vy + gy, qz0 = x[i + 2] + vz + gz;
        let px = qx0 + (targets[i] - qx0) * k, py = qy0 + (targets[i + 1] - qy0) * k, pz = qz0 + (targets[i + 2] - qz0) * k;
        // 3. the groom's segment, turned by the turn of the previous segment from the groom
        let rx = targets[i] - targets[a], ry = targets[i + 1] - targets[a + 1], rz = targets[i + 2] - targets[a + 2];
        if (j >= 2) {
          const b = a - 3;
          const ux = targets[a] - targets[b], uy = targets[a + 1] - targets[b + 1], uz = targets[a + 2] - targets[b + 2];
          const wx = x[a] - x[b], wy = x[a + 1] - x[b + 1], wz = x[a + 2] - x[b + 2];
          const uw = Math.sqrt((ux * ux + uy * uy + uz * uz) * (wx * wx + wy * wy + wz * wz));
          if (uw > 1e-18) {
            const inv = 1 / uw, c = (ux * wx + uy * wy + uz * wz) * inv;
            if (c > -0.99) {
              // rotate r by the turn u -> w: r c + a x r + a (a . r) / (1 + c), with a = u x w / (|u| |w|)
              const ax = (uy * wz - uz * wy) * inv, ay = (uz * wx - ux * wz) * inv, az = (ux * wy - uy * wx) * inv;
              const d = (ax * rx + ay * ry + az * rz) / (1 + c);
              const qx = rx * c + (ay * rz - az * ry) + ax * d;
              const qy = ry * c + (az * rx - ax * rz) + ay * d;
              const qz = rz * c + (ax * ry - ay * rx) + az * d;
              rx = qx; ry = qy; rz = qz;
            }
          }
        }
        // the pull moves the point without changing its velocity: its target turns with the strand, so as a force
        // (a follower force) it would pump energy into the strand
        const lx = (x[a] + rx - px) * kl, ly = (x[a + 1] + ry - py) * kl, lz = (x[a + 2] + rz - pz) * kl;
        px += lx; py += ly; pz += lz; xp[i] += lx; xp[i + 1] += ly; xp[i + 2] += lz;
        // the colliders
        for (let q = 0; q < nc; q++) {
          const c0 = q * 10, mx = px - C[c0 + 7], my = py - C[c0 + 8], mz = pz - C[c0 + 9];
          if (mx * mx + my * my + mz * mz > reach[q]) continue;
          const cx = C[c0], cy = C[c0 + 1], cz = C[c0 + 2], ex = C[c0 + 3], ey = C[c0 + 4], ez = C[c0 + 5], ee = C[c0 + 6];
          const t = ee > 0 ? Math.min(1, Math.max(0, ((px - cx) * ex + (py - cy) * ey + (pz - cz) * ez) / ee)) : 0;
          const dx = px - (cx + t * ex), dy = py - (cy + t * ey), dz = pz - (cz + t * ez);
          const dd = dx * dx + dy * dy + dz * dz, r = rad[(s * P + j) * nc + q];
          if (dd < r * r && dd > 1e-18) {
            const fct = r / Math.sqrt(dd) - 1;
            px += dx * fct; py += dy * fct; pz += dz * fct;
          }
        }
        // the length, and the velocity correction of the previous point
        const dx = px - x[a], dy = py - x[a + 1], dz = pz - x[a + 2];
        const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (l >= 1e-12) {
          const e = seg / l, nx = x[a] + dx * e, ny = x[a + 1] + dy * e, nz = x[a + 2] + dz * e;
          if (j - 1 >= f) { xp[a] += sd * (nx - px); xp[a + 1] += sd * (ny - py); xp[a + 2] += sd * (nz - pz); }
          px = nx; py = ny; pz = nz;
        }
        const mx = px - x[i], my = py - x[i + 1], mz = pz - x[i + 2];
        moved = Math.max(moved, mx * mx + my * my + mz * mz);
        x[i] = px; x[i + 1] = py; x[i + 2] = pz;
      }
    }
    return moved;
  }
}
