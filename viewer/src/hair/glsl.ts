// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The shaders of anny's hair (anny.hair.styles repeats them in NumPy):
//   pass A: the guides follow the body (the frames of AnnyBody.followStrands, as anny.hair.StrandBinding with tips),
//           take the pose of their root and add the motion of the physics (blended from three simulated guides, as
//           anny.hair.styles.sim_offsets does); a texture row of P + 2 texels per guide: the points, the root normal,
//           and the direction of the first third (for the blend);
//   pass B: each render strand from its four guides, with its length, clumps, curl, waves, flyaways, flick and
//           volume; P texels per strand, the occlusion of the density volume in w (-1: no strand);
//   STRAND_VS: the ribbons of the strands, from the texels of pass B (HAIR_FS shades them).
// Every texture is addressed by a linear index i at (i % 2048, i / 2048).

const COMMON = /* glsl */`
precision highp float; precision highp int; precision highp usampler2D; precision highp sampler3D;
layout(location = 0) out highp vec4 outData;
#define TW 2048
vec4 at(sampler2D t, int i) { return texelFetch(t, ivec2(i % TW, i / TW), 0); }
uvec4 atu(usampler2D t, int i) { return texelFetch(t, ivec2(i % TW, i / TW), 0); }
int fragIndex() { return int(gl_FragCoord.y) * TW + int(gl_FragCoord.x); }
`;

// the guides of the style: their rest points follow the frames of the body; then the pose of their root
export const HAIR_PASS_A = /* glsl */`
${COMMON}
uniform highp sampler2D uLocal;    // per guide point: in the frame of the root triangle / scalp size, w: the tip follows
uniform highp sampler2D uTipLocal; // per guide point: in the frame of the tip triangle / scalp size
uniform highp sampler2D uFrames;   // 6 texels per guide: rows [scalp * frame | anchor] of the root and tip triangles
uniform highp sampler2D uMat;      // 3 texels per guide: the rows of the pose of its root
uniform highp sampler2D uMotion;   // per point of the simulated guides: the motion of the simulation (m)
uniform highp sampler2D uSimInfo;  // 2 texels per guide: its three simulated guides, and their weights
uniform highp sampler2D uGuideInfo; // 2 texels per guide (pass B): segment, ...; pivot
uniform int uP; uniform int uG; uniform int uMoving;
#define NO_PIVOT 100.0
// the motion of the simulated guides at point j of guide g; it starts at the pivot of the guide (or at its root) and
// grows over a centimetre (hair/sim.ts)
vec3 motion(int g, int j) {
  vec4 si = at(uSimInfo, g * 2), sw = at(uSimInfo, g * 2 + 1);
  vec3 m = sw.x * at(uMotion, int(si.x) * uP + j).xyz + sw.y * at(uMotion, int(si.y) * uP + j).xyz + sw.z * at(uMotion, int(si.z) * uP + j).xyz;
  float seg = at(uGuideInfo, g * 2).x, piv = at(uGuideInfo, g * 2 + 1).x;
  float start = piv < NO_PIVOT ? piv : 0.0;
  return m * clamp((float(j) * seg - start) / 0.01, 0.0, 1.0);
}
vec3 frameApply(int g, int base, vec3 L) {
  vec4 m0 = at(uFrames, g * 6 + base), m1 = at(uFrames, g * 6 + base + 1), m2 = at(uFrames, g * 6 + base + 2);
  return vec3(dot(m0.xyz, L) + m0.w, dot(m1.xyz, L) + m1.w, dot(m2.xyz, L) + m2.w);
}
vec3 restPoint(int g, int j) {
  vec4 L = at(uLocal, g * uP + j);
  vec3 p = frameApply(g, 0, L.xyz);
  if (L.w > 0.5) {
    float t = float(j) / float(uP - 1);
    p = mix(p, frameApply(g, 3, at(uTipLocal, g * uP + j).xyz), t * t);
  }
  return p;
}
vec3 pose(int g, vec3 p) {
  vec4 r0 = at(uMat, g * 3), r1 = at(uMat, g * 3 + 1), r2 = at(uMat, g * 3 + 2);
  return vec3(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}
vec3 poseDir(int g, vec3 d) {
  vec4 r0 = at(uMat, g * 3), r1 = at(uMat, g * 3 + 1), r2 = at(uMat, g * 3 + 2);
  return vec3(dot(r0.xyz, d), dot(r1.xyz, d), dot(r2.xyz, d));
}
void main() {
  int i = fragIndex(), C = uP + 2, g = i / C, j = i - g * C;
  if (g >= uG) { outData = vec4(0.0); return; }
  if (j < uP) {
    vec3 p = pose(g, restPoint(g, j));
    if (uMoving > 0) p += motion(g, j);
    outData = vec4(p, 0.0);
  } else if (j == uP) {
    vec4 m0 = at(uFrames, g * 6), m1 = at(uFrames, g * 6 + 1), m2 = at(uFrames, g * 6 + 2);
    outData = vec4(normalize(poseDir(g, vec3(m0.z, m1.z, m2.z))), 0.0);
  } else {
    outData = vec4(normalize(poseDir(g, restPoint(g, uP / 3) - restPoint(g, 0))), 0.0);
  }
}`;

export const HAIR_PASS_B = /* glsl */`
${COMMON}
uniform highp sampler2D uGuides;    // pass A
uniform highp sampler2D uGuideInfo; // two texels per guide: segment and default length (anny's default body), flick, group; pivot
uniform highp sampler2D uFrames;    // the rest frames of the guides (for the clump sectors)
uniform highp sampler2D uMat;       // the pose of the guide roots
uniform highp sampler2D uRootRest;  // per render root: rest position and packed normal
uniform highp usampler2D uRootData; // per render root: four guides, weights, chart
uniform highp sampler3D uOcc;       // the density volume (anny's default head): occlusion in r
uniform vec3 uOccLo; uniform vec3 uOccSize;
uniform mat4 uHeadInv;              // the current posed head -> anny's default head
uniform int uP; uniform int uCount;
uniform float uScale; uniform vec3 uHeadC;
uniform float uLength, uCurl, uVolume, uFadeShift;
uniform vec2 uJitter; uniform vec3 uThin; uniform vec2 uCoarse; uniform float uCoarseTip; uniform vec2 uFine; uniform vec2 uFinePow;
uniform float uSectors; uniform float uSectorR; uniform float uSimilarity;
uniform float uPivotBlend; // m: the turn to the root normal fades out over this arc before the pivot of a guide
uniform vec3 uCurlP;   // radius, period (m), ramp
uniform vec4 uFrizz;   // amplitude range (m), cycles range
uniform vec3 uFly;     // share, drift range (m)
uniform float uHLPhi[24]; uniform float uHLEl[24];
uniform float uHairline[19]; uniform float uFadeStart[19]; uniform vec4 uFade;   // width, clipper, top, on
#define PI 3.141592653589793

// random values (anny.hair.styles.rnd)
uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd(uint key, uint stream) { return float(pcg(key * 0x9E3779B9u ^ pcg(stream)) >> 8u) / 16777216.0; }
float int16(uint v) { int x = int(v & 0xFFFFu); return float(x >= 32768 ? x - 65536 : x); }
float sstep(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
// the style fields (anny.hair.chart)
float hairline(float phi) {
  float a = abs(phi), el = uHLEl[23];
  for (int k = 1; k < 24; k++) if (a <= uHLPhi[k]) { el = mix(uHLEl[k - 1], uHLEl[k], (a - uHLPhi[k - 1]) / (uHLPhi[k] - uHLPhi[k - 1])); break; }
  float f = clamp(a / 10.0, 0.0, 18.0); int k = min(int(f), 17);
  return el + mix(uHairline[k], uHairline[k + 1], f - float(k));
}
float fadeLength(float phi, float el) {
  if (uFade.w < 0.5) return 1e9;
  float a = abs(phi), f = clamp(a / 10.0, 0.0, 18.0); int k = min(int(f), 17);
  float lo = mix(uFadeStart[k], uFadeStart[k + 1], f - float(k)) + uFadeShift;
  float u = clamp((el - hairline(phi) - lo) / uFade.x, 0.0, 30.0);
  return uFade.y * pow(uFade.z / uFade.y, u);
}
vec3 rotBetween(vec3 u, vec3 v, vec3 x) {
  float c = dot(u, v); vec3 a = cross(u, v);
  return x * c + cross(a, x) + a * dot(a, x) / (1.0 + c);
}
vec3 unpackNormal(float w) {
  uint p = uint(round(w));   // w holds an exact integer (w + 0.5 would round in float32 above 2^23)
  float x = float(p & 4095u) / 4095.0 * 2.0 - 1.0, y = float(p >> 12u) / 4095.0 * 2.0 - 1.0;
  float z = 1.0 - abs(x) - abs(y), t = max(-z, 0.0);
  x += x >= 0.0 ? -t : t; y += y >= 0.0 ? -t : t;
  return normalize(vec3(x, y, z));
}
vec3 pose(int g, vec3 p) {
  vec4 r0 = at(uMat, g * 3), r1 = at(uMat, g * 3 + 1), r2 = at(uMat, g * 3 + 2);
  return vec3(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}
vec3 poseDir(int g, vec3 d) {
  vec4 r0 = at(uMat, g * 3), r1 = at(uMat, g * 3 + 1), r2 = at(uMat, g * 3 + 2);
  return vec3(dot(r0.xyz, d), dot(r1.xyz, d), dot(r2.xyz, d));
}
vec3 guidePoint(int g, int j) { return at(uGuides, g * (uP + 2) + j).xyz; }

void main() {
  int i = fragIndex(), r = i / uP, j = i - r * uP;
  if (r >= uCount) { outData = vec4(0.0, 0.0, 0.0, -1.0); return; }
  uvec4 rd = atu(uRootData, r);
  int gid[4] = int[4](int(rd.x & 0xFFFFu), int(rd.x >> 16u), int(rd.y & 0xFFFFu), int(rd.y >> 16u));
  float w[4];
  float phi = int16(rd.w) * 0.01, el = int16(rd.w >> 16u) * 0.01;
  uint key = uint(r) * 4u;
  // the hairline: a share of the roots near it carries no strand
  float cover = sstep(-0.8, 3.0, el - hairline(phi));
  if (!(rnd(key, 4u) < cover)) { outData = vec4(0.0, 0.0, 0.0, -1.0); return; }
  // guides of another group, or that flow another way, do not blend in
  vec4 info[4]; vec3 gn[4]; vec3 g0[4];
  vec3 dir0 = at(uGuides, gid[0] * (uP + 2) + uP + 1).xyz;
  float wsum = 0.0;
  for (int k = 0; k < 4; k++) {
    info[k] = at(uGuideInfo, gid[k] * 2);
    gn[k] = at(uGuides, gid[k] * (uP + 2) + uP).xyz;
    g0[k] = guidePoint(gid[k], 0);
    w[k] = float((rd.z >> (8u * uint(k))) & 0xFFu) / 255.0;
    if (k > 0) {
      vec3 dk = at(uGuides, gid[k] * (uP + 2) + uP + 1).xyz;
      if (info[k].w != info[0].w || !(dot(dk, dir0) > uSimilarity)) w[k] = 0.0;
    }
    wsum += w[k];
  }
  // the length
  float jit = mix(uJitter.x, uJitter.y, rnd(key, 1u));
  if (rnd(key, 2u) < uThin.x) jit *= mix(uThin.y, uThin.z, rnd(key, 3u));
  float lsum = 0.0, asum = 0.0;
  for (int k = 0; k < 4; k++) { w[k] /= wsum; lsum += w[k] * info[k].y; asum += w[k] * info[k].x * float(uP - 1); }
  float ell = min(uLength * lsum * jit * uScale, asum * uScale);
  ell = min(ell, fadeLength(phi, el) * uScale);
  // strands under 0.1 mm are left out (the skin shade of the density volume shows them)
  if (ell <= 1e-4 * uScale) { outData = vec4(0.0, 0.0, 0.0, -1.0); return; }
  float t = float(j) / float(uP - 1), s = ell * t;
  // the root, posed with its nearest guide
  vec4 rr = at(uRootRest, r);
  vec3 xr = pose(gid[0], rr.xyz), nr = normalize(poseDir(gid[0], unpackNormal(rr.w)));
  // the blended offsets of the guides, turned to the root normal
  vec3 base = xr, tng = vec3(0.0);
  float lift = 0.0, turn = 0.0;
  for (int k = 0; k < 4; k++) {
    if (w[k] <= 0.0) continue;
    float piv = at(uGuideInfo, gid[k] * 2 + 1).x * uScale;
    float rho = 1.0 - sstep(piv - uPivotBlend * uScale, piv, s);
    float seg = info[k].x * uScale, u = s / max(seg, 1e-12);
    int i0 = clamp(int(floor(u)), 0, uP - 2);
    float f = u - float(i0);
    vec3 a = guidePoint(gid[k], i0), b = guidePoint(gid[k], i0 + 1), st = b - a, off = a + f * st - g0[k], dn = normalize(st);
    base += w[k] * mix(off, rotBetween(gn[k], nr, off), rho);
    tng += w[k] * mix(dn, rotBetween(gn[k], nr, dn), rho);
    lift += w[k] * (length(a + f * st - uHeadC) - length(g0[k] - uHeadC));
    turn += w[k] * rho;
  }
  tng = normalize(tng);
  // on the head, each point keeps the height of the guides over the root (anny.hair.styles.strands)
  vec3 hrel = base - uHeadC;
  float hrad = length(hrel);
  base = uHeadC + hrel * (1.0 + turn * ((length(xr - uHeadC) + lift) / max(hrad, 1e-9) - 1.0));
  // clumps: the lateral offset of the root from its guide shrinks along the strand, first toward the centre of its
  // sector (fine clumps), then toward the guide (coarse clumps); the sectors are found at rest
  vec4 m0 = at(uFrames, gid[0] * 6), m1 = at(uFrames, gid[0] * 6 + 1), m2 = at(uFrames, gid[0] * 6 + 2);
  vec3 n0 = normalize(vec3(m0.z, m1.z, m2.z)), d = rr.xyz - vec3(m0.w, m1.w, m2.w);
  d -= dot(d, n0) * n0;
  vec3 ref = abs(n0.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  vec3 ta = normalize(cross(ref, n0)), tb = cross(n0, ta);
  float S = uSectors, ang = atan(dot(d, tb), dot(d, ta));
  float q = clamp(floor((ang / (2.0 * PI) + 0.5) * S), 0.0, S - 1.0), qa = (q + 0.5) / S * 2.0 * PI - PI;
  vec3 centre = poseDir(gid[0], (cos(qa) * ta + sin(qa) * tb) * uSectorR * uScale);
  d = poseDir(gid[0], d);
  uint skey = (uint(gid[0]) * uint(S) + uint(q)) * 4u + 2u, gkey = uint(gid[0]) * 4u + 1u;
  float fs = mix(uFine.x, uFine.y, rnd(skey, 1u)), fp = mix(uFinePow.x, uFinePow.y, rnd(skey, 2u));
  float cs = mix(uCoarse.x, uCoarse.y, rnd(gkey, 1u));
  float pf = fs * pow(t, fp), pc = cs * pow(sstep(0.0, 0.5, t), 0.9) + (1.0 - cs) * uCoarseTip * pow(t, 2.2);
  vec3 lateral = (centre + (d - centre) * (1.0 - pf)) * (1.0 - pc);
  vec3 p = base - (d - lateral);
  // a frame along the strand
  vec3 outward = normalize(p - uHeadC), b1 = normalize(cross(tng, outward)), b2 = cross(tng, b1);
  // curl: a helix about the strand, in phase within a fine clump
  float rad = uCurl * uCurlP.x * uScale * mix(0.8, 1.2, rnd(key, 5u));
  if (rad > 0.0) {
    float per = uCurlP.y * uScale * mix(0.85, 1.15, rnd(skey, 3u));
    float th = 2.0 * PI * s / per + 2.0 * PI * (rnd(skey, 4u) + 0.08 * rnd(key, 6u));
    p += sstep(0.0, uCurlP.z, t) * rad * (cos(th) * b1 + sin(th) * b2);
  }
  // small waves
  float amp = mix(uFrizz.x, uFrizz.y, rnd(key, 7u)) * uScale * clamp(ell / (0.03 * uScale), 0.2, 1.0);
  float cyc = mix(uFrizz.z, uFrizz.w, rnd(key, 8u));
  float p1 = 2.0 * PI * rnd(key, 9u), p2 = 2.0 * PI * rnd(key, 10u);
  p += (sin(2.0 * PI * cyc * t + p1) * b1 + sin(2.0 * PI * cyc * 0.7 * t + p2) * b2) * amp * pow(t, 1.3);
  // flyaways
  if (rnd(key, 11u) < uFly.x && ell > 0.025 * uScale) {
    vec3 drift = normalize(xr - uHeadC) * mix(uFly.y, uFly.z, rnd(key, 12u));
    drift += (vec3(rnd(key, 13u), rnd(key, 14u), rnd(key, 15u)) - 0.5) * 0.0036;
    p += drift * uScale * t * t;
  }
  // flick: the ends of the perimeter pieces turn outward
  float fk = sstep(0.6, 1.0, t);
  p += outward * info[0].z * uScale * fk * fk;
  // volume: points move away from the head centre in proportion to their height above the root
  vec3 rel = p - uHeadC;
  float rho = length(rel), rho0 = length(xr - uHeadC);
  p = uHeadC + rel * (1.0 + (uVolume - 1.0) * max(rho - rho0, 0.0) / max(rho, 1e-9));
  // occlusion from the density volume, on anny's default head
  vec3 q0 = (uHeadInv * vec4(p, 1.0)).xyz;
  float occ = texture(uOcc, (q0 - uOccLo) / uOccSize).r;
  outData = vec4(p, occ);
}`;

// ribbons facing the camera, one instance per strand, from the points of pass B
export const STRAND_VS = /* glsl */`
precision highp float; precision highp int;
#define TW 2048
uniform highp sampler2D uPoints;
uniform int uP;
uniform float uWidth; uniform float uTipWidth; uniform float uViewportH; uniform float uMinPix;
varying vec3 vWPos; varying vec3 vT; varying vec4 vH; varying float vCov;
vec4 pointAt(int r, int j) { int i = r * uP + j; return texelFetch(uPoints, ivec2(i % TW, i / TW), 0); }
float hash11(uint v) { v = v * 747796405u + 2891336453u; uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u; return float(((w >> 22u) ^ w) >> 8u) / 16777216.0; }
void main() {
  int r = gl_InstanceID, j = gl_VertexID >> 1;
  float side = float(gl_VertexID & 1);
  vec4 pt = pointAt(r, j);
  if (pt.w < 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vCov = 0.0; return; }
  vec3 pa = pointAt(r, max(j - 1, 0)).xyz, pb = pointAt(r, min(j + 1, uP - 1)).xyz;
  vec3 T = pb - pa;
  float tl = length(T);
  if (tl < 1e-9) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vCov = 0.0; return; }
  T /= tl;
  vec3 wp = pt.xyz;
  bool ortho = projectionMatrix[3][3] > 0.5;
  vec3 V = ortho ? normalize(vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2])) : normalize(cameraPosition - wp);
  vec3 sd = cross(T, V);
  float sl = length(sd);
  sd = sl > 1e-4 ? sd / sl : normalize(cross(T, vec3(0.0, 1.0, 0.0)));
  float t = float(j) / float(uP - 1);
  float w = mix(uWidth, uTipWidth, t * t);
  vec4 clip = projectionMatrix * viewMatrix * vec4(wp, 1.0);
  float pix = ortho ? 2.0 / (projectionMatrix[1][1] * uViewportH) : clip.w * 2.0 / (projectionMatrix[1][1] * uViewportH);
  float wEff = max(w, pix * uMinPix);
  vCov = w / wEff;
  wp += sd * (side * 2.0 - 1.0) * wEff * 0.5;
  vWPos = wp; vT = T; vH = vec4(t, side, hash11(uint(r)), pt.w);
  gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
}`;
