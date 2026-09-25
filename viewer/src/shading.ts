// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The shaders of the viewer, carried over from the legacy 3D Model viewer (web/app.js): the skin with its
// screen-space scattering, the two specular lobes, the light through thin parts and the procedural pores,
// veins and vellus sheen; the eyes; the hair; the backdrop and the props. The hair vertex shader places each
// point from the frame of its strand on the current body (AnnyBody.followStrands).

export const BG_GLSL = /* glsl */`
uniform vec3 uBgA; uniform vec3 uBgB; uniform vec2 uBgCenter; uniform vec2 uRes;
vec3 bgColor() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 d = (uv - uBgCenter) * vec2(uRes.x / uRes.y, 1.0);
  float r = length(d * vec2(0.8, 1.0));
  return mix(uBgA, uBgB, smoothstep(0.0, 0.75, r));
}`;

export const HEAD_OUT_GLSL = /* glsl */`
vec3 headOutRest(vec3 p) { float cy = p.y > 0.43 ? min(p.y, 0.55) : p.y; return normalize(p - vec3(0.0, cy, 0.02) + vec3(0.0, 1e-4, 0.0)); }`;

// skinning (linear blend) for the custom vertex shaders; three.js defines USE_SKINNING for skinned meshes
export const SKIN_GLSL = /* glsl */`
#include <skinning_pars_vertex>
#ifdef USE_SKINNING
#ifdef SKIN8
attribute vec4 skinIndex2; attribute vec4 skinWeight2;   // the body and the wearables use eight bones per vertex
#endif
mat4 skinMat() {
  mat4 m = skinWeight.x * getBoneMatrix(skinIndex.x) + skinWeight.y * getBoneMatrix(skinIndex.y)
         + skinWeight.z * getBoneMatrix(skinIndex.z) + skinWeight.w * getBoneMatrix(skinIndex.w);
#ifdef SKIN8
  m += skinWeight2.x * getBoneMatrix(skinIndex2.x) + skinWeight2.y * getBoneMatrix(skinIndex2.y)
     + skinWeight2.z * getBoneMatrix(skinIndex2.z) + skinWeight2.w * getBoneMatrix(skinIndex2.w);
#endif
  return bindMatrixInverse * m * bindMatrix;
}
#else
mat4 skinMat() { return mat4(1.0); }
#endif`;

export const GLSL_COMMON = /* glsl */`
#define PI 3.141592653589793
uniform vec3 uKeyDir; uniform vec3 uKeyColor; uniform float uKeyTan;
uniform vec3 uRimDir; uniform vec3 uRimColor;
uniform vec3 uSH[9];
uniform sampler2D uEnv; uniform sampler2D uEnvKey; uniform float uEnvSpec; uniform float uEnvDiff;
uniform float uFrame;
uniform sampler2D uShadowRaw; uniform sampler2DShadow uShadowCmp; uniform sampler2D uHairShadow; uniform mat4 uShadowMat; uniform vec4 uShadowP;
uniform mat3 uShadowRot;
uniform sampler2DShadow uBodyShadowCmp; uniform mat4 uBodyShadowMat; uniform vec4 uBodyShadowP; uniform sampler2D uBodyShadowRaw;
uniform float uHairShadowK; uniform float uMannequin;
uniform mat4 uHeadInv; uniform mat3 uHeadRot;
#include <cube_uv_reflection_fragment>
#include <packing>
vec3 shIrr(vec3 n) {
  return uSH[0] + uSH[1]*n.y + uSH[2]*n.z + uSH[3]*n.x + uSH[4]*(n.x*n.y) + uSH[5]*(n.y*n.z) + uSH[6]*(3.0*n.z*n.z-1.0) + uSH[7]*(n.x*n.z) + uSH[8]*(n.x*n.x-n.y*n.y);
}
float D_GGX(float NoH, float a) { float a2 = a*a; float d = NoH*NoH*(a2-1.0)+1.0; return a2/(PI*d*d); }
float V_Smith(float NoV, float NoL, float a) { float gv = NoL*(NoV*(1.0-a)+a); float gl = NoV*(NoL*(1.0-a)+a); return 0.5/max(gv+gl, 1e-5); }
float F_Schlick(float f0, float VoH) { float f = pow(1.0-VoH, 5.0); return f0 + (1.0-f0)*f; }
float specSphere(vec3 N, vec3 V, vec3 L, float rough, float f0, float tanR) {
  float a = rough*rough; float a2 = clamp(a + 0.5*tanR, 0.0, 1.0);
  vec3 H = normalize(V+L);
  float NoL = max(dot(N,L),0.0), NoV = max(dot(N,V),0.03), NoH = max(dot(N,H),0.0), VoH = max(dot(V,H),0.0);
  return D_GGX(NoH, a2) * V_Smith(NoV, NoL, a2) * F_Schlick(f0, VoH) * NoL;
}
vec2 envBRDF(float rough, float NoV) {
  const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
  const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
  vec4 r = rough*c0 + c1;
  float a004 = min(r.x*r.x, exp2(-9.28*NoV))*r.x + r.y;
  return vec2(-1.04, 1.04)*a004 + r.zw;
}
vec3 envSpecular(vec3 N, vec3 V, float rough, float f0) {
  vec3 R = reflect(-V, N);
  R = normalize(mix(R, N, rough*rough));
  vec3 pre = textureCubeUV(uEnv, R, rough).rgb;
  vec2 ab = envBRDF(rough, max(dot(N,V),1e-4));
  return pre * (f0*ab.x + ab.y);
}
// reflections that head back into the skull are blocked by the head itself (sphere approximation).
// headOutRest works in the rest pose; the skinned surfaces carry it through their own bones (vHOut), and the parts
// that move with the head (the eyes) map their position back into the head's rest pose.
${HEAD_OUT_GLSL}
vec3 headOut(vec3 wp) { return normalize(uHeadRot * headOutRest((uHeadInv * vec4(wp, 1.0)).xyz)); }
float headEnvOccDir(vec3 hOut, vec3 N, vec3 V) {
  return smoothstep(-0.35, 0.15, dot(reflect(-V, N), hOut));
}
float headEnvOcc(vec3 wp, vec3 N, vec3 V) { return headEnvOccDir(headOut(wp), N, V); }
float ign(vec2 p) { p += uFrame * 5.588238; return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
const vec2 PD[16] = vec2[16](
  vec2(-0.94201624,-0.39906216), vec2(0.94558609,-0.76890725), vec2(-0.09418410,-0.92938870), vec2(0.34495938,0.29387760),
  vec2(-0.91588581,0.45771432), vec2(-0.81544232,-0.87912464), vec2(-0.38277543,0.27676845), vec2(0.97484398,0.75648379),
  vec2(0.44323325,-0.97511554), vec2(0.53742981,-0.47373420), vec2(-0.26496911,-0.41893023), vec2(0.79197514,0.19090188),
  vec2(-0.24188840,0.99706507), vec2(-0.81409955,0.91437590), vec2(0.19984126,0.78641367), vec2(0.14383161,-0.14100790));
// whole-body cascade: plain rotated-disk PCF (the head cascade handles the fine detail above the neck)
float rawDepthBody(vec2 uv);
// height in the head's rest pose: the head cascade covers the head wherever the pose puts it
float headY(vec3 wp) { return (uHeadInv * vec4(wp, 1.0)).y; }
// (with a blocker search, so shadows from parts further away, such as a hand in front of the body, get softer)
float bodyShadow(vec3 wp, vec3 n) {
  vec3 p = wp + n * 0.002 + normalize(uKeyDir) * 0.001;
  vec4 s = uBodyShadowMat * vec4(p, 1.0);
  vec3 sc = s.xyz / s.w;
  if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return 1.0;
  float ang = ign(gl_FragCoord.xy + 17.0) * 6.2831853;
  mat2 rot = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
  float uvm = length(vec3(uBodyShadowMat[0][0], uBodyShadowMat[1][0], uBodyShadowMat[2][0]));   // uv per metre
  float range = uBodyShadowP.w;
  vec3 nl = uShadowRot * n;
  vec2 dzduv = clamp(nl.xy / (uvm * max(nl.z, 0.25) * range), -8.0, 8.0);
  float z = sc.z - uBodyShadowP.y;
  float r0 = uBodyShadowP.x * uBodyShadowP.z;
  float searchR = max(r0, uvm * 0.4 * uKeyTan);
  float bsum = 0.0, bn = 0.0;
  for (int i = 0; i < 9; i++) {
    vec2 o = i == 8 ? vec2(0.0) : rot * PD[i * 2 + 1] * searchR;
    float zr = z + dot(dzduv, o);
    float d = rawDepthBody(sc.xy + o);
    if (d < zr - 0.0002) { bsum += zr - d; bn += 1.0; }
  }
  if (bn < 0.5) return 1.0;
  float pen = clamp(uvm * (bsum / bn * range) * uKeyTan, r0, searchR);
  float lit = 0.0;
  for (int i = 0; i < 12; i++) {
    vec2 o = rot * PD[i] * pen;
    lit += texture(uBodyShadowCmp, vec3(sc.xy + o, z + dot(dzduv, o)));
  }
  return lit / 12.0;
}
float rawDepth(vec2 uv) {
#ifdef SHADOW_PACKED
  return unpackRGBAToDepth(texture2D(uShadowRaw, uv));
#else
  return texture2D(uShadowRaw, uv).x;
#endif
}
float rawDepthBody(vec2 uv) {
#ifdef SHADOW_PACKED
  return unpackRGBAToDepth(texture2D(uBodyShadowRaw, uv));
#else
  return texture2D(uBodyShadowRaw, uv).x;
#endif
}
// how far (m) the key light travels inside the body before it reaches this point, read from the shadow maps
float keyThickness(vec3 wp, vec3 n) {
  vec3 p = wp - n * 0.0004;
  float wHead = smoothstep(0.345, 0.375, headY(wp));
  float ang = ign(gl_FragCoord.xy + 5.0) * 6.2831853;
  mat2 rot = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
  float tHead = 1.0, tBody = 1.0;
  if (wHead > 0.0) {
    vec4 s = uShadowMat * vec4(p, 1.0); vec3 sc = s.xyz / s.w;
    if (sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0) {
      tHead = 0.0;
      for (int i = 0; i < 4; i++) tHead += max(sc.z - rawDepth(sc.xy + rot * PD[i * 4 + 2] * uShadowP.x * 2.0), 0.0);
      tHead *= 0.25 * uShadowP.z;
    }
  }
  if (wHead < 1.0) {
    vec4 s = uBodyShadowMat * vec4(p, 1.0); vec3 sc = s.xyz / s.w;
    if (sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0) {
      tBody = 0.0;
      for (int i = 0; i < 4; i++) tBody += max(sc.z - rawDepthBody(sc.xy + rot * PD[i * 4 + 2] * uBodyShadowP.x * 1.5), 0.0);
      tBody *= 0.25 * uBodyShadowP.w;
    }
  }
  return mix(tBody, tHead, wHead);
}
// light that crosses a thin layer of skin (s in mm): red travels furthest
vec3 skinTransmit(float s) { return vec3(0.35, 0.16, 0.10) * exp(-s / vec3(3.6, 1.8, 1.2)); }
// PCSS soft shadow from the key light + deep-opacity hair transmittance. returns (opaque vis, hair T)
vec2 keyShadowHead(vec3 wp, vec3 n, float soft);
vec2 keyShadow(vec3 wp, vec3 n, float soft) {
  float wHead = smoothstep(0.345, 0.375, headY(wp));
  float vb = wHead < 1.0 ? bodyShadow(wp, n) : 1.0;
  if (wHead <= 0.0) return vec2(vb, 1.0);
  vec2 h = keyShadowHead(wp, n, soft);
  if (h.x < -0.5) h = vec2(vb, 1.0);
  return vec2(mix(vb, h.x, wHead), mix(1.0, h.y, wHead));
}
vec2 keyShadowHead(vec3 wp, vec3 n, float soft) {
  vec3 p = wp + n * uShadowP.w * 1.5;
  vec4 s = uShadowMat * vec4(p, 1.0);
  vec3 sc = s.xyz / s.w;
  if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return vec2(-1.0, 1.0);
  vec3 nl = uShadowRot * n;
  float W = uShadowP.w / uShadowP.x;
  vec2 dzduv = clamp(nl.xy * W / (max(nl.z, 0.25) * uShadowP.z), -8.0, 8.0);
  float z = sc.z - 0.00006;
  float ang = ign(gl_FragCoord.xy) * 6.2831853;
  mat2 rot = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
  float searchR = uShadowP.y * (0.025 / uShadowP.z);
  float bsum = 0.0, bn = 0.0;
  for (int i = 0; i < 16; i++) {
    vec2 o = rot * PD[i] * searchR;
    float zr = z + dot(dzduv, o);
    float d = rawDepth(sc.xy + o);
    if (d < zr - 0.0001) { bsum += zr - d; bn += 1.0; }
  }
  float vis = 1.0;
  if (bn > 0.5) {
    float dz = bsum / bn;
    float pen = clamp(dz * uShadowP.y * soft, uShadowP.x * 1.0, searchR);
    float lit = 0.0;
    for (int i = 0; i < 16; i++) {
      vec2 o = rot * PD[i] * pen;
      lit += texture(uShadowCmp, vec3(sc.xy + o, z + dot(dzduv, o)));
    }
    vis = lit / 16.0;
  }
  // hair: soft deep-opacity transmittance, filter radius grows with distance to the hair layer
  float dh0 = texture2D(uHairShadow, sc.xy).x;
  float dzh = max(0.0, z - dh0) * uShadowP.z;
  float hr = clamp(dzh * uShadowP.y / uShadowP.z * 1.0, uShadowP.x * 6.0, uShadowP.x * 24.0);
  float T = 0.0;
  for (int i = 0; i < 12; i++) {
    vec2 o = rot * PD[i] * hr;
    float dh = texture2D(uHairShadow, sc.xy + o).x;
    T += exp(-420.0 * uHairShadowK * max(0.0, z + dot(dzduv, o) - dh) * uShadowP.z);
  }
  T /= 12.0;
  return vec2(vis, T);
}
// cheaper variant for hair fragments (heavy overdraw)
vec2 keyShadowLite(vec3 wp, vec3 n) {
  vec3 p = wp + n * uShadowP.w * 2.0;
  vec4 s = uShadowMat * vec4(p, 1.0);
  vec3 sc = s.xyz / s.w;
  if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return vec2(1.0);
  float z = sc.z - 0.0002;
  float ang = ign(gl_FragCoord.xy) * 6.2831853;
  mat2 rot = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
  float r = uShadowP.x * 5.0;
  float lit = 0.0, T = 0.0;
  for (int i = 0; i < 4; i++) {
    vec2 o = rot * PD[i * 4 + 1] * r;
    lit += texture(uShadowCmp, vec3(sc.xy + o, z));
    float dh = texture2D(uHairShadow, sc.xy + o * 0.6).x;
    T += exp(-700.0 * uHairShadowK * max(0.0, z - dh) * uShadowP.z);
  }
  return vec2(lit * 0.25, T * 0.25);
}
`;

// Diffuse light goes to a second render target and is spread by a screen-space diffusion pass (SSS_SCREEN).
// Without SSS_SCREEN the shader falls back to the pre-integrated curvature LUT.
// skin under a worn garment sinks a few millimetres, so bends of the body cannot push it through the fabric; the
// sinking fades out over a few millimetres past the edge, where an elastic band would press into the skin
export const COVER_GLSL = /* glsl */`
vec3 coveredPosition() {
  vec4 c = attrV * uWearOn;
  return position - normal * (0.004 * max(max(c.x, c.y), max(c.z, c.w)));
}`;

export const SKIN_VS = /* glsl */`
${SKIN_GLSL}
${HEAD_OUT_GLSL}
attribute vec3 nsmooth; attribute vec3 albedo; attribute vec4 attrA; attribute vec4 attrB; attribute vec4 attrC; attribute vec4 attrD; attribute vec4 attrW; attribute vec4 attrV;
attribute vec3 rest;   // rest shape: skin detail, moles and creases stay fixed to the surface while the body shape changes
uniform vec4 uWearOn;
varying vec3 vWPos; varying vec3 vN; varying vec3 vNs; varying vec3 vAlb; varying vec3 vObj; varying vec4 vA; varying vec4 vB; varying vec4 vC; varying vec4 vD; varying vec3 vAx;
varying vec3 vN0; varying vec3 vR0; varying vec3 vR1; varying vec3 vR2; varying vec3 vHOut;
${COVER_GLSL}
void main() {
  mat4 S = skinMat();
  mat3 M = mat3(modelMatrix) * mat3(S);
  vec4 wp = modelMatrix * (S * vec4(coveredPosition(), 1.0));
  vWPos = wp.xyz; vObj = rest;
  vN = normalize(M * normal);
  vNs = normalize(M * nsmooth);
  // the rest normal and the turn of the skin: surface detail is worked out in the rest pose and turned with the skin
  vN0 = normal; vR0 = M[0]; vR1 = M[1]; vR2 = M[2];
  vHOut = M * headOutRest(rest);   // the head of anny's default body (uHeadInv maps to it)
  vAlb = pow(albedo, vec3(2.2));
  vA = attrA; vB = attrB; vC = attrC; vD = attrD;
  // contact shade from the wearables that are on (one slot each)
  vec4 wz = mix(vec4(1.0), attrW, uWearOn);
  vA.x *= wz.x * wz.y * wz.z * wz.w;
  // main direction of the limb under this point (veins and skin lines run along it)
  vec3 ax = vec3(0.0, 1.0, 0.0);
  if (abs(rest.x) > 0.19 && rest.y > -0.3) ax = normalize(vec3(sign(rest.x) * 0.52, -0.5, 0.73));
  if (abs(rest.x) > 0.3) ax = normalize(vec3(sign(rest.x) * 0.45, -0.35, 0.8));
  if (rest.y < -0.75) ax = vec3(0.0, 0.0, 1.0);
  vAx = ax;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

export const SKIN_FS = /* glsl */`
${GLSL_COMMON}
${BG_GLSL}
layout(location = 0) out highp vec4 outColor;
layout(location = 1) out highp vec4 outSSS;
uniform sampler2D uDetail; uniform float uDetailScale; uniform float uDetailAmt;
uniform sampler2D uPores; uniform float uPoreScale; uniform float uPoreAmt;
uniform sampler2D uLUT; uniform float uCurvScale;
uniform vec4 uMoleA[16]; uniform vec4 uMoleB[16];
uniform float uTrans; uniform float uFuzz; uniform float uUnd;
uniform vec3 uSkinTint; uniform vec3 uScalpCol; uniform vec3 uScalp0; uniform float uMelanin;
uniform vec4 uKnA[28]; uniform vec4 uKnB[28];
uniform vec4 uEyeL; uniform vec4 uEyeR;
varying vec3 vWPos; varying vec3 vN; varying vec3 vNs; varying vec3 vAlb; varying vec3 vObj; varying vec4 vA; varying vec4 vB; varying vec4 vC; varying vec4 vD; varying vec3 vAx;
varying vec3 vN0; varying vec3 vR0; varying vec3 vR1; varying vec3 vR2; varying vec3 vHOut;
float h31(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
float vn3(vec3 p) { vec3 i = floor(p); vec3 f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(h31(i), h31(i + vec3(1,0,0)), f.x), mix(h31(i + vec3(0,1,0)), h31(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(h31(i + vec3(0,0,1)), h31(i + vec3(1,0,1)), f.x), mix(h31(i + vec3(0,1,1)), h31(i + vec3(1,1,1)), f.x), f.y), f.z); }
vec3 hash33(vec3 p) { p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return fract((p.xxy + p.yxx) * p.zyx) * 2.0 - 1.0; }
float gnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(mix(mix(dot(hash33(i), f), dot(hash33(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                 mix(dot(hash33(i + vec3(0,1,0)), f - vec3(0,1,0)), dot(hash33(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
             mix(mix(dot(hash33(i + vec3(0,0,1)), f - vec3(0,0,1)), dot(hash33(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                 mix(dot(hash33(i + vec3(0,1,1)), f - vec3(0,1,1)), dot(hash33(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
}
vec3 tnormal(sampler2D tex, vec2 uv, float amt, out float cav) {
  vec4 t = texture2D(tex, uv);
  cav = t.a;
  return vec3((t.xy * 2.0 - 1.0) * amt, 1.0);
}
vec3 triplanar(sampler2D tex, vec3 p, vec3 n, float scale, float amt, out float cav) {
  vec3 w = pow(abs(n), vec3(4.0)); w /= (w.x + w.y + w.z);
  float c1, c2, c3;
  vec3 tx = tnormal(tex, p.zy * scale, amt, c1);
  vec3 ty = tnormal(tex, p.xz * scale + 0.31, amt, c2);
  vec3 tz = tnormal(tex, p.xy * scale + 0.67, amt, c3);
  cav = c1 * w.x + c2 * w.y + c3 * w.z;
  tx = vec3(tx.xy + n.zy, abs(tx.z) * n.x);
  ty = vec3(ty.xy + n.xz, abs(ty.z) * n.y);
  tz = vec3(tz.xy + n.xy, abs(tz.z) * n.z);
  return normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
}
vec3 lutDiffuse(float ndl, float curv) {
  float v = sqrt(clamp(curv / 0.5, 0.0, 1.0));
  vec3 s = texture2D(uLUT, vec2(ndl * 0.5 + 0.5, 1.0 - v)).rgb;
  return s * s;
}
// thin superficial veins: zero crossings of smooth noise, stretched along the limb
float veins(vec3 p, vec3 ax) {
  vec3 q = p - ax * dot(p, ax) * 0.62;
  float n = gnoise(q * 34.0 + vec3(3.1, 7.7, 1.3)) + 0.35 * gnoise(q * 90.0 + vec3(9.2, 1.4, 4.4));
  float line = 1.0 - smoothstep(0.02, 0.085, abs(n));
  float fade = smoothstep(-0.15, 0.25, gnoise(p * 16.0 + vec3(1.7, 5.3, 2.9)));
  return line * fade;
}
// creases over the finger joints: several fine lines on the back of each joint, one or two deep lines on the palm side
vec3 knuckles(vec3 p, vec3 N, inout float cavK) {
  vec3 dn = vec3(0.0);
  float sx = sign(p.x);
  vec3 dors0 = -normalize(vec3(-sx * 0.55, -0.75, 0.35));
  for (int i = 0; i < 28; i++) {
    vec4 A = uKnA[i]; vec4 B = uKnB[i];
    vec3 d = p - A.xyz;
    if (dot(d, d) > 0.00025) continue;
    float u = dot(d, B.xyz);
    vec3 radial = d - B.xyz * u;
    float r = length(radial);
    if (r > B.w * 1.7) continue;
    vec3 dors = normalize(dors0 - B.xyz * dot(dors0, B.xyz));
    vec3 lat = cross(B.xyz, dors);
    float side = dot(radial, dors) / max(r, 1e-5);
    float la = dot(radial, lat);
    float kind = A.w;
    // back of the joint: arcs that curve toward the fingertip at the sides
    float span = kind < 0.5 ? 0.0055 : (kind < 1.5 ? 0.0042 : 0.0028);
    float spacing = kind < 1.5 ? 0.00105 : 0.0009;
    float uu = u - la * la * 14.0 + 0.00018 * sin(la * 1400.0 + float(i) * 1.7);
    float env = exp(-0.5 * (u / span) * (u / span)) * smoothstep(0.05, 0.55, side) * (kind < 0.5 ? 0.45 : 1.0);
    float ph = uu / spacing * 6.2831853 + float(i) * 2.1;
    float c = 0.5 + 0.5 * cos(ph);
    float dh = 0.00006 * 4.0 * c * c * c * 0.5 * sin(ph) * (6.2831853 / spacing);
    // palm side: deep flexion lines right at the joint
    float envP = smoothstep(0.1, 0.6, -side) * (kind < 0.5 ? 0.0 : 1.0);
    float up = (u - la * la * 6.0) / 0.0006;
    float g1 = exp(-0.5 * (up - 1.0) * (up - 1.0)), g2 = exp(-0.5 * (up + 1.0) * (up + 1.0));
    float dhP = -0.00005 * (-(up - 1.0) * g1 - (up + 1.0) * g2 * 0.7) / 0.0006;
    dn -= B.xyz * (dh * env + dhP * envP);
    cavK *= 1.0 - 0.35 * env * c * c * c * c - 0.4 * envP * (g1 + 0.7 * g2);
  }
  return normalize(N + dn);
}
void main() {
  vec3 N = normalize(vN);
  vec3 Ns = normalize(vNs);
  vec3 V = normalize(cameraPosition - vWPos);
  // (with multisampling the varyings can be extrapolated past the triangle, so they are clamped before pow and sqrt)
  vec4 A = clamp(vA, 0.0, 1.0), Bv = clamp(vB, 0.0, 1.0), Cv = clamp(vC, 0.0, 1.0), Dv = clamp(vD, 0.0, 1.0);
  float ao = A.x, lip = A.y, hocc = A.z, scalp = A.w;
  float wet = Bv.x, ear = Bv.y, curv = Bv.z * 0.4 * uCurvScale, fuzz = Bv.w;
  float thickB = Cv.x * Cv.x * 0.03, oil = Cv.y, vein = Cv.z, face = Cv.w;
  float pore = Dv.x * uPoreAmt;
  vec3 L = normalize(uKeyDir), Lr = normalize(uRimDir);
  vec2 sh = keyShadow(vWPos, N, 1.0);
  float vis = sh.x * sh.y;
  vec3 hOut = normalize(vHOut);
  float rimOcc = smoothstep(-0.05, 0.3, dot(Ns, Lr)) * smoothstep(-0.25, 0.2, dot(hOut, Lr));
  float keyOcc = smoothstep(-0.12, 0.2, dot(Ns, L)) * smoothstep(-0.3, 0.1, dot(hOut, L));

  // ---------------- mannequin: matte grey clay, no skin colour or detail
  if (uMannequin > 0.5) {
    vec3 clay = vec3(0.46, 0.45, 0.43);
    vec3 c = clay * (uKeyColor * max(dot(N, L), 0.0) * vis + uRimColor * max(dot(N, Lr), 0.0) * 0.5 * ao * rimOcc + shIrr(N) * uEnvDiff * pow(ao, 0.8));
    c += PI * uKeyColor * specSphere(N, V, L, 0.55, 0.03, uKeyTan) * vis * keyOcc * 0.5;
    c += envSpecular(N, V, 0.55, 0.03) * uEnvSpec * ao * 0.5;
    outColor = vec4(min(c, vec3(64.0)), 1.0);
    outSSS = vec4(0.0);
    return;
  }

  // ---------------- skin microstructure: fine lines everywhere, pores on the face
  // (worked out in the rest pose with the rest normal N0, then turned with the skin)
  float cav;
  vec3 N0 = normalize(vN0);
  float dScale = uDetailScale * mix(0.78, 1.0, face);
  vec3 Nd0 = triplanar(uDetail, vObj, N0, dScale, uDetailAmt * mix(0.85, 1.0, face) * (1.0 - 0.75 * lip), cav);
  // (sampled everywhere: texture lookups inside a branch lose their mip level at the region border)
  float cp;
  vec3 Np0 = triplanar(uPores, vObj, N0, uPoreScale, 1.4 * pore, cp);
  Nd0 = normalize(Nd0 + Np0 - N0);
  cav *= mix(1.0, cp, clamp(pore, 0.0, 1.0));
  if (lip > 0.01) {
    float g = sin(vObj.x * 2600.0 + (sin(vObj.y * 900.0 + vObj.x * 300.0) * 0.6 + sin(vObj.x * 1700.0) * 0.4) * 1.5);
    vec3 tx = normalize(cross(vec3(0.0, 1.0, 0.0), N0));
    Nd0 = normalize(Nd0 + tx * smoothstep(0.55, 1.0, abs(g)) * sign(g) * 0.10 * lip);
    cav *= 1.0 - 0.15 * lip * smoothstep(0.7, 1.0, abs(g));
  }
  if (abs(vObj.x) > 0.33) Nd0 = knuckles(vObj, Nd0, cav);
  vec3 Nd = normalize(N + mat3(vR0, vR1, vR2) * (Nd0 - N0));
  // gentle unevenness of the surface (a few millimetres across), strongest on the body
  // (screen-space bump: the tilt is limited, and it fades where the surface folds sharply within a pixel)
  vec3 sx = dFdx(vWPos), sy = dFdy(vWPos);
  float pixW = max(length(sx), length(sy));
  float hU = (gnoise(vObj * 150.0) * 0.00009 + gnoise(vObj * 380.0 + 4.0) * 0.00003) * mix(1.0, 0.35, face) * (1.0 - lip) * (1.0 - wet) * smoothstep(0.0025, 0.0008, pixW) * uUnd;
  vec3 R1 = cross(sy, N), R2 = cross(N, sx);
  float det = dot(sx, R1);
  vec3 gradU = sign(det) * (dFdx(hU) * R1 + dFdy(hU) * R2) / max(abs(det), 1e-20);
  float fold = length(fwidth(N));
  gradU *= min(1.0, 0.1 / max(length(gradU), 1e-6)) * smoothstep(0.35, 0.1, fold) * step(1e-16, abs(det));
  Nd = normalize(Nd - gradU);

  // ---------------- albedo
  // skin tone: a tint over the baked colour (the moist lining around the eyes keeps most of its pink),
  // and the scalp under the hair takes a colour between the skin and the hair
  vec3 tint = mix(uSkinTint, pow(uSkinTint, vec3(0.3)), wet);
  vec3 albedo = clamp(vAlb, 0.0, 1.0) * tint + (uScalpCol - uScalp0 * tint) * 0.9 * scalp;
  albedo = max(albedo, vec3(0.0)) * mix(0.90, 1.0, cav);
  float m1 = vn3(vObj * 260.0) * 0.6 + vn3(vObj * 620.0) * 0.4;
  float m2 = vn3(vObj * 1500.0 + 7.0);
  albedo *= mix(vec3(1.06, 0.97, 0.96), vec3(0.95, 1.02, 1.03), m1) * (0.97 + 0.06 * m2);
  if (vein > 0.01) albedo *= mix(vec3(1.0), vec3(0.68, 0.76, 0.86), veins(vObj, normalize(vAx)) * vein * 0.6);
  for (int i = 0; i < 16; i++) {
    vec4 a = uMoleA[i];
    float d = length(vObj - a.xyz);
    if (d < a.w * 1.3) albedo = mix(albedo, albedo * vec3(0.25, 0.21, 0.22), smoothstep(a.w, a.w * 0.5, d) * uMoleB[i].x);
  }

  // ---------------- diffuse light that enters the skin
#ifdef SSS_SCREEN
  // the screen-space pass spreads this light, so the normals only get a light softening here
  vec3 Nr = normalize(mix(Nd, Ns, 0.3));
  vec3 Ng = normalize(mix(Nd, N, 0.4));
  vec3 Nb = Nd;
  vec3 dk = vec3(max(dot(Nr, L), 0.0), max(dot(Ng, L), 0.0), max(dot(Nb, L), 0.0));
  vec3 E = uKeyColor * dk * vis;
#else
  vec3 Nr = normalize(mix(Ns, N, 0.35));
  vec3 Ng = normalize(mix(N, Nd, 0.35));
  vec3 Nb = normalize(mix(N, Nd, 0.75));
  vec3 visC = pow(vec3(sh.x), vec3(0.72, 0.96, 1.0)) * pow(vec3(sh.y), vec3(0.9, 0.98, 1.0));
  vec3 dk = vec3(lutDiffuse(dot(Nr, L), curv).r, lutDiffuse(dot(Ng, L), curv).g, lutDiffuse(dot(Nb, L), curv).b);
  vec3 E = uKeyColor * dk * visC;
#endif
  E += uRimColor * max(dot(Ng, Lr), 0.0) * ao * hocc * rimOcc;
  vec3 amb = vec3(shIrr(Nr).r, shIrr(Ng).g, shIrr(Nb).b) * uEnvDiff;
  E += amb * pow(vec3(ao), vec3(0.7, 0.92, 1.0)) * hocc;
  // light that passes through thin parts (ears, nostrils, fingers, toes)
  // the baked thickness keeps light from passing through the eyeballs and other solid parts
  float nearEye = max(smoothstep(uEyeL.w + 0.0045, uEyeL.w + 0.002, distance(vObj, uEyeL.xyz)), smoothstep(uEyeR.w + 0.0045, uEyeR.w + 0.002, distance(vObj, uEyeR.xyz)));
  float tK = max(max(keyThickness(vWPos, N), 0.6 * thickB), 0.03 * nearEye) * 1124.0;
  // (only where the key light does not reach the surface directly)
  vec3 trans = uKeyColor * skinTransmit(tK) * max(-dot(Ns, L), 0.0) * (1.0 - sh.x) * sh.y;
  vec3 tb = skinTransmit(thickB * 1124.0);
  trans += uRimColor * tb * sqrt(max(-dot(Ns, Lr), 0.0) * max(-dot(N, Lr), 0.0)) * hocc;
  trans += shIrr(-Ns) * uEnvDiff * tb * 0.6;
  E += trans * uTrans * (1.0 - 0.65 * uMelanin) * sqrt(ao) / max(albedo, vec3(0.05));
  vec3 diffSkin = albedo * E;

  // ---------------- specular: two lobes, drier on the body, oily in the T-zone
  float r1 = mix(0.56, 0.50, face), r2 = mix(0.36, 0.30, face);
  float rough1 = mix(mix(mix(r1, 0.38, oil), 0.32, lip), 0.12, wet);
  float rough2 = mix(mix(mix(r2, 0.21, oil), 0.16, lip), 0.06, wet);
  float rv = vn3(vObj * 70.0 + 3.0);
  rough1 *= mix(0.9, 1.1, rv); rough2 *= mix(0.88, 1.12, rv);
  rough1 = mix(rough1, 0.72, (1.0 - cav) * 0.6);
  rough2 = mix(rough2, 0.5, (1.0 - cav) * 0.6);
  vec3 dnx = dFdx(Nd), dny = dFdy(Nd);
  float kvar = min(0.5 * (dot(dnx, dnx) + dot(dny, dny)), 0.2);
  rough1 = sqrt(sqrt(rough1 * rough1 * rough1 * rough1 + kvar));
  rough2 = sqrt(sqrt(rough2 * rough2 * rough2 * rough2 + kvar));
  float f0 = 0.028;
  float specOcc = clamp(pow(ao, 1.5) * mix(0.55, 1.0, cav), 0.0, 1.0) * hocc;
  float w2 = mix(mix(mix(0.10, 0.15, face), 0.35, lip), 0.6, wet);
  float ks = mix(specSphere(Nd, V, L, rough1, f0, uKeyTan), specSphere(Nd, V, L, rough2, f0, uKeyTan), w2);
  float kr = mix(specSphere(Nd, V, Lr, rough1, f0, 0.05), specSphere(Nd, V, Lr, rough2, f0, 0.05), 0.18);
  float gz = smoothstep(0.02, 0.25, dot(N, V));
  float hor = clamp(1.0 + dot(reflect(-V, Nd), N), 0.0, 1.0); hor *= hor;
  vec3 spec = PI * (uKeyColor * min(ks, 6.0) * vis * keyOcc * mix(0.5, 1.0, cav) * mix(specOcc, 1.0, 0.35) + uRimColor * min(kr, 3.0) * ao * ao * hocc * gz * rimOcc);
  spec += (envSpecular(Nd, V, rough1, f0) * (1.0 - w2) + envSpecular(Nd, V, rough2, f0) * w2) * uEnvSpec * specOcc * headEnvOccDir(hOut, Nd, V) * hor;
  // fine vellus hair: a soft sheen at grazing angles, strongest with light from behind
  float nvs = clamp(dot(Ns, V), 0.0, 1.0);
  float edgeF = pow(1.0 - nvs, 3.0);
  float fwd = 0.45 + 1.1 * pow(clamp(0.5 - 0.5 * dot(L, V), 0.0, 1.0), 3.0);
  float fwdR = 0.45 + 1.1 * pow(clamp(0.5 - 0.5 * dot(Lr, V), 0.0, 1.0), 3.0);
  vec3 fuzzLight = uKeyColor * clamp((dot(Ns, L) + 0.45) / 1.45, 0.0, 1.0) * vis * fwd
                 + uRimColor * clamp((dot(Ns, Lr) + 0.45) / 1.45, 0.0, 1.0) * rimOcc * hocc * fwdR
                 + shIrr(Ns) * uEnvDiff * ao;
  spec += fuzz * uFuzz * 0.07 * edgeF * fuzzLight * vec3(1.0, 0.88, 0.76) * hocc;

  vec3 color = spec;
#ifdef SSS_SCREEN
  outSSS = vec4(min(diffSkin, vec3(64.0)), 1.0);
#else
  color += diffSkin;
  outSSS = vec4(0.0);
#endif
  color = min(color, vec3(64.0));
  if (any(isnan(color))) color = vec3(0.0);
  if (any(isnan(outSSS))) outSSS = vec4(0.0);
  outColor = vec4(color, 1.0);
}`;

export const EYE_VS = /* glsl */`
attribute vec3 eyelocal; attribute float eyeao;
varying vec3 vWPos; varying vec3 vL; varying float vAO;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWPos = wp.xyz; vL = eyelocal; vAO = eyeao;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

export const EYE_FS = /* glsl */`
${GLSL_COMMON}
uniform mat3 uEyeRot;
uniform float uZcc; uniform float uRL; uniform float uIrisZ; uniform float uPupil;
uniform sampler2D uIris;
varying vec3 vWPos; varying vec3 vL; varying float vAO;
void main() {
  vec3 lp = vL;
  float ao = clamp(vAO, 0.0, 1.0);
  vec3 Vw = normalize(cameraPosition - vWPos);
  mat3 toLocal = transpose(uEyeRot);
  vec3 V = toLocal * Vw;
  vec3 L = toLocal * normalize(uKeyDir);
  vec3 Lr = toLocal * normalize(uRimDir);
  float rho = length(lp.xy);
  float isCornea = step(0.0, lp.z) * (1.0 - smoothstep(uRL - 0.0002, uRL + 0.0004, rho));
  vec3 Nsc = normalize(lp);
  vec3 Nco = normalize(lp - vec3(0.0, 0.0, uZcc));
  vec3 Nl = normalize(mix(Nsc, Nco, isCornea));
  vec3 Nw = uEyeRot * Nl;
  vec2 sh = keyShadow(vWPos, Nw, 0.6);
  float vis = sh.x * sh.y;
  if (uMannequin > 0.5) {
    vec3 Lw = normalize(uKeyDir);
    vec3 c = vec3(0.46, 0.45, 0.43) * (uKeyColor * max(dot(Nw, Lw), 0.0) * vis + shIrr(Nw) * uEnvDiff * ao);
    gl_FragColor = vec4(min(c, vec3(64.0)), 1.0);
    return;
  }
  vec3 Rf = refract(-V, Nco, 1.0 / 1.376);
  float t = (uIrisZ - lp.z) / min(Rf.z, -1e-3);
  vec3 ip = lp + Rf * t;
  float ir = length(ip.xy);
  vec3 irisC = texture2D(uIris, ip.xy / (2.0 * uRL) + 0.5).rgb;
  irisC *= smoothstep(uPupil, uPupil + 0.00018, ir);
  vec3 In = normalize(vec3(-ip.xy * 0.25 / uRL, 1.0));
  float caus = 1.0 + 0.5 * pow(max(dot(normalize(ip.xy + 1e-6), -normalize(L.xy + 1e-6)), 0.0), 2.0) * smoothstep(0.0, 0.5, L.z) * smoothstep(0.2 * uRL, uRL, ir);
  vec3 irisLight = uKeyColor * max(dot(In, L), 0.0) * caus * vis + shIrr(uEyeRot * In) * uEnvDiff * 0.8 * ao;
  vec3 sclera = vec3(0.56, 0.51, 0.47);
  float corner = smoothstep(0.30, 0.95, abs(lp.x) / 0.0123);
  float angA = atan(lp.y, lp.x);
  float polar = acos(clamp(lp.z / length(lp), -1.0, 1.0));
  float wv = sin(angA * 23.0 + sin(polar * 40.0 + angA * 7.0) * 1.3) * 0.5 + sin(angA * 51.0 + polar * 25.0) * 0.5;
  float vein = smoothstep(0.93, 1.0, abs(wv)) * smoothstep(0.55, 1.1, polar) * (0.35 + 0.65 * corner);
  sclera = mix(sclera, vec3(0.60, 0.40, 0.36), corner * 0.35);
  sclera = mix(sclera, vec3(0.45, 0.13, 0.11), vein * 0.55);
  sclera *= mix(0.82, 1.0, smoothstep(uRL, uRL + 0.0022, rho));
  float rimEye = smoothstep(-0.1, 0.2, Lr.z);
  vec3 scl = sclera * (uKeyColor * max(dot(Nsc, L) * 0.8 + 0.2, 0.0) * vis + (shIrr(uEyeRot * Nsc) * uEnvDiff + uRimColor * max(dot(Nsc, Lr), 0.0) * rimEye) * ao);
  float limb = smoothstep(uRL - 0.0006, uRL + 0.0003, ir);
  vec3 col = mix(scl, mix(irisC * irisLight, scl, limb), isCornea);
  float rough = mix(0.08, 0.03, isCornea);
  vec3 dnx = dFdx(Nw), dny = dFdy(Nw);
  rough = sqrt(sqrt(rough * rough * rough * rough + min(0.5 * (dot(dnx, dnx) + dot(dny, dny)), 0.2)));
  float f0 = 0.025;
  vec2 abk = envBRDF(rough, max(dot(Nw, Vw), 1e-4));
  vec3 spec = textureCubeUV(uEnvKey, reflect(-Vw, Nw), rough).rgb * (f0 * abk.x + abk.y) * vis * smoothstep(-0.1, 0.2, L.z);
  spec += PI * uRimColor * specSphere(Nl, V, Lr, rough, f0, 0.05) * ao * rimEye;
  spec += envSpecular(Nw, Vw, rough, f0) * uEnvSpec * pow(ao, 1.5) * headEnvOcc(vWPos, Nw, Vw);
  col += spec;
  col = min(col, vec3(64.0));
  if (any(isnan(col))) col = vec3(0.0);
  gl_FragColor = vec4(col, 1.0);
}`;

export const HAIR_VS = /* glsl */`
${SKIN_GLSL}
attribute vec4 tangent4; attribute vec4 hattr; attribute float strand;
uniform float uWidth; uniform float uTipWidth; uniform float uViewportH; uniform float uMinPix;
// per strand the rows of [scalp size * frame of the triangle under the root | root] on the current body, and with
// HAIR_TIPS the same rows for the triangle under the tip (AnnyBody.followStrands)
uniform highp sampler2D uStrands;
#ifdef HAIR_TIPS
attribute vec3 tipLocal;
#define STRAND_TEXELS 6
#else
#define STRAND_TEXELS 3
#endif
varying vec3 vWPos; varying vec3 vT; varying vec4 vH; varying float vCov;
vec4 strandTexel(int i) { int w = textureSize(uStrands, 0).x; return texelFetch(uStrands, ivec2(i % w, i / w), 0); }
void main() {
  // the point and its tangent keep their coordinates in the frame of the root triangle; with HAIR_TIPS the point
  // blends toward its place in the frame of the tip triangle with the weight t^2 (anny.hair.StrandBinding)
  int ts = int(strand + 0.5) * STRAND_TEXELS;
  vec4 m0 = strandTexel(ts), m1 = strandTexel(ts + 1), m2 = strandTexel(ts + 2);
  vec3 rest = vec3(dot(m0.xyz, position) + m0.w, dot(m1.xyz, position) + m1.w, dot(m2.xyz, position) + m2.w);
#ifdef HAIR_TIPS
  vec4 n0 = strandTexel(ts + 3), n1 = strandTexel(ts + 4), n2 = strandTexel(ts + 5);
  vec3 atTip = vec3(dot(n0.xyz, tipLocal) + n0.w, dot(n1.xyz, tipLocal) + n1.w, dot(n2.xyz, tipLocal) + n2.w);
  rest = mix(rest, atTip, hattr.x * hattr.x);
#endif
  vec3 tl = vec3(dot(m0.xyz, tangent4.xyz), dot(m1.xyz, tangent4.xyz), dot(m2.xyz, tangent4.xyz));
  mat4 S = skinMat();
  vec3 wp = (modelMatrix * (S * vec4(rest, 1.0))).xyz;
  vec3 T = normalize(mat3(modelMatrix) * mat3(S) * tl);
  bool ortho = projectionMatrix[3][3] > 0.5;
  vec3 V = ortho ? normalize(vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2])) : normalize(cameraPosition - wp);
  vec3 side = cross(T, V);
  float sl = length(side);
  side = sl > 1e-4 ? side / sl : normalize(cross(T, vec3(0.0, 1.0, 0.0)));
  float t = hattr.x;
  float w = mix(uWidth, uTipWidth, t * t);
  vec4 clip = projectionMatrix * viewMatrix * vec4(wp, 1.0);
  float pix = ortho ? 2.0 / (projectionMatrix[1][1] * uViewportH) : clip.w * 2.0 / (projectionMatrix[1][1] * uViewportH);
  float wEff = max(w, pix * uMinPix);
  vCov = w / wEff;
  wp += side * (hattr.y * 2.0 - 1.0) * wEff * 0.5;
  vWPos = wp; vT = T; vH = hattr;
  gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
}`;

export const HAIR_FS = /* glsl */`
${GLSL_COMMON}
uniform vec3 uHairColor; uniform float uHairRough; uniform vec3 uHeadC; uniform float uSpecScale; uniform float uDiffScale;
varying vec3 vWPos; varying vec3 vT; varying vec4 vH; varying float vCov;
float hairG(float B, float th) { return exp(-0.5 * th * th / (B * B)) / (2.5066283 * B); }
float hairF(float c) { const float F0 = 0.04652; return F0 + (1.0 - F0) * pow(1.0 - c, 5.0); }
vec3 hairBSDF(vec3 T, vec3 V, vec3 L, vec3 base, float rough) {
  float sL = clamp(dot(T, L), -1.0, 1.0), sV = clamp(dot(T, V), -1.0, 1.0);
  float cosTD = cos(0.5 * abs(asin(sV) - asin(sL)));
  vec3 Lp = L - sL * T, Vp = V - sV * T;
  float cosPhi = dot(Lp, Vp) * inversesqrt(dot(Lp, Lp) * dot(Vp, Vp) + 1e-4);
  float cosHalfPhi = sqrt(clamp(0.5 + 0.5 * cosPhi, 0.0, 1.0));
  float n_p = 1.19 / cosTD + 0.36 * cosTD;
  float shift = 0.035;
  float r2 = rough * rough;
  float sa = sin(-2.0 * shift), ca = cos(-2.0 * shift);
  float shR = 2.0 * sa * (ca * cosHalfPhi * sqrt(max(1.0 - sV * sV, 0.0)) + sa * sV);
  vec3 S = vec3(hairG(r2, sL + sV - shR) * 0.25 * cosHalfPhi * min(hairF(sqrt(clamp(0.5 + 0.5 * dot(V, L), 0.0, 1.0))), 0.2)) * uSpecScale;
  float a = 1.0 / n_p;
  float h = cosHalfPhi * (1.0 + a * (0.6 - 0.8 * cosPhi));
  float f = hairF(cosTD * sqrt(clamp(1.0 - h * h, 0.0, 1.0)));
  vec3 Tp = pow(base, vec3(0.5 * sqrt(max(1.0 - h * h * a * a, 0.0)) / cosTD));
  S += hairG(r2 * 0.5, sL + sV - shift) * exp(-3.65 * cosPhi - 3.98) * (1.0 - f) * (1.0 - f) * Tp;
  f = hairF(cosTD * 0.5);
  Tp = pow(base, vec3(0.8 / cosTD));
  S += hairG(r2 * 2.0, sL + sV - 4.0 * shift) * exp(17.0 * cosPhi - 16.78) * (1.0 - f) * (1.0 - f) * f * Tp;
  return max(S, vec3(0.0));
}
vec3 hairDiffuse(vec3 T, vec3 V, vec3 L, vec3 base, float shadow) {
  float kd = 1.0 - abs(dot(T, L));
  vec3 fn = normalize(V - T * dot(V, T));
  float nl = clamp((dot(fn, L) + 1.0) / 4.0, 0.0, 1.0);
  float ds = (1.0 / PI) * mix(nl, kd, 0.33);
  float luma = dot(base, vec3(0.2126, 0.7152, 0.0722));
  vec3 tint = pow(base / max(luma, 1e-4), vec3(1.0 - shadow));
  return base * uDiffScale * ds * tint;
}
float hash13(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
void main() {
  float t = vH.x;
  float cov = vCov * (1.0 - smoothstep(0.9, 1.0, t));
  float rnd0 = hash13(vec3(gl_FragCoord.xy, uFrame * 7.31 + vH.z * 131.7));
#ifdef HAIR_DEPTH
  if (cov < 0.12) discard;
  gl_FragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
#else
  if (rnd0 > cov) discard;
  vec3 T = normalize(vT);
  vec3 V = normalize(cameraPosition - vWPos);
  vec3 L = normalize(uKeyDir);
  vec3 Lr = normalize(uRimDir);
  vec3 nOut = normalize(vWPos - uHeadC);
  float rnd = vH.z;
  float ao = vH.w;
  vec3 base = uHairColor * mix(0.75, 1.3, rnd);
  float rough = uHairRough * mix(0.9, 1.15, fract(rnd * 7.13));
  vec2 sh = keyShadowLite(vWPos, nOut);
  float vis = sh.x * sh.y;
  vec3 col = PI * uKeyColor * (hairBSDF(T, V, L, base, rough) + hairDiffuse(T, V, L, base, vis)) * vis;
  float rimVis = smoothstep(-0.15, 0.35, dot(nOut, Lr));
  col += PI * uRimColor * (hairBSDF(T, V, Lr, base, rough) + hairDiffuse(T, V, Lr, base, 1.0)) * mix(ao, 1.0, 0.5) * rimVis;
  col += shIrr(nOut) * uEnvDiff * base * uDiffScale * 0.6 * ao;
  vec3 Rn = normalize(V - T * dot(V, T));
  col += textureCubeUV(uEnv, reflect(-V, Rn), 0.35).rgb * 0.05 * ao * uEnvSpec;
  col = min(col, vec3(64.0));
  if (any(isnan(col))) col = vec3(0.0);
  gl_FragColor = vec4(col, 1.0);
#endif
}`;

export const NOISE_GLSL = /* glsl */`
vec2 hash2(vec2 p) { p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3))); return fract(sin(p) * 43758.5453); }
float hash1(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec2 worley(vec2 uv, float n) {
  vec2 p = uv * n; vec2 ip = floor(p); vec2 fp = fract(p);
  float f1 = 8.0, f2 = 8.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 g = vec2(float(i), float(j)); vec2 cell = mod(ip + g, n); vec2 o = hash2(cell);
    vec2 r = g + o - fp; float d = dot(r, r);
    if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
  }
  return sqrt(vec2(f1, f2));
}
float vnoise(vec2 uv, float n) {
  vec2 p = uv * n; vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash1(mod(i, n)), b = hash1(mod(i + vec2(1, 0), n)), c = hash1(mod(i + vec2(0, 1), n)), d = hash1(mod(i + vec2(1, 1), n));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}`;

export const PROP_VS = /* glsl */`
varying vec3 vWPos; varying vec3 vN; varying vec3 vObj;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWPos = wp.xyz; vObj = position; vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

// oiled beech: warm and fairly smooth, with a faint grain along the legs and across the seat
export const PROP_FS = /* glsl */`
${GLSL_COMMON}
uniform vec3 uWood;
varying vec3 vWPos; varying vec3 vN; varying vec3 vObj;
float hp(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
void main() {
  vec3 N = normalize(vN);
  if (!gl_FrontFacing) N = -N;
  vec3 V = normalize(cameraPosition - vWPos);
  vec3 L = normalize(uKeyDir), Lr = normalize(uRimDir);
  vec2 sh = keyShadow(vWPos, N, 0.8);
  float vis = sh.x * sh.y;
  float g = sin(vObj.x * 380.0 + sin(vObj.z * 60.0) * 2.0) * 0.5 + 0.5;
  vec3 base = uWood * (0.86 + 0.14 * g) * (0.96 + 0.08 * hp(floor(vObj * 900.0)));
  vec3 col = base * (uKeyColor * max(dot(N, L), 0.0) * vis + uRimColor * max(dot(N, Lr), 0.0) * 0.3 + shIrr(N) * uEnvDiff);
  col += PI * uKeyColor * specSphere(N, V, L, 0.38, 0.04, uKeyTan) * vis * 0.6;
  col += envSpecular(N, V, 0.38, 0.04) * uEnvSpec * 0.5;
  gl_FragColor = vec4(min(col, vec3(64.0)), 1.0);
}`;

export const SSS_GLSL = (SSS_N: number) => /* glsl */`
#include <packing>
uniform sampler2D tDepth; uniform vec4 uKernel[${SSS_N}]; uniform vec2 uTexel; uniform float uProj; uniform float uWidth;
uniform float uNear; uniform float uFar; uniform float uFrameS; uniform float uSSSRaw;
float linZ(vec2 uv) { return -perspectiveDepthToViewZ(texture2D(tDepth, uv).x, uNear, uFar); }
float ignS(vec2 p) { p += uFrameS * 5.588238; return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
// one direction of the separable blur. Colours are premultiplied by skin coverage (alpha), and the blur is a
// normalised convolution: pixels at the edge of the skin count by their coverage, and gaps (eyes, hair, fabric)
// or samples on another surface fall back to the centre pixel
vec4 sssBlur(sampler2D tex, vec2 uv, vec2 dir) {
  vec4 cM = texture2D(tex, uv);
  if (cM.a < 0.002) return vec4(0.0);
  if (uSSSRaw > 0.5) return cM;
  float zM = linZ(uv);
  vec2 stepv = dir * uTexel * (uWidth * uProj / zM) * (0.85 + 0.3 * ignS(gl_FragCoord.xy + dir * 37.0));
  float tol = uWidth * 3.0;
  vec3 acc = cM.rgb * uKernel[0].rgb, wsum = cM.a * uKernel[0].rgb;
  for (int i = 1; i < ${SSS_N}; i++) {
    vec2 o = uv + uKernel[i].a * stepv;
    vec4 c = texture2D(tex, o);
    float s = max(clamp(abs(linZ(o) - zM) / tol, 0.0, 1.0), 1.0 - smoothstep(0.0, 0.5, c.a));
    c = mix(c, cM, s);
    acc += uKernel[i].rgb * c.rgb; wsum += uKernel[i].rgb * c.a;
  }
  return vec4(acc / max(wsum, vec3(1e-4)) * cM.a, cM.a);
}`;
