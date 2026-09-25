// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// =====================================================================================
//  Anny Viewer - real-time renderer of anny's body model
//  Body: anny's phenotype sliders (anny_shape.ts), the mixed Catmull-Clark subdivision of anny's body with its
//        detail layers (subdivision.ts, body.ts), anny's rig and pose library, and anny's soft-tissue correctives
//  Skin: screen-space subsurface scattering (separable d'Eon profile) + light through thin parts from the shadow maps
//        + dual-lobe specular + procedural pores, skin lines, finger creases, veins and vellus sheen
//  Eyes: refractive cornea, procedural iris, soft-box catchlights
//  Hair: ~55k strands bound to the skin, Marschner-style shading, deep-opacity shadows
//  Image: progressive temporal accumulation (soft shadows, AA, stochastic hair coverage)
//  Ported from the legacy 3D Model viewer (legacy/3d_model/baseline_body/source.zip, web/app.js). The page draws anny
//  in the frame of the legacy figure (anny.poses.authoring.rig), so the tuning of the lights and shaders carries over.
// =====================================================================================
// @ts-nocheck -- the renderer below is the legacy JavaScript; the new modules (anny_shape, subdivision, body) are typed
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { AnnyBody, vertexNormals as normalsOf } from './body.ts';

const qs = new URLSearchParams(location.search);
const DEG = Math.PI / 180;
const $ = (id) => document.getElementById(id);
const isSmall = Math.min(screen.width, screen.height) < 700 || /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
const SHOT = qs.has('shot');
// skin diffusion: screen-space pass by default, '?sss=lut' switches to the pre-integrated curvature LUT
const SSS_ON = qs.get('sss') !== 'lut';

// ------------------------------------------------------------------ renderer
const canvas = $('c');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: SHOT });
} catch (e) {
  showError('WebGL 2 is not available in this browser, so the 3D portrait cannot be shown.');
  throw e;
}
const PR = Math.min(window.devicePixelRatio || 1, isSmall ? 1.5 : 2);
renderer.setPixelRatio(PR);
renderer.setSize(innerWidth, innerHeight, false);
const TONEMAP = { agx: THREE.AgXToneMapping, aces: THREE.ACESFilmicToneMapping, neutral: THREE.NeutralToneMapping };
renderer.toneMapping = TONEMAP[qs.get('tm') || 'agx'];
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const camera = new THREE.PerspectiveCamera(24, innerWidth / innerHeight, 0.02, 20);
const TARGET = new THREE.Vector3(0, 0.495, 0.035);
// framings: 'body' shows the whole figure, 'face' the head and hair (half extents in metres)
const FRAMES = {
  body: { target: [0, -0.19, 0.03], yaw: 24, pitch: 4, halfH: 0.95, halfW: 0.58 },
  face: { target: [0, 0.495, 0.035], yaw: 24, pitch: 4, halfH: 0.17, halfW: 0.15 },
};
let currentFrame = 'body';
function frameDistance(f) {
  const aspect = innerWidth / innerHeight;
  const vfov = 24 * DEG;
  const dv = f.halfH / Math.tan(vfov / 2);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
  const dh = f.halfW / Math.tan(hfov / 2);
  return Math.max(dv, dh);
}
const HOME = { yaw: 24, pitch: 4, dist: 0.8 };
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.085;
controls.minDistance = SHOT ? 0.03 : 0.22; controls.maxDistance = 6.5;
controls.minPolarAngle = 42 * DEG; controls.maxPolarAngle = 118 * DEG;
controls.zoomToCursor = true;
controls.enablePan = false; controls.rotateSpeed = 0.7; controls.zoomSpeed = 0.8;
controls.autoRotateSpeed = 0.9;
function placeCamera(yaw, pitch, dist, tx = 0, ty = TARGET.y, tz = TARGET.z) {
  controls.target.set(tx, ty, tz);
  const y = yaw * DEG, p = pitch * DEG;
  camera.position.set(tx + dist * Math.sin(y) * Math.cos(p), ty + dist * Math.sin(p), tz + dist * Math.cos(y) * Math.cos(p));
  camera.lookAt(controls.target);
  controls.update();
}
function frameCamera(name) {
  const f = FRAMES[name]; currentFrame = name;
  placeCamera(f.yaw, f.pitch, frameDistance(f), f.target[0], f.target[1], f.target[2]);
}
// smooth move between framings
let tween = null;
function flyTo(name) {
  const f = name === 'body' && typeof MOTION !== 'undefined' && MOTION.cur ? bodyFrameFor(MOTION.cur) : FRAMES[name]; currentFrame = name;
  if (name === 'body' && typeof MOTION !== 'undefined') MOTION.lastFramed = MOTION.cur;
  const d = frameDistance(f) * (name === 'face' && RIG.ready ? HEADMAP.k : 1), y = f.yaw * DEG, p = f.pitch * DEG;
  const t1 = new THREE.Vector3(...f.target);
  if (name === 'face' && RIG.ready) t1.applyMatrix4(_headFull);
  const p1 = new THREE.Vector3(t1.x + d * Math.sin(y) * Math.cos(p), t1.y + d * Math.sin(p), t1.z + d * Math.cos(y) * Math.cos(p));
  tween = { t0: performance.now(), dur: 900, fromT: controls.target.clone(), fromP: camera.position.clone(), toT: t1, toP: p1 };
}
frameCamera('body');
const scene = new THREE.Scene();

// every material drawn into the scene target writes two outputs: the final colour and the skin light for the diffusion pass
const MRT_HEAD = 'layout(location = 0) out highp vec4 outColor;\nlayout(location = 1) out highp vec4 outSSS;\n';
function toMRT(fs) { return MRT_HEAD + fs.replace(/gl_FragColor\s*=/g, 'outSSS = vec4(0.0); outColor ='); }

// ------------------------------------------------------------------ backdrop (drawn in-scene)
const BG = { uBgA: { value: new THREE.Vector3() }, uBgB: { value: new THREE.Vector3() }, uBgCenter: { value: new THREE.Vector2(0.5, 0.56) }, uRes: { value: new THREE.Vector2(1, 1) } };
const BG_GLSL = /* glsl */`
uniform vec3 uBgA; uniform vec3 uBgB; uniform vec2 uBgCenter; uniform vec2 uRes;
vec3 bgColor() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 d = (uv - uBgCenter) * vec2(uRes.x / uRes.y, 1.0);
  float r = length(d * vec2(0.8, 1.0));
  return mix(uBgA, uBgB, smoothstep(0.0, 0.75, r));
}`;
const bgMat = new THREE.ShaderMaterial({
  uniforms: BG,
  vertexShader: 'void main(){ gl_Position = vec4(position.xy, 0.9999, 1.0); }',
  fragmentShader: toMRT(`${BG_GLSL}\nvoid main(){ gl_FragColor = vec4(bgColor(), 1.0); }`), glslVersion: THREE.GLSL3,
  depthWrite: false, depthTest: false, toneMapped: false, blending: THREE.NoBlending,
});
const triGeo = new THREE.BufferGeometry();
triGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
const bgMesh = new THREE.Mesh(triGeo, bgMat); bgMesh.frustumCulled = false; bgMesh.renderOrder = -10;
scene.add(bgMesh);

// ------------------------------------------------------------------ lighting presets
// az: degrees from the face direction (+z) toward the model's left (+x); el: degrees up.
const PRESETS = {
  studio: {
    label: 'Studio', exposure: 0.88,
    sky: [0.022, 0.022, 0.025], horizon: [0.014, 0.013, 0.013], ground: [0.006, 0.0055, 0.005],
    boxes: [
      { az: 78, el: 5, w: 30, h: 40, c: [0.12, 0.123, 0.135] },
      { az: 180, el: 40, w: 20, h: 12, c: [0.30, 0.31, 0.34] },
      { az: 0, el: 70, w: 30, h: 30, c: [0.03, 0.03, 0.032] },
    ],
    key: { az: -22, el: 26, size: 9, c: [4.4, 4.2, 4.0] },
    rim: { az: 140, el: 35, c: [1.8, 1.85, 2.0] },
    bg: [[0.050, 0.052, 0.058], [0.005, 0.0053, 0.0062]],
  },
  window: {
    label: 'Window', exposure: 1.05,
    sky: [0.045, 0.043, 0.040], horizon: [0.030, 0.027, 0.023], ground: [0.012, 0.010, 0.008],
    boxes: [
      { az: 95, el: 0, w: 45, h: 35, c: [0.085, 0.075, 0.062] },
      { az: -10, el: -30, w: 40, h: 20, c: [0.05, 0.045, 0.04] },
    ],
    key: { az: -68, el: 14, size: 20, c: [3.2, 3.4, 3.8] },
    rim: { az: 160, el: 20, c: [0.25, 0.25, 0.27] },
    bg: [[0.070, 0.066, 0.060], [0.009, 0.008, 0.0075]],
  },
  chiaroscuro: {
    label: 'Chiaroscuro', exposure: 1.0,
    sky: [0.004, 0.0038, 0.0035], horizon: [0.003, 0.0027, 0.0024], ground: [0.0015, 0.0013, 0.0012],
    boxes: [
      { az: 60, el: 10, w: 25, h: 30, c: [0.012, 0.010, 0.008] },
    ],
    key: { az: -55, el: 38, size: 4.5, c: [5.6, 4.7, 3.6] },
    rim: { az: 150, el: 30, c: [0.35, 0.3, 0.25] },
    bg: [[0.030, 0.024, 0.018], [0.0022, 0.0018, 0.0015]],
  },
  backlight: {
    label: 'Backlight', exposure: 1.0,
    sky: [0.015, 0.016, 0.019], horizon: [0.010, 0.010, 0.012], ground: [0.004, 0.004, 0.0045],
    boxes: [
      { az: 25, el: 8, w: 28, h: 34, c: [0.20, 0.19, 0.18] },
      { az: -120, el: 20, w: 16, h: 30, c: [0.35, 0.37, 0.42] },
    ],
    key: { az: 128, el: 28, size: 8, c: [5.0, 5.3, 6.0] },
    rim: { az: -135, el: 25, c: [1.6, 1.7, 1.9] },
    bg: [[0.040, 0.046, 0.058], [0.004, 0.0045, 0.0058]],
  },
};
function dirAE(az, el) {
  const a = az * DEG, e = el * DEG;
  return new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e));
}
function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function buildEnvironment(preset, keyOnly = false) {
  const W = 1024, H = 512;
  const data = new Float32Array(W * H * 4);
  let src = preset.boxes;
  if (keyOnly) {
    const k = preset.key, t = Math.tan(k.size * DEG);
    const Lr = Math.PI / (4 * t * t);
    src = [{ az: k.az, el: k.el, w: k.size, h: k.size, c: k.c.map(v => v * Lr) }];
  }
  const boxes = src.map(b => {
    const c = dirAE(b.az, b.el);
    const up0 = Math.abs(c.y) > 0.95 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    const r = new THREE.Vector3().crossVectors(up0, c).normalize();
    const u = new THREE.Vector3().crossVectors(c, r).normalize();
    return { cx: c.x, cy: c.y, cz: c.z, rx: r.x, ry: r.y, rz: r.z, ux: u.x, uy: u.y, uz: u.z, tw: Math.tan(b.w * DEG), th: Math.tan(b.h * DEG), col: b.c };
  });
  const sh = new Float64Array(27);
  for (let j = 0; j < H; j++) {
    const lat = ((j + 0.5) / H - 0.5) * Math.PI;
    const cl = Math.cos(lat), sl = Math.sin(lat);
    const dO = (2 * Math.PI / W) * (Math.PI / H) * cl;
    for (let i = 0; i < W; i++) {
      const a = ((i + 0.5) / W - 0.5) * 2 * Math.PI;
      const dx = Math.cos(a) * cl, dy = sl, dz = Math.sin(a) * cl;
      let c0, c1, c2;
      if (keyOnly) { c0 = c1 = c2 = 0; }
      else if (dy > 0) { const t = Math.pow(dy, 0.6); c0 = preset.horizon[0] + (preset.sky[0] - preset.horizon[0]) * t; c1 = preset.horizon[1] + (preset.sky[1] - preset.horizon[1]) * t; c2 = preset.horizon[2] + (preset.sky[2] - preset.horizon[2]) * t; }
      else { const t = Math.min(1, -dy * 4); c0 = preset.horizon[0] + (preset.ground[0] - preset.horizon[0]) * t; c1 = preset.horizon[1] + (preset.ground[1] - preset.horizon[1]) * t; c2 = preset.horizon[2] + (preset.ground[2] - preset.horizon[2]) * t; }
      for (const b of boxes) {
        const cd = dx * b.cx + dy * b.cy + dz * b.cz;
        if (cd <= 0.05) continue;
        const x = (dx * b.rx + dy * b.ry + dz * b.rz) / cd / b.tw, y = (dx * b.ux + dy * b.uy + dz * b.uz) / cd / b.th;
        const ax = Math.abs(x), ay = Math.abs(y);
        if (ax > 1.1 || ay > 1.1) continue;
        const m = keyOnly ? smooth(1.03, 0.97, ax) * smooth(1.03, 0.97, ay) : smooth(1.1, 0.92, ax) * smooth(1.1, 0.92, ay);
        const f = keyOnly ? 1 - 0.1 * (x * x + y * y) : 1 - 0.25 * (x * x + y * y);
        c0 += b.col[0] * m * f; c1 += b.col[1] * m * f; c2 += b.col[2] * m * f;
      }
      const o = (j * W + i) * 4;
      data[o] = c0; data[o + 1] = c1; data[o + 2] = c2; data[o + 3] = 1;
      if (!keyOnly) {
        const Y = [0.282095, 0.488603 * dy, 0.488603 * dz, 0.488603 * dx, 1.092548 * dx * dy, 1.092548 * dy * dz, 0.315392 * (3 * dz * dz - 1), 1.092548 * dx * dz, 0.546274 * (dx * dx - dy * dy)];
        for (let k = 0; k < 9; k++) { const w = Y[k] * dO; sh[k * 3] += c0 * w; sh[k * 3 + 1] += c1 * w; sh[k * 3 + 2] += c2 * w; }
      }
    }
  }
  const A = [1, 2 / 3, 2 / 3, 2 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4];
  const Cb = [0.282095, 0.488603, 0.488603, 0.488603, 1.092548, 1.092548, 0.315392, 1.092548, 0.546274];
  const shc = Array.from({ length: 9 }, (_, k) => new THREE.Vector3(sh[k * 3] * A[k] * Cb[k], sh[k * 3 + 1] * A[k] * Cb[k], sh[k * 3 + 2] * A[k] * Cb[k]));
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const pm = new THREE.PMREMGenerator(renderer);
  const rt = pm.fromEquirectangular(tex);
  pm.dispose(); tex.dispose();
  return { pmrem: rt.texture, sh: shc };
}

// ------------------------------------------------------------------ shared uniforms
const U = {
  uKeyDir: { value: new THREE.Vector3() }, uKeyColor: { value: new THREE.Vector3() }, uKeyTan: { value: 0.2 },
  uRimDir: { value: new THREE.Vector3() }, uRimColor: { value: new THREE.Vector3() },
  uSH: { value: Array.from({ length: 9 }, () => new THREE.Vector3()) },
  uEnv: { value: null }, uEnvKey: { value: null }, uEnvSpec: { value: 1.0 }, uEnvDiff: { value: 1.0 },
  uShadowRaw: { value: null }, uShadowCmp: { value: null }, uHairShadow: { value: null }, uShadowMat: { value: new THREE.Matrix4() },
  uShadowP: { value: new THREE.Vector4() }, uShadowRot: { value: new THREE.Matrix3() },
  uBodyShadowCmp: { value: null }, uBodyShadowMat: { value: new THREE.Matrix4() }, uBodyShadowP: { value: new THREE.Vector4() },
  uDetail: { value: null }, uDetailScale: { value: 48.0 }, uDetailAmt: { value: 1.0 },
  uPores: { value: null }, uPoreScale: { value: 41.0 }, uPoreAmt: { value: parseFloat(qs.get('pores') || '1.0') },
  uLUT: { value: null }, uFrame: { value: 0 }, uCurvScale: { value: parseFloat(qs.get('curv') || '1.6') },
  uBodyShadowRaw: { value: null },
  uMoleA: { value: Array.from({ length: 16 }, () => new THREE.Vector4(0, -9, 0, 0.0001)) },
  uMoleB: { value: Array.from({ length: 16 }, () => new THREE.Vector4()) },
  uWearOn: { value: new THREE.Vector4(0, 0, 0, 0) },
  // look: skin tint over the baked skin colour, scalp colour under the hair, melanin (0 light to 1 deep), hair shadow density
  uSkinTint: { value: new THREE.Vector3(1, 1, 1) }, uScalpCol: { value: new THREE.Vector3() }, uScalp0: { value: new THREE.Vector3() },
  uMelanin: { value: 0 }, uHairShadowK: { value: 1 },
  uMannequin: { value: 0 },   // 1: plain grey mannequin (shown while no base layer is worn)
  uTrans: { value: parseFloat(qs.get('trans') || '1.0') }, uFuzz: { value: parseFloat(qs.get('fuzz') || '1.0') }, uUnd: { value: parseFloat(qs.get('und') || '1.0') },
  uKnA: { value: Array.from({ length: 28 }, () => new THREE.Vector4(0, -9, 0, 0)) }, uKnB: { value: Array.from({ length: 28 }, () => new THREE.Vector4(0, 1, 0, 0.001)) },
  uEyeL: { value: new THREE.Vector4(0, -9, 0, 0.012) }, uEyeR: { value: new THREE.Vector4(0, -9, 0, 0.012) },
  // contact shade of each foot on the floor: centre x, z, radius, strength
  uFootL: { value: new THREE.Vector4(0.168, 0.07, 0.08, 1) }, uFootR: { value: new THREE.Vector4(-0.168, 0.07, 0.08, 1) },
  // the head's pose: from world space back to the rest pose of the head, and the head's rotation
  uHeadInv: { value: new THREE.Matrix4() }, uHeadRot: { value: new THREE.Matrix3() },
};
Object.assign(U, BG);
const FLOAT_RT = renderer.extensions.has('EXT_color_buffer_float');
let ENV_DEFINES = null;
function envDefines(tex) {
  const h = tex.image.height;
  const maxMip = Math.log2(h) - 2;
  const d = { ENVMAP_TYPE_CUBE_UV: '', CUBEUV_TEXEL_WIDTH: 1.0 / (3 * Math.max(Math.pow(2, maxMip), 7 * 16)), CUBEUV_TEXEL_HEIGHT: 1.0 / h, CUBEUV_MAX_MIP: maxMip.toFixed(1) };
  if (!FLOAT_RT) d.SHADOW_PACKED = '';
  return d;
}

// ------------------------------------------------------------------ GLSL common
const HEAD_OUT_GLSL = /* glsl */`
vec3 headOutRest(vec3 p) { float cy = p.y > 0.43 ? min(p.y, 0.55) : p.y; return normalize(p - vec3(0.0, cy, 0.02) + vec3(0.0, 1e-4, 0.0)); }`;
// skinning (linear blend) for the custom vertex shaders; three.js defines USE_SKINNING for skinned meshes
const SKIN_GLSL = /* glsl */`
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
const GLSL_COMMON = /* glsl */`
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

// ------------------------------------------------------------------ skin material
// Diffuse light goes to a second render target and is spread by a screen-space diffusion pass (SSS_SCREEN).
// Without SSS_SCREEN the shader falls back to the pre-integrated curvature LUT.
// skin under a worn garment sinks a few millimetres, so bends of the body cannot push it through the fabric; the
// sinking fades out over a few millimetres past the edge, where an elastic band would press into the skin
const COVER_GLSL = /* glsl */`
vec3 coveredPosition() {
  vec4 c = attrV * uWearOn;
  return position - normal * (0.004 * max(max(c.x, c.y), max(c.z, c.w)));
}`;
const SKIN_VS = /* glsl */`
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
const SKIN_FS = /* glsl */`
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

// ------------------------------------------------------------------ eye material
const EYE_VS = /* glsl */`
attribute vec3 eyelocal; attribute float eyeao;
varying vec3 vWPos; varying vec3 vL; varying float vAO;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWPos = wp.xyz; vL = eyelocal; vAO = eyeao;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;
const EYE_FS = /* glsl */`
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

// ------------------------------------------------------------------ hair material
const HAIR_VS = /* glsl */`
${SKIN_GLSL}
attribute vec4 tangent4; attribute vec4 hattr; attribute float strand;
uniform float uWidth; uniform float uTipWidth; uniform float uViewportH; uniform float uMinPix;
// three texels per strand: the rows of [scalp size * frame of the root triangle | root] on the current body
uniform highp sampler2D uStrands;
varying vec3 vWPos; varying vec3 vT; varying vec4 vH; varying float vCov;
void main() {
  // the point and its tangent keep their coordinates in the frame of the root triangle (AnnyBody.followStrands)
  int ts = int(strand + 0.5) * 3, tw = textureSize(uStrands, 0).x;
  vec4 m0 = texelFetch(uStrands, ivec2(ts % tw, ts / tw), 0);
  vec4 m1 = texelFetch(uStrands, ivec2((ts + 1) % tw, (ts + 1) / tw), 0);
  vec4 m2 = texelFetch(uStrands, ivec2((ts + 2) % tw, (ts + 2) / tw), 0);
  vec3 rest = vec3(dot(m0.xyz, position) + m0.w, dot(m1.xyz, position) + m1.w, dot(m2.xyz, position) + m2.w);
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
const HAIR_FS = /* glsl */`
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

// ------------------------------------------------------------------ procedural textures
function fsQuadRender(fs, size, type = THREE.UnsignedByteType, mip = true) {
  const rt = new THREE.WebGLRenderTarget(size, size, { type, generateMipmaps: mip, minFilter: mip ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter, magFilter: THREE.LinearFilter, wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, depthBuffer: false });
  const mat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: `in vec3 position; out vec2 vUv; void main(){ vUv = position.xy*0.5+0.5; gl_Position = vec4(position.xy,0.0,1.0); }`,
    fragmentShader: `precision highp float; in vec2 vUv; out vec4 outColor; ${fs}`,
  });
  const m = new THREE.Mesh(triGeo, mat); m.frustumCulled = false;
  const sc = new THREE.Scene(); sc.add(m);
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(rt); renderer.render(sc, new THREE.OrthographicCamera()); renderer.setRenderTarget(prev);
  mat.dispose();
  rt.texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  rt.texture.userData.rt = rt;
  return rt.texture;
}
const NOISE_GLSL = /* glsl */`
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
function makeDetailTexture(size) {
  return fsQuadRender(`${NOISE_GLSL}
  float height(vec2 uv, out float cav) {
    vec2 w = worley(uv, 72.0);
    float pr = hash1(floor(uv * 72.0) + 0.5);
    float pore = smoothstep(0.30, 0.02, w.x) * (0.55 + 0.9 * pr);
    vec2 w2 = worley(uv + 0.37, 26.0);
    float furrow = smoothstep(0.08, 0.0, w2.y - w2.x) * (0.5 + 0.5 * vnoise(uv, 40.0));
    vec2 w3 = worley(uv + 0.71, 9.0);
    float furrow2 = smoothstep(0.035, 0.0, w3.y - w3.x) * smoothstep(0.3, 0.8, vnoise(uv + 0.5, 12.0));
    float bump = vnoise(uv, 160.0) * 0.5 + vnoise(uv, 320.0) * 0.25 + vnoise(uv, 48.0) * 0.6;
    float und = vnoise(uv, 12.0) * 0.6 + vnoise(uv, 24.0) * 0.4;
    cav = 1.0 - 0.5 * pore - 0.2 * furrow - 0.15 * furrow2;
    return -pore - furrow * 0.35 - furrow2 * 0.25 + bump * 0.3 + und * 9.0;
  }
  void main() {
    float e = 1.0 / ${size}.0; float c, c2;
    float s = 0.55 * ${size}.0 / 1024.0;
    float h = height(vUv, c);
    float hx = height(vUv + vec2(e, 0.0), c2);
    float hy = height(vUv + vec2(0.0, e), c2);
    vec3 n = normalize(vec3((h - hx) * s, (h - hy) * s, 1.0));
    outColor = vec4(n.xy * 0.5 + 0.5, n.z, c);
  }`, size);
}
// facial pores: sparse round pits with a faint rim, and a denser set of smaller pits
function makePoreTexture(size) {
  return fsQuadRender(`${NOISE_GLSL}
  vec2 poreCell(vec2 uv, float n) {
    vec2 p = uv * n; vec2 ip = floor(p); vec2 fp = fract(p);
    float f1 = 8.0, id = 0.0;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
      vec2 g = vec2(float(i), float(j)); vec2 cell = mod(ip + g, n); vec2 o = hash2(cell);
      vec2 r = g + o - fp; float d = dot(r, r);
      if (d < f1) { f1 = d; id = hash1(cell + 0.37); }
    }
    return vec2(sqrt(f1), id);
  }
  float height(vec2 uv, out float cav) {
    vec2 c = poreCell(uv, 34.0);
    float r = 0.085 + 0.075 * c.y;
    float keep = step(0.15, c.y);
    float pit = smoothstep(r, r * 0.3, c.x) * keep;
    float rim = smoothstep(r * 2.2, r * 1.2, c.x) * (1.0 - pit) * keep;
    vec2 c2 = poreCell(uv + 0.21, 61.0);
    float r2 = 0.07 + 0.05 * c2.y;
    float pit2 = smoothstep(r2, r2 * 0.3, c2.x) * step(0.4, c2.y);
    cav = 1.0 - 0.7 * pit - 0.35 * pit2;
    return -pit + rim * 0.12 - pit2 * 0.45;
  }
  void main() {
    float e = 1.0 / ${size}.0; float c, c2;
    float s = 0.55 * ${size}.0 / 1024.0;
    float h = height(vUv, c);
    float hx = height(vUv + vec2(e, 0.0), c2);
    float hy = height(vUv + vec2(0.0, e), c2);
    vec3 n = normalize(vec3((h - hx) * s, (h - hy) * s, 1.0));
    outColor = vec4(n.xy * 0.5 + 0.5, n.z, c);
  }`, size);
}
const glv3 = (c) => `vec3(${c.map(v => v.toFixed(5)).join(', ')})`;
// mid: linear iris colour; the inner ring is lighter and warmer, the outer ring darker
function makeIrisTexture(mid = [0.038, 0.018, 0.0075]) {
  const inner = [mid[0] * 1.97, mid[1] * 1.89, mid[2] * 1.47], outer = [mid[0] * 0.53, mid[1] * 0.56, mid[2] * 0.73];
  const tex = fsQuadRender(`${NOISE_GLSL}
  void main() {
    vec2 p = vUv * 2.0 - 1.0; float r = length(p); float a = atan(p.y, p.x) / 6.2831853 + 0.5;
    float rp = 0.29;
    float t = clamp((r - rp) / (1.0 - rp), 0.0, 1.0);
    float fib = vnoise(vec2(a, r * 0.15), 180.0) * 0.6 + vnoise(vec2(a, r * 0.3), 90.0) * 0.4;
    float fib2 = vnoise(vec2(a, r * 0.5), 400.0);
    vec3 inner = ${glv3(inner)};
    vec3 mid = ${glv3(mid)};
    vec3 outer = ${glv3(outer)};
    vec3 col = mix(inner, mid, smoothstep(0.0, 0.45, t));
    col = mix(col, outer, smoothstep(0.45, 1.0, t));
    float coll = 0.30 + 0.05 * sin(a * 6.2831853 * 11.0 + vnoise(vec2(a, 0.0), 30.0) * 4.0);
    col *= 1.0 + 0.55 * smoothstep(0.06, 0.0, abs(t - coll));
    col *= 0.65 + 0.7 * fib;
    col *= 0.85 + 0.3 * fib2;
    float crypt = smoothstep(0.55, 0.85, vnoise(vec2(a, r), 26.0)) * smoothstep(0.25, 0.4, t) * smoothstep(0.85, 0.6, t);
    col *= 1.0 - 0.45 * crypt;
    col *= mix(1.0, 0.25, smoothstep(0.82, 1.0, t));
    col = mix(col, vec3(0.02, 0.012, 0.008), smoothstep(0.06, 0.0, t));
    outColor = vec4(col, 1.0);
  }`, 512, THREE.HalfFloatType, true);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}

// ------------------------------------------------------------------ shadows
const SHADOW_SIZE = (!isSmall && renderer.capabilities.maxTextureSize >= 4096) ? 4096 : 2048;
function makeShadowRT(size) {
  const rt = new THREE.WebGLRenderTarget(size, size, { type: FLOAT_RT ? THREE.FloatType : THREE.UnsignedByteType, format: FLOAT_RT ? THREE.RedFormat : THREE.RGBAFormat, depthBuffer: true, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  rt.depthTexture = new THREE.DepthTexture(size, size, THREE.UnsignedIntType);
  rt.depthTexture.compareFunction = THREE.LessEqualCompare;
  rt.depthTexture.minFilter = THREE.LinearFilter; rt.depthTexture.magFilter = THREE.LinearFilter;
  return rt;
}
const shadowRT = makeShadowRT(SHADOW_SIZE);
const hairShadowRT = makeShadowRT(SHADOW_SIZE / 2);
const SHADOW_HALF = 0.17;
const shadowCam = new THREE.OrthographicCamera(-SHADOW_HALF, SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, 0.3, 1.3);
const depthMat = new THREE.ShaderMaterial({
  defines: FLOAT_RT ? {} : { SHADOW_PACKED: '' },
  vertexShader: SKIN_GLSL + '\nvoid main(){ gl_Position = projectionMatrix * modelViewMatrix * (skinMat() * vec4(position, 1.0)); }',
  fragmentShader: '#include <packing>\nvoid main(){\n#ifdef SHADOW_PACKED\n gl_FragColor = packDepthToRGBA(gl_FragCoord.z);\n#else\n gl_FragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);\n#endif\n}',
  side: THREE.DoubleSide,
});
U.uShadowRaw.value = shadowRT.texture;
U.uShadowCmp.value = shadowRT.depthTexture;
U.uHairShadow.value = hairShadowRT.texture;
const bodyShadowRT = makeShadowRT(SHADOW_SIZE);
const BODY_HALF = 0.95;
const bodyShadowCam = new THREE.OrthographicCamera(-BODY_HALF, BODY_HALF, BODY_HALF, -BODY_HALF, 0.3, 3.6);
U.uBodyShadowCmp.value = bodyShadowRT.depthTexture;
U.uBodyShadowRaw.value = bodyShadowRT.texture;
// floor: a shadow catcher that takes the backdrop colour, so only the shadow and the contact shade show
let GROUND_Y = -0.8415;
const floorMat = new THREE.ShaderMaterial({
  uniforms: U,
  vertexShader: 'varying vec3 vWPos; void main(){ vec4 wp = modelMatrix * vec4(position, 1.0); vWPos = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }',
  fragmentShader: `${GLSL_COMMON}
${BG_GLSL}
varying vec3 vWPos;
uniform vec4 uFootL; uniform vec4 uFootR;
float footBlob(vec2 p, vec4 f) { vec2 d = (p - f.xy) / f.z; return f.w * exp(-0.5 * dot(d, d)); }
void main() {
  vec3 bg = bgColor();
  float vis = bodyShadow(vWPos, vec3(0.0, 1.0, 0.0));
  float fade = smoothstep(1.4, 0.3, length(vWPos.xz - vec2(0.0, 0.05)));
  float contact = max(footBlob(vWPos.xz, uFootL), footBlob(vWPos.xz, uFootR));
  float dark = clamp((1.0 - vis) * 0.55 * fade + contact * 0.45, 0.0, 0.85);
  gl_FragColor = vec4(bg * (1.0 - dark), 1.0);
}`,
  defines: FLOAT_RT ? {} : { SHADOW_PACKED: '' }, glslVersion: THREE.GLSL3, blending: THREE.NoBlending,
});
floorMat.fragmentShader = toMRT(floorMat.fragmentShader);
const floorGeo = new THREE.PlaneGeometry(4, 4); floorGeo.rotateX(-Math.PI / 2);
const floorMesh = new THREE.Mesh(floorGeo, floorMat);
floorMesh.position.y = GROUND_Y; floorMesh.renderOrder = -5;
scene.add(floorMesh);
// what the two shadow maps look at: the head, and the whole figure (both follow the pose)
const SHADOW_FOCUS = { head: new THREE.Vector3(0, 0.5, 0.03), body: new THREE.Vector3(0, -0.12, 0.03), bodyHalf: 0.95 };
function updateShadowCamera() {
  const L = U.uKeyDir.value;
  const c = SHADOW_FOCUS.head.clone();
  shadowCam.position.copy(c).addScaledVector(L, 0.8);
  shadowCam.up.set(0, 1, 0);
  if (Math.abs(L.y) > 0.95) shadowCam.up.set(0, 0, 1);
  shadowCam.lookAt(c);
  shadowCam.updateMatrixWorld(); shadowCam.updateProjectionMatrix();
  const bias = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
  U.uShadowMat.value.copy(bias).multiply(shadowCam.projectionMatrix).multiply(shadowCam.matrixWorldInverse);
  U.uShadowRot.value.setFromMatrix4(shadowCam.matrixWorldInverse);
  const range = shadowCam.far - shadowCam.near, width = SHADOW_HALF * 2;
  U.uShadowP.value.set(1 / SHADOW_SIZE, range * 2 * U.uKeyTan.value / width, range, width / SHADOW_SIZE);
  const cb = SHADOW_FOCUS.body.clone();
  const bh = SHADOW_FOCUS.bodyHalf;
  bodyShadowCam.left = -bh; bodyShadowCam.right = bh; bodyShadowCam.top = bh; bodyShadowCam.bottom = -bh;
  bodyShadowCam.position.copy(cb).addScaledVector(L, 1.9);
  bodyShadowCam.up.set(0, 1, 0);
  if (Math.abs(L.y) > 0.95) bodyShadowCam.up.set(0, 0, 1);
  bodyShadowCam.lookAt(cb);
  bodyShadowCam.updateMatrixWorld(); bodyShadowCam.updateProjectionMatrix();
  U.uBodyShadowMat.value.copy(bias).multiply(bodyShadowCam.projectionMatrix).multiply(bodyShadowCam.matrixWorldInverse);
  const brange = bodyShadowCam.far - bodyShadowCam.near;
  // x: texel size in uv, y: depth bias, z: filter radius in texels (wider for softer key lights), w: depth range (m)
  U.uBodyShadowP.value.set(1 / SHADOW_SIZE, 0.0012 / brange, 1.5 + 18.0 * U.uKeyTan.value * (SHADOW_SIZE / 4096), brange);
}

// ------------------------------------------------------------------ data decoding
function b64ToBytes(b64) {
  const bin = atob(b64); const n = bin.length; const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function gunzip(bytes) {
  if (typeof DecompressionStream !== 'undefined') {
    const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  const { gunzipSync } = await import('https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js');
  return gunzipSync(bytes);
}
function parseContainer(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const hlen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hlen)));
  const base = 8 + hlen + ((4 - (hlen % 4)) % 4);
  const B = {};
  for (const b of header.buffers) {
    const src = u8.subarray(base + b.byteOffset, base + b.byteOffset + b.byteLength);
    if (b.enc === 'mv') { const out = new Uint8Array(b.count * b.stride); MeshoptDecoder.decodeVertexBuffer(out, b.count, b.stride, src); B[b.name] = { info: b, data: out }; }
    else if (b.enc === 'mi') { const out = new Uint32Array(b.count); MeshoptDecoder.decodeIndexBuffer(new Uint8Array(out.buffer), b.count, 4, src); B[b.name] = { info: b, data: out }; }
    else B[b.name] = { info: b, data: src };
  }
  return { meta: header.meta, B };
}

// ------------------------------------------------------------------ geometry builders
function buildHead(buf) {
  const { info, data } = buf;
  const n = info.count, st = info.stride, lo = info.lo, hi = info.hi;
  const pos = new Float32Array(n * 3), alb = new Uint8Array(n * 3), aA = new Uint8Array(n * 4), aB = new Uint8Array(n * 4), aC = new Uint8Array(n * 4), aD = new Uint8Array(n * 4), aW = new Uint8Array(n * 4), ns = new Float32Array(n * 3);
  const aV = new Uint8Array(n * 4);
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let i = 0; i < n; i++) {
    const o = i * st;
    for (let k = 0; k < 3; k++) pos[i * 3 + k] = lo[k] + dv.getUint16(o + k * 2, true) / 65535 * (hi[k] - lo[k]);
    alb[i * 3] = data[o + 6]; alb[i * 3 + 1] = data[o + 7]; alb[i * 3 + 2] = data[o + 8];
    aA[i * 4] = data[o + 9]; aA[i * 4 + 1] = data[o + 10]; aA[i * 4 + 2] = data[o + 11]; aA[i * 4 + 3] = data[o + 12];
    aB[i * 4] = data[o + 13]; aB[i * 4 + 1] = data[o + 14]; aB[i * 4 + 2] = data[o + 15]; aB[i * 4 + 3] = data[o + 22];
    aC[i * 4] = data[o + 18]; aC[i * 4 + 1] = data[o + 19]; aC[i * 4 + 2] = data[o + 20]; aC[i * 4 + 3] = data[o + 21];
    aD[i * 4] = data[o + 23];
    aW[i * 4] = data[o + 24]; aW[i * 4 + 1] = data[o + 25]; aW[i * 4 + 2] = data[o + 26]; aW[i * 4 + 3] = data[o + 27];
    if (st >= 32) { aV[i * 4] = data[o + 28]; aV[i * 4 + 1] = data[o + 29]; aV[i * 4 + 2] = data[o + 30]; aV[i * 4 + 3] = data[o + 31]; }
    let x = dv.getInt8(o + 16) / 127, y = dv.getInt8(o + 17) / 127;
    let z = 1 - Math.abs(x) - Math.abs(y);
    if (z < 0) { const ox = x, oy = y; x = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1); y = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1); }
    const l = Math.hypot(x, y, z) || 1;
    ns[i * 3] = x / l; ns[i * 3 + 1] = y / l; ns[i * 3 + 2] = z / l;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('rest', new THREE.BufferAttribute(pos.slice(), 3));
  g.setAttribute('albedo', new THREE.BufferAttribute(alb, 3, true));
  g.setAttribute('attrA', new THREE.BufferAttribute(aA, 4, true));
  g.setAttribute('attrB', new THREE.BufferAttribute(aB, 4, true));
  g.setAttribute('attrC', new THREE.BufferAttribute(aC, 4, true));
  g.setAttribute('attrD', new THREE.BufferAttribute(aD, 4, true));
  g.setAttribute('attrW', new THREE.BufferAttribute(aW, 4, true));
  g.setAttribute('attrV', new THREE.BufferAttribute(aV, 4, true));
  g.setAttribute('nsmooth', new THREE.BufferAttribute(ns, 3));
  return g;
}
function addSkin(g, buf) {
  if (!buf) return 0;
  const n = buf.info.count, d = buf.data, st = buf.info.stride, K = st / 2;
  for (let set = 0; set < K / 4; set++) {
    const si = new Uint8Array(n * 4), sw = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) { si[i * 4 + k] = d[i * st + set * 4 + k]; sw[i * 4 + k] = d[i * st + K + set * 4 + k]; }
    const sfx = set ? String(set + 1) : '';
    g.setAttribute('skinIndex' + sfx, new THREE.BufferAttribute(si, 4));
    g.setAttribute('skinWeight' + sfx, new THREE.BufferAttribute(sw, 4, true));
  }
  return K;
}
const SKIN8_DEF = { SKIN8: '' };
// a mesh on the rig (a plain mesh when the model has no rig)
function skinned(geo, mat) {
  if (!RIG.ready || !geo.attributes.skinIndex) return new THREE.Mesh(geo, mat);
  const m = new THREE.SkinnedMesh(geo, mat);
  m.bind(RIG.skel, new THREE.Matrix4());
  m.frustumCulled = false;
  return m;
}
function buildEye(e, meta, aoBytes) {
  const nlat = meta.eye_mesh.nlat, nlon = meta.eye_mesh.nlon;
  const Rs = meta.eyes.R, Rc = meta.eyes.Rc, zcc = meta.eyes.zcc;
  const R = e.rot, c = e.center;
  const nv = (nlat + 1) * (nlon + 1);
  const pos = new Float32Array(nv * 3), loc = new Float32Array(nv * 3), ao = new Float32Array(nv);
  let v = 0;
  for (let i = 0; i <= nlat; i++) {
    const th = Math.PI * Math.pow(i / nlat, 1.6);
    for (let j = 0; j <= nlon; j++, v++) {
      const ph = 2 * Math.PI * j / nlon;
      const dx = Math.sin(th) * Math.cos(ph), dy = Math.sin(th) * Math.sin(ph), dz = Math.cos(th);
      const b = dz * zcc, disc = b * b - (zcc * zcc - Rc * Rc);
      const tc = disc >= 0 ? b + Math.sqrt(disc) : 0;
      const k = 0.00025, hh = Math.min(1, Math.max(0, 0.5 + 0.5 * (tc - Rs) / k));
      const t = tc * hh + Rs * (1 - hh) + k * hh * (1 - hh);
      const px = dx * t, py = dy * t, pz = dz * t;
      loc[v * 3] = px; loc[v * 3 + 1] = py; loc[v * 3 + 2] = pz;
      pos[v * 3] = R[0][0] * px + R[0][1] * py + R[0][2] * pz + c[0];
      pos[v * 3 + 1] = R[1][0] * px + R[1][1] * py + R[1][2] * pz + c[1];
      pos[v * 3 + 2] = R[2][0] * px + R[2][1] * py + R[2][2] * pz + c[2];
      ao[v] = aoBytes[v] / 255;
    }
  }
  const idx = [];
  const id = (i, j) => i * (nlon + 1) + j;
  for (let i = 0; i < nlat; i++) for (let j = 0; j < nlon; j++) {
    const a = id(i, j), b = id(i + 1, j), cc = id(i + 1, j + 1), d = id(i, j + 1);
    idx.push(a, b, cc, a, cc, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('eyelocal', new THREE.BufferAttribute(loc, 3));
  g.setAttribute('eyeao', new THREE.BufferAttribute(ao, 1));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}
function decodeStrands(B, name) {
  const r = B[name + '_r'], d = B[name + '_d'], n = B[name + '_n'];
  const nS = n.info.strands, lo = r.info.lo, hi = r.info.hi, dq = d.info.dq;
  const counts = n.data.subarray(0, nS);
  let total = 0; for (let i = 0; i < nS; i++) total += counts[i];
  const P = new Float32Array(total * 3);
  const rv = new DataView(r.data.buffer, r.data.byteOffset, r.data.byteLength);
  const dd = new Int8Array(d.data.buffer, d.data.byteOffset, d.data.byteLength);
  let p = 0, q = 0;
  for (let i = 0; i < nS; i++) {
    let x = lo[0] + rv.getUint16(i * 6, true) / 65535 * (hi[0] - lo[0]);
    let y = lo[1] + rv.getUint16(i * 6 + 2, true) / 65535 * (hi[1] - lo[1]);
    let z = lo[2] + rv.getUint16(i * 6 + 4, true) / 65535 * (hi[2] - lo[2]);
    P[p++] = x; P[p++] = y; P[p++] = z;
    for (let j = 1; j < counts[i]; j++) {
      x += dd[q++] * dq; y += dd[q++] * dq; z += dd[q++] * dq;
      P[p++] = x; P[p++] = y; P[p++] = z;
    }
  }
  const sk = B[name + '_skin'];
  return { P, counts, nS, total, skin: sk ? sk.data.subarray(0, nS * 8) : null };
}
// density-based self occlusion for hair points (how much hair lies outward of a point)
function hairOcclusion(P, total, center, k = 0.1) {
  let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (let i = 0; i < total; i++) for (let a = 0; a < 3; a++) { const v = P[i * 3 + a]; if (v < mn[a]) mn[a] = v; if (v > mx[a]) mx[a] = v; }
  const h = 0.0015; mn = mn.map(v => v - 0.01);
  const dims = mx.map((v, a) => Math.ceil((v + 0.01 - mn[a]) / h) + 1);
  const [dx, dy, dz] = dims;
  let grid = new Float32Array(dx * dy * dz);
  for (let i = 0; i < total; i++) {
    const gx = Math.floor((P[i * 3] - mn[0]) / h), gy = Math.floor((P[i * 3 + 1] - mn[1]) / h), gz = Math.floor((P[i * 3 + 2] - mn[2]) / h);
    grid[(gx * dy + gy) * dz + gz] += 1;
  }
  // separable [1,2,1]/4 blur x2 (approx gaussian)
  const tmp = new Float32Array(grid.length);
  const blur = (src, dst, stride, len, outer) => {
    for (let o = 0; o < src.length; o++) {
      const c = Math.floor(o / stride) % len;
      const a = c > 0 ? src[o - stride] : 0, b = c < len - 1 ? src[o + stride] : 0;
      dst[o] = 0.25 * a + 0.5 * src[o] + 0.25 * b;
    }
  };
  for (let pass = 0; pass < 2; pass++) { blur(grid, tmp, dy * dz, dx); blur(tmp, grid, dz, dy); blur(grid, tmp, 1, dz); grid.set(tmp); }
  const ao = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    let x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
    let vx = x - center[0], vy = y - center[1], vz = z - center[2];
    const l = Math.hypot(vx, vy, vz) || 1; vx /= l; vy /= l; vz /= l;
    let acc = 0;
    for (let s = 1; s < 14; s++) {
      const gx = Math.floor((x + vx * s * h - mn[0]) / h), gy = Math.floor((y + vy * s * h - mn[1]) / h), gz = Math.floor((z + vz * s * h - mn[2]) / h);
      if (gx < 0 || gy < 0 || gz < 0 || gx >= dx || gy >= dy || gz >= dz) continue;
      acc += grid[(gx * dy + gy) * dz + gz];
    }
    ao[i] = Math.exp(-k * acc);
  }
  return ao;
}
// lighter hair for phones: keep every k-th strand (the caller widens the strands to keep the same coverage)
function subsetStrands(S, ao, k) {
  const { P, counts, nS } = S;
  const starts = new Uint32Array(nS); let p = 0;
  for (let i = 0; i < nS; i++) { starts[i] = p; p += counts[i]; }
  let n = 0, total = 0;
  for (let i = 0; i < nS; i += k) { n++; total += counts[i]; }
  const P2 = new Float32Array(total * 3), ao2 = new Float32Array(total), c2 = new Uint8Array(n);
  const sk2 = S.skin ? new Uint8Array(n * 8) : null;
  let q = 0, j = 0;
  for (let i = 0; i < nS; i += k, j++) {
    const c = counts[i]; c2[j] = c;
    P2.set(P.subarray(starts[i] * 3, (starts[i] + c) * 3), q * 3);
    ao2.set(ao.subarray(starts[i], starts[i] + c), q);
    if (sk2) sk2.set(S.skin.subarray(i * 8, i * 8 + 8), j * 8);
    q += c;
  }
  const keep = Uint32Array.from({ length: n }, (_, q) => q * k);
  return { S: { P: P2, counts: c2, nS: n, total, skin: sk2 }, ao: ao2, keep };
}
// S.P: the points in the frames of their root triangles; ids: the strand of each (for a subset of the strands)
function buildRibbons(S, ao, seed, ids = null) {
  const { P, counts, nS, total } = S;
  const nv = total * 2;
  const pos = new Float32Array(nv * 3), tan = new Int8Array(nv * 4), ha = new Uint8Array(nv * 4), sid = new Float32Array(nv);
  // skin: each strand takes the weights of the skin under its root (brows and lashes ride on the head bone)
  const skI = new Uint8Array(nv * 4), skW = new Uint8Array(nv * 4);
  { let v2 = 0;
    for (let i = 0; i < nS; i++) {
      for (let j = 0; j < counts[i] * 2; j++, v2++) {
        if (S.skin) for (let k = 0; k < 4; k++) { skI[v2 * 4 + k] = S.skin[i * 8 + k]; skW[v2 * 4 + k] = S.skin[i * 8 + 4 + k]; }
        else { skI[v2 * 4] = Math.max(RIG.head, 0); skW[v2 * 4] = 255; }
      }
    }
  }
  let segs = 0; for (let i = 0; i < nS; i++) segs += counts[i] - 1;
  const idx = new Uint32Array(segs * 6);
  let p = 0, v = 0, ii = 0; let rs = seed >>> 0;
  const rand = () => { rs = (rs * 1664525 + 1013904223) >>> 0; return rs / 4294967296; };
  for (let i = 0; i < nS; i++) {
    const n = counts[i], r = Math.floor(rand() * 255);
    for (let j = 0; j < n; j++) {
      const a = p + Math.max(j - 1, 0), b = p + Math.min(j + 1, n - 1);
      let tx = P[b * 3] - P[a * 3], ty = P[b * 3 + 1] - P[a * 3 + 1], tz = P[b * 3 + 2] - P[a * 3 + 2];
      const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
      const t = Math.round(j / (n - 1) * 255), o = Math.round((ao ? ao[p + j] : 0.8) * 255);
      for (let s = 0; s < 2; s++, v++) {
        pos[v * 3] = P[(p + j) * 3]; pos[v * 3 + 1] = P[(p + j) * 3 + 1]; pos[v * 3 + 2] = P[(p + j) * 3 + 2];
        tan[v * 4] = Math.round(tx * 127); tan[v * 4 + 1] = Math.round(ty * 127); tan[v * 4 + 2] = Math.round(tz * 127);
        ha[v * 4] = t; ha[v * 4 + 1] = s * 255; ha[v * 4 + 2] = r; ha[v * 4 + 3] = o; sid[v] = ids ? ids[i] : i;
      }
      if (j < n - 1) {
        const b0 = (p + j) * 2;
        idx[ii++] = b0; idx[ii++] = b0 + 1; idx[ii++] = b0 + 2; idx[ii++] = b0 + 1; idx[ii++] = b0 + 3; idx[ii++] = b0 + 2;
      }
    }
    p += n;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('tangent4', new THREE.BufferAttribute(tan, 4, true));
  g.setAttribute('hattr', new THREE.BufferAttribute(ha, 4, true));
  g.setAttribute('strand', new THREE.BufferAttribute(sid, 1));
  g.setAttribute('skinIndex', new THREE.BufferAttribute(skI, 4));
  g.setAttribute('skinWeight', new THREE.BufferAttribute(skW, 4, true));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.52, 0.03), 0.2);
  return g;
}

// ------------------------------------------------------------------ scene assembly
const opaque = []; const hairObjs = [];
let shadowsDirty = true;
let currentPreset = 'studio';
function applyPreset(name) {
  const p = PRESETS[name]; currentPreset = name;
  const env = buildEnvironment(p);
  if (U.uEnv.value) U.uEnv.value.dispose();
  U.uEnv.value = env.pmrem;
  const envk = buildEnvironment(p, true);
  if (U.uEnvKey.value) U.uEnvKey.value.dispose();
  U.uEnvKey.value = envk.pmrem;
  env.sh.forEach((v, i) => U.uSH.value[i].copy(v));
  U.uKeyDir.value.copy(dirAE(p.key.az, p.key.el));
  U.uKeyColor.value.set(...p.key.c);
  U.uKeyTan.value = Math.tan(p.key.size * DEG);
  U.uRimDir.value.copy(dirAE(p.rim.az, p.rim.el));
  U.uRimColor.value.set(...p.rim.c);
  BG.uBgA.value.set(...p.bg[0]); BG.uBgB.value.set(...p.bg[1]);
  renderer.toneMappingExposure = qs.get('exp') ? parseFloat(qs.get('exp')) : p.exposure;
  updateShadowCamera();
  shadowsDirty = true; resetAccum();
}

function makeHairMesh(geo, p, defines, strands) {
  const hu = Object.assign({}, U, {
    uStrands: { value: strands },
    uWidth: { value: p.width }, uTipWidth: { value: p.width * p.tip }, uViewportH: { value: innerHeight * PR }, uMinPix: { value: p.minPix ?? 0.7 },
    uHairColor: { value: new THREE.Vector3(...p.color) }, uHairRough: { value: p.rough }, uSpecScale: { value: p.spec ?? 1.0 }, uDiffScale: { value: p.diff ?? 1.3 },
    uHeadC: { value: new THREE.Vector3(...(p.center || [0, 0.515, 0.035])) },
  });
  const m = new THREE.ShaderMaterial({ uniforms: hu, vertexShader: HAIR_VS, fragmentShader: toMRT(HAIR_FS), defines, side: THREE.DoubleSide, glslVersion: THREE.GLSL3, blending: THREE.NoBlending });
  const dm = new THREE.ShaderMaterial({ uniforms: Object.assign({}, hu, { uViewportH: { value: SHADOW_SIZE / 2 }, uMinPix: { value: 1.2 } }), vertexShader: HAIR_VS, fragmentShader: toMRT(HAIR_FS), defines: Object.assign({ HAIR_DEPTH: '' }, defines), side: THREE.DoubleSide, glslVersion: THREE.GLSL3, blending: THREE.NoBlending });
  const mesh = skinned(geo, m);
  mesh.frustumCulled = false; mesh.userData.depthMat = dm; mesh.userData.hu = hu;
  mesh.userData.center = new THREE.Vector3(...(p.center || [0, 0.515, 0.035]));
  scene.add(mesh); hairObjs.push(mesh);
  return mesh;
}

// ------------------------------------------------------------------ body shape: anny's phenotype sliders
// The body comes from anny (body.ts): the sliders set anny's blend-shape coefficients, the coarse body and the
// joints follow, the fine surface is rebuilt by subdivision with its detail layers, and the eyes, the hair, the
// skeleton and the corrective shapes follow the body. Every slider runs from 0 to 1, and anny's default is 0.5.
const BODY: any = { ready: false, anny: null as AnnyBody | null, geo: null, values: null, sliders: [], lastMs: 0, lastTotalMs: 0, lastTiming: {} };
function sliderEnds(tables: any, label: string) {
  const v = tables.variations.find((x: any) => x[0] === label);
  const a = tables.anchors[label];
  if (!v || !a) return ['0', '1'];
  const at = (x: number) => { let k = 0; for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - x) < Math.abs(a[k] - x)) k = i; return v[1][k]; };
  const nice = (s: string) => s.replace(/(min|max|average)(\w+)/, '$1 $2').replace(/^./, (c: string) => c.toUpperCase());
  return [nice(at(0)), nice(at(1))];
}
function initBody(meta: any, B: any, geo: any) {
  const f32 = (name: string) => { const b = B[name]; return new Float32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4); };
  const i16 = (name: string) => { const b = B[name]; return new Int16Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 2); };
  const u32 = (name: string) => { const b = B[name]; return new Uint32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4); };
  const sm = meta.shape;
  const at = geo.attributes;
  const nF = at.position.count;
  // the detail records hold 4 values (the fourth pads the record); AnnyBody reads 3 per vertex
  const d4 = new Int16Array(B.head_detail.data.buffer, B.head_detail.data.byteOffset, nF * 4), detailRaw = new Int16Array(nF * 3);
  for (let i = 0; i < nF; i++) { detailRaw[i * 3] = d4[i * 4]; detailRaw[i * 3 + 1] = d4[i * 4 + 1]; detailRaw[i * 3 + 2] = d4[i * 4 + 2]; }
  const rows = new Uint32Array(B.head_row.data.buffer, B.head_row.data.byteOffset, nF);
  // the shape components come as records of 4 values (x, y, z and a pad)
  const c4 = i16('shape_components'), nComp = sm.components * sm.coarse_vertices, components = new Int16Array(nComp * 3);
  for (let i = 0; i < nComp; i++) { components[i * 3] = c4[i * 4]; components[i * 3 + 1] = c4[i * 4 + 1]; components[i * 3 + 2] = c4[i * 4 + 2]; }
  const body = new AnnyBody(meta, {
    template: f32('coarse_template'), components,
    projection: f32('shape_projection').subarray(0, sm.components * sm.blend_shapes),
    jointTemplate: f32('joint_template'), jointBlend: f32('joint_blend'),
    quads: u32('coarse_quads'), rows, detail: detailRaw,
    index: geo.index.array, rest: at.rest.array, nsmooth: at.nsmooth.array.slice(), coarseSkin: B.coarse_skin.data,
  }, at.position.array, at.normal.array, at.nsmooth.array);
  body.detailStep = B.head_detail.info.step || 1e-5;
  BODY.anny = body; BODY.geo = geo;
  BODY.sliders = sm.sliders.map((name: string) => ({ name, label: name.charAt(0).toUpperCase() + name.slice(1), ends: sliderEnds(sm.tables, name) }));
  BODY.ready = true;
}
function sameValues(a: any, b: any) {
  if (!a || !b) return false;
  return BODY.sliders.every((s: any) => Math.abs((a[s.name] ?? 0.5) - (b[s.name] ?? 0.5)) < 1e-6);
}
// the head of anny's default body -> the head of the current body (both at rest): a move and a uniform scale
const HEADMAP = { def: new THREE.Vector3(), cur: new THREE.Vector3(), k: 1, m: new THREE.Matrix4(), inv: new THREE.Matrix4() };
function updateHeadMap() {
  const b = BODY.anny, J = b.joints, J0 = b.jointsDefault, h = RIG.head;
  if (h < 0) return;
  HEADMAP.def.set(J0[h * 3], J0[h * 3 + 1], J0[h * 3 + 2]);
  HEADMAP.cur.set(J[h * 3], J[h * 3 + 1], J[h * 3 + 2]);
  HEADMAP.k = b.headScale();
  const k = HEADMAP.k;
  HEADMAP.m.makeTranslation(HEADMAP.cur.x, HEADMAP.cur.y, HEADMAP.cur.z).multiply(new THREE.Matrix4().makeScale(k, k, k))
    .multiply(new THREE.Matrix4().makeTranslation(-HEADMAP.def.x, -HEADMAP.def.y, -HEADMAP.def.z));
  HEADMAP.inv.copy(HEADMAP.m).invert();
}
function applyBodyShape(values: any) {
  if (!BODY.ready) return false;
  const v: any = {};
  for (const s of BODY.sliders) v[s.name] = typeof values?.[s.name] === 'number' ? values[s.name] : 0.5;
  if (sameValues(v, BODY.values)) return false;
  const t0 = performance.now(), steps: any = {};
  let last = t0;
  const mark = (name: string) => { const t = performance.now(); steps[name] = Math.round((t - last) * 10) / 10; last = t; };
  BODY.values = v;
  clearCorrectives();
  const r = BODY.anny.update(v);
  last = performance.now();
  const at = BODY.geo.attributes;
  markFull(at.position); markFull(at.normal); markFull(at.nsmooth);
  // the bounds from the coarse body (the fine surface lies within a few millimetres of it)
  bodySphere(BODY.geo.boundingSphere, BODY.anny.coarse, BODY.anny.nBody);
  mark('bounds');
  updateHairShape();
  mark('ribbons');
  updateEyeShape();
  rebaseCorrectives();
  mark('correctives');
  rigForBody();
  mark('rig');
  shadowsDirty = true; resetAccum();
  // the time of the last update: the body alone (with its steps) and everything that follows it
  BODY.lastMs = r.ms; BODY.lastTotalMs = performance.now() - t0;
  BODY.lastTiming = Object.assign({}, BODY.anny.timing, steps);
  return true;
}
function bodySphere(sphere: any, V: Float32Array, n: number) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { const x = V[i * 3 + k]; if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x; }
  sphere.center.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
  sphere.radius = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) + 0.02;
}
function updateEyeShape() {
  for (const e of RIG.eyes) {
    const s = e.side, c0 = e.center0, g = BODY.anny.eyes[s];
    if (!g) continue;
    e.shape.makeTranslation(g.center[0], g.center[1], g.center[2]).multiply(new THREE.Matrix4().makeScale(g.scale, g.scale, g.scale))
      .multiply(new THREE.Matrix4().makeTranslation(-c0[0], -c0[1], -c0[2]));
  }
}

// ------------------------------------------------------------------ rig
// Every bone rests with the world axes at its head, so a bone's rest offset is its head minus its parent's head and
// its inverse bind matrix is a plain translation. The rest heads come from anny's joints for the current sliders.
const RIG: any = { ready: false, bones: [], skel: null, parents: null, heads: null, root: -1, head: -1, eyes: [], holder: null };
function buildRig(rig: any, J: Float32Array) {
  RIG.parents = rig.parents;
  RIG.heads = rig.names.map((_: string, i: number) => [J[i * 3], J[i * 3 + 1], J[i * 3 + 2]]);
  RIG.bones = rig.names.map((name: string) => { const b = new THREE.Bone(); b.name = name; return b; });
  rig.parents.forEach((p: number, i: number) => { if (p >= 0) RIG.bones[p].add(RIG.bones[i]); });
  RIG.root = rig.parents.indexOf(-1); RIG.head = rig.names.indexOf('head');
  RIG.holder = new THREE.Group(); RIG.holder.add(RIG.bones[RIG.root]); scene.add(RIG.holder);
  RIG.skel = new THREE.Skeleton(RIG.bones, RIG.bones.map(() => new THREE.Matrix4()));
  RIG.ready = true;
  setRigRest();
}
function setRigRest() {
  const H = RIG.heads, P = RIG.parents;
  RIG.bones.forEach((b: any, i: number) => {
    const p = P[i];
    if (p >= 0) b.position.set(H[i][0] - H[p][0], H[i][1] - H[p][1], H[i][2] - H[p][2]);
    RIG.skel.boneInverses[i].makeTranslation(-H[i][0], -H[i][1], -H[i][2]);
  });
  placeRoot();
}
function placeRoot() {
  const r = RIG.root, h = RIG.heads[r], o = MOTION.rootOff;
  RIG.bones[r].position.set(h[0] + o[0], h[1] + o[1], h[2] + o[2]);
}
// the joints follow the body: anny's bone heads for the current sliders
function rigForBody() {
  if (!RIG.ready) return;
  const J = BODY.anny.joints;
  for (let i = 0; i < RIG.heads.length; i++) { RIG.heads[i][0] = J[i * 3]; RIG.heads[i][1] = J[i * 3 + 1]; RIG.heads[i][2] = J[i * 3 + 2]; }
  updateHeadMap();
  updateCorrectiveScales();
  groundMotion();
  setRigRest();
  evalMotion();
}
// the height of the root above the floor: the library scales root offsets, stools and bounds by it
function hipHeight() { return RIG.heads[RIG.root][1] - BODY.anny.floor; }
// world matrices (column-major, one per bone, rest heads removed) for library rotations and a root offset
function poseMatrices(q: Float32Array, root: number[]) {
  const n = RIG.heads.length, H = RIG.heads, P = RIG.parents;
  const R = new Float64Array(n * 9), T = new Float64Array(n * 3), out = new Float32Array(n * 16);
  for (let i = 0; i < n; i++) {
    const x = q[i * 4], y = q[i * 4 + 1], z = q[i * 4 + 2], w = q[i * 4 + 3];
    const l = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
    const p = P[i];
    if (p < 0) { for (let k = 0; k < 9; k++) R[i * 9 + k] = l[k]; for (let k = 0; k < 3; k++) T[i * 3 + k] = H[i][k] + root[k]; }
    else {
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        let s = 0; for (let k = 0; k < 3; k++) s += R[p * 9 + r * 3 + k] * l[k * 3 + c];
        R[i * 9 + r * 3 + c] = s;
      }
      const d = [H[i][0] - H[p][0], H[i][1] - H[p][1], H[i][2] - H[p][2]];
      for (let r = 0; r < 3; r++) T[i * 3 + r] = T[p * 3 + r] + R[p * 9 + r * 3] * d[0] + R[p * 9 + r * 3 + 1] * d[1] + R[p * 9 + r * 3 + 2] * d[2];
    }
    // skin matrix: x -> R (x - head) + T, column-major
    const m = i * 16;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[m + c * 4 + r] = R[i * 9 + r * 3 + c];
    for (let r = 0; r < 3; r++) out[m + 12 + r] = T[i * 3 + r] - (R[i * 9 + r * 3] * H[i][0] + R[i * 9 + r * 3 + 1] * H[i][1] + R[i * 9 + r * 3 + 2] * H[i][2]);
    out[m + 15] = 1;
  }
  return out;
}

// ------------------------------------------------------------------ corrective shapes
// anny's soft-tissue shapes (anny.correctives) correct plain skinning at the shoulders, elbows, hips and knees.
// Drivers read the skeleton on every pose change: an elbow or a knee by its bend angle, a shoulder or a hip by the
// direction of the upper bone in the frame of the torso bone above it. The shapes add to the rest shape before
// skinning, as morph targets do in a game engine, and each one scales with the size of the body around it
// (anny.correctives.SoftTissueCorrectives.shape_scales). CORR.base holds the rest shape with no correction.
const CORR: any = { ready: false, on: true, shapes: [], joints: [], w: null, wn: null, wPrev: null, scale: null, touched: null, nt: 0, mark: null, acc: null,
  base: null, nbase: null, sbase: null };
function initCorrectives(meta: any, B: any) {
  const cm = meta.correctives, buf = B.corr, sup = B.corr_support;
  if (!cm || !buf || !BODY.ready || !RIG.ready) return;
  const dv = new DataView(buf.data.buffer, buf.data.byteOffset, buf.data.byteLength);
  const sv = new Uint32Array(sup.data.buffer, sup.data.byteOffset, sup.data.byteLength / 4);
  const st = buf.info.stride, step = cm.step, nstep = cm.nstep;
  CORR.shapes = cm.shapes.map((s: any) => {
    const n = s.count, idx = new Uint32Array(n), d = new Float32Array(n * 3), nn = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const o = (s.start + i) * st;
      idx[i] = dv.getUint32(o, true);
      for (let k = 0; k < 3; k++) { d[i * 3 + k] = dv.getInt16(o + 4 + 2 * k, true) * step; nn[i * 3 + k] = dv.getInt8(o + 10 + k) * nstep; }
    }
    return { name: s.name, idx, d, n: nn, support: sv.slice(s.support_start, s.support_start + s.support_count), radius: s.radius };
  });
  const byName = Object.fromEntries(CORR.shapes.map((s: any, i: number) => [s.name, i]));
  const bone = (n: string) => RIG.bones.findIndex((b: any) => b.name === n);
  for (const j of cm.spec.joints) {
    if (j.type === 'hinge') {
      CORR.joints.push({ name: j.name, type: 'hinge', b: j.bones.map(bone), keys: j.keys.map((k: any) => ({ a: k.angle * DEG, s: k.shape ? byName[k.shape] : -1, rest: !!k.rest })) });
    } else {
      CORR.joints.push({ name: j.name, type: 'cone', b: bone(j.bone), e: bone(j.end), f: bone(j.frame), shapes: j.targets.map((t: any) => t.shape ? byName[t.shape] : -1),
        dirs: j.targets.map((t: any) => t.dir.slice()), rest: j.targets.findIndex((t: any) => t.rest), triangles: j.triangles, tris: [] });
    }
  }
  const n = BODY.anny.nFine;
  Object.assign(CORR, { w: new Float32Array(CORR.shapes.length), wn: new Float32Array(CORR.shapes.length), wPrev: new Float32Array(CORR.shapes.length).fill(-1),
    scale: new Float32Array(CORR.shapes.length).fill(1), touched: new Uint32Array(n), mark: new Uint8Array(n), acc: new Float32Array(n * 3),
    base: BODY.anny.pos.slice(), nbase: BODY.anny.nrm.slice(), sbase: BODY.anny.ns.slice() });
  CORR.on = qs.get('corr') !== 'off';
  CORR.ready = true;
  updateCorrectiveScales();
}
// the rest keys follow the rest skeleton of the current body, and each shape scales with the body around it
function updateCorrectiveScales() {
  if (!CORR.ready) return;
  const H = RIG.heads, V = BODY.anny.coarse;
  const sub = (a: number, b: number) => [H[b][0] - H[a][0], H[b][1] - H[a][1], H[b][2] - H[a][2]];
  const unit = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  for (const J of CORR.joints) {
    if (J.type === 'hinge') {
      const u = unit(sub(J.b[0], J.b[1])), f = unit(sub(J.b[1], J.b[2]));
      const rest = Math.acos(Math.min(1, Math.max(-1, u[0] * f[0] + u[1] * f[1] + u[2] * f[2])));
      const K = J.keys, i = K.findIndex((k: any) => k.rest);
      K[i].a = Math.min(Math.max(rest, i > 0 ? K[i - 1].a + 1e-5 : rest), i + 1 < K.length ? K[i + 1].a - 1e-5 : rest);
    } else {
      J.dirs[J.rest] = unit(sub(J.b, J.e));
      const D = J.dirs;
      J.tris = J.triangles.map((t: number[]) => ({ t, inv: new THREE.Matrix3().set(
        D[t[0]][0], D[t[1]][0], D[t[2]][0], D[t[0]][1], D[t[1]][1], D[t[2]][1], D[t[0]][2], D[t[1]][2], D[t[2]][2]).invert() }));
    }
  }
  CORR.shapes.forEach((s: any, k: number) => {
    const n = s.support.length;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) { const o = s.support[i] * 3; cx += V[o]; cy += V[o + 1]; cz += V[o + 2]; }
    cx /= n; cy /= n; cz /= n;
    let q = 0;
    for (let i = 0; i < n; i++) { const o = s.support[i] * 3; q += (V[o] - cx) ** 2 + (V[o + 1] - cy) ** 2 + (V[o + 2] - cz) ** 2; }
    CORR.scale[k] = Math.sqrt(q / n) / s.radius;
  });
  CORR.wPrev.fill(-1);
}
const _ja = new THREE.Vector3(), _jb = new THREE.Vector3(), _jc = new THREE.Vector3(), _jd = new THREE.Vector3(), _je = new THREE.Vector3(), _jm = new THREE.Matrix3();
// the weight of every shape for the current pose (the drivers of anny.correctives)
function correctiveWeights(w: Float32Array) {
  w.fill(0);
  const bw = (i: number, out: any) => out.setFromMatrixPosition(RIG.bones[i].matrixWorld);
  for (const J of CORR.joints) {
    if (J.type === 'hinge') {
      bw(J.b[0], _ja); bw(J.b[1], _jb); bw(J.b[2], _jc);
      _jd.subVectors(_jb, _ja).normalize(); _je.subVectors(_jc, _jb).normalize();
      const K = J.keys;
      const ang = Math.min(Math.max(Math.acos(Math.min(1, Math.max(-1, _jd.dot(_je)))), K[0].a), K[K.length - 1].a);
      for (let k = 0; k < K.length - 1; k++) {
        if (ang > K[k + 1].a) continue;
        const t = (ang - K[k].a) / Math.max(K[k + 1].a - K[k].a, 1e-6);
        if (K[k].s >= 0) w[K[k].s] += 1 - t;
        if (K[k + 1].s >= 0) w[K[k + 1].s] += t;
        break;
      }
    } else {
      bw(J.b, _ja); bw(J.e, _jb);
      _jm.setFromMatrix4(RIG.bones[J.f].matrixWorld).transpose();     // the rig has no scale: the transpose inverts the turn
      _jd.subVectors(_jb, _ja).applyMatrix3(_jm).normalize();
      let best = -Infinity, bt = null, b0 = 0, b1 = 0, b2 = 0;
      for (const T of J.tris) {
        _je.copy(_jd).applyMatrix3(T.inv);
        const s = _je.x + _je.y + _je.z;
        if (s <= 0) continue;
        const x = _je.x / s, y = _je.y / s, z = _je.z / s, m = Math.min(x, y, z);
        if (m > best) { best = m; bt = T; b0 = x; b1 = y; b2 = z; }
      }
      if (!bt) continue;
      b0 = Math.max(b0, 0); b1 = Math.max(b1, 0); b2 = Math.max(b2, 0);
      const s = b0 + b1 + b2 || 1, S = J.shapes, t = bt.t;
      if (S[t[0]] >= 0) w[S[t[0]]] += b0 / s;
      if (S[t[1]] >= 0) w[S[t[1]]] += b1 / s;
      if (S[t[2]] >= 0) w[S[t[2]]] += b2 / s;
    }
  }
}
// base + the weighted shapes for the vertices that this call or the one before moved; returns the index range.
// w weighs the positions (with the size of the body) and wn the normals.
function sparseApply(st: any, shapes: any[], w: Float32Array, wn: Float32Array, P: Float32Array, N: Float32Array, S: Float32Array, base: Float32Array, nbase: Float32Array, sbase: Float32Array) {
  const T = st.touched, mark = st.mark, acc = st.acc;
  let lo = Infinity, hi = -1;
  for (let k = 0; k < st.nt; k++) {
    const v = T[k], o = v * 3;
    P[o] = base[o]; P[o + 1] = base[o + 1]; P[o + 2] = base[o + 2];
    N[o] = nbase[o]; N[o + 1] = nbase[o + 1]; N[o + 2] = nbase[o + 2];
    if (S) { S[o] = sbase[o]; S[o + 1] = sbase[o + 1]; S[o + 2] = sbase[o + 2]; }
    mark[v] = 0;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  let nt = 0;
  for (let s = 0; s < shapes.length; s++) {
    const ws = w[s], wns = wn[s];
    if (wns < 1e-4) continue;
    const { idx, d, n } = shapes[s];
    for (let i = 0, q = 0; i < idx.length; i++, q += 3) {
      const v = idx[i], o = v * 3;
      if (!mark[v]) {
        mark[v] = 1; T[nt++] = v; acc[o] = 0; acc[o + 1] = 0; acc[o + 2] = 0;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      P[o] += ws * d[q]; P[o + 1] += ws * d[q + 1]; P[o + 2] += ws * d[q + 2];
      acc[o] += wns * n[q]; acc[o + 1] += wns * n[q + 1]; acc[o + 2] += wns * n[q + 2];
    }
  }
  for (let k = 0; k < nt; k++) {
    const o = T[k] * 3;
    const ax = nbase[o], ay = nbase[o + 1], az = nbase[o + 2];
    let x = ax + acc[o], y = ay + acc[o + 1], z = az + acc[o + 2];
    const l = Math.sqrt(x * x + y * y + z * z) || 1;
    x /= l; y /= l; z /= l;
    N[o] = x; N[o + 1] = y; N[o + 2] = z;
    if (S) {
      // The smoothed normal (it softens the red light) turns with the normal, by the shortest turn from the old
      // normal to the new one. Where a shape turns the normal far, the fold of the posed skin differs from the rest
      // pose that the smoothing saw, so the smoothed normal moves over to the normal there.
      const kx = ay * z - az * y, ky = az * x - ax * z, kz = ax * y - ay * x;
      const c = ax * x + ay * y + az * z, s2 = kx * kx + ky * ky + kz * kz;
      let vx = sbase[o], vy = sbase[o + 1], vz = sbase[o + 2];
      if (s2 > 1e-12 && c > -0.9) {
        const f = (kx * vx + ky * vy + kz * vz) * (1 - c) / s2;
        const rx = vx * c + (ky * vz - kz * vy) + kx * f, ry = vy * c + (kz * vx - kx * vz) + ky * f, rz = vz * c + (kx * vy - ky * vx) + kz * f;
        vx = rx; vy = ry; vz = rz;
      }
      const d = Math.sqrt(acc[o] * acc[o] + acc[o + 1] * acc[o + 1] + acc[o + 2] * acc[o + 2]);
      const t = Math.min(Math.max((d - 0.1) / 0.4, 0), 1), u = t * t * (3 - 2 * t);
      vx += (x - vx) * u; vy += (y - vy) * u; vz += (z - vz) * u;
      const m = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
      S[o] = vx / m; S[o + 1] = vy / m; S[o + 2] = vz / m;
    }
  }
  st.nt = nt;
  return [lo, hi];
}
function applyCorrectives(force = false) {
  if (!CORR.ready) return;
  const w = CORR.wn;
  if (CORR.on) correctiveWeights(w); else w.fill(0);
  let same = !force;
  for (let i = 0; same && i < w.length; i++) if (Math.abs(w[i] - CORR.wPrev[i]) > 2e-4) same = false;
  if (same) return;
  CORR.wPrev.set(w);
  for (let i = 0; i < w.length; i++) CORR.w[i] = w[i] * CORR.scale[i];
  const at = BODY.geo.attributes;
  const [lo, hi] = sparseApply(CORR, CORR.shapes, CORR.w, CORR.wn, BODY.anny.pos, BODY.anny.nrm, BODY.anny.ns, CORR.base, CORR.nbase, CORR.sbase);
  markRange(at.position, lo, hi); markRange(at.normal, lo, hi); markRange(at.nsmooth, lo, hi);
}
// uploads of the body arrays: a slider change sends the whole array, the correctives send the vertex range they
// touch. Three.js sends only the ranges when there are any, so a range must not cut short a pending full upload.
const FULL_UPLOAD = new Set<any>();
function markFull(attr: any) {
  if (!FULL_UPLOAD.has(attr)) { FULL_UPLOAD.add(attr); attr.onUpload(() => FULL_UPLOAD.delete(attr)); }
  attr.clearUpdateRanges();
  attr.needsUpdate = true;
}
function markRange(attr: any, a: number, b: number) {
  if (b < a) return;
  if (!FULL_UPLOAD.has(attr)) attr.addUpdateRange(a * 3, (b - a + 1) * 3);
  attr.needsUpdate = true;
}
// take the correction off (before the sliders rebuild the rest shape)
function clearCorrectives() {
  if (!CORR.ready) return;
  CORR.w.fill(0); CORR.wn.fill(0);
  sparseApply(CORR, CORR.shapes, CORR.w, CORR.wn, BODY.anny.pos, BODY.anny.nrm, BODY.anny.ns, CORR.base, CORR.nbase, CORR.sbase);
  CORR.wPrev.fill(-1);
}
// after the sliders: the new rest shape is the base for the correction
function rebaseCorrectives() {
  if (!CORR.ready) return;
  CORR.base.set(BODY.anny.pos); CORR.nbase.set(BODY.anny.nrm); CORR.sbase.set(BODY.anny.ns);
  CORR.nt = 0;
  CORR.wPrev.fill(-1);
}
function setCorrectivesOn(on: boolean) {
  if (!CORR.ready) return;
  CORR.on = !!on;
  applyCorrectives(true);
  shadowsDirty = true; resetAccum();
  syncPoser();
}

// ------------------------------------------------------------------ motion: anny's pose library
// Poses and clips hold one rotation per bone relative to its parent (every bone rests along the world axes) and a
// root offset relative to the height of the hips (anny.poses). The page grounds each pose on anny's coarse body
// skinned by the pose, and a clip with one floor for all its frames, so a jump stays in the air.
const MOTION: any = { clips: [], byName: {}, Q: null, R: null, nb: 0, cur: null, t: 0, playing: true, speed: 1,
  q: null, rootOff: [0, 0, 0], from: null, fade: 1, fadeDur: 0.4, tmp: null, stool: null, lastT: 0, ground: 0, stoolInfo: null };
function initMotion(meta: any, B: any) {
  if (!meta.motion || !B.motion_q || !RIG.ready) return;
  MOTION.clips = meta.motion; MOTION.clips.forEach((c: any) => { MOTION.byName[c.name] = c; });
  const qb = B.motion_q, rb = B.motion_root;
  MOTION.Q = new Int16Array(qb.data.buffer, qb.data.byteOffset, qb.data.byteLength / 2);
  MOTION.R = new Float32Array(rb.data.buffer, rb.data.byteOffset, rb.data.byteLength / 4);
  MOTION.nb = qb.info.bones;
  MOTION.q = new Float32Array(MOTION.nb * 4); MOTION.tmp = new Float32Array(MOTION.nb * 4);
  MOTION.from = { q: new Float32Array(MOTION.nb * 4), root: [0, 0, 0] };
  for (let i = 0; i < MOTION.nb; i++) MOTION.q[i * 4 + 3] = 1;
  MOTION.cur = MOTION.byName.a_pose || MOTION.clips[0];
  MOTION.seat = Uint32Array.from((meta.stool && meta.stool.seat_vertices) || []);
  MOTION.stoolForward = (meta.stool && meta.stool.forward) || 0;
}
function slerpInto(out: Float32Array, o: number, a: Float32Array, ao: number, b: Float32Array, bo: number, t: number) {
  let ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3];
  let bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  let d = ax * bx + ay * by + az * bz + aw * bw;
  if (d < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d; }
  let k0 = 1 - t, k1 = t;
  if (d < 0.9995) { const th = Math.acos(d), s = Math.sin(th); k0 = Math.sin((1 - t) * th) / s; k1 = Math.sin(t * th) / s; }
  const x = ax * k0 + bx * k1, y = ay * k0 + by * k1, z = az * k0 + bz * k1, w = aw * k0 + bw * k1;
  const l = Math.hypot(x, y, z, w) || 1;
  out[o] = x / l; out[o + 1] = y / l; out[o + 2] = z / l; out[o + 3] = w / l;
}
const _qa = new Float32Array(4), _qb = new Float32Array(4);
// one frame of a clip: rotations into outQ, and the root offset of this body (with its ground) into outRoot
function sampleClip(c: any, t: number, outQ: Float32Array, outRoot: number[], ground = MOTION.ground) {
  const nb = MOTION.nb, Q = MOTION.Q, R = MOTION.R;
  let f0 = 0, f1 = 0, a = 0;
  if (c.count > 1) {
    const ft = ((t % c.duration) + c.duration) % c.duration / c.duration * c.count;
    f0 = Math.floor(ft) % c.count; f1 = (f0 + 1) % c.count; a = ft - Math.floor(ft);
  }
  const o0 = (c.start + f0) * nb * 4, o1 = (c.start + f1) * nb * 4;
  for (let i = 0; i < nb; i++) {
    for (let k = 0; k < 4; k++) { _qa[k] = Q[o0 + i * 4 + k] / 32767; _qb[k] = Q[o1 + i * 4 + k] / 32767; }
    slerpInto(outQ, i * 4, _qa, 0, _qb, 0, a);
  }
  const r0 = (c.start + f0) * 3, r1 = (c.start + f1) * 3, hip = hipHeight();
  for (let k = 0; k < 3; k++) outRoot[k] = (R[r0 + k] + (R[r1 + k] - R[r0 + k]) * a) * hip;
  outRoot[1] += ground;
}
// the vertical shift that puts this body's lowest point on the floor in a pose, or over all frames of a clip
function groundFor(c: any) {
  const q = new Float32Array(MOTION.nb * 4), root = [0, 0, 0];
  const n = c.kind === 'loop' ? Math.min(c.count, 24) : 1;
  let low = Infinity;
  for (let i = 0; i < n; i++) {
    sampleClip(c, c.kind === 'loop' ? i / n * c.duration : 0, q, root, 0);
    low = Math.min(low, BODY.anny.lowestSkinned(poseMatrices(q, root))[1]);
  }
  return BODY.anny.floor - low;
}
// the stool of a seated pose meets the lowest point of the seat; its size follows the hips
function stoolFor(c: any) {
  if (!c.stool || !MOTION.seat.length) return null;
  const q = new Float32Array(MOTION.nb * 4), root = [0, 0, 0];
  sampleClip(c, 0, q, root, MOTION.ground);
  const p = BODY.anny.lowestSkinned(poseMatrices(q, root), MOTION.seat), hip = hipHeight();
  return { top: p[1], center: [p[0], p[2] + MOTION.stoolForward * hip], radius: c.stool.radius * hip, thickness: c.stool.thickness * hip,
    leg_radius: c.stool.leg_radius * hip, legs: c.stool.legs };
}
function groundMotion() {
  if (!MOTION.cur || !BODY.ready) return;
  MOTION.ground = groundFor(MOTION.cur);
  if (MOTION.cur.stool) showStool(stoolFor(MOTION.cur));
}
function evalMotion() {
  const c = MOTION.cur; if (!c) return;
  const root = [0, 0, 0];
  sampleClip(c, MOTION.t, MOTION.tmp, root);
  if (MOTION.fade < 1) {
    const w = smooth(0, 1, MOTION.fade);
    for (let i = 0; i < MOTION.nb; i++) slerpInto(MOTION.q, i * 4, MOTION.from.q, i * 4, MOTION.tmp, i * 4, w);
    for (let k = 0; k < 3; k++) MOTION.rootOff[k] = MOTION.from.root[k] + (root[k] - MOTION.from.root[k]) * w;
  } else {
    MOTION.q.set(MOTION.tmp);
    for (let k = 0; k < 3; k++) MOTION.rootOff[k] = root[k];
  }
  const q = MOTION.q;
  RIG.bones.forEach((b: any, i: number) => b.quaternion.set(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]));
  placeRoot();
  poseChanged();
}
// start a pose or a clip, blending from what the figure shows now
function setMotion(name: string, opts: any = {}) {
  const c = MOTION.byName[name];
  if (!c) return false;
  MOTION.from.q.set(MOTION.q); MOTION.from.root = MOTION.rootOff.slice();
  MOTION.fade = opts.instant ? 1 : 0;
  MOTION.cur = c; MOTION.t = opts.t || 0;
  MOTION.ground = groundFor(c);
  if (!opts.instant) refitBody(c);
  if (c.stool) showStool(stoolFor(c));
  else if (opts.instant) showStool(null);
  evalMotion();
  if (opts.remember !== false) rememberMotion();
  syncPoser();
  return true;
}
// the Body view fits each pose or clip: a raised arm or a jump needs more room above, a seated figure less
function bodyFrameFor(c: any) {
  const b = c && c.bounds;
  if (!b || !BODY.ready) return FRAMES.body;
  const hip = hipHeight(), fl = BODY.anny.floor;
  const lo = fl + b[0] * hip, hi = fl + b[1] * hip, wx = b[2] * hip;
  return { target: [0, (lo + hi) / 2 - 0.11 * hip / MANIFEST.authoring.hip_height, 0.03], yaw: 24, pitch: 4, halfH: (hi - lo) / 2 + 0.21 * hip / MANIFEST.authoring.hip_height,
    halfW: Math.max(0.58 * hip / MANIFEST.authoring.hip_height, wx + 0.1) };
}
function refitBody(c: any, force = false) {
  if (SHOT || currentFrame !== 'body' || !c.bounds) return;
  const f = bodyFrameFor(c), dNow = camera.position.distanceTo(controls.target);
  const dOld = frameDistance(bodyFrameFor(MOTION.lastFramed)), dNew = frameDistance(f);
  // a view the viewer has zoomed well away from the fitted one stays as it is
  if (!force && Math.abs(dNow - dOld) / dOld > 0.3) return;
  MOTION.lastFramed = c;
  const t1 = new THREE.Vector3(...f.target);
  if (t1.distanceTo(controls.target) < 0.03 && Math.abs(dNew - dNow) < 0.08) return;
  const dir = camera.position.clone().sub(controls.target).normalize();
  tween = { t0: performance.now(), dur: 800, fromT: controls.target.clone(), fromP: camera.position.clone(), toT: t1, toP: t1.clone().addScaledVector(dir, dNew) };
}
// advance the clock; returns true while the figure moves
function tickMotion(dt: number) {
  if (!MOTION.cur) return false;
  let moving = false;
  if (MOTION.fade < 1) {
    MOTION.fade = Math.min(1, MOTION.fade + dt / MOTION.fadeDur); moving = true;
    if (MOTION.fade >= 1 && !MOTION.cur.stool) showStool(null);
  }
  if (MOTION.cur.kind === 'loop' && MOTION.playing) { MOTION.t += dt * MOTION.speed; moving = true; }
  if (moving) evalMotion();
  return moving;
}
// everything that rides on the head, the shadow cameras, the floor shade and the stool follow the pose
const _headSkin = new THREE.Matrix4(), _headFull = new THREE.Matrix4(), _m3 = new THREE.Matrix3(), _v = new THREE.Vector3();
function poseChanged() {
  if (!RIG.ready) return;
  RIG.holder.updateMatrixWorld(true);
  applyCorrectives();
  const hb = RIG.bones[RIG.head];
  _headSkin.multiplyMatrices(hb.matrixWorld, RIG.skel.boneInverses[RIG.head]);
  // the head of anny's default body, carried to the current body and then posed
  _headFull.multiplyMatrices(_headSkin, HEADMAP.m);
  U.uHeadInv.value.copy(_headFull).invert();
  U.uHeadRot.value.setFromMatrix4(_headSkin);
  _m3.setFromMatrix4(_headSkin);
  for (const e of RIG.eyes) {
    e.mesh.matrix.multiplyMatrices(_headSkin, e.shape); e.mesh.matrixWorldNeedsUpdate = true;
    e.u.uEyeRot.value.copy(_m3).multiply(e.mesh.userData.rot0);
  }
  for (const h of hairObjs) h.userData.hu.uHeadC.value.copy(h.userData.center).applyMatrix4(_headFull);
  // shadow cameras: the head map follows the head, the body map covers every joint
  SHADOW_FOCUS.head.set(0, 0.5, 0.03).applyMatrix4(_headFull);
  const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (const b of RIG.bones) {
    _v.setFromMatrixPosition(b.matrixWorld);
    mn[0] = Math.min(mn[0], _v.x); mn[1] = Math.min(mn[1], _v.y); mn[2] = Math.min(mn[2], _v.z);
    mx[0] = Math.max(mx[0], _v.x); mx[1] = Math.max(mx[1], _v.y); mx[2] = Math.max(mx[2], _v.z);
  }
  mn[1] = Math.min(mn[1], GROUND_Y);
  SHADOW_FOCUS.body.set((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
  SHADOW_FOCUS.bodyHalf = Math.max(0.95 * hipHeight() / MANIFEST.authoring.hip_height, 0.5 * Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) + 0.2);
  // contact shade on the floor under each foot: between the ankle and the toes, as large as the foot
  for (const [s, key] of [['L', 'uFootL'], ['R', 'uFootR']]) {
    const a = RIG.bones[RIG.bones.findIndex((b: any) => b.name === 'foot.' + s)], t = RIG.bones[RIG.bones.findIndex((b: any) => b.name === 'toe3-1.' + s)];
    if (!a || !t) continue;
    const pa = new THREE.Vector3().setFromMatrixPosition(a.matrixWorld), pt = new THREE.Vector3().setFromMatrixPosition(t.matrixWorld);
    const len = Math.hypot(pt.x - pa.x, pt.z - pa.z), lift = Math.min(pa.y, pt.y) - GROUND_Y;
    const on = smooth(0.25, 0.08, lift);
    U[key].value.set(0.5 * (pa.x + pt.x), 0.5 * (pa.z + pt.z), 0.4 * len + 0.02, on);
  }
  updateShadowCamera();
  shadowsDirty = true; resetAccum();
}

// ------------------------------------------------------------------ the stool for the seated pose
let STOOL = null;
const PROP_VS = /* glsl */`
varying vec3 vWPos; varying vec3 vN; varying vec3 vObj;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWPos = wp.xyz; vObj = position; vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;
// oiled beech: warm and fairly smooth, with a faint grain along the legs and across the seat
const PROP_FS = /* glsl */`
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
function buildStool() {
  const g = new THREE.Group();
  const mat = new THREE.ShaderMaterial({ uniforms: Object.assign({}, U, { uWood: { value: new THREE.Vector3(0.36, 0.2, 0.1) } }), vertexShader: PROP_VS,
    fragmentShader: toMRT(PROP_FS), defines: Object.assign({}, ENV_DEFINES), glslVersion: THREE.GLSL3, blending: THREE.NoBlending });
  g.userData.mat = mat;
  scene.add(g);
  g.visible = false;
  return g;
}
function showStool(info) {
  if (!STOOL) return;
  const was = STOOL.visible;
  if (!info) { STOOL.visible = false; STOOL.children.forEach(m => { m.visible = false; }); if (was) { shadowsDirty = true; resetAccum(); } return; }
  for (const m of STOOL.children) { const k = opaque.indexOf(m); if (k >= 0) opaque.splice(k, 1); m.geometry.dispose(); }
  STOOL.clear();
  const mat = STOOL.userData.mat, r = info.radius, th = info.thickness, top = info.top, h = top - GROUND_Y;
  const cx = info.center[0], cz = info.center[1];
  const seat = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.97, th, 64, 1), mat);
  seat.position.set(cx, top - th / 2, cz);
  STOOL.add(seat);
  for (let k = 0; k < info.legs; k++) {
    const ang = Math.PI / 4 + k * Math.PI * 2 / info.legs;
    const rt = r * 0.62, rb = r * 0.86;
    const x0 = cx + Math.cos(ang) * rt, z0 = cz + Math.sin(ang) * rt, x1 = cx + Math.cos(ang) * rb, z1 = cz + Math.sin(ang) * rb;
    const len = Math.hypot(x1 - x0, h - th, z1 - z0);
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(info.leg_radius * 0.9, info.leg_radius * 1.1, len, 20), mat);
    leg.position.set((x0 + x1) / 2, GROUND_Y + (h - th) / 2, (z0 + z1) / 2);
    leg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(x0 - x1, h - th, z0 - z1).normalize());
    STOOL.add(leg);
  }
  const ring = new THREE.Mesh(new THREE.TorusGeometry(r * 0.76, info.leg_radius * 0.55, 10, 64), mat);
  ring.rotation.x = Math.PI / 2; ring.position.set(cx, GROUND_Y + h * 0.36, cz);
  STOOL.add(ring);
  STOOL.children.forEach(m => { opaque.includes(m) || opaque.push(m); });
  STOOL.visible = true;
  shadowsDirty = true; resetAccum();
}

// ------------------------------------------------------------------ look: the parameters a preset carries
// A preset is plain JSON: sRGB hex colours for the skin, the hair and the eyes, and the values of anny's phenotype
// sliders, so the same file can set up anny in Python (phenotype_kwargs) and this viewer.
const LOOK_FORMAT = 'anny-viewer/look@1';
const srgb2lin = (c: number) => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const hexToRgb = (h: string) => { const n = parseInt(h.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255]; };
const hexToLin = (h: string) => hexToRgb(h).map(srgb2lin);
const isHex = (h: any) => typeof h === 'string' && /^#[0-9a-fA-F]{6}$/.test(h);
// skin: base colour along the tone scale (sRGB, neutral undertone); the baked skin colour was painted at tone 0.35
const TONE_STOPS = [[0.00, 0.93, 0.80, 0.72], [0.18, 0.87, 0.71, 0.60], [0.35, 0.78, 0.595, 0.462], [0.55, 0.69, 0.50, 0.36], [0.75, 0.53, 0.37, 0.25], [1.00, 0.33, 0.22, 0.15]];
function skinBase(tone: number, under: number) {
  let i = 0; while (i < TONE_STOPS.length - 2 && tone > TONE_STOPS[i + 1][0]) i++;
  const a = TONE_STOPS[i], b = TONE_STOPS[i + 1];
  const t = Math.min(1, Math.max(0, (tone - a[0]) / (b[0] - a[0])));
  const c = [1, 2, 3].map(k => a[k] + (b[k] - a[k]) * t);
  // a warm undertone leans golden (more green, less blue); a cool one leans pink
  return [c[0] * (1 - 0.012 * under), Math.min(1, c[1] * (1 + 0.045 * under)), c[2] * (1 - 0.13 * under)];
}
const SKIN_REF = skinBase(0.35, 0).map(srgb2lin);
const SCALP_REF = [0.24, 0.18, 0.15].map(srgb2lin);   // scalp colour baked under the baseline hair
const PALETTES: any = {
  hair: [['Black', '#141110'], ['Dark brown', '#271f16'], ['Brown', '#3d2a1d'], ['Ash brown', '#4a4038'], ['Light brown', '#6b4a30'],
         ['Dark blond', '#86704f'], ['Blond', '#b09a72'], ['Auburn', '#5a2716'], ['Copper', '#8a3d1c'], ['Grey', '#8c8a86'], ['White', '#d8d4cc']],
  eyes: [['Dark brown', '#4a3020'], ['Brown', '#875f3d'], ['Amber', '#b07a2c'], ['Hazel', '#8a7a45'], ['Green', '#5e7a4c'],
         ['Grey', '#7d8790'], ['Blue', '#5f84a8'], ['Pale blue', '#9db8cc']],
};
const PHENOTYPE_DEFAULT = 0.5;
function baselinePhenotype() { return Object.fromEntries((MANIFEST?.shape?.sliders || []).map((k: string) => [k, PHENOTYPE_DEFAULT])); }
const BASELINE_LOOK: any = { format: LOOK_FORMAT, name: 'Baseline', skin: { tone: 0.35, undertone: 0 }, hair: { color: '#271f16' }, eyes: { color: '#875f3d' }, phenotype: {} };
const EXAMPLE_LOOKS: any[] = [
  BASELINE_LOOK,
  { format: LOOK_FORMAT, name: 'Fair', skin: { tone: 0.12, undertone: -0.3 }, hair: { color: '#6b4a30' }, eyes: { color: '#5f84a8' } },
  { format: LOOK_FORMAT, name: 'Olive', skin: { tone: 0.5, undertone: 0.45 }, hair: { color: '#141110' }, eyes: { color: '#4a3020' } },
  { format: LOOK_FORMAT, name: 'Deep', skin: { tone: 0.9, undertone: 0.1 }, hair: { color: '#141110' }, eyes: { color: '#4a3020' } },
];
const LOOK_PARTS: any = { hair: null, brows: null, lashes: null, hairMeshes: [], eyes: [], iris: null };
let currentLook: any = JSON.parse(JSON.stringify(BASELINE_LOOK));
let hairWanted = true;
function updateHairVisibility() {
  LOOK_PARTS.hairMeshes.forEach((m: any, i: number) => { m.visible = i > 0 || hairWanted; });
  shadowsDirty = true; resetAccum();
}
// read a preset: unknown fields are ignored, missing ones fall back to the baseline and to anny's defaults
function normaliseLook(o: any) {
  if (!o || typeof o !== 'object') throw new Error('The preset is not a JSON object.');
  const b = BASELINE_LOOK, num = (v: any, lo: number, hi: number, d: number) => (typeof v === 'number' && isFinite(v)) ? Math.min(hi, Math.max(lo, v)) : d;
  const col = (v: any, d: string) => isHex(v) ? v.toLowerCase() : d;
  const ph = o.phenotype && typeof o.phenotype === 'object' ? o.phenotype : {};
  return {
    format: LOOK_FORMAT, name: typeof o.name === 'string' && o.name.trim() ? o.name.trim().slice(0, 60) : 'Custom',
    skin: { tone: num(o.skin && o.skin.tone, 0, 1, b.skin.tone), undertone: num(o.skin && o.skin.undertone, -1, 1, b.skin.undertone) },
    hair: { color: col(o.hair && o.hair.color, b.hair.color) },
    eyes: { color: col(o.eyes && o.eyes.color, b.eyes.color) },
    phenotype: Object.fromEntries(Object.keys(baselinePhenotype()).map((k) => [k, Math.round(num(ph[k], 0, 1, PHENOTYPE_DEFAULT) * 1000) / 1000])),
  };
}
function applyLook(look: any, remember = true) {
  const L = normaliseLook(look);
  const irisChanged = !LOOK_PARTS.iris || L.eyes.color !== currentLook.eyes.color;
  currentLook = L;
  // skin
  const base = skinBase(L.skin.tone, L.skin.undertone);
  const lin = base.map(srgb2lin);
  U.uSkinTint.value.set(lin[0] / SKIN_REF[0], lin[1] / SKIN_REF[1], lin[2] / SKIN_REF[2]);
  U.uMelanin.value = smooth(0.35, 1.0, L.skin.tone);
  // hair, brows and lashes (the brows a little darker than the hair, the lashes much darker)
  const h = hexToLin(L.hair.color);
  if (LOOK_PARTS.hair) LOOK_PARTS.hair.uHairColor.value.set(...h);
  if (LOOK_PARTS.brows) LOOK_PARTS.brows.uHairColor.value.set(h[0] * 0.8, h[1] * 0.72, h[2] * 0.73);
  if (LOOK_PARTS.lashes) LOOK_PARTS.lashes.uHairColor.value.set(h[0] * 0.3 + 0.001, h[1] * 0.32 + 0.001, h[2] * 0.37 + 0.001);
  const lum = 0.2126 * h[0] + 0.7152 * h[1] + 0.0722 * h[2];
  U.uHairShadowK.value = 1.0 - 0.45 * smooth(0.02, 0.25, lum);   // light hair lets more light through
  // scalp under the hair: between the skin and the hair colour
  const hs = hexToRgb(L.hair.color);
  U.uScalpCol.value.set(...[0, 1, 2].map(k => srgb2lin(base[k] * 0.18 + hs[k] * 0.82)));
  U.uScalp0.value.set(...SCALP_REF);
  // eyes: the swatch shows the iris as it reads under light; the texture holds the darker pigment colour
  if (irisChanged && LOOK_PARTS.eyes.length) {
    const tex = makeIrisTexture(hexToLin(L.eyes.color).map(v => v * 0.16));
    const old = LOOK_PARTS.iris;
    LOOK_PARTS.eyes.forEach((eu: any) => { eu.uIris.value = tex; });
    LOOK_PARTS.iris = tex;
    if (old && old.userData.rt) old.userData.rt.dispose();
  }
  U.uWearOn.value.set(0, 0, 0, 0);
  U.uMannequin.value = 0;
  applyBodyShape(L.phenotype);
  updateHairVisibility();
  if (remember) { try { localStorage.setItem('anny-look', JSON.stringify(L)); } catch (e) { /* storage unavailable */ } }
  shadowsDirty = true;
  resetAccum();
  syncEditor();
}
function loadRememberedMotion() {
  try { const t = localStorage.getItem('anny-motion'); return t ? JSON.parse(t) : null; } catch (e) { return null; }
}
function loadRememberedLook() {
  try { const t = localStorage.getItem('anny-look'); return t ? normaliseLook(JSON.parse(t)) : null; } catch (e) { return null; }
}

// ------------------------------------------------------------------ hair that follows the body
// Each ribbon point keeps its coordinates in the frame of the triangle under its strand's root. A float texture
// holds the frame of every strand on the current body (AnnyBody.followStrands writes into it), and the hair
// vertex shader places the points, so a slider step moves ~58k frames instead of ~1.7M ribbon vertices.
const HAIRSETS: any[] = [];
const STRAND_TEX_W = 2048;
function strandTexture(nS: number) {
  const h = Math.ceil(nS * 3 / STRAND_TEX_W);
  const tex = new THREE.DataTexture(new Float32Array(STRAND_TEX_W * h * 4), STRAND_TEX_W, h, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter; tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace; tex.needsUpdate = true;
  return tex;
}
function updateHairShape() { for (const h of HAIRSETS) h.tex.needsUpdate = true; }
// bind a strand set to the skin; returns the ribbon source (local points) and the texture of its frames
function strandBinding(B: any, name: string, S: any) {
  const bb = B[name + '_bind'];
  const nS = S.nS, dv = new DataView(bb.data.buffer, bb.data.byteOffset, bb.data.byteLength);
  const corners = new Uint32Array(nS * 3), bary = new Float32Array(nS * 3);
  for (let i = 0; i < nS; i++) {
    for (let k = 0; k < 3; k++) corners[i * 3 + k] = dv.getUint32(i * 20 + k * 4, true);
    const u = dv.getFloat32(i * 20 + 12, true), w = dv.getFloat32(i * 20 + 16, true);
    bary[i * 3] = u; bary[i * 3 + 1] = w; bary[i * 3 + 2] = 1 - u - w;
  }
  const tex = strandTexture(nS);
  BODY.anny.bindStrands(name, S.P, S.counts, corners, bary, (tex.image.data as Float32Array).subarray(0, nS * 12));
  HAIRSETS.push({ name, tex });
  return { local: { P: BODY.anny.strands[name].local, counts: S.counts, nS, total: S.total, skin: S.skin }, tex };
}

let MeshoptDecoder: any = null;
let MANIFEST: any = null;
async function init() {
  setProgress(0.04, 'Decoding model');
  MeshoptDecoder = (window as any).MeshoptDecoderRef;
  const b64 = (window as any).MODEL_B64 || await (await fetch('build/model.b64')).text();
  await nextFrame();
  const raw = await gunzip(b64ToBytes(b64));
  setProgress(0.18, 'Unpacking geometry');
  await nextFrame();
  const { meta, B } = parseContainer(raw);
  MANIFEST = meta;
  GROUND_Y = meta.shape.frame.floor - 0.0012;
  floorMesh.position.y = GROUND_Y;
  setProgress(0.32, 'Building skin');
  await nextFrame();
  // textures
  const lutInfo = meta.lut;
  const lut = new THREE.DataTexture(B.lut.data.slice(0, lutInfo.w * lutInfo.h * 4), lutInfo.w, lutInfo.h, THREE.RGBAFormat, THREE.UnsignedByteType);
  lut.colorSpace = THREE.NoColorSpace; lut.minFilter = THREE.LinearFilter; lut.magFilter = THREE.LinearFilter; lut.generateMipmaps = false;
  lut.wrapS = lut.wrapT = THREE.ClampToEdgeWrapping; lut.needsUpdate = true;
  U.uLUT.value = lut;
  U.uDetail.value = makeDetailTexture(isSmall ? 512 : 1024);
  U.uPores.value = makePoreTexture(isSmall ? 512 : 1024);
  // skin metadata: moles and knuckles in the rest shape of anny's default body
  const skinMeta = meta.skin || {};
  (skinMeta.moles || []).slice(0, 16).forEach((m: number[], i: number) => { U.uMoleA.value[i].set(m[0], m[1], m[2], m[3]); U.uMoleB.value[i].set(m[4], 0, 0, 0); });
  (skinMeta.knuckles || []).slice(0, 28).forEach((k: number[], i: number) => { U.uKnA.value[i].set(k[0], k[1], k[2], k[3]); U.uKnB.value[i].set(k[4], k[5], k[6], k[7]); });
  U.uEyeL.value.set(...meta.eyes.eyes.l.center, meta.eyes.R); U.uEyeR.value.set(...meta.eyes.eyes.r.center, meta.eyes.R);
  const irisTex = makeIrisTexture();
  LOOK_PARTS.iris = irisTex;
  applyPreset(currentPreset);
  ENV_DEFINES = envDefines(U.uEnv.value);
  floorMat.defines = Object.assign({}, ENV_DEFINES); floorMat.needsUpdate = true;
  // the fine body of anny's default body, then anny's body for the sliders
  const hg = buildHead(B.head_v);
  hg.setIndex(new THREE.BufferAttribute(B.head_i.data, 1));
  addSkin(hg, B.head_skin);
  hg.setAttribute('normal', new THREE.BufferAttribute(normalsOf(hg.attributes.position.array, B.head_i.data), 3));
  hg.computeBoundingSphere();
  initBody(meta, B, hg);
  buildRig(meta.rig, BODY.anny.jointsDefault);
  updateHeadMap();
  const skinMat = new THREE.ShaderMaterial({ uniforms: U, vertexShader: SKIN_VS, fragmentShader: SKIN_FS, defines: Object.assign({}, ENV_DEFINES, SSS_ON ? { SSS_SCREEN: '' } : {}, hg.attributes.skinIndex2 ? SKIN8_DEF : {}), glslVersion: THREE.GLSL3, blending: THREE.NoBlending });
  const head = skinned(hg, skinMat);
  head.userData.depthMat = new THREE.ShaderMaterial({
    uniforms: { uWearOn: U.uWearOn }, defines: Object.assign({}, depthMat.defines, hg.attributes.skinIndex2 ? SKIN8_DEF : {}), side: THREE.DoubleSide, fragmentShader: depthMat.fragmentShader,
    vertexShader: SKIN_GLSL + '\nattribute vec4 attrV; uniform vec4 uWearOn;\n' + COVER_GLSL + '\nvoid main(){ gl_Position = projectionMatrix * modelViewMatrix * (skinMat() * vec4(coveredPosition(), 1.0)); }',
  });
  scene.add(head); opaque.push(head);
  buildBodySliders();
  initMotion(meta, B);
  initCorrectives(meta, B);
  STOOL = buildStool();
  // eyes: built on anny's default body; a matrix carries each one to the current body (body.ts)
  for (const s of ['l', 'r']) {
    const e = meta.eyes.eyes[s];
    const eu = Object.assign({}, U, {
      uEyeRot: { value: new THREE.Matrix3().set(...e.rot.flat()) },
      uZcc: { value: meta.eyes.zcc }, uRL: { value: meta.eyes.RL }, uIrisZ: { value: meta.eyes.zL - 0.0004 }, uPupil: { value: 0.0017 },
      uIris: { value: irisTex },
    });
    const em = new THREE.ShaderMaterial({ uniforms: eu, vertexShader: EYE_VS, fragmentShader: toMRT(EYE_FS), defines: ENV_DEFINES, glslVersion: THREE.GLSL3, blending: THREE.NoBlending });
    const eye = new THREE.Mesh(buildEye(e, meta, B['eye_ao_' + s].data), em);
    eye.matrixAutoUpdate = false; eye.frustumCulled = false;
    eye.userData.rot0 = eu.uEyeRot.value.clone();
    scene.add(eye); opaque.push(eye);
    LOOK_PARTS.eyes.push(eu); RIG.eyes.push({ mesh: eye, u: eu, side: s, center0: e.center, shape: new THREE.Matrix4() });
  }
  setProgress(0.5, 'Growing hair');
  await nextFrame();
  const center = meta.authoring.hair_center;
  const hairS = decodeStrands(B, 'hair');
  const hairAO = hairOcclusion(hairS.P, hairS.total, center);
  const hairB = strandBinding(B, 'hair', hairS);
  setProgress(0.75, 'Placing strands');
  await nextFrame();
  const lite = isSmall ? subsetStrands(hairB.local, hairAO, 2) : { S: hairB.local, ao: hairAO, keep: null };
  const hairMesh = makeHairMesh(buildRibbons(lite.S, lite.ao, 17, lite.keep), { width: isSmall ? 0.00017 : 0.00012, tip: 0.45, color: [0.020, 0.0125, 0.0082], rough: 0.38, diff: 1.3, spec: 0.38, center }, ENV_DEFINES, hairB.tex);
  const browB = strandBinding(B, 'brows', decodeStrands(B, 'brows'));
  const browMesh = makeHairMesh(buildRibbons(browB.local, null, 5), { width: 0.00010, tip: 0.3, color: [0.016, 0.009, 0.006], rough: 0.62, spec: 0.15, diff: 1.3, center: [0, 0.515, 0.06] }, ENV_DEFINES, browB.tex);
  const lashB = strandBinding(B, 'lashes', decodeStrands(B, 'lashes'));
  const lashMesh = makeHairMesh(buildRibbons(lashB.local, null, 9), { width: 0.00012, tip: 0.25, color: [0.006, 0.004, 0.003], rough: 0.55, spec: 0.3, diff: 1.3, center: [0, 0.51, 0.10] }, ENV_DEFINES, lashB.tex);
  (window as any).__hair = hairMesh;
  LOOK_PARTS.hair = hairMesh.userData.hu; LOOK_PARTS.brows = browMesh.userData.hu; LOOK_PARTS.lashes = lashMesh.userData.hu;
  LOOK_PARTS.hairMeshes = [hairMesh, browMesh, lashMesh];
  const q = new URLSearchParams(location.search);
  const start = loadRememberedLook() || normaliseLook(BASELINE_LOOK);
  for (const s of BODY.sliders) if (q.has(s.name)) start.phenotype[s.name] = Math.min(1, Math.max(0, parseFloat(q.get(s.name))));
  applyLook(start, false);
  if (MOTION.cur) {
    const rm = SHOT ? null : loadRememberedMotion();
    if (rm) { MOTION.playing = rm.playing !== false; MOTION.speed = rm.speed || 1; }
    setMotion(rm && MOTION.byName[rm.name] ? rm.name : 'a_pose', { instant: true, remember: false });
    if (!SHOT) { const f = bodyFrameFor(MOTION.cur); placeCamera(f.yaw, f.pitch, frameDistance(f), ...f.target); MOTION.lastFramed = MOTION.cur; }
    buildPoser();
  }
  setProgress(0.95, 'Compiling shaders');
  await nextFrame();
  renderer.compile(scene, camera);
  setProgress(1, '');
  document.body.classList.add('ready');
  (window as any).__ready = true;
  if (!SHOT) animate();
}
function nextFrame() { return new Promise<void>(r => requestAnimationFrame(() => r())); }

function renderShadows() {
  bgMesh.visible = false; floorMesh.visible = false;
  const saved = opaque.map(o => o.material);
  const ov = opaque.map(o => o.visible);   // wearables that are off stay off
  opaque.forEach(o => o.material = o.userData.depthMat || depthMat);
  const hv = hairObjs.map(o => o.visible);
  hairObjs.forEach(o => o.visible = false);
  renderer.setClearColor(0xffffff, 1);
  renderer.setRenderTarget(shadowRT); renderer.clear(); renderer.render(scene, shadowCam);
  renderer.setRenderTarget(bodyShadowRT); renderer.clear(); renderer.render(scene, bodyShadowCam);
  opaque.forEach((o, i) => { o.material = saved[i]; o.visible = false; });
  hairObjs.forEach((o, i) => { o.visible = hv[i]; o.userData.mainMat = o.material; o.material = o.userData.depthMat; });
  renderer.setRenderTarget(hairShadowRT); renderer.clear(); renderer.render(scene, shadowCam);
  hairObjs.forEach(o => { o.material = o.userData.mainMat; });
  opaque.forEach((o, i) => { o.visible = ov[i]; });
  renderer.setRenderTarget(null);
  renderer.setClearColor(0x000000, 1);
  bgMesh.visible = true; floorMesh.visible = true;
  shadowsDirty = false;
}

// ------------------------------------------------------------------ progressive accumulation
let accCount = 0;
const MAX_ACC = SHOT ? parseInt(qs.get('acc') || '16') : (isSmall ? 20 : 40);
const rtOpts = { type: THREE.HalfFloatType, depthBuffer: true, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
let sceneRT, sssRT, accRT = [];
function allocRTs() {
  const w = Math.max(1, Math.floor(innerWidth * PR)), h = Math.max(1, Math.floor(innerHeight * PR));
  if (sceneRT) { sceneRT.dispose(); accRT.forEach(r => r.dispose()); if (sssRT) sssRT.dispose(); }
  sceneRT = new THREE.WebGLRenderTarget(w, h, Object.assign({ samples: 4, count: SSS_ON ? 2 : 1 }, rtOpts));
  if (SSS_ON) {
    sceneRT.depthTexture = new THREE.DepthTexture(w, h);
    sceneRT.textures[1].minFilter = sceneRT.textures[1].magFilter = THREE.LinearFilter;
    sssRT = new THREE.WebGLRenderTarget(w, h, Object.assign({}, rtOpts, { depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter }));
  }
  accRT = [0, 1].map(() => new THREE.WebGLRenderTarget(w, h, Object.assign({}, rtOpts, { depthBuffer: false })));
  BG.uRes.value.set(w, h);
  for (const o of hairObjs) o.userData.hu.uViewportH.value = h;
}
const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const FSQ_VS = 'varying vec2 vUv; void main(){ vUv = position.xy*0.5+0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }';
// separable diffusion kernel from the d'Eon skin profile (Jimenez et al., "Separable Subsurface Scattering", 2015)
function sssKernel(n, strength, falloff) {
  const RANGE = n > 20 ? 3.0 : 2.0, EXP = 2.0;
  const gauss = (v, r) => [0, 1, 2].map(i => { const rr = r / (0.001 + falloff[i]); return Math.exp(-(rr * rr) / (2 * v)) / (2 * Math.PI * v); });
  const terms = [[0.100, 0.0484], [0.118, 0.187], [0.113, 0.567], [0.358, 1.99], [0.078, 7.41]];
  const profile = (r) => { const out = [0, 0, 0]; for (const [w, v] of terms) gauss(v, r).forEach((g, i) => { out[i] += w * g; }); return out; };
  const k = [];
  const step = 2 * RANGE / (n - 1);
  for (let i = 0; i < n; i++) { const o = -RANGE + i * step; k.push({ o: RANGE * Math.sign(o) * Math.pow(Math.abs(o), EXP) / Math.pow(RANGE, EXP), w: [0, 0, 0] }); }
  for (let i = 0; i < n; i++) {
    const w0 = i > 0 ? Math.abs(k[i].o - k[i - 1].o) : 0, w1 = i < n - 1 ? Math.abs(k[i].o - k[i + 1].o) : 0;
    const area = (w0 + w1) / 2;
    k[i].w = profile(k[i].o).map(v => v * area);
  }
  const c = k.splice((n - 1) / 2, 1)[0]; k.unshift(c);
  const sum = [0, 0, 0]; k.forEach(s => s.w.forEach((v, j) => { sum[j] += v; }));
  k.forEach(s => { s.w = s.w.map((v, j) => v / sum[j]); });
  k[0].w = k[0].w.map((v, j) => (1 - strength[j]) + strength[j] * v);
  for (let i = 1; i < n; i++) k[i].w = k[i].w.map((v, j) => v * strength[j]);
  return k.map(s => new THREE.Vector4(s.w[0], s.w[1], s.w[2], s.o));
}
const SSS_N = isSmall ? 17 : 25;
const qv = (key, def) => (qs.get(key) || def).split(',').map(Number);
const sssU = {
  tSSS: { value: null }, tDepth: { value: null },
  uKernel: { value: sssKernel(SSS_N, qv('ssss', '0.62,0.46,0.30'), qv('sssf', '1.0,0.37,0.3')) },
  uTexel: { value: new THREE.Vector2(1, 1) }, uProj: { value: 1 },
  // world metres per millimetre of the profile (the model is built at 1.45 m and shown for a 1.63 m boy)
  uWidth: { value: 0.001 / 1.124 * parseFloat(qs.get('sssw') || '1.15') },
  uNear: { value: camera.near }, uFar: { value: camera.far }, uFrameS: { value: 0 },
  uSSSRaw: { value: qs.get('dbg') === 'raw' ? 1 : 0 },
};
const SSS_GLSL = /* glsl */`
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
const sssMat = new THREE.ShaderMaterial({
  uniforms: sssU, vertexShader: FSQ_VS,
  fragmentShader: SSS_GLSL + '\nuniform sampler2D tSSS; varying vec2 vUv; void main(){ gl_FragColor = sssBlur(tSSS, vUv, vec2(1.0, 0.0)); }',
  depthTest: false, depthWrite: false, toneMapped: false, blending: THREE.NoBlending,
});
allocRTs();
const accMat = new THREE.ShaderMaterial({
  uniforms: Object.assign({ tNew: { value: null }, tAcc: { value: null }, uW: { value: 1 }, tSSSH: { value: null } }, sssU),
  defines: SSS_ON ? { SSS: '' } : {},
  vertexShader: FSQ_VS,
  fragmentShader: SSS_GLSL + `
uniform sampler2D tNew; uniform sampler2D tAcc; uniform float uW; uniform sampler2D tSSSH; varying vec2 vUv;
void main(){
  vec4 c = texture2D(tNew, vUv);
#ifdef SSS
  c.rgb += sssBlur(tSSSH, vUv, vec2(0.0, 1.0)).rgb;
#endif
  gl_FragColor = mix(texture2D(tAcc, vUv), c, uW);
}`,
  depthTest: false, depthWrite: false, toneMapped: false, blending: THREE.NoBlending,
});
const dispMat = new THREE.ShaderMaterial({
  uniforms: { tAcc: { value: null }, uSeed: { value: 0 }, uSat: { value: parseFloat(qs.get('sat') || '1.15') }, uContrast: { value: parseFloat(qs.get('con') || '1.05') }, uVignette: { value: 0.22 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = position.xy*0.5+0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }',
  fragmentShader: `uniform sampler2D tAcc; uniform float uSeed; uniform float uSat; uniform float uContrast; uniform float uVignette; varying vec2 vUv;
    float h12(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + uSeed) * 43758.5453); }
    void main(){
      vec3 rgb = texture2D(tAcc, vUv).rgb;
      vec2 q = vUv - 0.5; rgb *= 1.0 - uVignette * dot(q, q) * 1.6;
      rgb = toneMapping(rgb);
      float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
      rgb = max(mix(vec3(lum), rgb, uSat), 0.0);
      rgb = sRGBTransferOETF(vec4(rgb, 1.0)).rgb;
      rgb = clamp((rgb - 0.5) * uContrast + 0.5, 0.0, 1.0);
      rgb += (h12(gl_FragCoord.xy) - 0.5) / 255.0;
      gl_FragColor = vec4(rgb, 1.0);
    }`,
  depthTest: false, depthWrite: false, toneMapped: true, blending: THREE.NoBlending,
});
const quadMesh = new THREE.Mesh(triGeo, accMat); quadMesh.frustumCulled = false;
const quadScene = new THREE.Scene(); quadScene.add(quadMesh);
function halton(i, b) { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; }
function renderPass() {
  if (shadowsDirty) renderShadows();
  const w = sceneRT.width, h = sceneRT.height;
  const jx = accCount === 0 ? 0 : halton(accCount, 2) - 0.5;
  const jy = accCount === 0 ? 0 : halton(accCount, 3) - 0.5;
  camera.setViewOffset(w, h, jx, jy, w, h);
  U.uFrame.value = accCount;
  renderer.setClearColor(0x000000, 0);
  renderer.setRenderTarget(sceneRT); renderer.clear(); renderer.render(scene, camera);
  camera.clearViewOffset();
  if (SSS_ON) {
    sssU.tSSS.value = sceneRT.textures[1]; sssU.tDepth.value = sceneRT.depthTexture;
    sssU.uProj.value = h / (2 * Math.tan(camera.fov * DEG / 2));
    sssU.uTexel.value.set(1 / w, 1 / h);
    sssU.uFrameS.value = accCount;
    quadMesh.material = sssMat;
    renderer.setRenderTarget(sssRT); renderer.render(quadScene, quadCam);
    accMat.uniforms.tSSSH.value = sssRT.texture;
  }
  const src = accRT[0], dst = accRT[1];
  accMat.uniforms.tNew.value = sceneRT.textures[0]; accMat.uniforms.tAcc.value = src.texture; accMat.uniforms.uW.value = 1 / (accCount + 1);
  quadMesh.material = accMat;
  renderer.setRenderTarget(dst); renderer.render(quadScene, quadCam);
  accRT = [dst, src];
  dispMat.uniforms.tAcc.value = dst.texture; dispMat.uniforms.uSeed.value = (accCount * 0.618) % 1;
  quadMesh.material = dispMat;
  renderer.setRenderTarget(null); renderer.render(quadScene, quadCam);
  accCount++;
  updateStatus();
}
function resetAccum() { accCount = 0; }
controls.addEventListener('change', resetAccum);
controls.addEventListener('start', () => { document.body.classList.add('interacted'); });
let lastFrameT = performance.now();
const _hc = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now(), dt = Math.min(0.1, (now - lastFrameT) / 1000); lastFrameT = now;
  if (tickMotion(dt) && currentFrame === 'face' && !tween) {
    // the close view of the face keeps the head in frame while the figure moves
    _hc.set(0, 0.495, 0.035).applyMatrix4(_headFull);
    const d = _hc.sub(controls.target).multiplyScalar(Math.min(1, dt * 6));
    controls.target.add(d); camera.position.add(d);
  }
  if (tween) {
    const k = Math.min(1, (performance.now() - tween.t0) / tween.dur);
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    controls.target.lerpVectors(tween.fromT, tween.toT, e);
    camera.position.lerpVectors(tween.fromP, tween.toP, e);
    resetAccum();
    if (k >= 1) tween = null;
  }
  const moved = controls.update();
  if (controls.autoRotate) resetAccum();
  if (accCount < MAX_ACC) renderPass();
}
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  if (!tween && !document.body.classList.contains('interacted')) {
    if (currentFrame === 'body' && MOTION.cur) { const f = bodyFrameFor(MOTION.cur); placeCamera(f.yaw, f.pitch, frameDistance(f), ...f.target); }
    else frameCamera(currentFrame);
  }
  renderer.setSize(innerWidth, innerHeight, false); allocRTs(); resetAccum();
});

// ------------------------------------------------------------------ UI wiring
function setProgress(f, label) {
  const bar = $('loadbar'); if (bar) bar.style.transform = `scaleX(${f})`;
  const l = $('loadlabel'); if (l && label) l.textContent = label;
}
function showError(msg) {
  const l = $('loadlabel'); if (l) l.textContent = msg;
  document.body.classList.add('failed');
}
let lastStatus = '';
function updateStatus() {
  const el = $('status'); if (!el) return;
  const moving = MOTION.cur && ((MOTION.cur.kind === 'loop' && MOTION.playing) || MOTION.fade < 1);
  const txt = moving ? 'Playing' : controls.autoRotate ? 'Turntable' : accCount >= MAX_ACC ? 'Refined' : `Refining ${Math.round(accCount / MAX_ACC * 100)}%`;
  if (txt !== lastStatus) { el.textContent = txt; lastStatus = txt; el.dataset.done = accCount >= MAX_ACC ? '1' : '0'; }
}
function wireUI() {
  // panels and the hint sit above the control bar, which can wrap to more rows on a phone
  const bar = document.querySelector('.bar');
  if (bar && typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => document.documentElement.style.setProperty('--bar-h', bar.offsetHeight + 'px')).observe(bar);
  }
  document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('[data-preset]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    applyPreset(b.dataset.preset);
  }));
  const hairBtn = $('toggle-hair');
  hairBtn?.addEventListener('click', () => {
    const on = hairBtn.getAttribute('aria-pressed') !== 'true';
    hairBtn.setAttribute('aria-pressed', String(on));
    hairWanted = on; updateHairVisibility();
    shadowsDirty = true; resetAccum();
  });
  const turn = $('toggle-turn');
  turn?.addEventListener('click', () => {
    const on = turn.getAttribute('aria-pressed') !== 'true';
    turn.setAttribute('aria-pressed', String(on));
    controls.autoRotate = on; resetAccum();
  });
  document.querySelectorAll('[data-frame]').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('[data-frame]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    flyTo(b.dataset.frame);
  }));
  canvas.addEventListener('dblclick', () => flyTo(currentFrame));
}
wireUI();

// ------------------------------------------------------------------ character panel
const lookName = (list, hex) => { const m = PALETTES[list].find(p => p[1] === hex); return m ? m[0] : 'Custom ' + hex; };
function setEdMsg(text, kind = '') { const m = $('ed-msg'); if (m) { m.textContent = text; m.dataset.kind = kind; } }
function buildSwatches(list, key, pick) {
  const box = $('ed-' + list);
  box.textContent = '';
  for (const [name, hex] of PALETTES[list]) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ed-sw'; b.style.setProperty('--sw', hex);
    b.setAttribute('aria-label', name); b.title = name; b.dataset.hex = hex; b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => pick(hex));
    box.appendChild(b);
  }
  // any other colour through the system colour picker
  const c = document.createElement('label');
  c.className = 'ed-sw ed-custom'; c.title = 'Custom colour';
  const inp = document.createElement('input');
  inp.type = 'color'; inp.id = 'ed-' + list + '-custom'; inp.setAttribute('aria-label', 'Custom ' + key + ' colour');
  inp.addEventListener('input', () => pick(inp.value.toLowerCase()));
  c.appendChild(inp);
  box.appendChild(c);
}
function editLook(mut) {
  const next = JSON.parse(JSON.stringify(currentLook));
  mut(next);
  // an edited example look becomes a custom one; a pasted preset keeps its own name
  const ex = EXAMPLE_LOOKS.find(l => l.name === next.name);
  if (ex && !sameLook(ex, next)) next.name = 'Custom';
  applyLook(next);
}
// the example looks set the colours and keep anny's slider values, so this comparison leaves the sliders out
function sameLook(a, b) {
  const A = normaliseLook(a), B = normaliseLook(b);
  return Math.abs(A.skin.tone - B.skin.tone) < 1e-3 && Math.abs(A.skin.undertone - B.skin.undertone) < 1e-3 &&
    A.hair.color === B.hair.color && A.eyes.color === B.eyes.color;
}
function toneTrack() {
  const stops = [];
  for (let i = 0; i <= 10; i++) { const c = skinBase(i / 10, currentLook.skin.undertone); stops.push(`rgb(${c.map(v => Math.round(v * 255)).join(',')}) ${i * 10}%`); }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}
function underTrack() {
  const c0 = skinBase(currentLook.skin.tone, -1), c1 = skinBase(currentLook.skin.tone, 0), c2 = skinBase(currentLook.skin.tone, 1);
  const f = (c) => `rgb(${c.map(v => Math.round(v * 255)).join(',')})`;
  return `linear-gradient(90deg, ${f(c0)}, ${f(c1)}, ${f(c2)})`;
}
function syncEditor() {
  const ed = $('editor'); if (!ed || !BODY.ready) return;
  const L = currentLook;
  const tone = $('ed-tone'), under = $('ed-under');
  if (document.activeElement !== tone) tone.value = L.skin.tone;
  if (document.activeElement !== under) under.value = L.skin.undertone;
  $('ed-tone-v').textContent = L.skin.tone.toFixed(2);
  $('ed-under-v').textContent = (L.skin.undertone > 0 ? '+' : '') + L.skin.undertone.toFixed(2);
  const sk = skinBase(L.skin.tone, L.skin.undertone).map(v => Math.round(v * 255));
  tone.style.setProperty('--track', toneTrack()); tone.style.setProperty('--thumb', `rgb(${sk.join(',')})`);
  under.style.setProperty('--track', underTrack()); under.style.setProperty('--thumb', `rgb(${sk.join(',')})`);
  for (const [list, hex] of [['hair', L.hair.color], ['eyes', L.eyes.color]]) {
    $('ed-' + list).querySelectorAll('.ed-sw[data-hex]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.hex === hex)));
    const custom = $('ed-' + list + '-custom');
    if (custom && document.activeElement !== custom) custom.value = hex;
  }
  $('ed-hair-v').textContent = lookName('hair', L.hair.color);
  $('ed-eyes-v').textContent = lookName('eyes', L.eyes.color);
  $('ed-looks').querySelectorAll('.ed-chip').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.name === L.name && sameLook(EXAMPLE_LOOKS.find(l => l.name === b.dataset.name), L))));
  for (const s of BODY.sliders) {
    const inp = $('ed-b-' + s.name), out = $('ed-b-' + s.name + '-v');
    if (!inp) continue;
    const v = L.phenotype[s.name] ?? 0.5;
    if (document.activeElement !== inp) inp.value = v;
    const txt = shapeText(s, v);
    out.textContent = txt; inp.setAttribute('aria-valuetext', txt);
  }
  const rb = $('ed-shape-reset');
  if (rb) rb.disabled = BODY.sliders.every(s => Math.abs((L.phenotype[s.name] ?? 0.5) - 0.5) < 1e-6);
}
function shapeText(s, v) {
  return v.toFixed(2);
}
// body sliders, built once the model has loaded; the shape updates at most once a frame while a slider moves
let bodyPending = null;
function queueBody(name, v) {
  if (!bodyPending) {
    bodyPending = {};
    requestAnimationFrame(() => { const p = bodyPending; bodyPending = null; editLook(n => { Object.assign(n.phenotype, p); }); });
  }
  bodyPending[name] = v;
}
function buildBodySliders() {
  const box = $('ed-shape');
  if (!box || !BODY.ready) return;
  box.textContent = '';
  for (const s of BODY.sliders) {
    const wrap = document.createElement('div'); wrap.className = 'ed-slider';
    const row = document.createElement('div'); row.className = 'ed-row';
    const lab = document.createElement('label'); lab.htmlFor = 'ed-b-' + s.name; lab.textContent = s.label;
    const out = document.createElement('output'); out.id = 'ed-b-' + s.name + '-v'; out.setAttribute('for', 'ed-b-' + s.name);
    row.append(lab, out);
    const inp = document.createElement('input');
    Object.assign(inp, { type: 'range', id: 'ed-b-' + s.name, min: '0', max: '1', step: '0.01', value: '0.5' });
    inp.title = 'Double-click to return to anny\'s default (0.5)';
    inp.addEventListener('input', () => queueBody(s.name, parseFloat(inp.value)));
    inp.addEventListener('dblclick', () => { inp.value = '0.5'; queueBody(s.name, 0.5); });
    const ends = document.createElement('div'); ends.className = 'ed-ends'; ends.setAttribute('aria-hidden', 'true');
    for (const t of s.ends) { const sp = document.createElement('span'); sp.textContent = t; ends.appendChild(sp); }
    wrap.append(row, inp, ends);
    box.appendChild(wrap);
  }
  $('ed-shape-sec').hidden = false;
  syncEditor();
}
function wireEditor() {
  const ed = $('editor'), btn = $('toggle-editor');
  if (!ed || !btn) return;
  // the Character and Pose panels share the space at the side, so one closes when the other opens
  const po = $('poser'), pbtn = $('toggle-poser');
  const setOpen = (panel, button, on, focusEl) => {
    panel.hidden = !on; button.setAttribute('aria-expanded', String(on));
    if (on) focusEl.focus({ preventScroll: true }); else button.focus({ preventScroll: true });
  };
  const open = (on) => {
    if (on && po && !po.hidden) setOpen(po, pbtn, false, pbtn);
    setOpen(ed, btn, on, $('ed-close'));
    if (on) syncEditor();
    document.body.classList.toggle('editing', !ed.hidden || (po && !po.hidden));
  };
  const openPoser = (on) => {
    if (on && !ed.hidden) setOpen(ed, btn, false, btn);
    setOpen(po, pbtn, on, $('po-close'));
    if (on) syncPoser();
    document.body.classList.toggle('editing', !ed.hidden || !po.hidden);
  };
  btn.addEventListener('click', () => open(ed.hidden));
  $('ed-close').addEventListener('click', () => open(false));
  if (po && pbtn) {
    pbtn.addEventListener('click', () => openPoser(po.hidden));
    $('po-close').addEventListener('click', () => openPoser(false));
  }
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!ed.hidden) open(false); else if (po && !po.hidden) openPoser(false);
  });
  // example looks
  const chips = $('ed-looks');
  for (const l of EXAMPLE_LOOKS) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ed-chip'; b.dataset.name = l.name; b.setAttribute('aria-pressed', 'false');
    const dot = document.createElement('i');
    const c = skinBase(l.skin.tone, l.skin.undertone).map(v => Math.round(v * 255));
    dot.style.background = `linear-gradient(135deg, rgb(${c.join(',')}) 50%, ${l.hair.color} 50%)`;
    b.append(dot, document.createTextNode(l.name));
    b.addEventListener('click', () => { applyLook(Object.assign({}, l, { phenotype: currentLook.phenotype })); setEdMsg(''); });
    chips.appendChild(b);
  }
  // skin sliders (the view refreshes while dragging)
  $('ed-tone').addEventListener('input', (e) => editLook(n => { n.skin.tone = parseFloat(e.target.value); }));
  $('ed-under').addEventListener('input', (e) => editLook(n => { n.skin.undertone = parseFloat(e.target.value); }));
  buildSwatches('hair', 'hair', (hex) => editLook(n => { n.hair.color = hex; }));
  buildSwatches('eyes', 'eye', (hex) => editLook(n => { n.eyes.color = hex; }));
  // preset text: copy, paste, reset
  const box = $('ed-json');
  $('ed-copy').addEventListener('click', () => {
    const text = JSON.stringify(currentLook, null, 2);
    box.value = text;
    const fallback = () => { box.hidden = false; box.focus(); box.select(); setEdMsg('Select the text above and copy it.'); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => { box.hidden = true; setEdMsg('Preset copied.'); }, fallback);
    } else fallback();
  });
  $('ed-paste').addEventListener('click', () => {
    box.hidden = false; box.value = ''; box.focus();
    setEdMsg('Paste a preset into the box. It loads as soon as it reads as a preset.');
  });
  const tryLoad = () => {
    const t = box.value.trim();
    if (!t) return;
    try {
      const look = normaliseLook(JSON.parse(t));
      applyLook(look);
      box.hidden = true;
      setEdMsg('Loaded "' + look.name + '".');
    } catch (err) {
      setEdMsg('This text is not a preset yet. A preset is JSON with skin, hair, eyes and phenotype fields.', 'error');
    }
  };
  box.addEventListener('input', tryLoad);
  $('ed-reset').addEventListener('click', () => { applyLook(BASELINE_LOOK); box.hidden = true; setEdMsg('Back to the baseline look and anny\'s defaults.'); });
  $('ed-shape-reset').addEventListener('click', () => editLook(n => { n.phenotype = baselinePhenotype(); }));
}
wireEditor();

// ------------------------------------------------------------------ pose panel
function buildPoser() {
  const box = $('po-poses'), an = $('po-anims');
  if (!box || !an || !MOTION.clips.length) return;
  box.textContent = ''; an.textContent = '';
  const chip = (c) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ed-chip'; b.dataset.motion = c.name; b.textContent = c.label; b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => { if (c.kind === 'loop' && MOTION.cur === c) togglePlay(); else { if (c.kind === 'loop') MOTION.playing = true; setMotion(c.name); } });
    return b;
  };
  // pose groups fold open and shut; the group of the current pose opens by itself
  const groups = [];
  for (const c of MOTION.clips) {
    if (c.kind !== 'pose') { an.appendChild(chip(c)); continue; }
    let g = groups.find(x => x.name === c.group);
    if (!g) {
      const wrap = document.createElement('details'); wrap.className = 'po-group'; wrap.dataset.group = c.group;
      const sum = document.createElement('summary');
      const name = document.createElement('span'); name.textContent = c.group;
      const n = document.createElement('span'); n.className = 'po-n';
      sum.append(name, n);
      const row = document.createElement('div'); row.className = 'ed-chips'; row.setAttribute('role', 'group'); row.setAttribute('aria-label', c.group + ' poses');
      wrap.append(sum, row); box.appendChild(wrap);
      g = { name: c.group, row, n, count: 0 }; groups.push(g);
    }
    g.row.appendChild(chip(c));
    g.n.textContent = String(++g.count);
  }
  $('po-play').addEventListener('click', togglePlay);
  $('po-speed').querySelectorAll('.ed-chip').forEach(b => b.addEventListener('click', () => {
    MOTION.speed = parseFloat(b.dataset.speed); rememberMotion(); syncPoser();
  }));
  if (CORR.ready) {
    $('po-tissue-sec').hidden = false;
    $('po-tissue').querySelectorAll('.ed-chip').forEach(b => b.addEventListener('click', () => setCorrectivesOn(b.dataset.corr === 'on')));
  }
  $('toggle-poser').hidden = false;
  syncPoser();
}
function rememberMotion() {
  try { localStorage.setItem('anny-motion', JSON.stringify({ name: MOTION.cur.name, speed: MOTION.speed, playing: MOTION.playing })); } catch (e) { /* storage unavailable */ }
}
function togglePlay() {
  MOTION.playing = !MOTION.playing;
  rememberMotion(); syncPoser(); resetAccum();
}
function syncPoser() {
  const box = $('poser'); if (!box || !MOTION || !MOTION.cur) return;
  box.querySelectorAll('.ed-chip[data-motion]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.motion === MOTION.cur.name)));
  const loop = MOTION.cur.kind === 'loop';
  const play = $('po-play');
  play.disabled = !loop;
  play.textContent = loop && !MOTION.playing ? 'Play' : 'Pause';
  play.setAttribute('aria-pressed', String(loop && MOTION.playing));
  $('po-speed').querySelectorAll('.ed-chip').forEach(b => b.setAttribute('aria-pressed', String(parseFloat(b.dataset.speed) === MOTION.speed)));
  $('po-state').textContent = MOTION.cur.label + (loop ? (MOTION.playing ? '' : ', paused') : '');
  if (!loop && MOTION.cur.group && syncPoser.openFor !== MOTION.cur.name) {
    syncPoser.openFor = MOTION.cur.name;
    const d = box.querySelector(`details.po-group[data-group="${MOTION.cur.group}"]`);
    if (d) d.open = true;
  }
  const cr = $('po-credit');
  if (cr) cr.textContent = MOTION.cur.credit ? 'Pose by ' + MOTION.cur.credit + ', from the MakeHuman community (CC0).' : '';
  const tb = $('po-tissue');
  if (tb) tb.querySelectorAll('.ed-chip').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.corr === 'on') === CORR.on)));
}

// test hook
window.setView = (yaw = 25, pitch = 3, dist = 0.8, ty = TARGET.y, tz = TARGET.z, fov = 24, tx = 0) => {
  camera.fov = fov; camera.updateProjectionMatrix();
  placeCamera(yaw, pitch, dist, tx, ty, tz);
  resetAccum();
  while (accCount < MAX_ACC) renderPass();
  return true;
};
// a framing of the page ('body' or 'face'), without the camera move
window.setFrame = (name) => {
  flyTo(name);
  controls.target.copy(tween.toT); camera.position.copy(tween.toP); tween = null;
  camera.lookAt(controls.target); controls.update();
  resetAccum();
  while (accCount < MAX_ACC) renderPass();
  return true;
};
window.setPreset = (n) => { applyPreset(n); return true; };
window.setCorrectives = (on) => { setCorrectivesOn(on); return CORR.ready; };
window.__CORR = CORR;
window.__U = U;
window.setLook = (look) => { applyLook(look, false); return true; };
window.setMotion = (name, t = 0, paused = true) => { MOTION.playing = !paused; return setMotion(name, { instant: true, t, remember: false }); };
window.__MOTION = MOTION; window.__RIG = RIG;
window.__BODY = BODY;
// anny's sliders from a test: returns the times of the update (ms)
window.setSliders = (v) => { applyLook(Object.assign({}, currentLook, { phenotype: Object.assign({}, currentLook.phenotype, v) }), false); return { body: BODY.lastMs, total: BODY.lastTotalMs, steps: BODY.lastTiming }; };
window.__look = () => currentLook;
init().catch(e => { console.error(e); showError('The model could not be loaded: ' + e.message); });
