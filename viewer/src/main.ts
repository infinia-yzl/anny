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
//  Ported from the viewer of the 3D Model experiment (git commit b10538d, 3D Model/baseline_body/source.zip,
//  web/app.js). The page draws anny
//  in the frame of the legacy figure (anny.poses.authoring.rig), so the tuning of the lights and shaders carries over.
// =====================================================================================
// @ts-nocheck -- the renderer below is the legacy JavaScript; the new modules (anny_shape, subdivision, body) are typed
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { AnnyBody, vertexNormals as normalsOf } from './body.ts';
import { sampleFace, type FaceData } from './anny_shape.ts';
import {
  BG_GLSL, COVER_GLSL, EYE_FS, EYE_VS, GLSL_COMMON, HAIR_FS, HAIR_VS, HEAD_OUT_GLSL, NOISE_GLSL, PROP_FS, PROP_VS, SKIN_FS, SKIN_GLSL,
  SKIN_VS, SSS_GLSL,
} from './shading.ts';
import { Hair } from './hair/gpu.ts';
import { COLLIDERS, fitColliders } from './hair/colliders.ts';
import { STRAND_VS } from './hair/glsl.ts';

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
  // the density volume of the hair (anny.hair.styles.density_volume): occlusion and scalp tint of the skin under it
  uHairOcc: { value: null }, uHairOccLo: { value: new THREE.Vector3() }, uHairOccSize: { value: new THREE.Vector3(1, 1, 1) }, uHairOn: { value: 0 },
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

// ------------------------------------------------------------------ skin material

// ------------------------------------------------------------------ eye material

// ------------------------------------------------------------------ hair material

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
// S.P: the points in the frames of their root triangles; ids: the strand of each (for a subset of the strands)
function buildRibbons(S, ao, seed, ids = null) {
  const { P, counts, nS, total } = S;
  const nv = total * 2;
  const pos = new Float32Array(nv * 3), tan = new Int8Array(nv * 4), ha = new Uint8Array(nv * 4), sid = new Float32Array(nv);
  const Q = S.Q || null, tip = Q ? new Float32Array(nv * 3) : null;
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
        if (tip) { tip[v * 3] = Q[(p + j) * 3]; tip[v * 3 + 1] = Q[(p + j) * 3 + 1]; tip[v * 3 + 2] = Q[(p + j) * 3 + 2]; }
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
  if (tip) g.setAttribute('tipLocal', new THREE.BufferAttribute(tip, 3));
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
// the strands of anny's hair (hair/gpu.ts): instanced ribbons that read the points of pass B
let HAIR: Hair | null = null;
function makeStrandMesh(hair: Hair, p, defines) {
  const hu = Object.assign({}, U, {
    uPoints: hair.uPoints, uP: hair.uP,
    uWidth: { value: p.width }, uTipWidth: { value: p.width * p.tip }, uViewportH: { value: innerHeight * PR }, uMinPix: { value: p.minPix ?? 0.7 },
    uHairColor: { value: new THREE.Vector3(...p.color) }, uHairRough: { value: p.rough }, uSpecScale: { value: p.spec ?? 1.0 }, uDiffScale: { value: p.diff ?? 1.3 },
    uHeadC: hair.uHeadC,
  });
  const opts = { vertexShader: STRAND_VS, fragmentShader: toMRT(HAIR_FS), side: THREE.DoubleSide, glslVersion: THREE.GLSL3, blending: THREE.NoBlending };
  const m = new THREE.ShaderMaterial(Object.assign({ uniforms: hu, defines }, opts));
  const dm = new THREE.ShaderMaterial(Object.assign({ uniforms: Object.assign({}, hu, { uViewportH: { value: SHADOW_SIZE / 2 }, uMinPix: { value: 1.2 } }),
    defines: Object.assign({ HAIR_DEPTH: '' }, defines) }, opts));
  const mesh = new THREE.Mesh(hair.geometry, m);
  mesh.frustumCulled = false; mesh.userData.depthMat = dm; mesh.userData.hu = hu;
  mesh.userData.center = new THREE.Vector3().copy(hair.uHeadC.value);
  hair.onGeometry = (g) => { mesh.geometry = g; };
  scene.add(mesh); hairObjs.push(mesh);
  return mesh;
}

// ------------------------------------------------------------------ body shape: anny's phenotype sliders
// The body comes from anny (body.ts): the sliders set anny's blend-shape coefficients, the coarse body and the
// joints follow, the fine surface is rebuilt by subdivision with its detail layers, and the eyes, the hair, the
// skeleton and the corrective shapes follow the body. Every slider runs from 0 to 1, and anny's default is 0.5.
const BODY: any = { ready: false, anny: null as AnnyBody | null, geo: null, values: null, face: null, faceSliders: [], sliders: [], lastMs: 0, lastTotalMs: 0, lastTiming: {} };
// anny's race phenotypes: their values mix by their shares
const RACES = ['african', 'asian', 'caucasian'];
function raceShare(phenotype: any, name: string) {
  const v = (k: string) => phenotype[k] ?? PHENOTYPE_DEFAULT;
  const sum = RACES.reduce((a, k) => a + v(k), 0);
  return sum > 0 ? v(name) / sum : 1 / RACES.length;
}
function sliderEnds(tables: any, label: string) {
  const v = tables.variations.find((x: any) => x[0] === label);
  const a = tables.anchors[label];
  if (!v || !a) return ['0', '1'];
  const at = (x: number) => { let k = 0; for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - x) < Math.abs(a[k] - x)) k = i; return v[1][k]; };
  const nice = (s: string) => s.replace(/(min|max|average|ideal|uncommon)(\w+)/, '$1 $2').replace(/^./, (c: string) => c.toUpperCase());
  return [nice(at(0)), nice(at(1))];
}
function initBody(meta: any, B: any, geo: any) {
  const f32 = (name: string) => { const b = B[name]; return new Float32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4); };
  const i16 = (name: string) => { const b = B[name]; return new Int16Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 2); };
  const u32 = (name: string) => { const b = B[name]; return new Uint32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4); };
  const sm = meta.shape;
  const at = geo.attributes;
  const nF = at.position.count;
  // the detail records hold (t1, t2, n) without the relief, then the height of the relief along n
  const d4 = new Int16Array(B.head_detail.data.buffer, B.head_detail.data.byteOffset, nF * 4), detailRaw = new Int16Array(nF * 3), relief = new Int16Array(nF);
  for (let i = 0; i < nF; i++) { detailRaw[i * 3] = d4[i * 4]; detailRaw[i * 3 + 1] = d4[i * 4 + 1]; detailRaw[i * 3 + 2] = d4[i * 4 + 2]; relief[i] = d4[i * 4 + 3]; }
  const rows = new Uint32Array(B.head_row.data.buffer, B.head_row.data.byteOffset, nF);
  // the shape components come as records of 4 values (x, y, z and a pad)
  const c4 = i16('shape_components'), nComp = sm.components * sm.coarse_vertices, components = new Int16Array(nComp * 3);
  for (let i = 0; i < nComp; i++) { components[i * 3] = c4[i * 4]; components[i * 3 + 1] = c4[i * 4 + 1]; components[i * 3 + 2] = c4[i * 4 + 2]; }
  // anny's face shapes: sparse offsets on the coarse body (records of x, y, z and a pad), bone-head deltas, the
  // landmarks that size the scale groups, and the factors of the face-shape distribution
  let face: FaceData | undefined;
  if (sm.face && B.face_offsets) {
    const o4 = i16('face_offsets'), n = o4.length / 4, offsets = new Int16Array(n * 3);
    for (let i = 0; i < n; i++) { offsets[i * 3] = o4[i * 4]; offsets[i * 3 + 1] = o4[i * 4 + 1]; offsets[i * 3 + 2] = o4[i * 4 + 2]; }
    face = { tables: sm.face, lmTemplate: f32('face_lm_template'), lmBlend: f32('face_lm_blend'), ids: u32('face_ids'), offsets,
      boneDeltas: f32('face_bones'), prior: B.face_prior ? f32('face_prior') : undefined };
  }
  const body = new AnnyBody(meta, {
    template: f32('coarse_template'), components,
    projection: f32('shape_projection').subarray(0, sm.components * sm.blend_shapes),
    jointTemplate: f32('joint_template'), jointBlend: f32('joint_blend'),
    quads: u32('coarse_quads'), rows, detail: detailRaw, relief,
    index: geo.index.array, rest: at.rest.array, nsmooth: at.nsmooth.array.slice(), coarseSkin: B.coarse_skin.data, face,
  }, at.position.array, at.normal.array, at.nsmooth.array);
  body.detailStep = B.head_detail.info.step || 1e-5;
  BODY.anny = body; BODY.geo = geo;
  BODY.sliders = sm.sliders.map((name: string) => ({ name, label: name.charAt(0).toUpperCase() + name.slice(1),
    race: RACES.includes(name), ends: RACES.includes(name) ? [] : sliderEnds(sm.tables, name) }));
  BODY.face = face || null;
  BODY.faceSliders = face ? face.tables.names.map((name: string, i: number) => ({ name, group: face.tables.groups[i],
    label: faceLabel(name, face.tables.groups[i]), range: face.tables.ranges[i], ends: (face.tables.ends || [])[i] || ['', ''] })) : [];
  BODY.ready = true;
}
// a readable label for a face-shape name, without the group's own word: 'nose-scale-vert' -> 'Height'
const FACE_WORDS: Record<string, string> = { 'scale-vert': 'height', 'scale-horiz': 'width', 'scale-depth': 'depth', 'trans': 'position',
  'down-up': 'up and down', 'in-out': 'in and out', 'backward-forward': 'back and forth', 'decr-incr': '', 'invertedtriangular': 'inverted triangular' };
const FACE_GROUP_WORDS: Record<string, string[]> = { head: ['head'], forehead: ['forehead'], brows: ['eyebrows'], eyes: ['eye'], nose: ['nose'],
  cheeks: ['cheek'], mouth: ['mouth'], chin: ['chin'], ears: ['ear'] };
function faceLabel(name: string, group: string) {
  let t = name;
  for (const w of FACE_GROUP_WORDS[group] || []) if (t.startsWith(w + '-')) t = t.slice(w.length + 1);
  for (const [k, v] of Object.entries(FACE_WORDS)) t = t.split(k).join(v);
  t = t.replace(/-/g, ' ').replace(/\s+/g, ' ').trim() || name;
  return t.charAt(0).toUpperCase() + t.slice(1);
}
const FACE_GROUP_TITLES: Record<string, string> = { head: 'Head', forehead: 'Forehead', brows: 'Brows', eyes: 'Eyes', nose: 'Nose',
  cheeks: 'Cheeks', mouth: 'Mouth', chin: 'Chin and jaw', ears: 'Ears', detail: 'Detail' };
function sameValues(a: any, b: any) {
  if (!a || !b) return false;
  return BODY.sliders.every((s: any) => Math.abs((a[s.name] ?? 0.5) - (b[s.name] ?? 0.5)) < 1e-6);
}
function sameFace(a: any, b: any) {
  if (!a || !b) return false;
  return BODY.faceSliders.every((s: any) => Math.abs((a[s.name] ?? 0) - (b[s.name] ?? 0)) < 1e-6);
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
function applyBodyShape(values: any, face: any = {}) {
  if (!BODY.ready) return false;
  const v: any = {}, f: any = {};
  for (const s of BODY.sliders) v[s.name] = typeof values?.[s.name] === 'number' ? values[s.name] : 0.5;
  for (const s of BODY.faceSliders) if (typeof face?.[s.name] === 'number' && face[s.name] !== 0) f[s.name] = face[s.name];
  if (sameValues(v, BODY.values) && sameFace(f, BODY.faceValues)) return false;
  const t0 = performance.now(), steps: any = {};
  let last = t0;
  const mark = (name: string) => { const t = performance.now(); steps[name] = Math.round((t - last) * 10) / 10; last = t; };
  BODY.values = v; BODY.faceValues = f;
  clearCorrectives();
  const r = BODY.anny.update(v, f);
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
  fitHairColliders();
  groundMotion();
  setRigRest();
  evalMotion();
}
// the colliders of the hair's physics on the current body at rest (hair/colliders.ts): the head sphere sits at the
// cranium's centre of anny's default body, carried to the current head
function fitHairColliders() {
  if (!HAIR || !RIG.ready) return;
  const c = new THREE.Vector3().fromArray(MANIFEST.hair.centre).applyMatrix4(HEADMAP.m);
  const names = RIG.bones.map((b: any) => b.name), b = BODY.anny;
  HAIR.setColliders(fitColliders(COLLIDERS, names, b.joints, { cranium: [c.x, c.y, c.z] }, b.coarse, b.nBody, HEADMAP.k));
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
  if (HAIR) { RIG.skel.update(); HAIR.setPose(RIG.skel.boneMatrices, RIG.head); }
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
// look@2 adds the face-shape values (face: {name: value}); a look@1 preset reads as a face of anny's defaults
const LOOK_FORMAT = 'anny-viewer/look@3';   // @3 adds the hair style and its parameters; @2 looks still load
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
const BASELINE_LOOK: any = { format: LOOK_FORMAT, name: 'Baseline', skin: { tone: 0.35, undertone: 0 }, hair: { color: '#271f16', style: 'medium_tousled' }, eyes: { color: '#875f3d' }, phenotype: {}, face: {} };
const EXAMPLE_LOOKS: any[] = [
  BASELINE_LOOK,
  { format: LOOK_FORMAT, name: 'Fair', skin: { tone: 0.12, undertone: -0.3 }, hair: { color: '#6b4a30' }, eyes: { color: '#5f84a8' } },
  { format: LOOK_FORMAT, name: 'Olive', skin: { tone: 0.5, undertone: 0.45 }, hair: { color: '#141110' }, eyes: { color: '#4a3020' } },
  { format: LOOK_FORMAT, name: 'Deep', skin: { tone: 0.9, undertone: 0.1 }, hair: { color: '#141110' }, eyes: { color: '#4a3020' } },
];
// characters: presets that set everything a look carries (the phenotype sliders, the face and the colours), as the
// presets of a character creator do; the example looks set the colours only. The faces are random faces of the
// calibrated distribution for each body, chosen on the review renders.
const CHARACTERS: any[] = [
  { format: LOOK_FORMAT, name: 'Asian woman', skin: { tone: 0.28, undertone: 0.4 }, hair: { color: '#141110', style: 'lob' }, eyes: { color: '#4a3020' },
    phenotype: { gender: 1, age: 0.78, muscle: 0.45, weight: 0.45, height: 0.5, proportions: 0.5, african: 0, asian: 1, caucasian: 0 },
    face: { 'head-scale-vert': 0.51, 'head-fat': -0.02, 'head-invertedtriangular': 0.03, 'head-back-scale-depth': -0.96, 'head-round': 0.02, 'head-scale-depth': 1.76, 'head-scale-horiz': 0.43, 'forehead-temple': -0.01, 'forehead-trans': 0.25, 'forehead-scale-vert': 0.06, 'forehead-nubian': -0.01, 'eyebrows-angle': -0.01, 'eyebrows-trans-backward-forward': 0.04, 'eye-corner1': 0.34, 'eye-height1': 0.1, 'eye-scale': 0.04, 'eye-push1': -0.01, 'eye-eyefold-down-up': 0.03, 'eye-height3': 0.16, 'eye-height2': 0.13, 'eye-bag-decr-incr': -0.01, 'eye-push2': -0.06, 'eye-trans-down-up': 0.18, 'eye-bag-height': 0.03, 'eye-epicanthus': 0.02, 'eye-eyefold-angle': -0.1, 'eye-bag-in-out': -0.03, 'eye-corner2': -0.26, 'eye-trans-in-out': -0.22, 'eye-eyefold-concave-convex': 0.02, 'nose-nostrils-angle': 0.04, 'nose-flaring': -0.11, 'nose-curve': -0.02, 'nose-scale-depth': 0.01, 'nose-scale-vert': 0.04, 'nose-point-width': -0.11, 'nose-scale-horiz': 0.07, 'nose-width1': -0.01, 'nose-trans-backward-forward': 0.08, 'nose-trans-down-up': -0.06, 'nose-hump': -0.01, 'nose-nostrils-width': 0.05, 'nose-greek': 0.03, 'nose-point': -0.04, 'nose-width2': -0.05, 'nose-width3': 0.03, 'nose-septumangle': 0.06, 'nose-base': -0.04, 'nose-compression': -0.03, 'nose-volume': 0.01, 'cheek-trans': 0.08, 'cheek-inner': -0.04, 'cheek-bones': -0.17, 'mouth-lowerlip-height': -0.01, 'mouth-scale-horiz': -0.08, 'mouth-cupidsbow': -0.01, 'mouth-laugh-lines': 0.01, 'mouth-scale-depth': -0.17, 'mouth-lowerlip-volume': -0.18, 'mouth-angles': 0.1, 'mouth-upperlip-ext': 0.01, 'mouth-lowerlip-width': -0.01, 'mouth-trans-backward-forward': 0.06, 'mouth-scale-vert': 0.04, 'mouth-dimples': 0.05, 'mouth-upperlip-height': 0.03, 'mouth-upperlip-volume': 0.03, 'mouth-cupidsbow-width': 0.03, 'mouth-lowerlip-ext': -0.06, 'mouth-upperlip-middle': -0.1, 'mouth-trans-down-up': -0.04, 'mouth-lowerlip-middle': -0.06, 'mouth-philtrum-volume': 0.04, 'chin-jaw-drop': 0.01, 'chin-height': 0.09, 'chin-prominent': 0.01, 'chin-bones': -0.01, 'chin-width': 0.02, 'chin-triangle': 0.02, 'chin-prognathism': 0.09, 'ear-lobe': 0.11, 'ear-shape-square-round': -0.04, 'ear-trans-backward-forward': -0.18, 'ear-rot': 0.01, 'ear-trans-down-up': 0.03, 'ear-scale-vert': 0.04, 'ear-flap': 0.02, 'ear-shape-pointed-triangle': 0.11, 'ear-scale': 0.14, 'ear-scale-depth': -0.06, 'ear-wing': 0.03, 'detail-1': 0.08, 'detail-2': -0.17, 'detail-3': -0.11, 'detail-4': 0.12, 'detail-5': 0.03, 'detail-6': -0.2, 'detail-7': 0.01, 'detail-8': -0.21, 'detail-9': -0.33, 'detail-10': 0.09 } },
  { format: LOOK_FORMAT, name: 'Asian man', skin: { tone: 0.32, undertone: 0.4 }, hair: { color: '#141110', style: 'low_taper_fade' }, eyes: { color: '#4a3020' },
    phenotype: { gender: 0, age: 0.79, muscle: 0.55, weight: 0.5, height: 0.5, proportions: 0.5, african: 0, asian: 1, caucasian: 0 },
    face: { 'head-scale-vert': 0.11, 'head-fat': -0.09, 'head-invertedtriangular': 0.07, 'head-back-scale-depth': -1.43, 'head-scale-depth': 0.81, 'head-scale-horiz': 0.11, 'forehead-temple': 0.08, 'forehead-trans': 0.07, 'forehead-scale-vert': -0.01, 'forehead-nubian': 0.01, 'eyebrows-angle': -0.03, 'eyebrows-trans-down-up': 0.02, 'eyebrows-trans-backward-forward': -0.03, 'eye-corner1': 0.02, 'eye-height1': 0.02, 'eye-scale': 0.14, 'eye-push1': -0.12, 'eye-eyefold-down-up': 0.02, 'eye-height3': -0.06, 'eye-height2': -0.06, 'eye-bag-decr-incr': 0.05, 'eye-push2': -0.26, 'eye-bag-height': 0.05, 'eye-epicanthus': 0.05, 'eye-eyefold-angle': 0.03, 'eye-bag-in-out': 0.04, 'eye-corner2': -0.04, 'eye-trans-in-out': 0.01, 'eye-eyefold-concave-convex': -0.13, 'nose-nostrils-angle': 0.01, 'nose-flaring': 0.04, 'nose-curve': -0.01, 'nose-scale-depth': -0.05, 'nose-scale-vert': 0.02, 'nose-point-width': -0.02, 'nose-scale-horiz': -0.02, 'nose-width1': -0.04, 'nose-trans-backward-forward': -0.08, 'nose-trans-down-up': -0.04, 'nose-hump': 0.02, 'nose-nostrils-width': -0.01, 'nose-greek': 0.04, 'nose-point': -0.04, 'nose-width2': 0.01, 'nose-width3': 0.06, 'nose-septumangle': -0.01, 'nose-base': 0.01, 'nose-compression': 0.01, 'cheek-trans': 0.02, 'cheek-volume': 0.01, 'cheek-inner': 0.07, 'cheek-bones': -0.04, 'mouth-lowerlip-height': 0.04, 'mouth-scale-horiz': -0.21, 'mouth-cupidsbow': 0.01, 'mouth-laugh-lines': 0.01, 'mouth-scale-depth': -0.13, 'mouth-lowerlip-volume': -0.07, 'mouth-angles': -0.04, 'mouth-upperlip-ext': -0.01, 'mouth-trans-backward-forward': 0.2, 'mouth-upperlip-width': -0.04, 'mouth-scale-vert': 0.06, 'mouth-dimples': 0.02, 'mouth-upperlip-height': -0.02, 'mouth-upperlip-volume': 0.02, 'mouth-cupidsbow-width': 0.03, 'mouth-upperlip-middle': 0.08, 'mouth-trans-down-up': 0.02, 'mouth-lowerlip-middle': 0.02, 'mouth-philtrum-volume': 0.03, 'chin-jaw-drop': 0.01, 'chin-prominent': 0.05, 'chin-bones': 0.18, 'chin-width': 0.04, 'chin-triangle': 0.04, 'chin-prognathism': 0.07, 'ear-lobe': 0.12, 'ear-shape-square-round': -0.13, 'ear-trans-backward-forward': 0.1, 'ear-rot': 0.03, 'ear-trans-down-up': 0.1, 'ear-scale-vert': 0.07, 'ear-flap': 0.05, 'ear-shape-pointed-triangle': -0.16, 'ear-scale': -0.09, 'ear-scale-depth': -0.18, 'ear-wing': 0.14, 'detail-1': 0.24, 'detail-2': -0.19, 'detail-3': -0.05, 'detail-4': -0.04, 'detail-5': -0.01, 'detail-6': 0.09, 'detail-7': -0.11, 'detail-8': -0.02, 'detail-9': -0.28, 'detail-10': 0.09 } },
  { format: LOOK_FORMAT, name: 'Eurasian woman', skin: { tone: 0.22, undertone: 0.25 }, hair: { color: '#271f16', style: 'long_layers' }, eyes: { color: '#875f3d' },
    phenotype: { gender: 1, age: 0.78, muscle: 0.45, weight: 0.45, height: 0.5, proportions: 0.5, african: 0, asian: 1, caucasian: 1 },
    face: { 'head-scale-vert': 0.34, 'head-fat': 0.07, 'head-invertedtriangular': 0.12, 'head-back-scale-depth': -0.82, 'head-round': 0.03, 'head-scale-depth': 1.48, 'head-scale-horiz': 0.21, 'forehead-temple': -0.23, 'forehead-trans': 0.05, 'forehead-scale-vert': 0.05, 'forehead-nubian': 0.01, 'eyebrows-angle': 0.01, 'eyebrows-trans-down-up': -0.02, 'eyebrows-trans-backward-forward': 0.07, 'eye-corner1': 0.12, 'eye-height1': -0.1, 'eye-scale': 0.24, 'eye-push1': 0.03, 'eye-eyefold-down-up': -0.04, 'eye-height3': -0.18, 'eye-height2': -0.14, 'eye-bag-decr-incr': -0.02, 'eye-push2': -0.04, 'eye-trans-down-up': 0.15, 'eye-bag-height': -0.04, 'eye-epicanthus': -0.01, 'eye-eyefold-angle': -0.06, 'eye-bag-in-out': -0.06, 'eye-corner2': -0.22, 'eye-trans-in-out': 0.01, 'eye-eyefold-concave-convex': 0.26, 'nose-nostrils-angle': -0.04, 'nose-flaring': -0.01, 'nose-curve': -0.07, 'nose-scale-depth': -0.03, 'nose-scale-vert': -0.1, 'nose-point-width': -0.06, 'nose-scale-horiz': -0.02, 'nose-width1': -0.04, 'nose-trans-backward-forward': 0.06, 'nose-trans-down-up': 0.02, 'nose-hump': -0.03, 'nose-nostrils-width': 0.09, 'nose-greek': -0.13, 'nose-point': -0.07, 'nose-width2': -0.03, 'nose-width3': -0.03, 'nose-septumangle': -0.11, 'nose-base': 0.16, 'nose-compression': 0.01, 'nose-volume': -0.14, 'cheek-trans': 0.04, 'cheek-volume': -0.08, 'cheek-inner': 0.07, 'cheek-bones': -0.12, 'mouth-lowerlip-height': -0.02, 'mouth-scale-horiz': -0.04, 'mouth-scale-depth': -0.01, 'mouth-lowerlip-volume': 0.12, 'mouth-angles': 0.1, 'mouth-lowerlip-width': 0.02, 'mouth-trans-backward-forward': -0.14, 'mouth-upperlip-width': 0.06, 'mouth-scale-vert': 0.01, 'mouth-dimples': -0.05, 'mouth-upperlip-height': 0.04, 'mouth-upperlip-volume': -0.08, 'mouth-lowerlip-ext': 0.02, 'mouth-upperlip-middle': 0.03, 'mouth-trans-down-up': 0.06, 'mouth-philtrum-volume': 0.03, 'chin-jaw-drop': -0.05, 'chin-height': -0.05, 'chin-prominent': -0.1, 'chin-width': -0.03, 'chin-prognathism': -0.17, 'ear-lobe': 0.02, 'ear-shape-square-round': 0.01, 'ear-trans-backward-forward': 0.12, 'ear-rot': -0.25, 'ear-trans-down-up': -0.11, 'ear-scale-vert': -0.01, 'ear-flap': 0.03, 'ear-shape-pointed-triangle': -0.1, 'ear-scale': 0.03, 'ear-scale-depth': -0.09, 'ear-wing': 0.16, 'detail-1': 0.06, 'detail-2': -0.16, 'detail-3': 0.12, 'detail-4': -0.04, 'detail-5': -0.22, 'detail-6': -0.38, 'detail-7': -0.03, 'detail-8': -0.21, 'detail-9': 0.14, 'detail-10': -0.2 } },
  { format: LOOK_FORMAT, name: 'Eurasian man', skin: { tone: 0.26, undertone: 0.25 }, hair: { color: '#271f16', style: 'textured_quiff' }, eyes: { color: '#4a3020' },
    phenotype: { gender: 0, age: 0.79, muscle: 0.55, weight: 0.5, height: 0.5, proportions: 0.5, african: 0, asian: 1, caucasian: 1 },
    face: { 'head-scale-vert': 0.41, 'head-fat': -0.04, 'head-rectangular': 0.02, 'head-back-scale-depth': -1.21, 'head-scale-depth': 1.42, 'head-scale-horiz': 0.21, 'head-triangular': 0.06, 'forehead-temple': 0.01, 'forehead-trans': -0.2, 'forehead-scale-vert': 0.02, 'eyebrows-trans-down-up': -0.05, 'eyebrows-trans-backward-forward': 0.01, 'eye-corner1': 0.05, 'eye-height1': -0.05, 'eye-scale': 0.08, 'eye-push1': -0.09, 'eye-eyefold-down-up': -0.09, 'eye-height3': -0.16, 'eye-height2': -0.06, 'eye-bag-decr-incr': -0.05, 'eye-push2': -0.11, 'eye-trans-down-up': 0.19, 'eye-bag-height': -0.14, 'eye-epicanthus': 0.07, 'eye-eyefold-angle': 0.07, 'eye-bag-in-out': 0.04, 'eye-corner2': -0.05, 'eye-trans-in-out': 0.11, 'eye-eyefold-concave-convex': 0.27, 'nose-nostrils-angle': -0.09, 'nose-flaring': 0.03, 'nose-curve': -0.07, 'nose-scale-depth': -0.07, 'nose-scale-vert': -0.03, 'nose-point-width': -0.09, 'nose-scale-horiz': 0.07, 'nose-width1': 0.01, 'nose-trans-backward-forward': -0.25, 'nose-trans-down-up': 0.06, 'nose-hump': -0.03, 'nose-nostrils-width': 0.01, 'nose-greek': -0.13, 'nose-point': 0.06, 'nose-width2': 0.07, 'nose-width3': -0.1, 'nose-septumangle': 0.1, 'nose-base': 0.12, 'nose-compression': -0.01, 'nose-volume': -0.09, 'cheek-trans': -0.01, 'cheek-volume': 0.01, 'cheek-inner': 0.12, 'cheek-bones': 0.21, 'mouth-lowerlip-height': 0.22, 'mouth-scale-horiz': -0.07, 'mouth-scale-depth': -0.04, 'mouth-lowerlip-volume': 0.04, 'mouth-angles': -0.08, 'mouth-upperlip-ext': 0.02, 'mouth-lowerlip-width': 0.01, 'mouth-trans-backward-forward': 0.33, 'mouth-upperlip-width': 0.01, 'mouth-scale-vert': 0.2, 'mouth-dimples': 0.02, 'mouth-upperlip-height': 0.09, 'mouth-upperlip-volume': 0.03, 'mouth-cupidsbow-width': 0.02, 'mouth-lowerlip-ext': 0.1, 'mouth-upperlip-middle': 0.27, 'mouth-trans-down-up': -0.01, 'mouth-lowerlip-middle': 0.07, 'mouth-philtrum-volume': -0.03, 'chin-jaw-drop': 0.01, 'chin-prominent': -0.05, 'chin-bones': 0.14, 'chin-width': -0.06, 'chin-cleft': 0.01, 'chin-triangle': 0.04, 'chin-prognathism': -0.02, 'ear-lobe': 0.21, 'ear-shape-square-round': -0.02, 'ear-trans-backward-forward': -0.12, 'ear-rot': -0.01, 'ear-trans-down-up': -0.16, 'ear-scale-vert': -0.13, 'ear-flap': 0.21, 'ear-shape-pointed-triangle': 0.17, 'ear-scale': -0.14, 'ear-scale-depth': 0.11, 'ear-wing': 0.19, 'detail-1': 0.12, 'detail-2': 0.04, 'detail-3': -0.14, 'detail-4': -0.14, 'detail-5': -0.16, 'detail-6': -0.03, 'detail-7': 0.52, 'detail-8': 0.05, 'detail-9': 0.15, 'detail-10': 0.18 } },
];
const LOOK_PARTS: any = { hair: null, brows: null, lashes: null, hairMeshes: [], eyes: [], iris: null };
let currentLook: any = JSON.parse(JSON.stringify(BASELINE_LOOK));
let hairWanted = true;
function updateHairVisibility() {
  LOOK_PARTS.hairMeshes.forEach((m: any, i: number) => { m.visible = i > 0 || hairWanted; });
  U.uHairOn.value = hairWanted && HAIR ? 1 : 0;
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
    hair: normaliseHair(o.hair, col(o.hair && o.hair.color, b.hair.color)),
    eyes: { color: col(o.eyes && o.eyes.color, b.eyes.color) },
    phenotype: Object.fromEntries(Object.keys(baselinePhenotype()).map((k) => [k, Math.round(num(ph[k], 0, 1, PHENOTYPE_DEFAULT) * 1000) / 1000])),
    face: normaliseFace(o.face),
  };
}
// the hair of a preset: a known style (the baseline's otherwise), its parameters within the style's ranges (the
// style's defaults when missing), and the side of the part for the styles that have one
function hairSpecs(): any[] { return MANIFEST?.hair?.styles || []; }
function normaliseHair(h: any, color: string) {
  const specs = hairSpecs(), o = h && typeof h === 'object' ? h : {};
  const spec = specs.find((x) => x.name === o.style) || specs.find((x) => x.name === BASELINE_LOOK.hair.style) || specs[0];
  if (!spec) return { color, style: BASELINE_LOOK.hair.style };
  const d = HAIR ? HAIR.defaults(spec.name) : { length: 1, curl: 0, volume: 1, density: 1, fade: 0 };
  const c = spec.controls, num = (v: any, r: number[], def: number) => (typeof v === 'number' && isFinite(v)) ? Math.round(Math.min(r[1], Math.max(r[0], v)) * 1000) / 1000 : def;
  const out: any = { color, style: spec.name, length: num(o.length, c.length, d.length), curl: num(o.curl, c.curl, d.curl),
    volume: num(o.volume, c.volume, d.volume), density: num(o.density, c.density, d.density) };
  if (spec.render.fade) out.fade = num(o.fade, FADE_RANGE, 0);
  if (spec.mirror) out.part = o.part === 'right' ? 'right' : 'left';
  return out;
}
const FADE_RANGE = [-8, 12];   // degrees: the fade band moves down or up
// the face-shape values of a preset: known names within their ranges, zeros left out
function normaliseFace(face: any) {
  const out: any = {};
  if (!face || typeof face !== 'object') return out;
  const tables = MANIFEST?.shape?.face;
  if (!tables) return out;
  tables.names.forEach((name: string, i: number) => {
    const v = face[name];
    if (typeof v !== 'number' || !isFinite(v)) return;
    const [lo, hi] = tables.ranges[i], r = Math.round(Math.min(hi, Math.max(lo, v)) * 1000) / 1000;
    if (r !== 0) out[name] = r;
  });
  return out;
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
  // the hairstyle and its parameters (hair/gpu.ts): a new style binds its guides, the parameters are uniforms
  if (HAIR) {
    const hh = L.hair, mirror = hh.part === 'right';
    if (!HAIR.style || HAIR.style.spec.name !== hh.style || HAIR.style.mirrored !== mirror) HAIR.setStyle(hh.style, mirror);
    HAIR.setParams({ length: hh.length, curl: hh.curl, volume: hh.volume, density: hh.density, fade: hh.fade ?? 0 });
  }
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
  applyBodyShape(L.phenotype, L.face);
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
// texels: 3 per strand (the root frame), or 6 with a tip binding (the root frame, then the tip frame)
function strandTexture(nS: number, texels: number) {
  const h = Math.ceil(nS * texels / STRAND_TEX_W);
  const tex = new THREE.DataTexture(new Float32Array(STRAND_TEX_W * h * 4), STRAND_TEX_W, h, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter; tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace; tex.needsUpdate = true;
  return tex;
}
function updateHairShape() { for (const h of HAIRSETS) h.tex.needsUpdate = true; HAIR?.bodyChanged(); }
// triangles and barycentric coordinates of a binding record (3 corners, 2 coordinates)
function bindingRecords(buf: any, n: number) {
  const dv = new DataView(buf.data.buffer, buf.data.byteOffset, buf.data.byteLength);
  const corners = new Uint32Array(n * 3), bary = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) corners[i * 3 + k] = dv.getUint32(i * 20 + k * 4, true);
    const u = dv.getFloat32(i * 20 + 12, true), w = dv.getFloat32(i * 20 + 16, true);
    bary[i * 3] = u; bary[i * 3 + 1] = w; bary[i * 3 + 2] = 1 - u - w;
  }
  return { corners, bary };
}
// bind a strand set to the skin; returns the ribbon source (local points) and the texture of its frames
function strandBinding(B: any, name: string, S: any) {
  const nS = S.nS, root = bindingRecords(B[name + '_bind'], nS);
  const tip = B[name + '_tip'] ? bindingRecords(B[name + '_tip'], nS) : undefined;
  const texels = tip ? 6 : 3, tex = strandTexture(nS, texels);
  BODY.anny.bindStrands(name, S.P, S.counts, root.corners, root.bary, (tex.image.data as Float32Array).subarray(0, nS * texels * 4), tip);
  HAIRSETS.push({ name, tex });
  const set = BODY.anny.strands[name];
  return { local: { P: set.local, Q: set.tipLocal || null, counts: S.counts, nS, total: S.total, skin: S.skin }, tex, tips: !!tip };
}

// the model data as base64 text: inline in the single-file page, or in text files next to the page (the page of
// build.mjs --parts, for hosts that limit the size of a file)
async function modelData(): Promise<string> {
  const w = window as any;
  if (w.MODEL_B64) return w.MODEL_B64;
  const parts: string[] = w.MODEL_PARTS || ['build/model.b64'];
  const texts = await Promise.all(parts.map(async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`the data file ${url} did not load (${r.status})`);
    return r.text();
  }));
  return texts.join('');
}

let MeshoptDecoder: any = null;
let MANIFEST: any = null;
async function init() {
  setProgress(0.04, 'Decoding model');
  MeshoptDecoder = (window as any).MeshoptDecoderRef;
  const b64 = await modelData();
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
  buildFaceSliders();
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
  // the hair: the scalp layout and its styles (hair/gpu.ts); it binds first, since its guide roots set the scalp size
  HAIR = new Hair(meta.hair, (name) => B[name].data, BODY.anny, U.uHeadInv);
  HAIR.lod = isSmall ? 0.5 : 1;
  HAIR.setStyle(meta.hair.styles.some((x) => x.name === 'medium_tousled') ? 'medium_tousled' : meta.hair.styles[0].name);
  // the physics runs unless the page takes still pictures (tests and reviews) or the address turns it off
  HAIR.setPhysics(qs.has('physics') ? qs.get('physics') !== 'off' : !SHOT);
  $('toggle-physics')?.setAttribute('aria-pressed', String(HAIR.physics));
  fitHairColliders();
  buildHairUI();
  Object.assign(U, HAIR.volumeUniforms());
  U.uHairOn.value = 1;
  setProgress(0.75, 'Placing strands');
  await nextFrame();
  const hairMesh = makeStrandMesh(HAIR, { width: 0.00012 / Math.sqrt(HAIR.lod), tip: 0.45, color: [0.020, 0.0125, 0.0082], rough: 0.38, diff: 1.3, spec: 0.38 }, ENV_DEFINES);
  const browB = strandBinding(B, 'brows', decodeStrands(B, 'brows'));
  const browMesh = makeHairMesh(buildRibbons(browB.local, null, 5), { width: 0.00010, tip: 0.3, color: [0.016, 0.009, 0.006], rough: 0.62, spec: 0.15, diff: 1.3, center: [0, 0.515, 0.06] }, ENV_DEFINES, browB.tex);
  const lashB = strandBinding(B, 'lashes', decodeStrands(B, 'lashes'));
  const lashMesh = makeHairMesh(buildRibbons(lashB.local, null, 9), { width: 0.00012, tip: 0.25, color: [0.006, 0.004, 0.003], rough: 0.55, spec: 0.3, diff: 1.3, center: [0, 0.51, 0.10] }, ENV_DEFINES, lashB.tex);
  (window as any).__hair = hairMesh; (window as any).__BUFFERS = B; (window as any).__HAIR = HAIR;
  LOOK_PARTS.hair = hairMesh.userData.hu; LOOK_PARTS.brows = browMesh.userData.hu; LOOK_PARTS.lashes = lashMesh.userData.hu;
  LOOK_PARTS.hairMeshes = [hairMesh, browMesh, lashMesh];
  const q = new URLSearchParams(location.search);
  const start = loadRememberedLook() || normaliseLook(BASELINE_LOOK);
  for (const s of BODY.sliders) if (q.has(s.name)) start.phenotype[s.name] = Math.min(1, Math.max(0, parseFloat(q.get(s.name))));
  applyLook(start, false);
  // tests wait for this: the start look is on the body, so a look set from now on stays
  (window as any).__READY = true;
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
const sssMat = new THREE.ShaderMaterial({
  uniforms: sssU, vertexShader: FSQ_VS,
  fragmentShader: SSS_GLSL(SSS_N) + '\nuniform sampler2D tSSS; varying vec2 vUv; void main(){ gl_FragColor = sssBlur(tSSS, vUv, vec2(1.0, 0.0)); }',
  depthTest: false, depthWrite: false, toneMapped: false, blending: THREE.NoBlending,
});
allocRTs();
const accMat = new THREE.ShaderMaterial({
  uniforms: Object.assign({ tNew: { value: null }, tAcc: { value: null }, uW: { value: 1 }, tSSSH: { value: null } }, sssU),
  defines: SSS_ON ? { SSS: '' } : {},
  vertexShader: FSQ_VS,
  fragmentShader: SSS_GLSL(SSS_N) + `
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
  if (HAIR && HAIR.update(renderer)) shadowsDirty = true;
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
let hairClock = false;   // a test steps the hair's physics itself (window.stepHair)
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
  if (HAIR && HAIR.due()) resetAccum();
  // the hair's physics: it sleeps once the hair rests, so the picture can refine
  if (HAIR && hairWanted && !hairClock && HAIR.stepPhysics(dt)) { shadowsDirty = true; resetAccum(); }
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
  const phys = $('toggle-physics');
  phys?.addEventListener('click', () => {
    const on = phys.getAttribute('aria-pressed') !== 'true';
    phys.setAttribute('aria-pressed', String(on));
    HAIR?.setPhysics(on);
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
  const ch = CHARACTERS.find(l => l.name === next.name);
  if (ch && !sameCharacter(ch, next)) next.name = 'Custom';
  applyLook(next);
}
// the example looks set the colours and keep anny's slider values, so this comparison leaves the sliders out
function sameLook(a, b) {
  const A = normaliseLook(a), B = normaliseLook(b);
  return Math.abs(A.skin.tone - B.skin.tone) < 1e-3 && Math.abs(A.skin.undertone - B.skin.undertone) < 1e-3 &&
    A.hair.color === B.hair.color && A.eyes.color === B.eyes.color;
}
// a character also carries the slider values and the face
function sameCharacter(a, b) {
  const A = normaliseLook(a), B = normaliseLook(b);
  const close = (x, y) => Object.keys(Object.assign({}, x, y)).every(k => Math.abs((x[k] ?? 0) - (y[k] ?? 0)) < 1e-3);
  return sameLook(A, B) && close(A.phenotype, B.phenotype) && close(A.face, B.face) && A.hair.style === B.hair.style;
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
  syncHair(L);
  $('ed-eyes-v').textContent = lookName('eyes', L.eyes.color);
  $('ed-looks').querySelectorAll('.ed-chip').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.name === L.name && sameLook(EXAMPLE_LOOKS.find(l => l.name === b.dataset.name), L))));
  $('ed-chars').querySelectorAll('.ed-chip').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.name === L.name && sameCharacter(CHARACTERS.find(l => l.name === b.dataset.name), L))));
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
  for (const s of BODY.faceSliders) {
    const inp = $('ed-f-' + s.name), out = $('ed-f-' + s.name + '-v');
    if (!inp) continue;
    const v = L.face?.[s.name] ?? 0;
    if (document.activeElement !== inp) inp.value = v;
    out.textContent = v.toFixed(2); inp.setAttribute('aria-valuetext', v.toFixed(2));
  }
  const fr = $('ed-face-reset');
  if (fr) fr.disabled = !L.face || Object.keys(L.face).length === 0;
  document.querySelectorAll('#ed-face details.po-group').forEach((d: any) => {
    const n = BODY.faceSliders.filter(s => s.group === d.dataset.group && (L.face?.[s.name] ?? 0) !== 0).length;
    const c = d.querySelector('.po-n'); if (c) c.textContent = n ? `${n} set` : '';
  });
}
function shapeText(s, v) {
  return s.race ? `${Math.round(100 * raceShare(currentLook.phenotype, s.name))} %` : v.toFixed(2);
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
  let raceHead = false;
  for (const s of BODY.sliders) {
    if (s.race && !raceHead) {
      raceHead = true;
      const h = document.createElement('div'); h.className = 'ed-sub'; h.textContent = 'Ethnicity';
      const note = document.createElement('p'); note.className = 'ed-note';
      note.textContent = 'The three values mix by their shares, shown on the right. Eurasian is Asian and Caucasian at equal values, with African at 0.';
      box.append(h, note);
    }
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
    wrap.append(row, inp);
    if (s.ends.length) {
      const ends = document.createElement('div'); ends.className = 'ed-ends'; ends.setAttribute('aria-hidden', 'true');
      for (const t of s.ends) { const sp = document.createElement('span'); sp.textContent = t; ends.appendChild(sp); }
      wrap.appendChild(ends);
    }
    box.appendChild(wrap);
  }
  $('ed-shape-sec').hidden = false;
  syncEditor();
}
// ------------------------------------------------------------------ hair styles and their sliders
const HAIR_FAMILIES: [string, string][] = [['short', 'Short'], ['medium', 'Medium'], ['long', 'Long'], ['tied', 'Tied']];
const HAIR_SLIDERS: any[] = [
  { key: 'length', label: 'Length', ends: ['Shorter', 'Longer'] },
  { key: 'curl', label: 'Curl', ends: ['Straight', 'Curly'] },
  { key: 'volume', label: 'Volume', ends: ['Flat', 'Full'] },
  { key: 'density', label: 'Density', ends: ['Thin', 'Thick'] },
  { key: 'fade', label: 'Fade height', ends: ['Low', 'High'], fade: true },
];
function buildHairUI() {
  const box = $('ed-hair-styles'), params = $('ed-hair-params');
  if (!box || !HAIR) return;
  box.textContent = ''; params.textContent = '';
  for (const [family, title] of HAIR_FAMILIES) {
    const specs = hairSpecs().filter((s) => s.family === family);
    if (!specs.length) continue;
    const g = document.createElement('div');
    const h = document.createElement('div'); h.className = 'ed-sub'; h.textContent = title;
    const chips = document.createElement('div'); chips.className = 'ed-chips';
    for (const s of specs) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'ed-chip'; b.dataset.style = s.name; b.setAttribute('aria-pressed', 'false');
      b.textContent = s.label;
      b.addEventListener('click', () => editLook((n) => { n.hair = { color: n.hair.color, style: s.name, part: n.hair.part }; }));
      chips.appendChild(b);
    }
    g.append(h, chips);
    box.appendChild(g);
  }
  for (const s of HAIR_SLIDERS) {
    const wrap = document.createElement('div'); wrap.className = 'ed-slider'; wrap.id = 'ed-h-' + s.key + '-w';
    const row = document.createElement('div'); row.className = 'ed-row';
    const lab = document.createElement('label'); lab.htmlFor = 'ed-h-' + s.key; lab.textContent = s.label;
    const out = document.createElement('output'); out.id = 'ed-h-' + s.key + '-v'; out.setAttribute('for', 'ed-h-' + s.key);
    row.append(lab, out);
    const inp = document.createElement('input');
    Object.assign(inp, { type: 'range', id: 'ed-h-' + s.key, step: '0.01' });
    inp.title = 'Double-click to return to the style\'s default';
    inp.addEventListener('input', () => queueHair(s.key, parseFloat(inp.value)));
    inp.addEventListener('dblclick', () => { const d = s.key === 'fade' ? 0 : (HAIR.defaults(currentLook.hair.style) as any)[s.key]; queueHair(s.key, d); });
    const ends = document.createElement('div'); ends.className = 'ed-ends'; ends.setAttribute('aria-hidden', 'true');
    for (const t of s.ends) { const sp = document.createElement('span'); sp.textContent = t; ends.appendChild(sp); }
    wrap.append(row, inp, ends);
    params.appendChild(wrap);
  }
  // the side of the part, for the styles that have one
  const part = document.createElement('div'); part.className = 'ed-row'; part.id = 'ed-h-part-w';
  const pl = document.createElement('span'); pl.className = 'ed-sub'; pl.textContent = 'Part';
  const pc = document.createElement('div'); pc.className = 'ed-chips';
  for (const side of ['left', 'right']) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ed-chip'; b.dataset.part = side; b.textContent = side === 'left' ? 'Left' : 'Right';
    b.addEventListener('click', () => editLook((n) => { n.hair.part = side; }));
    pc.appendChild(b);
  }
  part.append(pl, pc);
  params.appendChild(part);
  syncEditor();
}
// the hair sliders apply at most once a frame
let hairPending: any = null;
function queueHair(key: string, v: number) {
  if (!hairPending) {
    hairPending = {};
    requestAnimationFrame(() => { const p = hairPending; hairPending = null; editLook((n) => { Object.assign(n.hair, p); }); });
  }
  hairPending[key] = v;
}
function syncHair(L: any) {
  const spec = hairSpecs().find((s) => s.name === L.hair.style);
  if (!spec || !$('ed-hair-styles')) return;
  $('ed-hair-style-v').textContent = spec.label;
  $('ed-hair-styles').querySelectorAll('.ed-chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.style === spec.name)));
  for (const s of HAIR_SLIDERS) {
    const w = $('ed-h-' + s.key + '-w'), inp = $('ed-h-' + s.key), out = $('ed-h-' + s.key + '-v');
    if (!w) continue;
    const range = s.fade ? FADE_RANGE : spec.controls[s.key];
    const show = s.fade ? !!spec.render.fade : range[1] > range[0];
    w.hidden = !show;
    if (!show) continue;
    inp.min = String(range[0]); inp.max = String(range[1]);
    const v = L.hair[s.key] ?? 0;
    if (document.activeElement !== inp) inp.value = String(v);
    out.textContent = s.fade ? (v > 0 ? '+' : '') + v.toFixed(0) + '°' : v.toFixed(2);
  }
  const part = $('ed-h-part-w');
  if (part) {
    part.hidden = !spec.mirror;
    part.querySelectorAll('.ed-chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.part === (L.hair.part || 'left'))));
  }
}
// face sliders: one group per part of the face, closed at first; the face updates at most once a frame
let facePending = null;
function queueFace(name, v) {
  if (!facePending) {
    facePending = {};
    requestAnimationFrame(() => {
      const p = facePending; facePending = null;
      editLook(n => { n.face = Object.assign({}, n.face || {}, p); for (const k of Object.keys(n.face)) if (n.face[k] === 0) delete n.face[k]; });
    });
  }
  facePending[name] = v;
}
function buildFaceSliders() {
  const box = $('ed-face');
  if (!box || !BODY.ready || !BODY.face) return;
  box.textContent = '';
  for (const group of BODY.face.tables.group_order) {
    const d = document.createElement('details'); d.className = 'po-group'; d.dataset.group = group;
    const sum = document.createElement('summary'); sum.textContent = FACE_GROUP_TITLES[group] || group;
    const count = document.createElement('span'); count.className = 'po-n'; sum.appendChild(count);
    const list = document.createElement('div'); list.className = 'ed-shape ed-face-list';
    for (const s of BODY.faceSliders.filter(x => x.group === group)) {
      const wrap = document.createElement('div'); wrap.className = 'ed-slider';
      const row = document.createElement('div'); row.className = 'ed-row';
      const lab = document.createElement('label'); lab.htmlFor = 'ed-f-' + s.name; lab.textContent = s.label; lab.title = s.name;
      const out = document.createElement('output'); out.id = 'ed-f-' + s.name + '-v'; out.setAttribute('for', 'ed-f-' + s.name);
      row.append(lab, out);
      const inp = document.createElement('input');
      Object.assign(inp, { type: 'range', id: 'ed-f-' + s.name, min: String(s.range[0]), max: String(s.range[1]), step: '0.01', value: '0' });
      if (s.range[0] < 0) inp.classList.add('ed-centred');
      inp.title = 'Double-click to return to 0';
      inp.addEventListener('input', () => queueFace(s.name, parseFloat(inp.value)));
      inp.addEventListener('dblclick', () => { inp.value = '0'; queueFace(s.name, 0); });
      const ends = document.createElement('div'); ends.className = 'ed-ends'; ends.setAttribute('aria-hidden', 'true');
      for (const t of s.ends) { const sp = document.createElement('span'); sp.textContent = t; ends.appendChild(sp); }
      wrap.append(row, inp, ends);
      list.appendChild(wrap);
    }
    d.append(sum, list);
    box.appendChild(d);
  }
  $('ed-face-random').hidden = !BODY.face.prior;
  $('ed-face-sec').hidden = false;
  syncEditor();
}
// a face drawn from anny's calibrated face-shape distribution for the current age, gender, weight and muscle
function randomFace() {
  if (!BODY.face?.prior) return;
  const f = sampleFace(BODY.face, currentLook.phenotype);
  const out: any = {};
  BODY.face.tables.names.forEach((name: string, i: number) => { const r = Math.round(f[i] * 1000) / 1000; if (r !== 0) out[name] = r; });
  editLook(n => { n.face = out; });
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
  // characters
  const charChips = $('ed-chars');
  for (const l of CHARACTERS) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ed-chip'; b.dataset.name = l.name; b.setAttribute('aria-pressed', 'false');
    const dot = document.createElement('i');
    const c = skinBase(l.skin.tone, l.skin.undertone).map(v => Math.round(v * 255));
    dot.style.background = `linear-gradient(135deg, rgb(${c.join(',')}) 50%, ${l.hair.color} 50%)`;
    b.append(dot, document.createTextNode(l.name));
    b.addEventListener('click', () => { applyLook(JSON.parse(JSON.stringify(l))); setEdMsg(''); });
    charChips.appendChild(b);
  }
  // example looks
  const chips = $('ed-looks');
  for (const l of EXAMPLE_LOOKS) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ed-chip'; b.dataset.name = l.name; b.setAttribute('aria-pressed', 'false');
    const dot = document.createElement('i');
    const c = skinBase(l.skin.tone, l.skin.undertone).map(v => Math.round(v * 255));
    dot.style.background = `linear-gradient(135deg, rgb(${c.join(',')}) 50%, ${l.hair.color} 50%)`;
    b.append(dot, document.createTextNode(l.name));
    b.addEventListener('click', () => { applyLook(Object.assign({}, l, { phenotype: currentLook.phenotype, face: currentLook.face })); setEdMsg(''); });
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
  $('ed-face-reset').addEventListener('click', () => editLook(n => { n.face = {}; }));
  $('ed-face-random').addEventListener('click', () => { randomFace(); setEdMsg(''); });
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
// a frontal portrait of the head for the photo benchmark (anny.faces.authoring.photos): the face framing seen from
// the front, with the head of the current body
window.setPortrait = (fov = 24, margin = 1.0) => {
  const f = FRAMES.face, t = new THREE.Vector3(...f.target);
  if (RIG.ready) t.applyMatrix4(_headFull);
  camera.fov = fov; camera.updateProjectionMatrix();
  placeCamera(0, 0, frameDistance(f) * (RIG.ready ? HEADMAP.k : 1) * margin, t.x, t.y, t.z);
  tween = null;
  resetAccum();
  while (accCount < MAX_ACC) renderPass();
  return true;
};
window.setHair = (on) => { hairWanted = !!on; updateHairVisibility(); return true; };
// the hair's style and its parameters (length, curl, volume, density, fade), and the points of pass B for tests
window.setHairStyle = (name, mirror = false) => { HAIR.setStyle(name, mirror); shadowsDirty = true; resetAccum(); return HAIR.stats(); };
window.setHairParams = (p) => { HAIR.setParams(p); shadowsDirty = true; resetAccum(); return HAIR.params; };
window.hairStats = () => HAIR ? HAIR.stats() : null;
window.hairPoints = (n) => { HAIR.update(renderer); return Array.from(HAIR.readPoints(renderer, n)); };
window.hairRest = (n) => HAIR.readRest(renderer, n);
window.hairGuides = () => HAIR.readGuides(renderer);
window.stepHair = (dt = 1 / 60) => { hairClock = true; if (HAIR.stepPhysics(dt)) { shadowsDirty = true; resetAccum(); } return HAIR.stats(); };
window.setHairPhysics = (on) => { HAIR.setPhysics(!!on); const b = $('toggle-physics'); if (b) b.setAttribute('aria-pressed', String(!!on)); return HAIR.physics; };
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
// anny's face-shape values from a test (missing names take 0), and a face drawn from the distribution
window.setFace = (v) => { applyLook(Object.assign({}, currentLook, { face: v }), false); return { body: BODY.lastMs, total: BODY.lastTotalMs }; };
window.randomFace = () => { randomFace(); return currentLook.face; };
init().catch(e => { console.error(e); showError('The model could not be loaded: ' + e.message); });
