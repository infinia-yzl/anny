// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// anny's hair on the GPU. Each piece of data lives at the level where it changes:
//   - the scalp layout: the render roots, their guides and weights (textures made once);
//   - the style: the guide curves in the frames of their triangles, and the density volume (on a style change);
//   - the body: the frames of the guide triangles and the render roots on the current body (on a slider step);
//   - the pose: the pose of each guide root, and the motion of the simulation (each moving frame).
// Pass A poses the guides and pass B builds the points of every render strand into a float texture; the ribbons read
// two texels per vertex, so a still frame costs no hair work at all. The parameters of a style are uniforms: the
// length, curl, volume and fade change pass B alone, and the density changes the number of instances.

import * as THREE from 'three';
import type { AnnyBody } from '../body.ts';
import {
  TEX_W, available, decodeLayout, decodeStyle, densityVolume, guideMatrices, mirrorStyle, rootsOnBody,
  type Bytes, type HairLayout, type HairMeta, type HairStyle, type StyleSpec, type Volume,
} from './data.ts';
import { HAIR_PASS_A, HAIR_PASS_B } from './glsl.ts';

export interface HairParams { length: number; curl: number; volume: number; density: number; fade: number }
export const DEFAULT_PARAMS: HairParams = { length: 1, curl: 1, volume: 1, density: 1, fade: 0 };

const FSQ_VS = 'void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }';
const rows = (n: number) => Math.max(1, Math.ceil(n / TEX_W));

function floatTexture(n: number) {
  const data = new Float32Array(TEX_W * rows(n) * 4);
  const tex = new THREE.DataTexture(data, TEX_W, rows(n), THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.generateMipmaps = false; tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return { tex, data };
}
function uintTexture(src: Uint32Array, n: number) {
  const data = new Uint32Array(TEX_W * rows(n) * 4);
  data.set(src.subarray(0, Math.min(src.length, data.length)));
  const tex = new THREE.DataTexture(data, TEX_W, rows(n), THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
  tex.internalFormat = 'RGBA32UI';
  tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}
function floatTarget(n: number) {
  return new THREE.WebGLRenderTarget(TEX_W, rows(n), {
    type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: false, stencilBuffer: false,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false,
  });
}
function pass(fs: string, uniforms: any) {
  const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: FSQ_VS, fragmentShader: fs, glslVersion: THREE.GLSL3,
    depthTest: false, depthWrite: false, toneMapped: false, blending: THREE.NoBlending });
  const tri = new THREE.BufferGeometry();
  tri.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
  const mesh = new THREE.Mesh(tri, mat); mesh.frustumCulled = false;
  const scene = new THREE.Scene(); scene.add(mesh);
  return { mat, scene };
}

// the ribbons of P points: one instance per strand, two vertices per point (the shader reads gl_VertexID)
export function strandTemplate(P: number, count: number) {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(P * 2 * 3), 3));
  const idx: number[] = [];
  for (let j = 0; j < P - 1; j++) { const b = j * 2; idx.push(b, b + 1, b + 2, b + 1, b + 3, b + 2); }
  g.setIndex(idx);
  g.instanceCount = count;
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.5, 0.03), 0.6);
  return g;
}

export class Hair {
  meta: HairMeta; layout: HairLayout; body: AnnyBody; bytes: Bytes;
  specs: Map<string, StyleSpec> = new Map();
  cache: Map<string, HairStyle> = new Map();
  style: HairStyle | null = null;
  params: HairParams = { ...DEFAULT_PARAMS };
  count = 0;
  lod = 1;   // the share of the strands drawn (phones draw half, with wider strands)
  frames = floatTexture(1); local = floatTexture(1); tipLocal = floatTexture(1); info = floatTexture(1);
  mat: { tex: THREE.DataTexture; data: Float32Array }; motion = floatTexture(1);
  rootRest: { tex: THREE.DataTexture; data: Float32Array };
  rootData: THREE.DataTexture;
  occ: THREE.Data3DTexture; volume: Volume | null = null;
  rtA: THREE.WebGLRenderTarget | null = null; rtB: THREE.WebGLRenderTarget;
  A: { mat: THREE.ShaderMaterial; scene: THREE.Scene }; B: { mat: THREE.ShaderMaterial; scene: THREE.Scene };
  cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  geometry: THREE.InstancedBufferGeometry | null = null;
  uHeadC = { value: new THREE.Vector3() };
  uPoints: { value: THREE.Texture | null } = { value: null };
  uP = { value: 2 };
  dirtyA = true; dirtyB = true; volumeDirty = false; lastChange = 0;
  onGeometry: ((g: THREE.InstancedBufferGeometry) => void) | null = null;

  constructor(meta: HairMeta, bytes: Bytes, body: AnnyBody, headInv: { value: THREE.Matrix4 }) {
    this.meta = meta; this.bytes = bytes; this.body = body;
    for (const s of meta.styles) this.specs.set(s.name, s);
    const L = this.layout = decodeLayout(bytes, meta);
    this.uHeadC.value.fromArray(meta.centre);
    this.mat = floatTexture(L.G * 3);
    for (let g = 0; g < L.G; g++) { this.mat.data[g * 12] = 1; this.mat.data[g * 12 + 5] = 1; this.mat.data[g * 12 + 10] = 1; }
    this.rootRest = floatTexture(L.R);
    this.rootData = uintTexture(L.rootData, L.R);
    this.occ = new THREE.Data3DTexture(new Uint8Array(16), 2, 2, 2);
    this.rtB = floatTarget(1);
    const hl = meta.hairline;
    this.A = pass(HAIR_PASS_A, {
      uLocal: { value: this.local.tex }, uTipLocal: { value: this.tipLocal.tex }, uFrames: { value: this.frames.tex },
      uMat: { value: this.mat.tex }, uMotion: { value: this.motion.tex }, uP: this.uP, uG: { value: L.G }, uMoving: { value: 0 },
    });
    this.B = pass(HAIR_PASS_B, {
      uGuides: { value: null }, uGuideInfo: { value: this.info.tex }, uFrames: { value: this.frames.tex }, uMat: { value: this.mat.tex },
      uRootRest: { value: this.rootRest.tex }, uRootData: { value: this.rootData },
      uOcc: { value: this.occ }, uOccLo: { value: new THREE.Vector3() }, uOccSize: { value: new THREE.Vector3(1, 1, 1) }, uHeadInv: headInv,
      uP: this.uP, uCount: { value: 0 }, uScale: { value: 1 }, uHeadC: this.uHeadC,
      uLength: { value: 1 }, uCurl: { value: 1 }, uVolume: { value: 1 }, uFadeShift: { value: 0 },
      uJitter: { value: new THREE.Vector2(1, 1) }, uThin: { value: new THREE.Vector3() }, uCoarse: { value: new THREE.Vector2() }, uCoarseTip: { value: 0 },
      uFine: { value: new THREE.Vector2() }, uFinePow: { value: new THREE.Vector2(1, 1) }, uSectors: { value: 6 },
      uSectorR: { value: meta.sector_radius }, uSimilarity: { value: meta.similarity },
      uPivotBlend: { value: 0.02 }, uCurlP: { value: new THREE.Vector3(0, 1, 0.15) }, uFrizz: { value: new THREE.Vector4() }, uFly: { value: new THREE.Vector3() },
      uHLPhi: { value: hl.phi.slice() }, uHLEl: { value: hl.el.slice() },
      uHairline: { value: new Array(19).fill(0) }, uFadeStart: { value: new Array(19).fill(0) }, uFade: { value: new THREE.Vector4(1, 1, 1, 0) },
    });
  }

  names(): string[] { return [...this.specs.keys()]; }

  // the parameters of a style at their defaults: a straight style starts without curl (anny.hair.styles.style_params)
  defaults(name: string): HairParams {
    const spec = this.specs.get(name)!, cu = spec.render.curl;
    return { ...DEFAULT_PARAMS, curl: cu && cu.radius_mm > 0 ? 1 : 0 };
  }

  // switch to a style (mirror: the part on the other side)
  setStyle(name: string, mirror = false) {
    const spec = this.specs.get(name);
    if (!spec) throw new Error('no hairstyle ' + name);
    const key = name + (mirror ? ':m' : '');
    let s = this.cache.get(key);
    if (!s) {
      const base = this.cache.get(name) || decodeStyle(this.bytes, this.layout, spec);
      this.cache.set(name, base);
      s = mirror ? mirrorStyle(base, this.layout) : base;
      this.cache.set(key, s);
    }
    this.style = s;
    const L = this.layout, G = L.G, P = s.P, body = this.body;
    // bind the guides: the page's body keeps them in the frames of their triangles (AnnyBody.bindStrands)
    const nF = G * 6;
    if (this.frames.data.length < rows(nF) * TEX_W * 4) {
      this.frames = floatTexture(nF);
      this.A.mat.uniforms.uFrames.value = this.frames.tex; this.B.mat.uniforms.uFrames.value = this.frames.tex;
    }
    const tip = s.mirrored ? { corners: L.guideCorners, bary: L.guideBary } : { corners: s.tipCorners, bary: s.tipBary };
    body.bindStrands('hair', s.points, new Uint8Array(G).fill(P), L.guideCorners, L.guideBary, this.frames.data.subarray(0, nF * 4), tip);
    body.followStrands();
    const set = body.strands.hair;
    if (this.local.data.length < rows(G * P) * TEX_W * 4) {
      this.local = floatTexture(G * P); this.tipLocal = floatTexture(G * P); this.motion = floatTexture(G * P);
      this.A.mat.uniforms.uLocal.value = this.local.tex; this.A.mat.uniforms.uTipLocal.value = this.tipLocal.tex;
      this.A.mat.uniforms.uMotion.value = this.motion.tex;
    }
    for (let g = 0; g < G; g++) for (let j = 0; j < P; j++) {
      const i = g * P + j;
      for (let k = 0; k < 3; k++) { this.local.data[i * 4 + k] = set.local[i * 3 + k]; this.tipLocal.data[i * 4 + k] = set.tipLocal![i * 3 + k]; }
      this.local.data[i * 4 + 3] = s.tipOn[g];
    }
    this.local.tex.needsUpdate = true; this.tipLocal.tex.needsUpdate = true; this.frames.tex.needsUpdate = true;
    // two texels per guide: (segment, default length, flick, group) and (pivot)
    if (this.info.data.length < rows(G * 2) * TEX_W * 4) { this.info = floatTexture(G * 2); this.B.mat.uniforms.uGuideInfo.value = this.info.tex; }
    for (let g = 0; g < G; g++) {
      this.info.data[g * 8] = s.seg[g]; this.info.data[g * 8 + 1] = s.length[g];
      this.info.data[g * 8 + 2] = s.flick[g]; this.info.data[g * 8 + 3] = s.group[g];
      this.info.data[g * 8 + 4] = s.pivot[g];
    }
    this.info.tex.needsUpdate = true;
    // pass A: P + 2 texels per guide; pass B: P texels per strand
    this.uP.value = P;
    if (!this.rtA || this.rtA.height !== rows(G * (P + 2))) { this.rtA?.dispose(); this.rtA = floatTarget(G * (P + 2)); }
    if (this.rtB.height !== rows(L.R * P)) { this.rtB.dispose(); this.rtB = floatTarget(L.R * P); }
    this.B.mat.uniforms.uGuides.value = this.rtA.texture;
    this.uPoints.value = this.rtB.texture;
    this.styleUniforms();
    this.applyCount();
    if (!this.geometry || this.geometry.attributes.position.count !== P * 2) {
      this.geometry?.dispose();
      this.geometry = strandTemplate(P, this.count);
      this.onGeometry?.(this.geometry);
    }
    this.geometry.instanceCount = this.count;
    this.rebuildVolume();
    this.bodyChanged();
  }

  private styleUniforms() {
    const u = this.B.mat.uniforms, r = this.style!.spec.render, MM = 0.001;
    u.uJitter.value.set(r.strand_jitter[0], r.strand_jitter[1]);
    const th = r.thinning; u.uThin.value.set(th ? th.share : 0, th ? th.range[0] : 1, th ? th.range[1] : 1);
    const cl = r.clump;
    u.uCoarse.value.set(cl.coarse[0], cl.coarse[1]); u.uCoarseTip.value = cl.coarse_tip ?? 0.55;
    u.uFine.value.set(cl.fine[0], cl.fine[1]); u.uFinePow.value.set(cl.fine_power[0], cl.fine_power[1]); u.uSectors.value = cl.sectors;
    // the curl at the parameter 1: the style's own, or the waves that the slider adds to a straight style
    // (anny.hair.styles.curl_shape)
    const cu = r.curl, own = cu && cu.radius_mm > 0;
    u.uCurlP.value.set((own ? cu.radius_mm : 4.0) * MM, (own ? cu.period_mm : 40.0) * MM, cu ? (cu.ramp ?? 0.15) : 0.15);
    const fr = r.frizz; u.uFrizz.value.set(fr ? fr.mm[0] * MM : 0, fr ? fr.mm[1] * MM : 0, fr ? fr.cycles[0] : 0, fr ? fr.cycles[1] : 0);
    const fl = r.flyaways; u.uFly.value.set(fl ? fl.share : 0, fl ? fl.mm[0] * MM : 0, fl ? fl.mm[1] * MM : 0);
    u.uHairline.value = r.hairline ? r.hairline.slice() : new Array(19).fill(0);
    const fd = r.fade;
    u.uFadeStart.value = fd ? fd.start.slice() : new Array(19).fill(0);
    u.uFade.value.set(fd ? fd.width : 1, fd ? fd.clipper : 1, fd ? fd.top : 1, fd ? 1 : 0);
    this.paramUniforms();
  }

  private paramUniforms() {
    const u = this.B.mat.uniforms, p = this.params;
    u.uLength.value = p.length; u.uCurl.value = p.curl; u.uVolume.value = p.volume; u.uFadeShift.value = p.fade;
  }

  private applyCount() {
    const d = this.style ? (this.style.spec.render.density ?? 1) : 1;
    this.count = Math.max(0, Math.min(this.layout.R, Math.round(this.layout.R * d * this.params.density * this.lod)));
    this.B.mat.uniforms.uCount.value = this.count;
    if (this.geometry) this.geometry.instanceCount = this.count;
  }

  setParams(p: Partial<HairParams>) {
    const before = { ...this.params };
    Object.assign(this.params, p);
    this.paramUniforms();
    if (before.density !== this.params.density) this.applyCount();
    if (before.length !== this.params.length || before.density !== this.params.density || before.fade !== this.params.fade) {
      this.volumeDirty = true; this.lastChange = performance.now();
    }
    this.dirtyB = true;
  }

  private rebuildVolume() {
    const s = this.style!;
    const v = this.volume = densityVolume(this.meta, this.layout, s, this.params, this.count);
    this.occ.dispose();
    const [nx, ny, nz] = v.dims;
    this.occ = new THREE.Data3DTexture(v.data, nx, ny, nz);
    this.occ.format = THREE.RGFormat; this.occ.type = THREE.UnsignedByteType;
    this.occ.minFilter = this.occ.magFilter = THREE.LinearFilter;
    this.occ.wrapS = this.occ.wrapT = this.occ.wrapR = THREE.ClampToEdgeWrapping;
    this.occ.unpackAlignment = 1; this.occ.needsUpdate = true;
    const u = this.B.mat.uniforms;
    u.uOcc.value = this.occ;
    u.uOccLo.value.set(v.lo[0] - v.h / 2, v.lo[1] - v.h / 2, v.lo[2] - v.h / 2);
    u.uOccSize.value.set(nx * v.h, ny * v.h, nz * v.h);
    this.volumeDirty = false; this.dirtyB = true;
  }

  // the uniforms the skin shader needs to read the density volume
  volumeUniforms() { const u = this.B.mat.uniforms; return { uHairOcc: u.uOcc, uHairOccLo: u.uOccLo, uHairOccSize: u.uOccSize }; }

  // after a slider step: AnnyBody.update has moved the guide frames; the render roots follow
  bodyChanged() {
    rootsOnBody(this.body.pos, this.layout, this.rootRest.data);
    this.rootRest.tex.needsUpdate = true; this.frames.tex.needsUpdate = true;
    this.B.mat.uniforms.uScale.value = this.body.headScale();
    this.dirtyA = true;
  }

  // the pose: three.js bone matrices (Skeleton.boneMatrices)
  setPose(bones: Float32Array | null) {
    const L = this.layout;
    if (bones) guideMatrices(L, bones, this.mat.data);
    this.mat.tex.needsUpdate = true;
    this.dirtyA = true;
  }

  // a rebuild of the density volume waits until the parameters rest for 150 ms
  due(): boolean { return this.volumeDirty && performance.now() - this.lastChange > 150; }

  // run the passes that the changes need; returns true when the strands moved
  update(renderer: THREE.WebGLRenderer): boolean {
    if (!this.style) return false;
    if (this.due()) this.rebuildVolume();
    if (!this.dirtyA && !this.dirtyB) return false;
    const prev = renderer.getRenderTarget();
    if (this.dirtyA) { renderer.setRenderTarget(this.rtA); renderer.render(this.A.scene, this.cam); }
    renderer.setRenderTarget(this.rtB); renderer.render(this.B.scene, this.cam);
    renderer.setRenderTarget(prev);
    this.dirtyA = this.dirtyB = false;
    return true;
  }

  // the points of the first n strands (pass B), for tests
  readPoints(renderer: THREE.WebGLRenderer, n: number): Float32Array {
    return this.read(renderer, this.rtB, n * this.uP.value);
  }

  private read(renderer: THREE.WebGLRenderer, rt: THREE.WebGLRenderTarget, texels: number): Float32Array {
    const h = rows(texels), out = new Float32Array(TEX_W * h * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, TEX_W, h, out);
    return out.subarray(0, texels * 4);
  }

  // the passes at rest (no pose, the head centre of anny's default body) for the parity tests: pass A, the render roots
  // and pass B of the first n strands; the pose comes back with the next setPose
  readRest(renderer: THREE.WebGLRenderer, n: number) {
    const L = this.layout, P = this.uP.value, keep = this.mat.data.slice(), c = this.uHeadC.value.clone();
    this.mat.data.fill(0);
    for (let g = 0; g < L.G; g++) { this.mat.data[g * 12] = 1; this.mat.data[g * 12 + 5] = 1; this.mat.data[g * 12 + 10] = 1; }
    this.mat.tex.needsUpdate = true;
    this.uHeadC.value.fromArray(this.meta.centre);
    this.dirtyA = true; this.update(renderer);
    const out = { guides: Array.from(this.read(renderer, this.rtA!, L.G * (P + 2))), roots: Array.from(this.rootRest.data.subarray(0, n * 4)),
      points: Array.from(this.read(renderer, this.rtB, n * P)), P, G: L.G, scale: this.B.mat.uniforms.uScale.value, params: { ...this.params } };
    this.mat.data.set(keep); this.mat.tex.needsUpdate = true; this.uHeadC.value.copy(c);
    this.dirtyA = true; this.update(renderer);
    return out;
  }

  // GPU bytes of the hair: targets, textures and the template
  stats() {
    const tex = (t: THREE.DataTexture) => (t.image.data as any).byteLength;
    const rt = (r: THREE.WebGLRenderTarget | null) => r ? r.width * r.height * 16 : 0;
    const v = this.volume ? this.volume.data.byteLength : 0;
    const geo = this.geometry ? (this.geometry.index!.array as any).byteLength + (this.geometry.attributes.position.array as any).byteLength : 0;
    const bytes = rt(this.rtA) + rt(this.rtB) + tex(this.frames.tex) + tex(this.local.tex) + tex(this.tipLocal.tex) + tex(this.info.tex)
      + tex(this.mat.tex) + tex(this.motion.tex) + tex(this.rootRest.tex) + tex(this.rootData) + v + geo;
    return { gpu_bytes: bytes, strands: this.count, points: this.uP.value, guides: this.layout.G, style: this.style?.spec.name,
      available: this.style ? Array.from(available(this.style)).reduce((a, b) => a + b, 0) / this.layout.G : 0 };
  }
}
