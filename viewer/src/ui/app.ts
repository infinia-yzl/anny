// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// What the interface needs from the renderer (main.ts): the look and its edits, the body, the hair, the motion, the
// lights, the camera and the frame counters. main.ts builds this object once and hands it to initUI.

import type * as THREE from 'three';

export interface Look {
  format: string;
  name: string;
  skin: { tone: number; undertone: number };
  hair: { color: string; style: string; length?: number; curl?: number; volume?: number; density?: number; fade?: number; part?: string };
  eyes: { color: string };
  phenotype: Record<string, number>;
  face: Record<string, number>;
}

export interface BodySlider { name: string; label: string; race: boolean; ends: string[] }
export interface FaceSlider { name: string; group: string; label: string; range: [number, number]; ends: string[] }
export interface Clip { name: string; label: string; kind: 'pose' | 'loop'; group?: string; duration: number; credit?: string }
export interface Preset { label: string; exposure: number; key: { az: number; el: number }; bg: number[][] }

export interface App {
  THREE: typeof THREE;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: any;
  canvas: HTMLCanvasElement;
  shot: boolean;
  small: boolean;
  qs: URLSearchParams;

  // the look
  look(): Look;
  applyLook(look: any, remember?: boolean): void;
  editLook(mut: (n: Look) => void): void;
  normaliseLook(o: any): Look;
  baseline(): Look;
  characters: Look[];
  looks: Look[];
  palettes: { hair: [string, string][]; eyes: [string, string][] };
  skinBase(tone: number, under: number): number[];
  sameLook(a: any, b: any): boolean;
  sameCharacter(a: any, b: any): boolean;
  baselinePhenotype(): Record<string, number>;

  // the body and the face
  bodyReady(): boolean;
  bodySliders(): BodySlider[];
  faceSliders(): FaceSlider[];
  faceGroups(): string[];
  faceGroupTitle(g: string): string;
  facePrior(): boolean;
  sampleFace(phenotype: Record<string, number>, spread: number): Record<string, number>;
  raceShare(phenotype: Record<string, number>, name: string): number;
  stature(): number;                 // metres, of the current body standing
  bodyTiming(): { ms: number; total: number; steps: Record<string, number> };

  // the hair
  hairSpecs(): any[];
  hairDefaults(style: string): any;
  fadeRange: number[];
  hairVisible(): boolean;
  setHairVisible(on: boolean): void;
  hairPhysics(): boolean;
  setHairPhysics(on: boolean): void;
  hairStats(): any;

  // the motion and the soft tissue
  clips(): Clip[];
  motion(): { cur: Clip | null; t: number; playing: boolean; speed: number };
  setMotion(name: string): void;
  togglePlay(): void;
  setSpeed(s: number): void;
  seek(t: number): void;
  correctives(): { ready: boolean; on: boolean };
  setCorrectives(on: boolean): void;

  // the lights and the picture
  presets: Record<string, Preset>;
  preset(): string;
  applyPreset(name: string): void;
  lightRotation(): number;
  setLightRotation(deg: number, rebuild: boolean): void;
  exposureEV(): number;
  setExposureEV(ev: number): void;
  quality(): string;
  setQuality(q: string): void;
  turntable(): boolean;
  setTurntable(on: boolean): void;
  // the view of the rig ('off', 'skeleton', or 'weights': the body in the colours of its skinning weights, with the
  // skeleton), the chosen bone (kept: chosen by a click), and the name of the bone under a point of the canvas
  rigView(): string;
  boneNames(): string[];
  setRigView(mode: string): void;
  rigBone(): string | null;
  rigBoneKept(): boolean;
  setRigBone(name: string | null, kept?: boolean): void;
  pickBone(x: number, y: number): string | null;
  weightStats(): { vertices: number; area: number; bones: { name: string; touched: number; area: number }[] } | null;   // area in m²
  // the frame rate limit (0: the display's rate)
  fpsCaps: number[];
  fpsCap(): number;
  setFpsCap(cap: number): void;

  // the camera
  groundY(): number;
  figure(): { lo: THREE.Vector3; hi: THREE.Vector3; top: number; head: THREE.Vector3; joints: { name: string; p: THREE.Vector3 }[] };
  // the framings ('body', 'upper', 'face') at the view's yaw, and the state that the navigation shares with main.ts
  flyTo(name: string): void;
  currentFrame(): string;
  view: { yaw: number; freeW: number; userMoved: boolean };
  tweenTo(target: THREE.Vector3, position: THREE.Vector3, ms?: number): void;
  tweening(): boolean;
  resetAccum(): void;

  // the picture and its counters
  renderPass(): void;
  samples(): { n: number; max: number };
  scale(): { s: number; w: number; h: number };
  pixelRatio(): number;
  shadowSize(): number;
  sssMode(): string;
  loadTimes(): [string, number][];
}

// what main.ts calls back into the interface
export interface Hooks {
  look(): void;
  motion(): void;
  hair(): void;
  frame(now: number, dt: number, drew: boolean, jsMs: number): void;
  passBegin(): void;
  passEnd(): void;
  interacted(): void;
}
