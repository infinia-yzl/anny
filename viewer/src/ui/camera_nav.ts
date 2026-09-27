// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Navigation around the figure: orbit, pan (right-drag, Shift+drag or two fingers) within a box around the posed
// figure, a camera that stays above the floor, a height rail at the side of the view, a double-click that focuses on
// the nearest joint, the angles (front, three-quarter, side, back), keyboard steps, and the film offset that keeps the
// figure centred in the free space between the two inspectors.

import type { App } from './app.ts';
import { clamp, h } from './dom.ts';

const DEG = Math.PI / 180;
export const ANGLES: Record<string, number> = { front: 0, three: 24, side: 90, back: 180 };
// the ticks of the height rail: a label and the joints it averages
const TICKS: [string, string[]][] = [['Head', ['@head']], ['Chest', ['spine01']], ['Hips', ['root']], ['Knees', ['lowerleg01.L', 'lowerleg01.R']], ['Feet', ['foot.L', 'foot.R']]];

export interface CameraNav {
  before(): void;
  after(dt: number): void;
  orbit(deg: number): void;
  tilt(deg: number): void;
  move(frac: number): void;
  zoom(k: number): void;
  frame(name: string): void;
  angle(name: string): void;
  reset(): void;
  setInsets(left: number, right: number): void;
  rail: HTMLElement;
  onView: () => void;
}

export function cameraNav(app: App): CameraNav {
  const { THREE, camera, controls, canvas } = app;
  const V = app.view;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.panSpeed = 0.8;
  controls.minPolarAngle = 12 * DEG;
  controls.maxPolarAngle = 150 * DEG;
  const lastT = new THREE.Vector3().copy(controls.target);
  const off = new THREE.Vector3(), sph = new THREE.Spherical();
  let insetL = 0, insetR = 0, filmPx = 0, filmGoal = 0, lastAspect = 0;
  let railT = 0;
  // a pan or a zoom toward the pointer moves the pivot: movement between the user's first touch of the controls and
  // the next camera move of the page counts as the user's
  let userWindow = false;
  controls.addEventListener('start', () => { userWindow = true; });

  // ---------------------------------------------------------------- the height rail
  const thumb = h('button.hr-thumb', { type: 'button', 'aria-label': 'Camera height. Drag, or use the arrow keys.', role: 'slider', 'aria-orientation': 'vertical' });
  const ticks = h('div.hr-ticks');
  const track = h('div.hr-track', {}, h('div.hr-line'), ticks, thumb);
  const rail = h('div.hrail', { role: 'group', 'aria-label': 'Camera height' }, track);
  let range = { lo: 0, hi: 1 };
  const yOf = (clientY: number) => {
    const r = track.getBoundingClientRect();
    return range.lo + (1 - clamp((clientY - r.top) / r.height, 0, 1)) * (range.hi - range.lo);
  };
  const setHeight = (y: number) => {
    const dy = clamp(y, range.lo, range.hi) - controls.target.y;
    controls.target.y += dy; camera.position.y += dy;
    V.userMoved = true;
    controls.update(); lastT.copy(controls.target); app.resetAccum();
  };
  let dragging = false;
  thumb.addEventListener('pointerdown', (e) => { dragging = true; rail.classList.add('dragging'); thumb.setPointerCapture(e.pointerId); e.preventDefault(); });
  thumb.addEventListener('pointermove', (e) => { if (dragging) setHeight(yOf(e.clientY)); });
  const stop = () => { dragging = false; rail.classList.remove('dragging'); };
  thumb.addEventListener('pointerup', stop); thumb.addEventListener('pointercancel', stop);
  track.addEventListener('pointerdown', (e) => {
    if (e.target !== track && !(e.target as HTMLElement).classList.contains('hr-line')) return;
    flyHeight(yOf(e.clientY));
  });
  thumb.addEventListener('keydown', (e) => {
    const s = (range.hi - range.lo) * (e.shiftKey ? 0.1 : 0.03);
    if (e.key === 'ArrowUp') setHeight(controls.target.y + s);
    else if (e.key === 'ArrowDown') setHeight(controls.target.y - s);
    else return;
    e.preventDefault(); e.stopPropagation();
  });
  function flyHeight(y: number) {
    const t1 = controls.target.clone(); t1.y = clamp(y, range.lo, range.hi);
    const p1 = camera.position.clone().add(new THREE.Vector3(0, t1.y - controls.target.y, 0));
    V.userMoved = true;
    app.tweenTo(t1, p1, 600);
  }
  const tickEls = TICKS.map(([label]) => {
    const b = h('button.hr-tick', { type: 'button', 'data-tip': 'Look at the ' + label.toLowerCase(), 'data-tip-pos': 'left' }, h('span', { text: label }));
    ticks.append(b);
    return b;
  });
  const tickY = new Array(TICKS.length).fill(0);
  tickEls.forEach((b, i) => b.addEventListener('click', () => flyHeight(tickY[i])));
  function updateRail() {
    const f = app.figure();
    range = { lo: app.groundY(), hi: f.top + 0.12 };
    const byName = new Map(f.joints.map((j) => [j.name, j.p]));
    TICKS.forEach(([, names], i) => {
      let y = 0, n = 0;
      for (const nm of names) { const p = nm === '@head' ? f.head : byName.get(nm); if (p) { y += p.y; n++; } }
      tickY[i] = n ? y / n : range.lo;
      tickEls[i].style.top = (100 * (1 - (tickY[i] - range.lo) / (range.hi - range.lo))).toFixed(2) + '%';
    });
  }
  function paintThumb() {
    const t = clamp((controls.target.y - range.lo) / (range.hi - range.lo), 0, 1);
    thumb.style.top = (100 * (1 - t)).toFixed(2) + '%';
    thumb.setAttribute('aria-valuenow', controls.target.y.toFixed(2));
    thumb.setAttribute('aria-valuetext', `${Math.round((controls.target.y - range.lo) * 100 / 0.8916)} cm above the floor`);
  }

  // ---------------------------------------------------------------- double-click: focus on the nearest joint
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), tmp = new THREE.Vector3();
  canvas.addEventListener('dblclick', (e) => {
    const r = canvas.getBoundingClientRect();
    ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const f = app.figure();
    let best: any = null, bestD = 0.12 * Math.max(0.7, app.stature() / 1.7);
    const cands = f.joints.concat([{ name: '@head', p: f.head }]);
    for (const j of cands) {
      if (/^(toe|finger|metacarpal|eye)/.test(j.name)) continue;
      const d = ray.ray.distanceToPoint(j.p);
      if (d < bestD) { bestD = d; best = j; }
    }
    if (!best) { V.userMoved = false; app.flyTo(app.currentFrame()); return; }
    const t1 = best.p.clone();
    const dir = tmp.copy(camera.position).sub(controls.target).normalize();
    const d0 = camera.position.distanceTo(controls.target);
    const want = best.name === '@head' || best.name.startsWith('neck') ? 0.5 : /wrist|lowerarm02/.test(best.name) ? 0.55 : 0.85;
    const d1 = clamp(Math.min(d0, want * Math.max(0.7, app.stature() / 1.6)), controls.minDistance, controls.maxDistance);
    V.userMoved = true;
    app.tweenTo(t1, t1.clone().addScaledVector(dir, d1), 700);
    N.onView();
  });

  // ---------------------------------------------------------------- steps
  function around(dTheta: number, dPhi: number) {
    off.copy(camera.position).sub(controls.target);
    sph.setFromVector3(off);
    sph.theta += dTheta; sph.phi = clamp(sph.phi + dPhi, controls.minPolarAngle, controls.maxPolarAngle);
    off.setFromSpherical(sph);
    camera.position.copy(controls.target).add(off);
    camera.lookAt(controls.target);
    controls.update(); app.resetAccum();
  }
  const N: CameraNav = {
    rail,
    onView: () => {},
    before() {
      // the camera stays 3 cm above the floor: the lowest polar angle for this target and distance
      const d = Math.max(1e-3, camera.position.distanceTo(controls.target));
      const c = clamp((app.groundY() + 0.03 - controls.target.y) / d, -1, 1);
      controls.maxPolarAngle = Math.max(controls.minPolarAngle + 0.01, Math.min(150 * DEG, Math.acos(c)));
      lastT.copy(controls.target);
    },
    after(dt: number) {
      // a pan or a zoom toward the cursor moved the pivot: the view is the user's now
      if (app.tweening()) userWindow = false;
      else {
        if (userWindow && controls.target.distanceToSquared(lastT) > 1e-10) { if (!V.userMoved) { V.userMoved = true; N.onView(); } }
        // the pivot stays in a box around the posed figure
        const f = app.figure(), t = controls.target;
        const x = clamp(t.x, f.lo.x - 0.35, f.hi.x + 0.35), y = clamp(t.y, app.groundY(), f.top + 0.15), z = clamp(t.z, f.lo.z - 0.35, f.hi.z + 0.35);
        if (x !== t.x || y !== t.y || z !== t.z) {
          const dx = x - t.x, dy = y - t.y, dz = z - t.z;
          t.set(x, y, z); camera.position.x += dx; camera.position.y += dy; camera.position.z += dz;
          controls.update();
        }
      }
      // a hard limit as well: the inertia of a pan can carry the pivot after the polar limit of this frame
      const lim = app.groundY() + 0.03;
      if (camera.position.y < lim) {
        off.copy(camera.position).sub(controls.target);
        sph.setFromVector3(off);
        sph.phi = Math.min(sph.phi, Math.acos(clamp((lim - controls.target.y) / Math.max(1e-3, sph.radius), -1, 1)));
        off.setFromSpherical(sph);
        camera.position.copy(controls.target).add(off);
        camera.lookAt(controls.target);
      }
      // the film offset eases toward the centre of the free space
      const aspect = camera.aspect;
      if (Math.abs(filmGoal - filmPx) > 0.25 || aspect !== lastAspect) {
        filmPx += (filmGoal - filmPx) * Math.min(1, dt * 10);
        if (Math.abs(filmGoal - filmPx) <= 0.25) filmPx = filmGoal;
        lastAspect = aspect;
        const fw = camera.getFilmWidth();
        camera.filmOffset = -(filmPx / innerWidth) * 2 * Math.tan(camera.fov * DEG / 2) * aspect * fw;
        camera.updateProjectionMatrix();
        app.resetAccum();
      }
      railT += dt;
      if (railT > 0.1) { railT = 0; updateRail(); }
      paintThumb();
    },
    orbit(deg) { around(deg * DEG, 0); },
    tilt(deg) { around(0, -deg * DEG); },
    move(frac) {
      const f = app.figure();
      const dy = frac * (f.top - app.groundY());
      const y = clamp(controls.target.y + dy, app.groundY(), f.top + 0.15);
      camera.position.y += y - controls.target.y; controls.target.y = y;
      V.userMoved = true; controls.update(); lastT.copy(controls.target); app.resetAccum();
      N.onView();
    },
    zoom(k) {
      off.copy(camera.position).sub(controls.target);
      const d = clamp(off.length() * k, controls.minDistance, controls.maxDistance);
      camera.position.copy(controls.target).addScaledVector(off.normalize(), d);
      controls.update(); app.resetAccum();
    },
    frame(name) { V.userMoved = false; app.flyTo(name); N.onView(); },
    angle(name) {
      V.yaw = ANGLES[name] ?? 24;
      if (!V.userMoved) { app.flyTo(app.currentFrame()); N.onView(); return; }
      // a view of the user's: turn about the pivot at the same distance and height
      off.copy(camera.position).sub(controls.target);
      sph.setFromVector3(off);
      sph.theta = V.yaw * DEG;
      off.setFromSpherical(sph);
      app.tweenTo(controls.target.clone(), controls.target.clone().add(off), 700);
      N.onView();
    },
    reset() { V.yaw = ANGLES.three; V.userMoved = false; app.flyTo('body'); N.onView(); },
    setInsets(left, right) {
      insetL = left; insetR = right;
      V.freeW = Math.max(200, innerWidth - left - right);
      filmGoal = app.shot ? 0 : (left - right) / 2;
      document.documentElement.style.setProperty('--inset-l', left + 'px');
      document.documentElement.style.setProperty('--inset-r', right + 'px');
    },
  };
  updateRail();
  return N;
}
