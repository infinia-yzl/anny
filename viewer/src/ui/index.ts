// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The interface of the viewer: the top bar (identity and quick actions), the Character and Stage segments, the view
// bar, the height rail, the performance card, the help sheet, the first-run hint, the name of the bone under the
// pointer while the skeleton shows, and the keyboard shortcuts.
// main.ts calls initUI once the model is ready, and calls the returned hooks when the look, the motion or a frame
// changes.

import type { App, Hooks } from './app.ts';
import { cameraNav, type CameraNav } from './camera_nav.ts';
import { characterSections, randomBodyInto, randomColours, randomFaceInto, randomHair, randomPhenotype, type Editor } from './character_ui.ts';
import { segmented } from './controls.ts';
import { $, download, h, iconButton, saveUi, slug, typing, uiState } from './dom.ts';
import { lookHistory } from './history.ts';
import { icon } from './icons.ts';
import { isPhone, makeSegment, WIDE, type Segment } from './segments.ts';
import { helpSheet } from './shortcuts.ts';
import { stageSections } from './stage_ui.ts';
import { makeStats, type Stats } from './stats.ts';
import { toast } from './toast.ts';

export interface UI { hooks: Hooks; nav: CameraNav; stats: Stats }

// a menu that opens under a button
function menu(button: HTMLButtonElement, items: { label: string; icon?: string; on: () => void; hidden?: () => boolean }[], align: 'left' | 'right' = 'right') {
  const box = h('div.menu', { role: 'menu', hidden: true, dataset: { align } });
  const els = items.map((it) => {
    const b = h('button', { type: 'button', role: 'menuitem' });
    if (it.icon) b.append(icon(it.icon, 18));
    b.append(h('span', { text: it.label }));
    b.addEventListener('click', () => { hide(); it.on(); });
    box.append(b);
    return b;
  });
  button.setAttribute('aria-haspopup', 'menu'); button.setAttribute('aria-expanded', 'false');
  const hide = () => { box.hidden = true; button.setAttribute('aria-expanded', 'false'); };
  const show = () => {
    items.forEach((it, i) => { els[i].hidden = !!it.hidden?.(); });
    box.hidden = false; button.setAttribute('aria-expanded', 'true');
    const r = button.getBoundingClientRect();
    box.style.top = r.bottom + 6 + 'px';
    if (align === 'right') { box.style.right = Math.max(8, innerWidth - r.right) + 'px'; box.style.left = 'auto'; } else { box.style.left = r.left + 'px'; box.style.right = 'auto'; }
    (els.find((e) => !e.hidden) as HTMLElement)?.focus();
  };
  button.addEventListener('click', (e) => { e.stopPropagation(); if (box.hidden) show(); else hide(); });
  box.addEventListener('keydown', (e) => {
    const vis = els.filter((x) => !x.hidden), i = vis.indexOf(document.activeElement as any);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); e.stopPropagation(); vis[(i + (e.key === 'ArrowDown' ? 1 : vis.length - 1)) % vis.length].focus(); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); hide(); button.focus(); }
  });
  document.addEventListener('click', (e) => { if (!box.hidden && !box.contains(e.target as Node)) hide(); });
  document.body.append(box);
  return { box, hide };
}

export function initUI(app: App): UI {
  document.body.classList.toggle('shot', app.shot);
  const nav = cameraNav(app);
  const stats = makeStats(app);

  // ---------------------------------------------------------------- the edits and their history
  let pending: ((n: any) => void)[] = [];
  const hist = lookHistory(() => app.look(), (l) => app.applyLook(l));
  const ed: Editor = {
    app, origin: app.look().name, faceSpread: 1,
    begin: (label) => hist.begin(label),
    end: () => hist.end(),
    edit(label, mut) { hist.begin(label); app.editLook(mut); hist.end(); },
    commit(label, look) { hist.begin(label); app.applyLook(look); hist.end(); },
    queue(mut) {
      if (!pending.length) requestAnimationFrame(() => { const ms = pending; pending = []; app.editLook((n) => { for (const m of ms) m(n); }); });
      pending.push(mut);
    },
  };
  const undo = () => { const l = hist.undo(); if (l) toast('Undo: ' + l); };
  const redo = () => { const l = hist.redo(); if (l) toast('Redo: ' + l); };
  const randomAll = () => {
    const next = JSON.parse(JSON.stringify(app.look()));
    next.phenotype = randomPhenotype(app);
    if (app.facePrior()) next.face = app.sampleFace(next.phenotype, ed.faceSpread);
    randomHair(app, next); randomColours(app, next); next.name = 'Custom';
    ed.commit('Random character', next); ed.origin = 'Random character';
  };

  // ---------------------------------------------------------------- the segments
  const cs = characterSections(ed);
  const ss = stageSections(app);
  let prevL = false, prevR = false;
  const layout = () => {
    const lo = left.isOpen(), ro = right.isOpen();
    // on a narrower screen the two inspectors take turns
    if (lo && ro && innerWidth < WIDE) { if (!prevL) right.close(); else left.close(); return; }
    prevL = left.isOpen(); prevR = right.isOpen();
    if (isPhone() || app.shot) { nav.setInsets(0, 0); document.body.classList.toggle('sheet-open', prevL || prevR); document.body.classList.remove('vb-compact', 'vb-fold'); return; }
    document.body.classList.remove('sheet-open');
    // the free space lies between the right edge of the left segment and the left edge of the right one
    const r = (el: HTMLElement) => el.hidden ? null : el.getBoundingClientRect();
    const lr = [r(left.rail), r(left.panel)].filter(Boolean) as DOMRect[], rr = [r(right.rail), r(right.panel)].filter(Boolean) as DOMRect[];
    const l = lr.length ? Math.max(...lr.map((x) => x.right)) : 0, rt = rr.length ? innerWidth - Math.min(...rr.map((x) => x.left)) : 0;
    nav.setInsets(l, rt);
    // the view bar folds its labels, then itself, as the free space narrows
    const free = innerWidth - l - rt;
    document.body.classList.toggle('vb-compact', free < 820);
    document.body.classList.toggle('vb-fold', free < 560);
  };
  const left: Segment = makeSegment('left', 'Character', [
    { id: 'characters', label: 'Characters', icon: 'characters', key: 'c', el: cs.chars },
    { id: 'body', label: 'Body', icon: 'body', key: 'b', el: cs.body },
    { id: 'face', label: 'Face', icon: 'face', key: 'f', el: cs.face },
    { id: 'hair', label: 'Hair', icon: 'hair', key: 'h', el: cs.hair },
    { id: 'skin', label: 'Skin & eyes', icon: 'skin', key: 'e', el: cs.skin },
  ], layout);
  const right: Segment = makeSegment('right', 'Stage', [
    { id: 'pose', label: 'Pose', icon: 'pose', key: 'p', el: ss.pose },
    { id: 'scene', label: 'Scene', icon: 'scene', key: 'l', el: ss.scene },
  ], layout);
  if (!app.clips().length) { right.sections[0].el.remove(); (right.rail.querySelector('#tab-pose') as HTMLElement).hidden = true; }

  // ---------------------------------------------------------------- the top bar
  const identity = h('button.identity', { type: 'button', 'data-tip': 'Characters  ·  C', 'data-tip-pos': 'down' });
  identity.addEventListener('click', () => left.open('characters'));
  const undoB = iconButton('undo', 'Undo', { key: 'Ctrl+Z', on: undo });
  const redoB = iconButton('redo', 'Redo', { key: 'Ctrl+Shift+Z', on: redo, cls: 'wide-only' });
  const diceB = iconButton('dice', 'Randomise', { cls: 'wide-only' }) as HTMLButtonElement;
  menu(diceB, [
    { label: 'Random character', icon: 'characters', on: randomAll },
    { label: 'Random body', icon: 'body', on: () => { randomBodyInto(app, ed, app.look()); ed.origin = 'Random body'; } },
    { label: 'Random face', icon: 'face', on: () => randomFaceInto(app, ed), hidden: () => !app.facePrior() },
    { label: 'Random hair', icon: 'hair', on: () => ed.edit('Random hair', (n) => randomHair(app, n)) },
    { label: 'Random colours', icon: 'skin', on: () => ed.edit('Random colours', (n) => randomColours(app, n)) },
  ]);
  // a hosted page (the artifact build, with its data in parts) cannot start a download, so it shows the picture to
  // save by hand; the single-file page downloads it
  const hosted = !!(window as any).MODEL_PARTS;
  const shotDlg = h('dialog.help.shot-dlg', { 'aria-label': 'Saved picture' }) as HTMLDialogElement;
  shotDlg.addEventListener('click', (e) => { if (e.target === shotDlg) shotDlg.close(); });
  const saveImage = () => {
    app.renderPass();
    app.canvas.toBlob((b) => {
      if (!b) { toast('The picture could not be made', 'error'); return; }
      const name = `anny-${slug(app.look().name)}.png`;
      if (!hosted) { download(name, b); toast('Image saved'); return; }
      const url = URL.createObjectURL(b);
      const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left', on: () => shotDlg.close() });
      shotDlg.replaceChildren(h('header.help-head', {}, h('h2', { text: 'Picture' }), close),
        h('p.help-lead', { text: 'Right-click the picture and choose Save image, or press and hold it on a phone.' }),
        h('img.shot-img', { src: url, alt: 'The current view of ' + app.look().name }));
      shotDlg.addEventListener('close', () => URL.revokeObjectURL(url), { once: true });
      shotDlg.showModal();
    }, 'image/png');
  };
  const saveB = iconButton('save', 'Save image', { on: saveImage, cls: 'wide-only' });
  const statsB = iconButton('stats', 'Performance', { key: '`', on: () => stats.toggle() });
  const fullB = iconButton('fullscreen', 'Full screen', { cls: 'wide-only', on: () => {
    if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => toast('Full screen is not allowed here', 'error'));
  } });
  fullB.hidden = !document.fullscreenEnabled;
  const help = helpSheet();
  const helpB = iconButton('help', 'Help and shortcuts', { key: '?', cls: 'wide-only', on: () => help.showModal() });
  const moreB = iconButton('more', 'More', { cls: 'phone-only' }) as HTMLButtonElement;
  menu(moreB, [
    { label: 'Redo', icon: 'redo', on: redo },
    { label: 'Random character', icon: 'dice', on: randomAll },
    { label: 'Save image', icon: 'save', on: saveImage },
    { label: 'Full screen', icon: 'fullscreen', on: () => fullB.click(), hidden: () => !document.fullscreenEnabled },
    { label: 'Help and shortcuts', icon: 'help', on: () => help.showModal() },
  ]);
  const top = h('header.topbar', {},
    h('div.brand', {}, h('span.brand-mark', { text: 'anny' }), h('span.brand-name', { text: 'Viewer' })), identity,
    h('div.actions', {}, undoB, redoB, h('span.sep'), diceB, saveB, statsB, fullB, helpB, moreB));
  statsB.setAttribute('aria-pressed', 'false');
  stats.onToggle = (on) => statsB.setAttribute('aria-pressed', String(on));
  hist.onChange = () => {
    undoB.disabled = !hist.canUndo(); redoB.disabled = !hist.canRedo();
    undoB.dataset.tip = hist.canUndo() ? `Undo ${hist.nextUndo()}  ·  Ctrl+Z` : 'Nothing to undo';
    redoB.dataset.tip = hist.canRedo() ? `Redo ${hist.nextRedo()}  ·  Ctrl+Shift+Z` : 'Nothing to redo';
  };
  hist.onChange();

  // ---------------------------------------------------------------- the view bar
  const framing = segmented('Framing', [{ value: 'body', label: 'Full body', tip: 'Full body  ·  1' }, { value: 'upper', label: 'Upper body', tip: 'Upper body  ·  2' }, { value: 'face', label: 'Face', tip: 'Face  ·  3' }], (v) => nav.frame(v), 'seg-view');
  const angles = segmented('Angle', [
    { value: 'front', label: 'Front', short: 'Front', icon: 'angle_front', tip: 'Front  ·  4' }, { value: 'three', label: 'Three-quarter', short: '¾', icon: 'angle_three', tip: 'Three-quarter  ·  5' },
    { value: 'side', label: 'Side', short: 'Side', icon: 'angle_side', tip: 'Side  ·  6' }, { value: 'back', label: 'Back', short: 'Back', icon: 'angle_back', tip: 'Back  ·  7' }], (v) => nav.angle(v), 'seg-icons');
  const turnB = iconButton('turntable', 'Turntable', { key: 'T', tip: 'up', on: () => app.setTurntable(!app.turntable()) });
  const bonesB = iconButton('bones', 'Skeleton', { key: 'Shift+B', tip: 'up', on: () => app.setSkeleton(!app.skeleton()) });
  const resetB = iconButton('reset', 'Reset view', { key: '0', tip: 'up', on: () => nav.reset() });
  const ns = 'http://www.w3.org/2000/svg';
  const ring = document.createElementNS(ns, 'svg'); ring.setAttribute('viewBox', '0 0 24 24'); ring.setAttribute('class', 'ring'); ring.setAttribute('aria-hidden', 'true');
  ring.innerHTML = '<circle cx="12" cy="12" r="8.5" class="ring-bg"/><circle cx="12" cy="12" r="8.5" class="ring-fg" pathLength="100" stroke-dasharray="0 100" transform="rotate(-90 12 12)"/>';
  const ringLabel = h('span.ring-label');
  const ringB = h('button.ringb', { type: 'button', 'data-tip': 'Refinement · open Performance', 'data-tip-pos': 'up', 'aria-live': 'polite' }, ring, ringLabel);
  ringB.addEventListener('click', () => stats.toggle());
  const vbItems = h('div.vb-items', {}, framing.el, h('span.sep'), angles.el, h('span.sep'), turnB, bonesB, resetB, ringB);
  const vbToggle = iconButton('camera', 'View', { cls: 'vb-toggle', tip: 'up' });
  const viewbar = h('div.viewbar', { role: 'toolbar', 'aria-label': 'View' }, vbToggle, vbItems);
  vbToggle.addEventListener('click', () => { viewbar.classList.toggle('open'); document.body.classList.toggle('vb-open', viewbar.classList.contains('open')); vbToggle.setAttribute('aria-expanded', String(viewbar.classList.contains('open'))); });
  vbToggle.setAttribute('aria-expanded', 'false');
  const paintView = () => {
    framing.set(app.view.userMoved ? '' : app.currentFrame());
    const a = Object.entries({ front: 0, three: 24, side: 90, back: 180 }).find(([, y]) => y === app.view.yaw);
    angles.set(a ? a[0] : '');
    turnB.setAttribute('aria-pressed', String(app.turntable()));
    bonesB.setAttribute('aria-pressed', String(app.skeleton()));
  };
  nav.onView = paintView;
  let lastRing = '';
  const paintRing = () => {
    const s = app.samples(), m = app.motion();
    const moving = !!m.cur && m.cur.kind === 'loop' && m.playing;
    const done = s.n >= s.max;
    const txt = moving ? 'Playing' : app.turntable() ? 'Turntable' : done ? 'Refined' : `${Math.round(s.n / s.max * 100)} %`;
    const key = txt + (done ? 1 : 0);
    if (key === lastRing) return;
    lastRing = key;
    ringLabel.textContent = txt;
    (ring.querySelector('.ring-fg') as SVGElement).setAttribute('stroke-dasharray', `${moving || app.turntable() ? 100 : Math.min(100, s.n / s.max * 100)} 100`);
    ringB.dataset.state = moving || app.turntable() ? 'live' : done ? 'done' : 'busy';
    ringB.setAttribute('aria-label', `Picture ${txt.toLowerCase()}. Open the performance card.`);
  };

  // ---------------------------------------------------------------- the hint for first-time visitors
  let coach: HTMLElement | null = null;
  if (!uiState().coach && !app.shot) {
    const ok = h('button.tb', { type: 'button', text: 'Got it' });
    coach = h('div.coach', { role: 'note' },
      h('p', {}, h('b', { text: 'Drag' }), ' to orbit · ', h('b', { text: 'right-drag' }), ' or two fingers to move · ', h('b', { text: 'scroll' }), ' to zoom · ', h('b', { text: 'double-click' }), ' to focus · ', h('b', { text: '?' }), ' for shortcuts'), ok);
    ok.addEventListener('click', () => dismissCoach());
  }
  function dismissCoach() { if (!coach) return; coach.classList.add('out'); const c = coach; coach = null; setTimeout(() => c.remove(), 300); saveUi({ coach: 1 }); }

  // ---------------------------------------------------------------- the name of the bone under the pointer
  const boneTip = h('div.bone-tip', { 'aria-hidden': 'true', hidden: true });
  let tipAt: { x: number; y: number } | null = null, tipQueued = false;
  const paintBoneTip = () => {
    tipQueued = false;
    const name = tipAt && app.skeleton() ? app.pickBone(tipAt.x, tipAt.y) : null;
    boneTip.hidden = !name;
    if (!name || !tipAt) return;
    boneTip.textContent = name;
    boneTip.style.left = tipAt.x + 'px'; boneTip.style.top = tipAt.y + 'px';
  };
  app.canvas.addEventListener('pointermove', (e) => {
    // a drag orbits the camera: the name waits until the pointer rests on a bone again
    if (e.buttons || e.pointerType === 'touch') { tipAt = null; boneTip.hidden = true; return; }
    tipAt = { x: e.clientX, y: e.clientY };
    if (!tipQueued && app.skeleton()) { tipQueued = true; requestAnimationFrame(paintBoneTip); }
  });
  app.canvas.addEventListener('pointerleave', () => { tipAt = null; boneTip.hidden = true; });

  document.body.append(top, left.rail, left.panel, right.rail, right.panel, nav.rail, viewbar, stats.el, help, shotDlg, boneTip);
  if (coach) document.body.append(coach);

  // ---------------------------------------------------------------- the keyboard
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && !e.altKey && (e.key === 'z' || e.key === 'Z' || e.key === 'y')) {
      if (typing(e)) return;
      e.preventDefault();
      if (e.key === 'y' || e.shiftKey) redo(); else undo();
      return;
    }
    if (typing(e) || mod) return;
    if (help.open) return;
    if (e.altKey) {
      const m = /^Digit([1-4])$/.exec(e.code);
      if (m) { const names = Object.keys(app.presets); const n = names[+m[1] - 1]; if (n) { e.preventDefault(); app.applyPreset(n); } }
      return;
    }
    const k = e.key;
    const act: Record<string, () => void> = {
      ArrowLeft: () => nav.orbit(-8), ArrowRight: () => nav.orbit(8),
      ArrowUp: () => e.shiftKey ? nav.tilt(5) : nav.move(0.05), ArrowDown: () => e.shiftKey ? nav.tilt(-5) : nav.move(-0.05),
      '+': () => nav.zoom(0.88), '=': () => nav.zoom(0.88), '-': () => nav.zoom(1.14), '_': () => nav.zoom(1.14),
      '0': () => nav.reset(), '1': () => nav.frame('body'), '2': () => nav.frame('upper'), '3': () => nav.frame('face'),
      '4': () => nav.angle('front'), '5': () => nav.angle('three'), '6': () => nav.angle('side'), '7': () => nav.angle('back'),
      c: () => left.toggle('characters'), b: () => left.toggle('body'), f: () => left.toggle('face'), h: () => left.toggle('hair'), e: () => left.toggle('skin'),
      p: () => right.toggle('pose'), l: () => right.toggle('scene'),
      H: () => app.setHairVisible(!app.hairVisible()), t: () => app.setTurntable(!app.turntable()),
      B: () => app.setSkeleton(!app.skeleton()),
      '`': () => stats.toggle(), '?': () => help.showModal(),
      ' ': () => { const m = app.motion(); if (m.cur && m.cur.kind === 'loop') app.togglePlay(); },
      Escape: () => {
        const a = document.activeElement;
        if (right.panel.contains(a)) right.close(true);
        else if (left.panel.contains(a)) left.close(true);
        else if (right.isOpen()) right.close();
        else if (left.isOpen()) left.close();
        else if (stats.isOpen()) stats.close();
      },
    };
    // a focused button keeps Space and Enter; the arrows stay with the camera only outside the controls that use them
    if (k === ' ' && (e.target as HTMLElement).closest('button, summary')) return;
    if (k.startsWith('Arrow') && (e.target as HTMLElement).closest('input, .seg, .rail-list, .menu')) return;
    const fn = act[k] ?? (k.length === 1 ? act[k.toLowerCase()] : undefined);
    if (!fn) return;
    e.preventDefault();
    fn();
  });

  // ---------------------------------------------------------------- the start state
  const st = uiState();
  if (!app.shot) {
    if (st.left?.open ?? !isPhone()) left.open(st.left?.section);
    if (st.right?.open && innerWidth >= WIDE) right.open(st.right.section);
    if (st.stats || app.qs.get('stats') === '1') stats.open();
  }
  addEventListener('resize', () => layout());
  layout();
  const syncLook = () => {
    const L = app.look();
    cs.sync(L);
    let name = L.name;
    const named = app.characters.concat(app.looks).some((c) => c.name === ed.origin);
    if (L.name === 'Custom' && named && ed.origin !== 'Baseline') name = ed.origin + ' · edited';
    identity.replaceChildren(h('span.identity-dot', { style: { '--a': `rgb(${app.skinBase(L.skin.tone, L.skin.undertone).map((v) => Math.round(v * 255)).join(',')})`, '--b': L.hair.color } }),
      h('span', { text: name }));
    if (L.name !== 'Custom') ed.origin = L.name;
  };
  syncLook(); ss.sync(); paintView(); paintRing();
  let tFrame = 0;
  const hooks: Hooks = {
    look: syncLook,
    motion: () => { ss.sync(); if (!app.skeleton()) boneTip.hidden = true; },
    hair: () => { cs.sync(app.look()); ss.sync(); },
    frame(now, dt, drew, jsMs) {
      stats.frame(now, dt, drew, jsMs);
      if (now - tFrame > 100) { tFrame = now; ss.frame(); paintRing(); paintView(); }
    },
    passBegin: () => stats.passBegin(),
    passEnd: () => stats.passEnd(),
    interacted: () => { if (coach) setTimeout(dismissCoach, 1500); },
  };
  return { hooks, nav, stats };
}
