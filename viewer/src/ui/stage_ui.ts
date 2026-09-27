// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The sections of the Stage segment: Pose (the clips with a timeline, the poses, the soft tissue) and Scene (the
// lights, their rotation, the exposure, the quality of the picture, the frame rate limit and the display toggles,
// the skeleton among them).

import type { App } from './app.ts';
import { chip, foldGroup, searchField, sectionHead, segmented, slider, toggle } from './controls.ts';
import { h, saveUi, uiState } from './dom.ts';
import { icon } from './icons.ts';

const QUALITY: Record<string, string> = {
  auto: 'Draws at a lower resolution while the figure moves, then refines the still picture at full resolution.',
  sharp: 'Always draws at full resolution with every hair strand. Moving frames can be slower.',
  fast: 'Draws fewer pixels, fewer samples and fewer strands, for slower devices and longer battery life.',
};

export function stageSections(app: App) {
  const syncs: (() => void)[] = [];
  const frames: (() => void)[] = [];

  // ---------------------------------------------------------------- Pose
  const pose = h('section');
  {
    const head = sectionHead('Pose');
    const clips = app.clips();
    const loops = clips.filter((c) => c.kind === 'loop');
    const anim = h('div.chips', { role: 'group', 'aria-label': 'Animations' });
    const chips = new Map<string, HTMLButtonElement>();
    const choose = (name: string) => {
      const m = app.motion();
      const c = clips.find((x) => x.name === name)!;
      if (c.kind === 'loop' && m.cur && m.cur.name === name) app.togglePlay(); else app.setMotion(name);
    };
    for (const c of loops) { const b = chip(c.label, () => choose(c.name)); b.dataset.motion = c.name; anim.append(b); chips.set(c.name, b); }
    const play = h('button.play', { type: 'button', 'aria-label': 'Play', 'data-tip': 'Play or pause  ·  Space', 'data-tip-pos': 'up' }, icon('play', 18));
    play.addEventListener('click', () => app.togglePlay());
    let scrubbing = false;
    const scrub = h('input.scrub', { type: 'range', min: '0', max: '1', step: '0.001', value: '0', 'aria-label': 'Animation time' }) as HTMLInputElement;
    scrub.addEventListener('input', () => {
      scrubbing = true;
      const m = app.motion(); if (!m.cur || m.cur.kind !== 'loop') return;
      if (m.playing) app.togglePlay();
      app.seek(parseFloat(scrub.value) * m.cur.duration);
    });
    scrub.addEventListener('change', () => { scrubbing = false; });
    const time = h('span.scrub-time');
    const speed = segmented('Speed', [{ value: '0.25', label: '¼×' }, { value: '0.5', label: '½×' }, { value: '1', label: '1×' }], (v) => app.setSpeed(parseFloat(v)));
    const timeline = h('div.timeline', {}, play, scrub, time);
    // the poses: groups that fold, a search over their names
    const folds = new Map<string, ReturnType<typeof foldGroup>>();
    const wrap = h('div.folds');
    for (const c of clips) {
      if (c.kind !== 'pose') continue;
      let f = folds.get(c.group!);
      if (!f) { f = foldGroup(c.group!); f.body.classList.add('chips'); folds.set(c.group!, f); wrap.append(f.el); }
      const b = chip(c.label, () => choose(c.name)); b.dataset.motion = c.name; f.body.append(b); chips.set(c.name, b);
      f.count.textContent = String(f.body.children.length);
    }
    const empty = h('p.note.empty', { text: 'No pose matches.', hidden: true });
    const search = searchField('Search 50 poses', (q) => {
      let any = false;
      for (const [g, f] of folds) {
        let n = 0;
        for (const b of Array.from(f.body.children) as HTMLElement[]) { const hit = !q || b.textContent!.toLowerCase().includes(q) || g.toLowerCase().includes(q); b.hidden = !hit; if (hit) n++; }
        f.el.hidden = n === 0; if (q) f.el.open = n > 0; any = any || n > 0;
      }
      empty.hidden = any;
    });
    const credit = h('p.note.credit');
    const tissue = toggle('Soft tissue', (on) => app.setCorrectives(on), 'Corrective shapes keep the volume of the joints');
    const tissueNote = h('p.note', { text: 'Corrective shapes from a soft-tissue simulation keep the volume of the shoulders, elbows, hips and knees as they bend. Switch them off to compare with plain skinning.' });
    pose.append(head.el, h('h3.sub', { text: 'Animations' }), anim, timeline, h('div.row', {}, h('span.row-label', { text: 'Speed' }), speed.el),
      h('p.note', { text: 'The view sharpens while the figure holds still. Pause an animation to see any moment at full quality.' }),
      h('h3.sub', { text: 'Poses' }), search.el, wrap, empty, credit,
      h('h3.sub', { text: 'Soft tissue' }), tissue.el, tissueNote);
    let openFor = '';
    syncs.push(() => {
      const m = app.motion(); if (!m.cur) return;
      for (const [n, b] of chips) b.setAttribute('aria-pressed', String(n === m.cur.name));
      const loop = m.cur.kind === 'loop';
      timeline.toggleAttribute('data-off', !loop);
      play.disabled = !loop; scrub.disabled = !loop;
      play.replaceChildren(icon(loop && m.playing ? 'pause' : 'play', 18));
      play.setAttribute('aria-label', loop && m.playing ? 'Pause' : 'Play');
      speed.set(String(m.speed));
      head.note.textContent = m.cur.label + (loop && !m.playing ? ' · paused' : '');
      if (!loop && m.cur.group && openFor !== m.cur.name) { openFor = m.cur.name; const f = folds.get(m.cur.group); if (f) f.el.open = true; }
      credit.textContent = m.cur.credit ? `Pose by ${m.cur.credit}, from the MakeHuman community (CC0).` : '';
      const cr = app.correctives();
      tissue.el.hidden = tissueNote.hidden = !cr.ready; tissue.set(cr.on);
    });
    frames.push(() => {
      const m = app.motion(); if (!m.cur || m.cur.kind !== 'loop') { time.textContent = ''; return; }
      const d = m.cur.duration, t = ((m.t % d) + d) % d;
      if (!scrubbing) scrub.value = String(t / d);
      time.textContent = `${t.toFixed(1)} / ${d.toFixed(1)} s`;
    });
  }

  // ---------------------------------------------------------------- Scene
  const scene = h('section');
  {
    const head = sectionHead('Scene');
    const cards = h('div.light-cards', { role: 'group', 'aria-label': 'Lighting' });
    const lightEls = Object.entries(app.presets).map(([name, p], i) => {
      // a preview of the backdrop and of the direction of the key light as seen from the front
      const bg = (c: number[]) => `rgb(${c.map((v) => Math.round(255 * Math.min(1, Math.pow(v * 5, 1 / 2.2)))).join(',')})`;
      const a = p.key.az * Math.PI / 180, behind = Math.abs(p.key.az) > 90;
      const x = 50 + Math.sin(a) * 36, y = 50 - p.key.el / 90 * 40;
      const prev = h('span.light-prev', { style: { background: `radial-gradient(circle at 50% 58%, ${bg(p.bg[0])}, ${bg(p.bg[1])})` }, 'aria-hidden': 'true' },
        h('span.light-fig'), h('span.light-dot' + (behind ? '.behind' : ''), { style: { left: x + '%', top: y + '%' } }));
      const b = h('button.light-card', { type: 'button', 'aria-pressed': 'false', dataset: { preset: name }, 'data-tip': `${p.label}  ·  Alt+${i + 1}`, 'data-tip-pos': 'up' }, prev, h('span', { text: p.label }));
      b.addEventListener('click', () => app.applyPreset(name));
      cards.append(b);
      return b;
    });
    const rot = slider({ id: 'sc-rot', label: 'Light rotation (°)', min: -180, max: 180, step: 1, def: 0, digits: 0, ends: ['Left', 'Right'],
      onInput: (v) => app.setLightRotation(v, false), onEnd: () => app.setLightRotation(rot.value(), true) });
    const exp = slider({ id: 'sc-exp', label: 'Exposure (stops)', min: -1.5, max: 1.5, step: 0.05, def: 0, ends: ['Darker', 'Brighter'],
      onInput: (v) => app.setExposureEV(v) });
    const qNote = h('p.note');
    const qual = segmented('Quality', [{ value: 'auto', label: 'Auto' }, { value: 'sharp', label: 'Sharp' }, { value: 'fast', label: 'Fast' }], (v) => { app.setQuality(v); saveUi({ quality: v }); sync(); }, 'seg-fill');
    const fps = segmented('Frame rate limit', app.fpsCaps.map((c) => ({ value: String(c), label: c ? `${c} fps` : 'Display', tip: c ? `At most ${c} frames per second` : 'As fast as the display refreshes' })),
      (v) => { app.setFpsCap(+v); saveUi({ fpsCap: +v }); sync(); }, 'seg-fill');
    const hairT = toggle('Hair', (on) => app.setHairVisible(on));
    const physT = toggle('Hair physics', (on) => app.setHairPhysics(on));
    const turnT = toggle('Turntable', (on) => app.setTurntable(on));
    const skelT = toggle('Skeleton', (on) => app.setSkeleton(on), 'Shift+B');
    scene.append(head.el, h('h3.sub', { text: 'Lighting' }), cards, h('div.sliders', {}, rot.el, exp.el),
      h('h3.sub', { text: 'Quality' }), qual.el, qNote,
      h('h3.sub', { text: 'Frame rate limit' }), fps.el,
      h('p.note', { text: 'A lower limit saves energy while the figure, the hair or the camera moves. A still picture refines and then stops drawing at any limit.' }),
      h('h3.sub', { text: 'Display' }), h('div.switches', {}, hairT.el, physT.el, turnT.el, skelT.el),
      h('p.note', { text: 'The skeleton shows the bones of the rig in front of the body: the figure’s left side in teal, its right side in orange. Point at a bone to read its name.' }));
    const sync = () => {
      lightEls.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === app.preset())));
      head.note.textContent = app.presets[app.preset()]?.label || '';
      rot.set(app.lightRotation()); exp.set(app.exposureEV());
      qual.set(app.quality()); qNote.textContent = QUALITY[app.quality()] || '';
      fps.set(String(app.fpsCap()));
      hairT.set(app.hairVisible()); physT.set(app.hairPhysics()); turnT.set(app.turntable()); skelT.set(app.skeleton());
    };
    syncs.push(sync);
  }

  // the saved quality and frame rate limit apply once at start
  const q = uiState().quality;
  if (q && q !== app.quality() && !app.shot) app.setQuality(q);
  const cap = uiState().fpsCap;
  if (typeof cap === 'number' && cap !== app.fpsCap() && !app.shot) app.setFpsCap(cap);
  return { pose, scene, sync: () => { for (const s of syncs) s(); }, frame: () => { for (const f of frames) f(); } };
}
