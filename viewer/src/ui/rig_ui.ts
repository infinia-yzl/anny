// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The Rig section of the Stage segment: the one place for the bones and the skin weights. It holds the view of the
// rig (off, the skeleton, or the skin weights with the skeleton), the chosen bone with its weight ramp and the skin it
// moves, and the list of the bones by part of the body, with a search.

import type { App } from './app.ts';
import { chip, foldGroup, searchField, sectionHead, segmented } from './controls.ts';
import { h } from './dom.ts';

// the parts of the body, in the order of the list; a bone goes to the first part whose test it passes
const PARTS: [string, (n: string) => boolean][] = [
  ['Spine and head', (n) => !/\.[LR]$/.test(n)],
  ['Left arm', (n) => /^(clavicle|shoulder|upperarm|lowerarm|wrist).*\.L$/.test(n)],
  ['Left hand', (n) => /^(finger|metacarpal).*\.L$/.test(n)],
  ['Right arm', (n) => /^(clavicle|shoulder|upperarm|lowerarm|wrist).*\.R$/.test(n)],
  ['Right hand', (n) => /^(finger|metacarpal).*\.R$/.test(n)],
  ['Left leg', (n) => /^(pelvis|upperleg|lowerleg|foot).*\.L$/.test(n)],
  ['Left foot', (n) => /^toe.*\.L$/.test(n)],
  ['Right leg', (n) => /^(pelvis|upperleg|lowerleg|foot).*\.R$/.test(n)],
  ['Right foot', (n) => /^toe.*\.R$/.test(n)],
  ['Other', () => true],
];

const MODES: Record<string, string> = { off: 'Off', skeleton: 'Skeleton', weights: 'Skin weights' };

export function rigSection(app: App) {
  const el = h('section');
  const head = sectionHead('Rig');
  const show = segmented('Show', [
    { value: 'off', label: 'Off' },
    { value: 'skeleton', label: 'Skeleton', tip: 'Skeleton  ·  Shift+B' },
    { value: 'weights', label: 'Skin weights', tip: 'Skin weights  ·  Shift+W' },
  ], (v) => app.setRigView(v), 'seg-fill');
  const showNote = h('p.note');

  // the chosen bone
  const card = h('div.bone-card', { role: 'status' });
  const fmtN = (n: number) => n.toLocaleString('en');
  const paintCard = () => {
    const name = app.rigBone(), mode = app.rigView();
    if (!name) {
      card.replaceChildren(h('p.note', { text: mode === 'weights'
        ? 'Each colour is one bone, and blended colours show shared weights. Point at a bone in the view, or choose one below.'
        : 'Point at a bone in the view, or choose one below.' }));
      return;
    }
    const st = app.weightStats(), b = st?.bones.find((x) => x.name === name);
    const all = h('button.tb.tb-quiet', { type: 'button', text: 'Clear' });
    all.addEventListener('click', () => app.setRigBone(null));
    card.replaceChildren(
      h('div.bc-head', {}, h('b.bc-name', { text: name }), h('span.bc-kept', { text: app.rigBoneKept() ? 'kept' : '' }), all),
      mode === 'weights' ? h('div.wl-bar', {}, h('span', { text: '0' }), h('div.wl-ramp', { 'aria-hidden': 'true' }), h('span', { text: '1' })) : null,
      st && b ? h('p.note', { text: b.touched ? `Weighs on ${fmtN(b.touched)} vertices, and leads ${fmtN(Math.round(b.area * 1e4))} cm² of skin (${(b.area / st.area * 100).toFixed(1)} % of the body).` : 'This bone moves no vertex of the body.' }) : null);
  };

  // the bones by part of the body
  const names = app.boneNames();
  const chips = new Map<string, HTMLButtonElement>();
  const folds = PARTS.map(([title, test]) => ({ title, test, fold: foldGroup(title) }));
  for (const n of names) {
    const f = folds.find((x) => x.test(n))!;
    const b = chip(n, () => {
      if (app.rigBone() === n && app.rigBoneKept()) { app.setRigBone(null); return; }
      if (app.rigView() === 'off') app.setRigView('skeleton');
      app.setRigBone(n, true);
    });
    b.classList.add('chip-mono');
    f.fold.body.append(b); chips.set(n, b);
  }
  const list = h('div.folds');
  for (const f of folds) {
    const n = f.fold.body.childElementCount;
    if (!n) continue;
    f.fold.body.classList.add('chips'); f.fold.count.textContent = String(n);
    list.append(f.fold.el);
  }
  const empty = h('p.note.empty', { text: 'No bone matches.', hidden: true });
  const search = searchField(`Search ${names.length} bones`, (q) => {
    let any = false;
    for (const f of folds) {
      let n = 0;
      for (const b of Array.from(f.fold.body.children) as HTMLElement[]) { const hit = !q || b.textContent!.toLowerCase().includes(q); b.hidden = !hit; if (hit) n++; }
      // without a query, the part of the chosen bone stays open
      const bone = app.rigBone();
      f.fold.el.hidden = n === 0; f.fold.el.open = q ? n > 0 : !!bone && f === folds.find((x) => x.test(bone)); any = any || n > 0;
    }
    empty.hidden = any;
  });

  el.append(head.el, h('p.sec-lead', { text: 'The bones of anny’s rig and the skin weights that tie the body to them.' }),
    h('h3.sub', { text: 'Show' }), show.el, showNote,
    h('h3.sub', { text: 'Bone' }), card,
    h('h3.sub', { text: 'Bones' }), search.el, list, empty,
    h('p.note', { text: 'In the skeleton, the figure’s left side is teal, its right side orange, and the chosen bone white. A click on a bone in the view keeps it, and a click beside the bones clears it.' }));

  let openFor = '';
  const sync = () => {
    const mode = app.rigView(), bone = app.rigBone();
    show.set(mode);
    head.note.textContent = MODES[mode] || '';
    showNote.textContent = mode === 'weights'
      ? 'The body shows the weight of the chosen bone from blue (none) to red (all), or a colour per bone. The hair hides meanwhile.'
      : mode === 'skeleton' ? 'The bones stand in front of the body. Point at one to read its name.' : 'The picture shows the body alone.';
    for (const [n, b] of chips) b.setAttribute('aria-pressed', String(n === bone));
    // the part of the chosen bone opens, once per choice
    if (bone && bone !== openFor) { openFor = bone; const f = folds.find((x) => x.test(bone)); if (f) f.fold.el.open = true; }
    paintCard();
  };
  return { el, sync };
}
