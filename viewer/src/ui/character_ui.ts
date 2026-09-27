// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The sections of the Character segment: Characters (presets, colour looks, preset text), Body (anny's phenotype
// sliders), Face (the face shapes), Hair (the styles and their sliders) and Skin & eyes.

import type { App, Look } from './app.ts';
import { chip, foldGroup, searchField, sectionHead, segmented, slider, textButton, toggle, type Slider } from './controls.ts';
import { copyText, download, h, iconButton, slug } from './dom.ts';
import { icon } from './icons.ts';
import { toast } from './toast.ts';

// anny's age slider in years (anny.shape_distribution: the morphological age mapping of the calibration)
const AGE_A = [0, 0.05, 0.215, 0.415, 0.67, 0.77, 0.83, 1], AGE_Y = [0, 1, 4, 11, 16, 18, 64, 110];
export function ageYears(v: number) {
  let i = 0; while (i < AGE_A.length - 2 && v > AGE_A[i + 1]) i++;
  return AGE_Y[i] + (AGE_Y[i + 1] - AGE_Y[i]) * (v - AGE_A[i]) / (AGE_A[i + 1] - AGE_A[i]);
}
const yearsText = (v: number) => { const y = ageYears(v); return y < 2 ? `${Math.round(y * 12)} mo` : `${Math.round(y)} y`; };
const RACE_COLOURS: Record<string, string> = { african: '#b8845a', asian: '#d9b77e', caucasian: '#e6c9b4' };
const HAIR_FAMILIES: [string, string][] = [['short', 'Short'], ['medium', 'Medium'], ['long', 'Long'], ['tied', 'Tied']];
const HAIR_SLIDERS = [
  { key: 'length', label: 'Length', ends: ['Shorter', 'Longer'] },
  { key: 'curl', label: 'Curl', ends: ['Straight', 'Curly'] },
  { key: 'volume', label: 'Volume', ends: ['Flat', 'Full'] },
  { key: 'density', label: 'Density', ends: ['Thin', 'Thick'] },
  { key: 'fade', label: 'Fade height', ends: ['Low', 'High'] },
];

export interface Editor {
  app: App;
  begin(label: string): void;
  end(): void;
  // one edit with its history entry
  edit(label: string, mut: (n: Look) => void): void;
  // a whole look with its history entry (a character, a preset, a random draw)
  commit(label: string, look: any): void;
  // slider edits apply at most once a frame
  queue(mut: (n: Look) => void): void;
  // the name of the last character or look picked, for the "edited" state
  origin: string;
  faceSpread: number;
}

const rgb = (c: number[]) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;
const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)];
const rnd = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
const r3 = (v: number) => Math.round(v * 1000) / 1000;

// ------------------------------------------------------------------ random draws
export function randomPhenotype(app: App) {
  const ph: Record<string, number> = { ...app.baselinePhenotype() };
  const set = (k: string, v: number) => { if (k in ph) ph[k] = r3(v); };
  set('gender', Math.random() < 0.5 ? rnd(0, 0.25) : rnd(0.75, 1));
  set('age', rnd(0.45, 0.96));
  set('muscle', rnd(0.2, 0.8)); set('weight', rnd(0.2, 0.8));
  set('height', rnd(0.2, 0.8)); set('proportions', rnd(0.3, 0.7));
  // one main ethnicity, sometimes mixed with a second one
  const races = ['african', 'asian', 'caucasian'].filter((k) => k in ph), main = pick(races);
  for (const k of races) set(k, k === main ? 1 : Math.random() < 0.3 ? rnd(0.2, 0.8) : 0);
  return ph;
}
export function randomColours(app: App, n: Look) {
  n.skin = { tone: r3(rnd(0.05, 0.95)), undertone: r3(rnd(-0.6, 0.6)) };
  n.hair.color = pick(app.palettes.hair)[1];
  n.eyes.color = pick(app.palettes.eyes)[1];
}
export function randomHair(app: App, n: Look) {
  const spec = pick(app.hairSpecs().filter((s) => s.name !== 'bald'));
  n.hair = { color: pick(app.palettes.hair)[1], style: spec.name, part: Math.random() < 0.5 ? 'left' : 'right' };
}

// ------------------------------------------------------------------ the sections
export function characterSections(ed: Editor) {
  const app = ed.app;
  const syncs: ((L: Look) => void)[] = [];

  // ---------------------------------------------------------------- Characters
  const chars = h('section');
  {
    const head = sectionHead('Characters', textButton('Reset all', () => { ed.commit('Reset all', app.baseline()); ed.origin = 'Baseline'; }, 'undoSmall', 'tb-quiet'));
    const cards = h('div.cards');
    const cardEls = app.characters.map((c) => {
      const sk = app.skinBase(c.skin.tone, c.skin.undertone);
      const spec = app.hairSpecs().find((s) => s.name === c.hair.style);
      const b = h('button.card', { type: 'button', 'aria-pressed': 'false', dataset: { name: c.name } },
        h('span.avatar', { style: { '--a': rgb(sk), '--b': c.hair.color, '--c': c.eyes.color }, 'aria-hidden': 'true' }),
        h('span.card-text', {}, h('span.card-name', { text: c.name }),
          h('span.card-meta', { text: [spec?.label, c.phenotype?.age !== undefined ? yearsText(c.phenotype.age) : ''].filter(Boolean).join(' · ') })));
      b.addEventListener('click', () => { ed.commit(c.name, JSON.parse(JSON.stringify(c))); ed.origin = c.name; });
      cards.append(b);
      return b;
    });
    const looks = h('div.chips');
    const lookEls = app.looks.map((l) => {
      const b = chip(l.name, () => {
        const L = app.look();
        ed.commit('Look: ' + l.name, Object.assign({}, l, { phenotype: L.phenotype, face: L.face, hair: Object.assign({}, L.hair, { color: l.hair.color }) }));
        ed.origin = l.name;
      }, `linear-gradient(135deg, ${rgb(app.skinBase(l.skin.tone, l.skin.undertone))} 50%, ${l.hair.color} 50%)`);
      b.dataset.name = l.name;
      looks.append(b);
      return b;
    });
    // the preset text
    const box = h('textarea.preset-text', { rows: '7', spellcheck: 'false', 'aria-label': 'Preset JSON', hidden: true }) as HTMLTextAreaElement;
    const msg = h('p.inline-msg', { role: 'status' });
    const file = h('input', { type: 'file', accept: 'application/json,.json', hidden: true }) as HTMLInputElement;
    const loadText = (t: string, quiet = false) => {
      try {
        const look = app.normaliseLook(JSON.parse(t));
        ed.commit('Load preset', look); ed.origin = look.name;
        box.hidden = true; msg.textContent = ''; toast(`Loaded "${look.name}"`);
        return true;
      } catch (e) {
        if (!quiet) { msg.textContent = 'This text is not a preset. A preset is JSON with skin, hair, eyes and phenotype fields.'; msg.dataset.kind = 'error'; }
        return false;
      }
    };
    box.addEventListener('input', () => { if (box.value.trim()) loadText(box.value, true); });
    file.addEventListener('change', async () => { const f = file.files && file.files[0]; if (f) loadText(await f.text()); file.value = ''; });
    const tools = h('div.btn-row', {},
      textButton('Copy', async () => {
        const text = JSON.stringify(app.look(), null, 2);
        if (await copyText(text)) { box.hidden = true; toast('Preset copied'); }
        else { box.hidden = false; box.value = text; box.focus(); box.select(); msg.textContent = 'Select the text above and copy it.'; msg.dataset.kind = ''; }
      }, 'copy'),
      textButton('Paste', () => { box.hidden = false; box.value = ''; box.focus(); msg.textContent = 'Paste a preset into the box. It loads as soon as it reads as a preset.'; msg.dataset.kind = ''; }, 'paste'),
      // a hosted page (the artifact build) cannot start a download; Copy covers it there
      (window as any).MODEL_PARTS ? null : textButton('Download', () => { const L = app.look(); download(`anny-${slug(L.name)}.json`, new Blob([JSON.stringify(L, null, 2)], { type: 'application/json' })); }, 'download'),
      textButton('Open file', () => file.click(), 'upload'));
    chars.append(head.el,
      h('p.sec-lead', { text: 'A character sets the body, the face, the hair and the colours. A look changes the colours only.' }),
      cards, h('h3.sub', { text: 'Colour looks' }), looks,
      h('h3.sub', { text: 'Preset' }), h('p.note', { text: 'A preset is a short JSON text that holds this look and every slider value.' }), tools, box, msg, file);
    syncs.push((L) => {
      cardEls.forEach((b) => { const c = app.characters.find((x) => x.name === b.dataset.name); b.setAttribute('aria-pressed', String(!!c && L.name === c.name && app.sameCharacter(c, L))); });
      lookEls.forEach((b) => { const l = app.looks.find((x) => x.name === b.dataset.name); b.setAttribute('aria-pressed', String(!!l && L.name === l.name && app.sameLook(l, L))); });
    });
  }

  // ---------------------------------------------------------------- Body
  const body = h('section');
  {
    const head = sectionHead('Body',
      iconButton('dice', 'Random body', { tip: 'down', on: () => { const L = app.look(); randomBodyInto(app, ed, L); } }),
      textButton('Reset', () => ed.edit('Reset body', (n) => { n.phenotype = app.baselinePhenotype(); }), 'undoSmall', 'tb-quiet'));
    const list = h('div.sliders');
    const sliders: Slider[] = [];
    const races = app.bodySliders().filter((s) => s.race);
    const shares = h('div.share-bar', { 'aria-hidden': 'true' });
    const shareSegs = races.map((s) => { const seg = h('span', { style: { background: RACE_COLOURS[s.name] || '#999' } }, h('b', { text: s.label })); shares.append(seg); return seg; });
    for (const s of app.bodySliders()) {
      if (s.race && s === races[0]) list.append(h('h3.sub', { text: 'Ethnicity' }), h('p.note', { text: 'The three values mix by their shares, shown in the bar. Eurasian is Asian and Caucasian at equal values.' }), shares);
      const sl = slider({
        id: 'ed-b-' + s.name, label: s.label, min: 0, max: 1, def: 0.5, ends: s.ends,
        readout: s.race ? (v) => `${Math.round(100 * app.raceShare({ ...app.look().phenotype, [s.name]: v }, s.name))} %` : s.name === 'age' ? yearsText : undefined,
        onStart: () => ed.begin(s.label), onEnd: () => ed.end(),
        onInput: (v) => ed.queue((n) => { n.phenotype[s.name] = v; }),
      });
      sliders.push(sl);
      list.append(sl.el);
    }
    body.append(head.el, h('p.sec-lead', { text: 'anny\'s phenotype sliders. Each one runs from 0 to 1, and anny\'s default is 0.5.' }), list);
    syncs.push((L) => {
      app.bodySliders().forEach((s, i) => sliders[i].set(L.phenotype[s.name] ?? 0.5));
      races.forEach((s, i) => { const p = app.raceShare(L.phenotype, s.name); shareSegs[i].style.flexGrow = String(Math.max(p, 0.0001)); shareSegs[i].title = `${s.label} ${Math.round(p * 100)} %`; });
      const st = app.stature();
      head.note.textContent = st > 0 ? `${(st).toFixed(2)} m tall` : '';
    });
  }

  // ---------------------------------------------------------------- Face
  const face = h('section');
  {
    const head = sectionHead('Face',
      iconButton('dice', 'Random face', { tip: 'down', on: () => randomFaceInto(app, ed) }),
      textButton('Reset', () => ed.edit('Reset face', (n) => { n.face = {}; }), 'undoSmall', 'tb-quiet'));
    const spread = slider({ id: 'ed-face-spread', label: 'Random variation', min: 0.25, max: 1.5, step: 0.05, def: 1, readout: (v) => `× ${v.toFixed(2)}`,
      onInput: (v) => { ed.faceSpread = v; } });
    spread.el.classList.add('sl-compact');
    const empty = h('p.note.empty', { text: 'No face shape matches.', hidden: true });
    const groups: { g: string; fold: ReturnType<typeof foldGroup>; rows: { s: any; sl: Slider }[] }[] = [];
    const search = searchField('Search 103 face shapes', (q) => {
      let any = false;
      for (const G of groups) {
        let n = 0;
        for (const r of G.rows) {
          const hit = !q || r.s.label.toLowerCase().includes(q) || r.s.name.includes(q) || G.g.includes(q);
          r.sl.el.hidden = !hit; if (hit) n++;
        }
        G.fold.el.hidden = n === 0;
        if (q) G.fold.el.open = n > 0;
        any = any || n > 0;
      }
      empty.hidden = any;
    });
    const wrap = h('div.folds');
    for (const g of app.faceGroups()) {
      const reset = h('button.fold-reset', { type: 'button', 'aria-label': 'Reset the ' + app.faceGroupTitle(g).toLowerCase(), 'data-tip': 'Reset group', 'data-tip-pos': 'left' }, icon('undoSmall', 14));
      const fold = foldGroup(app.faceGroupTitle(g), [reset]);
      const rows = app.faceSliders().filter((s) => s.group === g).map((s) => {
        const sl = slider({ id: 'ed-f-' + s.name, label: s.label, title: s.name, min: s.range[0], max: s.range[1], def: 0, ends: s.ends,
          onStart: () => ed.begin('Face: ' + s.label), onEnd: () => ed.end(),
          onInput: (v) => ed.queue((n) => { n.face = { ...(n.face || {}), [s.name]: v }; if (v === 0) delete n.face[s.name]; }) });
        fold.body.append(sl.el);
        return { s, sl };
      });
      reset.addEventListener('click', () => ed.edit('Reset ' + app.faceGroupTitle(g).toLowerCase(), (n) => { for (const r of rows) delete n.face[r.s.name]; }));
      groups.push({ g, fold, rows });
      wrap.append(fold.el);
    }
    face.append(head.el, h('p.sec-lead', { text: 'Named shapes of the head and the face, which scale with the size of the head. Random face draws from anny\'s distribution, calibrated against measured faces, for the current body.' }),
      spread.el, search.el, wrap, empty);
    syncs.push((L) => {
      let total = 0;
      for (const G of groups) {
        let n = 0;
        for (const r of G.rows) { const v = L.face?.[r.s.name] ?? 0; r.sl.set(v); if (v !== 0) n++; }
        G.fold.count.textContent = n ? `${n} changed` : '';
        G.fold.el.toggleAttribute('data-changed', n > 0);
        total += n;
      }
      head.note.textContent = total ? `${total} changed` : '';
      spread.el.hidden = !app.facePrior();
    });
  }

  // ---------------------------------------------------------------- Hair
  const hair = h('section');
  {
    const show = toggle('Show', (on) => app.setHairVisible(on), 'Show or hide the hair  ·  Shift+H');
    const phys = toggle('Physics', (on) => app.setHairPhysics(on), 'The hair follows the motion of the head');
    const head = sectionHead('Hair');
    let family = '';
    const grid = h('div.style-grid', { role: 'group', 'aria-label': 'Hair styles' });
    const styleEls = app.hairSpecs().map((s) => {
      const b = h('button.style-card', { type: 'button', 'aria-pressed': 'false', dataset: { style: s.name, family: s.family } }, h('span', { text: s.label }));
      b.addEventListener('click', () => ed.edit('Hair: ' + s.label, (n) => { n.hair = { color: n.hair.color, style: s.name, part: n.hair.part }; }));
      grid.append(b);
      return b;
    });
    const fam = segmented('Hair length', HAIR_FAMILIES.map(([v, l]) => ({ value: v, label: l })), (v) => { family = v; paintFamily(); }, 'seg-fill');
    const paintFamily = () => { fam.set(family); styleEls.forEach((b) => { b.hidden = b.dataset.family !== family; }); };
    const params = h('div.sliders');
    const hs = HAIR_SLIDERS.map((p) => {
      const sl = slider({ id: 'ed-h-' + p.key, label: p.label, min: 0, max: 1, def: 0, ends: p.ends, digits: p.key === 'fade' ? 0 : 2, step: p.key === 'fade' ? 1 : 0.01,
        readout: p.key === 'fade' ? (v) => `${v > 0 ? '+' : ''}${v.toFixed(0)}°` : undefined,
        onStart: () => ed.begin('Hair ' + p.label.toLowerCase()), onEnd: () => ed.end(),
        onInput: (v) => ed.queue((n) => { (n.hair as any)[p.key] = v; }) });
      params.append(sl.el);
      return { p, sl };
    });
    const part = segmented('Part', [{ value: 'left', label: 'Left' }, { value: 'right', label: 'Right' }], (v) => ed.edit('Hair part', (n) => { n.hair.part = v; }));
    const partRow = h('div.row', {}, h('span.row-label', { text: 'Part' }), part.el);
    const colour = swatches(app, 'hair', 'Hair colour', (hex) => ed.edit('Hair colour', (n) => { n.hair.color = hex; }));
    hair.append(head.el, h('div.switch-row', {}, show.el, phys.el), fam.el, grid, h('h3.sub', { text: 'Shape' }), params, partRow, h('div.sub-row', {}, h('h3.sub', { text: 'Colour' }), colour.name), colour.el);
    let lastStyle = '';
    syncs.push((L) => {
      const spec = app.hairSpecs().find((s) => s.name === L.hair.style);
      if (!spec) return;
      head.note.textContent = spec.label;
      if (spec.name !== lastStyle) { lastStyle = spec.name; family = spec.family; paintFamily(); }
      styleEls.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.style === spec.name)));
      const d = app.hairDefaults(spec.name) || {};
      for (const { p, sl } of hs) {
        const range = p.key === 'fade' ? app.fadeRange : spec.controls[p.key];
        const on = p.key === 'fade' ? !!spec.render.fade : range[1] > range[0];
        sl.el.hidden = !on;
        if (!on) continue;
        sl.setRange(range[0], range[1], p.key === 'fade' ? 0 : d[p.key]);
        sl.set((L.hair as any)[p.key] ?? 0);
      }
      partRow.hidden = !spec.mirror;
      part.set(L.hair.part || 'left');
      colour.sync(L.hair.color);
      show.set(app.hairVisible()); phys.set(app.hairPhysics());
    });
  }

  // ---------------------------------------------------------------- Skin and eyes
  const skin = h('section');
  {
    const head = sectionHead('Skin & eyes');
    const tone = slider({ id: 'ed-tone', label: 'Tone', min: 0, max: 1, def: 0.35, ends: ['Fair', 'Deep'], track: true,
      onStart: () => ed.begin('Skin tone'), onEnd: () => ed.end(), onInput: (v) => ed.queue((n) => { n.skin.tone = v; }) });
    const under = slider({ id: 'ed-under', label: 'Undertone', min: -1, max: 1, def: 0, ends: ['Cool', 'Warm'], track: true,
      onStart: () => ed.begin('Undertone'), onEnd: () => ed.end(), onInput: (v) => ed.queue((n) => { n.skin.undertone = v; }) });
    const eyes = swatches(app, 'eyes', 'Eye colour', (hex) => ed.edit('Eye colour', (n) => { n.eyes.color = hex; }));
    skin.append(head.el, h('h3.sub', { text: 'Skin' }), h('div.sliders', {}, tone.el, under.el), h('div.sub-row', {}, h('h3.sub', { text: 'Eyes' }), eyes.name), eyes.el);
    syncs.push((L) => {
      tone.set(L.skin.tone); under.set(L.skin.undertone);
      const stops = []; for (let i = 0; i <= 10; i++) stops.push(`${rgb(app.skinBase(i / 10, L.skin.undertone))} ${i * 10}%`);
      const sk = rgb(app.skinBase(L.skin.tone, L.skin.undertone));
      tone.input.style.setProperty('--track', `linear-gradient(90deg, ${stops.join(', ')})`);
      under.input.style.setProperty('--track', `linear-gradient(90deg, ${rgb(app.skinBase(L.skin.tone, -1))}, ${rgb(app.skinBase(L.skin.tone, 0))}, ${rgb(app.skinBase(L.skin.tone, 1))})`);
      tone.input.style.setProperty('--thumb', sk); under.input.style.setProperty('--thumb', sk);
      eyes.sync(L.eyes.color);
    });
  }

  return { chars, body, face, hair, skin, sync: (L: Look) => { for (const s of syncs) s(L); } };
}

export function randomBodyInto(app: App, ed: Editor, L: Look) {
  const next = JSON.parse(JSON.stringify(L));
  next.phenotype = randomPhenotype(app);
  if (app.facePrior()) next.face = app.sampleFace(next.phenotype, ed.faceSpread);
  next.name = 'Custom';
  ed.commit('Random body', next);
}
export function randomFaceInto(app: App, ed: Editor) {
  if (!app.facePrior()) return;
  const L = app.look();
  ed.edit('Random face', (n) => { n.face = app.sampleFace(L.phenotype, ed.faceSpread); });
}

// colour swatches with a custom colour picker
function swatches(app: App, list: 'hair' | 'eyes', label: string, onPick: (hex: string) => void) {
  const el = h('div.swatches', { role: 'group', 'aria-label': label });
  const name = h('span.sub-val');
  const btns = app.palettes[list].map(([nm, hex]) => {
    const b = h('button.sw', { type: 'button', 'aria-label': nm, 'data-tip': nm, 'data-tip-pos': 'up', 'aria-pressed': 'false', style: { '--sw': hex }, dataset: { hex } });
    b.addEventListener('click', () => onPick(hex));
    el.append(b);
    return b;
  });
  const inp = h('input', { type: 'color', 'aria-label': 'Custom ' + label.toLowerCase() }) as HTMLInputElement;
  inp.addEventListener('input', () => onPick(inp.value.toLowerCase()));
  el.append(h('label.sw.sw-custom', { 'data-tip': 'Custom colour', 'data-tip-pos': 'up' }, inp));
  return {
    el, name,
    sync(hex: string) {
      btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.hex === hex)));
      if (document.activeElement !== inp) inp.value = hex;
      const m = app.palettes[list].find((p) => p[1] === hex);
      name.textContent = m ? m[0] : 'Custom ' + hex;
    },
  };
}
