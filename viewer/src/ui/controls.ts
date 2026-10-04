// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The controls of the inspectors: a slider with an editable value, a reset button and a mark at its default; a
// segmented control; an on/off switch; a search field; a folding group.

import { clamp, h } from './dom.ts';
import { icon } from './icons.ts';

export interface SliderOptions {
  id: string;
  label: string;
  min: number;
  max: number;
  step?: number;
  def: number;
  ends?: string[];
  title?: string;
  digits?: number;
  // a readable form of the value (per cent, degrees), shown next to the number
  readout?: (v: number) => string;
  // a custom track (a colour gradient): no fill from the default
  track?: boolean;
  onInput: (v: number) => void;
  // an edit starts (the first input of a drag, a typed value) and ends (the change event)
  onStart?: () => void;
  onEnd?: () => void;
}

export interface Slider {
  el: HTMLElement;
  input: HTMLInputElement;
  opts: SliderOptions;
  set(v: number): void;
  setRange(min: number, max: number, def?: number): void;
  value(): number;
}

const fmt = (v: number, d: number) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);

export function slider(o: SliderOptions): Slider {
  const digits = o.digits ?? 2;
  const input = h('input', { type: 'range', id: o.id, min: String(o.min), max: String(o.max), step: String(o.step ?? 0.01), value: String(o.def) }) as HTMLInputElement;
  const num = h('input.sl-num', { type: 'text', inputmode: 'decimal', autocomplete: 'off', spellcheck: 'false', 'aria-label': o.label + ' value' }) as HTMLInputElement;
  const read = h('span.sl-read');
  const reset = h('button.sl-reset', { type: 'button', 'aria-label': 'Reset ' + o.label, 'data-tip': 'Reset to ' + fmt(o.def, digits), 'data-tip-pos': 'up' }, icon('undoSmall', 14));
  const mark = h('span.sl-def', { 'aria-hidden': 'true' });
  const bubble = h('span.sl-bubble', { 'aria-hidden': 'true' });
  const head = h('div.sl-head', {}, h('label', { for: o.id, text: o.label, title: o.title || '' }), read, reset, num);
  const track = h('div.sl-track', {}, input, mark, bubble);
  const el = h('div.sl' + (o.track ? '.sl-custom' : ''), {}, head, track);
  if (o.ends && o.ends.some(Boolean)) el.append(h('div.sl-ends', { 'aria-hidden': 'true' }, ...o.ends.map((t) => h('span', { text: t }))));
  let editing = false, cur = o.def;
  const start = () => { if (!editing) { editing = true; o.onStart?.(); } };
  const end = () => { if (editing) { editing = false; o.onEnd?.(); } };
  const pct = (v: number) => (o.max > o.min ? (v - o.min) / (o.max - o.min) * 100 : 0);
  const paint = (v: number) => {
    const p = pct(v), d = pct(clamp(o.def, o.min, o.max));
    input.style.setProperty('--lo', Math.min(p, d) + '%');
    input.style.setProperty('--hi', Math.max(p, d) + '%');
    input.style.setProperty('--p', p + '%');
    mark.style.left = `calc(${d}% + ${(0.5 - d / 100) * 18}px)`;
    bubble.style.left = `calc(${p}% + ${(0.5 - p / 100) * 18}px)`;
    const text = fmt(v, digits);
    if (document.activeElement !== num) num.value = text;
    bubble.textContent = o.readout ? o.readout(v) : text;
    read.textContent = o.readout ? o.readout(v) : '';
    input.setAttribute('aria-valuetext', o.readout ? `${text}, ${o.readout(v)}` : text);
    el.toggleAttribute('data-changed', Math.abs(v - o.def) > 1e-6);
  };
  const apply = (v: number) => { cur = v; paint(v); o.onInput(v); };
  input.addEventListener('input', () => { start(); apply(parseFloat(input.value)); });
  input.addEventListener('change', end);
  input.addEventListener('pointerdown', () => el.classList.add('dragging'));
  const up = () => el.classList.remove('dragging');
  input.addEventListener('pointerup', up); input.addEventListener('pointercancel', up); input.addEventListener('blur', up);
  input.addEventListener('dblclick', () => { start(); input.value = String(o.def); apply(o.def); end(); });
  reset.addEventListener('click', () => { start(); input.value = String(o.def); apply(o.def); end(); input.focus({ preventScroll: true }); });
  const commitNum = () => {
    const v = parseFloat(num.value.replace(',', '.').replace(/[^\d.+-]/g, ''));
    if (!isFinite(v)) { num.value = fmt(cur, digits); return; }
    const c = clamp(v, o.min, o.max);
    start(); input.value = String(c); apply(c); end();
  };
  num.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commitNum(); num.select(); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); num.value = fmt(cur, digits); num.blur(); }
    else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const s = (o.step ?? 0.01) * (e.shiftKey ? 10 : 1) * (e.key === 'ArrowUp' ? 1 : -1);
      num.value = fmt(clamp(cur + s, o.min, o.max), digits); commitNum();
    }
  });
  num.addEventListener('blur', commitNum);
  num.addEventListener('focus', () => num.select());
  paint(o.def);
  return {
    el, input, opts: o,
    set(v: number) {
      cur = v;
      if (document.activeElement !== input) input.value = String(v);
      paint(v);
    },
    setRange(min: number, max: number, def?: number) {
      o.min = min; o.max = max; if (def !== undefined) o.def = def;
      input.min = String(min); input.max = String(max);
      reset.setAttribute('data-tip', 'Reset to ' + fmt(o.def, digits));
      paint(cur);
    },
    value: () => cur,
  };
}

// a row of mutually exclusive buttons (aria-pressed); arrow keys move between them
export interface Segmented { el: HTMLElement; set(value: string): void }
export function segmented(label: string, items: { value: string; label: string; short?: string; icon?: string; tip?: string }[], onPick: (v: string) => void, cls = ''): Segmented {
  const el = h('div.seg' + (cls ? '.' + cls : ''), { role: 'group', 'aria-label': label });
  const buttons = items.map((it) => {
    const b = h('button', { type: 'button', 'aria-pressed': 'false', dataset: { value: it.value }, 'aria-label': it.short ? it.label : null,
      'data-tip': it.tip || null, 'data-tip-pos': 'up' });
    if (it.icon) b.append(icon(it.icon, 18));
    b.append(h('span', { text: it.short || it.label, 'aria-hidden': it.short ? 'true' : null }));
    b.addEventListener('click', () => onPick(it.value));
    return b;
  });
  el.append(...buttons);
  el.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const i = buttons.indexOf(document.activeElement as any);
    if (i < 0) return;
    e.preventDefault(); e.stopPropagation();
    const j = (i + (e.key === 'ArrowRight' ? 1 : buttons.length - 1)) % buttons.length;
    buttons[j].focus(); buttons[j].click();
  });
  return { el, set(v: string) { for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.value === v)); } };
}

// an on/off switch
export interface Switch { el: HTMLElement; set(on: boolean): void }
export function toggle(label: string, onChange: (on: boolean) => void, tip = ''): Switch {
  const b = h('button.switch', { type: 'button', role: 'switch', 'aria-checked': 'false', 'data-tip': tip || null, 'data-tip-pos': 'up' },
    h('span.switch-track', { 'aria-hidden': 'true' }, h('span.switch-knob')), h('span.switch-label', { text: label }));
  b.addEventListener('click', () => onChange(b.getAttribute('aria-checked') !== 'true'));
  return { el: b, set(on: boolean) { b.setAttribute('aria-checked', String(on)); } };
}

// a search field that calls back on each keystroke
export function searchField(placeholder: string, onQuery: (q: string) => void) {
  const input = h('input.search-input', { type: 'search', placeholder, 'aria-label': placeholder, autocomplete: 'off', spellcheck: 'false' }) as HTMLInputElement;
  input.addEventListener('input', () => onQuery(input.value.trim().toLowerCase()));
  input.addEventListener('keydown', (e) => { if (e.key === 'Escape' && input.value) { e.stopPropagation(); input.value = ''; onQuery(''); } });
  return { el: h('label.search', {}, icon('search', 16), input), input };
}

// a folding group: a summary with a title, a count and actions, and a body
export function foldGroup(title: string, actions: HTMLElement[] = []) {
  const count = h('span.fold-n');
  const summary = h('summary', {}, icon('chevron', 14), h('span.fold-title', { text: title }), count, ...actions);
  for (const a of actions) a.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
  const body = h('div.fold-body');
  const el = h('details.fold', {}, summary, body) as HTMLDetailsElement;
  return { el, body, count };
}

// a pressed-state chip button
export function chip(label: string, onClick: () => void, dot?: string) {
  const b = h('button.chip', { type: 'button', 'aria-pressed': 'false' });
  if (dot) b.append(h('i', { style: { background: dot } }));
  b.append(document.createTextNode(label));
  b.addEventListener('click', onClick);
  return b;
}

// a section header inside an inspector: a title, an optional live note and actions
export function sectionHead(title: string, ...actions: HTMLElement[]) {
  const note = h('span.sec-note');
  return { el: h('header.sec-head', {}, h('h2', { text: title }), note, h('div.sec-actions', {}, ...actions)), note };
}

export function textButton(label: string, onClick: () => void, iconName?: string, cls = '') {
  const b = h('button.tb' + (cls ? '.' + cls : ''), { type: 'button' });
  if (iconName) b.append(icon(iconName, 16));
  b.append(h('span', { text: label }));
  b.addEventListener('click', onClick);
  return b;
}
