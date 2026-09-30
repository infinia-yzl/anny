// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Small DOM helpers for the interface, and the saved state of the interface (localStorage 'anny-ui').

import { icon } from './icons.ts';

type Attrs = Record<string, any>;
type Child = Node | string | null | undefined | false;

// h('button.cls#id', { attrs, on: { click } }, children...)
export function h<K extends keyof HTMLElementTagNameMap>(spec: K, attrs?: Attrs, ...children: (Child | Child[])[]): HTMLElementTagNameMap[K];
export function h(spec: string, attrs?: Attrs, ...children: (Child | Child[])[]): any;
export function h(spec: string, attrs: Attrs = {}, ...children: (Child | Child[])[]): any {
  const m = /^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i.exec(spec);
  const tag = (m && m[1]) || 'div';
  const el = document.createElement(tag) as any;
  if (m && m[2]) for (const part of m[2].match(/[.#][\w-]+/g) || []) {
    if (part[0] === '.') el.classList.add(part.slice(1)); else el.id = part.slice(1);
  }
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'on') { for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn as any); continue; }
    if (k === 'style' && typeof v === 'object') { for (const [sk, sv] of Object.entries(v)) el.style.setProperty(sk, String(sv)); continue; }
    if (k === 'text') { el.textContent = v; continue; }
    if (k === 'dataset') { Object.assign(el.dataset, v); continue; }
    if (k in el && typeof v !== 'string') el[k] = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c as any);
  return el;
}

// an icon button with an accessible name and a tooltip (the shortcut shows in the tooltip)
export function iconButton(name: string, label: string, opts: { key?: string; tip?: 'down' | 'up' | 'left' | 'right'; cls?: string; text?: boolean; on?: () => void } = {}) {
  const b = h('button', { type: 'button', class: 'ib' + (opts.cls ? ' ' + opts.cls : ''), 'aria-label': label,
    'data-tip': label + (opts.key ? '  ·  ' + opts.key : ''), 'data-tip-pos': opts.tip || 'down' });
  b.append(icon(name));
  if (opts.text) b.append(h('span.ib-label', { text: label }));
  if (opts.on) b.addEventListener('click', opts.on);
  return b;
}

export const $ = (id: string) => document.getElementById(id);

// the saved state of the interface; storage can be missing (private windows, previews), so every access is guarded
const KEY = 'anny-ui';
let cache: Record<string, any> | null = null;
export function uiState(): Record<string, any> {
  if (cache) return cache;
  try { cache = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { cache = {}; }
  return cache!;
}
export function saveUi(patch: Record<string, any>) {
  Object.assign(uiState(), patch);
  try { localStorage.setItem(KEY, JSON.stringify(cache)); } catch (e) { /* storage unavailable */ }
}

// true while the user types into a field, where the page's shortcuts stay off
export function typing(e: Event) {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  const tag = t.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable) return true;
  if (tag === 'INPUT') {
    const type = (t as HTMLInputElement).type;
    return !['range', 'checkbox', 'radio', 'button', 'color'].includes(type);
  }
  return false;
}

export function clamp(v: number, lo: number, hi: number) { return Math.min(hi, Math.max(lo, v)); }

export function download(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function slug(s: string) { return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'corporis'; }

// copy text: the clipboard where the page may use it, else false (the caller shows the text to copy by hand)
export async function copyText(text: string): Promise<boolean> {
  try { if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); return true; } } catch (e) { /* denied */ }
  return false;
}
