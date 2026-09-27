// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The two segments of the page: the Character segment on the left (who the figure is) and the Stage segment on the
// right (how it is shown). Each has an icon rail and an inspector that shows one section at a time. On a wide screen
// both inspectors can stay open; on a narrower one, opening one closes the other; on a phone the rails form one
// bottom tab bar and the inspectors become bottom sheets.

import { h, saveUi, uiState } from './dom.ts';
import { icon } from './icons.ts';

export interface SectionDef {
  id: string;
  label: string;
  icon: string;
  key: string;
  el: HTMLElement;
  onShow?: () => void;
}

export interface Segment {
  side: 'left' | 'right';
  rail: HTMLElement;
  panel: HTMLElement;
  sections: SectionDef[];
  open(id?: string, focus?: boolean): void;
  close(focusRail?: boolean): void;
  toggle(id: string): void;
  isOpen(): boolean;
  current(): string;
}

export const PHONE = 720, WIDE = 1280;
export const isPhone = () => innerWidth < PHONE;

export function makeSegment(side: 'left' | 'right', label: string, sections: SectionDef[], onLayout: () => void): Segment {
  const list = h('div.rail-list', { role: 'tablist', 'aria-label': label, 'aria-orientation': 'vertical' });
  const rail = h('nav.rail.rail-' + side, { 'aria-label': label + ' sections' }, list);
  const scroll = h('div.insp-scroll');
  const handle = h('button.sheet-handle', { type: 'button', 'aria-label': 'Resize the panel' }, h('span'));
  const close = h('button.ib.insp-x', { type: 'button', 'aria-label': 'Close the ' + label.toLowerCase() + ' panel', 'data-tip': 'Close  ·  Esc', 'data-tip-pos': side === 'left' ? 'right' : 'left' }, icon('close', 16));
  const panel = h('aside.insp.insp-' + side, { id: 'insp-' + side, 'aria-label': label, hidden: true, dataset: { snap: 'half' } }, handle, close, scroll);
  const saved = uiState()[side] || {};
  let cur = sections.some((s) => s.id === saved.section) ? saved.section : sections[0].id;
  const tabs = sections.map((s) => {
    const b = h('button.rail-tab', { type: 'button', role: 'tab', id: 'tab-' + s.id, 'aria-selected': 'false', 'aria-controls': 'sec-' + s.id,
      'data-tip': s.label + '  ·  ' + s.key.toUpperCase(), 'data-tip-pos': side === 'left' ? 'right' : 'left', tabindex: '-1' },
      icon(s.icon, 22), h('span.rail-label', { text: s.label }));
    b.addEventListener('click', () => seg.toggle(s.id));
    list.append(b);
    s.el.id = 'sec-' + s.id;
    s.el.setAttribute('role', 'tabpanel');
    s.el.setAttribute('aria-labelledby', 'tab-' + s.id);
    s.el.classList.add('sec');
    s.el.hidden = true;
    scroll.append(s.el);
    return b;
  });
  // the rail is one tab stop; the arrow keys move between its tabs
  list.addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement as any);
    if (i < 0) return;
    let j = -1;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') j = (i + 1) % tabs.length;
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') j = (i + tabs.length - 1) % tabs.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = tabs.length - 1;
    if (j < 0) return;
    e.preventDefault(); e.stopPropagation();
    tabs[j].focus();
    seg.open(sections[j].id);
  });
  const paintTabs = () => {
    sections.forEach((s, i) => {
      const on = s.id === cur && !panel.hidden;
      tabs[i].setAttribute('aria-selected', String(on));
      tabs[i].tabIndex = s.id === cur ? 0 : -1;
      s.el.hidden = s.id !== cur;
    });
  };
  close.addEventListener('click', () => seg.close(true));
  // the sheet on a phone: a tap on the handle switches between half and full height; a drag picks the nearer
  let dragY = -1, startH = 0;
  handle.addEventListener('click', () => { if (dragY === -2) { dragY = -1; return; } panel.dataset.snap = panel.dataset.snap === 'full' ? 'half' : 'full'; });
  handle.addEventListener('pointerdown', (e) => { if (!isPhone()) return; dragY = e.clientY; startH = panel.getBoundingClientRect().height; handle.setPointerCapture(e.pointerId); });
  handle.addEventListener('pointermove', (e) => {
    if (dragY < 0) return;
    panel.style.height = Math.max(120, startH + dragY - e.clientY) + 'px';
  });
  handle.addEventListener('pointerup', (e) => {
    if (dragY < 0) return;
    const moved = Math.abs(e.clientY - dragY) > 6;
    const hgt = startH + dragY - e.clientY;
    panel.style.height = '';
    if (moved) {
      dragY = -2;
      if (hgt < innerHeight * 0.22) seg.close();
      else panel.dataset.snap = hgt > innerHeight * 0.62 ? 'full' : 'half';
    } else dragY = -1;
  });
  const seg: Segment = {
    side, rail, panel, sections,
    open(id, focus = false) {
      if (id) cur = id;
      const s = sections.find((x) => x.id === cur)!;
      const wasHidden = panel.hidden;
      panel.hidden = false;
      paintTabs();
      if (wasHidden) panel.classList.add('entering'), requestAnimationFrame(() => panel.classList.remove('entering'));
      s.onShow?.();
      saveUi({ [side]: { section: cur, open: true } });
      onLayout();
      if (focus) (s.el.querySelector('input, button, [tabindex]') as HTMLElement | null)?.focus({ preventScroll: true });
    },
    close(focusRail = false) {
      if (panel.hidden) return;
      panel.hidden = true;
      paintTabs();
      saveUi({ [side]: { section: cur, open: false } });
      onLayout();
      if (focusRail) tabs[sections.findIndex((s) => s.id === cur)].focus({ preventScroll: true });
    },
    toggle(id) { if (!panel.hidden && cur === id) seg.close(); else seg.open(id); },
    isOpen: () => !panel.hidden,
    current: () => cur,
  };
  paintTabs();
  return seg;
}
