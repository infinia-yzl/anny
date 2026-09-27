// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Short messages at the top of the page ("Preset copied", "Undo: Age"): they fade after a few seconds, and an error
// stays until it is dismissed.

import { h } from './dom.ts';
import { icon } from './icons.ts';

let box: HTMLElement | null = null;

export function toast(text: string, kind: 'info' | 'error' = 'info', ms = 2500) {
  if (!box) {
    box = h('div.toasts', { role: 'status', 'aria-live': 'polite' });
    document.body.append(box);
  }
  // one message at a time for the same text (a held key repeats it)
  for (const old of Array.from(box.children)) if ((old as HTMLElement).dataset.text === text) old.remove();
  const close = h('button.toast-x', { type: 'button', 'aria-label': 'Dismiss' }, icon('close', 14));
  const t = h('div.toast', { dataset: { kind, text } }, h('span', { text }), close);
  const remove = () => { t.classList.add('out'); setTimeout(() => t.remove(), 220); };
  close.addEventListener('click', remove);
  box.append(t);
  while (box.children.length > 3) box.firstElementChild!.remove();
  if (kind !== 'error') setTimeout(remove, ms);
}
