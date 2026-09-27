// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The keyboard shortcuts of the page and the help sheet that lists them.

import { h, iconButton } from './dom.ts';

export const SHORTCUTS: [string, [string, string][]][] = [
  ['Camera', [
    ['Drag', 'Orbit around the figure'],
    ['Right-drag, Shift+drag, two fingers', 'Move the view'],
    ['Scroll, pinch', 'Zoom toward the pointer'],
    ['Double-click', 'Focus on the nearest joint (the background reframes)'],
    ['← →', 'Orbit'],
    ['↑ ↓', 'Move up or down the body'],
    ['Shift+↑ ↓', 'Tilt'],
    ['+ −', 'Zoom'],
    ['0', 'Reset the view'],
  ]],
  ['View', [
    ['1 2 3', 'Full body, upper body, face'],
    ['4 5 6 7', 'Front, three-quarter, side, back'],
    ['T', 'Turntable'],
    ['Shift+H', 'Show or hide the hair'],
    ['Shift+B', 'Show or hide the skeleton'],
    ['Shift+W', 'Show or hide the skin weights'],
    ['Alt+1 to 4', 'Lighting presets'],
    ['Space', 'Play or pause the animation'],
  ]],
  ['Panels', [
    ['C B F H E', 'Characters, Body, Face, Hair, Skin & eyes'],
    ['P L', 'Pose, Scene'],
    ['`', 'Performance'],
    ['?', 'This sheet'],
    ['Esc', 'Close the panel or the sheet'],
  ]],
  ['Editing', [
    ['Ctrl+Z, ⌘Z', 'Undo'],
    ['Ctrl+Shift+Z, Ctrl+Y', 'Redo'],
    ['Double-click a slider', 'Return it to its default'],
    ['Enter in a value', 'Apply a typed value'],
  ]],
];

export function helpSheet() {
  const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left' });
  const dlg = h('dialog.help', { 'aria-labelledby': 'help-title' },
    h('header.help-head', {}, h('h2', { id: 'help-title', text: 'Anny Viewer' }), close),
    h('p.help-lead', { text: 'Anny is a differentiable human body model for every age, built on the MakeHuman assets. This page renders anny live in your browser: skin with subsurface scattering, eyes with a refractive cornea, and up to 64,000 hair strands that follow the body and its motion.' }),
    h('div.help-grid', {}, ...SHORTCUTS.map(([title, rows]) => h('section', {}, h('h3', { text: title }),
      h('dl', {}, ...rows.flatMap(([k, v]) => [h('dt', {}, ...k.split(', ').map((x, i) => [i ? ', ' : '', h('kbd', { text: x })]).flat()), h('dd', { text: v })]))))),
  ) as HTMLDialogElement;
  close.addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  return dlg;
}
