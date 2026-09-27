// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The icons of the page: one outline style on a 24 px grid (1.6 px stroke, round caps and joins), drawn as inline SVG
// so the page needs no icon font and no request.

const PATHS: Record<string, string> = {
  characters: '<circle cx="9" cy="8" r="3.2"/><path d="M3.5 19.5c0-3.3 2.5-5.6 5.5-5.6s5.5 2.3 5.5 5.6"/><circle cx="16.5" cy="9" r="2.6"/><path d="M15.4 14.2c3 .1 5.1 2.2 5.1 5.3"/>',
  body: '<circle cx="12" cy="4.6" r="2.1"/><path d="M8 9.2h8M12 8v6.2M12 14.2l-2.6 6.3M12 14.2l2.6 6.3M8 9.2l-1.6 5.4M16 9.2l1.6 5.4"/>',
  face: '<path d="M12 3.5c-3.9 0-6.5 3-6.5 7.2 0 4.8 3 9.8 6.5 9.8s6.5-5 6.5-9.8c0-4.2-2.6-7.2-6.5-7.2z"/><path d="M9.3 10.6h.01M14.7 10.6h.01M12 11.5v2.6l-1 .5M10 16.7c1.2.8 2.8.8 4 0"/>',
  hair: '<path d="M4.5 20c0-7 1.5-15 7.5-15s7.5 8 7.5 15"/><path d="M8 20c0-5 .6-9.3 4-11M16 20c0-5-.6-9.3-4-11M12 9v11"/>',
  skin: '<path d="M2.8 11.5S6 6.5 12 6.5s9.2 5 9.2 5-3.2 5-9.2 5-9.2-5-9.2-5z"/><circle cx="12" cy="11.5" r="2.6"/><path d="M18.5 16.5c.9 1.3 1.6 2.3 1.6 3a1.6 1.6 0 01-3.2 0c0-.7.7-1.7 1.6-3z"/>',
  pose: '<circle cx="14.5" cy="4.3" r="2"/><path d="M8.5 9.3l3.8-2 2.8 2.9 3 1M12.3 7.3l-1.3 6 3.6 2.5-1.2 5M11 13.3l-3.2 2.4-3.3-.4"/>',
  scene: '<circle cx="12" cy="12" r="3.8"/><path d="M12 2.8v2M12 19.2v2M2.8 12h2M19.2 12h2M5.5 5.5l1.4 1.4M17.1 17.1l1.4 1.4M5.5 18.5l1.4-1.4M17.1 6.9l1.4-1.4"/>',
  stats: '<path d="M3 12h3.5l2.2-5.5 4.2 11 2.6-7 1.6 1.5H21"/>',
  undo: '<path d="M8.5 5.5L4.5 9.5l4 4"/><path d="M4.5 9.5h9.3a5.2 5.2 0 010 10.4H9"/>',
  redo: '<path d="M15.5 5.5l4 4-4 4"/><path d="M19.5 9.5h-9.3a5.2 5.2 0 000 10.4H15"/>',
  dice: '<rect x="4" y="4" width="16" height="16" rx="3.2"/><path d="M8.6 8.6h.01M15.4 8.6h.01M12 12h.01M8.6 15.4h.01M15.4 15.4h.01" stroke-width="2.6"/>',
  save: '<path d="M12 4v10.5M7.5 10l4.5 4.5 4.5-4.5"/><path d="M4.5 15.5v2.3c0 1.2.9 2.2 2.2 2.2h10.6c1.3 0 2.2-1 2.2-2.2v-2.3"/>',
  fullscreen: '<path d="M4.5 9V4.5H9M15 4.5h4.5V9M19.5 15v4.5H15M9 19.5H4.5V15"/>',
  help: '<circle cx="12" cy="12" r="8.6"/><path d="M9.6 9.4a2.5 2.5 0 014.8.9c0 1.7-2.4 2.2-2.4 3.8M12 16.8h.01"/>',
  more: '<path d="M6 12h.01M12 12h.01M18 12h.01" stroke-width="2.8"/>',
  turntable: '<path d="M4.5 12a7.5 7.5 0 0113-5.1M19.5 12a7.5 7.5 0 01-13 5.1"/><path d="M17.8 3.8v3.4h-3.4M6.2 20.2v-3.4h3.4"/>',
  bones: '<circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="6" r="1.5"/><path d="M7.56 16.44L11.2 16.48 16.94 7.06 7.52 12.8zM7.52 12.8l3.68 3.68"/>',
  reset: '<circle cx="12" cy="12" r="6.5"/><path d="M12 2.8v4M12 17.2v4M2.8 12h4M17.2 12h4"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
  camera: '<path d="M4 8.5c0-1 .8-1.8 1.8-1.8h2.4L9.8 4.5h4.4l1.6 2.2h2.4c1 0 1.8.8 1.8 1.8v9.2c0 1-.8 1.8-1.8 1.8H5.8c-1 0-1.8-.8-1.8-1.8z"/><circle cx="12" cy="12.8" r="3.4"/>',
  close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
  collapse: '<path d="M6 15l6-6 6 6"/>',
  expand: '<path d="M6 9l6 6 6-6"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  play: '<path d="M8 5.5v13l10.5-6.5z"/>',
  pause: '<path d="M8.5 5.5v13M15.5 5.5v13" stroke-width="2.4"/>',
  search: '<circle cx="10.8" cy="10.8" r="6"/><path d="M15.3 15.3l4.7 4.7"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6.5c0-1.1-.9-2-2-2h-7c-1.1 0-2 .9-2 2v7c0 1.1.9 2 2 2h2"/>',
  paste: '<rect x="5.5" y="4.5" width="13" height="16" rx="2"/><path d="M9 4.5v-.3c0-.7.5-1.2 1.2-1.2h3.6c.7 0 1.2.5 1.2 1.2v.3M9 11h6M9 15h4"/>',
  download: '<path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M5 20h14"/>',
  upload: '<path d="M12 15V4M7.5 8.5L12 4l4.5 4.5M5 20h14"/>',
  undoSmall: '<path d="M8.5 6.5L5 10l3.5 3.5"/><path d="M5 10h8.5a4.5 4.5 0 010 9H10"/>',
  drag: '<path d="M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01" stroke-width="2.6"/>',
  hud: '<rect x="3.5" y="5" width="17" height="14" rx="2.5"/><path d="M6.5 14.5l3-3 2.5 2 5-5"/>',
  head: '<path d="M12 4.5c-3.3 0-5.5 2.6-5.5 6 0 4.1 2.6 8.9 5.5 8.9s5.5-4.8 5.5-8.9c0-3.4-2.2-6-5.5-6z"/>',
  focus: '<path d="M4.5 9V6.3c0-1 .8-1.8 1.8-1.8H9M15 4.5h2.7c1 0 1.8.8 1.8 1.8V9M19.5 15v2.7c0 1-.8 1.8-1.8 1.8H15M9 19.5H6.3c-1 0-1.8-.8-1.8-1.8V15"/><circle cx="12" cy="12" r="2.2"/>',
};
// front, three-quarter, side and back: a head seen from above, the nose pointing at the camera (down)
const ANGLE_NOSE: Record<string, number> = { front: 0, three: 38, side: 90, back: 180 };
for (const [name, deg] of Object.entries(ANGLE_NOSE)) {
  const a = deg * Math.PI / 180, x = 12 + 7.8 * Math.sin(a), y = 12 + 7.8 * Math.cos(a);
  const back = deg === 180;
  PATHS['angle_' + name] = `<circle cx="12" cy="12" r="6.2"/><path d="M${(12 + 6.2 * Math.sin(a - 0.3)).toFixed(2)} ${(12 + 6.2 * Math.cos(a - 0.3)).toFixed(2)}L${x.toFixed(2)} ${y.toFixed(2)}L${(12 + 6.2 * Math.sin(a + 0.3)).toFixed(2)} ${(12 + 6.2 * Math.cos(a + 0.3)).toFixed(2)}"${back ? ' stroke-dasharray="1.6 1.8"' : ''}/><path d="M12 21.5v.01" stroke-width="2.4"/>`;
}

export type IconName = keyof typeof PATHS | string;

// an <svg> element of the icon (decorative: the button that holds it carries the name)
export function icon(name: IconName, size = 20): SVGSVGElement {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', String(size)); svg.setAttribute('height', String(size));
  svg.setAttribute('aria-hidden', 'true'); svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', 'ico');
  svg.innerHTML = PATHS[name] || '';
  return svg;
}
