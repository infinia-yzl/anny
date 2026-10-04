// OpenSculptBoy
// Apache License, Version 2.0
//
// The silhouette of the posed figure as an SVG path, the page's counterpart of opensculptboy.render.flat.silhouette_svg:
// the projected mesh fills a mask, the mask's outline runs along the pixel edges, and Douglas-Peucker simplifies each
// ring.

export type Ring = [number, number][];

// the outlines of a mask (w x h, row-major, true inside) along the pixel edges, each with the inside on its left
export function traceMask(mask: Uint8Array, w: number, h: number): Ring[] {
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] !== 0;
  // directed edges between corners (x, y) in 0..w x 0..h, keyed by their start
  const next = new Map<number, number[]>();
  const key = (x: number, y: number) => y * (w + 1) + x;
  const edge = (x0: number, y0: number, x1: number, y1: number) => {
    const k = key(x0, y0), e = next.get(k);
    if (e) e.push(key(x1, y1)); else next.set(k, [key(x1, y1)]);
  };
  for (let y = 0; y <= h; y++) for (let x = 0; x <= w; x++) {
    // a horizontal edge from (x, y) to (x + 1, y): inside below runs to the right, inside above to the left
    if (x < w) {
      const below = inside(x, y), above = inside(x, y - 1);
      if (below && !above) edge(x + 1, y, x, y); else if (above && !below) edge(x, y, x + 1, y);
    }
    if (y < h) {
      const right = inside(x, y), left = inside(x - 1, y);
      if (right && !left) edge(x, y, x, y + 1); else if (left && !right) edge(x, y + 1, x, y);
    }
  }
  const rings: Ring[] = [];
  for (const [start, ends] of next) {
    while (ends.length) {
      const ring: Ring = [];
      let at = start, to = ends.pop()!;
      ring.push([at % (w + 1), Math.floor(at / (w + 1))]);
      while (to !== start) {
        const out = next.get(to);
        if (!out || !out.length) break;
        at = to; to = out.pop()!;
        ring.push([at % (w + 1), Math.floor(at / (w + 1))]);
      }
      if (ring.length > 2) rings.push(ring);
    }
  }
  return rings;
}

// Douglas-Peucker on a closed ring
export function simplify(ring: Ring, tolerance: number): Ring {
  if (ring.length < 4) return ring;
  // split the ring at its point farthest from the first, then simplify both halves as open lines
  let far = 0, fd = -1;
  for (let i = 1; i < ring.length; i++) { const d = Math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]); if (d > fd) { fd = d; far = i; } }
  const a = open(ring.slice(0, far + 1), tolerance), b = open(ring.slice(far).concat([ring[0]]), tolerance);
  return a.slice(0, -1).concat(b.slice(0, -1));
}
function open(points: Ring, tolerance: number): Ring {
  const keep = new Uint8Array(points.length); keep[0] = keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [i, j] = stack.pop()!;
    const [x0, y0] = points[i], [x1, y1] = points[j], dx = x1 - x0, dy = y1 - y0, l = Math.hypot(dx, dy) || 1;
    let best = -1, bd = tolerance;
    for (let k = i + 1; k < j; k++) {
      const d = Math.abs((points[k][0] - x0) * dy - (points[k][1] - y0) * dx) / l;
      if (d > bd) { bd = d; best = k; }
    }
    if (best >= 0) { keep[best] = 1; stack.push([i, best], [best, j]); }
  }
  return points.filter((_, k) => keep[k]);
}

// an SVG path of rings, scaled by k (even-odd fill keeps the holes)
export function svgPath(rings: Ring[], k = 1): string {
  const f = (v: number) => String(Math.round(v * k * 10) / 10);
  return rings.map((r) => 'M' + r.map(([x, y]) => `${f(x)} ${f(y)}`).join('L') + 'Z').join('');
}

// the silhouette of projected quads (points px: x right, y down, in pixels of a w x h picture) as a standalone SVG
export function silhouetteSvg(px: Float64Array, quads: Uint32Array, w: number, h: number, fill: string, desc: string, supersample = 3): string {
  const W = w * supersample, H = h * supersample;
  const canvas = new OffscreenCanvas(W, H), ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  // one fill per quad: the front and the back of the body overlap with opposite windings, which one path would cancel
  for (let q = 0; q < quads.length; q += 4) {
    ctx.beginPath();
    ctx.moveTo(px[quads[q] * 2] * supersample, px[quads[q] * 2 + 1] * supersample);
    for (let k = 1; k < 4; k++) ctx.lineTo(px[quads[q + k] * 2] * supersample, px[quads[q + k] * 2 + 1] * supersample);
    ctx.fill();
  }
  const data = ctx.getImageData(0, 0, W, H).data, mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) mask[i] = data[i * 4 + 3] > 127 ? 1 : 0;
  const rings = traceMask(mask, W, H).map((r) => simplify(r, 0.6 * supersample)).filter((r) => r.length > 2);
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">\n` +
    `  <title>OpenSculptBoy</title>\n  <desc>${esc(desc)}</desc>\n` +
    `  <path fill="${fill}" fill-rule="evenodd" d="${svgPath(rings, 1 / supersample)}"/>\n</svg>\n`;
}
