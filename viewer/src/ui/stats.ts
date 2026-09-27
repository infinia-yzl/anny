// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The performance card: the frame rate and a graph of the frame times, the CPU and GPU time of a frame (the GPU time
// from EXT_disjoint_timer_query_webgl2 where the browser has it), the render scale and the refinement, the draw calls,
// the memory, the hair, the device and the load times, a 10-second benchmark, and a compact HUD. The graphs are SVG,
// so the page keeps one canvas (the screenshot tools select it).

import type { App } from './app.ts';
import { copyText, h, iconButton, saveUi, uiState } from './dom.ts';
import { toast } from './toast.ts';

const N = 240;
const quantile = (a: number[], q: number) => { if (!a.length) return 0; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const mb = (b: number) => `${(b / 1048576).toFixed(b > 104857600 ? 0 : 1)} MB`;

export interface Stats {
  el: HTMLElement;
  passBegin(): void;
  passEnd(): void;
  frame(now: number, dt: number, drew: boolean, jsMs: number): void;
  open(): void;
  close(): void;
  toggle(): void;
  isOpen(): boolean;
  snapshot(): Record<string, any>;
  benchmark(): void;
  onToggle: (open: boolean) => void;
}

export function makeStats(app: App): Stats {
  const { renderer } = app;
  const gl = renderer.getContext() as WebGL2RenderingContext;
  const tq: any = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  let gpuName = '';
  try {
    const dbg: any = gl.getExtension('WEBGL_debug_renderer_info');
    gpuName = String(gl.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  } catch (e) { gpuName = 'unknown'; }
  // ANGLE reports 'ANGLE (vendor, device ..., driver)': keep the device, without the API and the ids
  const gpuFull = gpuName;
  const m = /^ANGLE \((.*)\)$/.exec(gpuName);
  if (m) {
    const parts: string[] = []; let depth = 0, cur = '';
    for (const ch of m[1]) { if (ch === '(') depth++; if (ch === ')') depth--; if (ch === ',' && depth === 0) { parts.push(cur.trim()); cur = ''; } else cur += ch; }
    parts.push(cur.trim());
    gpuName = (parts[1] || parts[0]).replace(/^ANGLE Metal Renderer:\s*/, '').replace(/\s*Direct3D.*$/, '').replace(/\s*\(0x[0-9a-f]+\)/gi, '')
      .replace(/^Vulkan [\d.]+ \((.*)\)$/, '$1').trim() || gpuFull;
  }

  // ---------------------------------------------------------------- measurements
  const dts: number[] = [], cpu: number[] = [], gpu: number[] = [];
  let lastDraw = 0, draws = 0, tris = 0, gpuLast = 0, pending: WebGLQuery[] = [], active: WebGLQuery | null = null;
  let bench: null | { t0: number; dts: number[]; gpu: number[]; scale: number[]; turn: boolean } = null;
  const push = (a: number[], v: number) => { a.push(v); if (a.length > N) a.shift(); };
  const timing = () => !!tq && (S.isOpen() || !!bench);

  // ---------------------------------------------------------------- the card
  const fps = h('span.st-big'), fpsUnit = h('span.st-unit', { text: 'fps' });
  const state = h('span.st-state');
  const ms = h('span.st-v'), gpuMs = h('span.st-v'), p95 = h('span.st-v'), slow = h('span.st-v');
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${N} 60`); svg.setAttribute('preserveAspectRatio', 'none'); svg.setAttribute('class', 'st-graph');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Frame times of the last 240 frames');
  // guides at 16.7 ms (60 fps) and 33.3 ms (30 fps) on a scale of 0 to 50 ms
  const guide = (msv: number, cls: string) => { const l = document.createElementNS(ns, 'line'); const y = 60 - msv / 50 * 60; l.setAttribute('x1', '0'); l.setAttribute('x2', String(N)); l.setAttribute('y1', String(y)); l.setAttribute('y2', String(y)); l.setAttribute('class', cls); svg.append(l); };
  guide(16.7, 'st-g60'); guide(33.3, 'st-g30');
  const line = document.createElementNS(ns, 'polyline'); line.setAttribute('class', 'st-line'); svg.append(line);
  const gline = document.createElementNS(ns, 'polyline'); gline.setAttribute('class', 'st-gline'); svg.append(gline);
  const legend = h('div.st-legend', {}, h('span.k-cpu', { text: 'Frame' }), h('span.k-gpu', { text: 'GPU' }), h('span.k-g', { text: '60 and 30 fps' }));
  const rows = (pairs: [string, HTMLElement][]) => h('dl.st-grid', {}, ...pairs.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', {}, v)]));
  const v = () => h('span');
  const R: Record<string, HTMLElement> = {};
  for (const k of ['scale', 'res', 'samples', 'draws', 'tris', 'hair', 'lod', 'hverts', 'phys', 'bodyms', 'geo', 'heap', 'gpu', 'pr', 'shadow', 'sss', 'quality', 'cap', 'canvas']) R[k] = v();
  const load = h('dl.st-grid');
  const benchOut = h('div.st-bench', { hidden: true });
  const benchBtn = h('button.tb', { type: 'button', text: 'Run benchmark (10 s)' });
  benchBtn.addEventListener('click', () => S.benchmark());
  const copyBtn = h('button.tb', { type: 'button', text: 'Copy stats' });
  copyBtn.addEventListener('click', async () => { const t = JSON.stringify((window as any).frameStats ? (window as any).frameStats() : S.snapshot(), null, 2); toast(await copyText(t) ? 'Stats copied' : 'The clipboard is not available here', 'info'); });
  const body = h('div.st-body', {},
    h('div.st-top', {}, h('div.st-fps', {}, fps, fpsUnit), state),
    rows([['Frame', ms], ['GPU', gpuMs], ['p95 frame', p95], ['Over 33 ms', slow]]),
    svg, legend,
    h('h4', { text: 'Picture' }), rows([['Render scale', R.scale], ['Resolution', R.res], ['Refinement', R.samples], ['Draw calls', R.draws], ['Triangles', R.tris]]),
    h('h4', { text: 'Scene' }), rows([['Hair strands', R.hair], ['Hair detail', R.lod], ['Hair vertices', R.hverts], ['Hair physics', R.phys], ['Body update', R.bodyms], ['GPU memory', R.geo], ['JS heap', R.heap]]),
    h('h4', { text: 'Device' }), rows([['GPU', R.gpu], ['Pixel ratio', R.pr], ['Canvas', R.canvas], ['Shadow map', R.shadow], ['Skin diffusion', R.sss], ['Quality', R.quality], ['Frame limit', R.cap]]),
    h('h4', { text: 'Load' }), load,
    h('div.btn-row', {}, benchBtn, copyBtn), benchOut);
  const hudFps = h('span.hud-fps'), hudMs = h('span.hud-ms');
  const hsvg = svg.cloneNode(false) as SVGSVGElement; hsvg.setAttribute('class', 'st-graph hud-graph');
  const hline = line.cloneNode(false) as SVGPolylineElement; hsvg.append(hline);
  const hud = h('button.st-hud', { type: 'button', 'aria-label': 'Expand the performance card' }, hudFps, hudMs, hsvg);
  const title = h('div.st-title', {}, h('h2', { text: 'Performance' }));
  const collapse = iconButton('hud', 'Collapse to a HUD', { tip: 'down' });
  const closeBtn = iconButton('close', 'Close', { tip: 'down', key: '`' });
  const head = h('header.st-head', {}, title, collapse, closeBtn);
  const el = h('section.stats', { id: 'stats', 'aria-label': 'Performance', hidden: true }, head, body, hud);
  let compact = !!uiState().statsCompact;
  const paintMode = () => { el.classList.toggle('compact', compact); };
  collapse.addEventListener('click', () => { compact = true; saveUi({ statsCompact: true }); paintMode(); });
  hud.addEventListener('click', () => { compact = false; saveUi({ statsCompact: false }); paintMode(); });
  closeBtn.addEventListener('click', () => S.close());
  // drag the card by its header
  let drag: null | { x: number; y: number; l: number; t: number } = null;
  head.addEventListener('pointerdown', (e) => {
    if ((e.target as HTMLElement).closest('button')) return;
    const r = el.getBoundingClientRect();
    drag = { x: e.clientX, y: e.clientY, l: r.left, t: r.top }; head.setPointerCapture(e.pointerId);
  });
  head.addEventListener('pointermove', (e) => {
    if (!drag) return;
    el.style.left = Math.max(4, Math.min(innerWidth - 80, drag.l + e.clientX - drag.x)) + 'px';
    el.style.top = Math.max(4, Math.min(innerHeight - 40, drag.t + e.clientY - drag.y)) + 'px';
    el.style.right = 'auto';
  });
  head.addEventListener('pointerup', () => { drag = null; });
  paintMode();

  const points = (a: number[]) => a.map((d, i) => `${i + N - a.length},${(60 - Math.min(50, d) / 50 * 60).toFixed(1)}`).join(' ');
  let tText = 0, tGraph = 0;
  function paint(now: number) {
    const idle = now - lastDraw > 500;
    const recent = dts.slice(-60);
    const mean = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
    const f = idle || !mean ? 0 : 1000 / mean;
    fps.textContent = idle ? '–' : f.toFixed(0);
    state.textContent = idle ? 'Idle · picture refined' : bench ? `Benchmark · ${Math.ceil(10 - (now - bench.t0) / 1000)} s` : app.fpsCap() ? `Drawing · limit ${app.fpsCap()} fps` : 'Drawing';
    state.dataset.kind = idle ? 'idle' : 'on';
    const c = cpu.slice(-60), cm = c.length ? c.reduce((a, b) => a + b, 0) / c.length : 0;
    ms.textContent = `${mean.toFixed(1)} ms · CPU ${cm.toFixed(1)} ms`;
    gpuMs.textContent = !tq ? 'not supported' : gpu.length ? `${quantile(gpu.slice(-60), 0.5).toFixed(2)} ms` : 'measuring';
    p95.textContent = `${quantile(dts, 0.95).toFixed(1)} ms`;
    slow.textContent = `${dts.filter((d) => d > 33.4).length} of ${dts.length}`;
    hudFps.textContent = idle ? 'idle' : `${f.toFixed(0)} fps`;
    hudMs.textContent = idle ? 'refined' : `${mean.toFixed(1)} ms`;
    const sc = app.scale(), sm = app.samples();
    R.scale.textContent = `${Math.round(sc.s * 100)} %`;
    R.res.textContent = `${sc.w} × ${sc.h}`;
    R.samples.textContent = `${Math.min(sm.n, sm.max)} of ${sm.max} samples`;
    R.draws.textContent = String(draws); R.tris.textContent = tris.toLocaleString();
    const hs = app.hairStats();
    if (hs) {
      R.hair.textContent = hs.strands.toLocaleString();
      R.lod.textContent = `${Math.round(hs.lod * 100)} %`;
      R.hverts.textContent = `${(hs.vertices / 1e6).toFixed(2)} M per draw`;
      R.phys.textContent = hs.physics ? (hs.asleep ? 'asleep' : `${(hs.sim?.ms ?? 0).toFixed(2)} ms`) : 'off';
    }
    const bt = app.bodyTiming();
    R.bodyms.textContent = bt.ms ? `${bt.ms.toFixed(1)} ms (${bt.total.toFixed(1)} ms in all)` : '–';
    const mem = (renderer.info as any).memory;
    R.geo.textContent = `${mem.geometries} geometries, ${mem.textures} textures${hs ? ', hair ' + mb(hs.gpu_bytes) : ''}`;
    const pm = (performance as any).memory;
    R.heap.textContent = pm ? `${mb(pm.usedJSHeapSize)} of ${mb(pm.jsHeapSizeLimit)}` : 'not reported';
    R.gpu.textContent = gpuName; R.gpu.title = gpuFull;
    R.pr.textContent = String(app.pixelRatio());
    R.canvas.textContent = `${renderer.domElement.width} × ${renderer.domElement.height}`;
    R.shadow.textContent = `${app.shadowSize()} px`;
    R.sss.textContent = app.sssMode();
    R.quality.textContent = app.quality();
    R.cap.textContent = app.fpsCap() ? `${app.fpsCap()} fps` : 'display rate';
    if (!load.childElementCount) {
      const L = app.loadTimes();
      for (let i = 0; i + 1 < L.length; i++) load.append(h('dt', { text: L[i][0] || 'Other' }), h('dd', { text: `${Math.round(L[i + 1][1] - L[i][1])} ms` }));
      if (L.length > 1) load.append(h('dt', { text: 'Total' }), h('dd', { text: `${Math.round(L[L.length - 1][1] - L[0][1])} ms` }));
    }
  }

  const S: Stats = {
    el,
    onToggle: () => {},
    passBegin() {
      (renderer.info as any).reset();
      if (timing() && !active && pending.length < 4) {
        active = gl.createQuery();
        if (active) gl.beginQuery(tq.TIME_ELAPSED_EXT, active);
      }
    },
    passEnd() {
      draws = renderer.info.render.calls; tris = renderer.info.render.triangles;
      if (active) { gl.endQuery(tq.TIME_ELAPSED_EXT); pending.push(active); active = null; }
    },
    frame(now, dt, drew, jsMs) {
      if (drew) {
        lastDraw = now;
        push(dts, dt * 1000); push(cpu, jsMs);
        if (bench) { bench.dts.push(dt * 1000); bench.scale.push(app.scale().s); }
      }
      if (pending.length) {
        const disjoint = gl.getParameter(tq.GPU_DISJOINT_EXT);
        while (pending.length && gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)) {
          const q = pending.shift()!;
          if (!disjoint) { gpuLast = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6; push(gpu, gpuLast); if (bench) bench.gpu.push(gpuLast); }
          gl.deleteQuery(q);
        }
      }
      if (bench && now - bench.t0 > 10000) finishBench();
      if (el.hidden) return;
      if (now - tGraph > 100) { tGraph = now; line.setAttribute('points', points(dts)); gline.setAttribute('points', points(gpu)); hline.setAttribute('points', points(dts.slice(-N))); }
      if (now - tText > 250) { tText = now; paint(now); }
    },
    open() { el.hidden = false; tText = 0; S.onToggle(true); saveUi({ stats: true }); paint(performance.now()); },
    close() { el.hidden = true; S.onToggle(false); saveUi({ stats: false }); },
    toggle() { if (el.hidden) S.open(); else S.close(); },
    isOpen: () => !el.hidden,
    snapshot() {
      return { gpu_ms: gpu.length ? +quantile(gpu.slice(-60), 0.5).toFixed(3) : null, p95_ms: +quantile(dts, 0.95).toFixed(2), draws, triangles: tris,
        samples: app.samples().n, gpu: gpuName, quality: app.quality() };
    },
    benchmark() {
      if (bench) return;
      bench = { t0: performance.now(), dts: [], gpu: [], scale: [], turn: app.turntable() };
      app.setTurntable(true);
      benchBtn.disabled = true; benchBtn.textContent = 'Running…';
      benchOut.hidden = true;
      toast('Benchmark: 10 seconds of turntable');
    },
  };
  function finishBench() {
    const b = bench!; bench = null;
    app.setTurntable(b.turn);
    benchBtn.disabled = false; benchBtn.textContent = 'Run benchmark (10 s)';
    const n = b.dts.length, mean = n ? b.dts.reduce((a, c) => a + c, 0) / n : 0;
    const res = {
      frames: n, fps: mean ? +(1000 / mean).toFixed(1) : 0, p50_ms: +quantile(b.dts, 0.5).toFixed(2), p95_ms: +quantile(b.dts, 0.95).toFixed(2),
      p99_ms: +quantile(b.dts, 0.99).toFixed(2), gpu_ms: b.gpu.length ? +quantile(b.gpu, 0.5).toFixed(2) : null, fps_cap: app.fpsCap(),
      scale: b.scale.length ? +(b.scale.reduce((a, c) => a + c, 0) / b.scale.length).toFixed(2) : 1, quality: app.quality(), gpu: gpuName,
    };
    benchOut.hidden = false;
    benchOut.replaceChildren(h('h4', { text: 'Benchmark result' }),
      rows([['Average', h('span', { text: `${res.fps} fps` })], ['Median frame', h('span', { text: `${res.p50_ms} ms` })], ['p95 / p99', h('span', { text: `${res.p95_ms} / ${res.p99_ms} ms` })],
        ['GPU median', h('span', { text: res.gpu_ms === null ? 'not supported' : `${res.gpu_ms} ms` })], ['Mean scale', h('span', { text: `${Math.round(res.scale * 100)} %` })]]),
      h('button.tb', { type: 'button', text: 'Copy result', on: { click: async () => toast(await copyText(JSON.stringify(res, null, 2)) ? 'Result copied' : 'The clipboard is not available here') } }));
    (window as any).__benchmark = res;
    toast(`Benchmark: ${res.fps} fps, p95 ${res.p95_ms} ms`);
  }
  return S;
}
