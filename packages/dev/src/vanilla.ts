// attachDebugOverlays - vanilla (no-React) debug overlay for non-React hosts.
//
// Creates DOM nodes directly on top of the provided canvas container element.
// Returns a dispose() handle for cleanup.

import type {
  FrameInput,
  FrameOutput,
  Scene,
  ScenePrimitive,
} from '@vitrum/core';
import type { DebuggableEngine, FrameStats } from './types.js';

type OverlayId =
  | 'frameTime'
  | 'denoiserToggle'
  | 'ddgiAtlas'
  | 'bvhVisualizer'
  | 'giSignalSplit'
  | 'materialInspector';

const DIAGNOSTIC_OVERLAYS = new Set<OverlayId>([
  'ddgiAtlas',
  'giSignalSplit',
]);

export interface AttachDebugOverlaysOptions {
  /**
   * Which overlays to activate. Defaults to frame time + denoiser diagnostics.
   * Each optional debug overlay renders a truthful disabled/not-ready state
   * when its backend hook is absent.
   */
  overlays?: ReadonlyArray<OverlayId>;
  /**
   * Moving average window size for FrameTimeHUD. Default: 60.
   */
  frameTimeAvgWindow?: number;
  /**
   * Scene snapshot for MaterialInspector. Pass the same object passed to
   * engine.setScene() so picked primitive IDs can be resolved locally.
   */
  scene?: Scene;
  /**
   * Render canvas used for material click-picking. If omitted, the vanilla
   * overlay uses `container` when it is a canvas, or the first canvas inside it.
   */
  canvas?: HTMLCanvasElement | null;
}

export interface DebugOverlaysHandle {
  /** Remove all overlay DOM nodes and unsubscribe from all engine events. */
  dispose(): void;
}

/**
 * Attach debug overlays on top of a canvas element.
 *
 * The `container` element must have `position: relative` (or any non-static
 * position) so that the absolutely-positioned overlay nodes land inside the
 * canvas area. This mirrors how the React components work.
 *
 * @example
 * ```ts
 * const handle = attachDebugOverlays(engine, canvasContainer);
 * // later:
 * handle.dispose();
 * ```
 */
export function attachDebugOverlays(
  engine: DebuggableEngine,
  container: HTMLElement,
  opts: AttachDebugOverlaysOptions = {},
): DebugOverlaysHandle {
  const {
    overlays = ['frameTime', 'denoiserToggle'],
    frameTimeAvgWindow = 60,
    scene,
    canvas,
  } = opts;

  const overlaySet = new Set<OverlayId>(overlays);
  const cleanupFns: Array<() => void> = [];
  const nodes: HTMLElement[] = [];

  const add = (el: HTMLElement): void => {
    container.appendChild(el);
    nodes.push(el);
  };

  const frameMonitor = needsFrameMonitor(overlaySet)
    ? createFrameMonitor(engine)
    : null;
  if (frameMonitor != null) cleanupFns.push(() => frameMonitor.dispose());

  if (overlaySet.has('frameTime')) {
    addFrameTimeHud(engine, add, cleanupFns, frameTimeAvgWindow);
  }

  if (overlaySet.has('denoiserToggle')) {
    addDenoiserDiagnostics(engine, add, cleanupFns);
  }

  if (overlaySet.has('ddgiAtlas')) {
    addDdgiDiagnostics(engine, add, cleanupFns);
  }

  if (overlaySet.has('bvhVisualizer')) {
    addBvhDiagnostics(engine, add, cleanupFns);
  }

  if (overlaySet.has('giSignalSplit')) {
    addGiSignalDiagnostics(engine, add, cleanupFns, frameMonitor);
  }

  if (overlaySet.has('materialInspector')) {
    addMaterialInspectorFallback(engine, container, canvas ?? null, scene, add, cleanupFns);
  }

  return {
    dispose(): void {
      for (const fn of cleanupFns) fn();
      for (const el of nodes) el.parentNode?.removeChild(el);
      nodes.length = 0;
      cleanupFns.length = 0;
    },
  };
}

function needsFrameMonitor(overlays: ReadonlySet<OverlayId>): boolean {
  for (const key of DIAGNOSTIC_OVERLAYS) {
    if (overlays.has(key)) return true;
  }
  return false;
}

function addFrameTimeHud(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
  frameTimeAvgWindow: number,
): void {
  const ring = new NumberRing(frameTimeAvgWindow);
  const div = makePanel({
    top: '8px',
    right: '8px',
    minWidth: '140px',
    pointerEvents: 'none',
  });

  const frameRow = makeRow('frame', '- ms');
  const avgRow = makeRow('avg', '- ms');
  const fpsRow = makeRow('fps', '-');
  div.append(frameRow.el, avgRow.el, fpsRow.el);
  add(div);

  const update = (stats: FrameStats): void => {
    ring.push(stats.frameTimeMs);
    const avg = ring.mean();
    frameRow.setValue(`${stats.frameTimeMs.toFixed(2)} ms`);
    avgRow.setValue(`${avg.toFixed(2)} ms`);
    fpsRow.setValue(avg > 0 ? (1000 / avg).toFixed(1) : '-');
  };

  if (typeof engine.onFrame === 'function') {
    const unsub = engine.onFrame(update);
    cleanupFns.push(unsub);
    return;
  }

  let lastTime: number | null = null;
  let rafId: number | null = null;
  const tick = (now: number): void => {
    if (lastTime !== null) {
      update({ frameTimeMs: now - lastTime });
    }
    lastTime = now;
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
  cleanupFns.push(() => {
    if (rafId !== null) cancelAnimationFrame(rafId);
  });
}

function addDenoiserDiagnostics(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const hasSetter = typeof engine.debug?.setDenoiserEnabled === 'function';
  const hasGetter = typeof engine.debug?.isDenoiserEnabled === 'function';
  let enabled = hasGetter ? engine.debug?.isDenoiserEnabled?.() ?? true : true;

  const badge = makeDiv({
    position: 'absolute',
    top: '8px',
    left: '8px',
    background: 'rgba(0,0,0,0.72)',
    color: '#e0e0e0',
    fontFamily: 'monospace',
    fontSize: '11px',
    padding: '4px 8px',
    borderRadius: '4px',
    userSelect: 'none',
    cursor: hasSetter ? 'pointer' : 'default',
    zIndex: '9998',
  });

  const render = (): void => {
    const color = !hasSetter ? '#ffb347' : enabled ? '#7dfa7d' : '#fa7d7d';
    badge.style.borderLeft = `3px solid ${color}`;
    badge.textContent = hasSetter
      ? `denoiser ${enabled ? 'on' : 'off'} [D]`
      : 'denoiser unavailable';
    badge.title = hasSetter
      ? 'Click or press D to toggle engine.debug.setDenoiserEnabled().'
      : 'This backend does not expose engine.debug.setDenoiserEnabled().';
  };

  const toggle = (): void => {
    if (!hasSetter) return;
    enabled = !enabled;
    engine.debug?.setDenoiserEnabled?.(enabled);
    render();
  };

  render();
  add(badge);

  badge.addEventListener('click', toggle);
  const keyHandler = (e: KeyboardEvent): void => {
    if (e.key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      toggle();
    }
  };
  window.addEventListener('keydown', keyHandler);
  cleanupFns.push(() => {
    badge.removeEventListener('click', toggle);
    window.removeEventListener('keydown', keyHandler);
  });
}

function addDdgiDiagnostics(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const panel = makePanel({
    top: '40px',
    left: '8px',
    minWidth: '230px',
  });
  panel.append(makeTitle('DDGI / capabilities'));

  const debugSurfaceRow = makeRow('debugSurface', '-');
  const auxRow = makeRow('aux buffers', '-');
  const presentationRow = makeRow('present', '-');
  const deviceRow = makeRow('device', '-');
  const irradianceRow = makeRow('irr atlas', '-');
  const visibilityRow = makeRow('vis atlas', '-');
  const memoryRow = makeRow('gpu memory', '-');
  const experimentalRow = makeRow('experimental', '-');
  panel.append(
    debugSurfaceRow.el,
    auxRow.el,
    presentationRow.el,
    deviceRow.el,
    irradianceRow.el,
    visibilityRow.el,
    memoryRow.el,
    experimentalRow.el,
  );
  add(panel);

  const render = (): void => {
    const caps = engine.capabilities;
    debugSurfaceRow.setValue(caps.debugSurface === true ? 'yes' : caps.debugSurface === false ? 'no' : 'omitted');
    auxRow.setValue(caps.supportsAuxBuffers ? 'yes' : 'no');
    presentationRow.setValue(caps.presentationMode ?? 'unspecified');
    deviceRow.setValue(debugValueStatus(
      typeof engine.debug?.device === 'function' ? () => engine.debug?.device?.() ?? null : undefined,
      'device',
    ));
    irradianceRow.setValue(debugValueStatus(
      typeof engine.debug?.atlasTexture === 'function' ? () => engine.debug?.atlasTexture?.() ?? null : undefined,
      'texture',
    ));
    visibilityRow.setValue(debugValueStatus(
      typeof engine.debug?.visibilityAtlasTexture === 'function'
        ? () => engine.debug?.visibilityAtlasTexture?.() ?? null
        : undefined,
      'texture',
    ));

    const memory = safeDebugCall(
      typeof engine.debug?.estimatedGpuMemoryBytes === 'function'
        ? () => engine.debug?.estimatedGpuMemoryBytes?.() ?? null
        : undefined,
    );
    memoryRow.setValue(memory.status === 'ready' && memory.value != null
      ? formatBytes(memory.value.total)
      : memory.status);
    experimentalRow.setValue(formatSet(caps.experimentalFeatures));
  };

  render();
  const interval = setInterval(render, 500);
  cleanupFns.push(() => clearInterval(interval));
}

function addBvhDiagnostics(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const panel = makePanel({
    bottom: '8px',
    right: '8px',
    minWidth: '230px',
  });
  panel.append(makeTitle('BVH structure'));

  const apiRow = makeRow('api', '-');
  const nodesRow = makeRow('nodes', '-');
  const depthRow = makeRow('max depth', '-');
  const avgRow = makeRow('avg depth', '-');
  const histogram = makeDiv({
    display: 'flex',
    alignItems: 'end',
    gap: '1px',
    height: '42px',
    marginTop: '6px',
    borderTop: '1px solid rgba(255,255,255,0.16)',
    paddingTop: '4px',
  });
  panel.append(apiRow.el, nodesRow.el, depthRow.el, avgRow.el, histogram);
  add(panel);

  const render = (): void => {
    const result = safeDebugCall(
      typeof engine.debug?.bvhNodes === 'function'
        ? () => engine.debug?.bvhNodes?.() ?? null
        : undefined,
    );
    apiRow.setValue(result.status === 'unsupported' ? 'unavailable' : result.status);
    if (result.status !== 'ready' || result.value == null) {
      nodesRow.setValue(result.status === 'ready' ? 'not built' : '-');
      depthRow.setValue('-');
      avgRow.setValue('-');
      histogram.replaceChildren(makeMuted('no node table'));
      return;
    }

    const stats = computeBvhStats(result.value);
    nodesRow.setValue(String(stats.nodeCount));
    depthRow.setValue(String(stats.maxDepth));
    avgRow.setValue(stats.avgDepth.toFixed(2));
    renderBvhBars(histogram, stats);
  };

  render();
  const interval = setInterval(render, 500);
  cleanupFns.push(() => clearInterval(interval));
}

function addGiSignalDiagnostics(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
  frameMonitor: FrameMonitor | null,
): void {
  const panel = makePanel({
    bottom: '8px',
    left: '8px',
    minWidth: '250px',
  });
  panel.append(makeTitle('GI / frame textures'));

  const giApiRow = makeRow('gi debug', '-');
  const directRow = makeRow('direct', '-');
  const indirectRow = makeRow('indirect', '-');
  const aoRow = makeRow('ao', '-');
  const totalRow = makeRow('total', '-');
  const frameRow = makeRow('frame', '-');
  const primaryRow = makeRow('primary', '-');
  const normalRow = makeRow('normalDepth', '-');
  const albedoRow = makeRow('albedo', '-');
  const varianceRow = makeRow('variance', '-');
  const motionRow = makeRow('motion', '-');
  panel.append(
    giApiRow.el,
    directRow.el,
    indirectRow.el,
    aoRow.el,
    totalRow.el,
    makeDivider(),
    frameRow.el,
    primaryRow.el,
    normalRow.el,
    albedoRow.el,
    varianceRow.el,
    motionRow.el,
  );
  add(panel);

  const render = (): void => {
    const gi = safeDebugCall(
      typeof engine.debug?.giSignalTextures === 'function'
        ? () => engine.debug?.giSignalTextures?.() ?? null
        : undefined,
    );
    giApiRow.setValue(gi.status === 'ready' ? 'ready' : gi.status);
    const textures = gi.status === 'ready' ? gi.value : null;
    directRow.setValue(textureAvailability(textures?.direct));
    indirectRow.setValue(textureAvailability(textures?.indirect));
    aoRow.setValue(textureAvailability(textures?.ao));
    totalRow.setValue(textureAvailability(textures?.total));

    if (frameMonitor == null || !frameMonitor.supported) {
      frameRow.setValue(frameMonitor?.message ?? 'not observed');
      primaryRow.setValue('-');
      normalRow.setValue('-');
      albedoRow.setValue('-');
      varianceRow.setValue('-');
      motionRow.setValue('-');
      return;
    }

    const frame = frameMonitor.get();
    if (frame == null) {
      frameRow.setValue('waiting');
      primaryRow.setValue('-');
      normalRow.setValue('-');
      albedoRow.setValue('-');
      varianceRow.setValue('-');
      motionRow.setValue('-');
      return;
    }

    frameRow.setValue(`${frame.kind}, spp ${frame.samplesAccumulated}, ${frame.isConverged ? 'converged' : 'active'}`);
    if (frame.kind === 'skipped') {
      primaryRow.setValue('skipped');
      normalRow.setValue('skipped');
      albedoRow.setValue('skipped');
      varianceRow.setValue('skipped');
      motionRow.setValue('skipped');
      return;
    }

    primaryRow.setValue(textureAvailability(frame.primaryRadiance));
    normalRow.setValue(textureAvailability(frame.normalDepth));
    albedoRow.setValue(textureAvailability(frame.albedo));
    varianceRow.setValue(textureAvailability(frame.variance));
    motionRow.setValue(textureAvailability(frame.motionVectors));
  };

  render();
  if (frameMonitor != null) cleanupFns.push(frameMonitor.onFrame(render));
  const interval = setInterval(render, 250);
  cleanupFns.push(() => clearInterval(interval));
}

function addMaterialInspectorFallback(
  engine: DebuggableEngine,
  container: HTMLElement,
  explicitCanvas: HTMLCanvasElement | null,
  scene: Scene | undefined,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const panel = makePanel({
    top: '96px',
    right: '8px',
    minWidth: '250px',
    maxWidth: '300px',
  });
  panel.append(makeTitle('Material inspector'));

  const pickRow = makeRow('pick api', '-');
  const selectedRow = makeRow('selected', '-');
  const kindRow = makeRow('kind', '-');
  const materialRow = makeRow('material', '-');
  const detailsRow = makeRow('details', '-');
  const editRow = makeRow('mat patch', '-');
  panel.append(pickRow.el, selectedRow.el, kindRow.el, materialRow.el, detailsRow.el, editRow.el);
  add(panel);

  const canvas = explicitCanvas ?? findCanvas(container);
  const hasPickApi = typeof engine.debug?.pickPrimitive === 'function';

  const renderSelection = (primitiveId: string | null): void => {
    pickRow.setValue(hasPickApi ? 'ready' : 'unavailable');
    editRow.setValue(materialPatchStatus(engine));

    if (!hasPickApi) {
      selectedRow.setValue('disabled');
      kindRow.setValue('debug.pickPrimitive missing');
      materialRow.setValue(scene == null ? 'pass scene for lookup' : `${scene.primitives.length} scene prims`);
      detailsRow.setValue('-');
      return;
    }
    if (canvas == null) {
      selectedRow.setValue('disabled');
      kindRow.setValue('pass canvas');
      materialRow.setValue(scene == null ? 'pass scene for lookup' : `${scene.primitives.length} scene prims`);
      detailsRow.setValue('-');
      return;
    }
    if (scene == null) {
      selectedRow.setValue(primitiveId ?? 'none');
      kindRow.setValue('no scene snapshot');
      materialRow.setValue('-');
      detailsRow.setValue('-');
      return;
    }
    if (primitiveId == null) {
      selectedRow.setValue('none');
      kindRow.setValue('click canvas');
      materialRow.setValue('-');
      detailsRow.setValue('-');
      return;
    }

    const primitive = scene.primitives.find((p) => p.id === primitiveId);
    selectedRow.setValue(primitiveId);
    if (primitive == null) {
      kindRow.setValue('not in scene');
      materialRow.setValue('-');
      detailsRow.setValue('-');
      return;
    }

    kindRow.setValue(primitive.kind);
    materialRow.setValue(formatMaterial(primitive.material));
    detailsRow.setValue(formatPrimitiveDetails(primitive));
  };

  renderSelection(null);

  if (!hasPickApi || canvas == null) return;

  const onClick = (e: MouseEvent): void => {
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || canvas.width || 1;
    const height = rect.height || canvas.height || 1;
    const x = (e.clientX - rect.left) * ((canvas.width || width) / width);
    const y = (e.clientY - rect.top) * ((canvas.height || height) / height);
    const primitiveId = engine.debug?.pickPrimitive?.(x, y) ?? null;
    renderSelection(primitiveId);
  };
  canvas.addEventListener('click', onClick);
  cleanupFns.push(() => canvas.removeEventListener('click', onClick));
}

interface FrameMonitor {
  readonly supported: boolean;
  readonly message?: string;
  get(): FrameOutput | null;
  onFrame(cb: () => void): () => void;
  dispose(): void;
}

function createFrameMonitor(engine: DebuggableEngine): FrameMonitor {
  const originalRenderFrame = engine.renderFrame;
  let lastFrame: FrameOutput | null = null;
  let installed = false;
  let message = 'ready';
  const listeners = new Set<() => void>();

  const monitoredRenderFrame = function monitoredRenderFrame(
    this: DebuggableEngine,
    input: FrameInput,
  ): FrameOutput {
    const output = originalRenderFrame.call(this, input);
    lastFrame = output;
    for (const listener of listeners) listener();
    return output;
  };

  try {
    (engine as { renderFrame: DebuggableEngine['renderFrame'] }).renderFrame = monitoredRenderFrame;
    installed = true;
  } catch {
    message = 'renderFrame readonly';
  }

  return {
    get supported() { return installed; },
    get message() { return message; },
    get() { return lastFrame; },
    onFrame(cb: () => void): () => void {
      listeners.add(cb);
      return () => { listeners.delete(cb); };
    },
    dispose(): void {
      listeners.clear();
      if (!installed) return;
      const writable = engine as { renderFrame: DebuggableEngine['renderFrame'] };
      if (writable.renderFrame === monitoredRenderFrame) {
        writable.renderFrame = originalRenderFrame;
      }
      installed = false;
    },
  };
}

interface DebugCallResult<T> {
  readonly status: 'unsupported' | 'ready' | 'missing' | 'error';
  readonly value: T | null;
}

function safeDebugCall<T>(fn: (() => T | null) | undefined): DebugCallResult<T> {
  if (typeof fn !== 'function') return { status: 'unsupported', value: null };
  try {
    const value = fn();
    return value == null
      ? { status: 'missing', value: null }
      : { status: 'ready', value };
  } catch {
    return { status: 'error', value: null };
  }
}

function debugValueStatus<T>(fn: (() => T | null) | undefined, noun: string): string {
  const result = safeDebugCall(fn);
  if (result.status === 'unsupported') return `${noun} api unavailable`;
  if (result.status === 'missing') return `${noun} missing`;
  return result.status;
}

interface BvhStats {
  readonly nodeCount: number;
  readonly maxDepth: number;
  readonly avgDepth: number;
  readonly histogram: ReadonlyArray<number>;
}

function computeBvhStats(nodes: Float32Array): BvhStats {
  const nodeCount = Math.floor(nodes.length / 8);
  if (nodeCount === 0) {
    return { nodeCount: 0, maxDepth: 0, avgDepth: 0, histogram: [] };
  }

  let maxDepth = 0;
  let sumDepth = 0;
  const histogram: number[] = [];
  for (let i = 0; i < nodeCount; i++) {
    const depth = Math.max(0, Math.floor(nodes[i * 8 + 6] ?? 0));
    maxDepth = Math.max(maxDepth, depth);
    sumDepth += depth;
    while (histogram.length <= depth) histogram.push(0);
    histogram[depth] = (histogram[depth] ?? 0) + 1;
  }

  return { nodeCount, maxDepth, avgDepth: sumDepth / nodeCount, histogram };
}

function renderBvhBars(target: HTMLElement, stats: BvhStats): void {
  target.replaceChildren();
  if (stats.histogram.length === 0) {
    target.append(makeMuted('empty'));
    return;
  }

  const maxCount = Math.max(...stats.histogram, 1);
  for (let depth = 0; depth < stats.histogram.length; depth++) {
    const count = stats.histogram[depth] ?? 0;
    const bar = makeDiv({
      width: '10px',
      minWidth: '4px',
      height: `${Math.max(2, Math.round((count / maxCount) * 36))}px`,
      background: `hsl(${(depth * 30) % 360}, 80%, 60%)`,
      opacity: count > 0 ? '1' : '0.25',
    });
    bar.title = `depth ${depth}: ${count}`;
    target.append(bar);
  }
}

class NumberRing {
  #values: number[];
  #index = 0;
  #filled = 0;

  constructor(size: number) {
    this.#values = new Array(Math.max(1, size)).fill(0);
  }

  push(value: number): void {
    this.#values[this.#index] = value;
    this.#index = (this.#index + 1) % this.#values.length;
    this.#filled = Math.min(this.#filled + 1, this.#values.length);
  }

  mean(): number {
    if (this.#filled === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.#filled; i++) sum += this.#values[i] ?? 0;
    return sum / this.#filled;
  }
}

function findCanvas(container: HTMLElement): HTMLCanvasElement | null {
  if (typeof HTMLCanvasElement !== 'undefined' && container instanceof HTMLCanvasElement) {
    return container;
  }
  return container.querySelector('canvas');
}

function textureAvailability(value: unknown): string {
  return value == null ? 'missing' : 'ready';
}

function materialPatchStatus(engine: DebuggableEngine): string {
  const support = engine.capabilities.incrementalPatchSupport?.material;
  if (support === true) return 'native';
  if (engine.capabilities.supportsIncrementalScene) return 'check backend';
  return 'unsupported';
}

function formatMaterial(material: ScenePrimitive['material']): string {
  const color = material.baseColor.map((v) => v.toFixed(2)).join(',');
  return `base [${color}] r ${material.roughness.toFixed(2)} m ${material.metallic.toFixed(2)}`;
}

function formatPrimitiveDetails(primitive: ScenePrimitive): string {
  if (primitive.kind === 'analytic') return primitive.shape;
  if (primitive.kind === 'instanced-mesh') return `${primitive.instances.length} instances`;
  const vertexCount = Math.floor(primitive.positions.length / 3);
  const triangleCount = primitive.indices != null
    ? Math.floor(primitive.indices.length / 3)
    : Math.floor(vertexCount / 3);
  if (primitive.kind === 'skinned-mesh') return `${vertexCount} verts, ${primitive.bones.length / 16} bones`;
  return `${vertexCount} verts, ${triangleCount} tris`;
}

function formatSet(values: ReadonlySet<string> | undefined): string {
  if (values == null || values.size === 0) return 'none';
  const entries = Array.from(values);
  return entries.length <= 3 ? entries.join(', ') : `${entries.slice(0, 3).join(', ')} +${entries.length - 3}`;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '-';
  if (bytes < 1024) return `${bytes} B`;
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib.toFixed(1)} KiB`;
  const mib = kib / 1024;
  if (mib < 1024) return `${mib.toFixed(1)} MiB`;
  return `${(mib / 1024).toFixed(2)} GiB`;
}

function makePanel(style: Record<string, string>): HTMLDivElement {
  return makeDiv({
    position: 'absolute',
    background: 'rgba(0,0,0,0.76)',
    color: '#e0e0e0',
    fontFamily: 'monospace',
    fontSize: '11px',
    padding: '8px 10px',
    borderRadius: '4px',
    userSelect: 'none',
    zIndex: '9997',
    lineHeight: '1.55',
    boxSizing: 'border-box',
    ...style,
  });
}

function makeDiv(style: Record<string, string>): HTMLDivElement {
  const el = document.createElement('div');
  Object.assign(el.style, style);
  return el;
}

function makeTitle(text: string): HTMLDivElement {
  const el = document.createElement('div');
  el.style.cssText = 'font-weight:bold;margin-bottom:5px;color:#fff';
  el.textContent = text;
  return el;
}

function makeDivider(): HTMLDivElement {
  return makeDiv({
    borderTop: '1px solid rgba(255,255,255,0.16)',
    margin: '5px 0',
    height: '0',
  });
}

function makeMuted(text: string): HTMLSpanElement {
  const el = document.createElement('span');
  el.style.cssText = 'color:#888;font-style:italic';
  el.textContent = text;
  return el;
}

function makeRow(
  label: string,
  initialValue: string,
): { el: HTMLDivElement; setValue(v: string): void } {
  const el = document.createElement('div');
  el.style.cssText = 'display:flex;justify-content:space-between;gap:10px';
  const labelSpan = document.createElement('span');
  labelSpan.style.cssText = 'color:#888;white-space:nowrap';
  labelSpan.textContent = label;
  const valueSpan = document.createElement('span');
  valueSpan.style.cssText = 'color:#e0e0e0;text-align:right;overflow-wrap:anywhere';
  valueSpan.textContent = initialValue;
  el.append(labelSpan, valueSpan);
  return {
    el,
    setValue(v: string): void {
      valueSpan.textContent = v;
    },
  };
}
