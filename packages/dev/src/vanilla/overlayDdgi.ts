// overlayDdgi.ts — DDGI capabilities panel overlay.

import type { DebuggableEngine } from '../types.js';
import { startGpuTextureBlit } from '../react/gpuTextureBlit.js';
import { makePanel, makeTitle, makeRow, makeDiv } from './domUtils.js';
import { safeDebugCall, debugValueStatus, formatSet, formatBytes } from './debugUtils.js';

export function addDdgiDiagnostics(
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
  const activeRow = makeRow('active', '-');
  panel.append(
    debugSurfaceRow.el,
    auxRow.el,
    presentationRow.el,
    deviceRow.el,
    irradianceRow.el,
    visibilityRow.el,
    memoryRow.el,
    activeRow.el,
  );

  const atlasGrid = makeDiv({
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '6px',
    marginTop: '6px',
  });
  const makeAtlasPreview = (label: string): {
    readonly root: HTMLDivElement;
    readonly canvas: HTMLCanvasElement;
  } => {
    const root = makeDiv({ minWidth: '0' });
    const caption = makeDiv({
      color: '#aaa',
      fontSize: '10px',
      marginBottom: '2px',
    });
    caption.textContent = label;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', `${label} DDGI atlas preview`);
    Object.assign(canvas.style, {
      display: 'block',
      width: '108px',
      maxWidth: '100%',
      height: '72px',
      objectFit: 'contain',
      imageRendering: 'pixelated',
      background: 'rgba(255,255,255,0.04)',
    });
    root.append(caption, canvas);
    return { root, canvas };
  };
  const irradiancePreview = makeAtlasPreview('irradiance');
  const visibilityPreview = makeAtlasPreview('visibility');
  atlasGrid.append(irradiancePreview.root, visibilityPreview.root);
  panel.append(atlasGrid);
  add(panel);

  let blitDevice: GPUDevice | null = null;
  let blitIrradiance: GPUTexture | null = null;
  let blitVisibility: GPUTexture | null = null;
  let stopIrradiance: (() => void) | null = null;
  let stopVisibility: (() => void) | null = null;

  const syncAtlasBlits = (
    device: GPUDevice | null,
    irradiance: GPUTexture | null,
    visibility: GPUTexture | null,
  ): void => {
    if (device !== blitDevice || irradiance !== blitIrradiance) {
      stopIrradiance?.();
      stopIrradiance = null;
      blitIrradiance = irradiance;
      if (device != null && irradiance != null) {
        stopIrradiance = startGpuTextureBlit(
          irradiancePreview.canvas,
          device,
          irradiance,
          { throttleMs: 100, label: 'vanilla-ddgi-irr-atlas' },
        );
      }
    }
    if (device !== blitDevice || visibility !== blitVisibility) {
      stopVisibility?.();
      stopVisibility = null;
      blitVisibility = visibility;
      if (device != null && visibility != null) {
        stopVisibility = startGpuTextureBlit(
          visibilityPreview.canvas,
          device,
          visibility,
          { throttleMs: 100, label: 'vanilla-ddgi-vis-atlas' },
        );
      }
    }
    blitDevice = device;
  };

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

    const device = safeDebugCall(
      typeof engine.debug?.device === 'function'
        ? () => engine.debug?.device?.() ?? null
        : undefined,
    );
    const irradiance = safeDebugCall(
      typeof engine.debug?.atlasTexture === 'function'
        ? () => engine.debug?.atlasTexture?.() ?? null
        : undefined,
    );
    const visibility = safeDebugCall(
      typeof engine.debug?.visibilityAtlasTexture === 'function'
        ? () => engine.debug?.visibilityAtlasTexture?.() ?? null
        : undefined,
    );
    syncAtlasBlits(
      device.status === 'ready' ? device.value : null,
      irradiance.status === 'ready' ? irradiance.value : null,
      visibility.status === 'ready' ? visibility.value : null,
    );

    const memory = safeDebugCall(
      typeof engine.debug?.estimatedGpuMemoryBytes === 'function'
        ? () => engine.debug?.estimatedGpuMemoryBytes?.() ?? null
        : undefined,
    );
    memoryRow.setValue(memory.status === 'ready' && memory.value != null
      ? formatBytes(memory.value.total)
      : memory.status);
    activeRow.setValue(formatSet(caps.activeFeatures));
  };

  render();
  const interval = setInterval(render, 500);
  cleanupFns.push(() => clearInterval(interval));
  cleanupFns.push(() => {
    stopIrradiance?.();
    stopVisibility?.();
    stopIrradiance = null;
    stopVisibility = null;
  });
}
