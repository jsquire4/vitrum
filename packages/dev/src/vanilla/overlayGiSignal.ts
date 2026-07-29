// overlayGiSignal.ts — GI / frame textures availability panel overlay.

import type { DebuggableEngine } from '../types.js';
import { startGpuTextureBlit } from '../react/gpuTextureBlit.js';
import type { FrameMonitor } from './frameMonitor.js';
import { makePanel, makeTitle, makeRow, makeDivider } from './domUtils.js';
import { safeDebugCall, textureAvailability } from './debugUtils.js';

const GI_CHANNELS = ['direct', 'indirect', 'ao', 'total'] as const;
type GiChannel = (typeof GI_CHANNELS)[number];

export function addGiSignalDiagnostics(
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

  const previewGrid = document.createElement('div');
  Object.assign(previewGrid.style, {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gridTemplateRows: '72px 72px',
    gap: '4px',
    marginTop: '6px',
  });
  const previewCanvases = {} as Record<GiChannel, HTMLCanvasElement>;
  for (const channel of GI_CHANNELS) {
    const preview = document.createElement('div');
    Object.assign(preview.style, {
      position: 'relative',
      minWidth: '0',
      overflow: 'hidden',
      background: 'rgba(255,255,255,0.04)',
    });
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', `${channel} GI signal preview`);
    Object.assign(canvas.style, {
      width: '100%',
      height: '100%',
      objectFit: 'contain',
      imageRendering: 'pixelated',
    });
    const label = document.createElement('span');
    label.textContent = channel === 'ao' ? 'AO' : channel;
    Object.assign(label.style, {
      position: 'absolute',
      top: '2px',
      left: '3px',
      color: '#ffb347',
      fontSize: '10px',
      textShadow: '0 0 3px #000',
      pointerEvents: 'none',
    });
    preview.append(canvas, label);
    previewGrid.append(preview);
    previewCanvases[channel] = canvas;
  }
  panel.append(previewGrid);
  add(panel);

  let blitDevice: GPUDevice | null = null;
  const blitTextures: Record<GiChannel, GPUTexture | null> = {
    direct: null,
    indirect: null,
    ao: null,
    total: null,
  };
  const stopBlits: Partial<Record<GiChannel, () => void>> = {};

  const syncSignalBlits = (
    device: GPUDevice | null,
    textures: {
      readonly direct: GPUTexture | null;
      readonly indirect: GPUTexture | null;
      readonly ao: GPUTexture | null;
      readonly total: GPUTexture | null;
    } | null,
  ): void => {
    for (const channel of GI_CHANNELS) {
      const texture = textures?.[channel] ?? null;
      if (device === blitDevice && texture === blitTextures[channel]) continue;
      stopBlits[channel]?.();
      delete stopBlits[channel];
      blitTextures[channel] = texture;
      if (device != null && texture != null) {
        stopBlits[channel] = startGpuTextureBlit(
          previewCanvases[channel],
          device,
          texture,
          { throttleMs: 100, label: `vanilla-gi-${channel}` },
        );
      }
    }
    blitDevice = device;
  };

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
    const device = safeDebugCall(
      typeof engine.debug?.device === 'function'
        ? () => engine.debug?.device?.() ?? null
        : undefined,
    );
    syncSignalBlits(
      device.status === 'ready' ? device.value : null,
      textures,
    );

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
  cleanupFns.push(() => {
    for (const channel of GI_CHANNELS) {
      stopBlits[channel]?.();
      delete stopBlits[channel];
    }
  });
}
