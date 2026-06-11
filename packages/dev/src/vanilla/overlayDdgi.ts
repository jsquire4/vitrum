// overlayDdgi.ts — DDGI capabilities panel overlay.

import type { DebuggableEngine } from '../types.js';
import { makePanel, makeTitle, makeRow } from './domUtils.js';
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
