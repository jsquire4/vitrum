// overlayGiSignal.ts — GI / frame textures availability panel overlay.

import type { DebuggableEngine } from '../types.js';
import type { FrameMonitor } from './frameMonitor.js';
import { makePanel, makeTitle, makeRow, makeDivider } from './domUtils.js';
import { safeDebugCall, textureAvailability } from './debugUtils.js';

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
