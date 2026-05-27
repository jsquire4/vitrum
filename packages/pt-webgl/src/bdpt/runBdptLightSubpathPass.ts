/**
 * Drive the fork's GPU light-subpath draw pass into a BdptLightPathBuffer target.
 */

import type { WebGLRenderer, WebGLRenderTarget } from 'three';

export interface BdptLightSubpathTracer {
  renderBdptLightSubpathPass(
    lightPathTarget: WebGLRenderTarget,
    maxLightBounces: number,
    frameSeed: number,
  ): void;
}

export function runBdptLightSubpathPass(
  renderer: WebGLRenderer,
  pathTracer: BdptLightSubpathTracer,
  target: WebGLRenderTarget,
  maxLightBounces: number,
  frameSeed: number,
): void {
  const prevRt = renderer.getRenderTarget();
  try {
    pathTracer.renderBdptLightSubpathPass(target, maxLightBounces, frameSeed);
  } finally {
    renderer.setRenderTarget(prevRt);
  }
}
