/**
 * C3 (2026-05-19) — BdptLightPathBuffer + PTEngineWebGL2.bdptAdvanceFrame.
 *
 * Pre-C3 the host had to: allocate an RGBA32F texture themselves; carry the
 * width = maxLightBounces / height = 3 convention by hand; remember to clear
 * `lightPathTex` on dispose; call `driveForkMaterialUniforms` manually each
 * frame with the texture handle. Nothing in the package made this discoverable
 * or safe.
 *
 * The new helper owns those details. These structural tests pin the public
 * contract so a future refactor can't silently change them.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { BdptLightPathBuffer } from '../legacy/three/bdptLightPathBuffer.js';

describe('BdptLightPathBuffer (C3)', () => {
  it('allocates a width × 3 RGBA32F WebGLRenderTarget with the requested maxLightBounces', () => {
    const buf = new BdptLightPathBuffer({ maxLightBounces: 3 });
    expect(buf.maxLightBounces).toBe(3);
    expect(buf.renderTarget.width).toBe(3);
    expect(buf.renderTarget.height).toBe(3);
    expect(buf.renderTarget.texture.type).toBe(THREE.FloatType);
    expect(buf.renderTarget.texture.format).toBe(THREE.RGBAFormat);
    // Nearest filter — bilinear interpolation of light-vertex data would
    // corrupt the cached path geometry.
    expect(buf.renderTarget.texture.minFilter).toBe(THREE.NearestFilter);
    expect(buf.renderTarget.texture.magFilter).toBe(THREE.NearestFilter);
    buf.dispose();
  });

  it('defaults maxLightBounces to 3 when omitted', () => {
    const buf = new BdptLightPathBuffer();
    expect(buf.maxLightBounces).toBe(3);
    expect(buf.renderTarget.width).toBe(3);
    buf.dispose();
  });

  it('accepts maxLightBounces in [1, 3]', () => {
    for (const max of [1, 2, 3] as const) {
      const buf = new BdptLightPathBuffer({ maxLightBounces: max });
      expect(buf.maxLightBounces).toBe(max);
      expect(buf.renderTarget.width).toBe(max);
      buf.dispose();
    }
  });

  it('rejects maxLightBounces outside [1, 3] — fork hard-cap', () => {
    expect(() => new BdptLightPathBuffer({ maxLightBounces: 0 }))
      .toThrow(/BDPT_MAX_LIGHT_BOUNCES = 3/);
    expect(() => new BdptLightPathBuffer({ maxLightBounces: 4 }))
      .toThrow(/BDPT_MAX_LIGHT_BOUNCES = 3/);
    expect(() => new BdptLightPathBuffer({ maxLightBounces: NaN }))
      .toThrow(/must be in 1\.\.3/);
  });

  it('exposes `.texture` as a direct alias for `.renderTarget.texture`', () => {
    const buf = new BdptLightPathBuffer({ maxLightBounces: 3 });
    expect(buf.texture).toBe(buf.renderTarget.texture);
    expect(buf.texture.name).toBe('vitrum.bdpt.lightPath');
    buf.dispose();
  });

  it('dispose() is idempotent', () => {
    const buf = new BdptLightPathBuffer();
    expect(buf.disposed).toBe(false);
    buf.dispose();
    expect(buf.disposed).toBe(true);
    // Second call must not throw.
    expect(() => buf.dispose()).not.toThrow();
    expect(buf.disposed).toBe(true);
  });
});
