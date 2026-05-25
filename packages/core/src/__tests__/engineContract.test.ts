import { describe, expect, it } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  asBackendTexture,
  asBackendTextureFormat,
  narrowToBackendTexture,
  narrowToBackendTextureFormat,
} from '../index.js';
import { asMat4, isMat4 } from '../scene/math.js';

describe('backend promise ledger', () => {
  it('contains all expected backend IDs', () => {
    expect(Object.keys(BACKEND_PROMISE_LEDGER).sort()).toEqual([
      'pt-webgl',
      'pt-webgpu',
      'walkaround-hybrid',
    ]);
  });

  it('keeps incremental patch support exhaustive for every backend', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(typeof rec.incrementalPatchSupport.transform).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.positions).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.material).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.emitter).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.topology).toBe('boolean');
    }
  });
});

describe('mat4 branding', () => {
  it('accepts exactly 16-element matrices', () => {
    const m = asMat4(new Float32Array(16));
    expect(isMat4(m)).toBe(true);
  });

  it('rejects non-16-element arrays', () => {
    expect(() => asMat4(new Float32Array(9))).toThrow(/Mat4 requires 16 elements/);
  });
});

describe('backend texture branding', () => {
  it('round-trips branded backend texture handles through narrow helpers', () => {
    const h = { texture: true as const };
    const branded = asBackendTexture<'webgpu', { texture: true }>(h);
    expect(narrowToBackendTexture<'webgpu', { texture: true }>(branded)).toBe(h);
  });

  it('round-trips branded backend texture formats through narrow helpers', () => {
    const f = 'rgba16float';
    const branded = asBackendTextureFormat<'webgpu', string>(f);
    expect(narrowToBackendTextureFormat<'webgpu', string>(branded)).toBe(f);
  });
});

