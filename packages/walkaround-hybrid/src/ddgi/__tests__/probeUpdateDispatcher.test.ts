import { describe, expect, it } from 'vitest';
import { packProbeUpdateBorderUbo } from '../probeUpdateDispatcher.js';
import { DDGI_BORDER_UBO_BYTES } from '../probeUpdateUbos.js';

describe('probeUpdateDispatcher', () => {
  it('packProbeUpdateBorderUbo is 32 bytes', () => {
    const buf = packProbeUpdateBorderUbo({
      probeCount: 4,
      atlasWidth: 128,
      atlasHeight: 64,
      gridDimX: 2,
      gridDimY: 2,
      gridDimZ: 1,
    });
    expect(buf.byteLength).toBe(DDGI_BORDER_UBO_BYTES);
    const u32 = new Uint32Array(buf);
    expect(u32[0]).toBe(4);
    expect(u32[1]).toBe(128);
    expect(u32[2]).toBe(64);
    expect(u32[4]).toBe(2);
    expect(u32[5]).toBe(2);
    expect(u32[6]).toBe(1);
  });
});
