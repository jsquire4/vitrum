import { describe, expect, it } from 'vitest';
import { demodulateAlbedo, remodulateAlbedo } from '../src/index.js';

describe('@vitrum/shared-denoisers barrel exports', () => {
  it('exports canonical albedo modulation helpers', () => {
    const color = new Float32Array([0.4, 0.2, 0.1]);
    const albedo = new Float32Array([0.5, 0.25, 0.2]);
    const demodulated = demodulateAlbedo(color, albedo, 1);
    const remodulated = remodulateAlbedo(demodulated, albedo, 1);

    expect(remodulated[0]).toBeCloseTo(color[0]!);
    expect(remodulated[1]).toBeCloseTo(color[1]!);
    expect(remodulated[2]).toBeCloseTo(color[2]!);
  });
});
