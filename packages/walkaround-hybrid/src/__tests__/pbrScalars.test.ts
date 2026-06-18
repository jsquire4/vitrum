import { describe, expect, it } from 'vitest';

import { extractPbrScalars } from '../pbrScalars.js';

describe('extractPbrScalars', () => {
  it('preserves core MaterialSpec spelling for fallback DDGI material packing', () => {
    const out = extractPbrScalars({
      baseColor: [0.2, 0.4, 0.6],
      roughness: 0.35,
      metallic: 0.75,
      emissive: [0.05, 0.1, 0.2],
      emissiveIntensity: 3,
      transmission: 0.25,
      ior: 1.62,
      attenuationColor: [0.9, 0.8, 0.7],
      attenuationDistance: 3,
      thickness: 0.4,
    });

    expect(out.baseColor).toEqual([0.2, 0.4, 0.6]);
    expect(out.roughness).toBe(0.35);
    expect(out.metallic).toBe(0.75);
    expect(out.emissive).toEqual([0.05, 0.1, 0.2]);
    expect(out.emissiveIntensity).toBe(3);
    expect(out.transmission).toBe(0.25);
    expect(out.ior).toBe(1.62);
    expect(out.attenuationColor).toEqual([0.9, 0.8, 0.7]);
    expect(out.attenuationDistance).toBe(3);
    expect(out.thickness).toBe(0.4);
  });

  it('keeps legacy THREE-style color and metalness extraction stable', () => {
    const out = extractPbrScalars({
      color: { r: 0.7, g: 0.6, b: 0.5 },
      roughness: 0.2,
      metalness: 0.4,
    });

    expect(out.baseColor).toEqual([0.7, 0.6, 0.5]);
    expect(out.roughness).toBe(0.2);
    expect(out.metallic).toBe(0.4);
  });

  it('prefers core metallic over legacy metalness when both are present', () => {
    const out = extractPbrScalars({
      baseColor: [1, 1, 1],
      metallic: 0.8,
      metalness: 0.2,
    });

    expect(out.metallic).toBe(0.8);
  });
});
