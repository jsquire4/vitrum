import { describe, expect, it } from 'vitest';
import { rgbToApproxSpectralCoefficients, evaluateSpectrum } from '../jakobHanika.js';

describe('jakobHanika spectral upsampling placeholder', () => {
  it('produces finite polynomial coefficients from sRGB-corner RGB', () => {
    const c = rgbToApproxSpectralCoefficients(1, 0, 0);
    expect(c).toHaveLength(3);
    expect(Array.from(c).every((x) => Number.isFinite(x))).toBe(true);
  });

  it('evaluates bounded spectrum samples in visible range', () => {
    const c = rgbToApproxSpectralCoefficients(0.2, 0.5, 0.9);
    for (const lambda of [380, 550, 780]) {
      const s = evaluateSpectrum(c, lambda);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});
