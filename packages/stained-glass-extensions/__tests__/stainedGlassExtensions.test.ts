import { describe, expect, it, vi } from 'vitest';
import {
  SURFACE_TEXTURE_ID,
  VITRUM_USER_DATA_KEYS,
  packCameUBO,
} from '../src/index.js';

describe('@vitrum/stained-glass-extensions', () => {
  it('keeps surface texture IDs stable and sequential', () => {
    expect(SURFACE_TEXTURE_ID).toEqual({
      smooth: 0,
      hammered: 1,
      ripple: 2,
      granite: 3,
      baroque: 4,
      waterglass: 5,
      catspaw: 6,
      flemish: 7,
    });
  });

  it('exports the canonical Three.js userData bridge keys', () => {
    expect(VITRUM_USER_DATA_KEYS.DISPERSION_ABBE).toBe('vitrumDispersionAbbeNumber');
    expect(VITRUM_USER_DATA_KEYS.SCATTERING_RGB).toBe('vitrumScatteringCoefficientRGB');
    expect(VITRUM_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT).toBe('vitrumDichroicTransmittanceLUT');
  });

  it('packs came segments and nodes into std140-friendly lanes', () => {
    const packed = packCameUBO(
      [{
        startWorld: [1, 2, 3],
        endWorld: [4, 5, 6],
        railWidth: 0.25,
        blockHeight: 0.5,
        webThickness: 0.125,
      }],
      [{
        position: [7, 8, 9],
        radius: 0.75,
      }],
    );

    expect(packed.segmentCount).toBe(1);
    expect(packed.nodeCount).toBe(1);
    expect(Array.from(packed.segments.slice(0, 9))).toEqual([
      1, 2, 3, 0.25,
      4, 5, 6, 0.5,
      0.125,
    ]);
    expect(Array.from(packed.nodes)).toEqual([7, 8, 9, 0.75]);
  });

  it('truncates came uploads to host-provided caps and warns once per capped input', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const packed = packCameUBO(
        [
          { startWorld: [0, 0, 0], endWorld: [1, 0, 0], railWidth: 1, blockHeight: 2, webThickness: 3 },
          { startWorld: [0, 1, 0], endWorld: [1, 1, 0], railWidth: 4, blockHeight: 5, webThickness: 6 },
        ],
        [
          { position: [0, 0, 0], radius: 1 },
          { position: [1, 1, 1], radius: 2 },
        ],
        { maxSegments: 1, maxNodes: 1 },
      );

      expect(packed.segmentCount).toBe(1);
      expect(packed.nodeCount).toBe(1);
      expect(warn).toHaveBeenCalledTimes(2);
    } finally {
      warn.mockRestore();
    }
  });
});
