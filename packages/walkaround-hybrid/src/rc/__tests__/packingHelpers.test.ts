import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import type { DDGILight } from '../../ddgi/types.js';
import { packRCLights } from '../packingHelpers.js';

const makeFixtureLights = (count: number): DDGILight[] =>
  Array.from({ length: count }, (_, i) => ({
    kind: 'fixture',
    on: true,
    intensity: 1,
    position: { x: i, y: 0, z: 0 },
  }));

describe('packRCLights diagnostics', () => {
  it('routes fixture-cap truncation through a structured warning sink', () => {
    const warnings: EngineWarning[] = [];
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const packed = packRCLights(makeFixtureLights(17), {
        onWarning: (warning) => warnings.push(warning),
        phase: 'renderFrame',
        method: 'renderFrame',
      });

      expect(new Uint32Array(packed)[0]).toBe(16);
      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toMatchObject({
        code: 'walkaround-hybrid.rc-light-cap-exceeded',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'renderFrame',
        details: {
          activeFixtureCount: 17,
          maxLights: 16,
          droppedLightCount: 1,
          fallback: 'drop-extra-rc-lights',
        },
      });
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('keeps console fallback for standalone helper use', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      packRCLights(makeFixtureLights(17));

      expect(warnSpy).toHaveBeenCalledOnce();
      expect(String(warnSpy.mock.calls[0]?.[0] ?? '')).toContain('active fixtures');
    } finally {
      warnSpy.mockRestore();
    }
  });
});
