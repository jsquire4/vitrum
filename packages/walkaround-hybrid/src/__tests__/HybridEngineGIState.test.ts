import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import type { DDGI } from '../ddgi/DDGI.js';
import type { GIStateSnapshot } from '../giStateSnapshot.js';
import { importGIStateImpl, type GIStateDeps } from '../HybridEngineGIState.js';

function makeSnapshot(): GIStateSnapshot {
  return {
    dims: { x: 3, y: 3, z: 3 },
    origin: [10, 0, 0],
    spacing: 2,
    irrW: 9,
    irrH: 9,
    visW: 48,
    visH: 48,
    irrData: new Uint16Array(9 * 9 * 4),
    visData: new Uint16Array(48 * 48 * 4),
  };
}

function makeDeps(warnings: EngineWarning[]): GIStateDeps {
  const ddgi = {
    gridParams: {
      dims: { x: 3, y: 3, z: 3 },
      origin: { x: 0, y: 0, z: 0 },
      spacing: 2,
      irradianceAtlasW: 9,
      irradianceAtlasH: 9,
      visibilityAtlasW: 48,
      visibilityAtlasH: 48,
    },
    importAtlasData: vi.fn(() => true),
  } as unknown as DDGI;

  return {
    device: {} as GPUDevice,
    ddgi,
    pipeline: null,
    onWarning: (warning) => warnings.push(warning),
  };
}

describe('HybridEngineGIState import diagnostics', () => {
  it('routes grid-layout mismatch rejection through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const deps = makeDeps(warnings);

    const ok = importGIStateImpl(deps, makeSnapshot());

    expect(ok).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-grid-mismatch',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      details: {
        snapshot: {
          dims: { x: 3, y: 3, z: 3 },
          origin: [10, 0, 0],
          spacing: 2,
        },
        current: {
          dims: { x: 3, y: 3, z: 3 },
          origin: [0, 0, 0],
          spacing: 2,
        },
      },
    });
    expect(warnings[0]!.message).toContain('restore rejected');
    warnSpy.mockRestore();
  });
});
