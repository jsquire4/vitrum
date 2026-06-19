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

function makeMatchingSnapshot(extra: Partial<GIStateSnapshot> = {}): GIStateSnapshot {
  return {
    ...makeSnapshot(),
    origin: [0, 0, 0],
    ...extra,
  };
}

function makeRestirGISnapshot(): NonNullable<GIStateSnapshot['restirGI']> {
  return {
    halfW: 2,
    halfH: 2,
    strideU32: 30,
    current: new Uint32Array(2 * 2 * 30),
    previous: new Uint32Array(2 * 2 * 30),
    spatial: new Uint32Array(2 * 2 * 30),
  };
}

function makePpgSnapshot(): NonNullable<GIStateSnapshot['ppg']> {
  return {
    maxSpatialCells: 128,
    maxDTreeNodesPerCell: 64,
    sTreeBuf: new Float32Array(16),
    dTreeBuf: new Float32Array(64),
    dTreeOffsets: new Uint32Array([0]),
    sceneBoundsMin: [-1, -1, -1],
    sceneBoundsMax: [1, 1, 1],
  };
}

function makeDeps(
  warnings: EngineWarning[],
  options: {
    readonly importAtlasData?: () => boolean;
    readonly pipeline?: GIStateDeps['pipeline'];
  } = {},
): GIStateDeps {
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
    importAtlasData: vi.fn(options.importAtlasData ?? (() => true)),
  } as unknown as DDGI;

  return {
    device: {} as GPUDevice,
    ddgi,
    pipeline: options.pipeline ?? null,
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

  it('routes DDGI atlas restore rejection through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const deps = makeDeps(warnings, { importAtlasData: () => false });

    const ok = importGIStateImpl(deps, makeMatchingSnapshot());

    expect(ok).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-atlas-rejected',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      details: {
        fallback: 'cold-start-gi',
        snapshot: {
          irradianceAtlas: [9, 9],
          visibilityAtlas: [48, 48],
        },
      },
    });
    expect(warnings[0]!.message).toContain('cold-start');
    warnSpy.mockRestore();
  });

  it('routes ReSTIR-GI reservoir restore rejection through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const pipeline = {
      importRestirGIReservoirs: vi.fn(() => false),
    } as unknown as GIStateDeps['pipeline'];
    const deps = makeDeps(warnings, { pipeline });

    const ok = importGIStateImpl(deps, makeMatchingSnapshot({
      restirGI: makeRestirGISnapshot(),
    }));

    expect(ok).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(pipeline!.importRestirGIReservoirs).toHaveBeenCalledTimes(1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-restir-reservoir-rejected',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      details: {
        fallback: 'cold-start-gi',
        hasPipeline: true,
      },
    });
    expect(warnings[0]!.message).toContain('partial restore');
    warnSpy.mockRestore();
  });

  it('keeps PPG restore best-effort but warns when the guide cold-starts', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const pipeline = {
      importRestirGIReservoirs: vi.fn(() => true),
      importPPGSTree: vi.fn(() => false),
    } as unknown as GIStateDeps['pipeline'];
    const deps = makeDeps(warnings, { pipeline });

    const ok = importGIStateImpl(deps, makeMatchingSnapshot({
      restirGI: makeRestirGISnapshot(),
      ppg: makePpgSnapshot(),
    }));

    expect(ok).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(pipeline!.importRestirGIReservoirs).toHaveBeenCalledTimes(1);
    expect(pipeline!.importPPGSTree).toHaveBeenCalledTimes(1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-ppg-guide-rejected',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      details: {
        fallback: 'cold-start-ppg',
        hasPipeline: true,
        maxSpatialCells: 128,
        maxDTreeNodesPerCell: 64,
      },
    });
    expect(warnings[0]!.message).toContain('PPG will cold-start');
    warnSpy.mockRestore();
  });
});
