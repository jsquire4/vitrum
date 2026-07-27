import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import type { DDGI } from '../ddgi/DDGI.js';
import {
  deserializeGIState,
  serializeGIState,
  type GIStateSnapshot,
} from '../giStateSnapshot.js';
import { importGIStateImpl, type GIStateDeps } from '../HybridEngineGIState.js';

const DIMS = { x: 3, y: 3, z: 3 } as const;
const IRR_W = 15;
const IRR_H = 45;
const VIS_W = 54;
const VIS_H = 162;

function makeSnapshot(): GIStateSnapshot {
  return {
    dims: DIMS,
    origin: [10, 0, 0],
    spacing: 2,
    irrW: IRR_W,
    irrH: IRR_H,
    visW: VIS_W,
    visH: VIS_H,
    irrData: new Uint16Array(IRR_W * IRR_H * 4),
    visData: new Uint16Array(VIS_W * VIS_H * 4),
    probeStateW: 3,
    probeStateH: 9,
    probeStateData: new Float32Array(3 * 9 * 4),
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
    strideU32: 28,
    current: new Uint32Array(2 * 2 * 28),
    previous: new Uint32Array(2 * 2 * 28),
    spatial: new Uint32Array(2 * 2 * 28),
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
    readonly atlasAvailable?: boolean;
    readonly origin?: readonly [number, number, number];
    readonly spacing?: number;
  } = {},
): GIStateDeps {
  const origin = options.origin ?? [0, 0, 0];
  const ddgi = {
    gridParams: {
      dims: DIMS,
      origin: { x: origin[0], y: origin[1], z: origin[2] },
      spacing: options.spacing ?? 2,
      irradianceAtlasW: IRR_W,
      irradianceAtlasH: IRR_H,
      visibilityAtlasW: VIS_W,
      visibilityAtlasH: VIS_H,
    },
    getReadAtlasGPUTextures: vi.fn(() =>
      options.atlasAvailable === false
        ? null
        : {
            irradiance: {} as GPUTexture,
            visibility: {} as GPUTexture,
          }),
    importAtlasData: vi.fn(options.importAtlasData ?? (() => true)),
  } as unknown as DDGI;

  return {
    device: {} as GPUDevice,
    ddgi,
    pipeline: options.pipeline ?? null,
    onWarning: (warning) => warnings.push(warning),
  };
}

function makePreparedPipeline(options: {
  readonly canImport?: boolean;
  readonly prepare?: () => {
    commit(): void;
    abort(): void;
  } | null;
  readonly importPPG?: () => boolean;
} = {}) {
  const transaction = {
    commit: vi.fn(),
    abort: vi.fn(),
  };
  const pipeline = {
    canImportRestirGIReservoirs: vi.fn(() => options.canImport ?? true),
    prepareRestirGIReservoirImport: vi.fn(
      options.prepare ?? (() => transaction),
    ),
    importPPGSTree: vi.fn(options.importPPG ?? (() => true)),
  } as unknown as NonNullable<GIStateDeps['pipeline']>;
  return { pipeline, transaction };
}

describe('HybridEngineGIState import diagnostics', () => {
  it.each([
    ['null', null],
    ['empty object', {}],
    [
      'missing payload arrays',
      {
        dims: DIMS,
        origin: [0, 0, 0],
        spacing: 2,
        irrW: IRR_W,
        irrH: IRR_H,
        visW: VIS_W,
        visH: VIS_H,
      },
    ],
  ])('rejects a structurally malformed public input without throwing: %s', (_label, value) => {
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline();
    const deps = makeDeps(warnings, { pipeline: made.pipeline });

    expect(
      importGIStateImpl(deps, value as unknown as GIStateSnapshot),
    ).toBe(false);
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(made.pipeline.canImportRestirGIReservoirs).not.toHaveBeenCalled();
    expect(warnings.at(-1)).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-malformed-snapshot',
      details: { fallback: 'retain-current-gi' },
    });
  });

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
          irradianceAtlas: [IRR_W, IRR_H],
          visibilityAtlas: [VIS_W, VIS_H],
        },
      },
    });
    expect(warnings[0]!.message).toContain('cold-start');
    warnSpy.mockRestore();
  });

  it('routes ReSTIR-GI reservoir restore rejection through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline({ canImport: false });
    const pipeline = made.pipeline;
    const deps = makeDeps(warnings, { pipeline });

    const ok = importGIStateImpl(deps, makeMatchingSnapshot({
      restirGI: makeRestirGISnapshot(),
    }));

    expect(ok).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(pipeline.canImportRestirGIReservoirs).toHaveBeenCalledTimes(1);
    expect(pipeline.prepareRestirGIReservoirImport).not.toHaveBeenCalled();
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(made.transaction.commit).not.toHaveBeenCalled();
    expect(made.transaction.abort).not.toHaveBeenCalled();
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

  it('keeps PPG restore best-effort and warns that the current guide is retained', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline({ importPPG: () => false });
    const pipeline = made.pipeline;
    const deps = makeDeps(warnings, { pipeline });

    const ok = importGIStateImpl(deps, makeMatchingSnapshot({
      restirGI: makeRestirGISnapshot(),
      ppg: makePpgSnapshot(),
    }));

    expect(ok).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(pipeline.prepareRestirGIReservoirImport).toHaveBeenCalledTimes(1);
    expect(made.transaction.commit).toHaveBeenCalledTimes(1);
    expect(made.transaction.abort).not.toHaveBeenCalled();
    expect(pipeline.importPPGSTree).toHaveBeenCalledTimes(1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-ppg-guide-rejected',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      details: {
        fallback: 'retain-current-ppg',
        hasPipeline: true,
        maxSpatialCells: 128,
        maxDTreeNodesPerCell: 64,
      },
    });
    expect(warnings[0]!.message).toContain('retaining the current PPG guide');
    warnSpy.mockRestore();
  });

  it.each([
    ['NaN origin', { origin: [Number.NaN, 0, 0] as const }],
    ['infinite origin', { origin: [0, Number.POSITIVE_INFINITY, 0] as const }],
    ['NaN spacing', { spacing: Number.NaN }],
    ['zero spacing', { spacing: 0 }],
    ['infinite spacing', { spacing: Number.POSITIVE_INFINITY }],
  ])('rejects non-finite grid metadata before touching either GI cohort: %s', (_label, metadata) => {
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline();
    const deps = makeDeps(warnings, { pipeline: made.pipeline });
    const snapshot = makeMatchingSnapshot({
      ...metadata,
      restirGI: makeRestirGISnapshot(),
    });

    expect(importGIStateImpl(deps, snapshot)).toBe(false);
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(made.pipeline.canImportRestirGIReservoirs).not.toHaveBeenCalled();
    expect(made.transaction.commit).not.toHaveBeenCalled();
    expect(made.transaction.abort).not.toHaveBeenCalled();
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.import-gi-state-malformed-snapshot',
    );
  });

  it.each([
    [
      'irradiance length',
      () => ({ irrData: new Uint16Array(IRR_W * IRR_H * 4 - 1) }),
    ],
    [
      'visibility non-finite half',
      () => {
        const visData = new Uint16Array(VIS_W * VIS_H * 4);
        visData[17] = 0x7e00;
        return { visData };
      },
    ],
    ['probe-state length', () => ({ probeStateData: new Float32Array(3) })],
    ['unallocated live atlases', () => ({})],
  ])('preflights malformed DDGI payloads without invoking its importer: %s', (_label, build) => {
    const warnings: EngineWarning[] = [];
    const deps = makeDeps(warnings, {
      atlasAvailable: _label !== 'unallocated live atlases',
    });
    const snapshot = makeMatchingSnapshot(build());

    expect(importGIStateImpl(deps, snapshot)).toBe(false);
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(warnings.at(-1)?.code).toBe(
      _label === 'unallocated live atlases'
        ? 'walkaround-hybrid.import-gi-state-atlas-rejected'
        : 'walkaround-hybrid.import-gi-state-malformed-snapshot',
    );
  });

  it('accepts a legitimate large-world f32 serialization round-trip', () => {
    const origin = [
      1_000_000_033,
      -2_000_000_017,
      3_000_000_049,
    ] as const;
    const spacing = 1_000_000.03125;
    const serialized = serializeGIState(
      makeMatchingSnapshot({ origin, spacing }),
    );
    const restored = deserializeGIState(serialized);
    const warnings: EngineWarning[] = [];
    const deps = makeDeps(warnings, { origin, spacing });

    expect(restored.origin).toEqual(origin.map(Math.fround));
    expect(restored.spacing).toBe(Math.fround(spacing));
    expect(importGIStateImpl(deps, restored)).toBe(true);
    expect(deps.ddgi.importAtlasData).toHaveBeenCalledOnce();
    expect(warnings).toEqual([]);
  });

  it('rejects a meaningfully different large-world grid', () => {
    const origin = [
      1_000_000_033,
      -2_000_000_017,
      3_000_000_049,
    ] as const;
    const spacing = 1_000_000.03125;
    const restored = deserializeGIState(
      serializeGIState(makeMatchingSnapshot({ origin, spacing })),
    );
    const mismatched: GIStateSnapshot = {
      ...restored,
      origin: [restored.origin[0] + 256, restored.origin[1], restored.origin[2]],
    };
    const warnings: EngineWarning[] = [];
    const deps = makeDeps(warnings, { origin, spacing });

    expect(importGIStateImpl(deps, mismatched)).toBe(false);
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.import-gi-state-grid-mismatch',
    );
  });

  it('does not publish DDGI when required-reservoir preparation throws', () => {
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline({
      prepare: () => {
        throw new Error('injected reservoir allocation failure');
      },
    });
    const deps = makeDeps(warnings, { pipeline: made.pipeline });

    expect(() =>
      importGIStateImpl(
        deps,
        makeMatchingSnapshot({ restirGI: makeRestirGISnapshot() }),
      ),
    ).toThrow('injected reservoir allocation failure');
    expect(deps.ddgi.importAtlasData).not.toHaveBeenCalled();
    expect(made.transaction.commit).not.toHaveBeenCalled();
    expect(made.transaction.abort).not.toHaveBeenCalled();
  });

  it('aborts the prepared reservoir cohort when DDGI rejects without publication', () => {
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline();
    const deps = makeDeps(warnings, {
      pipeline: made.pipeline,
      importAtlasData: () => false,
    });

    expect(
      importGIStateImpl(
        deps,
        makeMatchingSnapshot({ restirGI: makeRestirGISnapshot() }),
      ),
    ).toBe(false);
    expect(made.transaction.abort).toHaveBeenCalledOnce();
    expect(made.transaction.commit).not.toHaveBeenCalled();
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.import-gi-state-atlas-rejected',
    );
  });

  it('aborts the prepared reservoir cohort and rethrows when DDGI import throws', () => {
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline();
    const deps = makeDeps(warnings, {
      pipeline: made.pipeline,
      importAtlasData: () => {
        throw new Error('injected DDGI upload failure');
      },
    });

    expect(() =>
      importGIStateImpl(
        deps,
        makeMatchingSnapshot({ restirGI: makeRestirGISnapshot() }),
      ),
    ).toThrow('injected DDGI upload failure');
    expect(made.transaction.abort).toHaveBeenCalledOnce();
    expect(made.transaction.commit).not.toHaveBeenCalled();
  });

  it('publishes ReSTIR only after DDGI has accepted its candidate atlases', () => {
    const events: string[] = [];
    const transaction = {
      commit: vi.fn(() => events.push('restir-published')),
      abort: vi.fn(),
    };
    const made = makePreparedPipeline({ prepare: () => transaction });
    const deps = makeDeps([], {
      pipeline: made.pipeline,
      importAtlasData: () => {
        events.push('ddgi-published');
        return true;
      },
    });

    expect(
      importGIStateImpl(
        deps,
        makeMatchingSnapshot({ restirGI: makeRestirGISnapshot() }),
      ),
    ).toBe(true);
    expect(events).toEqual(['ddgi-published', 'restir-published']);
    expect(transaction.commit).toHaveBeenCalledOnce();
    expect(transaction.abort).not.toHaveBeenCalled();
  });

  it('contains an unexpected optional PPG throw after required GI publication', () => {
    const warnings: EngineWarning[] = [];
    const made = makePreparedPipeline({
      importPPG: () => {
        throw new Error('injected optional PPG failure');
      },
    });
    const deps = makeDeps(warnings, { pipeline: made.pipeline });

    expect(
      importGIStateImpl(
        deps,
        makeMatchingSnapshot({
          restirGI: makeRestirGISnapshot(),
          ppg: makePpgSnapshot(),
        }),
      ),
    ).toBe(true);
    expect(made.transaction.commit).toHaveBeenCalledOnce();
    expect(made.transaction.abort).not.toHaveBeenCalled();
    expect(warnings.at(-1)).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-ppg-guide-rejected',
      details: {
        fallback: 'retain-current-ppg',
        failure: 'injected optional PPG failure',
      },
    });
  });
});
