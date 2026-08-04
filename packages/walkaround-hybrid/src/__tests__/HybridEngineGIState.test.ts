import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import type { DDGI } from '../ddgi/DDGI.js';
import type { ProbeAtlasSnapshot } from '../ddgi/probeUpdatePass.js';
import {
  deserializeGIState,
  serializeGIState,
  type GIStateSnapshot,
  type PpgSnapshot,
} from '../giStateSnapshot.js';
import {
  GI_STATE_COMPATIBILITY_SCHEMA,
  GI_STATE_COMPATIBILITY_WORDS,
} from '../giStateCompatibility.js';
import {
  nrcStateShape,
  type NrcLearnedStateSnapshot,
  type NrcStateConfig,
} from '../neural/nrc/nrcStateSnapshot.js';
import type { RestirDISnapshot } from '../restir/restirDiStateSnapshot.js';
import {
  exportGIStateImpl,
  importGIStateImpl,
  type GIStateDeps,
} from '../HybridEngineGIState.js';

const DIMS = { x: 3, y: 3, z: 3 } as const;
const IRR_W = 15;
const IRR_H = 45;
const VIS_W = 54;
const VIS_H = 162;

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makeCompatibility(seed = 0): Uint32Array {
  const compatibility = new Uint32Array(GI_STATE_COMPATIBILITY_WORDS);
  compatibility[0] = GI_STATE_COMPATIBILITY_SCHEMA;
  compatibility[1] = seed;
  return compatibility;
}

function makeRestirGISnapshot(): NonNullable<GIStateSnapshot['restirGI']> {
  const length = 2 * 2 * 28;
  return {
    representationVersion: 1,
    halfW: 2,
    halfH: 2,
    strideU32: 28,
    current: new Uint32Array(length),
    previous: new Uint32Array(length),
    spatial: new Uint32Array(length),
  };
}

function makeRestirDISnapshot(): RestirDISnapshot {
  const length = 4 * 4 * 8;
  return {
    representationVersion: 1,
    width: 4,
    height: 4,
    strideU32: 8,
    current: new Uint32Array(length),
    previous: new Uint32Array(length),
    spatial: new Uint32Array(length),
  };
}

function makePpgSnapshot(): PpgSnapshot {
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

const NRC_CONFIG: NrcStateConfig = {
  levels: 2,
  featuresPerEntry: 2,
  tableSize: 4,
  nMin: 2,
  growth: 2,
  oneBlobBins: 2,
  width: 16,
  hidden: 2,
  spreadC: 0.01,
  recordCap: 8,
  learningRate: 0.01,
  tableLearningRate: 0.1,
  useF16: false,
  tileB: 4,
  warmupSteps: 3,
};

function makeNrcSnapshot(): NrcLearnedStateSnapshot {
  const shape = nrcStateShape(NRC_CONFIG);
  return {
    config: NRC_CONFIG,
    sceneBoundsMin: [-1, -2, -3],
    sceneBoundsMax: [4, 5, 6],
    trainedSteps: 9,
    mlp: {
      weights: new Float32Array(shape.weightScalars).fill(0.25),
      biases: new Float32Array(shape.biasScalars).fill(-0.5),
      firstMomentWeights: new Float32Array(shape.weightScalars).fill(-0.01),
      secondMomentWeights: new Float32Array(shape.weightScalars).fill(0.02),
      firstMomentBiases: new Float32Array(shape.biasScalars).fill(0.03),
      secondMomentBiases: new Float32Array(shape.biasScalars).fill(0.04),
      adamT: 9,
    },
    hashGrid: {
      tables: new Float32Array(shape.tableScalars).fill(0.001),
      firstMoment: new Float32Array(shape.tableScalars).fill(-0.002),
      secondMoment: new Float32Array(shape.tableScalars).fill(0.003),
      adamT: 9,
    },
  };
}

function makeSnapshot(
  extra: Partial<GIStateSnapshot> = {},
): GIStateSnapshot {
  const probeStateData = new Float32Array(DIMS.x * DIMS.y * DIMS.z * 4);
  for (let probe = 0; probe < DIMS.x * DIMS.y * DIMS.z; probe += 1) {
    probeStateData[probe * 4 + 3] = 1;
  }
  return {
    dims: DIMS,
    origin: [0, 0, 0],
    spacing: 2,
    irrW: IRR_W,
    irrH: IRR_H,
    visW: VIS_W,
    visH: VIS_H,
    irrData: new Uint16Array(IRR_W * IRR_H * 4),
    visData: new Uint16Array(VIS_W * VIS_H * 4),
    probeStateW: DIMS.x,
    probeStateH: DIMS.y * DIMS.z,
    probeStateData,
    compatibility: makeCompatibility(),
    restirGI: makeRestirGISnapshot(),
    restirDI: makeRestirDISnapshot(),
    ...extra,
  };
}

function makeAtlasSnapshot(): ProbeAtlasSnapshot {
  const snapshot = makeSnapshot();
  return {
    irrW: snapshot.irrW,
    irrH: snapshot.irrH,
    visW: snapshot.visW,
    visH: snapshot.visH,
    probeStateW: snapshot.probeStateW,
    probeStateH: snapshot.probeStateH,
    irrData: snapshot.irrData,
    visData: snapshot.visData,
    probeStateData: snapshot.probeStateData,
  };
}

function makeSnapshotWithout(
  section: 'compatibility' | 'restirGI' | 'restirDI',
): unknown {
  const snapshot = { ...makeSnapshot() } as Record<string, unknown>;
  delete snapshot[section];
  return snapshot;
}

interface ImportTransaction {
  commit(): void;
  rollback(): void;
  finalize(): void;
}

function makeTransaction(
  label: string,
  events: string[],
  throws: Partial<Record<'commit' | 'rollback' | 'finalize', Error>> = {},
): ImportTransaction {
  return {
    commit: vi.fn(() => {
      events.push(`${label}:commit`);
      if (throws.commit) throw throws.commit;
    }),
    rollback: vi.fn(() => {
      events.push(`${label}:rollback`);
      if (throws.rollback) throw throws.rollback;
    }),
    finalize: vi.fn(() => {
      events.push(`${label}:finalize`);
      if (throws.finalize) throw throws.finalize;
    }),
  };
}

interface HarnessOptions {
  readonly ppg?: boolean;
  readonly nrc?: boolean;
  readonly origin?: readonly [number, number, number];
  readonly spacing?: number;
  readonly atlasAvailable?: boolean;
  readonly finishError?: Error;
  readonly submitError?: Error;
  readonly transactionErrors?: Partial<
    Record<
      'atlas' | 'gi' | 'di' | 'ppg' | 'nrc',
      Partial<Record<'commit' | 'rollback' | 'finalize', Error>>
    >
  >;
}

function makeHarness(
  warnings: EngineWarning[],
  options: HarnessOptions = {},
) {
  const events: string[] = [];
  const transactions = {
    atlas: makeTransaction(
      'atlas',
      events,
      options.transactionErrors?.atlas,
    ),
    gi: makeTransaction('gi', events, options.transactionErrors?.gi),
    di: makeTransaction('di', events, options.transactionErrors?.di),
    ppg: makeTransaction('ppg', events, options.transactionErrors?.ppg),
    nrc: makeTransaction('nrc', events, options.transactionErrors?.nrc),
  };
  const finish = vi.fn(() => {
    events.push('encoder:finish');
    if (options.finishError) throw options.finishError;
    return {} as GPUCommandBuffer;
  });
  const submit = vi.fn(() => {
    events.push('queue:submit');
    if (options.submitError) throw options.submitError;
  });
  const device = {
    createCommandEncoder: vi.fn(() => ({ finish })),
    queue: { submit },
  } as unknown as GPUDevice;
  const origin = options.origin ?? [0, 0, 0];
  const atlasSnapshot = makeSnapshot();
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
    exportAtlasData: vi.fn(async () => ({
      irrW: atlasSnapshot.irrW,
      irrH: atlasSnapshot.irrH,
      visW: atlasSnapshot.visW,
      visH: atlasSnapshot.visH,
      probeStateW: atlasSnapshot.probeStateW,
      probeStateH: atlasSnapshot.probeStateH,
      irrData: atlasSnapshot.irrData,
      visData: atlasSnapshot.visData,
      probeStateData: atlasSnapshot.probeStateData,
    })),
    prepareAtlasImport: vi.fn(() => transactions.atlas),
  } as unknown as DDGI;
  const pipeline = {
    ppgStateRequired: options.ppg ?? false,
    nrcStateRequired: options.nrc ?? false,
    exportRestirGIReservoirs: vi.fn(async () => makeRestirGISnapshot()),
    exportRestirDIReservoirs: vi.fn(async () => makeRestirDISnapshot()),
    exportPPGSTree: vi.fn(() => (options.ppg ? makePpgSnapshot() : null)),
    exportNrcLearnedState: vi.fn(async () =>
      options.nrc ? makeNrcSnapshot() : null),
    canImportRestirGIReservoirs: vi.fn(() => true),
    canImportRestirDIReservoirs: vi.fn(() => true),
    canImportPPGSTree: vi.fn(() => true),
    canImportNrcLearnedState: vi.fn(() => true),
    prepareRestirGIReservoirImport: vi.fn(() => transactions.gi),
    prepareRestirDIReservoirImport: vi.fn(() => transactions.di),
    preparePPGSTreeImport: vi.fn(() => transactions.ppg),
    prepareNrcLearnedStateImport: vi.fn(() => transactions.nrc),
  } as unknown as NonNullable<GIStateDeps['pipeline']>;
  const deps: GIStateDeps = {
    device,
    ddgi,
    pipeline,
    compatibility: makeCompatibility(),
    onWarning: (warning) => warnings.push(warning),
  };
  return {
    deps,
    ddgi,
    pipeline,
    transactions,
    events,
    finish,
    submit,
  };
}

describe('HybridEngine complete estimator-state import', () => {
  it.each([
    ['null', null],
    ['empty object', {}],
    ['missing compatibility', makeSnapshotWithout('compatibility')],
    ['missing ReSTIR-GI', makeSnapshotWithout('restirGI')],
    ['missing ReSTIR-DI', makeSnapshotWithout('restirDI')],
  ])('rejects malformed or incomplete state before preparation: %s', (_label, value) => {
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings);

    expect(
      importGIStateImpl(made.deps, value as GIStateSnapshot),
    ).toBe(false);
    expect(made.ddgi.prepareAtlasImport).not.toHaveBeenCalled();
    expect(made.pipeline.prepareRestirGIReservoirImport).not.toHaveBeenCalled();
    expect(warnings.at(-1)).toMatchObject({
      details: { fallback: 'retain-current-gi' },
    });
  });

  it('rejects grid and exact live-input compatibility mismatches', () => {
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings);

    expect(
      importGIStateImpl(made.deps, makeSnapshot({ origin: [10, 0, 0] })),
    ).toBe(false);
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.import-gi-state-grid-mismatch',
    );

    expect(
      importGIStateImpl(
        made.deps,
        makeSnapshot({ compatibility: makeCompatibility(7) }),
      ),
    ).toBe(false);
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.import-gi-state-scene-mismatch',
    );
    expect(made.ddgi.prepareAtlasImport).not.toHaveBeenCalled();
  });

  it.each([
    [
      'PPG snapshot absent while PPG is live',
      { ppg: true },
      {},
      'ppg-mode-mismatch',
    ],
    [
      'PPG snapshot present while PPG is disabled',
      {},
      { ppg: makePpgSnapshot() },
      'ppg-mode-mismatch',
    ],
    [
      'NRC snapshot absent while NRC is live',
      { nrc: true },
      {},
      'nrc-mode-mismatch',
    ],
    [
      'NRC snapshot present while NRC is disabled',
      {},
      { nrc: makeNrcSnapshot() },
      'nrc-mode-mismatch',
    ],
  ] as const)(
    'rejects estimator-mode mismatch atomically: %s',
    (_label, harnessOptions, snapshotExtra, reason) => {
      const warnings: EngineWarning[] = [];
      const made = makeHarness(warnings, harnessOptions);

      expect(importGIStateImpl(made.deps, makeSnapshot(snapshotExtra))).toBe(
        false,
      );
      expect(made.ddgi.prepareAtlasImport).not.toHaveBeenCalled();
      expect(warnings.at(-1)).toMatchObject({
        code: 'walkaround-hybrid.import-gi-state-complete-state-rejected',
        details: { reason },
      });
    },
  );

  it('rolls back already-prepared candidates in reverse order when preparation rejects', () => {
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings, { ppg: true });
    vi.mocked(made.pipeline.preparePPGSTreeImport).mockReturnValueOnce(null);

    expect(
      importGIStateImpl(
        made.deps,
        makeSnapshot({ ppg: makePpgSnapshot() }),
      ),
    ).toBe(false);
    expect(made.events).toEqual([
      'di:rollback',
      'gi:rollback',
      'atlas:rollback',
    ]);
    expect(made.submit).not.toHaveBeenCalled();
    expect(warnings.at(-1)).toMatchObject({
      details: { reason: 'ppg-prepare-rejected' },
    });
  });

  it('rolls every candidate back when publication or queue submission throws', () => {
    const commitError = new Error('injected DI publication failure');
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings, {
      transactionErrors: { di: { commit: commitError } },
    });

    expect(() => importGIStateImpl(made.deps, makeSnapshot())).toThrow(
      commitError,
    );
    expect(made.events).toEqual([
      'atlas:commit',
      'gi:commit',
      'di:commit',
      'di:rollback',
      'gi:rollback',
      'atlas:rollback',
    ]);
    expect(made.submit).not.toHaveBeenCalled();

    const submitError = new Error('injected queue submission failure');
    const second = makeHarness([], {
      nrc: true,
      submitError,
    });
    expect(() =>
      importGIStateImpl(
        second.deps,
        makeSnapshot({ nrc: makeNrcSnapshot() }),
      ),
    ).toThrow(submitError);
    expect(second.events).toEqual([
      'encoder:finish',
      'atlas:commit',
      'gi:commit',
      'di:commit',
      'nrc:commit',
      'queue:submit',
      'nrc:rollback',
      'di:rollback',
      'gi:rollback',
      'atlas:rollback',
    ]);
  });

  it('rolls every prepared candidate back when NRC command encoding cannot finish', () => {
    const finishError = new Error('injected encoder finish failure');
    const made = makeHarness([], { nrc: true, finishError });

    expect(() =>
      importGIStateImpl(
        made.deps,
        makeSnapshot({ nrc: makeNrcSnapshot() }),
      ),
    ).toThrow(finishError);
    expect(made.events).toEqual([
      'encoder:finish',
      'nrc:rollback',
      'di:rollback',
      'gi:rollback',
      'atlas:rollback',
    ]);
  });

  it('commits one complete cohort, submits encoded state, then retires old resources in reverse order', () => {
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings, { ppg: true, nrc: true });

    expect(
      importGIStateImpl(
        made.deps,
        makeSnapshot({
          ppg: makePpgSnapshot(),
          nrc: makeNrcSnapshot(),
        }),
      ),
    ).toBe(true);
    expect(made.events).toEqual([
      'encoder:finish',
      'atlas:commit',
      'gi:commit',
      'di:commit',
      'ppg:commit',
      'nrc:commit',
      'queue:submit',
      'nrc:finalize',
      'ppg:finalize',
      'di:finalize',
      'gi:finalize',
      'atlas:finalize',
    ]);
    expect(warnings).toEqual([]);
  });

  it('reports retirement failures without lying that the committed state was rejected', () => {
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings, {
      transactionErrors: {
        gi: { finalize: new Error('injected old-buffer destroy failure') },
      },
    });

    expect(importGIStateImpl(made.deps, makeSnapshot())).toBe(true);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.import-gi-state-finalization-failed',
      details: { committed: true, failureCount: 1 },
    });
  });

  it('aggregates rollback failure with the primary publication failure', () => {
    const primary = new Error('injected commit failure');
    const rollback = new Error('injected rollback failure');
    const made = makeHarness([], {
      transactionErrors: {
        di: { commit: primary },
        gi: { rollback },
      },
    });

    expect(() => importGIStateImpl(made.deps, makeSnapshot())).toThrow(
      AggregateError,
    );
    try {
      importGIStateImpl(made.deps, makeSnapshot());
    } catch (error) {
      expect(error).toBeInstanceOf(AggregateError);
      expect((error as AggregateError).errors).toEqual([primary, rollback]);
    }
  });

  it('accepts a legitimate large-world float32 serialization round-trip', () => {
    const origin = [
      1_000_000_033,
      -2_000_000_017,
      3_000_000_049,
    ] as const;
    const spacing = 1_000_000.03125;
    const restored = deserializeGIState(
      serializeGIState(makeSnapshot({ origin, spacing })),
    );
    const warnings: EngineWarning[] = [];
    const made = makeHarness(warnings, { origin, spacing });

    expect(restored.origin).toEqual(origin.map(Math.fround));
    expect(restored.spacing).toBe(Math.fround(spacing));
    expect(importGIStateImpl(made.deps, restored)).toBe(true);
    expect(warnings).toEqual([]);
  });
});

describe('HybridEngine complete estimator-state export', () => {
  it('captures every live estimator section and owns the compatibility key', async () => {
    const made = makeHarness([], { ppg: true, nrc: true });

    const snapshot = await exportGIStateImpl(made.deps);

    expect(snapshot).toMatchObject({
      dims: DIMS,
      origin: [0, 0, 0],
      spacing: 2,
      restirGI: { halfW: 2, halfH: 2, strideU32: 28 },
      restirDI: { width: 4, height: 4, strideU32: 8 },
      ppg: {
        maxSpatialCells: 128,
        maxDTreeNodesPerCell: 64,
      },
      nrc: {
        trainedSteps: 9,
      },
    });
    expect(snapshot?.compatibility).toEqual(made.deps.compatibility);
    expect(snapshot?.compatibility).not.toBe(made.deps.compatibility);
  });

  it('keeps CPU metadata and optional-mode ownership from the queued generation', async () => {
    const made = makeHarness([], { ppg: true, nrc: true });
    const atlasGate = deferred<ProbeAtlasSnapshot | null>();
    vi.mocked(made.ddgi.exportAtlasData).mockReturnValueOnce(
      atlasGate.promise,
    );

    const pending = exportGIStateImpl(made.deps);

    const grid = made.ddgi.gridParams as unknown as {
      dims: { x: number; y: number; z: number };
      origin: { x: number; y: number; z: number };
      spacing: number;
    };
    grid.dims = { x: 9, y: 8, z: 7 };
    grid.origin = { x: 10, y: 20, z: 30 };
    grid.spacing = 99;
    made.deps.compatibility[1] = 1234;
    (made.pipeline as unknown as { ppgStateRequired: boolean })
      .ppgStateRequired = false;
    (made.pipeline as unknown as { nrcStateRequired: boolean })
      .nrcStateRequired = false;
    atlasGate.resolve(makeAtlasSnapshot());

    const snapshot = await pending;
    expect(snapshot).toMatchObject({
      dims: DIMS,
      origin: [0, 0, 0],
      spacing: 2,
      ppg: { maxSpatialCells: 128 },
      nrc: { trainedSteps: 9 },
    });
    expect(snapshot?.compatibility?.[1]).toBe(0);
  });

  it('does not publish a partial cohort when a required flag changes during readback', async () => {
    const made = makeHarness([], { ppg: true });
    const atlasGate = deferred<ProbeAtlasSnapshot | null>();
    vi.mocked(made.ddgi.exportAtlasData).mockReturnValueOnce(
      atlasGate.promise,
    );
    vi.mocked(made.pipeline.exportPPGSTree).mockReturnValueOnce(null);

    const pending = exportGIStateImpl(made.deps);
    (made.pipeline as unknown as { ppgStateRequired: boolean })
      .ppgStateRequired = false;
    atlasGate.resolve(makeAtlasSnapshot());

    await expect(pending).resolves.toBeNull();
  });

  it.each([
    ['DDGI', 'exportAtlasData'],
    ['ReSTIR-GI', 'exportRestirGIReservoirs'],
    ['ReSTIR-DI', 'exportRestirDIReservoirs'],
    ['PPG', 'exportPPGSTree'],
    ['NRC', 'exportNrcLearnedState'],
  ] as const)(
    'refuses to publish a partial snapshot when required %s state is unavailable',
    async (_label, method) => {
      const options = {
        ppg: method === 'exportPPGSTree',
        nrc: method === 'exportNrcLearnedState',
      };
      const made = makeHarness([], options);
      if (method === 'exportAtlasData') {
        vi.mocked(made.ddgi.exportAtlasData).mockResolvedValueOnce(null);
      } else if (method === 'exportPPGSTree') {
        vi.mocked(made.pipeline.exportPPGSTree).mockReturnValueOnce(null);
      } else {
        vi.mocked(made.pipeline[method]).mockResolvedValueOnce(null);
      }

      await expect(exportGIStateImpl(made.deps)).resolves.toBeNull();
    },
  );
});
