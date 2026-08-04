/**
 * nrcSubsystem.ts — per-engine host driver for LIVE Neural Radiance Caching
 * (Müller, Rousselle, Novák, Keller 2021, "Real-time Neural Radiance Caching for
 * Path Tracing", ACM TOG 40(4)).
 *
 * Owns the GPU resources the gi-ris NRC query reads + the {@link FusedMlpTrainer}
 * that learns the cache, and exposes:
 *   • {@link bindGroup} — the `@group(4)` NRC bind group the gi-ris NRC pipeline
 *     binds (MLP weights/biases + hash-grid tables + level descs + record gather
 *     + encoding-config UBO),
 *   • {@link trainFromRecords} — read back the per-frame self-training records
 *     the gi-ris pass wrote and run ONE {@link FusedMlpTrainer.trainStep} FOLLOWED
 *     by a hash-grid TABLE training step (encode-backward scatter + a separate
 *     Adam over the feature tables). Both the MLP AND the multiresolution
 *     hash-grid encoding LEARN (Müller 2022 Instant-NGP §4); the tables are no
 *     longer frozen at random init. (host-owns-cadence: once per frame.)
 *
 * HOST-OWNS-LIFECYCLE: like the rest of vitrum, this accepts a device handle but
 * does NOT own the device. It is constructed only when the engine was created
 * with `nrcEnabled` (full-tier only); when NRC is off it is never instantiated,
 * so the default pipeline allocates none of these buffers.
 *
 * BIAS CAVEAT: NRC is a BIASED cache — the MLP prediction REPLACES the true path
 * suffix at the spread-termination vertex. Acceptance is perceptual closeness +
 * faster convergence, NOT equal converged mean. See HARDWARE-VALIDATION-NEEDS
 * V20.
 */

import {
  FusedMlpTrainer,
  heInit,
  type FusedNetSpec,
  type FusedTrainerConfig,
} from './fusedMlpTrainer.js';
import { HashGridTableTrainer } from './hashGridTableTrainer.js';
import { unpackRecords } from './recordUnpack.js';
import { computeNrcResourceFootprint, preflightNrcResources, validateNrcAabb } from './nrcPreflight.js';
import type { RisGiNrcConfig } from '../../shaders/risGiNrc.wgsl.js';
import type { BGLCache } from '../../pipeline/bindGroupLayouts.js';
import type { PipelineSubsystem } from '../../pipeline/PipelineSubsystem.js';
import {
  rethrowWithSceneMutationCleanup,
  runSceneMutationCleanups,
  type PreparedSceneMutation,
  type SceneMutationCleanup,
} from '../../SceneMutationTransaction.js';
import type { FramePublication } from '../../pipeline/FramePublication.js';
import {
  NRC_DIAGNOSTIC_BYTES,
  NRC_DIAGNOSTIC_COUNT,
  NRC_DIAGNOSTIC_INDEX,
  type NrcDiagnostics,
} from './nrcDiagnostics.js';
import {
  buildNrcInferenceArenaHeader,
  buildNrcRuntimeArenaHeader,
  nextNrcArenaEpoch,
  type NrcInferenceArenaLayout,
  type NrcRuntimeArenaLayout,
} from './nrcArena.js';
import {
  assertNrcLearnedStateSnapshot,
  nrcStateBoundsMatch,
  nrcStateConfigMatches,
  type NrcLearnedStateSnapshot,
  type NrcStateConfig,
} from './nrcStateSnapshot.js';

/** Resolved NRC config (encoding + MLP + self-training cadence). The WGSL gi-ris
 *  NRC variant and the trainer net-spec are both derived from this so the query
 *  evaluates the same network the trainer learns on the same encoding. */
export interface NrcConfig {
  /** Hash-grid resolution levels L. */
  readonly levels: number;
  /** Features per hash-grid entry F. */
  readonly featuresPerEntry: number;
  /** Hash table rows per level (T). */
  readonly tableSize: number;
  /** Coarsest level resolution N_min. */
  readonly nMin: number;
  /** Per-level geometric growth b. */
  readonly growth: number;
  /** One-blob bins k per encoded scalar. */
  readonly oneBlobBins: number;
  /** MLP hidden width W (Müller: 64). */
  readonly width: number;
  /** MLP hidden node-layers (Müller: 6). */
  readonly hidden: number;
  /** Müller §5 spread-termination constant c. */
  readonly spreadC: number;
  /** Max self-training records gathered per frame (= record buffer capacity).
   *  The gi-ris pass partitions half-res pixels into disjoint slot-owned blocks
   *  and traces at most one independent suffix per slot each frame. */
  readonly recordCap: number;
  /** Adam learning rate per train step (the MLP weights). */
  readonly learningRate: number;
  /** Adam learning rate for the hash-grid feature TABLES. Instant-NGP (Müller
   *  2022 §4) trains the embedding faster than the MLP (lr_embed ≈ 0.1 vs
   *  lr_mlp ≈ 0.01); the table grad magnitudes are sparse + small so a higher LR
   *  is standard. Separate from {@link learningRate} so the two can be tuned. */
  readonly tableLearningRate: number;
  /** Use f16 mixed-precision in the trainer (adapter must support shader-f16). */
  readonly useF16: boolean;
  /** Trainer tile size (samples per workgroup). */
  readonly tileB: number;
  /** Completed trainer windows required before NRC predictions may replace DDGI suffixes. */
  /** Optional host policy for total NRC GPU-buffer residency at the ordinary
   * readback peak. WebGPU does not expose an adapter-wide VRAM budget, so hosts
   * that have one must provide it explicitly; per-resource adapter limits are
   * always checked independently. */
  readonly maxNrcResidentBytes?: number;
  readonly warmupSteps?: number;
}

/** Müller-core-sized default NRC config, sized to be full-tier viable. */
export const DEFAULT_NRC_CONFIG: NrcConfig = {
  levels: 8,
  featuresPerEntry: 2,
  tableSize: 4096,
  nMin: 4,
  growth: 2.0,
  oneBlobBins: 8,
  width: 64,
  hidden: 6,
  spreadC: 0.01,
  recordCap: 4096,
  learningRate: 0.01,
  tableLearningRate: 0.1,   // Instant-NGP §4: embedding LR ≈ 10× the MLP LR.
  useF16: false,
  tileB: 32,
  warmupSteps: 8,
};

/** Resolve and validate a partial NRC configuration without allocating GPU
 * resources. This is the single construction-time source used by the public
 * HybridEngine option parser, device negotiation, shader compilation, and the
 * subsystem itself. */
export function resolveNrcConfig(partial: Partial<NrcConfig> = {}): NrcConfig {
  const cfg: NrcConfig = { ...DEFAULT_NRC_CONFIG, ...partial };
  const positiveIntegers: ReadonlyArray<keyof Pick<
    NrcConfig,
    'levels' | 'featuresPerEntry' | 'tableSize' | 'nMin' | 'oneBlobBins' |
    'width' | 'recordCap' | 'tileB'
  >> = [
    'levels', 'featuresPerEntry', 'tableSize', 'nMin', 'oneBlobBins',
    'width', 'recordCap', 'tileB',
  ];
  for (const key of positiveIntegers) {
    const value = cfg[key];
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`NRC ${key} must be a positive safe integer; got ${value}`);
    }
  }
  if (!Number.isSafeInteger(cfg.hidden) || cfg.hidden < 0) {
    throw new RangeError(`NRC hidden must be a non-negative safe integer; got ${cfg.hidden}`);
  }
  const positiveFinite: ReadonlyArray<keyof Pick<
    NrcConfig,
    'growth' | 'learningRate' | 'tableLearningRate'
  >> = ['growth', 'learningRate', 'tableLearningRate'];
  for (const key of positiveFinite) {
    const value = cfg[key];
    if (!Number.isFinite(value) || value <= 0) {
      throw new RangeError(`NRC ${key} must be finite and positive; got ${value}`);
    }
  }
  if (!Number.isFinite(cfg.spreadC) || cfg.spreadC < 0) {
    throw new RangeError(`NRC spreadC must be finite and non-negative; got ${cfg.spreadC}`);
  }
  const warmup = cfg.warmupSteps ?? DEFAULT_NRC_CONFIG.warmupSteps ?? 8;
  if (!Number.isSafeInteger(warmup) || warmup < 0 || warmup > 0xffff_ffff) {
    throw new RangeError(
      `NRC warmupSteps must be a non-negative u32 integer; got ${warmup}`,
    );
  }
  if (typeof cfg.useF16 !== 'boolean') {
    throw new TypeError(`NRC useF16 must be a boolean; got ${String(cfg.useF16)}`);
  }
  if (cfg.maxNrcResidentBytes !== undefined
      && (!Number.isSafeInteger(cfg.maxNrcResidentBytes)
          || cfg.maxNrcResidentBytes <= 0)) {
    throw new RangeError(
      `NRC maxNrcResidentBytes must be a positive safe integer; got ${cfg.maxNrcResidentBytes}`,
    );
  }
  const inputWidth =
    cfg.levels * cfg.featuresPerEntry + 2 * cfg.oneBlobBins + 7;
  if (!Number.isSafeInteger(inputWidth) || inputWidth > cfg.width) {
    throw new RangeError(
      `NRC encoded input width ${inputWidth} exceeds MLP width ${cfg.width}`,
    );
  }
  return cfg;
}

const OUT_W = 3; // RGB radiance
const U32_MAX = 0xffff_ffff;


function initialHashGridTableData(tableScalars: number): Float32Array {
  const tableData = new Float32Array(tableScalars);
  let s = 0x9e3779b1 >>> 0;
  for (let i = 0; i < tableScalars; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    tableData[i] = (s / 0x100000000 - 0.5) * 2e-4;
  }
  return tableData;
}

function packNrcConfigUbo(
  cfg: NrcConfig,
  trainedSteps: number,
  recordStride: number,
  aabbMin: readonly [number, number, number],
  aabbMax: readonly [number, number, number],
): ArrayBuffer {
  const ab = new ArrayBuffer(48);
  const f = new Float32Array(ab);
  const u = new Uint32Array(ab);
  f[0] = aabbMin[0]; f[1] = aabbMin[1]; f[2] = aabbMin[2]; f[3] = cfg.spreadC;
  f[4] = aabbMax[0]; f[5] = aabbMax[1]; f[6] = aabbMax[2];
  u[7] = cfg.recordCap >>> 0;
  u[8] = recordStride >>> 0;
  f[9] = 1.0;

  u[10] = Math.min(U32_MAX, trainedSteps) >>> 0;
  u[11] = Math.min(
    U32_MAX,
    cfg.warmupSteps ?? DEFAULT_NRC_CONFIG.warmupSteps ?? 8,
  ) >>> 0;
  return ab;
}

export type NrcSubsystemLifecycleState =
  | 'new'
  | 'initializing'
  | 'ready'
  | 'disposed';

export interface NrcLearnedStateImportTransaction {
  commit(): void;
  rollback(): void;
  finalize(): void;
}

interface NrcReadbackTicket {
  readonly buffer: GPUBuffer;
  readonly generation: number;
  readonly sequence: number;
  destroyed: boolean;
}

type NrcReadbackState =
  | { readonly kind: 'idle' }
  | { readonly kind: 'copy-pending'; readonly ticket: NrcReadbackTicket }
  | { readonly kind: 'copy-recorded'; readonly ticket: NrcReadbackTicket }
  | { readonly kind: 'mapping'; readonly ticket: NrcReadbackTicket }
  | { readonly kind: 'disposed' };
interface NrcInitializationCandidate {
  trainer?: FusedMlpTrainer;
  tableTrainer?: HashGridTableTrainer;
  trainerDiagnosticsBuffer?: GPUBuffer;
  trainerDiagnosticsReadback?: GPUBuffer;
  tablesBuf?: GPUBuffer;
  levelsBuf?: GPUBuffer;
  inferenceArenaA?: GPUBuffer;
  inferenceArenaB?: GPUBuffer;
  runtimeArena?: GPUBuffer;
  cfgUbo?: GPUBuffer;
}

function destroyCandidateBuffer(buffer: GPUBuffer | undefined): void {
  if (!buffer) return;
  try {
    if (buffer.mapState === 'mapped' || buffer.mapState === 'pending') {
      buffer.unmap();
    }
  } catch {
    // Rollback must continue so one broken wrapper cannot leak later buffers.
  }
  try {
    buffer.destroy();
  } catch {
    // Best-effort cleanup continues through every candidate resource.
  }
}

function rollbackNrcInitialization(candidate: NrcInitializationCandidate): void {
  try { candidate.tableTrainer?.dispose(); } catch { /* continue rollback */ }
  try { candidate.trainer?.dispose(); } catch { /* continue rollback */ }
  destroyCandidateBuffer(candidate.trainerDiagnosticsReadback);
  destroyCandidateBuffer(candidate.trainerDiagnosticsBuffer);
  destroyCandidateBuffer(candidate.cfgUbo);
  destroyCandidateBuffer(candidate.runtimeArena);
  destroyCandidateBuffer(candidate.inferenceArenaB);
  destroyCandidateBuffer(candidate.inferenceArenaA);
  destroyCandidateBuffer(candidate.levelsBuf);
  destroyCandidateBuffer(candidate.tablesBuf);
}
export class NrcSubsystem implements PipelineSubsystem {
  readonly cfg: NrcConfig;
  private readonly _device: GPUDevice;

  private _trainer: FusedMlpTrainer | undefined;
  /** Raw encoded input width (MLP inW). */
  private _inW = 0;
  /** Record stride in f32s (= inW + OUT_W + 3 query-world-pos). */
  private _recordStride = 0;

  // Trainer-owned trainables remain private. The renderer sees one immutable,
  // versioned inference snapshot and one packed mutable runtime arena.
  private _tablesBuf: GPUBuffer | undefined;   // hash-grid feature tables (f32, concatenated)
  private _levelsBuf: GPUBuffer | undefined;   // NrcLevelDesc[] (resolution, tableSize, tableOffset, _pad)
  private _cfgUbo: GPUBuffer | undefined;      // NrcCfgUBO
  private _activeInferenceArena: GPUBuffer | undefined;
  private _spareInferenceArena: GPUBuffer | undefined;
  private _runtimeArena: GPUBuffer | undefined;
  /** Trainer-only counters are separate from per-frame query counters so the
   *  next frame's slot reset cannot erase optimizer health telemetry before
   *  it is copied and mapped. */
  private _trainerDiagnosticsBuffer: GPUBuffer | undefined;
  private _trainerDiagnosticsReadback: GPUBuffer | undefined;
  private _inferenceLayout: NrcInferenceArenaLayout | undefined;
  private _runtimeLayout: NrcRuntimeArenaLayout | undefined;
  private _inferenceEpoch = 0;
  private _runtimeEpoch = 0;
  private _recordByteSize = 0;
  private _readbackByteSize = 0;
  private _sceneBoundsMin: [number, number, number] | undefined;
  private _sceneBoundsMax: [number, number, number] | undefined;

  // ── Hash-grid TABLE training (the trainable encoding). ──
  // Encode-backward scatter → grad finalize → Adam over _tablesBuf with its own
  // moment state. This is what makes the multiresolution encoding LEARN (Müller
  // 2022 Instant-NGP §4); without it the tables stay frozen at random init. The
  // pipeline + its GPU buffers live in the peer {@link HashGridTableTrainer}.
  private _tableTrainer: HashGridTableTrainer | undefined;

  // Host-side staging for the train batch (re-used each frame).
  private _batchX: Float32Array | undefined;
  private _batchY: Float32Array | undefined;
  private _batchPos: Float32Array | undefined;  // [recordCap × 3] dense query positions
  private _readbackState: NrcReadbackState = { kind: 'idle' };
  private _readbackSequence = 0;
  private _lastGpuDiagnostics = new Uint32Array(NRC_DIAGNOSTIC_COUNT);
  /** Last completed optimizer/table-training diagnostic epoch. Kept separate
   *  from query diagnostics because the next query readback replaces the latter. */
  private _lastTrainerDiagnostics = new Uint32Array(NRC_DIAGNOSTIC_COUNT);
  private _hostDroppedNonFiniteRecords = 0;
  private _hostClampedTargets = 0;
  private _readbackOverlapSkips = 0;
  private _staleReadbacks = 0;
  private _trainingFailures = 0;
  private _trainedSteps = 0;
  private _generation = 0;
  private _lifecycleState: NrcSubsystemLifecycleState = 'new';

  constructor(device: GPUDevice, _bglCache: BGLCache, cfg: Partial<NrcConfig> = {}) {
    this._device = device;
    this.cfg = resolveNrcConfig(cfg);
  }

  /**
   * Explicit lifecycle contract. Failed initialization rolls back to `new` and
   * may be retried. `disposed` is terminal; a disposed subsystem never rebuilds.
   */
  get lifecycleState(): NrcSubsystemLifecycleState {
    return this._lifecycleState;
  }

  private _assertReady(method: string): void {
    if (this._lifecycleState !== 'ready') {
      throw new Error(
        `NrcSubsystem.${method}() requires state 'ready'; current state is ` +
        `'${this._lifecycleState}'`,
      );
    }
  }

  private _destroyReadbackTicket(ticket: NrcReadbackTicket): void {
    if (ticket.destroyed) return;
    ticket.destroyed = true;
    try {
      if (ticket.buffer.mapState === 'mapped' || ticket.buffer.mapState === 'pending') ticket.buffer.unmap();
    } catch {
      // Mapping may already have been cancelled by device loss.
    }
    try { ticket.buffer.destroy(); } catch { /* wrapper invalidation is terminal */ }
  }

  /** Last completed GPU epoch plus cumulative host-side rejection telemetry. */
  diagnostics(): NrcDiagnostics {
    const footprint = computeNrcResourceFootprint(this.cfg);
    const gpuDiagnostic = (index: number): number => Math.min(
      U32_MAX,
      (this._lastGpuDiagnostics[index] ?? 0) +
        (this._lastTrainerDiagnostics[index] ?? 0),
    );
    return {
      droppedRecords: gpuDiagnostic(NRC_DIAGNOSTIC_INDEX.droppedRecords),
      saturatedValues: gpuDiagnostic(NRC_DIAGNOSTIC_INDEX.saturatedValues),
      nonFiniteValues: gpuDiagnostic(NRC_DIAGNOSTIC_INDEX.nonFiniteValues),
      invalidPdfs: gpuDiagnostic(NRC_DIAGNOSTIC_INDEX.invalidPdfs),
      droppedUpdates: gpuDiagnostic(NRC_DIAGNOSTIC_INDEX.droppedUpdates),
      hostDroppedNonFiniteRecords: this._hostDroppedNonFiniteRecords,
      hostClampedTargets: this._hostClampedTargets,
      readbackOverlapSkips: this._readbackOverlapSkips,
      staleReadbacks: this._staleReadbacks,
      trainingFailures: this._trainingFailures,
      trainedSteps: this._trainedSteps,
      persistentBufferCount: footprint.persistentBufferCount,
      persistentBufferBytes: footprint.persistentBufferBytes,
      peakResidentBufferCount: footprint.peakResidentBufferCount,
      peakResidentBufferBytes: footprint.peakResidentBufferBytes,
    };
  }

  private _assertInitializationActive(): void {
    if (this._lifecycleState !== 'initializing') {
      throw new Error(
        this._lifecycleState === 'disposed'
          ? 'NrcSubsystem was disposed during initialize(); disposed is terminal'
          : `NrcSubsystem initialization left the initializing state unexpectedly`,
      );
    }
  }

  /** The encoding/MLP config the gi-ris NRC WGSL bakes its sizes from. MUST be
   *  passed to compilePipelines so the shader and these buffers agree. */
  wgslConfig(): RisGiNrcConfig {
    return {
      levels: this.cfg.levels,
      featuresPerEntry: this.cfg.featuresPerEntry,
      oneBlobBins: this.cfg.oneBlobBins,
      width: this.cfg.width,
      outWidth: OUT_W,
      hidden: this.cfg.hidden,
    };
  }

  /** Allocate GPU resources + build the trainer. `aabb*` is the scene bounds the
   *  hash grid normalises query positions into (NRC queries the world vertex). */
  async initialize(
    aabbMin: readonly [number, number, number],
    aabbMax: readonly [number, number, number],
  ): Promise<void> {
    if (this._lifecycleState === 'disposed') {
      throw new Error(
        'NrcSubsystem.initialize() called after dispose(); disposed is terminal',
      );
    }
    if (this._lifecycleState === 'initializing') {
      throw new Error('NrcSubsystem.initialize() is already in progress');
    }
    if (this._lifecycleState === 'ready') {
      throw new Error(
        'NrcSubsystem.initialize() called while ready; use resetForSceneBounds()',
      );
    }

    this._lifecycleState = 'initializing';
    const candidate: NrcInitializationCandidate = {};
    try {
      const d = this._device;
      const cfg = this.cfg;
      const footprint = preflightNrcResources(d, cfg, aabbMin, aabbMax);
      const inW = footprint.inW;
    // record = [inW encoded input | OUT_W radiance target | 3 query world pos].
    // The +3 carries the raw query position so the hash-grid encode-backward can
    // recompute the trilinear corners (the encoded input alone is not invertible
    // — the hash forward collides). Data-only: the gate-OFF path writes no
      // records, so this stride change is byte-invisible when nrcEnabled=0.
      const recordStride = footprint.recordStride;
      const recordByteSize = footprint.recordBytes;
      const readbackByteSize = recordByteSize + NRC_DIAGNOSTIC_BYTES;
      const runtimeLayout = footprint.runtimeArenaLayout;
      const runtimeEpoch = 1;
      const runtimeArena = d.createBuffer({
        label: 'nrc-runtime-arena',
        size: runtimeLayout.byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      candidate.runtimeArena = runtimeArena;
      d.queue.writeBuffer(
        runtimeArena,
        runtimeLayout.headerByteOffset,
        buildNrcRuntimeArenaHeader(
          runtimeLayout, runtimeEpoch, 0, cfg.recordCap, recordStride,
        ) as unknown as BufferSource,
      );

      const trainerDiagnosticsBuffer = d.createBuffer({
        label: 'nrc-trainer-diagnostics',
        size: NRC_DIAGNOSTIC_BYTES,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      candidate.trainerDiagnosticsBuffer = trainerDiagnosticsBuffer;
      d.queue.writeBuffer(
        trainerDiagnosticsBuffer,
        0,
        new Uint32Array(NRC_DIAGNOSTIC_COUNT),
      );
      const trainerDiagnosticsReadback = d.createBuffer({
        label: 'nrc-trainer-diagnostics-readback',
        size: NRC_DIAGNOSTIC_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      candidate.trainerDiagnosticsReadback = trainerDiagnosticsReadback;

    // ── Trainer (the cache MLP) ──
      const spec: FusedNetSpec = { inW, W: cfg.width, outW: OUT_W, hidden: cfg.hidden };
      const tcfg: FusedTrainerConfig = { useF16: cfg.useF16, tileB: cfg.tileB };
      const trainer = new FusedMlpTrainer(
        d,
        spec,
        tcfg,
        trainerDiagnosticsBuffer,
      );
      candidate.trainer = trainer;
      await trainer.build(cfg.recordCap);
      this._assertInitializationActive();
      // He-init the MLP so the query is well-conditioned from frame 0.
      const { w, b } = heInit(trainer);
      trainer.setWeights(w, b);

    // ── Hash-grid feature tables (concatenated f32, all levels) ──
    const F = cfg.featuresPerEntry;
    let totalRows = 0;
    const levelDescs = new Uint32Array(cfg.levels * 4);
    for (let l = 0; l < cfg.levels; l++) {
      levelDescs[l * 4 + 0] = footprint.levelResolutions[l]! >>> 0;
      levelDescs[l * 4 + 1] = cfg.tableSize >>> 0;
      levelDescs[l * 4 + 2] = (totalRows * F) >>> 0; // tableOffset in scalar units
      levelDescs[l * 4 + 3] = 0;
      totalRows += cfg.tableSize;
    }
    const tableScalars = footprint.tableScalars;
    // Small random table init (Instant-NGP §3: U(-1e-4, 1e-4)).
    const tableData = initialHashGridTableData(tableScalars);

    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    // The tables buffer is now WRITTEN-EVERY-FRAME by the table Adam step (it was
    // previously write-once → frozen). It also needs COPY_SRC for liveness probes.
      const tablesBuf = d.createBuffer({
        label: 'nrc-tables', size: Math.max(16, tableScalars * 4),
        usage: ST | GPUBufferUsage.COPY_SRC,
      });

      candidate.tablesBuf = tablesBuf;
      d.queue.writeBuffer(tablesBuf, 0, tableData as unknown as BufferSource);
      const levelsBuf = d.createBuffer({
        // Republished into the spare inference arena after every successful
        // training transaction, so the descriptor source must be copyable.
        label: 'nrc-levels',
        size: Math.max(16, cfg.levels * 16),
        usage: ST | GPUBufferUsage.COPY_SRC,
      });
      candidate.levelsBuf = levelsBuf;
      d.queue.writeBuffer(levelsBuf, 0, levelDescs);

    // ── Hash-grid TABLE trainer (the trainable encoding). ──
    // Owns the encode-backward scatter + grad-finalize + table-Adam pipeline and
    // its GPU buffers (gradTablesFx/F, m/vTables, posBuf, encBwdParams + the two
    // persistent UBOs). It Adam-updates _tablesBuf in place using _trainer's
    // finalized dL/dX. (Müller 2022 Instant-NGP §4.)
      const tableTrainer = new HashGridTableTrainer(d, {
        levels: cfg.levels,
        featuresPerEntry: cfg.featuresPerEntry,
        inW,
        tableScalars,
        recordCap: cfg.recordCap,
        tableLearningRate: cfg.tableLearningRate,
        }, trainerDiagnosticsBuffer);
      candidate.tableTrainer = tableTrainer;
      await tableTrainer.build(
        { gradInputF: trainer.gradInputF!, tablesBuf, levelsBuf },
        aabbMin, aabbMax,
      );
      this._assertInitializationActive();

      const inferenceLayout = footprint.inferenceArenaLayout;
      const inferencePayload = {
        weightsBytes: w.byteLength,
        biasesBytes: b.byteLength,
        tablesBytes: tableData.byteLength,
        levelsBytes: levelDescs.byteLength,
      };
      const inferenceEpoch = 1;
      const makeInferenceArena = (label: string): GPUBuffer => {
        const buffer = d.createBuffer({
          label,
          size: inferenceLayout.byteSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        });
        const bytes = new Uint8Array(buffer.getMappedRange());
        bytes.set(new Uint8Array(
          buildNrcInferenceArenaHeader(inferenceLayout, inferencePayload, inferenceEpoch, 0).buffer,
        ));
        bytes.set(new Uint8Array(w.buffer, w.byteOffset, w.byteLength), inferenceLayout.weightsByteOffset);
        bytes.set(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), inferenceLayout.biasesByteOffset);
        bytes.set(new Uint8Array(tableData.buffer), inferenceLayout.tablesByteOffset);
        bytes.set(new Uint8Array(levelDescs.buffer), inferenceLayout.levelsByteOffset);
        buffer.unmap();
        return buffer;
      };
      const inferenceArenaA = makeInferenceArena('nrc-inference-arena-active');
      candidate.inferenceArenaA = inferenceArenaA;
      const inferenceArenaB = makeInferenceArena('nrc-inference-arena-spare');
      candidate.inferenceArenaB = inferenceArenaB;

    // ── Config UBO (matches NrcCfgUBO in nrcQuery.wgsl: vec3+f32, vec3+u32, ...) ──
      const cfgUbo = d.createBuffer({
        label: 'nrc-cfg', size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      candidate.cfgUbo = cfgUbo;
      // f[9] = cameraPixelPdf — initialised to 1.0 (pinhole, unit resolution).
      // Updated every frame by updateCameraPixelPdf() once the host supplies the
      // camera projection matrix and render resolution.
      d.queue.writeBuffer(cfgUbo, 0, packNrcConfigUbo(
        cfg, 0, recordStride, aabbMin, aabbMax,
      ));

      // Host staging is also candidate state: allocation failure rolls back GPU ownership.
      const batchX = new Float32Array(cfg.recordCap * inW);
      const batchY = new Float32Array(cfg.recordCap * OUT_W);
      const batchPos = new Float32Array(cfg.recordCap * 3);
      this._assertInitializationActive();

      // Publish only after every async build, allocation, upload, and bind succeeds.
      this._trainer = trainer;
      this._tableTrainer = tableTrainer;
      this._tablesBuf = tablesBuf;
      this._levelsBuf = levelsBuf;
      this._cfgUbo = cfgUbo;
      this._activeInferenceArena = inferenceArenaA;
      this._spareInferenceArena = inferenceArenaB;
      this._runtimeArena = runtimeArena;
      this._trainerDiagnosticsBuffer = trainerDiagnosticsBuffer;
      this._trainerDiagnosticsReadback = trainerDiagnosticsReadback;
      this._inferenceLayout = inferenceLayout;
      this._runtimeLayout = runtimeLayout;
      this._inferenceEpoch = inferenceEpoch;
      this._runtimeEpoch = runtimeEpoch;
      this._batchX = batchX;
      this._batchY = batchY;
      this._batchPos = batchPos;
      this._inW = inW;
      this._recordStride = recordStride;
      this._recordByteSize = recordByteSize;
      this._readbackByteSize = readbackByteSize;
      this._sceneBoundsMin = [
        Math.fround(aabbMin[0]),
        Math.fround(aabbMin[1]),
        Math.fround(aabbMin[2]),
      ];
      this._sceneBoundsMax = [
        Math.fround(aabbMax[0]),
        Math.fround(aabbMax[1]),
        Math.fround(aabbMax[2]),
      ];
      this._trainedSteps = 0;
      this._readbackState = { kind: 'idle' };
      this._lifecycleState = 'ready';
    } catch (error) {
      rollbackNrcInitialization(candidate);
      if (this.lifecycleState !== 'disposed') {
        this._lifecycleState = 'new';
      }
      throw error;
    }
  }
  prepareSceneReset(
    encoder: GPUCommandEncoder,
    aabbMin: readonly [number, number, number],
    aabbMax: readonly [number, number, number],
  ): PreparedSceneMutation {
    this._assertReady('prepareSceneReset');
    validateNrcAabb(aabbMin, aabbMax);
    const d = this._device;
    const trainer = this._trainer!;
    const tableTrainer = this._tableTrainer!;
    const tablesBuf = this._tablesBuf!;
    const levelsBuf = this._levelsBuf!;
    const cfgUbo = this._cfgUbo!;
    const inferenceLayout = this._inferenceLayout!;
    const runtimeLayout = this._runtimeLayout!;
    const inferenceCandidate = this._spareInferenceArena!;
    const runtimeArena = this._runtimeArena!;
    const nextEpoch = nextNrcArenaEpoch(Math.max(this._inferenceEpoch, this._runtimeEpoch));
    const nextGeneration = (this._generation + 1) >>> 0;
    const staging: GPUBuffer[] = [];
    const stage = (data: ArrayBufferView | ArrayBuffer): GPUBuffer => {
      const bytes = data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const buffer = d.createBuffer({
        label: 'nrc-scene-reset-staging',
        size: Math.max(4, (bytes.byteLength + 3) & ~3),
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      staging.push(buffer);
      new Uint8Array(buffer.getMappedRange()).set(bytes);
      buffer.unmap();
      return buffer;
    };

    let trainerReset: ReturnType<FusedMlpTrainer['prepareSceneReset']> | null = null;
    let tableReset: ReturnType<HashGridTableTrainer['prepareSceneReset']> | null = null;
    try {
      trainerReset = trainer.prepareSceneReset(encoder);
      tableReset = tableTrainer.prepareSceneReset(encoder, aabbMin, aabbMax);
      const tableScalars = this.cfg.levels * this.cfg.tableSize * this.cfg.featuresPerEntry;
      const tableData = initialHashGridTableData(tableScalars);
      const tableSource = stage(tableData);
      encoder.copyBufferToBuffer(tableSource, 0, tablesBuf, 0, tableData.byteLength);
      // Build the complete spare snapshot from the reset trainer/table state.
      // Header publication is encoded last, so an interrupted command build can
      // never make a partially copied candidate live.
      encoder.copyBufferToBuffer(
        trainer.wMasterGpu!, 0, inferenceCandidate,
        inferenceLayout.weightsByteOffset, trainer.wMaster.length * 4,
      );
      encoder.copyBufferToBuffer(
        trainer.bMasterGpu!, 0, inferenceCandidate,
        inferenceLayout.biasesByteOffset, trainer.bMaster.length * 4,
      );
      encoder.copyBufferToBuffer(
        tablesBuf, 0, inferenceCandidate,
        inferenceLayout.tablesByteOffset, tableData.byteLength,
      );
      encoder.copyBufferToBuffer(
        levelsBuf, 0, inferenceCandidate,
        inferenceLayout.levelsByteOffset, this.cfg.levels * 16,
      );
      const cfgData = packNrcConfigUbo(
        this.cfg, 0, this._recordStride, aabbMin, aabbMax,
      );
      const cfgSource = stage(cfgData);
      encoder.copyBufferToBuffer(cfgSource, 0, cfgUbo, 0, cfgData.byteLength);
      encoder.clearBuffer(runtimeArena, runtimeLayout.recordsByteOffset, runtimeLayout.recordsBytes);
      encoder.clearBuffer(runtimeArena, runtimeLayout.diagnosticsByteOffset, runtimeLayout.diagnosticsBytes);
      encoder.clearBuffer(runtimeArena, runtimeLayout.claimsByteOffset, runtimeLayout.claimsBytes);
      encoder.clearBuffer(
        this._trainerDiagnosticsBuffer!,
        0,
        NRC_DIAGNOSTIC_BYTES,
      );
      const inferenceHeader = buildNrcInferenceArenaHeader(inferenceLayout, {
        weightsBytes: trainer.wMaster.length * 4,
        biasesBytes: trainer.bMaster.length * 4,
        tablesBytes: tableData.byteLength,
        levelsBytes: this.cfg.levels * 16,
      }, nextEpoch, nextGeneration);
      const runtimeHeader = buildNrcRuntimeArenaHeader(
        runtimeLayout, nextEpoch, nextGeneration, this.cfg.recordCap, this._recordStride,
      );
      const inferenceHeaderSource = stage(inferenceHeader);
      const runtimeHeaderSource = stage(runtimeHeader);
      encoder.copyBufferToBuffer(inferenceHeaderSource, 0, inferenceCandidate, 0, inferenceHeader.byteLength);
      encoder.copyBufferToBuffer(
        runtimeHeaderSource, 0, runtimeArena, runtimeLayout.headerByteOffset, runtimeHeader.byteLength,
      );
    } catch (error) {
      const cleanups: SceneMutationCleanup[] = [];
      if (trainerReset) {
        const reset = trainerReset;
        cleanups.push(() => reset.rollback());
      }
      if (tableReset) {
        const reset = tableReset;
        cleanups.push(() => reset.rollback());
      }
      cleanups.push(...staging.map((buffer) => () => buffer.destroy()));
      rethrowWithSceneMutationCleanup(
        error,
        cleanups,
        'NRC scene-reset preparation failed and cleanup also failed',
      );
    }

    const oldGeneration = this._generation;
    const oldTrainedSteps = this._trainedSteps;
    const oldActiveInference = this._activeInferenceArena!;
    const oldSpareInference = this._spareInferenceArena!;
    const oldInferenceEpoch = this._inferenceEpoch;
    const oldRuntimeEpoch = this._runtimeEpoch;
    const oldSceneBoundsMin = this._sceneBoundsMin;
    const oldSceneBoundsMax = this._sceneBoundsMax;
    const nextSceneBoundsMin: [number, number, number] = [
      Math.fround(aabbMin[0]),
      Math.fround(aabbMin[1]),
      Math.fround(aabbMin[2]),
    ];
    const nextSceneBoundsMax: [number, number, number] = [
      Math.fround(aabbMax[0]),
      Math.fround(aabbMax[1]),
      Math.fround(aabbMax[2]),
    ];
    let committed = false;
    const oldGpuDiagnostics = this._lastGpuDiagnostics.slice();
    const oldTrainerDiagnostics = this._lastTrainerDiagnostics.slice();
    let closed = false;
    const destroyStaging = (): void => {
      runSceneMutationCleanups(
        [
          () => trainerReset.finalize(),
          () => tableReset.finalize(),
          ...staging.map((buffer) => () => buffer.destroy()),
        ],
        'NRC scene-reset retirement failed',
      );
    };
    return {
      commit: () => {
        if (closed || committed) return;
        trainerReset.commitCpu();
        tableReset.commitCpu();
        this._generation = nextGeneration;
        this._activeInferenceArena = oldSpareInference;
        this._spareInferenceArena = oldActiveInference;
        this._inferenceEpoch = nextEpoch;
        this._runtimeEpoch = nextEpoch;
        this._sceneBoundsMin = nextSceneBoundsMin;
        this._sceneBoundsMax = nextSceneBoundsMax;
        this._lastGpuDiagnostics.fill(0);
        this._lastTrainerDiagnostics.fill(0);
        this._trainedSteps = 0;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          [
            () => trainerReset.rollback(),
            () => tableReset.rollback(),
            () => {
              if (!committed) return;
              this._generation = oldGeneration;
              this._activeInferenceArena = oldActiveInference;
              this._spareInferenceArena = oldSpareInference;
              this._inferenceEpoch = oldInferenceEpoch;
              this._runtimeEpoch = oldRuntimeEpoch;
              this._sceneBoundsMin = oldSceneBoundsMin;
              this._sceneBoundsMax = oldSceneBoundsMax;
              this._trainedSteps = oldTrainedSteps;
              this._lastGpuDiagnostics.set(oldGpuDiagnostics);
              this._lastTrainerDiagnostics.set(oldTrainerDiagnostics);
            },
            ...staging.map((buffer) => () => buffer.destroy()),
          ],
          'NRC scene-reset rollback failed',
        );
      },
      finalize: () => {
        if (closed) return;
        try {
          void d.queue.onSubmittedWorkDone().then(destroyStaging, destroyStaging);
        } catch {
          destroyStaging();
        }
        closed = true;
      },
    };
  }

  /**
   * Cold-restart learned NRC state after scene-geometry mutation.
   *
   * The compiled query/training pipelines are independent of scene bounds, but
   * the hash-grid normalization, cached records, warmup gate, optimizer moments,
   * MLP weights, and trainable tables are not. Reset those in place so mutation
   * never keeps predictions trained against the previous scene volume.
   */
  resetForSceneBounds(
    aabbMin: readonly [number, number, number],
    aabbMax: readonly [number, number, number],
  ): void {
    this._assertReady('resetForSceneBounds');
    validateNrcAabb(aabbMin, aabbMax);
    const encoder = this._device.createCommandEncoder({ label: 'nrc-scene-reset' });
    const mutation = this.prepareSceneReset(encoder, aabbMin, aabbMax);
    try {
      mutation.commit();
      this._device.queue.submit([encoder.finish()]);
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        [() => mutation.rollback()],
        'NRC scene reset failed and rollback also failed',
      );
    }
    mutation.finalize();
  }

  private _snapshotConfig(): NrcStateConfig {
    return {
      levels: this.cfg.levels,
      featuresPerEntry: this.cfg.featuresPerEntry,
      tableSize: this.cfg.tableSize,
      nMin: this.cfg.nMin,
      growth: this.cfg.growth,
      oneBlobBins: this.cfg.oneBlobBins,
      width: this.cfg.width,
      hidden: this.cfg.hidden,
      spreadC: this.cfg.spreadC,
      recordCap: this.cfg.recordCap,
      learningRate: this.cfg.learningRate,
      tableLearningRate: this.cfg.tableLearningRate,
      useF16: this.cfg.useF16,
      tileB: this.cfg.tileB,
      warmupSteps:
        this.cfg.warmupSteps ?? DEFAULT_NRC_CONFIG.warmupSteps ?? 8,
    };
  }

  /**
   * Read back one coherent learned NRC generation. All trainable sources are
   * copied by one command buffer, so the result cannot mix MLP/table Adam
   * epochs. Frame scratch and diagnostics are intentionally not persistent.
   */
  async exportLearnedState(): Promise<NrcLearnedStateSnapshot | null> {
    if (this._lifecycleState !== 'ready') return null;
    const trainerState = this._trainer!.stateBuffers();
    const tableState = this._tableTrainer!.stateBuffers();
    const boundsMin = this._sceneBoundsMin;
    const boundsMax = this._sceneBoundsMax;
    if (!boundsMin || !boundsMax) {
      throw new Error('NRC scene bounds are unavailable for learned-state export.');
    }
    const config = this._snapshotConfig();
    const sceneBoundsMin: [number, number, number] = [...boundsMin];
    const sceneBoundsMax: [number, number, number] = [...boundsMax];
    const trainedSteps = this._trainedSteps;
    const bytesPerScalar = Float32Array.BYTES_PER_ELEMENT;
    let totalBytes = 0;
    const reserve = (scalars: number, label: string): number => {
      if (!Number.isSafeInteger(scalars) || scalars < 0) {
        throw new RangeError(`${label} has an invalid scalar count.`);
      }
      const bytes = scalars * bytesPerScalar;
      const offset = totalBytes;
      totalBytes += bytes;
      if (!Number.isSafeInteger(totalBytes)) {
        throw new RangeError('NRC learned-state readback exceeds the safe-integer domain.');
      }
      return offset;
    };
    const offsets = {
      weights: reserve(trainerState.weightScalars, 'NRC MLP weights'),
      biases: reserve(trainerState.biasScalars, 'NRC MLP biases'),
      firstMomentWeights: reserve(
        trainerState.weightScalars,
        'NRC MLP firstMomentWeights',
      ),
      secondMomentWeights: reserve(
        trainerState.weightScalars,
        'NRC MLP secondMomentWeights',
      ),
      firstMomentBiases: reserve(
        trainerState.biasScalars,
        'NRC MLP firstMomentBiases',
      ),
      secondMomentBiases: reserve(
        trainerState.biasScalars,
        'NRC MLP secondMomentBiases',
      ),
      tables: reserve(tableState.tableScalars, 'NRC hash-grid tables'),
      tableFirstMoment: reserve(
        tableState.tableScalars,
        'NRC hash-grid firstMoment',
      ),
      tableSecondMoment: reserve(
        tableState.tableScalars,
        'NRC hash-grid secondMoment',
      ),
    };
    const sources: ReadonlyArray<readonly [GPUBuffer, number, number]> = [
      [trainerState.weights, offsets.weights, trainerState.weightScalars],
      [trainerState.biases, offsets.biases, trainerState.biasScalars],
      [
        trainerState.firstMomentWeights,
        offsets.firstMomentWeights,
        trainerState.weightScalars,
      ],
      [
        trainerState.secondMomentWeights,
        offsets.secondMomentWeights,
        trainerState.weightScalars,
      ],
      [
        trainerState.firstMomentBiases,
        offsets.firstMomentBiases,
        trainerState.biasScalars,
      ],
      [
        trainerState.secondMomentBiases,
        offsets.secondMomentBiases,
        trainerState.biasScalars,
      ],
      [tableState.tables, offsets.tables, tableState.tableScalars],
      [
        tableState.firstMoment,
        offsets.tableFirstMoment,
        tableState.tableScalars,
      ],
      [
        tableState.secondMoment,
        offsets.tableSecondMoment,
        tableState.tableScalars,
      ],
    ];
    for (const [source, , scalars] of sources) {
      if (source.size < scalars * bytesPerScalar) {
        throw new Error('An NRC optimizer buffer is smaller than its declared state.');
      }
    }
    const staging = this._device.createBuffer({
      label: 'nrc-learned-state-readback',
      size: Math.max(4, totalBytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let mapped = false;
    try {
      const encoder = this._device.createCommandEncoder({
        label: 'nrc-learned-state-readback',
      });
      for (const [source, offset, scalars] of sources) {
        encoder.copyBufferToBuffer(
          source,
          0,
          staging,
          offset,
          scalars * bytesPerScalar,
        );
      }
      this._device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ, 0, totalBytes);
      mapped = true;
      const range = staging.getMappedRange(0, totalBytes);
      const copy = (offset: number, scalars: number): Float32Array =>
        new Float32Array(range, offset, scalars).slice();
      const snapshot: NrcLearnedStateSnapshot = {
        config,
        sceneBoundsMin,
        sceneBoundsMax,
        trainedSteps,
        mlp: {
          weights: copy(offsets.weights, trainerState.weightScalars),
          biases: copy(offsets.biases, trainerState.biasScalars),
          firstMomentWeights: copy(
            offsets.firstMomentWeights,
            trainerState.weightScalars,
          ),
          secondMomentWeights: copy(
            offsets.secondMomentWeights,
            trainerState.weightScalars,
          ),
          firstMomentBiases: copy(
            offsets.firstMomentBiases,
            trainerState.biasScalars,
          ),
          secondMomentBiases: copy(
            offsets.secondMomentBiases,
            trainerState.biasScalars,
          ),
          adamT: trainerState.adamT,
        },
        hashGrid: {
          tables: copy(offsets.tables, tableState.tableScalars),
          firstMoment: copy(offsets.tableFirstMoment, tableState.tableScalars),
          secondMoment: copy(offsets.tableSecondMoment, tableState.tableScalars),
          adamT: tableState.adamT,
        },
      };
      assertNrcLearnedStateSnapshot(snapshot);
      return snapshot;
    } finally {
      if (mapped || staging.mapState === 'mapped') {
        try { staging.unmap(); } catch { /* preserve export outcome */ }
      }
      try { staging.destroy(); } catch { /* preserve export outcome */ }
    }
  }

  canImportLearnedState(snapshot: unknown): snapshot is NrcLearnedStateSnapshot {
    if (
      this._lifecycleState !== 'ready' ||
      !this._sceneBoundsMin ||
      !this._sceneBoundsMax
    ) {
      return false;
    }
    try {
      assertNrcLearnedStateSnapshot(snapshot);
    } catch {
      return false;
    }
    return (
      nrcStateConfigMatches(snapshot.config, this._snapshotConfig()) &&
      nrcStateBoundsMatch(snapshot, this._sceneBoundsMin, this._sceneBoundsMax)
    );
  }

  /**
   * Prepare a complete candidate learned-state generation. No live handle or
   * CPU convergence counter changes until commit, and no encoded GPU mutation
   * executes until the caller submits the shared command encoder.
   */
  prepareLearnedStateImport(
    encoder: GPUCommandEncoder,
    snapshot: NrcLearnedStateSnapshot,
  ): NrcLearnedStateImportTransaction | null {
    if (!this.canImportLearnedState(snapshot)) return null;
    const trainer = this._trainer!;
    const tableTrainer = this._tableTrainer!;
    let mlpTransaction:
      | ReturnType<FusedMlpTrainer['prepareStateRestore']>
      | null = null;
    let tableTransaction:
      | ReturnType<HashGridTableTrainer['prepareStateRestore']>
      | null = null;
    const staging: GPUBuffer[] = [];
    const stage = (data: ArrayBufferView): GPUBuffer => {
      const buffer = this._device.createBuffer({
        label: 'nrc-learned-state-import-staging',
        size: Math.max(4, (data.byteLength + 3) & ~3),
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      staging.push(buffer);
      new Uint8Array(buffer.getMappedRange()).set(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      );
      buffer.unmap();
      return buffer;
    };
    const oldActiveInference = this._activeInferenceArena!;
    const oldSpareInference = this._spareInferenceArena!;
    const inferenceCandidate = oldSpareInference;
    const inferenceLayout = this._inferenceLayout!;
    const runtimeLayout = this._runtimeLayout!;
    const nextEpoch = nextNrcArenaEpoch(
      Math.max(this._inferenceEpoch, this._runtimeEpoch),
    );
    const nextGeneration = (this._generation + 1) >>> 0;
    try {
      mlpTransaction = trainer.prepareStateRestore(encoder, snapshot.mlp);
      tableTransaction = tableTrainer.prepareStateRestore(
        encoder,
        snapshot.hashGrid,
      );
      const weightsBytes = snapshot.mlp.weights.byteLength;
      const biasesBytes = snapshot.mlp.biases.byteLength;
      const tablesBytes = snapshot.hashGrid.tables.byteLength;
      const levelsBytes = this.cfg.levels * 16;
      encoder.copyBufferToBuffer(
        mlpTransaction.candidateWeightBuffer,
        0,
        inferenceCandidate,
        inferenceLayout.weightsByteOffset,
        weightsBytes,
      );
      encoder.copyBufferToBuffer(
        mlpTransaction.candidateBiasBuffer,
        0,
        inferenceCandidate,
        inferenceLayout.biasesByteOffset,
        biasesBytes,
      );
      encoder.copyBufferToBuffer(
        tableTransaction.candidateTableBuffer,
        0,
        inferenceCandidate,
        inferenceLayout.tablesByteOffset,
        tablesBytes,
      );
      encoder.copyBufferToBuffer(
        this._levelsBuf!,
        0,
        inferenceCandidate,
        inferenceLayout.levelsByteOffset,
        levelsBytes,
      );
      const inferenceHeader = buildNrcInferenceArenaHeader(
        inferenceLayout,
        { weightsBytes, biasesBytes, tablesBytes, levelsBytes },
        nextEpoch,
        nextGeneration,
      );
      const runtimeHeader = buildNrcRuntimeArenaHeader(
        runtimeLayout,
        nextEpoch,
        nextGeneration,
        this.cfg.recordCap,
        this._recordStride,
      );
      const gate = new Uint32Array([
        snapshot.trainedSteps >>> 0,
        (this.cfg.warmupSteps ?? DEFAULT_NRC_CONFIG.warmupSteps ?? 8) >>> 0,
      ]);
      const inferenceHeaderSource = stage(inferenceHeader);
      const runtimeHeaderSource = stage(runtimeHeader);
      const gateSource = stage(gate);
      encoder.copyBufferToBuffer(
        inferenceHeaderSource,
        0,
        inferenceCandidate,
        0,
        inferenceHeader.byteLength,
      );
      encoder.copyBufferToBuffer(
        runtimeHeaderSource,
        0,
        this._runtimeArena!,
        runtimeLayout.headerByteOffset,
        runtimeHeader.byteLength,
      );
      encoder.copyBufferToBuffer(gateSource, 0, this._cfgUbo!, 40, gate.byteLength);
      encoder.clearBuffer(
        this._runtimeArena!,
        runtimeLayout.recordsByteOffset,
        runtimeLayout.recordsBytes,
      );
      encoder.clearBuffer(
        this._runtimeArena!,
        runtimeLayout.diagnosticsByteOffset,
        runtimeLayout.diagnosticsBytes,
      );
      encoder.clearBuffer(
        this._trainerDiagnosticsBuffer!,
        0,
        NRC_DIAGNOSTIC_BYTES,
      );
      encoder.clearBuffer(
        this._runtimeArena!,
        runtimeLayout.claimsByteOffset,
        runtimeLayout.claimsBytes,
      );
    } catch (error) {
      try { tableTransaction?.rollback(); } catch { /* continue rollback */ }
      try { mlpTransaction?.rollback(); } catch { /* continue rollback */ }
      for (const buffer of staging) {
        try { buffer.destroy(); } catch { /* continue rollback */ }
      }
      throw error;
    }

    const oldTables = this._tablesBuf!;
    const oldGeneration = this._generation;
    const oldInferenceEpoch = this._inferenceEpoch;
    const oldRuntimeEpoch = this._runtimeEpoch;
    const oldTrainedSteps = this._trainedSteps;
    const oldDiagnostics = this._lastGpuDiagnostics.slice();
    const oldTrainerDiagnostics = this._lastTrainerDiagnostics.slice();
    let committed = false;
    let closed = false;
    const destroyStaging = (): void => {
      for (const buffer of staging) {
        try { buffer.destroy(); } catch { /* continue retirement */ }
      }
    };
    return {
      commit: () => {
        if (closed || committed) return;
        mlpTransaction.commitCpu();
        tableTransaction.commitCpu();
        this._tablesBuf = tableTransaction.candidateTableBuffer;
        this._activeInferenceArena = oldSpareInference;
        this._spareInferenceArena = oldActiveInference;
        this._generation = nextGeneration;
        this._inferenceEpoch = nextEpoch;
        this._runtimeEpoch = nextEpoch;
        this._trainedSteps = snapshot.trainedSteps;
        this._lastGpuDiagnostics.fill(0);
        this._lastTrainerDiagnostics.fill(0);
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        try { tableTransaction.rollback(); } finally {
          mlpTransaction.rollback();
        }
        if (committed) {
          this._tablesBuf = oldTables;
          this._activeInferenceArena = oldActiveInference;
          this._spareInferenceArena = oldSpareInference;
          this._generation = oldGeneration;
          this._inferenceEpoch = oldInferenceEpoch;
          this._runtimeEpoch = oldRuntimeEpoch;
          this._trainedSteps = oldTrainedSteps;
          this._lastGpuDiagnostics.set(oldDiagnostics);
          this._lastTrainerDiagnostics.set(oldTrainerDiagnostics);
        }
        closed = true;
        destroyStaging();
      },
      finalize: () => {
        if (closed) return;
        mlpTransaction.finalizeSuccess();
        tableTransaction.finalizeSuccess();
        closed = true;
        const retire = (): void => destroyStaging();
        try {
          void this._device.queue.onSubmittedWorkDone().then(retire, retire);
        } catch {
          retire();
        }
      },
    };
  }

  importLearnedState(snapshot: NrcLearnedStateSnapshot): boolean {
    if (!this.canImportLearnedState(snapshot)) return false;
    const encoder = this._device.createCommandEncoder({
      label: 'nrc-learned-state-import',
    });
    const transaction = this.prepareLearnedStateImport(encoder, snapshot);
    if (!transaction) return false;
    try {
      transaction.commit();
      this._device.queue.submit([encoder.finish()]);
    } catch (error) {
      transaction.rollback();
      throw error;
    }
    transaction.finalize();
    return true;
  }

  /** Resources appended to the NRC-specific hybrid-layers group at bindings 7..9. */
  queryBindings(): {
    readonly inferenceArenaBuffer: GPUBuffer;
    readonly runtimeArenaBuffer: GPUBuffer;
    readonly configBuffer: GPUBuffer;
  } {
    this._assertReady('queryBindings');
    return {
      inferenceArenaBuffer: this._activeInferenceArena!,
      runtimeArenaBuffer: this._runtimeArena!,
      configBuffer: this._cfgUbo!,
    };
  }

  /**
   * Clear the per-slot claim buffer and gathered record payloads to zero so
   * every slot is available for this frame's NRC records. Must be called BEFORE
   * the gi-ris NRC pass runs each frame (or at the start of each training
   * window).
   *
   * H27 first-writer-wins: the GPU shader uses atomicCompareExchangeWeak against
   * this buffer — a 0-value means unclaimed, 1 means claimed. Writing zeros here
   * resets all slots so the shader can claim them fresh each frame. Clearing the
   * record buffer at the same boundary preserves the all-zero encoded-input
   * empty-slot contract for slots no invocation fills this frame.
   */
  clearSlotClaims(encoder: GPUCommandEncoder): void {
    this._assertReady('clearSlotClaims');
    const layout = this._runtimeLayout!;
    encoder.clearBuffer(this._runtimeArena!, layout.claimsByteOffset, layout.claimsBytes);
    encoder.clearBuffer(this._runtimeArena!, layout.recordsByteOffset, layout.recordsBytes);
    encoder.clearBuffer(
      this._runtimeArena!, layout.diagnosticsByteOffset, layout.diagnosticsBytes,
    );
  }

  /**
   * Update the per-pixel camera solid-angle pdf in the NRC config UBO.
   * Must be called once per frame (or after every resize) BEFORE the gi-ris
   * NRC pass runs so the WGSL a0 footprint uses the correct camera pdf instead
   * of the hard-coded 1.0 fallback.
   *
   * For a pinhole camera with a column-major perspective projection matrix:
   *   - projMatrix[0], [5] are the x/y focal-length elements fx/fy.
   *   - Centre-pixel solid angle ≈ 4 / (|fx·fy| · W · H).
   *   - The corresponding within-pixel directional pdf is its reciprocal.
   *
   * @param projMatrix  Column-major 4×4 perspective matrix.
   * @param renderWidth  Render resolution width in pixels (internal, not CSS).
   * @param renderHeight Render resolution height in pixels (internal, not CSS).
   */
  updateCameraPixelPdf(
    projMatrix: Float32Array | readonly number[],
    renderWidth: number,
    renderHeight: number,
  ): void {
    this._assertReady('updateCameraPixelPdf');
    if (projMatrix.length < 16) {
      throw new RangeError(
        `NRC projection matrix must contain 16 elements; got ${projMatrix.length}`,
      );
    }
    const focalX = projMatrix[0]!;
    const focalY = projMatrix[5]!;
    if (!Number.isFinite(focalX) || focalX === 0
        || !Number.isFinite(focalY) || focalY === 0) {
      throw new RangeError(
        `NRC projection focal terms must be finite and non-zero; got [${focalX}, ${focalY}]`,
      );
    }
    if (!Number.isSafeInteger(renderWidth) || renderWidth <= 0
        || !Number.isSafeInteger(renderHeight) || renderHeight <= 0) {
      throw new RangeError(
        `NRC render dimensions must be positive safe integers; got ${renderWidth}x${renderHeight}`,
      );
    }
    // Centre-pixel solid angle is 4/(fx*fy*W*H). Using fy² is only correct for
    // a square viewport and overestimates the camera PDF by the aspect ratio.
    const rawPdf = Math.abs(focalX * focalY) * renderWidth * renderHeight / 4;
    const pdf = Math.fround(rawPdf);
    if (!Number.isFinite(pdf) || !(pdf > 0)) {
      throw new RangeError(
        `NRC camera pixel PDF must be finite, positive, and representable as f32; got ${rawPdf}`,
      );
    }
    const tmp = new Float32Array(1);
    tmp[0] = pdf;
    // Byte offset 36 = f32 index 9 in the UBO (after aabbMin, spreadC, aabbMax,
    // recordCap, recordStride — see nrcQuery.wgsl NrcCfgUBO layout).
    this._device.queue.writeBuffer(this._cfgUbo!, 36, tmp);
  }

  private _writeTrainingGateState(trainedSteps = this._trainedSteps): void {
    const tmp = new Uint32Array(2);
    tmp[0] = Math.min(U32_MAX, trainedSteps) >>> 0;
    tmp[1] = Math.min(
      U32_MAX,
      this.cfg.warmupSteps ?? DEFAULT_NRC_CONFIG.warmupSteps ?? 8,
    ) >>> 0;
    this._device.queue.writeBuffer(this._cfgUbo!, 40, tmp);
  }

  /** Record one generation-tagged record+diagnostic snapshot for later mapping. */
  recordCopyForReadback(
    encoder: GPUCommandEncoder,
    publication?: FramePublication,
  ): void {
    if (this._lifecycleState === 'disposed') return;
    this._assertReady('recordCopyForReadback');
    if (this._readbackState.kind !== 'idle') {
      this._readbackOverlapSkips++;
      return;
    }
    const buffer = this._device.createBuffer({
      label: `nrc-readback-${this._readbackSequence + 1}`,
      size: this._readbackByteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const ticket: NrcReadbackTicket = {
      buffer,
      generation: this._generation,
      sequence: this._readbackSequence + 1,
      destroyed: false,
    };
    // Reserve the unique readback slot before publication. A frame transaction
    // may remain open while another caller attempts to encode work, and leaving
    // the state idle here would allocate overlapping tickets that can overwrite
    // each other (or be resurrected by a late publication after dispose()).
    this._readbackState = { kind: 'copy-pending', ticket };
    try {
      const layout = this._runtimeLayout!;
      encoder.copyBufferToBuffer(
        this._runtimeArena!, layout.recordsByteOffset,
        buffer, 0, this._recordByteSize,
      );
      encoder.copyBufferToBuffer(
        this._runtimeArena!, layout.diagnosticsByteOffset,
        buffer, this._recordByteSize, NRC_DIAGNOSTIC_BYTES,
      );
      if (publication == null) {
        // Standalone NRC harnesses submit their own encoder immediately.
        this._readbackSequence = ticket.sequence;
        this._readbackState = { kind: 'copy-recorded', ticket };
      } else {
        publication.stage(
          () => {
            if (this._lifecycleState === 'ready'
                && ticket.generation === this._generation
                && this._readbackState.kind === 'copy-pending'
                && this._readbackState.ticket === ticket
                && !ticket.destroyed) {
              this._readbackSequence = ticket.sequence;
              this._readbackState = { kind: 'copy-recorded', ticket };
              return;
            }
            this._destroyReadbackTicket(ticket);
            if (this._lifecycleState === 'ready'
                && this._readbackState.kind === 'copy-pending'
                && this._readbackState.ticket === ticket) {
              this._readbackState = { kind: 'idle' };
            }
          },
          () => {
            this._destroyReadbackTicket(ticket);
            if (this._lifecycleState === 'ready'
                && this._readbackState.kind === 'copy-pending'
                && this._readbackState.ticket === ticket) {
              this._readbackState = { kind: 'idle' };
            }
          },
        );
      }
    } catch (error) {
      this._destroyReadbackTicket(ticket);
      if (this._lifecycleState === 'ready'
          && this._readbackState.kind === 'copy-pending'
          && this._readbackState.ticket === ticket) {
        this._readbackState = { kind: 'idle' };
      }
      throw error;
    }
  }

  /** Map the unique snapshot, discard stale generations, then submit both trainers atomically. */
  async trainFromRecords(): Promise<void> {
    if (this._lifecycleState === 'disposed') return;
    this._assertReady('trainFromRecords');
    if (this._readbackState.kind !== 'copy-recorded') return;
    const ticket = this._readbackState.ticket;
    this._readbackState = { kind: 'mapping', ticket };
    const trainer = this._trainer!;
    const tableTrainer = this._tableTrainer!;
    let mlpTransaction: ReturnType<FusedMlpTrainer['recordTrainStep']> = null;
    let tableTransaction: ReturnType<HashGridTableTrainer['recordStep']> | null = null;
    let publicationHeaderStaging: GPUBuffer | undefined;
    let published = false;
    let failureAlreadyCounted = false;
    try {
      await ticket.buffer.mapAsync(GPUMapMode.READ);
      if (ticket.destroyed || this.lifecycleState === 'disposed') return;
      const mapped = ticket.buffer.getMappedRange(0, this._readbackByteSize);
      if (ticket.generation !== this._generation) {
        this._staleReadbacks++;
        return;
      }
      this._lastGpuDiagnostics.set(new Uint32Array(
        mapped, this._recordByteSize, NRC_DIAGNOSTIC_COUNT,
      ));
      const unpacked = unpackRecords(
        new Float32Array(mapped, 0, this._recordByteSize / Float32Array.BYTES_PER_ELEMENT),
        this.cfg.recordCap,
        this._recordStride,
        this._inW,
        { x: this._batchX!, y: this._batchY!, pos: this._batchPos! },
      );
      this._hostDroppedNonFiniteRecords += unpacked.droppedNonFinite;
      this._hostClampedTargets += unpacked.clampedTargets;
      if (unpacked.filled === 0) return;

      trainer.setBatch(this._batchX!, this._batchY!);
      const encoder = this._device.createCommandEncoder({ label: `nrc-train-${ticket.sequence}` });
      encoder.clearBuffer(
        this._trainerDiagnosticsBuffer!,
        0,
        NRC_DIAGNOSTIC_BYTES,
      );
      mlpTransaction = trainer.recordTrainStep(encoder, this.cfg.learningRate, unpacked.filled);
      if (!mlpTransaction) return;
      tableTransaction = tableTrainer.recordStep(encoder, this._batchPos!, unpacked.filled);
      const inferenceLayout = this._inferenceLayout!;
      const inferenceCandidate = this._spareInferenceArena!;
      const nextInferenceEpoch = nextNrcArenaEpoch(this._inferenceEpoch);
      const weightsBytes = trainer.wMaster.length * Float32Array.BYTES_PER_ELEMENT;
      const biasesBytes = trainer.bMaster.length * Float32Array.BYTES_PER_ELEMENT;
      const tablesBytes = this.cfg.levels * this.cfg.tableSize
        * this.cfg.featuresPerEntry * Float32Array.BYTES_PER_ELEMENT;
      const levelsBytes = this.cfg.levels * 16;
      encoder.copyBufferToBuffer(
        mlpTransaction.candidateWeightBuffer, 0,
        inferenceCandidate, inferenceLayout.weightsByteOffset, weightsBytes,
      );
      encoder.copyBufferToBuffer(
        mlpTransaction.candidateBiasBuffer, 0,
        inferenceCandidate, inferenceLayout.biasesByteOffset, biasesBytes,
      );
      encoder.copyBufferToBuffer(
        tableTransaction.candidateTableBuffer, 0,
        inferenceCandidate, inferenceLayout.tablesByteOffset, tablesBytes,
      );
      encoder.copyBufferToBuffer(
        this._levelsBuf!, 0,
        inferenceCandidate, inferenceLayout.levelsByteOffset, levelsBytes,
      );
      const publicationHeader = buildNrcInferenceArenaHeader(
        inferenceLayout,
        { weightsBytes, biasesBytes, tablesBytes, levelsBytes },
        nextInferenceEpoch,
        this._generation,
      );
      publicationHeaderStaging = this._device.createBuffer({
        label: `nrc-inference-header-${ticket.sequence}`,
        size: publicationHeader.byteLength,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      new Uint8Array(publicationHeaderStaging.getMappedRange()).set(
        new Uint8Array(publicationHeader.buffer),
      );
      publicationHeaderStaging.unmap();
      // Publication header is the final encoded copy. The active arena remains
      // unchanged until submit + completion + training-gate write all succeed.
      encoder.copyBufferToBuffer(
        publicationHeaderStaging, 0, inferenceCandidate, 0, publicationHeader.byteLength,
      );
      encoder.copyBufferToBuffer(
        this._trainerDiagnosticsBuffer!,
        0,
        this._trainerDiagnosticsReadback!,
        0,
        NRC_DIAGNOSTIC_BYTES,
      );
      const commandBuffer = encoder.finish();
      if (ticket.generation !== this._generation || this._lifecycleState !== 'ready') {
        this._staleReadbacks++;
        mlpTransaction.rollback();
        tableTransaction.rollback();
        return;
      }
      this._device.queue.submit([commandBuffer]);
      await this._device.queue.onSubmittedWorkDone();
      if (ticket.generation !== this._generation || this._lifecycleState !== 'ready') {
        this._staleReadbacks++;
        mlpTransaction.rollback();
        tableTransaction.rollback();
        return;
      }
      const trainerDiagnosticsReadback = this._trainerDiagnosticsReadback!;
      await trainerDiagnosticsReadback.mapAsync(
        GPUMapMode.READ,
        0,
        NRC_DIAGNOSTIC_BYTES,
      );
      let trainerDiagnostics: Uint32Array;
      try {
        trainerDiagnostics = new Uint32Array(
          trainerDiagnosticsReadback.getMappedRange(0, NRC_DIAGNOSTIC_BYTES),
        ).slice();
      } finally {
        trainerDiagnosticsReadback.unmap();
      }
      // Scene reset can commit while the persistent diagnostics buffer is
      // mapping. Revalidate after that asynchronous boundary before publishing
      // either diagnostics or the trained candidate from the old generation.
      if (ticket.generation !== this._generation || this._lifecycleState !== 'ready') {
        this._staleReadbacks++;
        mlpTransaction.rollback();
        tableTransaction.rollback();
        return;
      }
      this._lastTrainerDiagnostics.set(trainerDiagnostics);
      // The gate write is the final fallible operation. Once the live handles
      // below are swapped, publication is irreversible and retirement cleanup
      // must never try to resurrect buffers that may already have been freed.
      const nextTrainedSteps = Math.min(U32_MAX, this._trainedSteps + 1);
      this._writeTrainingGateState(nextTrainedSteps);
      mlpTransaction.commitCpu();
      tableTransaction.commitCpu();
      this._tablesBuf = tableTransaction.candidateTableBuffer;
      const retiredInference = this._activeInferenceArena!;
      this._activeInferenceArena = inferenceCandidate;
      this._spareInferenceArena = retiredInference;
      this._inferenceEpoch = nextInferenceEpoch;
      this._trainedSteps = nextTrainedSteps;
      published = true;

      // Retirement is best-effort after publication. A failed destroy must not
      // roll back to an already-partially-destroyed live set.
      const retirementErrors: unknown[] = [];
      try { mlpTransaction.finalizeSuccess(); } catch (error) { retirementErrors.push(error); }
      try { tableTransaction.finalizeSuccess(); } catch (error) { retirementErrors.push(error); }
      if (retirementErrors.length > 0) {
        this._trainingFailures += retirementErrors.length;
        failureAlreadyCounted = true;
        throw new AggregateError(
          retirementErrors,
          'NRC published a training generation but failed to retire one or more previous resources',
        );
      }
    } catch (error) {
      if (!published) {
        try { mlpTransaction?.rollback(); } catch { /* continue rollback */ }
        try { tableTransaction?.rollback(); } catch { /* continue rollback */ }
      }
      if (this.lifecycleState !== 'disposed' && !failureAlreadyCounted) {
        this._trainingFailures++;
      }
      // The pipeline owns the engine-facing error channel. Do not convert a
      // failed transaction into a fulfilled promise: callers need the original
      // failure after rollback so HybridEngine.onError can report it.
      throw error;
    } finally {
      destroyCandidateBuffer(publicationHeaderStaging);
      this._destroyReadbackTicket(ticket);
      if (this._lifecycleState === 'ready'
          && this._readbackState.kind === 'mapping'
          && this._readbackState.ticket === ticket) {
        this._readbackState = { kind: 'idle' };
      }
    }
  }

  dispose(): void {
    if (this._lifecycleState === 'disposed') return;
    const pendingTicket = this._readbackState.kind === 'copy-pending'
      || this._readbackState.kind === 'copy-recorded'
      || this._readbackState.kind === 'mapping'
      ? this._readbackState.ticket
      : undefined;
    this._lifecycleState = 'disposed';
    this._readbackState = { kind: 'disposed' };
    this._generation = (this._generation + 1) >>> 0;
    if (pendingTicket) this._destroyReadbackTicket(pendingTicket);
    try { this._tableTrainer?.dispose(); } catch { /* continue disposing */ }
    try { this._trainer?.dispose(); } catch { /* continue disposing */ }
    destroyCandidateBuffer(this._trainerDiagnosticsReadback);
    destroyCandidateBuffer(this._trainerDiagnosticsBuffer);
    destroyCandidateBuffer(this._cfgUbo);
    destroyCandidateBuffer(this._runtimeArena);
    destroyCandidateBuffer(this._spareInferenceArena);
    destroyCandidateBuffer(this._activeInferenceArena);
    destroyCandidateBuffer(this._levelsBuf);
    destroyCandidateBuffer(this._tablesBuf);
    this._trainer = undefined;
    this._tableTrainer = undefined;
    this._trainerDiagnosticsReadback = undefined;
    this._trainerDiagnosticsBuffer = undefined;
    this._lastTrainerDiagnostics.fill(0);
    this._tablesBuf = undefined;
    this._levelsBuf = undefined;
    this._cfgUbo = undefined;
    this._activeInferenceArena = undefined;
    this._spareInferenceArena = undefined;
    this._runtimeArena = undefined;
    this._inferenceLayout = undefined;
    this._runtimeLayout = undefined;
    this._inferenceEpoch = 0;
    this._runtimeEpoch = 0;
    this._batchX = undefined;
    this._batchY = undefined;
    this._batchPos = undefined;
    this._inW = 0;
    this._recordStride = 0;
    this._recordByteSize = 0;
    this._readbackByteSize = 0;
    this._sceneBoundsMin = undefined;
    this._sceneBoundsMax = undefined;
  }
}
