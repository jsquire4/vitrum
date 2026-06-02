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
import {
  levelResolution,
  nrcInputWidth,
  type NrcEncodingConfig,
} from './nrcEncoding.js';
import type { RisGiNrcConfig } from '../../shaders/risGiNrc.wgsl.js';
import { getNrcBindGroupLayout, type BGLCache } from '../../pipeline/bindGroupLayouts.js';

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
   *  The gi-ris pass writes one record per half-res pixel into slot
   *  (pixelIdx % recordCap); a larger cap captures more distinct vertices. */
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
};

const OUT_W = 3; // RGB radiance

/** The encoding config the WGSL sizes + the trainer input width derive from. */
function encodingConfig(cfg: NrcConfig, aabbMin: readonly [number, number, number], aabbMax: readonly [number, number, number]): NrcEncodingConfig {
  const levels = [];
  for (let l = 0; l < cfg.levels; l++) {
    levels.push({
      resolution: levelResolution(cfg.nMin, cfg.growth, l),
      tableSize: cfg.tableSize,
      table: new Float32Array(cfg.tableSize * cfg.featuresPerEntry),
    });
  }
  return {
    hashGrid: {
      dim: 3,
      featuresPerEntry: cfg.featuresPerEntry,
      levels,
      aabbMin,
      aabbMax,
    },
    oneBlob: { bins: cfg.oneBlobBins, sigma: 1 / cfg.oneBlobBins },
  };
}

export class NrcSubsystem {
  readonly cfg: NrcConfig;
  private readonly _device: GPUDevice;
  private readonly _bglCache: BGLCache;

  private _trainer!: FusedMlpTrainer;
  /** Raw encoded input width (MLP inW). */
  private _inW = 0;
  /** Record stride in f32s (= inW + OUT_W + 3 query-world-pos). */
  private _recordStride = 0;

  // GPU resources the gi-ris NRC query @group(4) binds.
  private _tablesBuf!: GPUBuffer;   // hash-grid feature tables (f32, concatenated)
  private _levelsBuf!: GPUBuffer;   // NrcLevelDesc[] (resolution, tableSize, tableOffset, _pad)
  private _recordsBuf!: GPUBuffer;  // self-training records (read_write)
  private _cfgUbo!: GPUBuffer;      // NrcCfgUBO
  private _recordReadback!: GPUBuffer; // MAP_READ staging for the record gather
  private _bindGroup!: GPUBindGroup;

  // ── Hash-grid TABLE training (the trainable encoding). ──
  // Encode-backward scatter → grad finalize → Adam over _tablesBuf with its own
  // moment state. This is what makes the multiresolution encoding LEARN (Müller
  // 2022 Instant-NGP §4); without it the tables stay frozen at random init. The
  // pipeline + its GPU buffers live in the peer {@link HashGridTableTrainer}.
  private _tableTrainer!: HashGridTableTrainer;

  // Host-side staging for the train batch (re-used each frame).
  private _batchX!: Float32Array;
  private _batchY!: Float32Array;
  private _batchPos!: Float32Array;  // [recordCap × 3] dense query positions
  private _readPending = false;

  constructor(device: GPUDevice, bglCache: BGLCache, cfg: NrcConfig = DEFAULT_NRC_CONFIG) {
    this._device = device;
    this._bglCache = bglCache;
    this.cfg = cfg;
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
    const d = this._device;
    const cfg = this.cfg;
    const enc = encodingConfig(cfg, aabbMin, aabbMax);
    this._inW = nrcInputWidth(enc);
    // record = [inW encoded input | OUT_W radiance target | 3 query world pos].
    // The +3 carries the raw query position so the hash-grid encode-backward can
    // recompute the trilinear corners (the encoded input alone is not invertible
    // — the hash forward collides). Data-only: the gate-OFF path writes no
    // records, so this stride change is byte-invisible when nrcEnabled=0.
    this._recordStride = this._inW + OUT_W + 3;

    // ── Trainer (the cache MLP) ──
    const spec: FusedNetSpec = { inW: this._inW, W: cfg.width, outW: OUT_W, hidden: cfg.hidden };
    const tcfg: FusedTrainerConfig = { useF16: cfg.useF16, tileB: cfg.tileB };
    this._trainer = new FusedMlpTrainer(d, spec, tcfg);
    await this._trainer.build(cfg.recordCap);
    // He-init the MLP so the query is well-conditioned from frame 0.
    const { w, b } = heInit(this._trainer);
    this._trainer.setWeights(w, b);

    // ── Hash-grid feature tables (concatenated f32, all levels) ──
    const F = cfg.featuresPerEntry;
    let totalRows = 0;
    const levelDescs = new Uint32Array(cfg.levels * 4);
    const tableOffsets: number[] = [];
    for (let l = 0; l < cfg.levels; l++) {
      tableOffsets.push(totalRows * F);
      levelDescs[l * 4 + 0] = levelResolution(cfg.nMin, cfg.growth, l) >>> 0;
      levelDescs[l * 4 + 1] = cfg.tableSize >>> 0;
      levelDescs[l * 4 + 2] = (totalRows * F) >>> 0; // tableOffset in scalar units
      levelDescs[l * 4 + 3] = 0;
      totalRows += cfg.tableSize;
    }
    const tableScalars = totalRows * F;
    // Small random table init (Instant-NGP §3: U(-1e-4, 1e-4)).
    const tableData = new Float32Array(tableScalars);
    let s = 0x9e3779b1 >>> 0;
    for (let i = 0; i < tableScalars; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      tableData[i] = (s / 0x100000000 - 0.5) * 2e-4;
    }

    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    // The tables buffer is now WRITTEN-EVERY-FRAME by the table Adam step (it was
    // previously write-once → frozen). It also needs COPY_SRC for liveness probes.
    this._tablesBuf = d.createBuffer({
      label: 'nrc-tables', size: Math.max(16, tableScalars * 4),
      usage: ST | GPUBufferUsage.COPY_SRC,
    });
    d.queue.writeBuffer(this._tablesBuf, 0, tableData);
    this._levelsBuf = d.createBuffer({ label: 'nrc-levels', size: Math.max(16, cfg.levels * 16), usage: ST });
    d.queue.writeBuffer(this._levelsBuf, 0, levelDescs);

    // ── Hash-grid TABLE trainer (the trainable encoding). ──
    // Owns the encode-backward scatter + grad-finalize + table-Adam pipeline and
    // its GPU buffers (gradTablesFx/F, m/vTables, posBuf, encBwdParams + the two
    // persistent UBOs). It Adam-updates _tablesBuf in place using _trainer's
    // finalized dL/dX. (Müller 2022 Instant-NGP §4.)
    this._tableTrainer = new HashGridTableTrainer(d, {
      levels: cfg.levels,
      featuresPerEntry: cfg.featuresPerEntry,
      inW: this._inW,
      tableScalars,
      recordCap: cfg.recordCap,
      tableLearningRate: cfg.tableLearningRate,
    });
    await this._tableTrainer.build(
      { gradInputF: this._trainer.gradInputF, tablesBuf: this._tablesBuf, levelsBuf: this._levelsBuf },
      aabbMin, aabbMax,
    );

    // ── Record-gather buffer (read_write from the shader; COPY_SRC for readback) ──
    const recordScalars = cfg.recordCap * this._recordStride;
    this._recordsBuf = d.createBuffer({
      label: 'nrc-records',
      size: Math.max(16, recordScalars * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._recordReadback = d.createBuffer({
      label: 'nrc-records-readback',
      size: Math.max(16, recordScalars * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // ── Config UBO (matches NrcCfgUBO in nrcQuery.wgsl: vec3+f32, vec3+u32, ...) ──
    this._cfgUbo = d.createBuffer({ label: 'nrc-cfg', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const ab = new ArrayBuffer(48);
    const f = new Float32Array(ab);
    const u = new Uint32Array(ab);
    f[0] = aabbMin[0]; f[1] = aabbMin[1]; f[2] = aabbMin[2]; f[3] = cfg.spreadC;
    f[4] = aabbMax[0]; f[5] = aabbMax[1]; f[6] = aabbMax[2];
    u[7] = cfg.recordCap >>> 0;
    u[8] = this._recordStride >>> 0;
    d.queue.writeBuffer(this._cfgUbo, 0, ab);

    // ── The @group(4) bind group. nrcWeights/nrcBiases are the trainer's f32
    //    MASTER buffers (always full-precision regardless of useF16). ──
    this._bindGroup = d.createBindGroup({
      label: 'nrc-bind-group',
      layout: getNrcBindGroupLayout(d, this._bglCache),
      entries: [
        { binding: 0, resource: { buffer: this._trainer.wMasterGpu } },
        { binding: 1, resource: { buffer: this._trainer.bMasterGpu } },
        { binding: 2, resource: { buffer: this._tablesBuf } },
        { binding: 3, resource: { buffer: this._levelsBuf } },
        { binding: 4, resource: { buffer: this._recordsBuf } },
        { binding: 5, resource: { buffer: this._cfgUbo } },
      ],
    });

    // (The EncBwdParams UBO + the grad-finalize count UBO are written once inside
    // HashGridTableTrainer.build() above.)

    this._batchX = new Float32Array(cfg.recordCap * this._inW);
    this._batchY = new Float32Array(cfg.recordCap * OUT_W);
    this._batchPos = new Float32Array(cfg.recordCap * 3);
  }

  /** The `@group(4)` NRC bind group the gi-ris NRC pipeline binds at slot 4. */
  bindGroup(): GPUBindGroup {
    return this._bindGroup;
  }

  /** Copy this frame's gathered records into the MAP_READ staging buffer. Called
   *  by the engine on its own command encoder AFTER the gi-ris pass ran (so the
   *  records the gi-ris pass wrote are present). Cheap (one B2B copy). */
  recordCopyForReadback(encoder: GPUCommandEncoder): void {
    encoder.copyBufferToBuffer(
      this._recordsBuf, 0, this._recordReadback, 0, this._recordReadback.size,
    );
  }

  /**
   * Read back the gathered records and run ONE train step (host-owns-cadence).
   * Async (maps the readback buffer); the engine awaits or fires-and-forgets per
   * its cadence policy. Re-entrancy guarded: a still-pending readback skips this
   * frame's train (the next frame picks up fresh records). A record whose RGB
   * target is all-zero is treated as empty (an unfilled slot) and skipped — the
   * batch is zero-padded so the trainer's fixed-size dispatch is unaffected.
   */
  async trainFromRecords(): Promise<void> {
    if (this._readPending) return;
    this._readPending = true;
    try {
      await this._recordReadback.mapAsync(GPUMapMode.READ);
      const raw = new Float32Array(this._recordReadback.getMappedRange());
      const cap = this.cfg.recordCap;
      const stride = this._recordStride;
      const inW = this._inW;
      let filled = 0;
      // Zero the batch, then pack the non-empty records densely. The pos column
      // (offset inW + OUT_W) is repacked in lockstep so sample index s in the
      // trainer batch == sample index s in posBuf == dL/dX row s.
      this._batchX.fill(0);
      this._batchY.fill(0);
      this._batchPos.fill(0);
      for (let rIdx = 0; rIdx < cap; rIdx++) {
        const base = rIdx * stride;
        const tx = raw[base + inW + 0]!;
        const ty = raw[base + inW + 1]!;
        const tz = raw[base + inW + 2]!;
        if (tx === 0 && ty === 0 && tz === 0) continue; // empty slot
        for (let i = 0; i < inW; i++) this._batchX[filled * inW + i] = raw[base + i]!;
        this._batchY[filled * OUT_W + 0] = tx;
        this._batchY[filled * OUT_W + 1] = ty;
        this._batchY[filled * OUT_W + 2] = tz;
        this._batchPos[filled * 3 + 0] = raw[base + inW + OUT_W + 0]!;
        this._batchPos[filled * 3 + 1] = raw[base + inW + OUT_W + 1]!;
        this._batchPos[filled * 3 + 2] = raw[base + inW + OUT_W + 2]!;
        filled++;
      }
      this._recordReadback.unmap();
      if (filled === 0) return; // nothing to learn this frame
      // The trainer dispatch is fixed-size (recordCap samples); the zero-padded
      // tail contributes zero-target samples. Those degrade slowly toward the
      // zero prediction, which is acceptable — most slots fill in a steady-state
      // walkaround frame, and the host could later mask the tail by numSamples.
      this._trainer.setBatch(this._batchX, this._batchY);
      // MLP step — also finalizes dL/dX into trainer.gradInputF (the encode-
      // backward upstream signal). Then scatter dL/dfeature into the trainable
      // hash-grid tables + run the TABLE Adam step (Müller 2022 Instant-NGP §4).
      this._trainer.trainStep(this.cfg.learningRate);
      // TABLE training step (encode-backward scatter → grad finalize → table
      // Adam). Owned by the peer HashGridTableTrainer; it reads the trainer's
      // finalized dL/dX + the dense query positions and Adam-updates _tablesBuf.
      this._tableTrainer.step(this._batchPos, filled);
    } finally {
      this._readPending = false;
    }
  }

  dispose(): void {
    this._tablesBuf?.destroy();
    this._levelsBuf?.destroy();
    this._recordsBuf?.destroy();
    this._recordReadback?.destroy();
    this._cfgUbo?.destroy();
    // The hash-grid TABLE trainer owns its ~8 GPU buffers (grad scatter/finalize,
    // Adam moments, pos staging, persistent UBOs); release them now. dispose() is
    // idempotent + safe if init never ran.
    this._tableTrainer?.dispose();
    // The MLP trainer owns its own ~18 GPU buffers; release them now (it was built
    // in initialize()). dispose() is idempotent + safe if init never ran.
    this._trainer?.dispose();
  }
}

