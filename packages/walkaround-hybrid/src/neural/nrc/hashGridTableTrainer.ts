// hashGridTableTrainer.ts — host driver for the NRC hash-grid feature-TABLE
// training step (the half that makes the multiresolution encoding LEARN).
//
// Müller, Rousselle, Novák, Keller 2021 (NRC) + Müller, Evans, Schied, Keller
// 2022 (Instant-NGP multiresolution hash encoding §4). Peer to {@link
// FusedMlpTrainer}: where that trainer learns the cache MLP, this one runs the
// per-frame TABLE Adam (encode-backward scatter → grad finalize → Adam over the
// feature tables with its own moment state + higher LR). Extracted from
// NrcSubsystem._tableTrainStep (Task 4.5 Theme I) so the subsystem just wires the
// two trainers together.
//
// HOST-OWNS-LIFECYCLE: accepts the device handle but does NOT own it. It owns the
// ~8 GPU buffers it allocates in {@link build} (grad scatter/finalize buffers,
// Adam moment state, query-position staging, and the two persistent UBOs);
// {@link dispose} releases them and {@link NrcSubsystem.dispose} forwards here.
//
// UBOs-ONCE (perf, output-identical): the grad-finalize count UBO and the Adam
// params UBO are allocated ONCE in build() and re-used every frame. The previous
// inline code created two throwaway UBOs PER frame. The count UBO is written once
// at build (its value — tableScalars — is constant). The Adam UBO is re-written
// each step into the SAME buffer because bc1/bc2 advance with the step counter;
// the written bytes are byte-identical to the old per-frame throwaway write.

import { gradFinalizeWgsl } from './wgsl/fusedMlp.wgsl.js';
import { nrcEncodeBackwardWgsl } from './wgsl/nrcEncodeBackward.wgsl.js';
import { ADAM_WGSL } from './fusedMlpTrainer.js';
import { packAdamUbo } from './adamUbo.js';
import { NRC_DIAGNOSTIC_BYTES } from './nrcDiagnostics.js';

/** Sizing + external buffers the table trainer needs. */
export interface HashGridTableTrainerConfig {
  /** Hash-grid resolution levels L. */
  readonly levels: number;
  /** Features per hash-grid entry F. */
  readonly featuresPerEntry: number;
  /** Raw encoded MLP input width inW (= dL/dX row stride). */
  readonly inW: number;
  /** Total hash-grid table feature scalars (Σ_l tableSize·F). */
  readonly tableScalars: number;
  /** Max self-training records gathered per frame (= dense pos capacity). */
  readonly recordCap: number;
  /** Adam learning rate for the feature tables (Instant-NGP §4: ≈10× the MLP LR). */
  readonly tableLearningRate: number;
}

/** External buffers the trainer reads/writes but does NOT own. */
export interface HashGridTableTrainerExternals {
  /** Finalized dL/dX from the MLP backward — the encode-backward upstream signal. */
  readonly gradInputF: GPUBuffer;
  /** The hash-grid feature tables the table Adam updates in place. */
  readonly tablesBuf: GPUBuffer;
  /** Per-level descriptors (resolution, tableSize, tableOffset). */
  readonly levelsBuf: GPUBuffer;
}

interface HashGridTrainableSet {
  tables: GPUBuffer;
  m: GPUBuffer;
  v: GPUBuffer;
  destroyed: boolean;
}

export interface HashGridTableTrainTransaction {
  readonly candidateTableBuffer: GPUBuffer;
  commitCpu(): void;
  rollback(): void;
  finalizeSuccess(): void;
}

export class HashGridTableTrainer {
  private readonly _device: GPUDevice;
  private readonly _cfg: HashGridTableTrainerConfig;
    private _diagnosticsBuffer: GPUBuffer | undefined;
    private _ownsDiagnosticsBuffer = false;

  // OWNED GPU resources.
  private _gradTablesFx!: GPUBuffer;  // i32 fixed-point atomic scatter target
  private _gradTablesF!: GPUBuffer;   // finalized f32
  private _mTables!: GPUBuffer;       // Adam first moment
  private _inFlightCandidate: HashGridTrainableSet | undefined;
  private _spareTrainableSet: HashGridTrainableSet | undefined;
  private _trainableSets: readonly [HashGridTrainableSet, HashGridTrainableSet] | undefined;
  private _encodeBackwardBindGroup: GPUBindGroup | undefined;
  private _gradFinalizeBindGroup: GPUBindGroup | undefined;
  private readonly _adamBindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private _vTables!: GPUBuffer;       // Adam second moment
  private _posBuf!: GPUBuffer;        // [recordCap × 3] dense query world positions
  private _encBwdParamsUbo!: GPUBuffer;
  // Persistent UBOs (allocated once — were per-frame throwaways before).
  private _gradFinUbo!: GPUBuffer;    // grad-finalize count (constant → written once)

  private _allocateTrainableCandidate(): HashGridTrainableSet {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const bytes = Math.max(16, this._cfg.tableScalars * Float32Array.BYTES_PER_ELEMENT);
    const allocated: GPUBuffer[] = [];
    const make = (): GPUBuffer => {
      const buffer = this._device.createBuffer({ label: 'nrc-table-train-candidate', size: bytes, usage });
      allocated.push(buffer);
      return buffer;
    };
    try {
      return { tables: make(), m: make(), v: make(), destroyed: false };
    } catch (error) {
      for (const buffer of allocated) try { buffer.destroy(); } catch { /* continue rollback */ }
      throw error;
    }
  }

  private _liveTrainableSet(): HashGridTrainableSet {
    const existing = this._trainableSets?.find((set) => set.tables === this._ext.tablesBuf);
    if (existing) return existing;
    return {
      tables: this._ext.tablesBuf,
      m: this._mTables,
      v: this._vTables,
      destroyed: false,
    };
  }

  private _buildTrainBindGroups(): void {
    const d = this._device;
    const sets = this._trainableSets;
    if (!sets) throw new Error('NRC table trainable generations are unavailable');
    this._encodeBackwardBindGroup = d.createBindGroup({
      layout: this._pEncodeBackward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._posBuf } },
        { binding: 1, resource: { buffer: this._ext.gradInputF } },
        { binding: 2, resource: { buffer: this._ext.levelsBuf } },
        { binding: 3, resource: { buffer: this._gradTablesFx } },
        { binding: 4, resource: { buffer: this._encBwdParamsUbo } },
        { binding: 5, resource: { buffer: this.diagnosticsBuffer } },
      ],
    });
    this._gradFinalizeBindGroup = d.createBindGroup({
      layout: this._pTableGradFin.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._gradTablesFx } },
        { binding: 1, resource: { buffer: this._gradTablesF } },
        { binding: 2, resource: { buffer: this._gradFinUbo } },
      ],
    });
    for (const set of sets) {
      this._adamBindGroups.set(set.tables, d.createBindGroup({
        layout: this._pTableAdam.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: set.tables } },
          { binding: 1, resource: { buffer: this._gradTablesF } },
          { binding: 2, resource: { buffer: set.m } },
          { binding: 3, resource: { buffer: set.v } },
          { binding: 4, resource: { buffer: this._adamUbo } },
          { binding: 5, resource: { buffer: this.diagnosticsBuffer } },
        ],
      }));
    }
  }

  private _destroyTrainableSet(set: HashGridTrainableSet | undefined): void {
    if (!set || set.destroyed) return;
    set.destroyed = true;
    for (const buffer of [set.tables, set.m, set.v]) {
      try { buffer.destroy(); } catch { /* continue candidate cleanup */ }
    }
  }
  private _adamUbo!: GPUBuffer;       // Adam params (re-written per step: bc1/bc2)

  private _ext!: HashGridTableTrainerExternals;

  private _pEncodeBackward!: GPUComputePipeline;
  private _pTableGradFin!: GPUComputePipeline;
  private _pTableAdam!: GPUComputePipeline;

  private _tableAdamT = 0;
  #disposed = false;

    constructor(device: GPUDevice, cfg: HashGridTableTrainerConfig, diagnosticsBuffer?: GPUBuffer) {
      this._device = device;
      this._cfg = cfg;
      this._diagnosticsBuffer = diagnosticsBuffer;
    }

    private get diagnosticsBuffer(): GPUBuffer {
      if (!this._diagnosticsBuffer) throw new Error('HashGridTableTrainer diagnostics are unavailable before build()');
      return this._diagnosticsBuffer;
    }

  /** Allocate the owned buffers + compile the three pipelines + write the
   *  static encBwdParams AABB + the grad-finalize count UBO (both written once).
   *  `aabb*` is the scene bounds the hash grid normalises query positions into. */
  async build(
    ext: HashGridTableTrainerExternals,
    aabbMin: readonly [number, number, number],
    aabbMax: readonly [number, number, number],
  ): Promise<void> {
    const d = this._device;
      if (!this._diagnosticsBuffer) {
        this._diagnosticsBuffer = d.createBuffer({
          label: 'nrc-table-trainer-diagnostics',
          size: NRC_DIAGNOSTIC_BYTES,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._ownsDiagnosticsBuffer = true;
        d.queue.writeBuffer(this._diagnosticsBuffer, 0, new Uint32Array(NRC_DIAGNOSTIC_BYTES / 4));
      }
    const cfg = this._cfg;
    this._ext = ext;

    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const tableScalars = cfg.tableScalars;

    // gradTablesFx: one atomic<i32> per table scalar (scatter target). gradTablesF:
    // finalized f32. mTables/vTables: Adam moment state (zero-init). posBuf: dense
    // query positions for the encode-backward. (Müller 2022 Instant-NGP §4.)
    this._gradTablesFx = d.createBuffer({ label: 'nrc-gradTablesFx', size: Math.max(16, tableScalars * 4), usage: ST | GPUBufferUsage.COPY_SRC });
    this._gradTablesF = d.createBuffer({ label: 'nrc-gradTablesF', size: Math.max(16, tableScalars * 4), usage: ST | GPUBufferUsage.COPY_SRC });
    // Every transaction seeds its spare generation from the live Adam moments.
    // COPY_SRC is therefore part of the live-generation contract, not merely a
    // readback/debug flag. Real WebGPU validation rejects the generation copy
    // when either moment buffer is STORAGE|COPY_DST only.
    const trainableUsage = ST | GPUBufferUsage.COPY_SRC;
    this._mTables = d.createBuffer({
      label: 'nrc-mTables', size: Math.max(16, tableScalars * 4), usage: trainableUsage,
    });
    this._vTables = d.createBuffer({
      label: 'nrc-vTables', size: Math.max(16, tableScalars * 4), usage: trainableUsage,
    });
    this._posBuf = d.createBuffer({ label: 'nrc-posBuf', size: Math.max(16, cfg.recordCap * 3 * 4), usage: ST });
    this._encBwdParamsUbo = d.createBuffer({ label: 'nrc-encBwdParams', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Persistent UBOs (once-allocated; the previous code threw these away every frame).
    this._gradFinUbo = d.createBuffer({ label: 'nrc-gradFinUbo', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._adamUbo = d.createBuffer({ label: 'nrc-tableAdamUbo', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._spareTrainableSet = this._allocateTrainableCandidate();
    this._trainableSets = [this._liveTrainableSet(), this._spareTrainableSet];

    const buildPipe = async (code: string, entry: string) => {
      const mod = d.createShaderModule({ label: entry, code });
      const ci = await mod.getCompilationInfo?.();
      const errs = ci?.messages?.filter((x) => x.type === 'error') ?? [];
      if (errs.length) {
        throw new Error(`shader '${entry}' compile error: ${errs.map((e) => `${e.lineNum}:${e.message}`).join('; ')}`);
      }
      return d.createComputePipelineAsync({ label: entry, layout: 'auto', compute: { module: mod, entryPoint: entry } });
    };
    this._pEncodeBackward = await buildPipe(
      nrcEncodeBackwardWgsl({ levels: cfg.levels, featuresPerEntry: cfg.featuresPerEntry, inWidth: cfg.inW }),
      'nrcEncodeBackward',
    );
    this._pTableGradFin = await buildPipe(gradFinalizeWgsl(), 'gradFinalize');
    this._pTableAdam = await buildPipe(ADAM_WGSL, 'adamMain');

    // EncBwdParams UBO: aabbMin (vec3) + numActive (u32) + aabbMax (vec3) + pad.
    // numActive is rewritten per frame in step(); the AABB is static.
    {
      const ab = new ArrayBuffer(32);
      const f = new Float32Array(ab);
      f[0] = aabbMin[0]; f[1] = aabbMin[1]; f[2] = aabbMin[2]; // [3] numActive (u32) set per frame
      f[4] = aabbMax[0]; f[5] = aabbMax[1]; f[6] = aabbMax[2];
      d.queue.writeBuffer(this._encBwdParamsUbo, 0, ab);
    }

    // Grad-finalize count UBO — constant (= tableScalars). Written ONCE here.
    {
      const u = new Uint32Array(4); u[0] = tableScalars;
      d.queue.writeBuffer(this._gradFinUbo, 0, u);
    }
    this._buildTrainBindGroups();
  }

  /**
   * Cold-restart the trainable hash-grid for a mutated scene without
   * recompiling pipelines. The scene-dependent data is the AABB in the
   * encode-backward params UBO plus optimizer/gradient buffers.
   */
  prepareSceneReset(
    encoder: GPUCommandEncoder,
    aabbMin: readonly [number, number, number],
    aabbMax: readonly [number, number, number],
  ): { commitCpu(): void; rollback(): void; finalize(): void } {
    const params = new ArrayBuffer(32);
    const floats = new Float32Array(params);
    floats[0] = aabbMin[0]; floats[1] = aabbMin[1]; floats[2] = aabbMin[2];
    floats[4] = aabbMax[0]; floats[5] = aabbMax[1]; floats[6] = aabbMax[2];
    const staging = this._device.createBuffer({
      label: 'nrc-table-reset-staging',
      size: 32,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    try {
      new Uint8Array(staging.getMappedRange()).set(new Uint8Array(params));
      staging.unmap();
      encoder.copyBufferToBuffer(staging, 0, this._encBwdParamsUbo, 0, 32);
      encoder.clearBuffer(this._gradTablesFx);
      encoder.clearBuffer(this._gradTablesF);
      encoder.clearBuffer(this._mTables);
      encoder.clearBuffer(this._vTables);
    } catch (error) {
      try { staging.destroy(); } catch { /* best-effort candidate cleanup */ }
      throw error;
    }
    const oldAdamT = this._tableAdamT;
    let committed = false;
    let closed = false;
    return {
      commitCpu: () => {
        if (closed || committed) return;
        this._tableAdamT = 0;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) this._tableAdamT = oldAdamT;
        closed = true;
        try { staging.destroy(); } catch { /* best-effort cleanup */ }
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        try { staging.destroy(); } catch { /* best-effort cleanup */ }
      },
    };
  }

  resetForSceneBounds(
    aabbMin: readonly [number, number, number],
    aabbMax: readonly [number, number, number],
  ): void {
    const ab = new ArrayBuffer(32);
    const f = new Float32Array(ab);
    f[0] = aabbMin[0]; f[1] = aabbMin[1]; f[2] = aabbMin[2];
    f[4] = aabbMax[0]; f[5] = aabbMax[1]; f[6] = aabbMax[2];
    this._device.queue.writeBuffer(this._encBwdParamsUbo, 0, ab);
    this._tableAdamT = 0;

    const encoder = this._device.createCommandEncoder({ label: 'nrc-table-trainer-scene-reset' });
    encoder.clearBuffer(this._gradTablesFx);
    encoder.clearBuffer(this._gradTablesF);
    encoder.clearBuffer(this._mTables);
    encoder.clearBuffer(this._vTables);
    this._device.queue.submit([encoder.finish()]);
  }

  /** Encode into isolated candidate table/moments; publication is explicit. */
  recordStep(encoder: GPUCommandEncoder, batchPos: Float32Array, numActive: number): HashGridTableTrainTransaction {
    if (this.#disposed) throw new Error('HashGridTableTrainer.recordStep() called after dispose()');
    if (this._inFlightCandidate) throw new Error('HashGridTableTrainer already has an in-flight candidate');
    const d = this._device;
    const cfg = this._cfg;
    if (!Number.isSafeInteger(numActive) || numActive <= 0 || numActive > cfg.recordCap) {
      throw new RangeError(`NRC table active sample count must be in [1, ${cfg.recordCap}]; got ${numActive}`);
    }
    const requiredPositions = numActive * 3;
    if (batchPos.length < requiredPositions) {
      throw new RangeError(`NRC table position batch has ${batchPos.length} scalars; requires ${requiredPositions}`);
    }
    for (let i = 0; i < requiredPositions; i++) {
      if (!Number.isFinite(batchPos[i])) throw new RangeError('NRC table positions must all be finite');
    }
    const old = this._liveTrainableSet();
    const candidate = this._spareTrainableSet ?? this._allocateTrainableCandidate();
    this._spareTrainableSet = undefined;
    this._inFlightCandidate = candidate;
    const bytes = cfg.tableScalars * Float32Array.BYTES_PER_ELEMENT;
    const previousAdamT = this._tableAdamT;
    const nextAdamT = previousAdamT + 1;
    try {
      d.queue.writeBuffer(this._posBuf, 0, batchPos.subarray(0, requiredPositions) as unknown as BufferSource);
      d.queue.writeBuffer(this._encBwdParamsUbo, 12, new Uint32Array([numActive >>> 0]));
      encoder.copyBufferToBuffer(old.tables, 0, candidate.tables, 0, bytes);
      encoder.copyBufferToBuffer(old.m, 0, candidate.m, 0, bytes);
      encoder.copyBufferToBuffer(old.v, 0, candidate.v, 0, bytes);
      encoder.clearBuffer(this._gradTablesFx);
      {
        const bg = this._encodeBackwardBindGroup;
        if (!bg) throw new Error('NRC table encode-backward binding is unavailable');
        const pass = encoder.beginComputePass();
        pass.setPipeline(this._pEncodeBackward); pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(numActive / 64));
        pass.end();
      }
      {
        const bg = this._gradFinalizeBindGroup;
        if (!bg) throw new Error('NRC table grad-finalize binding is unavailable');
        const pass = encoder.beginComputePass();
        pass.setPipeline(this._pTableGradFin); pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(cfg.tableScalars / 64));
        pass.end();
      }
      const bc1 = 1 - Math.pow(0.9, nextAdamT);
      const bc2 = 1 - Math.pow(0.999, nextAdamT);
      d.queue.writeBuffer(this._adamUbo, 0, packAdamUbo(cfg.tableScalars, cfg.tableLearningRate, bc1, bc2));
      {
        const bg = this._adamBindGroups.get(candidate.tables);
        if (!bg) throw new Error('NRC table Adam binding generation is unavailable');
        const pass = encoder.beginComputePass();
        pass.setPipeline(this._pTableAdam); pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(cfg.tableScalars / 64));
        pass.end();
      }
    } catch (error) {
      this._spareTrainableSet = candidate;
      this._inFlightCandidate = undefined;
      throw error;
    }

    let committed = false;
    let closed = false;
    return {
      candidateTableBuffer: candidate.tables,
      commitCpu: () => {
        if (closed || committed) return;
        this._ext = { ...this._ext, tablesBuf: candidate.tables };
        this._mTables = candidate.m;
        this._vTables = candidate.v;
        this._tableAdamT = nextAdamT;
        this._inFlightCandidate = undefined;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) {
          this._ext = { ...this._ext, tablesBuf: old.tables };
          this._mTables = old.m; this._vTables = old.v;
          this._tableAdamT = previousAdamT;
        }
        this._spareTrainableSet = candidate;
        if (this._inFlightCandidate === candidate) this._inFlightCandidate = undefined;
        closed = true;
      },
      finalizeSuccess: () => {
        if (closed) return;
        if (!committed) throw new Error('Cannot finalize an unpublished NRC table candidate');
        this._spareTrainableSet = old;
        closed = true;
      },
    };
  }

  /** Compatibility wrapper: one command buffer and one queue submission. */
  step(batchPos: Float32Array, numActive: number): void {
    const encoder = this._device.createCommandEncoder({ label: 'nrc-table-train-step' });
    const transaction = this.recordStep(encoder, batchPos, numActive);
    try {
      this._device.queue.submit([encoder.finish()]);
      transaction.commitCpu();
      transaction.finalizeSuccess();
    } catch (error) {
      transaction.rollback();
      throw error;
    }
  }

  /** Release every GPU buffer this trainer allocated in {@link build}. Idempotent;
   *  safe before build ran (fields are undefined and skipped). */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const sets = this._trainableSets;
    const liveTable = sets ? this._ext.tablesBuf : undefined;
    const destroyed = new Set<GPUBuffer>();
    const destroyOnce = (buffer: GPUBuffer | undefined): void => {
      if (!buffer || destroyed.has(buffer)) return;
      destroyed.add(buffer);
      try { buffer.destroy(); } catch { /* continue releasing owned buffers */ }
    };

    if (sets) {
      for (const set of sets) {
        // NrcSubsystem owns exactly the published table generation and destroys
        // it after trainer disposal. The trainer owns both moment generations
        // and the non-published spare table.
        if (set.tables !== liveTable) destroyOnce(set.tables);
        destroyOnce(set.m);
        destroyOnce(set.v);
        set.destroyed = true;
      }
    } else {
      this._destroyTrainableSet(this._inFlightCandidate);
      this._destroyTrainableSet(this._spareTrainableSet);
      destroyOnce(this._mTables);
      destroyOnce(this._vTables);
    }
    this._inFlightCandidate = undefined;
    this._spareTrainableSet = undefined;
    this._trainableSets = undefined;

    for (const buffer of [
      this._gradTablesFx, this._gradTablesF,
      this._posBuf, this._encBwdParamsUbo,
      this._gradFinUbo, this._adamUbo,
    ]) destroyOnce(buffer);
    if (this._ownsDiagnosticsBuffer) destroyOnce(this._diagnosticsBuffer);
    this._diagnosticsBuffer = undefined;
    this._ownsDiagnosticsBuffer = false;
    this._encodeBackwardBindGroup = undefined;
    this._gradFinalizeBindGroup = undefined;
    this._adamBindGroups.clear();
  }
}
