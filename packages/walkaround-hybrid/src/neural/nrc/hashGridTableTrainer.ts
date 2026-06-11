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

export class HashGridTableTrainer {
  private readonly _device: GPUDevice;
  private readonly _cfg: HashGridTableTrainerConfig;

  // OWNED GPU resources.
  private _gradTablesFx!: GPUBuffer;  // i32 fixed-point atomic scatter target
  private _gradTablesF!: GPUBuffer;   // finalized f32
  private _mTables!: GPUBuffer;       // Adam first moment
  private _vTables!: GPUBuffer;       // Adam second moment
  private _posBuf!: GPUBuffer;        // [recordCap × 3] dense query world positions
  private _encBwdParamsUbo!: GPUBuffer;
  // Persistent UBOs (allocated once — were per-frame throwaways before).
  private _gradFinUbo!: GPUBuffer;    // grad-finalize count (constant → written once)
  private _adamUbo!: GPUBuffer;       // Adam params (re-written per step: bc1/bc2)

  private _ext!: HashGridTableTrainerExternals;

  private _pEncodeBackward!: GPUComputePipeline;
  private _pTableGradFin!: GPUComputePipeline;
  private _pTableAdam!: GPUComputePipeline;

  private _tableAdamT = 0;
  #disposed = false;

  constructor(device: GPUDevice, cfg: HashGridTableTrainerConfig) {
    this._device = device;
    this._cfg = cfg;
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
    const cfg = this._cfg;
    this._ext = ext;

    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const tableScalars = cfg.tableScalars;

    // gradTablesFx: one atomic<i32> per table scalar (scatter target). gradTablesF:
    // finalized f32. mTables/vTables: Adam moment state (zero-init). posBuf: dense
    // query positions for the encode-backward. (Müller 2022 Instant-NGP §4.)
    this._gradTablesFx = d.createBuffer({ label: 'nrc-gradTablesFx', size: Math.max(16, tableScalars * 4), usage: ST | GPUBufferUsage.COPY_SRC });
    this._gradTablesF = d.createBuffer({ label: 'nrc-gradTablesF', size: Math.max(16, tableScalars * 4), usage: ST | GPUBufferUsage.COPY_SRC });
    this._mTables = d.createBuffer({ label: 'nrc-mTables', size: Math.max(16, tableScalars * 4), usage: ST });
    this._vTables = d.createBuffer({ label: 'nrc-vTables', size: Math.max(16, tableScalars * 4), usage: ST });
    this._posBuf = d.createBuffer({ label: 'nrc-posBuf', size: Math.max(16, cfg.recordCap * 3 * 4), usage: ST });
    this._encBwdParamsUbo = d.createBuffer({ label: 'nrc-encBwdParams', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Persistent UBOs (once-allocated; the previous code threw these away every frame).
    this._gradFinUbo = d.createBuffer({ label: 'nrc-gradFinUbo', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._adamUbo = d.createBuffer({ label: 'nrc-tableAdamUbo', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

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
  }

  /**
   * Run ONE hash-grid TABLE training step (the half that makes the encoding LEARN):
   *   1. upload the dense query positions + active count;
   *   2. clear gradTablesFx, dispatch the encode-backward scatter (reads the
   *      MLP's finalized dL/dX, the first L·F columns are dL/dfeature);
   *   3. finalize gradTablesFx → gradTablesF;
   *   4. Adam over the external tables buffer with this trainer's moment state +
   *      (higher) table LR.
   * EXACT mirror of nrcEncoding.ts hashGridBackward + a standard Adam.
   *
   * @param batchPos  dense query world positions [recordCap × 3]; only the first
   *                  `numActive` rows are uploaded + scattered.
   * @param numActive densely-packed active sample count.
   */
  step(batchPos: Float32Array, numActive: number): void {
    const d = this._device;
    const cfg = this._cfg;
    // (1) upload dense positions + active count.
    d.queue.writeBuffer(this._posBuf, 0, batchPos.subarray(0, numActive * 3) as unknown as BufferSource);
    d.queue.writeBuffer(this._encBwdParamsUbo, 12, new Uint32Array([numActive >>> 0])); // numActive at byte 12

    // (2) clear scatter target, dispatch encode-backward.
    const encClear = d.createCommandEncoder();
    encClear.clearBuffer(this._gradTablesFx);
    d.queue.submit([encClear.finish()]);

    const enc = d.createCommandEncoder();
    {
      const bg = d.createBindGroup({
        layout: this._pEncodeBackward.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._posBuf } },
          { binding: 1, resource: { buffer: this._ext.gradInputF } },
          { binding: 2, resource: { buffer: this._ext.levelsBuf } },
          { binding: 3, resource: { buffer: this._gradTablesFx } },
          { binding: 4, resource: { buffer: this._encBwdParamsUbo } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(this._pEncodeBackward); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(numActive / 64));
      pass.end();
    }
    // (3) finalize gradTablesFx → gradTablesF (i32 fixed-point → f32, clears fx).
    {
      const bg = d.createBindGroup({
        layout: this._pTableGradFin.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._gradTablesFx } },
          { binding: 1, resource: { buffer: this._gradTablesF } },
          { binding: 2, resource: { buffer: this._gradFinUbo } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(this._pTableGradFin); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(cfg.tableScalars / 64));
      pass.end();
    }
    // (4) table Adam over the external tables buffer with this trainer's moments + table LR.
    this._tableAdamT++;
    const bc1 = 1 - Math.pow(0.9, this._tableAdamT);
    const bc2 = 1 - Math.pow(0.999, this._tableAdamT);
    {
      const ab = new ArrayBuffer(48);
      new Uint32Array(ab, 0, 1)[0] = cfg.tableScalars;
      const f = new Float32Array(ab);
      f[4] = cfg.tableLearningRate; f[5] = 0.9; f[6] = 0.999; f[7] = 1e-8; f[8] = bc1; f[9] = bc2;
      d.queue.writeBuffer(this._adamUbo, 0, ab);
      const bg = d.createBindGroup({
        layout: this._pTableAdam.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._ext.tablesBuf } },
          { binding: 1, resource: { buffer: this._gradTablesF } },
          { binding: 2, resource: { buffer: this._mTables } },
          { binding: 3, resource: { buffer: this._vTables } },
          { binding: 4, resource: { buffer: this._adamUbo } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(this._pTableAdam); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(cfg.tableScalars / 64));
      pass.end();
    }
    d.queue.submit([enc.finish()]);
  }

  /** Release every GPU buffer this trainer allocated in {@link build}. Idempotent;
   *  safe before build ran (fields are undefined and skipped). */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this._gradTablesFx?.destroy();
    this._gradTablesF?.destroy();
    this._mTables?.destroy();
    this._vTables?.destroy();
    this._posBuf?.destroy();
    this._encBwdParamsUbo?.destroy();
    this._gradFinUbo?.destroy();
    this._adamUbo?.destroy();
  }
}
