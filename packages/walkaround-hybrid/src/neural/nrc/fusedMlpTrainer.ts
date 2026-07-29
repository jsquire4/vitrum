// fusedMlpTrainer.ts — host driver for the FUSED / TILED NRC MLP training kernel.
//
// Builds the layer-fused forward+backward+Adam GPU pipeline (wgsl/fusedMlp.wgsl.ts)
// for the Müller 2021 Neural Radiance Caching MLP and exposes a single-dispatch-
// per-direction train step. This is the perf-engineering successor to the
// per-layer spike (tools/nrc-spike): activations stay resident in workgroup
// shared memory across all layers, so there is no global-memory round-trip
// between layers — only the per-layer weight matrices are streamed.
//
// The harness is backend-agnostic WebGPU; it runs on lavapipe (CPU, correctness
// only) via Deno today and on a real adapter unchanged. It IS wired into the
// path tracer via {@link NrcSubsystem} (nrcSubsystem.ts), which owns one trainer
// per engine and drives one train step per frame from the gi-ris self-training
// records. The trainer owns the ~18 main GPU buffers it allocates in {@link
// FusedMlpTrainer.build} plus 8 persistent UBOs (paramsUniform, 3 grad-finalize
// counts, 2 Adam slots, 2 downcast counts). {@link FusedMlpTrainer.dispose}
// releases all of them (host-owns-lifecycle), and NrcSubsystem.dispose() forwards.

import {
  fusedForwardWgsl, fusedBackwardWgsl, gradFinalizeWgsl, downcastF16Wgsl,
  fusedMlpWorkgroupStorageBytes,
  type FusedMlpWgslOptions,
} from "./wgsl/fusedMlp.wgsl.js";
import { packAdamUbo } from "./adamUbo.js";
import { NRC_DIAGNOSTIC_BYTES, NRC_DIAGNOSTIC_CONSTANTS_WGSL } from './nrcDiagnostics.js';
import type { NrcMlpStateSnapshot } from './nrcStateSnapshot.js';

// Adam optimizer kernel (same math as the spike; operates on the finalized f32
// grad buffers). Kept inline so the module is self-contained.
const ADAM_WGSL = /* wgsl */`
struct AdamParams {
  count : u32, _p0 : u32, _p1 : u32, _p2 : u32,
  lr : f32, beta1 : f32, beta2 : f32, eps : f32,
  bc1 : f32, bc2 : f32, _p3 : f32, _p4 : f32,
}
@group(0) @binding(0) var<storage, read_write> params : array<f32>;
@group(0) @binding(1) var<storage, read>       grads  : array<f32>;
@group(0) @binding(2) var<storage, read_write> m      : array<f32>;
@group(0) @binding(3) var<storage, read_write> v      : array<f32>;
@group(0) @binding(4) var<uniform>             p      : AdamParams;
@group(0) @binding(5) var<storage, read_write> nrcDiagnostics : array<atomic<u32>>;
${NRC_DIAGNOSTIC_CONSTANTS_WGSL}
fn nrcAdamFinite(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
@compute @workgroup_size(64, 1, 1)
fn adamMain(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.count) { return; }
  let current = params[idx];
  let oldM = m[idx];
  let oldV = v[idx];
  let g = grads[idx];
  if (!nrcAdamFinite(current) || !nrcAdamFinite(oldM) || !nrcAdamFinite(oldV) || oldV < 0.0) {
    params[idx] = select(0.0, current, nrcAdamFinite(current));
    m[idx] = 0.0; v[idx] = 0.0;
    atomicAdd(&nrcDiagnostics[NRC_DIAG_NONFINITE], 1u);
    return;
  }
  if (!nrcAdamFinite(g) || !nrcAdamFinite(p.lr) || !nrcAdamFinite(p.bc1) || !nrcAdamFinite(p.bc2)
      || !(p.lr > 0.0) || !(p.bc1 > 0.0) || !(p.bc2 > 0.0)) {
    atomicAdd(&nrcDiagnostics[NRC_DIAG_NONFINITE], 1u);
    return;
  }
  let mi = p.beta1 * oldM + (1.0 - p.beta1) * g;
  let vi = p.beta2 * oldV + (1.0 - p.beta2) * g * g;
  let mhat = mi / p.bc1;
  let vhat = vi / p.bc2;
  if (!nrcAdamFinite(mi) || !nrcAdamFinite(vi) || !nrcAdamFinite(mhat)
      || !nrcAdamFinite(vhat) || vi < 0.0 || vhat < 0.0) {
    atomicAdd(&nrcDiagnostics[NRC_DIAG_NONFINITE], 1u);
    return;
  }
  let updated = current - p.lr * mhat / (sqrt(vhat) + p.eps);
  if (!nrcAdamFinite(updated)) {
    atomicAdd(&nrcDiagnostics[NRC_DIAG_NONFINITE], 1u);
    return;
  }
  let bounded = clamp(updated, -65504.0, 65504.0);
  if (bounded != updated) { atomicAdd(&nrcDiagnostics[NRC_DIAG_SATURATED], 1u); }
  m[idx] = mi; v[idx] = vi; params[idx] = bounded;
}
`;

// f16 params are stored on the GPU as f16, but Adam math + grad finalize run in
// f32. We keep a *master f32 copy* of weights/biases (standard mixed-precision
// training) and downcast to the f16 forward/backward buffers each step. This
// matches tiny-cuda-nn's mixed-precision recipe and avoids f16 Adam drift.

export interface FusedNetSpec {
  /** raw input width (encoded features). Padded to W internally. */
  inW: number;
  /** hidden width (Müller: 64). */
  W: number;
  /** output width (Müller NRC RGB: 3). */
  outW: number;
  /** hidden node-layers (Müller: 6). */
  hidden: number;
}

export interface FusedTrainerConfig {
  useF16: boolean;
  tileB: number;
}

interface MlpTrainableSet {
  weights: GPUBuffer;
  biases: GPUBuffer;
  wMasterGpu: GPUBuffer;
  bMasterGpu: GPUBuffer;
  mW: GPUBuffer;
  vW: GPUBuffer;
  mB: GPUBuffer;
  vB: GPUBuffer;
  destroyed: boolean;
}

export interface FusedMlpTrainTransaction {
  readonly candidateWeightBuffer: GPUBuffer;
  readonly candidateBiasBuffer: GPUBuffer;
  commitCpu(): void;
  rollback(): void;
  finalizeSuccess(): void;
}

export interface FusedMlpStateBuffers {
  readonly weights: GPUBuffer;
  readonly biases: GPUBuffer;
  readonly firstMomentWeights: GPUBuffer;
  readonly secondMomentWeights: GPUBuffer;
  readonly firstMomentBiases: GPUBuffer;
  readonly secondMomentBiases: GPUBuffer;
  readonly weightScalars: number;
  readonly biasScalars: number;
  readonly adamT: number;
}

export interface FusedMlpStateImportTransaction {
  readonly candidateWeightBuffer: GPUBuffer;
  readonly candidateBiasBuffer: GPUBuffer;
  commitCpu(): void;
  rollback(): void;
  finalizeSuccess(): void;
}

// Per-layer weight/bias offsets in the concatenated param buffers.
interface LayerPlan {
  wOff: number[]; bOff: number[];
  inW: number[]; outW: number[];
  totalW: number; totalB: number;
  wlayers: number;
}

function planLayers(spec: FusedNetSpec): LayerPlan {
  const { W, outW, hidden } = spec;
  // weight layers: layer 0 maps W(in-padded) -> W; hidden-1 more W->W; last W->outW.
  // We pad the raw input to W, so the first GEMM is W->W.
  const widths: number[] = [W]; // node-layer 0 (input, padded to W)
  for (let h = 0; h < hidden; h++) widths.push(W);
  widths.push(outW);
  const wlayers = widths.length - 1;
  const wOff: number[] = [], bOff: number[] = [], li: number[] = [], lo: number[] = [];
  let tw = 0, tb = 0;
  for (let l = 0; l < wlayers; l++) {
    const i = widths[l]!, o = widths[l + 1]!;
    wOff.push(tw); bOff.push(tb);
    li.push(i); lo.push(o);
    tw += o * i; tb += o;
  }
  return { wOff, bOff, inW: li, outW: lo, totalW: tw, totalB: tb, wlayers };
}

function resolveActiveSampleWindow(
  maxSamples: number,
  tileB: number,
  activeSamples?: number,
): { samples: number; tiles: number } {
  const capacity = Math.max(0, Math.floor(maxSamples));
  const requested = activeSamples === undefined
    ? capacity
    : Number.isFinite(activeSamples)
      ? Math.floor(activeSamples)
      : capacity;
  const samples = Math.min(capacity, Math.max(0, requested));
  const tileSize = Math.max(1, Math.floor(tileB));
  return { samples, tiles: samples === 0 ? 0 : Math.ceil(samples / tileSize) };
}

// Pad a Uint16Array to an even length so its byte length is a multiple of 4
// (WebGPU writeBuffer/copy alignment for f16 storage buffers). The trailing
// padding element is never read by the kernel (buffers are sized to match).
function padEvenU16(a: Uint16Array): Uint16Array {
  if ((a.length & 1) === 0) return a;
  const out = new Uint16Array(a.length + 1);
  out.set(a);
  return out;
}

// IEEE-754 binary32 -> binary16, round-to-nearest ties-to-even. NRC normally
// keeps its optimizer master in f32, but the opt-in f16 forward path still
// needs correct subnormal, tie, NaN, and mantissa-carry handling.
function f32ToF16Bits(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  const dv = new DataView(new ArrayBuffer(4));
  for (let k = 0; k < src.length; k++) {
    dv.setFloat32(0, src[k]!, true);
    const x = dv.getUint32(0, true);
    const sign = (x >>> 16) & 0x8000;
    const exp32 = (x >>> 23) & 0xff;
    const mant32 = x & 0x7fffff;

    if (exp32 === 0xff) {
      // Preserve infinities; canonicalise every NaN to a quiet half NaN.
      out[k] = sign | (mant32 === 0 ? 0x7c00 : 0x7e00);
      continue;
    }

    let exp16 = exp32 - 127 + 15;
    if (exp16 >= 0x1f) {
      out[k] = sign | 0x7c00;
      continue;
    }

    if (exp16 <= 0) {
      if (exp16 < -10) {
        out[k] = sign;
        continue;
      }
      const mantissa = mant32 | 0x800000;
      const shift = 14 - exp16;
      let rounded = mantissa >>> shift;
      const remainder = mantissa & ((1 << shift) - 1);
      const halfway = 1 << (shift - 1);
      if (remainder > halfway || (remainder === halfway && (rounded & 1) !== 0)) {
        rounded++;
      }
      out[k] = sign | rounded;
      continue;
    }

    let mant16 = mant32 >>> 13;
    const remainder = mant32 & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (mant16 & 1) !== 0)) {
      mant16++;
      if (mant16 === 0x400) {
        mant16 = 0;
        exp16++;
        if (exp16 >= 0x1f) {
          out[k] = sign | 0x7c00;
          continue;
        }
      }
    }
    out[k] = sign | (exp16 << 10) | mant16;
  }
  return out;
}

function assertMlpStateArray(
  value: unknown,
  expectedLength: number,
  label: string,
  nonNegative = false,
): asserts value is Float32Array {
  if (!(value instanceof Float32Array) || value.length !== expectedLength) {
    throw new RangeError(
      `NRC MLP ${label} must be a ${expectedLength}-element Float32Array.`,
    );
  }
  for (let index = 0; index < value.length; index++) {
    const scalar = value[index]!;
    if (!Number.isFinite(scalar) || (nonNegative && scalar < 0)) {
      throw new RangeError(
        `NRC MLP ${label}[${index}] must be finite` +
        `${nonNegative ? ' and non-negative' : ''}.`,
      );
    }
  }
}

function assertMlpAdamT(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError('NRC MLP adamT must be an unsigned 32-bit integer.');
  }
}

function assertF16Representable(value: Float32Array, label: string): void {
  for (let index = 0; index < value.length; index++) {
    if (Math.abs(value[index]!) > 65504) {
      throw new RangeError(
        `NRC MLP ${label}[${index}] is outside the finite binary16 range.`,
      );
    }
  }
}

export class FusedMlpTrainer {
  device: GPUDevice;
  spec: FusedNetSpec;
  cfg: FusedTrainerConfig;
  plan: LayerPlan;
  node: number; // node-layers incl input+output

  // master f32 weights/biases (mixed precision)
  wMaster!: Float32Array;
  bMaster!: Float32Array;

  // GPU buffers. Typed `GPUBuffer | undefined` honestly (D7.5): undefined before
  // build() and again after dispose(). Public entry points guard with
  // #assertUsable(); internal hot-path reads use `!` (build() guarantees them).
  weights: GPUBuffer | undefined; biases: GPUBuffer | undefined; // f16 or f32 (forward/backward)
  // f32 MASTER copy for Adam (mixed precision). On the f32 path this IS the same
  // role as weights/biases but kept separate so Adam math always runs in f32.
  wMasterGpu: GPUBuffer | undefined; bMasterGpu: GPUBuffer | undefined;
  inputs: GPUBuffer | undefined; targets: GPUBuffer | undefined;
  actsGlob: GPUBuffer | undefined; zGlob: GPUBuffer | undefined;
  gradWfx: GPUBuffer | undefined; gradBfx: GPUBuffer | undefined;  // i32 fixed-point
  gradWf: GPUBuffer | undefined; gradBf: GPUBuffer | undefined;    // f32 finalized
  // dL/dX — gradient w.r.t. the raw (padded) network INPUT, [numSamples × inW].
  // Fixed-point i32 atomic accumulator + finalized f32. The first L·F columns of
  // each sample's row are dL/dfeature for the hash-grid encode (Müller 2022 §4);
  // the NRC encode-backward scatters them into the trainable feature tables.
  gradInputFx: GPUBuffer | undefined; gradInputF: GPUBuffer | undefined;
  mW: GPUBuffer | undefined; vW: GPUBuffer | undefined;
  mB: GPUBuffer | undefined; vB: GPUBuffer | undefined; // Adam state (f32)

  pFwd!: GPUComputePipeline; pBwd!: GPUComputePipeline;
  pGradFin!: GPUComputePipeline; pAdam!: GPUComputePipeline;
  pDowncast?: GPUComputePipeline; // f16 path only

  // Persistent UBOs — allocated ONCE in build(), rewritten in-place per step
  // rather than discarding and re-creating each call (was: 9–11 throwaway
  // GPUBuffers per trainStep). Mirrors the identical fix in HashGridTableTrainer.
  // _paramsUbo: layer-plan + numSamples header (constant after build — written once).
  _paramsUbo: GPUBuffer | undefined;
  // _gradFinUbo{W,B,X}: grad-finalize count UBOs for weights / biases / dL/dX.
  // Weight/bias counts are stable; dL/dX is rewritten for sparse active NRC
  // record windows so the partial final tile does not finalize padded tail rows.
  _gradFinUboW: GPUBuffer | undefined; _gradFinUboB: GPUBuffer | undefined; _gradFinUboX: GPUBuffer | undefined;
  // _adamUbo{W,B}: Adam params UBOs for the weight and bias Adam passes (per-step
  // bc1/bc2 + lr rewritten into the same buffer each step; count is constant).
  _adamUboW: GPUBuffer | undefined; _adamUboB: GPUBuffer | undefined;
  // _downcastUbo{W,B}: count UBOs for the f16-downcast passes (constant after build).
  _downcastUboW?: GPUBuffer; _downcastUboB?: GPUBuffer;

  numSamples = 0;
  adamT = 0;
    #disposed = false;

  private _diagnosticsBuffer: GPUBuffer | undefined;
  private _ownsDiagnosticsBuffer = false;
  private _inFlightCandidate: MlpTrainableSet | undefined;
  private _spareTrainableSet: MlpTrainableSet | undefined;
  private _trainableSets: readonly [MlpTrainableSet, MlpTrainableSet] | undefined;
  private readonly _forwardBindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private readonly _backwardBindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private readonly _gradFinalizeBindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private readonly _adamBindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private readonly _downcastBindGroups = new Map<GPUBuffer, GPUBindGroup>();
  constructor(device: GPUDevice, spec: FusedNetSpec, cfg: FusedTrainerConfig, diagnosticsBuffer?: GPUBuffer) {
    this.device = device; this.spec = spec; this.cfg = cfg;
    this.plan = planLayers(spec);
    this.node = spec.hidden + 2;
    this._diagnosticsBuffer = diagnosticsBuffer;
  }

  get diagnosticsBuffer(): GPUBuffer {
    if (!this._diagnosticsBuffer) throw new Error('FusedMlpTrainer diagnostics are unavailable before build()');
    return this._diagnosticsBuffer;
  }

  private wgslOpts(): FusedMlpWgslOptions {
    return {
      useF16: this.cfg.useF16, W: this.spec.W, OUT_W: this.spec.outW,
      HIDDEN: this.spec.hidden, TILE_B: this.cfg.tileB,
    };
  }

  /** Bytes of one resident scalar (f16=2, f32=4). */

  private _allocateTrainableCandidate(): MlpTrainableSet {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const allocated: GPUBuffer[] = [];
    const make = (bytes: number): GPUBuffer => {
      const buffer = this.device.createBuffer({
        label: 'nrc-mlp-train-candidate',
        size: Math.max(16, (bytes + 3) & ~3),
        usage,
      });
      allocated.push(buffer);
      return buffer;
    };
    const sc = this.scBytes();
    try {
      return {
        weights: make(this.plan.totalW * sc), biases: make(this.plan.totalB * sc),
        wMasterGpu: make(this.plan.totalW * 4), bMasterGpu: make(this.plan.totalB * 4),
        mW: make(this.plan.totalW * 4), vW: make(this.plan.totalW * 4),
        mB: make(this.plan.totalB * 4), vB: make(this.plan.totalB * 4),
        destroyed: false,
      };
    } catch (error) {
      for (const buffer of allocated) try { buffer.destroy(); } catch { /* continue rollback */ }
      throw error;
    }
  }

  private _liveTrainableSet(): MlpTrainableSet {
    return {
      weights: this.weights!, biases: this.biases!,
      wMasterGpu: this.wMasterGpu!, bMasterGpu: this.bMasterGpu!,
      mW: this.mW!, vW: this.vW!, mB: this.mB!, vB: this.vB!,
      destroyed: false,
    };
  }

  private _publishTrainableSet(set: MlpTrainableSet): void {
    this.weights = set.weights; this.biases = set.biases;
    this.wMasterGpu = set.wMasterGpu; this.bMasterGpu = set.bMasterGpu;
    this.mW = set.mW; this.vW = set.vW; this.mB = set.mB; this.vB = set.vB;
  }

  private _buildTrainBindGroups(): void {
    const d = this.device;
    const sets = this._trainableSets;
    if (!sets) throw new Error('NRC MLP trainable generations are unavailable');

    for (const set of sets) {
      this._forwardBindGroups.set(set.weights, d.createBindGroup({
        layout: this.pFwd.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: set.weights } },
          { binding: 1, resource: { buffer: set.biases } },
          { binding: 2, resource: { buffer: this.inputs! } },
          { binding: 3, resource: { buffer: this.actsGlob! } },
          { binding: 4, resource: { buffer: this.zGlob! } },
          { binding: 5, resource: { buffer: this._paramsUbo! } },
        ],
      }));
      this._backwardBindGroups.set(set.weights, d.createBindGroup({
        layout: this.pBwd.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: set.weights } },
          { binding: 1, resource: { buffer: this.targets! } },
          { binding: 2, resource: { buffer: this.actsGlob! } },
          { binding: 3, resource: { buffer: this.zGlob! } },
          { binding: 4, resource: { buffer: this.gradWfx! } },
          { binding: 5, resource: { buffer: this.gradBfx! } },
          { binding: 6, resource: { buffer: this._paramsUbo! } },
          { binding: 7, resource: { buffer: this.gradInputFx! } },
          { binding: 8, resource: { buffer: this.diagnosticsBuffer } },
        ],
      }));
      this._adamBindGroups.set(set.wMasterGpu, d.createBindGroup({
        layout: this.pAdam.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: set.wMasterGpu } },
          { binding: 1, resource: { buffer: this.gradWf! } },
          { binding: 2, resource: { buffer: set.mW } },
          { binding: 3, resource: { buffer: set.vW } },
          { binding: 4, resource: { buffer: this._adamUboW! } },
          { binding: 5, resource: { buffer: this.diagnosticsBuffer } },
        ],
      }));
      this._adamBindGroups.set(set.bMasterGpu, d.createBindGroup({
        layout: this.pAdam.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: set.bMasterGpu } },
          { binding: 1, resource: { buffer: this.gradBf! } },
          { binding: 2, resource: { buffer: set.mB } },
          { binding: 3, resource: { buffer: set.vB } },
          { binding: 4, resource: { buffer: this._adamUboB! } },
          { binding: 5, resource: { buffer: this.diagnosticsBuffer } },
        ],
      }));
      if (this.cfg.useF16) {
        this._downcastBindGroups.set(set.wMasterGpu, d.createBindGroup({
          layout: this.pDowncast!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: set.wMasterGpu } },
            { binding: 1, resource: { buffer: set.weights } },
            { binding: 2, resource: { buffer: this._downcastUboW! } },
          ],
        }));
        this._downcastBindGroups.set(set.bMasterGpu, d.createBindGroup({
          layout: this.pDowncast!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: set.bMasterGpu } },
            { binding: 1, resource: { buffer: set.biases } },
            { binding: 2, resource: { buffer: this._downcastUboB! } },
          ],
        }));
      }
    }

    for (const [ubo, fx, f] of [
      [this._gradFinUboW!, this.gradWfx!, this.gradWf!],
      [this._gradFinUboB!, this.gradBfx!, this.gradBf!],
      [this._gradFinUboX!, this.gradInputFx!, this.gradInputF!],
    ] as const) {
      this._gradFinalizeBindGroups.set(ubo, d.createBindGroup({
        layout: this.pGradFin.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fx } },
          { binding: 1, resource: { buffer: f } },
          { binding: 2, resource: { buffer: ubo } },
        ],
      }));
    }
  }

  private _destroyTrainableSet(set: MlpTrainableSet | undefined): void {
    if (!set || set.destroyed) return;
    set.destroyed = true;
    for (const buffer of [set.weights, set.biases, set.wMasterGpu, set.bMasterGpu, set.mW, set.vW, set.mB, set.vB]) {
      try { buffer.destroy(); } catch { /* continue candidate cleanup */ }
    }
  }
  private scBytes(): number { return this.cfg.useF16 ? 2 : 4; }

    /** Guard at every public-method entry: a disposed trainer's buffers are
     *  destroyed, so any further use would touch destroyed GPU resources. Fail
     *  loudly with a clear error instead (D7.5). */
    #assertUsable(method: string): void {
      if (this.#disposed) {
        throw new Error(`FusedMlpTrainer.${method}() called after dispose() — the trainer's GPU buffers are destroyed`);
      }
    }

    async build(numSamples: number) {
      this.#assertUsable('build');
      const d = this.device;
      if (!this._diagnosticsBuffer) {
        this._diagnosticsBuffer = d.createBuffer({
          label: 'nrc-trainer-diagnostics',
          size: NRC_DIAGNOSTIC_BYTES,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._ownsDiagnosticsBuffer = true;
        d.queue.writeBuffer(this._diagnosticsBuffer, 0, new Uint32Array(NRC_DIAGNOSTIC_BYTES / 4));
      }
      this.numSamples = numSamples;
      const opts = this.wgslOpts();
      const sc = this.scBytes();

      // Exact shader footprint: two TILE_B x W arrays; weights stay global.
      const sharedBytes = fusedMlpWorkgroupStorageBytes({
        useF16: this.cfg.useF16,
        W: this.spec.W,
        TILE_B: this.cfg.tileB,
      });
    const lim = d.limits.maxComputeWorkgroupStorageSize ?? 16384;
    if (sharedBytes > lim) {
      throw new Error(`fused MLP shared-mem ${sharedBytes}B exceeds adapter limit ${lim}B ` +
        `(tileB=${this.cfg.tileB}, W=${this.spec.W}, f16=${this.cfg.useF16}). Reduce tileB.`);
    }

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    // WebGPU requires storage-buffer sizes (and B2B copy sizes) to be 4-byte
    // aligned. f16 buffers with an odd element count (e.g. 19 biases = 38 B)
    // violate this, so round every allocation up to a multiple of 4.
    const mk = (bytes: number, extra = 0) =>
      d.createBuffer({ size: Math.max(16, (bytes + 3) & ~3), usage: usage | extra });

    const node = this.node, W = this.spec.W;
    this.weights = mk(this.plan.totalW * sc);
    this.biases = mk(this.plan.totalB * sc);
    this.wMasterGpu = mk(this.plan.totalW * 4); // f32 master for Adam
    this.bMasterGpu = mk(this.plan.totalB * 4);
    this.inputs = mk(numSamples * this.spec.inW * sc);
    this.targets = mk(numSamples * this.spec.outW * 4); // f32 targets
    this.actsGlob = mk(numSamples * node * W * sc);
    this.zGlob = mk(numSamples * node * W * sc);
    this.gradWfx = mk(this.plan.totalW * 4);
    this.gradBfx = mk(this.plan.totalB * 4);
    this.gradWf = mk(this.plan.totalW * 4);
    this.gradBf = mk(this.plan.totalB * 4);
    // dL/dX accumulators, [numSamples × inW] (raw encoded input width). i32 fixed
    // point (matches the grad-atomic discipline) + f32 finalized for the scatter.
    this.gradInputFx = mk(numSamples * this.spec.inW * 4);
    this.gradInputF = mk(numSamples * this.spec.inW * 4);
    this.mW = mk(this.plan.totalW * 4); this.vW = mk(this.plan.totalW * 4);
    this.mB = mk(this.plan.totalB * 4); this.vB = mk(this.plan.totalB * 4);
    // Allocate the rollback target once. Training ping-pongs live/spare sets so
    // an epoch never allocates or destroys GPU buffers on the hot path.
    this._spareTrainableSet = this._allocateTrainableCandidate();
    this._trainableSets = [this._liveTrainableSet(), this._spareTrainableSet];

    const pipe = async (code: string, entry: string) => {
      const m = d.createShaderModule({ label: entry, code });
      const cinfo = await m.getCompilationInfo?.();
      const errs = cinfo?.messages?.filter((x) => x.type === "error") ?? [];
      if (errs.length) {
        throw new Error(`shader '${entry}' compile error: ${errs.map((e) => `${e.lineNum}:${e.message}`).join("; ")}`);
      }
      return await d.createComputePipelineAsync({ label: entry, layout: "auto", compute: { module: m, entryPoint: entry } });
    };
    this.pFwd = await pipe(fusedForwardWgsl(opts), "fusedForward");
    this.pBwd = await pipe(fusedBackwardWgsl(opts), "fusedBackward");
    this.pGradFin = await pipe(gradFinalizeWgsl(), "gradFinalize");
    this.pAdam = await pipe(ADAM_WGSL, "adamMain");
    if (this.cfg.useF16) this.pDowncast = await pipe(downcastF16Wgsl(), "downcast");

    // Persistent UBOs — allocated once here; rewritten in-place per step.
    // (Previously each record* call created a throwaway buffer: 9–11 leaks/frame.)
    const mkUbo = (size: number) =>
      d.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // paramsUniform: 4-u32 header + array<vec4<u32>, wlayers>. Content is
    // constant after build, so written once here (not per step).
    {
      const wl = this.plan.wlayers;
      const u = new Uint32Array(4 + wl * 4);
      const win = this.activeWindow(numSamples);
      u[0] = win.samples; u[1] = this.spec.inW; u[2] = 0; u[3] = win.tiles;
      for (let l = 0; l < wl; l++) {
        const base = 4 + l * 4;
        u[base + 0] = this.plan.wOff[l]! >>> 0;
        u[base + 1] = this.plan.bOff[l]! >>> 0;
        u[base + 2] = this.plan.inW[l]! >>> 0;
        u[base + 3] = this.plan.outW[l]! >>> 0;
      }
      this._paramsUbo = mkUbo(u.byteLength);
      d.queue.writeBuffer(this._paramsUbo, 0, u);
    }

    // Grad-finalize count UBOs. Seed them with full-capacity counts; recordGradFinalize()
    // rewrites the same buffers per call so dL/dX can use sparse active batches.
    const writeCount = (ub: GPUBuffer, count: number) => {
      const u = new Uint32Array(4); u[0] = count;
      d.queue.writeBuffer(ub, 0, u);
    };
    this._gradFinUboW = mkUbo(16); writeCount(this._gradFinUboW, this.plan.totalW);
    this._gradFinUboB = mkUbo(16); writeCount(this._gradFinUboB, this.plan.totalB);
    this._gradFinUboX = mkUbo(16); writeCount(this._gradFinUboX, numSamples * this.spec.inW);

    // Adam UBOs: count is constant; bc1/bc2/lr are rewritten each step.
    this._adamUboW = mkUbo(48);
    this._adamUboB = mkUbo(48);

    // Downcast UBOs (f16 path only): count is constant, written once here.
    if (this.cfg.useF16) {
      this._downcastUboW = mkUbo(16); writeCount(this._downcastUboW, this.plan.totalW);
      this._downcastUboB = mkUbo(16); writeCount(this._downcastUboB, this.plan.totalB);
    }
    this._buildTrainBindGroups();
  }

  /**
   * Release every GPU buffer this trainer allocated in {@link build}
   * (host-owns-lifecycle). Idempotent: a second call is a safe no-op, and it is
   * also safe to call before {@link build} has run (the buffer fields are simply
   * undefined and skipped). The owning {@link NrcSubsystem.dispose} forwards here
   * so the trainer's ~26 buffers (18 main + 8 persistent UBOs) don't leak until device teardown.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const trainableSets = this._trainableSets;
    if (trainableSets) {
      for (const set of trainableSets) this._destroyTrainableSet(set);
    } else {
      this._destroyTrainableSet(this._inFlightCandidate);
      this._destroyTrainableSet(this._spareTrainableSet);
    }
    this._spareTrainableSet = undefined;
    this._inFlightCandidate = undefined;
    // If both generations were installed, their buffers were destroyed through
    // the generation sets above. The fallback covers a partial build.
    const buffers: (GPUBuffer | undefined)[] = [
      ...(trainableSets ? [] : [
        this.weights, this.biases,
        this.wMasterGpu, this.bMasterGpu,
        this.mW, this.vW, this.mB, this.vB,
      ]),
      this.inputs, this.targets,
      this.actsGlob, this.zGlob,
      this.gradWfx, this.gradBfx,
      this.gradWf, this.gradBf,
      this.gradInputFx, this.gradInputF,
      // Persistent UBOs (BUG-1 fix — these were previously thrown away per step).
      this._paramsUbo,
      this._gradFinUboW, this._gradFinUboB, this._gradFinUboX,
      this._adamUboW, this._adamUboB,
      this._downcastUboW, this._downcastUboB,
    ];
    for (const buf of buffers) {
      try {
        buf?.destroy();
      } catch {
        // Initialization rollback must keep releasing later owned buffers even
        // if a test double or invalidated wrapper rejects one destroy call.
      }
    }
      if (this._ownsDiagnosticsBuffer) {
        try { this._diagnosticsBuffer?.destroy(); } catch { /* continue cleanup */ }
      }
      this._diagnosticsBuffer = undefined;
      this._ownsDiagnosticsBuffer = false;
    // Null the references (honest `GPUBuffer | undefined` types — no casts) so
      this._forwardBindGroups.clear();
      this._backwardBindGroups.clear();
      this._gradFinalizeBindGroups.clear();
      this._adamBindGroups.clear();
      this._downcastBindGroups.clear();
      this._trainableSets = undefined;
    // the GC can reclaim the wrappers; #assertUsable at every public-method
    // entry makes a stray post-dispose use fail loudly with a clear error.
    this.weights = this.biases = undefined;
    this.wMasterGpu = this.bMasterGpu = undefined;
    this.inputs = this.targets = undefined;
    this.actsGlob = this.zGlob = undefined;
    this.gradWfx = this.gradBfx = undefined;
    this.gradWf = this.gradBf = undefined;
    this.gradInputFx = this.gradInputF = undefined;
    this.mW = this.vW = this.mB = this.vB = undefined;
    this._paramsUbo = undefined;
    this._gradFinUboW = this._gradFinUboB = this._gradFinUboX = undefined;
    this._adamUboW = this._adamUboB = undefined;
    // exactOptionalPropertyTypes: assigning `undefined` directly is not valid;
    // delete the optional fields instead so they revert to absent.
    delete this._downcastUboW;
    delete this._downcastUboB;
  }

  /** Upload master weights/biases (f32) and downcast to the GPU forward buffers. */
  prepareSceneReset(encoder: GPUCommandEncoder): {
    commitCpu(): void;
    rollback(): void;
    finalize(): void;
  } {
    this.#assertUsable('prepareSceneReset');
    const { w, b } = heInit(this);
    const uploads: GPUBuffer[] = [];
    const stage = (data: ArrayBufferView): GPUBuffer => {
      const byteLength = Math.max(4, (data.byteLength + 3) & ~3);
      const buffer = this.device.createBuffer({
        label: 'nrc-reset-staging',
        size: byteLength,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      uploads.push(buffer);
      new Uint8Array(buffer.getMappedRange()).set(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      );
      buffer.unmap();
      return buffer;
    };
    try {
      const wMaster = stage(w);
      const bMaster = stage(b);
      const wForwardData = this.cfg.useF16 ? padEvenU16(f32ToF16Bits(w)) : w;
      const bForwardData = this.cfg.useF16 ? padEvenU16(f32ToF16Bits(b)) : b;
      const wForward = stage(wForwardData);
      const bForward = stage(bForwardData);
      encoder.copyBufferToBuffer(wMaster, 0, this.wMasterGpu!, 0, w.byteLength);
      encoder.copyBufferToBuffer(bMaster, 0, this.bMasterGpu!, 0, b.byteLength);
      encoder.copyBufferToBuffer(wForward, 0, this.weights!, 0, wForwardData.byteLength);
      encoder.copyBufferToBuffer(bForward, 0, this.biases!, 0, bForwardData.byteLength);
      for (const buffer of [
        this.inputs, this.targets, this.actsGlob, this.zGlob,
        this.gradWfx, this.gradBfx, this.gradWf, this.gradBf,
        this.gradInputFx, this.gradInputF,
        this.mW, this.vW, this.mB, this.vB,
      ]) {
        encoder.clearBuffer(buffer!);
      }
    } catch (error) {
      for (const buffer of uploads) {
        try {
          buffer.destroy();
        } catch { /* continue releasing every staging buffer */ }
      }
      throw error;
    }

    const oldW = this.wMaster;
    const oldB = this.bMaster;
    const oldAdamT = this.adamT;
    let committed = false;
    let closed = false;
    const destroyUploads = (): void => {
      for (const buffer of uploads) {
        try {
          buffer.destroy();
        } catch { /* cleanup is best effort and must not skip later buffers */ }
      }
    };
    return {
      commitCpu: () => {
        if (closed || committed) return;
        this.wMaster = w.slice();
        this.bMaster = b.slice();
        this.adamT = 0;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) {
          this.wMaster = oldW;
          this.bMaster = oldB;
          this.adamT = oldAdamT;
        }
        closed = true;
        destroyUploads();
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        destroyUploads();
      },
    };
  }

  setWeights(w: Float32Array, b: Float32Array) {
    this.#assertUsable('setWeights');
    if (w.length !== this.plan.totalW || b.length !== this.plan.totalB) {
      throw new RangeError(`FusedMlpTrainer.setWeights expected ${this.plan.totalW} weights and ${this.plan.totalB} biases; got ${w.length} and ${b.length}`);
    }
    for (const value of w) if (!Number.isFinite(value)) throw new RangeError('FusedMlpTrainer weights must all be finite');
    for (const value of b) if (!Number.isFinite(value)) throw new RangeError('FusedMlpTrainer biases must all be finite');
    this.wMaster = w.slice();
    this.bMaster = b.slice();
    this.pushWeightsToGpu();
  }

  /** Live f32 optimizer buffers used by the engine-level coherent readback. */
  stateBuffers(): FusedMlpStateBuffers {
    this.#assertUsable('stateBuffers');
    return {
      weights: this.wMasterGpu!,
      biases: this.bMasterGpu!,
      firstMomentWeights: this.mW!,
      secondMomentWeights: this.vW!,
      firstMomentBiases: this.mB!,
      secondMomentBiases: this.vB!,
      weightScalars: this.plan.totalW,
      biasScalars: this.plan.totalB,
      adamT: this.adamT,
    };
  }

  /**
   * Record a complete optimizer generation into the isolated spare set.
   * Publication is pointer/CPU-state only; callers may therefore compose this
   * transaction with DDGI, DI/GI, PPG, and NRC inference publication before
   * submitting one command buffer.
   */
  prepareStateRestore(
    encoder: GPUCommandEncoder,
    state: NrcMlpStateSnapshot,
  ): FusedMlpStateImportTransaction {
    this.#assertUsable('prepareStateRestore');
    if (this._inFlightCandidate) {
      throw new Error('Cannot restore NRC MLP state while a training candidate is in flight.');
    }
    assertMlpStateArray(state.weights, this.plan.totalW, 'weights');
    assertMlpStateArray(state.biases, this.plan.totalB, 'biases');
    assertMlpStateArray(
      state.firstMomentWeights,
      this.plan.totalW,
      'firstMomentWeights',
    );
    assertMlpStateArray(
      state.secondMomentWeights,
      this.plan.totalW,
      'secondMomentWeights',
      true,
    );
    assertMlpStateArray(
      state.firstMomentBiases,
      this.plan.totalB,
      'firstMomentBiases',
    );
    assertMlpStateArray(
      state.secondMomentBiases,
      this.plan.totalB,
      'secondMomentBiases',
      true,
    );
    assertMlpAdamT(state.adamT);
    if (this.cfg.useF16) {
      assertF16Representable(state.weights, 'weights');
      assertF16Representable(state.biases, 'biases');
    }

    const old = this._liveTrainableSet();
    const candidate = this._spareTrainableSet ?? this._allocateTrainableCandidate();
    this._spareTrainableSet = undefined;
    this._inFlightCandidate = candidate;
    const uploads: GPUBuffer[] = [];
    const stage = (data: ArrayBufferView): GPUBuffer => {
      const buffer = this.device.createBuffer({
        label: 'nrc-mlp-state-import-staging',
        size: Math.max(4, (data.byteLength + 3) & ~3),
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      uploads.push(buffer);
      new Uint8Array(buffer.getMappedRange()).set(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      );
      buffer.unmap();
      return buffer;
    };
    try {
      const weights = stage(state.weights);
      const biases = stage(state.biases);
      const firstMomentWeights = stage(state.firstMomentWeights);
      const secondMomentWeights = stage(state.secondMomentWeights);
      const firstMomentBiases = stage(state.firstMomentBiases);
      const secondMomentBiases = stage(state.secondMomentBiases);
      const forwardWeightsData = this.cfg.useF16
        ? padEvenU16(f32ToF16Bits(state.weights))
        : state.weights;
      const forwardBiasesData = this.cfg.useF16
        ? padEvenU16(f32ToF16Bits(state.biases))
        : state.biases;
      const forwardWeights = stage(forwardWeightsData);
      const forwardBiases = stage(forwardBiasesData);
      encoder.copyBufferToBuffer(
        weights, 0, candidate.wMasterGpu, 0, state.weights.byteLength,
      );
      encoder.copyBufferToBuffer(
        biases, 0, candidate.bMasterGpu, 0, state.biases.byteLength,
      );
      encoder.copyBufferToBuffer(
        firstMomentWeights, 0, candidate.mW, 0,
        state.firstMomentWeights.byteLength,
      );
      encoder.copyBufferToBuffer(
        secondMomentWeights, 0, candidate.vW, 0,
        state.secondMomentWeights.byteLength,
      );
      encoder.copyBufferToBuffer(
        firstMomentBiases, 0, candidate.mB, 0,
        state.firstMomentBiases.byteLength,
      );
      encoder.copyBufferToBuffer(
        secondMomentBiases, 0, candidate.vB, 0,
        state.secondMomentBiases.byteLength,
      );
      encoder.copyBufferToBuffer(
        forwardWeights, 0, candidate.weights, 0, forwardWeightsData.byteLength,
      );
      encoder.copyBufferToBuffer(
        forwardBiases, 0, candidate.biases, 0, forwardBiasesData.byteLength,
      );
    } catch (error) {
      for (const upload of uploads) {
        try { upload.destroy(); } catch { /* continue candidate cleanup */ }
      }
      this._spareTrainableSet = candidate;
      this._inFlightCandidate = undefined;
      throw error;
    }

    const oldW = this.wMaster;
    const oldB = this.bMaster;
    const oldAdamT = this.adamT;
    let committed = false;
    let closed = false;
    const destroyUploads = (): void => {
      for (const upload of uploads) {
        try { upload.destroy(); } catch { /* continue staging retirement */ }
      }
    };
    return {
      candidateWeightBuffer: candidate.wMasterGpu,
      candidateBiasBuffer: candidate.bMasterGpu,
      commitCpu: () => {
        if (closed || committed) return;
        this._publishTrainableSet(candidate);
        this.wMaster = state.weights.slice();
        this.bMaster = state.biases.slice();
        this.adamT = state.adamT;
        this._inFlightCandidate = undefined;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) {
          this._publishTrainableSet(old);
          this.wMaster = oldW;
          this.bMaster = oldB;
          this.adamT = oldAdamT;
        }
        this._spareTrainableSet = candidate;
        if (this._inFlightCandidate === candidate) this._inFlightCandidate = undefined;
        closed = true;
        destroyUploads();
      },
      finalizeSuccess: () => {
        if (closed) return;
        if (!committed) throw new Error('Cannot finalize an unpublished NRC MLP restore.');
        this._spareTrainableSet = old;
        closed = true;
        const retire = (): void => destroyUploads();
        try {
          void this.device.queue.onSubmittedWorkDone().then(retire, retire);
        } catch {
          retire();
        }
      },
    };
  }

  private pushWeightsToGpu() {
    const q = this.device.queue;
    q.writeBuffer(this.wMasterGpu!, 0, this.wMaster as unknown as BufferSource);
    q.writeBuffer(this.bMasterGpu!, 0, this.bMaster as unknown as BufferSource);
    if (this.cfg.useF16) {
      q.writeBuffer(this.weights!, 0, padEvenU16(f32ToF16Bits(this.wMaster)) as unknown as BufferSource);
      q.writeBuffer(this.biases!, 0, padEvenU16(f32ToF16Bits(this.bMaster)) as unknown as BufferSource);
    } else {
      q.writeBuffer(this.weights!, 0, this.wMaster as unknown as BufferSource);
      q.writeBuffer(this.biases!, 0, this.bMaster as unknown as BufferSource);
    }
  }

  /** Upload one full-capacity batch; active rows are selected by recordTrainStep. */
  setBatch(x: Float32Array, y: Float32Array) {
    this.#assertUsable('setBatch');
    const expectedX = this.numSamples * this.spec.inW;
    const expectedY = this.numSamples * this.spec.outW;
    if (x.length !== expectedX || y.length !== expectedY) {
      throw new RangeError(`FusedMlpTrainer.setBatch expected ${expectedX} inputs and ${expectedY} targets; got ${x.length} and ${y.length}`);
    }
    for (const value of x) if (!Number.isFinite(value)) throw new RangeError('FusedMlpTrainer inputs must all be finite');
    for (const value of y) if (!Number.isFinite(value)) throw new RangeError('FusedMlpTrainer targets must all be finite');
    const q = this.device.queue;
    if (this.cfg.useF16) {
      q.writeBuffer(this.inputs!, 0, padEvenU16(f32ToF16Bits(x)) as unknown as BufferSource);
    } else {
      q.writeBuffer(this.inputs!, 0, x as unknown as BufferSource);
    }
    q.writeBuffer(this.targets!, 0, y as unknown as BufferSource);
  }

  private activeWindow(activeSamples?: number): { samples: number; tiles: number } {
    return resolveActiveSampleWindow(this.numSamples, this.cfg.tileB, activeSamples);
  }

  private writeParamsHeader(activeSamples?: number): { samples: number; tiles: number } {
    const win = this.activeWindow(activeSamples);
    const u = new Uint32Array(4);
    u[0] = win.samples;
    u[1] = this.spec.inW;
    u[2] = 0;
    u[3] = win.tiles;
    this.device.queue.writeBuffer(this._paramsUbo!, 0, u);
    return win;
  }

  // Returns the persistent params uniform buffer (allocated once in build).
  // Layer-plan offsets are constant after build; trainStep rewrites the first
  // 16-byte header so sparse NRC record batches train only on filled samples.
  private paramsUniform(): GPUBuffer { return this._paramsUbo!; }

  /** Record the fused forward pass (one workgroup per tile).
   *  @internal — used by {@link FusedMlpTrainerProbe} (the FD/loss debug surface). */
  recordForward(enc: GPUCommandEncoder, activeSamples?: number) {
    this.#assertUsable('recordForward');
    const win = this.writeParamsHeader(activeSamples);
    const bg = this._forwardBindGroups.get(this.weights!);
    if (!bg) throw new Error('NRC MLP forward binding generation is unavailable');
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pFwd); pass.setBindGroup(0, bg);
    if (win.tiles > 0) pass.dispatchWorkgroups(win.tiles);
    pass.end();
  }

  /** Record the fused backward pass (accumulates fixed-point grads).
   *  @internal — used by {@link FusedMlpTrainerProbe}. */
  recordBackward(enc: GPUCommandEncoder, activeSamples?: number) {
    this.#assertUsable('recordBackward');
    const win = this.writeParamsHeader(activeSamples);
    const bg = this._backwardBindGroups.get(this.weights!);
    if (!bg) throw new Error('NRC MLP backward binding generation is unavailable');
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pBwd); pass.setBindGroup(0, bg);
    if (win.tiles > 0) pass.dispatchWorkgroups(win.tiles);
    pass.end();
  }

  /** @internal — used by {@link FusedMlpTrainerProbe} + trainStep. */
  recordGradFinalize(enc: GPUCommandEncoder, _fx: GPUBuffer, _f: GPUBuffer, count: number, ubo: GPUBuffer) {
    this.#assertUsable('recordGradFinalize');
    const u = new Uint32Array(4);
    u[0] = count >>> 0;
    this.device.queue.writeBuffer(ubo, 0, u);
    const bg = this._gradFinalizeBindGroups.get(ubo);
    if (!bg) throw new Error('NRC MLP grad-finalize binding is unavailable');
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pGradFin); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
  }

  /** Encode into isolated candidate parameters/moments; publication is explicit. */
  recordTrainStep(enc: GPUCommandEncoder, lr: number, activeSamples?: number): FusedMlpTrainTransaction | null {
    this.#assertUsable('recordTrainStep');
    if (this._inFlightCandidate) throw new Error('FusedMlpTrainer already has an in-flight candidate');
    if (!Number.isFinite(lr) || lr <= 0) {
      throw new RangeError(`FusedMlpTrainer learning rate must be finite and positive; got ${lr}`);
    }
    const win = this.activeWindow(activeSamples);
    if (win.samples === 0) return null;
    const old = this._liveTrainableSet();
    const candidate = this._spareTrainableSet ?? this._allocateTrainableCandidate();
    this._spareTrainableSet = undefined;
    this._inFlightCandidate = candidate;
    const previousAdamT = this.adamT;
    const nextAdamT = previousAdamT + 1;
    try {
      enc.copyBufferToBuffer(old.wMasterGpu, 0, candidate.wMasterGpu, 0, this.plan.totalW * 4);
      enc.copyBufferToBuffer(old.bMasterGpu, 0, candidate.bMasterGpu, 0, this.plan.totalB * 4);
      enc.copyBufferToBuffer(old.mW, 0, candidate.mW, 0, this.plan.totalW * 4);
      enc.copyBufferToBuffer(old.vW, 0, candidate.vW, 0, this.plan.totalW * 4);
      enc.copyBufferToBuffer(old.mB, 0, candidate.mB, 0, this.plan.totalB * 4);
      enc.copyBufferToBuffer(old.vB, 0, candidate.vB, 0, this.plan.totalB * 4);
      enc.clearBuffer(this.gradWfx!);
      enc.clearBuffer(this.gradBfx!);
      enc.clearBuffer(this.gradInputFx!);
      this.recordForward(enc, win.samples);
      this.recordBackward(enc, win.samples);
      this.recordGradFinalize(enc, this.gradWfx!, this.gradWf!, this.plan.totalW, this._gradFinUboW!);
      this.recordGradFinalize(enc, this.gradBfx!, this.gradBf!, this.plan.totalB, this._gradFinUboB!);
      this.recordGradFinalize(enc, this.gradInputFx!, this.gradInputF!, win.samples * this.spec.inW, this._gradFinUboX!);
      const bc1 = 1 - Math.pow(0.9, nextAdamT);
      const bc2 = 1 - Math.pow(0.999, nextAdamT);
      this.recordAdam(enc, candidate.wMasterGpu, this.gradWf!, candidate.mW, candidate.vW, this.plan.totalW, lr, bc1, bc2, this._adamUboW!);
      this.recordAdam(enc, candidate.bMasterGpu, this.gradBf!, candidate.mB, candidate.vB, this.plan.totalB, lr, bc1, bc2, this._adamUboB!);
      if (this.cfg.useF16) {
        this.recordDowncast(enc, candidate.wMasterGpu, candidate.weights, this.plan.totalW, this._downcastUboW!);
        this.recordDowncast(enc, candidate.bMasterGpu, candidate.biases, this.plan.totalB, this._downcastUboB!);
      } else {
        enc.copyBufferToBuffer(candidate.wMasterGpu, 0, candidate.weights, 0, this.plan.totalW * 4);
        enc.copyBufferToBuffer(candidate.bMasterGpu, 0, candidate.biases, 0, this.plan.totalB * 4);
      }
    } catch (error) {
      this._spareTrainableSet = candidate;
      this._inFlightCandidate = undefined;
      throw error;
    }

    let committed = false;
    let closed = false;
    return {
      candidateWeightBuffer: candidate.wMasterGpu,
      candidateBiasBuffer: candidate.bMasterGpu,
      commitCpu: () => {
        if (closed || committed) return;
        this._publishTrainableSet(candidate);
        this.adamT = nextAdamT;
        this._inFlightCandidate = undefined;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) {
          this._publishTrainableSet(old);
          this.adamT = previousAdamT;
        }
        this._spareTrainableSet = candidate;
        if (this._inFlightCandidate === candidate) this._inFlightCandidate = undefined;
        closed = true;
      },
      finalizeSuccess: () => {
        if (closed) return;
        if (!committed) throw new Error('Cannot finalize an unpublished NRC MLP candidate');
        this._spareTrainableSet = old;
        closed = true;
      },
    };
  }

  /** Compatibility wrapper: one command buffer and one queue submission. */
  trainStep(lr: number, activeSamples?: number): void {
    this.#assertUsable('trainStep');
    const enc = this.device.createCommandEncoder({ label: 'nrc-mlp-train-step' });
    const transaction = this.recordTrainStep(enc, lr, activeSamples);
    if (!transaction) return;
    try {
      this.device.queue.submit([enc.finish()]);
      transaction.commitCpu();
      transaction.finalizeSuccess();
    } catch (error) {
      transaction.rollback();
      throw error;
    }
  }

  // Uses a pre-allocated persistent UBO (count is constant after build — written once).
  private recordDowncast(
    enc: GPUCommandEncoder,
    src: GPUBuffer,
    _dst: GPUBuffer,
    count: number,
    _ubo: GPUBuffer,
  ) {
    const bg = this._downcastBindGroups.get(src);
    if (!bg) throw new Error('NRC MLP downcast binding generation is unavailable');
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pDowncast!); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
  }

  // Uses a pre-allocated persistent UBO. count is constant per slot; bc1/bc2/lr
  // are rewritten into the same buffer each step (no buffer allocation per step).
  private recordAdam(
    enc: GPUCommandEncoder,
    params: GPUBuffer,
    _grads: GPUBuffer,
    _m: GPUBuffer,
    _v: GPUBuffer,
    count: number,
    lr: number,
    bc1: number,
    bc2: number,
    ubo: GPUBuffer,
  ) {
    this.device.queue.writeBuffer(ubo, 0, packAdamUbo(count, lr, bc1, bc2));
    const bg = this._adamBindGroups.get(params);
    if (!bg) throw new Error('NRC MLP Adam binding generation is unavailable');
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pAdam); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
  }

  // The FD/loss/readback debug helpers (computeGradsStep / computeLoss /
  // readGrads / readInputGrads / readScalar / readF32) moved to
  // {@link FusedMlpTrainerProbe} (Task 4.5 Theme I) so the production trainer
  // surface is just build / setWeights / setBatch / trainStep / dispose.

  get layerPlan(): LayerPlan { return this.plan; }
}

// f16 bits -> f32 decode (for readback).
function f16BitsToF32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let k = 0; k < bits.length; k++) {
    const h = bits[k]!;
    const neg = (h & 0x8000) !== 0;
    const exp = (h >>> 10) & 0x1f;
    const mant = h & 0x3ff;
    let f: number;
    if (exp === 0) {
      f = (mant / 1024) * Math.pow(2, -14);   // subnormal / zero
    } else if (exp === 0x1f) {
      f = mant ? NaN : Infinity;
    } else {
      f = (1 + mant / 1024) * Math.pow(2, exp - 15);
    }
    out[k] = neg ? -f : f;
  }
  return out;
}

/** He-init the master f32 weights/biases for a resolved layer plan.
 *  Seed is fixed (12345) for reproducibility; small positive biases (0.1) keep
 *  ReLU units active so the FD gradient check gets clean signal. */
function heInit(trainer: FusedMlpTrainer): { w: Float32Array; b: Float32Array } {
  const plan = trainer.layerPlan;
  const w = new Float32Array(plan.totalW);
  const b = new Float32Array(plan.totalB);
  let s = 12345 >>> 0;
  const rng = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  for (let l = 0; l < plan.wlayers; l++) {
    const inW = plan.inW[l]!, outW = plan.outW[l]!;
    const scale = Math.sqrt(2 / inW);
    for (let k = 0; k < inW * outW; k++) w[plan.wOff[l]! + k] = (rng() * 2 - 1) * scale;
  }
  for (let i = 0; i < b.length; i++) b[i] = 0.1;
  return { w, b };
}

// The Adam optimizer WGSL is exported so the NRC subsystem can run a SEPARATE
// Adam on the hash-grid feature tables with its own (higher) learning rate +
// moment buffers (Instant-NGP §4: lr_embed ≈ 0.1 vs lr_mlp ≈ 0.01).
export { planLayers, resolveActiveSampleWindow, f32ToF16Bits, f16BitsToF32, heInit, ADAM_WGSL };
export type { LayerPlan };
