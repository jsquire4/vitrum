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
// records. The trainer owns the ~18 GPU buffers it allocates in {@link
// FusedMlpTrainer.build}; {@link FusedMlpTrainer.dispose} releases them
// (host-owns-lifecycle), and NrcSubsystem.dispose() forwards to it.

import {
  fusedForwardWgsl, fusedBackwardWgsl, gradFinalizeWgsl, downcastF16Wgsl,
  type FusedMlpWgslOptions,
} from "./wgsl/fusedMlp.wgsl.js";

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
@compute @workgroup_size(64, 1, 1)
fn adamMain(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.count) { return; }
  let g = grads[idx];
  let mi = p.beta1 * m[idx] + (1.0 - p.beta1) * g;
  let vi = p.beta2 * v[idx] + (1.0 - p.beta2) * g * g;
  m[idx] = mi; v[idx] = vi;
  let mhat = mi / p.bc1;
  let vhat = vi / p.bc2;
  params[idx] = params[idx] - p.lr * mhat / (sqrt(vhat) + p.eps);
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

// Pad a Uint16Array to an even length so its byte length is a multiple of 4
// (WebGPU writeBuffer/copy alignment for f16 storage buffers). The trailing
// padding element is never read by the kernel (buffers are sized to match).
function padEvenU16(a: Uint16Array): Uint16Array {
  if ((a.length & 1) === 0) return a;
  const out = new Uint16Array(a.length + 1);
  out.set(a);
  return out;
}

// Half-precision pack: round-to-nearest f32->f16 bit cast (subset; no subnormal
// flush concerns for our value range). Returns Uint16Array of f16 bits.
function f32ToF16Bits(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  const dv = new DataView(new ArrayBuffer(4));
  for (let k = 0; k < src.length; k++) {
    dv.setFloat32(0, src[k]!, true);
    const x = dv.getUint32(0, true);
    const sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 0xff) - 127 + 15;
    let mant = x & 0x7fffff;
    if (exp <= 0) { out[k] = sign; continue; }            // flush tiny to ±0
    if (exp >= 0x1f) { out[k] = sign | 0x7c00; continue; } // inf/overflow
    // round to nearest even
    const round = (mant & 0x1000) !== 0;
    mant = mant >>> 13;
    let h = sign | (exp << 10) | mant;
    if (round) h += 1;
    out[k] = h & 0xffff;
  }
  return out;
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

  // GPU buffers
  weights!: GPUBuffer; biases!: GPUBuffer;        // f16 or f32 (forward/backward)
  // f32 MASTER copy for Adam (mixed precision). On the f32 path this IS the same
  // role as weights/biases but kept separate so Adam math always runs in f32.
  wMasterGpu!: GPUBuffer; bMasterGpu!: GPUBuffer;
  inputs!: GPUBuffer; targets!: GPUBuffer;
  actsGlob!: GPUBuffer; zGlob!: GPUBuffer;
  gradWfx!: GPUBuffer; gradBfx!: GPUBuffer;        // i32 fixed-point
  gradWf!: GPUBuffer; gradBf!: GPUBuffer;          // f32 finalized
  // dL/dX — gradient w.r.t. the raw (padded) network INPUT, [numSamples × inW].
  // Fixed-point i32 atomic accumulator + finalized f32. The first L·F columns of
  // each sample's row are dL/dfeature for the hash-grid encode (Müller 2022 §4);
  // the NRC encode-backward scatters them into the trainable feature tables.
  gradInputFx!: GPUBuffer; gradInputF!: GPUBuffer;
  mW!: GPUBuffer; vW!: GPUBuffer; mB!: GPUBuffer; vB!: GPUBuffer; // Adam state (f32)

  pFwd!: GPUComputePipeline; pBwd!: GPUComputePipeline;
  pGradFin!: GPUComputePipeline; pAdam!: GPUComputePipeline;
  pDowncast?: GPUComputePipeline; // f16 path only

  numSamples = 0;
  adamT = 0;

  #disposed = false;

  constructor(device: GPUDevice, spec: FusedNetSpec, cfg: FusedTrainerConfig) {
    this.device = device; this.spec = spec; this.cfg = cfg;
    this.plan = planLayers(spec);
    this.node = spec.hidden + 2;
  }

  private wgslOpts(): FusedMlpWgslOptions {
    return {
      useF16: this.cfg.useF16, W: this.spec.W, OUT_W: this.spec.outW,
      HIDDEN: this.spec.hidden, TILE_B: this.cfg.tileB,
    };
  }

  /** Bytes of one resident scalar (f16=2, f32=4). */
  private scBytes(): number { return this.cfg.useF16 ? 2 : 4; }

  async build(numSamples: number) {
    const d = this.device;
    this.numSamples = numSamples;
    const opts = this.wgslOpts();
    const sc = this.scBytes();

    // shared-mem budget sanity (3 tiles of TILE_B×W): assert it fits.
    const sharedBytes = 3 * this.cfg.tileB * this.spec.W * sc;
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
  }

  /**
   * Release every GPU buffer this trainer allocated in {@link build}
   * (host-owns-lifecycle). Idempotent: a second call is a safe no-op, and it is
   * also safe to call before {@link build} has run (the buffer fields are simply
   * undefined and skipped). The owning {@link NrcSubsystem.dispose} forwards here
   * so the trainer's ~18 buffers don't leak until device teardown.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    // Every GPUBuffer allocated in build(). Guarded for the never-built case.
    const buffers: (GPUBuffer | undefined)[] = [
      this.weights, this.biases,
      this.wMasterGpu, this.bMasterGpu,
      this.inputs, this.targets,
      this.actsGlob, this.zGlob,
      this.gradWfx, this.gradBfx,
      this.gradWf, this.gradBf,
      this.gradInputFx, this.gradInputF,
      this.mW, this.vW, this.mB, this.vB,
    ];
    for (const buf of buffers) buf?.destroy();
    // Null the references so a stray post-dispose use fails loudly rather than
    // touching a destroyed buffer, and so the GC can reclaim the wrappers.
    this.weights = this.biases = undefined as unknown as GPUBuffer;
    this.wMasterGpu = this.bMasterGpu = undefined as unknown as GPUBuffer;
    this.inputs = this.targets = undefined as unknown as GPUBuffer;
    this.actsGlob = this.zGlob = undefined as unknown as GPUBuffer;
    this.gradWfx = this.gradBfx = undefined as unknown as GPUBuffer;
    this.gradWf = this.gradBf = undefined as unknown as GPUBuffer;
    this.gradInputFx = this.gradInputF = undefined as unknown as GPUBuffer;
    this.mW = this.vW = this.mB = this.vB = undefined as unknown as GPUBuffer;
  }

  /** Upload master weights/biases (f32) and downcast to the GPU forward buffers. */
  setWeights(w: Float32Array, b: Float32Array) {
    this.wMaster = w.slice();
    this.bMaster = b.slice();
    this.pushWeightsToGpu();
  }

  private pushWeightsToGpu() {
    const q = this.device.queue;
    // f32 master (Adam operand) always gets the full-precision values.
    q.writeBuffer(this.wMasterGpu, 0, this.wMaster as unknown as BufferSource);
    q.writeBuffer(this.bMasterGpu, 0, this.bMaster as unknown as BufferSource);
    // forward/backward operand buffers: downcast to f16 or copy f32.
    if (this.cfg.useF16) {
      q.writeBuffer(this.weights, 0, padEvenU16(f32ToF16Bits(this.wMaster)) as unknown as BufferSource);
      q.writeBuffer(this.biases, 0, padEvenU16(f32ToF16Bits(this.bMaster)) as unknown as BufferSource);
    } else {
      q.writeBuffer(this.weights, 0, this.wMaster as unknown as BufferSource);
      q.writeBuffer(this.biases, 0, this.bMaster as unknown as BufferSource);
    }
  }

  /** Upload a batch of inputs (raw inW) + f32 targets (outW). */
  setBatch(x: Float32Array, y: Float32Array) {
    const q = this.device.queue;
    if (this.cfg.useF16) {
      q.writeBuffer(this.inputs, 0, padEvenU16(f32ToF16Bits(x)) as unknown as BufferSource);
    } else {
      q.writeBuffer(this.inputs, 0, x as unknown as BufferSource);
    }
    q.writeBuffer(this.targets, 0, y as unknown as BufferSource);
  }

  private numTiles(): number { return Math.ceil(this.numSamples / this.cfg.tileB); }

  // Build the params uniform: 4-u32 header + array<vec4<u32>, WLAYERS> of
  // (wOff, bOff, inW, outW). 16-byte aligned throughout.
  private paramsUniform(): GPUBuffer {
    const d = this.device;
    const wl = this.plan.wlayers;
    const u = new Uint32Array(4 + wl * 4);
    u[0] = this.numSamples; u[1] = this.spec.inW; u[2] = 0; u[3] = this.numTiles();
    for (let l = 0; l < wl; l++) {
      const base = 4 + l * 4;
      u[base + 0] = this.plan.wOff[l]! >>> 0;
      u[base + 1] = this.plan.bOff[l]! >>> 0;
      u[base + 2] = this.plan.inW[l]! >>> 0;
      u[base + 3] = this.plan.outW[l]! >>> 0;
    }
    const b = d.createBuffer({ size: u.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(b, 0, u);
    return b;
  }

  /** Record the fused forward pass (one workgroup per tile).
   *  @internal — used by {@link FusedMlpTrainerProbe} (the FD/loss debug surface). */
  recordForward(enc: GPUCommandEncoder) {
    const d = this.device;
    const ub = this.paramsUniform();
    const bg = d.createBindGroup({
      layout: this.pFwd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weights } },
        { binding: 1, resource: { buffer: this.biases } },
        { binding: 2, resource: { buffer: this.inputs } },
        { binding: 3, resource: { buffer: this.actsGlob } },
        { binding: 4, resource: { buffer: this.zGlob } },
        { binding: 5, resource: { buffer: ub } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pFwd); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(this.numTiles());
    pass.end();
  }

  /** Record the fused backward pass (accumulates fixed-point grads).
   *  @internal — used by {@link FusedMlpTrainerProbe}. */
  recordBackward(enc: GPUCommandEncoder) {
    const d = this.device;
    const ub = this.paramsUniform();
    const bg = d.createBindGroup({
      layout: this.pBwd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weights } },
        { binding: 1, resource: { buffer: this.targets } },
        { binding: 2, resource: { buffer: this.actsGlob } },
        { binding: 3, resource: { buffer: this.zGlob } },
        { binding: 4, resource: { buffer: this.gradWfx } },
        { binding: 5, resource: { buffer: this.gradBfx } },
        { binding: 6, resource: { buffer: ub } },
        { binding: 7, resource: { buffer: this.gradInputFx } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pBwd); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(this.numTiles());
    pass.end();
  }

  /** @internal — used by {@link FusedMlpTrainerProbe} + trainStep. */
  recordGradFinalize(enc: GPUCommandEncoder, fx: GPUBuffer, f: GPUBuffer, count: number) {
    const d = this.device;
    const u = new Uint32Array(4); u[0] = count;
    const ub = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ub, 0, u);
    const bg = d.createBindGroup({
      layout: this.pGradFin.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: fx } },
        { binding: 1, resource: { buffer: f } },
        { binding: 2, resource: { buffer: ub } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pGradFin); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
  }

  /** Full fused train step: forward, backward, finalize grads, Adam, push back.
   *  Also finalizes dL/dX into {@link gradInputF} so the NRC encode-backward can
   *  scatter it into the trainable hash-grid tables. The MLP-weight Adam runs
   *  here; the host runs the TABLE Adam separately after the encode-backward. */
  trainStep(lr: number) {
    const d = this.device;
    const enc0 = d.createCommandEncoder();
    enc0.clearBuffer(this.gradWfx); enc0.clearBuffer(this.gradBfx);
    enc0.clearBuffer(this.gradInputFx);
    d.queue.submit([enc0.finish()]);

    const enc = d.createCommandEncoder();
    this.recordForward(enc);
    this.recordBackward(enc);
    this.recordGradFinalize(enc, this.gradWfx, this.gradWf, this.plan.totalW);
    this.recordGradFinalize(enc, this.gradBfx, this.gradBf, this.plan.totalB);
    this.recordGradFinalize(enc, this.gradInputFx, this.gradInputF, this.numSamples * this.spec.inW);
    d.queue.submit([enc.finish()]);

    // Adam ALWAYS runs on the f32 MASTER weight/bias buffers (mixed precision);
    // on the f16 path a downcast kernel then refreshes the f16 forward/backward
    // operand buffers. This keeps optimizer state full-precision (tiny-cuda-nn
    // recipe) and is fully on-GPU — no CPU round-trip in the step.
    this.adamT++;
    const bc1 = 1 - Math.pow(0.9, this.adamT);
    const bc2 = 1 - Math.pow(0.999, this.adamT);

    const enc2 = d.createCommandEncoder();
    this.recordAdam(enc2, this.wMasterGpu, this.gradWf, this.mW, this.vW, this.plan.totalW, lr, bc1, bc2);
    this.recordAdam(enc2, this.bMasterGpu, this.gradBf, this.mB, this.vB, this.plan.totalB, lr, bc1, bc2);
    if (this.cfg.useF16) {
      this.recordDowncast(enc2, this.wMasterGpu, this.weights, this.plan.totalW);
      this.recordDowncast(enc2, this.bMasterGpu, this.biases, this.plan.totalB);
    } else {
      // f32 path: master IS the operand; copy master back into the forward buffer.
      enc2.copyBufferToBuffer(this.wMasterGpu, 0, this.weights, 0, this.plan.totalW * 4);
      enc2.copyBufferToBuffer(this.bMasterGpu, 0, this.biases, 0, this.plan.totalB * 4);
    }
    d.queue.submit([enc2.finish()]);
  }

  private recordDowncast(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, count: number) {
    const d = this.device;
    const u = new Uint32Array(4); u[0] = count;
    const ub = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ub, 0, u);
    const bg = d.createBindGroup({
      layout: this.pDowncast!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: ub } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pDowncast!); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
  }

  private recordAdam(enc: GPUCommandEncoder, params: GPUBuffer, grads: GPUBuffer,
    m: GPUBuffer, v: GPUBuffer, count: number, lr: number, bc1: number, bc2: number) {
    const d = this.device;
    const ab = new ArrayBuffer(48);
    new Uint32Array(ab, 0, 1)[0] = count;
    const f = new Float32Array(ab);
    f[4] = lr; f[5] = 0.9; f[6] = 0.999; f[7] = 1e-8; f[8] = bc1; f[9] = bc2;
    const ub = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ub, 0, ab);
    const bg = d.createBindGroup({
      layout: this.pAdam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: grads } },
        { binding: 2, resource: { buffer: m } },
        { binding: 3, resource: { buffer: v } },
        { binding: 4, resource: { buffer: ub } },
      ],
    });
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
export { planLayers, f32ToF16Bits, f16BitsToF32, heInit, ADAM_WGSL };
export type { LayerPlan };
