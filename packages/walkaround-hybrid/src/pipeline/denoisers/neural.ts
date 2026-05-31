/**
 * NeuralDenoiser registry entry.
 *
 * Enabled iff the pipeline supplied a pre-initialized InferenceGraph.
 *
 * Dispatch path:
 *   1) pack current frame textures (noisy/albedo/normalDepth) into three
 *      f32 storage buffers
 *   2) run InferenceGraph on those buffers
 *   3) unpack denoised RGB buffer back into an rgba16float output texture
 *
 * The InferenceGraph itself remains owned by pipeline lifecycle code.
 */

import {
  DENOISER_PASS_LABELS,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
} from './index.js';
import type { InferenceGraph } from '../../neural/InferenceGraph.js';

export class NeuralDenoiser implements Denoiser {
  readonly id = 'neural' as const;
  readonly disabled: boolean;
  /** The 2 passes this denoiser dispatches: `neural-pack` (input-pack) +
   *  `neural-unpack` (output-unpack). The slot allocator inspects this even
   *  for `disabled` entries when buildPassLayout sizes the querySet. */
  readonly passLabels = DENOISER_PASS_LABELS['neural'];
  private readonly _inferenceGraph: InferenceGraph | undefined;
  private _width = 0;
  private _height = 0;
  private _loggedSizeMismatch = false;

  private _packPipeline: GPUComputePipeline | null = null;
  private _unpackPipeline: GPUComputePipeline | null = null;
  private _packParamsBuf: GPUBuffer | null = null;
  private _unpackParamsBuf: GPUBuffer | null = null;
  private _noisyBuf: GPUBuffer | null = null;
  private _albedoBuf: GPUBuffer | null = null;
  private _normalsBuf: GPUBuffer | null = null;
  private _outputBuf: GPUBuffer | null = null;
  private _outputTex: GPUTexture | null = null;
  private _outputTexW = 0;
  private _outputTexH = 0;

  constructor(options?: { inferenceGraph?: InferenceGraph }) {
    this._inferenceGraph = options?.inferenceGraph;
    this.disabled = this._inferenceGraph === undefined;
  }

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    if (this._inferenceGraph == null) return;
    this._width = ctx.width;
    this._height = ctx.height;
    const device = ctx.device;

    const packSM = device.createShaderModule({
      label: 'neural-denoiser-pack',
      code: /* wgsl */`
struct PackParams {
  width: u32,
  height: u32,
  pixelCount: u32,
  _pad0: u32,
}
@group(0) @binding(0) var noisyTex: texture_2d<f32>;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var normalDepthTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> noisyOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> albedoOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> normalsOut: array<f32>;
@group(0) @binding(6) var<uniform> params: PackParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }
  let x = p % params.width;
  let y = p / params.width;
  let n = textureLoad(noisyTex, vec2i(i32(x), i32(y)), 0).rgb;
  let a = textureLoad(albedoTex, vec2i(i32(x), i32(y)), 0).rgb;
  let nd = textureLoad(normalDepthTex, vec2i(i32(x), i32(y)), 0).xyz;
  let nrm = normalize(nd * 2.0 - 1.0);
  let base = p * 3u;
  noisyOut[base + 0u] = n.r;
  noisyOut[base + 1u] = n.g;
  noisyOut[base + 2u] = n.b;
  albedoOut[base + 0u] = a.r;
  albedoOut[base + 1u] = a.g;
  albedoOut[base + 2u] = a.b;
  normalsOut[base + 0u] = nrm.r;
  normalsOut[base + 1u] = nrm.g;
  normalsOut[base + 2u] = nrm.b;
}
`,
    });
    const unpackSM = device.createShaderModule({
      label: 'neural-denoiser-unpack',
      code: /* wgsl */`
struct UnpackParams {
  width: u32,
  height: u32,
  pixelCount: u32,
  _pad0: u32,
}
@group(0) @binding(0) var<storage, read> denoisedIn: array<f32>;
@group(0) @binding(1) var denoisedOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: UnpackParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }
  let x = p % params.width;
  let y = p / params.width;
  let base = p * 3u;
  let c = vec3f(
    max(0.0, denoisedIn[base + 0u]),
    max(0.0, denoisedIn[base + 1u]),
    max(0.0, denoisedIn[base + 2u]),
  );
  textureStore(denoisedOut, vec2u(x, y), vec4f(c, 1.0));
}
`,
    });
    this._packPipeline = await device.createComputePipelineAsync({
      label: 'neural-denoiser-pack-pipeline',
      layout: 'auto',
      compute: { module: packSM, entryPoint: 'main' },
    });
    this._unpackPipeline = await device.createComputePipelineAsync({
      label: 'neural-denoiser-unpack-pipeline',
      layout: 'auto',
      compute: { module: unpackSM, entryPoint: 'main' },
    });

    this._packParamsBuf = device.createBuffer({
      label: 'neural-denoiser-pack-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._unpackParamsBuf = device.createBuffer({
      label: 'neural-denoiser-unpack-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._reallocForSize(device, ctx.width, ctx.height);
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null {
    if (this._inferenceGraph == null) return null;
    if (
      this._packPipeline == null ||
      this._unpackPipeline == null ||
      this._packParamsBuf == null ||
      this._unpackParamsBuf == null
    ) {
      return ctx.resources.common.hdrColorTexture;
    }
    // InferenceGraph tensor buffers are fixed at initialization dimensions.
    if (ctx.width !== this._width || ctx.height !== this._height) {
      if (!this._loggedSizeMismatch) {
        this._loggedSizeMismatch = true;
        console.warn(
          `[NeuralDenoiser] size changed from ${this._width}x${this._height} ` +
          `to ${ctx.width}x${ctx.height}; falling back to hdrColorTexture. ` +
          `Recreate engine to reinitialize neural tensor buffers.`,
        );
      }
      return ctx.resources.common.hdrColorTexture;
    }
    const device = ctx.device;
    this._reallocForSize(device, ctx.width, ctx.height);
    if (
      this._noisyBuf == null ||
      this._albedoBuf == null ||
      this._normalsBuf == null ||
      this._outputBuf == null ||
      this._outputTex == null
    ) {
      return ctx.resources.common.hdrColorTexture;
    }
    const pixelCount = ctx.width * ctx.height;
    const params = new Uint32Array([ctx.width, ctx.height, pixelCount, 0]);
    device.queue.writeBuffer(this._packParamsBuf, 0, params);
    device.queue.writeBuffer(this._unpackParamsBuf, 0, params);

    const packBG = device.createBindGroup({
      label: 'neural-denoiser-pack-bg',
      layout: this._packPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: ctx.resources.common.hdrColorTexture.createView() },
        { binding: 1, resource: ctx.resources.common.albedoTexture.createView() },
        { binding: 2, resource: ctx.resources.common.gNormalDepthTexture.createView() },
        { binding: 3, resource: { buffer: this._noisyBuf } },
        { binding: 4, resource: { buffer: this._albedoBuf } },
        { binding: 5, resource: { buffer: this._normalsBuf } },
        { binding: 6, resource: { buffer: this._packParamsBuf } },
      ],
    });
    {
      const pass = ctx.encoder.beginComputePass(ctx.computeDesc('neural-pack'));
      pass.setPipeline(this._packPipeline);
      pass.setBindGroup(0, packBG);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / 256), 1, 1);
      pass.end();
    }

    this._inferenceGraph.run(
      this._noisyBuf,
      this._albedoBuf,
      this._normalsBuf,
      this._outputBuf,
      ctx.encoder,
    );

    const unpackBG = device.createBindGroup({
      label: 'neural-denoiser-unpack-bg',
      layout: this._unpackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._outputBuf } },
        { binding: 1, resource: this._outputTex.createView() },
        { binding: 2, resource: { buffer: this._unpackParamsBuf } },
      ],
    });
    {
      const pass = ctx.encoder.beginComputePass(ctx.computeDesc('neural-unpack'));
      pass.setPipeline(this._unpackPipeline);
      pass.setBindGroup(0, unpackBG);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / 256), 1, 1);
      pass.end();
    }
    return this._outputTex;
  }

  resize(w: number, h: number): void {
    // Reallocate bridge buffers/output texture. InferenceGraph tensors are fixed
    // to init dimensions; dispatch guards and falls back on mismatch.
    this._outputTex?.destroy();
    this._outputTex = null;
    this._outputTexW = 0;
    this._outputTexH = 0;
    this._noisyBuf?.destroy();
    this._albedoBuf?.destroy();
    this._normalsBuf?.destroy();
    this._outputBuf?.destroy();
    this._noisyBuf = null;
    this._albedoBuf = null;
    this._normalsBuf = null;
    this._outputBuf = null;
    this._width = w;
    this._height = h;
    this._loggedSizeMismatch = false;
  }

  dispose(): void {
    this._outputTex?.destroy();
    this._outputTex = null;
    this._noisyBuf?.destroy();
    this._albedoBuf?.destroy();
    this._normalsBuf?.destroy();
    this._outputBuf?.destroy();
    this._packParamsBuf?.destroy();
    this._unpackParamsBuf?.destroy();
    this._noisyBuf = null;
    this._albedoBuf = null;
    this._normalsBuf = null;
    this._outputBuf = null;
    this._packParamsBuf = null;
    this._unpackParamsBuf = null;
    this._packPipeline = null;
    this._unpackPipeline = null;
  }

  private _reallocForSize(device: GPUDevice, w: number, h: number): void {
    if (this._noisyBuf != null && this._outputTex != null && this._outputTexW === w && this._outputTexH === h) {
      return;
    }
    this._outputTex?.destroy();
    this._outputTex = null;
    this._noisyBuf?.destroy();
    this._albedoBuf?.destroy();
    this._normalsBuf?.destroy();
    this._outputBuf?.destroy();
    this._noisyBuf = null;
    this._albedoBuf = null;
    this._normalsBuf = null;
    this._outputBuf = null;
    const pixelCount = w * h;
    const bytes = Math.max(4, pixelCount * 3 * 4);
    const mkStorage = (label: string) =>
      device.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    this._noisyBuf = mkStorage('neural-denoiser-noisy');
    this._albedoBuf = mkStorage('neural-denoiser-albedo');
    this._normalsBuf = mkStorage('neural-denoiser-normals');
    this._outputBuf = mkStorage('neural-denoiser-output');
    this._outputTex = device.createTexture({
      label: 'neural-denoiser-output-texture',
      size: [w, h],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._outputTexW = w;
    this._outputTexH = h;
  }
}
