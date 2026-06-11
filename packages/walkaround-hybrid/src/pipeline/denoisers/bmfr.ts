/**
 * BmfrDenoiser — BMFR (Koskela et al. 2019) blockwise multi-order feature
 * regression for the walkaround-hybrid realtime pipeline.
 *
 * Per frame, a single compute pass fits each 32×32 screen block's noisy 1-spp
 * color to a 10-feature matrix [1, p.xyz, n.xyz, p².xyz] via Householder QR on
 * the normal equations and reconstructs `color = T·α`, then temporally
 * accumulates the reconstruction against the previous frame's result (EMA).
 *
 * Inputs (from FrameResources.common):
 *   - hdrColorTexture     — noisy direct-channel HDR radiance (the fit target)
 *   - gNormalDepthTexture — .xyz = packed world normal, .w = signed linear depth
 *
 * The walkaround pipeline has no dedicated world-position G-buffer, so this
 * entry runs the kernel in `positionMode = 1` (screen-space position proxy):
 * the regression position is `(pixelX, pixelY, depth)`, reading depth from the
 * gNormalDepth `.w` channel. Per-screen-block regression is well-posed with the
 * screen-space proxy (positions are block-local-normalised before the squared
 * features), and it avoids allocating + filling a full world-position buffer
 * every frame.
 *
 * Temporal history is a private ping-pong pair owned by this denoiser
 * (rgba16float, full internal-res). It is reset (hasHistory = 0) on the first
 * frame and whenever the camera moves (`ctx.isMoving`), so a fresh disocclusion
 * does not blend in stale geometry — mirroring BMFR's per-frame accept/reject.
 *
 * Pipeline uses `layout: 'auto'`; the UBO is allocated once and re-packed only
 * when the moving / history state changes.
 *
 * Refs: Koskela, Immonen, Mäkitalo, Foi, Viitanen, Jääskeläinen, Kultala,
 * Takala. "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing
 * Reconstruction." ACM TOG 38(5), 2019.
 */

import {
  BMFR_DEFAULT_UNIFORMS,
  BMFR_UNIFORMS_SIZE_BYTES,
  packBmfrUniforms,
  type BmfrUniforms,
} from '@vitrum/shared-denoisers';
import { composeWgsl } from '../wgslComposer.js';
import { BMFR_MODULE, WGSL_MODULES } from '../wgslModules.js';
import { checkShaderCompile } from '../shaderUtils.js';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';

export class BmfrDenoiser implements Denoiser {
  readonly id = 'bmfr' as const;
  readonly passLabels = DENOISER_PASS_LABELS['bmfr'];

  private _device!: GPUDevice;
  private _pipeline!: GPUComputePipeline;
  private _ubo!: GPUBuffer;
  /** Last-packed (movingOrFirst) state so the UBO is only re-written on change. */
  private _lastHasHistory = -1;

  // Private persistent temporal-history ping-pong (full internal-res).
  private _historyA: GPUTexture | null = null;
  private _historyB: GPUTexture | null = null;
  private _pingPong = 0;
  /** True until the first dispatch / after a resize — forces hasHistory = 0. */
  private _historyValid = false;

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    const { device, width, height } = ctx;
    this._device = device;

    const code = composeWgsl(BMFR_MODULE, WGSL_MODULES);
    const sm = device.createShaderModule({ label: 'bmfr', code });
    await checkShaderCompile(sm, 'bmfr');
    this._pipeline = await device.createComputePipelineAsync({
      label: 'bmfr',
      layout: 'auto',
      compute: { module: sm, entryPoint: 'bmfrMain' },
    });

    this._ubo = device.createBuffer({
      label: 'bmfr-ubo',
      size: BMFR_UNIFORMS_SIZE_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._allocHistory(width, height);
  }

  state(): DenoiserState {
    return DENOISER_READY_STATE;
  }

  private _allocHistory(width: number, height: number): void {
    this._historyA?.destroy();
    this._historyB?.destroy();
    const usage =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST;
    this._historyA = this._device.createTexture({ label: 'bmfr-history-a', size: [width, height], format: 'rgba16float', usage });
    this._historyB = this._device.createTexture({ label: 'bmfr-history-b', size: [width, height], format: 'rgba16float', usage });
    this._pingPong = 0;
    this._historyValid = false;
  }

  private _packUniforms(hasHistory: number): void {
    if (this._lastHasHistory === hasHistory) return;
    const scratch = new ArrayBuffer(BMFR_UNIFORMS_SIZE_BYTES);
    const u: BmfrUniforms = {
      ...BMFR_DEFAULT_UNIFORMS,
      // Walkaround pipeline → screen-space position proxy from gNormalDepth.w.
      positionMode: 1,
      hasHistory,
    };
    packBmfrUniforms(u, scratch);
    this._device.queue.writeBuffer(this._ubo, 0, scratch);
    this._lastHasHistory = hasHistory;
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture {
    const { device, encoder, resources, computeDesc, isMoving } = ctx;
    const common = resources.common;
    const w = ctx.width;
    const h = ctx.height;

    // Camera motion / first frame → drop history so a disocclusion does not
    // blend in stale geometry (BMFR accept/reject analogue).
    const useHistory = this._historyValid && !isMoving;
    this._packUniforms(useHistory ? 1 : 0);

    const histRead = this._pingPong === 0 ? this._historyA! : this._historyB!;
    const histWrite = this._pingPong === 0 ? this._historyB! : this._historyA!;

    const bg = device.createBindGroup({
      label: 'bmfr-bg',
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: common.hdrColorTexture.createView() },     // noisy color (fit target)
        { binding: 1, resource: common.gNormalDepthTexture.createView() }, // normal .xyz
        { binding: 2, resource: common.gNormalDepthTexture.createView() }, // depth .w (screen-space proxy)
        { binding: 3, resource: histRead.createView() },                   // temporal history
        { binding: 4, resource: histWrite.createView() },                  // reconstructed + accumulated out
        { binding: 5, resource: { buffer: this._ubo } },
      ],
    });

    // One workgroup per 32×32 block. Each thread owns a 2×2 patch, so a
    // 16×16 workgroup covers a full block.
    const block = BMFR_DEFAULT_UNIFORMS.blockSize;
    const blocksX = Math.ceil(w / block);
    const blocksY = Math.ceil(h / block);

    const pass = encoder.beginComputePass(computeDesc('bmfr'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(blocksX, blocksY, 1);
    pass.end();

    // The write half is BOTH this frame's denoised output AND next frame's
    // history-read. Flip the ping-pong so next frame reads what we just wrote.
    this._pingPong = 1 - this._pingPong;
    this._historyValid = true;

    return histWrite;
  }

  resize(width: number, height: number): void {
    if (this._historyA == null) return; // not yet initialized
    this._allocHistory(width, height);
    this._lastHasHistory = -1; // force re-pack on next dispatch
  }

  dispose(): void {
    this._historyA?.destroy();
    this._historyB?.destroy();
    this._historyA = null;
    this._historyB = null;
    this._ubo?.destroy();
  }
}
