/**
 * Pipeline-wide temporal luminance variance tracker.
 *
 * `atrous-variance` already owns the same Welford update as part of its
 * denoiser dispatch. This pass supplies that producer for every other
 * denoiser mode (and for the raw-denoiser A/B bypass) so adaptive sampling
 * and the public FrameOutput variance view never read an unwritten texture.
 */

import { defineUbo } from '@vitrum/shared-samplers';
import { WELFORD_TEMPORAL_MODULE } from '../../shaders/welfordTemporal.wgsl.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';
import { publishFrameState } from '../FramePublication.js';
import { checkShaderCompile } from '../shaderUtils.js';
import { composeWgsl } from '../wgslComposer.js';
import { WGSL_MODULES } from '../wgslModules.js';
import type { PassLabel } from '../timestampQueries.js';
import type { PingPongRef } from './passRefs.js';

const WELFORD_TEMPORAL_UBO = defineUbo([
  { name: 'sampleN', type: 'u32' },
  { name: 'forceReset', type: 'u32' },
] as const);

function buildVarianceTrackerBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  currentRadiance: GPUTextureView,
  previousState: GPUTextureView,
  nextState: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'variance-tracker-bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: currentRadiance },
      { binding: 1, resource: previousState },
      { binding: 2, resource: nextState },
      { binding: 3, resource: { buffer: ubo } },
    ],
  });
}

export class VarianceTrackerPass implements Pass {
  readonly id = 'variance-tracker' as const;
  readonly dependencies: readonly string[] = ['shade'];

  private readonly _pingPongRef: PingPongRef;
  private readonly _shouldDispatch: () => boolean;
  private _pipeline: GPUComputePipeline | null = null;
  private _ubo: GPUBuffer | null = null;

  constructor(
    pingPongRef: PingPongRef,
    shouldDispatch: () => boolean,
  ) {
    this._pingPongRef = pingPongRef;
    this._shouldDispatch = shouldDispatch;
  }

  gates(): boolean {
    return this._shouldDispatch();
  }

  get passLabels(): readonly PassLabel[] {
    return this._shouldDispatch() ? ['welford-temporal'] : [];
  }

  async initialize(ctx: PassInitContext): Promise<void> {
    const module = ctx.device.createShaderModule({
      label: 'variance-tracker',
      code: composeWgsl(WELFORD_TEMPORAL_MODULE, WGSL_MODULES),
    });
    await checkShaderCompile(module, 'variance-tracker');
    const pipeline = await ctx.device.createComputePipelineAsync({
      label: 'variance-tracker',
      layout: 'auto',
      compute: { module, entryPoint: 'welfordTemporalMain' },
    });
    const ubo = ctx.device.createBuffer({
      label: 'variance-tracker-ubo',
      size: WELFORD_TEMPORAL_UBO.sizeBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._pipeline = pipeline;
    this._ubo = ubo;
  }

  dispatch(ctx: PassDispatchContext): void {
    const pipeline = this._pipeline;
    const ubo = this._ubo;
    if (pipeline == null || ubo == null) {
      throw new Error('VarianceTrackerPass.dispatch called before initialize.');
    }

    const common = ctx.resources.common;
    const read = this._pingPongRef.value === 0
      ? common.varianceBuffer
      : common.varianceBufferAux;
    const write = this._pingPongRef.value === 0
      ? common.varianceBufferAux
      : common.varianceBuffer;

    const bytes = new ArrayBuffer(WELFORD_TEMPORAL_UBO.sizeBytes);
    WELFORD_TEMPORAL_UBO.pack(new DataView(bytes), 0, {
      sampleN: ctx.frameIndex + 1,
      forceReset: ctx.frameIndex === 0 || ctx.frameState.isMoving ? 1 : 0,
    });
    ctx.device.queue.writeBuffer(ubo, 0, bytes);

    const build = (): GPUBindGroup => buildVarianceTrackerBindGroup(
      ctx.device,
      pipeline,
      ctx.resourceCache?.textureView(common.hdrTotalTexture)
        ?? common.hdrTotalTexture.createView(),
      ctx.resourceCache?.textureView(read) ?? read.createView(),
      ctx.resourceCache?.textureView(write) ?? write.createView(),
      ubo,
    );
    const bindGroup = cachedBindGroup(
      ctx.resourceCache,
      'pass:variance-tracker',
      [common.hdrTotalTexture, read, write, ubo],
      build,
    );
    const pass = ctx.encoder.beginComputePass(ctx.computeDesc('welford-temporal'));
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(ctx.wgX16, ctx.wgY16, 1);
    pass.end();

    const next = 1 - this._pingPongRef.value;
    publishFrameState(ctx.publication, () => {
      this._pingPongRef.value = next;
    });
  }

  dispose(): void {
    try { this._ubo?.destroy(); } catch { /* best-effort teardown */ }
    this._ubo = null;
    this._pipeline = null;
  }
}
