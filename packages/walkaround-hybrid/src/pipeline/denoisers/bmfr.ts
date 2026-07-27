/**
 * Persistent BMFR denoiser for the walkaround pipeline.
 *
 * The fit pass performs direct, chunked Householder QR on each overlapping
 * block and stores one private coefficient record. The resolve pass evaluates
 * and averages all records covering each pixel, then applies temporal EMA.
 * Signed gNormalDepth values are converted to absolute depth for regression so
 * negative glass depths remain valid surfaces; only zero denotes sky.
 *
 * Reference: Koskela et al. Blockwise Multi-Order Feature Regression for
 * Real-Time Path-Tracing Reconstruction. ACM TOG 38(5), 2019.
 */

import {
  BMFR_BLOCK_FIT_SIZE_BYTES,
  BMFR_DEFAULT_UNIFORMS,
  BMFR_RESOLVE_WORKGROUP_SIZE,
  BMFR_UNIFORMS_SIZE_BYTES,
  packBmfrUniforms,
  type BmfrUniforms,
} from '@vitrum/shared-denoisers';
import { composeWgsl } from '../wgslComposer.js';
import { BMFR_MODULE, WGSL_MODULES } from '../wgslModules.js';
import { checkShaderCompile } from '../shaderUtils.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';
import { publishFrameState } from '../FramePublication.js';
import { shouldResetDenoiserHistory } from './historyReset.js';

interface BmfrSizedResources {
  readonly historyA: GPUTexture;
  readonly historyB: GPUTexture;
  readonly blockFits: GPUBuffer;
  readonly width: number;
  readonly height: number;
}

function destroyResource(resource: { destroy(): void } | null | undefined): void {
  try {
    resource?.destroy();
  } catch {
    // GPU resource retirement is best-effort. A lost-device implementation or
    // hostile host wrapper may throw from destroy(); independent resources must
    // still be retired and an already-published replacement stays successful.
  }
}

function destroySizedResources(resources: BmfrSizedResources | null): void {
  destroyResource(resources?.historyA);
  destroyResource(resources?.historyB);
  destroyResource(resources?.blockFits);
}

function blockFitBufferSize(width: number, height: number): number {
  const stride = BMFR_DEFAULT_UNIFORMS.blockStride;
  return (
    Math.ceil(width / stride) *
    Math.ceil(height / stride) *
    BMFR_BLOCK_FIT_SIZE_BYTES
  );
}

export class BmfrDenoiser implements Denoiser {
  readonly id = 'bmfr' as const;
  readonly passLabels = DENOISER_PASS_LABELS.bmfr;

  private _device: GPUDevice | null = null;
  private _fitPipeline: GPUComputePipeline | null = null;
  private _resolvePipeline: GPUComputePipeline | null = null;
  private _ubo: GPUBuffer | null = null;
  private _sized: BmfrSizedResources | null = null;
  private _lastHasHistory = -1;
  private _pingPong = 0;
  private _historyValid = false;
  private _lifecycleGeneration = 0;

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    const generation = ++this._lifecycleGeneration;
    const { device, width, height } = ctx;
    const code = composeWgsl(BMFR_MODULE, WGSL_MODULES);
    const shader = device.createShaderModule({ label: 'bmfr', code });
    await checkShaderCompile(shader, 'bmfr');
    const [fitPipeline, resolvePipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: 'bmfr-fit',
        layout: 'auto',
        compute: { module: shader, entryPoint: 'bmfrMain' },
      }),
      device.createComputePipelineAsync({
        label: 'bmfr-resolve',
        layout: 'auto',
        compute: { module: shader, entryPoint: 'bmfrResolve' },
      }),
    ]);
    if (generation !== this._lifecycleGeneration) {
      throw new Error('BMFR initialization was superseded by dispose or reinitialize');
    }

    let nextUbo: GPUBuffer | null = null;
    let nextSized: BmfrSizedResources | null = null;
    try {
      nextUbo = device.createBuffer({
        label: 'bmfr-ubo',
        size: BMFR_UNIFORMS_SIZE_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      nextSized = this._createSizedResources(device, width, height);
    } catch (error) {
      destroyResource(nextUbo);
      destroySizedResources(nextSized);
      throw error;
    }

    const previousUbo = this._ubo;
    const previousSized = this._sized;
    this._device = device;
    this._fitPipeline = fitPipeline;
    this._resolvePipeline = resolvePipeline;
    this._ubo = nextUbo;
    this._sized = nextSized;
    this._lastHasHistory = -1;
    this._pingPong = 0;
    this._historyValid = false;
    destroyResource(previousUbo);
    destroySizedResources(previousSized);
  }

  state(): DenoiserState {
    return DENOISER_READY_STATE;
  }

  private _createSizedResources(
    device: GPUDevice,
    width: number,
    height: number,
  ): BmfrSizedResources {
    if (!Number.isSafeInteger(width) || width <= 0) {
      throw new RangeError('BMFR width must be a positive safe integer');
    }
    if (!Number.isSafeInteger(height) || height <= 0) {
      throw new RangeError('BMFR height must be a positive safe integer');
    }
    const fitBytes = blockFitBufferSize(width, height);
    if (!Number.isSafeInteger(fitBytes)) {
      throw new RangeError('BMFR block-fit buffer size exceeds the safe integer range');
    }
    if (
      typeof device.limits?.maxBufferSize === 'number' &&
      fitBytes > device.limits.maxBufferSize
    ) {
      throw new RangeError('BMFR block-fit buffer exceeds device maxBufferSize');
    }
    if (
      typeof device.limits?.maxStorageBufferBindingSize === 'number' &&
      fitBytes > device.limits.maxStorageBufferBindingSize
    ) {
      throw new RangeError(
        'BMFR block-fit buffer exceeds device maxStorageBufferBindingSize',
      );
    }

    const textureUsage =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST;
    let historyA: GPUTexture | null = null;
    let historyB: GPUTexture | null = null;
    let blockFits: GPUBuffer | null = null;
    try {
      historyA = device.createTexture({
        label: 'bmfr-history-a',
        size: [width, height],
        format: 'rgba16float',
        usage: textureUsage,
      });
      historyB = device.createTexture({
        label: 'bmfr-history-b',
        size: [width, height],
        format: 'rgba16float',
        usage: textureUsage,
      });
      blockFits = device.createBuffer({
        label: 'bmfr-block-fits',
        size: fitBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      return { historyA, historyB, blockFits, width, height };
    } catch (error) {
      destroyResource(historyA);
      destroyResource(historyB);
      destroyResource(blockFits);
      throw error;
    }
  }

  private _packUniforms(hasHistory: number): void {
    if (this._device == null || this._ubo == null) {
      throw new Error('BMFR denoiser is not initialized');
    }
    if (this._lastHasHistory === hasHistory) return;
    const scratch = new ArrayBuffer(BMFR_UNIFORMS_SIZE_BYTES);
    const uniforms: BmfrUniforms = {
      ...BMFR_DEFAULT_UNIFORMS,
      positionMode: 1,
      hasHistory,
    };
    packBmfrUniforms(uniforms, scratch);
    this._device.queue.writeBuffer(this._ubo, 0, scratch);
    this._lastHasHistory = hasHistory;
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture {
    const fitPipeline = this._fitPipeline;
    const resolvePipeline = this._resolvePipeline;
    const ubo = this._ubo;
    const sized = this._sized;
    if (
      this._device == null ||
      fitPipeline == null ||
      resolvePipeline == null ||
      ubo == null ||
      sized == null
    ) {
      throw new Error('BMFR denoiser is not initialized');
    }
    if (ctx.width !== sized.width || ctx.height !== sized.height) {
      throw new Error(
        'BMFR dispatch dimensions do not match allocated resources; call resize first',
      );
    }

    const { device, encoder, resources, computeDesc, isMoving, resourceCache } =
      ctx;
    const common = resources.common;
    const resetHistory = shouldResetDenoiserHistory(ctx.frameIndex, isMoving);
    const useHistory = this._historyValid && !resetHistory;
    this._packUniforms(useHistory ? 1 : 0);

    const historyRead =
      this._pingPong === 0 ? sized.historyA : sized.historyB;
    const historyWrite =
      this._pingPong === 0 ? sized.historyB : sized.historyA;
    const colorView =
      resourceCache?.textureView(common.hdrColorTexture) ??
      common.hdrColorTexture.createView();
    const normalDepthView =
      resourceCache?.textureView(common.gNormalDepthTexture) ??
      common.gNormalDepthTexture.createView();

    const fitBindGroup = cachedBindGroup(
      resourceCache,
      'denoiser:bmfr-fit',
      [common.hdrColorTexture, common.gNormalDepthTexture, sized.blockFits, ubo],
      () =>
        device.createBindGroup({
          label: 'bmfr-fit-bg',
          layout: fitPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: colorView },
            { binding: 1, resource: normalDepthView },
            { binding: 2, resource: normalDepthView },
            { binding: 4, resource: { buffer: sized.blockFits } },
            { binding: 5, resource: { buffer: ubo } },
          ],
        }),
    );
    const resolveBindGroup = cachedBindGroup(
      resourceCache,
      'denoiser:bmfr-resolve',
      [
        common.hdrColorTexture,
        common.gNormalDepthTexture,
        historyRead,
        historyWrite,
        sized.blockFits,
        ubo,
      ],
      () =>
        device.createBindGroup({
          label: 'bmfr-resolve-bg',
          layout: resolvePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: colorView },
            { binding: 1, resource: normalDepthView },
            { binding: 2, resource: normalDepthView },
            {
              binding: 3,
              resource:
                resourceCache?.textureView(historyRead) ??
                historyRead.createView(),
            },
            { binding: 4, resource: { buffer: sized.blockFits } },
            { binding: 5, resource: { buffer: ubo } },
            {
              binding: 6,
              resource:
                resourceCache?.textureView(historyWrite) ??
                historyWrite.createView(),
            },
          ],
        }),
    );

    const stride = BMFR_DEFAULT_UNIFORMS.blockStride;
    const fitPass = encoder.beginComputePass(computeDesc('bmfr-fit'));
    fitPass.setPipeline(fitPipeline);
    fitPass.setBindGroup(0, fitBindGroup);
    fitPass.dispatchWorkgroups(
      Math.ceil(ctx.width / stride),
      Math.ceil(ctx.height / stride),
      1,
    );
    fitPass.end();

    const resolvePass = encoder.beginComputePass(computeDesc('bmfr-resolve'));
    resolvePass.setPipeline(resolvePipeline);
    resolvePass.setBindGroup(0, resolveBindGroup);
    resolvePass.dispatchWorkgroups(
      Math.ceil(ctx.width / BMFR_RESOLVE_WORKGROUP_SIZE),
      Math.ceil(ctx.height / BMFR_RESOLVE_WORKGROUP_SIZE),
      1,
    );
    resolvePass.end();

    const nextPingPong = 1 - this._pingPong;
    publishFrameState(ctx.publication, () => {
      this._pingPong = nextPingPong;
      this._historyValid = true;
    });
    return historyWrite;
  }

  resize(width: number, height: number): void {
    const device = this._device;
    if (device == null || this._sized == null) return;
    const next = this._createSizedResources(device, width, height);
    const previous = this._sized;
    this._sized = next;
    this._pingPong = 0;
    this._historyValid = false;
    this._lastHasHistory = -1;
    destroySizedResources(previous);
  }

  dispose(): void {
    this._lifecycleGeneration += 1;
    const sized = this._sized;
    const ubo = this._ubo;
    this._sized = null;
    this._ubo = null;
    this._fitPipeline = null;
    this._resolvePipeline = null;
    this._device = null;
    this._lastHasHistory = -1;
    this._pingPong = 0;
    this._historyValid = false;
    destroySizedResources(sized);
    destroyResource(ubo);
  }
}
