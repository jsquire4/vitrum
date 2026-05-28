/**
 * GpuResources — the cohesive GPU-resource-lifecycle cluster extracted from the
 * `PTEngineWebGPU` god-class (T14-followup, mirrors the W1-R2 FrameResources split
 * on `@vitrum/walkaround-hybrid`).
 *
 * Owns, as a single sub-struct on the engine (`#gpu`):
 *   - the accumulation + aux textures (accum / normalDepth / albedo / variance /
 *     motionVectors) and their cached views,
 *   - the accum / varianceMoments storage buffers + the params uniform buffer,
 *   - the compute pipeline(s) (path-trace + optional BDPT light-subpath) and the
 *     auto-derived group-0 bind-group layout,
 *   - the cached per-frame bind groups (group 0/1/2), and
 *   - the current accum dims (width / height / byte size).
 *
 * Behavior is preserved verbatim from the prior inline implementation. The only
 * cross-cutting state that stays on the engine is `#samplesAccumulated`: methods
 * that reset it (`ensureAccumResources` on recreate) report that back to the
 * caller (return `recreated: boolean`) rather than reaching into engine state.
 * Bind-group *construction* takes the scene buffers + BDPT light-path view as
 * explicit parameters (those live on the engine), but the resulting groups are
 * cached here because their lifetime is tied to the accum views + pipeline.
 */

import type { PtWebgpuTraceTier } from './traceTier.js';
import type { UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import { PT_WEBGPU_TRACE_WGSL } from './wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from './wgsl/pathTraceBruteforceLite.wgsl.js';

export class GpuResources {
  readonly #device: GPUDevice;
  readonly #traceTier: PtWebgpuTraceTier;
  readonly #bdpt: boolean;

  accumTexture: GPUTexture | null = null;
  accumView: GPUTextureView | null = null;
  normalDepthTexture: GPUTexture | null = null;
  normalDepthView: GPUTextureView | null = null;
  albedoTexture: GPUTexture | null = null;
  albedoView: GPUTextureView | null = null;
  varianceTexture: GPUTexture | null = null;
  varianceView: GPUTextureView | null = null;
  motionVectorsTexture: GPUTexture | null = null;
  motionVectorsView: GPUTextureView | null = null;
  accumBuffer: GPUBuffer | null = null;
  varianceMomentsBuffer: GPUBuffer | null = null;
  accumBufferByteSize = 0;
  accumWidth = 0;
  accumHeight = 0;

  /** Reused bind groups until scene buffers or accum views are recreated. */
  pathTraceBindGroup: GPUBindGroup | null = null;
  pathTraceBindGroup1: GPUBindGroup | null = null;
  pathTraceBindGroup2: GPUBindGroup | null = null;

  paramsBuffer: GPUBuffer | null = null;
  computePipeline: GPUComputePipeline | null = null;
  bdptSubpathPipeline: GPUComputePipeline | null = null;
  bindGroupLayout: GPUBindGroupLayout | null = null;

  constructor(device: GPUDevice, traceTier: PtWebgpuTraceTier, bdpt: boolean) {
    this.#device = device;
    this.#traceTier = traceTier;
    this.#bdpt = bdpt;
  }

  destroyAccumTexture(): void {
    this.accumTexture?.destroy();
    this.accumTexture = null;
    this.accumView = null;
    this.normalDepthTexture?.destroy();
    this.normalDepthTexture = null;
    this.normalDepthView = null;
    this.albedoTexture?.destroy();
    this.albedoTexture = null;
    this.albedoView = null;
    this.varianceTexture?.destroy();
    this.varianceTexture = null;
    this.varianceView = null;
    this.motionVectorsTexture?.destroy();
    this.motionVectorsTexture = null;
    this.motionVectorsView = null;
    this.accumBuffer?.destroy();
    this.accumBuffer = null;
    this.varianceMomentsBuffer?.destroy();
    this.varianceMomentsBuffer = null;
    this.accumBufferByteSize = 0;
    this.accumWidth = 0;
    this.accumHeight = 0;
  }

  clearAccumBuffer(): void {
    if (this.accumBuffer == null) return;
    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.clearAccum' });
    encoder.clearBuffer(this.accumBuffer);
    if (this.varianceMomentsBuffer != null) {
      encoder.clearBuffer(this.varianceMomentsBuffer);
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  /**
   * (Re)allocate the accum + aux textures and the accum / varianceMoments
   * buffers to the requested dims. Returns `true` if a recreate happened (which
   * means the caller must reset its sample counter — the prior inline version
   * set `#samplesAccumulated = 0` here; that one piece of engine state is now
   * reported back rather than reached into). Returns `false` on the cache-hit
   * fast path where nothing was touched.
   */
  ensureAccumResources(width: number, height: number): boolean {
    const targetByteSize = width * height * 16;
    const textureReady =
      this.accumTexture != null && this.accumWidth === width && this.accumHeight === height;
    const bufferReady = this.accumBuffer != null && this.accumBufferByteSize === targetByteSize;
    if (textureReady && bufferReady) {
      return false;
    }
    this.destroyAccumTexture();
    this.accumTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.accum',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.accumView = this.accumTexture.createView();
    this.normalDepthTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.normalDepth',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.normalDepthView = this.normalDepthTexture.createView();
    this.albedoTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.albedo',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.albedoView = this.albedoTexture.createView();
    this.varianceTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.variance',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.varianceView = this.varianceTexture.createView();
    if (this.#traceTier === 'full') {
      this.motionVectorsTexture = this.#device.createTexture({
        label: 'vitrum.pt-webgpu.motionVectors',
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.motionVectorsView = this.motionVectorsTexture.createView();
    }
    this.accumBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.accum.buffer',
      size: Math.max(16, targetByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (this.#traceTier === 'full') {
      this.varianceMomentsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.varianceMoments.buffer',
        size: Math.max(16, targetByteSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.accumBufferByteSize = targetByteSize;
    this.accumWidth = width;
    this.accumHeight = height;
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
    this.clearAccumBuffer();
    return true;
  }

  ensurePipeline(): void {
    if (this.computePipeline != null && this.bindGroupLayout != null && this.paramsBuffer != null) {
      return;
    }
    this.paramsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.params',
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const traceWgsl =
      this.#traceTier === 'lite' ? PT_WEBGPU_TRACE_LITE_WGSL : PT_WEBGPU_TRACE_WGSL;
    const module = this.#device.createShaderModule({
      label: `vitrum.pt-webgpu.pathTrace.${this.#traceTier}`,
      code: traceWgsl,
    });
    this.computePipeline = this.#device.createComputePipeline({
      label: 'vitrum.pt-webgpu.pathTrace.pipeline',
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main',
      },
    });
    this.bindGroupLayout = this.computePipeline.getBindGroupLayout(0);
    if (this.#traceTier === 'full' && this.#bdpt) {
      this.bdptSubpathPipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.bdptLightSubpath.pipeline',
        layout: 'auto',
        compute: {
          module,
          entryPoint: 'bdptExtendLightSubpath',
        },
      });
    } else {
      this.bdptSubpathPipeline = null;
    }
  }

  /**
   * Build (and cache) the path-trace bind group(s) from the current accum views,
   * params buffer, pipeline layout, the supplied scene buffers, and the BDPT
   * light-path view. Returns group 0 (the always-present group). Groups 1/2 are
   * only created on the `full` tier and are read back off this struct by the
   * caller. Idempotent: if group 0 is already cached, returns it unchanged.
   *
   * `bdptLightPathView` is a thunk so the engine's lazy placeholder-texture
   * creation only fires on the construction branch (matching the prior inline
   * code, which only called `#bdptLightPathView()` when the group was rebuilt).
   *
   * Callers must have already run `ensureAccumResources` + `ensurePipeline` and
   * validated that the views / pipeline / layout / params / scene buffers are
   * non-null (renderFrame's preconditions handle this).
   */
  buildBindGroups(sb: UploadedSceneBuffers, bdptLightPathView: () => GPUTextureView): GPUBindGroup {
    if (this.pathTraceBindGroup != null) return this.pathTraceBindGroup;
    const liteEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: this.accumView! },
      { binding: 1, resource: { buffer: this.paramsBuffer! } },
      { binding: 2, resource: { buffer: this.accumBuffer! } },
      { binding: 3, resource: { buffer: sb.positionsBuffer } },
      { binding: 4, resource: { buffer: sb.indicesBuffer } },
      { binding: 5, resource: { buffer: sb.triMaterialIdsBuffer } },
      { binding: 6, resource: { buffer: sb.materialsBuffer } },
      { binding: 7, resource: { buffer: sb.bvhNodesBuffer } },
      { binding: 8, resource: { buffer: sb.normalsBuffer } },
      { binding: 9, resource: this.normalDepthView! },
      { binding: 10, resource: this.albedoView! },
      { binding: 11, resource: this.varianceView! },
    ];
    const fullGroup0Entries: GPUBindGroupEntry[] = [
      ...liteEntries,
      { binding: 12, resource: this.motionVectorsView! },
      { binding: 13, resource: { buffer: this.varianceMomentsBuffer! } },
    ];
    const fullGroup1Entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: sb.analyticHeadersBuffer } },
      { binding: 1, resource: { buffer: sb.analyticParamsBuffer } },
      { binding: 2, resource: { buffer: sb.analyticLocalToWorldBuffer } },
      { binding: 3, resource: { buffer: sb.analyticWorldToLocalBuffer } },
      { binding: 4, resource: { buffer: sb.environmentMapTexelsBuffer } },
      { binding: 5, resource: { buffer: sb.environmentMapCdfBuffer } },
      { binding: 6, resource: { buffer: sb.pointLightsBuffer } },
      { binding: 7, resource: { buffer: sb.spotLightsBuffer } },
      { binding: 8, resource: { buffer: sb.rectAreaLightsBuffer } },
      { binding: 9, resource: { buffer: sb.meshAreaLightsBuffer } },
    ];
    const fullGroup2Entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: sb.tlasNodesBuffer } },
      { binding: 1, resource: { buffer: sb.tlasInstanceIndicesBuffer } },
      { binding: 2, resource: { buffer: sb.tlasBlasRootsBuffer } },
      { binding: 3, resource: { buffer: sb.tlasInstanceWorldToLocalBuffer } },
      { binding: 4, resource: { buffer: sb.tlasInstanceLocalToWorldBuffer } },
      { binding: 5, resource: bdptLightPathView() },
    ];
    const bindGroup = this.#device.createBindGroup({
      label: `vitrum.pt-webgpu.pathTrace.bindgroup0.${this.#traceTier}`,
      layout: this.bindGroupLayout!,
      entries: this.#traceTier === 'lite' ? liteEntries : fullGroup0Entries,
    });
    this.pathTraceBindGroup = bindGroup;
    if (this.#traceTier === 'full') {
      this.pathTraceBindGroup1 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup1.full',
        layout: this.computePipeline!.getBindGroupLayout(1),
        entries: fullGroup1Entries,
      });
      this.pathTraceBindGroup2 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup2.full',
        layout: this.computePipeline!.getBindGroupLayout(2),
        entries: fullGroup2Entries,
      });
    }
    return bindGroup;
  }

  /** Invalidate the cached bind groups (scene-buffer / accum-view recreation). */
  invalidateBindGroups(): void {
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
  }

  /**
   * Full GPU-resource teardown for engine `dispose()`: destroy the accum textures
   * + buffers, drop the cached bind groups, destroy + null the params buffer, and
   * null the pipeline / layout handles. Mirrors the prior inline dispose order.
   */
  dispose(): void {
    this.destroyAccumTexture();
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
    this.paramsBuffer?.destroy();
    this.paramsBuffer = null;
    this.computePipeline = null;
    this.bdptSubpathPipeline = null;
    this.bindGroupLayout = null;
  }
}
