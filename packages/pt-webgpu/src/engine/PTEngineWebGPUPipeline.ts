import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import type { AccumResources } from './PTEngineWebGPUResources.js';

/**
 * Compute-pipeline ownership for `PTEngineWebGPU`.
 *
 * Compiles the path-trace shader module once (lazily, on first use) and caches
 * the resulting `GPUComputePipeline`, its auto-derived bind-group layout, and
 * the 512-byte params UBO. The orchestrator calls `ensure()` per frame; the
 * call is a fast no-op once the pipeline is hot.
 *
 * Bind-group construction lives here too — there are 24 bindings drawn from
 * `AccumResources` + `UploadedSceneBuffers` + the params buffer, and isolating
 * the entry list keeps the dispatch path in the orchestrator readable.
 */
export interface PathTracePipeline {
  readonly computePipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly paramsBuffer: GPUBuffer;
}

const PARAMS_BUFFER_BYTES = 512;

export class PTEngineWebGPUPipeline {
  readonly #device: GPUDevice;

  #computePipeline: GPUComputePipeline | null = null;
  #bindGroupLayout: GPUBindGroupLayout | null = null;
  #paramsBuffer: GPUBuffer | null = null;

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  get paramsBuffer(): GPUBuffer | null {
    return this.#paramsBuffer;
  }

  get computePipeline(): GPUComputePipeline | null {
    return this.#computePipeline;
  }

  /**
   * Compile the path-trace pipeline if it has not yet been built. Idempotent:
   * after the first call, all subsequent calls are O(1) field-presence checks.
   */
  ensure(): void {
    if (
      this.#computePipeline != null &&
      this.#bindGroupLayout != null &&
      this.#paramsBuffer != null
    ) {
      return;
    }
    this.#paramsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.params',
      size: PARAMS_BUFFER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const module = this.#device.createShaderModule({
      label: 'vitrum.pt-webgpu.pathTrace',
      code: PT_WEBGPU_TRACE_WGSL,
    });
    this.#computePipeline = this.#device.createComputePipeline({
      label: 'vitrum.pt-webgpu.pathTrace.pipeline',
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main',
      },
    });
    this.#bindGroupLayout = this.#computePipeline.getBindGroupLayout(0);
  }

  snapshot(): PathTracePipeline {
    if (
      this.#computePipeline == null ||
      this.#bindGroupLayout == null ||
      this.#paramsBuffer == null
    ) {
      throw new Error('PTEngineWebGPUPipeline.snapshot: pipeline not compiled');
    }
    return {
      computePipeline: this.#computePipeline,
      bindGroupLayout: this.#bindGroupLayout,
      paramsBuffer: this.#paramsBuffer,
    };
  }

  /**
   * Build the 24-entry path-trace bind group. The orchestrator caches the
   * returned handle until the underlying resources are reallocated.
   *
   * Layout (matches `@binding(...)` annotations in pathTraceBruteforce.wgsl):
   *   0  accum image (storage texture, rgba16float)
   *   1  params (uniform buffer, 512 bytes)
   *   2  accum buffer (storage, RW)
   *   3..8  scene geometry (positions, indices, triMaterialIds, materials,
   *         bvh, normals)
   *   9..12 aux storage textures (normalDepth, albedo, variance,
   *         motionVectors)
   *   13    variance moments buffer (storage, RW)
   *   14..17 analytic-shape arrays (headers, params, localToWorld,
   *         worldToLocal)
   *   18..19 environment HDRI (texels, CDF)
   *   20..23 per-light arrays (point, spot, rectArea, meshArea)
   */
  buildPathTraceBindGroup(
    pipeline: PathTracePipeline,
    accum: AccumResources,
    sceneBuffers: UploadedSceneBuffers,
  ): GPUBindGroup {
    return this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.pathTrace.bindgroup',
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: accum.accumView },
        { binding: 1, resource: { buffer: pipeline.paramsBuffer } },
        { binding: 2, resource: { buffer: accum.accumBuffer } },
        { binding: 3, resource: { buffer: sceneBuffers.positionsBuffer } },
        { binding: 4, resource: { buffer: sceneBuffers.indicesBuffer } },
        { binding: 5, resource: { buffer: sceneBuffers.triMaterialIdsBuffer } },
        { binding: 6, resource: { buffer: sceneBuffers.materialsBuffer } },
        { binding: 7, resource: { buffer: sceneBuffers.bvhNodesBuffer } },
        { binding: 8, resource: { buffer: sceneBuffers.normalsBuffer } },
        { binding: 9, resource: accum.normalDepthView },
        { binding: 10, resource: accum.albedoView },
        { binding: 11, resource: accum.varianceView },
        { binding: 12, resource: accum.motionVectorsView },
        { binding: 13, resource: { buffer: accum.varianceMomentsBuffer } },
        { binding: 14, resource: { buffer: sceneBuffers.analyticHeadersBuffer } },
        { binding: 15, resource: { buffer: sceneBuffers.analyticParamsBuffer } },
        { binding: 16, resource: { buffer: sceneBuffers.analyticLocalToWorldBuffer } },
        { binding: 17, resource: { buffer: sceneBuffers.analyticWorldToLocalBuffer } },
        { binding: 18, resource: { buffer: sceneBuffers.environmentMapTexelsBuffer } },
        { binding: 19, resource: { buffer: sceneBuffers.environmentMapCdfBuffer } },
        { binding: 20, resource: { buffer: sceneBuffers.pointLightsBuffer } },
        { binding: 21, resource: { buffer: sceneBuffers.spotLightsBuffer } },
        { binding: 22, resource: { buffer: sceneBuffers.rectAreaLightsBuffer } },
        { binding: 23, resource: { buffer: sceneBuffers.meshAreaLightsBuffer } },
      ],
    });
  }

  /** Release the params buffer and forget the cached pipeline. Idempotent. */
  destroy(): void {
    this.#paramsBuffer?.destroy();
    this.#paramsBuffer = null;
    this.#computePipeline = null;
    this.#bindGroupLayout = null;
  }
}
