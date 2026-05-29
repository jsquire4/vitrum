/**
 * GpuResources — the cohesive GPU-resource-lifecycle cluster extracted from the
 * `PTEngineWebGPU` god-class (T14-followup, mirrors the W1-R2 FrameResources split
 * on `@vitrum/walkaround-hybrid`).
 *
 * Owns, as a single sub-struct on the engine (`#gpu`):
 *   - the accumulation + aux textures (accum / normalDepth / albedo / variance /
 *     motionVectors) and their cached views,
 *   - the accum / varianceMoments storage buffers + the params uniform buffer,
 *   - the compute pipeline(s) (path-trace + optional BDPT light-subpath) sharing
 *     ONE explicit GPUPipelineLayout, and the explicit per-group bind-group
 *     layouts that pipeline layout is built from,
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
import { composePtWebgpuTraceWgsl } from './wgsl/pathTraceBruteforce.wgsl.js';
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
  /** WS2 — light-tree node buffer bind group (full tier only). */
  pathTraceBindGroup3: GPUBindGroup | null = null;

  paramsBuffer: GPUBuffer | null = null;
  computePipeline: GPUComputePipeline | null = null;
  bdptSubpathPipeline: GPUComputePipeline | null = null;
  /**
   * Explicit group-0 bind-group layout. Used to build `pathTraceBindGroup`.
   *
   * Both compute pipelines (path-trace `main` + BDPT `bdptExtendLightSubpath`)
   * share ONE explicit `GPUPipelineLayout` built from these layouts. A bind
   * group built against an explicit layout is NOT pipeline-exclusive, so the
   * SAME bind groups can be set on both pipelines — which is exactly what the
   * BDPT light-subpath pass needs (it reuses the path-trace scene/params/light
   * bindings). Auto-generated layouts (`layout:'auto'`) are pipeline-exclusive
   * per the WebGPU spec, so they would reject a cross-pipeline `setBindGroup`.
   */
  bindGroupLayout: GPUBindGroupLayout | null = null;
  /** Explicit group-1 layout (full tier only): analytics + env + area lights. */
  bindGroupLayout1: GPUBindGroupLayout | null = null;
  /** Explicit group-2 layout (full tier only): TLAS table + BDPT scratch buffers. */
  bindGroupLayout2: GPUBindGroupLayout | null = null;
  /** Explicit group-3 layout (full tier only): WS2 light-tree node buffer. */
  bindGroupLayout3: GPUBindGroupLayout | null = null;

  /**
   * BDPT eye-subpath scratch stack (D2): a per-pixel × maxEyeDepth read_write
   * storage buffer (2× vec4 / eye vertex = 32 B). Bound at group(2) binding(6) on
   * the full tier. A 32-byte placeholder is kept when BDPT is off so the explicit
   * group-2 layout (which always declares binding 6) stays satisfied.
   */
  bdptEyeStackBuffer: GPUBuffer | null = null;
  bdptEyeStackByteSize = 0;

  /** Bytes per eye vertex in the scratch stack: 2× vec4f = 32. */
  static readonly BDPT_EYE_VERTEX_BYTES = 32;
  /**
   * Safety ceiling for the eye-stack allocation. The full-depth (8) per-pixel
   * stack is ~530 MB at 1920×1080; above this ceiling we refuse to grow the
   * buffer and warn rather than silently allocating a multi-hundred-MB region.
   */
  static readonly BDPT_EYE_STACK_MAX_BYTES = 384 * 1024 * 1024; // 384 MiB

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
    this.pathTraceBindGroup3 = null;
    this.clearAccumBuffer();
    return true;
  }

  /**
   * Build the explicit `GPUBindGroupLayout`s for the current tier and wrap them
   * in a single `GPUPipelineLayout` shared by both compute pipelines. The
   * binding indices / resource types / visibility MUST match what the prior
   * `layout:'auto'` derived from the WGSL `@group/@binding` decls (see
   * `wgsl/pathTrace/material.wgsl.ts`) so the existing bind-group construction
   * in `buildBindGroups` stays valid unchanged.
   *
   * All bindings are COMPUTE-visible (the entire kernel is one compute stage).
   * The explicit layout is a SUPERSET that satisfies both `main` (uses every
   * binding) and `bdptExtendLightSubpath` (uses a subset) — an explicit layout
   * may declare bindings an entry point doesn't statically use, so one layout
   * serves both pipelines.
   */
  #buildSharedPipelineLayout(): GPUPipelineLayout {
    const VIS = GPUShaderStage.COMPUTE;
    // WGSL `var<storage, read>`  → 'read-only-storage'.
    const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
    // WGSL `var<storage, read_write>` → 'storage'.
    const rw: GPUBufferBindingLayout = { type: 'storage' };
    // WGSL `var<uniform>` → 'uniform'.
    const uniform: GPUBufferBindingLayout = { type: 'uniform' };
    // WGSL `texture_storage_2d<rgba16float, write>`.
    const storageTex: GPUStorageTextureBindingLayout = {
      access: 'write-only',
      format: 'rgba16float',
      viewDimension: '2d',
    };
    const buf = (binding: number, layout: GPUBufferBindingLayout): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: VIS,
      buffer: layout,
    });
    const tex = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: VIS,
      storageTexture: storageTex,
    });

    // Group 0 — bindings 0..11 (lite) / 0..13 (full). Mirrors material.wgsl.ts.
    const group0Entries: GPUBindGroupLayoutEntry[] = [
      tex(0), // outputTexture (storage texture, write)
      buf(1, uniform), // params (uniform)
      buf(2, rw), // accumBuffer (read_write)
      buf(3, ro), // positions
      buf(4, ro), // indices
      buf(5, ro), // triMaterialIds
      buf(6, ro), // materials
      buf(7, ro), // bvhNodes
      buf(8, ro), // normals
      tex(9), // normalDepthTexture
      tex(10), // albedoTexture
      tex(11), // varianceTexture
    ];
    if (this.#traceTier === 'full') {
      group0Entries.push(
        tex(12), // motionVectorsTexture
        buf(13, rw), // varianceMomentsBuffer (read_write)
      );
    }
    this.bindGroupLayout = this.#device.createBindGroupLayout({
      label: `vitrum.pt-webgpu.layout.group0.${this.#traceTier}`,
      entries: group0Entries,
    });
    const bindGroupLayouts: GPUBindGroupLayout[] = [this.bindGroupLayout];

    if (this.#traceTier === 'full') {
      // Group 1 — 10 read-only storage buffers (analytics + env + area lights).
      this.bindGroupLayout1 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group1.full',
        entries: Array.from({ length: 10 }, (_unused, binding) => buf(binding, ro)),
      });
      // Group 2 — TLAS table (5 read-only) + BDPT light-path + eye-stack (read_write).
      this.bindGroupLayout2 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group2.full',
        entries: [
          buf(0, ro), // tlasNodes
          buf(1, ro), // tlasInstanceIndices
          buf(2, ro), // tlasBlasRoots
          buf(3, ro), // tlasInstanceWorldToLocal
          buf(4, ro), // tlasInstanceLocalToWorld
          buf(5, rw), // bdptLightPath (read_write)
          buf(6, rw), // bdptEyeStack (read_write)
        ],
      });
      // Group 3 — WS2 light-tree node buffer (one read-only storage buffer). A
      // DEDICATED group so the lite tier (which never reaches this branch) carries
      // no group-3 layout, and so adding it leaves groups 0/1/2 byte-identical.
      this.bindGroupLayout3 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group3.full',
        entries: [buf(0, ro)], // lightTree (read-only storage)
      });
      bindGroupLayouts.push(this.bindGroupLayout1, this.bindGroupLayout2, this.bindGroupLayout3);
    } else {
      this.bindGroupLayout1 = null;
      this.bindGroupLayout2 = null;
      this.bindGroupLayout3 = null;
    }

    return this.#device.createPipelineLayout({
      label: `vitrum.pt-webgpu.pipelineLayout.${this.#traceTier}`,
      bindGroupLayouts,
    });
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
    // WS4 — the full-tier kernel is composed for this engine's integrator
    // config: the volumetric SSS random walk is compiled in only when BDPT is
    // OFF (structural gate — energy conservation; BDPT has no medium logic).
    const traceWgsl =
      this.#traceTier === 'lite'
        ? PT_WEBGPU_TRACE_LITE_WGSL
        : composePtWebgpuTraceWgsl(this.#bdpt);
    const module = this.#device.createShaderModule({
      label: `vitrum.pt-webgpu.pathTrace.${this.#traceTier}`,
      code: traceWgsl,
    });
    // ONE explicit pipeline layout shared by BOTH pipelines. Auto layouts are
    // pipeline-exclusive (WebGPU spec), so a bind group built against the
    // path-trace pipeline's auto layout cannot be set on the BDPT pipeline (and
    // vice versa). The BDPT light-subpath pass reuses the path-trace bind groups,
    // so it requires a shared explicit layout to dispatch on real hardware.
    const pipelineLayout = this.#buildSharedPipelineLayout();
    this.computePipeline = this.#device.createComputePipeline({
      label: 'vitrum.pt-webgpu.pathTrace.pipeline',
      layout: pipelineLayout,
      compute: {
        module,
        entryPoint: 'main',
      },
    });
    if (this.#traceTier === 'full' && this.#bdpt) {
      this.bdptSubpathPipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.bdptLightSubpath.pipeline',
        layout: pipelineLayout,
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
   * (Re)allocate the BDPT eye-subpath scratch stack for the given render dims and
   * per-pixel eye depth. Returns `true` if BDPT connections may proceed this
   * frame, `false` if the allocation would exceed the safety ceiling (caller must
   * skip the BDPT pass; a stale/placeholder buffer remains bound so the pipeline
   * still validates). Sizes the buffer to `width·height·maxDepth·32 B`. When
   * BDPT is off, keeps only a 32-byte placeholder.
   *
   * The full-tier explicit group-2 layout always declares the `bdptEyeStack`
   * binding (6), so a non-null buffer must always exist on the full tier.
   */
  ensureBdptEyeStack(width: number, height: number, maxDepth: number, bdptActive: boolean): boolean {
    if (this.#traceTier !== 'full') {
      return false;
    }
    const targetBytes = bdptActive
      ? Math.max(
          GpuResources.BDPT_EYE_VERTEX_BYTES,
          width * height * Math.max(1, maxDepth) * GpuResources.BDPT_EYE_VERTEX_BYTES,
        )
      : GpuResources.BDPT_EYE_VERTEX_BYTES;

    if (bdptActive && targetBytes > GpuResources.BDPT_EYE_STACK_MAX_BYTES) {
      // Non-silent refusal: report the size and skip BDPT this frame rather than
      // allocating a multi-hundred-MB scratch region. The caller falls back to
      // the unidirectional path; a placeholder buffer keeps the layout valid.
      const mib = (targetBytes / (1024 * 1024)).toFixed(1);
      console.warn(
        `[vitrum/pt-webgpu] BDPT eye-stack scratch would be ${mib} MiB ` +
          `(${width}×${height} × depth ${maxDepth} × 32 B), exceeding the ` +
          `${(GpuResources.BDPT_EYE_STACK_MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB ceiling. ` +
          'Skipping BDPT connections this frame — lower resolutionFactor, cap bounces, or tile.',
      );
      if (this.bdptEyeStackBuffer == null) {
        this.bdptEyeStackBuffer = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.bdpt.eyeStack.placeholder',
          size: GpuResources.BDPT_EYE_VERTEX_BYTES,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.bdptEyeStackByteSize = GpuResources.BDPT_EYE_VERTEX_BYTES;
        this.invalidateBindGroups();
      }
      return false;
    }

    if (this.bdptEyeStackBuffer != null && this.bdptEyeStackByteSize === targetBytes) {
      return bdptActive;
    }
    this.bdptEyeStackBuffer?.destroy();
    this.bdptEyeStackBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.bdpt.eyeStack',
      size: targetBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bdptEyeStackByteSize = targetBytes;
    this.invalidateBindGroups();
    return bdptActive;
  }

  /**
   * Build (and cache) the path-trace bind group(s) from the current accum views,
   * params buffer, pipeline layout, the supplied scene buffers, and the BDPT
   * light-path buffer. Returns group 0 (the always-present group). Groups 1/2 are
   * only created on the `full` tier and are read back off this struct by the
   * caller. Idempotent: if group 0 is already cached, returns it unchanged.
   *
   * `bdptLightPathBuffer` is a thunk so the engine's lazy placeholder-buffer
   * creation only fires on the construction branch (matching the prior inline
   * code, which only called `#bdptLightPathBuffer()` when the group was rebuilt).
   *
   * Callers must have already run `ensureAccumResources` + `ensurePipeline` and
   * validated that the views / pipeline / layout / params / scene buffers are
   * non-null (renderFrame's preconditions handle this).
   */
  buildBindGroups(sb: UploadedSceneBuffers, bdptLightPathBuffer: () => GPUBuffer): GPUBindGroup {
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
      { binding: 5, resource: { buffer: bdptLightPathBuffer() } },
      { binding: 6, resource: { buffer: this.bdptEyeStackBuffer! } },
    ];
    const bindGroup = this.#device.createBindGroup({
      label: `vitrum.pt-webgpu.pathTrace.bindgroup0.${this.#traceTier}`,
      layout: this.bindGroupLayout!,
      entries: this.#traceTier === 'lite' ? liteEntries : fullGroup0Entries,
    });
    this.pathTraceBindGroup = bindGroup;
    if (this.#traceTier === 'full') {
      // Built against the SAME explicit layouts the shared pipeline layout uses,
      // so these groups set cleanly on BOTH the path-trace and BDPT pipelines.
      this.pathTraceBindGroup1 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup1.full',
        layout: this.bindGroupLayout1!,
        entries: fullGroup1Entries,
      });
      this.pathTraceBindGroup2 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup2.full',
        layout: this.bindGroupLayout2!,
        entries: fullGroup2Entries,
      });
      this.pathTraceBindGroup3 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup3.full',
        layout: this.bindGroupLayout3!,
        entries: [{ binding: 0, resource: { buffer: sb.lightTreeBuffer } }],
      });
    }
    return bindGroup;
  }

  /** Invalidate the cached bind groups (scene-buffer / accum-view recreation). */
  invalidateBindGroups(): void {
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
    this.pathTraceBindGroup3 = null;
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
    this.pathTraceBindGroup3 = null;
    this.bdptEyeStackBuffer?.destroy();
    this.bdptEyeStackBuffer = null;
    this.bdptEyeStackByteSize = 0;
    this.paramsBuffer?.destroy();
    this.paramsBuffer = null;
    this.computePipeline = null;
    this.bdptSubpathPipeline = null;
    this.bindGroupLayout = null;
    this.bindGroupLayout1 = null;
    this.bindGroupLayout2 = null;
    this.bindGroupLayout3 = null;
  }
}
