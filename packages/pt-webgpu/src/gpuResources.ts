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
import {
  composePtWebgpuTraceWgsl,
  composePtWebgpuCompositeTraceWgsl,
} from './wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from './wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_SEED_BLIT_WGSL } from './wgsl/seedBlit.wgsl.js';
import {
  composeRestirPtProducerWgsl,
  composeRestirPtTemporalWgsl,
  composeRestirPtSpatialWgsl,
  composeRestirPtResolveWgsl,
  RPT_GROUP0_BINDING_BASE,
} from './wgsl/pathTrace/restirPtCompose.wgsl.js';

export class GpuResources {
  readonly #device: GPUDevice;
  readonly #traceTier: PtWebgpuTraceTier;
  readonly #bdpt: boolean;
  /**
   * Compile-time opt-in for the ReSTIR-PT reservoir/reuse pre-passes (the hero-
   * stack temporal reconnection-reuse path). OFF by default and full-tier only.
   * When OFF, NONE of the reuse resources/pipelines below are ever created, and
   * `renderFrame`'s default megakernel path is byte-identical — the wgslContract
   * SHA pin + every existing test stay green. Mirrors the `#bdpt` flag.
   */
  readonly #restirPtReuse: boolean;

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

  // ── ReSTIR-PT reuse resources (gated by #restirPtReuse; full tier only) ──────
  /**
   * Full-res ReservoirPTHero ping-pong buffers (144 B/px = 36 u32). `Cur` is the
   * producer output that the temporal pass fuses in place; `Prev` is last frame's
   * temporal output (read-only this frame). `swapReservoirs()` exchanges them at
   * frame end so this frame's resolved reservoir becomes next frame's history.
   * STORAGE | COPY_SRC | COPY_DST (COPY_* so a future readback / clear works).
   */
  rptReservoirCur: GPUBuffer | null = null;
  rptReservoirPrev: GPUBuffer | null = null;
  /** A1 — the SPATIAL pass output (post-temporal → spatial → resolve). The spatial
   *  pass reads the temporal output (`Cur`) for neighbour sampling and writes here;
   *  resolve reads this. A dedicated buffer (not a ping-pong slot) so the spatial
   *  pass never writes the slot it samples (hazard-free neighbour reads). */
  rptReservoirSpatial: GPUBuffer | null = null;
  /** `rpt_result`: one vec4f / px (16 B) — the resolve pass's reconnection
   *  indirect (.rgb) + contributing flag (.a). STORAGE | COPY_SRC. */
  rptResultBuffer: GPUBuffer | null = null;
  /** RestirPtParams UBO (32 B: width/height/mClamp/_padA u32 + wCap/3×_pad f32). */
  rptParamsBuffer: GPUBuffer | null = null;
  rptReservoirByteSize = 0;
  rptResultByteSize = 0;
  /** The three reuse compute pipelines (lazy, gated). Each carries its OWN
   *  minimal explicit pipeline layout (groups it statically uses + group-0 reuse
   *  bindings) — see #buildReservoirPipelines. */
  rptProducerPipeline: GPUComputePipeline | null = null;
  rptTemporalPipeline: GPUComputePipeline | null = null;
  rptSpatialPipeline: GPUComputePipeline | null = null;
  rptResolvePipeline: GPUComputePipeline | null = null;
  /** A1 — the COMPOSITE megakernel: E0-direct-only + adds the resolve indirect from
   *  rpt_result (relocated @group(0)@binding(23)). Built alongside the reuse passes
   *  (it needs the reuse-extended group-0 layout for the rpt_result binding). When
   *  reuse is active the engine dispatches THIS instead of the default megakernel. */
  rptCompositePipeline: GPUComputePipeline | null = null;
  /** Explicit group-0 layout for the reuse passes: the megakernel's group-0
   *  bindings PLUS the relocated reuse bindings (20..25). Built once. */
  #rptGroup0Layout: GPUBindGroupLayout | null = null;
  /** Cached per-pass reuse bind groups (rebuilt on scene-buffer / reservoir
   *  recreation via invalidateBindGroups). */
  rptProducerGroup0: GPUBindGroup | null = null;
  rptTemporalGroup0: GPUBindGroup | null = null;
  rptSpatialGroup0: GPUBindGroup | null = null;
  rptResolveGroup0: GPUBindGroup | null = null;

  /** Bytes per ReservoirPTHero (36 u32). MUST equal RESERVOIR_PT_HERO_STRIDE·4
   *  in reservoirPtHero.wgsl.ts (pinned by reservoirPtHeroLayout.test.ts). */
  static readonly RESERVOIR_PT_HERO_BYTES = 144;
  /** RestirPtParams UBO byte size (8 × 4-byte fields). */
  static readonly RESTIR_PT_PARAMS_BYTES = 32;
  /**
   * Safety ceiling for EACH reservoir ping-pong buffer. 144 B/px is ~298 MB at
   * 1920×1080; above this ceiling we refuse to grow (skip reuse this frame) rather
   * than silently allocate. Mirrors the BDPT eye-stack ceiling discipline.
   */
  static readonly RESTIR_PT_RESERVOIR_MAX_BYTES = 384 * 1024 * 1024; // 384 MiB

  /**
   * Progressive walkaround→PT handoff (P8) — the seed-blit compute pipeline +
   * its uniform buffer + filtering sampler. Lazily created on the first
   * `seedAccumBuffer` call and reused thereafter (engine-owned; freed in
   * dispose). The bind group is per-call (it references the caller's seed
   * texture + the current accum buffers), so it is not cached.
   */
  #seedBlitPipeline: GPUComputePipeline | null = null;
  #seedBlitParamsBuffer: GPUBuffer | null = null;
  #seedBlitSampler: GPUSampler | null = null;
  /**
   * Placeholder storage buffer bound at the seed-blit's varianceMoments slot on
   * the lite tier (which has no real `varianceMomentsBuffer`). Keeps the single
   * seed-blit bind-group layout satisfied for both tiers; the seed luminance
   * moments written here are discarded.
   */
  #seedBlitVarPlaceholder: GPUBuffer | null = null;

  /** Bytes per eye vertex in the scratch stack: 2× vec4f = 32. */
  static readonly BDPT_EYE_VERTEX_BYTES = 32;
  /**
   * Safety ceiling for the eye-stack allocation. The full-depth (8) per-pixel
   * stack is ~530 MB at 1920×1080; above this ceiling we refuse to grow the
   * buffer and warn rather than silently allocating a multi-hundred-MB region.
   */
  static readonly BDPT_EYE_STACK_MAX_BYTES = 384 * 1024 * 1024; // 384 MiB

  /**
   * H14-F — once-gate set for per-frame buffer-ceiling console.warns. A warn
   * fires at most once per key (e.g. 'bdptEyeStack', 'restirPtReservoir') for
   * the lifetime of this GpuResources instance. Keys are added on the first
   * warn; subsequent frames that hit the same ceiling are silently suppressed.
   */
  readonly #ceilingWarnedKeys = new Set<string>();

  constructor(
    device: GPUDevice,
    traceTier: PtWebgpuTraceTier,
    bdpt: boolean,
    restirPtReuse = false,
  ) {
    this.#device = device;
    this.#traceTier = traceTier;
    this.#bdpt = bdpt;
    // Reuse is full-tier only: the per-pass layouts bind the full-tier scene
    // groups (analytics/TLAS/lights). On the lite tier the flag is inert.
    this.#restirPtReuse = restirPtReuse && traceTier === 'full';
  }

  /** Whether ReSTIR-PT reuse is active for this engine (compile-time + full-tier). */
  get restirPtReuseEnabled(): boolean {
    return this.#restirPtReuse;
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
   * Item 2e — Clear all allocated ReSTIR-PT reservoir buffers (Cur/Prev/Spatial)
   * when the scene changes or the engine resets so stale temporal history from a
   * previous scene does not bleed into the new one. No-op when the buffers have
   * not yet been allocated. Called by `index.ts reset()` and full setScene.
   */
  clearReservoirBuffers(): void {
    if (this.rptReservoirCur == null) return;
    const encoder = this.#device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.restirPt.clearReservoirs',
    });
    encoder.clearBuffer(this.rptReservoirCur);
    if (this.rptReservoirPrev != null) encoder.clearBuffer(this.rptReservoirPrev);
    if (this.rptReservoirSpatial != null) encoder.clearBuffer(this.rptReservoirSpatial);
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
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.normalDepthView = this.normalDepthTexture.createView();
    this.albedoTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.albedo',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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
    // Drop ALL cached bind groups: the path-trace groups AND (when reuse is on)
    // the reuse group-0, which references the just-recreated accum/aux views.
    this.invalidateBindGroups();
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
      // Group 3 — WS2 light-tree node buffer + P2 material textures (per-vertex
      // UVs, per-material descriptors, the baseColor texture_2d_array, a sampler).
      // A DEDICATED group so the lite tier (which never reaches this branch) carries
      // no group-3 layout, and so adding it leaves groups 0/1/2 byte-identical.
      this.bindGroupLayout3 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group3.full',
        entries: [
          buf(0, ro), // lightTree (read-only storage)
          buf(1, ro), // meshUvs (P2)
          buf(2, ro), // materialTexDescriptors (P2)
          { binding: 3, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTextures sRGB (P2)
          { binding: 4, visibility: VIS, sampler: { type: 'filtering' } }, // materialTexSampler (P2)
          { binding: 5, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTexturesLinear normal/ORM (P2)
        ],
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
      // H14-F: once-gate — warn only on the first frame that hits the ceiling.
      if (!this.#ceilingWarnedKeys.has('bdptEyeStack')) {
        this.#ceilingWarnedKeys.add('bdptEyeStack');
        const mib = (targetBytes / (1024 * 1024)).toFixed(1);
        console.warn(
          `[vitrum/pt-webgpu] BDPT eye-stack scratch would be ${mib} MiB ` +
            `(${width}×${height} × depth ${maxDepth} × 32 B), exceeding the ` +
            `${(GpuResources.BDPT_EYE_STACK_MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB ceiling. ` +
            'Skipping BDPT connections this frame — lower resolutionFactor, cap bounces, or tile. ' +
            '(This warning fires once per engine instance.)',
        );
      }
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

  // ── ReSTIR-PT reuse resource lifecycle (gated; full tier only) ───────────────

  /**
   * (Re)allocate the ReSTIR-PT reservoir ping-pong buffers (`Cur` / `Prev`,
   * 144 B/px), the `rpt_result` buffer (16 B/px), and the RestirPtParams UBO to
   * the requested dims. No-op (returns `false`) when reuse is OFF. Returns `true`
   * when buffers for the requested frame dimensions are available, whether they
   * were freshly created or already cached. Refuses to grow past the per-buffer
   * safety ceiling (returns `false`, leaving any prior buffers as-is); the caller
   * skips the reuse passes that frame instead of dispatching against stale sizes.
   *
   * Both reservoir buffers are zero-cleared on (re)allocation so the FIRST frame's
   * temporal pass reads an empty (M=0) history rather than garbage.
   */
  ensureReservoirBuffers(width: number, height: number): boolean {
    if (!this.#restirPtReuse) return false;
    const px = Math.max(1, width) * Math.max(1, height);
    const reservoirBytes = px * GpuResources.RESERVOIR_PT_HERO_BYTES;
    const resultBytes = px * 16;
    if (reservoirBytes > GpuResources.RESTIR_PT_RESERVOIR_MAX_BYTES) {
      // H14-F: once-gate — warn only on the first frame that hits the ceiling.
      if (!this.#ceilingWarnedKeys.has('restirPtReservoir')) {
        this.#ceilingWarnedKeys.add('restirPtReservoir');
        const mib = (reservoirBytes / (1024 * 1024)).toFixed(1);
        console.warn(
          `[vitrum/pt-webgpu] ReSTIR-PT reservoir buffer would be ${mib} MiB ` +
            `(${width}×${height} × 144 B), exceeding the ` +
            `${(GpuResources.RESTIR_PT_RESERVOIR_MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB ceiling. ` +
            'Skipping ReSTIR-PT reuse this frame — lower resolutionFactor or tile. ' +
            '(This warning fires once per engine instance.)',
        );
      }
      return false;
    }
    const ready =
      this.rptReservoirCur != null &&
      this.rptReservoirByteSize === reservoirBytes &&
      this.rptResultByteSize === resultBytes;
    if (ready) return true;

    this.rptReservoirCur?.destroy();
    this.rptReservoirPrev?.destroy();
    this.rptReservoirSpatial?.destroy();
    this.rptResultBuffer?.destroy();
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.rptReservoirCur = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.reservoir.cur',
      size: reservoirBytes,
      usage,
    });
    this.rptReservoirPrev = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.reservoir.prev',
      size: reservoirBytes,
      usage,
    });
    this.rptReservoirSpatial = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.reservoir.spatial',
      size: reservoirBytes,
      usage,
    });
    this.rptResultBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.result',
      size: resultBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    if (this.rptParamsBuffer == null) {
      this.rptParamsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.restirPt.params',
        size: GpuResources.RESTIR_PT_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.rptReservoirByteSize = reservoirBytes;
    this.rptResultByteSize = resultBytes;
    // Zero the ping-pong so frame 0's temporal history is an empty reservoir.
    const enc = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.restirPt.clear' });
    enc.clearBuffer(this.rptReservoirCur);
    enc.clearBuffer(this.rptReservoirPrev);
    enc.clearBuffer(this.rptReservoirSpatial);
    this.#device.queue.submit([enc.finish()]);
    // New buffers → the cached reuse bind groups are stale.
    this.rptProducerGroup0 = null;
    this.rptTemporalGroup0 = null;
    this.rptSpatialGroup0 = null;
    this.rptResolveGroup0 = null;
    return true;
  }

  /**
   * Write the RestirPtParams UBO (width/height/mClamp + wCap). No-op when reuse is
   * OFF or the buffer is absent. Called per-frame by the engine before dispatch.
   */
  writeReservoirParams(width: number, height: number, mClamp: number, wCap: number): void {
    if (!this.#restirPtReuse || this.rptParamsBuffer == null) return;
    const ubo = new ArrayBuffer(GpuResources.RESTIR_PT_PARAMS_BYTES);
    const u = new Uint32Array(ubo);
    const f = new Float32Array(ubo);
    u[0] = width >>> 0;
    u[1] = height >>> 0;
    u[2] = Math.max(1, Math.floor(mClamp)) >>> 0;
    u[3] = 0; // _padA
    f[4] = wCap;
    f[5] = 0; f[6] = 0; f[7] = 0; // _padB/_padC/_padD
    this.#device.queue.writeBuffer(this.rptParamsBuffer, 0, ubo);
  }

  /**
   * Build the extended group-0 bind-group layout for the reuse passes: the
   * megakernel's full-tier group-0 bindings (0..13, IDENTICAL to #buildShared-
   * PipelineLayout's group0) PLUS the relocated reuse bindings (20..24). This is
   * the ONE group the reuse passes carry their own resources in; groups 1/2/3 are
   * the megakernel's existing explicit layouts (reused verbatim), so a reuse
   * pipeline's layout is [g0', g1, g2, g3] — exactly 4 groups, portable on a
   * guaranteed maxBindGroups = 4 adapter. Full tier only (the reuse passes
   * statically use the full-tier scene groups).
   */
  #buildReservoirGroup0Layout(): GPUBindGroupLayout {
    const VIS = GPUShaderStage.COMPUTE;
    const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
    const rw: GPUBufferBindingLayout = { type: 'storage' };
    const uniform: GPUBufferBindingLayout = { type: 'uniform' };
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
    const B = RPT_GROUP0_BINDING_BASE; // 20
    // Megakernel group-0 (0..13) — byte-for-byte the same as the trace layout's
    // group0 (full tier) so the same group-0 scene/G-buffer resources bind.
    const entries: GPUBindGroupLayoutEntry[] = [
      tex(0),
      buf(1, uniform),
      buf(2, rw),
      buf(3, ro),
      buf(4, ro),
      buf(5, ro),
      buf(6, ro),
      buf(7, ro),
      buf(8, ro),
      tex(9),
      tex(10),
      tex(11),
      tex(12),
      buf(13, rw),
      // Relocated reuse bindings (20..24). The composed WGSL keeps rpt_resPrev /
      // rpt_resResolved as `var<storage, read>` (the naga-gap fix is to MONOMORPHISE
      // the reservoir helpers so the read-only global is indexed DIRECTLY rather
      // than passed as a storage pointer — see restirPtCompose.wgsl.ts
      // monomorphiseReservoirHelpers; the access modes are unchanged). So the layout
      // matches the shader access modes exactly: rpt_resPrev (b22) is read-only.
      buf(B + 0, rw), // rpt_reservoirOut  (producer write)
      buf(B + 1, rw), // rpt_resCurrent    (temporal in/out; spatial reads same slot as `read`)
      buf(B + 2, ro), // rpt_resPrev       (temporal history, read-only)
      buf(B + 3, rw), // rpt_result        (resolve write)
      buf(B + 4, uniform), // rptParams
      buf(B + 5, rw), // rpt_resSpatial    (spatial write; resolve reads same slot as `read`)
    ];
    return this.#device.createBindGroupLayout({
      label: 'vitrum.pt-webgpu.restirPt.layout.group0',
      entries,
    });
  }

  /**
   * Lazily build the three reuse compute pipelines + their shared 4-group layout
   * [g0', g1, g2, g3]. Requires `ensurePipeline()` to have run first (so the
   * megakernel's group-1/2/3 explicit layouts exist — the reuse passes reuse them
   * for the scene/TLAS/light bindings their trace + NEE statically use). No-op
   * when reuse is OFF, on the lite tier, or already built. Each entry point gets
   * its own module (the combined unit has duplicate @group/@binding slots — see
   * restirPtCompose.wgsl.ts), composed standalone with the reuse bindings
   * relocated into group 0.
   */
  ensureReservoirPipelines(): void {
    if (!this.#restirPtReuse) return;
    if (this.rptProducerPipeline != null) return;
    if (
      this.bindGroupLayout1 == null ||
      this.bindGroupLayout2 == null ||
      this.bindGroupLayout3 == null
    ) {
      // ensurePipeline() must build the full-tier group-1/2/3 layouts first.
      throw new Error(
        'ensureReservoirPipelines: full-tier group-1/2/3 layouts missing — call ensurePipeline() first.',
      );
    }
    this.#rptGroup0Layout = this.#buildReservoirGroup0Layout();
    const pipelineLayout = this.#device.createPipelineLayout({
      label: 'vitrum.pt-webgpu.restirPt.pipelineLayout',
      bindGroupLayouts: [
        this.#rptGroup0Layout,
        this.bindGroupLayout1,
        this.bindGroupLayout2,
        this.bindGroupLayout3,
      ],
    });
    const mk = (label: string, code: string, entryPoint: string): GPUComputePipeline => {
      const module = this.#device.createShaderModule({ label, code });
      return this.#device.createComputePipeline({
        label,
        layout: pipelineLayout,
        compute: { module, entryPoint },
      });
    };
    this.rptProducerPipeline = mk(
      'vitrum.pt-webgpu.restirPt.producer',
      composeRestirPtProducerWgsl(),
      'restirPtProduce',
    );
    this.rptTemporalPipeline = mk(
      'vitrum.pt-webgpu.restirPt.temporal',
      composeRestirPtTemporalWgsl(),
      'restirPtTemporal',
    );
    this.rptSpatialPipeline = mk(
      'vitrum.pt-webgpu.restirPt.spatial',
      composeRestirPtSpatialWgsl(),
      'restirPtSpatial',
    );
    this.rptResolvePipeline = mk(
      'vitrum.pt-webgpu.restirPt.resolve',
      composeRestirPtResolveWgsl(),
      'restirPtResolve',
    );
    // A1 — the COMPOSITE megakernel uses the SAME [g0', g1, g2, g3] layout (it reads
    // rpt_result at the relocated group-0 binding 23 + the scene groups). Composed
    // for this engine's BDPT mode (matches the default megakernel's SSS/BDPT gate).
    this.rptCompositePipeline = mk(
      'vitrum.pt-webgpu.restirPt.compositeMegakernel',
      composePtWebgpuCompositeTraceWgsl(this.#bdpt),
      'main',
    );
  }

  /**
   * Build (and cache) the per-pass reuse group-0 bind groups. Each provides the
   * megakernel group-0 scene/G-buffer resources (IDENTICAL to the trace group 0)
   * PLUS the reuse bindings (20..24). The producer reads `Cur` as its OUTPUT slot
   * (binding 21 is the "current" reservoir in every pass by binding number); the
   * temporal pass reads `Cur`(21)+`Prev`(22); the resolve pass reads `Cur`(21) +
   * writes `result`(23). All three also bind 20/23/24 even if a given pass does
   * not declare them (extra layout entries are legal; the layout is uniform), so
   * one group-0 bind-group construction serves all three with the SAME resources
   * except the producer's binding-20 write target. Returns nothing; the engine
   * reads the cached groups off this struct. Idempotent.
   *
   * NOTE on binding 20 vs 21: the producer writes `rpt_reservoirOut` at b20 and
   * the temporal/resolve read `rpt_resCurrent`/`rpt_resResolved` at b21 — these
   * are the SAME logical "current" reservoir. We bind `rptReservoirCur` to BOTH
   * b20 and b21 so the producer's output IS the temporal's input (one buffer),
   * and `rptReservoirPrev` to b22.
   */
  buildReservoirBindGroups(sb: UploadedSceneBuffers): void {
    if (!this.#restirPtReuse || this.#rptGroup0Layout == null) return;
    if (this.rptProducerGroup0 != null) return;
    if (
      this.rptReservoirCur == null ||
      this.rptReservoirPrev == null ||
      this.rptReservoirSpatial == null ||
      this.rptResultBuffer == null ||
      this.rptParamsBuffer == null ||
      this.accumView == null ||
      this.normalDepthView == null ||
      this.albedoView == null ||
      this.varianceView == null ||
      this.motionVectorsView == null ||
      this.varianceMomentsBuffer == null ||
      this.paramsBuffer == null
    ) {
      return;
    }
    const B = RPT_GROUP0_BINDING_BASE;
    // Shared megakernel group-0 scene/G-buffer entries (0..13).
    const sceneG0: GPUBindGroupEntry[] = [
      { binding: 0, resource: this.accumView },
      { binding: 1, resource: { buffer: this.paramsBuffer } },
      { binding: 2, resource: { buffer: this.accumBuffer! } },
      { binding: 3, resource: { buffer: sb.positionsBuffer } },
      { binding: 4, resource: { buffer: sb.indicesBuffer } },
      { binding: 5, resource: { buffer: sb.triMaterialIdsBuffer } },
      { binding: 6, resource: { buffer: sb.materialsBuffer } },
      { binding: 7, resource: { buffer: sb.bvhNodesBuffer } },
      { binding: 8, resource: { buffer: sb.normalsBuffer } },
      { binding: 9, resource: this.normalDepthView },
      { binding: 10, resource: this.albedoView },
      { binding: 11, resource: this.varianceView },
      { binding: 12, resource: this.motionVectorsView },
      { binding: 13, resource: { buffer: this.varianceMomentsBuffer } },
      { binding: B + 4, resource: { buffer: this.rptParamsBuffer } },
    ];
    // The reuse-reservoir slots differ only in which buffer is the "current"
    // ping-pong half. All three passes share ONE bind group: b20 = b21 = Cur (the
    // producer writes b20, temporal/resolve read b21 — same buffer), b22 = Prev,
    // b23 = result.
    const group0 = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.restirPt.bindgroup0',
      layout: this.#rptGroup0Layout,
      entries: [
        ...sceneG0,
        { binding: B + 0, resource: { buffer: this.rptReservoirCur } }, // rpt_reservoirOut → Cur
        { binding: B + 1, resource: { buffer: this.rptReservoirCur } }, // rpt_resCurrent  → Cur
        { binding: B + 2, resource: { buffer: this.rptReservoirPrev } }, // rpt_resPrev    → Prev
        { binding: B + 3, resource: { buffer: this.rptResultBuffer } }, // rpt_result      → result
        { binding: B + 5, resource: { buffer: this.rptReservoirSpatial } }, // rpt_resSpatial → Spatial
      ],
    });
    // Same resources for all four passes (the layout + bindings are uniform).
    this.rptProducerGroup0 = group0;
    this.rptTemporalGroup0 = group0;
    this.rptSpatialGroup0 = group0;
    this.rptResolveGroup0 = group0;
  }

  /**
   * Ping-pong the reservoir buffers: this frame's RESOLVED reservoir is the SPATIAL
   * pass output (producer→Cur, temporal Prev→Cur, spatial Cur→Spatial, resolve reads
   * Spatial). For the temporal feedback loop to carry the spatially-improved estimate
   * forward, next frame's `Prev` history must be THIS frame's `Spatial` — so we swap
   * Prev↔Spatial (the old Prev buffer becomes the new Spatial scratch). `Cur` is
   * producer-overwritten every frame, so it does NOT rotate. Invalidates the cached
   * reuse bind groups (they reference the now-swapped buffers). No-op when reuse OFF.
   */
  swapReservoirs(): void {
    if (!this.#restirPtReuse) return;
    const tmp = this.rptReservoirPrev;
    this.rptReservoirPrev = this.rptReservoirSpatial;
    this.rptReservoirSpatial = tmp;
    this.rptProducerGroup0 = null;
    this.rptTemporalGroup0 = null;
    this.rptSpatialGroup0 = null;
    this.rptResolveGroup0 = null;
  }

  /** Tear down all ReSTIR-PT reuse resources. Called from dispose(). */
  #disposeReservoirResources(): void {
    this.rptReservoirCur?.destroy();
    this.rptReservoirCur = null;
    this.rptReservoirPrev?.destroy();
    this.rptReservoirPrev = null;
    this.rptReservoirSpatial?.destroy();
    this.rptReservoirSpatial = null;
    this.rptResultBuffer?.destroy();
    this.rptResultBuffer = null;
    this.rptParamsBuffer?.destroy();
    this.rptParamsBuffer = null;
    this.rptReservoirByteSize = 0;
    this.rptResultByteSize = 0;
    this.rptProducerPipeline = null; // GPUComputePipeline has no destroy()
    this.rptTemporalPipeline = null;
    this.rptSpatialPipeline = null;
    this.rptResolvePipeline = null;
    this.rptCompositePipeline = null;
    this.#rptGroup0Layout = null;
    this.rptProducerGroup0 = null;
    this.rptTemporalGroup0 = null;
    this.rptSpatialGroup0 = null;
    this.rptResolveGroup0 = null;
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
        entries: [
          { binding: 0, resource: { buffer: sb.lightTreeBuffer } },
          { binding: 1, resource: { buffer: sb.uvsBuffer } },
          { binding: 2, resource: { buffer: sb.materialTexDescriptorsBuffer } },
          { binding: 3, resource: sb.materialTextureView },
          { binding: 4, resource: sb.materialTextureSampler },
          { binding: 5, resource: sb.materialLinearTextureView },
        ],
      });
    }
    return bindGroup;
  }

  /**
   * H9 — Reconstruct ONLY bind group 2 (TLAS table + BDPT light-path/eye-stack)
   * while leaving groups 0, 1, and 3 intact.  Called by `bdptAdvanceFrame` when
   * the host supplies a new external light-path buffer for the NEXT frame; the
   * full-group rebuild in `buildBindGroups` is NOT triggered because group 0 is
   * still cached (and returning it early is the correct fast path for all other
   * frames).
   *
   * Fast-out: if `lightPathBuffer` is the same reference that was used to build
   * the currently-cached group 2, the group is left in place (pointer equality
   * suffices because GPUBuffer identity is stable for the same host allocation).
   *
   * Preconditions (enforced by the `bdptAdvanceFrame` caller):
   *  - full tier only (`this.#traceTier === 'full'`)
   *  - `bindGroupLayout2` non-null (ensurePipeline must have run)
   *  - `sb` non-null (a scene has been set)
   *  - `bdptEyeStackBuffer` non-null (ensureBdptEyeStack must have run)
   */
  rebuildGroup2Only(sb: UploadedSceneBuffers, lightPathBuffer: GPUBuffer): void {
    if (this.#traceTier !== 'full' || this.bindGroupLayout2 == null) return;
    // Pointer-equality fast-out: if the buffer didn't change, the cached group
    // is still valid — avoid a redundant createBindGroup call.
    if (this.#lastBdptLightPathBuffer === lightPathBuffer && this.pathTraceBindGroup2 != null) {
      return;
    }
    this.#lastBdptLightPathBuffer = lightPathBuffer;
    this.pathTraceBindGroup2 = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.pathTrace.bindgroup2.full.bdptRebuild',
      layout: this.bindGroupLayout2,
      entries: [
        { binding: 0, resource: { buffer: sb.tlasNodesBuffer } },
        { binding: 1, resource: { buffer: sb.tlasInstanceIndicesBuffer } },
        { binding: 2, resource: { buffer: sb.tlasBlasRootsBuffer } },
        { binding: 3, resource: { buffer: sb.tlasInstanceWorldToLocalBuffer } },
        { binding: 4, resource: { buffer: sb.tlasInstanceLocalToWorldBuffer } },
        { binding: 5, resource: { buffer: lightPathBuffer } },
        { binding: 6, resource: { buffer: this.bdptEyeStackBuffer! } },
      ],
    });
  }
  /** The light-path buffer reference used to build the most-recent group 2.
   *  Enables the pointer-equality fast-out in `rebuildGroup2Only`. */
  #lastBdptLightPathBuffer: GPUBuffer | null = null;

  /**
   * Progressive walkaround→PT handoff (P8) — seed the accumulation buffers from
   * `seedTex` as a DECAYING PRIOR of virtual weight `weight`. Writes
   * `accumBuffer[i] = vec4f(seedRGB·W, W)` and
   * `varianceMomentsBuffer[i] = vec3(lum·W, lum²·W, W)` (full tier; lite tier
   * discards the variance write to a placeholder). The converged mean is
   * UNCHANGED because the seed's influence is W/(W+M) → 0 as M real samples land
   * (see seedBlit.wgsl.ts header for the derivation).
   *
   * MUST be called AFTER `ensureAccumResources` (so the accum buffers exist) and
   * AFTER `clearAccumBuffer`/`reset` (so the seed isn't subsequently zeroed). A
   * no-op if the accum buffer is absent. `width`/`height` are the accum
   * (destination) dims; `seedTex` may be any size (bilinearly resampled).
   *
   * Does NOT touch the engine's `#samplesAccumulated`: `weight` is a
   * virtual-sample prior, distinct from the real-SPP counter — the converged-mean
   * math depends on that separation (the caller in `index.ts` enforces it).
   */
  seedAccumBuffer(seedTex: GPUTexture, weight: number, width: number, height: number): void {
    if (this.accumBuffer == null) return;
    const W = Math.max(0, weight);

    // Lazily build the seed-blit pipeline + sampler + params UBO (engine-owned).
    if (this.#seedBlitPipeline == null) {
      const module = this.#device.createShaderModule({
        label: 'vitrum.pt-webgpu.seedBlit',
        code: PT_WEBGPU_SEED_BLIT_WGSL,
      });
      this.#seedBlitPipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.seedBlit.pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    if (this.#seedBlitSampler == null) {
      // Filtering sampler so a differently-sized seed is bilinearly resampled
      // onto the accum grid; clamp so edge UVs don't wrap.
      this.#seedBlitSampler = this.#device.createSampler({
        label: 'vitrum.pt-webgpu.seedBlit.sampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
    }
    if (this.#seedBlitParamsBuffer == null) {
      this.#seedBlitParamsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.seedBlit.params',
        size: 32, // vec4u seedDim (16) + vec4f seedWeight (16)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    // SeedParams UBO: seedDim (accum dims) as uvec4, seedWeight as vec4f.
    const ubo = new ArrayBuffer(32);
    new Uint32Array(ubo, 0, 4).set([width >>> 0, height >>> 0, 0, 0]);
    new Float32Array(ubo, 16, 4).set([W, 0, 0, 0]);
    this.#device.queue.writeBuffer(this.#seedBlitParamsBuffer, 0, ubo);

    // varianceMoments slot: the real buffer on the full tier; a discardable
    // placeholder on the lite tier (which has none) so the layout stays valid.
    let varBuffer = this.varianceMomentsBuffer;
    if (varBuffer == null) {
      if (this.#seedBlitVarPlaceholder == null) {
        this.#seedBlitVarPlaceholder = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.seedBlit.varPlaceholder',
          size: this.accumBufferByteSize > 0 ? this.accumBufferByteSize : 16,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
      varBuffer = this.#seedBlitVarPlaceholder;
    }

    const bindGroup = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.seedBlit.bindgroup',
      layout: this.#seedBlitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#seedBlitParamsBuffer } },
        { binding: 1, resource: seedTex.createView() },
        { binding: 2, resource: this.#seedBlitSampler },
        { binding: 3, resource: { buffer: this.accumBuffer } },
        { binding: 4, resource: { buffer: varBuffer } },
      ],
    });

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.seedBlit.encoder' });
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.seedBlit.pass' });
    pass.setPipeline(this.#seedBlitPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
    pass.end();
    this.#device.queue.submit([encoder.finish()]);
  }

  /** Invalidate the cached bind groups (scene-buffer / accum-view recreation). */
  invalidateBindGroups(): void {
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
    this.pathTraceBindGroup3 = null;
    // A full invalidation also clears the fast-out reference so the next
    // rebuildGroup2Only call unconditionally rebuilds against fresh scene buffers.
    this.#lastBdptLightPathBuffer = null;
    // The reuse bind groups reference the same scene buffers + accum views, so a
    // scene-buffer / accum-view recreation invalidates them too.
    this.rptProducerGroup0 = null;
    this.rptTemporalGroup0 = null;
    this.rptSpatialGroup0 = null;
    this.rptResolveGroup0 = null;
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
    this.#seedBlitParamsBuffer?.destroy();
    this.#seedBlitParamsBuffer = null;
    this.#seedBlitVarPlaceholder?.destroy();
    this.#seedBlitVarPlaceholder = null;
    this.#seedBlitSampler = null; // GPUSampler has no destroy(); drop the ref
    this.#seedBlitPipeline = null; // GPUComputePipeline has no destroy(); drop the ref
    this.paramsBuffer?.destroy();
    this.paramsBuffer = null;
    this.computePipeline = null;
    this.bdptSubpathPipeline = null;
    this.bindGroupLayout = null;
    this.bindGroupLayout1 = null;
    this.bindGroupLayout2 = null;
    this.bindGroupLayout3 = null;
    this.#disposeReservoirResources();
  }
}
