/**
 * WalkaroundGPUPipeline — manages all WebGPU resources + compute passes for
 * the ReSTIR DI/GI pipeline.
 *
 * Uses the fully-manual `device.createShaderModule()` path (§10.7) since
 * wgslFn composition with three's TSL compute() is unvalidated. Web-RTRT
 * confirms this approach works in browser. Exposes a simple `renderFrame()`
 * method that the WalkaroundStage calls per-frame.
 *
 * Pipeline shape per frame:
 *   1. RIS: primary-ray-cast primary visibility + initial candidate sampling
 *   2. Temporal reuse: merge with previous-frame reservoir
 *   3. Spatial reuse (2 separable passes)
 *   4. Shade + GI: compute DI + one indirect bounce, write HDR color
 *   5. À-trous denoiser (5 iterations)
 *   6. Composite render pass: blit denoised HDR to the swap-chain texture
 *
 * Note: we use primary-ray-casting mode (§10.7 fallback) instead of a G-buffer
 * raster pass. The G-buffer bind group slots are filled with 1×1 placeholder
 * textures for layout compatibility; the RIS + shade passes generate their own
 * primary visibility by casting rays through the BVH.
 */

import type { SceneBVHBuffers } from './bvhCompute';
import { COMMON_WGSL } from './shaders/common.wgsl';
import { RIS_WGSL } from './shaders/ris.wgsl';
import { TEMPORAL_WGSL } from './shaders/temporal.wgsl';
import { SPATIAL_WGSL } from './shaders/spatial.wgsl';
import { SHADE_WGSL } from './shaders/shade.wgsl';
import { ATROUS_WGSL } from './shaders/atrous.wgsl';
import { TEMPORAL_ACCUM_WGSL } from './shaders/temporalAccum.wgsl';
import { COMPOSITE_VERT_WGSL, COMPOSITE_FRAG_WGSL } from './shaders/composite.wgsl';

/**
 * WebGPU device limits required by the layered-hybrid shade pipeline.
 *
 * The ReSTIR shade pass binds 13 storage buffers in the live path:
 *   - 5 frame buffers (current + previous + spatial reservoirs + 2 G-buffer placeholders)
 *   - 5 RC cascade buffers (one per cascade level)
 *   - 3 BVH buffers (nodes / index / position)
 *
 * WebGPU's default `maxStorageBuffersPerShaderStage` is 8. Library
 * consumers must pass these `requiredLimits` when calling
 * `navigator.gpu.requestAdapter().requestDevice({...})` or via the
 * three.js WebGPURenderer constructor's `requiredLimits` option.
 *
 * Caller pattern (from StudioScene.tsx gl factory):
 *   const renderer = new WebGPURenderer({
 *     ...,
 *     requiredLimits: HYBRID_WEBGPU_REQUIRED_LIMITS,
 *   });
 */
export const HYBRID_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: 16,
};

export interface PipelineFrameInputs {
  /** Camera view matrix (column-major mat4x4f, 16 floats). The pipeline
   *  composes VP = projMatrix * viewMatrix internally; do NOT pre-multiply. */
  viewMatrix: Float32Array;
  /** Camera projection matrix (column-major mat4x4f, 16 floats). */
  projMatrix: Float32Array;
  /** Previous-frame view matrix — drives temporal reservoir reuse. Pass
   *  the same matrix as viewMatrix on the first frame to avoid a one-frame
   *  ghost from uninitialized previous-frame state. */
  prevViewMatrix: Float32Array;
  /** Previous-frame projection matrix; same first-frame note as prevViewMatrix. */
  prevProjMatrix: Float32Array;
  /** World-space camera position [x, y, z]. */
  cameraPos: [number, number, number];
  /** Render-target dimensions in pixels. Used by all compute kernels for
   *  workgroup dispatch sizing — must match the swap chain's actual size. */
  screenWidth: number;
  screenHeight: number;
  /** u32 frame counter / per-frame randomness seed. Drives PCG hash inits
   *  for ray jitter, RIS candidate sampling, and temporal reservoir update.
   *  Caller may use a frame index, performance.now()|0, or any monotone u32. */
  frameSeed: number;
  /** Sum of (Le * area) over all emitter triangles, computed at BVH build
   *  time. Used by RIS importance-sampling weight normalization. Must match
   *  the value baked into the emitter CDF in SceneBVHBuffers. */
  totalEmissivePower: number;
  /** Number of entries in the emitter list (length of EmitterTri[] array
   *  in SceneBVHBuffers.emitters). Used by RIS to bound candidate selection. */
  emitterCount: number;
  /** Primary directional light direction [x, y, z] in world space, normalized.
   *  Today this is the sun for cathedral-window glass tracing; the field is
   *  named generically because the path tracer is light-source-agnostic. */
  primaryLightDir: [number, number, number];
  /** Primary directional light irradiance multiplier (linear, unitless).
   *  Must match the value passed to buildSceneBVH({primaryLightIntensity})
   *  at BVH-build time so the shader's self-emission Le for primary panel
   *  hits reproduces exactly the Le baked into the emitter list. */
  primaryLightIntensity: number;
  /** Diffuse-sky-dome RGB tint, derived from computeLightingState. Replaces
   *  four formerly-hardcoded sky tints in WGSL. Consumed by sky-aperture
   *  probe + second-bounce sky-miss paths. */
  skyTint: [number, number, number];
  /** Sky-dome irradiance scalar paired with skyTint. ~0.5×sun at noon. */
  skyIrradiance: number;
  /** The WebGPU swap-chain texture view to render into for this frame.
   *  Caller must obtain via context.getCurrentTexture().createView()
   *  inside the same animation-frame callback that calls renderFrame. */
  swapChainView: GPUTextureView;
  /** The format of swapChainView. The composite pass's render-pipeline
   *  is recompiled if this changes (rare — usually fixed at canvas mount). */
  swapChainFormat: GPUTextureFormat;
}

export class WalkaroundGPUPipeline {
  private device: GPUDevice;
  private width: number;
  private height: number;

  // Storage buffers (BVH + emitters — static across frames)
  // Note: material colors are packed into bvhIndex[*].w (vec4u), not separate buffers.
  private bvhNodesBuffer!: GPUBuffer;
  private bvhIndexBuffer!: GPUBuffer;
  private bvhBeerBuffer!: GPUBuffer;
  private bvhPositionBuffer!: GPUBuffer;
  private bvhNormalBuffer!: GPUBuffer;
  private bvhUvBuffer!: GPUBuffer;
  private emitterBuffer!: GPUBuffer;
  private emitterCdfBuffer!: GPUBuffer;

  // Per-frame storage buffers
  private reservoirCurrentBuffer!: GPUBuffer;
  private reservoirPreviousBuffer!: GPUBuffer;
  private reservoirSpatialBuffer!: GPUBuffer;
  private hdrColorTexture!: GPUTexture;
  // Real G-buffer authored by the shade pass alongside hdrColorTexture.
  // rgba16float: .xyz = encoded primary-hit world normal (n*0.5+0.5),
  //              .w   = primary-hit distance (depth along ray).
  // Consumed by the à-trous denoiser for normal/depth edge stopping.
  // (Separate from `placeholderTexture` which is bound to G-buffer slots in
  //  the compute frame bind group for layout compatibility — those slots are
  //  not actually read by RIS/temporal/spatial/shade in primary-ray-cast mode.)
  private gNormalDepthTexture!: GPUTexture;
  private denoisedPingTexture!: GPUTexture;
  private denoisedPongTexture!: GPUTexture;
  // Temporal accumulator (ping-pong). Stores 16f HDR across frames so
  // we can blend current_atrous_output with last frame's accumulated
  // HDR — drives the per-pixel noise floor down by ~10× over a few
  // frames of static camera. Camera-motion detection resets via α=1.
  private accumTextureA!: GPUTexture;
  private accumTextureB!: GPUTexture;
  private accumPingPongIndex = 0;       // 0 = read A, write B; 1 = swap
  private accumFrameIndex = 0;
  private lastCameraPos: [number, number, number] = [0, 0, 0];
  private accumPipeline!: GPUComputePipeline;
  private _accumBGL!: GPUBindGroupLayout;
  private _accumUboBuffer!: GPUBuffer;
  private placeholderTexture!: GPUTexture;  // 1×1 placeholder for G-buffer slots

  // DDGI inputs (layered hybrid). Default to placeholder textures so the
  // shade pipeline always validates even when HybridLayeredStage hasn't
  // wired its DDGI atlas through setDDGIInputs() yet. shade.wgsl consumes
  // them via ddgiSampleFromBindings() gated by isDDGIWired() (returns
  // false when the placeholder UBO is bound, true once real grid params
  // are written by setDDGIInputs).
  private _ddgiIrrTex: GPUTexture | null = null;       // user-provided atlas (sampled)
  private _ddgiVisTex: GPUTexture | null = null;
  private _ddgiPlaceholderRgba16f!: GPUTexture;        // 1×1 fallback for irradiance
  private _ddgiPlaceholderRg16f!: GPUTexture;          // 1×1 fallback for visibility
  private _ddgiUboBuffer!: GPUBuffer;                  // 64-byte grid params UBO

  // Uniform buffer
  private uboBuffer!: GPUBuffer;

  // Compute pipelines
  private risPipeline!: GPUComputePipeline;
  private temporalPipeline!: GPUComputePipeline;
  private spatialPipeline!: GPUComputePipeline;
  private shadePipeline!: GPUComputePipeline;
  private atrousPipeline!: GPUComputePipeline;

  // Composite render pipeline
  private compositePipeline!: GPURenderPipeline;
  private compositeLinearSampler!: GPUSampler;

  // Samplers
  private nearestSampler!: GPUSampler;

  private frameCount = 0;
  private initialized = false;
  private _swapChainFormat: GPUTextureFormat = 'bgra8unorm';

  // ── GPU timestamp queries (DEV-only, feature-gated) ─────────────────
  // 10 passes per frame: ris, temporal, spatial-1, spatial-2, shade,
  // atrous-0..2 (3 iterations, stepWidths 1/2/4), temporalAccum, composite.
  // Atrous count history: 5→7→5→3→2→3. The final 2→3 bump is for
  // collective edge softening — 2 iters was too sharp/aliased.
  private static readonly PASS_LABELS = [
    'ris', 'temporal', 'spatial-1', 'spatial-2', 'shade',
    'atrous-0', 'atrous-1', 'atrous-2',
    'temporalAccum', 'composite',
  ] as const;
  private static readonly PASS_COUNT = WalkaroundGPUPipeline.PASS_LABELS.length;
  private timestampQuerySet: GPUQuerySet | null = null;
  private timestampResolveBuffer: GPUBuffer | null = null;
  // Ping-pong of two readback buffers — one in flight (mapped/mapping),
  // one being written this frame. Avoids the "buffer in use" stall that
  // would happen if a single readback buffer hadn't completed mapping
  // before the next frame's resolve+copy targets it.
  private timestampReadbackA: GPUBuffer | null = null;
  private timestampReadbackB: GPUBuffer | null = null;
  private timestampReadbackInFlight: 'A' | 'B' | null = null;
  private timestampPeriodNs: number = 1.0;
  /** Last successfully-read timestamp values, ms per pass. */
  public lastGpuTimings: Record<string, number> = {};
  /** Frame index of the last completed timestamp read — lets the bridge
   *  show "fresh" vs "stale" data. */
  public lastGpuTimingsFrame: number = -1;

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device;
    this.width  = width;
    this.height = height;
  }

  /** Upload BVH data + compile shaders. Must be called once before renderFrame. */
  async initialize(bvhBuffers: SceneBVHBuffers, swapChainFormat: GPUTextureFormat = 'bgra8unorm'): Promise<void> {
    const d = this.device;
    const { width: W, height: H } = this;
    this._swapChainFormat = swapChainFormat;

    // ── Upload BVH buffers ────────────────────────────────────────────────
    this.bvhNodesBuffer = this.uploadBuffer(
      bvhBuffers.bvhNodes.cpuData, GPUBufferUsage.STORAGE,
    );
    this.bvhIndexBuffer = this.uploadBuffer(
      bvhBuffers.bvhIndex.cpuData, GPUBufferUsage.STORAGE,
    );
    this.bvhBeerBuffer = this.uploadBuffer(
      bvhBuffers.bvhBeerColors.cpuData, GPUBufferUsage.STORAGE,
    );
    this.bvhPositionBuffer = this.uploadBuffer(
      bvhBuffers.bvhPositions.cpuData, GPUBufferUsage.STORAGE,
    );
    this.bvhNormalBuffer = this.uploadBuffer(
      bvhBuffers.bvhNormals.cpuData, GPUBufferUsage.STORAGE,
    );
    this.bvhUvBuffer = this.uploadBuffer(
      bvhBuffers.bvhUvs.cpuData, GPUBufferUsage.STORAGE,
    );
    this.emitterBuffer = this.uploadBuffer(
      bvhBuffers.emitters.cpuData, GPUBufferUsage.STORAGE,
    );
    this.emitterCdfBuffer = this.uploadBuffer(
      bvhBuffers.emitterCdf.cpuData, GPUBufferUsage.STORAGE,
    );
    // triangleMatIds and materialColors are packed into bvhIndex[*].w — no separate GPU buffers.

    // ── Per-frame buffers ─────────────────────────────────────────────────
    // Reservoir DI: 16 bytes/pixel (4 × u32)
    const RESERVOIR_STRIDE = 16;
    const totalReservoirBytes = Math.max(W * H * RESERVOIR_STRIDE, 256);

    this.reservoirCurrentBuffer = d.createBuffer({
      size: totalReservoirBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.reservoirPreviousBuffer = d.createBuffer({
      size: totalReservoirBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.reservoirSpatialBuffer = d.createBuffer({
      size: totalReservoirBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // HDR color output (rgba16float — written by shade, read by atrous).
    // COPY_SRC enables GPU pixel readback for the caustic validation harness.
    this.hdrColorTexture = d.createTexture({
      size: [W, H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    // G-buffer (normal + depth) — written by shade, read by atrous denoiser.
    this.gNormalDepthTexture = d.createTexture({
      size: [W, H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Ping-pong denoised textures.
    // COPY_SRC enables GPU pixel readback for the caustic validation harness.
    this.denoisedPingTexture = d.createTexture({
      size: [W, H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.denoisedPongTexture = d.createTexture({
      size: [W, H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    // Temporal accumulator ping-pong (rgba16float). Read prev / write
    // current within a single dispatch — must be separate textures.
    this.accumTextureA = d.createTexture({
      size: [W, H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.accumTextureB = d.createTexture({
      size: [W, H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    // 1×1 placeholder texture for G-buffer bind group slots.
    this.placeholderTexture = d.createTexture({
      size: [1, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // Fill placeholder with a valid forward-facing normal so the atrous denoiser
    // does not produce NaN. The atrous shader decodes normal as: n = raw * 2 - 1.
    // For a forward-facing normal (0,0,1): raw = (0.5, 0.5, 1.0, 0.0).
    // Using (0,0,0) for the zero-depth (sky) placeholder causes dot(n,n) = 3 →
    // pow(3, sigmaN=128) → Inf, and Inf/Inf = NaN propagation through the denoiser.
    const placeholderData = new Float32Array([0.5, 0.5, 1.0, 0.0]); // encodes normal=(0,0,1), depth=0
    d.queue.writeTexture({ texture: this.placeholderTexture }, placeholderData, { bytesPerRow: 16 }, [1, 1]);

    // UBO: camera matrices + per-frame params (256 bytes).
    this.uboBuffer = d.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.nearestSampler = d.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
    });
    this.compositeLinearSampler = d.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Phase 1.2A — DDGI placeholder textures + UBO. The shade pipeline's
    // 4th bind group always binds these, so the pipeline validates even
    // when no real DDGI atlas has been supplied via setDDGIInputs.
    this._ddgiPlaceholderRgba16f = d.createTexture({
      label: 'ddgi-placeholder-irr',
      size: [1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // Visibility placeholder must match the live atlas format
    // (probeUpdatePass._getOrCreateAtlasTexture creates rgba16float).
    // WebGPU's 'unfilterable-float' sample type tolerates either rg16float
    // or rgba16float at the bind-group slot today, but format consistency
    // is the safer invariant — bug fix isolated 2026-05-07 sweep (B5).
    this._ddgiPlaceholderRg16f = d.createTexture({
      label: 'ddgi-placeholder-vis',
      size: [1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this._ddgiUboBuffer = d.createBuffer({
      label: 'ddgi-ubo',
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Default DDGI uniform — origin (0,0,0), spacing 24, dims (1,1,1),
    // atlas dims 1×1. shade.wgsl gates DDGI consumption on isDDGIWired()
    // which checks dimsX > 1u; the placeholder writes dimsX=1 so the gate
    // returns false and Lo_ddgi=0 until setDDGIInputs() supplies real
    // grid params from HybridLayeredStage.
    const defaultDdgiUbo = new Float32Array(16);
    defaultDdgiUbo[3] = 24; // spacing
    new Uint32Array(defaultDdgiUbo.buffer)[4] = 1; // dimsX
    new Uint32Array(defaultDdgiUbo.buffer)[5] = 1; // dimsY
    new Uint32Array(defaultDdgiUbo.buffer)[6] = 1; // dimsZ
    defaultDdgiUbo[8]  = 1; // irrW
    defaultDdgiUbo[9]  = 1; // irrH
    defaultDdgiUbo[10] = 1; // visW
    defaultDdgiUbo[11] = 1; // visH
    d.queue.writeBuffer(this._ddgiUboBuffer, 0, defaultDdgiUbo.buffer);

    // ── Compile shaders ───────────────────────────────────────────────────
    await this.compilePipelines();

    // ── Timestamp queries (DEV-only, feature-gated) ──────────────────────
    if (import.meta.env.DEV && d.features.has('timestamp-query')) {
      const N = WalkaroundGPUPipeline.PASS_COUNT;
      this.timestampQuerySet = d.createQuerySet({
        type: 'timestamp',
        count: N * 2,
      });
      this.timestampResolveBuffer = d.createBuffer({
        size: N * 2 * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.timestampReadbackA = d.createBuffer({
        size: N * 2 * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.timestampReadbackB = d.createBuffer({
        size: N * 2 * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      // WebGPU exposes timestampPeriod via adapter info (ns per tick).
      // Three.js's renderer doesn't surface this directly; pull from
      // adapterInfo on the device if available, else default to 1.0
      // (some browsers normalize to 1ns by spec).
      const adapterInfo = (d as unknown as { adapterInfo?: { timestampPeriod?: number } }).adapterInfo;
      this.timestampPeriodNs = adapterInfo?.timestampPeriod ?? 1.0;
      console.log('[hybrid:debug] timestamp queries enabled',
        { passes: N, periodNs: this.timestampPeriodNs });
    } else if (import.meta.env.DEV) {
      console.log('[hybrid:debug] timestamp queries unavailable on this adapter; falling back to JS-submit timing only');
    }

    this.initialized = true;
    console.log('[ReSTIR] Pipeline initialized', { W, H, bvhNodes: bvhBuffers.bvhNodes.count, emitters: bvhBuffers.emitterCount });
  }

  /**
   * Build the optional timestampWrites struct for a pass at the given
   * pipeline-level pass index. Returns undefined when timestamp queries
   * aren't enabled, so the spread `...(this.tsWrites(...) && { timestampWrites: this.tsWrites(...) })`
   * trick doesn't degrade pass descriptors on adapters without the feature.
   */
  private tsWrites(passIndex: number): GPUComputePassTimestampWrites | undefined {
    if (!this.timestampQuerySet) return undefined;
    return {
      querySet: this.timestampQuerySet,
      beginningOfPassWriteIndex: passIndex * 2,
      endOfPassWriteIndex: passIndex * 2 + 1,
    };
  }

  /**
   * Read back the most recent timestamp results into `lastGpuTimings`.
   * Async; uses ping-pong to avoid stalling the next frame's submit.
   * Called from renderFrame; safe to invoke every frame because the
   * mapAsync is fire-and-forget (no await on the hot path).
   */
  private kickTimestampReadback(): void {
    if (!this.timestampResolveBuffer || !this.timestampReadbackA || !this.timestampReadbackB) return;
    if (this.timestampReadbackInFlight) return; // skip — prior readback still pending
    const target = this.frameCount % 2 === 0 ? this.timestampReadbackA : this.timestampReadbackB;
    const slot: 'A' | 'B' = this.frameCount % 2 === 0 ? 'A' : 'B';
    this.timestampReadbackInFlight = slot;
    const periodNs = this.timestampPeriodNs;
    const labels = WalkaroundGPUPipeline.PASS_LABELS;
    const N = labels.length;
    target.mapAsync(GPUMapMode.READ).then(() => {
      try {
        const range = target.getMappedRange();
        const view = new BigInt64Array(range.slice(0));
        target.unmap();
        const next: Record<string, number> = {};
        let total = 0;
        for (let i = 0; i < N; i++) {
          const begin = view[i * 2];
          const end = view[i * 2 + 1];
          // Defensive: monotonic clocks should never decrement, but at
          // boot the first frame's begin/end can be 0n. Skip those.
          if (end <= begin) continue;
          const ms = Number(end - begin) * periodNs / 1_000_000;
          next[labels[i]] = +ms.toFixed(3);
          total += ms;
        }
        next['total'] = +total.toFixed(3);
        this.lastGpuTimings = next;
        this.lastGpuTimingsFrame = this.frameCount;
      } catch {
        // ignore — buffer was likely unmapped during a dispose race
      } finally {
        this.timestampReadbackInFlight = null;
      }
    }).catch(() => {
      this.timestampReadbackInFlight = null;
    });
  }

  /** Re-upload emitter data (called on sun/light/panel change — §8.5). */
  updateEmitters(bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf'>): void {
    this.emitterBuffer.destroy();
    this.emitterCdfBuffer.destroy();
    this.emitterBuffer = this.uploadBuffer(
      bvhBuffers.emitters.cpuData, GPUBufferUsage.STORAGE,
    );
    this.emitterCdfBuffer = this.uploadBuffer(
      bvhBuffers.emitterCdf.cpuData, GPUBufferUsage.STORAGE,
    );
  }

  /**
   * Run one frame of the ReSTIR compute pipeline + composite render pass.
   * Returns true on success, false if pipeline not ready.
   */
  renderFrame(inputs: PipelineFrameInputs): boolean {
    if (!this.initialized) return false;

    const d = this.device;
    const { width: W, height: H } = this;

    // ── Update UBO ────────────────────────────────────────────────────────
    this.updateUBO(inputs);

    // ── Build placeholder texture view ────────────────────────────────────
    const placeholderView = this.placeholderTexture.createView();

    // ── Build bind groups ─────────────────────────────────────────────────
    const bgFrame  = this.buildFrameBindGroup(placeholderView);
    const bgScene  = this.buildSceneBindGroup();
    const bgUbo    = this.buildUboBindGroup();

    // ── Dispatch compute passes ───────────────────────────────────────────
    const encoder = d.createCommandEncoder({ label: 'walkaround-restir' });

    const wgX = Math.ceil(W / 8);
    const wgY = Math.ceil(H / 8);

    // Pass 1: RIS (primary ray cast + reservoir sampling)
    {
      const pass = encoder.beginComputePass({ label: 'ris', timestampWrites: this.tsWrites(0) });
      pass.setPipeline(this.risPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 2: Temporal reuse
    {
      const pass = encoder.beginComputePass({ label: 'temporal', timestampWrites: this.tsWrites(1) });
      pass.setPipeline(this.temporalPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 3a + 3b: Spatial reuse — TWO passes restored 2026-05-08 for
    // fidelity. Per-pass cost on Lovelace was ~22ms each; halved
    // budget by going back to one pass earlier, but soft falloff
    // and AO-like coherence visibly degraded. With NEIGHBORS=5 (was
    // 3), each pass is heavier but the visual win is the dominant
    // variance reducer in the pipeline. Total spatial budget back to
    // ~45ms; net frame still under headroom.
    {
      const pass = encoder.beginComputePass({ label: 'spatial-1', timestampWrites: this.tsWrites(2) });
      pass.setPipeline(this.spatialPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: 'spatial-2', timestampWrites: this.tsWrites(3) });
      pass.setPipeline(this.spatialPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 4: Shading + GI (re-traces primary ray, evaluates ReSTIR).
    // Phase 2B — single combined bind group at slot 3 holds DDGI + RC
    // inputs (Lovelace caps maxBindGroups=4 so we can't use slot 4).
    // shade.wgsl gates on isDDGIWired() / isRCWired().
    const bgHybrid = this.buildHybridLayersBindGroup();
    {
      const pass = encoder.beginComputePass({ label: 'shade', timestampWrites: this.tsWrites(4) });
      pass.setPipeline(this.shadePipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.setBindGroup(3, bgHybrid);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 5: À-trous denoiser (5 iterations with stepWidths 1,2,4,8,16).
    // The denoiser now consumes the real per-pixel normal+depth G-buffer
    // written by the shade pass (`gNormalDepthTexture`) so its edge-stopping
    // weights actually fire.  Pre-fix, this was bound to a 1×1 placeholder
    // texture (constant value for every pixel) which collapsed σn and σz to
    // no-ops and forced the filter to rely entirely on σc — at the absurdly
    // narrow value 0.01, which barely smoothed the noisy ReSTIR output.
    const wgX16 = Math.ceil(W / 16);
    const wgY16 = Math.ceil(H / 16);
    let inputTex = this.hdrColorTexture;
    const gNormalDepthView = this.gNormalDepthTexture.createView();

    // 3 atrous iterations (stepWidths 1, 2, 4). 2-iter was technically
    // crisper but read as too-sharp/aliased on caustic edges. 3 iters
    // gives a 1-2px collective softening across all colors equally
    // (σc=0.05 still blocks distinct-cell color jumps). Caustic shapes
    // stay distinct, edges read as smooth instead of pixelated.
    for (let iter = 0; iter < 3; iter++) {
      const stepWidth = 1 << iter;
      const outputTex = iter % 2 === 0 ? this.denoisedPingTexture : this.denoisedPongTexture;
      const bgAtrous = this.buildAtrousBindGroup(
        inputTex.createView(), outputTex.createView(),
        gNormalDepthView, gNormalDepthView, stepWidth,
      );
      const pass = encoder.beginComputePass({
        label: `atrous-${iter}`,
        timestampWrites: this.tsWrites(5 + iter),
      });
      pass.setPipeline(this.atrousPipeline);
      pass.setBindGroup(0, bgAtrous);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outputTex;
    }

    // Pass 5.5: Temporal accumulation. Blend current_atrous_output with
    // last frame's accumulated HDR. On big camera motion (delta > 5
    // world units) reset the accumulator to avoid ghosting; otherwise
    // α=0.1 (10% new, 90% history) — ~30-frame EMA window. Briefly
    // tried α=0.05 for stronger noise suppression after the bidirectional
    // Lo_emit landed, but user reported convergence "took forever" (the
    // EMA window doubled to ~60 frames, ≥4s on heavy GPU load). The
    // signed-sunDot fix means chroma now has 2× the pixel coverage, so
    // the per-pixel noise needs less temporal smoothing for the same
    // perceived clarity — α=0.1 is the better speed/clarity trade.
    const atrousFinalTex = this.denoisedPingTexture;
    const dx = inputs.cameraPos[0] - this.lastCameraPos[0];
    const dy = inputs.cameraPos[1] - this.lastCameraPos[1];
    const dz = inputs.cameraPos[2] - this.lastCameraPos[2];
    const camMoveSq = dx * dx + dy * dy + dz * dz;
    const isFirstFrame = this.accumFrameIndex === 0;
    // Threshold tuned to OrbitControls damping. After the user releases
    // a drag, OrbitControls' damping continues to update the camera by
    // ~0.1-0.5" per frame for ~30 frames before it fully settles. At
    // threshold 0.001 (any motion) the accumulator never gets to
    // integrate noise — sun-cone jitter and ReSTIR DI variance stay
    // visible forever. Threshold 1.0 (= 1" per frame, moveSq=1.0) lets
    // OrbitControls damping ride through to α=0.1 while still resetting
    // history on actual user pan/orbit (which moves multiple inches).
    const isMoving = camMoveSq > 1.0;
    const alpha = (isFirstFrame || isMoving) ? 1.0 : 0.1;
    if (isMoving) this.accumFrameIndex = 0;
    this.lastCameraPos = [...inputs.cameraPos];

    const readAccum  = this.accumPingPongIndex === 0 ? this.accumTextureA : this.accumTextureB;
    const writeAccum = this.accumPingPongIndex === 0 ? this.accumTextureB : this.accumTextureA;
    {
      const bgAccum = this.buildAccumBindGroup(
        atrousFinalTex.createView(),
        readAccum.createView(),
        writeAccum.createView(),
        alpha,
      );
      const pass = encoder.beginComputePass({
        label: 'temporalAccum',
        timestampWrites: this.tsWrites(8),
      });
      pass.setPipeline(this.accumPipeline);
      pass.setBindGroup(0, bgAccum);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }
    this.accumPingPongIndex = 1 - this.accumPingPongIndex;
    this.accumFrameIndex++;

    // Pass 6: Composite render pass — blit accumulated HDR to swap-chain.
    const finalTex = writeAccum;
    const bgComposite = this.buildCompositeBindGroup(finalTex.createView());
    {
      const pass = encoder.beginRenderPass({
        label: 'composite',
        colorAttachments: [{
          view: inputs.swapChainView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
        timestampWrites: this.tsWrites(9),
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bgComposite);
      pass.draw(3, 1, 0, 0);  // 3 vertices, fullscreen triangle
      pass.end();
    }

    // Resolve timestamps + copy into the inactive readback buffer (the
    // ping-pong slot NOT currently in flight). Skip if the prior frame's
    // readback for this slot is still pending — drops one frame of GPU
    // timings but never stalls submit.
    if (this.timestampQuerySet && this.timestampResolveBuffer) {
      const N = WalkaroundGPUPipeline.PASS_COUNT;
      const target = this.frameCount % 2 === 0 ? this.timestampReadbackA : this.timestampReadbackB;
      const slot: 'A' | 'B' = this.frameCount % 2 === 0 ? 'A' : 'B';
      // Skip resolve+copy if the target buffer is still mapped from a
      // prior readback (i.e. its slot matches `timestampReadbackInFlight`).
      if (target && this.timestampReadbackInFlight !== slot) {
        encoder.resolveQuerySet(this.timestampQuerySet, 0, N * 2, this.timestampResolveBuffer, 0);
        encoder.copyBufferToBuffer(this.timestampResolveBuffer, 0, target, 0, N * 2 * 8);
      }
    }

    d.queue.submit([encoder.finish()]);

    // Swap reservoir ping-pong for next frame (copy current → previous).
    const enc2 = d.createCommandEncoder({ label: 'reservoir-swap' });
    enc2.copyBufferToBuffer(
      this.reservoirCurrentBuffer, 0,
      this.reservoirPreviousBuffer, 0,
      this.reservoirCurrentBuffer.size,
    );
    d.queue.submit([enc2.finish()]);

    // Kick async readback of the timestamp buffer we just copied into.
    // Fires once per frame max; readback completion takes a few frames
    // because GPU work is queued. Result lands in this.lastGpuTimings.
    this.kickTimestampReadback();

    this.frameCount++;
    return true;
  }

  dispose(): void {
    this.bvhNodesBuffer?.destroy();
    this.bvhIndexBuffer?.destroy();
    this.bvhBeerBuffer?.destroy();
    this.bvhPositionBuffer?.destroy();
    this.bvhNormalBuffer?.destroy();
    this.bvhUvBuffer?.destroy();
    this.emitterBuffer?.destroy();
    this.emitterCdfBuffer?.destroy();
    this.reservoirCurrentBuffer?.destroy();
    this.reservoirPreviousBuffer?.destroy();
    this.reservoirSpatialBuffer?.destroy();
    this.hdrColorTexture?.destroy();
    this.gNormalDepthTexture?.destroy();
    this.denoisedPingTexture?.destroy();
    this.denoisedPongTexture?.destroy();
    this.accumTextureA?.destroy();
    this.accumTextureB?.destroy();
    this.placeholderTexture?.destroy();
    this.uboBuffer?.destroy();
    this._atrousUboBuffer?.destroy();
    this._ddgiPlaceholderRgba16f?.destroy();
    this._ddgiPlaceholderRg16f?.destroy();
    this._ddgiUboBuffer?.destroy();
    // Timestamp infrastructure — destroy whichever of A/B isn't mapped
    // (mapped buffers can't be destroy()'d; the GC reclaims them when
    // their mapped range goes out of scope). QuerySet.destroy() is safe
    // unconditionally.
    this.timestampQuerySet?.destroy();
    this.timestampResolveBuffer?.destroy();
    if (this.timestampReadbackInFlight !== 'A') this.timestampReadbackA?.destroy();
    if (this.timestampReadbackInFlight !== 'B') this.timestampReadbackB?.destroy();
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private uploadBuffer(data: ArrayBuffer, usage: number): GPUBuffer {
    const size = Math.max(data.byteLength, 16); // min 16 bytes
    const buf = this.device.createBuffer({
      size,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
    buf.unmap();
    return buf;
  }

  private updateUBO(inputs: PipelineFrameInputs): void {
    // UBO layout (all f32 / u32, 256 bytes):
    //   offset  0: viewMatrix      (mat4×4f = 64 bytes)
    //   offset 64: projMatrix      (64 bytes)
    //  offset 128: prevViewMatrix  (64 bytes)
    //  offset 192: cameraPos       (vec3f = 12 bytes)
    //  offset 204: frameSeed       (u32 = 4 bytes)
    //  offset 208: screenSize      (vec2u = 8 bytes)
    //  offset 216: emitterCount    (u32 = 4 bytes)
    //  offset 220: totalEmPower    (f32 = 4 bytes)
    //  offset 224: primaryLightDir       (vec3f = 12 bytes)
    //  offset 236: primaryLightIntensity (f32 = 4 bytes) — irradiance multiplier
    //  offset 240: skyTint               (vec3f = 12 bytes) — diffuse sky dome RGB
    //  offset 252: skyIrradiance         (f32 = 4 bytes) — dome brightness scalar
    // Total: 256 bytes (fully consumed).
    const data = new ArrayBuffer(256);
    const f32  = new Float32Array(data);
    const u32  = new Uint32Array(data);

    f32.set(inputs.viewMatrix,     0);    //  0..15 (64 bytes)
    f32.set(inputs.projMatrix,    16);    // 16..31 (64 bytes)
    f32.set(inputs.prevViewMatrix, 32);   // 32..47 (64 bytes)
    f32[48] = inputs.cameraPos[0];
    f32[49] = inputs.cameraPos[1];
    f32[50] = inputs.cameraPos[2];
    u32[51] = inputs.frameSeed >>> 0;
    u32[52] = inputs.screenWidth;
    u32[53] = inputs.screenHeight;
    u32[54] = inputs.emitterCount;
    f32[55] = inputs.totalEmissivePower;
    f32[56] = inputs.primaryLightDir[0];
    f32[57] = inputs.primaryLightDir[1];
    f32[58] = inputs.primaryLightDir[2];
    f32[59] = inputs.primaryLightIntensity;
    f32[60] = inputs.skyTint[0];
    f32[61] = inputs.skyTint[1];
    f32[62] = inputs.skyTint[2];
    f32[63] = inputs.skyIrradiance;

    this.device.queue.writeBuffer(this.uboBuffer, 0, data);
  }

  private async compilePipelines(): Promise<void> {
    const d = this.device;

    // Compile all shader modules (common WGSL is prepended to each).
    const risSM      = d.createShaderModule({ label: 'ris',      code: COMMON_WGSL + RIS_WGSL });
    const temporalSM = d.createShaderModule({ label: 'temporal', code: COMMON_WGSL + TEMPORAL_WGSL });
    const spatialSM  = d.createShaderModule({ label: 'spatial',  code: COMMON_WGSL + SPATIAL_WGSL });
    const shadeSM    = d.createShaderModule({ label: 'shade',    code: COMMON_WGSL + SHADE_WGSL });
    const atrousSM   = d.createShaderModule({ label: 'atrous',   code: COMMON_WGSL + ATROUS_WGSL });
    const compVertSM = d.createShaderModule({ label: 'comp-vert', code: COMPOSITE_VERT_WGSL });
    const compFragSM = d.createShaderModule({ label: 'comp-frag', code: COMPOSITE_FRAG_WGSL });

    // Check for compile errors.
    const modules: [string, GPUShaderModule][] = [
      ['ris', risSM], ['temporal', temporalSM], ['spatial', spatialSM],
      ['shade', shadeSM], ['atrous', atrousSM],
      ['comp-vert', compVertSM], ['comp-frag', compFragSM],
    ];
    for (const [label, sm] of modules) {
      const info = await sm.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === 'error');
      if (errors.length > 0) {
        console.error(`[ReSTIR] Shader compile errors in '${label}':`, errors.map(e => `line ${e.lineNum}: ${e.message}`));
        throw new Error(`[ReSTIR] Shader compile error in '${label}': ${errors[0].message} (line ${errors[0].lineNum})`);
      }
      const warns = info.messages.filter(m => m.type === 'warning');
      if (warns.length > 0) {
        console.warn(`[ReSTIR] Shader warnings in '${label}':`, warns.map(w => w.message));
      }
    }

    // Pipeline layouts.
    // - computeLayout: shared by RIS, temporal, spatial. 3 bind groups.
    // - shadeLayout (Phase 1.2A): adds DDGI as 4th bind group. Used only
    //   by the shade pipeline. RIS/temporal/spatial don't need DDGI inputs.
    const computeLayout = d.createPipelineLayout({
      bindGroupLayouts: [
        this.getFrameBindGroupLayout(),
        this.getSceneBindGroupLayout(),
        this.getUboBindGroupLayout(),
      ],
    });
    const shadeLayout = d.createPipelineLayout({
      bindGroupLayouts: [
        this.getFrameBindGroupLayout(),
        this.getSceneBindGroupLayout(),
        this.getUboBindGroupLayout(),
        this.getHybridLayersBindGroupLayout(),
      ],
    });

    const atrousLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.getAtrousBindGroupLayout()],
    });

    const accumLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.getAccumBindGroupLayout()],
    });

    const compositeLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.getCompositeBindGroupLayout()],
    });

    // Compile compute pipelines in parallel.
    [this.risPipeline, this.temporalPipeline, this.spatialPipeline, this.shadePipeline] =
      await Promise.all([
        d.createComputePipelineAsync({ label: 'ris',      layout: computeLayout, compute: { module: risSM,      entryPoint: 'risMain'      } }),
        d.createComputePipelineAsync({ label: 'temporal', layout: computeLayout, compute: { module: temporalSM, entryPoint: 'temporalMain' } }),
        d.createComputePipelineAsync({ label: 'spatial',  layout: computeLayout, compute: { module: spatialSM,  entryPoint: 'spatialMain'  } }),
        d.createComputePipelineAsync({ label: 'shade',    layout: shadeLayout,   compute: { module: shadeSM,    entryPoint: 'shadeMain'    } }),
      ]);

    this.atrousPipeline = await d.createComputePipelineAsync({
      label: 'atrous', layout: atrousLayout,
      compute: { module: atrousSM, entryPoint: 'atrousMain' },
    });

    const accumSM = d.createShaderModule({ label: 'accum', code: TEMPORAL_ACCUM_WGSL });
    this.accumPipeline = await d.createComputePipelineAsync({
      label: 'temporalAccum', layout: accumLayout,
      compute: { module: accumSM, entryPoint: 'temporalAccumMain' },
    });

    // Composite render pipeline.
    this.compositePipeline = await d.createRenderPipelineAsync({
      label: 'composite',
      layout: compositeLayout,
      vertex:   { module: compVertSM, entryPoint: 'vertMain' },
      fragment: {
        module: compFragSM,
        entryPoint: 'fragMain',
        targets: [{ format: this._swapChainFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    console.log('[ReSTIR] All pipelines compiled successfully');
  }

  // ── Bind group layout factories ───────────────────────────────────────────

  private _frameBGL?: GPUBindGroupLayout;
  private getFrameBindGroupLayout(): GPUBindGroupLayout {
    if (this._frameBGL) return this._frameBGL;
    this._frameBGL = this.device.createBindGroupLayout({
      label: 'frame-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'non-filtering' } },
        // gNormalDepth — written by shade pass (normal in xyz, primary-hit
        // distance in w); read by the à-trous denoiser for edge stopping.
        // Declared in all four compute pass bind groups (RIS / temporal /
        // spatial / shade) for layout compatibility, but only shade actually
        // writes to it.  Bound to the same texture in every dispatch — the
        // unused write bindings are validated and inert.
        { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });
    return this._frameBGL;
  }

  private _sceneBGL?: GPUBindGroupLayout;
  private getSceneBindGroupLayout(): GPUBindGroupLayout {
    if (this._sceneBGL) return this._sceneBGL;
    this._sceneBGL = this.device.createBindGroupLayout({
      label: 'scene-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvhNodes
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvhIndex (vec4u: [0..2]=indices, [3]=RGBA8 raw attCol)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvhPositions
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // emitters
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // emitterCdf
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_beer (Beer-Lambert visible color)
      ],
    });
    return this._sceneBGL;
  }

  private _uboBGL?: GPUBindGroupLayout;
  private getUboBindGroupLayout(): GPUBindGroupLayout {
    if (this._uboBGL) return this._uboBGL;
    this._uboBGL = this.device.createBindGroupLayout({
      label: 'ubo-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    return this._uboBGL;
  }

  private _atrousBGL?: GPUBindGroupLayout;
  private getAtrousBindGroupLayout(): GPUBindGroupLayout {
    if (this._atrousBGL) return this._atrousBGL;
    this._atrousBGL = this.device.createBindGroupLayout({
      label: 'atrous-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    return this._atrousBGL;
  }

  private _compositeBGL?: GPUBindGroupLayout;
  private getCompositeBindGroupLayout(): GPUBindGroupLayout {
    if (this._compositeBGL) return this._compositeBGL;
    this._compositeBGL = this.device.createBindGroupLayout({
      label: 'composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
      ],
    });
    return this._compositeBGL;
  }

  private getAccumBindGroupLayout(): GPUBindGroupLayout {
    if (this._accumBGL) return this._accumBGL;
    this._accumBGL = this.device.createBindGroupLayout({
      label: 'accum-bgl',
      entries: [
        // 0: currentAtrous (read)
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        // 1: prevAccum (read)
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        // 2: accumOut (write)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        // 3: AccumUBO
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    return this._accumBGL;
  }

  private buildAccumBindGroup(
    currentAtrousView: GPUTextureView,
    prevAccumView: GPUTextureView,
    accumOutView: GPUTextureView,
    alpha: number,
  ): GPUBindGroup {
    if (!this._accumUboBuffer) {
      this._accumUboBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    // varianceK: 1.5 — std-dev multiplier for the temporal-history clamp
    // box. Karis recommends k=1 in his original temporal-AA writeup;
    // 1.5 is a small relaxation that lets stable samples blend more
    // smoothly without re-introducing the sparkle/cross-talk at edges.
    // Tunable. Higher = more history weight (more smoothing, more
    // sparkle risk), lower = stricter clamp (sharper edges, more noise).
    this.device.queue.writeBuffer(
      this._accumUboBuffer, 0,
      new Float32Array([alpha, 1.5, 0, 0]),
    );
    return this.device.createBindGroup({
      label: 'accum-bg',
      layout: this.getAccumBindGroupLayout(),
      entries: [
        { binding: 0, resource: currentAtrousView },
        { binding: 1, resource: prevAccumView },
        { binding: 2, resource: accumOutView },
        { binding: 3, resource: { buffer: this._accumUboBuffer } },
      ],
    });
  }

  // Phase 2B — combined hybrid-layers bind group (slot 3, shade only).
  // DDGI + RC inputs packed into one group because Lovelace's adapter
  // caps `maxBindGroups = 4` (verified empirically); a 5th group is
  // rejected. maxBindingsPerBindGroup is 1000 so 10 bindings is fine.
  // Layout:
  //   DDGI section
  //     0 — irradiance atlas (texture_2d<f32>, unfilterable)
  //     1 — visibility atlas (texture_2d<f32>, unfilterable)
  //     2 — non-filtering sampler
  //     3 — DDGI grid uniform (64 bytes)
  //   RC section
  //     4 — cascade C0 storage buffer
  //     5 — cascade C1
  //     6 — cascade C2
  //     7 — cascade C3
  //     8 — cascade C4
  //     9 — RC params uniform (256 bytes)
  private _hybridLayersBGL?: GPUBindGroupLayout;
  private getHybridLayersBindGroupLayout(): GPUBindGroupLayout {
    if (this._hybridLayersBGL) return this._hybridLayersBGL;
    this._hybridLayersBGL = this.device.createBindGroupLayout({
      label: 'hybrid-layers-bgl',
      entries: [
        // DDGI (RC bindings dropped step 4 of restructure — Lo_rc was
        // computed and discarded; the cascade buffers, params UBO, and
        // setRCInputs wiring all retired together. The RC subsystem
        // itself remains live for the standalone 'rc' walkaround engine.)
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'non-filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    return this._hybridLayersBGL;
  }

  /**
   * Phase 1.2A — bind a DDGI atlas. Pass `null` to revert to placeholder
   * (no-DDGI fallback). The shade pass continues to render with hardcoded
   * sky color until Phase 1.2B updates shade.wgsl to actually sample
   * the atlas.
   *
   * Caller (HybridLayeredStage) provides:
   *  - irradianceTex / visibilityTex: GPUTexture instances with
   *    TEXTURE_BINDING usage. ProbeGrid's StorageTexture exposes these
   *    via the .image property after .createForRendering — caller must
   *    pass the renderer-resolved GPUTexture, not the StorageTexture
   *    wrapper.
   *  - gridParams: 64-byte ArrayBuffer matching the WGSL DDGIGridUniform
   *    layout (origin vec3 + spacing f32 + dims vec3u + pad u32 +
   *    irradianceAtlasW/H + visibilityAtlasW/H).
   */
  setDDGIInputs(inputs: {
    irradianceTex: GPUTexture;
    visibilityTex: GPUTexture;
    gridParams: ArrayBuffer;
  } | null): void {
    if (inputs === null) {
      this._ddgiIrrTex = null;
      this._ddgiVisTex = null;
      // Restore placeholder UBO (dims=1×1×1) so shade.wgsl's
      // isDDGIWired() check returns false and the DDGI contribution
      // drops to zero.
      const placeholder = new Float32Array(16);
      placeholder[3] = 24;
      new Uint32Array(placeholder.buffer)[4] = 1;
      new Uint32Array(placeholder.buffer)[5] = 1;
      new Uint32Array(placeholder.buffer)[6] = 1;
      placeholder[8]  = 1;
      placeholder[9]  = 1;
      placeholder[10] = 1;
      placeholder[11] = 1;
      this.device.queue.writeBuffer(this._ddgiUboBuffer, 0, placeholder.buffer);
    } else {
      this._ddgiIrrTex = inputs.irradianceTex;
      this._ddgiVisTex = inputs.visibilityTex;
      if (inputs.gridParams.byteLength > 0) {
        this.device.queue.writeBuffer(this._ddgiUboBuffer, 0, inputs.gridParams);
      }
    }
  }

  // ── Bind group builders ───────────────────────────────────────────────────

  private buildFrameBindGroup(placeholderView: GPUTextureView): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'frame-bg',
      layout: this.getFrameBindGroupLayout(),
      entries: [
        { binding: 0, resource: placeholderView },  // gDepth (placeholder — not used in primary-ray-cast mode)
        { binding: 1, resource: placeholderView },  // gNormal
        { binding: 2, resource: placeholderView },  // gAlbedo
        { binding: 3, resource: placeholderView },  // gRough
        { binding: 4, resource: placeholderView },  // motionVec
        { binding: 5, resource: { buffer: this.reservoirCurrentBuffer } },
        { binding: 6, resource: { buffer: this.reservoirPreviousBuffer } },
        { binding: 7, resource: { buffer: this.reservoirSpatialBuffer } },
        { binding: 8, resource: this.hdrColorTexture.createView() },
        { binding: 9, resource: this.nearestSampler },
        // gNormalDepth — only the shade pass writes to it; other passes
        // declare it (in the BGL) but never reference the symbol, so it's
        // inert for them.  Bound to the same texture in every dispatch.
        { binding: 10, resource: this.gNormalDepthTexture.createView() },
      ],
    });
  }

  private buildSceneBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'scene-bg',
      layout: this.getSceneBindGroupLayout(),
      entries: [
        { binding: 0, resource: { buffer: this.bvhNodesBuffer } },
        { binding: 1, resource: { buffer: this.bvhIndexBuffer } },   // vec4u: [0..2]=indices, [3]=RGBA8 raw attCol
        { binding: 2, resource: { buffer: this.bvhPositionBuffer } },
        { binding: 3, resource: { buffer: this.emitterBuffer } },
        { binding: 4, resource: { buffer: this.emitterCdfBuffer } },
        { binding: 5, resource: { buffer: this.bvhBeerBuffer } },    // u32: per-tri Beer-Lambert visible color
      ],
    });
  }

  private buildUboBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'ubo-bg',
      layout: this.getUboBindGroupLayout(),
      entries: [{ binding: 0, resource: { buffer: this.uboBuffer } }],
    });
  }

  private _atrousUboBuffer?: GPUBuffer;
  private buildAtrousBindGroup(
    inputView: GPUTextureView,
    outputView: GPUTextureView,
    gNormalView: GPUTextureView,
    gDepthView: GPUTextureView,
    stepWidth: number,
  ): GPUBindGroup {
    if (!this._atrousUboBuffer) {
      this._atrousUboBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    // sigmaN=128, sigmaZ=5.0, sigmaC=0.25 are tuned for an HDR-linear ReSTIR
    // à-trous denoiser at this scene's scale (camera-relative ray distances
    // ~30..200 units).  The shade pass now writes a real per-pixel
    // normal+depth G-buffer into `gNormalDepthTexture`, so the normal-based
    // edge-stop (`pow(dot(n,n)^k, sigmaN)`) and depth-based stop
    // (`exp(-|Δz|/sigmaZ)`) are meaningful again.
    //   σn=128 → tight normal-stop; only near-coplanar surfaces blur together.
    //   σz=5   → tolerates ~5 unit depth differences (one floor-tile receding
    //            from camera at stepWidth=16); rejects floor↔wall transitions.
    //   σc=0.25 → tightened FURTHER from 0.5.  Floor caustic patches have
    //            smaller chroma deltas than panel cells (a red caustic on
    //            oak floor is ≈ (0.79, 0.34, 0.28) vs the un-caustic floor
    //            ≈ (0.33, 0.31, 0.26) — Δ≈0.46, not 1.0+ like adjacent
    //            saturated panel cells).  At σc=0.5 the gaussian
    //            exp(-0.46²/0.25) ≈ 0.43 still blends caustic patches
    //            with un-caustic floor — the "smooth warm blobs without
    //            per-cell discrimination" symptom.  σc=0.25 → exp(-0.46²/0.0625)
    //            ≈ 0.034, which cleanly separates caustic from non-caustic
    //            while still smoothing within-patch noise (Δ<0.05 keeps
    //            weight ≈0.96).  Panel cells (Δ≈1.0+) remain trivially
    //            separable — exp(-1/0.0625) ≈ 1e-7 — so the previous
    //            cell-discrimination property is preserved.
    //
    // Sky pixels write depth=0 in shade; non-sky pixels write `primaryHit.dist`
    // (positive).  The depth edge-stop therefore separates sky from non-sky
    // automatically — important so the denoiser does not blur sky color into
    // floor caustic regions.
    // σc 0.4 (was 0.25) — relax the chroma edge-stop slightly so flat
    // regions (uniform glass interior, smooth wall surfaces) blend more
    // smoothly without losing the came/glass boundary preservation
    // (still strongly rejected by the tight σn=128 normal stop). The
    // 0.25 → 0.4 change increases the gaussian width from σ=0.25 to
    // σ=0.4: at Δ=0.32 between came (gray) and glass (red), weight goes
    // from 0.19 → 0.53 — but normal-stop at this boundary is essentially
    // zero (came normal +Z vs glass back-face normal -Z, dot=-1, pow=0),
    // so came-glass blending stays blocked. The benefit is in low-Δ
    // regions where intra-cell noise can blur out without fighting the
    // chroma stop.
    // σc tightened further 0.15 → 0.05 to handle low-chroma-delta
    // caustic boundaries. At 0.15, RED caustics on warm-oak floor
    // (huge R delta) blocked cleanly, but BLUE/GREEN caustics on the
    // same floor had small Δc (both have low B/G) — atrous bled
    // cool-tinted caustic into the warm floor over 3 iterations,
    // producing the "some sharp, some blurry" symptom user flagged.
    // At σc=0.05: even Δ=0.36 blue-caustic-vs-warm-floor chroma gap
    // weighs ≈ 0 (blocked); Δ=0.05 within-patch noise still weighs
    // ≈ 0.37 so within-patch denoising continues.
    const uboData = new Float32Array([stepWidth, 128.0, 5.0, 0.05]);
    this.device.queue.writeBuffer(this._atrousUboBuffer, 0, uboData);

    return this.device.createBindGroup({
      label: `atrous-bg-step${stepWidth}`,
      layout: this.getAtrousBindGroupLayout(),
      entries: [
        { binding: 0, resource: inputView },
        { binding: 1, resource: outputView },
        { binding: 2, resource: gNormalView },
        { binding: 3, resource: gDepthView },
        { binding: 4, resource: { buffer: this._atrousUboBuffer } },
      ],
    });
  }

  // DDGI bind group (group 3). RC bindings dropped — see
  // getHybridLayersBindGroupLayout comment.
  private buildHybridLayersBindGroup(): GPUBindGroup {
    const irrTex = this._ddgiIrrTex ?? this._ddgiPlaceholderRgba16f;
    const visTex = this._ddgiVisTex ?? this._ddgiPlaceholderRg16f;
    return this.device.createBindGroup({
      label: 'hybrid-layers-bg',
      layout: this.getHybridLayersBindGroupLayout(),
      entries: [
        { binding: 0, resource: irrTex.createView() },
        { binding: 1, resource: visTex.createView() },
        { binding: 2, resource: this.nearestSampler },
        { binding: 3, resource: { buffer: this._ddgiUboBuffer } },
      ],
    });
  }

  private buildCompositeBindGroup(texView: GPUTextureView): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'composite-bg',
      layout: this.getCompositeBindGroupLayout(),
      entries: [
        { binding: 0, resource: texView },
        { binding: 1, resource: this.compositeLinearSampler },
      ],
    });
  }
}
