/**
 * WalkaroundGPUPipeline — manages all WebGPU resources + compute passes for
 * the ReSTIR DI/GI pipeline.
 *
 * Uses the fully-manual `device.createShaderModule()` path since wgslFn
 * composition with three's TSL compute() is unvalidated. Web-RTRT confirms
 * this approach works in browser. Exposes a simple `renderFrame()` method
 * that the WalkaroundStage calls per-frame.
 *
 * Pipeline shape per frame:
 *   1. RIS: primary-ray-cast primary visibility + initial candidate sampling
 *   2. Temporal reuse: merge with previous-frame reservoir
 *   3. Spatial reuse (2 separable passes)
 *   4. Shade + GI: compute DI + one indirect bounce, write HDR color
 *   5. À-trous denoiser (3 iterations, stepWidths 1, 2, 4)
 *   6. Temporal accumulation: EMA blend with previous frame's HDR
 *   7. Composite render pass: blit accumulated HDR to the swap-chain texture
 *
 * Note: we use primary-ray-casting mode instead of a G-buffer raster pass.
 * The G-buffer bind group slots are filled with 1×1 placeholder textures for
 * layout compatibility; the RIS + shade passes generate their own primary
 * visibility by casting rays through the BVH.
 */

import type { SceneBVHBuffers } from '../restir/bvhCompute.js';
import { updateUBO } from './uboUpdater.js';
import { compilePipelines } from './pipelineCompiler.js';
import {
  uploadBuffer,
  buildDDGIPlaceholderUBO,
  createFrameResources,
  destroyFrameResources,
  type FrameResources,
} from './resourceManager.js';
import {
  type BGLCache,
} from './bindGroupLayouts.js';
import {
  buildFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
  buildAtrousBindGroup,
  buildAccumBindGroup,
  buildHybridLayersBindGroup,
  buildCompositeBindGroup,
  type UboRef,
} from './bindGroupBuilders.js';
import {
  tsWrites,
  initTimestampQueries,
  kickTimestampReadback,
  resolveTimestamps,
  disposeTimestampState,
  makeTimestampState,
  type TimestampState,
} from './timestampQueries.js';

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
 * Caller pattern:
 *   const renderer = new WebGPURenderer({
 *     ...,
 *     requiredLimits: HYBRID_WEBGPU_REQUIRED_LIMITS,
 *   });
 */
export const HYBRID_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: 16,
};

/**
 * Camera squared-distance threshold for temporal accumulator reset.
 * Tuned to OrbitControls damping: after a drag release, damping continues
 * to move the camera by ~0.1–0.5" per frame for ~30 frames before settling.
 * Threshold 1.0 lets damping ride through to α=0.1 while still resetting
 * history on actual pan/orbit. (WARM-4 fix: was inline magic literal `1.0`.)
 */
const CAMERA_MOVE_RESET_THRESHOLD_SQ = 1.0;

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

  // Static BVH + emitter buffers (uploaded once at initialize time)
  private bvhNodesBuffer!: GPUBuffer;
  private bvhIndexBuffer!: GPUBuffer;
  private bvhBeerBuffer!: GPUBuffer;
  private bvhPositionBuffer!: GPUBuffer;
  private bvhNormalBuffer!: GPUBuffer;
  private bvhUvBuffer!: GPUBuffer;
  private emitterBuffer!: GPUBuffer;
  private emitterCdfBuffer!: GPUBuffer;
  /**
   * Sprint 2 (Phase 6): per-emitter total radiant flux buffer (f32[]).
   * cellPower[i] = luminance(Le[i]) × area[i] for each emitter triangle.
   * Uploaded once at initialize time alongside the emitter CDF. Sprint 3's
   * light tree will build a power-weighted CDF over this buffer; Sprint 2
   * just makes it available in the pipeline. Not yet bound to any shader.
   */
  private cellPowerBuffer!: GPUBuffer;

  // Per-frame GPU resources (created by resourceManager.createFrameResources)
  private res!: FrameResources;

  // Temporal accumulator ping-pong state
  private accumPingPongIndex = 0;       // 0 = read A, write B; 1 = swap
  private accumFrameIndex = 0;
  private lastCameraPos: [number, number, number] = [0, 0, 0];

  // DDGI inputs (layered hybrid). Null → placeholder textures.
  private _ddgiIrrTex: GPUTexture | null = null;
  private _ddgiVisTex: GPUTexture | null = null;

  // Cached DDGI placeholder UBO — reused by setDDGIInputs(null) so we don't
  // allocate a fresh Float32Array(16) every frame when DDGI is disabled.
  // Populated lazily on first setDDGIInputs(null) call.
  private _ddgiPlaceholderUBO: Float32Array | null = null;

  // Compiled compute + render pipelines
  private risPipeline!: GPUComputePipeline;
  private temporalPipeline!: GPUComputePipeline;
  private spatialPipeline!: GPUComputePipeline;
  private shadePipeline!: GPUComputePipeline;
  private atrousPipeline!: GPUComputePipeline;
  private accumPipeline!: GPUComputePipeline;
  private compositePipeline!: GPURenderPipeline;
  private _swapChainFormat: GPUTextureFormat = 'bgra8unorm';

  // Bind group layout memoisation cache
  private bglCache: BGLCache = {};

  // Lazily-created per-builder UBO buffers
  private atrousUboRef: UboRef = { buf: undefined };
  private accumUboRef: UboRef  = { buf: undefined };

  // GPU timestamp query state (DEV-only, feature-gated)
  private tsState: TimestampState = makeTimestampState();
  /** Last successfully-read timestamp values, ms per pass. */
  public lastGpuTimings: Record<string, number> = {};
  /** Frame index of the last completed timestamp read. */
  public lastGpuTimingsFrame: number = -1;

  private frameCount = 0;
  private initialized = false;

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
    this.bvhNodesBuffer    = uploadBuffer(d, bvhBuffers.bvhNodes.cpuData,     GPUBufferUsage.STORAGE);
    this.bvhIndexBuffer    = uploadBuffer(d, bvhBuffers.bvhIndex.cpuData,     GPUBufferUsage.STORAGE);
    this.bvhBeerBuffer     = uploadBuffer(d, bvhBuffers.bvhBeerColors.cpuData, GPUBufferUsage.STORAGE);
    this.bvhPositionBuffer = uploadBuffer(d, bvhBuffers.bvhPositions.cpuData, GPUBufferUsage.STORAGE);
    this.bvhNormalBuffer   = uploadBuffer(d, bvhBuffers.bvhNormals.cpuData,   GPUBufferUsage.STORAGE);
    this.bvhUvBuffer       = uploadBuffer(d, bvhBuffers.bvhUvs.cpuData,       GPUBufferUsage.STORAGE);
    this.emitterBuffer     = uploadBuffer(d, bvhBuffers.emitters.cpuData,     GPUBufferUsage.STORAGE);
    this.emitterCdfBuffer  = uploadBuffer(d, bvhBuffers.emitterCdf.cpuData,   GPUBufferUsage.STORAGE);
    // Sprint 2 (Phase 6): cellPower[i] = luminance(Le[i]) * area[i].
    // Not yet bound to any WGSL shader — Sprint 3 light tree will consume it.
    this.cellPowerBuffer   = uploadBuffer(d, bvhBuffers.cellPower.cpuData,    GPUBufferUsage.STORAGE);
    // triangleMatIds are packed into bvhIndex[*].w — no separate GPU buffer.

    // ── Per-frame GPU resources ───────────────────────────────────────────
    this.res = createFrameResources(d, W, H);

    // ── Compile shaders ───────────────────────────────────────────────────
    const compiled = await compilePipelines(d, this.bglCache, swapChainFormat);
    this.risPipeline       = compiled.risPipeline;
    this.temporalPipeline  = compiled.temporalPipeline;
    this.spatialPipeline   = compiled.spatialPipeline;
    this.shadePipeline     = compiled.shadePipeline;
    this.atrousPipeline    = compiled.atrousPipeline;
    this.accumPipeline     = compiled.accumPipeline;
    this.compositePipeline = compiled.compositePipeline;

    // ── Timestamp queries (DEV-only, feature-gated) ──────────────────────
    initTimestampQueries(d, this.tsState);

    this.initialized = true;
    console.log('[ReSTIR] Pipeline initialized', { W, H, bvhNodes: bvhBuffers.bvhNodes.count, emitters: bvhBuffers.emitterCount });
  }

  /** Re-upload emitter data (called on light/panel change). */
  updateEmitters(bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf'>): void {
    this.emitterBuffer.destroy();
    this.emitterCdfBuffer.destroy();
    this.emitterBuffer    = uploadBuffer(this.device, bvhBuffers.emitters.cpuData,    GPUBufferUsage.STORAGE);
    this.emitterCdfBuffer = uploadBuffer(this.device, bvhBuffers.emitterCdf.cpuData,  GPUBufferUsage.STORAGE);
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
    updateUBO(d, this.res.uboBuffer, inputs);

    // ── Build placeholder texture view ────────────────────────────────────
    const placeholderView = this.res.placeholderTexture.createView();

    // ── Build bind groups ─────────────────────────────────────────────────
    const bgFrame = buildFrameBindGroup(d, this.bglCache, {
      placeholderView,
      reservoirCurrentBuffer:  this.res.reservoirCurrentBuffer,
      reservoirPreviousBuffer: this.res.reservoirPreviousBuffer,
      reservoirSpatialBuffer:  this.res.reservoirSpatialBuffer,
      hdrColorTexture:         this.res.hdrColorTexture,
      nearestSampler:          this.res.nearestSampler,
      gNormalDepthTexture:     this.res.gNormalDepthTexture,
    });
    const bgScene = buildSceneBindGroup(d, this.bglCache, {
      bvhNodesBuffer:    this.bvhNodesBuffer,
      bvhIndexBuffer:    this.bvhIndexBuffer,
      bvhPositionBuffer: this.bvhPositionBuffer,
      emitterBuffer:     this.emitterBuffer,
      emitterCdfBuffer:  this.emitterCdfBuffer,
      bvhBeerBuffer:     this.bvhBeerBuffer,
    });
    const bgUbo   = buildUboBindGroup(d, this.bglCache, this.res.uboBuffer);

    // ── Dispatch compute passes ───────────────────────────────────────────
    const encoder = d.createCommandEncoder({ label: 'walkaround-restir' });

    const wgX = Math.ceil(W / 8);
    const wgY = Math.ceil(H / 8);

    // Helper: build a GPUComputePassDescriptor without an undefined timestampWrites
    // property — required by exactOptionalPropertyTypes. We spread the optional
    // timestampWrites field only when it has a value.
    const computeDesc = (label: string, passIdx: number): GPUComputePassDescriptor => {
      const ts = tsWrites(this.tsState.querySet, passIdx);
      return ts ? { label, timestampWrites: ts } : { label };
    };

    // Pass 1: RIS (primary ray cast + reservoir sampling)
    {
      const pass = encoder.beginComputePass(computeDesc('ris', 0));
      pass.setPipeline(this.risPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 2: Temporal reuse
    {
      const pass = encoder.beginComputePass(computeDesc('temporal', 1));
      pass.setPipeline(this.temporalPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 3a + 3b: Spatial reuse — two passes for fidelity. Per-pass cost
    // on Lovelace was ~22ms each. With NEIGHBORS=5, each pass is heavier
    // but the visual win is the dominant variance reducer in the pipeline.
    {
      const pass = encoder.beginComputePass(computeDesc('spatial-1', 2));
      pass.setPipeline(this.spatialPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass(computeDesc('spatial-2', 3));
      pass.setPipeline(this.spatialPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 4: Shading + GI (re-traces primary ray, evaluates ReSTIR).
    // Combined hybrid-layers bind group at slot 3 holds DDGI inputs
    // (Lovelace caps maxBindGroups=4 so we can't use slot 4).
    // shade.wgsl gates on isDDGIWired().
    const bgHybrid = buildHybridLayersBindGroup(d, this.bglCache, {
      ddgiIrrTex:              this._ddgiIrrTex,
      ddgiVisTex:              this._ddgiVisTex,
      ddgiPlaceholderRgba16f:  this.res.ddgiPlaceholderRgba16f,
      ddgiPlaceholderRg16f:    this.res.ddgiPlaceholderRg16f,
      nearestSampler:          this.res.nearestSampler,
      ddgiUboBuffer:           this.res.ddgiUboBuffer,
    });
    {
      const pass = encoder.beginComputePass(computeDesc('shade', 4));
      pass.setPipeline(this.shadePipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.setBindGroup(3, bgHybrid);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 5: À-trous denoiser — 3 iterations (stepWidths 1, 2, 4).
    // Consumes the real per-pixel normal+depth G-buffer written by the shade
    // pass so edge-stopping weights fire correctly. 3 iters gives a 1-2px
    // collective softening across all colors equally (σc=0.05 still blocks
    // distinct-cell color jumps). Caustic shapes stay distinct.
    const wgX16 = Math.ceil(W / 16);
    const wgY16 = Math.ceil(H / 16);
    let inputTex = this.res.hdrColorTexture;
    const gNormalDepthView = this.res.gNormalDepthTexture.createView();

    for (let iter = 0; iter < 3; iter++) {
      const stepWidth = 1 << iter;
      const outputTex = iter % 2 === 0 ? this.res.denoisedPingTexture : this.res.denoisedPongTexture;
      const bgAtrous = buildAtrousBindGroup(
        d, this.bglCache, this.atrousUboRef,
        inputTex.createView(), outputTex.createView(),
        gNormalDepthView, gNormalDepthView, stepWidth,
      );
      const pass = encoder.beginComputePass(computeDesc(`atrous-${iter}`, 5 + iter));
      pass.setPipeline(this.atrousPipeline);
      pass.setBindGroup(0, bgAtrous);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outputTex;
    }

    // Pass 5.5: Temporal accumulation. Blend current_atrous_output with
    // last frame's accumulated HDR. On big camera motion (delta > 5 world
    // units) reset the accumulator to avoid ghosting; otherwise α=0.1
    // (10% new, 90% history) — ~30-frame EMA window. α=0.1 is the better
    // speed/clarity trade at the current signed-sunDot chroma coverage.
    const atrousFinalTex = this.res.denoisedPingTexture;
    const dx = inputs.cameraPos[0] - this.lastCameraPos[0];
    const dy = inputs.cameraPos[1] - this.lastCameraPos[1];
    const dz = inputs.cameraPos[2] - this.lastCameraPos[2];
    const camMoveSq = dx * dx + dy * dy + dz * dz;
    const isFirstFrame = this.accumFrameIndex === 0;
    // See CAMERA_MOVE_RESET_THRESHOLD_SQ for tuning rationale.
    const isMoving = camMoveSq > CAMERA_MOVE_RESET_THRESHOLD_SQ;
    const alpha = (isFirstFrame || isMoving) ? 1.0 : 0.1;
    if (isMoving) this.accumFrameIndex = 0;
    this.lastCameraPos = [...inputs.cameraPos];

    const readAccum  = this.accumPingPongIndex === 0 ? this.res.accumTextureA : this.res.accumTextureB;
    const writeAccum = this.accumPingPongIndex === 0 ? this.res.accumTextureB : this.res.accumTextureA;
    {
      const bgAccum = buildAccumBindGroup(
        d, this.bglCache, this.accumUboRef,
        atrousFinalTex.createView(),
        readAccum.createView(),
        writeAccum.createView(),
        alpha,
      );
      const pass = encoder.beginComputePass(computeDesc('temporalAccum', 8));
      pass.setPipeline(this.accumPipeline);
      pass.setBindGroup(0, bgAccum);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }
    this.accumPingPongIndex = 1 - this.accumPingPongIndex;
    this.accumFrameIndex++;

    // Pass 6: Composite render pass — blit accumulated HDR to swap-chain.
    const finalTex = writeAccum;
    const bgComposite = buildCompositeBindGroup(d, this.bglCache, finalTex.createView(), this.res.compositeLinearSampler);
    {
      const tsComp = tsWrites(this.tsState.querySet, 9);
      const pass = encoder.beginRenderPass({
        label: 'composite',
        colorAttachments: [{
          view: inputs.swapChainView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
        ...(tsComp ? { timestampWrites: tsComp } : {}),
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bgComposite);
      pass.draw(3, 1, 0, 0);  // 3 vertices, fullscreen triangle
      pass.end();
    }

    // Resolve timestamps + copy into the inactive readback buffer.
    resolveTimestamps(encoder, this.tsState, this.frameCount);

    d.queue.submit([encoder.finish()]);

    // Swap reservoir ping-pong for next frame (copy current → previous).
    const enc2 = d.createCommandEncoder({ label: 'reservoir-swap' });
    enc2.copyBufferToBuffer(
      this.res.reservoirCurrentBuffer, 0,
      this.res.reservoirPreviousBuffer, 0,
      this.res.reservoirCurrentBuffer.size,
    );
    d.queue.submit([enc2.finish()]);

    // Kick async readback of the timestamp buffer we just copied into.
    kickTimestampReadback(this.tsState, this.frameCount);
    // Mirror public telemetry fields from the state object so callers
    // can read them as before.
    this.lastGpuTimings      = this.tsState.lastGpuTimings;
    this.lastGpuTimingsFrame = this.tsState.lastGpuTimingsFrame;

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
    this.cellPowerBuffer?.destroy();
    if (this.res) destroyFrameResources(this.res);
    this.atrousUboRef.buf?.destroy();
    this.accumUboRef.buf?.destroy();
    disposeTimestampState(this.tsState);
  }

  /**
   * Bind a DDGI atlas. Pass `null` to revert to placeholder (no-DDGI
   * fallback). The shade pass continues to render with hardcoded sky color
   * when isDDGIWired() returns false (dimsX ≤ 1 in the placeholder UBO).
   *
   * Caller (HybridLayeredStage) provides:
   *  - irradianceTex / visibilityTex: GPUTexture with TEXTURE_BINDING usage.
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
      // isDDGIWired() check returns false and Lo_ddgi drops to zero.
      // Cache the placeholder to avoid allocating Float32Array(16) every
      // frame when DDGI is disabled (HOT-1 fix).
      if (this._ddgiPlaceholderUBO === null) {
        this._ddgiPlaceholderUBO = buildDDGIPlaceholderUBO();
      }
      this.device.queue.writeBuffer(this.res.ddgiUboBuffer, 0, this._ddgiPlaceholderUBO.buffer);
    } else {
      this._ddgiIrrTex = inputs.irradianceTex;
      this._ddgiVisTex = inputs.visibilityTex;
      if (inputs.gridParams.byteLength > 0) {
        this.device.queue.writeBuffer(this.res.ddgiUboBuffer, 0, inputs.gridParams);
      }
    }
  }
}
