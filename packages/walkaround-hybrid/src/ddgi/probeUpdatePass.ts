/**
 * ProbeUpdatePass — DDGI probe update via raw WebGPU compute.
 *
 * Uses the raw GPUDevice to run two compute passes per frame:
 *  Pass 1 (probeUpdateRays): for each active probe, fire 192 rays via
 *          inline BVH traversal, collect radiance at hit points.
 *  Pass 2 (probeUpdateBlend): blend ray results into the L2-SH irradiance
 *          atlas and the octahedral visibility atlas with EWMA temporal
 *          hysteresis.
 *
 * Raw WebGPU is used because the compute shaders have custom @group/@binding
 * layouts that the engine manages directly. The GPUDevice is supplied by the
 * caller (HybridEngine); this class does not own the device lifetime.
 *
 * The irradiance atlas stores 9-coefficient L2 spherical harmonics in a 3×3
 * texel block per probe (IRR_CELL=3); the visibility atlas uses a 16×16
 * octahedral layout per probe (VIS_CELL=16). Both atlases are allocated and
 * owned here; the TSL-side applyDDGIShading.ts consumer reads them as
 * StorageTexture handles — that is the only remaining three/webgpu import in
 * the DDGI subsystem.
 */

import { type SceneBvh } from '@vitrum/shared-bvh';
import {
  rebuildProbeBvhFromRestir,
  rebuildProbeBvhFromScene,
  refitProbeTlasBuffersInPlace,
} from './probeUpdateBvhBuffers.js';
import {
  DDGI_MAX_MATERIALS,
  DDGI_MATERIAL_STRIDE_BYTES,
  packDDGIMaterialsFromCoreN,
  packDDGIMaterialsN,
} from './probeUpdateMaterials.js';
import type { MaterialSpec } from '@vitrum/core';
import type { PbrScalarSource } from '../pbrScalars.js';
import type { ProbeGrid } from './probeGrid.js';
import type { DDGILight } from './types.js';
import { isDdgiRestirTlasOnlyRefit, type DdgiRestirBvhSnapshot } from './ddgiRestirBvh.js';
import { makeProbeUpdateRaysWGSL } from './wgsl/probeUpdateRays.wgsl.js';
import { makeProbeUpdateBlendIrrWGSL, makeProbeUpdateBlendVisWGSL } from './wgsl/probeUpdateBlend.wgsl.js';
import { makeProbeUpdateBorderVisWGSL } from './wgsl/probeUpdateBorder.wgsl.js';
import { packDDGIGridParams } from './ddgiGridUbo.js';
import { detectGpu } from '@vitrum/core';
import { RAYS_PER_PROBE } from './ddgiConstants.js';
import { DDGI_PROBE_LIGHTS_BUFFER_BYTES, packDDGIProbeLights } from './probeUpdateLights.js';
import {
  packProbeUpdateBlendParams,
  packProbeUpdateFrameParams,
} from './probeUpdateFrameParams.js';
import { ProbeUpdateAtlasTextureCache } from './probeUpdateAtlasCache.js';
import {
  dispatchProbeUpdateBlendIrrPass,
  dispatchProbeUpdateBlendVisPass,
  dispatchProbeUpdateBorderVisPass,
  dispatchProbeUpdateRaysPass,
  uploadProbeUpdateBorderUbo,
} from './probeUpdateDispatcher.js';
import type { ProbeUpdateGpuState } from './probeUpdateGpuState.js';
import {
  DDGI_BORDER_UBO_BYTES,
  DDGI_FRAME_PARAMS_UBO,
  PROBE_RAY_STRIDE_BYTES,
} from './probeUpdateUbos.js';

/** Options accepted by ProbeUpdatePass constructor. */
export interface ProbeUpdatePassOptions {
  /**
   * When true, exposes DDGI internal state to `window.__DDGI__` for
   * debugging and e2e inspection. Defaults to false.
   *
   * Gate rationale (Q-WA-3): the original source used an unconditional
   * window global assignment. A constructor option is the simplest
   * library-safe pattern — consumers opt in explicitly rather than
   * having a debug global appear unconditionally in production builds.
   */
  debug?: boolean;
  /**
   * Maximum number of distinct materials supported by the DDGI probe pass.
   * Injected as a compile-time constant into the `probeUpdateRays.wgsl`
   * shader (`array<MaterialEntry, N>`). Must match the `materialsBuf` size
   * allocated at init (M × DDGI_MATERIAL_STRIDE_BYTES bytes).
   *
   * Defaults to 64. Raise for scenes with more unique materials.
   * Do NOT raise above WebGPU's uniform-buffer array limit (~4096).
   *
   * @since Sprint 16 (M9 audit remediation)
   */
  maxMaterials?: number;
}

export class ProbeUpdatePass {
  private _bvh:  SceneBvh;
  private _grid: ProbeGrid;
  private _gpu:  ProbeUpdateGpuState | null = null;
  private _atlasCache = new ProbeUpdateAtlasTextureCache();
  /** When set, probe rays use ReSTIR buffers (PR-5.1) instead of SceneBvh rebuild. */
  private _restirSnapshot: DdgiRestirBvhSnapshot | null = null;
  private _lastBvhVersion = -1;
  private _lastBlasVersion = -1;
  private _lastTlasVersion = -1;
  private _frameIndex = 0;
  private _maxProbes = 0;
  private _lights: DDGILight[] = [];
  private _debug: boolean;
  // Guard: set to true on the first call to init() so a failed GPU init
  // (which leaves _gpu null) does not re-issue navigator.gpu.requestAdapter
  // on every subsequent runFrame call. Pattern matches DDGI._inited / _gpuOk.
  private _initAttempted = false;
  // Multiplier applied to every light's intensity when packing the
  // probe-update light UBO. Defaults to 1 so unrelated callers keep the
  // original behaviour. The hybrid pipeline calls setSunIntensityMultiplier(5.0)
  // so DDGI's per-probe Le matches the same primaryLightIntensity used by
  // shade.wgsl — without this, DDGI runs at 1/5 the magnitude of Lo_emit
  // and walls render dark.
  private _sunIntensityMul = 1;

  // Sky tint and irradiance scale used by sampleSkyColor() in
  // probeUpdateRays.wgsl for miss rays. Defaults replicate the legacy
  // hardcoded gradient midpoint so Cornell results are unchanged.
  // (B2 audit remediation — previously hardcoded in WGSL).
  private _skyTint: [number, number, number] = [0.4, 0.6, 1.0];
  private _skyIrradiance = 2.0;

  // 2026-05-18 sweep — glass-transmission perceptual mix used inside
  // probeUpdateRays. Cornell-tuned default 0.7 preserves current behaviour;
  // hosts override via HybridEngineOptions.glassMixScale.
  private _glassMixScale = 0.7;

  // H46-A — DDGI indirect-feedback gate (maxBounces semantics for this regime).
  // true (default) = fold the previous-frame irradiance atlas into the bounce
  // radiance (infinite-bounce diffuse EMA; maxBounces >= 2). false = direct-only
  // probes (maxBounces == 1). Set from HybridEngine._cfg.maxBounces.
  private _indirectFeedback = true;

  // Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
  // When hasEnv=true, _uploadFrameParams passes the env-map view + rotation/intensity
  // into the WGSL FrameParams; sampleSkyColor() samples the equirect map on miss.
  // Default false → procedural gradient (byte-identical for non-HDRI scenes).
  private _hasEnv      = false;
  private _envRotationY = 0;
  private _envIntensity = 0;
  // Optional externally-provided env map view. When null, a 1×1 placeholder is
  // created at init time and destroyed on dispose.
  private _envMapView:    GPUTextureView | null = null;
  private _envSampler:    GPUSampler | null = null;

  // Phase-0 productization — DDGI round-robin probe-update divisor. The ray
  // pass + the blend pass MUST agree on `probesPerFrame = ceil(total/N)`, so
  // both pack functions read this single field. Default 4 reproduces the
  // historical hardcoded `/4` (a 4-frame full-grid update cycle). Higher
  // values (8/16) update fewer probes per frame for the medium/low presets.
  private _probeUpdateDivisor = 4;

  // H16 — when true, the next _uploadBlendParams call uploads hysteresis=0.0
  // (full replace) instead of the steady-state 0.97, then clears this flag.
  // Set by requestFullBlend() (called from DDGI.invalidateProbeCache()) so
  // lighting changes converge in ONE stride window rather than hundreds of frames.
  private _pendingFullBlend = false;

  // Max materials for the WGSL compile-time array size (M9 audit remediation).
  private _ddgiMaxMaterials: number;

  // H18 Stage 2 — packed EmitterTri array for area-emitter NEE in the probe kernel.
  // Each EmitterTri is 80 bytes (5 × vec4f) matching the RC probeRayCast layout.
  // Updated via setEmitterTris() from HybridEngine after BVH rebuild.
  private _emitterTrisData: Float32Array = new Float32Array(0);
  private _emitterTrisCount = 0;

  constructor(bvh: SceneBvh, grid: ProbeGrid, opts: ProbeUpdatePassOptions = {}) {
    this._bvh  = bvh;
    this._grid = grid;
    this._debug = opts.debug ?? false;
    this._ddgiMaxMaterials = opts.maxMaterials ?? DDGI_MAX_MATERIALS;
    if (this._ddgiMaxMaterials < 1) {
      console.warn(`[DDGI] ProbeUpdatePass: maxMaterials=${this._ddgiMaxMaterials} is invalid; clamping to 1.`);
      this._ddgiMaxMaterials = 1;
    }
  }

  /** PR-5.1 — share ReSTIR scene buffers; pass `null` to fall back to SceneBvh. */
  setRestirBvhSnapshot(snapshot: DdgiRestirBvhSnapshot | null): void {
    this._restirSnapshot = snapshot;
  }

  setLights(lights: DDGILight[]): void {
    this._lights = lights;
  }

  /** Multiplier on the sun's stored intensity. Hybrid pipeline
   *  calls this with primaryLightIntensity (5.0) so DDGI's bake of
   *  the sun's Le matches shade.wgsl's Lo_emit. */
  setSunIntensityMultiplier(mul: number): void {
    this._sunIntensityMul = mul;
  }

  /**
   * Override the sky tint and irradiance scale used by probe miss-rays.
   * Written into {@link FrameParams} at offsets 32–44 so `sampleSkyColor`
   * in WGSL reads scene-specific values rather than a Cornell-tuned gradient.
   *
   * Defaults: tint=(0.4,0.6,1.0) and irradiance=2.0 (matches the former
   * hardcoded WGSL values — no regression for Cornell builds).
   *
   * @param tint      Linear-sRGB sky colour (HDR; values above 1.0 are valid).
   * @param irradiance Scalar multiplier on top of the tint. At 1.0 the tint
   *                   is used as-is. At 2.0 the sky is twice as bright as a
   *                   unit-white sky, which suits open-outdoor scenes.
   */
  setSkyParams(tint: [number, number, number], irradiance: number): void {
    this._skyTint = tint;
    this._skyIrradiance = irradiance;
  }

  /**
   * Override the glass-transmission perceptual mix scale used inside
   * probeUpdateRays when a probe ray hits glass. Written into FrameParams
   * as `glassMixScale`. Cornell-tuned default 0.7 leaves 30 % of room
   * radiance on transmission=1 glass; raise toward 1.0 for fully glass-
   * dominated transmission scenes (the indirect bounce becomes the sky
   * tint entirely) or lower toward 0 to suppress sky tinting through glass.
   */
  setGlassMixScale(value: number): void {
    this._glassMixScale = value;
  }

  /**
   * H46-A — set the DDGI indirect-feedback gate from the engine's `maxBounces`.
   *
   * `maxBounces == 1` ⇒ `false` (direct-only probes: each probe carries one
   * bounce of direct light, no infinite-bounce EMA). `maxBounces >= 2` ⇒ `true`
   * (the default multi-bounce diffuse equilibrium). This is the only control
   * surface `maxBounces` has on the walkaround stack: it is NOT a path-tracer
   * bounce cap — the realtime DDGI/ReSTIR passes have a fixed pass-graph budget.
   */
  setIndirectFeedback(enabled: boolean): void {
    this._indirectFeedback = enabled;
  }

  /**
   * Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
   *
   * Supply the equirect env-map texture view and matching sampler so probe
   * miss-rays sample the actual HDRI instead of the procedural sky gradient.
   *
   * Call with `hasEnv = false` (or omit the call entirely) to keep the
   * procedural gradient — byte-identical to pre-Wave-4 behaviour for scenes
   * without an HDRI.
   *
   * The texture view must remain valid for the lifetime of the ProbeUpdatePass
   * (or until the next `setEnvironment` call). ProbeUpdatePass does NOT take
   * ownership of the view — the caller is responsible for its lifetime (the
   * pipeline's BvhBufferHost owns the env textures).
   *
   * If the GPU state is already initialised, the new view / sampler is picked
   * up on the NEXT `runFrame` call (the bind group is rebuilt each frame so
   * there is no explicit invalidation step).
   *
   * UV convention: matches `environmentSample.wgsl envRadiance` exactly —
   *   lookupDir = RY(-rotationY) · worldDir   [H6 world→map]
   *   u = fract(atan2(z,x)/(2π) + 0.5),  v = clamp(acos(y)/π, 0, 1)
   *
   * @param view       GPUTextureView for the rgba16float equirect radiance map.
   * @param sampler    GPUSampler (clamp-to-edge, linear) for the same texture.
   *                   Pass null to reuse the pass's internal linearSampler.
   * @param rotationY  Y-axis rotation in radians (H6 — CCW dome rotation means
   *                   lookupDir = RY(-rotationY)·worldDir). Pass 0 for no rotation.
   * @param intensity  Radiance multiplier applied to the texel after lookup.
   *                   Matches `envParams.intensity` in `environmentSample.wgsl`.
   * @param hasEnv     `true` to activate the HDRI sample path; `false` to keep
   *                   the procedural gradient (the default).
   */
  setEnvironment(
    view: GPUTextureView | null,
    sampler: GPUSampler | null,
    rotationY: number,
    intensity: number,
    hasEnv: boolean,
  ): void {
    this._hasEnv       = hasEnv;
    this._envRotationY = rotationY;
    this._envIntensity = intensity;
    this._envMapView   = view;
    this._envSampler   = sampler;
    // If GPU is already up, update the env fields in the GpuState immediately
    // so the next runFrame bind-group creation picks them up without a re-init.
    if (this._gpu) {
      this._syncEnvViewsToGpu();
    }
  }

  /**
   * H18 Stage 2 — supply rect/disc area-emitter triangles for per-probe NEE.
   *
   * Called by HybridEngine after each BVH rebuild (setScene / updateEmitter).
   * The emitter layout matches the RC probeRayCast EmitterTri struct
   * (5 × vec4f = 80 bytes per tri):
   *   [0..2]  vA.xyz  + pad
   *   [4..6]  vB.xyz  + pad
   *   [8..10] vC.xyz  + pad
   *   [12..14] normal.xyz + area
   *   [16..18] Le.rgb + pad
   *
   * Pass an empty array (or omit the call) for sun-only scenes — the shader
   * guards on emitterCount == 0 so those scenes are byte-identical with the
   * pre-H18 path.
   *
   * @param tris Packed 80-byte-stride Float32Array (20 floats per emitter).
   * @param count Number of emitter triangles in `tris`.
   */
  setEmitterTris(tris: Float32Array, count: number): void {
    this._emitterTrisData = tris;
    this._emitterTrisCount = count;
    // If the GPU state is already initialised, re-upload immediately.
    if (this._gpu) {
      this._uploadEmitterTris(this._gpu.device);
    }
  }

  /**
   * Phase-0 productization — set the DDGI round-robin probe-update divisor
   * (`probesPerFrame = ceil(totalProbes / divisor)`). Higher ⇒ fewer probes
   * updated per frame ⇒ cheaper per-frame cost but slower GI response to
   * lighting changes. Default 4 (the historical hardcoded value). The ray pass
   * and the blend pass both read this so their coverage stays in lockstep.
   * Clamped to ≥ 1 (a divisor < 1 would request more probes than exist).
   */
  setProbeUpdateDivisor(divisor: number): void {
    this._probeUpdateDivisor = Math.max(1, Math.floor(divisor));
  }

  /**
   * H16 — request a full-replace blend on the next runFrame call.
   *
   * Sets `_pendingFullBlend`, which makes `_uploadBlendParams` write
   * `hysteresis = 0.0` (EMA weight = 0 → full replace of every texel)
   * for exactly one frame, then reverts to the steady-state 0.97.
   * Called by `DDGI.invalidateProbeCache()` so lighting changes converge
   * in a single stride window rather than fading over hundreds of frames.
   */
  requestFullBlend(): void {
    this._pendingFullBlend = true;
  }

  /**
   * Initialize GPU resources. Returns false if WebGPU is unavailable.
   * Tries the renderer's WebGPU backend first; falls back to navigator.gpu.
   */
  async init(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }): Promise<boolean> {
    this._initAttempted = true;
    // Hardware-GPU gate. detectGpu() publishes window.__WG__ BEFORE we
    // touch the device, so e2e validation can read the flag even if we
    // refuse to proceed. SwiftShader (Chromium's software rasterizer)
    // compiles WGSL but produces low-chroma "almost passes" — fail-fast
    // here so validation rounds never silently mistake software output
    // for hardware-GPU output.
    const gpu = await detectGpu();
    if (gpu.isWebGPU && gpu.adapterKind === 'swiftshader') {
      console.error(
        `[DDGI] SwiftShader detected (vendor='${gpu.adapterVendor}', architecture='${gpu.adapterArchitecture}'). ` +
        `Refusing to initialize DDGI on software rasterizer. Launch Chrome with hardware GPU enabled to validate DDGI output.`,
      );
      return false;
    }

    // Try renderer's backend first (three.js WebGPURenderer).
    let device: GPUDevice | null = null;
    const backend = renderer.backend;
    if (backend?.isWebGPUBackend && backend.device) {
      device = backend.device;
    } else if (typeof navigator !== 'undefined' && navigator.gpu) {
      // Direct WebGPU path — works when three.js uses WebGLRenderer but
      // the browser exposes WebGPU (Chromium with --enable-unsafe-webgpu).
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          device = await adapter.requestDevice();
        }
      } catch (e) {
        console.warn('[DDGI] navigator.gpu.requestAdapter failed:', e);
      }
    }
    if (!device) {
      console.warn('[DDGI] WebGPU not available (no renderer backend and no navigator.gpu).');
      return false;
    }

    // Compile shaders.
    let raysPipeline: GPUComputePipeline;
    let blendIrrPipeline: GPUComputePipeline;
    let blendVisPipeline: GPUComputePipeline;
    let borderVisPipeline: GPUComputePipeline;   // irradiance is SH (seam-free) — no border pass
    try {
      // M9: compile with the host-specified material array size so scenes with
      // more than 64 materials don't overflow the uniform buffer.
      const raysModule = device.createShaderModule({ code: makeProbeUpdateRaysWGSL(this._ddgiMaxMaterials) });
      raysPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: raysModule, entryPoint: 'probeUpdateRays' },
      });

      const blendIrrModule = device.createShaderModule({ code: makeProbeUpdateBlendIrrWGSL() });
      blendIrrPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: blendIrrModule, entryPoint: 'probeUpdateBlendIrradiance' },
      });
      const blendVisModule = device.createShaderModule({ code: makeProbeUpdateBlendVisWGSL() });
      blendVisPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: blendVisModule, entryPoint: 'probeUpdateBlendVisibility' },
      });
      // No irradiance border pipeline — SH irradiance is seam-free.
      const borderVisModule = device.createShaderModule({ code: makeProbeUpdateBorderVisWGSL() });
      borderVisPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: borderVisModule, entryPoint: 'probeUpdateBorderVisibility' },
      });
    } catch (e) {
      console.error('[DDGI] Shader compilation failed:', e);
      return false;
    }

    const linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Allocate placeholder buffers (1 element each, replaced on first update).
    const makeBuffer = (size: number, usage: number) =>
      device.createBuffer({ size: Math.max(size, 16), usage });

    const RO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const RW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const UB = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

    // Wave 4 — HDRI into DDGI probe misses: create a 1×1 placeholder env
    // map texture so the bind group is always valid (hasEnv=0 in FrameParams
    // prevents it from ever being sampled when no HDRI is present).
    const TEX_BINDING = 0x04;
    const COPY_DST_TEX = 0x02;
    const placeholderEnvTex = device.createTexture({
      label: 'vitrum.ddgi.env.placeholder',
      size: { width: 1, height: 1, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: TEX_BINDING | COPY_DST_TEX,
    });

    this._gpu = {
      device,
      raysPipeline,
      blendIrrPipeline,
      blendVisPipeline,
      borderVisPipeline,
      irrScratchTex:  null,
      visScratchTex:  null,
      bvhBuf:          makeBuffer(16, RO),
      posBuf:          makeBuffer(12, RO),
      idxBuf:          makeBuffer(12, RO),
      normBuf:         makeBuffer(12, RO),
      matIdBuf:        makeBuffer(4,  RO),
      tlasNodesBuf:    makeBuffer(16, RO),
      tlasInstIdxBuf:  makeBuffer(16, RO),
      tlasBlasRootsBuf: makeBuffer(16, RO),
      tlasW2lBuf:      makeBuffer(16, RO),
      tlasL2wBuf:      makeBuffer(16, RO),
      traceParamsBuf:  makeBuffer(16, UB),
      materialsBuf:    makeBuffer(this._ddgiMaxMaterials * DDGI_MATERIAL_STRIDE_BYTES, UB),
      lightsBuf:       makeBuffer(DDGI_PROBE_LIGHTS_BUFFER_BYTES, UB),
      gridParamsBuf:   makeBuffer(64, UB),
      frameParamsBuf:  makeBuffer(DDGI_FRAME_PARAMS_UBO.sizeBytes, UB),
      blendParamsBuf:  makeBuffer(16, UB),
      borderVisUboBuf: makeBuffer(DDGI_BORDER_UBO_BYTES, UB),
      rayResultsBuf:   makeBuffer(PROBE_RAY_STRIDE_BYTES, RW),
      activeProbesBuf: makeBuffer(4, RO),
      // H18 — placeholder (16 bytes); real data uploaded by setEmitterTris().
      emitterTrisBuf:  makeBuffer(16, RO),
      emitterTrisCount: 0,
      linearSampler,
      // Wave 4 — env map: placeholder initially; real view + sampler wired via
      // setEnvironment() (called from the engine before runFrame).
      envMapView:         placeholderEnvTex.createView(),
      envMapOwnedByPass:  true,
      envMapPlaceholderTex: placeholderEnvTex,
      envSamplerForProbe:  linearSampler,
    };
    // If setEnvironment() was called before init (engine wires env before GPU is
    // ready), apply those values now.
    this._syncEnvViewsToGpu();
    return true;
  }

  /**
   * Run one frame of probe updates.
   *
   * The active-probe set for this frame is `{ i : i ≡ offset (mod stride) }`,
   * i.e. one round-robin stratum of the grid. `stride` is the probe-update
   * divisor chosen by DDGI (default 8; overridden by setProbeUpdateDivisor /
   * the quality preset). `offset` cycles 0…stride-1 across consecutive frames
   * so the whole grid refreshes every `stride` frames.
   * @param renderer  The WebGPU renderer
   * @param offset    Which stratum to update this frame (0 … stride-1)
   * @param stride    Number of update strata (the probe-update divisor)
   */
  async runFrame(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }, offset: number, stride: number): Promise<void> {
    if (!this._gpu) {
      // If init was already attempted and failed (e.g. hard WebGPU-init failure
      // where navigator.gpu.requestAdapter returned null), do not re-issue the
      // adapter request on every subsequent frame.  Pattern mirrors DDGI._inited
      // / _gpuOk so the two classes have consistent guards.
      if (this._initAttempted) return;
      await this.init(renderer);
      if (!this._gpu) return;
    }
    const { device } = this._gpu;
    const snap = this._restirSnapshot;
    const legacyBuffers = snap == null ? this._bvh.buffers : null;
    if (snap == null && legacyBuffers == null) return;
    if (this._grid.dirty) this._grid.allocateAtlases();

    const probeCount = this._grid.probeCount;
    if (probeCount === 0) return;

    if (snap != null) {
      if (snap.contentVersion !== this._lastBvhVersion) {
        const tlasOnly =
          this._gpu != null &&
          isDdgiRestirTlasOnlyRefit(snap, {
            blasContentVersion: this._lastBlasVersion,
            tlasContentVersion: this._lastTlasVersion,
          });
        if (tlasOnly && snap.tlas != null) {
          refitProbeTlasBuffersInPlace(device, this._gpu, snap.tlas);
        } else {
          rebuildProbeBvhFromRestir(device, this._gpu, snap);
        }
        this._lastBvhVersion = snap.contentVersion;
        this._lastBlasVersion = snap.blasContentVersion;
        this._lastTlasVersion = snap.tlasContentVersion;
      }
      this._uploadTraceParams(device, snap);
    } else if (legacyBuffers != null) {
      // H24-C — always rebuild so normal/position updates land in the probe-BVH.
      // The length-based gate was a fragile proxy that missed in-place geometry
      // changes (positions fast-path, normal recompute). Rebuild cost is small
      // relative to probe-ray dispatch; this is the safe default.
      rebuildProbeBvhFromScene(device, this._gpu, legacyBuffers);
      this._uploadTraceParams(device, { bvhMode: 'merged', tlasNodeCount: 0 });
    }

    // Reallocate ray results buffer if probe count changed.
    const maxProbes = probeCount;
    if (maxProbes !== this._maxProbes) {
      this._gpu.rayResultsBuf.destroy();
      this._gpu.rayResultsBuf = device.createBuffer({
        size: maxProbes * RAYS_PER_PROBE * PROBE_RAY_STRIDE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._maxProbes = maxProbes;
    }

    // Determine which probes are active this frame.
    const activeProbes: number[] = [];
    for (let i = offset; i < probeCount; i += stride) {
      activeProbes.push(i);
    }
    if (activeProbes.length === 0) return;

    // Upload active probe indices.
    const activeArr = new Uint32Array(activeProbes);
    const activeSize = activeArr.byteLength;
    if (this._gpu.activeProbesBuf.size < activeSize) {
      this._gpu.activeProbesBuf.destroy();
      this._gpu.activeProbesBuf = device.createBuffer({
        size: activeSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this._gpu.activeProbesBuf, 0, activeArr);

    // Update uniforms. Material source priority (core-first wherever available):
    //   1. PRODUCTION ReSTIR snapshot, core path — `snap.coreMaterials` (deduped
    //      MaterialSpec[], slot-aligned with `snap.materials`/`triMaterialIds`),
    //      filled by the core TLAS build `buildReSTIRSceneBVHForCoreScene`. Pack via
    //      `coreMaterialToMaterialEntry`. This is the production material list
    //      path that mirrors the ReSTIR-DI emitter `46a0078` + standalone DDGI
    //      `15070cd` decouples.
    //   2. core-first STANDALONE path — `SceneBvh.updateFromCore` filled
    //      `legacyBuffers.coreMaterials` (no ReSTIR snapshot present).
    //   3. ReSTIR snapshot structural-material path — `snap.materials`.
    //   4. standalone SceneBvh structural-material path — `legacyBuffers.sourceMaterials`.
    const snapCoreMats = snap?.coreMaterials;
    const legacyCoreMats = legacyBuffers?.materials ?? legacyBuffers?.coreMaterials;
    if (snapCoreMats != null && snapCoreMats.length > 0) {
      this._uploadCoreMaterials(device, snapCoreMats);
    } else if (legacyCoreMats != null && legacyCoreMats.length > 0) {
      this._uploadCoreMaterials(device, legacyCoreMats);
    } else {
      const materials = snap?.materials ?? legacyBuffers?.sourceMaterials;
      this._uploadMaterials(device, [...(materials ?? [])] as PbrScalarSource[]);
    }
    this._uploadLights(device);
    this._uploadEmitterTris(device);  // H18 — area-emitter NEE tris
    this._uploadGridParams(device);
    this._uploadFrameParams(device);
    this._uploadBlendParams(device);

    // Ensure grid atlases are allocated.
    if (!this._grid.irradianceA) this._grid.allocateAtlases();

    // Get/create GPU textures for the atlases.
    const irrReadTex  = this._atlasCache.getOrCreateAtlasTexture(device, this._grid.irradianceReadTex, 'rgba16float');
    const irrWriteTex = this._atlasCache.getOrCreateAtlasTexture(device, this._grid.irradianceWriteTex, 'rgba16float');
    // Visibility atlas: allocated as RGBAFormat (rgba16float) because WebGPU does not
    // support rg16float as a storage texture. The WGSL shader declares rgba16float too.
    const visReadTex  = this._atlasCache.getOrCreateAtlasTexture(device, this._grid.visibilityReadTex, 'rgba16float');
    const visWriteTex = this._atlasCache.getOrCreateAtlasTexture(device, this._grid.visibilityWriteTex, 'rgba16float');

    // Run compute passes.
    const encoder = device.createCommandEncoder();
    const gpu = this._gpu;
    dispatchProbeUpdateRaysPass(encoder, gpu, activeProbes.length, irrReadTex);

    // CRITICAL: copy read→write so inactive probes' cells stay in sync.
    // Without this, the per-frame ping-pong (`swap()` below) combined with
    // round-robin updates (STRIDE=8, only 1/8 of probes active per frame)
    // produces a structural bug: every probe always writes to the SAME
    // atlas (parity-matched by probeIdx % STRIDE × frame % 2), and reads
    // its `prev` from the OTHER atlas which it never writes to → prev=0.
    // Hysteresis 0.97 with prev=0 collapses the EMA to 0.03 × newColor,
    // a 33× attenuation that explains why DDGI atlas peaked at ~0.05
    // despite per-ray red-wall radiance of ~6.0. Copying read→write
    // before blend means inactive probes' cells already hold the latest
    // value (from when they were last active, one full STRIDE cycle ago).
    encoder.copyTextureToTexture(
      { texture: irrReadTex },
      { texture: irrWriteTex },
      { width: irrReadTex.width, height: irrReadTex.height, depthOrArrayLayers: 1 },
    );
    encoder.copyTextureToTexture(
      { texture: visReadTex },
      { texture: visWriteTex },
      { width: visReadTex.width, height: visReadTex.height, depthOrArrayLayers: 1 },
    );

    dispatchProbeUpdateBlendIrrPass(encoder, gpu, activeProbes.length, irrReadTex, irrWriteTex);
    dispatchProbeUpdateBlendVisPass(encoder, gpu, activeProbes.length, visReadTex, visWriteTex);

    // Border fill pass (Item 3 — Majercik 2019 §3.2) — VISIBILITY ONLY.
    //
    // The irradiance atlas now stores L2 SH coefficients (seam-free); receivers
    // sample it at exact texel centres (bilinear collapses to the exact coeff),
    // never across a cell edge, so it needs no octahedral border ring and the
    // irradiance border pass is skipped entirely.
    //
    // The visibility atlas IS still octahedral (sharp depth), so its bordered
    // bilinear reads need the seam-mirror ring. We can't bind the same texture
    // as both `texture_2d` (read) and `texture_storage_2d` (write) in one pass,
    // so use a scratch ping-pong: copy write → scratch, then the border pass
    // reads scratch and writes the border pixels into write.
    const visScratch = this._atlasCache.getOrCreateScratchTexture(device, gpu, visWriteTex, 'vis');
    encoder.copyTextureToTexture(
      { texture: visWriteTex },
      { texture: visScratch },
      { width: visWriteTex.width, height: visWriteTex.height, depthOrArrayLayers: 1 },
    );
    uploadProbeUpdateBorderUbo(device, gpu, this._grid, visWriteTex, 'vis');
    dispatchProbeUpdateBorderVisPass(encoder, gpu, probeCount, visScratch, visWriteTex);

    device.queue.submit([encoder.finish()]);

    // Swap ping-pong atlases.
    this._grid.swap();
    this._frameIndex++;

    // Expose internal state for debug/e2e inspection when opted in.
    if (this._debug && typeof window !== 'undefined') {
      (window as unknown as Record<string, unknown>)['__DDGI__'] = {
        frameIndex: this._frameIndex,
        probeCount,
        activeCount: activeProbes.length,
      };
    }
  }

  private _uploadTraceParams(
    device: GPUDevice,
    params: { bvhMode: 'merged' | 'tlas'; tlasNodeCount: number },
  ): void {
    const u = new Uint32Array([
      params.bvhMode === 'tlas' ? 1 : 0,
      params.tlasNodeCount,
      // H18 Stage 2 — emitter triangle count (WGSL DdgiTraceParams.emitterTriCount).
      // Use the TypeScript-side _emitterTrisCount (not gpu.emitterTrisCount which is
      // updated by _uploadEmitterTris called after this). Both writes go to the GPU
      // queue in the same frame so the shader sees a consistent (bvhMode, emitterTriCount).
      this._emitterTrisCount,
      0,
    ]);
    device.queue.writeBuffer(this._gpu!.traceParamsBuf, 0, u);
  }

  private _uploadMaterials(device: GPUDevice, mats: PbrScalarSource[]): void {
    // M9: runtime warning when scene exceeds the compiled-in cap.
    if (mats.length > this._ddgiMaxMaterials) {
      console.warn(
        `[DDGI] Scene has ${mats.length} materials but ddgiMaxMaterials=${this._ddgiMaxMaterials}. ` +
        `Materials beyond the cap are ignored. Raise ddgiMaxMaterials in HybridEngineOptions to fix.`,
      );
    }
    const buf = packDDGIMaterialsN(mats, this._ddgiMaxMaterials);
    device.queue.writeBuffer(this._gpu!.materialsBuf, 0, buf);
  }

  /** Core-first material upload (THREE-decouple): pack a deduped `MaterialSpec[]`
   *  directly via `coreMaterialToMaterialEntry` + the `emissiveIntensity = 1`
   *  production convention — no `THREE.Material` round-trip. See
   *  `packDDGIMaterialsFromCoreN`. */
  private _uploadCoreMaterials(device: GPUDevice, mats: readonly MaterialSpec[]): void {
    if (mats.length > this._ddgiMaxMaterials) {
      console.warn(
        `[DDGI] Scene has ${mats.length} materials but ddgiMaxMaterials=${this._ddgiMaxMaterials}. ` +
        `Materials beyond the cap are ignored. Raise ddgiMaxMaterials in HybridEngineOptions to fix.`,
      );
    }
    const buf = packDDGIMaterialsFromCoreN(mats, this._ddgiMaxMaterials);
    device.queue.writeBuffer(this._gpu!.materialsBuf, 0, buf);
  }

  private _uploadLights(device: GPUDevice): void {
    const buf = packDDGIProbeLights(this._lights, this._sunIntensityMul);
    device.queue.writeBuffer(this._gpu!.lightsBuf, 0, buf);
  }

  /** H18 Stage 2 — upload the packed emitter-tri array (or keep a dummy if count==0). */
  private _uploadEmitterTris(device: GPUDevice): void {
    const gpu = this._gpu!;
    const count = this._emitterTrisCount;
    const data  = this._emitterTrisData;
    if (count === 0) {
      // Sun-only scene: keep the existing 16-byte dummy; update the count only.
      gpu.emitterTrisCount = 0;
      return;
    }
    const needed = data.byteLength;
    if (gpu.emitterTrisBuf.size < needed) {
      gpu.emitterTrisBuf.destroy();
      gpu.emitterTrisBuf = device.createBuffer({
        size: needed,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(gpu.emitterTrisBuf, 0, data.buffer, data.byteOffset, data.byteLength);
    gpu.emitterTrisCount = count;
  }

  private _uploadGridParams(device: GPUDevice): void {
    // Use the canonical packer shared with HybridEngine — single source
    // for the 64-byte DDGI grid-params UBO layout.
    const buf = packDDGIGridParams(this._grid.params);
    device.queue.writeBuffer(this._gpu!.gridParamsBuf, 0, buf);
  }

  private _uploadFrameParams(device: GPUDevice): void {
    const data = packProbeUpdateFrameParams({
      frameIndex: this._frameIndex,
      totalProbes: this._grid.probeCount,
      skyTint: this._skyTint,
      skyIrradiance: this._skyIrradiance,
      glassMixScale: this._glassMixScale,
      updateDivisor: this._probeUpdateDivisor,
      indirectFeedback: this._indirectFeedback,
      // Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
      hasEnv:       this._hasEnv,
      envRotationY: this._envRotationY,
      envIntensity: this._envIntensity,
    });
    device.queue.writeBuffer(this._gpu!.frameParamsBuf, 0, data);
  }

  /** Wave 4 — sync the external env-map view + sampler into GpuState.
   *  Called after init() (to apply early setEnvironment() calls) and
   *  directly from setEnvironment() (when gpu is already live). */
  private _syncEnvViewsToGpu(): void {
    const gpu = this._gpu;
    if (!gpu) return;
    if (this._envMapView !== null) {
      // External view provided: destroy the pass-owned placeholder (once).
      if (gpu.envMapOwnedByPass && gpu.envMapPlaceholderTex !== null) {
        gpu.envMapPlaceholderTex.destroy();
        gpu.envMapPlaceholderTex = null;
        gpu.envMapOwnedByPass = false;
      }
      gpu.envMapView = this._envMapView;
      gpu.envSamplerForProbe = this._envSampler ?? gpu.linearSampler;
    }
    // When _envMapView is null (setEnvironment with hasEnv=false, or reset),
    // the placeholder is kept as-is — it's already the right view and sampler.
  }

  private _uploadBlendParams(device: GPUDevice): void {
    // Same divisor as the ray pass so the blend coverage matches the rays
    // written this frame (a mismatch would blend uncovered probes).
    // H16 — when _pendingFullBlend is set, use hysteresis=0.0 (EMA weight=0,
    // full replace) for exactly this one frame so a lighting change takes
    // effect immediately instead of fading in over hundreds of frames.
    // The flag is cleared HERE (after the upload) so only one frame fires
    // with hysteresis=0; subsequent frames revert to the steady-state 0.97.
    const hysteresisOverride = this._pendingFullBlend ? 0.0 : undefined;
    this._pendingFullBlend = false;
    const data = packProbeUpdateBlendParams(
      this._grid.probeCount,
      this._probeUpdateDivisor,
      hysteresisOverride,
    );
    device.queue.writeBuffer(this._gpu!.blendParamsBuf, 0, data);
  }

  /**
   * Expose the cached read-side GPUTextures so external consumers
   * (e.g. the hybrid pipeline) can bind them into the ReSTIR shade pass
   * via WalkaroundGPUPipeline.setDDGIInputs(...). Returns null if compute
   * hasn't yet allocated the textures (called before first runFrame).
   *
   * The textures are flagged with TEXTURE_BINDING + STORAGE_BINDING usage
   * (see _getOrCreateAtlasTexture above), so they're directly bindable as
   * `texture_2d<f32>` in the shade pass.
   */
  getReadAtlasGPUTextures(): { irradiance: GPUTexture; visibility: GPUTexture } | null {
    const irrTex = this._grid.irradianceReadTex;
    const visTex = this._grid.visibilityReadTex;
    if (!irrTex || !visTex) return null;
    const irrGpu = this._atlasCache.getCachedAtlas(irrTex);
    const visGpu = this._atlasCache.getCachedAtlas(visTex);
    if (!irrGpu || !visGpu) return null;
    return { irradiance: irrGpu, visibility: visGpu };
  }

  /**
   * Read the converged probe atlases back to CPU (the "cached light field" export).
   * Returns the raw rgba16float texels + atlas dims, or null if the atlases aren't
   * allocated yet. Async (mapAsync).
   */
  async exportAtlasData(device: GPUDevice): Promise<{
    irrW: number; irrH: number; visW: number; visH: number;
    irrData: Uint16Array; visData: Uint16Array;
  } | null> {
    const tex = this.getReadAtlasGPUTextures();
    const irrSlot = this._grid.irradianceReadTex;
    const visSlot = this._grid.visibilityReadTex;
    if (!tex || !irrSlot || !visSlot) return null;
    const irrData = await this.#readbackRgba16f(device, tex.irradiance, irrSlot.width, irrSlot.height);
    const visData = await this.#readbackRgba16f(device, tex.visibility, visSlot.width, visSlot.height);
    return { irrW: irrSlot.width, irrH: irrSlot.height, visW: visSlot.width, visH: visSlot.height, irrData, visData };
  }

  /**
   * Upload previously-exported probe atlases into the read-side textures (the
   * restore). Returns false (no-op) if the atlases aren't allocated or the
   * snapshot's dims don't match the current grid (a different scene). The restored
   * state seeds the temporal blend, so subsequent frames continue from it.
   */
  importAtlasData(
    device: GPUDevice,
    snap: { irrW: number; irrH: number; visW: number; visH: number; irrData: Uint16Array; visData: Uint16Array },
  ): boolean {
    const tex = this.getReadAtlasGPUTextures();
    const irrSlot = this._grid.irradianceReadTex;
    const visSlot = this._grid.visibilityReadTex;
    if (!tex || !irrSlot || !visSlot) return false;
    if (irrSlot.width !== snap.irrW || irrSlot.height !== snap.irrH ||
        visSlot.width !== snap.visW || visSlot.height !== snap.visH) {
      return false; // grid mismatch — cannot restore into a differently-sized atlas
    }
    this.#uploadRgba16f(device, tex.irradiance, snap.irrW, snap.irrH, snap.irrData);
    this.#uploadRgba16f(device, tex.visibility, snap.visW, snap.visH, snap.visData);
    return true;
  }

  /** copyTextureToBuffer (256-aligned rows) → unpadded Uint16Array of rgba16float. */
  async #readbackRgba16f(device: GPUDevice, tex: GPUTexture, w: number, h: number): Promise<Uint16Array> {
    const unpadded = w * 8; // rgba16float = 8 bytes/texel
    const bytesPerRow = Math.ceil(unpadded / 256) * 256;
    const staging = device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: staging, bytesPerRow, rowsPerImage: h }, [w, h, 1]);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const out = new Uint16Array(w * h * 4);
    const outBytes = new Uint8Array(out.buffer);
    for (let y = 0; y < h; y++) {
      outBytes.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + unpadded), y * unpadded);
    }
    return out;
  }

  /** writeTexture (no row alignment needed) from a tightly-packed rgba16float array. */
  #uploadRgba16f(device: GPUDevice, tex: GPUTexture, w: number, h: number, data: Uint16Array): void {
    device.queue.writeTexture(
      { texture: tex },
      data as Uint16Array<ArrayBuffer>, // our snapshot arrays are always ArrayBuffer-backed
      { offset: 0, bytesPerRow: w * 8, rowsPerImage: h },
      [w, h, 1],
    );
  }

  dispose(): void {
    if (!this._gpu) return;
    const g = this._gpu;
    g.bvhBuf.destroy();
    g.posBuf.destroy();
    g.idxBuf.destroy();
    g.normBuf.destroy();
    g.matIdBuf.destroy();
    // TLAS buffers (allocated in init / rebuildProbeBvhFrom*) — previously
    // omitted, causing a GPU memory leak on dispose. All five are real
    // GPUBuffers from ProbeUpdateBvhGpuBuffers; traceParamsBuf is from
    // ProbeUpdateGpuState.
    g.tlasNodesBuf.destroy();
    g.tlasInstIdxBuf.destroy();
    g.tlasBlasRootsBuf.destroy();
    g.tlasW2lBuf.destroy();
    g.tlasL2wBuf.destroy();
    g.traceParamsBuf.destroy();
    g.materialsBuf.destroy();
    g.lightsBuf.destroy();
    g.emitterTrisBuf.destroy();  // H18
    g.gridParamsBuf.destroy();
    g.frameParamsBuf.destroy();
    g.blendParamsBuf.destroy();
    g.borderVisUboBuf.destroy();
    g.irrScratchTex?.destroy();
    g.visScratchTex?.destroy();
    g.rayResultsBuf.destroy();
    g.activeProbesBuf.destroy();
    // Wave 4 — destroy the pass-owned placeholder env texture (if we own it).
    if (g.envMapOwnedByPass && g.envMapPlaceholderTex !== null) {
      g.envMapPlaceholderTex.destroy();
    }
    this._gpu = null;
    this._atlasCache.dispose();
  }
}
