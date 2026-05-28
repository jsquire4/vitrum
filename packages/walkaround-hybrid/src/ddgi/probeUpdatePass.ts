/**
 * ProbeUpdatePass — DDGI probe update via raw WebGPU compute.
 *
 * Uses the renderer's raw GPUDevice to run two compute passes per frame:
 *  Pass 1 (probeUpdateRays): for each active probe, fire 96 rays via
 *          inline BVH traversal, collect radiance at hit points.
 *  Pass 2 (probeUpdateBlend): blend ray results into octahedral atlas
 *          textures with EWMA temporal hysteresis.
 *
 * This uses raw WebGPU rather than TSL/wgslFn because the compute shader
 * has custom @group/@binding layouts that don't compose naturally with
 * three.js's binding system. The WebGPU backend's `device` property is
 * used directly.
 *
 * three/webgpu coupling: `renderer.backend.device` is accessed directly,
 * but ProbeGrid atlas slots are now backend-agnostic `AtlasTextureSlot`
 * records (see probeGrid.ts). The GPU texture per slot is allocated lazily
 * in `_getOrCreateAtlasTexture` and cached in `_textureCache` keyed on the
 * slot identity. TSL binding (if applyDDGIShading is in use) is the only
 * site still importing from three/webgpu.
 */

import * as THREE from 'three';
import { type SceneBvh } from '@vitrum/shared-bvh';
import {
  rebuildProbeBvhFromRestir,
  rebuildProbeBvhFromScene,
  refitProbeTlasBuffersInPlace,
} from './probeUpdateBvhBuffers.js';
import {
  DDGI_MAX_MATERIALS,
  DDGI_MATERIAL_STRIDE_BYTES,
  packDDGIMaterialsN,
} from './probeUpdateMaterials.js';
import type { ProbeGrid } from './probeGrid.js';
import type { DDGILight } from './types.js';
import { isDdgiRestirTlasOnlyRefit, type DdgiRestirBvhSnapshot } from './ddgiRestirBvh.js';
import { makeProbeUpdateRaysWGSL } from './wgsl/probeUpdateRays.wgsl.js';
import { makeProbeUpdateBlendIrrWGSL, makeProbeUpdateBlendVisWGSL } from './wgsl/probeUpdateBlend.wgsl.js';
import { makeProbeUpdateBorderIrrWGSL, makeProbeUpdateBorderVisWGSL } from './wgsl/probeUpdateBorder.wgsl.js';
import { packDDGIGridParams } from './ddgiGridUbo.js';
import { detectGpu } from '@vitrum/core';
import { RAYS_PER_PROBE } from './ddgiConstants.js';
import { packDDGIProbeLights } from './probeUpdateLights.js';
import {
  packProbeUpdateBlendParams,
  packProbeUpdateFrameParams,
} from './probeUpdateFrameParams.js';
import { ProbeUpdateAtlasTextureCache } from './probeUpdateAtlasCache.js';
import {
  dispatchProbeUpdateBlendIrrPass,
  dispatchProbeUpdateBlendVisPass,
  dispatchProbeUpdateBorderIrrPass,
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

  // Phase-0 productization — DDGI round-robin probe-update divisor. The ray
  // pass + the blend pass MUST agree on `probesPerFrame = ceil(total/N)`, so
  // both pack functions read this single field. Default 4 reproduces the
  // historical hardcoded `/4` (a 4-frame full-grid update cycle). Higher
  // values (8/16) update fewer probes per frame for the medium/low presets.
  private _probeUpdateDivisor = 4;

  // Max materials for the WGSL compile-time array size (M9 audit remediation).
  private _ddgiMaxMaterials: number;

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
   * Initialize GPU resources. Returns false if WebGPU is unavailable.
   * Tries the renderer's WebGPU backend first; falls back to navigator.gpu.
   */
  async init(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }): Promise<boolean> {
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
    let borderIrrPipeline: GPUComputePipeline;
    let borderVisPipeline: GPUComputePipeline;
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
      const borderIrrModule = device.createShaderModule({ code: makeProbeUpdateBorderIrrWGSL() });
      borderIrrPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: borderIrrModule, entryPoint: 'probeUpdateBorderIrradiance' },
      });
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

    this._gpu = {
      device,
      raysPipeline,
      blendIrrPipeline,
      blendVisPipeline,
      borderIrrPipeline,
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
      lightsBuf:       makeBuffer(16 * 80 + 16, UB),
      gridParamsBuf:   makeBuffer(64, UB),
      frameParamsBuf:  makeBuffer(DDGI_FRAME_PARAMS_UBO.sizeBytes, UB),
      blendParamsBuf:  makeBuffer(16, UB),
      borderIrrUboBuf: makeBuffer(DDGI_BORDER_UBO_BYTES, UB),
      borderVisUboBuf: makeBuffer(DDGI_BORDER_UBO_BYTES, UB),
      rayResultsBuf:   makeBuffer(PROBE_RAY_STRIDE_BYTES, RW),
      activeProbesBuf: makeBuffer(4, RO),
      linearSampler,
    };
    return true;
  }

  /**
   * Run one frame of probe updates.
   * @param renderer  The WebGPU renderer
   * @param offset    Which 1/4 of probes to update (0-3)
   * @param stride    Number of update strata (usually 4)
   */
  async runFrame(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }, offset: number, stride: number): Promise<void> {
    if (!this._gpu) {
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
      const bvhVersion = legacyBuffers.bvhNodes.length + legacyBuffers.positions.length;
      if (bvhVersion !== this._lastBvhVersion) {
        rebuildProbeBvhFromScene(device, this._gpu, legacyBuffers);
        this._lastBvhVersion = bvhVersion;
      }
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

    // Update uniforms.
    const materials = snap?.materials ?? legacyBuffers!.materials;
    this._uploadMaterials(device, [...materials]);
    this._uploadLights(device);
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

    // Border fill pass (Item 3 — Majercik 2019 §3.2).
    //
    // After blend, `irrWriteTex` and `visWriteTex` have correct interior pixels
    // but zeroed border pixels. We can't bind the same texture as both
    // `texture_2d` (read) and `texture_storage_2d` (write) in a single pipeline
    // pass, so we use a scratch ping-pong:
    //   1. copy write → scratch (so border pass reads complete interior from scratch)
    //   2. border pass reads from scratch, writes border pixels into write
    //
    // The scratch textures are allocated lazily and cached in `_gpu.irrScratchTex`
    // / `_gpu.visScratchTex`, reused every frame as long as atlas size is stable.
    const irrScratch = this._atlasCache.getOrCreateScratchTexture(device, gpu, irrWriteTex, 'irr');
    const visScratch = this._atlasCache.getOrCreateScratchTexture(device, gpu, visWriteTex, 'vis');
    encoder.copyTextureToTexture(
      { texture: irrWriteTex },
      { texture: irrScratch },
      { width: irrWriteTex.width, height: irrWriteTex.height, depthOrArrayLayers: 1 },
    );
    encoder.copyTextureToTexture(
      { texture: visWriteTex },
      { texture: visScratch },
      { width: visWriteTex.width, height: visWriteTex.height, depthOrArrayLayers: 1 },
    );
    uploadProbeUpdateBorderUbo(device, gpu, this._grid, irrWriteTex, 'irr');
    uploadProbeUpdateBorderUbo(device, gpu, this._grid, visWriteTex, 'vis');
    dispatchProbeUpdateBorderIrrPass(encoder, gpu, probeCount, irrScratch, irrWriteTex);
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
      0,
      0,
    ]);
    device.queue.writeBuffer(this._gpu!.traceParamsBuf, 0, u);
  }

  private _uploadMaterials(device: GPUDevice, mats: THREE.Material[]): void {
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

  private _uploadLights(device: GPUDevice): void {
    const buf = packDDGIProbeLights(this._lights, this._sunIntensityMul);
    device.queue.writeBuffer(this._gpu!.lightsBuf, 0, buf);
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
    });
    device.queue.writeBuffer(this._gpu!.frameParamsBuf, 0, data);
  }

  private _uploadBlendParams(device: GPUDevice): void {
    // Same divisor as the ray pass so the blend coverage matches the rays
    // written this frame (a mismatch would blend uncovered probes).
    const data = packProbeUpdateBlendParams(this._grid.probeCount, this._probeUpdateDivisor);
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

  dispose(): void {
    if (!this._gpu) return;
    const g = this._gpu;
    g.bvhBuf.destroy();
    g.posBuf.destroy();
    g.idxBuf.destroy();
    g.normBuf.destroy();
    g.matIdBuf.destroy();
    g.materialsBuf.destroy();
    g.lightsBuf.destroy();
    g.gridParamsBuf.destroy();
    g.frameParamsBuf.destroy();
    g.blendParamsBuf.destroy();
    g.borderIrrUboBuf.destroy();
    g.borderVisUboBuf.destroy();
    g.irrScratchTex?.destroy();
    g.visScratchTex?.destroy();
    g.rayResultsBuf.destroy();
    g.activeProbesBuf.destroy();
    this._gpu = null;
    this._atlasCache.dispose();
  }
}
