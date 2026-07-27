/**
 * ProbeUpdatePass — DDGI probe update via raw WebGPU compute.
 *
 * Uses the raw GPUDevice to run three compute stages per frame:
 *  Pass 1 (probeUpdateRays): for each active probe, fire 192 rays via
 *          inline BVH traversal, collect radiance at hit points.
 *  Pass 2 (probeClassifyRelocate): update bounded per-probe offsets and
 *          active/inactive classification from those ray records.
 *  Pass 3 (probeUpdateBlend): blend valid ray results into the L2-SH irradiance
 *          atlas and the octahedral visibility atlas with EWMA temporal
 *          hysteresis.
 *
 * Raw WebGPU is used because the compute shaders have custom @group/@binding
 * layouts that the engine manages directly. A GPUDevice supplied by the caller
 * is borrowed. In the standalone navigator fallback only, this pass requests
 * and owns a private device and releases it after all pass-owned resources.
 *
 * The irradiance atlas stores 9-coefficient L2 spherical harmonics in a 3×3
 * texel block per probe (IRR_CELL=3); the visibility atlas uses a 16×16
 * octahedral layout per probe (VIS_CELL=16). Both atlases are allocated and
 * owned here; applyDDGIShading.ts reads them via the atlas-slot WeakMap
 * pattern (ProbeUpdateAtlasTextureCache / ProbeGrid).
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
  assertDDGIMaterialCapacity,
  packDDGIMaterialsFromCoreN,
  packDDGIMaterialsN,
} from './probeUpdateMaterials.js';
import { buildAliasTable, luminance } from '@vitrum/shared-samplers';
import type { EngineWarning, MaterialSpec } from '@vitrum/core';
import type { PbrScalarSource } from '../pbrScalars.js';
import { rethrowWithSceneMutationCleanup } from '../SceneMutationTransaction.js';
import type { AtlasTextureSlot, ProbeGrid } from './probeGrid.js';
import { snapshotDdgiLights, type DDGILight } from './types.js';
import { isRestirTlasOnlyRefit, type RestirBvhSnapshot } from '../restir/restirBvhSnapshot.js';
import { makeProbeUpdateRaysWGSL } from './wgsl/probeUpdateRays.wgsl.js';
import { PROBE_CLASSIFY_RELOCATE_WGSL } from './wgsl/probeClassifyRelocate.wgsl.js';
import { makeProbeUpdateBlendIrrWGSL, makeProbeUpdateBlendVisWGSL } from './wgsl/probeUpdateBlend.wgsl.js';
import { makeProbeUpdateBorderVisWGSL } from './wgsl/probeUpdateBorder.wgsl.js';
import { packDDGIGridParams } from './ddgiGridUbo.js';
import { RAYS_PER_PROBE } from './ddgiConstants.js';
import { DDGI_PROBE_LIGHTS_BUFFER_BYTES, packDDGIProbeLights } from './probeUpdateLights.js';
import {
  uploadMaterialTextureAtlas,
  type MaterialTextureAtlasPayload,
} from '../pipeline/materialTextureAtlas.js';
import { uploadTangentTexture } from '../pipeline/bvhTangentTexture.js';
import { uploadVertexColorTexture } from '../pipeline/bvhVertexColorTexture.js';
import {
  packProbeUpdateBlendParams,
  packProbeUpdateFrameParams,
} from './probeUpdateFrameParams.js';
import { ProbeUpdateAtlasTextureCache } from './probeUpdateAtlasCache.js';
import {
  copyProbeIrradianceAndPackedStateForward,
  dispatchProbeUpdateBlendIrrPass,
  dispatchProbeUpdateBlendVisPass,
  dispatchProbeUpdateBorderVisPass,
  dispatchProbeClassifyRelocatePass,
  dispatchProbeUpdateRaysPass,
  uploadProbeUpdateBorderUbo,
} from './probeUpdateDispatcher.js';
import type { ProbeUpdateGpuState } from './probeUpdateGpuState.js';
import {
  DDGI_BORDER_UBO_BYTES,
  DDGI_FRAME_PARAMS_UBO,
  PROBE_RAY_STRIDE_BYTES,
} from './probeUpdateUbos.js';
import { EMITTER_TRI_STRIDE_BYTES } from '../restir/emitterList.js';
import type { PreparedSceneMutation } from '../SceneMutationTransaction.js';
import {
  assertDdgiBoolean,
  assertDdgiInteger,
  assertDdgiUnitInterval,
  assertFiniteDdgiNumber,
  assertFiniteDdgiVec3,
  assertNonNegativeDdgiNumber,
  assertPositiveDdgiInteger,
  assertValidDdgiLights,
} from './inputValidation.js';
import {
  isValidProbeStateData,
  readPackedProbeStateFromIrradianceAtlas,
  writePackedProbeStateToIrradianceAtlas,
} from './probeState.js';

const DDGI_PLACEHOLDER_MATERIAL_ATLAS: MaterialTextureAtlasPayload = {
  atlasData: new Float32Array([1, 1, 1, 1]),
  atlasDim: 1,
  atlasLayerCount: 1,
  atlasMipLevelCount: 1,
  gpuSourceLayers: [],
  baseColorMetaData: new Float32Array([-1, 0, 0, 0]),
  baseColorMetaWidth: 1,
  baseColorMetaHeight: 1,
  readableBaseColorLayerCount: 0,
  readableNormalLayerCount: 0,
  readableRoughnessLayerCount: 0,
  readableMetallicLayerCount: 0,
  readableAoLayerCount: 0,
  readableAlphaLayerCount: 0,
  readableEmissiveLayerCount: 0,
  readableTransmissionLayerCount: 0,
  readableLightLayerCount: 0,
  readableSpecularColorLayerCount: 0,
  readableSpecularIntensityLayerCount: 0,
  readableClearcoatLayerCount: 0,
  readableClearcoatRoughnessLayerCount: 0,
  readableClearcoatNormalLayerCount: 0,
  readableSheenColorLayerCount: 0,
  readableSheenRoughnessLayerCount: 0,
  readableAnisotropyLayerCount: 0,
  readableIridescenceLayerCount: 0,
  readableIridescenceThicknessLayerCount: 0,
  readableThicknessLayerCount: 0,
  readableBumpLayerCount: 0,
  diagnostics: [],
};

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
  /**
   * Optional structured warning sink. When omitted, standalone pass callers keep
   * the historical console.warn behavior for DDGI probe-light/material caps.
   */
  onWarning?: (warning: EngineWarning) => void;
}

/** Transaction-safe snapshot of a full-blend invalidation generation. */
export interface ProbeFullBlendState {
  readonly generation: number;
  readonly stride: number;
  readonly pendingStrata: readonly number[];
}

/**
 * Finish and submit one probe-update command buffer, then publish CPU-side
 * state. Keeping the publication callback behind both fallible operations makes
 * retries exact: an encoder or queue failure cannot consume invalidation state,
 * advance frame counters, or swap the atlas identities.
 *
 * @internal Exported for deterministic failure-injection tests.
 */
export function submitProbeUpdateCommand(
  encoder: Pick<GPUCommandEncoder, 'finish'>,
  queue: Pick<GPUQueue, 'submit'>,
  publishAccepted: () => void,
): void {
  const commandBuffer = encoder.finish();
  queue.submit([commandBuffer]);
  publishAccepted();
}

function destroyProbeResourceBestEffort(resource: GPUBuffer | GPUTexture | null): void {
  if (resource == null) return;
  try { resource.destroy(); } catch { /* preserve the resource transaction outcome */ }
}

function destroyProbeDeviceBestEffort(device: GPUDevice | null): void {
  if (device == null) return;
  try { device.destroy(); } catch { /* terminal fallback-device teardown */ }
}

function assertEmitterTriPayload(tris: Float32Array, count: number): number {
  if (!Number.isSafeInteger(count) || count < 0) {
    throw new RangeError(
      'DDGI emitter triangle count must be a non-negative safe integer.',
    );
  }
  const floatsPerTri = EMITTER_TRI_STRIDE_BYTES / Float32Array.BYTES_PER_ELEMENT;
  const requiredFloats = count * floatsPerTri;
  if (!Number.isSafeInteger(requiredFloats)) {
    throw new RangeError('DDGI emitter triangle payload size is not safe.');
  }
  if (tris.length < requiredFloats) {
    throw new RangeError(
      `DDGI emitter triangle payload contains ${tris.length} floats for ${count} ` +
      `triangles; expected at least ${requiredFloats}.`,
    );
  }
  for (let index = 0; index < requiredFloats; index += 1) {
    assertFiniteDdgiNumber(tris[index]!, `DDGI emitter triangles[${index}]`);
  }
  for (let tri = 0; tri < count; tri += 1) {
    const base = tri * floatsPerTri;
    assertNonNegativeDdgiNumber(
      tris[base + 15]!,
      `DDGI emitter triangles[${tri}].area`,
    );
    for (let channel = 0; channel < 3; channel += 1) {
      assertNonNegativeDdgiNumber(
        tris[base + 16 + channel]!,
        `DDGI emitter triangles[${tri}].radiance[${channel}]`,
      );
    }
  }
  return requiredFloats;
}

function packEmitterTriSamplingPayload(
  tris: Float32Array,
  count: number,
): Float32Array<ArrayBuffer> {
  const recordFloats = assertEmitterTriPayload(tris, count);
  const aliasFloats = count * 4;
  const packed = new Float32Array(recordFloats + aliasFloats);
  packed.set(tris.subarray(0, recordFloats));
  if (count === 0) return packed;

  const floatsPerTri = EMITTER_TRI_STRIDE_BYTES / Float32Array.BYTES_PER_ELEMENT;
  const weights = Array.from({ length: count }, (_, tri) => {
    const base = tri * floatsPerTri;
    return packed[base + 15]! * luminance(
      packed[base + 16]!,
      packed[base + 17]!,
      packed[base + 18]!,
    );
  });
  const alias = buildAliasTable(weights);
  new Uint8Array(packed.buffer).set(
    new Uint8Array(alias.data),
    recordFloats * Float32Array.BYTES_PER_ELEMENT,
  );
  return packed;
}

export class ProbeUpdatePass {
  private _bvh:  SceneBvh;
  private _grid: ProbeGrid;
  private _gpu:  ProbeUpdateGpuState | null = null;
  /** Non-null only for a navigator fallback device requested by this pass. */
  private _ownedDevice: GPUDevice | null = null;
  private _atlasCache = new ProbeUpdateAtlasTextureCache();
  /** When set, probe rays use ReSTIR buffers (PR-5.1) instead of SceneBvh rebuild. */
  private _restirSnapshot: RestirBvhSnapshot | null = null;
  private _lastBvhVersion = -1;
  private _lastBlasVersion = -1;
  private _lastTlasVersion = -1;
  private _lastMaterialVersion = -1;
  private _lastMaterialAtlasPayload: MaterialTextureAtlasPayload | null = null;
  private _lastTangentSource: ArrayBuffer | null = null;
  private _lastVertexColorSource: ArrayBuffer | null = null;
  private _frameIndex = 0;
  private _maxProbes = 0;
  private _lights: DDGILight[] = [];
  private _debug: boolean;
  // Guard: set to true on the first call to init() so a failed GPU init
  // (which leaves _gpu null) does not re-issue navigator.gpu.requestAdapter
  // on every subsequent runFrame call. Pattern matches DDGI._inited / _gpuOk.
  private _initAttempted = false;
  private _initInFlight: Promise<boolean> | null = null;
  private _disposed = false;
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

  // DDGI round-robin probe-update divisor. This directly controls the active
  // probe stratum assembled and dispatched by runFrame.
  private _probeUpdateDivisor = 4;

  // Full-blend invalidation is generation + stratum based. Every stratum in the
  // active round-robin must submit once with hysteresis=0 before the generation
  // is complete; clearing a one-shot boolean after stratum zero leaves the other
  // strata carrying stale lighting indefinitely.
  private _fullBlendGeneration = 0;
  private _fullBlendStride = 1;
  private _pendingFullBlendStrata = new Set<number>();

  // Max materials for the WGSL compile-time array size (M9 audit remediation).
  private _ddgiMaxMaterials: number;
  private readonly _onWarning: ((warning: EngineWarning) => void) | undefined;

  // H18 Stage 2 — packed EmitterTri array for area-emitter NEE in the probe kernel.
  // Each EmitterTri is 80 bytes (5 × vec4f) matching the RC probeRayCast layout.
  // Updated via setEmitterTris() from HybridEngine after BVH rebuild.
  private _emitterTrisData: Float32Array<ArrayBuffer> = new Float32Array(0);
  private _emitterTrisCount = 0;

  constructor(bvh: SceneBvh, grid: ProbeGrid, opts: ProbeUpdatePassOptions = {}) {
    if (opts.debug !== undefined) {
      assertDdgiBoolean(opts.debug, 'DDGI probe pass debug');
    }
    this._bvh  = bvh;
    this._grid = grid;
    this._debug = opts.debug ?? false;
    this._onWarning = opts.onWarning;
    this._ddgiMaxMaterials = opts.maxMaterials ?? DDGI_MAX_MATERIALS;
    if (
      !Number.isFinite(this._ddgiMaxMaterials) ||
      !Number.isInteger(this._ddgiMaxMaterials)
    ) {
      throw new RangeError('DDGI maxMaterials must be a finite integer.');
    }
    if (this._ddgiMaxMaterials < 1) {
      throw new RangeError('DDGI maxMaterials must be at least 1.');
    }
  }

  private _warn(warning: EngineWarning): void {
    if (this._onWarning) {
      this._onWarning(warning);
      return;
    }
    console.warn(warning.message);
  }

  /** PR-5.1 — share ReSTIR scene buffers; pass `null` to fall back to SceneBvh. */
  setRestirBvhSnapshot(snapshot: RestirBvhSnapshot | null): void {
    if (snapshot != null) {
      const materialCount = snapshot.coreMaterials.length > 0
        ? snapshot.coreMaterials.length
        : snapshot.materials.length;
      assertDDGIMaterialCapacity(materialCount, this._ddgiMaxMaterials);
    }
    if (snapshot == null && this._restirSnapshot != null) {
      this._lastBvhVersion = -1;
      this._lastBlasVersion = -1;
      this._lastTlasVersion = -1;
      this._lastMaterialVersion = -1;
    }
    this._restirSnapshot = snapshot;
  }

  setLights(lights: DDGILight[]): void {
    assertValidDdgiLights(lights);
    this._lights = snapshotDdgiLights(lights);
  }

  /** Multiplier on the sun's stored intensity. Hybrid pipeline
   *  calls this with primaryLightIntensity (5.0) so DDGI's bake of
   *  the sun's Le matches shade.wgsl's Lo_emit. */
  setSunIntensityMultiplier(mul: number): void {
    assertNonNegativeDdgiNumber(mul, 'DDGI sun intensity multiplier');
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
    assertFiniteDdgiVec3(tint, 'DDGI sky tint');
    tint.forEach((channel, index) => {
      assertNonNegativeDdgiNumber(channel, `DDGI sky tint[${index}]`);
    });
    assertNonNegativeDdgiNumber(irradiance, 'DDGI sky irradiance');
    this._skyTint = [...tint];
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
    assertDdgiUnitInterval(value, 'DDGI glass mix scale');
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
    assertDdgiBoolean(enabled, 'DDGI indirect feedback');
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
    assertFiniteDdgiNumber(rotationY, 'DDGI environment rotation');
    assertNonNegativeDdgiNumber(intensity, 'DDGI environment intensity');
    assertDdgiBoolean(hasEnv, 'DDGI environment hasEnv');
    if (hasEnv && view == null) {
      throw new TypeError('DDGI environment view is required when hasEnv is true.');
    }
    const previous = {
      hasEnv: this._hasEnv,
      rotationY: this._envRotationY,
      intensity: this._envIntensity,
      view: this._envMapView,
      sampler: this._envSampler,
    };
    this._hasEnv       = hasEnv;
    this._envRotationY = rotationY;
    this._envIntensity = intensity;
    this._envMapView   = view;
    this._envSampler   = sampler;
    try {
      // If GPU is already up, update the env fields in the GpuState immediately
      // so the next runFrame bind-group creation picks them up without a re-init.
      if (this._gpu) this._syncEnvViewsToGpu();
    } catch (error) {
      // _syncEnvViewsToGpu prepares any owned placeholder before publication,
      // so restoring the host mirrors is enough to leave CPU and GPU state on
      // the same accepted generation.
      this._hasEnv = previous.hasEnv;
      this._envRotationY = previous.rotationY;
      this._envIntensity = previous.intensity;
      this._envMapView = previous.view;
      this._envSampler = previous.sampler;
      throw error;
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
    const nextTris = packEmitterTriSamplingPayload(tris, count);
    if (this._gpu == null) {
      // The pass owns the deferred upload payload. Never retain a host-owned
      // ArrayBuffer/SharedArrayBuffer alias across the setter boundary.
      this._emitterTrisData = nextTris;
      this._emitterTrisCount = count;
      return;
    }

    // A live pass must not publish its CPU payload before the replacement GPU
    // buffer is completely allocated and populated. Reuse the same prepared
    // transaction as the engine-level lighting mutation.
    const mutation = this.prepareLightingMutation(
      this._lights,
      this._sunIntensityMul,
      tris,
      count,
    );
    try {
      mutation.commit();
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        [() => mutation.rollback()],
        'DDGI emitter publication failed and rollback also failed',
      );
    }
    mutation.finalize();
  }

  /**
   * Stage a complete DDGI lighting replacement without touching the live
   * light list or emitter-triangle GPU buffer.
   *
   * A fresh emitter buffer is allocated and populated at preparation time.
   * Commit is therefore pointer/CPU-state publication only; rollback can
   * restore the previous buffer and host mirrors without issuing GPU work.
   */
  prepareLightingMutation(
    lights: readonly DDGILight[],
    sunIntensityMultiplier: number,
    tris: Float32Array,
    count: number,
  ): PreparedSceneMutation {
    assertValidDdgiLights(lights);
    assertNonNegativeDdgiNumber(
      sunIntensityMultiplier,
      'DDGI sun intensity multiplier',
    );
    const nextLights = snapshotDdgiLights(lights);
    const nextTris = packEmitterTriSamplingPayload(tris, count);
    const gpu = this._gpu;
    const previousGpuBuffer = gpu?.emitterTrisBuf ?? null;
    const previousGpuCount = gpu?.emitterTrisCount ?? 0;
    let candidateGpuBuffer: GPUBuffer | null = null;

    if (gpu != null && count > 0) {
      const byteLength = nextTris.byteLength;
      const candidate = gpu.device.createBuffer({
        label: 'ddgi.emitter-tris.candidate',
        size: Math.max(EMITTER_TRI_STRIDE_BYTES, byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      if (candidate === previousGpuBuffer) {
        throw new Error('DDGI emitter candidate aliases the live GPU buffer.');
      }
      try {
        new Uint8Array(candidate.getMappedRange()).set(
          new Uint8Array(nextTris.buffer, nextTris.byteOffset, nextTris.byteLength),
        );
        candidate.unmap();
        candidateGpuBuffer = candidate;
      } catch (error) {
        rethrowWithSceneMutationCleanup(
          error,
          [() => candidate.destroy()],
          'DDGI emitter preparation failed and candidate cleanup also failed',
        );
      }
    }

    const previousLights = this._lights;
    const previousSunIntensityMultiplier = this._sunIntensityMul;
    const previousTris = this._emitterTrisData;
    const previousCount = this._emitterTrisCount;
    let committed = false;
    let closed = false;

    const destroyCandidate = (): void => {
      if (candidateGpuBuffer == null) return;
      const candidate = candidateGpuBuffer;
      candidateGpuBuffer = null;
      candidate.destroy();
    };

    return {
      commit: () => {
        if (closed || committed) return;
        this._lights = nextLights;
        this._sunIntensityMul = sunIntensityMultiplier;
        this._emitterTrisData = nextTris;
        this._emitterTrisCount = count;
        if (gpu != null) {
          if (candidateGpuBuffer != null) gpu.emitterTrisBuf = candidateGpuBuffer;
          gpu.emitterTrisCount = count;
        }
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        closed = true;
        if (committed) {
          this._lights = previousLights;
          this._sunIntensityMul = previousSunIntensityMultiplier;
          this._emitterTrisData = previousTris;
          this._emitterTrisCount = previousCount;
          if (gpu != null) {
            gpu.emitterTrisBuf = previousGpuBuffer!;
            gpu.emitterTrisCount = previousGpuCount;
          }
        }
        destroyCandidate();
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        if (committed && candidateGpuBuffer != null) {
          candidateGpuBuffer = null;
          previousGpuBuffer?.destroy();
        } else {
          destroyCandidate();
        }
      },
    };
  }

  /**
   * Set the DDGI round-robin probe-update divisor. Higher ⇒ fewer probes
   * updated per frame ⇒ cheaper per-frame cost but slower GI response to
   * lighting changes. Default 4 (the historical hardcoded value). The active
   * stratum built by runFrame is the single source of dispatch coverage.
   */
  setProbeUpdateDivisor(divisor: number): void {
    assertPositiveDdgiInteger(divisor, 'DDGI probe update divisor');
    const next = divisor;
    if (next === this._probeUpdateDivisor) return;
    this._probeUpdateDivisor = next;
    if (this._pendingFullBlendStrata.size > 0) {
      this.requestFullBlend(next);
    }
  }

  /**
   * Request a full-replace blend for every round-robin stratum.
   *
   * A new generation supersedes any older pending set. A stratum is removed
   * only after its command buffer is accepted by `queue.submit`; encoder finish
   * or submit failures leave it pending for an exact retry.
   */
  requestFullBlend(stride = this._probeUpdateDivisor): void {
    assertPositiveDdgiInteger(stride, 'DDGI full-blend stride');
    const normalizedStride = stride;
    this._fullBlendGeneration = (this._fullBlendGeneration + 1) >>> 0;
    if (this._fullBlendGeneration === 0) this._fullBlendGeneration = 1;
    this._fullBlendStride = normalizedStride;
    this._pendingFullBlendStrata = new Set(
      Array.from({ length: normalizedStride }, (_, index) => index),
    );
  }

  get pendingFullBlend(): boolean {
    return this._pendingFullBlendStrata.size > 0;
  }

  setPendingFullBlend(value: boolean): void {
    if (value) {
      this.requestFullBlend(this._probeUpdateDivisor);
    } else {
      this._pendingFullBlendStrata.clear();
    }
  }

  get pendingFullBlendCount(): number {
    return this._pendingFullBlendStrata.size;
  }

  get fullBlendGeneration(): number {
    return this._fullBlendGeneration;
  }

  captureFullBlendState(): ProbeFullBlendState {
    return {
      generation: this._fullBlendGeneration,
      stride: this._fullBlendStride,
      pendingStrata: [...this._pendingFullBlendStrata].sort((a, b) => a - b),
    };
  }

  restoreFullBlendState(state: ProbeFullBlendState): void {
    assertPositiveDdgiInteger(state.stride, 'DDGI full-blend stride');
    if (!Number.isSafeInteger(state.generation) || state.generation < 0) {
      throw new RangeError('DDGI full-blend generation must be a non-negative safe integer.');
    }
    const nextStrata = new Set<number>();
    for (const stratum of state.pendingStrata) {
      assertDdgiInteger(stratum, 'DDGI full-blend stratum');
      if (stratum < 0 || stratum >= state.stride) {
        throw new RangeError('DDGI full-blend stratum must be within the stride.');
      }
      nextStrata.add(stratum);
    }
    this._fullBlendGeneration = state.generation >>> 0;
    this._fullBlendStride = state.stride;
    this._pendingFullBlendStrata = nextStrata;
  }

  /**
   * Initialize GPU resources. Returns false if WebGPU is unavailable.
   * Tries the renderer's WebGPU backend first; falls back to navigator.gpu.
   */
  init(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }): Promise<boolean> {
    if (this._disposed) return Promise.resolve(false);
    if (this._gpu != null) return Promise.resolve(true);
    if (this._initInFlight != null) return this._initInFlight;
    const operation = this._initOwned(renderer);
    const owned = operation.finally(() => {
      if (this._initInFlight === owned) this._initInFlight = null;
    });
    this._initInFlight = owned;
    return owned;
  }

  private async _initOwned(
    renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } },
  ): Promise<boolean> {
    this._initAttempted = true;
    const acquired = await this._acquireDevice(renderer);
    if (acquired == null) return false;
    const { device, ownedByPass } = acquired;
    let published = false;
    try {
      if (this._disposed) return false;
      const pipelines = await this._compilePipelines(device);
      if (!pipelines) {
        // Pipeline compilation can fail transiently during device recovery. Do
        // not poison the pass the way a definitive no-device result does.
        this._initAttempted = false;
        return false;
      }
      if (this._disposed) return false;
      this._allocateResources(device, pipelines);
      if (ownedByPass) this._ownedDevice = device;
      published = true;
      return true;
    } catch (error) {
      // Allocation failure is transient (OOM/device pressure/failure injection),
      // unlike an explicit unsupported-device result. Leave the pass retryable.
      this._initAttempted = false;
      throw error;
    } finally {
      if (ownedByPass && !published) destroyProbeDeviceBestEffort(device);
    }
  }

  /** Acquire a GPUDevice from the renderer backend or navigator.gpu. */
  private async _acquireDevice(
    renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } },
  ): Promise<{ readonly device: GPUDevice; readonly ownedByPass: boolean } | null> {
    // Adapter provenance is a validation-harness concern, not a production
    // capability gate. Any WebGPU-conformant renderer device is accepted.
    // Try the renderer's backend first.
    const backend = renderer.backend;
    if (backend?.isWebGPUBackend && backend.device) {
      return { device: backend.device, ownedByPass: false };
    }
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      // Direct WebGPU path — used when the renderer uses WebGL but the browser
      // exposes WebGPU (Chromium with --enable-unsafe-webgpu).
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          return { device: await adapter.requestDevice(), ownedByPass: true };
        }
      } catch (e) {
        this._warn({
          code: 'walkaround-hybrid.ddgi-request-adapter-failed',
          backend: 'walkaround-hybrid',
          phase: 'renderFrame',
          method: 'ProbeUpdatePass.init',
          message: `[DDGI] navigator.gpu.requestAdapter failed: ${String(e)}`,
          details: { source: 'navigator.gpu', fallback: 'disable-ddgi-probe-update' },
          raw: e,
        });
      }
    }
    this._warn({
      code: 'walkaround-hybrid.ddgi-webgpu-unavailable',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'ProbeUpdatePass.init',
      message: '[DDGI] WebGPU not available (no renderer backend and no navigator.gpu).',
      details: { fallback: 'disable-ddgi-probe-update' },
    });
    return null;
  }

  /** Compile all five compute pipelines. Returns null on compilation failure. */
  private async _compilePipelines(device: GPUDevice): Promise<{
    raysPipeline: GPUComputePipeline;
    classifyRelocatePipeline: GPUComputePipeline;
    blendIrrPipeline: GPUComputePipeline;
    blendVisPipeline: GPUComputePipeline;
    borderVisPipeline: GPUComputePipeline;
  } | null> {
    try {
      // M9: compile with the host-specified material array size so scenes with
      // more than 64 materials don't overflow the uniform buffer.
      const raysModule = device.createShaderModule({ code: makeProbeUpdateRaysWGSL(this._ddgiMaxMaterials) });
      const raysPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: raysModule, entryPoint: 'probeUpdateRays' },
      });

      const classifyRelocateModule = device.createShaderModule({
        code: PROBE_CLASSIFY_RELOCATE_WGSL,
      });
      const classifyRelocatePipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
          module: classifyRelocateModule,
          entryPoint: 'probeClassifyRelocate',
        },
      });

      const blendIrrModule = device.createShaderModule({ code: makeProbeUpdateBlendIrrWGSL() });
      const blendIrrPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: blendIrrModule, entryPoint: 'probeUpdateBlendIrradiance' },
      });
      const blendVisModule = device.createShaderModule({ code: makeProbeUpdateBlendVisWGSL() });
      const blendVisPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: blendVisModule, entryPoint: 'probeUpdateBlendVisibility' },
      });
      // No irradiance border pipeline — SH irradiance is seam-free.
      const borderVisModule = device.createShaderModule({ code: makeProbeUpdateBorderVisWGSL() });
      const borderVisPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: borderVisModule, entryPoint: 'probeUpdateBorderVisibility' },
      });
      return {
        raysPipeline,
        classifyRelocatePipeline,
        blendIrrPipeline,
        blendVisPipeline,
        borderVisPipeline,
      };
    } catch (e) {
      this._warn({
        code: 'walkaround-hybrid.ddgi-shader-compilation-failed',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'ProbeUpdatePass.init',
        message: `[DDGI] Shader compilation failed: ${String(e)}`,
        details: { fallback: 'disable-ddgi-probe-update' },
        raw: e,
      });
      return null;
    }
  }

  /** Allocate all GPU resources and wire this._gpu. */
  private _allocateResources(
    device: GPUDevice,
    pipelines: {
      raysPipeline: GPUComputePipeline;
      classifyRelocatePipeline: GPUComputePipeline;
      blendIrrPipeline: GPUComputePipeline;
      blendVisPipeline: GPUComputePipeline;
      borderVisPipeline: GPUComputePipeline;
    },
  ): void {
    const candidateBuffers = new Set<GPUBuffer>();
    const candidateTextures = new Set<GPUTexture>();
    const cleanupCandidates = (): void => {
      for (const buffer of candidateBuffers) destroyProbeResourceBestEffort(buffer);
      for (const texture of candidateTextures) destroyProbeResourceBestEffort(texture);
    };
    const makeBuffer = (label: string, size: number, usage: number): GPUBuffer => {
      const buffer = device.createBuffer({ label, size: Math.max(size, 16), usage });
      candidateBuffers.add(buffer);
      return buffer;
    };
    const registerTexture = (texture: GPUTexture): GPUTexture => {
      candidateTextures.add(texture);
      return texture;
    };

    try {
      const linearSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });

      // Allocate placeholder buffers (1 element each, replaced on first update).
      // WebGPU validates unsized storage-array bindings against at least one
      // element of the declared WGSL type. BVHNode is 32 bytes, so a generic
      // 16-byte placeholder is invalid for array<BVHNode> even when TLAS is off.
      const BVH_NODE_PLACEHOLDER_BYTES = 32;
      const VEC4_PLACEHOLDER_BYTES = 16;
      const U32_PLACEHOLDER_BYTES = 16;
      const RO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const RW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const UB = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

      let placeholderEnvTex: GPUTexture | null = null;
      let envMapView = this._envMapView;
      if (envMapView == null) {
        placeholderEnvTex = registerTexture(device.createTexture({
          label: 'vitrum.ddgi.env.placeholder',
          size: { width: 1, height: 1, depthOrArrayLayers: 1 },
          format: 'rgba16float',
          usage: 0x04 | 0x02, // TEXTURE_BINDING | COPY_DST
        }));
        envMapView = placeholderEnvTex.createView();
      }

      const placeholderMaterialAtlas =
        uploadMaterialTextureAtlas(device, DDGI_PLACEHOLDER_MATERIAL_ATLAS);
      registerTexture(placeholderMaterialAtlas.atlasTexture);
      registerTexture(placeholderMaterialAtlas.baseColorMetaTexture);
      const placeholderTangents = uploadTangentTexture(device, new Float32Array(4), 1);
      registerTexture(placeholderTangents.texture);
      const placeholderTangentView = placeholderTangents.texture.createView();
      const placeholderVertexColors = uploadVertexColorTexture(
        device,
        new Float32Array([1, 1, 1, 1]),
        1,
      );
      registerTexture(placeholderVertexColors.texture);
      const placeholderVertexColorView = placeholderVertexColors.texture.createView();

      const candidateGpu: ProbeUpdateGpuState = {
        device,
        ...pipelines,
        visScratchTex: null,
        bvhBuf: makeBuffer('ddgi.bvh.nodes.placeholder', BVH_NODE_PLACEHOLDER_BYTES, RO),
        posBuf: makeBuffer('ddgi.bvh.positions.placeholder', VEC4_PLACEHOLDER_BYTES, RO),
        idxBuf: makeBuffer('ddgi.bvh.indices.placeholder', VEC4_PLACEHOLDER_BYTES, RO),
        normBuf: makeBuffer('ddgi.bvh.normals.placeholder', VEC4_PLACEHOLDER_BYTES, RO),
        matIdBuf: makeBuffer('ddgi.bvh.material-ids.placeholder', U32_PLACEHOLDER_BYTES, RO),
        tlasNodesBuf: makeBuffer('ddgi.tlas.nodes.placeholder', BVH_NODE_PLACEHOLDER_BYTES, RO),
        tlasInstIdxBuf: makeBuffer('ddgi.tlas.instance-indices.placeholder', U32_PLACEHOLDER_BYTES, RO),
        tlasBlasRootsBuf: makeBuffer('ddgi.tlas.blas-roots.placeholder', U32_PLACEHOLDER_BYTES, RO),
        tlasW2lBuf: makeBuffer('ddgi.tlas.world-to-local.placeholder', VEC4_PLACEHOLDER_BYTES, RO),
        tlasL2wBuf: makeBuffer('ddgi.tlas.local-to-world.placeholder', VEC4_PLACEHOLDER_BYTES, RO),
        traceParamsBuf: makeBuffer('ddgi.trace-params', 16, UB),
        materialsBuf: makeBuffer(
          'ddgi.materials',
          this._ddgiMaxMaterials * DDGI_MATERIAL_STRIDE_BYTES,
          UB,
        ),
        lightsBuf: makeBuffer('ddgi.lights', DDGI_PROBE_LIGHTS_BUFFER_BYTES, RO),
        lightsCapacityBytes: DDGI_PROBE_LIGHTS_BUFFER_BYTES,
        gridParamsBuf: makeBuffer('ddgi.grid-params', 64, UB),
        frameParamsBuf: makeBuffer('ddgi.frame-params', DDGI_FRAME_PARAMS_UBO.sizeBytes, UB),
        blendParamsBuf: makeBuffer('ddgi.blend-params', 16, UB),
        borderVisUboBuf: makeBuffer('ddgi.border-vis-ubo', DDGI_BORDER_UBO_BYTES, UB),
        rayResultsBuf: makeBuffer('ddgi.ray-results', PROBE_RAY_STRIDE_BYTES, RW),
        activeProbesBuf: makeBuffer('ddgi.active-probes', 4, RO),
        // H18 — one full packed EmitterTri record. The shader statically indexes
        // five vec4 lanes per candidate, so layout:auto validates a 32+ byte
        // storage binding even when traceParams.emitterTriCount is zero.
        emitterTrisBuf: makeBuffer(
          'ddgi.emitter-tris.placeholder',
          EMITTER_TRI_STRIDE_BYTES,
          RO,
        ),
        emitterTrisCount: 0,
        materialTextureAtlas: placeholderMaterialAtlas.atlasTexture,
        materialTextureAtlasView: placeholderMaterialAtlas.atlasTextureView,
        materialTextureAtlasMeta: placeholderMaterialAtlas.baseColorMetaTexture,
        materialTextureAtlasMetaView: placeholderMaterialAtlas.baseColorMetaTextureView,
        bvhTangentTexture: placeholderTangents.texture,
        bvhTangentTextureView: placeholderTangentView,
        bvhVertexColorTexture: placeholderVertexColors.texture,
        bvhVertexColorTextureView: placeholderVertexColorView,
        linearSampler,
        envMapView,
        envMapOwnedByPass: placeholderEnvTex != null,
        envMapPlaceholderTex: placeholderEnvTex,
        envSamplerForProbe: this._envSampler ?? linearSampler,
      };

      // Publish only after every buffer, texture upload, and view creation has
      // completed. The candidate sets become owned by _gpu at this point.
      this._gpu = candidateGpu;
      this._lastMaterialAtlasPayload = DDGI_PLACEHOLDER_MATERIAL_ATLAS;
    } catch (error) {
      cleanupCandidates();
      throw error;
    }
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
  async runFrame(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }, offset: number, stride: number): Promise<boolean> {
    if (this._disposed) return false;
    assertPositiveDdgiInteger(stride, 'DDGI probe update stride');
    assertDdgiInteger(offset, 'DDGI probe update offset');
    if (!this._gpu) {
      // If init was already attempted and failed (e.g. hard WebGPU-init failure
      // where navigator.gpu.requestAdapter returned null), do not re-issue the
      // adapter request on every subsequent frame.  Pattern mirrors DDGI._inited
      // / _gpuOk so the two classes have consistent guards.
      if (this._initAttempted) return false;
      await this.init(renderer);
      if (!this._gpu) return false;
    }
    const { device } = this._gpu;
    const snap = this._restirSnapshot;
    const legacyBuffers = snap == null ? this._bvh.buffers : null;
    if (snap == null && legacyBuffers == null) return false;
    if (this._grid.dirty) this.reallocateGridAtlases();

    const probeCount = this._grid.probeCount;
    if (probeCount === 0) return false;

    const normalizedStride = stride;
    const normalizedOffset =
      ((offset % normalizedStride) + normalizedStride) % normalizedStride;
    const fullBlendGeneration = this._fullBlendGeneration;
    const fullBlendThisStratum =
      this._fullBlendStride === normalizedStride &&
      this._pendingFullBlendStrata.has(normalizedOffset);

    if (snap != null) {
      if (snap.contentVersion !== this._lastBvhVersion) {
        const tlasOnly =
          this._gpu != null &&
          isRestirTlasOnlyRefit(snap, {
            blasContentVersion: this._lastBlasVersion,
            tlasContentVersion: this._lastTlasVersion,
            materialContentVersion: this._lastMaterialVersion,
          });
        if (tlasOnly && snap.tlas != null) {
          refitProbeTlasBuffersInPlace(device, this._gpu, snap.tlas);
        } else {
          rebuildProbeBvhFromRestir(device, this._gpu, snap);
          this._syncTangentTexture(device, snap.tangents);
          this._syncVertexColorTexture(device, snap.vertexColors);
        }
        this._lastBvhVersion = snap.contentVersion;
        this._lastBlasVersion = snap.blasContentVersion;
        this._lastTlasVersion = snap.tlasContentVersion;
        this._lastMaterialVersion = snap.materialContentVersion;
      }
      this._uploadTraceParams(device, snap);
      this._syncMaterialTextureAtlas(device, snap.materialTextureAtlas);
    } else if (legacyBuffers != null) {
      // H24-C — always rebuild so normal/position updates land in the probe-BVH.
      // The length-based gate was a fragile proxy that missed in-place geometry
      // changes (positions fast-path, normal recompute). Rebuild cost is small
      // relative to probe-ray dispatch; this is the safe default.
      rebuildProbeBvhFromScene(device, this._gpu, legacyBuffers);
      this._uploadTraceParams(device, { bvhMode: 'merged', tlasNodeCount: 0 });
      this._syncMaterialTextureAtlas(device, DDGI_PLACEHOLDER_MATERIAL_ATLAS);
      this._syncTangentTexture(device, null);
      this._syncVertexColorTexture(device, null);
    }

    // Reallocate ray results buffer if probe count changed.
    const maxProbes = probeCount;
    this._ensureRayResultsCapacity(device, maxProbes);

    // Determine which probes are active this frame.
    const activeProbes: number[] = [];
    for (let i = normalizedOffset; i < probeCount; i += normalizedStride) {
      activeProbes.push(i);
    }
    if (activeProbes.length === 0) {
      submitProbeUpdateCommand(
        device.createCommandEncoder(),
        device.queue,
        () => {
          this._frameIndex++;
          this._acknowledgeFullBlendStratum(
            fullBlendGeneration,
            normalizedStride,
            normalizedOffset,
            fullBlendThisStratum,
          );
        },
      );
      return true;
    }

    // Upload active probe indices.
    const activeArr = new Uint32Array(activeProbes);
    this._uploadActiveProbeIndices(device, activeArr);

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
    this._uploadBlendParams(device, fullBlendThisStratum);

    // Ensure grid atlases are allocated.
    if (!this._grid.irradianceA) {
      this.reallocateGridAtlases();
    }

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
    // Preserve every inactive probe's SH coefficients and packed state before
    // the classifier/blend overwrite only this frame's active stratum. This
    // copy must precede the classifier because state lives in the irradiance
    // atlas's reserved (4,4) ring texel.
    copyProbeIrradianceAndPackedStateForward(encoder, irrReadTex, irrWriteTex);
    encoder.copyTextureToTexture(
      { texture: visReadTex },
      { texture: visWriteTex },
      { width: visReadTex.width, height: visReadTex.height, depthOrArrayLayers: 1 },
    );

    dispatchProbeUpdateRaysPass(
      encoder,
      gpu,
      activeProbes.length,
      irrReadTex,
    );
    dispatchProbeClassifyRelocatePass(
      encoder,
      gpu,
      activeProbes.length,
      irrReadTex,
      irrWriteTex,
    );

    // CRITICAL: the read→write copies above keep inactive probes' cells in sync.
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
    dispatchProbeUpdateBlendIrrPass(
      encoder,
      gpu,
      activeProbes.length,
      irrReadTex,
      irrWriteTex,
    );
    dispatchProbeUpdateBlendVisPass(
      encoder,
      gpu,
      activeProbes.length,
      visReadTex,
      visWriteTex,
    );

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
    const visScratch = this._atlasCache.getOrCreateScratchTexture(device, gpu, visWriteTex);
    encoder.copyTextureToTexture(
      { texture: visWriteTex },
      { texture: visScratch },
      { width: visWriteTex.width, height: visWriteTex.height, depthOrArrayLayers: 1 },
    );
    uploadProbeUpdateBorderUbo(device, gpu, this._grid, visWriteTex);
    dispatchProbeUpdateBorderVisPass(encoder, gpu, probeCount, visScratch, visWriteTex);

    submitProbeUpdateCommand(encoder, device.queue, () => {
      // Publish the ping-pong identities, accepted-frame counter, and
      // invalidation acknowledgement atomically after queue.submit succeeds.
      this._grid.swap();
      this._frameIndex++;
      this._acknowledgeFullBlendStratum(
        fullBlendGeneration,
        normalizedStride,
        normalizedOffset,
        fullBlendThisStratum,
      );
    });

    // Expose internal state for debug/e2e inspection when opted in.
    if (this._debug && typeof window !== 'undefined') {
      (window as unknown as Record<string, unknown>)['__DDGI__'] = {
        frameIndex: this._frameIndex,
        probeCount,
        activeCount: activeProbes.length,
      };
    }
    return true;
  }

  private _ensureRayResultsCapacity(device: GPUDevice, maxProbes: number): void {
    if (maxProbes === this._maxProbes) return;
    const gpu = this._gpu!;
    const previous = gpu.rayResultsBuf;
    const candidate = device.createBuffer({
      label: 'ddgi.ray-results',
      size: maxProbes * RAYS_PER_PROBE * PROBE_RAY_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (candidate === previous) {
      throw new Error('DDGI ray-results candidate aliases the live buffer.');
    }
    gpu.rayResultsBuf = candidate;
    this._maxProbes = maxProbes;
    destroyProbeResourceBestEffort(previous);
  }

  private _uploadActiveProbeIndices(
    device: GPUDevice,
    activeArr: Uint32Array<ArrayBuffer>,
  ): void {
    const gpu = this._gpu!;
    const previous = gpu.activeProbesBuf;
    if (previous.size >= activeArr.byteLength) {
      device.queue.writeBuffer(previous, 0, activeArr);
      return;
    }
    const candidate = device.createBuffer({
      label: 'ddgi.active-probes',
      size: activeArr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (candidate === previous) {
      throw new Error('DDGI active-probes candidate aliases the live buffer.');
    }
    try {
      device.queue.writeBuffer(candidate, 0, activeArr);
    } catch (error) {
      destroyProbeResourceBestEffort(candidate);
      throw error;
    }
    gpu.activeProbesBuf = candidate;
    destroyProbeResourceBestEffort(previous);
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
    const buf = packDDGIMaterialsN(mats, this._ddgiMaxMaterials);
    device.queue.writeBuffer(this._gpu!.materialsBuf, 0, buf);
  }

  /** Core-first material upload (THREE-decouple): pack a deduped `MaterialSpec[]`
   *  directly via `coreMaterialToMaterialEntry` + the `emissiveIntensity = 1`
   *  production convention — no `THREE.Material` round-trip. See
   *  `packDDGIMaterialsFromCoreN`. */
  private _uploadCoreMaterials(device: GPUDevice, mats: readonly MaterialSpec[]): void {
    const buf = packDDGIMaterialsFromCoreN(mats, this._ddgiMaxMaterials);
    device.queue.writeBuffer(this._gpu!.materialsBuf, 0, buf);
  }

  private _uploadLights(device: GPUDevice): void {
    const payload = packDDGIProbeLights(this._lights, this._sunIntensityMul);
    const gpu = this._gpu!;
    if (payload.byteLength <= gpu.lightsCapacityBytes) {
      device.queue.writeBuffer(gpu.lightsBuf, 0, payload);
      return;
    }

    const maxBufferSize = device.limits?.maxBufferSize;
    if (typeof maxBufferSize === 'number' && payload.byteLength > maxBufferSize) {
      throw new RangeError(
        `[DDGI] ${this._lights.length} lights require ${payload.byteLength} bytes, ` +
        `exceeding device.limits.maxBufferSize=${maxBufferSize}.`,
      );
    }
    const maxStorageBinding = device.limits?.maxStorageBufferBindingSize;
    if (typeof maxStorageBinding === 'number' && payload.byteLength > maxStorageBinding) {
      throw new RangeError(
        `[DDGI] ${this._lights.length} lights require a ${payload.byteLength}-byte storage binding, ` +
        `exceeding device.limits.maxStorageBufferBindingSize=${maxStorageBinding}.`,
      );
    }

    const hardwareLimit = Math.min(
      typeof maxBufferSize === 'number' ? maxBufferSize : Number.MAX_SAFE_INTEGER,
      typeof maxStorageBinding === 'number' ? maxStorageBinding : Number.MAX_SAFE_INTEGER,
    );
    const doubled = gpu.lightsCapacityBytes * 2;
    const nextCapacity = Math.min(
      hardwareLimit,
      Math.max(payload.byteLength, Number.isSafeInteger(doubled) ? doubled : payload.byteLength),
    );
    let candidate: GPUBuffer | null = null;
    try {
      candidate = device.createBuffer({
        label: 'ddgi.lights',
        size: nextCapacity,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(candidate, 0, payload);
    } catch (error) {
      destroyProbeResourceBestEffort(candidate);
      throw error;
    }

    const previous = gpu.lightsBuf;
    gpu.lightsBuf = candidate;
    gpu.lightsCapacityBytes = nextCapacity;
    destroyProbeResourceBestEffort(previous);
  }

  private _syncMaterialTextureAtlas(device: GPUDevice, payload: MaterialTextureAtlasPayload): void {
    const gpu = this._gpu;
    if (gpu == null || this._lastMaterialAtlasPayload === payload) return;
    const next = uploadMaterialTextureAtlas(device, payload);
    const previousAtlas = gpu.materialTextureAtlas;
    const previousMeta = gpu.materialTextureAtlasMeta;
    if (next.atlasTexture === previousAtlas || next.baseColorMetaTexture === previousMeta) {
      destroyProbeResourceBestEffort(
        next.atlasTexture === previousAtlas ? null : next.atlasTexture,
      );
      destroyProbeResourceBestEffort(
        next.baseColorMetaTexture === previousMeta ? null : next.baseColorMetaTexture,
      );
      throw new Error('DDGI material-atlas candidate aliases a live texture.');
    }
    gpu.materialTextureAtlas = next.atlasTexture;
    gpu.materialTextureAtlasView = next.atlasTextureView;
    gpu.materialTextureAtlasMeta = next.baseColorMetaTexture;
    gpu.materialTextureAtlasMetaView = next.baseColorMetaTextureView;
    this._lastMaterialAtlasPayload = payload;
    destroyProbeResourceBestEffort(previousAtlas);
    destroyProbeResourceBestEffort(previousMeta);
  }

  private _syncTangentTexture(device: GPUDevice, tangents: ArrayBuffer | null): void {
    const gpu = this._gpu;
    if (gpu == null) return;
    if (tangents == null && this._lastTangentSource == null) return;
    const data = tangents != null && tangents.byteLength >= 16
      ? new Float32Array(tangents)
      : new Float32Array(4);
    const vertexCount = Math.max(1, Math.floor(data.length / 4));
    const next = uploadTangentTexture(device, data, vertexCount);
    const previous = gpu.bvhTangentTexture;
    if (next.texture === previous) {
      throw new Error('DDGI tangent candidate aliases the live texture.');
    }
    let nextView: GPUTextureView;
    try {
      nextView = next.texture.createView();
    } catch (error) {
      destroyProbeResourceBestEffort(next.texture);
      throw error;
    }
    gpu.bvhTangentTexture = next.texture;
    gpu.bvhTangentTextureView = nextView;
    this._lastTangentSource = tangents;
    destroyProbeResourceBestEffort(previous);
  }

  private _syncVertexColorTexture(device: GPUDevice, colors: ArrayBuffer | null): void {
    const gpu = this._gpu;
    if (gpu == null) return;
    if (colors == null && this._lastVertexColorSource == null) return;
    const data = colors != null && colors.byteLength >= 16
      ? new Float32Array(colors)
      : new Float32Array([1, 1, 1, 1]);
    const vertexCount = Math.max(1, Math.floor(data.length / 4));
    const next = uploadVertexColorTexture(device, data, vertexCount);
    const previous = gpu.bvhVertexColorTexture;
    if (next.texture === previous) {
      throw new Error('DDGI vertex-color candidate aliases the live texture.');
    }
    let nextView: GPUTextureView;
    try {
      nextView = next.texture.createView();
    } catch (error) {
      destroyProbeResourceBestEffort(next.texture);
      throw error;
    }
    gpu.bvhVertexColorTexture = next.texture;
    gpu.bvhVertexColorTextureView = nextView;
    this._lastVertexColorSource = colors;
    destroyProbeResourceBestEffort(previous);
  }

  /** H18 Stage 2 — upload the packed emitter-tri array (or keep a dummy if count==0). */
  private _uploadEmitterTris(device: GPUDevice): void {
    const gpu = this._gpu!;
    const count = this._emitterTrisCount;
    const data  = this._emitterTrisData;
    if (count === 0) {
      // Sun-only scene: keep the existing full-record dummy; update the count only.
      gpu.emitterTrisCount = 0;
      return;
    }
    const needed = data.byteLength;
    if (gpu.emitterTrisBuf.size < needed) {
      const previous = gpu.emitterTrisBuf;
      const candidate = device.createBuffer({
        label: 'ddgi.emitter-tris',
        size: needed,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      if (candidate === previous) {
        throw new Error('DDGI emitter-triangle candidate aliases the live buffer.');
      }
      try {
        device.queue.writeBuffer(candidate, 0, data.buffer, data.byteOffset, data.byteLength);
      } catch (error) {
        destroyProbeResourceBestEffort(candidate);
        throw error;
      }
      gpu.emitterTrisBuf = candidate;
      gpu.emitterTrisCount = count;
      destroyProbeResourceBestEffort(previous);
      return;
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
      skyTint: this._skyTint,
      skyIrradiance: this._skyIrradiance,
      glassMixScale: this._glassMixScale,
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
      // Publish the borrowed view first; retirement of the old placeholder is
      // best-effort and cannot prevent the environment transition.
      const previousPlaceholder = gpu.envMapOwnedByPass
        ? gpu.envMapPlaceholderTex
        : null;
      gpu.envMapView = this._envMapView;
      gpu.envSamplerForProbe = this._envSampler ?? gpu.linearSampler;
      gpu.envMapPlaceholderTex = null;
      gpu.envMapOwnedByPass = false;
      destroyProbeResourceBestEffort(previousPlaceholder);
    } else {
      // HDRI disabled/reset. If the current view came from an external pipeline
      // texture, drop that borrowed view immediately and bind a fresh pass-owned
      // 1x1 placeholder so future probe-ray bind groups cannot retain a stale or
      // destroyed environment view after updateEnvironment(null).
      if (gpu.envMapOwnedByPass) {
        gpu.envSamplerForProbe = gpu.linearSampler;
        return;
      }
      const placeholder = gpu.device.createTexture({
        label: 'ddgi.env-placeholder',
        size: [1, 1, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      let placeholderView: GPUTextureView;
      try {
        placeholderView = placeholder.createView();
      } catch (error) {
        destroyProbeResourceBestEffort(placeholder);
        throw error;
      }
      gpu.envMapPlaceholderTex = placeholder;
      gpu.envMapView = placeholderView;
      gpu.envMapOwnedByPass = true;
      gpu.envSamplerForProbe = gpu.linearSampler;
    }
  }

  private _uploadBlendParams(
    device: GPUDevice,
    fullBlend = false,
  ): void {
    // Same divisor as the ray pass so the blend coverage matches the rays
    // written this frame (a mismatch would blend uncovered probes).
    // Full-replace is staged for this stratum. The pending set is acknowledged
    // only after encoder.finish + queue.submit succeed in runFrame.
    const hysteresisOverride = fullBlend ? 0.0 : undefined;
    const data = packProbeUpdateBlendParams(hysteresisOverride);
    device.queue.writeBuffer(this._gpu!.blendParamsBuf, 0, data);
  }

  private _acknowledgeFullBlendStratum(
    generation: number,
    stride: number,
    offset: number,
    wasFullBlend: boolean,
  ): void {
    if (!wasFullBlend) return;
    if (generation !== this._fullBlendGeneration || stride !== this._fullBlendStride) {
      return;
    }
    this._pendingFullBlendStrata.delete(offset);
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
  getReadAtlasGPUTextures(): {
    irradiance: GPUTexture;
    visibility: GPUTexture;
  } | null {
    const irrTex = this._grid.irradianceReadTex;
    const visTex = this._grid.visibilityReadTex;
    if (!irrTex || !visTex) return null;
    const irrGpu = this._atlasCache.getCachedAtlas(irrTex);
    const visGpu = this._atlasCache.getCachedAtlas(visTex);
    if (!irrGpu || !visGpu) return null;
    return { irradiance: irrGpu, visibility: visGpu };
  }

  /**
   * Replace the ProbeGrid's backend-agnostic slots and retire every GPU atlas
   * cached for the displaced slots. ProbeUpdatePass owns those GPU textures;
   * ProbeGrid intentionally owns no backend handles.
   */
  reallocateGridAtlases(): void {
    const displaced = [
      this._grid.irradianceA,
      this._grid.irradianceB,
      this._grid.visibilityA,
      this._grid.visibilityB,
    ].filter((slot): slot is AtlasTextureSlot => slot != null);
    this._grid.allocateAtlases();
    this._atlasCache.retireAtlasSlots(displaced);
  }

  /**
   * Read the converged probe atlases back to CPU (the "cached light field" export).
   * Returns the raw rgba16float texels + atlas dims, or null if the atlases aren't
   * allocated yet. Async (mapAsync).
   */
  async exportAtlasData(device: GPUDevice): Promise<{
    irrW: number; irrH: number; visW: number; visH: number;
    probeStateW: number; probeStateH: number;
    irrData: Uint16Array; visData: Uint16Array;
    probeStateData: Float32Array;
  } | null> {
    const tex = this.getReadAtlasGPUTextures();
    const irrSlot = this._grid.irradianceReadTex;
    const visSlot = this._grid.visibilityReadTex;
    if (!tex || !irrSlot || !visSlot) return null;
    const irrData = await this._readbackRgba16f(device, tex.irradiance, irrSlot.width, irrSlot.height);
    const visData = await this._readbackRgba16f(device, tex.visibility, visSlot.width, visSlot.height);
    const probeStateData = readPackedProbeStateFromIrradianceAtlas(irrData, {
      dimsX: this._grid.dims.x,
      dimsY: this._grid.dims.y,
      dimsZ: this._grid.dims.z,
      irradianceWidth: irrSlot.width,
      irradianceHeight: irrSlot.height,
      spacing: this._grid.worldSpacing,
    });
    if (!isValidProbeStateData(probeStateData, this._grid.worldSpacing)) {
      throw new Error('DDGI packed probe state readback is invalid.');
    }
    return {
      irrW: irrSlot.width,
      irrH: irrSlot.height,
      visW: visSlot.width,
      visH: visSlot.height,
      probeStateW: this._grid.dims.x,
      probeStateH: this._grid.dims.y * this._grid.dims.z,
      irrData,
      visData,
      probeStateData,
    };
  }

  /**
   * Upload previously-exported probe atlases into the read-side textures (the
   * restore). Returns false (no-op) if the atlases aren't allocated or the
   * snapshot's dims don't match the current grid (a different scene). The restored
   * state seeds the temporal blend, so subsequent frames continue from it.
   */
  importAtlasData(
    device: GPUDevice,
    snap: {
      irrW: number; irrH: number; visW: number; visH: number;
      probeStateW: number; probeStateH: number;
      irrData: Uint16Array; visData: Uint16Array;
      probeStateData: Float32Array;
    },
  ): boolean {
    const tex = this.getReadAtlasGPUTextures();
    const irrSlot = this._grid.irradianceReadTex;
    const visSlot = this._grid.visibilityReadTex;
    if (!tex || !irrSlot || !visSlot) return false;
    if (irrSlot.width !== snap.irrW || irrSlot.height !== snap.irrH ||
        visSlot.width !== snap.visW || visSlot.height !== snap.visH ||
        this._grid.dims.x !== snap.probeStateW ||
        this._grid.dims.y * this._grid.dims.z !== snap.probeStateH) {
      return false; // grid mismatch — cannot restore into a differently-sized atlas
    }
    if (snap.irrData.length !== snap.irrW * snap.irrH * 4 ||
        snap.visData.length !== snap.visW * snap.visH * 4 ||
        snap.probeStateData.length !== snap.probeStateW * snap.probeStateH * 4 ||
        !isValidProbeStateData(snap.probeStateData, this._grid.worldSpacing)) {
      return false;
    }
    const irradianceData = snap.irrData.slice();
    try {
      writePackedProbeStateToIrradianceAtlas(
        irradianceData,
        snap.probeStateData,
        {
          dimsX: this._grid.dims.x,
          dimsY: this._grid.dims.y,
          dimsZ: this._grid.dims.z,
          irradianceWidth: snap.irrW,
          irradianceHeight: snap.irrH,
          spacing: this._grid.worldSpacing,
        },
      );
    } catch {
      return false;
    }

    const usage =
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST;
    const candidates: GPUTexture[] = [];
    try {
      const irradiance = device.createTexture({
        label: 'ddgi.import.irradiance.candidate',
        size: [snap.irrW, snap.irrH, 1],
        format: 'rgba16float',
        usage,
      });
      candidates.push(irradiance);
      const visibility = device.createTexture({
        label: 'ddgi.import.visibility.candidate',
        size: [snap.visW, snap.visH, 1],
        format: 'rgba16float',
        usage,
      });
      candidates.push(visibility);
      if (irradiance === tex.irradiance || irradiance === tex.visibility ||
          visibility === tex.irradiance || visibility === tex.visibility ||
          visibility === irradiance) {
        throw new Error('DDGI atlas-import candidate aliases a live or sibling texture.');
      }

      this.#uploadRgba16f(device, irradiance, snap.irrW, snap.irrH, irradianceData);
      this.#uploadRgba16f(device, visibility, snap.visW, snap.visH, snap.visData);
      this._atlasCache.replaceCachedAtlasCohort([
        { slot: irrSlot, texture: irradiance },
        { slot: visSlot, texture: visibility },
      ]);
      candidates.length = 0;
    } catch (error) {
      for (const candidate of new Set(candidates)) {
        if (
          candidate === tex.irradiance ||
          candidate === tex.visibility
        ) continue;
        destroyProbeResourceBestEffort(candidate);
      }
      throw error;
    }
    return true;
  }

  /** copyTextureToBuffer (256-aligned rows) → unpadded Uint16Array of rgba16float. */
  private async _readbackRgba16f(
    device: GPUDevice,
    tex: GPUTexture,
    w: number,
    h: number,
  ): Promise<Uint16Array> {
    const unpadded = w * 8; // rgba16float = 8 bytes/texel
    const bytesPerRow = Math.ceil(unpadded / 256) * 256;
    const staging = device.createBuffer({
      label: 'ddgi.readback.rgba16f',
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let mapped = false;
    let padded: Uint8Array;
    try {
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: tex },
        { buffer: staging, bytesPerRow, rowsPerImage: h },
        [w, h, 1],
      );
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      mapped = true;
      padded = new Uint8Array(staging.getMappedRange().slice(0));
    } finally {
      if (mapped) {
        try { staging.unmap(); } catch { /* continue retirement */ }
      }
      destroyProbeResourceBestEffort(staging);
    }
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
    if (this._disposed && this._gpu == null && this._ownedDevice == null) return;
    this._disposed = true;
    this._pendingFullBlendStrata.clear();
    const g = this._gpu;
    const ownedDevice = this._ownedDevice;
    this._gpu = null;
    this._ownedDevice = null;
    this._lastMaterialAtlasPayload = null;
    this._lastTangentSource = null;
    this._lastVertexColorSource = null;
    if (g != null) {
      const owned = new Set<GPUBuffer | GPUTexture>([
        g.bvhBuf,
        g.posBuf,
        g.idxBuf,
        g.normBuf,
        g.matIdBuf,
        g.tlasNodesBuf,
        g.tlasInstIdxBuf,
        g.tlasBlasRootsBuf,
        g.tlasW2lBuf,
        g.tlasL2wBuf,
        g.traceParamsBuf,
        g.materialsBuf,
        g.lightsBuf,
        g.emitterTrisBuf,
        g.materialTextureAtlas,
        g.materialTextureAtlasMeta,
        g.bvhTangentTexture,
        g.bvhVertexColorTexture,
        g.gridParamsBuf,
        g.frameParamsBuf,
        g.blendParamsBuf,
        g.borderVisUboBuf,
        g.rayResultsBuf,
        g.activeProbesBuf,
      ]);
      if (g.visScratchTex != null) owned.add(g.visScratchTex);
      if (g.envMapOwnedByPass && g.envMapPlaceholderTex != null) {
        owned.add(g.envMapPlaceholderTex);
      }
      for (const resource of owned) destroyProbeResourceBestEffort(resource);
    }
    try { this._atlasCache.dispose(); } catch { /* terminal teardown remains complete */ }
    destroyProbeDeviceBestEffort(ownedDevice);
  }
}
