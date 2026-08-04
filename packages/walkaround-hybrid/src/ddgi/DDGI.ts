/**
 * DDGI — class-based DDGI probe grid lifecycle.
 *
 * De-React-ified from `useDDGI.ts`. All React hooks stripped:
 *   - useRef       → private class fields
 *   - useEffect    → constructor + dispose()
 *   - useCallback  → plain method
 *   - useFrame     → updateFrame()
 *   - useThree     → renderer / scene passed as arguments
 *   - useSelector  → lights passed explicitly to setLights()
 *
 * Normal usage via HybridEngine:
 *   1. `new DDGI(opts)` — creates BVH, ProbeGrid, ProbeUpdatePass in memory.
 *   2. `setLights(lights)` whenever the light list changes.
 *   3. `HybridEngine.renderFrame()` drives `updateFrame()` internally.
 *   4. `dispose()` when the canvas is torn down.
 *
 * Standalone (advanced) usage — DDGI without the full HybridEngine:
 *   Call `updateFrame(inputs)` directly once per animation frame.
 *
 * Debug global: when `opts.debug === true` the class writes
 * `window.__DDGI__` after each updateFrame, guarded by
 * `typeof window !== 'undefined'`. The host is responsible for exposing
 * the `window.__WALKAROUND__.layers.ddgi` and `window.__SET_CAMERA__`
 * globals that the original hook published — these are host-side bridge
 * concerns, not library concerns.
 */

import { SceneBvh } from '@vitrum/shared-bvh';
import type { EngineError, EngineWarning, Scene } from '@vitrum/core';
import { ProbeGrid } from './probeGrid.js';
import type { ProbeGridParams } from './probeGrid.js';
import {
  ProbeUpdatePass,
  type ProbeAtlasImportTransaction,
  type ProbeAtlasSnapshot,
} from './probeUpdatePass.js';
import { snapshotDdgiLights, type DDGILight } from './types.js';
import {
  isRestirTlasOnlySnapshotChange,
  refreshRestirBvhSnapshot,
  restirBvhSnapshotStateEqual,
  type RestirBvhSnapshot,
} from '../restir/restirBvhSnapshot.js';
import {
  rethrowWithSceneMutationCleanup,
  type PreparedSceneMutation,
} from '../SceneMutationTransaction.js';
import type { SceneBVHBuffers } from '../restir/bvhCore.js';
import {
  assertDdgiBoolean,
  assertDdgiU32,
  assertDdgiUnitInterval,
  assertFiniteDdgiNumber,
  assertFiniteDdgiVec3,
  assertNonNegativeDdgiNumber,
  assertPositiveDdgiInteger,
  assertValidDdgiLights,
  packDdgiProbeSpacingFloat32,
  validateDdgiProbeUpdateDivisor,
} from './inputValidation.js';
import {
  assertWalkaroundEnvironmentScaleF32,
  packWalkaroundEnvironmentRotationF32,
} from '../environment/environmentRadianceScale.js';
import {
  packLightingRgbScaleEnvelopeF32,
  packNonNegativeLightingFloat32,
} from '../lightingFloat32.js';

// Default probe round-robin stride. STRIDE=8 means each probe updates every
// 8th frame (~133ms at 60fps). This is the cadence the engine has always
// actually run — it is the divisor used to build the per-frame `activeProbes`
// set (see updateFrame). `setProbeUpdateDivisor` overrides it at runtime; when
// no divisor is set the round-robin falls back to this value.
const DEFAULT_STRIDE = 8;

// Target frame interval for the 60 FPS cap.
const TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

export interface DDGIOptions {
  /**
   * When true, writes debug state to `window.__DDGI__` after each frame.
   * Guarded by `typeof window !== 'undefined'`.
   */
  debug?: boolean;
  /**
   * Maximum number of distinct materials supported by the DDGI probe pass.
   * Forwarded to {@link ProbeUpdatePassOptions.maxMaterials}. Defaults to 64.
   * @since Sprint 16 (M9 audit remediation)
   */
  maxMaterials?: number;
  /**
   * Probe spacing (world units). Passed to `ProbeGrid.computeFromBounds`.
   * `undefined` = auto-derived from scene AABB (`maxSize / 12`).
   * @since Sprint 16 (M11 audit remediation)
   */
  probeSpacing?: number;
  /**
   * Hard cap on probes per axis. Defaults to 16.
   * @since Sprint 16 (M11 audit remediation)
   */
  maxProbesPerAxis?: number;
  /**
   * Optional structured error sink. HybridEngine wires this to Engine.onError;
   * standalone DDGI consumers may use it for the same non-fatal diagnostics.
   */
  onError?: (error: EngineError) => void;
  /**
   * Optional structured warning sink. HybridEngine wires this to Engine.onWarning;
   * standalone DDGI consumers may use it for probe-light/material cap telemetry.
   */
  onWarning?: (warning: EngineWarning) => void;
}

/** Per-frame inputs supplied by the host for a DDGI update tick. */
export interface DDGIFrameInputs {
  /**
   * Optional `@vitrum/core` `Scene` — when present (and no ReSTIR snapshot is
   * active), the standalone DDGI BVH is built core-first via
   * {@link SceneBvh.updateFromCore} (`mergeWorldSpaceFromCore` + core
   * materials). When absent, standalone DDGI skips BVH update until a core scene
   * or ReSTIR snapshot is available.
   */
  coreScene?: Scene;
  /** Raw WebGPU device. Supply this when the host owns the device directly. */
  device?: GPUDevice;
  /**
   * Whether DDGI compute is enabled this frame. When false, updateFrame
   * returns immediately without dispatching any GPU work.
   */
  enabled: boolean;
}

export class DDGI {
  private _bvh:         SceneBvh;
  private _grid:        ProbeGrid;
  private _pass:        ProbeUpdatePass;
  /** PR-5.1 — when set, skips SceneBvh rebuild and uses ReSTIR GPU buffers. */
  private _restirSnapshot: RestirBvhSnapshot | null = null;
  private _ready:       boolean  = false;
  private _lastFrameMs: number   = 0;
  private _frame:       number   = 0;
  private _inited:      boolean  = false;
  private _gpuOk:       boolean  = false;
  private _lastFrameTs: number   = 0;
  private _debug:       boolean;
  private readonly _onError: ((error: EngineError) => void) | undefined;
  private readonly _onWarning: ((warning: EngineWarning) => void) | undefined;
  private _reportedMissingDevice: boolean = false;
  // M11: probe grid parameters forwarded to computeFromBounds each frame.
  private _probeSpacing:      number | undefined;
  private _maxProbesPerAxis:  number;
  // Phase-0 productization (H1) — round-robin probe-update stride. THIS is the
  // load-bearing cadence knob: `updateFrame` builds the per-frame active-probe
  // set from `offset = _frame % _stride; stride = _stride`, and ProbeUpdatePass
  // constructs `activeProbes` from that stride. Defaults to DEFAULT_STRIDE (8);
  // `setProbeUpdateDivisor` overrides it so the quality preset's divisor
  // actually changes how many probes update per frame (previously the divisor
  // only fed a UBO field that no shader reads).
  private _stride:            number = DEFAULT_STRIDE;
  private _disposed = false;
  private _lifecycleGeneration = 1;
  /** Invalidates warmup accounting captured by an older in-flight submission. */
  private _contentEpoch = 1;
  private _updateInFlight: Promise<void> | null = null;
  // Standalone setters are part of the public DDGI surface. Keep semantic
  // snapshots here so repeated per-frame synchronization is a no-op, while a
  // real lighting change invalidates any older in-flight probe submission.
  private _configuredLights: DDGILight[] = [];
  private _configuredEmitterTris = new Float32Array(0);
  private _configuredEmitterCount = 0;
  private _configuredSunIntensityMultiplier = 1;
  private _configuredSkyTint: [number, number, number] = [0.4, 0.6, 1.0];
  private _configuredSkyIrradiance = 2;
  private _configuredGlassMixScale = 0.7;
  private _configuredIndirectFeedback = true;
  private _configuredEnvironment: {
    view: GPUTextureView | null;
    sampler: GPUSampler | null;
    rotationY: number;
    intensity: number;
    hasEnv: boolean;
  } = { view: null, sampler: null, rotationY: 0, intensity: 0, hasEnv: false };

  constructor(opts: DDGIOptions = {}) {
    if (opts.debug !== undefined) assertDdgiBoolean(opts.debug, 'DDGI debug');
    const packedProbeSpacing = opts.probeSpacing === undefined
      ? undefined
      : packDdgiProbeSpacingFloat32(opts.probeSpacing, 'DDGI probe spacing');
    if (opts.maxProbesPerAxis !== undefined) {
      assertPositiveDdgiInteger(
        opts.maxProbesPerAxis,
        'DDGI max probes per axis',
      );
      assertDdgiU32(opts.maxProbesPerAxis, 'DDGI max probes per axis');
      if (opts.maxProbesPerAxis < 3) {
        throw new RangeError('DDGI max probes per axis must be >= 3.');
      }
    }
    this._debug = opts.debug ?? false;
    this._onError = opts.onError;
    this._onWarning = opts.onWarning;
    this._probeSpacing     = packedProbeSpacing;
    this._maxProbesPerAxis = opts.maxProbesPerAxis ?? 16;
    this._bvh   = new SceneBvh({
      onWarning: (warning) => this._warn({
        code: warning.includes('displacementMap')
          ? 'walkaround-hybrid.vertex-displacement-skipped'
          : 'walkaround-hybrid.scene-pack-warning',
        backend: 'walkaround-hybrid',
        phase: 'setScene',
        method: 'DDGI.setScene',
        message: `[vitrum/walkaround-hybrid/DDGI] ${warning}`,
        details: {
          warning,
          source: 'shared-bvh',
          subsystem: 'ddgi',
        },
      }),
    });
    this._grid  = new ProbeGrid();
    this._pass  = new ProbeUpdatePass(this._bvh, this._grid, {
      debug: this._debug,
      ...(opts.maxMaterials !== undefined ? { maxMaterials: opts.maxMaterials } : {}),
      onWarning: (warning) => this._warn(warning),
    });
    this._pass.setProbeUpdateDivisor(this._stride);
  }

  private _reportError(message: string, raw?: unknown): void {
    try {
      this._onError?.({
        kind: 'render',
        message,
        fatal: false,
        ...(raw !== undefined ? { raw } : {}),
      });
    } catch {
      // Host error callbacks must not break the DDGI frame loop.
    }
  }

  private _warn(warning: EngineWarning): void {
    if (this._onWarning) {
      try {
        this._onWarning(warning);
      } catch {
        // Host warning callbacks must not break the DDGI frame loop.
      }
      return;
    }
    console.warn(warning.message);
  }

  private _advanceContentEpoch(): void {
    this._contentEpoch = (this._contentEpoch + 1) >>> 0;
    if (this._contentEpoch === 0) this._contentEpoch = 1;
  }

  private _mutateWithLightingContentInvalidation(mutate: () => void): void {
    const invalidation = this._pass.prepareFullBlendInvalidation(this._stride);
    try {
      mutate();
      // All fallible candidate work is complete. The remaining operations are
      // scalar/reference swaps and wrapping u32 arithmetic only.
      invalidation.commit();
      this._frame = 0;
      this._ready = false;
      this._advanceContentEpoch();
    } catch (error) {
      invalidation.rollback();
      throw error;
    }
    invalidation.finalize();
  }

  /**
   * Publish a fresh atlas cohort together with the temporal state that makes
   * every stratum replace, rather than blend against, its zero-initialized
   * history. Preparing the bounded pending-strata Set happens before the old
   * atlas cohort is displaced; after reallocation succeeds, commit is only a
   * reference/scalar publication and cannot strand fresh atlases behind stale
   * readiness state.
   */
  private _reallocateGridAtlasesWithWarmupReset(): void {
    const invalidation = this._pass.prepareFullBlendInvalidation(this._stride);
    try {
      this._pass.reallocateGridAtlases();
      invalidation.commit();
      this._frame = 0;
      this._ready = false;
    } catch (error) {
      invalidation.rollback();
      throw error;
    }
    invalidation.finalize();
  }

  /** Advance the long-running cadence counter with explicit u32 wrap. */
  private _advanceFrameIndex(): void {
    this._frame = (this._frame + 1) >>> 0;
  }

  private _commitPreparedLightingMutation(mutation: PreparedSceneMutation): void {
    try {
      mutation.commit();
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        [() => mutation.rollback()],
        'DDGI lighting publication failed and rollback also failed',
      );
    }
    mutation.finalize();
  }

  // ── Read-only accessors matching the old DDGIHandle shape ─────────────────

  get bvh():        SceneBvh    { return this._bvh; }
  /**
   * @internal — direct ProbeUpdatePass access for tests and legacy adapters.
   *   Production callers should prefer the facade methods
   *   ({@link exportAtlasData}, {@link importAtlasData},
   *   {@link getReadAtlasGPUTextures}) instead of reaching through to the pass.
   */
  get pass():       ProbeUpdatePass { return this._pass; }
  get ready():      boolean     { return this._ready; }
  get lastFrameMs(): number     { return this._lastFrameMs; }
  get probeCount(): number      { return this._grid.probeCount; }

  /**
   * H24-B — observable DDGI lifecycle state.
   *
   * - `'initializing'` — GPU init has not yet completed (before the first
   *   `updateFrame` call with a WebGPU device).
   * - `'ready'`        — GPU init succeeded AND the probe atlas has completed
   *   at least one full round-robin cycle (all `_stride` strata updated).
   *   Only set when `_gpuOk === true` to prevent a failed GPU init from
   *   silently advertising convergence.
   * - `'failed'`       — GPU init was attempted but returned false
   *   (`navigator.gpu` unavailable, adapter request failed, or
   *   ShaderModule compilation error). DDGI compute is disabled; the scene
   *   renders without indirect GI.
   */
  state(): 'initializing' | 'ready' | 'failed' {
    if (!this._inited) return 'initializing';
    if (!this._gpuOk)  return 'failed';
    return this._ready ? 'ready' : 'initializing';
  }

  /** Number of probe-update command buffers accepted since the last
   *  `invalidateProbeCache()` (or construction). Increments only after an
   *  enabled `updateFrame` successfully finishes and submits its GPU work;
   *  reset to 0 by `invalidateProbeCache()`. Read by
   *  `HybridEngine.onProgress` to compute the `'ddgi-warmup'` fraction —
   *  after `warmupStride` passes the round-robin has touched every probe at
   *  least once (one stratum of `1/stride` probes per pass). */
  get warmupFrame(): number     { return this._frame; }

  /** Round-robin probe-update stride (= the divisor). After `warmupStride`
   *  enabled passes every probe has received ≥1 update, which is exactly when
   *  `ready` flips true. Target for the `'ddgi-warmup'` progress metric. */
  get warmupStride(): number    { return this._stride; }

  // ── Light configuration ───────────────────────────────────────────────────

  /** Replace the current light list. Forwarded to ProbeUpdatePass. */
  setLights(lights: DDGILight[]): void {
    assertValidDdgiLights(lights);
    const nextLights = snapshotDdgiLights(lights);
    if (sameDdgiLights(this._configuredLights, nextLights)) return;
    this._commitPreparedLightingMutation(this.prepareLightingMutation({
      lights: nextLights,
      sunIntensityMultiplier: this._configuredSunIntensityMultiplier,
      emitterTris: this._configuredEmitterTris,
      emitterCount: this._configuredEmitterCount,
    }));
  }

  /**
   * H18 Stage 2 — supply rect/disc area-emitter triangles for per-probe NEE.
   * Forwarded to ProbeUpdatePass.setEmitterTris(). Call after each BVH rebuild.
   */
  setEmitterTris(tris: Float32Array, count: number): void {
    if (sameEmitterPayload(
      this._configuredEmitterTris,
      this._configuredEmitterCount,
      tris,
      count,
    )) return;
    this._commitPreparedLightingMutation(this.prepareLightingMutation({
      lights: this._configuredLights,
      sunIntensityMultiplier: this._configuredSunIntensityMultiplier,
      emitterTris: tris,
      emitterCount: count,
    }));
  }

  /**
   * Prepare an atomic replacement of every emitter-derived DDGI lighting input.
   * The underlying pass allocates a private emitter-triangle buffer before this
   * returns, so commit only publishes prepared CPU/GPU identities.
   */
  prepareLightingMutation(inputs: {
    readonly lights: readonly DDGILight[];
    readonly sunIntensityMultiplier: number;
    readonly emitterTris: Float32Array;
    readonly emitterCount: number;
    readonly skyTint?: readonly [number, number, number];
    readonly skyIrradiance?: number;
  }): PreparedSceneMutation {
    assertValidDdgiLights(inputs.lights);
    assertNonNegativeDdgiNumber(
      inputs.sunIntensityMultiplier,
      'DDGI sun intensity multiplier',
    );
    const packedSunIntensityMultiplier = packNonNegativeLightingFloat32(
      inputs.sunIntensityMultiplier,
      'DDGI sun intensity multiplier',
    );
    const nextLights = snapshotDdgiLights(inputs.lights);
    if (
      (inputs.skyTint === undefined) !==
      (inputs.skyIrradiance === undefined)
    ) {
      throw new TypeError(
        'DDGI prepared lighting mutation must supply skyTint and skyIrradiance together.',
      );
    }
    const packedSky =
      inputs.skyTint === undefined || inputs.skyIrradiance === undefined
        ? null
        : packLightingRgbScaleEnvelopeF32(
            inputs.skyTint,
            inputs.skyIrradiance,
            'DDGI sky radiance',
          );
    // Complete every allocation/snapshot that can fail before the pass stages
    // a private GPU candidate. Nothing between pass preparation and returning
    // the transaction performs user-code access or another fallible copy.
    const nextEmitterTris = inputs.emitterTris.slice(0, inputs.emitterCount * 20);
    const previousConfiguredLights = this._configuredLights;
    const previousConfiguredSunIntensityMultiplier =
      this._configuredSunIntensityMultiplier;
    const previousConfiguredEmitterTris = this._configuredEmitterTris;
    const previousConfiguredEmitterCount = this._configuredEmitterCount;
    const previousConfiguredSkyTint = this._configuredSkyTint;
    const previousConfiguredSkyIrradiance = this._configuredSkyIrradiance;
    const previousFrame = this._frame;
    const previousReady = this._ready;
    const previousContentEpoch = this._contentEpoch;
    const invalidationMutation = this._pass.prepareFullBlendInvalidation(
      this._stride,
    );
    let passMutation: PreparedSceneMutation;
    try {
      passMutation = this._pass.prepareLightingMutation(
        nextLights,
        packedSunIntensityMultiplier,
        inputs.emitterTris,
        inputs.emitterCount,
        packedSky == null
          ? undefined
          : { tint: packedSky.value, irradiance: packedSky.scale },
      );
    } catch (error) {
      invalidationMutation.rollback();
      throw error;
    }
    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || committed) return;
        passMutation.commit();
        this._configuredLights = nextLights;
        this._configuredSunIntensityMultiplier = packedSunIntensityMultiplier;
        this._configuredEmitterTris = nextEmitterTris;
        this._configuredEmitterCount = inputs.emitterCount;
        if (packedSky != null) {
          this._configuredSkyTint = packedSky.value;
          this._configuredSkyIrradiance = packedSky.scale;
        }
        committed = true;
        this._frame = 0;
        this._ready = false;
        invalidationMutation.commit();
        this._advanceContentEpoch();
      },
      rollback: () => {
        if (closed) return;
        try {
          passMutation.rollback();
        } finally {
          if (committed) {
            this._frame = previousFrame;
            this._ready = previousReady;
            this._contentEpoch = previousContentEpoch;
            this._configuredLights = previousConfiguredLights;
            this._configuredSunIntensityMultiplier =
              previousConfiguredSunIntensityMultiplier;
            this._configuredEmitterTris = previousConfiguredEmitterTris;
            this._configuredEmitterCount = previousConfiguredEmitterCount;
            this._configuredSkyTint = previousConfiguredSkyTint;
            this._configuredSkyIrradiance =
              previousConfiguredSkyIrradiance;
          }
          invalidationMutation.rollback();
          closed = true;
        }
      },
      finalize: () => {
        if (closed) return;
        try {
          passMutation.finalize();
        } finally {
          invalidationMutation.finalize();
          closed = true;
        }
      },
    };
  }

  /**
   * Prepare a runtime-only sun/sky mutation while retaining the current
   * emitter-triangle payload and GPU buffer identity.
   */
  prepareRuntimeLightingMutation(inputs: {
    readonly lights: readonly DDGILight[];
    readonly sunIntensityMultiplier: number;
    readonly skyTint: readonly [number, number, number];
    readonly skyIrradiance: number;
  }): PreparedSceneMutation {
    return this.prepareLightingMutation({
      ...inputs,
      emitterTris: this._configuredEmitterTris,
      emitterCount: this._configuredEmitterCount,
    });
  }

  /**
   * Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
   *
   * Forward the directional env-map texture view + sampler to the underlying
   * ProbeUpdatePass so probe miss-rays sample the actual HDRI instead of the
   * procedural sky gradient. Call from the engine whenever the directional
   * environment changes (in `_applyDirectionalEnvironment`).
   *
   * Call with `hasEnv = false` (or pass `null` view) to revert to the
   * procedural gradient — byte-identical to the pre-Wave-4 path.
   *
   * UV convention: matches `environmentSample.wgsl envRadiance` (H6):
   *   lookupDir = RY(-rotationY) · worldDir   [world → unrotated-map space]
   *
   * @param view       GPUTextureView for the rgba16float equirect radiance map,
   *                   or null when reverting to the procedural gradient.
   * @param sampler    GPUSampler for the map, or null to use the pass's own
   *                   linear+clamp sampler.
   * @param rotationY  Y-axis rotation in radians (H6 convention). 0 = no rotation.
   * @param intensity  Radiance multiplier applied to the texel after lookup.
   * @param hasEnv     `true` to activate HDRI sampling; `false` for procedural sky.
   */
  setEnvironment(
    view: GPUTextureView | null,
    sampler: GPUSampler | null,
    rotationY: number,
    intensity: number,
    hasEnv: boolean,
  ): void {
    const packedRotationY = packWalkaroundEnvironmentRotationF32(
      rotationY,
      'DDGI environment rotation',
    );
    assertNonNegativeDdgiNumber(intensity, 'DDGI environment intensity');
    const packedIntensity = assertWalkaroundEnvironmentScaleF32(
      intensity,
      'DDGI environment intensity',
    );
    assertDdgiBoolean(hasEnv, 'DDGI environment hasEnv');
    if (hasEnv && view == null) {
      throw new TypeError('DDGI environment view is required when hasEnv is true.');
    }
    const previous = this._configuredEnvironment;
    if (
      previous.view === view &&
      previous.sampler === sampler &&
      Object.is(previous.rotationY, packedRotationY) &&
      Object.is(previous.intensity, packedIntensity) &&
      previous.hasEnv === hasEnv
    ) {
      // Preserve the facade's forwarding contract even for the default
      // procedural environment. The pass may have acquired GPU state since the
      // previous call; forwarding lets it repair that binding without treating
      // an idempotent host synchronization as new lighting content.
      this._pass.setEnvironment(
        view,
        sampler,
        packedRotationY,
        packedIntensity,
        hasEnv,
      );
      return;
    }
    const nextEnvironment = {
      view,
      sampler,
      rotationY: packedRotationY,
      intensity: packedIntensity,
      hasEnv,
    };
    this._mutateWithLightingContentInvalidation(() => {
      this._pass.setEnvironment(
        view,
        sampler,
        packedRotationY,
        packedIntensity,
        hasEnv,
      );
      this._configuredEnvironment = nextEnvironment;
    });
  }

  // ── Forwarding façade — callers go through DDGI, not DDGI.pass/probeGrid ──

  /**
   * Set the sun-intensity multiplier on the underlying ProbeUpdatePass.
   * Forwarded from `HybridEngine` / `HybridEngineLifecycle` so they don't
   * reach through to `DDGI.pass` directly.
   */
  setSunIntensityMultiplier(m: number): void {
    assertNonNegativeDdgiNumber(m, 'DDGI sun intensity multiplier');
    const packed = packNonNegativeLightingFloat32(
      m,
      'DDGI sun intensity multiplier',
    );
    if (Object.is(this._configuredSunIntensityMultiplier, packed)) return;
    this._commitPreparedLightingMutation(this.prepareLightingMutation({
      lights: this._configuredLights,
      sunIntensityMultiplier: packed,
      emitterTris: this._configuredEmitterTris,
      emitterCount: this._configuredEmitterCount,
    }));
  }

  setSkyParams(tint: [number, number, number], irradiance: number): void {
    assertFiniteDdgiVec3(tint, 'DDGI sky tint');
    tint.forEach((channel, index) => {
      assertNonNegativeDdgiNumber(channel, `DDGI sky tint[${index}]`);
    });
    assertNonNegativeDdgiNumber(irradiance, 'DDGI sky irradiance');
    const packed = packLightingRgbScaleEnvelopeF32(
      tint,
      irradiance,
      'DDGI sky radiance',
    );
    if (
      sameNumber(this._configuredSkyTint[0], packed.value[0]) &&
      sameNumber(this._configuredSkyTint[1], packed.value[1]) &&
      sameNumber(this._configuredSkyTint[2], packed.value[2]) &&
      sameNumber(this._configuredSkyIrradiance, packed.scale)
    ) return;
    this._commitPreparedLightingMutation(this.prepareLightingMutation({
      lights: this._configuredLights,
      sunIntensityMultiplier: this._configuredSunIntensityMultiplier,
      emitterTris: this._configuredEmitterTris,
      emitterCount: this._configuredEmitterCount,
      skyTint: packed.value,
      skyIrradiance: packed.scale,
    }));
  }

  /**
   * Set the glass mix scale on the underlying ProbeUpdatePass.
   * Forwarded from `HybridEngineFrameOrchestrator` so it doesn't reach
   * through to `DDGI.pass` directly.
   */
  setGlassMixScale(s: number): void {
    assertDdgiUnitInterval(s, 'DDGI glass mix scale');
    if (Object.is(this._configuredGlassMixScale, s)) return;
    this._mutateWithLightingContentInvalidation(() => {
      this._pass.setGlassMixScale(s);
      this._configuredGlassMixScale = s;
    });
  }

  /**
   * H46-A — forward the engine's `maxBounces`-derived indirect-feedback gate to
   * the underlying ProbeUpdatePass. `true` = multi-bounce diffuse EMA
   * (maxBounces >= 2, default); `false` = direct-only probes (maxBounces == 1).
   * Forwarded from `HybridEngineFrameOrchestrator` so it doesn't reach through
   * to `DDGI.pass` directly.
   */
  setIndirectFeedback(enabled: boolean): void {
    assertDdgiBoolean(enabled, 'DDGI indirect feedback');
    if (this._configuredIndirectFeedback === enabled) return;
    this._mutateWithLightingContentInvalidation(() => {
      this._pass.setIndirectFeedback(enabled);
      this._configuredIndirectFeedback = enabled;
    });
  }

  /**
   * Return the read-side atlas GPU textures from the underlying
   * ProbeUpdatePass. Forwarded from `HybridEngineFrameOrchestrator`.
   */
  getReadAtlasGPUTextures(): {
    irradiance: GPUTexture;
    visibility: GPUTexture;
  } | null {
    return this._pass.getReadAtlasGPUTextures();
  }

  /**
   * I5.3 facade — export the converged DDGI probe atlases to CPU
   * ({irrW, irrH, visW, visH, irrData, visData}). Returns null if the atlases
   * are not yet allocated. Async (atlas readback uses mapAsync).
   *
   * Callers (e.g. `HybridEngineGIState.exportGIStateImpl`) use this instead of
   * reaching through `DDGI.pass.exportAtlasData`.
   */
  async exportAtlasData(device: GPUDevice): Promise<ProbeAtlasSnapshot | null> {
    return this._pass.exportAtlasData(device);
  }

  /**
   * I5.3 facade — restore a previously-exported atlas snapshot into the live
   * probe atlases. Returns false (no-op) when the atlases are not allocated or
   * the snapshot dims do not match the current grid.
   *
   * Callers (e.g. `HybridEngineGIState.importGIStateImpl`) use this instead of
   * reaching through `DDGI.pass.importAtlasData`.
   */
  importAtlasData(
    device: GPUDevice,
    snap: ProbeAtlasSnapshot,
  ): boolean {
    return this._pass.importAtlasData(device, snap);
  }

  prepareAtlasImport(
    device: GPUDevice,
    snap: ProbeAtlasSnapshot,
  ): ProbeAtlasImportTransaction | null {
    return this._pass.prepareAtlasImport(device, snap);
  }

  /**
   * Probe-grid parameters (origin, spacing, dims, atlas sizes).
   * The returned cached object and its nested origin/dims are deeply frozen.
   */
  get gridParams(): ProbeGridParams {
    return this._grid.params;
  }

  /** Phase-0 productization (H1) — set the round-robin probe-update divisor.
   *  This is now the LOAD-BEARING cadence knob: it sets the round-robin
   *  `_stride`, which directly determines how many probes update each frame
   *  (`ceil(totalProbes / stride)` probes, one stratum of the grid). A higher
   *  divisor ⇒ more strata ⇒ fewer probes per frame ⇒ cheaper but slower GI
   *  response.
   *
   *  Must be a finite positive integer. The default (no call) is
   *  DEFAULT_STRIDE = 8. The quality
   *  presets thread an explicit divisor across a 2→32 spread: ultra=2 (fastest
   *  GI cadence), high=4, medium=8 (= the default), low=32 (cheapest). Because
   *  this knob is load-bearing, those preset values directly set how many probes
   *  update per frame. See HybridEngineQualityPreset.ts. */
  setProbeUpdateDivisor(divisor: number): void {
    const normalized = validateDdgiProbeUpdateDivisor(divisor);
    if (normalized === this._stride) return;
    this._pass.setProbeUpdateDivisor(normalized);
    // Warmup readiness proves one complete cycle of the cadence that produced
    // `_frame`. Reinterpreting a partial old cycle modulo a new divisor can
    // otherwise mark untouched strata ready (for example frame=4 at stride=8
    // changed to stride=2). A ready atlas already has complete coverage, so it
    // remains valid across a cadence-only change.
    if (!this._ready) this._frame = 0;
    this._stride = normalized;
    this._advanceContentEpoch();
  }

  // ── Probe cache invalidation ──────────────────────────────────────────────

  /**
   * Invalidate the DDGI probe atlas so it re-converges from scratch.
   *
   * Mechanism:
   *   1. Resets `_frame` to 0 and `_ready` to false so the warmup gate
   *      re-arms and the progress bar re-runs.
   *   2. Calls `ProbeUpdatePass.requestFullBlend()`, which arms every stratum
   *      in a new invalidation generation. Each stratum keeps uploading
   *      `hysteresis = 0.0` until its command buffer is accepted by
   *      `queue.submit` (`EMA weight = 0 → blendedValue = freshSample`).
   *      This overwrites every atlas texel in one successful probe-update
   *      stride window (~8 accepted frames at the default cadence) rather
   *      than fading in over hundreds of frames.
   *
   * Does NOT deallocate GPU textures or touch the BVH — cost is three JS
   * field writes only. Called by `HybridEngine.updateLighting()` when
   * lighting parameters change at runtime.
   */
  invalidateProbeCache(): void {
    this._frame = 0;
    this._ready = false;
    // H16 — fire hysteresis=0 for every pending stratum so the atlas clears
    // in one successful stride window instead of fading over hundreds of frames.
    this._pass.requestFullBlend(this._stride);
    this._advanceContentEpoch();
  }

  /**
   * PR-5 — TLAS transform-only refit: nudge probe temporal blend without a
   * full atlas wipe (geometry unchanged; instance matrices moved).
   */
  markInstancesDirty(): void {
    this._frame = Math.max(0, this._frame - Math.floor(this._stride / 2));
    this._advanceContentEpoch();
  }

  /**
   * PR-5.1 — point DDGI probe rays at the same BLAS/TLAS as ReSTIR.
   * Call each frame from HybridEngine when `_bvhBuffers` is ready.
   */
  syncRestirBvhBuffers(buffers: SceneBVHBuffers | null, scene?: Scene): void {
    const previous = this._restirSnapshot;
    const next = buffers == null
      ? null
      : refreshRestirBvhSnapshot(previous, buffers, scene);
    if (sameRestirSnapshotVersion(previous, next)) return;

    this._pass.setRestirBvhSnapshot(next);
    this._restirSnapshot = next;
    const tlasOnly = next != null &&
      isRestirTlasOnlySnapshotChange(previous, next);
    if (tlasOnly) {
      this._frame = Math.max(0, this._frame - Math.floor(this._stride / 2));
    } else {
      this._frame = 0;
      this._ready = false;
      this._pass.requestFullBlend(this._stride);
    }
    this._advanceContentEpoch();
  }

  // ── Per-frame update ──────────────────────────────────────────────────────
  prepareSceneMutation(
    buffers: SceneBVHBuffers | null,
    scene: Scene | undefined,
    options: { readonly invalidate: boolean; readonly instancesDirty: boolean },
  ): PreparedSceneMutation {
    const previousSnapshot = this._restirSnapshot;
    const nextSnapshot = buffers == null
      ? null
      : refreshRestirBvhSnapshot(previousSnapshot, buffers, scene);
    const previousFrame = this._frame;
    const previousReady = this._ready;
    const previousContentEpoch = this._contentEpoch;
    const previousFullBlend = this._pass.captureFullBlendState();
    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || committed) return;
        this._pass.setRestirBvhSnapshot(nextSnapshot);
        this._restirSnapshot = nextSnapshot;
        if (options.instancesDirty) {
          this._frame = Math.max(0, this._frame - Math.floor(this._stride / 2));
        }
        if (options.invalidate) {
          this._frame = 0;
          this._ready = false;
          this._pass.requestFullBlend(this._stride);
        }
        this._advanceContentEpoch();
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) {
          this._restirSnapshot = previousSnapshot;
          this._pass.setRestirBvhSnapshot(previousSnapshot);
          this._frame = previousFrame;
          this._ready = previousReady;
          this._contentEpoch = previousContentEpoch;
          this._pass.restoreFullBlendState(previousFullBlend);
        }
        closed = true;
      },
      finalize: () => {
        closed = true;
      },
    };
  }

  /**
   * Run one frame of DDGI compute.
   *
   * **Called internally by `HybridEngine.renderFrame` once per frame.**
   * The host does NOT need to call this when using `HybridEngine` — the
   * engine drives it as part of the normal `renderFrame` tick.
   *
   * Advanced standalone DDGI consumers (e.g. a DDGI-only host without the
   * full ReSTIR pipeline) may call this directly. The 60 FPS cap is enforced
   * internally — callers on high-refresh-rate displays will get a no-op on
   * frames that arrive too quickly.
   */
  updateFrame(inputs: DDGIFrameInputs): Promise<void> {
    assertDdgiBoolean(inputs.enabled, 'DDGI frame enabled');
    if (!inputs.enabled || this._disposed) return Promise.resolve();
    if (this._updateInFlight != null) return this._updateInFlight;

    const generation = this._lifecycleGeneration;
    const contentEpoch = this._contentEpoch;
    const operation = this._updateFrameOwned(inputs, generation, contentEpoch);
    const owned = operation
      .catch((error: unknown) => {
        if (this._isLifecycleCurrent(generation)) {
          this._ready = false;
          this._reportError(`[DDGI] unexpected updateFrame error: ${ddgiErrorMessage(error)}`, error);
        }
      })
      .finally(() => {
        if (this._updateInFlight === owned) this._updateInFlight = null;
      });
    this._updateInFlight = owned;
    return owned;
  }

  private _isLifecycleCurrent(generation: number): boolean {
    return !this._disposed && generation === this._lifecycleGeneration;
  }

  private async _updateFrameOwned(
    inputs: DDGIFrameInputs,
    generation: number,
    contentEpoch: number,
  ): Promise<void> {

    // 60 FPS frame cap.
    const now = performance.now();
    if (this._lastFrameTs !== 0 &&
        now - this._lastFrameTs < TARGET_FRAME_INTERVAL_MS) {
      return;
    }
    this._lastFrameTs = now;

    const t0 = now;

    const rendererAdapter = inputs.device
      ? { backend: { device: inputs.device, isWebGPUBackend: true as const } }
      : undefined;
    if (!rendererAdapter) {
      this._warn({
        code: 'walkaround-hybrid.ddgi-missing-device',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'DDGI.updateFrame',
        message: '[DDGI] updateFrame called without device; skipping.',
        details: { fallback: 'skip-ddgi-frame' },
      });
      if (!this._reportedMissingDevice) {
        this._reportedMissingDevice = true;
        this._reportError('[DDGI] updateFrame called without device; skipping.');
      }
      return;
    }
    this._reportedMissingDevice = false;

    // Initialize GPU on the first enabled frame (only try once). Lifecycle
    // state is published only after the awaited init belongs to the current
    // generation. A dispose during init invalidates the generation; any late
    // allocation is immediately retired and cannot resurrect this instance.
    if (!this._inited) {
      let ok = false;
      let initError: unknown;
      try {
        ok = await this._pass.init(rendererAdapter);
      } catch (e) {
        initError = e;
      }

      if (!this._isLifecycleCurrent(generation)) {
        // init() may have allocated after dispose() observed an empty pass.
        // ProbeUpdatePass.dispose() is idempotent and owns those resources.
        this._pass.dispose();
        return;
      }

      if (initError !== undefined) {
        console.error('[DDGI] GPU init threw:', initError);
        this._reportError(
          `[DDGI] GPU init threw: ${ddgiErrorMessage(initError)}`,
          initError,
        );
        this._inited = false;
        this._gpuOk = false;
        this._lastFrameMs = performance.now() - t0;
        return;
      }
      if (!ok && this._pass.initializationRetryable) {
        // ProbeUpdatePass uses false for both definitive no-device failures and
        // transient pipeline-compilation failures. Only the latter clears its
        // init-attempt guard. Keep this facade initializing so the next owned
        // frame retries instead of latching a recoverable failure forever.
        this._inited = false;
        this._gpuOk = false;
        this._lastFrameMs = performance.now() - t0;
        return;
      }
      this._inited = true;
      this._gpuOk = ok;
      if (!ok) {
        this._warn({
          code: 'walkaround-hybrid.ddgi-init-disabled',
          backend: 'walkaround-hybrid',
          phase: 'renderFrame',
          method: 'DDGI.updateFrame',
          message: '[DDGI] GPU init failed — DDGI compute disabled (scene still renders without indirect).',
          details: { fallback: 'disable-ddgi-compute' },
        });
        this._reportError('[DDGI] GPU init failed — DDGI compute disabled (scene still renders without indirect).');
        // Don't return — still update BVH and frame timing so standalone
        // hosts can observe the failed state without hanging their loop.
      }
    }

    let bvhOk = true;
    if (this._restirSnapshot == null) {
      try {
        if (inputs.coreScene != null) {
          this._bvh.updateFromCore(inputs.coreScene);
        } else {
          this._warn({
            code: 'walkaround-hybrid.ddgi-missing-core-scene',
            backend: 'walkaround-hybrid',
            phase: 'renderFrame',
            method: 'DDGI.updateFrame',
            message: '[DDGI] updateFrame called without a core scene; skipping BVH update.',
            details: { fallback: 'skip-ddgi-bvh-update' },
          });
        }
      } catch (e) {
        console.error('[DDGI] BVH update failed:', e);
        bvhOk = false;
        this._ready = false;
        this._reportError(`[DDGI] BVH update failed: ${ddgiErrorMessage(e)}`, e);
      }
    }

    if (!bvhOk) {
      this._lastFrameMs = performance.now() - t0;
      return;
    }

    const boundsBox = this._restirSnapshot?.boundingBox ?? this._bvh.buffers?.boundingBox;
    if (boundsBox) {
      this._grid.computeFromBounds(boundsBox, this._probeSpacing, this._maxProbesPerAxis);
      if (this._grid.dirty || !this._grid.irradianceA) {
        this._reallocateGridAtlasesWithWarmupReset();
      }
    }

    // Round-robin: update 1/_stride of probes this frame. `_stride` is the
    // probe-update divisor set via setProbeUpdateDivisor (default 8).
    const stride = this._stride;
    let probeFrameOk = false;
    if (this._gpuOk) {
      const offset = this._frame % stride;
      try {
        const submitted = await this._pass.runFrame(rendererAdapter, offset, stride);
        if (!this._isLifecycleCurrent(generation)) return;
        if (submitted && contentEpoch === this._contentEpoch) {
          this._advanceFrameIndex();
          probeFrameOk = true;
        }
      } catch (e) {
        if (!this._isLifecycleCurrent(generation)) return;
        console.error('[DDGI] runFrame error:', e);
        this._ready = false;
        this._reportError(`[DDGI] runFrame error: ${ddgiErrorMessage(e)}`, e);
      }
    }

    // Mark ready after the first full cycle (`_stride` frames).
    // H24-B — only flip _ready when _gpuOk so a failed GPU init cannot silently
    // advertise convergence (the state() accessor returns 'failed' in that case).
    if (this._gpuOk && probeFrameOk && this._frame >= stride) {
      this._ready = true;
    }

    this._lastFrameMs = performance.now() - t0;

    // Debug window global (guarded by typeof + debug flag).
    if (this._debug && typeof window !== 'undefined') {
      (window as unknown as Record<string, unknown>)['__DDGI__'] = {
        ready: this._ready,
        lastFrameMs: this._lastFrameMs,
        probeCount: this._grid.probeCount,
      };
    }
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  /** Free all GPU resources. Safe to call even before updateFrame. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._lifecycleGeneration = (this._lifecycleGeneration + 1) >>> 0;
    if (this._lifecycleGeneration === 0) this._lifecycleGeneration = 1;
    try { this._pass.dispose(); } catch { /* continue retiring independent owners */ }
    try { this._grid.dispose(); } catch { /* continue retiring independent owners */ }
    try { this._bvh.dispose(); } catch { /* lifecycle state still becomes terminal */ }
    this._ready = false;
    this._inited = false;
    this._gpuOk = false;
  }
}

function sameNumber(a: number | undefined, b: number | undefined): boolean {
  return Object.is(a, b);
}

function sameRestirSnapshotVersion(
  a: RestirBvhSnapshot | null,
  b: RestirBvhSnapshot | null,
): boolean {
  return restirBvhSnapshotStateEqual(a, b);
}

function sameVec3(
  a: { readonly x: number; readonly y: number; readonly z: number } | undefined,
  b: { readonly x: number; readonly y: number; readonly z: number } | undefined,
): boolean {
  return a === undefined
    ? b === undefined
    : b !== undefined &&
      sameNumber(a.x, b.x) && sameNumber(a.y, b.y) && sameNumber(a.z, b.z);
}

function sameColor(
  a: { readonly r: number; readonly g: number; readonly b: number } | undefined,
  b: { readonly r: number; readonly g: number; readonly b: number } | undefined,
): boolean {
  return a === undefined
    ? b === undefined
    : b !== undefined &&
      sameNumber(a.r, b.r) && sameNumber(a.g, b.g) && sameNumber(a.b, b.b);
}

function sameDdgiLight(a: DDGILight, b: DDGILight): boolean {
  return a.kind === b.kind &&
    a.id === b.id &&
    sameNumber(a.intensity, b.intensity) &&
    a.on === b.on &&
    a.castShadow === b.castShadow &&
    sameVec3(a.position, b.position) &&
    sameVec3(a.direction, b.direction) &&
    sameNumber(a.angularRadius, b.angularRadius) &&
    sameColor(a.color, b.color) &&
    sameVec3(a.spotAxis, b.spotAxis) &&
    sameNumber(a.spotCosInner, b.spotCosInner) &&
    sameNumber(a.spotCosOuter, b.spotCosOuter) &&
    sameNumber(a.distance, b.distance) &&
    sameNumber(a.decay, b.decay);
}

function sameDdgiLights(a: readonly DDGILight[], b: readonly DDGILight[]): boolean {
  return a.length === b.length && a.every((light, index) => {
    const other = b[index];
    return other !== undefined && sameDdgiLight(light, other);
  });
}

function sameEmitterPayload(
  previous: Float32Array,
  previousCount: number,
  next: Float32Array,
  nextCount: number,
): boolean {
  if (!Number.isInteger(nextCount) || nextCount < 0 || previousCount !== nextCount) {
    return false;
  }
  const requiredFloats = nextCount * 20;
  if (next.length < requiredFloats || previous.length !== requiredFloats) return false;
  for (let i = 0; i < requiredFloats; i++) {
    if (!Object.is(previous[i], next[i])) return false;
  }
  return true;
}

function ddgiErrorMessage(error: unknown): string {
  if (error instanceof Error && typeof error.message === 'string') return error.message;
  if (typeof error === 'string') return error;
  try {
    return String(error);
  } catch {
    return 'unknown error';
  }
}
