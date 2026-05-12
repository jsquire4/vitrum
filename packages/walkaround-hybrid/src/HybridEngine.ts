/**
 * HybridEngine — WebGPU layered DDGI + ReSTIR DI engine.
 *
 * Class-based extraction of `useHybridLayeredGI.ts`. All React hooks stripped:
 *   - useRef      → private class fields
 *   - useState    → private class fields
 *   - useEffect   → initialize() + dispose() + setScene()
 *   - useFrame    → renderFrame()
 *   - useThree    → GPUDevice + canvas dimensions passed via factory opts
 *
 * Implements `@vitrum/core`'s `Engine` interface so a host can swap this
 * backend interchangeably with `@vitrum/pt-webgl`.
 *
 * RC subsystem: shade pass does not sample Lo_rc — see
 * `plan/walkaround-without-three.md` for the re-integration plan.
 *
 * Debug globals:
 *   The original hook wrote to `window.__WGPU__.walkaround` and
 *   `window.__HYBRID_LAYERS__`. Those are host-bridge responsibilities.
 *   This class exposes `setLayerEnabled()` so the host can wire layer
 *   toggles; it calls `window.__WGPU__` only inside a debug branch
 *   guarded by `typeof window !== 'undefined'` and the `debug` option.
 */

import * as THREE from 'three';
import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineOptions,
  EngineState,
} from '@vitrum/core';
import type { Scene } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import { packDDGIGridParams } from './pipeline/resourceManager.js';
import { buildReSTIRSceneBVH, disposeSceneBVH } from './restir/bvhCompute.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import { vitrumSceneToThree, disposeVitrumThreeSceneRoot } from '@vitrum/three-bindings';
import { InferenceGraph } from './neural/InferenceGraph.js';
import type { ModelWeights } from './neural/weights.js';

/** Default per-frame target interval (~60 FPS soft-cap). */
const DEFAULT_TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/**
 * Runtime-mutable lighting parameters for {@link HybridEngine.updateLighting}.
 * All fields are optional; omitting a field leaves the corresponding engine
 * parameter unchanged.
 */
export interface LightingOptions {
  /** Primary directional light direction (world-space, normalised). */
  primaryLightDir?: [number, number, number];
  /** Primary directional light intensity (linear, unitless). */
  primaryLightIntensity?: number;
  /** Diffuse-sky-dome RGB tint. */
  skyTint?: [number, number, number];
  /** Sky-dome irradiance scalar paired with {@link skyTint}. */
  skyIrradiance?: number;
}

export interface HybridEngineOptions extends EngineOptions {
  /** WebGPU device (narrowed from the opaque `device: unknown` on EngineOptions). */
  readonly device: GPUDevice;

  /** Physical pixel width of the render surface. */
  readonly width: number;

  /** Physical pixel height of the render surface. */
  readonly height: number;

  /**
   * Predicate the engine polls before kicking off ReSTIR pipeline init.
   * Returns true when the scene has enough geometry to build a BVH.
   * Defaults to the `defaultIsSceneReady` heuristic (any triangle present).
   * Override if your scene loads asynchronously and you need a different
   * signal (e.g. wait for a specific async asset, or require N triangles).
   */
  readonly isSceneReady?: () => boolean;

  /**
   * Stable signal sampled at ctor (`pipelineRebuildKey`) and/or dynamically
   * via {@link getPipelineRebuildKey}. When the effective value changes compared
   * to the previous frame's sample, {@link HybridEngine.reset} runs so the GPU
   * pipeline is recreated (same `_lastScene` / `THREE` graph).
   */
  readonly pipelineRebuildKey?: string | number | null;

  /**
   * Optional callback polled at the **start** of each {@link HybridEngine.renderFrame}
   * (after state guards). Takes precedence over {@link pipelineRebuildKey} when
   * supplied. Enables hosts to invalidate the pipeline without `setScene()`.
   */
  readonly getPipelineRebuildKey?: () => string | number | null | undefined;

  /**
   * Primary directional light direction (world-space, normalised).
   * Used for both BVH-build-time emitter list construction AND per-frame
   * sun-shadow casting. The two MUST match exactly for self-emission Le
   * to reproduce correctly.
   */
  readonly primaryLightDir: [number, number, number];

  /** Primary directional light intensity (linear, unitless). */
  readonly primaryLightIntensity: number;

  /**
   * Diffuse-sky-dome RGB tint. Consumed by the sky-aperture probe and
   * second-bounce sky-miss paths.
   */
  readonly skyTint: [number, number, number];

  /** Sky-dome irradiance scalar paired with skyTint. */
  readonly skyIrradiance: number;

  /** Host `THREE.Scene` — BVH / DDGI fallback when `setScene` has no mesh primitives. */
  readonly threeScene: THREE.Scene;

  /** Light list for DDGI probe update pass. */
  readonly lights?: DDGILight[];

  /** When true, enables informational ReSTIR pipeline logs (initialization / shader compile). */
  readonly verbose?: boolean;

  /**
   * When true, enables debug logging and exposes
   * `window.__DDGI__` inside `typeof window !== 'undefined'` guards.
   */
  readonly debug?: boolean;

  /**
   * Post-shade denoiser:
   *
   *   `'atrous-variance'` (default) — temporal Welford + à-trous + variance
   *   scalar lookup; honest about what it does (not Schied 2017 SVGF).
   *
   *   `'atrous'` — legacy three-pass edge-stopping à-trous only.
   *
   *   `'svgf-real'` — T2.H1 — full Schied 2017 SVGF: bilinear motion-vector
   *   reprojection, depth+normal+objId disocclusion test (Eq. 2), per-pixel
   *   history-length texture (Eq. 3), EMA α=max(α_min, 1/(h+1)) (Eq. 4),
   *   variance-from-moments (Eq. 5), 7×7 spatial fallback for disoccluded pixels
   *   (§4.3). Requires historyLength (r16uint) + momentsHistory (rg32float) +
   *   prevRadiance (rgba16float) persistent textures: ~52 MB at 1080p.
   *
   *   `'svgf'` is a deprecated alias for `'atrous-variance'`; triggers a
   *   one-time console warning.
   *
   *   `'neural'` — T2.H2 — GPU U-Net denoiser (Chaitanya et al. 2017 / Ronneberger
   *   et al. 2015). Requires `neuralWeights` to be provided. Default still
   *   `'atrous-variance'`; neural is opt-in. See tools/neural-denoiser-training/README.md.
   */
  readonly denoiser?: 'atrous' | 'atrous-variance' | 'svgf-real' | 'svgf' | 'neural';

  /**
   * Pre-loaded model weights for the neural denoiser (T2.H2).
   * Required when `denoiser === 'neural'`. Load via `loadWeightsFromArrayBuffer()`
   * from the vitrum binary format exported by `tools/neural-denoiser-training/export_weights.py`.
   *
   * If `denoiser === 'neural'` and `neuralWeights` is undefined, the engine
   * constructor throws with a helpful error pointing to the training README.
   */
  readonly neuralWeights?: ModelWeights;

  // ── Library-generality knobs (audit follow-up) ──────────────────────────
  // All optional; defaults preserve Cornell-test-scene behaviour byte-for-
  // byte. Hosts targeting other scene scales / intensities should set them.

  /**
   * Per-frame render-interval cap in **milliseconds**. Null disables the
   * cap (every rAF call dispatches a frame). Pass `1000/30 - 1` for a 30
   * FPS ceiling, `1000/120 - 1` for 120 FPS.  Default `1000/60 - 1` (~60
   * FPS soft-cap).  Scene-independent — purely a host-side governor.
   *
   * @default 1000/60 - 1
   */
  readonly targetFrameIntervalMs?: number | null;

  /**
   * Camera squared-distance threshold (**world-space units²**) for
   * resetting the temporal accumulator.  When the camera moves more than
   * `sqrt(threshold)` units in one frame, the accumulator's history is
   * discarded and accumulation restarts at α=1.
   *
   * **Scene-scale-sensitive**.  Default `1.0` is tuned to Cornell's ~2-unit
   * room. For a 100-unit city block, this never trips (permanent ghosting);
   * for a 1-unit jewellery scene, every micro-movement trips it. Recommended
   * default for hosts is `(sceneDiagonal × 0.001)²`.
   *
   * @default 1.0
   */
  readonly cameraMoveResetThresholdSq?: number;

  /**
   * Per-frame temporal-accumulator EMA weight.  `1.0` = no history (single
   * frame), `0.01` = 99% history retain.
   *
   * **Framerate-sensitive**.  Default `0.01` is tuned for ~60 FPS Cornell
   * convergence. At 30 FPS the same α doubles temporal lag; at 120 FPS it
   * halves convergence-back-to-steady-state after a camera stop. For
   * FPS-independent feel, set `1 - exp(-frameTime × k)` for a chosen k.
   *
   * @default 0.01
   */
  readonly temporalAccumAlpha?: number;

  /**
   * Emitter-geometry-term distance² floor (audit M12).  Clamps
   * `G = (n_l · ω) / max(dist², emitterDist2Floor)` to prevent G blowup
   * for receivers within sqrt(floor) of an emitter.
   *
   * **Scene-scale-sensitive**.  Default `0.01` (10 cm minimum effective
   * distance) for Cornell-scale.  Hosts on different scales should pass
   * `(sceneDiagonal × 1e-3)²` so the floor scales with scene extent.
   *
   * @default 0.01
   */
  readonly emitterDist2Floor?: number;

  /**
   * Per-channel HDR clamp on the direct radiance channel before the
   * atrous-variance denoiser (audit B4). Suppresses fireflies from ReSTIR-DI's
   * stochastic light-point selection on glancing-angle BRDF evaluations.
   *
   * **Light-intensity-sensitive**.  Default `4.0` is calibrated for
   * Le=12 (`4 / π × 12 ≈ 15`, clamped at 4).  For brighter scenes
   * compute `~4 × luminance(maxEmitterLe)`.
   *
   * @default 4.0
   */
  readonly directFireflyClamp?: number;

  /**
   * Stained-glass caustic boost (audit B1).  Multiplies the through-glass
   * sun-shadow-ray contribution.  Cornell's stained-glass test scene uses
   * `{ boost: 22, visClamp: 0.6 }` to compensate for Brown-Beer-Lambert
   * attenuation; generic scenes should leave this at defaults (no boost,
   * no clamp).
   *
   * @default { boost: 1.0, visClamp: 1.0 }
   */
  readonly caustic?: {
    readonly boost?: number;
    readonly visClamp?: number;
  };

  /**
   * ReSTIR-DI temporal M-clamp (audit M6).  Caps the previous-frame
   * reservoir's `M` before combining into this frame's reservoir.
   * Higher = stickier history (slower to respond to lighting changes
   * but lower variance).
   *
   * **Framerate-sensitive**.  Default 20 frames ≈ 333 ms history at
   * 60 FPS.  At 15 FPS this stretches to 1.3 s; at 120 FPS it compresses
   * to 167 ms.  For FPS-independent feel: `round(0.3 / frameTimeSeconds)`.
   *
   * @default 20
   */
  readonly temporalMClampDI?: number;

  /**
   * ReSTIR-DI spatial-reuse radius in **pixels** (audit M7).  The Poisson
   * disk for neighbour sampling extends this far from the centre pixel.
   *
   * **Resolution-sensitive**.  Default `30` is calibrated for ~1080p–4K.
   * At 480p reuse stretches across geometry boundaries; at 8K it stays
   * very local.  Suggested host derivation: `screenHeight × 0.025`.
   *
   * @default 30
   */
  readonly spatialReuseRadiusPx?: number;

  /**
   * ReSTIR-DI spatial-reuse depth-tolerance world-units floor (audit M8).
   * Neighbours whose depth differs by less than this absolute value are
   * accepted regardless of relative tolerance.
   *
   * **Scene-scale-sensitive**.  Default `0.05` (5 cm) for Cornell-scale.
   * Hosts on cm-scale scenes should use ~`sceneDiagonal × 1e-3`.
   *
   * @default 0.05
   */
  readonly spatialDepthTolFloor?: number;

  /**
   * Adaptive-sampling tier classifier thresholds (audit M2).  The
   * sample-budget pass reads previous-frame Welford variance and writes
   * a per-pixel tier (1 / 2 / 4) used downstream by RIS to scale M_GI.
   *
   * **Light-intensity-sensitive** — variance scales with peak-radiance²,
   * so HDR scenes need higher thresholds.  Default `[0.01, 0.10]` is
   * calibrated for Cornell's variance dynamic range.
   *
   * @default [0.01, 0.10]
   */
  readonly adaptiveSamplingThresholds?: readonly [low: number, high: number];

  /**
   * GTAO (ground-truth ambient occlusion) tuning (audits M1, B3).  All
   * fields optional.  Defaults preserve Cornell behaviour.
   *
   * - `radiusPx` (resolution-sensitive): sampling radius in screen-space
   *   pixels. Default 32; consider `screenHeight × ~0.025` for
   *   resolution-independent feel.
   * - `intensity`: AO exponent (`ao = pow(raw, intensity)`). Default 2.0.
   * - `depthThresholdWorldUnits` (scene-scale-sensitive): max depth
   *   discontinuity to include in the horizon test.  Default 2.0 (Cornell
   *   ~2 m room); large-scale scenes should use ~`sceneDiagonal × 0.05`.
   * - `bilateralDepthSigma` (scene-scale-sensitive): σ for the bilateral
   *   upsample's depth-weight Gaussian (world units).  Default 0.25.
   *   Hosts should set ~`sceneDiagonal × 0.01`.
   */
  readonly gtao?: {
    readonly radiusPx?: number;
    readonly intensity?: number;
    readonly depthThresholdWorldUnits?: number;
    readonly bilateralDepthSigma?: number;
  };

  /**
   * Möller-Trumbore coplanarity epsilon (D12 / audit M3 follow-up).
   * Controls the `abs(det) < ε` near-zero determinant test in
   * `intersectTriangle` in the ReSTIR WGSL.  A too-small value causes
   * grazing-angle rays to incorrectly miss coplanar triangles; a too-large
   * value rejects valid near-coplanar hits.
   *
   * **Scene-scale-sensitive.**  Default `1e-5` is correct for metre-scale.
   * For millimetre-scale geometry, try `1e-7`; for kilometre-scale, `1e-3`.
   *
   * @default 1e-5
   */
  readonly triIntersectEpsilon?: number;

  // ── PPG (T2.H3 — Practical Path Guiding, Müller et al. 2017) ──────────────

  /**
   * Enable the Müller 2017 Practical Path Guiding subsystem.
   *
   * When `true`, the engine instantiates an adaptive spatial tree (sTree)
   * and per-cell directional trees (dTree) per §3.1–3.2. Training runs
   * via `ppgUpdate.wgsl.ts` (incoming radiance L_i, world frame). Guiding
   * mixes the learned PDF with BSDF sampling via MIS (§3.4).
   *
   * Default: `false` (PPG is an opt-in feature; default remains BSDF-only).
   */
  readonly ppgEnabled?: boolean;

  /**
   * Maximum number of sTree spatial cells (hard cap on adaptive splits).
   * Each cell consumes memory for a flat dTree node buffer on the GPU.
   *
   * Default: 16 384 (matches `PPG_MAX_SPATIAL_CELLS`).
   */
  readonly ppgMaxSpatialCells?: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Scene-readiness helper
// ────────────────────────────────────────────────────────────────────────────

/** Default scene-readiness predicate: counts triangles via THREE.Scene.traverse. */
function defaultIsSceneReady(scene: THREE.Scene): boolean {
  let total = 0;
  scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const idx = mesh.geometry.index;
    total += idx ? idx.count / 3 : (mesh.geometry.attributes['position']?.count ?? 0) / 3;
  });
  // Audit M5: was `total >= 200`, calibrated to Cornell-scale scenes.
  // Procedurally-generated terrain or sparse-geometry scenes may have far
  // fewer triangles and a perfectly valid BVH; the 200 floor silently
  // blocked them. Hosts with stricter readiness signals supply their own
  // `isSceneReady` callback.
  return total > 0;
}

/** Get the preferred swap-chain format from the browser GPU. */
function getPreferredSwapChainFormat(): GPUTextureFormat {
  return (typeof navigator !== 'undefined' && 'gpu' in navigator
    ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
        .getPreferredCanvasFormat?.() ?? 'bgra8unorm'
    : 'bgra8unorm') as GPUTextureFormat;
}

// ────────────────────────────────────────────────────────────────────────────
// HybridEngine
// ────────────────────────────────────────────────────────────────────────────

export class HybridEngine implements Engine {

  // ── Engine contract fields ─────────────────────────────────────────────
  private _state: EngineState = 'uninitialized';
  readonly capabilities: EngineCapabilities;

  get state(): EngineState {
    return this._state;
  }

  // ── Creation-time options (immutable after construction) ───────────────
  private readonly _device:               GPUDevice;
  private readonly _width:                number;
  private readonly _height:               number;
  private readonly _threeScene:           THREE.Scene;
  private readonly _isSceneReady:         () => boolean;
  // Lighting fields are NOT readonly — updateLighting() mutates them at runtime.
  private _primaryLightDir:               [number, number, number];
  private _primaryLightIntensity:         number;
  private _skyTint:                       [number, number, number];
  private _skyIrradiance:                 number;
  private readonly _ctorLights:           readonly DDGILight[];
  private readonly _debug:                boolean;
  private readonly _verbose:             boolean;
  private readonly _maxBounces:           number;

  /** Rolling window of per-frame timings (newest last, cap 240 entries).
   *  Only populated when `debug === true`. Hosts that want a UI gauge
   *  should poll {@link debugTimings} instead of reaching into globals. */
  private readonly _debugTimings: Array<{ t: number; ms: number }> = [];

  /** Read-only snapshot of recent frame timings collected when the engine
   *  was constructed with `debug: true`. Returns an empty array when debug
   *  is off (no allocation cost is paid in production). */
  get debugTimings(): readonly { t: number; ms: number }[] {
    return this._debugTimings;
  }

  /** Per-pass GPU timings in milliseconds, keyed by the same `PassLabel`
   *  set the timestamp-query subsystem uses. Empty record when the active
   *  adapter doesn't expose `timestamp-query` or when the engine is not yet
   *  initialised. Useful for dev panels + telemetry harnesses. */
  get lastGpuTimings(): Record<string, number> {
    const p = this._pipeline as unknown as {
      lastGpuTimings?: Record<string, number>;
    } | null;
    return p?.lastGpuTimings ?? {};
  }

  /** Frame index that produced {@link lastGpuTimings}; -1 if no readback
   *  has resolved yet. */
  get lastGpuTimingsFrame(): number {
    const p = this._pipeline as unknown as {
      lastGpuTimingsFrame?: number;
    } | null;
    return p?.lastGpuTimingsFrame ?? -1;
  }

  /**
   * Diagnostic one-shot GPU timing readback (P3-Vδ). Bypasses the
   * production fire-and-forget ping-pong path and synchronously awaits a
   * fresh staging-buffer mapAsync. Use this from telemetry probes that
   * need a confirmed-fresh per-pass timing snapshot. Returns empty
   * objects when the engine isn't ready or the device lacks the
   * `timestamp-query` feature.
   */
  async readGpuTimingsOnce(): Promise<{ perPass: Record<string, number>; rawBigints: string[] }> {
    const p = this._pipeline as unknown as {
      readGpuTimingsOnce?: () => Promise<{ perPass: Record<string, number>; rawBigints: string[] }>;
    } | null;
    if (!p?.readGpuTimingsOnce) return { perPass: {}, rawBigints: [] };
    return p.readGpuTimingsOnce();
  }

  private readonly _denoiser: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural';
  /** T2.H2 — neural denoiser weights (populated when _denoiser === 'neural'). */
  private readonly _neuralWeights: ModelWeights | undefined;
  /** Audit M4 — null disables the FPS cap; configured at construction. */
  private readonly _targetFrameIntervalMs: number | null;
  /** Audit B8 — passed to WalkaroundGPUPipeline at initialize() time. */
  private readonly _cameraMoveResetThresholdSq: number;
  /** Audit M3 — passed to WalkaroundGPUPipeline at initialize() time. */
  private readonly _temporalAccumAlpha: number;
  /** Audit M12 — written into WalkaroundUBO each frame. */
  private readonly _emitterDist2Floor: number;
  /** Audit B4 — written into WalkaroundUBO each frame. */
  private readonly _directFireflyClamp: number;
  /** Audit B1 — written into WalkaroundUBO each frame. */
  private readonly _causticBoost: number;
  /** Audit B1 — written into WalkaroundUBO each frame. */
  private readonly _causticVisClamp: number;
  /** Audit M6 — written into WalkaroundUBO each frame. */
  private readonly _temporalMClampDI: number;
  /** Audit M7 — written into WalkaroundUBO each frame. */
  private readonly _spatialReuseRadiusPx: number;
  /** Audit M8 — written into WalkaroundUBO each frame. */
  private readonly _spatialDepthTolFloor: number;
  /** Audit M1 — written into GTAO UBO each frame. */
  private readonly _gtaoRadiusPx: number;
  /** Audit M1 — written into GTAO UBO each frame. */
  private readonly _gtaoIntensity: number;
  /** Audit M1 — written into GTAO UBO each frame. */
  private readonly _gtaoDepthThreshold: number;
  /** Audit B3 — written into GTAO UBO each frame. */
  private readonly _gtaoBilateralDepthSigma: number;
  /** Audit M2 — written into sample-budget UBO each frame. */
  private readonly _adaptiveSamplingThresholdLow: number;
  /** Audit M2 — written into sample-budget UBO each frame. */
  private readonly _adaptiveSamplingThresholdHigh: number;
  /** D12 — written into WalkaroundUBO each frame. */
  private readonly _triIntersectEpsilon: number;

  // ── Pipeline state ─────────────────────────────────────────────────────
  private _pipeline:    WalkaroundGPUPipeline | null = null;
  private _bvhBuffers:  SceneBVHBuffers | null       = null;

  // ── Scene (from @vitrum/core contract) ────────────────────────────────
  /** Last scene passed via `setScene()`. When it contains `mesh` primitives,
   *  ReSTIR BVH + DDGI probe walks use `vitrumSceneToThree(this._lastScene)`. */
  private _lastScene: Scene | null = null;

  /** Owned `THREE.Scene` from `vitrumSceneToThree` when BVH/DDGI follow the core
   *  contract; disposed on pipeline teardown. Null when falling back to ctor `threeScene`. */
  private _ddgiTraversalScene: THREE.Scene | null = null;

  // ── DDGI subsystem ─────────────────────────────────────────────────────
  private _ddgi:    DDGI;
  private _ddgiOn:  boolean = true;

  // ── Per-frame throttle ─────────────────────────────────────────────────
  private _lastFrameTs = 0;

  // ── Diagnostic counters (debug only) ──────────────────────────────────
  private _dbg = {
    initStart:          0,
    initCount:          0,
    disposeCount:       0,
    skipNoPipeline:     0,
    skipNoBvh:          0,
    skipNoSwapView:     0,
    skipFrameInterval:  0,
    framesDispatched:   0,
    lastReportTs:       0,
  };

  // ── Layer toggles (debug console interface) ────────────────────────────
  private _layerEnabled: Map<string, boolean> = new Map([
    ['ddgi', true],
  ]);

  // ── Initialisation cancellation ───────────────────────────────────────
  /** Set to true by dispose() to cancel an in-flight async init. */
  private _disposed = false;

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _staticPipelineRebuildKey: string | number | null;
  private readonly _getPipelineRebuildKey: (() => string | number | null | undefined) | undefined;

  constructor(opts: HybridEngineOptions) {
    this._device                = opts.device;
    this._width                 = opts.width;
    this._height                = opts.height;
    this._threeScene            = opts.threeScene;
    this._primaryLightDir       = opts.primaryLightDir;
    this._primaryLightIntensity = opts.primaryLightIntensity;
    this._skyTint               = opts.skyTint;
    this._skyIrradiance         = opts.skyIrradiance;
    this._debug                 = opts.debug ?? false;
    this._verbose               = opts.verbose ?? false;
    this._maxBounces            = opts.maxBounces ?? 4;
    // Audit B7: validate the denoiser option at construction so an unsupported
    // value (e.g. `'none'`, `'bmfr'`, `'oidn-final'` from the @vitrum/core
    // EngineOptions contract) does not silently coerce to atrous-variance and produce
    // wrong output. Supported values are explicitly enumerated here.
    // 'svgf' is accepted as a deprecated alias — logs a one-time warning, then normalises.
    if (
      opts.denoiser !== undefined &&
      opts.denoiser !== 'atrous' &&
      opts.denoiser !== 'atrous-variance' &&
      opts.denoiser !== 'svgf-real' &&
      opts.denoiser !== 'svgf' &&
      opts.denoiser !== 'neural'
    ) {
      throw new TypeError(
        `[HybridEngine] unsupported denoiser '${opts.denoiser}'. ` +
        `walkaround-hybrid supports: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural'. ` +
        `If you need 'none' / 'bmfr' / 'oidn-final' from @vitrum/core, ` +
        `pick a backend that implements those modes.`,
      );
    }
    // T2.H2 — 'neural' requires neuralWeights to be provided.
    if (opts.denoiser === 'neural' && !opts.neuralWeights) {
      throw new TypeError(
        `[HybridEngine] denoiser: 'neural' requires neuralWeights to be provided. ` +
        `Load weights via loadWeightsFromArrayBuffer() from a .vitrum-model file, ` +
        `or train one with tools/neural-denoiser-training/train.py. ` +
        `See tools/neural-denoiser-training/README.md for instructions.`,
      );
    }
    if (opts.denoiser === 'svgf') {
      console.warn(
        `[walkaround-hybrid] denoiser: 'svgf' is deprecated; use 'atrous-variance'. ` +
        `The shipping implementation is à-trous + variance scalar lookup, NOT real Schied 2017 SVGF. ` +
        `For real Schied 2017 SVGF, pass denoiser: 'svgf-real' (T2.H1).`,
      );
    }
    this._denoiser = opts.denoiser === 'svgf' ? 'atrous-variance' : (opts.denoiser ?? 'atrous-variance');
    this._neuralWeights = opts.neuralWeights;
    this._targetFrameIntervalMs = opts.targetFrameIntervalMs !== undefined
      ? opts.targetFrameIntervalMs
      : DEFAULT_TARGET_FRAME_INTERVAL_MS;
    this._cameraMoveResetThresholdSq = opts.cameraMoveResetThresholdSq ?? 1.0;
    this._temporalAccumAlpha    = opts.temporalAccumAlpha ?? 0.01;
    // Library-generality tunables. Defaults preserve Cornell behaviour;
    // hosts on different scene scales / intensities should override.
    this._emitterDist2Floor     = opts.emitterDist2Floor    ?? 0.01;
    this._directFireflyClamp    = opts.directFireflyClamp   ?? 4.0;
    this._causticBoost          = opts.caustic?.boost       ?? 1.0;
    this._causticVisClamp       = opts.caustic?.visClamp    ?? 1.0;
    this._temporalMClampDI      = opts.temporalMClampDI     ?? 20;
    this._spatialReuseRadiusPx  = opts.spatialReuseRadiusPx ?? 30.0;
    this._spatialDepthTolFloor  = opts.spatialDepthTolFloor ?? 0.05;
    // GTAO defaults match the previous hard-coded values in the pipeline.
    this._gtaoRadiusPx          = opts.gtao?.radiusPx                ?? 32.0;
    this._gtaoIntensity         = opts.gtao?.intensity               ?? 2.0;
    this._gtaoDepthThreshold    = opts.gtao?.depthThresholdWorldUnits ?? 2.0;
    this._gtaoBilateralDepthSigma = opts.gtao?.bilateralDepthSigma   ?? 0.25;
    this._adaptiveSamplingThresholdLow  = opts.adaptiveSamplingThresholds?.[0] ?? 0.01;
    this._adaptiveSamplingThresholdHigh = opts.adaptiveSamplingThresholds?.[1] ?? 0.10;
    this._triIntersectEpsilon    = opts.triIntersectEpsilon ?? 1e-5;
    this._isSceneReady          = opts.isSceneReady ?? (() => defaultIsSceneReady(this._threeScene));

    this._staticPipelineRebuildKey = opts.pipelineRebuildKey ?? null;
    this._getPipelineRebuildKey     = opts.getPipelineRebuildKey;
    this._rebuildKeyFingerprintSeen = HybridEngine._fingerprintRebuildKey(
      opts.getPipelineRebuildKey?.() ?? opts.pipelineRebuildKey ?? null,
    );

    this._ddgi = new DDGI({ debug: this._debug });
    this._ctorLights = opts.lights ?? [];
    if (this._ctorLights.length > 0) {
      this._ddgi.setLights(this._ctorLights as DDGILight[]);
    }

    this.capabilities = {
      supportsIncrementalScene:  false,
      // supportsMotionBlur === false. WalkaroundGPUPipeline does allocate a
      // motionVectorTexture, but it's for SVGF temporal reprojection (encoded
      // as `motion-vectors-zero` — a 2D screen-space delta), not for accumu-
      // lating samples across a shutter interval. True motion-blur SPP
      // accumulation is incompatible with the walkaround engine's per-frame
      // cadence — see resourceManager.ts ("motion-vectors-zero" label).
      supportsMotionBlur:        false,
      supportsAuxBuffers:        false,
      accumulates:               false,
      maxSamplesPerPixel:        Infinity,
      maxBounces:                this._maxBounces,
      supportedAnalyticShapes:   new Set<string>(),
      // Emitter kinds handled by DDGI _uploadLights: sun, fixture, teaLight
      // mapped to core taxonomy: directional, point
      supportedEmitterKinds:     new Set<string>(['directional', 'point']),
      // RFE-05: Real-time caustic strategies (MNEE / photon-map) are not
      // compatible with the walkaround engine's frame cadence; the walkaround
      // engine always reports 'none'. Track via
      // external_requests/05-manifold-nee.md §4 for the approved approximation
      // path if real-time caustic approximation is added.
      causticStrategy: 'none',
    };
  }

  // ── Scene management ───────────────────────────────────────────────────

  /**
   * Replace the scene. Triggers a full pipeline reinitialisation
   * (BVH rebuild + ReSTIR pipeline re-init).
   *
   * **BVH + DDGI geometry:** When `setScene` receives a `Scene` with at least
   * one `mesh` primitive, ReSTIR `buildSceneBVH` and DDGI probe traversal use
   * `vitrumSceneToThree` (same path as `pt-webgl`). Otherwise the ctor
   * `threeScene` is the BVH + DDGI source (hosts with only Three.js graphs).
   *
   * **Host guidance:** For one `Scene` driving both `pt-webgl` and this engine,
   * pass `setScene(sceneFromThreeJS(yourThreeScene))` and keep ctor `threeScene`
   * in sync for the non-mesh case and for auxiliary Object3D state you have not
   * serialized into the core `Scene`.
   *
   * @param scene - The `@vitrum/core` scene (e.g. from `sceneFromThreeJS`).
   */
  setScene(scene: Scene): void {
    this._lastScene = scene;

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    void this._initPipeline();
  }

  updatePrimitive?: never;
  updateEmitter?: never;

  // ── Runtime lighting update ────────────────────────────────────────────

  /**
   * Runtime update of the primary directional light + sky parameters.
   *
   * Re-uploads the WalkaroundUBO at the next frame start (the existing UBO
   * uploader reads live engine fields each frame — no frozen snapshot to
   * bust). Invalidates the DDGI probe atlas so it re-converges over the
   * next ~8 frames. Resets the temporal accumulator (history discarded;
   * α=1 for the very next frame).
   *
   * **No pipelines or GPU buffers are recreated.** The only cost is:
   *   - 2 JS field writes per changed field (field + DDGI sun-intensity mirror)
   *   - DDGI re-convergence over the default `STRIDE` frames (~133 ms at
   *     60 FPS)
   *   - 1 frame of temporal-accumulator reset overhead
   *
   * **Use case:** time-of-day scrubbing in stainedGlass without engine
   * teardown. Eliminates the engine-recreation workaround documented at
   * `useVitrumWalkaroundEngine.ts:34` (stainedGlass audit Gap 1).
   *
   * Calling with an empty object (`{}`) is a safe no-op.
   *
   * @param opts - Partial lighting overrides. Omitted fields are unchanged.
   */
  updateLighting(opts: Partial<LightingOptions>): void {
    let changed = false;

    if (opts.primaryLightDir !== undefined) {
      this._primaryLightDir = opts.primaryLightDir;
      changed = true;
      // Mirror into DDGI's sun-intensity multiplier path on the ProbeUpdatePass.
      // The pass uses the sun direction implicitly via the light list; updating
      // the field here ensures renderFrame() passes the new value to the UBO.
    }
    if (opts.primaryLightIntensity !== undefined) {
      this._primaryLightIntensity = opts.primaryLightIntensity;
      changed = true;
      // Keep the DDGI ProbeUpdatePass sun-intensity multiplier in sync so the
      // irradiance atlas re-converges at the correct brightness.
      this._ddgi.pass.setSunIntensityMultiplier(opts.primaryLightIntensity);
    }
    if (opts.skyTint !== undefined) {
      this._skyTint = opts.skyTint;
      changed = true;
    }
    if (opts.skyIrradiance !== undefined) {
      this._skyIrradiance = opts.skyIrradiance;
      changed = true;
    }

    if (!changed) return;

    // Invalidate the DDGI probe atlas — re-converges from scratch over the
    // next STRIDE frames (~8 frames, ~133 ms at 60 FPS).
    this._ddgi.invalidateProbeCache();

    // Reset the temporal accumulator — history discarded, α=1 for next frame.
    // _pipeline may be null if the engine is still initialising; the flag is
    // applied as soon as the pipeline exists (set before any renderFrame call).
    this._pipeline?.requestAccumReset();
  }

  // ── Frame rendering ────────────────────────────────────────────────────

  /**
   * Render one walkaround frame. Drives:
   *   1. DDGI per-frame compute (fire-and-forget) — GPU command queueing is
   *      synchronous from JS's perspective; the actual GPU work runs after the
   *      JS tick returns. The atlas DDGI writes is double-buffered: this frame
   *      reads from the previous tick's write target while next tick will read
   *      from this one.
   *   2. DDGI atlas wire into the ReSTIR shade pass.
   *   3. ReSTIR pipeline.renderFrame().
   *
   * The host calls `engine.renderFrame(input)` and receives a complete frame.
   * The host does NOT separately call `ddgi.updateFrame()`.
   *
   * Returns a FrameOutput immediately — the pipeline writes directly into the
   * swap chain texture provided via `input.swapChainView`.
   *
   * The 60 FPS internal throttle is enforced here: on high-refresh-rate
   * displays, frames arriving faster than ~16.67 ms apart are skipped and
   * this returns a "skip" FrameOutput (samplesAccumulated: 0, isConverged:
   * false, primaryRadiance: null).
   */
  renderFrame(input: FrameInput): FrameOutput {
    const skipOutput: FrameOutput = {
      primaryRadiance: null,
      samplesAccumulated: 0,
      isConverged: false,
    };

    if (this._state === 'paused' || this._state === 'disposed' || this._state === 'error') {
      return skipOutput;
    }

    const fp = HybridEngine._fingerprintRebuildKey(
      this._getPipelineRebuildKey?.() ?? this._staticPipelineRebuildKey,
    );
    if (fp !== this._rebuildKeyFingerprintSeen) {
      this._rebuildKeyFingerprintSeen = fp;
      this.reset();
      return skipOutput;
    }

    const dbg = this._debug ? this._dbg : null;

    const pipeline = this._pipeline;
    const bvh = this._bvhBuffers;
    if (!pipeline) { if (dbg) dbg.skipNoPipeline++; return skipOutput; }
    if (!bvh)      { if (dbg) dbg.skipNoBvh++;      return skipOutput; }

    const now = performance.now();
    // Audit M4: configurable FPS cap. `null` disables the throttle so VR /
    // 90+ Hz displays get every frame. Default preserves Cornell's 60-FPS
    // soft-cap.
    if (this._targetFrameIntervalMs !== null &&
        this._lastFrameTs !== 0 &&
        now - this._lastFrameTs < this._targetFrameIntervalMs) {
      if (dbg) dbg.skipFrameInterval++;
      // CRITICAL on >60Hz displays: even though the heavy pipeline is
      // throttled to 60 FPS, the host's rAF still acquires a fresh swap-
      // chain texture each tick. Without writing to it, the texture is
      // presented as cleared black → visible dark-flash on every other
      // frame at 120Hz. Blit the most recent resolvedTexture so the
      // canvas keeps showing the previous frame.
      const skipSwapView = input.swapChainView as GPUTextureView | undefined;
      if (skipSwapView && this._pipeline) {
        this._pipeline.presentLastFrame(skipSwapView);
      }
      return skipOutput;
    }
    this._lastFrameTs = now;

    const t0 = now;

    // Core's FrameInput types swap-chain fields opaquely (BackendTexture /
    // BackendTextureFormat). The walkaround backend requires WebGPU; cast at
    // this boundary so the rest of HybridEngine works with concrete types.
    const swapView   = input.swapChainView as GPUTextureView | undefined;
    const swapFmt    = (input.swapChainFormat as GPUTextureFormat | undefined) ?? getPreferredSwapChainFormat();

    if (!swapView) {
      if (dbg) dbg.skipNoSwapView++;
      return skipOutput;
    }

    // Periodic 5s rate report.
    if (dbg) {
      dbg.framesDispatched++;
      if (dbg.lastReportTs === 0) dbg.lastReportTs = now;
      if (now - dbg.lastReportTs > 5_000) {
        const elapsed = (now - dbg.lastReportTs) / 1_000;
        const gpu = (pipeline as unknown as { lastGpuTimings?: Record<string, number> })
          .lastGpuTimings ?? {};
        const gpuTotal = gpu['total'];
        console.log('[hybrid:debug] rate (5s window)', {
          framesDispatched: dbg.framesDispatched,
          fps: (dbg.framesDispatched / elapsed).toFixed(2),
          skipReasons: {
            noPipeline: dbg.skipNoPipeline, noBvh: dbg.skipNoBvh,
            noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
          },
          gpuTotalMs: gpuTotal !== undefined ? +gpuTotal.toFixed(2) : 'n/a',
          gpuPerPassMs: gpu,
        });
        dbg.framesDispatched = 0;
        dbg.skipNoPipeline = 0;
        dbg.skipNoBvh = 0;
        dbg.skipNoSwapView = 0;
        dbg.skipFrameInterval = 0;
        dbg.lastReportTs = now;
      }
    }

    // ── DDGI per-frame compute ──────────────────────────────────────────
    // Drive DDGI probe updates as part of this frame tick (fire-and-forget).
    // GPU command queueing (writeBuffer / dispatchWorkgroups / queue.submit) is
    // synchronous from JS — the GPU executes the work asynchronously after the
    // JS tick returns. The atlas is double-buffered, so this frame reads the
    // previous tick's write target while the GPU processes this tick's update.
    // The host MUST NOT call ddgi.updateFrame() separately.
    const ddgiLayerOn = this._ddgiOn && (this._layerEnabled.get('ddgi') ?? true);
    if (ddgiLayerOn) {
      // DDGIFrameInputs now accepts a DDGIDeviceHandle (`device` or a
      // Three.js renderer adapter). HybridEngine owns the device directly.
      void this._ddgi.updateFrame({
        scene:   this._ddgiTraversalScene ?? this._threeScene,
        device:  this._device,
        enabled: true,
      });
    }

    // ── DDGI atlas wire ─────────────────────────────────────────────────
    if (!ddgiLayerOn) {
      pipeline.setDDGIInputs(null);
    } else if (this._ddgi.ready) {
      const atlas = this._ddgi.pass.getReadAtlasGPUTextures?.();
      if (atlas) {
        const gridParams = packDDGIGridParams(this._ddgi.probeGrid.params);
        pipeline.setDDGIInputs({
          irradianceTex: atlas.irradiance,
          visibilityTex: atlas.visibility,
          gridParams,
        });
      }
    }

    // ── ReSTIR pipeline renderFrame ─────────────────────────────────────
    const W = this._width;
    const H = this._height;

    const viewMatrix  = input.viewMatrix  as Float32Array;
    const projMatrix  = input.projMatrix  as Float32Array;
    const prevView    = (input.prevViewMatrix ?? input.viewMatrix) as Float32Array;
    const prevProj    = (input.prevProjMatrix ?? input.projMatrix) as Float32Array;
    const camPos      = input.cameraPosition as [number, number, number];

    // `FrameInput.quality.bounces` is ignored: ReSTIR + shade WGSL use a fixed
    // path depth baked at shader compile time (see `capabilities.maxBounces`).

    pipeline.renderFrame({
      viewMatrix:            new Float32Array(viewMatrix),
      projMatrix:            new Float32Array(projMatrix),
      prevViewMatrix:        new Float32Array(prevView),
      prevProjMatrix:        new Float32Array(prevProj),
      cameraPos:             camPos,
      screenWidth:           W,
      screenHeight:          H,
      frameSeed:             input.frameSeed,
      totalEmissivePower:    bvh.totalEmissivePower ?? 1.0,
      emitterCount:          bvh.emitters?.count ?? 0,
      primaryLightDir:       this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      skyTint:               this._skyTint,
      skyIrradiance:         this._skyIrradiance,
      // Library-generality tunables (audit follow-up). Defaults preserve
      // Cornell behaviour; hosts override via HybridEngineOptions.
      emitterDist2Floor:     this._emitterDist2Floor,
      directFireflyClamp:    this._directFireflyClamp,
      causticBoost:          this._causticBoost,
      causticVisClamp:       this._causticVisClamp,
      temporalMClampDI:      this._temporalMClampDI,
      spatialReuseRadiusPx:  this._spatialReuseRadiusPx,
      spatialDepthTolFloor:  this._spatialDepthTolFloor,
      gtaoRadiusPx:          this._gtaoRadiusPx,
      gtaoIntensity:         this._gtaoIntensity,
      gtaoDepthThreshold:    this._gtaoDepthThreshold,
      gtaoBilateralDepthSigma: this._gtaoBilateralDepthSigma,
      adaptiveSamplingThresholdLow:  this._adaptiveSamplingThresholdLow,
      adaptiveSamplingThresholdHigh: this._adaptiveSamplingThresholdHigh,
      triIntersectEpsilon:   this._triIntersectEpsilon,
      swapChainView:         swapView,
      swapChainFormat:       swapFmt,
    });

    const dt = performance.now() - t0;

    // Record the frame timing on the engine itself; hosts that want to surface
    // it in dev UI can poll `engine.debugTimings`. The legacy mirror into
    // window.__WGPU__.walkaround.frameTimings is preserved while
    // `_staging/legacy-source` host code reads from there — it will be dropped
    // when the host extraction lands.
    if (this._debug) {
      this._debugTimings.push({ t: now, ms: dt });
      if (this._debugTimings.length > 240) this._debugTimings.shift();

      if (typeof window !== 'undefined') {
        const w = window as unknown as { __WGPU__?: { walkaround?: { frameTimings: unknown } } };
        if (w.__WGPU__?.walkaround) {
          const ft = w.__WGPU__.walkaround.frameTimings as Array<{ t: number; ms: number }>;
          if (Array.isArray(ft)) {
            ft.push({ t: now, ms: dt });
            if (ft.length > 240) ft.shift();
          }
        }
      }
    }

    return {
      primaryRadiance:    swapView,   // swap chain is the output surface
      samplesAccumulated: 1,
      isConverged:        false,      // walkaround never converges; resamples every frame
    };
  }

  // ── Reset ──────────────────────────────────────────────────────────────

  /**
   * Tear down the pipeline and reinitialise from scratch.
   * Hosts call this when the scene changes significantly.
   */
  reset(): void {
    this._teardownPipeline();
    void this._initPipeline();
  }

  // ── Pause / resume ─────────────────────────────────────────────────────

  /**
   * Pause per-frame compute. Engine state transitions from `'ready'` →
   * `'paused'`. Calls in any other live state (e.g. `'initializing'`,
   * `'uninitialized'`) are no-ops; calls after `'disposed'` throw.
   *
   * Aligns with `PTEngineWebGL2.pause()`: both throw on disposed, both
   * no-op when the state transition doesn't apply.
   */
  pause(): void {
    if (this._state === 'disposed' || this._state === 'error') {
      throw new Error('pause: engine is disposed or in error state');
    }
    if (this._state === 'ready') {
      this._state = 'paused';
    }
    // no-op in 'uninitialized' | 'initializing' | 'paused'
  }

  /**
   * Resume per-frame compute. Engine state transitions from `'paused'` →
   * `'ready'`. Calls in any other live state are no-ops; calls after
   * `'disposed'` throw.
   *
   * Aligns with `PTEngineWebGL2.resume()`: both throw on disposed, both
   * no-op when the state transition doesn't apply.
   */
  resume(): void {
    if (this._state === 'disposed' || this._state === 'error') {
      throw new Error('resume: engine is disposed or in error state');
    }
    if (this._state === 'paused') {
      this._state = 'ready';
    }
    // no-op in 'uninitialized' | 'initializing' | 'ready'
  }

  // ── Layer toggles (host-accessible debug interface) ────────────────────

  /**
   * Enable or disable a named render layer. Currently recognised layers:
   *   - 'ddgi' — DDGI probe atlas wiring into the shade pass.
   *
   * Replaces `window.__HYBRID_LAYERS__` from the original host bridge.
   * The host sets up `window.__HYBRID_LAYERS__` → `engine.setLayerEnabled()`
   * forwarding if it wants console-accessible toggles.
   */
  setLayerEnabled(layer: string, enabled: boolean): void {
    this._layerEnabled.set(layer, enabled);
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  dispose(): void {
    this._disposed = true;
    this._teardownPipeline();
    this._ddgi.dispose();
    this._state = 'disposed';

    if (this._debug && typeof window !== 'undefined') {
      const dbg = this._dbg;
      dbg.disposeCount++;
      const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
      console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
        ranForMs: liveMs.toFixed(1),
        framesDispatched: dbg.framesDispatched,
        skipReasons: {
          noPipeline: dbg.skipNoPipeline, noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
        },
      });
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────

  /** True when `_lastScene` supplies at least one triangle mesh primitive. */
  private _coreSceneSuppliesMeshes(): boolean {
    const s = this._lastScene;
    return s != null && s.primitives.some((p) => p.kind === 'mesh');
  }

  /**
   * Scene-readiness for BVH build: core mesh payload **or** ctor `isSceneReady`
   * heuristic on the host `threeScene` (proxy-heavy hosts may rely on the latter).
   */
  private _sceneReadyForBvh(): boolean {
    if (this._coreSceneSuppliesMeshes()) return true;
    return this._isSceneReady();
  }

  private _teardownPipeline(): void {
    if (this._pipeline) {
      this._pipeline.dispose();
      this._pipeline = null;
    }
    if (this._bvhBuffers) {
      disposeSceneBVH(this._bvhBuffers);
      this._bvhBuffers = null;
    }
    if (this._ddgiTraversalScene) {
      disposeVitrumThreeSceneRoot(this._ddgiTraversalScene);
      this._ddgiTraversalScene = null;
    }
    if (this._state !== 'disposed') {
      this._state = 'initializing';
    }
  }

  private async _initPipeline(): Promise<void> {
    if (this._disposed) return;

    const device = this._device;

    if (this._debug) {
      const dbg = this._dbg;
      dbg.initCount++;
      dbg.initStart = performance.now();
      console.log(`[hybrid:debug] init #${dbg.initCount} START`, {
        W: this._width, H: this._height, device: !!device,
        t: dbg.initStart.toFixed(0),
      });
    }

    this._state = 'initializing';

    // Fire-and-forget: the engine transitions to 'ready' when the BVH and
    // pipeline finish setting up, or to 'error' on failure. Callers do not
    // await this — the engine state machine is the synchronization point.
    void (async () => {
      // Poll until scene has enough geometry (or 5s timeout).
      const pollStart = Date.now();
      let pollIters = 0;
      while (!this._disposed) {
        const elapsed = Date.now() - pollStart;
        if (elapsed >= 5_000) break;
        if (this._sceneReadyForBvh()) break;
        await new Promise<void>((r) => setTimeout(r, 50));
        pollIters++;
      }

      if (this._disposed) {
        if (this._debug) {
          console.log('[hybrid:debug] init aborted during scene-readiness poll', { pollIters });
        }
        return;
      }

      if (this._debug) {
        console.log('[hybrid:debug] scene-ready', { pollIters, elapsed: Date.now() - pollStart });
      }

      try {
        const bvhStart = performance.now();
        const bvhRoot: THREE.Object3D = this._coreSceneSuppliesMeshes()
          ? vitrumSceneToThree(this._lastScene!)
          : this._threeScene;
        if (bvhRoot !== this._threeScene) {
          this._ddgiTraversalScene = bvhRoot as THREE.Scene;
        }
        const bvh = buildReSTIRSceneBVH([bvhRoot], {
          primaryLightDir:       new THREE.Vector3(...this._primaryLightDir),
          primaryLightIntensity: this._primaryLightIntensity,
        });
        const bvhMs = performance.now() - bvhStart;
        this._bvhBuffers = bvh;

        if (this._debug) {
          console.log('[hybrid:debug] BVH built', {
            bvhMs: bvhMs.toFixed(1),
            triCount: bvh.bvhNodes?.count,
            emitterCount: bvh.emitters?.count,
          });
        }

        const pipeline = new WalkaroundGPUPipeline(device, this._width, this._height);
        const pipelineStart = performance.now();

        // T2.H2 — Neural denoiser: create and initialize InferenceGraph before pipeline init.
        let inferenceGraph: InferenceGraph | undefined;
        if (this._denoiser === 'neural' && this._neuralWeights) {
          const { buildUNetSpec } = await import('./neural/unetArchitecture.js');
          inferenceGraph = new InferenceGraph(buildUNetSpec());
          await inferenceGraph.initialize(device, this._neuralWeights, this._width, this._height);
        }

        await pipeline.initialize(
          bvh,
          getPreferredSwapChainFormat(),
          {
            verbose: this._verbose || this._debug,
            denoiser: this._denoiser,
            cameraMoveResetThresholdSq: this._cameraMoveResetThresholdSq,
            temporalAccumAlpha: this._temporalAccumAlpha,
            // exactOptionalPropertyTypes: omit the key entirely when undefined.
            ...(inferenceGraph !== undefined ? { inferenceGraph } : {}),
          },
        );
        const pipelineMs = performance.now() - pipelineStart;

        if (this._disposed) {
          if (this._debug) {
            console.log('[hybrid:debug] init aborted post-pipeline.initialize', {
              pipelineMs: pipelineMs.toFixed(1),
            });
          }
          pipeline.dispose();
          return;
        }

        // Wire the sun intensity multiplier into DDGI so its Le bake
        // matches shade.wgsl's Lo_emit. `setSunIntensityMultiplier` is a
        // public method on ProbeUpdatePass — no cast needed.
        this._ddgi.pass.setSunIntensityMultiplier(this._primaryLightIntensity);

        // Auto-collect THREE.RectAreaLight from the scene as DDGI point
        // lights (centroid + flux-equivalent intensity). DDGI's per-probe
        // ray-cast pass uses only 'sun' + 'fixture'/'teaLight' kinds —
        // without this bridge, rect-area lights from `vitrumSceneToThree`
        // never reach DDGI's `evalDirectLighting`, so probe rays hitting
        // walls return zero radiance and the irradiance atlas stays
        // black → no color bleed onto boxes, surfaces render flat-gray
        // even with DDGI mechanically running.
        //
        // For a 1×1 rect at intensity 12 (Le=(12,12,12) per channel),
        // the per-area-element flux is Le × dA. A point at the rect
        // centroid carrying flux ≈ Le × area integrates roughly the same
        // total downward power; the DDGI atlas captures the qualitative
        // colour bleed correctly, which is what the indirect bounce
        // depends on. ReSTIR DI still drives the high-frequency direct
        // term from the actual rect geometry — DDGI here only feeds the
        // low-frequency indirect.
        const ddgiRectLights = collectDDGILightsFromRectAreaLights(bvhRoot);
        if (ddgiRectLights.length > 0) {
          this._ddgi.setLights([...this._ctorLights, ...ddgiRectLights]);
        }

        this._pipeline     = pipeline;
        this._state        = 'ready';

        if (this._debug) {
          const dbg = this._dbg;
          const totalMs = performance.now() - dbg.initStart;
          console.log(`[hybrid:debug] init #${dbg.initCount} COMPLETE`, {
            pipelineMs: pipelineMs.toFixed(1), totalMs: totalMs.toFixed(1),
          });
        }
      } catch (err) {
        if (!this._disposed) {
          this._state = 'error';
        }
        console.error(
          '[HybridEngine] init failed — engine state set to error. Call dispose() and recreate the engine to retry.',
          err,
        );
      }
    })();
  }

  // ── Private static helpers ─────────────────────────────────────────────

  private static _fingerprintRebuildKey(key: string | number | null | undefined): string {
    if (key === null || key === undefined) return '__null';
    if (typeof key === 'number') return Number.isNaN(key) ? '__n:NaN' : `__n:${key}`;
    return `__s:${key}`;
  }
}

/**
 * Walk an Object3D tree for `THREE.RectAreaLight` instances and project each
 * onto a `DDGILight` point-light approximation so the DDGI probe-update pass
 * (which only switches on `kind === 'sun' | 'fixture' | 'teaLight'`) can
 * evaluate direct lighting at probe-ray hit points.
 *
 * Approximation rationale: DDGI provides low-frequency indirect bounce — the
 * actual rect geometry only matters for the high-frequency direct term, which
 * ReSTIR DI handles separately from the actual emitter triangles. A point at
 * the rect centroid carrying flux ≈ `color × intensity × area` gives a
 * qualitatively-correct downward irradiance for probes; colour bleed onto
 * surrounding walls (the visible signature of Cornell-style scenes) reaches
 * the irradiance atlas correctly. The remaining factor-of-π errors in
 * total-flux conversion are negligible against the multiple-of-10 dynamic
 * range that distinguishes "lit colour bleed" from "atlas reads zero".
 */
function collectDDGILightsFromRectAreaLights(root: THREE.Object3D): DDGILight[] {
  const out: DDGILight[] = [];
  const _wp = new THREE.Vector3();
  root.updateMatrixWorld(true);
  root.traverseVisible((obj) => {
    if (!(obj instanceof THREE.RectAreaLight)) return;
    const light = obj;
    const area = light.width * light.height;
    _wp.setFromMatrixPosition(light.matrixWorld);
    out.push({
      kind: 'fixture',
      intensity: light.intensity * area,
      on: true,
      position: { x: _wp.x, y: _wp.y, z: _wp.z },
    });
  });
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────────────

/**
 * Create a HybridEngine instance and begin asynchronous pipeline initialisation.
 *
 * The engine is returned immediately in `'initializing'` state. The host
 * should poll `engine.state` or listen for the `'ready'` transition before
 * calling `renderFrame`.
 *
 * @param opts  Creation-time options. `opts.device` must be a live GPUDevice.
 */
export const createWalkaroundEngine_Hybrid: EngineFactory<HybridEngineOptions> = async (
  opts: HybridEngineOptions,
): Promise<Engine> => {
  // Duck-type GPUDevice validation — `instanceof GPUDevice` is not reliable
  // across realms; checking for a known required method is more robust.
  if (
    !opts.device ||
    typeof (opts.device as { createCommandEncoder?: unknown }).createCommandEncoder !== 'function'
  ) {
    throw new TypeError(
      '[createWalkaroundEngine_Hybrid] opts.device must be a live GPUDevice. ' +
      'Received: ' + String(opts.device),
    );
  }

  const engine = new HybridEngine(opts);
  // _initPipeline is fire-and-forget; the engine transitions to 'ready'
  // when the BVH poll + pipeline compile finishes. The host polls
  // engine.state or observes renderFrame returning samplesAccumulated=0.
  // Bootstrap with a valid empty scene (no primitives, no emitters) so the
  // scene-readiness poll falls through to the host threeScene heuristic.
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
}
