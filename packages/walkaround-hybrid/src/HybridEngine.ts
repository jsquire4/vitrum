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
  EngineDebugSurface,
  EngineFactory,
  EngineOptions,
  EngineState,
  FrameStats,
} from '@vitrum/core';
import type { Scene, ScenePrimitive } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { refitBvhBounds } from '@vitrum/shared-bvh';
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

  /**
   * Optional escape hatch for hosts that need to provide a THREE.Scene as the
   * BVH / DDGI source directly (e.g. when the host's authoritative scene graph
   * is THREE-only and they intentionally omit `setScene(vitrumScene)`).
   *
   * **Most callers leave this undefined.** When `setScene` provides a vitrum
   * Scene with at least one mesh primitive, the engine derives the BVH source
   * via `vitrumSceneToThree()` and the `threeScene` field is never read. The
   * @vitrum/engine `createEngine()` facade always takes the latter path.
   *
   * Was required pre-T3.H (deprecated 2026-05-12, removed 2026-05-12). Hosts
   * that previously passed `threeScene: someScene` can drop the field if they
   * also call `setScene(sceneFromThreeJS(someScene))` afterwards. If they do
   * neither (no mesh primitives in setScene + no threeScene), the engine
   * throws on pipeline init with a clear error.
   */
  readonly threeScene?: THREE.Scene;

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
  // Mutable since T-resize: the host calls `setSize()` whenever the
  // canvas resizes; the pipeline reallocates its FrameResources without
  // a full engine teardown. See `setSize()` for the resize contract.
  private _width:                number;
  private _height:               number;
  /** Optional escape-hatch THREE.Scene from ctor opts. Null when the host
   *  goes through the canonical setScene(vitrumScene) path (T3.H removal). */
  private readonly _threeScene:           THREE.Scene | null;
  /** Lazily-synthesized THREE.Scene root from the most recent vitrum
   *  setScene() — caches `vitrumSceneToThree(_lastScene)` so DDGI updateFrame
   *  doesn't re-traverse on every frame. Reset on every setScene(). */
  private _synthesizedThreeScene:         THREE.Scene | null = null;
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

  /** T3.E — telemetry subscribers fired at end of each successful renderFrame. */
  private readonly _frameSubs: Array<(s: FrameStats) => void> = [];

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

  /**
   * Monotonic init sequence — incremented at the start of every
   * `_initPipeline()` call. Each in-flight async init captures the value
   * at entry; before it writes any shared state (`_ddgiTraversalScene`,
   * `_bvhBuffers`, `_pipeline`) it re-checks `mySeq === this._initSeq`.
   * If the value drifted, a newer init / teardown raced ahead and this
   * older chain MUST dispose its locals + bail without mutating shared
   * state. Without this guard, two concurrent inits both write — the loser
   * leaks ~1 GB of GPU resources (full-res rgba16float textures + BVH
   * buffers + DDGI atlases) per setScene/resize storm tick.
   */
  private _initSeq: number = 0;

  /**
   * Set true by `dispose()` when there's an in-flight `_initPipeline()`
   * chain. The init chain checks this flag after every `await` and, if
   * set, disposes any locals it owns (BVH, pipeline) AND finalises
   * teardown of `_pipeline` / `_bvhBuffers` / `_ddgiTraversalScene` if
   * something snuck into them between the dispose call and the await
   * resolution. Lets `dispose()` stay synchronous while still being honest
   * about late writers.
   */
  private _pendingTeardown: boolean = false;

  /**
   * True while an `_initPipeline()` async chain is mid-flight. The init
   * chain sets this to `true` at entry to the inner IIFE and to `false`
   * in its `finally`. Read by `dispose()` to decide whether to defer
   * teardown to the in-flight chain's finally block (via
   * `_pendingTeardown`) or to tear down synchronously here and now.
   */
  private _initRunning: boolean = false;

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _staticPipelineRebuildKey: string | number | null;
  private readonly _getPipelineRebuildKey: (() => string | number | null | undefined) | undefined;

  constructor(opts: HybridEngineOptions) {
    this._device                = opts.device;
    this._width                 = opts.width;
    this._height                = opts.height;
    this._threeScene            = opts.threeScene ?? null;
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
    // Default predicate: ready when EITHER the vitrum Scene supplies any mesh
    // primitive OR the optional escape-hatch THREE.Scene contains triangles.
    // Hosts override via opts.isSceneReady when they need a scene-specific
    // signal (e.g. wait for an async asset).
    this._isSceneReady          = opts.isSceneReady ?? (() => {
      if (this._coreSceneSuppliesMeshes()) return true;
      return this._threeScene != null && defaultIsSceneReady(this._threeScene);
    });

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
      // Set true once any updatePrimitive path ships. This branch
      // (`feat/a3-geometry-change-bvh-leaf-rebuild`) ships the transform
      // refit + topology rebuild paths; the sibling branch ships the
      // material-only fast path. Either alone is enough to advertise
      // incremental-scene support: hosts can call `updatePrimitive` and
      // get either the fast path (this branch's transform refit / sibling's
      // material bytes write) or the safe fall-through (full BVH rebuild).
      supportsIncrementalScene:  true,
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
    // T3.H removal: drop the cached synthesized THREE.Scene; the next BVH
    // build / DDGI updateFrame will re-derive it from the new vitrum Scene.
    if (this._synthesizedThreeScene != null) {
      try { disposeVitrumThreeSceneRoot(this._synthesizedThreeScene); } catch {}
      this._synthesizedThreeScene = null;
    }

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    void this._initPipeline();
  }

  /** T3.H removal: lazily synthesize a THREE.Scene from the most recent
   *  vitrum Scene if (a) the host did not pass `threeScene` at construction,
   *  and (b) the vitrum Scene supplies meshes. Caller is responsible for
   *  null-checking — if both threeScene and synthesizable are null, the
   *  pipeline will throw at BVH build with a clear message. */
  private _ensureThreeSceneRoot(): THREE.Scene | null {
    if (this._threeScene != null) return this._threeScene;
    if (this._synthesizedThreeScene != null) return this._synthesizedThreeScene;
    if (this._lastScene != null && this._coreSceneSuppliesMeshes()) {
      this._synthesizedThreeScene = vitrumSceneToThree(this._lastScene) as THREE.Scene;
      return this._synthesizedThreeScene;
    }
    return null;
  }

  // ── updatePrimitive — geometry-change path ─────────────────────────────
  //
  // **Scope of this branch (feat/a3-geometry-change-bvh-leaf-rebuild):**
  // implements the **transform-only fast path** + **topology-change full
  // rebuild path** of `Engine.updatePrimitive`. The **material-only fast
  // path** ships separately on `feat/a3-hybridengine-incremental-updates`
  // (commit `d0d22b0`); the merger combines both code paths into a single
  // dispatcher.
  //
  // **Routing rules (this branch alone)**:
  //  - `patch.transform` present AND no topology fields → fast-path (c):
  //     refit the BVH bounds in-place (no SAH rebuild, no pipeline
  //     recompile, no DDGI atlas invalidation), rewrite the affected
  //     primitive's vertex slice in `bvhPositions`, reset the accumulator.
  //  - any topology field present (`positions` / `normals` / `uvs` /
  //    `tangents` / `indices` / `instances` / `params` / `shape` /
  //    `fallbackMesh` / `kind`) → full-rebuild path (a): re-run
  //    `buildReSTIRSceneBVH`, destroy + reupload all four BVH GPU
  //    buffers, reset the accumulator.
  //  - material-only patches → throw with a clear pointer (the merger's
  //    job is to replace this branch with the material-fast-path branch's
  //    dispatch logic).
  //
  // Implements `Engine.updatePrimitive(id, patch)` from `@vitrum/core`.
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no scene set. ` +
        `Call setScene(scene) before updatePrimitive.`,
      );
    }
    const prim = this._lastScene.primitives.find((p) => String(p.id) === id);
    if (!prim) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive id not found in current scene.`,
      );
    }

    const topologyFields = [
      'positions', 'normals', 'uvs', 'tangents', 'indices',
      'instances', 'params', 'shape', 'fallbackMesh', 'kind',
    ] as const;
    const hasTopologyChange = topologyFields.some(
      (f) => (patch as Record<string, unknown>)[f] !== undefined,
    );
    const hasTransformChange = (patch as Record<string, unknown>)['transform'] !== undefined;
    const hasMaterialChange  = (patch as Record<string, unknown>)['material']  !== undefined;

    if (hasTopologyChange) {
      this._updatePrimitiveTopologyRebuild(id, patch);
      return;
    }
    if (hasTransformChange) {
      this._updatePrimitiveTransformRefit(id, patch);
      return;
    }
    if (hasMaterialChange) {
      // Reserved for the material-only fast-path branch
      // (`feat/a3-hybridengine-incremental-updates`). Until that branch
      // is merged, send a clear pointer rather than silently no-op.
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): material-only fast path lives on ` +
        `the sibling branch \`feat/a3-hybridengine-incremental-updates\` (commit d0d22b0). ` +
        `This branch (\`feat/a3-geometry-change-bvh-leaf-rebuild\`) handles geometry / ` +
        `transform patches only. The merger combines both code paths.`,
      );
    }

    // No recognised patch field — treat as a no-op rather than throw so
    // hosts can pass through optional patches without checking each
    // field's presence.
  }

  /**
   * Transform-only fast path (Option (c) per items_to_fix.md A3).
   *
   * The BVH topology is preserved — only AABB bounds are refit. Cost is
   * O(affectedVertices + totalBvhNodes), no pipeline recompile, no DDGI
   * atlas invalidation. For a single primitive on a 30k-tri scene this
   * runs in well under 1 ms vs. ~50 ms for a full SAH rebuild + pipeline
   * recompile.
   *
   * Steps:
   *  1. Look up the affected mesh by `name === id` in the synthesized
   *     THREE scene (or `_threeScene` for the host-Three-scene path).
   *  2. Apply the new transform to the THREE.Mesh (`matrix` + `matrixWorld`).
   *  3. Compute the matrix delta `D = matrixWorldNew · matrixWorldAtBuild⁻¹`.
   *  4. For each vertex `v` in `[vertexStart, vertexStart + vertexCount)`,
   *     read the old world-space position from `bvhPositions.cpuData`,
   *     apply `D`, write the new world-space position back. (UV in `.w`
   *     is preserved.)
   *  5. Update `matrixWorldAtBuild` snapshot to the new matrix world.
   *  6. Run `refitBvhBounds` on the BVH node buffer.
   *  7. Upload the refit nodes + position slice via the pipeline.
   *  8. Reset the accumulator (history is invalid — the primitive moved).
   */
  private _updatePrimitiveTransformRefit(id: string, patch: Partial<ScenePrimitive>): void {
    const bvh = this._bvhBuffers;
    if (bvh == null) {
      // Pipeline still initialising — nothing to refit. Fall through to a
      // full rebuild so the next setScene picks up the new transform.
      this._updatePrimitiveTopologyRebuild(id, patch);
      return;
    }

    const range = bvh.meshVertexRanges.find((r) => r.name === id);
    if (range == null || range.vertexCount === 0) {
      // No vertices for this primitive in the merged buffer (e.g. an
      // emitter-only primitive, or a name mismatch). Fall back to a
      // topology rebuild so the user's intent isn't silently dropped.
      this._updatePrimitiveTopologyRebuild(id, patch);
      return;
    }

    const root = this._ensureThreeSceneRoot();
    if (root == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no THREE scene available for refit.`,
      );
    }
    let mesh: THREE.Mesh | null = null;
    root.traverseVisible((obj) => {
      if (mesh == null && obj.name === id && (obj as THREE.Mesh).isMesh) {
        mesh = obj as THREE.Mesh;
      }
    });
    if (mesh == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
      );
    }
    const meshRef = mesh as THREE.Mesh;

    // Apply the new transform. The Scene contract says transform is a
    // 16-element column-major Mat4 (see core/src/scene.ts:MeshPrimitive).
    const newMat = new THREE.Matrix4();
    const transform = (patch as { transform?: ArrayLike<number> }).transform;
    if (transform && transform.length >= 16) {
      newMat.fromArray(Array.from(transform));
    } else {
      newMat.identity();
    }
    meshRef.matrix.copy(newMat);
    meshRef.matrixWorld.copy(newMat);
    meshRef.matrixAutoUpdate = false;

    // Compute matrix delta D = newMat · oldMat⁻¹. We transform each
    // already-baked world-space vertex through D to get the new
    // world-space vertex; equivalent to local⁻¹ → new-world round-trip
    // but without storing local-space positions.
    const oldMatWorld = new THREE.Matrix4().fromArray(Array.from(range.matrixWorldAtBuild));
    const oldMatWorldInv = new THREE.Matrix4().copy(oldMatWorld).invert();
    const delta = new THREE.Matrix4().multiplyMatrices(newMat, oldMatWorldInv);

    // Rewrite the affected vertex slice of bvhPositions.cpuData. The
    // stride-4 layout packs world-space xyz into [0..2] and UV-as-u32
    // into [3] (preserved here). Use a single typed-array view over the
    // shared ArrayBuffer so the changes land in cpuData.
    const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
    const STRIDE = 4;
    const baseVertex = range.vertexStart;
    const sliceVerts = range.vertexCount;
    const tmp = new THREE.Vector3();
    for (let v = 0; v < sliceVerts; v++) {
      const off = (baseVertex + v) * STRIDE;
      tmp.x = positionsF32[off + 0]!;
      tmp.y = positionsF32[off + 1]!;
      tmp.z = positionsF32[off + 2]!;
      tmp.applyMatrix4(delta);
      positionsF32[off + 0] = tmp.x;
      positionsF32[off + 1] = tmp.y;
      positionsF32[off + 2] = tmp.z;
      // .w (UV pack) preserved.
    }

    // Update the matrix snapshot in-place so subsequent transform
    // patches compute their delta against the latest matrix, not the
    // original build-time matrix.
    range.matrixWorldAtBuild.set(newMat.elements);

    // Refit BVH bounds in place against the freshly-updated positions.
    // Use the cached stride-3 index buffer (refit reads 3 u32 per
    // triangle, no padding).
    const bvhNodesF32 = new Float32Array(bvh.bvhNodes.cpuData);
    refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);

    // Upload the refit nodes + the affected position slice to GPU.
    // bvhNodes is small (~32 KB / 1k tris) — upload whole.
    // Positions: write only the affected byte range to honour the
    // "fast path" goal.
    const positionsByteOffset = baseVertex * STRIDE * 4; // f32 = 4 bytes
    const positionsByteLength = sliceVerts * STRIDE * 4;
    const positionsSlice = bvh.bvhPositions.cpuData.slice(
      positionsByteOffset,
      positionsByteOffset + positionsByteLength,
    );
    this._pipeline?.refreshBvhRefit(
      bvh.bvhNodes.cpuData.slice(0),
      { byteOffset: positionsByteOffset, data: positionsSlice },
    );

    // Reset the accumulator — temporal history is invalid because the
    // primitive moved (history pixels reference the old world position).
    this._pipeline?.requestAccumReset();
    // DDGI probes baked their irradiance against the old position;
    // invalidate so probes re-converge over the next STRIDE frames.
    this._ddgi.invalidateProbeCache();
  }

  /**
   * Topology-change full-rebuild path (Option (a) per items_to_fix.md A3).
   *
   * Picked over Option (b) ("`rebuildBvhLeaf(bvh, leafIndex, newTriangles)`
   * in shared-bvh") because:
   *  - three-mesh-bvh's MeshBVH constructor builds the whole tree
   *    monolithically; surgical leaf-replacement would require
   *    re-implementing SAH partitioning (Option (b) is genuinely
   *    invasive).
   *  - Topology changes are rarer than transform / material edits — the
   *    fast paths (this branch's (c) + the material branch's bytes-only
   *    re-upload) handle the common case. When topology DOES change,
   *    paying ~50 ms for a clean rebuild is the right trade vs. multi-
   *    sprint engineering on a custom partial-rebuilder.
   *
   * The CPU-side BVH builder runs; the pipeline shaders + bind-group
   * layouts + DDGI atlas + per-frame textures are preserved (no
   * `_initPipeline()` re-run). Cost: BVH build (~50 ms / 30k tris) +
   * 4 buffer destroy/recreate. No pipeline recompile.
   */
  private _updatePrimitiveTopologyRebuild(id: string, patch: Partial<ScenePrimitive>): void {
    const root = this._ensureThreeSceneRoot();
    if (root == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no THREE scene available for rebuild.`,
      );
    }

    // Apply the patch to the affected THREE.Mesh in the synthesized
    // scene so the BVH build picks up the new geometry / transform.
    // For now we support the most common topology patches:
    //   - transform (16-element Mat4)
    //   - positions, normals, uvs, tangents, indices (typed arrays from
    //     core/src/scene.ts MeshPrimitive)
    // Other fields (`instances`, `params`, `shape`, `fallbackMesh`,
    // `kind`) require a wholesale primitive replacement; throw with a
    // clear pointer so the host knows to use setScene().
    let mesh: THREE.Mesh | null = null;
    root.traverseVisible((obj) => {
      if (mesh == null && obj.name === id && (obj as THREE.Mesh).isMesh) {
        mesh = obj as THREE.Mesh;
      }
    });
    if (mesh == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
      );
    }
    const meshRef = mesh as THREE.Mesh;

    const p = patch as {
      transform?: ArrayLike<number>;
      positions?: ArrayLike<number>;
      normals?: ArrayLike<number>;
      uvs?: ArrayLike<number>;
      tangents?: ArrayLike<number>;
      indices?: ArrayLike<number>;
      instances?: unknown;
      params?: unknown;
      shape?: unknown;
      fallbackMesh?: unknown;
      kind?: unknown;
    };
    for (const f of ['instances', 'params', 'shape', 'fallbackMesh', 'kind'] as const) {
      if (p[f] !== undefined) {
        throw new Error(
          `HybridEngine.updatePrimitive("${id}"): patching '${f}' requires a primitive ` +
          `replacement, not just an attribute update. Call setScene() with the ` +
          `modified scene instead.`,
        );
      }
    }

    if (p.transform && p.transform.length >= 16) {
      const m = new THREE.Matrix4().fromArray(Array.from(p.transform));
      meshRef.matrix.copy(m);
      meshRef.matrixWorld.copy(m);
      meshRef.matrixAutoUpdate = false;
    }
    if (p.positions) {
      meshRef.geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(Array.from(p.positions)), 3),
      );
    }
    if (p.normals) {
      meshRef.geometry.setAttribute(
        'normal',
        new THREE.BufferAttribute(new Float32Array(Array.from(p.normals)), 3),
      );
    }
    if (p.uvs) {
      meshRef.geometry.setAttribute(
        'uv',
        new THREE.BufferAttribute(new Float32Array(Array.from(p.uvs)), 2),
      );
    }
    if (p.tangents) {
      meshRef.geometry.setAttribute(
        'tangent',
        new THREE.BufferAttribute(new Float32Array(Array.from(p.tangents)), 4),
      );
    }
    if (p.indices) {
      meshRef.geometry.setIndex(
        new THREE.BufferAttribute(new Uint32Array(Array.from(p.indices)), 1),
      );
    }

    // Rebuild the BVH from the patched THREE scene. The old buffers are
    // released after the new ones are uploaded.
    const oldBuffers = this._bvhBuffers;
    const newBuffers = buildReSTIRSceneBVH([root], {
      primaryLightDir:       new THREE.Vector3(...this._primaryLightDir),
      primaryLightIntensity: this._primaryLightIntensity,
    });
    this._bvhBuffers = newBuffers;
    if (oldBuffers) disposeSceneBVH(oldBuffers);

    // Refresh the four BVH GPU buffers + (in case emissive geometry
    // changed) the emitter buffers. Pipeline shaders + bind-group
    // layouts are NOT touched.
    this._pipeline?.refreshBvhFullRebuild(newBuffers);
    this._pipeline?.updateEmitters(newBuffers);

    // Reset the accumulator + invalidate DDGI — geometry topology
    // changed, history is meaningless.
    this._pipeline?.requestAccumReset();
    this._ddgi.invalidateProbeCache();
  }

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

  // ── Resize ─────────────────────────────────────────────────────────────

  /**
   * Resize the render surface WITHOUT rebuilding the BVH or recompiling
   * pipelines. The host calls this whenever the canvas (or device-pixel
   * ratio) changes — much cheaper than the previous resize-storm pattern
   * (engine teardown → recreate engine → poll for ready), which on a
   * single resize tick churned every BVH buffer + every pipeline shader
   * + every DDGI atlas + ~1 GB of FrameResources textures.
   *
   * Behaviour:
   *   - Updates `_width` / `_height` on the engine.
   *   - Calls `WalkaroundGPUPipeline.resize(W, H)` if the pipeline is
   *     live, which destroys + recreates per-frame GPU resources only
   *     (FrameResources textures + reservoir buffers + variance buffers
   *     + GTAO half/full + SVGF persistent textures) at the new size.
   *     The BVH, pipeline shaders, bind-group layouts, DDGI atlases,
   *     and per-pass UBOs are preserved.
   *   - Resets the temporal accumulator + ping-pong indices on the
   *     pipeline (the new textures are blank, so reusing prior history
   *     would sample undefined memory).
   *
   * No-op when called with the current size, or when the pipeline isn't
   * yet live (the new size is stored on the engine; `_initPipeline` will
   * use it when it constructs the pipeline).
   *
   * Cost: O(W·H) GPU memory churn for the FrameResources reallocation;
   * no shader recompile, no BVH rebuild. Typical resize tick: 5-30 ms
   * for the GPU allocations on a 4K surface, vs 500-2000 ms for a full
   * engine teardown + re-init.
   */
  setSize(width: number, height: number): void {
    if (width === this._width && height === this._height) return;
    if (width <= 0 || height <= 0) {
      // Defensive: WebGPU createTexture rejects zero-sized textures.
      // Hosts sometimes feed in transient 0-pixel sizes during resize
      // animations — silently ignore.
      return;
    }
    this._width = width;
    this._height = height;
    if (this._pipeline) {
      this._pipeline.resize(width, height);
    }
    // No DDGI invalidation — the irradiance atlas is world-space, not
    // screen-space, so it survives a resize unchanged.
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
      // Source order: traversal scene set during BVH init (vitrum-derived),
      // host-provided threeScene escape hatch, lazily-synthesized fallback.
      const ddgiScene = this._ddgiTraversalScene ?? this._ensureThreeSceneRoot();
      if (ddgiScene != null) {
        void this._ddgi.updateFrame({
          scene:   ddgiScene,
          device:  this._device,
          enabled: true,
        });
      }
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

    // T3.E telemetry. Fired before the legacy debug mirror so subscribers
    // see every frame, not just debug-mode ones. We pull GPU timings from
    // the pipeline's lastGpuTimings if it exposed any (they're populated
    // by the same timestamp-query infrastructure the debug log uses).
    if (this._frameSubs.length > 0) {
      const gpu = (pipeline as unknown as { lastGpuTimings?: Record<string, number> })
        .lastGpuTimings;
      const passTimings = gpu;
      const gpuTotal = gpu?.['total'];
      const stats: FrameStats = {
        frameTimeMs: dt,
        ...(gpuTotal !== undefined ? { gpuTimeMs: gpuTotal } : {}),
        ...(passTimings ? { passTimings } : {}),
        spp: 1,
      };
      for (const sub of this._frameSubs) {
        try { sub(stats); } catch (err) {
          if (this._verbose) console.warn('[HybridEngine] onFrame subscriber threw', err);
        }
      }
    }

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

  // ── Telemetry (T3.E) ───────────────────────────────────────────────────

  /** Subscribe to per-frame stats. Fired at the end of each successful
   *  renderFrame() call. Returns an unsubscribe function. Subscribers
   *  that throw are swallowed so the render loop stays alive. */
  onFrame(cb: (stats: FrameStats) => void): () => void {
    this._frameSubs.push(cb);
    return () => {
      const i = this._frameSubs.indexOf(cb);
      if (i >= 0) this._frameSubs.splice(i, 1);
    };
  }

  // Walkaround engines don't accumulate, so onProgress doesn't have a
  // meaningful 'pt-spp' to report. DDGI warm-up could surface here once
  // we expose probe-update progress; for now the optional method is
  // intentionally absent (consumers must typeof-check per the contract).

  // ── Debug introspection (T3.G followup) ────────────────────────────────

  /** Debug-introspection surface for @vitrum/dev overlays. Each method
   *  reaches into the live pipeline / DDGI / BVH state and returns the
   *  current handle (engine-owned; callers MUST NOT destroy). Returns
   *  null when the relevant subsystem isn't initialised yet.
   *
   *  Not implemented: pickPrimitive (needs a real picking pass) and the
   *  denoiser-toggle pair (needs pipeline bypass plumbing). Both stay
   *  absent so DenoiserABToggle / MaterialInspector fall back to their
   *  warn paths until those land. */
  readonly debug: EngineDebugSurface = {
    atlasTexture: (): GPUTexture | null => {
      const atlas = this._ddgi?.pass?.getReadAtlasGPUTextures?.();
      return atlas?.irradiance ?? null;
    },
    visibilityAtlasTexture: (): GPUTexture | null => {
      const atlas = this._ddgi?.pass?.getReadAtlasGPUTextures?.();
      return atlas?.visibility ?? null;
    },
    bvhNodes: (): Float32Array | null => {
      const bvh = this._bvhBuffers;
      const buf = bvh?.bvhNodes?.cpuData;
      if (buf == null) return null;
      // BVH node layout: 32 bytes/node, 8 × u32, shared by shared-bvh and
      // pt-webgpu's buildCpuBvh:
      //   f32[0..2] bounds.min xyz
      //   f32[3..5] bounds.max xyz
      //   u32[6]    rightChildOrTriOffset
      //   u32[7]    splitAxisOrTriCount
      // Repackage into the public contract: [min, max, depth=0 placeholder,
      // pad=0]. Depth would need a parent-link traversal we don't have
      // here; the visualiser can colour by node-index ratio as a passable
      // proxy.
      const src = new Float32Array(buf);
      const nodeCount = (buf as ArrayBuffer).byteLength / 32;
      const out = new Float32Array(nodeCount * 8);
      for (let i = 0; i < nodeCount; i++) {
        const so = i * 8;   // 8 f32 lanes per source node
        const oo = i * 8;   // 8 f32 lanes per output node
        out[oo + 0] = src[so + 0]!; // minX
        out[oo + 1] = src[so + 1]!; // minY
        out[oo + 2] = src[so + 2]!; // minZ
        out[oo + 3] = src[so + 3]!; // maxX
        out[oo + 4] = src[so + 4]!; // maxY
        out[oo + 5] = src[so + 5]!; // maxZ
        out[oo + 6] = 0;            // depth — TODO: parent traversal
        out[oo + 7] = 0;            // pad
      }
      return out;
    },
    giSignalTextures: () => {
      // W1-R2 — _res is the nested FrameResources struct. The debug surface
      // reaches in via a structural cast (cannot import FrameResources here
      // without a cyclic dep), so we mirror the sub-struct shape inline.
      const p = this._pipeline as unknown as {
        _res?: {
          common?: { hdrColorTexture?: GPUTexture; hdrIndirectTexture?: GPUTexture };
          gtao?: { aoFullTexture?: GPUTexture };
        };
      } | null;
      const res = p?._res;
      if (res == null) return null;
      // 'direct' = hdrColorTexture (raw shade output, before SVGF).
      // 'indirect' = hdrIndirectTexture (per-channel SVGF input, Sprint 18).
      // 'ao' = aoFullTexture (GTAO bilateral upsample output, Sprint 15).
      // 'total' = current swap chain — not exposed as a persistent
      //   texture; consumers can blit from the canvas directly.
      return {
        direct:   res.common?.hdrColorTexture    ?? null,
        indirect: res.common?.hdrIndirectTexture ?? null,
        ao:       res.gtao?.aoFullTexture        ?? null,
        total:    null,
      };
    },
  };

  // ── Dispose ────────────────────────────────────────────────────────────

  /**
   * Synchronous dispose — releases all engine-owned GPU resources.
   *
   * The contract intentionally remains synchronous (so hosts can call it
   * from React cleanup effects, finalizers, etc. without an async
   * paradigm shift). When an `_initPipeline()` chain is in flight, the
   * actual GPU-resource release for any work that chain hasn't yet
   * published is deferred to the chain's own `finally` block — the chain
   * checks `_pendingTeardown` after every await boundary and, if set,
   * disposes its locals AND finalises teardown of whatever did make it
   * to shared state. This avoids the async-dispose API ripple while
   * still being honest about late-resolving init work.
   *
   * Idempotent: a second `dispose()` call is a no-op.
   */
  dispose(): void {
    if (this._state === 'disposed' && !this._initRunning) {
      // Already disposed and no in-flight chain to coordinate with — no-op.
      return;
    }
    this._disposed = true;
    // We deliberately do NOT bump _initSeq here. The in-flight chain
    // captured `mySeq` at start; it relies on `mySeq === this._initSeq`
    // to know whether IT is the latest writer. If we bumped seq the
    // chain's `finally` would think a newer chain raced past — but
    // there isn't one — and it would skip the teardown finalisation.
    // Instead, dispose communicates intent via `_disposed` +
    // `_pendingTeardown`; the chain's checkpoints check all three.

    if (this._initRunning) {
      // An init is mid-flight. Defer teardown to that chain's finally
      // block — it will dispose its locals AND tear down whatever's
      // currently in shared state (BVH/pipeline/traversal scene that a
      // PRIOR chain published before being raced out by this one). We
      // can't safely call _teardownPipeline() here because the in-flight
      // chain's `await pipeline.initialize()` may still be holding a
      // live reference to a half-built pipeline.
      this._pendingTeardown = true;
      this._state = 'disposed';
      // Note: _ddgi.dispose() is deferred too; the in-flight chain may
      // still call _ddgi.pass.setSunIntensityMultiplier() after the
      // post-pipeline checkpoint, and we don't want a torn-down DDGI
      // under it. The chain's finally calls _ddgi.dispose() when it
      // sees _pendingTeardown.
    } else {
      // No in-flight init; tear down here and now.
      this._teardownPipeline();
      this._ddgi.dispose();
      this._state = 'disposed';
    }

    if (this._debug && typeof window !== 'undefined') {
      const dbg = this._dbg;
      dbg.disposeCount++;
      const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
      console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
        ranForMs: liveMs.toFixed(1),
        framesDispatched: dbg.framesDispatched,
        deferredTeardown: this._pendingTeardown,
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
    // Capture our sequence number — any newer _initPipeline()/teardown bump
    // invalidates the writes below. See `_initSeq` docs.
    const mySeq = ++this._initSeq;

    if (this._debug) {
      const dbg = this._dbg;
      dbg.initCount++;
      dbg.initStart = performance.now();
      console.log(`[hybrid:debug] init #${dbg.initCount} START`, {
        W: this._width, H: this._height, device: !!device,
        t: dbg.initStart.toFixed(0), seq: mySeq,
      });
    }

    this._state = 'initializing';

    // Fire-and-forget: the engine transitions to 'ready' when the BVH and
    // pipeline finish setting up, or to 'error' on failure. Callers do not
    // await this — the engine state machine is the synchronization point.
    this._initRunning = true;
    void (async () => {
      // Poll until scene has enough geometry (or 5s timeout).
      const pollStart = Date.now();
      let pollIters = 0;
      while (!this._disposed && mySeq === this._initSeq) {
        const elapsed = Date.now() - pollStart;
        if (elapsed >= 5_000) break;
        if (this._sceneReadyForBvh()) break;
        await new Promise<void>((r) => setTimeout(r, 50));
        pollIters++;
      }

      if (this._disposed || mySeq !== this._initSeq) {
        if (this._debug) {
          console.log('[hybrid:debug] init aborted during scene-readiness poll', {
            pollIters, disposed: this._disposed, raced: mySeq !== this._initSeq, seq: mySeq,
          });
        }
        return;
      }

      if (this._debug) {
        console.log('[hybrid:debug] scene-ready', { pollIters, elapsed: Date.now() - pollStart, seq: mySeq });
      }

      // Locals — must be disposed if we lose the race before publishing to
      // shared state. `bvhRoot` is owned-and-synthesized iff
      // (bvhRoot !== this._threeScene) — that's the same condition the
      // sync code uses to decide whether _ddgiTraversalScene gets it.
      let bvhRoot: THREE.Object3D | null = null;
      let bvhOwnedSynthesized = false;
      let bvh: SceneBVHBuffers | null = null;
      let pipeline: WalkaroundGPUPipeline | null = null;

      try {
        const bvhStart = performance.now();
        if (this._coreSceneSuppliesMeshes()) {
          bvhRoot = vitrumSceneToThree(this._lastScene!);
          bvhOwnedSynthesized = true;
        } else if (this._threeScene != null) {
          bvhRoot = this._threeScene;
          bvhOwnedSynthesized = false;
        } else {
          // T3.H removal: no vitrum mesh primitives AND no escape-hatch
          // threeScene. The host hasn't given us anything to render against.
          throw new Error(
            '[HybridEngine] BVH source unavailable: setScene(vitrumScene) ' +
            'supplied no mesh primitives and no `threeScene` was passed at ' +
            'construction. Call engine.setScene(sceneFromThreeJS(yourThreeScene)) ' +
            'or pass `threeScene` directly to the engine constructor.',
          );
        }
        bvh = buildReSTIRSceneBVH([bvhRoot], {
          primaryLightDir:       new THREE.Vector3(...this._primaryLightDir),
          primaryLightIntensity: this._primaryLightIntensity,
        });
        const bvhMs = performance.now() - bvhStart;

        // First shared-state write checkpoint. If a newer setScene/reset
        // bumped _initSeq while buildReSTIRSceneBVH ran (it's CPU-side but
        // not instantaneous on heavy scenes), discard our work locally.
        if (this._disposed || this._pendingTeardown || mySeq !== this._initSeq) {
          if (this._debug) {
            console.log('[hybrid:debug] init lost race pre-_ddgiTraversalScene write', {
              disposed: this._disposed, pendingTeardown: this._pendingTeardown,
              raced: mySeq !== this._initSeq, seq: mySeq,
            });
          }
          // Locals will be disposed by the finally block.
          return;
        }
        if (bvhOwnedSynthesized) {
          this._ddgiTraversalScene = bvhRoot as THREE.Scene;
          bvhOwnedSynthesized = false; // ownership transferred to engine
        }
        this._bvhBuffers = bvh;
        const bvhPublished = bvh;
        bvh = null; // ownership transferred to engine

        if (this._debug) {
          console.log('[hybrid:debug] BVH built', {
            bvhMs: bvhMs.toFixed(1),
            triCount: bvhPublished.bvhNodes?.count,
            emitterCount: bvhPublished.emitters?.count,
            seq: mySeq,
          });
        }

        pipeline = new WalkaroundGPUPipeline(device, this._width, this._height);
        const pipelineStart = performance.now();

        // T2.H2 — Neural denoiser: create and initialize InferenceGraph before pipeline init.
        let inferenceGraph: InferenceGraph | undefined;
        if (this._denoiser === 'neural' && this._neuralWeights) {
          const { buildUNetSpec } = await import('./neural/unetArchitecture.js');
          inferenceGraph = new InferenceGraph(buildUNetSpec());
          await inferenceGraph.initialize(device, this._neuralWeights, this._width, this._height);
        }

        await pipeline.initialize(
          bvhPublished,
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

        // Final shared-state write checkpoint — pipeline.initialize() awaits
        // shader compilation (~50–500 ms). A newer setScene/reset/dispose
        // raced ahead in the meantime → discard locally and dispose locals
        // in finally. We MUST NOT publish `pipeline` to `_pipeline` in this
        // case; the newer chain has its own pipeline coming.
        if (this._disposed || this._pendingTeardown || mySeq !== this._initSeq) {
          if (this._debug) {
            console.log('[hybrid:debug] init lost race post-pipeline.initialize', {
              pipelineMs: pipelineMs.toFixed(1),
              disposed: this._disposed, pendingTeardown: this._pendingTeardown,
              raced: mySeq !== this._initSeq, seq: mySeq,
            });
          }
          // Also tear down the BVH we already published to _bvhBuffers if
          // it's still ours; otherwise the newer chain has replaced it.
          if (this._bvhBuffers === bvhPublished) {
            disposeSceneBVH(bvhPublished);
            this._bvhBuffers = null;
          }
          // And the traversal scene we published if it's still ours.
          if (this._ddgiTraversalScene !== null
              && this._ddgiTraversalScene === bvhRoot
              && bvhRoot !== this._threeScene) {
            disposeVitrumThreeSceneRoot(this._ddgiTraversalScene);
            this._ddgiTraversalScene = null;
          }
          // pipeline disposed by the finally block.
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
        pipeline = null; // ownership transferred to engine
        this._state        = 'ready';

        if (this._debug) {
          const dbg = this._dbg;
          const totalMs = performance.now() - dbg.initStart;
          console.log(`[hybrid:debug] init #${dbg.initCount} COMPLETE`, {
            pipelineMs: pipelineMs.toFixed(1), totalMs: totalMs.toFixed(1), seq: mySeq,
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
      } finally {
        // Dispose any locals that did NOT make it to shared state. After
        // a successful run all three are null (ownership transferred). On
        // a race / error / dispose, whichever weren't transferred get freed
        // here so we don't leak ~1 GB of GPU resources per loser.
        if (pipeline) {
          try { pipeline.dispose(); } catch {}
        }
        if (bvh) {
          try { disposeSceneBVH(bvh); } catch {}
        }
        if (bvhRoot && bvhOwnedSynthesized) {
          try { disposeVitrumThreeSceneRoot(bvhRoot); } catch {}
        }
        // If dispose() raced and left _pendingTeardown set, finalise the
        // teardown now. The newest writer (us, if we published successfully)
        // is responsible for actually tearing down — we're the last live
        // reference. Note: if a newer chain is still running, its own
        // checkpoints will see _pendingTeardown and bail before publishing.
        if (this._pendingTeardown && mySeq === this._initSeq) {
          if (this._debug) {
            console.log('[hybrid:debug] init finally — finalising deferred teardown', { seq: mySeq });
          }
          this._teardownPipeline();
          // _ddgi.dispose() was deferred by dispose() since init was in-flight;
          // it's safe to call now because no chain is using it any more.
          try { this._ddgi.dispose(); } catch {}
          this._state = 'disposed';
        }
        // Always clear _initRunning at the end of OUR chain — but only if
        // we're still the latest. A newer chain will have set it back to
        // true; we MUST NOT clear another chain's flag.
        if (mySeq === this._initSeq) {
          this._initRunning = false;
        }
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
  // Bootstrap setScene with an empty vitrum Scene. Two callers depend on
  // this:
  //   1. Hosts that pass `threeScene` at construction and never call setScene
  //      themselves (e.g. examples/two-engines-one-scene). Without the
  //      bootstrap they'd never trigger _initPipeline → engine stays
  //      'uninitialized' → renderFrame returns skip output forever.
  //   2. Hosts that DO call setScene afterwards (e.g. @vitrum/engine.createEngine).
  //      The host's setScene fires init-B which races init-A. The init-flight
  //      guard (HybridEngine._initSeq, see _initPipeline()) ensures the loser
  //      bootstrap chain disposes its locals — no GPU resource leak. The
  //      bootstrap is wasted work but safe.
  //
  // We could remove the bootstrap and require all hosts to call setScene
  // explicitly, but that would silently break case 1 and offer no safety
  // benefit (Fix 1 already eliminates the race-leak class).
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
}
