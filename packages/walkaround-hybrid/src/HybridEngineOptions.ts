/**
 * HybridEngineOptions — construction-time option types for {@link HybridEngine}.
 *
 * Extracted from `HybridEngine.ts` (refactor sweep 2026-05-18). Pulled out
 * so the engine orchestrator file is not dominated by ~340 lines of
 * pure-JSDoc interface definitions. The construction-time validation
 * (denoiser enum check, neuralWeights presence check, oidnModelUrl
 * presence check, deprecated-svgf-alias warning) still lives in the
 * `HybridEngine` constructor — only the type shape itself moved.
 */

import type * as THREE from 'three';
import type { EngineOptions } from '@vitrum/core';
import type { DDGILight } from './ddgi/types.js';
import type { ModelWeights } from './neural/weights.js';

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
   *
   *   `'oidn-final'` — W11 — Intel Open Image Denoise final-pass via ONNX
   *   Runtime Web (`@vitrum/shared-denoisers/oidnBridge`). Async/stale-by-
   *   one-frame model: each dispatch reads back the HDR + albedo + normal
   *   buffers, runs OIDN on the CPU/GPU/WebNN, and uploads the denoised
   *   RGB back into a vt-owned output texture (≈50-200 ms inference).
   *   Requires `extensions['walkaround-hybrid'].oidnModelUrl` to be supplied
   *   (URL or path to the bundled .onnx model file). Optional peer dep
   *   `onnxruntime-web` must be installed at runtime.
   */
  readonly denoiser?: 'atrous' | 'atrous-variance' | 'svgf-real' | 'svgf' | 'neural' | 'oidn-final';

  /**
   * Pre-loaded model weights for the neural denoiser (T2.H2).
   * Required when `denoiser === 'neural'`. Load via `loadWeightsFromArrayBuffer()`
   * from the vitrum binary format exported by `tools/neural-denoiser-training/export_weights.py`.
   *
   * If `denoiser === 'neural'` and `neuralWeights` is undefined, the engine
   * constructor throws with a helpful error pointing to the training README.
   */
  readonly neuralWeights?: ModelWeights;

  /**
   * Backend-specific creation-time configuration (per the
   * `@vitrum/core` `EngineOptions.extensions` design-principle: backends
   * own their own extension namespace). Hosts pass
   * `extensions: { 'walkaround-hybrid': { ... } }` to thread non-generic
   * config (today: OIDN model URL) without polluting the generic
   * `EngineOptions` surface.
   *
   * Keys currently consumed:
   *   - `'walkaround-hybrid'.oidnModelUrl` (W11) — required when
   *     `denoiser === 'oidn-final'`. URL or path to the bundled OIDN
   *     `.onnx` model file (e.g. `'/models/oidn_rt_hdr_alb_nrm.onnx'`).
   *     The host is responsible for serving the file.
   *   - `'walkaround-hybrid'.oidnExecutionProviders` (W11) — optional
   *     override of the ONNX Runtime Web execution-provider order;
   *     default `['webnn', 'webgpu', 'wasm']` (Decision 11).
   */
  readonly extensions?: Readonly<Record<string, unknown>> & {
    readonly 'walkaround-hybrid'?: {
      readonly oidnModelUrl?: string;
      readonly oidnExecutionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
    };
  };

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

  /**
   * 2026-05-18 sweep — Probe-side glass-transmission perceptual mix scale.
   * When a DDGI probe ray hits a transmissive surface, the probe's radiance
   * is `mix(roomRadiance, transmitted, mat.transmission * glassMixScale)`.
   * Cornell default 0.7 leaves 30 % of the room radiance on fully-transparent
   * glass; raise toward 1 for sky-tint-dominated transmission, lower toward 0
   * to keep room-bounce contribution dominant.
   *
   * @default 0.7
   */
  readonly glassMixScale?: number;

  /**
   * 2026-05-18 sweep — ReSTIR-GI per-pixel unbiased weight cap.
   * Bounds firefly contribution from tiny `p̂` denominators. Cornell default
   * 16.0 admits legitimate variance the unbiased estimator needs while
   * bounding pathological grazing-angle samples.
   *
   * @default 16.0
   */
  readonly restirGiWCap?: number;

  /**
   * 2026-05-18 sweep — DDGI irradiance clamp at the ReSTIR-GI reconnection
   * vertex (`risGi`). Caps `sampleDDGIAtPoint(xs, ns)` per channel.
   *
   * **Light-intensity-sensitive.** Cornell default 5.0 is calibrated for
   * Le=12 indirect-band peaks.  Brighter emitters need higher caps.
   *
   * @default 5.0
   */
  readonly restirGiIrrClamp?: number;

  /**
   * 2026-05-18 sweep — ReSTIR-GI temporal previous-frame M clamp. Higher
   * makes the chosen sample change less often → less per-frame pattern
   * jitter; lower lets new samples take over faster.
   *
   * **Framerate-sensitive.** Cornell default 50 ≈ 0.83 s at 60 FPS.
   *
   * @default 50
   */
  readonly restirGiMClamp?: number;

  /**
   * 2026-05-18 sweep — ReSTIR-GI spatial-reuse disc radius (half-res pixels).
   * The spatial-reuse pass samples K=5 neighbours within this radius.
   *
   * **Resolution-sensitive.** Cornell default 12.0 px.  Hosts at very high
   * resolution may scale by `screenHeight / 1080`.
   *
   * @default 12.0
   */
  readonly restirGiSpatialRadiusPx?: number;

  /**
   * 2026-05-18 sweep — ReSTIR-GI spatial-reuse normal-alignment minimum
   * cosine.  Neighbour reservoirs whose normal makes an angle larger than
   * `acos(restirGiSpatialNormalDotMin)` with the centre pixel are rejected.
   *
   * Cornell default 0.906 ≈ cos(25°).
   *
   * @default 0.906
   */
  readonly restirGiSpatialNormalDotMin?: number;

  /**
   * 2026-05-18 sweep — ReSTIR-GI spatial-reuse coplanarity tolerance in
   * **world units**.  Neighbour reservoirs whose position deviates from the
   * centre pixel's tangent plane by more than this amount are rejected.
   *
   * **Scene-scale-sensitive.** Cornell default 0.05 (5 cm).  Hosts on
   * different scales should pass `sceneDiagonal × 1e-3` ish.
   *
   * @default 0.05
   */
  readonly restirGiSpatialCoplanarTol?: number;

  /**
   * 2026-05-18 sweep — per-channel HDR clamp on the indirect-radiance
   * channel before the atrous chain (`shade.wgsl`).  Kills firefly tails
   * that survive ReSTIR W-capping and atrous variance estimation.
   *
   * **Light-intensity-sensitive.** Cornell default `[1.0, 1.0, 1.0]`.
   * Brighter scenes need proportionally higher per-channel caps.
   *
   * @default [1.0, 1.0, 1.0]
   */
  readonly indirectFireflyClamp?: readonly [number, number, number];

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
