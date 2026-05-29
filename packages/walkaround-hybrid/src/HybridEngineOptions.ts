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
import type { CascadeDim } from '@vitrum/walkaround-rc';

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

/**
 * Source-of-truth key list for {@link LightingOptions}. The
 * `satisfies readonly (keyof LightingOptions)[]` constraint makes the compiler
 * reject any key that is not a real `LightingOptions` field, so this list can
 * only drift out of sync with the interface by deletion (a key removed from
 * the interface fails the `satisfies` check), never by addition of a typo.
 *
 * Consumed by {@link assertKnownLightingKeys} for the runtime unknown-key
 * guard in `HybridEngine.updateLighting`.
 */
export const LIGHTING_OPTION_KEYS = [
  'primaryLightDir',
  'primaryLightIntensity',
  'skyTint',
  'skyIrradiance',
] as const satisfies readonly (keyof LightingOptions)[];

const LIGHTING_OPTION_KEY_SET: ReadonlySet<string> = new Set(LIGHTING_OPTION_KEYS);

/**
 * Cheap, non-throwing unknown-key guard for {@link HybridEngine.updateLighting}.
 *
 * `Engine.updateLighting` is contractually opaque (`Readonly<Record<string,
 * unknown>>` in `@vitrum/core`) — backends own their own lighting vocabulary.
 * To keep that opacity while still surfacing silent drops, this helper
 * `console.warn`s once per unrecognised key per call so hosts notice typos /
 * stale keys instead of having them silently ignored. (No cross-call dedup:
 * `updateLighting` is host-driven scrubbing, not a per-frame hot path.)
 *
 * Does NOT throw and does NOT mutate `opts`; the caller's field-by-field
 * application logic still runs unchanged for the keys it recognises.
 */
export function assertKnownLightingKeys(opts: Readonly<Record<string, unknown>>): void {
  for (const k of Object.keys(opts)) {
    if (!LIGHTING_OPTION_KEY_SET.has(k)) {
      console.warn(
        `[@vitrum/walkaround-hybrid] updateLighting: ignoring unknown key "${k}"`,
      );
    }
  }
}

export interface HybridEngineOptions extends EngineOptions {
  /** WebGPU device (narrowed from the opaque `device: unknown` on EngineOptions). */
  readonly device: GPUDevice;

  /** Physical pixel width of the render surface. */
  readonly width: number;

  /** Physical pixel height of the render surface. */
  readonly height: number;

  /**
   * PR-7 — when true, run {@link solveSkin} for every `skinned-mesh` primitive
   * at the start of each {@link HybridEngine.renderFrame} and refit BLAS via
   * `updatePrimitive({ positions, normals })`.
   */
  readonly gpuSkinning?: boolean;

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
   * Was REQUIRED pre-T3.H; the requirement was dropped 2026-05-12 (the
   * field itself stays as an escape hatch). Hosts that previously passed
   * `threeScene: someScene` can drop it if they also call
   * `setScene(sceneFromThreeJS(someScene))` afterwards. If they do
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
   *   `'bmfr'` — Koskela et al. 2019, "Blockwise Multi-Order Feature
   *   Regression for Real-Time Path-Tracing Reconstruction" (ACM TOG 38(5)).
   *   Per 32×32 screen block, least-squares-fits the noisy 1-spp color to a
   *   10-feature matrix [1, p.xyz, n.xyz, p².xyz] via Householder QR on the
   *   normal equations and reconstructs `color = T·α`, then temporally
   *   accumulates (EMA, reset on camera motion). Uses a screen-space position
   *   proxy from the gNormalDepth depth channel — no dedicated world-position
   *   G-buffer required. Owns a private rgba16float history ping-pong.
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
  readonly denoiser?: 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final';

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
   *   - `'walkaround-hybrid'.bvhMode` (PR-2) — `'merged'` | `'tlas'` to
   *     force CPU pack mode. When omitted, multi-mesh / instanced vitrum
   *     scenes default to TLAS pack (GPU traversal still merged until PR-3).
   */
  readonly extensions?: Readonly<Record<string, unknown>> & {
    readonly 'walkaround-hybrid'?: {
      readonly oidnModelUrl?: string;
      readonly oidnExecutionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
      readonly bvhMode?: 'merged' | 'tlas';
    };
  };

  // ── Phase-0 productization — quality preset (roadmap §4.3 / §5.1) ────────

  /**
   * Coarse quality preset. Resolves to concrete tunable / UBO / pass-gate
   * values per the roadmap §4.3 table (see `HybridEngineQualityPreset.ts`).
   *
   * The preset is a BASELINE, not a lock: explicit per-knob options in this
   * same object OVERRIDE the preset (e.g. `{ qualityTier: 'low',
   * gtao: { radiusPx: 99 } }` keeps the explicit gtao radius while applying
   * the other low-tier values).
   *
   * Default: `'ultra'` — byte-identical to the pre-Phase-0 Cornell-baseline
   * defaults. Existing hosts that never set this are unaffected.
   *
   * @default 'ultra'
   */
  readonly qualityTier?: 'ultra' | 'high' | 'medium' | 'low';

  /**
   * Phase-0 productization — hybrid resource tier (Deliverable 3).
   *
   * - `'full'` (default) — the full pipeline: TLAS-capable, RC/PPG/neural
   *   allowed, requires `HYBRID_WEBGPU_REQUIRED_LIMITS` (16 buf / 8 tex).
   * - `'lite'` — a reduced-budget path for adapters that meet only
   *   `HYBRID_LITE_LIMITS` (≈10 buf / 5 tex). Lite runs the SAME shade pipeline
   *   (no WGSL fork) but:
   *     - forces `extensions['walkaround-hybrid'].bvhMode = 'merged'` (drops the
   *       5 TLAS scene-group storage buffers — the buffer-axis win) even when a
   *       needs-TLAS scene would otherwise default to TLAS, with a `console.warn`
   *       that instanced-scene fidelity is reduced;
   *     - FORBIDS `rcEnabled` / `ppgEnabled` / `denoiser:'neural'` (they need
   *       extra GPU resources / weights) — the constructor throws an actionable
   *       error if any is set with `tier:'lite'`;
   *     - biases the default `qualityTier` to `'medium'` (still overridable).
   *
   * `@vitrum/engine`'s `createEngine()` selects `'lite'` automatically when the
   * adapter profile reports `hybridCapable:false && hybridLiteCapable:true`.
   *
   * @default 'full'
   */
  readonly tier?: 'full' | 'lite';

  /**
   * Per-knob override for the preset's GTAO dispatch mode. `'on'` = half-res
   * (default), `'quarter'` = quarter-res dispatch (cheaper, softer AO),
   * `'off'` = skip GTAO + the bilateral upsample entirely. When omitted, the
   * value comes from {@link qualityTier}'s preset.
   */
  readonly gtaoMode?: 'on' | 'quarter' | 'off';

  /**
   * Per-knob override for the ReSTIR-DI spatial-reuse ping-pong pass count
   * (1 or 2). 2 is the full-fidelity variance reducer; 1 halves the spatial
   * cost. When omitted, comes from {@link qualityTier}'s preset.
   */
  readonly diSpatialPasses?: 1 | 2;

  /**
   * Per-knob override for the ReSTIR-GI spatial-reuse ping-pong pass count
   * (1 or 2). When omitted, comes from {@link qualityTier}'s preset.
   */
  readonly giSpatialPasses?: 1 | 2;

  /**
   * Per-knob override for the DDGI round-robin probe-update divisor
   * (`probesPerFrame = ceil(totalProbes / divisor)`). Higher ⇒ fewer probes
   * updated per frame ⇒ cheaper but slower GI response to lighting changes.
   * Default (and the historical hardcoded value) is 4. When omitted, comes
   * from {@link qualityTier}'s preset.
   */
  readonly ddgiUpdateDivisor?: number;

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
   * NOTE (T5): `caustic.boost` / `caustic.visClamp` only have any effect
   * when {@link stainedGlass}`.sunCaustic` is also `true`. When the
   * sun-caustic term is OFF (the default) the caustic-boost calibration is
   * never reached because `lo_sg_caustic` early-returns `vec3f(0)`.
   *
   * @default { boost: 1.0, visClamp: 1.0 }
   */
  readonly caustic?: {
    readonly boost?: number;
    readonly visClamp?: number;
  };

  /**
   * T5 — opt-in stained-glass-specific lighting physics.
   *
   * The hybrid shade pass historically ran two stained-glass-specific direct
   * lighting terms UNCONDITIONALLY: a sun-caustic term (sun directional light
   * reaching a receiver through tinted glass, with the {@link caustic}
   * boost/visClamp calibration) and a 5-tap sky-aperture probe (diffuse sky
   * illumination through a window cutout). Those terms were physically
   * appropriate only for cathedral-window / Cornell-stained-glass scenes; a
   * generic scene received them anyway, which is incorrect.
   *
   * T5 moved both terms into an opt-in WGSL module (`stainedGlassShade.wgsl.ts`,
   * `lo_sg_caustic` / `lo_sg_aperture`) gated by a UBO flag bit — mirroring the
   * Radiance-Cascades `sampleCascadeC0` precedent. When a flag is unset the
   * helper early-returns `vec3f(0)`; flag-OFF is therefore bit-identical to "no
   * such term" without a separate shader compile.
   *
   * **Default both `false`** → generic scenes get ZERO caustic / aperture
   * physics. Stained-glass hosts (e.g. the Cornell-stained-glass example) opt
   * in with `{ sunCaustic: true, skyAperture: true }`.
   *
   * - `sunCaustic` — enable the through-glass sun-caustic term. Pairs with the
   *   {@link caustic} boost/visClamp calibration.
   * - `skyAperture` — enable the 5-tap diffuse-sky-aperture probe. Pairs with
   *   {@link skyTint} / {@link skyIrradiance}.
   *
   * @default { sunCaustic: false, skyAperture: false }
   */
  readonly stainedGlass?: {
    readonly sunCaustic?: boolean;
    readonly skyAperture?: boolean;
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

  /**
   * Atrous DIRECT-channel sigmas `[sigmaN, sigmaZ, sigmaC]` for the
   * shadow/caustic-edge-preserving wavelet filter applied to the direct
   * radiance channel after RIS/spatial reuse.
   *
   * - `sigmaN` — normal-alignment falloff (cos-similarity sharpness)
   * - `sigmaZ` — depth-tolerance in world units
   * - `sigmaC` — color-distance tolerance (HDR linear scalar)
   *
   * Cornell default `[128.0, 5.0, 0.05]` — tight stops that preserve hard
   * shadow + caustic edges. Hosts on different scene scales should pass
   * a proportional `sigmaZ` (depth is the only world-unit-scaled axis).
   *
   * @default [128.0, 5.0, 0.05]
   */
  readonly atrousDirectSigmas?: readonly [number, number, number];

  /**
   * Atrous INDIRECT-channel sigmas `[sigmaN, sigmaZ, sigmaC]` for the
   * wavelet filter applied to the ReSTIR-GI indirect channel.
   *
   * Broader on every axis than the direct sigmas because ReSTIR-GI
   * already smooths the indirect signal temporally + spatially; the
   * remaining 2×2 quad variance (from half-res reservoir reads) just
   * needs a wide low-pass.
   *
   * Cornell default `[32.0, 20.0, 0.5]`. Hosts on different scene scales
   * should pass a proportional `sigmaZ`.
   *
   * @default [32.0, 20.0, 0.5]
   */
  readonly atrousIndirectSigmas?: readonly [number, number, number];

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

  /**
   * PPG train-pass dispatch cadence (Müller 2017 §3.3). The path-guiding
   * `guide` + `update` compute passes run only on frames where
   * `frameCount % ppgDispatchInterval === 0`. The learned sTree/dTree GPU
   * buffers PERSIST between train cycles and the gi-ris guided SAMPLING reads
   * them EVERY frame, so a higher interval is a pure training-cost lever — it
   * never changes whether guided sampling is active, only how often the tree
   * is retrained. `1` trains every frame (no behaviour change); `N > 1` trains
   * every Nth frame. Clamped to ≥ 1 internally. Only meaningful when
   * `ppgEnabled: true`.
   *
   * Default: resolved from `qualityTier` (`resolveQualityPreset`) —
   * ultra/high = 1, medium = 2, low = 4. An explicit value here OVERRIDES the
   * preset.
   */
  readonly ppgDispatchInterval?: number;

  // ── RC (Sannikov 2023 Radiance Cascades — W8 sprint, 2026-05-18) ──────────

  /**
   * Enable the Sannikov 2023 Radiance Cascades subsystem inside HybridEngine.
   *
   * When `true`, the engine instantiates an {@link RCSubsystem} that:
   *   - Builds a per-engine RC BVH (~50 ms for ~30K-tri scenes) on each
   *     `setScene` call. Today this builds a SEPARATE BVH from the
   *     ReSTIR-DI BVH — future W2-style work may unify them.
   *   - Allocates raw `GPUBuffer`s for each of the 5 cascades (per
   *     `CASCADE_DIMS`) — memory cost depends on cascade sizing.
   *   - Dispatches the cascade compute pipeline (5 cast passes + 4 merge
   *     passes) each frame.
   *
   * The cascade-0 buffer is exposed via the pipeline as the indirect-diffuse
   * source for shade.wgsl. RC inputs are wired into shade.wgsl via
   * balance-heuristic MIS (W8 Phase 3, rcWeight option). See
   * plan/w8-rc-mis-composition.md.
   *
   * Default: `false` — RC is opt-in until Phase 3 demonstrates first-bounce
   * indirect quality gain over DDGI-only.
   *
   * @see plan/w8-rc-mis-composition.md for the full sprint plan.
   */
  readonly rcEnabled?: boolean;

  /**
   * W8 Phase 3 — Track-A balance-heuristic MIS weight for the Radiance
   * Cascades contribution to `Lo_indirect`. The ReSTIR-GI contribution
   * receives `1 - rcWeight`; the two are forced to sum to 1 so the
   * estimator stays normalised regardless of host choice.
   *
   * Range: [0, 1]. Default: 0.5 when `rcEnabled: true`, 0 otherwise.
   *
   * Tuning notes:
   *   - 0.0 — pure ReSTIR-GI (matches pre-Phase-3 behaviour).
   *   - 1.0 — pure RC; ReSTIR-GI's reservoir read is multiplied by 0
   *     in the indirect sum. Useful for verifying cascade-0 alone.
   *   - 0.5 — equal-weight mix. Should look like ReSTIR-GI's diffuse
   *     gain damped by half + half of RC's smoother spatial signal.
   *
   * @default 0.5 (effective only when rcEnabled === true)
   */
  readonly rcWeight?: number;

  /**
   * B3b (2026-05-19) — Cornell-tuned cascade-pyramid dimensions override.
   * Each entry specifies `probes` (3D probe grid), `rays` per probe, and
   * the `intervalNear`/`intervalFar` world-space slab bounds for that
   * cascade level. The default 5-cascade pyramid in
   * `@vitrum/walkaround-rc/CASCADE_DIMS` is tuned for Cornell-aspect-ratio
   * scenes at metre-scale.
   *
   * Hosts on different scene shapes should pass dims sized to the scene's
   * aspect ratio (`probes` along the dominant axis); hosts on different
   * scene scales should pass proportional `intervalNear`/`intervalFar`
   * world-unit bounds.
   *
   * @default CASCADE_DIMS from @vitrum/walkaround-rc
   */
  readonly cascadeDims?: readonly CascadeDim[];
}
