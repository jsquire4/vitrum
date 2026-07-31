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

import type { EngineOptions } from '@vitrum/core';
import type { DDGILight } from './ddgi/types.js';
import type { ModelWeights } from './neural/weights.js';
import type { CascadeDim } from '@vitrum/walkaround-rc';

/**
 * All denoiser ids supported by walkaround-hybrid.
 * Single source of truth — the union type is derived from this array so
 * adding a new denoiser is a one-edit change.
 */
export const VALID_DENOISERS = [
  'none',
  'auto',
  'atrous',
  'atrous-variance',
  'svgf-real',
  'bmfr',
  'neural',
  'oidn-final',
] as const;

/** Union of all supported walkaround-hybrid denoiser identifiers. */
type ValidDenoiser = (typeof VALID_DENOISERS)[number];
import type { Tunables } from './HybridEngineTuning.js';
import type { HybridEnvironmentMapResolver } from './environment/resolveHybridEnvironment.js';
import type { NrcConfig } from './neural/nrc/nrcSubsystem.js';
import {
  canonicalizeLightingDirectionF32,
  packNonNegativeLightingFloat32,
  packNonNegativeLightingRgbF32,
} from './lightingFloat32.js';

/**
 * Runtime-mutable lighting parameters for {@link HybridEngine.updateLighting}.
 * All fields are optional; omitting a field leaves the corresponding engine
 * parameter unchanged.
 */
export interface LightingOptions {
  /** Explicit primary-direction override (world-space, normalised). Once set,
   *  it supersedes authored scene directional vectors for this engine. */
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
const LIGHTING_OPTION_KEYS = [
  'primaryLightDir',
  'primaryLightIntensity',
  'skyTint',
  'skyIrradiance',
] as const satisfies readonly (keyof LightingOptions)[];

const LIGHTING_OPTION_KEY_SET: ReadonlySet<string> = new Set(LIGHTING_OPTION_KEYS);

/**
 * Strict unknown-key guard for {@link HybridEngine.updateLighting}.
 *
 * `Engine.updateLighting` is contractually opaque (`Readonly<Record<string,
 * unknown>>` in `@vitrum/core`) — backends own their own lighting vocabulary.
 * Backend opacity must not become a typo sink: any unrecognised key throws
 * synchronously before the caller mutates lighting or GPU-facing state.
 */
export function assertKnownLightingKeys(
  opts: Readonly<Record<string, unknown>>,
): void {
  if (
    opts == null || typeof opts !== 'object' || Array.isArray(opts)
    || ArrayBuffer.isView(opts)
  ) {
    throw new TypeError(
      '[@vitrum/walkaround-hybrid] updateLighting: options must be a plain object.',
    );
  }
  for (const k of Object.keys(opts)) {
    if (!LIGHTING_OPTION_KEY_SET.has(k)) {
      throw new TypeError(
        `[@vitrum/walkaround-hybrid] updateLighting: unknown key "${k}". ` +
        `Supported keys: ${LIGHTING_OPTION_KEYS.join(', ')}.`,
      );
    }
  }

  const assertVec3 = (key: 'primaryLightDir' | 'skyTint', nonNegative: boolean): void => {
    const value = opts[key];
    if (value === undefined) return;
    if (!Array.isArray(value) || value.length !== 3) {
      throw new TypeError(
        `[@vitrum/walkaround-hybrid] updateLighting: ${key} must be an exact ` +
        `finite ${nonNegative ? 'non-negative ' : ''}[x, y, z] tuple.`,
      );
    }
    const lanes: readonly unknown[] = value;
    if (lanes.some((lane) =>
      typeof lane !== 'number' || !Number.isFinite(lane) || (nonNegative && lane < 0)
    )) {
      throw new TypeError(
        `[@vitrum/walkaround-hybrid] updateLighting: ${key} must be an exact ` +
        `finite ${nonNegative ? 'non-negative ' : ''}[x, y, z] tuple.`,
      );
    }
  };
  const assertNonNegativeScalar = (
    key: 'primaryLightIntensity' | 'skyIrradiance',
  ): void => {
    const value = opts[key];
    if (value === undefined) return;
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new TypeError(
        `[@vitrum/walkaround-hybrid] updateLighting: ${key} must be finite and non-negative.`,
      );
    }
  };
  assertVec3('primaryLightDir', false);
  assertVec3('skyTint', true);
  assertNonNegativeScalar('primaryLightIntensity');
  assertNonNegativeScalar('skyIrradiance');

  // Validate the exact binary32 publication boundary as part of the closed
  // runtime vocabulary check. The engine snapshots the returned canonical
  // values before publication; these calls keep malformed values from reaching
  // any transaction preparation.
  if (opts.primaryLightDir !== undefined) {
    canonicalizeLightingDirectionF32(
      opts.primaryLightDir as [number, number, number],
      'updateLighting.primaryLightDir',
    );
  }
  if (opts.primaryLightIntensity !== undefined) {
    packNonNegativeLightingFloat32(
      opts.primaryLightIntensity as number,
      'updateLighting.primaryLightIntensity',
    );
  }
  if (opts.skyTint !== undefined) {
    packNonNegativeLightingRgbF32(
      opts.skyTint as [number, number, number],
      'updateLighting.skyTint',
    );
  }
  if (opts.skyIrradiance !== undefined) {
    packNonNegativeLightingFloat32(
      opts.skyIrradiance as number,
      'updateLighting.skyIrradiance',
    );
  }
}

export interface HybridEngineOptions extends EngineOptions {
  /** WebGPU device (narrowed from the opaque `device: unknown` on EngineOptions). */
  readonly device: GPUDevice;

  /** Physical pixel width of the render surface. When neural denoising is
   *  selected, the quality-preset-scaled internal width must be >= 8 and a
   *  multiple of 8. */
  readonly width: number;

  /** Physical pixel height of the render surface. When neural denoising is
   *  selected, the quality-preset-scaled internal height must be >= 8 and a
   *  multiple of 8. */
  readonly height: number;

  /**
   * Resolution policy for the resolution-dependent walkaround frame graph.
   *
   * - `'auto'` (default) keeps the full swap-chain presentation size but
   *   selects the largest internal render resolution that fits the persistent
   *   frame-resource budget and reported WebGPU limits. It first increases the
   *   ReSTIR reservoir scale, preserving full-resolution shading, and only
   *   then lowers the complete internal graph.
   * - `'native'` requires the exact quality-scaled internal resolution and
   *   rejects before allocation when it does not fit. It never silently
   *   downscales an explicit native request.
   */
  readonly frameResourceResolutionPolicy?: 'auto' | 'native';

  /**
   * Steady-state byte ceiling for resolution-dependent persistent frame
   * resources. Defaults to 384 MiB. Transactional resize temporarily owns two
   * generations, so its reported peak is bounded by twice this value.
   */
  readonly maxPersistentFrameResourceBytes?: number;

  /**
   * Explicit integer ReSTIR grid scale. DI uses `floor(internal/scale)` and GI
   * uses `floor(internal/(2*scale))`. Omit to let the bounded resolver choose
   * the smallest scale in `[1,4]` that preserves the largest shading
   * resolution. PPG/NRC currently require scale 1.
   */
  readonly restirReservoirScale?: 1 | 2 | 3 | 4;

  /**
   * PR-7 — when true, run eligible `skinned-mesh` primitives through the GPU
   * skinning kernel at the start of each {@link HybridEngine.renderFrame}, then
   * refit the live BVH. Meshes with morph targets, tangents, non-identity bind
   * matrices, unavailable compute support, or missing BVH ranges fall back to
   * the canonical CPU {@link solveSkin} path and the same positions/normals
   * refit.
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
   * Legacy primary directional fallback (world-space, normalised) used when
   * the scene has no authored directional emitter. Scene directions are the
   * default source of truth; calling
   * `updateLighting({ primaryLightDir })` establishes an explicit persistent
   * host override for BVH, DDGI, RC, and per-frame sun shading together.
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

  /** Light list for DDGI probe update pass. */
  readonly lights?: DDGILight[];

  readonly ddgiMaxMaterials?: number;

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
   *   `'auto'` — choose a concrete denoiser from host-supplied assets at engine
   *   construction. Full tier selects neural only for a production-ready v2
   *   checkpoint, otherwise OIDN when configured, then resolves to the
   *   preset/default denoiser with a structured diagnostic.
   *
   *   `'atrous'` — legacy three-pass edge-stopping à-trous only.
 *
 *   `'none'` — pass-through mode; skips denoiser dispatch and composites the
 *   raw HDR target directly.
 *
 *   `'svgf-real'` — T2.H1 — full Schied 2017 SVGF: bilinear motion-vector
   *   reprojection, depth+normal+objId disocclusion test (Eq. 2), per-pixel
   *   history-length texture (Eq. 3), EMA α=max(α_min, 1/(h+1)) (Eq. 4),
   *   variance-from-moments (Eq. 5), 7×7 spatial fallback for disoccluded pixels
   *   (§4.3). Requires historyLength (r32uint) + momentsHistory (rgba32float) +
   *   prevRadiance (rgba16float) persistent textures: ~52 MB at 1080p.
   *
   *   `'bmfr'` — Koskela et al. 2019, "Blockwise Multi-Order Feature
   *   Regression for Real-Time Path-Tracing Reconstruction" (ACM TOG 38(5)).
   *   Overlapping 32×32 blocks least-squares-fit noisy 1-spp color to the
   *   10-feature matrix [1, p.xyz, n.xyz, p².xyz] by cooperative direct
   *   Householder TSQR (without forming normal equations). A second pass
   *   deterministically averages covering reconstructions before temporal EMA.
   *   The persistent walkaround path derives `p` from screen coordinates and
   *   signed gNormalDepth depth; the standalone shared-denoiser entry point
   *   instead requires world positions. BMFR owns a private rgba16float history
   *   ping-pong plus a per-block coefficient buffer. It is explicit-only and
   *   available on `tier:'full'`; it is never selected by `'auto'`.
   *
   *   `'neural'` — T2.H2 — GPU U-Net denoiser (Chaitanya et al. 2017 / Ronneberger
   *   et al. 2015). Requires `neuralWeights` to be provided. The canonical
   *   three-level U-Net requires internal render width/height >= 8 and divisible
   *   by 8. Construction rejects unsupported initial dimensions; an unsupported
   *   resize records a durable selected-mode failure without publishing a
   *   mismatched neural generation. Neural remains opt-in. See
   *   tools/neural-denoiser-training/README.md.
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
  readonly denoiser?: ValidDenoiser;

  /**
   * Pre-loaded model weights for the neural denoiser (T2.H2).
   * Required when `denoiser === 'neural'`. Load via `loadWeightsFromArrayBuffer()`
   * from the vitrum binary format exported by `tools/neural-denoiser-training/export_weights.py`.
   *
   * If `denoiser === 'neural'` and `neuralWeights` is undefined, the engine
   * constructor throws with a helpful error pointing to the training README.
   * The graph accepts every positive resolved internal width and height by
   * padding its private U-Net lattice and cropping the output; query
   * `capabilities.supportDetails.denoiserSpatialShapeRequirements.neural` for
   * the machine-readable requirement.
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
   *   - `'walkaround-hybrid'.resolveEnvironmentMap` — optional host callback
   *     for resolving opaque HDRI handles into walkaround-hybrid sky data. The
   *     callback may return diffuse sky scalars, or a CPU-readable `rawHdri`
   *     equirect payload that walkaround converts into its directional IBL map
   *     and importance CDFs. The core `EnvironmentMapRef` remains opaque; this
   *     hook is the typed escape hatch for hosts that already have CPU-side HDRI
   *     pixels or a precomputed average for that handle.
   */
  readonly extensions?: Readonly<Record<string, unknown>> & {
    readonly 'walkaround-hybrid'?: {
      readonly oidnModelUrl?: string;
      readonly oidnExecutionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
      /**
       * Neural input/activation/output storage. `'auto'` (default) uses f16
       * only for a certified checkpoint on a device with enabled shader-f16.
       * Explicit `'f16'` fails during construction if either condition is
       * missing; `'f32'` never silently upgrades.
       */
      readonly neuralTensorStorage?: 'auto' | 'f32' | 'f16';
      readonly bvhMode?: 'merged' | 'tlas';
      readonly resolveEnvironmentMap?: HybridEnvironmentMapResolver;
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
   *   allowed, requires `HYBRID_WEBGPU_REQUIRED_LIMITS`.
   * - `'lite'` — a reduced-work/memory path with the same structural device
   *   limits as full because it compiles the SAME explicit layouts (no WGSL or
   *   bind-group-layout fork), but:
   *     - forces `extensions['walkaround-hybrid'].bvhMode = 'merged'` (skips
   *       TLAS traversal while retaining dummy-compatible layout bindings) even
   *       when a needs-TLAS scene would otherwise default to TLAS, with a warning
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

  // ── Audit tuning knobs (nested `tuning` namespace) ───────────────────────

  /**
   * Audit-driven per-frame tuning knobs (Theme-H nested namespace, 2026-05-30).
   *
   * Each field is a fine-grained, scene-scale / framerate / light-intensity-
   * sensitive constant that was once a hardcoded magic number in the ReSTIR /
   * GTAO / caustic / atrous WGSL and is now host-overridable. The canonical
   * type + JSDoc + Cornell-baseline default for every knob lives on
   * {@link Tunables} (`HybridEngineTuning.ts`), the single source of truth;
   * this nested `Partial<Tunables>` is the host override path.
   *
   * Resolution (`readTunables`): for the three knobs that ALSO have a
   * dedicated subsystem sub-object — `caustic.{boost,visClamp}`,
   * `gtao.{radiusPx,intensity,depthThresholdWorldUnits,bilateralDepthSigma}`,
   * and the `adaptiveSamplingThresholds` tuple — the subsystem sub-object wins
   * over `tuning`, which wins over the {@link Tunables} default. For every
   * other knob, `tuning` wins over the default. Omitting the field (or the
   * whole `tuning` object) preserves Cornell behaviour byte-for-byte.
   *
   * @example
   *   tuning: { directFireflyClamp: 8, restirGiWCap: 32, triIntersectEpsilon: 1e-7 }
   */
  readonly tuning?: Partial<Tunables>;

  /**
   * Stained-glass caustic boost (audit B1).  Multiplies the through-glass
   * sun-shadow-ray contribution.  Cornell's stained-glass test scene uses
   * `{ boost: 22, visClamp: 0.6 }` to compensate for Brown-Beer-Lambert
   * attenuation; generic scenes should leave this at defaults (no boost,
   * no clamp).
   *
   * `caustic.boost` / `caustic.visClamp` only affect the explicitly selected
   * `causticStrategy:'refractive-trace'` estimator when
   * {@link stainedGlass}`.sunCaustic` is also `true`. They never enable a
   * caustic strategy by themselves.
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
   * The sky-aperture helper remains an opt-in WGSL term. `sunCaustic` now only
   * enables the historical stained-glass boost/clamp calibration inside the
   * separately selected `refractive-trace` estimator; it does not activate
   * caustics while `causticStrategy` is `'none'`.
   *
   * **Default both `false`** → generic scenes get ZERO caustic / aperture
   * physics. Stained-glass hosts (e.g. the Cornell-stained-glass example) opt
   * in with `{ sunCaustic: true, skyAperture: true }`.
   *
   * - `sunCaustic` — enable stained-glass boost/visClamp calibration for an
   *   active `refractive-trace` strategy.
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

  // ── Generalized reuse compatibility options ───────────────────────────────

  /**
   * @deprecated Generalized reconnection-shift reuse is now the sole
   * production ReSTIR-GI path and is always enabled. Omit this option.
   *
   * The estimator is deliberately narrower than ReSTIR PT: the reused sample is one
   * cosine-sampled direction whose suffix radiance comes from DDGI (or the
   * environment), and the receiver target is geometric diffuse
   * `luminance(Lo) * cos(theta) / PI`. Receiver material response is applied
   * later by shading and is not part of the reused target. PPG proposals and
   * NRC suffix substitution compose with this target using their exact source
   * mixture PDF and the same generalized reuse matrix.
   *
   * The enabled path uses a reconnection shift for surface samples, an
   * identity-direction shift for environment samples, exact shift Jacobians,
   * transformed-density generalized-balance weights, and visibility-aware
   * inverse-shift support. A history epoch prevents reservoirs from crossing
   * scene, material, emitter, environment, or lighting mutations.
   *
   * This option does not promise an unbiased path-tracing estimator. DDGI is a
   * cached irradiance approximation and the default finite
   * `restirGiIrrClamp` / `restirGiWCap` controls intentionally bound outliers.
   * `true` is accepted only for source compatibility and emits a deprecation
   * warning. `false` is rejected because it would request the retired biased
   * compact reuse implementation.
   *
   * @default true
   */
  readonly grisReuse?: boolean;

  /**
   * @deprecated Historical alias for {@link grisReuse}. Generalized reuse is
   * always enabled; omit both names. `true` is accepted with a warning and
   * `false` is rejected.
   */
  readonly restirPtReuse?: boolean;

  // ── Checkerboard half-res shading ─────────────────────────────────────────

  /**
   * Enable half-res CHECKERBOARD shading + temporal-resolve reconstruction.
   *
   * When `true`, the THREE expensive per-pixel passes — the shade pass and BOTH
   * ReSTIR-DI spatial-reuse passes — COMPACT their compute dispatch to one
   * checkerboard phase per frame (the active half, `(px+py)&1 == frameCount&1`),
   * genuinely skipping the gap-parity work rather than early-returning it (an
   * early-return still occupies the warp and saves nothing). The two spatial
   * passes are the pipeline's dominant cost — each thread casts 6 BVH rays
   * (centre + 5 Poisson neighbours), ×2 passes — so compacting them is where the
   * frame-time win comes from; `resolve.wgsl` reconstructs the gap pixels by
   * reprojecting the previous frame's radiance through the motion-vector
   * G-buffer. The spatial passes refine only the active-parity reservoir slots
   * shade then reads (same parity), so no gap-pixel reservoir passthrough is
   * needed.
   *
   * The ReSTIR-DI `ris` initial-candidate pass ALSO compacts (it carries forward
   * the gap-parity reservoir), so the full compacted set is FOUR passes — shade +
   * the two spatial passes + ris — which is where the whole-frame win comes from
   * (the BVH re-cast passes dominate).
   *
   * Default: `false` — OFF shades/refines EVERY pixel and the resolve pass passes
   * through, so the render is BIT-IDENTICAL to the pre-checkerboard pipeline
   * (the OFF-is-bit-identical opt-in pattern shared by `rcEnabled` /
   * `ppgEnabled` / `nrcEnabled` / `regir`). The flag flips
   * a few already-present UBO fields + the dispatch compaction + the ResolvePass
   * gate — it adds no bind groups, so it is NOT a compile-time structural decision.
   * The bare engine default (no quality preset ⇒ `ultra`) leaves this OFF; the
   * `medium`/`low` presets enable it (see below).
   *
   * Motion fallback: above {@link checkerboardMotionThresholdSq} of per-frame
   * camera motion the sparse path is forced FULL-RATE for that frame (shade +
   * spatial + ris + resolve all bit-identical to OFF), so a fast pan never
   * exposes the half-rate reservoir lag. Checkerboard's win is realised at static
   * / slow motion, where the reconstruction is faithful.
   *
   * VALIDATED + PROMOTED (objective whole-frame A/B, dzn RTX-4090). PERF (768px
   * Cornell, interleaved-paired, both swap orders agree): per-pass spatial-1
   * 1.87×, spatial-2 1.88×, ris 1.90×, shade 1.66×; WHOLE FRAME 1.46× (≈31%
   * GPU-time saved) — the win grows with resolution. QUALITY (384px motion A/B):
   * static/converged identical (64.34 dB); sustained motion ≈ full-rate; the
   * motion-ONSET transient worst-frame is 43.6 dB (sub-perceptible 0.00101 luma
   * gap) and recovers in 2-3 frames; fast motion forces full-rate (bit-identical).
   * On this evidence checkerboard is now ENABLED in the `medium` + `low` quality
   * presets (degradation tiers) and OFF in `ultra` + `high` (fidelity tiers); a
   * host overrides either way with this flag. See
   * `HybridEngineQualityPreset.ts` (`CHECKERBOARD_MEASURED_PERF_PROOF`). Harnesses
   * (wsl-gpu): checkerboard-ris-perf-ab.ts, checkerboard-spatial-perf-ab.ts,
   * checkerboard-motion-ab.ts, checkerboard-ris-isolate-ab.ts.
   */
  readonly checkerboardRendering?: boolean;

  /**
   * Camera squared-distance threshold (**world-space units²**) above which the
   * checkerboard sparse path (shade + the two ReSTIR-DI spatial passes) is
   * forced FULL-RATE for that frame. Only consulted when
   * {@link checkerboardRendering} is on.
   *
   * Checkerboard reuses last-frame reservoirs/radiance for the gap-parity half;
   * under perceptible camera motion that lag would show as softening, so when
   * the per-frame camera move exceeds `sqrt(threshold)` units the pipeline
   * shades every pixel that frame (and the resolve pass passes through), exactly
   * as if checkerboard were off — then resumes sparse shading once the camera
   * slows. The render stays a strict subset of the full-rate output.
   *
   * This is DELIBERATELY a separate, finer threshold than
   * {@link cameraMoveResetThresholdSq} (the temporal-accumulator reset): the
   * checkerboard reservoir lag becomes visible at MUCH smaller motion than a
   * full history discard, so the checkerboard fallback must trip earlier.
   *
   * **Scene-scale-sensitive** (same scaling rule as `cameraMoveResetThresholdSq`,
   * but much smaller). Default `0.004` (= `0.063²`) is tuned to Cornell's
   * ~2-unit room and GPU-validated on the dzn motion A/B: checkerboard stays
   * sparse through a SLOW drag (~0.04 units/frame, where the reconstruction held
   * ~50 dB vs full-rate) and flips to full-rate by a faster pan (~0.07
   * units/frame, where it had fallen to ~29 dB). Hosts on a different scene
   * scale should override (recommended `(sceneDiagonal × 3e-4)²`).
   *
   * @default 0.004
   */
  readonly checkerboardMotionThresholdSq?: number;

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
   * Default: 1 024 — large enough for meaningful spatial refinement while
   * keeping persistent PPG buffers bounded at ~12.2 MiB. The absolute ceiling is 16 384; raise this
   * only for dense, complex scenes where 1 024 cells are insufficient for
   * guided sampling coverage.
   *
   * This cap is **active**: `PPGCoordinator` calls `splitOverflowLeaves` with
   * this value as the hard ceiling, so the CPU sTree and the GPU flux/dTree
   * buffers always stay in lockstep. Raising this value allocates larger GPU
   * buffers; lowering it suppresses splits earlier, saving VRAM at the cost of
   * coarser spatial guidance.
   *
   * Requires `ppgEnabled: true`; construction rejects it otherwise.
   */
  readonly ppgMaxSpatialCells?: number;

  /**
   * Maximum number of adaptive dTree nodes allocated per spatial cell.
   *
   * Default: 341 — a full depth-4 directional quadtree
   * (1 + 4 + 16 + 64 + 256 nodes), matching `PPG_DTREE_MAX_DEPTH = 4`.
   *
   * This cap is **structural**: it sizes the dTree storage buffer and flux
   * atomics, templates the PPG update shader's per-cell stride, clamps CPU
   * tree uploads, and is recorded in GI snapshots so incompatible restores are
   * rejected. Lowering it saves VRAM and update/readback bandwidth but limits
   * directional refinement; raising it only helps once the CPU dTree max depth
   * is raised too.
   *
   * Requires `ppgEnabled: true`; construction rejects it otherwise.
   */
  readonly ppgMaxDTreeNodesPerCell?: number;

  /**
   * Practical Path Guiding MIS mixture weight alpha.
   *
   * The GI RIS source pdf is
   * `p_src = alpha * p_guide + (1 - alpha) * p_cosine`. Alpha must be finite
   * and strictly between `0` and `1`, preserving positive support from both
   * proposal components; invalid endpoints are rejected.
   *
   * Default: 0.5 (Muller 2017 section 3.4).
   *
   * Requires `ppgEnabled: true`; construction rejects it otherwise.
   */
  readonly ppgMixAlpha?: number;

  /**
   * PPG train-pass dispatch cadence (Müller 2017 §3.3). The path-guiding
   * `update` compute pass runs only on frames where
   * `frameCount % ppgDispatchInterval === 0`. The learned sTree/dTree GPU
   * buffers PERSIST between train cycles and the gi-ris guided SAMPLING reads
   * them EVERY frame, so a higher interval is a pure training-cost lever — it
   * never changes whether guided sampling is active, only how often the tree
   * is retrained. `1` trains every frame (no behaviour change); `N > 1` trains
   * every Nth frame. Values must be positive safe integers. Requires
   * `ppgEnabled: true`; construction rejects it otherwise.
   *
   * Default: resolved from `qualityTier` (`resolveQualityPreset`) —
   * ultra/high = 1, medium = 2, low = 4. An explicit value here OVERRIDES the
   * preset.
   */
  readonly ppgDispatchInterval?: number;

  // ── ReGIR (Boksansky, Wyman, Benty 2021 — grid-based reservoirs) ──────────

  /**
   * Enable ReGIR (Reservoir-based Grid Importance Resampling) for ReSTIR-DI
   * initial-candidate light SELECTION.
   *
   * ReGIR scales ReSTIR-DI to MANY lights by decoupling the per-pixel light-
   * selection cost from the light count. Each frame a grid-build compute pass
   * fills a world-space grid of light reservoirs — each cell pre-resamples
   * lights by power × proximity via WRS, SEEDED BY THE LIGHT TREE at the cell
   * centroid. RIS then draws its initial candidates from the cell containing
   * the shading point (O(1) per pixel) instead of traversing the light tree
   * per pixel × M candidates.
   *
   * Unbiased: RIS divides the target p̂ by the EXACT per-cell selection pmf the
   * grid stored (`q̂_c(e)/Ŝ`), the same discipline as the light-tree path. When
   * `enabled: false` (or omitted), RIS uses the light-tree path bit-identically.
   *
   * The grid is co-located in the SAME storage buffer as the light tree (so RIS
   * stays within its storage-buffer budget) and only goes live when the light
   * tree is live (≥ 2 emitters). Lower-light scenes gain nothing and stay on the
   * tree path automatically.
   *
   * Default: `undefined` ⇒ ReGIR off.
   *
   * @see Boksansky, Wyman, Benty 2021, "Rendering Many Lights with Grid-Based
   *      Reservoirs", Ray Tracing Gems II ch. 23.
   */
  readonly regir?: {
    /** Master gate. `false`/omitted ⇒ ReGIR off (light-tree fallback). */
    readonly enabled?: boolean;
    /** Cells per axis (cubic grid). Default 16 ⇒ up to 4096 cells. */
    readonly cellsPerAxis?: number;
    /** M — WRS candidates drawn per cell sub-reservoir at grid build. Default 32. */
    readonly candidatesPerCell?: number;
    /** K — survivors stored per cell (per-pixel candidate diversity). Default 8. */
    readonly survivorsPerCell?: number;
  };

  // ── RC (Sannikov 2023 Radiance Cascades — W8 sprint, 2026-05-18) ──────────

  /**
   * Enable the Sannikov 2023 Radiance Cascades subsystem inside HybridEngine.
   *
   * When `true`, the engine instantiates an {@link RCSubsystem} that:
   *   - Borrows the canonical hybrid scene-arena BLAS ranges. RC retains only
   *     a compact material/TLAS adapter, so enabling it does not duplicate the
   *     scene's large node/index/position/normal allocations.
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
   * Default: `false`; hosts opt into the additional cascade memory and compute
   * cost when its multiscale indirect signal is useful for their scene.
   *
   * @see plan/w8-rc-mis-composition.md for the full sprint plan.
   */
  readonly rcEnabled?: boolean;

  /**
   * Maximum number of dielectric interfaces one Radiance Cascades probe ray
   * may cross before the transmitted path is terminated. Thin sheets consume
   * two interfaces (entry plus their reciprocal virtual exit), while bulk
   * dielectric boundaries consume one each.
   *
   * Must be an integer in [1, 8]. The default of 8 matches the shader's
   * statically bounded medium stack and preserves the standalone RC default.
   * Requires {@link rcEnabled} to be `true`; construction rejects it otherwise.
   *
   * @default 8
   */
  readonly rcTransmittedInterfaceBudget?: number;

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
   * Requires `rcEnabled: true`; construction rejects it otherwise.
   *
   * @default 0.5 (when rcEnabled === true)
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
   * Requires `rcEnabled: true`; construction rejects it otherwise.
   *
   * @default CASCADE_DIMS from @vitrum/walkaround-rc
   */
  readonly cascadeDims?: readonly CascadeDim[];

  // ── NRC (Müller, Rousselle, Novák, Keller 2021 — Real-time Neural
  //    Radiance Caching for Path Tracing) ───────────────────────────────────

  /**
   * Enable the Müller et al. 2021 Neural Radiance Caching subsystem.
   *
   * The "full" version: a small fused MLP whose input is a multiresolution
   * **hash-grid positional encoding** of the cache-query vertex (Instant-NGP,
   * Müller et al. 2022) + a **one-blob** direction encoding (Müller et al. 2019)
   * + raw surface features (normal/roughness/albedo). When live, the GI suffix
   * is TERMINATED into the cache once Müller's **path-spread heuristic** fires
   * (a(x) > c·a₀), and the MLP's predicted outgoing radiance becomes the suffix
   * contribution; radiance records gathered along paths self-train the cache
   * (hash-grid feature tables + MLP weights are trained JOINTLY) once per frame.
   *
   * **NRC is a BIASED estimator** — the cache is a learned approximation of the
   * path suffix, not an unbiased Monte-Carlo estimate. Unlike the explicit RC/PPG paths (which
   * preserve the converged mean), the acceptance criterion for NRC is
   * *perceptual closeness to the no-NRC reference within tolerance, with faster
   * convergence / lower noise*. See `HARDWARE-VALIDATION-NEEDS.md` V20.
   *
   * Default: `false` — NRC is opt-in. **OFF is BIT-IDENTICAL** to the current GI:
   * the pipeline compiles/runs the non-NRC `gi-ris` variant, while the UBO bit in
   * the former `_ppgPad2` slot remains an informational mirror for diagnostics.
   * FORBIDDEN on `tier:'lite'` (the hash-grid feature tables + MLP weight/Adam
   * buffers exceed the lite resource budget) — the constructor throws if
   * `nrcEnabled` is set with `tier:'lite'`, mirroring rcEnabled / ppgEnabled /
   * denoiser:'neural'.
   *
   * When ON, the gi-ris pass compiles in its NRC variant (`risGiNrc`). Once
   * Müller's spread heuristic fires at an opaque suffix vertex, the MLP query
   * replaces the DDGI estimate. A private four-vertex Monte Carlo suffix with
   * finite/analytic/sun NEE, environment continuation, and Russian roulette
   * supplies the matched online training label; it reads neither DDGI nor the
   * cache prediction. `NrcSubsystem` trains the hash grid and MLP once per frame.
   * NRC remains a biased learned approximation, and its independent teacher
   * estimates a deliberately bounded rather than infinite path suffix. See
   * `HARDWARE-VALIDATION-NEEDS.md` V20 for GPU acceptance evidence.
   *
   * @default false
   */
  readonly nrcEnabled?: boolean;

  /**
   * Complete Neural Radiance Cache configuration.
   *
   * Every field is resolved against {@link DEFAULT_NRC_CONFIG} and drives the
   * matching executable path: hash-grid dimensions, MLP shape, spread gate,
   * record capacity, both Adam learning rates, trainer precision/tile size,
   * warmup, and the optional aggregate-residency policy. The same resolved
   * object is used for device preflight, GPU allocation, query-shader
   * compilation, and both trainers, so those layers cannot silently disagree.
   *
   * `useF16: true` requires a device with the `shader-f16` feature enabled.
   * Facade-owned device creation requests that feature automatically; hosts
   * supplying their own device must enable it when requesting the device.
   *
   * The legacy top-level `nrcWarmupSteps`, `nrcSpreadC`, and
   * `nrcMaxResidentBytes` aliases remain accepted. Supplying an alias together
   * with the corresponding `nrcConfig` field is allowed only when the values
   * agree exactly; disagreement throws synchronously.
   *
   * Requires `nrcEnabled: true`; construction rejects it otherwise.
   */
  readonly nrcConfig?: Partial<NrcConfig>;

  /**
   * Completed NRC trainer windows required before cache predictions may replace
   * the DDGI suffix.
   *
   * NRC gathers training records immediately, but the GI shader keeps using the
   * DDGI suffix until `trainedSteps >= nrcWarmupSteps`. Lower values promote the
   * biased cache earlier; higher values keep the explicit DDGI suffix longer
   * while the cache settles. Values must be safe integers `>= 0`; invalid
   * values are rejected.
   *
   * Default: 8.
   *
   * Requires `nrcEnabled: true`; construction rejects it otherwise.
   *
   * @deprecated Use `nrcConfig: { warmupSteps }`.
   */
  readonly nrcWarmupSteps?: number;

  /**
   * Müller et al. spread-termination constant `c` for the NRC suffix cache.
   *
   * Smaller values let the biased cache replace the DDGI suffix earlier along a
   * path; larger values keep more of the explicit DDGI suffix before querying the
   * cache. Values must be finite and `>= 0`; invalid values are rejected.
   *
   * Default: 0.01.
   *
   * Requires `nrcEnabled: true`; construction rejects it otherwise.
   *
   * @deprecated Use `nrcConfig: { spreadC }`.
   */
  readonly nrcSpreadC?: number;

  /**
   * Optional host policy for the NRC subsystem's peak resident GPU-buffer bytes,
   * including its one permitted readback buffer. The default leaves aggregate
   * residency uncapped because WebGPU reports no adapter-wide VRAM budget; all
   * real per-buffer and binding limits remain enforced independently.
   *
   * Requires `nrcEnabled: true`; construction rejects it otherwise. Must be a
   * positive safe integer.
   *
   * @deprecated Use `nrcConfig: { maxNrcResidentBytes }`.
   */
  readonly nrcMaxResidentBytes?: number;
}
