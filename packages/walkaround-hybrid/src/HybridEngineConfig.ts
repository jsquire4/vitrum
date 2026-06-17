/**
 * HybridEngineConfig — construction-time option parsing for {@link HybridEngine}.
 *
 * Extracted from `HybridEngine.ts` (R3 B-chain decomposition sweep).
 *
 * Contains:
 *   - {@link ParsedHybridEngineConfig} — the immutable derived-config record type.
 *   - {@link validateHybridEngineOptions} — pure throws (lite/neural/OIDN/denoiser guards).
 *   - {@link deriveHybridEngineConfig} — pure defaulting into the derived record.
 *   - {@link parseHybridEngineOptions} — thin orchestrator over the two halves.
 *
 * No `this` dependency, no GPU side effects. All behaviour-preserving.
 */

import type { HybridEngineOptions } from './HybridEngineOptions.js';
import { VALID_DENOISERS } from './HybridEngineOptions.js';
import { ATROUS_DIRECT_SIGMAS, ATROUS_INDIRECT_SIGMAS } from './pipeline/constants.js';
import { packStainedGlassFlags } from './pipeline/uboUpdater.js';
import { readTunables, readInitTunables, type Tunables, type InitTunables } from './HybridEngineTuning.js';
import { resolveQualityPreset } from './HybridEngineQualityPreset.js';
import { fingerprintHybridPipelineRebuildKey } from './HybridEngineFrameOrchestrator.js';
import type { ReSTIRBvhMode } from './restir/bvhCore.js';
import type { ModelWeights } from './neural/weights.js';
import { PPG_MIS_ALPHA } from './ppg/ppgConstants.js';

/** Default per-frame target interval (~60 FPS soft-cap). */
const DEFAULT_TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

/**
 * The construction-time-immutable config the engine derives PURELY from its
 * options — no `this` dependency, no GPU side effects. Extracting the ~80 LOC
 * of defaulting + validation that produced these out of the constructor (WD
 * decomposition sweep) keeps the constructor focused on object wiring (DDGI /
 * RC subsystem creation, capabilities, init coordinator, debug surface) that
 * genuinely needs `this`.
 *
 * Behaviour-preserving: `parseHybridEngineOptions` throws the same three
 * `TypeError`s in the same order as the inline constructor did, and applies
 * the same defaults. The constructor assigns each field verbatim from the
 * returned record.
 */
export interface ParsedHybridEngineConfig {
  readonly denoiser: 'none' | 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final';
  readonly neuralWeights: ModelWeights | undefined;
  readonly oidnModelUrl: string | undefined;
  readonly oidnExecutionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly restirBvhModeOverride: ReSTIRBvhMode | undefined;
  readonly targetFrameIntervalMs: number | null;
  readonly tunables: Tunables;
  readonly initTunables: InitTunables;
  readonly indirectFireflyClamp: readonly [number, number, number];
  readonly atrousDirectSigmas: readonly [number, number, number];
  readonly atrousIndirectSigmas: readonly [number, number, number];
  readonly stainedGlassFlags: number;
  /** GRIS / ReSTIR-PT reconnection-shift reuse gate (0 = off / legacy reuse,
   *  1 = unbiased GRIS shift + visibility + pairwise MIS). The STRUCTURE is
   *  COMPILE-TIME (the boolean selects the GI pipeline layout + shader variant
   *  at init); this number is also threaded into the per-frame UBO. */
  readonly restirPtReuse: number;
  /** NRC (Müller et al. 2021) cache flag mirrored into the per-frame UBO
   *  (0 = off / verbatim DDGI suffix, 1 = on). The load-bearing gate is
   *  COMPILE-TIME: `nrcEnabled` selects the `risGiNrc` GI shader variant at
   *  engine creation (a UBO flag alone can't add the @group(4) NRC bindings).
   *  When ON, the suffix cache-query + per-frame training passes are live.
   *  FORBIDDEN on tier:'lite'. */
  readonly nrcEnabled: number;
  /** PPG (Müller 2017) guided-sampling flag (0 = off, 1 = on). COMPILE-TIME
   *  at the pipeline level: `ppgEnabled` builds the ppg-update pipeline and
   *  drives the UBO gate; OFF is bit-identical to the cosine kernel.
   *  FORBIDDEN on tier:'lite'. (G-P1.1 follow-up: opts.ppgEnabled used to be
   *  read only by the lite-tier guard and never forwarded to the pipeline —
   *  PPG was inert through the public API.) */
  readonly ppgEnabled: number;
  /** H47 — maximum PPG sTree spatial cells, threaded to `allocatePPGResources`.
   *  `undefined` ⇒ use allocatePPGResources default (1 024). */
  readonly ppgMaxSpatialCells: number | undefined;
  /** H29 — maximum per-cell PPG dTree nodes, threaded to shader compile and
   *  `allocatePPGResources`. `undefined` ⇒ use the default 341-node stride. */
  readonly ppgMaxDTreeNodesPerCell: number | undefined;
  /** PPG guide/cosine MIS mixture alpha, clamped to [0,1]. */
  readonly ppgMixAlpha: number;
  /** Checkerboard half-res shading (HybridEngineOptions.checkerboardRendering).
   *  `false` by default (no preset ⇒ ultra ⇒ off); the `medium`/`low` presets
   *  enable it, `ultra`/`high` keep it off. Threaded into
   *  `pipeline.initialize({ checkerboard, checkerboardMotionThresholdSq })`; OFF
   *  is bit-identical to the pre-checkerboard pipeline (shade + both spatial
   *  passes + ris dispatch full-res and ResolvePass passes through).
   *  GPU-validated on dzn — see WalkaroundGPUPipeline `_checkerboard`. */
  readonly checkerboard: boolean;
  readonly staticPipelineRebuildKey: string | number | null;
  readonly getPipelineRebuildKey: (() => string | number | null | undefined) | undefined;
  readonly rebuildKeyFingerprintSeen: string;
  readonly maxBounces: number;
  readonly verbose: boolean;
  readonly debug: boolean;
  // ── Phase-0 productization — quality-preset-resolved knobs ───────────────
  /** Resolved GTAO dispatch mode (preset, overridden by `opts.gtaoMode`). */
  readonly gtaoMode: 'on' | 'quarter' | 'off';
  /** Resolved ReSTIR-DI spatial pass count (preset, overridden by opts). */
  readonly diSpatialPasses: 1 | 2;
  /** Resolved ReSTIR-GI spatial pass count (preset, overridden by opts). */
  readonly giSpatialPasses: 1 | 2;
  /** Resolved DDGI round-robin probe-update divisor (preset, overridden by opts). */
  readonly ddgiUpdateDivisor: number;
  /** Resolved PPG train-pass dispatch cadence (preset, overridden by opts).
   *  Threaded into `pipeline.initialize` so the ppg-update pass gates on
   *  `frameCount % ppgDispatchInterval`. Always >= 1. */
  readonly ppgDispatchInterval: number;
  /** ReGIR (Boksansky 2021) grid-based DI light-selection config (pass-through
   *  from opts; `undefined` ⇒ off). Threaded into `pipeline.initialize`. */
  readonly regirConfig: Partial<import('./pipeline/ReGIRCoordinator.js').ReGIRConfig> | undefined;
  /** Resolved initial internal-resolution factor (preset; per-frame
   *  `quality.resolutionFactor` still overrides at runtime). */
  readonly resolutionFactor: number;
}

/**
 * The extracted `extensions['walkaround-hybrid']` sub-object shape — read by
 * both {@link validateHybridEngineOptions} (oidnModelUrl presence) and
 * {@link deriveHybridEngineConfig} (oidnModelUrl / providers / bvhMode).
 */
type WalkaroundHybridExt = {
  oidnModelUrl?: string;
  oidnExecutionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  bvhMode?: 'merged' | 'tlas';
};

function resolvePpgMixAlpha(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return PPG_MIS_ALPHA;
  return Math.min(1, Math.max(0, value));
}

function readWalkaroundHybridExt(opts: HybridEngineOptions): WalkaroundHybridExt | undefined {
  return (opts.extensions as undefined | {
    'walkaround-hybrid'?: WalkaroundHybridExt;
  })?.['walkaround-hybrid'];
}

/**
 * Pure construction-time validation of `HybridEngineOptions` — throws the
 * three (well, six) `TypeError`s the constructor relies on, in the exact same
 * order as the pre-Theme-H inline path. No defaulting, no derived config, no
 * `this`, no GPU side effects: this is the independently-testable "does this
 * option object describe a buildable engine?" gate.
 *
 * Throw order (load-bearing — tests pin it):
 *   1. tier:'lite' forbids rcEnabled / ppgEnabled / denoiser:'neural' /
 *      nrcEnabled (lite validated FIRST so it is the host's first signal);
 *   2. unsupported denoiser enum;
 *   3. denoiser:'neural' without neuralWeights;
 *   4. denoiser:'oidn-final' without extensions['walkaround-hybrid'].oidnModelUrl.
 */
export function validateHybridEngineOptions(opts: HybridEngineOptions): void {
  // Phase-0 productization — hybrid LITE tier (Deliverable 3). Lite runs the
  // same pipeline but on a reduced resource budget: it forbids the
  // resource-heavy optional subsystems and forces the merged-BVH path (drops
  // the 5 TLAS scene-group buffers). Validated FIRST so the throws are the
  // host's first signal.
  if (opts.tier === 'lite') {
    if (opts.rcEnabled === true) {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids rcEnabled — Radiance Cascades ` +
        `allocate 5 extra cascade GPUBuffers + a separate BVH that the lite ` +
        `resource budget cannot fit. Use tier:'full' (needs a 16-buffer / ` +
        `8-texture adapter) for RC.`,
      );
    }
    if (opts.ppgEnabled === true) {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids ppgEnabled — Practical Path ` +
        `Guiding allocates an sTree/dTree GPU buffer set the lite budget ` +
        `cannot fit. Use tier:'full' for PPG.`,
      );
    }
    if (opts.denoiser === 'neural') {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids denoiser:'neural' — the U-Net ` +
        `InferenceGraph + weight buffers exceed the lite budget. Use ` +
        `'atrous-variance' / 'atrous' on lite, or tier:'full' for neural.`,
      );
    }
    if (opts.nrcEnabled === true) {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids nrcEnabled — Neural Radiance ` +
        `Caching allocates a multiresolution hash-grid feature-table set + the ` +
        `fused-MLP weight/Adam buffers the lite budget cannot fit. Use ` +
        `tier:'full' for NRC.`,
      );
    }
  }

  // Audit B7: validate the denoiser option at construction so an unsupported
  // value does not silently coerce to atrous-variance and produce wrong output.
  // Supported values are enumerated in VALID_DENOISERS (single source of truth
  // in HybridEngineOptions.ts). `'none'` is the pass-through denoiser, and
  // `'bmfr'` is a real denoiser (Koskela 2019 — see denoisers/bmfr.ts).
  if (
    opts.denoiser !== undefined &&
    !(VALID_DENOISERS as ReadonlyArray<string>).includes(opts.denoiser)
  ) {
    throw new TypeError(
      `[HybridEngine] unsupported denoiser '${String(opts.denoiser)}'. ` +
      `walkaround-hybrid supports: 'none' | 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final'.`,
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
  // W11 — 'oidn-final' requires extensions['walkaround-hybrid'].oidnModelUrl.
  const oidnModelUrl = readWalkaroundHybridExt(opts)?.oidnModelUrl;
  if (opts.denoiser === 'oidn-final' &&
      (typeof oidnModelUrl !== 'string' || oidnModelUrl.length === 0)) {
    throw new TypeError(
      `[HybridEngine] denoiser: 'oidn-final' requires ` +
      `extensions['walkaround-hybrid'].oidnModelUrl (non-empty string) ` +
      `pointing at the bundled OIDN ONNX model file ` +
      `(e.g. '/models/oidn_rt_hdr_alb_nrm.onnx'). ` +
      `See plan/premium-grade-refactor-20260517.md §W11 + ` +
      `packages/shared-denoisers/src/oidnBridge.ts for the model-URL convention.`,
    );
  }
}

/**
 * Pure defaulting of `HybridEngineOptions` into the immutable derived config,
 * given an already-resolved quality `preset`. ASSUMES the options have already
 * passed {@link validateHybridEngineOptions} (it does not re-throw the
 * validation `TypeError`s). Behaviour-preserving: every field is defaulted
 * exactly as the pre-Theme-H inline path produced it.
 *
 * @param preset resolved {@link resolveQualityPreset} output for the engine's
 *   effective quality tier (the caller resolves the tier so the lite-biased
 *   `'medium'` default + explicit `qualityTier` override live in one place).
 */
export function deriveHybridEngineConfig(
  opts: HybridEngineOptions,
  preset: ReturnType<typeof resolveQualityPreset>,
): ParsedHybridEngineConfig {
  const isLite = opts.tier === 'lite';
  // Effective options overlay: the preset supplies fallbacks for the knobs it
  // governs, so the existing table-driven `readTunables` / denoiser /
  // targetFrameInterval logic picks them up unchanged. Explicit opts win.
  const effectiveOpts: HybridEngineOptions = {
    ...opts,
    ...(opts.adaptiveSamplingThresholds === undefined && preset.adaptiveSamplingThresholds !== undefined
      ? { adaptiveSamplingThresholds: preset.adaptiveSamplingThresholds }
      : {}),
  };

  const whExt = readWalkaroundHybridExt(opts);
  const oidnModelUrl = whExt?.oidnModelUrl;

  return {
    // Preset supplies the denoiser fallback (low ⇒ 'atrous'); explicit
    // opts.denoiser wins, then the engine default 'atrous-variance'.
    denoiser: opts.denoiser ?? preset.denoiser ?? 'atrous-variance',
    neuralWeights: opts.neuralWeights,
    oidnModelUrl,
    oidnExecutionProviders: whExt?.oidnExecutionProviders,
    // Lite forces merged BVH (drops the 5 TLAS scene-group buffers — the lite
    // buffer-axis win) regardless of any host bvhMode override; warn so the
    // host knows instanced-scene fidelity is reduced on this weak adapter.
    restirBvhModeOverride: isLite
      ? (whExt?.bvhMode === 'tlas'
          ? (console.warn(
              `[HybridEngine] tier:'lite' overrides bvhMode:'tlas' → 'merged' ` +
              `(TLAS scene buffers exceed the lite resource budget). Instanced/` +
              `multi-mesh scene fidelity is reduced. Use tier:'full' for TLAS.`,
            ), 'merged')
          : 'merged')
      : whExt?.bvhMode,
    // Precedence: explicit opts → preset → engine default (~60 FPS cap).
    // The preset never carries `null`, so it cannot accidentally disable the
    // cap; only an explicit `opts.targetFrameIntervalMs: null` does that.
    targetFrameIntervalMs: opts.targetFrameIntervalMs !== undefined
      ? opts.targetFrameIntervalMs
      : preset.targetFrameIntervalMs !== undefined
        ? preset.targetFrameIntervalMs
        : DEFAULT_TARGET_FRAME_INTERVAL_MS,
    // Library-generality tunables — table-driven; defaults preserve Cornell
    // behaviour, hosts override via HybridEngineOptions. `effectiveOpts`
    // carries the preset's adaptiveSamplingThresholds fallback.
    tunables: readTunables(effectiveOpts),
    initTunables: readInitTunables(opts),
    // 2026-05-18 sweep — `indirectFireflyClamp` is tuple-typed so it lives
    // outside the number-typed Tunables table; default preserves Cornell.
    indirectFireflyClamp: opts.indirectFireflyClamp ?? [1.0, 1.0, 1.0],
    // 2026-05-19 B3a — atrous DIRECT/INDIRECT sigmas; tuple-typed same as
    // indirectFireflyClamp. Defaults sourced from the single-source-of-truth
    // constants in bindGroupBuilders.ts (no duplicated literals).
    atrousDirectSigmas: opts.atrousDirectSigmas
      ?? [ATROUS_DIRECT_SIGMAS.sigmaN, ATROUS_DIRECT_SIGMAS.sigmaZ, ATROUS_DIRECT_SIGMAS.sigmaC],
    atrousIndirectSigmas: opts.atrousIndirectSigmas
      ?? [ATROUS_INDIRECT_SIGMAS.sigmaN, ATROUS_INDIRECT_SIGMAS.sigmaZ, ATROUS_INDIRECT_SIGMAS.sigmaC],
    // T5 — stained-glass opt-in flag bits. Default 0 (both terms OFF); hosts
    // opt in via opts.stainedGlass. Packed once here (construction-time
    // config); threaded into pipeline.renderFrame via _denoiserFilterDeps.
    stainedGlassFlags: packStainedGlassFlags({
      sunCaustic: opts.stainedGlass?.sunCaustic,
      skyAperture: opts.stainedGlass?.skyAperture,
    }),
    // GRIS / ReSTIR-PT reconnection-shift reuse gate. Default 0 (OFF) so the
    // GI spatial/temporal reuse is bit-identical to the legacy clamped-Jacobian
    // path unless a host opts in via opts.restirPtReuse.
    restirPtReuse: opts.restirPtReuse === true ? 1 : 0,
    // NRC cache flag. Default 0 (OFF) so the gi-ris suffix is bit-identical to
    // the verbatim DDGI-atlas estimate unless a host opts in via opts.nrcEnabled
    // (which tier:'lite' forbids — validated above). The real gate is compile-time
    // (selects the risGiNrc variant); this value is mirrored into the UBO.
    nrcEnabled: opts.nrcEnabled === true ? 1 : 0,
    // PPG guided sampling. Default 0 (OFF) — bit-identical cosine kernel.
    // Forwarded to pipeline.initialize so the ppg-update pipeline is actually
    // built when a host opts in (tier:'lite' forbids it — validated above).
    ppgEnabled: opts.ppgEnabled === true ? 1 : 0,
    // PPG guide/cosine mixture weight. The default remains the paper value, but
    // hosts can tune/A-B favorable scenes without patching the coordinator.
    ppgMixAlpha: resolvePpgMixAlpha(opts.ppgMixAlpha),
    // Checkerboard half-res shading. Explicit opt wins, else the preset value
    // (ON for medium/low degradation tiers, OFF for ultra/high). No preset ⇒
    // ultra ⇒ OFF ⇒ shade + both spatial passes + ris shade every pixel +
    // ResolvePass passes through = bit-identical to the pre-checkerboard
    // pipeline. GPU-validated on dzn (whole-frame 1.46× at medium/low).
    checkerboard: opts.checkerboardRendering ?? preset.checkerboard,
    staticPipelineRebuildKey: opts.pipelineRebuildKey ?? null,
    getPipelineRebuildKey: opts.getPipelineRebuildKey,
    rebuildKeyFingerprintSeen: fingerprintHybridPipelineRebuildKey(
      opts.getPipelineRebuildKey?.() ?? opts.pipelineRebuildKey ?? null,
    ),
    maxBounces: opts.maxBounces ?? 4,
    verbose: opts.verbose ?? false,
    debug: opts.debug ?? false,
    // Phase-0 productization — quality-preset-resolved structural / gating
    // knobs. Explicit per-knob opts override the preset.
    gtaoMode: opts.gtaoMode ?? preset.gtaoMode,
    diSpatialPasses: opts.diSpatialPasses ?? preset.diSpatialPasses,
    giSpatialPasses: opts.giSpatialPasses ?? preset.giSpatialPasses,
    ddgiUpdateDivisor: opts.ddgiUpdateDivisor ?? preset.ddgiUpdateDivisor,
    // PPG train-pass cadence: explicit opt wins, else the preset value. Clamp
    // to ≥ 1 here too (the pipeline re-clamps, but keep the resolved config
    // honest so a debug surface reading it sees the effective value).
    ppgDispatchInterval: Math.max(
      1,
      Math.floor(opts.ppgDispatchInterval ?? preset.ppgDispatchInterval),
    ),
    // H47 — PPG max spatial cells. Pass-through; undefined = allocatePPGResources
    // default (1 024). No clamping here — the allocator handles its own floor.
    ppgMaxSpatialCells: opts.ppgMaxSpatialCells,
    // H29 — PPG max dTree nodes per spatial cell. Pass-through; undefined =
    // allocatePPGResources / buildPpgUpdateWgsl default (341). The pipeline
    // threads this to BOTH shader compile and resource allocation so the GPU
    // update kernel stride matches the buffers.
    ppgMaxDTreeNodesPerCell: opts.ppgMaxDTreeNodesPerCell,
    // ReGIR (Boksansky 2021) grid-based DI light selection. Pass-through from
    // opts; `undefined` ⇒ off (the pipeline's resolveReGIRConfig default).
    regirConfig: opts.regir,
    resolutionFactor: preset.resolutionFactor,
  };
}

/**
 * Parse + validate `HybridEngineOptions` into the immutable derived config.
 * Thin orchestrator over the two independently-testable halves:
 *   1. {@link validateHybridEngineOptions} — the pure throws (lite-mode
 *      violations, bad denoiser/neural/OIDN combos), in load-bearing order.
 *   2. {@link deriveHybridEngineConfig} — the 45-field defaulting record, given
 *      the already-resolved quality preset.
 *
 * The quality-tier resolution (`opts.qualityTier ?? (isLite ? 'medium' :
 * 'ultra')`) lives HERE — between the throws and the derive — so the
 * lite-biased default + explicit override exist in exactly one place. Pure
 * (no `this`, no GPU). Behaviour-preserving over the pre-Theme-H inline path:
 * same throws in the same order, same derived config. See
 * {@link ParsedHybridEngineConfig}.
 */
export function parseHybridEngineOptions(opts: HybridEngineOptions): ParsedHybridEngineConfig {
  validateHybridEngineOptions(opts);

  // Phase-0 productization — resolve the coarse quality preset, then let
  // explicit per-knob options OVERRIDE it inside `deriveHybridEngineConfig`
  // (preset is a baseline, not a lock). `ultra` (the default) is byte-identical
  // to the pre-Phase-0 defaults. Lite biases the default tier to `'medium'`
  // (still overridable by an explicit `qualityTier`).
  const effectiveQualityTier = opts.qualityTier ?? (opts.tier === 'lite' ? 'medium' : 'ultra');
  const preset = resolveQualityPreset(effectiveQualityTier);

  return deriveHybridEngineConfig(opts, preset);
}
