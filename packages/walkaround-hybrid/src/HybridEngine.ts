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
 * RC subsystem: cascade dispatch + shade-pass balance-heuristic MIS shipped
 * (W8 Phase 3). See plan/w8-rc-mis-composition.md.
 *
 * Debug globals:
 *   The original hook wrote to `window.__WGPU__.walkaround` and
 *   `window.__HYBRID_LAYERS__`. Those are host-bridge responsibilities.
 *   This class exposes `setLayerEnabled()` so the host can wire layer
 *   toggles; it calls `window.__WGPU__` only inside a debug branch
 *   guarded by `typeof window !== 'undefined'` and the `debug` option.
 *
 * Decomposition (refactor sweep 2026-05-18):
 *   - {@link PipelineInitCoordinator} (HybridEngineLifecycle.ts) owns the
 *     async pipeline-init race coordination + the multi-phase init chain.
 *   - {@link transformRefit} / {@link topologyRebuild}
 *     (HybridEnginePrimitiveUpdates.ts) implement the `updatePrimitive`
 *     fast / rebuild paths.
 *   - {@link TUNABLE_DEFINITIONS} (HybridEngineTuning.ts) is the single
 *     source of truth for audit-driven tuning knobs.
 *   - This file owns: public Engine API impl, construction-time options
 *     validation, debug surface, engine-state machine and reset
 *     coordination, scene synthesis + ownership, per-frame DDGI
 *     orchestration + frame throttle + telemetry mirror.
 */

import * as THREE from 'three';
import type {
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineFactory,
  EngineState,
  FrameStats,
  ProgressStats,
} from '@vitrum/core';
import type { Scene, ScenePrimitive, SceneEmitter, SceneEnvironment } from '@vitrum/core';
import { partitionSceneBySupport } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import { ATROUS_DIRECT_SIGMAS, ATROUS_INDIRECT_SIGMAS } from './pipeline/bindGroupBuilders.js';
import { packStainedGlassFlags } from './pipeline/uboUpdater.js';
import { createHybridEngineDebugSurface } from './HybridEngineDebug.js';
import {
  fingerprintHybridPipelineRebuildKey,
  getPreferredSwapChainFormat,
  resolveInternalRenderSize,
  runHybridEngineFrame,
  type HybridEngineFrameDeps,
  type HybridLightingDeps,
  type HybridDenoiserFilterDeps,
} from './HybridEngineFrameOrchestrator.js';
import { disposeSceneBVH } from './restir/bvhCompute.js';
import {
  rebuildEmitterBuffersFromSceneRoots,
  type ReSTIRBvhMode,
  type SceneBVHBuffers,
} from './restir/bvhCompute.js';
import { applyEmitterPatchToScene } from './scenePatch.js';
import {
  disposeVitrumThreeSceneRoot,
  solveSkin,
  vitrumSceneToThree,
} from '@vitrum/three-bindings';
import type { ModelWeights } from './neural/weights.js';
import {
  transformRefit,
  positionsRefit,
  topologyRebuild,
  materialPatch,
  refitSkinnedMeshAfterGpuWrite,
  TOPOLOGY_PATCH_FIELDS,
  type PrimitiveUpdateContext,
  type PrimitiveUpdateResult,
} from './HybridEnginePrimitiveUpdates.js';
import {
  PipelineInitCoordinator,
  collectDDGILightsFromThreeRoot,
  mergeDDGILightsDedupSun,
  type PipelineInitHost,
  type HybridInitStaticConfig,
} from './HybridEngineLifecycle.js';
import { readTunables, readInitTunables, type Tunables, type InitTunables } from './HybridEngineTuning.js';
import { resolveQualityPreset } from './HybridEngineQualityPreset.js';
import type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';
import { assertKnownLightingKeys } from './HybridEngineOptions.js';
import { RCSubsystem } from './HybridEngineRC.js';
import { propagateBvhToGiSubsystems } from './HybridEngineGiPropagation.js';
import { coreEmittersToDDGILights, directionalSunMultiplier } from './coreEmittersToDDGILights.js';
import { GpuSkinningSubsystem } from './skin/GpuSkinningSubsystem.js';

// Re-export the option / lighting interfaces from their dedicated module so
// the package's public surface (`./HybridEngine.js` import path) stays
// unchanged after the type split (refactor sweep 2026-05-18).
export type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';

/** Default per-frame target interval (~60 FPS soft-cap). */
const DEFAULT_TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

// `HybridEngineOptions` + `LightingOptions` interface bodies live in
// `HybridEngineOptions.ts` (~340 LOC of pure JSDoc, extracted refactor sweep
// 2026-05-18). Re-exported above so the package surface is unchanged.

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

// ────────────────────────────────────────────────────────────────────────────
// Option parsing + validation
// ────────────────────────────────────────────────────────────────────────────

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
interface ParsedHybridEngineConfig {
  readonly denoiser: 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final';
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
  /** NRC (Müller et al. 2021) cache gate (0 = off / verbatim DDGI suffix,
   *  1 = neural radiance cache eligible). Per-frame UBO gate, no pipeline
   *  rebuild. STAGED: gate plumbed end-to-end + bit-identity-pinned OFF; the
   *  query/record passes are the next phase. FORBIDDEN on tier:'lite'. */
  readonly nrcEnabled: number;
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
   *  Threaded into `pipeline.initialize` so the ppg-guide + ppg-update passes
   *  gate on `frameCount % ppgDispatchInterval`. Always ≥ 1. */
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
  // value (e.g. `'none'` from the @vitrum/core EngineOptions contract) does
  // not silently coerce to atrous-variance and produce wrong output. Supported
  // values are explicitly enumerated here. `'bmfr'` is now a real denoiser
  // (Koskela 2019 — see denoisers/bmfr.ts), so it is accepted.
  if (
    opts.denoiser !== undefined &&
    opts.denoiser !== 'atrous' &&
    opts.denoiser !== 'atrous-variance' &&
    opts.denoiser !== 'svgf-real' &&
    opts.denoiser !== 'bmfr' &&
    opts.denoiser !== 'neural' &&
    opts.denoiser !== 'oidn-final'
  ) {
    throw new TypeError(
      `[HybridEngine] unsupported denoiser '${opts.denoiser}'. ` +
      `walkaround-hybrid supports: 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final'. ` +
      `If you need 'none' from @vitrum/core, pick a backend that implements that mode.`,
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
    // NRC cache gate. Default 0 (OFF) so the gi-ris suffix is bit-identical to
    // the verbatim DDGI-atlas estimate unless a host opts in via opts.nrcEnabled
    // (which tier:'lite' forbids — validated above). STAGED: this gate is
    // plumbed to the UBO but the query/record passes are the next phase.
    nrcEnabled: opts.nrcEnabled === true ? 1 : 0,
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
function parseHybridEngineOptions(opts: HybridEngineOptions): ParsedHybridEngineConfig {
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
  //
  // Phase-0 productization — `_width/_height` are the CANVAS (swap-chain)
  // dimensions (what the composite pass blits TO + what `setSize` sets). The
  // INTERNAL render resolution (what the compute kernels dispatch over) is
  // `_internalWidth/_internalHeight = canvas × _resolutionFactor`; it equals
  // the canvas dims when no `quality.resolutionFactor` downscale is active.
  // The two are kept in sync by `setSize` (recomputes internal from the last
  // factor) and by the per-frame `quality.resolutionFactor` path in
  // `HybridEngineFrameOrchestrator` (debounced internal resize).
  private _width:                number;
  private _height:               number;
  /** Fires the one-time `FrameInput.viewport`-ignored dev warning at most once
   *  per engine instance (see {@link renderFrame}). */
  private _viewportMismatchWarned = false;
  /** Internal render width = `_width × _resolutionFactor`. Drives compute
   *  dispatch + UBO `screenSize`; the composite upscales to `_width`. */
  private _internalWidth:        number;
  /** Internal render height = `_height × _resolutionFactor`. */
  private _internalHeight:       number;
  /** Last-seen `FrameInput.quality.resolutionFactor` (clamped to (0,1]).
   *  Default 1.0 (internal == canvas). */
  private _resolutionFactor:     number = 1.0;
  /** `performance.now()` of the last accepted resolution-factor resize, for
   *  the debounce that prevents accumulator thrash (Risk R5). */
  private _lastResolutionResizeTs: number = 0;
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

  /** Rolling window of per-frame timings (newest last, cap 240 entries).
   *  Only populated when `debug === true`. Hosts that want a UI gauge
   *  should poll {@link debugTimings} instead of reaching into globals. */
  private readonly _debugTimings: Array<{ t: number; ms: number }> = [];

  /** T3.E — telemetry subscribers fired at end of each successful renderFrame. */
  private readonly _frameSubs: Array<(s: FrameStats) => void> = [];

  /** T3.E — long-running progress subscribers fired at the end of each
   *  dispatched frame, once per still-converging signal (`'ddgi-warmup'`
   *  while the probe grid warms, `'denoiser-converge'` while the temporal
   *  accumulator fills). Empty until a host calls {@link onProgress}. */
  private readonly _progressSubs: Array<(p: ProgressStats) => void> = [];

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
    return this._pipeline?.lastGpuTimings ?? {};
  }

  /** Frame index that produced {@link lastGpuTimings}; -1 if no readback
   *  has resolved yet. */
  get lastGpuTimingsFrame(): number {
    return this._pipeline?.lastGpuTimingsFrame ?? -1;
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
    if (!this._pipeline) return { perPass: {}, rawBigints: [] };
    return this._pipeline.readGpuTimingsOnce();
  }

  /**
   * The construction-time-immutable derived config (Task 4.2 / Theme A).
   * `parseHybridEngineOptions(opts)` produces this once in the constructor;
   * every tunable-cluster value the engine used to splat onto an individual
   * `this._x` field is now read from `this._cfg.x`. Holding the parsed record
   * directly collapses ~25 one-by-one ctor assignments + their field
   * declarations, so a single tunable no longer hops a private-field layer.
   *
   * ONLY construction-immutable values live here. Genuinely per-instance
   * MUTABLE runtime state (lighting, size/internal-size, accumulators, the
   * rebuild-key fingerprint, BVH/pipeline/scene handles) stays in its own
   * field below because it changes after construction.
   */
  private readonly _cfg: ParsedHybridEngineConfig;
  /** Dev A/B — mirrors `engine.debug.setDenoiserEnabled` (default on). */
  private _denoiserPassEnabled = true;

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

  // ── RC subsystem (W8 Phase 2 — opt-in via opts.rcEnabled) ───────────────
  private _rc: RCSubsystem | null = null;
  /** W8 Phase 3 — balance-heuristic MIS weight for RC in Lo_indirect.
   *  Effective only when _rc != null. Default 0.5 when rcEnabled is true. */
  private _rcWeight: number = 0;

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

  // ── Pipeline init coordinator (see HybridEngineLifecycle.ts) ──────────
  //
  // Owns the monotonic init-sequence + dispose race coordination + multi-
  // phase async init chain. The engine delegates init/dispose flow to it
  // via the {@link PipelineInitHost} surface built in `_buildInitHost()`.
  private readonly _initCoordinator: PipelineInitCoordinator;

  // Underscore-prefixed forwarders for the coordinator's race-tracking
  // state. These exist so existing dispose / init-race tests that reach
  // in via `engine['_initSeq']` / `engine['_pendingTeardown']` /
  // `engine['_initRunning']` continue to see the live values without
  // duplicating state on the engine. They're test seams, not part of any
  // documented engine surface.
  private get _initSeq(): number { return this._initCoordinator.initSeq; }
  private get _initRunning(): boolean { return this._initCoordinator.initRunning; }
  private get _pendingTeardown(): boolean { return this._initCoordinator.pendingTeardown; }
  private get _disposed(): boolean { return this._initCoordinator.disposed; }

  // Task 4.2 / Theme A — the construction-immutable tunable-cluster values live
  // on `_cfg` (one parsed record instead of ~25 splatted `_x` fields). Consumers
  // read `this._cfg.x` directly; tests pin resolved knobs via the `_cfg` seam.

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _skinning: GpuSkinningSubsystem | null;

  readonly debug: EngineDebugSurface;

  constructor(opts: HybridEngineOptions) {
    // Pure option parsing + validation (defaults, denoiser/neural/OIDN
    // validation throws) lives in `parseHybridEngineOptions` so the
    // constructor body stays focused on `this`-dependent wiring (subsystems,
    // capabilities, init coordinator, debug surface). Behaviour-preserving:
    // same throws in the same order, same defaults. (WD decomposition sweep.)
    // Task 4.2 / Theme A — hold the parsed config in one `_cfg` field rather
    // than splatting its ~25 construction-immutable values onto individual
    // `this._x` fields. Consumers read `this._cfg.x`; only genuinely-mutable
    // runtime state (device handle, size, lighting, accumulators, the rebuild-
    // key fingerprint) gets its own field.
    const cfg = parseHybridEngineOptions(opts);
    this._cfg = cfg;

    this._device                = opts.device;
    this._width                 = opts.width;
    this._height                = opts.height;
    // Phase-0 — the quality preset supplies the INITIAL internal-resolution
    // factor. A per-frame `quality.resolutionFactor` still overrides at runtime
    // (`_applyResolutionFactor`); this is just the starting point so a
    // `qualityTier:'low'` engine boots at 0.5 internal res.
    this._resolutionFactor      = cfg.resolutionFactor;
    this._internalWidth         = Math.max(1, Math.round(opts.width * cfg.resolutionFactor));
    this._internalHeight        = Math.max(1, Math.round(opts.height * cfg.resolutionFactor));
    this._skinning              = opts.gpuSkinning
      ? new GpuSkinningSubsystem(opts.device, true)
      : null;
    this._threeScene            = opts.threeScene ?? null;
    this._primaryLightDir       = opts.primaryLightDir;
    this._primaryLightIntensity = opts.primaryLightIntensity;
    this._skyTint               = opts.skyTint;
    this._skyIrradiance         = opts.skyIrradiance;
    // Default predicate: ready when EITHER the vitrum Scene supplies any mesh
    // primitive OR the optional escape-hatch THREE.Scene contains triangles.
    // Hosts override via opts.isSceneReady when they need a scene-specific
    // signal (e.g. wait for an async asset). Stays inline because it closes
    // over `this` (`_coreSceneSuppliesMeshes` / `_threeScene`).
    this._isSceneReady          = opts.isSceneReady ?? (() => {
      if (this._coreSceneSuppliesMeshes()) return true;
      return this._threeScene != null && defaultIsSceneReady(this._threeScene);
    });

    // `_rebuildKeyFingerprintSeen` is the one rebuild-key value that MUTATES
    // post-construction (`consumeRebuildKeyChange` rewrites it), so it stays a
    // mutable field seeded from the parsed config. The static key + getter live
    // on `_cfg` (immutable).
    this._rebuildKeyFingerprintSeen = cfg.rebuildKeyFingerprintSeen;

    this._ddgi = new DDGI({ debug: this._cfg.debug });
    // Phase-0 — apply the quality-preset DDGI probe-update divisor (default 4).
    this._ddgi.setProbeUpdateDivisor(this._cfg.ddgiUpdateDivisor);
    this._ctorLights = opts.lights ?? [];
    if (this._ctorLights.length > 0) {
      this._ddgi.setLights(this._ctorLights as DDGILight[]);
    }

    // W8 Phase 2 — opt-in RC subsystem. RCSubsystem owns its own BVH +
    // cascade GPUBuffers. setScene() rebuilds them when the source scene
    // changes; dispatch happens per-frame in renderFrame() below.
    if (opts.rcEnabled === true) {
      // B3b — Cornell-tuned CASCADE_DIMS default lives in walkaround-rc;
      // hosts override via opts.cascadeDims for non-Cornell aspect ratios
      // or scene scales.
      this._rc = opts.cascadeDims !== undefined
        ? new RCSubsystem(this._device, opts.cascadeDims)
        : new RCSubsystem(this._device);
      // W8 Phase 3 — host-overridable MIS weight (default 0.5 = equal
      // mix with ReSTIR-GI). When rcEnabled is false the weight stays 0
      // and pipeline.setRCInputs(null) routes the bind group to the
      // placeholder buffers (rcParams.enabled = 0u short-circuits the
      // shader's sample to vec3f(0)).
      this._rcWeight = Math.max(0, Math.min(1, opts.rcWeight ?? 0.5));
    }

    this.capabilities = {
      // Incremental scene patches are implemented: transform/positions fast
      // paths plus full-rebuild fallbacks for material/topology edits, and
      // emitter patching via scene-level rebuild.
      supportsIncrementalScene:  true,
      incrementalPatchSupport: {
        transform: true,
        positions: true,
        material: true,
        emitter: true,
        topology: true,
      },
      // Explicit whole-primitive add/remove IS implemented: addPrimitive appends
      // a new primitive and removePrimitive evicts one, each by routing a fresh
      // mutated `Scene` copy through this engine's existing `setScene` spine —
      // the same `partitionSceneBySupport` → `_teardownPipeline` → init-chain
      // (full BVH / DDGI / ReSTIR rebuild + temporal-accumulator reset) the
      // initial scene build runs. A geometry change invalidates every cached GI
      // signal, so on this realtime stack the work is a rebuild either way; the
      // value is API consistency with pt-webgl / pt-webgpu, not a perf win. The
      // DDGI / ReSTIR / RC subsystems index off the packed scene, so reusing the
      // setScene packing path re-syncs them all correct-by-construction — no
      // fragile per-array index remap. Distinct from
      // incrementalPatchSupport.topology (COUNT-change patches on an EXISTING
      // primitive). Kept in sync with the walkaround-hybrid row in
      // @vitrum/core's BACKEND_PROMISE_LEDGER.
      supportsAddRemovePrimitive: true,
      supportsAuxBuffers:        false,
      accumulates:               false,
      maxSamplesPerPixel:        Infinity,
      maxBounces:                this._cfg.maxBounces,
      supportedAnalyticShapes:   new Set(),
      // BVH + DDGI ingest via vitrumSceneToThree, which handles
      // mesh / skinned-mesh / instanced-mesh and throws on anything else
      // (analytic). `analytic` stays OUT — partitionSceneBySupport drops it
      // with a warning at setScene before the converter can throw.
      supportedPrimitiveKinds:   new Set<ScenePrimitive['kind']>(['mesh', 'skinned-mesh', 'instanced-mesh']),
      // Emitter kinds that genuinely reach a renderable state:
      //   - rect-area / disc-area → THREE.RectAreaLight, harvested as ReSTIR-DI
      //     direct emitter tris (collectRectAreaLightEmitterTris) AND projected
      //     to DDGI fixture lights (coreEmittersToDDGILights) for indirect bounce.
      //   - mesh-area → folded into the referenced mesh's emissive material by
      //     vitrumSceneToThree, so it reaches both the ReSTIR-DI emissive-triangle
      //     path and DDGI as emissive geometry.
      //   - point / spot → projected to DDGI fixture lights by
      //     coreEmittersToDDGILights (spot is a point-like approximation — the
      //     probe shader has no cone handling).
      //   - directional → projected to a DDGI `sun` light by
      //     coreEmittersToDDGILights, carrying the emitter's REAL direction
      //     (negated to a travel direction), intensity, and colour into the
      //     probe pass's sun path (replacing the packer's former hardcoded
      //     straight-down warm-white sun). Single-counted: the host sets the
      //     sun-intensity multiplier to 1 when a scene directional is present,
      //     so the emitter intensity is not double-applied. The directional
      //     still drives the shade-side Lo_emit via the WalkaroundUBO config
      //     path (primaryLightDir/Intensity) — those remain host config, not
      //     derived from the emitter, so there is no shade-side double-count.
      supportedEmitterKinds:     new Set<SceneEmitter['kind']>([
        'directional',
        'rect-area',
        'disc-area',
        'point',
        'spot',
        'mesh-area',
      ]),
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri']),
      presentationMode:          'swapchain-required',
      experimentalFeatures:      new Set(['svgf-real-conservative-objid']),
      // RFE-05: Real-time caustic strategies (MNEE / photon-map) are not
      // compatible with the walkaround engine's frame cadence; the walkaround
      // engine always reports 'none'. Track via
      // external_requests/05-manifold-nee.md §4 for the approved approximation
      // path if real-time caustic approximation is added.
      causticStrategy: 'none',
      // W3-D8 — this engine ships a `debug` surface (DDGI atlases, BVH nodes,
      // GI signal textures, and now estimatedGpuMemoryBytes). Hosts can
      // structurally opt-in to the dev-overlay panel without typeof-checking
      // every method.
      debugSurface: true,
    };

    // Pipeline-init coordinator: own the async init race state machine and
    // dispose coordination. The host adapter built below grants the
    // coordinator the small surface it needs without exposing private
    // engine fields directly.
    this._initCoordinator = new PipelineInitCoordinator(this._buildInitHost());

    this.debug = createHybridEngineDebugSurface({
      device: () => this._device,
      readAtlas: () => this._ddgi?.pass?.getReadAtlasGPUTextures?.() ?? null,
      bvhNodesCpu: () => this._bvhBuffers?.bvhNodes?.cpuData,
      pipelineResources: () => this._pipeline?.frameResources ?? null,
      denoiserPassEnabled: () => this._denoiserPassEnabled,
      setDenoiserPassEnabled: (enabled) => {
        this._denoiserPassEnabled = enabled;
      },
      setPipelineDenoiserPassEnabled: (enabled) => {
        this._pipeline?.setDenoiserPassEnabled(enabled);
      },
    });
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
   * **Capability filter:** the scene is first partitioned against this engine's
   * declared `supported*Kinds` (warn + skip). Unsupported kinds — notably
   * `analytic`, which has no THREE-conversion path — are dropped with a
   * `console.warn` before conversion, so they never reach (and throw from)
   * `vitrumSceneToThree`.
   *
   * @param inputScene - The `@vitrum/core` scene (e.g. from `sceneFromThreeJS`).
   */
  setScene(inputScene: Scene): void {
    // Capability filter (warn + skip) — consume this engine's OWN declared
    // support sets to drop kinds it cannot ingest (e.g. `analytic`, whose
    // THREE-conversion path does not exist) BEFORE `vitrumSceneToThree` runs
    // in `_ensureThreeSceneRoot` / the lifecycle BVH-build phase. Without this,
    // an analytic primitive would reach the converter and throw; the warn-skip
    // model matches pt-webgpu's `buildPackedScene` behaviour.
    const { supported: scene, warnings } = partitionSceneBySupport(inputScene, this.capabilities);
    for (const warning of warnings) {
      console.warn(`[vitrum/walkaround-hybrid] ${warning}`);
    }
    this._lastScene = scene;
    // T3.H removal: drop the cached synthesized THREE.Scene; the next BVH
    // build / DDGI updateFrame will re-derive it from the new vitrum Scene.
    if (this._synthesizedThreeScene != null) {
      try { disposeVitrumThreeSceneRoot(this._synthesizedThreeScene); } catch {}
      this._synthesizedThreeScene = null;
    }

    // W8 Phase 2 — rebuild the RC BVH + cascade buffers against the new
    // scene. Synthesise a fresh THREE root if needed (lazy via
    // `_ensureThreeSceneRoot`). When the host supplied no source, the RC
    // dispatcher stays idle until the next setScene.
    // RC BVH is wired after async ReSTIR BVH publish (see publishBvh).

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    this._initCoordinator.startInit();
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
      this._synthesizedThreeScene = vitrumSceneToThree(this._lastScene);
      return this._synthesizedThreeScene;
    }
    return null;
  }

  // ── updatePrimitive — geometry-change path ─────────────────────────────
  //
  // **Routing rules**:
  //  - `patch.transform` present AND no topology fields → fast-path (c):
  //     refit the BVH bounds in-place (no SAH rebuild, no pipeline
  //     recompile, no DDGI atlas invalidation), rewrite the affected
  //     primitive's vertex slice in `bvhPositions`, reset the accumulator.
  //  - any topology field present (`positions` / `normals` / `uvs` /
  //     `tangents` / `indices` / `instances` / `params` / `shape` /
  //     `fallbackMesh` / `kind`) → full-rebuild path (a): re-run
  //     `buildReSTIRSceneBVH`, destroy + reupload all four BVH GPU
  //     buffers, reset the accumulator.
  //  - material-only patches → patch scene primitive + rebuild via setScene()
  //     for correctness (material-byte upload optimization can still be
  //     layered later without changing host contract behavior).
  //
  // Implementations live in `HybridEnginePrimitiveUpdates.ts`; this method
  // is the routing dispatcher.
  //
  // Implements `Engine.updatePrimitive(id, patch)` from `@vitrum/core`.
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no scene set. ` +
        `Call setScene(scene) before updatePrimitive.`,
      );
    }
    const primIndex = this._lastScene.primitives.findIndex((p) => String(p.id) === id);
    if (primIndex < 0) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive id not found in current scene.`,
      );
    }

    // Three.js BVH refit (fast path) preserves topology when only AABB
    // bounds need updating. Three flavours:
    //   - Transform only       → transformRefit  (~1 ms / 30k tris)
    //   - Positions only       → positionsRefit  (A3, ~1 ms / 30k tris)
    //   - True topology change → topologyRebuild (~50 ms / 30k tris)
    // "Positions only" means new vertex data on the SAME index buffer +
    // SAME vertex count. The positionsRefit path falls through to
    // topologyRebuild internally if the count doesn't match.
    const result = this._routePrimitiveUpdate(id, patch);
    if (result == null) {
      // No recognised patch field — treat as a no-op rather than throw so
      // hosts can pass through optional patches without checking each
      // field's presence.
      return;
    }
    this._applyUpdateResult(result);
  }

  /**
   * Select the fast/full path for an `updatePrimitive` patch and run it.
   * Returns `null` for an unrecognised patch (no-op). Branch order is
   * load-bearing — topology beats positions beats transform beats material:
   *  - any topology field present → full SAH `topologyRebuild` (Option (a)),
   *    even if `positions` is also in the patch (the index buffer / vertex
   *    layout changed, so the count-preserving refit can't apply).
   *  - `positions` only → A3 `positionsRefit` (same topology, new verts).
   *  - `transform` only → `transformRefit` (refit AABB bounds in place).
   *  - `material` only → `materialPatch` (re-pack slices, NO GI propagation;
   *    the result carries `applySubsystems: false`).
   */
  private _routePrimitiveUpdate(
    id: string,
    patch: Partial<ScenePrimitive>,
  ): PrimitiveUpdateResult | null {
    const has = (f: string): boolean => (patch as Record<string, unknown>)[f] !== undefined;
    const ctx = this._buildPrimitiveUpdateContext();
    if (TOPOLOGY_PATCH_FIELDS.some((f) => has(f))) return topologyRebuild(id, patch, ctx);
    if (has('positions')) return positionsRefit(id, patch, ctx);
    if (has('transform')) return transformRefit(id, patch, ctx);
    if (has('material')) return materialPatch(id, patch, ctx);
    return null;
  }

  /**
   * Uniform epilogue for every primitive-update path: swap the freshly-built
   * BVH buffers + patched scene into engine state, then — unless the path
   * opted out (`applySubsystems === false`, the material-only fast path) —
   * re-sync the GI subsystems against the new BVH.
   */
  private _applyUpdateResult(result: PrimitiveUpdateResult): void {
    this._bvhBuffers = result.bvhBuffers;
    this._lastScene = result.updatedScene;
    if (result.applySubsystems !== false) {
      this._applyPrimitiveUpdateSubsystems(result);
    }
  }

  /**
   * PR-7 — GPU LBS wrote world positions into the live `bvhPositions` buffer;
   * refit BVH nodes and sync scene without re-uploading the position slice.
   */
  applyGpuSkinnedRefit(
    id: string,
    localPositions?: Float32Array,
    localNormals?: Float32Array,
  ): void {
    let positions = localPositions;
    let normals = localNormals;
    if (positions == null) {
      const prim = this._lastScene?.primitives.find(
        (p) => String(p.id) === id && p.kind === 'skinned-mesh',
      );
      if (prim?.kind !== 'skinned-mesh') {
        throw new Error(`applyGpuSkinnedRefit("${id}"): skinned-mesh primitive not found.`);
      }
      const solved = solveSkin(prim);
      positions = solved.positions;
      normals = solved.normals;
    }
    const result = refitSkinnedMeshAfterGpuWrite(
      id,
      positions,
      normals,
      this._buildPrimitiveUpdateContext(),
    );
    this._applyUpdateResult(result);
  }

  /** Merged BVH position SSBO for GPU skinning (null before pipeline init). */
  getGpuSkinningBvhBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhPositionBuffer() ?? null;
  }

  /** WS1 — merged BVH normal SSBO for GPU skinning (null before pipeline init).
   *  The skin kernel writes inverse-transpose skinned normals here at
   *  `baseVertex+vi` so the smooth shading-normal blend reads deformed normals. */
  getGpuSkinningNormalBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhNormalBuffer() ?? null;
  }

  /** Per-mesh vertex ranges in the merged BVH (for GPU skinning). */
  getMeshVertexRanges(): SceneBVHBuffers['meshVertexRanges'] | null {
    return this._bvhBuffers?.meshVertexRanges ?? null;
  }

  /** Active ReSTIR BVH layout (`merged` world positions vs `tlas` local BLAS). */
  getBvhMode(): ReSTIRBvhMode | null {
    return this._bvhBuffers?.bvhMode ?? null;
  }

  getPrimitiveTlasBindings(): SceneBVHBuffers['primitiveTlasBindings'] | null {
    return this._bvhBuffers?.primitiveTlasBindings ?? null;
  }

  /**
   * After geometry BVH updates: sync DDGI probe rays + RC cascades to the live
   * ReSTIR buffers without waiting for the next `renderFrame` tick.
   */
  private _applyPrimitiveUpdateSubsystems(result: PrimitiveUpdateResult): void {
    propagateBvhToGiSubsystems({
      ddgi: this._ddgi,
      rc: this._rc,
      bvhBuffers: this._bvhBuffers,
      lastScene: this._lastScene,
      syncDdgi: true,
      allowRcSceneRebuild: true,
      ensureThreeSceneRoot: () => this._ensureThreeSceneRoot(),
      rcRefitBounds: result.rcRefitBounds,
    });
  }

  /** Build the per-call resource context the primitive-update helpers consume. */
  private _buildPrimitiveUpdateContext(): PrimitiveUpdateContext {
    if (this._lastScene == null) {
      throw new Error(
        'HybridEngine.updatePrimitive: no scene set. Call setScene(scene) first.',
      );
    }
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers:            this._bvhBuffers,
      threeRoot:             this._ensureThreeSceneRoot(),
      pipeline:              this._pipeline,
      ddgi:                  this._ddgi,
      primaryLightDir:       this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      lastScene:             this._lastScene,
    };
    if (this._cfg.restirBvhModeOverride !== undefined) {
      return { ...ctx, restirBvhModeOverride: this._cfg.restirBvhModeOverride };
    }
    return ctx;
  }

  /**
   * Add one whole primitive to the live scene (contract:
   * {@link Engine.addPrimitive}).
   *
   * Design choice — full `setScene`-equivalent rebuild. A whole-primitive add
   * almost always introduces NEW geometry + a NEW material, and on this realtime
   * stack the geometry change invalidates EVERY cached GI signal — the DDGI
   * irradiance/visibility atlases, the ReSTIR-DI/GI reservoir history, the RC
   * cascade buffers, and the temporal accumulator all index off the packed
   * scene and must be rebuilt/reset. Rather than bolt a brittle "incremental BVH
   * splice + per-subsystem re-sync" onto the add path, we append the primitive
   * to a fresh `Scene` copy and route it through `setScene` — the same
   * already-correct `partitionSceneBySupport` → `_teardownPipeline` →
   * `_initCoordinator.startInit()` spine the initial scene build runs. That
   * spine rebuilds the BVH, re-synthesises the DDGI traversal scene, re-derives
   * DDGI lights, rebuilds the RC BVH (via `publishBvh` →
   * `propagateBvhToGiSubsystems`), and — because the pipeline (and its temporal
   * accumulator + reservoirs + history textures) is torn down and rebuilt blank
   * — resets all temporal/accumulation state. Mirrors pt-webgl / pt-webgpu's
   * full-repack choice: correct-by-construction, no fragile per-array index
   * remap. On a realtime engine the work is a rebuild either way; the value is
   * API consistency, not a perf win.
   *
   * Contract semantics honored:
   *   • Duplicate `id` throws BEFORE any mutation — the dup check runs against
   *     the live `_lastScene`, and `nextScene` is only built (and `setScene`
   *     only called) once it passes, so the scene is unchanged on throw.
   *   • Unsupported primitive kinds / analytic shapes are warn-skipped by the
   *     `partitionSceneBySupport` filter inside `setScene` (they do not throw).
   *   • Accumulation / temporal history resets — `setScene` tears down the
   *     pipeline (blank accumulator + reservoirs + DDGI/ReSTIR/RC rebuild) and
   *     reinitialises.
   */
  addPrimitive(primitive: ScenePrimitive): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.addPrimitive: engine is disposed.');
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.addPrimitive("${String(primitive.id)}"): no scene set. ` +
        `Call setScene(scene) before addPrimitive.`,
      );
    }
    if (this._lastScene.primitives.some((p) => String(p.id) === String(primitive.id))) {
      throw new Error(
        `HybridEngine.addPrimitive: a primitive with id "${String(primitive.id)}" ` +
        `already exists; use updatePrimitive to mutate an existing primitive.`,
      );
    }
    const nextScene: Scene = {
      ...this._lastScene,
      primitives: [...this._lastScene.primitives, primitive],
    };
    this.setScene(nextScene);
  }

  /**
   * Remove one whole primitive from the live scene by `id` (contract:
   * {@link Engine.removePrimitive}). The inverse of {@link addPrimitive}.
   *
   * Implementation: drop the primitive from a fresh `Scene` copy and route
   * through `setScene` (same full-rebuild approach as {@link addPrimitive}).
   * Reusing the `setScene` packing path re-packs the dense BVH / DDGI-light /
   * ReSTIR-emitter arrays correctly by construction rather than hand-rolling a
   * multi-array compaction. Removing the last primitive is legal and yields a
   * renderable sky-only scene (the empty scene `setScene` already supports — the
   * factory bootstraps with exactly that).
   *
   * Contract semantics honored:
   *   • A missing `id` throws BEFORE any mutation — the membership check runs
   *     against the live `_lastScene`; `setScene` is only called once it passes,
   *     so the scene is unchanged on throw.
   *   • Accumulation / temporal history resets exactly as for
   *     {@link addPrimitive}.
   */
  removePrimitive(id: ScenePrimitive['id']): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.removePrimitive: engine is disposed.');
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.removePrimitive("${String(id)}"): no scene set. ` +
        `Call setScene(scene) before removePrimitive.`,
      );
    }
    const nextPrimitives = this._lastScene.primitives.filter((p) => String(p.id) !== String(id));
    if (nextPrimitives.length === this._lastScene.primitives.length) {
      throw new Error(
        `HybridEngine.removePrimitive: no primitive with id "${String(id)}" ` +
        `in the live scene.`,
      );
    }
    const nextScene: Scene = {
      ...this._lastScene,
      primitives: nextPrimitives,
    };
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): no scene set. ` +
        `Call setScene(scene) before updateEmitter.`,
      );
    }
    const idx = this._lastScene.emitters.findIndex((e) => String(e.id) === id);
    if (idx < 0) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): emitter id not found in current scene.`,
      );
    }
    if (this._bvhBuffers == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): BVH not ready. Wait for setScene init to finish.`,
      );
    }
    const threeRoot = this._ensureThreeSceneRoot();
    if (threeRoot == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): no THREE scene available.`,
      );
    }

    this._lastScene = applyEmitterPatchToScene(this._lastScene, id, patch);

    const emitterSlice = rebuildEmitterBuffersFromSceneRoots(
      [threeRoot],
      this._bvhBuffers,
      {
        primaryLightDir: new THREE.Vector3(...this._primaryLightDir),
        primaryLightIntensity: this._primaryLightIntensity,
      },
    );

    this._bvhBuffers = {
      ...this._bvhBuffers,
      emitters: emitterSlice.emitters,
      emitterCdf: emitterSlice.emitterCdf,
      emitterCount: emitterSlice.emitterCount,
      totalEmissivePower: emitterSlice.totalEmissivePower,
      lightTree: emitterSlice.lightTree,
      lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
      lightTreeEnabled: emitterSlice.lightTreeEnabled,
    };

    this._pipeline?.updateEmitters(this._bvhBuffers);
    this._syncDdgiLightsFromThreeRoot();
    this._pipeline?.requestAccumReset();
  }

  /** Re-upload DDGI point/rect lights from the live THREE scene (no `setScene`). */
  refreshDdgiLightsFromThreeScene(): void {
    this._syncDdgiLightsFromThreeRoot();
    this._pipeline?.requestAccumReset();
  }

  private _syncDdgiLightsFromThreeRoot(): void {
    const root = this._ensureThreeSceneRoot();
    if (root == null) return;
    // Theme T16 — match the init path's gate (HybridEngineLifecycle): prefer
    // the lossless core-emitter projection when a core scene supplies meshes,
    // which preserves chroma, uses the true emissive area `4·|uAxis × vAxis|`
    // for rect emitters, and carries the source emitter id. Fall back to the
    // lossy THREE-walk ONLY for the raw-`threeScene` case (no core scene to
    // read emitters from). Without this, an incremental `updateEmitter` /
    // `refreshDdgiLightsFromThreeScene` re-derived lights via the lossy walk
    // and silently dropped chroma / used the wrong area, drifting from the
    // freshly-built init state.
    const sceneForSun =
      this._coreSceneSuppliesMeshes() && this._lastScene != null
        ? this._lastScene
        : null;
    const sceneLights =
      sceneForSun != null
        ? coreEmittersToDDGILights(sceneForSun)
        : collectDDGILightsFromThreeRoot(root);
    // Single-count the sun: a scene directional emits a `sun` DDGILight that
    // already carries its own intensity, so the multiplier must be 1; absent a
    // scene directional, keep the legacy config multiplier. Mirrors the init
    // coordinator's resolution so an incremental emitter edit can't drift the
    // sun magnitude away from the freshly-built init state.
    this._ddgi.pass.setSunIntensityMultiplier(
      directionalSunMultiplier(sceneForSun, this._primaryLightIntensity),
    );
    this._ddgi.setLights(mergeDDGILightsDedupSun(this._ctorLights, sceneLights));
    this._ddgi.invalidateProbeCache();
  }

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
    // `Engine.updateLighting` is contractually opaque (Record<string, unknown>),
    // so hosts can pass any key without a type error at the core-contract call
    // site. Warn (don't throw) on keys outside LightingOptions so silent drops
    // become visible; the field-by-field application below is unchanged.
    assertKnownLightingKeys(opts as Readonly<Record<string, unknown>>);

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
      // irradiance atlas re-converges at the correct brightness. Single-count:
      // when a scene `directional` drives the sun, its `sun` DDGILight already
      // carries the emitter intensity, so the multiplier stays 1 and config
      // primaryLightIntensity does NOT additionally scale the DDGI sun (it
      // still drives the shade-side Lo_emit via the WalkaroundUBO). Absent a
      // scene directional, the config intensity is the multiplier as before.
      const sceneForSun =
        this._coreSceneSuppliesMeshes() && this._lastScene != null
          ? this._lastScene
          : null;
      this._ddgi.pass.setSunIntensityMultiplier(
        directionalSunMultiplier(sceneForSun, opts.primaryLightIntensity),
      );
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
    // Recompute the internal render size from the last-seen resolution factor
    // so a canvas resize preserves the host's chosen downscale. The pipeline
    // renders at the INTERNAL size; the composite upscales to the canvas.
    this._internalWidth = Math.max(1, Math.round(width * this._resolutionFactor));
    this._internalHeight = Math.max(1, Math.round(height * this._resolutionFactor));
    if (this._pipeline) {
      this._pipeline.resize(this._internalWidth, this._internalHeight);
    }
    // No DDGI invalidation — the irradiance atlas is world-space, not
    // screen-space, so it survives a resize unchanged.
  }

  /**
   * §5.1 — apply a per-frame `quality.resolutionFactor`. Computes the target
   * internal render size (= canvas × clamped factor), and — when it changes
   * beyond a 2-px threshold and the debounce window has elapsed — resizes the
   * pipeline's per-frame resources to the internal size. The composite pass
   * upscales the internal-sized resolvedTexture to the full canvas swap-chain
   * view, so no swap-chain reconfigure is needed.
   *
   * Returns the internal dims to dispatch at this frame (unchanged when the
   * resize was debounced). Called once per frame from the orchestrator before
   * `pipeline.renderFrame`.
   */
  private _applyResolutionFactor(
    factor: number | undefined,
    nowMs: number,
  ): { width: number; height: number } {
    // Record the clamped factor so a subsequent setSize() preserves the host's
    // chosen downscale.
    this._resolutionFactor =
      typeof factor === 'number' && Number.isFinite(factor) && factor > 0
        ? Math.min(1, factor)
        : 1;

    const decision = resolveInternalRenderSize({
      swapW: this._width,
      swapH: this._height,
      factor,
      currentW: this._internalWidth,
      currentH: this._internalHeight,
      nowMs,
      lastResizeTs: this._lastResolutionResizeTs,
    });

    if (decision.shouldResize) {
      this._internalWidth = decision.targetW;
      this._internalHeight = decision.targetH;
      this._lastResolutionResizeTs = nowMs;
      this._pipeline?.resize(decision.targetW, decision.targetH);
    }
    return { width: this._internalWidth, height: this._internalHeight };
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
   * this returns a "skip" FrameOutput (`kind: 'skipped'`,
   * `samplesAccumulated: 0`, `isConverged: false`).
   *
   * Note: `input.viewport` (the CANVAS size) is ignored by HybridEngine — its
   * WebGPU render targets (DDGI atlas, ReSTIR reservoirs, history textures,
   * accumulation buffer) are sized to the canvas at construction and the
   * canvas size is changed only via {@link setSize}. Hosts MUST call
   * `engine.setSize(w, h)` when the canvas dimensions change; pushing a new
   * `viewport` per frame is silently dropped.
   *
   * However, `input.quality.resolutionFactor` IS honoured per-frame
   * (Phase-0 productization): it scales the INTERNAL render resolution
   * (= canvas × factor) via a debounced {@link _applyResolutionFactor}; the
   * composite pass upscales to the full canvas. See the `@vitrum/core`
   * FrameInput.viewport JSDoc for the cross-backend contract.
   */
  renderFrame(input: FrameInput): FrameOutput {
    // Ergonomics guard: HybridEngine sizes its render targets at construction /
    // `setSize()` and does NOT honour `FrameInput.viewport` per-frame (unlike the
    // converged PT backends — see the FrameInput.viewport contract note). A host
    // that resizes its canvas and pushes a new viewport, expecting the engine to
    // follow, would silently render at the stale size. Warn once so the misuse is
    // visible instead of mysterious. (attachVitrum wires setSize for you.)
    if (
      !this._viewportMismatchWarned &&
      (input.viewport.width !== this._width || input.viewport.height !== this._height)
    ) {
      this._viewportMismatchWarned = true;
      console.warn(
        `[HybridEngine] FrameInput.viewport (${input.viewport.width}×${input.viewport.height}) ` +
          `differs from the engine canvas size (${this._width}×${this._height}) and is IGNORED. ` +
          'HybridEngine sizes render targets at construction; call engine.setSize(width, height) ' +
          'on canvas resize (attachVitrum does this automatically). For per-frame internal-' +
          'resolution scaling use FrameInput.quality.resolutionFactor instead.',
      );
    }
    return runHybridEngineFrame(this._buildFrameDeps(), input);
  }

  /** Live lighting snapshot — the four runtime-mutable lighting fields
   *  (`updateLighting()` mutates them). Grouped so both DI builders and any
   *  future lighting consumer share one source of truth: adding a lighting
   *  field is a single edit here, not one per builder. Read at call time
   *  (per-frame snapshot semantics — see {@link _buildFrameDeps}). */
  private _lightingSnapshot(): HybridLightingDeps {
    return {
      primaryLightDir: this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      skyTint: this._skyTint,
      skyIrradiance: this._skyIrradiance,
    };
  }

  /** Tuple-typed denoiser-filter cluster (firefly clamp + per-channel atrous
   *  sigmas). These live outside the number-only {@link Tunables} table
   *  because they are tuple-valued; grouping them keeps {@link _buildFrameDeps}
   *  compact and makes a new tuple knob a single edit. */
  private _denoiserFilterDeps(): HybridDenoiserFilterDeps {
    return {
      indirectFireflyClamp: this._cfg.indirectFireflyClamp,
      atrousDirectSigmas: this._cfg.atrousDirectSigmas,
      atrousIndirectSigmas: this._cfg.atrousIndirectSigmas,
      stainedGlassFlags: this._cfg.stainedGlassFlags,
      restirPtReuse: this._cfg.restirPtReuse,
      nrcEnabled: this._cfg.nrcEnabled,
    };
  }

  private _buildFrameDeps(): HybridEngineFrameDeps {
    const self = this;
    return {
      get state() {
        return self._state;
      },
      debug: self._cfg.debug,
      dbg: self._cfg.debug ? self._dbg : null,
      pipeline: self._pipeline,
      bvhBuffers: self._bvhBuffers,
      consumeRebuildKeyChange: () => {
        const fp = fingerprintHybridPipelineRebuildKey(
          self._cfg.getPipelineRebuildKey?.() ?? self._cfg.staticPipelineRebuildKey,
        );
        if (fp !== self._rebuildKeyFingerprintSeen) {
          self._rebuildKeyFingerprintSeen = fp;
          self.reset();
          return true;
        }
        return false;
      },
      targetFrameIntervalMs: self._cfg.targetFrameIntervalMs,
      getLastFrameTs: () => self._lastFrameTs,
      setLastFrameTs: (ts) => {
        self._lastFrameTs = ts;
      },
      width: self._width,
      height: self._height,
      internalWidth: self._internalWidth,
      internalHeight: self._internalHeight,
      applyResolutionFactor: (factor, nowMs) => self._applyResolutionFactor(factor, nowMs),
      skinning: self._skinning,
      lastScene: self._lastScene,
      runSkinning: () => {
        if (self._skinning != null && self._lastScene != null) {
          self._skinning.run(self, self._lastScene);
        }
      },
      ddgiOn: self._ddgiOn,
      isLayerEnabled: (layer) => self._layerEnabled.get(layer) ?? true,
      ddgi: self._ddgi,
      ddgiTraversalScene: self._ddgiTraversalScene,
      ensureThreeSceneRoot: () => self._ensureThreeSceneRoot(),
      device: self._device,
      tunables: self._cfg.tunables,
      rc: self._rc,
      ...self._lightingSnapshot(),
      rcWeight: self._rcWeight,
      ...self._denoiserFilterDeps(),
      frameSubs: self._frameSubs,
      progressSubs: self._progressSubs,
      verbose: self._cfg.verbose,
      debugTimings: self._debugTimings,
      debugSurface: self.debug,
      presentLastFrame: (view) => {
        self._pipeline?.presentLastFrame(view);
      },
    };
  }

  // ── Reset ──────────────────────────────────────────────────────────────

  /**
   * Tear down the pipeline and reinitialise from scratch.
   * Hosts call this when the scene changes significantly.
   */
  reset(): void {
    this._teardownPipeline();
    this._initCoordinator.startInit();
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

  /**
   * Subscribe to long-running progress events. Returns an unsubscribe
   * function. Subscribers that throw are swallowed so the render loop stays
   * alive (mirrors {@link onFrame}).
   *
   * Walkaround engines don't accumulate samples, so there is no `'pt-spp'`
   * signal — but the two REAL warm-up signals this engine has ARE surfaced
   * (closing the contract's `'ddgi-warmup'` / `'denoiser-converge'`
   * zero-producer gap):
   *
   *   - `'ddgi-warmup'` — the DDGI probe round-robin updates `1/stride` of the
   *     grid per frame, so a freshly built / invalidated grid takes `stride`
   *     frames for every probe to receive its first update. `fraction` ramps
   *     `frame / stride` from 0→1 and emission STOPS once the grid is warm
   *     (`DDGI.ready`). Reset to 0 on `setScene()` (fresh DDGI) and
   *     `updateLighting()` (`invalidateProbeCache()`).
   *
   *   - `'denoiser-converge'` — the temporal accumulator blends `α` of the new
   *     frame with `1-α` of history (α≈0.01 ⇒ ~100-frame window). `fraction`
   *     ramps `accumFrameIndex / round(1/α)` and emission STOPS once the
   *     window is full. Reset to 0 on camera motion, `updateLighting()` /
   *     `updateEmitter()` (`requestAccumReset()`), and `setSize()` / resize.
   *
   * Both events fire at most once per dispatched frame, only while their
   * signal is still converging. A no-op when the host registered no callback.
   */
  onProgress(cb: (progress: ProgressStats) => void): () => void {
    this._progressSubs.push(cb);
    return () => {
      const i = this._progressSubs.indexOf(cb);
      if (i >= 0) this._progressSubs.splice(i, 1);
    };
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  /**
   * Synchronous dispose — releases all engine-owned GPU resources.
   *
   * The contract intentionally remains synchronous (so hosts can call it
   * from React cleanup effects, finalizers, etc. without an async
   * paradigm shift). When the {@link PipelineInitCoordinator} has an init
   * chain in flight, the actual GPU-resource release for any work that
   * chain hasn't yet published is deferred to the chain's own `finally`
   * block — the chain checks the coordinator's `_pendingTeardown` after
   * every await boundary and, if set, disposes its locals AND finalises
   * teardown of whatever did make it to shared state.
   *
   * Idempotent: a second `dispose()` call is a no-op.
   */
  dispose(): void {
    if (this._state === 'disposed' && !this._initCoordinator.initRunning) {
      // Already disposed and no in-flight chain to coordinate with — no-op.
      return;
    }

    // requestTeardown returns true when the coordinator has no chain in
    // flight (we tear down here and now), false when it has one in flight
    // and its finally block will tear down. The coordinator records the
    // dispose intent regardless so its phase checkpoints bail.
    const teardownNow = this._initCoordinator.requestTeardown();
    if (teardownNow) {
      // No in-flight init; tear down here and now.
      this._teardownPipeline();
      this._skinning?.dispose();
      this._ddgi.dispose();
      // W8 Phase 2 — also tear down RC subsystem when active.
      if (this._rc) {
        this._rc.dispose();
        this._rc = null;
      }
      this._state = 'disposed';
    } else {
      // An init is mid-flight. Defer teardown to that chain's finally
      // block — it will dispose its locals AND tear down whatever's
      // currently in shared state. We can't safely call
      // _teardownPipeline() here because the in-flight chain's
      // `await pipeline.initialize()` may still be holding a live
      // reference to a half-built pipeline.
      this._state = 'disposed';
      // Note: _ddgi.dispose() is deferred too; the in-flight chain may
      // still call _ddgi.pass.setSunIntensityMultiplier() after the
      // post-pipeline checkpoint, and we don't want a torn-down DDGI
      // under it. The chain's finally calls disposeDdgi() when it sees
      // pending teardown.
    }

    if (this._cfg.debug && typeof window !== 'undefined') {
      const dbg = this._dbg;
      dbg.disposeCount++;
      const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
      console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
        ranForMs: liveMs.toFixed(1),
        framesDispatched: dbg.framesDispatched,
        deferredTeardown: !teardownNow,
        skipReasons: {
          noPipeline: dbg.skipNoPipeline, noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
        },
      });
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────

  /** True when `_lastScene` supplies at least one triangle mesh primitive
   *  (rest-pose skinned meshes count — host pushes deformed positions via
   *  `updatePrimitive`, but the BVH still needs a non-empty scene to build). */
  private _coreSceneSuppliesMeshes(): boolean {
    const s = this._lastScene;
    return s != null && s.primitives.some((p) => p.kind === 'mesh' || p.kind === 'skinned-mesh');
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

  /** Construction-time immutable config consumed by the init coordinator.
   *  Every field here is assigned once in the constructor and never mutated,
   *  so plain values are behaviorally identical to live getters — grouping
   *  them collapses ~11 one-line getters into one spread and makes adding a
   *  ctor-immutable init input a single edit. The MUTABLE fields (width,
   *  height, lastScene, lighting, current BVH/traversal-scene) stay as live
   *  getters in {@link _buildInitHost} because the coordinator reads them
   *  across async phases. */
  private _initStaticConfig(): HybridInitStaticConfig {
    return {
      device: this._device,
      threeScene: this._threeScene,
      restirBvhModeOverride: this._cfg.restirBvhModeOverride,
      denoiser: this._cfg.denoiser,
      neuralWeights: this._cfg.neuralWeights,
      oidnModelUrl: this._cfg.oidnModelUrl,
      oidnExecutionProviders: this._cfg.oidnExecutionProviders,
      verbose: this._cfg.verbose,
      debug: this._cfg.debug,
      cameraMoveResetThresholdSq: this._cfg.initTunables.cameraMoveResetThresholdSq,
      temporalAccumAlpha: this._cfg.initTunables.temporalAccumAlpha,
      ctorLights: this._ctorLights,
      ddgi: this._ddgi,
      gtaoMode: this._cfg.gtaoMode,
      diSpatialPasses: this._cfg.diSpatialPasses,
      giSpatialPasses: this._cfg.giSpatialPasses,
      // GRIS / ReSTIR-PT reuse is a COMPILE-TIME structural gate at the pipeline
      // level (selects the GI pipeline layout + shader variant). The `restirPtReuse`
      // number (0/1) also drives the UBO flag; here we forward the boolean so the
      // pipeline builds the matching layout.
      restirPtReuse: this._cfg.restirPtReuse === 1,
      // NRC live cache — same COMPILE-TIME structural gate discipline as
      // restirPtReuse. `nrcEnabled` (0/1) also drives the UBO flag; here we
      // forward the boolean so the pipeline builds the matching gi-ris layout
      // (4-group DDGI default vs 5-group inline-MLP variant).
      nrcEnabled: this._cfg.nrcEnabled === 1,
      ppgDispatchInterval: this._cfg.ppgDispatchInterval,
      regirConfig: this._cfg.regirConfig,
    };
  }

  /** Build the back-reference the {@link PipelineInitCoordinator} consumes.
   *  Live-mutable inputs are getters closing over `this`; construction-time
   *  immutables are spread from {@link _initStaticConfig}. The coordinator
   *  never sees raw field references, only the small documented surface in
   *  `HybridEngineLifecycle.ts`. */
  private _buildInitHost(): PipelineInitHost {
    const self = this;
    return {
      ...this._initStaticConfig(),
      // Pipeline initializes at the INTERNAL render size (= canvas ×
      // resolutionFactor). Equal to the canvas size on first init (factor
      // 1.0); after a factor was applied, a reset() re-inits at the live
      // internal size so the composite upscale stays correct.
      get width() { return self._internalWidth; },
      get height() { return self._internalHeight; },
      get lastScene() { return self._lastScene; },
      get primaryLightDir() { return self._primaryLightDir; },
      get primaryLightIntensity() { return self._primaryLightIntensity; },
      get preferredSwapChainFormat() { return getPreferredSwapChainFormat(); },
      get currentBvhBuffers() { return self._bvhBuffers; },
      get currentTraversalScene() { return self._ddgiTraversalScene; },

      isSceneReadyForBvh: () => self._sceneReadyForBvh(),
      coreSceneSuppliesMeshes: () => self._coreSceneSuppliesMeshes(),

      publishBvh:             (bvh) => {
        self._bvhBuffers = bvh;
        propagateBvhToGiSubsystems({
          ddgi: self._ddgi,
          rc: self._rc,
          bvhBuffers: bvh,
          lastScene: self._lastScene,
          syncDdgi: true,
          allowRcSceneRebuild: true,
          ensureThreeSceneRoot: () => self._ensureThreeSceneRoot(),
        });
      },
      publishTraversalScene:  (s)   => { self._ddgiTraversalScene = s; },
      publishPipeline:        (p)   => { self._pipeline = p; },
      rollbackBvh:            ()    => { self._bvhBuffers = null; },
      rollbackTraversalScene: ()    => { self._ddgiTraversalScene = null; },
      setState:               (s)   => { self._state = s; },
      teardownPipeline:       ()    => { self._teardownPipeline(); },
      disposeDdgi:            ()    => { self._ddgi.dispose(); },

      recordInitStart: () => {
        const d = self._dbg;
        d.initCount++;
        d.initStart = performance.now();
        console.log(`[hybrid:debug] init #${d.initCount} START`, {
          W: self._width, H: self._height, device: !!self._device,
          t: d.initStart.toFixed(0),
        });
      },
      recordInitComplete: (pipelineMs, totalMs) => {
        const d = self._dbg;
        console.log(`[hybrid:debug] init #${d.initCount} COMPLETE`, {
          pipelineMs: pipelineMs.toFixed(1), totalMs: totalMs.toFixed(1),
        });
      },
    };
  }

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
  //      guard inside PipelineInitCoordinator (mySeq === _initSeq) ensures the
  //      loser bootstrap chain disposes its locals — no GPU resource leak.
  //      The bootstrap is wasted work but safe.
  //
  // We could remove the bootstrap and require all hosts to call setScene
  // explicitly, but that would silently break case 1 and offer no safety
  // benefit (the init-flight guard already eliminates the race-leak class).
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
}
