/**
 * HybridEngineLifecycle — async pipeline-init coordinator.
 *
 * Extracted from `HybridEngine.ts` (refactor sweep 2026-05-18). The
 * original code held three interlocking state variables for race
 * coordination directly on the engine class (`_initSeq`,
 * `_pendingTeardown`, `_initRunning`) plus a 240-line async IIFE inside
 * `_initPipeline`. That bundle owned everything from scene-readiness
 * polling to BVH construction to pipeline.initialize() to deferred-
 * teardown finalisation.
 *
 * Here, that bundle is repackaged into a {@link PipelineInitCoordinator}
 * with explicit phases. The engine owns one coordinator and delegates
 * init/dispose to it:
 *
 *   1. {@link PipelineInitCoordinator.startInit} — fire-and-forget the
 *      async init chain. Bumps the monotonic init-sequence counter,
 *      captures the new seq, and kicks off the phase machine.
 *   2. {@link PipelineInitCoordinator.requestTeardown} — synchronous
 *      dispose-side entry. Returns `true` if teardown should happen
 *      immediately, or `false` if the coordinator deferred teardown to
 *      its in-flight chain's `finally` block.
 *
 * Phase machine (internal to the IIFE):
 *
 *   a. awaitSceneReady   — poll engine.isSceneReady() (with 5s cap),
 *                          aborting on dispose or sequence-drift.
 *   b. buildBvh          — build from the core scene or pick the host THREE
 *                          root, then call buildReSTIRSceneBVH();
 *                          race-checkpoint at exit.
 *   c. publishBvh        — write `_bvhBuffers` to the engine.
 *   d. initializePipeline — instantiate WalkaroundGPUPipeline, optionally
 *                          load neural InferenceGraph, await
 *                          pipeline.initialize(); race-checkpoint at exit.
 *   e. publishPipeline   — write `_pipeline`, set engine state 'ready',
 *                          collect rect-area-light → DDGI bridge.
 *   f. finalizeOrRollback — finally block: dispose any locals that did
 *                          not get published; finalise deferred
 *                          teardown if requested.
 *
 * All inter-phase race checks (dispose flag, pending teardown, sequence
 * drift) follow the same pattern as the pre-refactor inlined code; the
 * coordinator does not alter race semantics, only shape.
 */

import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import { buildReSTIRSceneBVHForCoreScene, disposeSceneBVH } from './restir/bvhCore.js';
import type { ReSTIRBvhMode, SceneBVHBuffers } from './restir/bvhCore.js';
import { InferenceGraph } from './neural/InferenceGraph.js';
import type { ModelWeights } from './neural/weights.js';
import type { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { syncDdgiFromCoreScene } from './HybridEngineDdgiSync.js';
import type { EngineError, EngineWarning, Scene } from '@vitrum/core';

/**
 * Opaque back-reference into the engine. The coordinator only touches
 * the surface defined here; the engine implements it as light getters
 * + setters over its private fields.
 */
export interface PipelineInitHost {
  // ── inputs (read each phase) ─────────────────────────────────────────
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  /** Latest render-ingestion scene; analytic primitives have already been
   * converted to generated MeshPrimitive fallbacks. May be null pre-bootstrap. */
  readonly lastScene: Scene | null;
  readonly primaryLightDir: readonly [number, number, number];
  readonly primaryLightIntensity: number;
  /** Optional override from `extensions['walkaround-hybrid'].bvhMode`. */
  readonly restirBvhModeOverride: ReSTIRBvhMode | undefined;
  readonly denoiser: 'none' | 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final';
  readonly neuralWeights: ModelWeights | undefined;
  readonly oidnModelUrl: string | undefined;
  readonly oidnExecutionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly verbose: boolean;
  readonly debug: boolean;
  readonly cameraMoveResetThresholdSq: number;
  readonly temporalAccumAlpha: number;
  readonly checkerboardMotionThresholdSq: number;
  readonly ctorLights: readonly DDGILight[];
  readonly ddgi: DDGI;
  readonly preferredSwapChainFormat: GPUTextureFormat;
  // ── Phase-0 productization — quality-preset structural gating ──────────
  /** GTAO dispatch mode: `'on'` (half-res) / `'quarter'` / `'off'`. */
  readonly gtaoMode: 'on' | 'quarter' | 'off';
  /** ReSTIR-DI spatial-reuse ping-pong pass count (1 or 2). */
  readonly diSpatialPasses: 1 | 2;
  /** ReSTIR-GI spatial-reuse ping-pong pass count (1 or 2). */
  readonly giSpatialPasses: 1 | 2;
  /** GRIS / ReSTIR-PT reconnection-shift reuse (Lin et al. 2022) opt-in.
   *  COMPILE-TIME structural gate threaded into `pipeline.initialize`: when
   *  true the GI spatial + temporal pipelines are built with the two-group
   *  (scene-BVH) layout + GRIS shader; when false (default) they are the
   *  verbatim Sprint-17 single-group pipeline. */
  readonly restirPtReuse: boolean;
  /** NRC (Müller et al. 2021) live cache opt-in (full-tier only). COMPILE-TIME
   *  structural gate threaded into `pipeline.initialize`: when true the gi-ris
   *  pipeline is built with the 5th `@group(4)` NRC group + the inline-MLP
   *  shader variant; when false (default) it is the verbatim 4-group DDGI pass.
   *  Same compile-time discipline as `restirPtReuse` (a runtime flag binding an
   *  extra group on the default path is the GRIS-class regression). */
  readonly nrcEnabled: boolean;
  /** NRC trainer windows required before cache substitution may replace DDGI. */
  readonly nrcWarmupSteps: number;
  /** PPG (Müller 2017) guided sampling — when true the pipeline builds the
   *  ppg-update pipeline and enables the UBO gate; false = bit-identical
   *  cosine kernel. */
  readonly ppgEnabled: boolean;
  /** H47 — maximum PPG sTree spatial cells forwarded to allocatePPGResources.
   *  undefined ⇒ use the allocator's built-in default (1 024). */
  readonly ppgMaxSpatialCells: number | undefined;
  /** H29 — maximum per-cell PPG dTree nodes forwarded to shader compile and
   *  allocatePPGResources. undefined ⇒ default 341-node stride. */
  readonly ppgMaxDTreeNodesPerCell: number | undefined;
  /** PPG guide/cosine MIS mixture alpha, clamped by config. */
  readonly ppgMixAlpha: number;
  /** Checkerboard half-res shading — when true shade.wgsl AND the two DI spatial
   *  passes compact their dispatch to one checkerboard phase per frame and
   *  ResolvePass reprojects the gap; false (default) shades every pixel + passes
   *  through (bit-identical). Threaded into
   *  `pipeline.initialize({ checkerboard, checkerboardMotionThresholdSq })`.
   *  GPU-validated on dzn (~1.28× whole-frame at static/slow-motion; full-rate
   *  fallback under faster motion). */
  readonly checkerboard: boolean;
  /** PPG train-pass dispatch cadence (>= 1). The ppg-update pass dispatches
   *  on `frameCount % ppgDispatchInterval === 0`. */
  readonly ppgDispatchInterval: number;
  /** ReGIR (Boksansky 2021) grid-based DI light-selection config. `undefined`
   *  ⇒ ReGIR off (RIS uses the light-tree path). Threaded into
   *  `pipeline.initialize({ regirConfig })`. */
  readonly regirConfig: Partial<import('./pipeline/ReGIRCoordinator.js').ReGIRConfig> | undefined;

  /** Scene-readiness predicate (engine combines the core-scene mesh
   *  count + optional ctor `isSceneReady` heuristic). */
  isSceneReadyForBvh(): boolean;
  /** True when the vitrum scene supplies at least one mesh primitive. */
  coreSceneSuppliesMeshes(): boolean;

  // ── outputs (written by coordinator) ────────────────────────────────
  publishBvh(bvh: SceneBVHBuffers): void;
  publishPipeline(pipeline: WalkaroundGPUPipeline): void;
  /** Clear `_bvhBuffers` back to null (post-init-race rollback). */
  rollbackBvh(): void;
  setState(state: 'initializing' | 'ready' | 'error' | 'disposed'): void;
  /** Route async lifecycle failures through the engine's programmatic error channel. */
  reportError(error: EngineError): void;
  /** Route async lifecycle warnings through the engine's programmatic warning channel. */
  reportWarning(warning: EngineWarning): void;
  /** Engine's synchronous teardown — releases pipeline + BVH + traversal
   *  scene currently held on the engine. Called from the deferred
   *  teardown path inside the coordinator's finally block. */
  teardownPipeline(): void;
  /** Engine's DDGI dispose — called once, only if the coordinator
   *  finalises a deferred teardown. */
  disposeDdgi(): void;

  // ── debug counters (only touched when host.debug === true) ──────────
  recordInitStart(): void;
  recordInitComplete(pipelineMs: number, totalMs: number): void;

  /** Currently-held BVH buffers — coordinator's finalize step checks
   *  whether a locally-built one still matches before tearing it down. */
  readonly currentBvhBuffers: SceneBVHBuffers | null;
}

/**
 * The construction-time-immutable slice of {@link PipelineInitHost}. Every
 * field here is assigned once in the engine constructor and never mutated, so
 * the engine builds it as a plain-value snapshot spread into `_buildInitHost`
 * (rather than one live getter per field). Derived via `Pick` so it stays in
 * lockstep with the host interface — adding an immutable field to the host
 * surface that belongs here is a one-line edit to this union.
 */
export type HybridInitStaticConfig = Pick<
  PipelineInitHost,
  | 'device'
  | 'restirBvhModeOverride'
  | 'denoiser'
  | 'neuralWeights'
  | 'oidnModelUrl'
  | 'oidnExecutionProviders'
  | 'verbose'
  | 'debug'
  | 'cameraMoveResetThresholdSq'
  | 'temporalAccumAlpha'
  | 'checkerboardMotionThresholdSq'
  | 'ctorLights'
  | 'ddgi'
  | 'gtaoMode'
  | 'diSpatialPasses'
  | 'giSpatialPasses'
  | 'restirPtReuse'
  | 'nrcEnabled'
  | 'nrcWarmupSteps'
  | 'ppgEnabled'
  | 'ppgMaxSpatialCells'
  | 'ppgMaxDTreeNodesPerCell'
  | 'ppgMixAlpha'
  | 'checkerboard'
  | 'ppgDispatchInterval'
  | 'regirConfig'
>;

export class PipelineInitCoordinator {
  /** Monotonic init sequence — bumped at the start of every
   *  `startInit()` call. Each in-flight async chain captures the value
   *  at entry; before any shared-state write it re-checks `mySeq ===
   *  this._initSeq`. If the value drifted, a newer init / teardown raced
   *  ahead and the older chain MUST dispose its locals + bail without
   *  mutating shared state. */
  private _initSeq = 0;

  /** Set true by `requestTeardown()` when there's an in-flight init
   *  chain. The init chain checks this flag after every `await` and, if
   *  set, disposes any locals it owns AND finalises teardown of whatever
   *  did make it to shared state. */
  private _pendingTeardown = false;

  /** True while an init chain is mid-flight. Set true at entry to the
   *  IIFE, false in its `finally`. Read by `requestTeardown` to decide
   *  whether to defer teardown to the in-flight chain or tear down
   *  synchronously here. */
  private _initRunning = false;

  /** Set true by `requestTeardown()` to signal dispose intent. Init-chain
   *  phase checkpoints honour this in addition to seq-drift. */
  private _disposed = false;

  constructor(private readonly host: PipelineInitHost) {}

  // ── Public surface ─────────────────────────────────────────────────────

  /** True when an init chain is still running. The engine reads this to
   *  decide how to coordinate disposal. */
  get initRunning(): boolean { return this._initRunning; }
  /** True after dispose intent has been recorded. */
  get disposed(): boolean { return this._disposed; }
  /** True when the coordinator has deferred a teardown to a later
   *  init-chain finally. Surfaced for test introspection and for the
   *  engine's debug-log dispose summary. */
  get pendingTeardown(): boolean { return this._pendingTeardown; }
  /** Monotonic init-sequence counter. Surfaced for test introspection. */
  get initSeq(): number { return this._initSeq; }

  /**
   * Kick off async pipeline init. Fire-and-forget — the engine state
   * machine is the synchronisation point. Returns immediately.
   */
  startInit(): void {
    if (this._disposed) return;

    const host = this.host;
    const device = host.device;
    // Capture our sequence number — any newer startInit/requestTeardown
    // bump invalidates the writes below.
    const mySeq = ++this._initSeq;

    if (host.debug) host.recordInitStart();

    host.setState('initializing');

    // Fire-and-forget: the engine transitions to 'ready' when BVH and
    // pipeline finish setting up, or to 'error' on failure. Callers do
    // not await this — the engine state machine is the synchronisation
    // point.
    this._initRunning = true;
    void this._runInitChain(mySeq, device);
  }

  /**
   * Synchronous teardown entry. Records dispose intent. Returns:
   *   - `true`  when no init chain is in flight — caller (engine.dispose)
   *             should run its synchronous teardown immediately.
   *   - `false` when an init chain is in flight — caller should NOT run
   *             a synchronous teardown; this coordinator's finally block
   *             will tear down whatever's in shared state.
   */
  requestTeardown(): boolean {
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
      // block. We can't safely call _teardownPipeline() here because
      // the in-flight chain's `await pipeline.initialize()` may still
      // be holding a live reference to a half-built pipeline.
      this._pendingTeardown = true;
      return false;
    }
    // No in-flight init; caller tears down here and now.
    return true;
  }

  // ── Internal phase machine ─────────────────────────────────────────────

  private async _runInitChain(mySeq: number, device: GPUDevice): Promise<void> {
    const host = this.host;
    const initStart = host.debug ? performance.now() : 0;

    // Poll until scene has enough geometry (or 5s timeout). A concrete empty
    // scene is valid: it gives hosts a clean sky-only / pre-geometry mount state
    // without turning a zero-mesh scene into a delayed init error.
    const pollStart = Date.now();
    let pollIters = 0;
    while (!this._disposed && mySeq === this._initSeq) {
      const elapsed = Date.now() - pollStart;
      if (elapsed >= 5_000) break;
      if (host.lastScene != null && !host.coreSceneSuppliesMeshes()) break;
      if (host.isSceneReadyForBvh()) break;
      await new Promise<void>((r) => setTimeout(r, 50));
      pollIters++;
    }

    if (this._disposed || mySeq !== this._initSeq) {
      if (host.debug) {
        console.log('[hybrid:debug] init aborted during scene-readiness poll', {
          pollIters, disposed: this._disposed, raced: mySeq !== this._initSeq, seq: mySeq,
        });
      }
      // Clear _initRunning if we're still latest; otherwise leave it
      // alone for the racer.
      if (mySeq === this._initSeq) this._initRunning = false;
      return;
    }

    if (host.debug) {
      console.log('[hybrid:debug] scene-ready', { pollIters, elapsed: Date.now() - pollStart, seq: mySeq });
    }

    if (host.lastScene != null && !host.coreSceneSuppliesMeshes()) {
      if (host.debug) {
        console.log('[hybrid:debug] empty scene; ready without BVH/pipeline', { seq: mySeq });
      }
      host.rollbackBvh();
      host.setState('ready');
      if (mySeq === this._initSeq) this._initRunning = false;
      return;
    }

    // Locals — must be disposed if we lose the race before publishing
    // to shared state.
    let bvh: SceneBVHBuffers | null = null;
    let pipeline: WalkaroundGPUPipeline | null = null;
    let bvhPublished: SceneBVHBuffers | null = null;
    let pipelineMs = 0;

    try {
      // ── Phase: buildBvh ──────────────────────────────────────────────
      const bvhStart = performance.now();
      if (host.lastScene == null || !host.coreSceneSuppliesMeshes()) {
        throw new Error(
          '[HybridEngine] BVH source unavailable: concrete walkaround-hybrid ' +
          'requires a core Scene with mesh primitives. Convert host scene data ' +
          'to @vitrum/core Scene before calling setScene().',
        );
      }
      const bvhBuildOpts = {
        primaryLightDir: {
          x: host.primaryLightDir[0],
          y: host.primaryLightDir[1],
          z: host.primaryLightDir[2],
        },
        primaryLightIntensity: host.primaryLightIntensity,
        ...(host.restirBvhModeOverride !== undefined
          ? { bvhMode: host.restirBvhModeOverride }
          : {}),
        onWarning: (warning: EngineWarning) => host.reportWarning(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      };
      bvh = buildReSTIRSceneBVHForCoreScene(host.lastScene, bvhBuildOpts);
      const bvhMs = performance.now() - bvhStart;

      // ── Phase: publishBvh (first shared-state checkpoint) ────────────
      // If a newer setScene/reset bumped _initSeq while
      // buildReSTIRSceneBVH ran (CPU-side but not instantaneous on heavy
      // scenes), discard our work locally.
      if (this._disposed || this._pendingTeardown || mySeq !== this._initSeq) {
        if (host.debug) {
          console.log('[hybrid:debug] init lost race pre-publishBvh', {
            disposed: this._disposed, pendingTeardown: this._pendingTeardown,
            raced: mySeq !== this._initSeq, seq: mySeq,
          });
        }
        // Locals will be disposed by the finally block.
        return;
      }
      host.publishBvh(bvh);
      bvhPublished = bvh;
      bvh = null; // ownership transferred to engine

      if (host.debug) {
        console.log('[hybrid:debug] BVH built', {
          bvhMs: bvhMs.toFixed(1),
          triCount: bvhPublished.bvhNodes?.count,
          emitterCount: bvhPublished.emitters?.count,
          seq: mySeq,
        });
      }

      // ── Phase: initializePipeline ────────────────────────────────────
      pipeline = new WalkaroundGPUPipeline(device, host.width, host.height, {
        onError: (error) => host.reportError(error),
        onWarning: (warning) => host.reportWarning(warning),
      });
      const pipelineStart = performance.now();

      // T2.H2 — Neural denoiser: create + initialize InferenceGraph before
      // pipeline init.
      let inferenceGraph: InferenceGraph | undefined;
      if (host.denoiser === 'neural' && host.neuralWeights) {
        const { buildUNetSpec } = await import('./neural/unetArchitecture.js');
        inferenceGraph = new InferenceGraph(buildUNetSpec());
        await inferenceGraph.initialize(device, host.neuralWeights, host.width, host.height);
      }

      await pipeline.initialize(
        bvhPublished,
        host.preferredSwapChainFormat,
        {
          verbose: host.verbose || host.debug,
          denoiser: host.denoiser,
          cameraMoveResetThresholdSq: host.cameraMoveResetThresholdSq,
          temporalAccumAlpha: host.temporalAccumAlpha,
          // Phase-0 — quality-preset structural gating (GTAO mode + spatial
          // pass counts). The pipeline gates GTAO + slices spatial labels and
          // sizes the timestamp layout off the same config.
          gtaoMode: host.gtaoMode,
          diSpatialPasses: host.diSpatialPasses,
          giSpatialPasses: host.giSpatialPasses,
          // GRIS / ReSTIR-PT reconnection-shift reuse — COMPILE-TIME structural
          // gate (selects the GI pipeline layout + shader variant). Default
          // OFF = the verbatim Sprint-17 GI pipeline (known-good default).
          restirPtReuse: host.restirPtReuse,
          // NRC (Müller et al. 2021) — COMPILE-TIME structural gate (selects the
          // gi-ris pipeline layout: 4-group DDGI default vs 5-group inline-MLP
          // variant). Default OFF = the verbatim DDGI-estimate gi-ris pass.
          nrcEnabled: host.nrcEnabled,
          // PPG (Müller 2017) guided sampling — builds the ppg-update pipeline
          // and the UBO gate. (G-P1.1 follow-up: this forward was missing, so
          // opts.ppgEnabled died at the lite-tier guard and PPG was inert
          // through the public API.)
          ppgEnabled: host.ppgEnabled,
          // Checkerboard half-res shading. OFF (default) ⇒ shade shades every
          // pixel + ResolvePass passes through = bit-identical to the
          // pre-checkerboard pipeline.
          checkerboard: host.checkerboard,
          // Checkerboard motion fallback threshold — camera move²/frame above
          // which checkerboard is forced full-rate (finer than the temporal
          // reset). Only consulted when checkerboard is on.
          checkerboardMotionThresholdSq: host.checkerboardMotionThresholdSq,
          nrcWarmupSteps: host.nrcWarmupSteps,
          // Phase-0 — PPG train-pass cadence (ppg-update gates on
          // `frameCount % N`). Only takes effect when PPG is enabled at the
          // pipeline level; harmless (= every frame) otherwise.
          ppgDispatchInterval: host.ppgDispatchInterval,
          // H47 — PPG max sTree spatial cells. Omit when undefined so the
          // pipeline's allocatePPGResources default (1 024) applies.
          ...(host.ppgMaxSpatialCells !== undefined
            ? { ppgMaxSpatialCells: host.ppgMaxSpatialCells }
            : {}),
          // H29 — PPG max per-cell dTree nodes. Omit when undefined so the
          // shader builder/resource allocator default (341) applies.
          ...(host.ppgMaxDTreeNodesPerCell !== undefined
            ? { ppgMaxDTreeNodesPerCell: host.ppgMaxDTreeNodesPerCell }
            : {}),
          ppgMixAlpha: host.ppgMixAlpha,
          // ReGIR (Boksansky 2021) grid-based DI light selection. Omit the key
          // entirely when undefined (exactOptionalPropertyTypes) so the
          // pipeline's resolveReGIRConfig default (off) applies.
          ...(host.regirConfig !== undefined ? { regirConfig: host.regirConfig } : {}),
          // exactOptionalPropertyTypes: omit the key entirely when undefined.
          ...(inferenceGraph !== undefined ? { inferenceGraph } : {}),
          ...(host.neuralWeights !== undefined ? { neuralWeights: host.neuralWeights } : {}),
          // W11 — forward OIDN config when denoiser === 'oidn-final'.
          // Validated upstream in the constructor; here we just thread.
          ...(host.oidnModelUrl !== undefined
            ? {
                oidn: {
                  modelUrl: host.oidnModelUrl,
                  ...(host.oidnExecutionProviders !== undefined
                    ? { executionProviders: host.oidnExecutionProviders }
                    : {}),
                },
              }
            : {}),
        },
      );
      pipelineMs = performance.now() - pipelineStart;

      // ── Phase: publishPipeline (final shared-state checkpoint) ───────
      // pipeline.initialize() awaits shader compilation (~50–500 ms). A
      // newer setScene/reset/dispose raced ahead in the meantime →
      // discard locally and dispose locals in finally. We MUST NOT
      // publish `pipeline` in this case; the newer chain has its own
      // pipeline coming.
      if (this._disposed || this._pendingTeardown || mySeq !== this._initSeq) {
        if (host.debug) {
          console.log('[hybrid:debug] init lost race post-pipeline.initialize', {
            pipelineMs: pipelineMs.toFixed(1),
            disposed: this._disposed, pendingTeardown: this._pendingTeardown,
            raced: mySeq !== this._initSeq, seq: mySeq,
          });
        }
        // Also tear down the BVH we already published if it's still
        // ours; otherwise the newer chain has replaced it.
        if (host.currentBvhBuffers === bvhPublished) {
          disposeSceneBVH(bvhPublished);
          host.rollbackBvh();
        }
        // pipeline disposed by the finally block.
        return;
      }

      // DDGI light sync (H18/H41/B3 — 5-step sequence, shared with engine
      // fast-update path via syncDdgiFromCoreScene in HybridEngineDdgiSync.ts).
      // R3 B-chain step 4.
      //
      // Intentional differences vs engine path (preserved):
      //   - setLightsConditional: true (only call setLights if scene has emitters)
      //   - No primaryLightDir supplied (lifecycle does NOT call orientDdgiSunLights)
      //   - No invalidateProbeCache (fresh init — probe cache is cold already)
      //
      // When sceneForSun is null (no core mesh scene), only set the multiplier
      // (preserves the legacy config intensity); skip the 4 mesh-dependent steps.
      const sceneForSun =
        host.coreSceneSuppliesMeshes() && host.lastScene != null
          ? host.lastScene
          : null;
      if (sceneForSun != null) {
        // H18/H41/sun-single-count: steps 1–4 via shared helper.
        // setLightsConditional=true: only merges if scene has ≥1 emitter.
        // No primaryLightDir: lifecycle omits orientDdgiSunLights (engine path adds it).
        syncDdgiFromCoreScene({
          ddgi: host.ddgi,
          pipeline,
          ctorLights: host.ctorLights,
          primaryLightIntensity: host.primaryLightIntensity,
          onWarning: (warning) => host.reportWarning(warning),
          setLightsConditional: true,
          ...(bvhPublished.bvhMode === 'tlas'
            ? { tlasPrimitiveBindings: bvhPublished.primitiveTlasBindings }
            : {}),
          // primaryLightDir intentionally absent — lifecycle does not orient sun lights here
        }, sceneForSun);
      } else {
        // No core mesh scene: still set the sun intensity multiplier to the
        // config value (legacy path — no scene directional to single-count).
        host.ddgi.setSunIntensityMultiplier(host.primaryLightIntensity);
      }

      host.publishPipeline(pipeline);
      pipeline = null; // ownership transferred to engine
      host.setState('ready');

      if (host.debug) {
        const totalMs = performance.now() - initStart;
        host.recordInitComplete(pipelineMs, totalMs);
      }
    } catch (err) {
      if (!this._disposed) {
        host.setState('error');
        host.reportError({
          kind: 'render',
          message:
            '[HybridEngine] async pipeline init failed; engine state set to error. ' +
            errorMessage(err),
          fatal: true,
          raw: err,
        });
      }
      console.error(
        '[HybridEngine] init failed — engine state set to error. Call dispose() and recreate the engine to retry.',
        err,
      );
    } finally {
      // ── Phase: finalizeOrRollback ────────────────────────────────────
      // Dispose any locals that did NOT make it to shared state. After a
      // successful run all three are null (ownership transferred). On a
      // race / error / dispose, whichever weren't transferred get freed
      // here so we don't leak ~1 GB of GPU resources per loser.
      if (pipeline) {
        try { pipeline.dispose(); } catch { /* best-effort cleanup of losing init branch — ignore */ }
      }
      if (bvh) {
        try { disposeSceneBVH(bvh); } catch { /* best-effort cleanup of losing init branch — ignore */ }
      }
      // If requestTeardown() raced and left _pendingTeardown set,
      // finalise the teardown now. The newest writer (us, if we
      // published successfully) is responsible for actually tearing
      // down — we're the last live reference. Note: if a newer chain
      // is still running, its own checkpoints will see _pendingTeardown
      // and bail before publishing.
      if (this._pendingTeardown && mySeq === this._initSeq) {
        if (host.debug) {
          console.log('[hybrid:debug] init finally — finalising deferred teardown', { seq: mySeq });
        }
        host.teardownPipeline();
        // host.disposeDdgi() was deferred by requestTeardown() since
        // init was in-flight; it's safe to call now because no chain
        // is using it any more.
        try { host.disposeDdgi(); } catch { /* best-effort cleanup — ignore */ }
        host.setState('disposed');
      }
      // Always clear _initRunning at the end of OUR chain — but only if
      // we're still the latest. A newer chain will have set it back to
      // true; we MUST NOT clear another chain's flag.
      if (mySeq === this._initSeq) {
        this._initRunning = false;
      }
    }
  }
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

/**
 * Merge the host-supplied (constructor `opts.lights`) DDGI lights with the
 * scene-derived ones, de-duplicating the SUN so DDGI only ever sees ONE.
 *
 * The double-sun bug: when a host passes `opts.lights` containing a `sun`
 * (manual override) AND the core scene also carries a `directional` emitter
 * (which `coreEmittersToDDGILights` converts into a `sun` DDGILight), BOTH
 * suns get spread into DDGI's `setLights`, and the probe-update pass evaluates
 * both — double-counting the sun's contribution to every probe's irradiance.
 *
 * Precedence: the SCENE-derived directional is the physical source of truth,
 * so it wins; a host-supplied `opts.lights` sun is treated as a manual default
 * that the scene overrides. When the scene contributes a `sun`, every
 * host-supplied `sun` is dropped (with a one-time `console.warn` naming the
 * conflict). When the scene contributes NO sun, host-supplied suns pass
 * through unchanged (the legacy host-only-sun configuration still works).
 *
 * Non-sun lights (fixtures / teaLights) from both sources are always kept —
 * the dedup is sun-specific, since the sun is the only single-instance
 * directional whose contribution would visibly double.
 */
export function mergeDDGILightsDedupSun(
  ctorLights: readonly DDGILight[],
  sceneLights: readonly DDGILight[],
  options: { readonly onWarning?: (warning: EngineWarning) => void } = {},
): DDGILight[] {
  const sceneHasSun = sceneLights.some((l) => l.kind === 'sun');
  if (!sceneHasSun) {
    return [...ctorLights, ...sceneLights];
  }
  const keptCtor: DDGILight[] = [];
  let droppedHostSun = false;
  for (const l of ctorLights) {
    if (l.kind === 'sun') {
      droppedHostSun = true;
      continue;
    }
    keptCtor.push(l);
  }
  if (droppedHostSun) {
    warnHostSunOverriddenOnce(options.onWarning);
  }
  return [...keptCtor, ...sceneLights];
}

/** One-time warning when a host-supplied `opts.lights` sun is dropped in favour
 *  of the scene-derived directional. Module-level latch so a per-frame
 *  re-sync can't spam the console. */
let _warnedHostSunOverridden = false;
function warnHostSunOverriddenOnce(onWarning: ((warning: EngineWarning) => void) | undefined): void {
  if (_warnedHostSunOverridden) return;
  _warnedHostSunOverridden = true;
  const warning: EngineWarning = {
    code: 'walkaround-hybrid.ddgi-host-sun-overridden',
    backend: 'walkaround-hybrid',
    phase: 'setScene',
    method: 'syncDdgiFromCoreScene',
    message:
      '[HybridEngine] Both a host-supplied `opts.lights` sun and a scene ' +
      '`directional` emitter were present. The scene directional is the ' +
      'physical source of truth and takes precedence; the host-supplied sun ' +
      'was dropped to avoid double-counting it in DDGI. Remove the `sun` from ' +
      '`opts.lights` (or the `directional` emitter from the scene) to silence ' +
      'this warning.',
    details: {
      fallback: 'drop-host-sun',
      sourceOfTruth: 'scene-directional-emitter',
    },
  };
  if (onWarning) {
    try {
      onWarning(warning);
    } catch {
      // Host warning callbacks must not break DDGI light sync.
    }
    return;
  }
  console.warn(warning.message);
}
