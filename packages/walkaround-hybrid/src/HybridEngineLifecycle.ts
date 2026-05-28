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
 *   b. buildBvh          — synthesize / pick the THREE root, call
 *                          buildReSTIRSceneBVH(); race-checkpoint at exit.
 *   c. publishBvh        — write `_bvhBuffers` / `_ddgiTraversalScene`
 *                          to the engine.
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

import * as THREE from 'three';
import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import { buildReSTIRSceneBVHForScene, disposeSceneBVH } from './restir/bvhCompute.js';
import type { ReSTIRBvhMode, SceneBVHBuffers } from './restir/bvhCompute.js';
import { vitrumSceneToThree, disposeVitrumThreeSceneRoot } from '@vitrum/three-bindings';
import { InferenceGraph } from './neural/InferenceGraph.js';
import type { ModelWeights } from './neural/weights.js';
import type { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { coreEmittersToDDGILights, directionalSunMultiplier } from './coreEmittersToDDGILights.js';
import type { Scene } from '@vitrum/core';

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
  /** Optional escape-hatch host-supplied THREE.Scene. */
  readonly threeScene: THREE.Scene | null;
  /** Latest vitrum scene supplied via setScene(); may be null pre-bootstrap. */
  readonly lastScene: Scene | null;
  readonly primaryLightDir: readonly [number, number, number];
  readonly primaryLightIntensity: number;
  /** Optional override from `extensions['walkaround-hybrid'].bvhMode`. */
  readonly restirBvhModeOverride: ReSTIRBvhMode | undefined;
  readonly denoiser: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural' | 'oidn-final';
  readonly neuralWeights: ModelWeights | undefined;
  readonly oidnModelUrl: string | undefined;
  readonly oidnExecutionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly verbose: boolean;
  readonly debug: boolean;
  readonly cameraMoveResetThresholdSq: number;
  readonly temporalAccumAlpha: number;
  readonly ctorLights: readonly DDGILight[];
  readonly ddgi: DDGI;
  readonly preferredSwapChainFormat: GPUTextureFormat;

  /** Scene-readiness predicate (engine combines the core-scene mesh
   *  count + optional ctor `isSceneReady` heuristic). */
  isSceneReadyForBvh(): boolean;
  /** True when the vitrum scene supplies at least one mesh primitive
   *  (drives the `vitrumSceneToThree` vs ctor `threeScene` branch). */
  coreSceneSuppliesMeshes(): boolean;

  // ── outputs (written by coordinator) ────────────────────────────────
  publishBvh(bvh: SceneBVHBuffers): void;
  publishTraversalScene(traversalScene: THREE.Scene): void;
  publishPipeline(pipeline: WalkaroundGPUPipeline): void;
  /** Clear `_bvhBuffers` back to null (post-init-race rollback). */
  rollbackBvh(): void;
  /** Clear `_ddgiTraversalScene` back to null (post-init-race rollback). */
  rollbackTraversalScene(): void;
  setState(state: 'initializing' | 'ready' | 'error' | 'disposed'): void;
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
  /** Currently-held traversal scene — same race-check role as above. */
  readonly currentTraversalScene: THREE.Scene | null;
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
  | 'threeScene'
  | 'restirBvhModeOverride'
  | 'denoiser'
  | 'neuralWeights'
  | 'oidnModelUrl'
  | 'oidnExecutionProviders'
  | 'verbose'
  | 'debug'
  | 'cameraMoveResetThresholdSq'
  | 'temporalAccumAlpha'
  | 'ctorLights'
  | 'ddgi'
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

    // Poll until scene has enough geometry (or 5s timeout).
    const pollStart = Date.now();
    let pollIters = 0;
    while (!this._disposed && mySeq === this._initSeq) {
      const elapsed = Date.now() - pollStart;
      if (elapsed >= 5_000) break;
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

    // Locals — must be disposed if we lose the race before publishing
    // to shared state.
    let bvhRoot: THREE.Object3D | null = null;
    let bvhOwnedSynthesized = false;
    let bvh: SceneBVHBuffers | null = null;
    let pipeline: WalkaroundGPUPipeline | null = null;
    let bvhPublished: SceneBVHBuffers | null = null;
    let publishedTraversalScene: THREE.Scene | null = null;
    let pipelineMs = 0;

    try {
      // ── Phase: buildBvh ──────────────────────────────────────────────
      const bvhStart = performance.now();
      if (host.coreSceneSuppliesMeshes()) {
        bvhRoot = vitrumSceneToThree(host.lastScene!);
        bvhOwnedSynthesized = true;
      } else if (host.threeScene != null) {
        bvhRoot = host.threeScene;
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
      const bvhBuildOpts = {
        primaryLightDir:       new THREE.Vector3(...host.primaryLightDir),
        primaryLightIntensity: host.primaryLightIntensity,
        ...(host.restirBvhModeOverride !== undefined
          ? { bvhMode: host.restirBvhModeOverride }
          : {}),
      };
      if (host.coreSceneSuppliesMeshes() && host.lastScene != null) {
        bvh = buildReSTIRSceneBVHForScene(host.lastScene, [bvhRoot], bvhBuildOpts);
      } else {
        bvh = buildReSTIRSceneBVHForScene(
          host.lastScene ?? {
            primitives: [],
            emitters: [],
            environment: { kind: 'none' },
          },
          [bvhRoot],
          { ...bvhBuildOpts, bvhMode: 'merged' },
        );
      }
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
      if (bvhOwnedSynthesized) {
        publishedTraversalScene = bvhRoot as THREE.Scene;
        host.publishTraversalScene(publishedTraversalScene);
        bvhOwnedSynthesized = false; // ownership transferred to engine
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
      pipeline = new WalkaroundGPUPipeline(device, host.width, host.height);
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
          // exactOptionalPropertyTypes: omit the key entirely when undefined.
          ...(inferenceGraph !== undefined ? { inferenceGraph } : {}),
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
        // And the traversal scene we published if it's still ours.
        if (publishedTraversalScene !== null
            && host.currentTraversalScene === publishedTraversalScene
            && publishedTraversalScene !== host.threeScene) {
          disposeVitrumThreeSceneRoot(publishedTraversalScene);
          host.rollbackTraversalScene();
        }
        // pipeline disposed by the finally block.
        return;
      }

      // Wire the sun intensity multiplier into DDGI. Single-count: when the
      // core scene supplies a `directional` emitter, `coreEmittersToDDGILights`
      // emits a `sun` DDGILight carrying `intensity = emitter.intensity`, so
      // the multiplier must be 1 (a multiplier of primaryLightIntensity would
      // double-apply). When NO scene directional is present, the legacy config
      // multiplier (primaryLightIntensity) is preserved — see
      // `directionalSunMultiplier`. `setSunIntensityMultiplier` is a public
      // method on ProbeUpdatePass — no cast needed.
      const sceneForSun =
        host.coreSceneSuppliesMeshes() && host.lastScene != null
          ? host.lastScene
          : null;
      host.ddgi.pass.setSunIntensityMultiplier(
        directionalSunMultiplier(sceneForSun, host.primaryLightIntensity),
      );

      // Collect the scene's analytic lights as DDGI lights. DDGI's per-probe
      // ray-cast pass uses 'sun' + 'fixture'/'teaLight' kinds — without this
      // bridge, area/point lights never reach DDGI's `evalDirectLighting`,
      // so probe rays hitting walls return zero radiance and the irradiance
      // atlas stays black → no colour bleed onto boxes, surfaces render
      // flat-gray even with DDGI mechanically running.
      //
      // Theme T16 — prefer the lossless core-emitter projection when a core
      // scene is available. `coreEmittersToDDGILights` consumes the
      // `@vitrum/core` `SceneEmitter` union directly, preserving chroma,
      // using the true emissive area `4·|uAxis × vAxis|` for rect emitters,
      // carrying the source emitter id, and — for `directional` emitters —
      // emitting a `sun` DDGILight with the emitter's REAL direction/colour
      // (replacing the packer's old hardcoded straight-down warm-white sun).
      //
      // The THREE-walk `collectDDGILightsFromThreeRoot(bvhRoot)` remains the
      // escape hatch ONLY when the host supplied a raw `threeScene` (no core
      // scene): in that case `bvhRoot === host.threeScene` and there is no
      // core emitter list to read, so we fall back to re-deriving from the
      // THREE light objects. The gate matches the BVH-source branch above
      // (`coreSceneSuppliesMeshes()` ⇒ `bvhRoot` came from `vitrumSceneToThree`).
      const ddgiSceneLights =
        sceneForSun != null
          ? coreEmittersToDDGILights(sceneForSun)
          : collectDDGILightsFromThreeRoot(bvhRoot);
      if (ddgiSceneLights.length > 0) {
        host.ddgi.setLights([...host.ctorLights, ...ddgiSceneLights]);
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
        try { pipeline.dispose(); } catch {}
      }
      if (bvh) {
        try { disposeSceneBVH(bvh); } catch {}
      }
      if (bvhRoot && bvhOwnedSynthesized) {
        try { disposeVitrumThreeSceneRoot(bvhRoot); } catch {}
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
        try { host.disposeDdgi(); } catch {}
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

/** Project `THREE.PointLight` instances to DDGI point-light fixtures. */
function collectDDGIPointLightsFromRoot(root: THREE.Object3D): DDGILight[] {
  const out: DDGILight[] = [];
  root.updateMatrixWorld(true);
  root.traverseVisible((obj) => {
    if (!(obj instanceof THREE.PointLight)) return;
    const pl = obj;
    out.push({
      kind: 'fixture',
      intensity: pl.intensity,
      on: true,
      position: { x: pl.position.x, y: pl.position.y, z: pl.position.z },
      color: { r: pl.color.r, g: pl.color.g, b: pl.color.b },
    });
  });
  return out;
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
export function collectDDGILightsFromThreeRoot(root: THREE.Object3D): DDGILight[] {
  return [
    ...collectDDGILightsFromRectAreaLights(root),
    ...collectDDGIPointLightsFromRoot(root),
  ];
}

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
