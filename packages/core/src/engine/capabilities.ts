// Engine capabilities (engine → host, queried after init).
//
// Split from the original `engine.ts` (sweep A-7). Hosts query the
// `capabilities` field once after engine construction and branch UI / feature
// gating on it. The shape is intentionally read-only — capabilities are an
// engine-identity property, not a per-frame dial.

import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';

export interface IncrementalPatchSupport {
  /** Primitive transform-only patch path (no full scene rebuild). */
  readonly transform: boolean;
  /** Primitive positions-only patch path (same topology). */
  readonly positions: boolean;
  /** Material-only patch path without full scene replacement. */
  readonly material: boolean;
  /** Emitter patch path without full scene replacement. */
  readonly emitter: boolean;
  /** Topology-changing patch path without full scene replacement. */
  readonly topology: boolean;
}

export type FramePresentationMode =
  /** Backend requires host-supplied swap-chain textures every frame. */
  | 'swapchain-required'
  /** Backend renders to an internal texture and returns it via `primaryRadiance`. */
  | 'offscreen-texture'
  /** Backend can use either path depending on host/plumbing. */
  | 'swapchain-optional';

export interface EngineCapabilities {
  /** Engine supports `updatePrimitive` / `updateEmitter` patches, falling
   *  back to full `setScene` for unsupported diffs. When false, hosts must
   *  always call `setScene` for any change. */
  readonly supportsIncrementalScene: boolean;
  /** Granular patch matrix. When omitted, callers should assume the
   *  conservative behavior implied by `supportsIncrementalScene`. */
  readonly incrementalPatchSupport?: IncrementalPatchSupport;

  /** Engine implements the explicit whole-primitive {@link Engine.addPrimitive}
   *  / {@link Engine.removePrimitive} API — adding or removing a complete
   *  primitive from the live scene without the host re-authoring the full
   *  {@link Scene} and calling `setScene`. This is orthogonal to
   *  {@link incrementalPatchSupport}, whose `topology` flag covers COUNT-change
   *  PATCHES on an existing primitive (e.g. growing an instanced-mesh's instance
   *  list) — NOT introducing or evicting a whole primitive. When false or
   *  omitted, hosts must call `setScene` for whole-primitive add/remove. Hosts
   *  MUST typeof-check `engine.addPrimitive` / `engine.removePrimitive` before
   *  calling regardless, since this is an optional method pair. */
  readonly supportsAddRemovePrimitive?: boolean;

  /** Engine reports `FrameOutput.variance` and `FrameOutput.motionVectors`,
   *  enabling external denoisers + adaptive sampling. */
  readonly supportsAuxBuffers: boolean;

  /** Engine continues accumulating samples after temporal stability is
   *  reached (PT-style hero render). When false, engine resamples every
   *  frame (walkaround-style real-time). */
  readonly accumulates: boolean;

  /** Structural cap: the maximum samples-per-pixel this engine instance was
   *  allocated for. PT engines stop accumulating at this ceiling; walkaround
   *  engines report Infinity (they resample every frame rather than
   *  accumulating). Per-frame `FrameInput.quality.samplesTarget` is clamped
   *  to this value. */
  readonly maxSamplesPerPixel: number;

  /** Structural cap: the maximum bounces per path this engine instance was
   *  allocated for. Determined at engine creation by `EngineOptions.maxBounces`
   *  (or the backend's default if omitted). Per-frame
   *  `FrameInput.quality.bounces` is clamped to this value. */
  readonly maxBounces: number;

  /** Set of analytic-primitive `shape` values this engine supports. Each
   *  element is one of the {@link AnalyticShape} union members (rather than
   *  raw `string`) so an engine that misspells a shape ID at construction
   *  fails to typecheck instead of silently advertising support. */
  readonly supportedAnalyticShapes: ReadonlySet<AnalyticShape>;

  /** Set of {@link SceneEmitter} `kind` values this engine supports. */
  readonly supportedEmitterKinds: ReadonlySet<SceneEmitter['kind']>;
  /** Set of scene primitive kinds this backend can ingest directly from
   *  `Scene.primitives` without adapter-side throws. */
  readonly supportedPrimitiveKinds?: ReadonlySet<ScenePrimitive['kind']>;
  /** Set of environment kinds this backend can consume as authored. */
  readonly supportedEnvironmentKinds?: ReadonlySet<SceneEnvironment['kind']>;
  /** Host-facing present mode for `FrameInput.swapChainView`. */
  readonly presentationMode?: FramePresentationMode;
  /** Backend-specific feature IDs that are intentionally non-final / approximate. */
  readonly experimentalFeatures?: ReadonlySet<string>;

  // ── Specular caustics (RFE-05) ──────────────────────────────────────────
  /**
   * Whether this engine instance was created with a caustic strategy.
   * 'none' means standard NEE only; consumers should not expect fast
   * caustic convergence. Other values indicate the active strategy.
   *
   * Reference: Hanika, Droske, Fascione, "Manifold Next Event Estimation,"
   * CGF 34(4), 2015.
   */
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';

  /**
   * True when the engine exposes {@link Engine.debug} with at least one
   * implemented introspection method (W3-D8). Hosts that show dev overlays
   * use this as a structural opt-in: when false, the overlay can hide the
   * panel entirely rather than typeof-checking every method. Inferred from
   * the engine's debug surface at construction time; backends that ship
   * `debug` set this to `true`, others leave it `false`. Defaults to
   * `false` so omitting it is a safe negative report.
   */
  readonly debugSurface?: boolean;
}
