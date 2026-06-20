// Engine capabilities (engine → host, queried after init).
//
// Split from the original `engine.ts` (sweep A-7). Hosts query the
// `capabilities` field once after engine construction and branch UI / feature
// gating on it. The shape is intentionally read-only — capabilities are an
// engine-identity property, not a per-frame dial.

import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type { MaterialSpec } from '../scene/material.js';

export type EngineDenoiserMode =
  | 'none'
  | 'auto'
  | 'atrous'
  | 'atrous-variance'
  | 'svgf-real'
  | 'bmfr'
  | 'oidn-final'
  | 'neural';

export const ENGINE_DENOISER_MODES = Object.freeze([
  'none',
  'auto',
  'atrous',
  'atrous-variance',
  'svgf-real',
  'bmfr',
  'oidn-final',
  'neural',
] as const satisfies readonly EngineDenoiserMode[]);

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

export type BackendSupportMode =
  /** Implemented as-authored by the backend's native scene/render path. */
  | 'native'
  /** Implemented by rebuilding backend scene state instead of a targeted patch. */
  | 'fallback-rebuild'
  /** Implemented by converting the authored primitive into generated mesh data. */
  | 'fallback-generated-mesh'
  /** Implemented with a documented approximation rather than full authored fidelity. */
  | 'approximate'
  /** Not supported by this backend. */
  | 'unsupported';

export interface BackendMutationSupportDetails {
  readonly transform: BackendSupportMode;
  readonly positions: BackendSupportMode;
  readonly material: BackendSupportMode;
  readonly emitter: BackendSupportMode;
  readonly topology: BackendSupportMode;
  readonly addPrimitive: BackendSupportMode;
  readonly removePrimitive: BackendSupportMode;
  readonly environment: BackendSupportMode;
  readonly resize: BackendSupportMode;
  readonly lighting: BackendSupportMode;
}

export interface BackendSupportDetails {
  /** Per primitive-kind fidelity. Existing broad capability sets stay available
   *  for host gating; this detail map says whether support is native,
   *  fallback-generated, approximate, or unsupported. */
  readonly primitives: Readonly<Partial<Record<ScenePrimitive['kind'], BackendSupportMode>>>;
  readonly emitters: Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>;
  readonly environments: Readonly<Partial<Record<SceneEnvironment['kind'], BackendSupportMode>>>;
  readonly analyticShapes: Readonly<Partial<Record<AnalyticShape, BackendSupportMode>>>;
  /** Per-material-field fidelity rows. This map is intentionally partial:
   *  omitted fields are not yet audited in this detail table, while present
   *  fields are a machine-checkable promise. */
  readonly materials: Readonly<Partial<Record<keyof MaterialSpec, BackendSupportMode>>>;
  /** Shadow-flag fidelity rows (SHADOW-01, 2026-06-11):
   *   - `primitiveCastShadow` — `MeshPrimitive.castShadow` (+ instanced/skinned
   *     variants): castShadow:false geometry is skipped by NEE/occlusion
   *     shadow rays while staying camera/radiance-visible.
   *   - `emitterCastShadow` — `EmitterBase.castShadow`: castShadow:false
   *     emitters skip their NEE shadow test (light passes through occluders).
   *   - `receiveShadow` — `MeshPrimitive.receiveShadow`: a "receiver ignores
   *     occlusion" toggle is non-physical in a GI path tracer; all shipping
   *     backends keep it `unsupported` and warn when a scene sets it. */
  readonly shadows: Readonly<
    Partial<Record<'primitiveCastShadow' | 'emitterCastShadow' | 'receiveShadow', BackendSupportMode>>
  >;
  /** Creation-time denoiser support rows. `none` means a first-class no-denoise
   *  mode; every other row says whether selecting that `EngineOptions.denoiser`
   *  value is implemented by this backend or will be rejected/degraded. */
  readonly denoisers: Readonly<Record<EngineDenoiserMode, BackendSupportMode>>;
  readonly mutations: BackendMutationSupportDetails;
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

  /** Engine implements {@link Engine.seedAccumulator} — injecting an initial
   *  image into the accumulator as a DECAYING PRIOR (the progressive
   *  walkaround→PT handoff primitive). The seed leaves the converged mean
   *  unchanged because its virtual weight W decays as W/(W+M) → 0 against M
   *  real samples. When false or omitted, hosts must not call
   *  `seedAccumulator` (a still camera starts from a 1-sample image). Hosts
   *  MUST typeof-check `engine.seedAccumulator` before calling regardless. */
  readonly supportsAccumulatorSeed?: boolean;

  /** True if this engine exposes `getProgressiveSeedTexture()` — its latest output
   *  as a seed SOURCE for another engine's `seedAccumulator` (the source side of the
   *  progressive walkaround→PT handoff). Real-time engines set this; converged PT
   *  backends omit it. Hosts MUST typeof-check `getProgressiveSeedTexture` too. */
  readonly supportsProgressiveSeedSource?: boolean;

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

  /** Fine-grained, host-readable implementation detail for professional
   *  conformance checks. This is additive to the legacy boolean/set fields:
   *  existing hosts can keep using the coarse shape, while diagnostics and
   *  conformance tests can distinguish native support from rebuild fallbacks,
   *  generated-mesh fallbacks, approximations, and unsupported rows. */
  readonly supportDetails?: BackendSupportDetails;

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
