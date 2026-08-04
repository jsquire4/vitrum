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
import type { InverseGradientMethod } from '../inverse.js';

export type EngineDenoiserMode =
  | 'none'
  | 'auto'
  | 'atrous'
  | 'atrous-variance'
  | 'svgf-real'
  | 'bmfr'
  | 'oidn-final'
  | 'neural';

/**
 * Construction-time caustic estimator selected by a backend.
 *
 * `refractive-trace` is deliberately distinct from `manifold-nee`: it is a
 * bounded realtime receiver-to-directional-light refractive path sampler, not
 * Hanika's Newton manifold walk and not an unbiased MNEE implementation.
 */
export type EngineCausticStrategy =
  | 'none'
  | 'bdpt'
  | 'manifold-nee'
  | 'photon-map'
  | 'refractive-trace';

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

/**
 * Stable IDs for construction-time features that are active on a concrete
 * engine instance. This channel reports resolved runtime selection, not
 * implementation maturity or backend support limits.
 */
export const ENGINE_FEATURE_IDS = Object.freeze([
  'pt-webgpu-bdpt',
  'pt-webgpu-one-edge-gris-reconnection',
  'pt-webgpu-sobol-sampling',
  'pt-webgpu-cwbvh-closest-traversal',
  'pt-webgpu-photon-map-sppm',
  'pt-webgpu-spectral',
  'pt-webgpu-oidn-final',
  'pt-webgl2-bdpt',
  'pt-webgl2-sobol-sampling',
  'pt-webgl2-spectral',
  'pt-webgl2-oidn-final',
  'walkaround-hybrid-gris-ddgi-proxy-reuse',
  'walkaround-hybrid-ppg-guided-gi',
  'walkaround-hybrid-nrc',
  'walkaround-hybrid-radiance-cascades',
  'walkaround-hybrid-regir',
  'walkaround-hybrid-gpu-skinning',
  'walkaround-hybrid-refractive-trace-caustics',
  'walkaround-hybrid-manifold-nee-caustics',
  'walkaround-hybrid-denoiser-atrous',
  'walkaround-hybrid-denoiser-atrous-variance',
  'walkaround-hybrid-denoiser-svgf-real',
  'walkaround-hybrid-denoiser-bmfr',
  'walkaround-hybrid-denoiser-neural',
  'walkaround-hybrid-denoiser-oidn-final',
] as const);

export type EngineFeatureId = (typeof ENGINE_FEATURE_IDS)[number];

export interface IncrementalPatchSupport {
  /** Primitive transform patch accepted through the incremental API. */
  readonly transform: boolean;
  /** Primitive positions/attribute patch accepted through the incremental API. */
  readonly positions: boolean;
  /** Material patch accepted through the incremental API. */
  readonly material: boolean;
  /** Emitter patch accepted through the incremental API. */
  readonly emitter: boolean;
  /**
   * Topology/count-changing patch accepted through the incremental API.
   *
   * This boolean is an API-surface promise, not a native in-place performance
   * promise. Use `EngineCapabilities.supportDetails.mutations` to distinguish
   * `native` paths from `fallback-rebuild` paths for each backend/profile.
   */
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

/**
 * Spatial-size contract for a denoiser whose network/layout cannot accept
 * arbitrary render dimensions. Values describe the backend's INTERNAL render
 * target, after any quality-preset resolution factor has been applied.
 */
export interface DenoiserSpatialShapeRequirement {
  readonly minWidth: number;
  readonly minHeight: number;
  readonly widthMultiple: number;
  readonly heightMultiple: number;
}

export interface SamplingSequenceSupportDetails {
  /** Default sequence when the host omits its backend sampling option. */
  readonly default: 'pcg' | 'sobol';
  /** Creation-time sequence modes implemented by this backend. */
  readonly modes: Readonly<Partial<Record<'pcg' | 'sobol', BackendSupportMode>>>;
  /** Exact low-discrepancy prefix and deterministic overflow contract. */
  readonly sobol?: {
    readonly lowDiscrepancyDimensions: number;
    readonly continuation: 'independent-pcg';
    readonly sampleBlockSize: number;
    readonly frameIndexPeriod: number;
  };
}

export interface CausticStrategySupportDetail {
  readonly mode: BackendSupportMode;
  /** Exact path family owned by the strategy; intended for estimator composition. */
  readonly estimatorScope: string;
  readonly emitterKinds: Readonly<Partial<Record<SceneEmitter['kind'] | 'environment', BackendSupportMode>>>;
  readonly volumeScattering: BackendSupportMode;
  /** Creation-time features that cannot be combined without cross-technique MIS. */
  readonly incompatibleFeatures: readonly string[];
}

/** Exact strategy boundary for a bounded bidirectional path tracer. */
export interface BidirectionalPathTracingSupportDetails {
  readonly mode: 'bounded-explicit-connections';
  /** Stored light vertices include the sampled source endpoint. */
  readonly maxLightVertices: number;
  /** Stored eye vertices count scene surface/medium vertices, not the camera. */
  readonly maxEyeVertices: number;
  /** Pure eye paths remain in the ordinary eye estimator where they have support. */
  readonly pureEyeStrategy: 'partitioned-eye-estimator';
  /** Whether light-subpath vertices are projected through camera We and splatted. */
  readonly cameraSplatStrategy: 'unsupported' | 'native';
  /** Which techniques enter the power-heuristic denominator. */
  readonly misDenominator: 'sampled-strategies-only';
}

/**
 * Meaning of the generic `maxBounces` capability for a concrete renderer
 * family. Progressive path tracers use an ordinary finite path-depth cap.
 * Walkaround's numeric surface instead selects one of two DDGI feedback
 * regimes and becomes inactive when its DDGI layer is disabled.
 */
export type BounceSemanticsSupportDetails =
  | {
      readonly kind: 'path-depth';
      readonly perFrameControl: 'finite-path-depth';
    }
  | {
      readonly kind: 'ddgi-feedback';
      readonly directOnlyValue: 1;
      readonly multiBounceEquilibriumValue: 2;
      readonly inactiveWhenLayerDisabled: 'ddgi';
    };

/**
 * Exact public contract for a backend's inverse-rendering implementation.
 *
 * `createInverseSession` is an optional method and remains the coarse feature
 * gate. This detail record tells professional hosts which gradient methods are
 * actually selectable and, for path replay, the certified end-to-end domain.
 * A local derivative or shader implementation is not enough to appear in the
 * field sets: the backend must have validated the complete session gradient
 * against its forward renderer.
 */
export interface InverseRenderingSupportDetails {
  readonly methods: Readonly<Partial<Record<InverseGradientMethod, BackendSupportMode>>>;
  readonly pathReplay?: {
    /** Behavior when a requested session falls outside the certified domain. */
    readonly failurePolicy: 'error' | 'finite-difference';
    /** Material fields with end-to-end certified path-replay gradients. */
    readonly materialFields: ReadonlySet<keyof MaterialSpec>;
    /** Emitter fields with end-to-end certified path-replay gradients. */
    readonly emitterFields: ReadonlySet<'color' | 'intensity'>;
    /** Maximum forward bounce count mirrored by the certified replay pass. */
    readonly maxBounces: number;
    readonly supportsSpectral: boolean;
    readonly supportsBdpt: boolean;
    readonly supportsRestirPtReuse: boolean;
    readonly supportsCausticStrategies: boolean;
  };
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
  /** Conditional material-profile fidelity that cannot be expressed by a
   * per-field row. For example, a backend may support roughness on opaque
   * surfaces while rejecting roughness combined with transmission. */
  readonly materialProfiles?: Readonly<Partial<Record<
    | 'deltaTransmission'
    | 'roughTransmission'
    | 'layeredTransmission'
    | 'normalMappedTransmission'
    | 'participatingMedia'
    | 'faceLayers',
    BackendSupportMode
  >>>;
  /** Shadow-flag fidelity rows (SHADOW-01, 2026-06-11):
   *   - `primitiveCastShadow` — `MeshPrimitive.castShadow` (+ instanced/skinned
   *     variants): castShadow:false geometry is skipped by NEE/occlusion
   *     shadow rays while staying camera/radiance-visible.
   *   - `emitterCastShadow` — `EmitterBase.castShadow`: castShadow:false
   *     emitters skip their NEE shadow test (light passes through occluders). */
  readonly shadows: Readonly<
    Partial<Record<'primitiveCastShadow' | 'emitterCastShadow', BackendSupportMode>>
  >;
  /** Creation-time denoiser support rows. `none` means a first-class no-denoise
   *  mode; every other row says whether selecting that `EngineOptions.denoiser`
   *  value is implemented by this backend or will be rejected/degraded. */
  readonly denoisers: Readonly<Record<EngineDenoiserMode, BackendSupportMode>>;
  /** Optional per-denoiser internal-render-size restrictions. An omitted row
   *  accepts the backend's ordinary positive render dimensions. Hosts that
   *  select a listed denoiser must satisfy the declared minima and multiples. */
  readonly denoiserSpatialShapeRequirements?: Readonly<
    Partial<Record<EngineDenoiserMode, DenoiserSpatialShapeRequirement>>
  >;
  /**
   * Exact convention and dynamic-geometry boundary for
   * `FrameOutput.motionVectors`. Omitted when a backend does not expose the
   * auxiliary motion field.
   */
  readonly motionVectors?: {
    readonly units: 'pixels';
    /** The vector stored at the current pixel before any consumer-side sign. */
    readonly direction:
      | 'previous-minus-current'
      | 'current-minus-previous';
    /** Which motion source is encoded by the producer. */
    readonly geometry: 'camera-only' | 'camera-and-object';
    /**
     * Correctness policy when geometry, transforms, or skinning change.
     * `reset-history` means temporal estimators discard their prior generation
     * instead of pretending camera-only motion describes object motion.
     */
    readonly sceneMutationPolicy:
      | 'reset-history'
      | 'motion-compensated';
  };
  readonly mutations: BackendMutationSupportDetails;
  /**
   * Renderer-family interpretation of `EngineCapabilities.maxBounces`.
   * Optional for source compatibility with third-party backends; Vitrum's
   * built-in backends publish it on every live support manifest.
   */
  readonly bounceSemantics?: BounceSemanticsSupportDetails;
  /**
   * End-to-end contract for authored closed homogeneous optical media.
   *
   * `maxNestedMedia` counts simultaneously live bulk media along a path.
   * A transmissive material is bulk when it has positive authored thickness,
   * positive effective scattering (RGB override when authored, otherwise
   * scalar), positive RGB absorption derived from finite
   * attenuationColor+attenuationDistance, or spectral attenuation.
   * Positive-thickness clear
   * dielectrics therefore consume a slot for nested-IOR tracking. Only
   * reciprocal zero-thickness materials with none of those bulk payloads are
   * sheets. The topology literal is a laminar-boundary promise: every pair of
   * closed oriented volumes is spatially disjoint or one volume wholly contains
   * the other. A backend publishing this record must synchronously reject open,
   * reversed, self/intersecting, touching, non-laminar, non-deterministically
   * covered, or over-capacity scenes before mutating GPU state; a shader-only
   * overflow fallback is not conformance.
   */
  readonly opticalMedia?: {
    readonly maxNestedMedia: number;
    readonly topology: 'closed-oriented-disjoint-or-nested';
    readonly overflowPolicy: 'reject-scene';
  };
  /** Sampling-sequence support and, when applicable, Sobol overflow semantics. */
  readonly samplingSequences?: SamplingSequenceSupportDetails;
  /** Optional estimator-scope and composition constraints for caustic modes. */
  readonly causticStrategies?: Readonly<
    Partial<Record<Exclude<EngineCausticStrategy, 'none'>, CausticStrategySupportDetail>>
  >;
  /** Optional strategy-set disclosure for a separately enabled BDPT pipeline. */
  readonly bidirectionalPathTracing?: BidirectionalPathTracingSupportDetails;
  /** D1 (2026-07-20) — maximum number of `MaterialSpec.thinFilmStack.layers`
   *  this backend can represent exactly. Scenes above this capacity are rejected
   *  before upload; the backend never truncates the authored coherent stack.
   *  Additive/optional: a backend that does not declare it makes no exact
   *  coherent-layer capacity promise. Walkaround supports `thinFilmStack`
   *  approximately, but does not expose an exact stack-layer capacity.
   *  pt-webgpu = 8 (WGSL loop stride); pt-webgl2 = 35 (GLSL stride). The value
   *  MUST stay in lockstep with each backend's packer constant — pinned by each
   *  backend's `thinFilmLayerLimit.test.ts` drift-guard. */
  readonly thinFilmLayerLimit?: number;
}

export type FramePresentationMode =
  /** Backend requires host-supplied swap-chain textures every frame. */
  | 'swapchain-required'
  /** Backend renders to an internal texture and returns it via `primaryRadiance`. */
  | 'offscreen-texture'
  /** Backend can use either path depending on host/plumbing. */
  | 'swapchain-optional';

export interface EngineCapabilities {
  /** Engine supports `updatePrimitive` / `updateEmitter` patch APIs. Some
   *  accepted patch classes may rebuild internal scene state; hosts should read
   *  `supportDetails.mutations` when they need native-vs-fallback routing. When
   *  false, hosts must always call `setScene` for any change. */
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

  /** Structural cap for the renderer-family depth control. For path tracers
   *  this is the maximum bounces per path; other renderer families may map the
   *  bounded numeric surface to a quality regime. The exact interpretation is
   *  published by `supportDetails.bounceSemantics`. Determined at engine
   *  creation by `EngineOptions.maxBounces` (or the backend default).
   *  Per-frame `FrameInput.quality.bounces` is clamped to this value. */
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
  /**
   * Construction-time features actually selected on this engine instance after
   * option validation, defaulting, tier gates, and automatic resolution.
   */
  readonly activeFeatures?: ReadonlySet<EngineFeatureId>;
  /** Fine-grained, host-readable implementation detail for professional
   *  conformance checks. This is additive to the legacy boolean/set fields:
   *  existing hosts can keep using the coarse shape, while diagnostics and
   *  conformance tests can distinguish native support from rebuild fallbacks,
   *  generated-mesh fallbacks, approximations, and unsupported rows. */
  readonly supportDetails?: BackendSupportDetails;

  /** Exact inverse-rendering method/domain contract for this engine instance.
   * Omitted when the backend does not publish a machine-checkable inverse
   * domain. Hosts must still typeof-check `createInverseSession`. */
  readonly inverseRendering?: InverseRenderingSupportDetails;

  // ── Specular caustics (RFE-05) ──────────────────────────────────────────
  /**
   * Whether this engine instance was created with a caustic strategy.
   * 'none' means standard NEE only; consumers should not expect fast
   * caustic convergence. Other values indicate the active strategy.
   *
   * Reference: Hanika, Droske, Fascione, "Manifold Next Event Estimation,"
   * CGF 34(4), 2015.
   */
  readonly causticStrategy: EngineCausticStrategy;

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
