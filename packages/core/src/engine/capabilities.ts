// Engine capabilities (engine → host, queried after init).
//
// Split from the original `engine.ts` (sweep A-7). Hosts query the
// `capabilities` field once after engine construction and branch UI / feature
// gating on it. The shape is intentionally read-only — capabilities are an
// engine-identity property, not a per-frame dial.

export interface EngineCapabilities {
  /** Engine supports `updatePrimitive` / `updateEmitter` patches, falling
   *  back to full `setScene` for unsupported diffs. When false, hosts must
   *  always call `setScene` for any change. */
  readonly supportsIncrementalScene: boolean;

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

  /** Set of analytic-primitive `kind` values this engine supports. */
  readonly supportedAnalyticShapes: ReadonlySet<string>;

  /** Set of emitter `kind` values this engine supports. */
  readonly supportedEmitterKinds: ReadonlySet<string>;

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
