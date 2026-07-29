// Backend factory contract.
//
// Split from the original `engine.ts` (sweep A-7). Every backend exposes an
// `EngineFactory<TOptions>` whose `TOptions` narrows `EngineOptions` to the
// concrete device handle and per-backend extension config. The factory is
// async because device negotiation (adapter request, shader compilation) is
// async.

import type { Engine } from './index.js';
import type { EngineWarning } from './telemetry.js';
import type { EngineCausticStrategy, EngineDenoiserMode } from './capabilities.js';

/** All engine-creation factories follow this shape. The `device` is opaque at
 *  the core level; each backend narrows `device` to its own concrete type.
 *  Examples: `@vitrum/pt-webgl2` narrows to `WebGL2RenderingContext`;
 *  `@vitrum/pt-webgpu` narrows to `GPUDevice` (the backend uses compute shaders).
 *  Each backend's package documents its concrete device type via the options
 *  interface that extends `EngineOptions`.
 *
 *  `TEngine` defaults to the erased {@link Engine} contract — that is the shape
 *  the host-agnostic `createEngine` facade returns. A NAMED backend factory
 *  (e.g. `createPTEngine_WebGL2`) narrows `TEngine` to `Engine & <backend
 *  surface>` so a host that deliberately picks one backend gets its stable,
 *  backend-specific public methods typed, while the universal contract stays
 *  free of backend specifics. `TEngine` MUST extend {@link Engine}: every
 *  factory still produces an Engine first. */
export type EngineFactory<
  TOptions extends EngineOptions = EngineOptions,
  TEngine extends Engine = Engine,
> = (
  opts: TOptions,
) => Promise<TEngine>;

/** Immutable creation-time configuration passed to an engine factory. Once
 *  the engine exists, this configuration does not change.
 *
 *  Per-frame quality dials — samplesTarget, bounces, resolutionFactor,
 *  filteredGlossyFactor — are NOT engine identity and do NOT belong here.
 *  They live on `FrameInput.quality` and are supplied by the host each frame.
 *
 *  What belongs here: the device handle (engine is bound to one device for
 *  its lifetime), the denoiser pipeline structure (changing it requires
 *  shader recompilation, i.e. a new engine), structural buffer-allocation
 *  caps (`maxBounces`, `maxSamplesPerPixel` — allocators may use these to
 *  size accumulator precision or sample-counter types), and extensions
 *  (backend-specific creation-time config). */
export interface EngineOptions {
  /** The graphics device handle. Backend-specific type is enforced via
   *  package-level overloads. */
  readonly device: unknown;

  // ── Structural caps (buffer allocation upper bounds) ─────────────────────
  /** Structural cap on the renderer-family depth control. Path tracers use a
   *  per-path bounce count; other renderer families may expose a bounded
   *  quality regime and publish that interpretation through
   *  `EngineCapabilities.supportDetails.bounceSemantics`. Backends may use the
   *  cap to size path-state buffers or accumulator array dimensions. Per-frame
   *  `FrameInput.quality.bounces` is clamped to this value.
   *  Default: backend-specific (e.g., pt-webgl2 defaults to 32). */
  readonly maxBounces?: number;

  /** Structural cap on samples-per-pixel. Backends may use this to choose
   *  accumulator precision (e.g., FP16 vs FP32) or size sample-counter
   *  types. Per-frame `FrameInput.quality.samplesTarget` is clamped to this
   *  value. Default: backend-specific (e.g., pt-webgl2 defaults to 4096). */
  readonly maxSamplesPerPixel?: number;

  // ── Denoiser composition ────────────────────────────────────────────────
  /** Denoiser pipeline wired at engine creation. Changing the denoiser
   *  requires recompiling shaders and resizing auxiliary buffers — so it is
   *  a creation-time structural decision, not a per-frame dial.
   *
   *  `'svgf-real'` — full Schied 2017 SVGF with bilinear motion-vector
   *  reprojection, depth+normal+objId disocclusion test (Eq. 2), per-pixel
   *  history-length texture (Eq. 3), EMA α=max(α_min, 1/(h+1)) (Eq. 4),
   *  variance-from-moments (Eq. 5), and 7×7 spatial fallback for disoccluded
   *  pixels (§4.3). Implemented in `@vitrum/shared-denoisers` and wired in
   *  `@vitrum/walkaround-hybrid`. GPU memory budget at 1080p: ~52 MB of
   *  new persistent textures.
   *
   *  `'auto'` — backend-owned creation-time resolver. Backends that support it
   *  must report the resolved concrete mode through structured warnings; backends
   *  that do not support it must reject or degrade explicitly.
   *
   *  `'neural'` — GPU U-Net denoiser. Requires backend-specific weight
   *  provisioning (e.g. `HybridEngineOptions.neuralWeights` in
   *  `@vitrum/walkaround-hybrid`). Opt-in; default remains `'atrous-variance'`.
   */
  readonly denoiser?: EngineDenoiserMode;

  /**
   * Opt in to backend diagnostic instrumentation and readback work intended for
   * development tooling. Built-in engines expose {@link Engine.debug} only when
   * this is exactly `true`; otherwise the property is absent and optional debug
   * instrumentation is not allocated or executed.
   * {@link EngineCapabilities.debugSurface} reports whether this engine
   * instance actually exposes that surface.
   */
  readonly debug?: boolean;

  /**
   * Optional construction-time warning sink. Backends call this for warnings
   * produced before an `Engine` instance is returned and for runtime warnings
   * on engines that also expose `Engine.onWarning`.
   *
   * Existing `console.warn` output is preserved; this callback is for host
   * telemetry and integration tests that need programmatic observability.
   */
  readonly onWarning?: (warning: EngineWarning) => void;

  // ── Specular caustics strategy (RFE-05) ────────────────────────────────
  /**
   * Strategy for handling specular-chain caustic paths (LS+E, LSS+E, …).
   *
   * 'none':          No special caustic handling. Standard NEE only. Caustics
   *                  accumulate slowly via BSDF-sampled paths (may require many
   *                  thousands of samples to converge).
   *
   * 'bdpt':          Bounded bidirectional path tracing with Veach MIS. The
   *                  backend reports its stored light/eye depth limits and
   *                  endpoint/medium coverage through supportDetails.
   *
   * 'manifold-nee':  Manifold Next-Event Estimation (Hanika et al. 2015).
   *                  At each diffuse vertex, launch a manifold walk to find
   *                  valid specular connections to sampled light positions.
   *                  Backends report whether their implementation is native or
   *                  approximate. Walkaround uses finite Newton/chain caps and
   *                  a bounded SMS inverse-basin recurrence correction; that is
   *                  a supported bounded approximation, not an unbiased claim.
   *
   * 'photon-map':    Progressive stochastic photon mapping on pt-webgpu full
   *                  tier. Each iteration emits a fresh bounded photon set,
   *                  hashes it into scene-relative cells, and updates per-pixel
   *                  (tau, radius^2, N) density-estimation state. Backends that
   *                  do not implement this estimator reject the option.
   *
   * 'refractive-trace': Bounded realtime receiver-to-directional-light path
   *                  sampling through refractive interfaces. This is an
   *                  approximate finite-work estimator, NOT Manifold NEE and
   *                  not covered by MNEE's unbiasedness claim. Backends report
   *                  the exact selected strategy through capabilities.
   *
   * Default: 'none'.
   *
   * Reference: Hanika, Droske, Fascione, "Manifold Next Event Estimation,"
   * Computer Graphics Forum 34(4), 2015. DOI: 10.1111/cgf.12681.
   */
  readonly causticStrategy?: EngineCausticStrategy;

  /**
   * Caustic-strategy-specific tuning knobs. Backends validate entries against
   * the selected `causticStrategy`; unsupported keys may be rejected.
   *
   * Known keys:
   *  - `mneeMaxIterations` (number, default 8) — MNEE Newton iterations per
   *    manifold walk attempt. Active when `causticStrategy === 'manifold-nee'`.
   *  - `mneeMaxChainLength` (number, default 3) — Maximum specular vertices
   *    in an MNEE chain. Active when `causticStrategy === 'manifold-nee'`.
   *  - `mneeMultiplicityTrials` (number, default 8) — Bounded independent
   *    recurrence trials used by the SMS inverse-basin approximation.
   *
   * The `[key: string]: unknown` index signature is the **intentional extension
   * point** for future caustic strategies (e.g. photon-map radius, PPG target
   * depth). New strategies add their tuning keys here without touching the core
   * contract — the same seam pattern as the top-level {@link extensions} bag.
   * It is NOT a typo-absorbing oversight; backends validate and warn on unknown
   * keys at runtime rather than rejecting them at the type layer.
   */
  readonly causticOptions?: Readonly<{
    mneeMaxIterations?: number;
    mneeMaxChainLength?: number;
    mneeMultiplicityTrials?: number;
    [key: string]: unknown;
  }>;

  // ── Backend-specific extensions ─────────────────────────────────────────
  /** Engines look up extension keys here for backend-specific creation-time
   *  config that doesn't fit the generic options. Backends document their own
   *  keys. */
  readonly extensions?: Readonly<Record<string, unknown>>;
}
