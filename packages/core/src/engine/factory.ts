// Backend factory contract.
//
// Split from the original `engine.ts` (sweep A-7). Every backend exposes an
// `EngineFactory<TOptions>` whose `TOptions` narrows `EngineOptions` to the
// concrete device handle and per-backend extension config. The factory is
// async because device negotiation (adapter request, shader compilation) is
// async.

import type { Engine } from './index.js';
import type { EngineWarning } from './telemetry.js';

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
  /** Structural cap on per-path bounce count. Backends may use this to size
   *  path-state buffers or accumulator array dimensions. Per-frame
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
   *  `'neural'` — GPU U-Net denoiser. Requires backend-specific weight
   *  provisioning (e.g. `HybridEngineOptions.neuralWeights` in
   *  `@vitrum/walkaround-hybrid`). Opt-in; default remains `'atrous-variance'`.
   */
  readonly denoiser?: 'none' | 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'oidn-final' | 'neural';

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
   * 'manifold-nee':  Manifold Next-Event Estimation (Hanika et al. 2015).
   *                  At each diffuse vertex, launch a manifold walk to find
   *                  valid specular connections to sampled light positions.
   *                  Unbiased. Adds per-shading-event cost proportional to
   *                  the number of specular interfaces (typically 2–5 Newton
   *                  steps per walk attempt). May fail for highly curved or
   *                  rough specular surfaces.
   *
   * 'photon-map':    Biased photon mapping for caustics. Trace forward photons
   *                  from lights; store caustic photons in a spatial data
   *                  structure; use density estimation at diffuse shading points
   *                  to reconstruct caustic radiance. APPROXIMATE / stylized — NOT
   *                  a radiometric reference: on `pt-webgpu`, GPU-A/B'd against the
   *                  forward-traced oracle that validated MNEE it recovers only
   *                  ~21% of the true caustic energy and fires on ~1% of caustic
   *                  pixels, with a hardcoded world-unit gather radius (~6×
   *                  firing-rate swing under a scale change) and a flat brightness
   *                  fudge (~20% of its reported energy). Backends that expose it
   *                  advertise it as approximate.
   *
   * Backend note: `pt-webgpu`'s `manifold-nee` path is the VALIDATED REFERENCE
   * caustic (~98.7% oracle energy, scale-invariant) — prefer it for fidelity.
   * `pt-webgl2`'s caustic modes and `pt-webgpu`'s `photon-map` mode are
   * approximate/experimental capability rows. Evidence for the pt-webgpu numbers:
   * GPU A/B dzn RTX-4090, 2026-06-07,
   * `wsl-gpu/captures/queue-2026-06-07/photon-map/RESULTS.md`.
   *
   * Default: 'none'.
   *
   * Reference: Hanika, Droske, Fascione, "Manifold Next Event Estimation,"
   * Computer Graphics Forum 34(4), 2015. DOI: 10.1111/cgf.12681.
   */
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';

  /**
   * Caustic-strategy-specific tuning knobs. Backends ignore entries that don't
   * apply to the selected `causticStrategy`.
   *
   * Known keys:
   *  - `mneeMaxIterations` (number, default 8) — MNEE Newton iterations per
   *    manifold walk attempt. Active when `causticStrategy === 'manifold-nee'`.
   *  - `mneeMaxChainLength` (number, default 3) — Maximum specular vertices
   *    in an MNEE chain. Active when `causticStrategy === 'manifold-nee'`.
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
    [key: string]: unknown;
  }>;

  // ── Backend-specific extensions ─────────────────────────────────────────
  /** Engines look up extension keys here for backend-specific creation-time
   *  config that doesn't fit the generic options. Backends document their own
   *  keys. */
  readonly extensions?: Readonly<Record<string, unknown>>;
}
