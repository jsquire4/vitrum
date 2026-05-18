// Engine lifecycle contract.
//
// Design principle (the white-whale insight):
// **The engine accepts a device handle but does NOT own the device's lifetime.**
// The host owns: when the device is created, when it's lost, when it's reset.
// The engine owns: GPU resources allocated against that device (BVH buffers,
// reservoirs, accumulation textures, MLP weights, etc.).
//
// This is the contract that resolves the "cells go grey + reaccumulate from
// frame 0" bug class. The engine survives any host topology change (Canvas
// remount, route change, tab visibility) as long as the device is still alive.
// If the device IS lost, the host calls `engine.dispose()` and creates a
// fresh engine with the new device.
//
// Frame cadence is the host's responsibility. The engine never starts its
// own RAF loop. The host calls `engine.renderFrame(input)` whenever it wants
// a frame — typically inside `requestAnimationFrame`, but a host that wants
// to drive frames manually (offline render, headless test, video encoder)
// can do so without the engine fighting it.

import type { Scene, ScenePrimitive, SceneEmitter } from './scene.js';
import type { FrameInput, FrameOutput } from './frame.js';

// ────────────────────────────────────────────────────────────────────────────
// Engine state
// ────────────────────────────────────────────────────────────────────────────

export type EngineState =
  | 'uninitialized'
  | 'initializing'
  | 'ready'
  | 'paused'
  /** Unrecoverable init/runtime failure — GPU resources torn down; recreate the engine. */
  | 'error'
  | 'disposed';

// ────────────────────────────────────────────────────────────────────────────
// Engine capabilities (engine → host, queried after init)
// ────────────────────────────────────────────────────────────────────────────

export interface EngineCapabilities {
  /** Engine supports `updatePrimitive` / `updateEmitter` patches, falling
   *  back to full `setScene` for unsupported diffs. When false, hosts must
   *  always call `setScene` for any change. */
  readonly supportsIncrementalScene: boolean;

  /** Engine consumes `FrameInput.shutterTime`. */
  readonly supportsMotionBlur: boolean;

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
}

// ────────────────────────────────────────────────────────────────────────────
// Engine — the public façade
// ────────────────────────────────────────────────────────────────────────────

/** There is intentionally NO `updateOptions()` method on this interface.
 *  Per-frame quality dials (samplesTarget, bounces, resolutionFactor,
 *  filteredGlossyFactor) live on `FrameInput.quality`. The host changes them
 *  by passing a different quality payload each frame — not by mutating the
 *  engine. Creation-time configuration (`EngineOptions`) is immutable for the
 *  engine's lifetime. If a structural change is needed (different denoiser
 *  pipeline, different structural caps), the host disposes the engine and
 *  creates a fresh one. */
export interface Engine {
  readonly state: EngineState;
  readonly capabilities: EngineCapabilities;

  // ── Scene management ────────────────────────────────────────────────────

  /** Replace the entire scene. Triggers a full BVH/light-tree rebuild. Cheap
   *  if the scene hasn't changed (engines may compare structural hashes). */
  setScene(scene: Scene): void;

  /** Patch a single primitive in-place. Engine MAY internally fall back to a
   *  full `setScene` rebuild if the diff is too disruptive (e.g., changing
   *  geometry vertex counts). Available only when
   *  `capabilities.supportsIncrementalScene = true`. */
  updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;

  /** Patch a single emitter. Same incremental-fallback semantics as above. */
  updateEmitter?(id: string, patch: Partial<SceneEmitter>): void;

  /** Apply an environment-only update (HDRI texture / intensity / rotation
   *  swap, or transition to `kind: 'none'`) without rebuilding the BVH or
   *  re-uploading geometry/materials. Backends that can update the IBL
   *  uniforms in place — pt-webgl wraps WebGLPathTracer.updateEnvironment(),
   *  which costs one accumulator reset and no BVH work — implement this for
   *  fast timeOfDay scrubs on the host side. Backends without a cheap env
   *  path (current HybridEngine is reactive to its own scene-source rather
   *  than host-driven env scrubs) may omit this method; hosts MUST
   *  typeof-check before calling. */
  updateEnvironment?(env: import('./scene.js').SceneEnvironment | null): void;

  // ── Frame-level rendering ───────────────────────────────────────────────

  /** Render one sample/frame and return references to the engine's output
   *  buffers. The host owns frame cadence; this method is the host's "tick."
   *
   *  PT-style engines: each call accumulates one sample into the running
   *  buffer. `FrameOutput.isConverged` flips true when `samplesAccumulated`
   *  reaches `min(input.quality.samplesTarget ?? Infinity, capabilities.maxSamplesPerPixel)`.
   *
   *  Walkaround-style engines: each call computes one fresh frame; output
   *  buffer is overwritten. */
  renderFrame(input: FrameInput): FrameOutput;

  /** Reset the accumulator. Hosts call this when the camera moves, the scene
   *  changes, or the user wants to start over. Engines may also reset
   *  internally on `setScene`. */
  reset(): void;

  // ── Pause / resume / dispose ────────────────────────────────────────────

  /** Skip per-frame compute work but keep all GPU resources allocated.
   *  Hosts call this when the engine's output isn't visible (Canvas hidden,
   *  tab backgrounded, route navigated away). */
  pause(): void;

  /** Resume per-frame compute. Engine state goes from 'paused' → 'ready'.
   *  Accumulator state is preserved across pause/resume. */
  resume(): void;

  /** Free all engine-owned GPU resources. The device handle remains valid;
   *  the host can dispose the device separately if it owns the device's
   *  lifetime. After dispose, the engine state is `'disposed'` and no method
   *  except `state` and `capabilities` is valid. For engines that surfaced
   *  `'error'`, callers should dispose before recreating. */
  dispose(): void;

  // ── Telemetry (T3.E) ─────────────────────────────────────────────────────

  /** Subscribe to per-frame stats. Backend invokes the callback synchronously
   *  at the end of each `renderFrame()` call. Returns an unsubscribe function;
   *  call it to stop receiving stats. Subscribers MUST NOT throw — engines
   *  catch and swallow throws to keep the render loop alive. Optional: a
   *  backend that does not implement this still satisfies the contract; the
   *  host should typeof-check before calling. */
  onFrame?(cb: (stats: FrameStats) => void): () => void;

  /** Subscribe to long-running progress events (PT samples-per-pixel
   *  accumulation, denoiser convergence, DDGI warm-up). Backends fire at
   *  their natural cadence (typically once per frame for SPP; less often
   *  for warm-up). Same throw-safety + optionality semantics as
   *  {@link onFrame}. */
  onProgress?(cb: (progress: ProgressStats) => void): () => void;

  /** Optional debug-introspection surface for dev overlays. When present,
   *  exposes engine-internal state (DDGI atlases, BVH nodes, GI signal
   *  textures, denoiser toggle) that visualisation tools can blit / draw.
   *  Backends that don't implement this still satisfy the core contract;
   *  consumers MUST typeof-check before calling any method. See
   *  {@link EngineDebugSurface}. */
  debug?: EngineDebugSurface;
}

// ────────────────────────────────────────────────────────────────────────────
// Debug-introspection surface (T3.G + T3.G followup)
// ────────────────────────────────────────────────────────────────────────────

/** Optional debug-introspection surface an engine may expose via `engine.debug`.
 *  Every method is optional — dev tools call only those they need and
 *  fall back when absent. Returned WebGPU texture handles are owned by the
 *  engine (the caller MUST NOT destroy them); they're invalidated on the
 *  next setScene() / dispose(). */
export interface EngineDebugSurface {
  /** DDGI irradiance atlas (the GPUTexture the probe-update pass writes
   *  to). Returns null when DDGI is disabled or not yet initialised. */
  atlasTexture?(): GPUTexture | null;

  /** DDGI visibility-atlas companion to {@link atlasTexture}. */
  visibilityAtlasTexture?(): GPUTexture | null;

  /** Flat per-node AABB list: 8 floats per node — `[minX, minY, minZ,
   *  maxX, maxY, maxZ, depth, pad]`. Returns null when the BVH is not
   *  built or introspection is not wired. */
  bvhNodes?(): Float32Array | null;

  /** Screen-space pixel hit-test. Returns the primitive ID at (x, y) or
   *  null when nothing is hit. Used by MaterialInspector for click-pick. */
  pickPrimitive?(x: number, y: number): string | null;

  /** Current denoiser-pass enabled state; mirrors the last
   *  {@link setDenoiserEnabled} call (or the engine default). */
  isDenoiserEnabled?(): boolean;

  /** Toggle the denoiser pass for the next frame. */
  setDenoiserEnabled?(enabled: boolean): void;

  /** Per-channel GI signal textures for split-screen visualisation. Any
   *  field may be null when the backend doesn't separate that signal. */
  giSignalTextures?(): {
    direct: GPUTexture | null;
    indirect: GPUTexture | null;
    ao: GPUTexture | null;
    total: GPUTexture | null;
  } | null;
}

// ────────────────────────────────────────────────────────────────────────────
// Telemetry (T3.E)
// ────────────────────────────────────────────────────────────────────────────

/** Per-frame statistics surfaced via {@link Engine.onFrame}.  Optional
 *  fields reflect backend capability — `gpuTimeMs` requires
 *  `timestamp-query` on the WebGPU side; `passTimings` requires per-pass
 *  instrumentation; `spp` is meaningful only for accumulating engines. */
export interface FrameStats {
  /** Wall-clock duration of `renderFrame()` in milliseconds. */
  readonly frameTimeMs: number;
  /** GPU-side execution time if timestamp queries are available. */
  readonly gpuTimeMs?: number;
  /** Optional per-pass breakdown (label → milliseconds). */
  readonly passTimings?: Readonly<Record<string, number>>;
  /** Samples accumulated this frame (PT-style engines).  Walkaround engines
   *  emit `1`. */
  readonly spp?: number;
  /** BVH max depth — diagnostic for traversal cost. */
  readonly bvhDepth?: number;
  /** Approximate engine-owned GPU memory (sum of texture + buffer bytes). */
  readonly estimatedGpuMemoryBytes?: number;
}

/** Progress event surfaced via {@link Engine.onProgress}.  The discriminator
 *  is `kind`; consumers switch on it to interpret `current` / `target`. */
export interface ProgressStats {
  readonly kind: 'pt-spp' | 'denoiser-converge' | 'ddgi-warmup';
  /** Current value in the kind-appropriate unit (samples for `pt-spp`,
   *  frames for `denoiser-converge`, probe-update passes for
   *  `ddgi-warmup`). */
  readonly current: number;
  /** Target value at which `fraction` reaches 1. May be `Infinity` for
   *  open-ended walkaround warm-ups. */
  readonly target: number;
  /** Convenience: `clamp(current / target, 0, 1)`. */
  readonly fraction: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Denoiser configuration (W3-D4)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Opaque handle to neural-denoiser model weights. The concrete shape lives
 * in the backend that ships the neural denoiser (`@vitrum/walkaround-hybrid`'s
 * `ModelWeights`); core declares the slot as `unknown` so the contract does
 * not pull a backend-specific dependency in. Backends MUST runtime-check the
 * shape when they pull it out of the {@link DenoiserConfig}.
 *
 * Hosts that statically know the concrete type (e.g. they `import type
 * { ModelWeights } from '@vitrum/walkaround-hybrid'`) can narrow at the call
 * site by writing `{ kind: 'neural', weights: myModelWeights }`.
 */
export type NeuralWeights = unknown;

/**
 * Discriminated union of denoiser pipeline configurations (W3-D4).
 *
 * The discriminator is `kind`; variants that need additional config
 * (`'neural'` requires `weights`; `'oidn-final'` requires `modelUrl`)
 * encode that in the type, so the compiler enforces per-mode
 * preconditions instead of leaving them as runtime throws.
 *
 * Backends:
 *   - `'none'`             — pass-through; sample raw HDR.
 *   - `'atrous'`           — legacy 3-iter à-trous.
 *   - `'atrous-variance'`  — Welford temporal + variance-scalar + à-trous (default).
 *   - `'svgf-real'`        — Schied 2017 SVGF (reproj + moments + 7×7 + 5×atrous).
 *   - `'neural'`           — T2.H2 GPU U-Net. **Requires** `weights`.
 *   - `'oidn-final'`       — Intel Open Image Denoise final pass.
 *                            **Requires** `modelUrl` (URL of the ONNX model).
 */
export type DenoiserConfig =
  | { readonly kind: 'none' }
  | { readonly kind: 'atrous' }
  | { readonly kind: 'atrous-variance' }
  | { readonly kind: 'svgf-real' }
  | { readonly kind: 'neural'; readonly weights: NeuralWeights }
  | { readonly kind: 'oidn-final'; readonly modelUrl: string };

/** Discriminator string union extracted from {@link DenoiserConfig}.
 *  Useful for backend registries that look up implementations by string id. */
export type DenoiserKind = DenoiserConfig['kind'];

/**
 * Backwards-compat accept-shape on `EngineOptions.denoiser`. Hosts may pass
 * either the structured {@link DenoiserConfig} (preferred — type-enforces
 * per-mode required config) or the legacy bare-string form which is
 * normalised internally and `@deprecated` for new code.
 *
 * The legacy-string accept-set includes:
 *   - every {@link DenoiserKind} (forwarded directly to the DU equivalent),
 *     EXCEPT `'neural'` / `'oidn-final'` which require additional config —
 *     {@link normalizeDenoiserConfig} throws on those bare strings.
 *   - `'svgf'`, a deprecated alias for `'atrous-variance'` carried over
 *     from the original string union for hosts that have not yet migrated.
 *
 * NOTE: passing `'neural'` or `'oidn-final'` as a bare string is rejected
 * at the core normaliser because there is nowhere to thread the required
 * `weights` / `modelUrl`. The DU form is the only typed path that gets
 * per-mode config end-to-end through the contract.
 *
 * @deprecated The bare-string form is retained for compatibility only.
 * Prefer the {@link DenoiserConfig} object form (`{kind: 'atrous-variance'}`).
 */
export type Denoiser = DenoiserConfig | DenoiserKind | 'svgf';

/**
 * Normalise either accepted form of `EngineOptions.denoiser` to the
 * canonical {@link DenoiserConfig} shape.
 *
 * Throws on the two string-form variants that require additional config
 * (`'neural'` / `'oidn-final'`): a bare-string of those kinds is
 * unsatisfiable without supplying weights / modelUrl, so we fail loudly
 * at engine creation instead of letting the backend reach a misconfigured
 * runtime state.
 *
 * `'svgf'` is the legacy deprecated alias for `'atrous-variance'`;
 * normaliser accepts it silently (the calling backend is expected to log
 * its own one-time deprecation warning so the wording can include
 * backend-specific guidance like the SVGF-real upgrade path).
 *
 * Passing `undefined` resolves to `{kind: 'atrous-variance'}`.
 */
export function normalizeDenoiserConfig(d: Denoiser | undefined): DenoiserConfig {
  if (d === undefined) return { kind: 'atrous-variance' };
  if (typeof d === 'string') {
    switch (d) {
      case 'none':
      case 'atrous':
      case 'atrous-variance':
      case 'svgf-real':
        return { kind: d };
      case 'svgf':
        return { kind: 'atrous-variance' };
      case 'neural':
        throw new TypeError(
          `[@vitrum/core] denoiser: 'neural' (bare string) is not supported; ` +
          `the neural denoiser requires weights — pass ` +
          `\`denoiser: { kind: 'neural', weights }\` instead.`,
        );
      case 'oidn-final':
        throw new TypeError(
          `[@vitrum/core] denoiser: 'oidn-final' (bare string) is not supported; ` +
          `the OIDN final-pass denoiser requires a modelUrl — pass ` +
          `\`denoiser: { kind: 'oidn-final', modelUrl }\` instead.`,
        );
      default: {
        const _exhaustive: never = d;
        throw new TypeError(
          `[@vitrum/core] unknown denoiser string '${_exhaustive as string}'.`,
        );
      }
    }
  }
  return d;
}

// ────────────────────────────────────────────────────────────────────────────
// Backend factory contract
// ────────────────────────────────────────────────────────────────────────────

/** All engine-creation factories follow this shape. The `device` is opaque at
 *  the core level; each backend narrows `device` to its own concrete type.
 *  Examples: `@vitrum/pt-webgl` narrows to `THREE.WebGLRenderer` (the backend
 *  wraps three-gpu-pathtracer and bakes IBL); `@vitrum/pt-webgpu` narrows to
 *  `GPUDevice` (the backend uses compute shaders with no Three.js coupling).
 *  Each backend's package documents its concrete device type via the options
 *  interface that extends `EngineOptions`. */
export type EngineFactory<TOptions extends EngineOptions = EngineOptions> = (
  opts: TOptions,
) => Promise<Engine>;

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
   *  Default: backend-specific (e.g., pt-webgl defaults to 12). */
  readonly maxBounces?: number;

  /** Structural cap on samples-per-pixel. Backends may use this to choose
   *  accumulator precision (e.g., FP16 vs FP32) or size sample-counter
   *  types. Per-frame `FrameInput.quality.samplesTarget` is clamped to this
   *  value. Default: backend-specific (e.g., pt-webgl defaults to 4096). */
  readonly maxSamplesPerPixel?: number;

  // ── Denoiser composition ────────────────────────────────────────────────
  /** Denoiser pipeline wired at engine creation. Changing the denoiser
   *  requires recompiling shaders and resizing auxiliary buffers — so it is
   *  a creation-time structural decision, not a per-frame dial.
   *
   *  W3-D4: this field accepts either a {@link DenoiserConfig} discriminated
   *  union (preferred) or the legacy bare string (deprecated). The DU form
   *  type-encodes per-mode required config — e.g. `{kind: 'neural'}` is a
   *  compile error because `weights` is missing — so a typed caller cannot
   *  reach runtime in an unsatisfiable configuration.
   *
   *  Default when absent: `{kind: 'atrous-variance'}`. */
  readonly denoiser?: Denoiser;

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
   *                  to reconstruct caustic radiance. Biased but robust.
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
   * The signature is open-ended so new strategies (photon-map params, etc.)
   * can add keys without churning the core contract.
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
