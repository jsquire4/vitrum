/**
 * oidnBridge.ts — OIDN final-pass denoiser via ONNX Runtime Web.
 *
 * Wraps `onnxruntime-web` to run an OIDN (Open Image Denoise) ONNX model in
 * the browser for PT_FINAL one-shot denoising.  Not real-time; the host
 * triggers this once on a converged path-traced frame.
 *
 * Design decisions:
 *
 *   1. Lazy import — `onnxruntime-web` is loaded via dynamic `await import()`
 *      on first denoiseFinal call. This means the package compiles cleanly
 *      without `onnxruntime-web` installed, and the bundle-size hit (5–20 MB)
 *      only occurs when the host triggers denoising.
 *
 *   2. Session caching — the InferenceSession is created once per modelUrl
 *      and reused across calls. Hosts can call clearOIDNCache() to release
 *      memory or swap the model.
 *
 *   3. Execution providers — default order: ['webnn', 'webgpu', 'wasm']
 *      per Decision 11 (phase-6-roadmap.md §Decision log). WebNN gives
 *      near-native acceleration on Edge/Chrome with WebNN behind-flag;
 *      WebGPU is the universal fallback; WASM ensures correctness everywhere.
 *
 *   4. Model file — host-side concern. The host bundles or fetches the OIDN
 *      ONNX model (typically the albedo/normal/color UNet from the official
 *      OpenImageDenoise project: https://github.com/OpenImageDenoise/oidn).
 *      The modelUrl passed to denoiseFinal / preloadOIDNModel is the path or
 *      URL to that file (e.g., '/models/oidn_rt_hdr.onnx').
 *
 *   5. Optional peerDependency — `onnxruntime-web` is listed in package.json
 *      as an optional peerDependency. Hosts opt in by installing it. If the
 *      package is absent at runtime, denoiseFinal throws a clear error.
 *
 * Usage flow:
 *   1. (Optional) Call preloadOIDNModel({ modelUrl }) to pre-warm the ONNX
 *      runtime before the user clicks "Denoise".
 *   2. When the user triggers denoising, call denoiseFinal(inputs, opts).
 *   3. The returned Float32Array is a flat HxWx3 RGB buffer (row-major,
 *      interleaved R G B). Host converts to ImageData or Blob for display/save.
 *
 * Type-checking note:
 *   The `onnxruntime-web` types are accessed via a dynamic import type assertion
 *   using the string literal 'onnxruntime-web'. TypeScript resolves this only
 *   when the package is installed. When it is absent, the module type falls back
 *   to `unknown`, which is safe — the runtime guard below catches the missing
 *   dependency and throws a descriptive error before any Tensor operations.
 *
 * References:
 *   Intel OpenImageDenoise: https://www.openimagedenoise.org/
 *   ONNX Runtime Web: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
 *   Sprint 10b spec: plan/archive/phase-6-roadmap.md §Sprint 10b.
 *   Decision 11: WebNN executionProviders rider.
 */

// ============================================================
// Public interfaces
// ============================================================

/**
 * Input tensors for OIDN final-pass denoising.
 *
 * All Float32Arrays are flat, row-major, interleaved RGB (no alpha):
 *   index = (row * width + col) * 3 + channel  (channel: 0=R 1=G 2=B)
 *
 * The color buffer is required; normal and albedo are optional auxiliary
 * inputs that improve denoising quality when available.  The OIDN UNet
 * model from the official project supports these aux inputs via separate
 * model variants (e.g., oidn_rt_hdr_alb_nrm.onnx).  If your model
 * variant doesn't support aux inputs, pass only color.
 */
export interface OIDNDenoiseInputs {
  /** Noisy accumulated radiance.  Layout: Float32[height × width × 3], RGB. */
  readonly color: Float32Array;
  /** World-space normals at primary hit (optional aux).  Same layout as color. */
  readonly normal?: Float32Array;
  /** Unlit base color / albedo at primary hit (optional aux).  Same layout. */
  readonly albedo?: Float32Array;
  /** Frame width in pixels. */
  readonly width: number;
  /** Frame height in pixels. */
  readonly height: number;
}

interface OIDNModelTensorNames {
  /** ONNX input name for the noisy color tensor. Default: `"color"`. */
  readonly color?: string;
  /** Optional normals aux input name. Default: `"normal"`. */
  readonly normal?: string;
  /** Optional albedo aux input name. Default: `"albedo"`. */
  readonly albedo?: string;
  /**
   * Primary output tensor name from the ONNX model.
   *
   * Default (when unset): the bridge looks up the tensor named `"output"`
   * first, then falls back to the literal `"color"` key. The `"color"`
   * fallback is a documented back-compat alias for legacy exports. Setting
   * this field pins one exact output name and disables every fallback.
   */
  readonly output?: string;
}

/**
 * Options for the OIDN ONNX bridge.
 */
export interface OIDNDenoiseOptions {
  /**
   * URL or path to the bundled OIDN ONNX model file.
   * Host is responsible for making this file available (bundled asset or
   * fetched from a CDN).  A common convention: '/models/oidn_rt_hdr.onnx'.
   */
  readonly modelUrl: string;

  /**
   * ONNX Runtime Web execution providers, in priority order.
   *
   * Default (Decision 11): ['webnn', 'webgpu', 'wasm']
   *   - 'webnn'  — Near-native acceleration on Edge/Chrome with WebNN flag.
   *   - 'webgpu' — GPU-accelerated inference, universally available on WebGPU browsers.
   *   - 'wasm'   — WASM fallback, guaranteed correctness everywhere.
   *
   * Override only when you need to pin a specific provider for testing or
   * when WebNN causes model compatibility issues on a target browser.
   */
  readonly executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;

  /**
   * ONNX graph I/O tensor names — exports vary (TorchScript, ONNXRuntime, OIDN zoo).
   * Defaults match Intel OIDN ONNX examples; override when Netron shows different keys.
   */
  readonly tensorNames?: OIDNModelTensorNames;
}

// ============================================================
// Module-level cache
// ============================================================

// Cached InferenceSession-creation PROMISE keyed by modelUrl (+ EP tuple).
// Using `unknown` avoids importing onnxruntime-web types at the module level,
// which would make the package hard-fail at compile time when the optional
// peer dep is absent.
//
// We cache the in-flight Promise (not the resolved session) so two concurrent
// first-use callers share ONE `InferenceSession.create` — otherwise both pass
// the `undefined` check, each creates a session, and the overwritten one leaks
// untracked (never released). The promise is inserted synchronously BEFORE the
// await; a rejected create deletes its own entry so a transient failure does
// not poison the cache.
interface _SessionCacheEntry {
  readonly key: string;
  readonly creation: Promise<unknown>;
  leaseCount: number;
  activeUsers: number;
  evictionRequested: boolean;
  releaseScheduled: boolean;
}

const _sessionCache = new Map<string, _SessionCacheEntry>();

/**
 * One engine's ownership claim on a shared OIDN session. Releasing a lease is
 * idempotent. The underlying ORT session is released only after every engine
 * lease and every in-flight denoise call for the cache key has ended.
 */
export interface OIDNSessionLease {
  release(): void;
}

// ============================================================
// Public API
// ============================================================

/**
 * Run OIDN denoising on a single converged PT_FINAL frame.
 *
 * Lazily loads the ONNX Runtime Web package on first call.  The InferenceSession
 * is cached by modelUrl; subsequent calls with the same URL skip session creation.
 *
 * @param inputs - Noisy color buffer + optional aux G-buffers (normal, albedo).
 * @param opts   - Model URL and optional execution provider override.
 * @returns Denoised RGB float buffer in the same layout as inputs.color.
 * @throws If `onnxruntime-web` is not installed or the model file cannot be fetched.
 */
export async function denoiseFinal(
  inputs: OIDNDenoiseInputs,
  opts: OIDNDenoiseOptions,
): Promise<Float32Array> {
  const expectedValues = _validateDenoiseRequest(inputs, opts);
  const ort = await _loadORT();
  const entry = _getOrCreateSessionEntry(ort, opts);
  // Acquire the active-use claim before awaiting creation. A concurrent engine
  // dispose/clear therefore cannot release a just-resolved session in the gap
  // between the creation promise resolving and run() beginning.
  entry.activeUsers += 1;

  try {
    const session = await entry.creation;
    const { color, normal, albedo, width, height } = inputs;

    const tn = opts.tensorNames ?? {};
    const colorKey = tn.color ?? 'color';
    const normalKey = tn.normal ?? 'normal';
    const albedoKey = tn.albedo ?? 'albedo';

    const typedSession = session as _OrtSession;
    const declaredInputs = _declaredSessionInputNames(typedSession);
    if (declaredInputs !== null && !declaredInputs.has(colorKey)) {
      throw new Error(
        `[oidnBridge] ONNX model does not declare the required color input ` +
        `'${colorKey}'. Declared inputs: ${[...declaredInputs].join(', ')}`,
      );
    }

    const colorNchw = _hwcToNchw(color, height, width, 3);
    const feeds: Record<string, unknown> = {
      [colorKey]: new ort.Tensor('float32', colorNchw, [1, 3, height, width]),
    };

    const acceptsNormal = declaredInputs === null || declaredInputs.has(normalKey);
    if (normal !== undefined && !acceptsNormal) {
      throw new Error(
        `[oidnBridge] normal guidance was supplied, but input '${normalKey}' is not ` +
        `declared by the ONNX model. Configure tensorNames.normal for models using ` +
        `a different name. Declared inputs: ${[...declaredInputs].join(', ')}`,
      );
    }
    if (normal !== undefined && acceptsNormal) {
      feeds[normalKey] = new ort.Tensor('float32', _hwcToNchw(normal, height, width, 3), [
        1,
        3,
        height,
        width,
      ]);
    }
    const acceptsAlbedo = declaredInputs === null || declaredInputs.has(albedoKey);
    if (albedo !== undefined && !acceptsAlbedo) {
      throw new Error(
        `[oidnBridge] albedo guidance was supplied, but input '${albedoKey}' is not ` +
        `declared by the ONNX model. Configure tensorNames.albedo for models using ` +
        `a different name. Declared inputs: ${[...declaredInputs].join(', ')}`,
      );
    }
    if (albedo !== undefined && acceptsAlbedo) {
      feeds[albedoKey] = new ort.Tensor('float32', _hwcToNchw(albedo, height, width, 3), [
        1,
        3,
        height,
        width,
      ]);
    }

    const results = await typedSession.run(feeds);
    if (results == null || typeof results !== 'object' || Array.isArray(results)) {
      throw new TypeError('[oidnBridge] ONNX session result must be an object.');
    }

    const explicitOutputKey = tn.output;
    const outputTensor = explicitOutputKey === undefined
      ? results['output'] ?? results['color']
      : results[explicitOutputKey];
    if (outputTensor == null) {
      const expected = explicitOutputKey === undefined
        ? "'output' or legacy alias 'color'"
        : `'${explicitOutputKey}'`;
      throw new Error(
        `[oidnBridge] ONNX model output not found. ` +
        `Expected output named ${expected}. Got keys: ${Object.keys(results).join(', ')}`,
      );
    }

    _assertNchwTensorShape('model output', outputTensor.dims, height, width);
    _assertFiniteFloat32Buffer('model output', outputTensor.data, expectedValues);
    return _nchwToHwc(outputTensor.data, height, width, 3);
  } finally {
    entry.activeUsers -= 1;
    _releaseEntryIfIdle(entry);
  }
}

/**
 * Pre-warm the ONNX runtime by loading and caching the InferenceSession.
 *
 * Optional — call this before the user triggers denoising so the first
 * denoiseFinal call doesn't block on model initialization.  Useful for
 * pre-fetching the ONNX model while the PT is still accumulating samples.
 *
 * @param opts - Same options as denoiseFinal (model URL + execution providers).
 */
export async function preloadOIDNModel(opts: OIDNDenoiseOptions): Promise<void> {
  _validateOIDNOptions(opts);
  const ort = await _loadORT();
  await _getOrCreateSession(ort, opts);
}

/**
 * Preload and retain one shared OIDN session on behalf of an engine instance.
 * Every successful acquisition must be paired with lease.release().
 */
export async function acquireOIDNSession(
  opts: OIDNDenoiseOptions,
): Promise<OIDNSessionLease> {
  _validateOIDNOptions(opts);
  const ort = await _loadORT();
  const entry = _getOrCreateSessionEntry(ort, opts);
  entry.leaseCount += 1;
  try {
    await entry.creation;
  } catch (err) {
    entry.leaseCount -= 1;
    _releaseEntryIfIdle(entry);
    throw err;
  }

  let released = false;
  return Object.freeze({
    release(): void {
      if (released) return;
      released = true;
      entry.leaseCount -= 1;
      // An engine release requests retirement, but other engine leases and
      // in-flight users keep the shared session alive until they finish.
      entry.evictionRequested = true;
      _releaseEntryIfIdle(entry);
    },
  });
}

/**
 * Clear the cached InferenceSession.
 *
 * Use when:
 *   - The host is under memory pressure and denoising won't be needed again soon.
 *   - Swapping to a different OIDN model (different modelUrl).
 *
 * After calling this, the next denoiseFinal call will re-create the session
 * from scratch (including another model load).
 */
export function clearOIDNCache(): void {
  for (const entry of _sessionCache.values()) {
    entry.evictionRequested = true;
    _releaseEntryIfIdle(entry);
  }
}

// ============================================================
// Internal helpers
// ============================================================

// ── Layout transform helpers ─────────────────────────────────────────────────
//
// Exported with an underscore prefix to signal internal / test-only status
// (not part of the public API surface). Direct export allows unit testing
// the layout transform without running ONNX inference.
//
// AUDIT FIX M-3 (2026-05-09): These were previously unexported and only
// "tested" indirectly via denoiseFinal's return type. A layout-transpose bug
// would silently produce scrambled output. Now exported for direct round-trip
// verification in oidnBridge.test.ts.
export { _hwcToNchw, _nchwToHwc };

/**
 * Lazily import onnxruntime-web.
 *
 * The dynamic import keeps the package optional at both compile and bundle time:
 *   - TypeScript compiles without onnxruntime-web in devDependencies.
 *   - Bundlers (Vite, Webpack, Rollup) include it only when the host installs it.
 *   - At runtime, if the package is absent, the import() throws a module-not-found
 *     error which we catch and re-throw with a helpful message.
 *
 * The return type is `unknown` to avoid importing ORT types at module level.
 * Callers cast to the subset they need (Tensor constructor, InferenceSession).
 */
// _OrtModule is the shape of the onnxruntime-web module that we consume.
// Declared here (not imported) so the package compiles cleanly without the
// optional peer dep installed. The dynamic import below is cast to this shape
// at runtime after we confirm the module loaded successfully.
interface _OrtModule {
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  InferenceSession: {
    create: (modelUrl: string, options: Record<string, unknown>) => Promise<unknown>;
  };
}

/** Subset of onnxruntime-web's InferenceSession we depend on. */
interface _OrtSession {
  /**
   * ONNX Runtime publishes the graph's declared input names on every session.
   * Older test doubles and compatible runtimes may omit the metadata, in which
   * case the bridge preserves its historical "forward every supplied input"
   * behaviour.
   */
  readonly inputNames?: readonly string[];
  run: (
    feeds: Record<string, unknown>,
  ) => Promise<Record<string, { data: unknown; dims: unknown }>>;
}

function _declaredSessionInputNames(session: _OrtSession): ReadonlySet<string> | null {
  const value: unknown = session.inputNames;
  if (value === undefined) return null;
  if (!Array.isArray(value) || value.some((name) => typeof name !== 'string')) {
    throw new TypeError('[oidnBridge] ONNX session inputNames must be an array of strings.');
  }
  return new Set(value);
}

function _validateOIDNOptions(opts: OIDNDenoiseOptions): void {
  if (opts == null || typeof opts !== 'object' || Array.isArray(opts)) {
    throw new TypeError('[oidnBridge] options must be an object.');
  }
  const record = opts as unknown as Record<string, unknown>;
  _rejectUnknownKeys(record, ['modelUrl', 'executionProviders', 'tensorNames'], 'options');

  if (typeof record['modelUrl'] !== 'string' || record['modelUrl'].trim().length === 0) {
    throw new TypeError('[oidnBridge] modelUrl must be a non-empty string.');
  }

  const executionProviders = record['executionProviders'];
  if (executionProviders !== undefined) {
    if (!Array.isArray(executionProviders) || executionProviders.length === 0) {
      throw new TypeError(
        '[oidnBridge] executionProviders must be a non-empty array.',
      );
    }
    const seen = new Set<string>();
    const providers: unknown[] = executionProviders;
    for (const provider of providers) {
      if (provider !== 'webnn' && provider !== 'webgpu' && provider !== 'wasm') {
        throw new TypeError(
          `[oidnBridge] unsupported execution provider ${String(provider)}.`,
        );
      }
      if (seen.has(provider)) {
        throw new TypeError(
          `[oidnBridge] executionProviders contains duplicate '${provider}'.`,
        );
      }
      seen.add(provider);
    }
  }

  const tensorNames = record['tensorNames'];
  if (tensorNames !== undefined) {
    if (
      tensorNames == null ||
      typeof tensorNames !== 'object' ||
      Array.isArray(tensorNames)
    ) {
      throw new TypeError('[oidnBridge] tensorNames must be an object.');
    }
    const names = tensorNames as Record<string, unknown>;
    _rejectUnknownKeys(names, ['color', 'normal', 'albedo', 'output'], 'tensorNames');
    for (const key of ['color', 'normal', 'albedo', 'output'] as const) {
      const name = names[key];
      if (name !== undefined && (typeof name !== 'string' || name.trim().length === 0)) {
        throw new TypeError(
          `[oidnBridge] tensorNames.${key} must be a non-empty string.`,
        );
      }
    }

    const configuredColor = names['color'];
    const configuredNormal = names['normal'];
    const configuredAlbedo = names['albedo'];
    const feedNames = [
      typeof configuredColor === 'string' ? configuredColor : 'color',
      typeof configuredNormal === 'string' ? configuredNormal : 'normal',
      typeof configuredAlbedo === 'string' ? configuredAlbedo : 'albedo',
    ];
    if (new Set(feedNames).size !== feedNames.length) {
      throw new TypeError(
        '[oidnBridge] color, normal, and albedo feed tensor names must be unique.',
      );
    }
  }
}

function _rejectUnknownKeys(
  record: Record<string, unknown>,
  allowed: readonly string[],
  name: string,
): void {
  const unknownKeys = Object.keys(record).filter((key) => !allowed.includes(key));
  if (unknownKeys.length !== 0) {
    throw new TypeError(
      `[oidnBridge] ${name} contains unknown key(s): ${unknownKeys.join(', ')}.`,
    );
  }
}

function _positiveSafeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `[oidnBridge] ${name} must be a positive safe integer; received ${String(value)}.`,
    );
  }
}

function _expectedTensorValues(
  height: number,
  width: number,
  channels: number,
): number {
  _positiveSafeInteger(width, 'width');
  _positiveSafeInteger(height, 'height');
  _positiveSafeInteger(channels, 'channels');
  const pixels = width * height;
  const values = pixels * channels;
  if (!Number.isSafeInteger(pixels) || !Number.isSafeInteger(values)) {
    throw new RangeError(
      '[oidnBridge] width × height × channels exceeds the safe integer range.',
    );
  }
  return values;
}

function _assertFiniteFloat32Buffer(
  name: string,
  buffer: unknown,
  expectedValues: number,
): asserts buffer is Float32Array {
  if (!(buffer instanceof Float32Array)) {
    throw new TypeError(`[oidnBridge] ${name} must be a Float32Array.`);
  }
  if (buffer.length !== expectedValues) {
    throw new RangeError(
      `[oidnBridge] ${name} length must be exactly ${expectedValues}; received ${buffer.length}.`,
    );
  }
  for (let i = 0; i < buffer.length; i += 1) {
    const value = buffer[i]!;
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `[oidnBridge] ${name}[${i}] must be finite; received ${String(value)}.`,
      );
    }
  }
}

function _assertNchwTensorShape(
  name: string,
  dims: unknown,
  height: number,
  width: number,
): void {
  const expected = [1, 3, height, width] as const;
  const valid = Array.isArray(dims) &&
    dims.length === expected.length &&
    dims.every((dimension, index) => dimension === expected[index]);
  if (!valid) {
    const received = Array.isArray(dims) ? `[${dims.join(', ')}]` : String(dims);
    throw new RangeError(
      `[oidnBridge] ${name} dims must be exactly [${expected.join(', ')}]; received ${received}.`,
    );
  }
}

function _validateDenoiseRequest(
  inputs: OIDNDenoiseInputs,
  opts: OIDNDenoiseOptions,
): number {
  _validateOIDNOptions(opts);
  if (inputs == null || typeof inputs !== 'object') {
    throw new TypeError('[oidnBridge] inputs must be an object.');
  }
  const expectedValues = _expectedTensorValues(inputs.height, inputs.width, 3);
  _assertFiniteFloat32Buffer('color', inputs.color, expectedValues);
  if (inputs.normal !== undefined) {
    _assertFiniteFloat32Buffer('normal', inputs.normal, expectedValues);
  }
  if (inputs.albedo !== undefined) {
    _assertFiniteFloat32Buffer('albedo', inputs.albedo, expectedValues);
  }
  return expectedValues;
}

function _releaseSession(session: unknown): void {
  try {
    const release = (session as { release?: unknown }).release;
    if (typeof release === 'function') {
      const result = release.call(session) as unknown;
      if (
        result !== null &&
        (typeof result === 'object' || typeof result === 'function') &&
        typeof (result as { then?: unknown }).then === 'function'
      ) {
        void Promise.resolve(result).catch(() => {});
      }
    }
  } catch {
    // Resource cleanup is best-effort and must never create an unhandled
    // rejection from the cache retirement microtask.
  }
}

async function _loadORT(): Promise<_OrtModule> {
  try {
    // Dynamic import of the optional peerDependency 'onnxruntime-web'.
    //
    // Conventional pattern: a direct `await import(...)` with a `@ts-expect-error`
    // suppressing the type-checker's "Cannot find module" complaint. Bundlers
    // (Vite, webpack, esbuild) handle the missing-package case via this package's
    // `peerDependenciesMeta.optional: true` declaration in package.json.
    //
    // (Previously used `new Function('id', 'return import(id)')` to evade the
    // type checker; that pattern is CSP-unsafe — `unsafe-eval` directives block
    // it. The direct-import + ts-expect-error idiom is CSP-clean and the
    // documented TypeScript pattern for optional peer deps.)
    // @ts-expect-error optional peer dependency — see peerDependenciesMeta in package.json
     
    const mod = await import('onnxruntime-web') as unknown as _OrtModule;
    return mod;
  } catch (cause) {
    throw new Error(
      `[oidnBridge] Could not load 'onnxruntime-web'. ` +
      `Install it as a project dependency: npm install onnxruntime-web. ` +
      `See Sprint 10b spec (plan/archive/phase-6-roadmap.md) for details.`,
      { cause },
    );
  }
}

/**
 * Get or create a cached ONNX InferenceSession for the given modelUrl.
 */
function _sessionCacheKey(opts: OIDNDenoiseOptions): string {
  const eps = (opts.executionProviders ?? ['webnn', 'webgpu', 'wasm']).join('|');
  return `${opts.modelUrl}\0${eps}`;
}

/**
 * Release one cached session (model URL + execution-provider tuple).
 * Prefer this over {@link clearOIDNCache} when disposing a single engine instance.
 */
export function releaseOIDNCacheEntry(
  opts: Pick<OIDNDenoiseOptions, 'modelUrl' | 'executionProviders'>,
): void {
  _validateOIDNOptions(opts);
  const key = _sessionCacheKey(opts);
  const entry = _sessionCache.get(key);
  if (entry === undefined) return;
  entry.evictionRequested = true;
  _releaseEntryIfIdle(entry);
}

function _getOrCreateSessionEntry(
  ort: Awaited<ReturnType<typeof _loadORT>>,
  opts: OIDNDenoiseOptions,
): _SessionCacheEntry {
  const { modelUrl, executionProviders = ['webnn', 'webgpu', 'wasm'] } = opts;
  const key = _sessionCacheKey(opts);

  const cached = _sessionCache.get(key);
  if (cached !== undefined) {
    // A concurrent first-use caller already started (or finished) creation;
    // share its in-flight promise instead of creating a duplicate session.
    return cached;
  }

  // Insert the creation promise SYNCHRONOUSLY (before the await) so a second
  // concurrent caller hits the cache above and shares this one create. Delete
  // the entry on rejection so a transient failure doesn't poison the cache.
  const creation = ort.InferenceSession.create(modelUrl, {
    executionProviders: executionProviders,
  });
  const entry: _SessionCacheEntry = {
    key,
    creation,
    leaseCount: 0,
    activeUsers: 0,
    evictionRequested: false,
    releaseScheduled: false,
  };
  _sessionCache.set(key, entry);
  void creation.catch(() => {
    if (_sessionCache.get(key) === entry) {
      _sessionCache.delete(key);
    }
  });
  return entry;
}

async function _getOrCreateSession(
  ort: Awaited<ReturnType<typeof _loadORT>>,
  opts: OIDNDenoiseOptions,
): Promise<unknown> {
  return await _getOrCreateSessionEntry(ort, opts).creation;
}

function _releaseEntryIfIdle(entry: _SessionCacheEntry): void {
  if (!entry.evictionRequested || entry.leaseCount !== 0 || entry.activeUsers !== 0) {
    return;
  }
  if (entry.releaseScheduled) return;
  entry.releaseScheduled = true;
  if (_sessionCache.get(entry.key) === entry) {
    _sessionCache.delete(entry.key);
  }
  void entry.creation.then(
    session => _releaseSession(session),
    () => {},
  );
}

/**
 * Convert a flat HWC (height × width × channels) buffer to NCHW
 * (1 × channels × height × width) for ONNX Runtime input.
 */
function _hwcToNchw(
  src: Float32Array,
  height: number,
  width: number,
  channels: number,
): Float32Array {
  const values = _expectedTensorValues(height, width, channels);
  _assertFiniteFloat32Buffer('HWC source', src, values);
  const dst = new Float32Array(values);
  for (let c = 0; c < channels; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = (h * width + w) * channels + c;
        const dstIdx = c * height * width + h * width + w;
        dst[dstIdx] = src[srcIdx]!;
      }
    }
  }
  return dst;
}

/**
 * Convert a flat NCHW (1 × channels × height × width) buffer back to HWC
 * (height × width × channels) for the caller.
 */
function _nchwToHwc(
  src: Float32Array,
  height: number,
  width: number,
  channels: number,
): Float32Array {
  const values = _expectedTensorValues(height, width, channels);
  _assertFiniteFloat32Buffer('NCHW source', src, values);
  const dst = new Float32Array(values);
  for (let c = 0; c < channels; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = c * height * width + h * width + w;
        const dstIdx = (h * width + w) * channels + c;
        dst[dstIdx] = src[srcIdx]!;
      }
    }
  }
  return dst;
}
