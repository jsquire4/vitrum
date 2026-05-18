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

export interface OIDNModelTensorNames {
  /** ONNX input name for the noisy color tensor. Default: `"color"`. */
  readonly color?: string;
  /** Optional normals aux input name. Default: `"normal"`. */
  readonly normal?: string;
  /** Optional albedo aux input name. Default: `"albedo"`. */
  readonly albedo?: string;
  /**
   * Primary output tensor name from the ONNX model.
   * Default tries `"output"` then `"color"` if unset.
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

// Cached InferenceSession keyed by modelUrl.
// Using `unknown` avoids importing onnxruntime-web types at the module level,
// which would make the package hard-fail at compile time when the optional
// peer dep is absent.
const _sessionCache = new Map<string, unknown>();

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
  const ort = await _loadORT();
  const session = await _getOrCreateSession(ort, opts);

  const { color, normal, albedo, width, height } = inputs;

  const tn = opts.tensorNames ?? {};
  const colorKey = tn.color ?? 'color';
  const normalKey = tn.normal ?? 'normal';
  const albedoKey = tn.albedo ?? 'albedo';

  const colorNchw = _hwcToNchw(color, height, width, 3);
  const feeds: Record<string, unknown> = {
    [colorKey]: new ort.Tensor('float32', colorNchw, [1, 3, height, width]),
  };

  if (normal !== undefined) {
    feeds[normalKey] = new ort.Tensor('float32', _hwcToNchw(normal, height, width, 3), [
      1,
      3,
      height,
      width,
    ]);
  }
  if (albedo !== undefined) {
    feeds[albedoKey] = new ort.Tensor('float32', _hwcToNchw(albedo, height, width, 3), [
      1,
      3,
      height,
      width,
    ]);
  }

  const results = await (session as _OrtSession).run(feeds);

  const outputPrimaryKey = tn.output ?? 'output';
  // `tn.output ?? 'output'` covers both the explicit-name path and the
  // 'output' fallback in a single lookup; 'color' remains as a secondary
  // alias some legacy models use.
  const outputTensor = results[outputPrimaryKey] ?? results['color'];
  if (outputTensor == null) {
    throw new Error(
      `[oidnBridge] ONNX model output not found. ` +
        `Expected output named '${outputPrimaryKey}' or 'color'. Got keys: ${Object.keys(results).join(', ')}`,
    );
  }

  return _nchwToHwc(outputTensor.data, height, width, 3);
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
  const ort = await _loadORT();
  await _getOrCreateSession(ort, opts);
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
  _sessionCache.clear();
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
  run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>>;
}

async function _loadORT(): Promise<_OrtModule> {
  try {
    // Dynamic import of the optional peerDependency 'onnxruntime-web'.
    //
    // We use Function() to construct the import call indirectly, preventing
    // TypeScript's static module resolver from attempting to type-check
    // 'onnxruntime-web' at compile time. This is the standard idiom for
    // optional runtime dependencies that must not cause compile errors when
    // the package is absent.
    //
    // At runtime: if the host has installed onnxruntime-web, the import
    // succeeds and we cast to _OrtModule. If not, it throws and we rethrow
    // with a descriptive error.
    //
    // See: https://www.typescriptlang.org/docs/handbook/2/modules.html
    const importFn = new Function('id', 'return import(id)') as (id: string) => Promise<unknown>;
    const mod = await importFn('onnxruntime-web');
    return mod as _OrtModule;
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
async function _getOrCreateSession(
  ort: Awaited<ReturnType<typeof _loadORT>>,
  opts: OIDNDenoiseOptions,
): Promise<unknown> {
  const { modelUrl, executionProviders = ['webnn', 'webgpu', 'wasm'] } = opts;

  const cached = _sessionCache.get(modelUrl);
  if (cached !== undefined) {
    return cached;
  }

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: executionProviders,
  });

  _sessionCache.set(modelUrl, session);
  return session;
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
  const dst = new Float32Array(height * width * channels);
  for (let c = 0; c < channels; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = (h * width + w) * channels + c;
        const dstIdx = c * height * width + h * width + w;
        // noUncheckedIndexedAccess: indices are computed from loop bounds,
        // so they are always in range. The ?? 0 guard satisfies the type checker.
        dst[dstIdx] = src[srcIdx] ?? 0;
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
  const dst = new Float32Array(height * width * channels);
  for (let c = 0; c < channels; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = c * height * width + h * width + w;
        const dstIdx = (h * width + w) * channels + c;
        // noUncheckedIndexedAccess: indices are computed from loop bounds,
        // so they are always in range. The ?? 0 guard satisfies the type checker.
        dst[dstIdx] = src[srcIdx] ?? 0;
      }
    }
  }
  return dst;
}
