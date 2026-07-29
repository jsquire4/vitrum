// @internal — pt-webgl2 backend constructor.
//
// Extracted from createEngine.ts (C1 / D1.6) to keep the public facade thin.
// The WebGL2 path-tracer is always dynamically imported to keep it out of the
// module graph for hosts that only ever take the WebGPU path — this module
// preserves that isolation seam.
//
// I1.4 — the dynamic-import cast now uses `EngineFactory<PTEngineWebGL2Options>`
// from @vitrum/core instead of the inline `PtWebgl2ModuleLike` + `as unknown` hack,
// so factory-signature drift fails typecheck rather than silently casting.

import type { Scene, Engine } from '@vitrum/core';
import type { EngineFactory } from '@vitrum/core';
import type { PTEngineWebGL2Options } from '@vitrum/pt-webgl2';
import {
  attachBackendId,
  BackendUnavailableError,
  resolveAdvancedForBackend,
  stripOwnershipCriticalKeys,
  wrapWithIdempotentDispose,
  type CreateEngineOptions,
  type EngineWithBackendId,
  type WebGL2PathTracerAdvancedOptions,
} from '../createEngineInternals.js';

/** Shape of the lazily-imported @vitrum/pt-webgl2 module.
 *  Using `EngineFactory<PTEngineWebGL2Options>` ensures factory-signature drift
 *  fails at compile time rather than casting through `unknown`.
 *  @internal */
interface PtWebgl2Module {
  readonly createPTEngine_WebGL2: EngineFactory<PTEngineWebGL2Options>;
  readonly validateWebgl2AdvancedOptions: (
    opts: WebGL2PathTracerAdvancedOptions,
  ) => void;
}

function createWebGL2ContextForCanvas(canvas: HTMLCanvasElement): WebGL2RenderingContext {
  const gl = canvas.getContext('webgl2', {
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (gl == null) {
    throw new BackendUnavailableError(
      'pt-webgl2',
      'createEngine: WebGL2 is unavailable; canvas.getContext("webgl2") returned null.',
    );
  }
  return gl;
}

/** @internal — pt-webgl2 backend constructor. */
export async function constructPathTracer(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
): Promise<Engine> {
  const advancedWebGL2Raw = resolveAdvancedForBackend(
    opts,
    'pt-webgl2',
  ) as WebGL2PathTracerAdvancedOptions | undefined;
  const advancedWebGL2 = stripOwnershipCriticalKeys(
    advancedWebGL2Raw as Record<string, unknown> | undefined,
    'pt-webgl2',
  ) as WebGL2PathTracerAdvancedOptions;
  const module: PtWebgl2Module = await import('@vitrum/pt-webgl2');
  module.validateWebgl2AdvancedOptions(advancedWebGL2);

  let gl: WebGL2RenderingContext;
  try {
    gl = createWebGL2ContextForCanvas(opts.canvas);
  } catch (cause) {
    if (cause instanceof BackendUnavailableError) throw cause;
    throw new BackendUnavailableError(
      'pt-webgl2',
      'createEngine: WebGL2 context acquisition failed',
      { cause },
    );
  }

  let engine: Engine | null = null;
  try {
    // V1-6 — pt-webgl2 was the only device-owning backend NOT stripping
    // ownership-critical keys (device/canvas/context) from `advanced`. Because
    // `advanced` was spread AFTER `device: gl`, a host-supplied `advanced.device`
    // could clobber the createEngine-owned WebGL2 context, causing a
    // double-dispose / owned-handle leak. Route it through the same
    // stripOwnershipCriticalKeys guard the walkaround/pt-webgpu backends use.
    const merged: PTEngineWebGL2Options = {
      device: gl,
      ...advancedWebGL2,
      ...(opts.debug != null ? { debug: opts.debug } : {}),
      ...(opts.onWarning != null ? { onWarning: opts.onWarning } : {}),
    };

    // Lazy runtime import keeps the WebGL2 path-tracer stack out of the module
    // graph for hosts that only ever take the WebGPU path.
    // I1.4 — cast via `PtWebgl2Module` (uses EngineFactory<PTEngineWebGL2Options>)
    // instead of `as unknown as PtWebgl2ModuleLike`, so the factory shape is
    // structurally verified by the TypeScript compiler.
    const { createPTEngine_WebGL2 } = module;
    engine = await createPTEngine_WebGL2(merged);
    engine.setScene(vitrumScene);

    const built = engine;
    engine = null;
    // Deleting the engine-owned programs, buffers, textures, and framebuffers is
    // sufficient teardown. Deliberately calling WEBGL_lose_context here turns an
    // ordinary dispose/recreate into an asynchronous context-loss cycle and can
    // make the replacement engine allocate against a still-lost canvas context.
    return wrapWithIdempotentDispose(built, () => {});
  } catch (err) {
    try { engine?.dispose(); } catch { /* best-effort cleanup before re-throw — ignore */ }
    throw err;
  }
}

/** Thin dispatch-table wrapper: builds, tags with backendId, returns EngineWithBackendId.
 *  @internal */
export async function constructPathTracerForDispatch(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  _aabb: import('../sceneAABB.js').SceneAABB, // unused but required by BackendConstructor signature
  _needsTlas: boolean,
): Promise<EngineWithBackendId> {
  // NOTE: no reportCreateEngineError here — every pt-webgl2 entry path goes
  // through createEngine's constructPathTracerWebGLFallback, which reports the
  // terminal failure exactly once. Reporting here too fired onError twice for
  // the same event (caught by the R5 audit). Matches the walkaround/pt-webgpu
  // dispatch wrappers, which also leave throw-reporting to the caller.
  const engine = await constructPathTracer(opts, vitrumScene);
  return attachBackendId(engine, 'pt-webgl2');
}
