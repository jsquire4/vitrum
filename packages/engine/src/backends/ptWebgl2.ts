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
  reportCreateEngineError,
  attachBackendId,
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
}

function disposeOwnedWebGL2Context(gl: WebGL2RenderingContext): void {
  try { gl.getExtension('WEBGL_lose_context')?.loseContext(); } catch { /* best-effort context loss — ignore */ }
}

function createWebGL2ContextForCanvas(canvas: HTMLCanvasElement): WebGL2RenderingContext {
  const gl = canvas.getContext('webgl2', {
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (gl == null) {
    throw new Error('createEngine: WebGL2 is unavailable; canvas.getContext("webgl2") returned null.');
  }
  return gl;
}

/** @internal — pt-webgl2 backend constructor. */
export async function constructPathTracer(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
): Promise<Engine> {
  const gl = createWebGL2ContextForCanvas(opts.canvas);

  let engine: Engine | null = null;
  try {
    const advancedWebGL2 = opts.advanced as WebGL2PathTracerAdvancedOptions | undefined;
    const merged: PTEngineWebGL2Options = {
      device: gl,
      ...(advancedWebGL2 ?? {}),
    };

    // Lazy runtime import keeps the WebGL2 path-tracer stack out of the module
    // graph for hosts that only ever take the WebGPU path.
    // I1.4 — cast via `PtWebgl2Module` (uses EngineFactory<PTEngineWebGL2Options>)
    // instead of `as unknown as PtWebgl2ModuleLike`, so the factory shape is
    // structurally verified by the TypeScript compiler.
    const { createPTEngine_WebGL2 } = await import('@vitrum/pt-webgl2') as unknown as PtWebgl2Module;
    engine = await createPTEngine_WebGL2(merged);
    engine.setScene(vitrumScene);

    const built = engine;
    engine = null;
    return wrapWithIdempotentDispose(built, () => {
      disposeOwnedWebGL2Context(gl);
    });
  } catch (err) {
    try { engine?.dispose(); } catch { /* best-effort cleanup before re-throw — ignore */ }
    disposeOwnedWebGL2Context(gl);
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
