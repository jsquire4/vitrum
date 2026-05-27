// Top-level drop-in factory.
//
// Given a canvas + scene (THREE.Scene or vitrum Scene), createEngine()
//   - probes WebGPU vs WebGL2,
//   - picks the walkaround-hybrid backend (real-time GI) or pt-webgl
//     backend (converged path tracer),
//   - derives scale-sensitive defaults (Möller-Trumbore epsilon, camera-
//     move-reset threshold, emitter dist² floor, GTAO sigma) from the
//     scene's AABB diagonal D,
//   - constructs and owns the backend's device handle (GPUDevice for
//     walkaround, THREE.WebGLRenderer for pt-webgl) so the host doesn't
//     have to plumb GPU primitives,
//   - returns the @vitrum/core Engine contract with an idempotent dispose.
//
// Note on resize: the Engine contract doesn't have a `setCanvasSize()` —
// viewport size lives on FrameInput.viewport per frame. The <VitrumCanvas>
// helper in T3.F owns the ResizeObserver + pushes width/height into the
// host's renderFrame call. createEngine() itself does NOT attach an
// observer; it just hands back an Engine.

import type { Scene, Engine } from '@vitrum/core';
import { auditSceneNeedsTlas, detectGpu } from '@vitrum/core';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import {
  createWalkaroundEngine_Hybrid,
  type HybridEngineOptions,
} from '@vitrum/walkaround-hybrid';
import {
  createPTEngine_WebGL2,
  type PTEngineWebGL2Options,
} from '@vitrum/pt-webgl';
import {
  createPTEngine_WebGPU,
  ptWebgpuRequiredLimitsForAdapter,
  type PTEngineWebGPUOptions,
} from '@vitrum/pt-webgpu';

import { computeSceneAABB, type SceneAABB } from './sceneAABB.js';
import {
  DEFAULT_PRIMARY_LIGHT_DIR,
  DEFAULT_PRIMARY_LIGHT_INTENSITY,
  DEFAULT_SKY_IRRADIANCE,
  DEFAULT_SKY_TINT,
  deriveScaleDefaults,
  pickBackend,
  type EnginePreference,
  type ScaleDefaults,
} from './createEngineScale.js';

export type { EnginePreference, ScaleDefaults };
export { pickBackend, deriveScaleDefaults };

// Deliberately structurally-typed to avoid a hard `import * as THREE` here —
// users may bring their own three.js version. The factory only reads the
// `isScene` flag (set by every THREE.Scene); it never invokes any methods.
interface ThreeSceneLike {
  readonly isScene: true;
  // remaining fields are passed through to three-bindings / threeScene
  readonly [key: string]: unknown;
}

export interface CreateEngineOptions {
  /** Canvas the engine renders into. Used to obtain the GPU context. */
  readonly canvas: HTMLCanvasElement;

  /** Scene description. Either a vitrum Scene or a THREE.Scene; THREE
   *  scenes are auto-converted via @vitrum/three-bindings. */
  readonly scene: Scene | ThreeSceneLike;

  /** Quality vs speed hint:
   *    'realtime' — prefer walkaround-hybrid (WebGPU; ~60fps target).
   *    'quality'  — prefer pt-webgl (WebGL2 path tracer; converged).
   *    'quality-webgpu' — prefer pt-webgpu when WebGPU is available, else pt-webgl.
   *    'auto'     — pick walkaround-hybrid if WebGPU + tris < 500k,
   *                 else pt-webgl. Default. */
  readonly prefer?: EnginePreference;

  /** Backend-specific overrides. Merged on top of the createEngine()-
   *  derived defaults; user-supplied keys win. Most users leave empty. */
  readonly advanced?: Partial<HybridEngineOptions> | Partial<PTEngineWebGL2Options>;

  /** Debug overlay opt-in. Forwarded to backend as `debug: true`. */
  readonly debug?: boolean;
}

export async function createEngine(opts: CreateEngineOptions): Promise<Engine> {
  if (opts.canvas == null) {
    throw new TypeError('createEngine: opts.canvas is required');
  }
  if (opts.scene == null) {
    throw new TypeError('createEngine: opts.scene is required');
  }

  const sceneInputIsThree = isThreeScene(opts.scene);
  const vitrumScene: Scene = sceneInputIsThree
    ? sceneFromThreeJS(opts.scene as unknown as Parameters<typeof sceneFromThreeJS>[0])
    : (opts.scene);

  const aabb = computeSceneAABB(vitrumScene);
  const tlasAudit = auditSceneNeedsTlas(vitrumScene);
  const gpu = await detectGpu({ publishToWindow: false });
  const backend = pickBackend(
    opts.prefer ?? 'auto',
    gpu.isWebGPU,
    aabb.triangleCount,
    tlasAudit.needsTlas,
  );
  if (tlasAudit.needsTlas && backend === 'pt-webgl') {
    console.warn(`[vitrum/createEngine] ${tlasAudit.detail}`);
  }

  if (backend === 'walkaround-hybrid') {
    return await constructWalkaround(opts, vitrumScene, aabb, sceneInputIsThree);
  }
  if (backend === 'pt-webgpu') {
    return await constructPathTracerWebGPU(opts, vitrumScene, sceneInputIsThree);
  }
  return await constructPathTracer(opts, vitrumScene, sceneInputIsThree);
}

// ────────────────────────────────────────────────────────────────────────────
// Backend constructors
// ────────────────────────────────────────────────────────────────────────────

async function constructWalkaround(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  aabb: SceneAABB,
  sceneInputIsThree: boolean,
): Promise<Engine> {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported support');
  }
  const device = await adapter.requestDevice();

  const D = aabb.diagonal;
  const scaleDefaults = deriveScaleDefaults(D);

  // The HybridEngine still expects mutable [number, number, number] for
  // these direction/tint fields (predates the readonly Vec3 alias on
  // @vitrum/core/scene). Spread into a fresh tuple to satisfy the older
  // signature without leaking mutable refs to our shared defaults.
  const primaryLightDir: [number, number, number] = [
    DEFAULT_PRIMARY_LIGHT_DIR[0], DEFAULT_PRIMARY_LIGHT_DIR[1], DEFAULT_PRIMARY_LIGHT_DIR[2],
  ];
  const skyTint: [number, number, number] = [
    DEFAULT_SKY_TINT[0], DEFAULT_SKY_TINT[1], DEFAULT_SKY_TINT[2],
  ];

  // T3.H removal: pass `threeScene` ONLY when the user gave us one — when
  // they passed a vitrum Scene the engine's setScene() path synthesizes the
  // THREE.Scene internally on first BVH build. Removes the round-trip
  // through vitrumSceneToThree() that we previously did at the facade.
  const threeSceneForCtor = sceneInputIsThree
    ? (opts.scene as unknown as Parameters<typeof createWalkaroundEngine_Hybrid>[0]['threeScene'])
    : undefined;

  const merged: HybridEngineOptions = {
    device,
    width: Math.max(1, opts.canvas.width),
    height: Math.max(1, opts.canvas.height),
    primaryLightDir,
    primaryLightIntensity: DEFAULT_PRIMARY_LIGHT_INTENSITY,
    skyTint,
    skyIrradiance: DEFAULT_SKY_IRRADIANCE,
    ...(threeSceneForCtor != null ? { threeScene: threeSceneForCtor } : {}),
    cameraMoveResetThresholdSq: scaleDefaults.cameraMoveResetThresholdSq,
    temporalAccumAlpha: scaleDefaults.temporalAccumAlpha,
    emitterDist2Floor: scaleDefaults.emitterDist2Floor,
    triIntersectEpsilon: scaleDefaults.triIntersectEpsilon,
    debug: opts.debug ?? false,
    ...(opts.advanced as Partial<HybridEngineOptions> | undefined),
  };

  const engine = await createWalkaroundEngine_Hybrid(merged);
  engine.setScene(vitrumScene);

  // A2 — configure the canvas's WebGPU context so the attachVitrum RAF tick
  // can acquire a fresh GPUTextureView per frame and pass it as
  // FrameInput.swapChainView. HybridEngine.renderFrame skips the WebGPU path
  // when input.swapChainView is undefined (HybridEngine.ts:979). Without
  // this configure step, a host using attachVitrum() against a WebGPU
  // backend gets a black canvas. We configure here (not in attachVitrum)
  // because createEngine owns the GPUDevice handle.
  try {
    const ctx = opts.canvas.getContext('webgpu');
    if (ctx != null) {
      const format = (typeof navigator !== 'undefined' && 'gpu' in navigator
        ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
            .getPreferredCanvasFormat?.() ?? ('bgra8unorm')
        : ('bgra8unorm' as GPUTextureFormat));
      ctx.configure({
        device,
        format,
        // OPAQUE compositing — matches HybridEngine.renderFrame's resolve pass
        // (which writes RGB with alpha = 1.0). PREMULTIPLIED would double-
        // composite the canvas over the page background.
        alphaMode: 'opaque',
      });
    }
  } catch {
    // Best-effort. Hosts that pre-configure their own context (or use a
    // headless test canvas with no getContext('webgpu') support) are fine —
    // attachVitrum will simply not plumb swapChainView and HybridEngine
    // will skip frames cleanly.
  }

  return wrapWithIdempotentDispose(engine, () => {
    try { device.destroy(); } catch {}
  });
}

async function constructPathTracerWebGPU(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  _sceneInputIsThree: boolean,
): Promise<Engine> {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported support');
  }
  const device = await adapter.requestDevice({
    requiredLimits: ptWebgpuRequiredLimitsForAdapter(adapter),
  });

  const merged: PTEngineWebGPUOptions = {
    device,
    ...(opts.advanced as Partial<PTEngineWebGPUOptions> | undefined),
  };

  const engine = await createPTEngine_WebGPU(merged);
  engine.setScene(vitrumScene);

  try {
    const ctx = opts.canvas.getContext('webgpu');
    if (ctx != null) {
      const format = (typeof navigator !== 'undefined' && 'gpu' in navigator
        ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
            .getPreferredCanvasFormat?.() ?? ('bgra8unorm')
        : ('bgra8unorm' as GPUTextureFormat));
      ctx.configure({ device, format, alphaMode: 'opaque' });
    }
  } catch {
    // Best-effort canvas configure for attachVitrum swap-chain plumbing.
  }

  return wrapWithIdempotentDispose(engine, () => {
    try { device.destroy(); } catch {}
  });
}

async function constructPathTracer(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  _sceneInputIsThree: boolean,
): Promise<Engine> {
  const renderer = await createWebGL2RendererForCanvas(opts.canvas);

  const merged: PTEngineWebGL2Options = {
    device: renderer,
    ...(opts.advanced as Partial<PTEngineWebGL2Options> | undefined),
  };

  const engine = await createPTEngine_WebGL2(merged);
  engine.setScene(vitrumScene);

  return wrapWithIdempotentDispose(engine, () => {
    try { renderer.dispose(); } catch {}
    // Some pt-webgl test paths hand back a renderer with forceContextLoss.
    const ext = (renderer as unknown as { forceContextLoss?: () => void }).forceContextLoss;
    if (typeof ext === 'function') {
      try { ext.call(renderer); } catch {}
    }
  });
}

async function createWebGL2RendererForCanvas(
  canvas: HTMLCanvasElement,
): Promise<import('three').WebGLRenderer> {
  // Late dynamic import keeps the @vitrum/engine bundle leaner for users
  // who only ever take the WebGPU path. The peer-dep guarantees `three`
  // resolves; if it doesn't, we surface a friendly error pointing at the
  // peer-dep block in package.json.
  let three: typeof import('three');
  try {
    three = await import('three');
  } catch (err) {
    throw new Error(
      'createEngine: failed to load three.js. @vitrum/engine has `three` as a peer dependency; install it in your host. Original error: ' + String(err),
    );
  }
  const renderer = new three.WebGLRenderer({
    canvas,
    antialias: false,
    preserveDrawingBuffer: false,
  });
  return renderer;
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function isThreeScene(s: Scene | ThreeSceneLike): s is ThreeSceneLike {
  return typeof s === 'object'
    && s != null
    && (s as { isScene?: unknown }).isScene === true;
}

/** Wrap an engine so that calling .dispose() multiple times is a no-op
 *  beyond the first call. The plan calls this out as an explicit
 *  acceptance criterion ("engine.dispose() followed by engine.dispose()
 *  is idempotent").
 *
 *  @internal Exported for unit-test access only. Not part of the public
 *  `@vitrum/engine` API surface; consumers should use {@link createEngine}
 *  / {@link attachVitrum}. */
export function wrapWithIdempotentDispose(
  engine: Engine,
  postDispose: () => void,
): Engine {
  let disposed = false;
  const patchSupport = engine.capabilities.incrementalPatchSupport;
  const primitivePatchAdvertised = patchSupport == null
    ? engine.capabilities.supportsIncrementalScene
    : (
        patchSupport.transform
        || patchSupport.positions
        || patchSupport.material
        || patchSupport.topology
      );
  const emitterPatchAdvertised = patchSupport == null
    ? engine.capabilities.supportsIncrementalScene
    : patchSupport.emitter;
  const proxy: Engine = {
    get state() { return engine.state; },
    get capabilities() { return engine.capabilities; },
    setScene(scene) { if (!disposed) engine.setScene(scene); },
    ...(primitivePatchAdvertised && engine.updatePrimitive
      ? {
          updatePrimitive: (id: string, patch: Parameters<NonNullable<Engine['updatePrimitive']>>[1]) => {
            if (!disposed) engine.updatePrimitive!(id, patch);
          },
        }
      : {}),
    ...(emitterPatchAdvertised && engine.updateEmitter
      ? {
          updateEmitter: (id: string, patch: Parameters<NonNullable<Engine['updateEmitter']>>[1]) => {
            if (!disposed) engine.updateEmitter!(id, patch);
          },
        }
      : {}),
    ...(engine.updateEnvironment
      ? {
          updateEnvironment: (env: Parameters<NonNullable<Engine['updateEnvironment']>>[0]) => {
            if (!disposed) engine.updateEnvironment!(env);
          },
        }
      : {}),
    ...(engine.setSize
      ? {
          setSize: (w: number, h: number) => {
            if (!disposed) engine.setSize!(w, h);
          },
        }
      : {}),
    ...(engine.updateLighting
      ? {
          updateLighting: (opts: Parameters<NonNullable<Engine['updateLighting']>>[0]) => {
            if (!disposed) engine.updateLighting!(opts);
          },
        }
      : {}),
    renderFrame(input) {
      if (disposed) {
        // Returning a no-op output keeps host RAF loops from crashing if
        // they race the dispose. The host is expected to stop rendering
        // when state === 'disposed'.
        return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      }
      return engine.renderFrame(input);
    },
    reset() { if (!disposed) engine.reset(); },
    pause() { if (!disposed) engine.pause(); },
    resume() { if (!disposed) engine.resume(); },
    dispose() {
      if (disposed) return;
      disposed = true;
      try { engine.dispose(); } catch {}
      try { postDispose(); } catch {}
    },
    ...(engine.onFrame
      ? {
          onFrame: (cb: Parameters<NonNullable<Engine['onFrame']>>[0]) => {
            if (disposed) return () => {};
            return engine.onFrame!(cb);
          },
        }
      : {}),
    ...(engine.onProgress
      ? {
          onProgress: (cb: Parameters<NonNullable<Engine['onProgress']>>[0]) => {
            if (disposed) return () => {};
            return engine.onProgress!(cb);
          },
        }
      : {}),
    // T3.G followup — pass the underlying engine.debug surface through
    // unchanged. Methods are bound to the engine instance, so calling
    // proxy.debug.atlasTexture() reads live state. After dispose, the
    // surface still exists but most methods will return null / empty
    // because the underlying _ddgi / _pipeline / _bvhBuffers are torn down.
    ...(engine.debug ? { debug: engine.debug } : {}),
  };
  return proxy;
}
