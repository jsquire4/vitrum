// Top-level drop-in factory.
//
// Given a canvas + vitrum Scene, createEngine()
//   - probes WebGPU vs WebGL2,
//   - picks the walkaround-hybrid backend (real-time GI) or pt-webgl2
//     backend (converged path tracer),
//   - derives scale-sensitive defaults (Möller-Trumbore epsilon, camera-
//     move-reset threshold, emitter dist² floor, GTAO sigma) from the
//     scene's AABB diagonal D,
//   - constructs and owns the backend's device handle (GPUDevice for
//     walkaround, WebGL2RenderingContext for pt-webgl2) so the host doesn't
//     have to plumb GPU primitives,
//   - returns the @vitrum/core Engine contract with an idempotent dispose.
//
// Note on resize: HybridEngine (WebGPU) requires setSize(w, h) for in-flight
// resizes; attachVitrum() owns the ResizeObserver and calls it automatically.
// Generic PT engines honour FrameInput.viewport per-frame. createEngine()
// itself does NOT attach an observer.
//
// C1 refactor: backend constructor bodies live in backends/{walkaround,ptWebgpu,
// ptWebgl2}.ts; shared types + helpers live in createEngineInternals.ts.
// createEngine.ts is now a thin facade + dispatch table.

import type { Scene } from '@vitrum/core';
import { auditSceneNeedsTlas, detectGpu } from '@vitrum/core';
import {
  pickBackend,
  deriveScaleDefaults,
  type EnginePreference,
  type ScaleDefaults,
} from './createEngineScale.js';
import { computeSceneAABB, type SceneAABB } from './sceneAABB.js';

// Shared types & utilities (no circular imports: internals ← backends ← this file)
export type {
  CreateEngineBackendId,
  CreateEngineErrorPhase,
  CreateEngineErrorEvent,
  CreateEngineOptions,
  EngineWithBackendId,
  SharedDeviceCtx,
} from './createEngineInternals.js';
export {
  mergeWalkaroundTlasExtension,
  stripOwnershipCriticalKeys,
  warnCrossBackendAdvanced,
} from './createEngineInternals.js';

import type {
  CreateEngineBackendId,
  CreateEngineOptions,
  EngineWithBackendId,
  BackendConstructor,
} from './createEngineInternals.js';
import {
  warnCrossBackendAdvanced,
  reportCreateEngineError,
} from './createEngineInternals.js';

// Backend constructors (extracted bodies)
import { constructWalkaround, constructWalkaroundForDispatch } from './backends/walkaround.js';
import { constructPathTracerWebGPU, constructPathTracerWebGPUForDispatch } from './backends/ptWebgpu.js';
import { constructPathTracerForDispatch } from './backends/ptWebgl2.js';

export type { EnginePreference, ScaleDefaults };
export { pickBackend, deriveScaleDefaults };

// Re-exported for back-compat — createProgressiveEngine imports these from this module's path.
// @internal — not part of the public `@vitrum/engine` API surface.
export { constructWalkaround, constructPathTracerWebGPU };

// Re-exported for unit-test access (tests import it from this module's path).
// @internal — not part of the public `@vitrum/engine` API surface.
export { wrapWithIdempotentDispose } from './idempotentDispose.js';

// ────────────────────────────────────────────────────────────────────────────
// C1 dispatch table: replaces the if/else branching in createEngine.
//
// Each entry is a BackendConstructor: (opts, scene, aabb, needsTlas, shared?)
// → Promise<EngineWithBackendId>. The table is the single place to add a new
// backend; the createEngine function body stays generic.
// ────────────────────────────────────────────────────────────────────────────

/** Dispatch table mapping each backend id to its constructor wrapper.
 *  @internal */
const BACKEND_CONSTRUCTORS: Record<CreateEngineBackendId, BackendConstructor> = {
  'walkaround-hybrid': constructWalkaroundForDispatch,
  'pt-webgpu': constructPathTracerWebGPUForDispatch,
  'pt-webgl2': constructPathTracerForDispatch,
};

export async function createEngine(opts: CreateEngineOptions): Promise<EngineWithBackendId> {
  if (opts.canvas == null) {
    throw new TypeError('createEngine: opts.canvas is required');
  }
  if (opts.scene == null) {
    throw new TypeError('createEngine: opts.scene is required');
  }

  const vitrumScene: Scene = opts.scene;

  const aabb = computeSceneAABB(vitrumScene);
  const tlasAudit = auditSceneNeedsTlas(vitrumScene);
  const gpu = await detectGpu({ publishToWindow: false });
  const backend = pickBackend(
    opts.prefer ?? 'auto',
    gpu.isWebGPU,
    aabb.triangleCount,
    tlasAudit.needsTlas,
  );
  // When the audit recommends a TLAS-capable backend but we resolved to pt-webgl2
  // (the only merged-BVH backend), surface the recommendation + detail so the host
  // can switch to walkaround-hybrid or pt-webgpu for correct instancing behaviour.
  if (tlasAudit.recommendation === 'prefer-tlas-backend' && backend === 'pt-webgl2') {
    console.warn(`[vitrum/createEngine] ${tlasAudit.detail}`);
  }

  if (backend === 'walkaround-hybrid') {
    try {
      return await BACKEND_CONSTRUCTORS['walkaround-hybrid'](opts, vitrumScene, aabb, tlasAudit.needsTlas);
    } catch (err) {
      reportCreateEngineError(opts, err, {
        phase: 'create:walkaround-hybrid',
        backend: 'walkaround-hybrid',
        recoverable: true,
      });
      const fallbackBackend = gpu.isWebGPU ? 'pt-webgpu' : 'pt-webgl2';
      console.warn(
        `[vitrum/createEngine] walkaround-hybrid unavailable; falling back to ${fallbackBackend}.`,
        err,
      );
      warnCrossBackendAdvanced(opts.advanced, 'walkaround-hybrid', fallbackBackend);
      return await constructPathTracerFallback(opts, vitrumScene, aabb, tlasAudit.needsTlas, gpu.isWebGPU);
    }
  }
  if (backend === 'pt-webgpu') {
    try {
      return await BACKEND_CONSTRUCTORS['pt-webgpu'](opts, vitrumScene, aabb, tlasAudit.needsTlas);
    } catch (err) {
      reportCreateEngineError(opts, err, {
        phase: 'create:pt-webgpu',
        backend: 'pt-webgpu',
        recoverable: true,
      });
      console.warn('[vitrum/createEngine] pt-webgpu unavailable; falling back to pt-webgl2.', err);
      warnCrossBackendAdvanced(opts.advanced, 'pt-webgpu', 'pt-webgl2');
      return await constructPathTracerWebGLFallback(opts, vitrumScene, aabb, tlasAudit.needsTlas);
    }
  }
  return await constructPathTracerWebGLFallback(opts, vitrumScene, aabb, tlasAudit.needsTlas);
}

// ────────────────────────────────────────────────────────────────────────────
// Fallback helpers (used only by the dispatch logic above)
// ────────────────────────────────────────────────────────────────────────────

async function constructPathTracerFallback(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  aabb: SceneAABB,
  needsTlas: boolean,
  tryWebGpuFirst: boolean,
): Promise<EngineWithBackendId> {
  if (tryWebGpuFirst) {
    try {
      return await BACKEND_CONSTRUCTORS['pt-webgpu'](opts, vitrumScene, aabb, needsTlas);
    } catch (err) {
      reportCreateEngineError(opts, err, {
        phase: 'create:pt-webgpu',
        backend: 'pt-webgpu',
        recoverable: true,
      });
      console.warn('[vitrum/createEngine] pt-webgpu fallback unavailable; falling back to pt-webgl2.', err);
    }
  }
  return await constructPathTracerWebGLFallback(opts, vitrumScene, aabb, needsTlas);
}

async function constructPathTracerWebGLFallback(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  aabb: SceneAABB,
  needsTlas: boolean,
): Promise<EngineWithBackendId> {
  try {
    return await BACKEND_CONSTRUCTORS['pt-webgl2'](opts, vitrumScene, aabb, needsTlas);
  } catch (err) {
    reportCreateEngineError(opts, err, {
      phase: 'create:pt-webgl2',
      backend: 'pt-webgl2',
      recoverable: false,
    });
    throw err;
  }
}
