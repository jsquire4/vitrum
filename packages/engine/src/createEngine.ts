// Top-level drop-in factory.
//
// Given a canvas + vitrum Scene, createEngine()
//   - probes WebGPU vs WebGL2,
//   - default (`prefer:'auto'`): stands the progressive viewer (walkaround
//     while the camera moves, PT when it settles) when the adapter satisfies
//     the shared-device limit union; otherwise a single PT engine (full or
//     lite) or pt-webgl2. Explicit `prefer` stays a single engine.
//   - derives scale-sensitive defaults (Möller-Trumbore epsilon, camera-
//     move-reset threshold, emitter dist² floor, GTAO sigma) from the
//     scene's AABB diagonal D,
//   - constructs and owns the backend's device handle (GPUDevice for
//     walkaround / progressive, WebGL2RenderingContext for pt-webgl2) so the
//     host doesn't have to plumb GPU primitives,
//   - returns the @vitrum/core Engine contract with an idempotent dispose.
//
// Note on resize: every backend honours FrameInput.viewport per-frame.
// HybridEngine also exposes setSize(w, h) as an eager allocation hook;
// attachVitrum() owns a ResizeObserver and calls it automatically. createEngine()
// itself does NOT attach an observer.
//
// C1 refactor: backend constructor bodies live in backends/{walkaround,ptWebgpu,
// ptWebgl2}.ts; shared types + helpers live in createEngineInternals.ts.
// createEngine.ts is now a thin facade + dispatch table.

import type { Scene } from '@vitrum/core';
import { auditSceneNeedsTlas, detectGpu, validateScene } from '@vitrum/core';
import {
  pickBackend,
  deriveScaleDefaults,
  recommendBackendForSceneMaterials,
  shouldAttemptProgressiveViewer,
  type EnginePreference,
  type ScaleDefaults,
} from './createEngineScale.js';
import { computeSceneAABB, type SceneAABB } from './sceneAABB.js';

// Shared types & utilities (no circular imports: internals ← backends ← this file)
export type {
  CreateEngineBackendId,
  CreateEngineAdvancedByBackend,
  CreateEngineGltfAssetHint,
  CreateEngineErrorPhase,
  CreateEngineErrorEvent,
  CreateEngineOptions,
  RuntimeEngineBackendId,
  RuntimeEngineWithBackendId,
  EngineWithBackendId,
  SharedDeviceCtx,
} from './createEngineInternals.js';
export {
  mergeWalkaroundTlasExtension,
  resolveAdvancedForBackend,
  stripOwnershipCriticalKeys,
} from './createEngineInternals.js';

import type {
  CreateEngineBackendId,
  CreateEngineOptions,
  EngineWithBackendId,
  BackendConstructor,
} from './createEngineInternals.js';
import {
  emitCreateEngineWarning,
  isBackendUnavailableError,
  reportCreateEngineError,
  validateCreateEngineOptionsShape,
  attachBackendId,
  wrapWithIdempotentDispose,
} from './createEngineInternals.js';
import { applyExperimentalPreset } from './experimentalPresets.js';

// Backend constructors (extracted bodies)
import { constructWalkaround, constructWalkaroundForDispatch } from './backends/walkaround.js';
import { constructPathTracerWebGPU, constructPathTracerWebGPUForDispatch } from './backends/ptWebgpu.js';
import { constructPathTracerForDispatch } from './backends/ptWebgl2.js';

export type { EnginePreference, ScaleDefaults };
export { pickBackend, deriveScaleDefaults, shouldAttemptProgressiveViewer };

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

async function validateCreateEngineBackendAdvancedOptions(
  opts: CreateEngineOptions,
): Promise<void> {
  const entries: Array<readonly [CreateEngineBackendId, Record<string, unknown>]> = [];
  if (opts.advancedByBackend != null) {
    for (const backend of ['walkaround-hybrid', 'pt-webgpu', 'pt-webgl2'] as const) {
      const bag = opts.advancedByBackend[backend];
      if (bag != null) entries.push([backend, bag]);
    }
  } else if (opts.advanced != null && opts.advancedBackend != null) {
    entries.push([opts.advancedBackend, opts.advanced]);
  }

  for (const [backend, bag] of entries) {
    if (backend === 'pt-webgpu') {
      const module = await import('@vitrum/pt-webgpu');
      module.validatePtWebgpuAdvancedOptions(bag);
      continue;
    }
    if (backend === 'pt-webgl2') {
      const module = await import('@vitrum/pt-webgl2');
      module.validateWebgl2AdvancedOptions(bag);
      continue;
    }
    const module: typeof import('@vitrum/walkaround-hybrid') =
      await import('@vitrum/walkaround-hybrid');
    module.validateHybridEngineAdvancedOptions(bag);
  }
}

export async function createEngine(opts: CreateEngineOptions): Promise<EngineWithBackendId> {
  validateCreateEngineOptionsShape(opts);
  opts = applyExperimentalPreset(opts);
  validateScene(opts.scene);
  await validateCreateEngineBackendAdvancedOptions(opts);

  const vitrumScene: Scene = opts.scene;

  const aabb = computeSceneAABB(vitrumScene);
  const tlasAudit = auditSceneNeedsTlas(vitrumScene);
  const gpu = await detectGpu();
  const gltfRecommendedBackend = opts.gltfAsset?.recommendedBackend?.backend;
  const materialRecommendation = gltfRecommendedBackend == null
    ? recommendBackendForSceneMaterials(vitrumScene, gpu.isWebGPU)
    : null;
  const prefer = opts.prefer ?? 'auto';
  const backend = pickBackend(
    prefer,
    gpu.isWebGPU,
    aabb.triangleCount,
    tlasAudit.needsTlas,
    gltfRecommendedBackend,
    materialRecommendation?.backend,
  );
  const backendWithoutMaterialRecommendation = materialRecommendation == null
    ? backend
    : pickBackend(
      prefer,
      gpu.isWebGPU,
      aabb.triangleCount,
      tlasAudit.needsTlas,
      gltfRecommendedBackend,
    );
  if (materialRecommendation != null && backend !== backendWithoutMaterialRecommendation) {
    emitCreateEngineWarning(opts.onWarning, {
      code: 'createEngine.material-feature-backend-recommended',
      backend: 'createEngine',
      phase: 'construction',
      method: 'createEngine',
      message:
        `[vitrum/createEngine] prefer:'auto' selected ${backend} because the scene uses ` +
        `material fields that walkaround-hybrid reports unsupported: ` +
        `${materialRecommendation.fields.join(', ')}.`,
      details: {
        fields: materialRecommendation.fields,
        defaultAutoBackend: backendWithoutMaterialRecommendation,
        resolvedBackend: backend,
      },
    });
  }
  if (prefer === 'realtime' && !gpu.isWebGPU && backend === 'pt-webgl2') {
    emitCreateEngineWarning(
      opts.onWarning,
      {
        code: 'createEngine.realtime-unavailable-fallback',
        backend: 'createEngine',
        phase: 'fallback',
        method: 'createEngine',
        message:
          '[vitrum/createEngine] prefer:\'realtime\' requires WebGPU; ' +
          'this host has no WebGPU adapter, so the engine is using the converged pt-webgl2 backend.',
        details: {
          preferredBackend: 'walkaround-hybrid',
          resolvedBackend: 'pt-webgl2',
          reason: 'webgpu-unavailable',
        },
      },
      '[vitrum/createEngine] prefer:\'realtime\' requires WebGPU; ' +
        'falling back to the converged pt-webgl2 backend.',
    );
  }
  // When the audit recommends a TLAS-capable backend but we resolved to pt-webgl2
  // (the only merged-BVH backend), surface the recommendation + detail so the host
  // can switch to walkaround-hybrid or pt-webgpu for correct instancing behaviour.
  if (tlasAudit.recommendation === 'prefer-tlas-backend' && backend === 'pt-webgl2') {
    emitCreateEngineWarning(opts.onWarning, {
      code: 'createEngine.tlas-backend-recommended',
      backend: 'createEngine',
      phase: 'construction',
      method: 'createEngine',
      message: `[vitrum/createEngine] ${tlasAudit.detail}`,
      details: { recommendation: tlasAudit.recommendation, resolvedBackend: backend },
    });
  }

  if (shouldAttemptProgressiveViewer(
    prefer,
    gpu.isWebGPU,
    gltfRecommendedBackend,
    materialRecommendation?.backend,
  )) {
    try {
      return await constructProgressiveViewer(opts);
    } catch (err) {
      emitCreateEngineWarning(
        opts.onWarning,
        {
          code: 'createEngine.progressive-unavailable-fallback',
          backend: 'createEngine',
          phase: 'fallback',
          method: 'createEngine',
          message:
            `[vitrum/createEngine] progressive viewer unavailable; falling back to ${backend}.`,
          details: {
            preferredBackend: 'progressive',
            resolvedBackend: backend,
          },
          raw: err,
        },
        `[vitrum/createEngine] progressive viewer unavailable; falling back to ${backend}.`,
        err,
      );
    }
  }

  if (backend === 'walkaround-hybrid') {
    try {
      return await BACKEND_CONSTRUCTORS['walkaround-hybrid'](opts, vitrumScene, aabb, tlasAudit.needsTlas);
    } catch (err) {
      if (!isBackendUnavailableError(err)) throw err;
      reportCreateEngineError(opts, err, {
        phase: 'create:walkaround-hybrid',
        backend: 'walkaround-hybrid',
        recoverable: true,
      });
      const fallbackBackend = gpu.isWebGPU ? 'pt-webgpu' : 'pt-webgl2';
      emitCreateEngineWarning(
        opts.onWarning,
        {
          code: 'createEngine.walkaround-fallback',
          backend: 'createEngine',
          phase: 'fallback',
          method: 'createEngine',
          message: `[vitrum/createEngine] walkaround-hybrid unavailable; falling back to ${fallbackBackend}.`,
          details: { preferredBackend: 'walkaround-hybrid', resolvedBackend: fallbackBackend },
          raw: err,
        },
        `[vitrum/createEngine] walkaround-hybrid unavailable; falling back to ${fallbackBackend}.`,
        err,
      );
      return await constructPathTracerFallback(opts, vitrumScene, aabb, tlasAudit.needsTlas, gpu.isWebGPU);
    }
  }
  if (backend === 'pt-webgpu') {
    try {
      return await BACKEND_CONSTRUCTORS['pt-webgpu'](opts, vitrumScene, aabb, tlasAudit.needsTlas);
    } catch (err) {
      if (!isBackendUnavailableError(err)) throw err;
      reportCreateEngineError(opts, err, {
        phase: 'create:pt-webgpu',
        backend: 'pt-webgpu',
        recoverable: true,
      });
      emitCreateEngineWarning(
        opts.onWarning,
        {
          code: 'createEngine.pt-webgpu-fallback',
          backend: 'createEngine',
          phase: 'fallback',
          method: 'createEngine',
          message: '[vitrum/createEngine] pt-webgpu unavailable; falling back to pt-webgl2.',
          details: { preferredBackend: 'pt-webgpu', resolvedBackend: 'pt-webgl2' },
          raw: err,
        },
        '[vitrum/createEngine] pt-webgpu unavailable; falling back to pt-webgl2.',
        err,
      );
      return await constructPathTracerWebGLFallback(opts, vitrumScene, aabb, tlasAudit.needsTlas);
    }
  }
  return await constructPathTracerWebGLFallback(opts, vitrumScene, aabb, tlasAudit.needsTlas);
}

// ────────────────────────────────────────────────────────────────────────────
// Fallback helpers (used only by the dispatch logic above)
// ────────────────────────────────────────────────────────────────────────────

async function constructProgressiveViewer(
  opts: CreateEngineOptions,
): Promise<EngineWithBackendId> {
  const { createProgressiveEngine, progressiveHandleAsEngine } = await import(
    './createProgressiveEngine.js'
  );
  const realtimeOptions = opts.advancedByBackend?.['walkaround-hybrid'];
  const convergedOptions = opts.advancedByBackend?.['pt-webgpu'];
  const handle = await createProgressiveEngine({
    canvas: opts.canvas,
    scene: opts.scene,
    ...(realtimeOptions != null ? { realtimeOptions } : {}),
    ...(convergedOptions != null ? { convergedOptions } : {}),
    ...(opts.debug != null ? { debug: opts.debug } : {}),
    ...(opts.onAdapterProfile != null ? { onAdapterProfile: opts.onAdapterProfile } : {}),
    ...(opts.onWarning != null ? { onWarning: opts.onWarning } : {}),
  });
  return attachBackendId(
    wrapWithIdempotentDispose(progressiveHandleAsEngine(handle), () => {}),
    'progressive',
  );
}

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
      if (!isBackendUnavailableError(err)) throw err;
      reportCreateEngineError(opts, err, {
        phase: 'create:pt-webgpu',
        backend: 'pt-webgpu',
        recoverable: true,
      });
      emitCreateEngineWarning(
        opts.onWarning,
        {
          code: 'createEngine.pt-webgpu-secondary-fallback',
          backend: 'createEngine',
          phase: 'fallback',
          method: 'createEngine',
          message: '[vitrum/createEngine] pt-webgpu fallback unavailable; falling back to pt-webgl2.',
          details: { preferredBackend: 'pt-webgpu', resolvedBackend: 'pt-webgl2' },
          raw: err,
        },
        '[vitrum/createEngine] pt-webgpu fallback unavailable; falling back to pt-webgl2.',
        err,
      );
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
