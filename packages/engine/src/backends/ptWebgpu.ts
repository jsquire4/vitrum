// @internal — pt-webgpu backend constructor.
//
// Extracted from createEngine.ts (C1 / D1.2) to keep the public facade thin.
// Re-exported from createEngine.ts for createProgressiveEngine back-compat;
// this module is NOT part of the public @vitrum/engine API surface.

import type { Scene, Engine } from '@vitrum/core';
import {
  createPTEngine_WebGPU,
  ptWebgpuRequiredLimitsForAdapter,
  type PTEngineWebGPUOptions,
} from '@vitrum/pt-webgpu';
import { configureWebGpuCanvas } from '../configureWebGpuCanvas.js';
import {
  stripOwnershipCriticalKeys,
  reportCreateEngineError,
  attachBackendId,
  wrapWithIdempotentDispose,
  type CreateEngineOptions,
  type SharedDeviceCtx,
  type EngineWithBackendId,
} from '../createEngineInternals.js';

function destroyOwnedWebGpuDevice(shared: SharedDeviceCtx | undefined, device: GPUDevice): void {
  if (shared == null) {
    try { device.destroy(); } catch { /* best-effort destroy — ignore */ }
  }
}

/** @internal — reused by `createProgressiveEngine` to build the converged
 *  engine on a shared device. Not part of the public `@vitrum/engine` API. */
export async function constructPathTracerWebGPU(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  shared?: SharedDeviceCtx,
): Promise<Engine> {
  const adapter = shared?.adapter ?? await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported support');
  }
  const advancedWebGPURaw = opts.advanced as Partial<PTEngineWebGPUOptions> | undefined;
  const advancedWebGPU = stripOwnershipCriticalKeys(advancedWebGPURaw, 'pt-webgpu') as Partial<PTEngineWebGPUOptions>;
  const device = shared?.device ?? await adapter.requestDevice({
    requiredLimits: ptWebgpuRequiredLimitsForAdapter(adapter, {
      restirPtReuse: advancedWebGPURaw?.restirPtReuse === true,
    }),
  });

  let engine: Engine | null = null;
  try {
    const merged: PTEngineWebGPUOptions = {
      device,
      ...advancedWebGPU,
      // A shared device is built with the limit UNION (≥ the full pt-webgpu
      // per-stage buffer/texture floor), so force the full trace tier here —
      // the union exists precisely so both engines run at full fidelity, and the
      // auto-resolver would also pick 'full' from these limits. Explicit is safer
      // (it surfaces a clear throw if a caller ever passes an under-spec device).
      ...(shared != null ? { traceTier: 'full' as const } : {}),
    };

    engine = await createPTEngine_WebGPU(merged);
    engine.setScene(vitrumScene);

    // The converged engine renders offscreen (presentationMode:'offscreen-texture');
    // it does not present to the canvas. When standing alone (createEngine) we keep
    // the historical canvas-configure for attachVitrum swap-chain plumbing. Under a
    // shared device the realtime engine owns the canvas, so skip it here to avoid
    // re-configuring the same context twice.
    if (shared == null) {
      configureWebGpuCanvas(opts.canvas, device, (err) => {
        reportCreateEngineError(opts, err, {
          phase: 'canvas-configure',
          backend: 'pt-webgpu',
          recoverable: true,
        });
      });
    }

    const built = engine;
    engine = null;
    return wrapWithIdempotentDispose(built, () => {
      destroyOwnedWebGpuDevice(shared, device);
    });
  } catch (err) {
    try { engine?.dispose(); } catch { /* best-effort cleanup before re-throw — ignore */ }
    destroyOwnedWebGpuDevice(shared, device);
    throw err;
  }
}

/** Thin dispatch-table wrapper: builds, tags with backendId, returns EngineWithBackendId.
 *  @internal */
export async function constructPathTracerWebGPUForDispatch(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  aabb: import('../sceneAABB.js').SceneAABB, // unused but required by BackendConstructor signature
  _needsTlas: boolean,
  shared?: SharedDeviceCtx,
): Promise<EngineWithBackendId> {
  const engine = await constructPathTracerWebGPU(opts, vitrumScene, shared);
  return attachBackendId(engine, 'pt-webgpu');
}
