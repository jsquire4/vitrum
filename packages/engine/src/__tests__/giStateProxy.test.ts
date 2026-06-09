// giStateProxy.test.ts — the createEngine idempotent-dispose proxy conditionally
// forwards the walkaround-hybrid GI-state persistence methods (exportGIState /
// importGIState — the cached light field), with dispose-safe fallbacks that
// match the methods' own no-op semantics (export → null, import → false).

import { describe, it, expect, vi } from 'vitest';
import type { Engine, EngineCapabilities, EngineState } from '@vitrum/core';
import type { GIStateSnapshot } from '@vitrum/walkaround-hybrid';
import { wrapWithIdempotentDispose } from '../createEngine.js';
import type { GIStatePersistable } from '../idempotentDispose.js';

function baseCapabilities(): EngineCapabilities {
  return {
    supportsIncrementalScene: false,
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers: false,
    accumulates: true,
    maxSamplesPerPixel: 1,
    maxBounces: 1,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    supportedPrimitiveKinds: new Set(),
    supportedEnvironmentKinds: new Set(),
    presentationMode: 'offscreen-texture',
    experimentalFeatures: new Set(),
    causticStrategy: 'none',
  } as unknown as EngineCapabilities;
}

const SNAPSHOT = {
  dims: { x: 2, y: 2, z: 2 },
  origin: [0, 0, 0],
  spacing: 1,
} as unknown as GIStateSnapshot;

function makeEngine(withGiState: boolean) {
  const exportFn = vi.fn(async (): Promise<GIStateSnapshot | null> => SNAPSHOT);
  const importFn = vi.fn((_s: GIStateSnapshot): boolean => true);
  const engine: Engine & Partial<GIStatePersistable> = {
    get state(): EngineState { return 'ready'; },
    get capabilities() { return baseCapabilities(); },
    setScene: vi.fn(),
    renderFrame: vi.fn(() => ({ kind: 'skipped', samplesAccumulated: 0, isConverged: false })),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    ...(withGiState ? { exportGIState: exportFn, importGIState: importFn } : {}),
  };
  return { engine, exportFn, importFn };
}

describe('createEngine proxy — GI-state (cached light field) forwarding', () => {
  it('forwards export/import when the backend implements them', async () => {
    const { engine, exportFn, importFn } = makeEngine(true);
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.exportGIState).toBe('function');
    expect(typeof proxy.importGIState).toBe('function');
    await expect(proxy.exportGIState!()).resolves.toBe(SNAPSHOT);
    expect(exportFn).toHaveBeenCalledTimes(1);
    expect(proxy.importGIState!(SNAPSHOT)).toBe(true);
    expect(importFn).toHaveBeenCalledWith(SNAPSHOT);
  });

  it('omits the methods when the backend does not implement them (e.g. pt-webgl2/pt-webgpu)', () => {
    const { engine } = makeEngine(false);
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.exportGIState).toBeUndefined();
    expect(proxy.importGIState).toBeUndefined();
  });

  it('is dispose-safe: export → null, import → false, without touching the torn-down backend', async () => {
    const { engine, exportFn, importFn } = makeEngine(true);
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();
    await expect(proxy.exportGIState!()).resolves.toBeNull();
    expect(proxy.importGIState!(SNAPSHOT)).toBe(false);
    // The disposed fallbacks must NOT forward to the (torn-down) engine.
    expect(exportFn).not.toHaveBeenCalled();
    expect(importFn).not.toHaveBeenCalled();
  });
});
