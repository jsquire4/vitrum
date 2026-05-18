import { describe, expect, it, vi } from 'vitest';
import type { FrameStats } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

describe('createPTEngine_WebGPU', () => {
  it('reports requested caustic strategy in capabilities', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'manifold-nee',
    });
    expect(engine.capabilities.causticStrategy).toBe('manifold-nee');
  });

  it('supports photon-map capability reporting path', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'photon-map',
    });
    expect(engine.capabilities.causticStrategy).toBe('photon-map');
  });

  it('transitions state ready → disposed across the lifecycle', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    expect(engine.state).toBe('ready');
    engine.dispose();
    expect(engine.state).toBe('disposed');
    // After dispose, lifecycle methods throw rather than no-op.
    expect(() => engine.pause()).toThrow(/disposed/);
    expect(() => engine.resume()).toThrow(/disposed/);
    expect(() => engine.renderFrame({} as never)).toThrow();
  });

  it('pause / resume toggle state when live', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    engine.pause();
    expect(engine.state).toBe('paused');
    engine.resume();
    expect(engine.state).toBe('ready');
    engine.dispose();
  });

  // ── W3-D16 telemetry ──────────────────────────────────────────────────
  // Full end-to-end onFrame firing requires a real GPUDevice + Scene
  // (covered by GPU integration tests). These tests pin the contract
  // surface: `onFrame` exists, returns an unsubscribe function, and the
  // canonical FrameStats shape pt-webgpu documents is assignable to the
  // core interface.

  it('exposes onFrame returning an unsubscribe function (W3-D16)', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    expect(typeof engine.onFrame).toBe('function');
    const off = engine.onFrame!(() => {});
    expect(typeof off).toBe('function');
    // Calling unsubscribe a second time is safe (idempotent).
    off();
    expect(() => off()).not.toThrow();
    engine.dispose();
  });

  it('canonical FrameStats shape pt-webgpu emits is core-assignable (W3-D16)', () => {
    // Type-level smoke. The runtime fixture mirrors what pt-webgpu's
    // renderFrame builds: `frameTimeMs`, `spp` / `samplesAccumulated`,
    // `frameIndex`, `backend: 'webgpu'`. `gpuTimeMs` / `passTimings` /
    // `bvhDepth` / `estimatedGpuMemoryBytes` are intentionally absent —
    // pt-webgpu doesn't have those signals yet, and the canonical contract
    // says "absent, not zero" for unavailable fields.
    const stats: FrameStats = {
      frameTimeMs: 4.2,
      spp: 3,
      samplesAccumulated: 3,
      frameIndex: 17,
      backend: 'webgpu',
    };
    expect(stats.backend).toBe('webgpu');
    expect(stats.samplesAccumulated).toBe(stats.spp);
    expect(stats.frameIndex).toBe(17);
    // Optional fields are absent (not zero) — host code can branch on this.
    expect(stats.gpuTimeMs).toBeUndefined();
    expect(stats.passTimings).toBeUndefined();
  });

  it('capabilities does NOT report qualityModes (W3-D16 — webgpu has no scheduler)', async () => {
    // pt-webgpu is honest: no adaptive scheduler, so qualityModes stays
    // absent. Hosts that want a "quality preset" UI introspect this and
    // hide the dropdown when it's missing.
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    expect(engine.capabilities.qualityModes).toBeUndefined();
    engine.dispose();
  });
});
