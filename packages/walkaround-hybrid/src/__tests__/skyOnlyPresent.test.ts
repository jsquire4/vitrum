/**
 * H20-A — sky-only present path for the empty-scene-ready state.
 *
 * When the scene has zero primitives the engine reaches `'ready'` WITHOUT a
 * pipeline or BVH (HybridEngineLifecycle: "empty scene; ready without
 * BVH/pipeline"). Before H20-A `runHybridEngineFrame` returned SKIP on every
 * call and never wrote the host swap chain. H20-A presents a flat sky-only
 * frame (a single device-level clear render pass to `skyTint × skyIrradiance`)
 * and returns a genuine `kind:'rendered'` FrameOutput.
 *
 * These tests drive `runHybridEngineFrame` directly with a minimal deps stub so
 * the empty-scene branch is exercised deterministically (no async init timing).
 */
import { describe, expect, it, vi } from 'vitest';
import type { FrameInput, Scene } from '@vitrum/core';
import {
  runHybridEngineFrame,
  type HybridEngineFrameDeps,
} from '../HybridEngineFrameOrchestrator.js';

interface RecordedClear {
  r: number;
  g: number;
  b: number;
  a: number;
}

function makeDeviceStub(record: { clears: RecordedClear[]; submits: number }): GPUDevice {
  return {
    createCommandEncoder: vi.fn(() => ({
      beginRenderPass: vi.fn((desc: GPURenderPassDescriptor) => {
        const atts = desc.colorAttachments as GPURenderPassColorAttachment[];
        const att = atts[0];
        const cv = att?.clearValue as RecordedClear | undefined;
        if (cv) record.clears.push({ r: cv.r, g: cv.g, b: cv.b, a: cv.a });
        return { end: vi.fn() } as unknown as GPURenderPassEncoder;
      }),
      finish: vi.fn(() => ({}) as GPUCommandBuffer),
    })),
    queue: {
      submit: vi.fn(() => {
        record.submits++;
      }),
      writeBuffer: vi.fn(),
    },
  } as unknown as GPUDevice;
}

function emptyScene(): Scene {
  return { primitives: [], emitters: [], environment: { kind: 'none' } };
}

function nonEmptyScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'm',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeDeps(
  device: GPUDevice,
  scene: Scene | null,
  skyTint: [number, number, number],
  skyIrradiance: number,
): HybridEngineFrameDeps {
  return {
    subsystems: {
      pipeline: null,
      bvhBuffers: null,
      ddgi: {} as never,
      rc: null,
      skinning: null,
      lastScene: scene,
    },
    lighting: {
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 3,
      skyTint,
      skyIrradiance,
    },
    filter: {
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [1, 1, 1],
      atrousIndirectSigmas: [1, 1, 1],
      stainedGlassFlags: 0,
      restirPtReuse: 0,
      nrcEnabled: 0,
    },
    telemetry: {
      frameSubs: [],
      progressSubs: [],
      verbose: false,
      debugTimings: [],
      debugSurface: {} as never,
      dbg: null,
      getDenoiserState: () => null,
    },
    dims: { width: 64, height: 64, internalWidth: 64, internalHeight: 64 },
    control: {
      consumeRebuildKeyChange: () => false,
      targetFrameIntervalMs: null,
      getLastFrameTs: () => 0,
      setLastFrameTs: () => {},
      applyResolutionFactor: () => ({ width: 64, height: 64 }),
      runSkinning: () => {},
      presentLastFrame: () => {},
    },
    flags: {
      state: 'ready',
      debug: false,
      ddgiOn: false,
      isLayerEnabled: () => false,
      device,
      tunables: {} as never,
      rcWeight: 0,
    },
  } as unknown as HybridEngineFrameDeps;
}

function frameInput(withSwapView: boolean): FrameInput {
  return {
    viewMatrix: new Float32Array(16),
    projMatrix: new Float32Array(16),
    cameraPosition: [0, 0, 0],
    frameSeed: 1,
    viewport: { width: 64, height: 64 },
    ...(withSwapView ? { swapChainView: {} as GPUTextureView } : {}),
  } as unknown as FrameInput;
}

describe('H20-A sky-only present', () => {
  it('empty-scene-ready frame presents (rendered, not skipped) and clears to skyTint × skyIrradiance', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const device = makeDeviceStub(record);
    const deps = makeDeps(device, emptyScene(), [0.4, 0.6, 1.0], 2.0);

    const out = runHybridEngineFrame(deps, frameInput(true));

    expect(out.kind).toBe('rendered');
    expect(out.samplesAccumulated).toBe(1);
    expect(out.isConverged).toBe(false);
    expect(record.submits).toBe(1);
    expect(record.clears).toHaveLength(1);
    // skyTint × skyIrradiance, clamped non-negative.
    expect(record.clears[0]).toEqual({ r: 0.8, g: 1.2, b: 2.0, a: 1 });
  });

  it('empty-scene-ready WITHOUT a swap-chain view skips (no present possible)', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const device = makeDeviceStub(record);
    const deps = makeDeps(device, emptyScene(), [1, 1, 1], 1);

    const out = runHybridEngineFrame(deps, frameInput(false));

    expect(out.kind).toBe('skipped');
    expect(record.submits).toBe(0);
  });

  it('non-empty scene with no pipeline yet still SKIPS (mid-init, not sky-only)', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const device = makeDeviceStub(record);
    const deps = makeDeps(device, nonEmptyScene(), [1, 1, 1], 1);

    const out = runHybridEngineFrame(deps, frameInput(true));

    expect(out.kind).toBe('skipped');
    expect(record.submits).toBe(0);
  });

  it('non-ready state does not present sky-only', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const device = makeDeviceStub(record);
    const deps = makeDeps(device, emptyScene(), [1, 1, 1], 1);
    (deps.flags as { state: string }).state = 'initializing';

    const out = runHybridEngineFrame(deps, frameInput(true));

    expect(out.kind).toBe('skipped');
    expect(record.submits).toBe(0);
  });
});
