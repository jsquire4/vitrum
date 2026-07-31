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
import {
  asBackendTextureFormat,
  type FrameInput,
  type Scene,
} from '@vitrum/core';
import { applyTonemap, linearToSrgb } from '@vitrum/shared-samplers';
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

function frameInput(
  withSwapView: boolean,
  swapChainFormat: GPUTextureFormat = 'bgra8unorm',
): FrameInput {
  return {
    viewMatrix: new Float32Array(16),
    projMatrix: new Float32Array(16),
    cameraPosition: [0, 0, 0],
    frameSeed: 1,
    viewport: { width: 64, height: 64 },
    swapChainFormat:
      asBackendTextureFormat<'webgpu', GPUTextureFormat>(swapChainFormat),
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
    const mapped = applyTonemap([0.8, 1.2, 2], 'aces', 1).map(linearToSrgb);
    expect(record.clears[0]?.r).toBeCloseTo(mapped[0]!, 7);
    expect(record.clears[0]?.g).toBeCloseTo(mapped[1]!, 7);
    expect(record.clears[0]?.b).toBeCloseTo(mapped[2]!, 7);
    expect(record.clears[0]?.a).toBe(1);
  });

  it('honors per-frame exposure, tonemap, and linear output on the empty scene', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const deps = makeDeps(makeDeviceStub(record), emptyScene(), [0.4, 0.6, 1], 2);
    const input = frameInput(true);
    (input as { quality?: FrameInput['quality'] }).quality = {
      tonemap: 'reinhard',
      exposure: 0.5,
      outputColorSpace: 'linear',
    };

    runHybridEngineFrame(deps, input);

    const expected = applyTonemap([0.8, 1.2, 2], 'reinhard', 0.5);
    expect(record.clears[0]?.r).toBeCloseTo(expected[0], 7);
    expect(record.clears[0]?.g).toBeCloseTo(expected[1], 7);
    expect(record.clears[0]?.b).toBeCloseTo(expected[2], 7);
  });

  it('uses the shader-mirrored f32 sky product while preserving surviving channels', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const deps = makeDeps(
      makeDeviceStub(record),
      emptyScene(),
      [1, 1, 2 ** -149],
      0.5,
    );
    const input = frameInput(true, 'rgba32float');
    (input as { quality?: FrameInput['quality'] }).quality = {
      tonemap: 'none',
      exposure: 1,
      outputColorSpace: 'linear',
    };

    runHybridEngineFrame(deps, input);

    expect(record.clears[0]).toEqual({ r: 0.5, g: 0.5, b: 0, a: 1 });
  });

  it('passes linear tonemapped values to an sRGB attachment for one hardware OETF', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const deps = makeDeps(
      makeDeviceStub(record),
      emptyScene(),
      [0.25, 0.5, 0.75],
      2,
    );
    const input = frameInput(true, 'bgra8unorm-srgb');
    (input as { quality?: FrameInput['quality'] }).quality = {
      tonemap: 'linear',
      exposure: 1,
      outputColorSpace: 'srgb',
    };

    runHybridEngineFrame(deps, input);

    // clearValue is interpreted as linear for an sRGB render attachment; the
    // output merger performs the sole OETF.
    expect(record.clears[0]).toEqual({ r: 0.5, g: 1, b: 1, a: 1 });
  });

  it('quantizes sky-only exposure to the same f32 value written to the UBO', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const deps = makeDeps(
      makeDeviceStub(record),
      emptyScene(),
      [0.5, 0.5, 0.5],
      1,
    );
    const input = frameInput(true);
    const halfwayAboveOne = 1 + 2 ** -24;
    expect(Math.fround(halfwayAboveOne)).toBe(1);
    (input as { quality?: FrameInput['quality'] }).quality = {
      tonemap: 'none',
      exposure: halfwayAboveOne,
      outputColorSpace: 'linear',
    };

    runHybridEngineFrame(deps, input);

    expect(record.clears[0]).toEqual({ r: 0.5, g: 0.5, b: 0.5, a: 1 });
  });

  it.each([
    ['rgba32float', 131_008],
    ['rgba16float', 65_504],
    ['rg11b10ufloat', 64_512],
  ] as const)(
    'clamps none+linear sky output at the concrete %s target boundary',
    (format, expectedBlue) => {
      const record = { clears: [] as RecordedClear[], submits: 0 };
      const deps = makeDeps(
        makeDeviceStub(record),
        emptyScene(),
        [65_504, 65_504, 65_504],
        1,
      );
      const input = frameInput(true, format);
      (input as { quality?: FrameInput['quality'] }).quality = {
        tonemap: 'none',
        exposure: 2,
        outputColorSpace: 'linear',
      };

      runHybridEngineFrame(deps, input);

      expect(record.clears[0]?.r).toBe(
        format === 'rg11b10ufloat' ? 65_024 : expectedBlue,
      );
      expect(record.clears[0]?.g).toBe(
        format === 'rg11b10ufloat' ? 65_024 : expectedBlue,
      );
      expect(record.clears[0]?.b).toBe(expectedBlue);
    },
  );

  it('rejects linear output on an sRGB target before encoder, submit, or debug mutation', () => {
    const record = { clears: [] as RecordedClear[], submits: 0 };
    const device = makeDeviceStub(record);
    const deps = makeDeps(device, emptyScene(), [1, 1, 1], 1);
    const dbg = {
      initStart: 0,
      initCount: 0,
      disposeCount: 0,
      skipNoPipeline: 0,
      skipNoBvh: 0,
      skipNoSwapView: 0,
      skipFrameInterval: 0,
      framesDispatched: 0,
      lastReportTs: 0,
    };
    (deps.telemetry as { dbg: typeof dbg | null }).dbg = dbg;
    const input = frameInput(true, 'rgba8unorm-srgb');
    (input as { quality?: FrameInput['quality'] }).quality = {
      outputColorSpace: 'linear',
    };

    expect(() => runHybridEngineFrame(deps, input)).toThrow(
      /outputColorSpace 'linear' is incompatible/,
    );
    expect(device.createCommandEncoder).not.toHaveBeenCalled();
    expect(record.submits).toBe(0);
    expect(dbg.framesDispatched).toBe(0);
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
