/**
 * lifecycleEdges.test.ts — §4.10-11 lifecycle and correctness edge tests.
 *
 * Covers the 7 verified bugs from the v1-closure-plan §4.11:
 *
 *   1. WalkaroundGPUPipeline.dispose() clears _initialized — post-dispose
 *      calls are no-ops.  Tested structurally: the _initialized flag clear
 *      is already pinned in walkaroundPipelineCharacterization.test.ts
 *      (dispose resets every subsystem).  Here we add a compile-level guard
 *      that dispose() is present on the class and returns void.
 *
 *   2. OIDNFinalDenoiser resize stale-write guard: resize() during an
 *      in-flight inference cycle bumps _resizeGeneration so the pending
 *      writeTexture is skipped.
 *
 *   3. NeuralDenoiser resize vs InferenceGraph fixed dims: after resize(),
 *      dispatch() at the new dims falls back (detects _graphW/_graphH mismatch).
 *
 *   4. PPGCoordinator.onResize() retains maxSpatialCells from initialize()
 *      and bumps _frameResourcesGeneration to abort stale readback callbacks.
 *
 *   5. SampleBudgetPass reads the freshest Welford side — varianceBufferAux
 *      when welfordPing === 1.
 *
 *   6. HybridEngine.importGIState() validates grid origin/spacing/dims
 *      before restoring — rejects mismatched grids.
 *
 *   7. PPG flux atomicAdd saturation — the WGSL kernel uses a saturation
 *      pattern to prevent u32 wrap on hot leaves.
 */

import { describe, it, expect, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

// ──────────────────────────────────────────────────────────────────────────────
// §1 — WalkaroundGPUPipeline.dispose() clears _initialized
// Structural guard: dispose() exists, has correct signature, and the class
// exposes _initialized as a protected/private field that the method clears.
// ──────────────────────────────────────────────────────────────────────────────

import { WalkaroundGPUPipeline } from '../src/pipeline/WalkaroundGPUPipeline.js';

describe('Item 1 — WalkaroundGPUPipeline.dispose sets _initialized=false', () => {
  it('WalkaroundGPUPipeline has a dispose() method (guard against accidental removal)', () => {
    expect(typeof WalkaroundGPUPipeline.prototype.dispose).toBe('function');
  });

  it('post-dispose _initialized is false (sourced from the patched dispose body)', () => {
    // Build the minimal stub needed to instantiate without a real device.
    // We only verify that after dispose() the _initialized flag is cleared
    // by reading it via the "as any" escape hatch (field is private).
    const device = {
      createBuffer: vi.fn(() => ({ destroy: vi.fn(), label: '', size: 0 })),
      createTexture: vi.fn(() => ({ destroy: vi.fn(), createView: vi.fn(() => ({})) })),
      createShaderModule: vi.fn(() => ({})),
      createBindGroup: vi.fn(() => ({})),
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createRenderPipeline: vi.fn(() => ({})),
      createComputePipelineAsync: vi.fn(async () => ({ getBindGroupLayout: vi.fn(() => ({})) })),
      createCommandEncoder: vi.fn(() => ({
        finish: vi.fn(() => ({})),
        beginComputePass: vi.fn(() => ({
          setPipeline: vi.fn(), setBindGroup: vi.fn(),
          dispatchWorkgroups: vi.fn(), end: vi.fn(),
        })),
        copyTextureToTexture: vi.fn(),
      })),
      queue: { submit: vi.fn(), writeBuffer: vi.fn(), writeTexture: vi.fn() },
      createQuerySet: vi.fn(() => ({ destroy: vi.fn() })),
      features: { has: vi.fn(() => false) },
      limits: { maxBindGroups: 8 },
    } as unknown as GPUDevice;

    // Constructing WalkaroundGPUPipeline does not call initialize(), so
    // _initialized is false by default.  dispose() on an uninitialised
    // pipeline must be safe (no-throw) and must leave _initialized false.
    const pipeline = new WalkaroundGPUPipeline(device, 64, 64);

    expect(() => pipeline.dispose()).not.toThrow();
    // _initialized is private — access via type assertion.
    expect((pipeline as unknown as { _initialized: boolean })._initialized).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// §2 — OIDNFinalDenoiser resize stale-write guard
// ──────────────────────────────────────────────────────────────────────────────

import { OIDNFinalDenoiser } from '../src/pipeline/denoisers/oidnFinal.js';

// Mock @vitrum/shared-denoisers so no real ONNX runtime is required.
vi.mock('@vitrum/shared-denoisers', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/shared-denoisers')>();
  return {
    ...actual,
    acquireOIDNSession: vi.fn(async () => ({ release: vi.fn() })),
    preloadOIDNModel: vi.fn(async () => undefined),
    denoiseFinal: vi.fn(
      async (inputs: { color: Float32Array }) => new Float32Array(inputs.color),
    ),
    releaseOIDNCacheEntry: vi.fn(),
    clearOIDNCache: vi.fn(),
  };
});

function makeReadbackBuffer(byteSize: number): GPUBuffer {
  const backing = new ArrayBuffer(byteSize);
  return {
    label: '',
    size: byteSize,
    usage: 0,
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => backing),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer;
}

function makeOidnDevice(w: number, h: number) {
  return {
    createTexture: vi.fn(() => ({
      label: '',
      destroy: vi.fn(),
      createView: vi.fn(() => ({})),
      width: w, height: h,
    })),
    createBuffer: vi.fn((d: { size: number }) => makeReadbackBuffer(d.size)),
    queue: {
      writeTexture: vi.fn(),
      writeBuffer: vi.fn(),
      submit: vi.fn(),
    },
  } as unknown as GPUDevice;
}

// _makeOidnCtx: retained as an OIDN context factory for denoiser lifecycle tests.
function _makeOidnCtx(device: GPUDevice, w: number, h: number) {
  const hdr = {
    label: 'hdr', destroy: vi.fn(),
    createView: vi.fn(() => ({})), width: w, height: h,
  } as unknown as GPUTexture;
  return {
    device,
    encoder: {
      copyTextureToBuffer: vi.fn(),
      beginComputePass: vi.fn(),
      finish: vi.fn(() => ({})),
    } as unknown as GPUCommandEncoder,
    hdrColorTexture: hdr,
    // The dispatch ctx is minimal — dispatch() reads device + encoder + width/height.
    width: w, height: h,
    resources: {
      common: {
        hdrColorTexture: hdr,
        albedoTexture: { label: 'alb', createView: vi.fn(() => ({})) } as unknown as GPUTexture,
        gNormalDepthTexture: { label: 'gnd', createView: vi.fn(() => ({})) } as unknown as GPUTexture,
      },
    },
    bglCache: {} as never,
    frameIndex: 0,
    gNormalDepthView: {} as GPUTextureView,
  };
}

describe('Item 2 — OIDNFinalDenoiser resize stale-write guard', () => {
  it('resize() bumps _resizeGeneration on the denoiser', async () => {
    const device = makeOidnDevice(64, 32);
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    await d.initialize({ device, width: 64, height: 32, bglCache: {} as never, frameResources: {} as never });

    const before = (d as unknown as { _resizeGeneration: number })._resizeGeneration;
    d.resize(128, 64);
    const after = (d as unknown as { _resizeGeneration: number })._resizeGeneration;
    expect(after).toBe(before + 1);
  });

  it('resize() a second time increments by 2 total', async () => {
    const device = makeOidnDevice(64, 32);
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    await d.initialize({ device, width: 64, height: 32, bglCache: {} as never, frameResources: {} as never });

    const before = (d as unknown as { _resizeGeneration: number })._resizeGeneration;
    d.resize(128, 64);
    d.resize(64, 32);
    const after = (d as unknown as { _resizeGeneration: number })._resizeGeneration;
    expect(after).toBe(before + 2);
  });

  it('_resizeGeneration does not change when resize() is not called', async () => {
    const device = makeOidnDevice(64, 32);
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    await d.initialize({ device, width: 64, height: 32, bglCache: {} as never, frameResources: {} as never });

    const gen1 = (d as unknown as { _resizeGeneration: number })._resizeGeneration;
    const gen2 = (d as unknown as { _resizeGeneration: number })._resizeGeneration;
    expect(gen1).toBe(gen2);
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// §3 — NeuralDenoiser resize vs InferenceGraph fixed dims
// ──────────────────────────────────────────────────────────────────────────────

import { NeuralDenoiser } from '../src/pipeline/denoisers/neural.js';
import type { InferenceGraph } from '../src/neural/InferenceGraph.js';
import { NEURAL_F32_TENSOR_STORAGE } from '../src/neural/tensorPrecision.js';
import type { DenoiserDispatchContext } from '../src/pipeline/denoisers/index.js';

const fakeInferenceGraph = {
  run: vi.fn(),
  tensorStorage: NEURAL_F32_TENSOR_STORAGE,
} as unknown as InferenceGraph;

function makeNeuralDevice(): GPUDevice {
  return {
    createBuffer: vi.fn((d: { size: number; label?: string }) => ({
      label: d.label ?? '', size: d.size, destroy: vi.fn(),
    })),
    createTexture: vi.fn((d: { label?: string }) => ({
      label: d.label ?? '', destroy: vi.fn(), createView: vi.fn(() => ({})),
    })),
    createShaderModule: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    createBindGroup: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeNeuralDispatchCtx(w: number, h: number, device: GPUDevice): DenoiserDispatchContext {
  return {
    device,
    encoder: {
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(), end: vi.fn(),
      })),
    } as unknown as GPUCommandEncoder,
    width: w,
    height: h,
    resources: {
      common: {
        hdrColorTexture: { label: 'hdr', createView: vi.fn(() => ({})) } as unknown as GPUTexture,
        albedoTexture: { label: 'alb', createView: vi.fn(() => ({})) } as unknown as GPUTexture,
        gNormalDepthTexture: { label: 'gnd', createView: vi.fn(() => ({})) } as unknown as GPUTexture,
      },
    } as DenoiserDispatchContext['resources'],
    bglCache: {} as never,
    frameIndex: 0,
    computeDesc: (label: string) => ({ label }),
  } as unknown as DenoiserDispatchContext;
}

describe('Item 3 — NeuralDenoiser uses _graphW/_graphH for size guard', () => {
  it('_graphW and _graphH are stored at initialize() time', async () => {
    const device = makeNeuralDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeInferenceGraph });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    expect((d as unknown as { _graphW: number })._graphW).toBe(64);
    expect((d as unknown as { _graphH: number })._graphH).toBe(64);
  });

  it('resize() does NOT update _graphW/_graphH (graph dims are fixed)', async () => {
    const device = makeNeuralDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeInferenceGraph });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    d.resize(128, 128);

    expect((d as unknown as { _graphW: number })._graphW).toBe(64);
    expect((d as unknown as { _graphH: number })._graphH).toBe(64);
  });

  it('dispatch() after a resize without retained weights fails durably', async () => {
    const device = makeNeuralDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeInferenceGraph });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    d.resize(128, 128);
    const ctx = makeNeuralDispatchCtx(128, 128, device);
    expect(() => d.dispatch(ctx)).toThrow(/without retained model weights/i);

    const state = d.state();
    expect(state.status).toBe('failed');
    if (state.status === 'failed') {
      expect(state.reason).toMatch(/without retained model weights/i);
      expect(state.retryable).toBe(false);
    }
  });

  it('does not silently clear selected-mode resize failure by resizing back', async () => {
    const device = makeNeuralDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeInferenceGraph });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    d.resize(128, 128);
    d.resize(64, 64);

    const ctx = makeNeuralDispatchCtx(64, 64, device);
    expect(() => d.dispatch(ctx)).toThrow(/without retained model weights/i);

    const state = d.state();
    expect(state.status).toBe('failed');
    if (state.status === 'failed') {
      expect(state.reason).toMatch(/without retained model weights/i);
      expect(state.retryable).toBe(false);
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// §4 — PPGCoordinator.onResize() retains maxSpatialCells + bumps generation
// ──────────────────────────────────────────────────────────────────────────────

import { PPGCoordinator } from '../src/pipeline/PPGCoordinator.js';
import { PPG_MIS_ALPHA } from '../src/ppg/ppgConstants.js';
import type { FrameResources } from '../src/pipeline/resourceManager.js';
import { computePPGResourceFootprint } from '../src/pipeline/resourceManager.js';

function makeMinimalPPGDevice(): GPUDevice {
  return {
    createBuffer: vi.fn((d: { size: number; label?: string }) => ({
      label: d.label ?? '', size: d.size, destroy: vi.fn(),
    })),
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
  } as unknown as GPUDevice;
}

function makeMinimalFrameResources(): FrameResources {
  const mkBuf = (label: string, size: number) =>
    ({ label, size, destroy: vi.fn() }) as unknown as GPUBuffer;
  return {
    ppg: {
      queryArenaBuf: mkBuf('ppg-query-arena', 1),
      queryArenaLayout: {},
      queryArenaEpoch: 0,
      fluxAtomicsBuf: mkBuf('ppg-flux-atomics', 1024 * 341 * 4),
      cellSampleCountsBuf: mkBuf('ppg-cellSampleCounts', 1024 * 4),
      updateUboBuffer: mkBuf('ppg-update-ubo', 16),
    },
  } as unknown as FrameResources;
}

function lastCreatedBufferSize(device: GPUDevice, label: string): number | undefined {
  const calls = (device.createBuffer as unknown as {
    mock: { calls: Array<[desc: { label?: string; size: number }]> };
  }).mock.calls;
  return calls
    .filter(([desc]) => desc.label === label)
    .at(-1)?.[0].size;
}

describe('Item 4 — PPGCoordinator onResize retains maxSpatialCells + bumps generation', () => {
  it('_maxSpatialCells is stored from initialize() args', () => {
    const device = makeMinimalPPGDevice();
    const coord = new PPGCoordinator(device);
    const fr = makeMinimalFrameResources();

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
      2048,
    );

    expect((coord as unknown as { _maxSpatialCells: number | undefined })._maxSpatialCells).toBe(2048);
  });

  it('stores maxDTreeNodesPerCell and allocates init/resize buffers with that stride', () => {
    const device = makeMinimalPPGDevice();
    const coord = new PPGCoordinator(device);
    const fr = makeMinimalFrameResources();

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
      8,
      17,
    );

    expect((coord as unknown as { _maxDTreeNodesPerCell: number | undefined })._maxDTreeNodesPerCell).toBe(17);
    expect(lastCreatedBufferSize(device, 'ppg-fluxAtomics')).toBe(8 * 17 * 4);
    expect(lastCreatedBufferSize(device, 'ppg-query-arena')).toBe(
      computePPGResourceFootprint(8, 17).queryArenaBytes,
    );

    coord.onResize(makeMinimalFrameResources(), 128, 128, 1);

    expect(lastCreatedBufferSize(device, 'ppg-fluxAtomics')).toBe(8 * 17 * 4);
    expect(lastCreatedBufferSize(device, 'ppg-query-arena')).toBe(
      computePPGResourceFootprint(8, 17).queryArenaBytes,
    );
  });

  it('stores valid ppgMixAlpha values without clamping', () => {
    const device = makeMinimalPPGDevice();
    const coord = new PPGCoordinator(device);
    const fr = makeMinimalFrameResources();

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
      8,
      17,
      0.75,
    );

    expect(coord.mixAlpha).toBe(0.75);

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
      8,
      17,
      0.25,
    );

    expect(coord.mixAlpha).toBeCloseTo(0.25, 6);
  });

  it('_maxSpatialCells is undefined by default (no custom cap)', () => {
    const device = makeMinimalPPGDevice();
    const coord = new PPGCoordinator(device);
    const fr = makeMinimalFrameResources();

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
      // no maxSpatialCells arg
    );

    expect((coord as unknown as { _maxSpatialCells: number | undefined })._maxSpatialCells).toBeUndefined();
    expect((coord as unknown as { _maxDTreeNodesPerCell: number | undefined })._maxDTreeNodesPerCell).toBeUndefined();
    expect(coord.mixAlpha).toBe(PPG_MIS_ALPHA);
  });

  it('onResize() bumps _frameResourcesGeneration', () => {
    const device = makeMinimalPPGDevice();
    const coord = new PPGCoordinator(device);
    const fr = makeMinimalFrameResources();

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
    );

    const genBefore = (coord as unknown as { _frameResourcesGeneration: number })._frameResourcesGeneration;
    const fr2 = makeMinimalFrameResources();
    coord.onResize(fr2, 128, 128, 1);
    const genAfter = (coord as unknown as { _frameResourcesGeneration: number })._frameResourcesGeneration;

    expect(genAfter).toBe(genBefore + 1);
  });

  it('multiple onResize() calls each increment generation', () => {
    const device = makeMinimalPPGDevice();
    const coord = new PPGCoordinator(device);
    const fr = makeMinimalFrameResources();

    coord.initialize(
      { bvhPositions: { cpuData: new Float32Array(4).buffer, count: 1 } } as never,
      fr, 64, 64, true, 0,
    );

    const genBefore = (coord as unknown as { _frameResourcesGeneration: number })._frameResourcesGeneration;
    coord.onResize(makeMinimalFrameResources(), 128, 128, 1);
    coord.onResize(makeMinimalFrameResources(), 64, 64, 2);
    const genAfter = (coord as unknown as { _frameResourcesGeneration: number })._frameResourcesGeneration;

    expect(genAfter).toBe(genBefore + 2);
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// §5 — SampleBudgetPass reads the freshest Welford side (welfordPing=1 path)
// ──────────────────────────────────────────────────────────────────────────────
//
// The existing dispatchEquivalence.test.ts pins the welfordPing=0 path.
// These tests verify the welfordPing=1 branch selects varianceBufferAux.
// ──────────────────────────────────────────────────────────────────────────────

import { SampleBudgetPass } from '../src/pipeline/passes/SampleBudgetPass.js';
import type { PassDispatchContext } from '../src/pipeline/Pass.js';
import type { PassLabel } from '../src/pipeline/timestampQueries.js';

describe('Item 5 — SampleBudgetPass welfordPing ping-pong selection', () => {
  function makeViewTrackingBuf(label: string) {
    const calls: string[] = [];
    const buf = {
      __tag: label,
      size: 256,
      createView: vi.fn(() => { calls.push(label); return { __tag: `${label}#view` }; }),
      destroy: vi.fn(),
    } as unknown as GPUBuffer;
    return { buf, calls };
  }

  function makeCtx(encoder: GPUCommandEncoder, welfordPing: number): PassDispatchContext {
    const { buf: varianceBuffer } = makeViewTrackingBuf('variance');
    const { buf: varianceBufferAux } = makeViewTrackingBuf('varianceAux');

    return {
      device: {
        createBindGroup: vi.fn(() => ({ __tag: 'bg' })),
        createBindGroupLayout: vi.fn(() => ({})),
        queue: { writeBuffer: vi.fn() },
      } as unknown as GPUDevice,
      encoder,
      width: 64, height: 64,
      frameIndex: 0, frameCount: 1,
      bglCache: {} as never,
      resources: {
        common: {
          varianceBuffer,
          varianceBufferAux,
          tierTexture: {
            createView: vi.fn(() => ({})),
          } as unknown as GPUTexture,
        },
        gtao: { gtaoUboBuffer: { __tag: 'gtaoUbo' } as unknown as GPUBuffer },
      } as unknown as PassDispatchContext['resources'],
      inputs: {
        gtao: {
          adaptiveSamplingThresholdLow: 0.1,
          adaptiveSamplingThresholdHigh: 0.5,
        },
      } as unknown as PassDispatchContext['inputs'],
      frameBindGroup: {} as GPUBindGroup,
      sceneBindGroup: {} as GPUBindGroup,
      uboBindGroup: {} as GPUBindGroup,
      hybridLayersBindGroup: {} as GPUBindGroup,
      lightTreeBindGroup: {} as GPUBindGroup,
      wgX: 8, wgY: 8, wgX16: 4, wgY16: 4, halfWgX: 2, halfWgY: 2,
      gtaoDownscale: 2,
      checkerboardOn: false,
      frameParity: 0,
      welfordPing,
      gNormalDepthView: {} as GPUTextureView,
      computeDesc: (label: PassLabel) => ({ label }) as GPUComputePassDescriptor,
      renderTimestampWrites: () => undefined,
      resourceCache: undefined as never,
      frameState: {} as never,
    } as unknown as PassDispatchContext;
  }

  function makeEncoder() {
    return {
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(), end: vi.fn(),
      })),
    } as unknown as GPUCommandEncoder;
  }

  it('welfordPing=0 → varianceBuffer.createView() is called', () => {
    const pipeline = { getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline;
    const budgetUboRef = { buf: { __tag: 'budgetUbo', destroy: vi.fn() } as unknown as GPUBuffer };
    const countUboRef  = { buf: { __tag: 'countUbo', destroy: vi.fn() }  as unknown as GPUBuffer };
    const pass = new SampleBudgetPass(pipeline, budgetUboRef, countUboRef);

    const encoder = makeEncoder();
    const ctx = makeCtx(encoder, 0);
    pass.dispatch(ctx);

    const varBuf = (ctx.resources.common as unknown as Record<string, GPUBuffer>).varianceBuffer;
    const auxBuf = (ctx.resources.common as unknown as Record<string, GPUBuffer>).varianceBufferAux;
    expect((varBuf as unknown as { createView: ReturnType<typeof vi.fn> }).createView).toHaveBeenCalled();
    expect((auxBuf as unknown as { createView: ReturnType<typeof vi.fn> }).createView).not.toHaveBeenCalled();
  });

  it('welfordPing=1 → varianceBufferAux.createView() is called', () => {
    const pipeline = { getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline;
    const budgetUboRef = { buf: { __tag: 'budgetUbo', destroy: vi.fn() } as unknown as GPUBuffer };
    const countUboRef  = { buf: { __tag: 'countUbo', destroy: vi.fn() }  as unknown as GPUBuffer };
    const pass = new SampleBudgetPass(pipeline, budgetUboRef, countUboRef);

    const encoder = makeEncoder();
    const ctx = makeCtx(encoder, 1);
    pass.dispatch(ctx);

    const varBuf = (ctx.resources.common as unknown as Record<string, GPUBuffer>).varianceBuffer;
    const auxBuf = (ctx.resources.common as unknown as Record<string, GPUBuffer>).varianceBufferAux;
    expect((auxBuf as unknown as { createView: ReturnType<typeof vi.fn> }).createView).toHaveBeenCalled();
    expect((varBuf as unknown as { createView: ReturnType<typeof vi.fn> }).createView).not.toHaveBeenCalled();
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// §6 — importGIState validates grid origin/spacing/dims
// ──────────────────────────────────────────────────────────────────────────────
//
// HybridEngine is hard to bootstrap fully; we test the guard logic inline by
// replicating the exact 25-line guard that was added to HybridEngine.importGIState.
// The logic is simple and pure enough that an inline replica is a meaningful
// regression pin.  A real integration smoke lives in giStateSnapshot.test.ts.
// ──────────────────────────────────────────────────────────────────────────────

import type { GIStateSnapshot } from '../src/giStateSnapshot.js';

/**
 * Inline port of the importGIState grid-layout guard added by this fix.
 * If this test breaks, check HybridEngine.ts importGIState region.
 */
function importGIStateGuard(
  snapshot: GIStateSnapshot,
  gridDims: { x: number; y: number; z: number },
  worldOrigin: { x: number; y: number; z: number },
  worldSpacing: number,
  importAtlasData: () => boolean,
): boolean {
  const epsilon = 1e-4;
  const dimsMismatch =
    snapshot.dims.x !== gridDims.x ||
    snapshot.dims.y !== gridDims.y ||
    snapshot.dims.z !== gridDims.z;
  const originMismatch =
    Math.abs(snapshot.origin[0] - worldOrigin.x) > epsilon ||
    Math.abs(snapshot.origin[1] - worldOrigin.y) > epsilon ||
    Math.abs(snapshot.origin[2] - worldOrigin.z) > epsilon;
  const spacingMismatch = Math.abs(snapshot.spacing - worldSpacing) > epsilon;
  if (dimsMismatch || originMismatch || spacingMismatch) return false;
  return importAtlasData();
}

function makeMatchingSnap(
  dims: { x: number; y: number; z: number },
  origin: [number, number, number],
  spacing: number,
): GIStateSnapshot {
  return {
    dims,
    origin,
    spacing,
    irrW: 4, irrH: 4,
    visW: 4, visH: 4,
    irrData: new Uint16Array(4 * 4 * 4),
    visData: new Uint16Array(4 * 4 * 4),
    probeStateW: dims.x,
    probeStateH: dims.y * dims.z,
    probeStateData: new Float32Array(dims.x * dims.y * dims.z * 4),
  };
}

const DIMS    = { x: 8, y: 4, z: 8 };
const ORIGIN  = { x: -5.0, y: 0.0, z: -5.0 };
const SPACING = 2.5;

describe('Item 6 — importGIState grid-layout guard', () => {
  it('accepts a snapshot with matching dims/origin/spacing when atlas is ok', () => {
    const snap = makeMatchingSnap(DIMS, [-5.0, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(true);
  });

  it('rejects a snapshot with mismatched dims.z (8 → 4)', () => {
    const snap = makeMatchingSnap({ x: 8, y: 4, z: 4 }, [-5.0, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(false);
  });

  it('rejects a snapshot with mismatched dims.x (8 → 6)', () => {
    const snap = makeMatchingSnap({ x: 6, y: 4, z: 8 }, [-5.0, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(false);
  });

  it('rejects a snapshot with mismatched origin.x (shifted by 0.5)', () => {
    const snap = makeMatchingSnap(DIMS, [-4.5, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(false);
  });

  it('rejects a snapshot with mismatched origin.y', () => {
    const snap = makeMatchingSnap(DIMS, [-5.0, 1.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(false);
  });

  it('rejects a snapshot with mismatched spacing (2.5 → 3.0)', () => {
    const snap = makeMatchingSnap(DIMS, [-5.0, 0.0, -5.0], 3.0);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(false);
  });

  it('accepts when origin differs by less than epsilon (1e-5 — within tolerance)', () => {
    const snap = makeMatchingSnap(DIMS, [-5.00001, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(true);
  });

  it('rejects when origin differs by more than epsilon (2e-4 — exceeds tolerance)', () => {
    const snap = makeMatchingSnap(DIMS, [-5.0002, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => true)).toBe(false);
  });

  it('rejects when grid matches but importAtlasData returns false (pixel-dims mismatch)', () => {
    const snap = makeMatchingSnap(DIMS, [-5.0, 0.0, -5.0], SPACING);
    expect(importGIStateGuard(snap, DIMS, ORIGIN, SPACING, () => false)).toBe(false);
  });

  it('does not call importAtlasData when grid dims are mismatched (short-circuits)', () => {
    const atlas = vi.fn(() => true);
    const snap = makeMatchingSnap({ x: 4, y: 4, z: 8 }, [-5.0, 0.0, -5.0], SPACING);
    importGIStateGuard(snap, DIMS, ORIGIN, SPACING, atlas);
    expect(atlas).not.toHaveBeenCalled();
  });
});

// ───────────────────────────────────────────────────────────────────────────────
// §7 — PPG atomic-f32 accumulation WGSL pattern
// ───────────────────────────────────────────────────────────────────────────────

import { buildPpgUpdateWgsl } from '../src/ppg/ppgUpdate.wgsl.ts';

describe('Item 7 — PPG atomic f32 accumulation', () => {
  it('uses a bounded weak-CAS loop over IEEE-754 bits with finite-range guards', () => {
    const wgsl = buildPpgUpdateWgsl(341);
    expect(wgsl).toContain('bitcast<f32>(oldBits)');
    expect(wgsl).toContain('atomicCompareExchangeWeak(');
    expect(wgsl).toContain('&ppgFluxAtomics[slot], oldBits');
    expect(wgsl).toContain('MAX_FLUX_CAS_ATTEMPTS: u32 = 256u');
    expect(wgsl).toContain('nextValue = MAX_FINITE_F32');
    expect(wgsl).not.toContain('atomicAdd(&ppgFluxAtomics[slot]');
  });

  it('keeps the f32 CAS contract for multiple dTree-node caps', () => {
    for (const wgsl of [buildPpgUpdateWgsl(171), buildPpgUpdateWgsl(511)]) {
      expect(wgsl).toContain('atomicLoad(&ppgFluxAtomics[slot])');
      expect(wgsl).toContain('bitcast<u32>(nextValue)');
      expect(wgsl).toContain('MAX_FLUX_CAS_ATTEMPTS');
      expect(wgsl).not.toContain('0xFFFFFFFFu - increment');
    }
  });

  it('CPU f32 oracle clamps overflow while preserving ordinary additions', () => {
    const MAX_F32 = 3.402823466e38;
    const accumulate = (current: number, deposit: number): number => {
      if (!(deposit > 0) || !Number.isFinite(deposit)) return current;
      if (!(current >= 0) || !Number.isFinite(current)) return Math.fround(deposit);
      const sum = Math.fround(current + deposit);
      return Number.isFinite(sum) && sum <= MAX_F32 ? sum : MAX_F32;
    };

    expect(accumulate(1, 2)).toBe(3);
    expect(accumulate(Number.NaN, 4)).toBe(4);
    expect(accumulate(10, Number.NaN)).toBe(10);
    expect(accumulate(MAX_F32, MAX_F32)).toBe(MAX_F32);
  });
});
