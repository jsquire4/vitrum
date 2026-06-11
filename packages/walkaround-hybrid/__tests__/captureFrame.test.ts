/**
 * captureFrame — walkaround-hybrid backend.
 *
 * Tests:
 *   1. captureFrame({ colorSpace: 'output' }) resolves to null before the first
 *      frame (pipeline not yet initialised — captureOutputFrame returns null).
 *   2. captureFrame() returns null before the first frame (getProgressiveSeedTexture
 *      returns null when the pipeline is not yet initialised).
 *   3. captureOutputFrame mock-device harness: verifies the offscreen render pass
 *      geometry (beginRenderPass called, fullscreen draw(3,1,0,0)), the staging
 *      copy (copyTextureToBuffer called with 256-aligned bytesPerRow), and the
 *      unorm [0,1] decode path.
 *
 * Tests 1–2 use the high-level HybridEngine via createWalkaroundEngine_Hybrid.
 * Test 3 drives WalkaroundGPUPipeline.captureOutputFrame directly with a
 * fine-grained mock device that simulates a completed frame state.
 */
import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

// Install GPUBufferUsage / GPUTextureUsage / GPUMapMode globals BEFORE importing
// any source module that reads them at init time (resourceManager, WalkaroundGPUPipeline).
installWebGPUPolyfills();

import type { HybridEngineOptions } from '../src/HybridEngineOptions.js';
import { createWalkaroundEngine_Hybrid } from '../src/index.js';
import { WalkaroundGPUPipeline } from '../src/pipeline/WalkaroundGPUPipeline.js';

// ── Shared stub device (minimal, used by tests 1–2) ─────────────────────────

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 8,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxBindGroupsPlusVertexBuffers: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxStorageBufferBindingSize: 1 << 30,
      maxUniformBufferBindingSize: 65536,
      maxDynamicStorageBuffersPerPipelineLayout: 0,
      maxDynamicUniformBuffersPerPipelineLayout: 0,
      maxTextureArrayLayers: 256,
      maxTextureDimension2D: 8192,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupsPerDimension: 65535,
      maxVertexBuffers: 8,
      maxBufferSize: 1 << 30,
    },
    features: new Set<string>(),
    createBuffer: vi.fn(() => ({
      destroy: vi.fn(),
      getMappedRange: vi.fn(() => new ArrayBuffer(16)),
      unmap: vi.fn(),
      mapAsync: vi.fn(() => Promise.resolve()),
    })),
    createTexture: vi.fn(() => ({
      destroy: vi.fn(),
      createView: vi.fn(() => ({})),
      width: 64, height: 64, depthOrArrayLayers: 1,
    })),
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({})),
    createRenderPipeline: vi.fn(() => ({})),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(), end: vi.fn(),
      })),
      beginRenderPass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(),
        draw: vi.fn(), end: vi.fn(),
      })),
      copyTextureToBuffer: vi.fn(),
      clearBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    createQuerySet: vi.fn(() => ({ destroy: vi.fn() })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
    destroy: vi.fn(),
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(() => Promise.resolve(null)),
  } as unknown as GPUDevice;
}

const MINIMAL_OPTS: Pick<HybridEngineOptions, 'device' | 'width' | 'height'> = {
  device: makeStubDevice(),
  width: 64,
  height: 64,
};

// ── Tests 1–2: high-level HybridEngine ────────────────────────────────────────

describe('captureFrame walkaround-hybrid', () => {
  it("captureFrame({ colorSpace: 'output' }) resolves to null before first frame", async () => {
    const engine = await createWalkaroundEngine_Hybrid(MINIMAL_OPTS as HybridEngineOptions);
    // No renderFrame call — pipeline._initialized is false → captureOutputFrame returns null.
    const frame = await engine.captureFrame?.({ colorSpace: 'output' });
    expect(frame).toBeNull();
    engine.dispose();
  });

  it("captureFrame() (linear default) returns null before first frame", async () => {
    const engine = await createWalkaroundEngine_Hybrid(MINIMAL_OPTS as HybridEngineOptions);
    // No renderFrame call — getProgressiveSeedTexture is null → captureFrame is null.
    const frame = await engine.captureFrame?.();
    expect(frame).toBeNull();
    engine.dispose();
  });

  it('engine exposes captureFrame as a function', async () => {
    const engine = await createWalkaroundEngine_Hybrid(MINIMAL_OPTS as HybridEngineOptions);
    expect(typeof engine.captureFrame).toBe('function');
    engine.dispose();
  });
});

// ── Test 3: WalkaroundGPUPipeline.captureOutputFrame mock-device harness ──────

/**
 * Build a mock device suitable for inspecting captureOutputFrame internals.
 *
 * The mock simulates a fully-initialised pipeline by:
 *   - returning a mock render pipeline from createRenderPipelineAsync
 *   - providing a staging buffer whose mapAsync resolves and getMappedRange
 *     returns a zeroed buffer (all pixels rgba8=0,0,0,0)
 *   - tracking beginRenderPass, draw, copyTextureToBuffer calls
 */
function makeCaptureHarness(W: number, H: number) {
  const drawCalls: Array<[number, number, number, number]> = [];
  const copyToBufferCalls: Array<{ bytesPerRow: number; width: number; height: number }> = [];
  const renderPassBeginCalls: number[] = [];

  // Enough bytes for a 256-row-aligned rgba8 readback.
  const bpr = Math.ceil((W * 4) / 256) * 256;
  const readSize = bpr * H;
  // All zeros → decoded pixels should be 0.0 (but format is valid).
  const mappedData = new ArrayBuffer(readSize);

  const stagingBuffer = {
    destroy: vi.fn(),
    mapAsync: vi.fn(() => Promise.resolve()),
    getMappedRange: vi.fn(() => mappedData.slice(0)),
    unmap: vi.fn(),
    size: readSize,
    usage: 0,
  };

  const offscreenTexView = {};
  const offscreenTex = {
    destroy: vi.fn(),
    createView: vi.fn(() => offscreenTexView),
    width: W,
    height: H,
    format: 'rgba8unorm',
  };

  const renderPass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    draw: vi.fn((vertexCount: number, instanceCount: number, firstVertex: number, firstInstance: number) => {
      drawCalls.push([vertexCount, instanceCount, firstVertex, firstInstance]);
    }),
    end: vi.fn(),
  };

  const encoder = {
    beginComputePass: vi.fn(() => ({
      setPipeline: vi.fn(), setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(), end: vi.fn(),
    })),
    beginRenderPass: vi.fn(() => {
      renderPassBeginCalls.push(1);
      return renderPass;
    }),
    copyTextureToBuffer: vi.fn((
      _src: unknown,
      dst: { buffer: unknown; bytesPerRow: number },
      extent: { width: number; height: number },
    ) => {
      copyToBufferCalls.push({ bytesPerRow: dst.bytesPerRow, width: extent.width, height: extent.height });
    }),
    clearBuffer: vi.fn(),
    finish: vi.fn(() => ({})),
  };

  const mockPipeline = {};
  const mockBgl = {};

  let textureCallCount = 0;
  const device = {
    limits: {
      maxStorageBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 8,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxBindGroupsPlusVertexBuffers: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxStorageBufferBindingSize: 1 << 30,
      maxUniformBufferBindingSize: 65536,
      maxDynamicStorageBuffersPerPipelineLayout: 0,
      maxDynamicUniformBuffersPerPipelineLayout: 0,
      maxTextureArrayLayers: 256,
      maxTextureDimension2D: 8192,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupsPerDimension: 65535,
      maxVertexBuffers: 8,
      maxBufferSize: 1 << 30,
    },
    features: new Set<string>(),
    createBuffer: vi.fn((desc: { label?: string }) => {
      if (desc?.label?.includes('capture-staging') || desc?.label?.includes('captureFrame')) {
        return stagingBuffer;
      }
      return { destroy: vi.fn(), getMappedRange: vi.fn(() => new ArrayBuffer(16)), unmap: vi.fn(), mapAsync: vi.fn(() => Promise.resolve()) };
    }),
    createTexture: vi.fn((desc: { label?: string }) => {
      textureCallCount++;
      if (desc?.label?.includes('capture-offscreen')) return offscreenTex;
      return { destroy: vi.fn(), createView: vi.fn(() => ({})), width: W, height: H };
    }),
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({ getCompilationInfo: vi.fn(async () => ({ messages: [] })) })),
    createComputePipeline: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(() => Promise.resolve({})),
    createRenderPipeline: vi.fn(() => mockPipeline),
    createRenderPipelineAsync: vi.fn(() => Promise.resolve(mockPipeline)),
    createBindGroupLayout: vi.fn(() => mockBgl),
    createPipelineLayout: vi.fn(() => ({})),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => encoder),
    createQuerySet: vi.fn(() => ({ destroy: vi.fn() })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
    destroy: vi.fn(),
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(() => Promise.resolve(null)),
  } as unknown as GPUDevice;

  return { device, drawCalls, copyToBufferCalls, renderPassBeginCalls, stagingBuffer, offscreenTex, bpr };
}

describe('WalkaroundGPUPipeline.captureOutputFrame — mock-device harness', () => {
  it('returns null when pipeline is not initialized', async () => {
    const harness = makeCaptureHarness(4, 4);
    const pipeline = new WalkaroundGPUPipeline(harness.device, 4, 4);
    const result = await pipeline.captureOutputFrame();
    expect(result).toBeNull();
    pipeline.dispose();
  });

  it('offscreen render pass geometry: beginRenderPass called, draw(3,1,0,0) issued', async () => {
    const W = 4;
    const H = 4;
    const harness = makeCaptureHarness(W, H);

    // Construct and manually force _initialized state by accessing internal fields
    // via a cast — this is a mock-device harness, not a real GPU test.
    const pipeline = new WalkaroundGPUPipeline(harness.device, W, H) as unknown as Record<string, unknown>;

    // Directly set the fields that captureOutputFrame guards on.
    pipeline['_initialized'] = true;
    pipeline['_width'] = W;
    pipeline['_height'] = H;
    pipeline['_swapChainFormat'] = 'bgra8unorm'; // != rgba8unorm → forces lazy compile path
    pipeline['_compositeUboRef'] = { buf: { destroy: vi.fn() } }; // non-null UBO ref
    // Provide a mock compositePass (pipeline getter is read for format check)
    pipeline['_compositePass'] = {
      pipeline: {},
    };
    // Provide mock frame resources (resolvedTexture + compositeSampler)
    pipeline['_res'] = {
      common: {
        resolvedTexture: { createView: vi.fn(() => ({})) },
        compositeSampler: {},
      },
    };
    // Provide a minimal _bglCache
    pipeline['_bglCache'] = {};

    const result = await (pipeline as unknown as WalkaroundGPUPipeline).captureOutputFrame();

    // The result is a Float32Array (all zeros since mapping returns zeroed data).
    expect(result).not.toBeNull();
    expect(result).toBeInstanceOf(Float32Array);
    expect(result!.length).toBe(W * H * 4);

    // beginRenderPass was called at least once (the capture pass).
    expect(harness.renderPassBeginCalls.length).toBeGreaterThanOrEqual(1);

    // draw(3, 1, 0, 0) — fullscreen triangle.
    expect(harness.drawCalls).toContainEqual([3, 1, 0, 0]);

    // copyTextureToBuffer was called with 256-row-aligned bytesPerRow.
    expect(harness.copyToBufferCalls.length).toBeGreaterThanOrEqual(1);
    const copyCall = harness.copyToBufferCalls[harness.copyToBufferCalls.length - 1]!;
    expect(copyCall.bytesPerRow % 256).toBe(0); // GPUTextureCopyView alignment
    expect(copyCall.width).toBe(W);
    expect(copyCall.height).toBe(H);

    // Null out _res and _initialized before dispose so destroyFrameResources is not
    // called on the partial-mock _res (the mock only stubs the fields that
    // captureOutputFrame accesses; the full FrameResources dispose path requires
    // all sub-structs including restirDI / restirGI).
    pipeline['_initialized'] = false;
    pipeline['_res'] = null;
    (pipeline as unknown as WalkaroundGPUPipeline).dispose();
  });

  it('unorm decode: all-255 staging bytes → output Float32 values all 1.0', async () => {
    const W = 2;
    const H = 2;
    const harness = makeCaptureHarness(W, H);

    // Override getMappedRange to return all-255 bytes.
    const bpr = harness.bpr;
    const allOnes = new Uint8Array(bpr * H).fill(255);
    const allOnesBuf = allOnes.buffer;
    harness.stagingBuffer.getMappedRange = vi.fn(() => allOnesBuf.slice(0));

    const pipeline = new WalkaroundGPUPipeline(harness.device, W, H) as unknown as Record<string, unknown>;
    pipeline['_initialized'] = true;
    pipeline['_width'] = W;
    pipeline['_height'] = H;
    pipeline['_swapChainFormat'] = 'rgba8unorm'; // matches CAPTURE_FORMAT → reuse existing pipeline
    pipeline['_compositeUboRef'] = { buf: { destroy: vi.fn() } };
    pipeline['_compositePass'] = { pipeline: {} };
    pipeline['_res'] = {
      common: {
        resolvedTexture: { createView: vi.fn(() => ({})) },
        compositeSampler: {},
      },
    };
    pipeline['_bglCache'] = {};

    const result = await (pipeline as unknown as WalkaroundGPUPipeline).captureOutputFrame();

    expect(result).not.toBeNull();
    // All channels should decode to 255/255 = 1.0.
    for (let i = 0; i < result!.length; i++) {
      expect(result![i]).toBeCloseTo(1.0, 5);
    }

    // Null out _res and _initialized before dispose (same reason as the geometry test).
    pipeline['_initialized'] = false;
    pipeline['_res'] = null;
    (pipeline as unknown as WalkaroundGPUPipeline).dispose();
  });
});
