import { describe, expect, it, vi } from 'vitest';
import type { BackendTexture, ProgressStats, Scene } from '@vitrum/core';
import { asBackendTexture, asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { GpuResources } from '../gpuResources.js';
import { PT_WEBGPU_SEED_BLIT_WGSL } from '../wgsl/seedBlit.wgsl.js';
import { composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

// ── seedAccumulator (P8 — progressive walkaround→PT handoff, increment 1) ─────
//
// The seed is a DECAYING PRIOR of virtual weight W: the accum buffers are
// written as `(seedRGB·W, W)` so the displayed mean after M real samples is
// `μ + W/(W+M)·(seed − μ)` and the CONVERGED mean is the no-seed result μ for
// any seed. The load-bearing correctness invariant — verified below — is that
// the virtual weight W is NOT counted as accumulated samples (it must not bump
// the engine's SPP counter, or convergence/telemetry would over-report).
//
// These are string-contract + counter checks against a mock device (no real
// GPU); the on-device A/B that the decay actually holds is
// wsl-gpu/scripts/progressive-seed-invariance-ab.ts.

describe('pt-webgpu seedBlit WGSL contract', () => {
  it('composes (luminance + the SeedParams/accum/variance bindings present)', () => {
    // Self-contained: pulls in the canonical Rec.709 luminance() it calls.
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('fn luminance(c: vec3f) -> f32');
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('struct SeedParams');
    // Resolution-mismatch handling: a filtering sampler + normalised-UV sample.
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('var seedTexture: texture_2d<f32>');
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('var seedSampler: sampler');
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('textureSampleLevel(seedTexture, seedSampler, uv, 0.0)');
    // The two read_write storage targets — the kernel's accum + variance buffers.
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain(
      'var<storage, read_write> accumBuffer: array<vec4f>',
    );
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain(
      'var<storage, read_write> varianceMomentsBuffer: array<vec4f>',
    );
  });

  it('writes the decaying-prior accum value vec4f(seedRGB·W, W)', () => {
    // The exact accum write the kernel-matched math depends on: count == W (the
    // virtual weight) and colour == seedRGB·W, so display = accum.xyz/accum.w
    // starts at seedRGB and the prior weight is W.
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('accumBuffer[pixelIndex] = vec4f(seedRgb * W, W);');
  });

  it('writes consistent variance moments vec3(lum·W, lum²·W, W)', () => {
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain('let lum = luminance(seedRgb);');
    expect(PT_WEBGPU_SEED_BLIT_WGSL).toContain(
      'varianceMomentsBuffer[pixelIndex] = vec4f(lum * W, lum * lum * W, W, 0.0);',
    );
  });

  it('matches the accumBuffer layout the path-trace kernel accumulates into', () => {
    // The kernel adds `vec4f(sampleColor, 1.0)` per real sample and reads
    // `accum.xyz / max(accum.w, 1.0)`. The seed must write into that SAME
    // (Σcolor, count) layout — pin both sides so a kernel layout change that
    // would silently break the seed is caught here.
    const kernel = composePtWebgpuTraceWgsl(false);
    expect(kernel).toContain('accum = accum + vec4f(sampleColor, 1.0);');
    expect(kernel).toContain('let display = accum.xyz / max(accum.w, 1.0);');
  });
});

describe('pt-webgpu seedAccumulator capability', () => {
  it('reports supportsAccumulatorSeed === true', async () => {
    const engine = await createPTEngine_WebGPU({
      device: {
        createCommandEncoder: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        lost: new Promise<never>(() => {}),
      } as unknown as GPUDevice,
    });
    expect(engine.capabilities.supportsAccumulatorSeed).toBe(true);
    expect(typeof engine.seedAccumulator).toBe('function');
    engine.dispose();
  });
});

// ── Mock device rich enough to run seedAccumulator + a paused renderFrame ──────
// seedAccumulator needs: createTexture/createView (accum + aux), createBuffer
// (accum/variance/params/placeholder), createShaderModule, createComputePipeline
// (+ getBindGroupLayout), createSampler, createBindGroup, createCommandEncoder →
// beginComputePass, queue.writeBuffer/submit. The PAUSED renderFrame path reads
// the SPP counter back via onProgress WITHOUT a compute dispatch.
function makeSeedCapableDevice(): GPUDevice {
  installGpuConstStubs();
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const encoder = {
    beginComputePass: vi.fn(() => pass),
    clearBuffer: vi.fn(),
    finish: vi.fn(() => ({})),
  };
  return {
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn(), submit: vi.fn() },
    createBuffer: vi.fn((desc?: { label?: string }) => ({
      label: desc?.label ?? '',
      destroy: vi.fn(),
    })),
    ...textureStubMethods(),
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => encoder),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0.1 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

const fakeSeed = (): BackendTexture =>
  asBackendTexture<'webgpu', GPUTexture>({
    createView: vi.fn(() => ({})),
  } as unknown as GPUTexture);

describe('pt-webgpu seedAccumulator does NOT advance the SPP counter', () => {
  it('reports 0 accumulated samples after a seed (the virtual weight is not real SPP)', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeSeedCapableDevice() });
    engine.setScene(makeScene());

    // Seed with a deliberately large virtual weight. If the implementation
    // (wrongly) folded W into #samplesAccumulated, the paused report below would
    // be 1000 (or the clamp), not 0.
    engine.seedAccumulator?.(fakeSeed(), { weight: 1000, width: 16, height: 16 });

    // Read the SPP counter back through the PAUSED renderFrame path: it emits
    // onProgress with current == #samplesAccumulated and does NOT dispatch.
    const progress: ProgressStats[] = [];
    engine.onProgress?.((p) => progress.push(p));
    engine.pause();
    engine.renderFrame({
      viewMatrix: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ])),
      projMatrix: asMat4(new Float32Array(16)),
      viewport: { width: 16, height: 16, devicePixelRatio: 1 },
      frameIndex: 0,
      frameSeed: 1,
      quality: { samplesTarget: 16, bounces: 2, resolutionFactor: 1 },
    });

    const last = progress.at(-1);
    expect(last?.kind).toBe('pt-spp');
    if (last?.kind === 'pt-spp') {
      // The seed weight W=1000 must NOT count as accumulated samples.
      expect(last.current).toBe(0);
    }
    engine.dispose();
  });

  it('runs the seed-blit compute pass (pipeline + sampler + params + dispatch)', async () => {
    const device = makeSeedCapableDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());
    engine.seedAccumulator?.(fakeSeed(), { weight: 8, width: 16, height: 16 });

    // The seed-blit shader was compiled + a compute pipeline + a filtering
    // sampler were created, and the seed-blit params UBO was written.
    expect(device.createShaderModule).toHaveBeenCalled();
    expect(device.createComputePipeline).toHaveBeenCalled();
    expect(device.createSampler).toHaveBeenCalled();
    expect(device.queue.writeBuffer).toHaveBeenCalled();
    engine.dispose();
  });

  it('starts a new cohort by clearing every temporal-history family', async () => {
    const clearTemporal = vi.spyOn(GpuResources.prototype, 'clearTemporalBuffers');
    const engine = await createPTEngine_WebGPU({ device: makeSeedCapableDevice() });
    engine.setScene(makeScene());
    clearTemporal.mockClear();

    engine.seedAccumulator?.(fakeSeed(), { weight: 8, width: 16, height: 16 });

    expect(clearTemporal).toHaveBeenCalledOnce();
    engine.dispose();
    clearTemporal.mockRestore();
  });
});
