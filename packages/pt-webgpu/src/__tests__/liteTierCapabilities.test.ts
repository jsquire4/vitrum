/**
 * H12 — lite-tier capabilities truth: the lite kernel only binds directional
 * lighting and procedural-sky; analytic shapes, HDRI, area-light emitters, and
 * BDPT are absent from the lite bind layout.
 *
 * Also covers the contract-honesty fix: lite-tier supportDetails must reflect
 * the actual lite binding budget (not the full-tier ledger).
 */
import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function makeLiteDevice(): GPUDevice {
  return {
    // Limits below the full-tier threshold → traceTier resolves to 'lite'.
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
    },
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

function makeFullDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

/** A lite device that also has the GPU stubs needed to call setScene(). */
function makeLiteDeviceForSetScene(): GPUDevice {
  installGpuConstStubs();
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
    },
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createCommandEncoder: vi.fn(),
    ...textureStubMethods(),
  } as unknown as GPUDevice;
}

describe('H12: lite-tier capabilities truth', () => {
  it('lite tier: supportedAnalyticShapes is empty', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    expect(engine.capabilities.supportedAnalyticShapes.size).toBe(0);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportedEmitterKinds contains only directional', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const kinds = engine.capabilities.supportedEmitterKinds;
    expect(kinds.has('directional')).toBe(true);
    expect(kinds.has('point')).toBe(false);
    expect(kinds.has('spot')).toBe(false);
    expect(kinds.has('rect-area')).toBe(false);
    expect(kinds.has('disc-area')).toBe(false);
    expect(kinds.has('mesh-area')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportedEnvironmentKinds is none + procedural-sky only (no hdri)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const envs = engine.capabilities.supportedEnvironmentKinds;
    expect(envs?.has('none')).toBe(true);
    expect(envs?.has('procedural-sky')).toBe(true);
    expect(envs?.has('hdri')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: pt-webgpu-bdpt is absent from experimentalFeatures even with bdpt:true', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice(), bdpt: true });
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-bdpt')).toBe(false);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-lite-tier')).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier: analytic shapes, all emitter kinds, hdri, and bdpt are available', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeFullDevice(), bdpt: true });
    expect(engine.capabilities.supportedAnalyticShapes.size).toBeGreaterThan(0);
    expect(engine.capabilities.supportedEmitterKinds.has('rect-area')).toBe(true);
    expect(engine.capabilities.supportedEnvironmentKinds?.has('hdri')).toBe(true);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-bdpt')).toBe(true);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-lite-tier')).toBe(false);
    engine.dispose();
  });

  it('lite tier: supportDetails emitters shows only directional as native, others unsupported', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.emitters.directional).toBe('native');
    expect(sd.emitters.point).toBe('unsupported');
    expect(sd.emitters.spot).toBe('unsupported');
    expect(sd.emitters['rect-area']).toBe('unsupported');
    expect(sd.emitters['disc-area']).toBe('unsupported');
    expect(sd.emitters['mesh-area']).toBe('unsupported');
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportDetails environments shows hdri as unsupported', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.environments.hdri).toBe('unsupported');
    expect(sd.environments.none).toBe('native');
    // procedural-sky is heuristic tint (not a full Preetham model), so 'approximate'
    expect(sd.environments['procedural-sky']).not.toBe('unsupported');
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportDetails analyticShapes all unsupported', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    for (const grade of Object.values(sd.analyticShapes)) {
      expect(grade).toBe('unsupported');
    }
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportDetails analytic primitive is unsupported', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.primitives.analytic).toBe('unsupported');
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier: supportDetails emitters all native', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeFullDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.emitters.directional).toBe('native');
    expect(sd.emitters.point).toBe('native');
    expect(sd.emitters['rect-area']).toBe('native');
    engine.dispose();
  });

  it('lite tier: setScene warns when scene contains analytic primitives', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDeviceForSetScene() });
    warn.mockClear();
    // The warn must fire BEFORE the GPU upload — use try/catch for GPU stub noise.
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'a',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),  // cx,cy,cz,radius
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('analytic') && c.includes('Lite tier'))).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene warns when scene contains non-directional emitters', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDeviceForSetScene() });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [{ kind: 'point', id: 'p', position: [0, 0, 0], color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('point') && c.includes('Lite tier'))).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene warns when scene has an hdri environment', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDeviceForSetScene() });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'hdri', hdri: undefined as unknown as NonNullable<unknown> },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('hdri') && c.includes('Lite tier'))).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });
});
