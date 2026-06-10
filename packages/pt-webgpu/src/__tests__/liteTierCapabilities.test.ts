/**
 * H12 — lite-tier capabilities truth: the lite kernel only binds directional
 * lighting and procedural-sky; analytic shapes, HDRI, area-light emitters, and
 * BDPT are absent from the lite bind layout.
 */
import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

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
});
