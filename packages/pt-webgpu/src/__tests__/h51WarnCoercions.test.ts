/**
 * H51-A/B/C — Coercion warns:
 *   A: warn when opts.maxBounces exceeds the engine's clamp cap.
 *   B: warn when a material has distinct roughnessMap + metallicMap handles
 *      (ORM single-slot drops metallicMap).
 *   C: warn once listing unknown opts.extensions keys.
 * H48: warn when opts.denoiser is neither 'none' nor 'oidn-final'.
 */
import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import { collectMaterialTextures } from '../scene/materialTextures.js';
import type { MaterialSpec } from '@vitrum/core';

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

// ── H48 ──────────────────────────────────────────────────────────────────────
// (H48 is already tested by unsupportedDenoiserDegrade.test.ts; this block
//  serves as the authoritative H48 statement.)
describe('H48: unsupported denoiser warns', () => {
  it("warns when denoiser is set to a non-oidn-final value ('atrous')", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice(), denoiser: 'atrous' });
    expect(warn.mock.calls.some((c) => String(c[0]).includes('denoiser="atrous"'))).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it("does not warn when denoiser is 'none'", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice(), denoiser: 'none' });
    expect(warn.mock.calls.some((c) => String(c[0]).includes('denoiser='))).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });
});

// ── H51-A ─────────────────────────────────────────────────────────────────────
describe('H51-A: maxBounces clamp warns', () => {
  it('warns when maxBounces exceeds the experimental cap (8)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice(), maxBounces: 20 });
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('maxBounces=20')),
    ).toBe(true);
    // Clamp stays: capability reports the capped value.
    expect(engine.capabilities.maxBounces).toBe(8);
    engine.dispose();
    warn.mockRestore();
  });

  it('does not warn when maxBounces is within the cap', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice(), maxBounces: 4 });
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('maxBounces=')),
    ).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });
});

// ── H51-B ─────────────────────────────────────────────────────────────────────
describe('H51-B: distinct roughnessMap + metallicMap warns', () => {
  const srgbTex = { handle: { label: 'baseColor' } };
  const roughTex = { handle: { label: 'roughness' } };
  const metallicTex = { handle: { label: 'metallic' } }; // different handle

  const matWithBoth = {
    baseColor: [0.8, 0.2, 0.1] as [number, number, number],
    roughness: 0.5,
    metallic: 0.0,
    roughnessMap: roughTex as unknown as import('@vitrum/core').TextureRef,
    metallicMap: metallicTex as unknown as import('@vitrum/core').TextureRef,
  } satisfies MaterialSpec;

  const matWithSame = {
    baseColor: [0.8, 0.2, 0.1] as [number, number, number],
    roughness: 0.5,
    metallic: 0.0,
    roughnessMap: roughTex as unknown as import('@vitrum/core').TextureRef,
    metallicMap: roughTex as unknown as import('@vitrum/core').TextureRef, // same handle
  } satisfies MaterialSpec;

  it('warns once when materials have distinct roughnessMap and metallicMap handles', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    collectMaterialTextures([matWithBoth, matWithBoth]); // two materials with the issue
    const warnCalls = warn.mock.calls.filter((c) => String(c[0]).includes('roughnessMap'));
    // Warn fires exactly once per collectMaterialTextures call, not once per material.
    expect(warnCalls.length).toBe(1);
    warn.mockRestore();
  });

  it('does not warn when roughnessMap and metallicMap share the same handle', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    collectMaterialTextures([matWithSame]);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('roughnessMap'))).toBe(false);
    warn.mockRestore();
  });

  it('does not warn when only roughnessMap is set (no metallicMap)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    collectMaterialTextures([{
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      roughnessMap: roughTex as unknown as import('@vitrum/core').TextureRef,
    }]);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('roughnessMap'))).toBe(false);
    warn.mockRestore();
  });

  it('does not warn when only metallicMap is set (no roughnessMap)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    collectMaterialTextures([{
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      metallicMap: metallicTex as unknown as import('@vitrum/core').TextureRef,
    }]);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('roughnessMap'))).toBe(false);
    warn.mockRestore();
  });

  it('roughnessMap is used (not metallicMap) when both are provided', () => {
    const { linearSources } = collectMaterialTextures([matWithBoth]);
    // roughnessMap.handle ('roughness') appears in the linear sources.
    expect(linearSources).toContain(roughTex.handle);
    // metallicMap.handle ('metallic') should NOT appear (it was dropped).
    expect(linearSources).not.toContain(metallicTex.handle);
  });
});

// ── H51-C ─────────────────────────────────────────────────────────────────────
describe('H51-C: unknown extensions keys warn once at construction', () => {
  it('warns on unknown extension keys', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      extensions: {
        'my.custom.key': 42,
        'another.unknown': 'value',
      },
    });
    const extWarn = warn.mock.calls.find((c) =>
      String(c[0]).includes('Unknown extensions keys'),
    );
    expect(extWarn).toBeDefined();
    expect(String(extWarn?.[0])).toContain('my.custom.key');
    expect(String(extWarn?.[0])).toContain('another.unknown');
    engine.dispose();
    warn.mockRestore();
  });

  it('does not warn when extensions is empty or absent', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const e1 = await createPTEngine_WebGPU({ device: makeStubDevice(), extensions: {} });
    const e2 = await createPTEngine_WebGPU({ device: makeStubDevice() });
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('Unknown extensions keys')),
    ).toBe(false);
    e1.dispose();
    e2.dispose();
    warn.mockRestore();
  });

  it('does not warn for graduated legacy extension keys (spectral, bdpt, oidn)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      extensions: {
        'vitrum.ptWebgpu.spectralHeroWavelength': true,
        'vitrum.ptWebgpu.bdpt': true,
        'vitrum.ptWebgpu.oidnModelUrl': '/model.onnx',
      },
    });
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('Unknown extensions keys')),
    ).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });
});
