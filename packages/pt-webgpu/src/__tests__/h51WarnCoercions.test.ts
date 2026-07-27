/**
 * H51-A/B/C — option validation:
 *   A: reject maxBounces outside the engine supported integer range.
 *   B: historical roughnessMap + metallicMap split no longer warns; both handles
 *      are packed into independent pt-webgpu texture slots.
 *   C: warn once listing unknown opts.extensions keys.
 *   D: reject bdptOptions.maxLightBounces outside its supported integer range.
 * H48: reject opts.denoiser when it is neither 'none', 'auto', nor 'oidn-final'.
 */
import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import { collectMaterialTextures } from '../scene/materialTextures.js';
import { PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';
import type { MaterialSpec } from '@vitrum/core';

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
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
describe('H48: unsupported denoiser rejects', () => {
  it("rejects a non-oidn-final value ('atrous') without warning", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(createPTEngine_WebGPU({
      device: makeStubDevice(),
      denoiser: 'atrous' as never,
    })).rejects.toThrow(/denoiser="atrous".*unsupported.*not degraded/s);
    expect(warn).not.toHaveBeenCalled();
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

// ── H51-A ──────────────────────────────────────────────────────────────────────
describe("H51-A: maxBounces validation", () => {
  it.each([0, 1.5, 9, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects an unsupported value (%s)",
    async (maxBounces) => {
      await expect(
        createPTEngine_WebGPU({ device: makeStubDevice(), maxBounces }),
      ).rejects.toThrow("maxBounces must be an integer in 1..8");
    },
  );

  it("accepts maxBounces within the supported range", async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      maxBounces: 4,
    });
    expect(engine.capabilities.maxBounces).toBe(4);
    engine.dispose();
  });
});

// ── H51-D ──────────────────────────────────────────────────────────────────────
describe("H51-D: bdptOptions.maxLightBounces validation", () => {
  it.each([0, 2.75, 9, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects an unsupported value (%s)",
    async (maxLightBounces) => {
      await expect(
        createPTEngine_WebGPU({
          device: makeStubDevice(),
          bdpt: true,
          bdptOptions: { maxLightBounces },
        }),
      ).rejects.toThrow(
        "bdptOptions.maxLightBounces must be an integer in 1..8",
      );
    },
  );

  it.each([1, 2, 3, 8])(
    "accepts every representative stable depth (%s)",
    async (maxLightBounces) => {
      const engine = await createPTEngine_WebGPU({
        device: makeStubDevice(),
        bdpt: true,
        bdptOptions: { maxLightBounces },
      });
      engine.dispose();
    },
  );
});

// ── H51-B ─────────────────────────────────────────────────────────────────────
describe('H51-B: distinct roughnessMap + metallicMap texture slots', () => {
  const _srgbTex = { handle: { label: 'baseColor' } }; // retained: use in baseColorMap test cases if added
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

  it('does not warn when materials have distinct roughnessMap and metallicMap handles', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    collectMaterialTextures([matWithBoth, matWithBoth]);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('roughnessMap'))).toBe(false);
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
      roughnessMap: roughTex,
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
      metallicMap: metallicTex,
    }]);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('roughnessMap'))).toBe(false);
    warn.mockRestore();
  });

  it('roughnessMap and metallicMap are both packed when both are provided', () => {
    const { linearSources, descriptors } = collectMaterialTextures([matWithBoth]);
    expect(linearSources).toEqual([roughTex.handle, metallicTex.handle]);
    expect(descriptors[2]).toBe(0);
    expect(descriptors[26]).toBe(1);
  });
});

// ── H51-C ─────────────────────────────────────────────────────────────────────
describe('H51-C: extensions are a strict construction boundary', () => {
  it('rejects all unknown extension keys in one error', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeStubDevice(),
      extensions: {
        'my.custom.key': 42,
        'another.unknown': 'value',
      },
    })).rejects.toThrow(/my\.custom\.key.*another\.unknown/);
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

  it('rejects graduated legacy keys with named-option migration guidance', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeStubDevice(),
      extensions: {
        'vitrum.ptWebgpu.spectralHeroWavelength': true,
        'vitrum.ptWebgpu.bdpt': true,
        'vitrum.ptWebgpu.oidnModelUrl': '/model.onnx',
      },
    })).rejects.toThrow(/spectral:true.*bdpt:true.*denoiser:'oidn-final'/);
  });
});
