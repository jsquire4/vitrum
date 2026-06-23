/**
 * H51-A/B/C — Coercion warns:
 *   A: warn when opts.maxBounces exceeds the engine's clamp cap.
 *   B: historical roughnessMap + metallicMap split no longer warns; both handles
 *      are packed into independent pt-webgpu texture slots.
 *   C: warn once listing unknown opts.extensions keys.
 *   D: validate/warn bdptOptions.maxLightBounces rather than silently coercing.
 * H48: warn when opts.denoiser is neither 'none', 'auto', nor 'oidn-final'.
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

// ── H51-D ─────────────────────────────────────────────────────────────────────
describe('H51-D: bdptOptions.maxLightBounces validates and warns predictably', () => {
  it('keeps bdpt:true default endpoint-only without a multi-vertex warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const onWarning = vi.fn();
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      bdpt: true,
      onWarning,
    });

    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('multi-vertex BDPT research path')),
    ).toBe(false);
    expect(onWarning).not.toHaveBeenCalledWith(expect.objectContaining({
      code: 'pt-webgpu.bdpt-multivertex-research-mode',
    }));

    engine.dispose();
    warn.mockRestore();
  });

  it('throws when maxLightBounces is below the structural minimum', async () => {
    await expect(
      createPTEngine_WebGPU({
        device: makeStubDevice(),
        bdpt: true,
        bdptOptions: { maxLightBounces: 0 },
      }),
    ).rejects.toThrow('bdptOptions.maxLightBounces must be a finite number >= 1');
  });

  it('warns when maxLightBounces exceeds the supported cap (8)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const onWarning = vi.fn();
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      bdpt: true,
      bdptOptions: { maxLightBounces: 20, experimentalMultiVertex: true },
      onWarning,
    });

    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('bdptOptions.maxLightBounces=20')),
    ).toBe(true);
    expect(onWarning).toHaveBeenCalledWith(expect.objectContaining({
      code: 'pt-webgpu.bdpt-max-light-bounces-clamped',
      details: { requested: 20, clampedTo: 8 },
    }));

    engine.dispose();
    warn.mockRestore();
  });

  it('warns when maxLightBounces explicitly opts into multi-vertex BDPT', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const onWarning = vi.fn();
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      bdpt: true,
      bdptOptions: { maxLightBounces: 2, experimentalMultiVertex: true },
      onWarning,
    });

    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('multi-vertex BDPT research path')),
    ).toBe(true);
    expect(onWarning).toHaveBeenCalledWith(expect.objectContaining({
      code: 'pt-webgpu.bdpt-multivertex-research-mode',
      details: expect.objectContaining({
        requested: 2,
        resolved: 2,
        safeDefault: 1,
        experimentalMultiVertex: true,
        promotionReady: false,
        currentEstimator: 'additive-sidecar-not-weighted-against-eye-path',
        blocker: 'not-weighted-against-regular-eye-path-strategy',
        requiredEstimator: 'multi-vertex-light-subpath-strategies-weighted-against-regular-eye-path-strategy',
        safeAlternative: 'omit bdptOptions.maxLightBounces or set maxLightBounces:1',
        evidencePath: 'tools/radiometric-ab/results-bdpt.json',
      }),
    }));

    engine.dispose();
    warn.mockRestore();
  });

  it('rejects multi-vertex BDPT unless the research flag is explicit', async () => {
    await expect(
      createPTEngine_WebGPU({
        device: makeStubDevice(),
        bdpt: true,
        bdptOptions: { maxLightBounces: 2 },
      }),
    ).rejects.toThrow('bdptOptions.experimentalMultiVertex=true');
  });

  it('warns when maxLightBounces is fractional and rounds down', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const onWarning = vi.fn();
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      bdpt: true,
      bdptOptions: { maxLightBounces: 2.75, experimentalMultiVertex: true },
      onWarning,
    });

    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('rounding down to integer 2')),
    ).toBe(true);
    expect(onWarning).toHaveBeenCalledWith(expect.objectContaining({
      code: 'pt-webgpu.bdpt-max-light-bounces-rounded',
      details: { requested: 2.75, roundedTo: 2 },
    }));

    engine.dispose();
    warn.mockRestore();
  });
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
