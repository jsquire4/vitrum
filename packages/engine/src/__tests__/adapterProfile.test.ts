import { describe, expect, it } from 'vitest';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
} from '@vitrum/walkaround-hybrid';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from '@vitrum/pt-webgpu';
import { probeAdapterProfile } from '../adapterProfile.js';

/** Build a fake GPUAdapter-shaped object with the given storage limits and
 *  optional adapter info. No `queue` ⇒ treated as a GPUAdapter (reads `.info`).
 *  This exercises the full verdict path without a real GPU. */
function fakeAdapter(
  buf: number,
  tex: number,
  info?: { vendor?: string; architecture?: string },
): GPUAdapter {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: buf,
      maxStorageTexturesPerShaderStage: tex,
    },
    ...(info ? { info } : {}),
  } as unknown as GPUAdapter;
}

const FULL_BUF = HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!;
const FULL_TEX = HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!;
const LITE_BUF = HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']!;
const LITE_TEX = HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!;

describe('probeAdapterProfile — verdict logic (no GPU)', () => {
  it('16/8 hardware is full hybrid and pt-webgpu lite', async () => {
    const p = await probeAdapterProfile(
      fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' }),
    );
    expect(p.hasWebGPU).toBe(true);
    expect(p.hybridCapable).toBe(true);
    expect(p.hybridLiteCapable).toBe(true);
    expect(p.ptWebgpuTier).toBe('lite');
    expect(p.isSoftwareAdapter).toBe(false);
    expect(p.recommendedRealtimeTier).toBe('ultra');
    expect(p.recommendedHeroBackend).toBe('pt-webgpu-lite');
    expect(p.maxStorageBuffersPerStage).toBe(16);
    expect(p.maxStorageTexturesPerStage).toBe(8);
  });

  it('10/6 is not full hybrid but is lite-capable for both realtime and pt-webgpu', async () => {
    // 10 buffers / 6 textures: below hybrid full (16/8), at/above hybrid lite
    // (10/6), and at/above pt-webgpu lite (8 buf / 4 tex).
    const p = await probeAdapterProfile(
      fakeAdapter(10, 6, { vendor: 'intel', architecture: 'gen12' }),
    );
    expect(p.hybridCapable).toBe(false);
    expect(p.hybridLiteCapable).toBe(true);
    expect(p.ptWebgpuTier).toBe('lite');
    expect(p.recommendedRealtimeTier).toBe('medium');
    expect(p.recommendedHeroBackend).toBe('pt-webgpu-lite');
  });

  it('full pt-webgpu tier flips at the stage-level full floor', async () => {
    const p = await probeAdapterProfile(
      fakeAdapter(
        PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
        { vendor: 'nvidia', architecture: 'ampere' },
      ),
    );
    expect(p.ptWebgpuTier).toBe('full');
    expect(p.recommendedHeroBackend).toBe('pt-webgpu-full');
  });

  it('8/4 → hybrid unavailable, pt-webgpu lite', async () => {
    const p = await probeAdapterProfile(
      fakeAdapter(
        PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
        { vendor: 'apple', architecture: 'm1' },
      ),
    );
    expect(p.hybridCapable).toBe(false);
    expect(p.hybridLiteCapable).toBe(false);
    expect(p.ptWebgpuTier).toBe('lite');
    expect(p.recommendedRealtimeTier).toBe('unavailable');
    expect(p.recommendedHeroBackend).toBe('pt-webgpu-lite');
  });

  it('SwiftShader software adapter → realtime unavailable even at full limits', async () => {
    const p = await probeAdapterProfile(
      fakeAdapter(16, 8, { vendor: 'google', architecture: 'swiftshader' }),
    );
    expect(p.isSoftwareAdapter).toBe(true);
    // Limits would pass full hybrid, but software adapters never run hybrid.
    expect(p.hybridCapable).toBe(true);
    expect(p.recommendedRealtimeTier).toBe('unavailable');
    expect(p.adapterKind).toBe('swiftshader');
  });

  it('below pt-webgpu lite floor → ptWebgpuTier none, hero falls to pt-webgl2', async () => {
    const belowBuf = PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1;
    const belowTex = PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE - 1;
    const p = await probeAdapterProfile(fakeAdapter(belowBuf, belowTex));
    expect(p.ptWebgpuTier).toBe('none');
    // No GPUAdapter info ⇒ unknown adapter kind, not software.
    expect(p.isSoftwareAdapter).toBe(false);
    // hasWebGL2 defaults true in the headless test env ⇒ pt-webgl2 fallback.
    expect(p.recommendedHeroBackend).toBe('pt-webgl2');
  });

  it('no navigator.gpu (no source, no WebGPU env) → hasWebGPU false', async () => {
    // The default vitest env has no navigator.gpu, so the no-source path's
    // probeWebGPU returns unsupported.
    const p = await probeAdapterProfile();
    expect(p.hasWebGPU).toBe(false);
    expect(p.recommendedRealtimeTier).toBe('unavailable');
    expect(p.recommendedHeroBackend).toBe('pt-webgl2'); // headless WebGL2 default
  });
});

describe('threshold-coupling guard (R1 — fails if a magic number is forked)', () => {
  it('hybridCapable flips exactly at HYBRID_WEBGPU_REQUIRED_LIMITS', async () => {
    const justBelow = await probeAdapterProfile(fakeAdapter(FULL_BUF - 1, FULL_TEX));
    const exact = await probeAdapterProfile(fakeAdapter(FULL_BUF, FULL_TEX));
    expect(justBelow.hybridCapable).toBe(false);
    expect(exact.hybridCapable).toBe(true);

    const texJustBelow = await probeAdapterProfile(fakeAdapter(FULL_BUF, FULL_TEX - 1));
    expect(texJustBelow.hybridCapable).toBe(false);
  });

  it('hybridLiteCapable flips exactly at HYBRID_LITE_LIMITS', async () => {
    const justBelow = await probeAdapterProfile(fakeAdapter(LITE_BUF - 1, LITE_TEX));
    const exact = await probeAdapterProfile(fakeAdapter(LITE_BUF, LITE_TEX));
    expect(justBelow.hybridLiteCapable).toBe(false);
    expect(exact.hybridLiteCapable).toBe(true);

    const texJustBelow = await probeAdapterProfile(fakeAdapter(LITE_BUF, LITE_TEX - 1));
    expect(texJustBelow.hybridLiteCapable).toBe(false);
  });

  it('pt-webgpu tier flips exactly at the imported lite consts', async () => {
    const liteBuf = PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
    const liteTex = PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE;
    const atLite = await probeAdapterProfile(fakeAdapter(liteBuf, liteTex));
    const belowLite = await probeAdapterProfile(fakeAdapter(liteBuf - 1, liteTex));
    expect(atLite.ptWebgpuTier).not.toBe('none');
    expect(belowLite.ptWebgpuTier).toBe('none');
  });

  it('HYBRID_LITE_LIMITS is strictly below HYBRID_WEBGPU_REQUIRED_LIMITS on both axes', () => {
    expect(LITE_BUF).toBeLessThan(FULL_BUF);
    expect(LITE_TEX).toBeLessThan(FULL_TEX);
  });
});
