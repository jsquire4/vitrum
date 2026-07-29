import { afterEach, describe, expect, it, vi } from 'vitest';
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
  sampled: number =
    HYBRID_WEBGPU_REQUIRED_LIMITS['maxSampledTexturesPerShaderStage']!,
): GPUAdapter {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: buf,
      maxStorageTexturesPerShaderStage: tex,
      maxSampledTexturesPerShaderStage: sampled,
    },
    ...(info ? { info } : {}),
  } as unknown as GPUAdapter;
}

const FULL_BUF = HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!;
const FULL_TEX = HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!;
const LITE_BUF = HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']!;
const LITE_TEX = HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!;

afterEach(() => {
  vi.unstubAllGlobals();
});

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

  it('10/6 is below the real hybrid layout floor but remains pt-webgpu lite-capable', async () => {
    // The current hybrid lite path compiles the same explicit layouts as full.
    const p = await probeAdapterProfile(
      fakeAdapter(10, 6, { vendor: 'intel', architecture: 'gen12' }),
    );
    expect(p.hybridCapable).toBe(false);
    expect(p.hybridLiteCapable).toBe(false);
    expect(p.ptWebgpuTier).toBe('lite');
    expect(p.recommendedRealtimeTier).toBe('unavailable');
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

  it('below pt-webgpu lite floor fails closed when the realm has no WebGL2 surface', async () => {
    const belowBuf = PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1;
    const belowTex = PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE - 1;
    const p = await probeAdapterProfile(fakeAdapter(belowBuf, belowTex));
    expect(p.ptWebgpuTier).toBe('none');
    // No GPUAdapter info ⇒ unknown adapter kind, not software.
    expect(p.isSoftwareAdapter).toBe(false);
    expect(p.hasWebGL2).toBe(false);
    expect(p.recommendedHeroBackend).toBe('none');
  });

  it('no navigator.gpu (no source, no WebGPU env) → hasWebGPU false', async () => {
    // The default vitest env has no navigator.gpu, so the no-source path's
    // probeWebGPU returns unsupported.
    const p = await probeAdapterProfile();
    expect(p.hasWebGPU).toBe(false);
    expect(p.recommendedRealtimeTier).toBe('unavailable');
    expect(p.hasWebGL2).toBe(false);
    expect(p.recommendedHeroBackend).toBe('none');
  });

  it('probes worker WebGL2 through OffscreenCanvas instead of assuming support', async () => {
    vi.stubGlobal('document', undefined);
    vi.stubGlobal('OffscreenCanvas', class {
      constructor(_width: number, _height: number) {}
      getContext(kind: string): object | null {
        return kind === 'webgl2' ? {} : null;
      }
    });

    const p = await probeAdapterProfile(fakeAdapter(
      PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1,
      PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE - 1,
    ));

    expect(p.ptWebgpuTier).toBe('none');
    expect(p.hasWebGL2).toBe(true);
    expect(p.recommendedHeroBackend).toBe('pt-webgl2');
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

    const sampledJustBelow = await probeAdapterProfile(fakeAdapter(
      FULL_BUF,
      FULL_TEX,
      undefined,
      HYBRID_WEBGPU_REQUIRED_LIMITS['maxSampledTexturesPerShaderStage']! - 1,
    ));
    expect(sampledJustBelow.hybridCapable).toBe(false);
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

  it('HYBRID_LITE_LIMITS equals FULL until the lite path has distinct layouts', () => {
    expect(HYBRID_LITE_LIMITS).toEqual(HYBRID_WEBGPU_REQUIRED_LIMITS);
  });
});
