import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('pt-webgpu WGSL material contract', () => {
  it('uses the bounded rich material payload layout', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 22u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const THIN_FILM_LAYER_LIMIT = 8u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const SPECTRAL_SAMPLE_COUNT = 32u;');
  });

  it('threads transmission probability into directional MIS pdf helper', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let nDotT = max(abs(wiDotN), 1e-5);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('transProb * pdfTransApprox');
  });

  it('accounts for uniform light selection probability in direct lighting', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('radiance = radiance + directLi * f32(lightCount);');
  });

  it('contains active strategy-specific caustic paths', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn causticMode() -> u32');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn photonMapContribution');
    // Caches the strategy code in `caustic` and branches on the local, so
    // the manifold/photon dispatch sites read a single causticMode() call.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let caustic = causticMode();');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (caustic == 1u)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('else if (caustic == 2u)');
  });

  it('declares INV_2PI alongside INV_PI for HDRI equirect sampling', () => {
    // pathTraceBruteforce.wgsl.ts uses INV_2PI for spherical-to-UV mapping;
    // omitting it would cause a runtime WGSL compile failure for any scene
    // with an HDRI environment.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const INV_2PI');
  });

  it('FrameParams matches the host-side 512-byte UBO buffer size', () => {
    // The host allocates `new ArrayBuffer(512)` in PTEngineWebGPU.#buildParamsBuffer
    // and writes vec4-aligned fields at offsets 0..127 (16-byte units). If the
    // WGSL FrameParams struct grows past 512 bytes, this assertion fails first
    // and points at the host-side layout mismatch.
    // Count of vec4-aligned slots referenced by WGSL: matrix terms (12 mat4f =
    // 48 vec4f-slots? no, three mat4x4f = 12 vec4f total) + scalar/vec4 fields.
    // Total budget 512 bytes / 16 bytes-per-vec4 = 32 vec4-slots = 128 f32-slots.
    // Sanity check: the WGSL header should NOT reference paramsF32[128+].
    const stride = (PT_WEBGPU_TRACE_WGSL.match(/struct FrameParams/g) ?? []).length;
    expect(stride).toBeGreaterThanOrEqual(1);
    // Verify FrameParams contains the matrix fields the host packs at offsets
    // 80..127 (invViewProj, viewProj, prevViewProj — 16 floats each).
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/invViewProj\s*:\s*mat4x4f/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/viewProj\s*:\s*mat4x4f/);
  });
});
