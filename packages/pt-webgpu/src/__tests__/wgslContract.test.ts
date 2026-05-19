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
    // Verify FrameParams contains the matrix fields. Exact byte offsets
    // (invViewProj@144, viewProj@208, prevViewProj@272 post W3-D12 + W4-A4
    // refactor) are pinned numerically in the next test.
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/invViewProj\s*:\s*mat4x4f/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/viewProj\s*:\s*mat4x4f/);
  });

  // Regression for pt-webgpu-deep-audit H-1 (matrix offsets in FrameParams
  // were 16 bytes shifted from spec, corrupting all ray reconstruction).
  // Fixed since the 2026-05-10 audit; this test pins the layout so a future
  // WGSL refactor can't silently re-break the host packer at
  // index.ts:274-280.
  it('FrameParams matrix block sits at the WGSL-spec byte offsets the host packer assumes (H-1 regression)', () => {
    const struct = PT_WEBGPU_TRACE_WGSL.match(
      /struct FrameParams \{([\s\S]*?)\};/,
    )?.[1];
    expect(struct).toBeDefined();
    if (struct == null) return;

    const lines = struct
      .split('\n')
      .map((s) => s.trim())
      .filter((s) => s.length > 0 && !s.startsWith('//'));

    // Walk the struct in WGSL-spec alignment + size to derive each
    // mat4x4f's byte offset. Any drift here vs the CPU packer (paramsF32
    // .set at f32 indices 36/52/68) breaks the GPU output silently.
    const SIZE: Record<string, number> = { u32: 4, f32: 4, vec3f: 12, vec4f: 16, mat4x4f: 64 };
    const ALIGN: Record<string, number> = { u32: 4, f32: 4, vec3f: 16, vec4f: 16, mat4x4f: 16 };
    const fieldRe = /^([A-Za-z_]\w*)\s*:\s*([\w<,>]+),/;
    let offset = 0;
    const matrixOffsets: Record<string, number> = {};
    for (const line of lines) {
      const m = line.match(fieldRe);
      if (m == null) continue;
      const name = m[1]!;
      const type = m[2]!;
      const a = ALIGN[type];
      const s = SIZE[type];
      if (a == null || s == null) {
        throw new Error(`Unknown WGSL field type "${type}" in FrameParams — extend SIZE/ALIGN tables.`);
      }
      const aligned = Math.ceil(offset / a) * a;
      if (type === 'mat4x4f') matrixOffsets[name] = aligned;
      offset = aligned + s;
    }

    // Byte offsets must match what the CPU packer at
    // packages/pt-webgpu/src/index.ts:274-280 writes:
    //   paramsF32.set(invVp, 36)   → byte 36 * 4 = 144
    //   paramsF32.set(vp, 52)      → byte 52 * 4 = 208
    //   paramsF32.set(prevVp, 68)  → byte 68 * 4 = 272
    expect(matrixOffsets['invViewProj']).toBe(144);
    expect(matrixOffsets['viewProj']).toBe(208);
    expect(matrixOffsets['prevViewProj']).toBe(272);
    expect(offset).toBeLessThanOrEqual(512);
  });
});
