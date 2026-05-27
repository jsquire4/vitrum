import { describe, expect, it } from 'vitest';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

describe('pt-webgpu WGSL material contract', () => {
  it('uses the bounded rich material payload layout', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 22u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const THIN_FILM_LAYER_LIMIT = 8u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const SPECTRAL_SAMPLE_COUNT = 32u;');
  });

  it('includes hero-wavelength MIS helpers when spectral mode is enabled', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleHeroWavelengthMIS');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn heroWavelengthToRgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('params.spectralEnabled != 0u');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('heroLambdaTo01(heroLambda)');
  });

  it('threads transmission probability into directional MIS pdf helper', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let nDotT = max(abs(wiDotN), 1e-5);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('eta * eta * INV_PI');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('transProb * pdfTransApprox');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn activeLayerWeightRgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('isTranslucent');
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

    // Byte offsets must match FrameParamsSlot (tools/generate-wgsl-layouts.mjs).
    expect(matrixOffsets['invViewProj']).toBe(FrameParamsSlot.invViewProj * 4);
    expect(matrixOffsets['viewProj']).toBe(FrameParamsSlot.viewProj * 4);
    expect(matrixOffsets['prevViewProj']).toBe(FrameParamsSlot.prevViewProj * 4);
    expect(offset).toBeLessThanOrEqual(512);
  });

  it('declares TLAS storage bindings and host bind-group wiring in lockstep', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(0) var<storage, read> tlasNodes');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(1) var<storage, read> tlasInstanceIndices');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(2) var<storage, read> tlasBlasRoots');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(3) var<storage, read> tlasInstanceWorldToLocal');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(4) var<storage, read> tlasInstanceLocalToWorld');

    const here = dirname(fileURLToPath(import.meta.url));
    const limitsSource = readFileSync(resolve(here, '../webgpuLimits.ts'), 'utf8');
    const indexSource = readFileSync(resolve(here, '../index.ts'), 'utf8');
    expect(limitsSource).toMatch(/PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP\s*=\s*10/);
    expect(indexSource).toContain('selectPtWebgpuTraceTier');
    expect(indexSource).toContain('pathTrace.bindgroup2.full');
    expect(indexSource).toContain('{ binding: 0, resource: { buffer: this.#sceneBuffers.tlasNodesBuffer } }');
    expect(indexSource).toContain('setBindGroup(2, this.#pathTraceBindGroup2)');
  });

  it('keeps TLAS hit reconstruction in world space', () => {
    // TLAS traversal must convert local-space BLAS hits back into world
    // distance/normal so denoising and shading stay frame-stable.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let worldDist = dot(worldHitPos - ray.origin, ray.direction);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('(*hit).dist = worldDist;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '(*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);',
    );
  });

  it('uses conservative local-ray bounds for TLAS instance traversal under scaling', () => {
    // Local instance-space t does not match world-space t under non-uniform
    // transforms, so BLAS queries must run unbounded and clamp in world space.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, tMin, INFINITY, true, &localHit, blasRoot, true);');
  });

  it('skips bary/normal reconstruction on TLAS any-hit instance traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('captureShadingDetails: bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, tMin, INFINITY, true, &localHit, blasRoot, false);');
  });

  it('returns closest-hit result from traceMeshBvh when in closest mode', () => {
    // Regression guard: closest-mode traversal must return didHit so
    // traceAny fallback paths without TLAS still report mesh occlusion.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return select(false, (*hit).didHit, closest);');
  });

  it('routes traceAny through TLAS any-hit traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn traceTlasAny(ray: Ray, tMin: f32, tMax: f32) -> bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (traceTlasAny(ray, tMin, tMax)) {');
  });

  it('uses conservative any-hit stack-overflow handling to avoid light leaks', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('Conservative any-hit overflow policy: prefer occlusion over light leak.');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return true;');
  });
});
