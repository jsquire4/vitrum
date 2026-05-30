import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

// ── Theme-C byte-identity pin ───────────────────────────────────────────────
// See wgslContract.test.ts for the rationale. The lite-tier composed trace must
// stay byte-identical across the intersectionCore / FrameParams-bindings /
// shadePrologue dedup. SHA256 captured from the pre-refactor code.
describe('pt-webgpu lite WGSL byte-identity (Theme-C dedup pin)', () => {
  it('composed lite-tier trace string matches the golden SHA256', () => {
    const digest = createHash('sha256').update(PT_WEBGPU_TRACE_LITE_WGSL).digest('hex');
    expect(digest).toBe('bdbb41f80acaad7b2e49a039914d869ff7a1ce6db296a17a3641c7271c5872c4');
    expect(PT_WEBGPU_TRACE_LITE_WGSL.length).toBe(71460);
  });
});

describe('pt-webgpu lite WGSL contract', () => {
  it('uses the reduced binding layout (no motion / TLAS / light buffers)', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 23u;'); // WS4: 22 → 23
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('motionVectorsTexture');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('varianceMomentsBuffer');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('tlasNodes');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('pointLights');
  });

  it('keeps mesh BVH trace and directional direct lighting', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn traceMeshBvh');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('params.lightDir.w');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn sampleSky');
  });

  it('gates emissive-on-hit to camera + refraction paths (camera-visible emitters)', () => {
    // Mirrors the full-tier gate so lite emitters are also camera-visible without
    // double-counting against the analytic BSDF↔light connection.
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('var prevSampleAllowsAreaMis = false;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('if (!prevSampleAllowsAreaMis) {');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('prevSampleAllowsAreaMis = sampleAllowsAreaMis;');
  });

  // Theme-D — the shared material FUNCS block (activeLayerWeightRgb) is composed
  // into the lite tier too; LUMINANCE_WGSL is composed ahead of it
  // (pathTraceBruteforceLite.wgsl.ts:24), so the dedup applies in lite as well.
  it('activeLayerWeightRgb uses canonical luminance() in the lite tier', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let lum = max(luminance(layerRgb), 0.0);');
    const inlineDots = (
      PT_WEBGPU_TRACE_LITE_WGSL.match(/dot\([^)]*vec3f\(0\.2126, 0\.7152, 0\.0722\)\)/g) ?? []
    ).length;
    expect(inlineDots).toBe(0);
  });
});
