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
    // Updated for P2: the shared intersectionCore gained SceneHit.baryVW, and the
    // shared shade-prologue's `let emissive`/`let metallic` became `var` (so the
    // full tier can modulate them by emissive/ORM maps). Lite NEVER reads baryVW
    // and never reassigns emissive/metallic (no group-3 texture bindings / sampler
    // / sample fns compose into lite — audited), so the lite RENDER is
    // byte-identical; only the dead baryVW compute + the let→var keywords moved the
    // SHA (length unchanged).
    expect(digest).toBe('d7b62216eb5764d888cdba0a1b20071db553b59e082f128eb8d2a92fc0c9dbb7');
    expect(PT_WEBGPU_TRACE_LITE_WGSL.length).toBe(72242);
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
