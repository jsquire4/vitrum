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
    // shared shade-prologue's `let emissive`/`metallic`/`normal` became `var` so
    // the full tier can modulate them by emissive / ORM / normal maps. Lite NEVER
    // reads baryVW and never reassigns those (no group-3 texture bindings / sampler
    // / sample fns compose into lite — audited), so the lite RENDER is
    // byte-identical; only the dead baryVW compute + the let→var keywords moved the
    // SHA (length unchanged).
    // Re-pinned 2026-06-06: FrameParams dropped the never-read heroStrategy slot
    // (replaced by _padAuto0, −3 chars). Render-neutral; lite never read it.
    // Re-pinned 2026-06-06 (2): intersectionCore closest-hit live-bound fix
    // (see wgslContract.test.ts) — lite composes the same core. RENDER-CHANGING
    // on purpose; GPU-validated via the G-P0.3 capture matrix.
    // Re-pinned 2026-06-08: SceneHit gained an invalid-or-real TLAS
    // instanceIndex for the full-tier normal-map tangent transform. Lite always
    // writes the invalid sentinel, so this is render-neutral for lite.
    // Re-pinned 2026-06-09: H13/D4 — brdfDirectionalPdf opposite-hemisphere
    // branch now returns 0.0 (delta lobe). Lite composes the same bsdf module.
    // Re-pinned 2026-06-09: H14-E — _padAuto0 renamed to environmentHdriIntensity (f32)
    // in the FrameParams struct; lite composes the same material.wgsl.ts.
    // Render-neutral for lite (lite never reads HDRI fields).
    // Re-pinned 2026-06-09: H52 — clearcoat/sheen/iridescence lobes + MATERIAL_VEC4_STRIDE
    // 23→26. Lite composes the same material.wgsl.ts and bsdf.wgsl.ts.
    // Zero-default invariant: render-neutral for materials without these extension fields.
    // Re-pinned 2026-06-10: Wave B — A3 (true spectral transport: baseColor
    // Jakob-Hanika spectral reflectance + spectralEmissionAtHero for emitters/env;
    // MATERIAL_VEC4_STRIDE 26→27), B9 (Kulla-Conty GGX multiscatter in the shared
    // bsdf module), B10 (physical refraction transmittance = baseColor). Lite
    // composes the same material.wgsl.ts + bsdf.wgsl.ts + shadePrologue. The RGB
    // (spectralEnabled=false) runtime path is byte-identical; B9/B10 are
    // render-changing on rough-metal/glass lite scenes (→ V28).
    // Re-pinned 2026-06-10 for B8 (light-tree orientation cones): the shared
    // light-tree traversal WGSL the lite tier composes grew the node stride 12→16
    // and gained the lt_coneFactor culling term. Default-path RUNTIME byte-
    // identical for unoriented scenes (full-sphere cone ⇒ factor ≡ 1); oriented
    // emitters get tighter SELECTION pdf only (divided out — unbiased).
    // Re-pinned 2026-06-10: caustic point/spot stride fix (H1-class) + shared light-stride constants
    // (POINT_LIGHT_VEC4_STRIDE / SPOT_LIGHT_VEC4_STRIDE added to material.wgsl.ts, composed in both
    // full and lite tiers). The lite-tier render is unchanged — caustic is not in the lite path.
    expect(digest).toBe('d4c47952d7ae0a0831ef88480a42e59a39c7e1047831d5c55281cb3424bf5752');
    expect(PT_WEBGPU_TRACE_LITE_WGSL.length).toBe(108215);
  });
});

describe('pt-webgpu lite WGSL contract', () => {
  it('uses the reduced binding layout (no motion / TLAS / light buffers)', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 27u;'); // A3: 26 → 27
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
