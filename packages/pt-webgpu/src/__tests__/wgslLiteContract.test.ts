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
    // Re-pinned 2026-06-10: B12 — lite-tier light/env texture packing. New texture bindings (12-14):
    // liteEnvTex (RGBA32F env radiance+pdf), liteEnvCdfTex (RGBA32F env CDF), liteLightTex
    // (RGBA32F point/spot/rect-area packed data). NEE loops added in kernelLite.wgsl.ts for
    // point/spot/rect-area + env importance sampling in connectLite.wgsl.ts.
    // RENDER-CHANGING for lite scenes with those lights; A/B pending V28-B.
    // Re-pinned R7b 2026-06-10: anisotropic GGX (Item 7) + envMapIntensity escape-half
    // (Item 8) + lightMap camera-only gate (Item 9). bsdf.wgsl.ts gained aniso GGX
    // functions (sampleGgxVndfAnisTangent / evalBrdfSpecAnisotropic / brdfAnisotropicSpecPdf
    // + extended evaluateBrdfFull / brdfDirectionalPdfFull / sampleNextBounceDirection
    // signatures). kernelLite.wgsl.ts passes 0.0, 0.0 for the aniso params (lite tier
    // has no aniso texture bindings — always isotropic). shadePrologue.wgsl.ts
    // gates lightMap to bounce==0 (litePrologue unchanged). RENDER-CHANGING for
    // full-tier aniso materials; A/B pending V28-B.
    // Re-pinned 2026-06-10: native analytic disc emitters in kernelLite.wgsl.ts —
    // rect/disc loop reads shape tag from liteLightTex texel .w, applies concentric-disc
    // map for disc records (same design as full-tier kernel.wgsl.ts). 32-triangle fan
    // removed. RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
    // Re-pinned 2026-06-10: D9.1 — computeAnisotropicAxes extracted in bsdf.wgsl.ts;
    // shared by both tiers (SEMANTICALLY EQUIVALENT dedup).
    // Re-pinned 2026-06-10: D9.11 — concentricDiscSample extracted to kernelCore.wgsl.ts;
    // kernelLite.wgsl.ts uses the shared helper. SEMANTICALLY EQUIVALENT; same Shirley-Chiu mapping.
    // Re-pinned 2026-06-10: D9.13 — rotateYNeg/rotateYPos moved to connectCore.wgsl.ts;
    // connectLite.wgsl.ts liteRotateY* replaced with canonical names. SEMANTICALLY EQUIVALENT.
    // Re-pinned 2026-06-11 (SHADOW-01): primitive castShadow any-hit gate in the
    // shared intersectionCore (traceMeshBvh !closest skip via triShadowCastDisabled)
    // + emitter castShadowDisabled gates in the lite NEE loops (point extra.z /
    // spot spExtra.z / rect texel-0 .w). Lite directional NEE decodes the
    // sign-encoded cameraPos.w mirror for the first directional flag.
    // Default (flag-less) scenes are behaviorally identical (all lanes pack 0.0).
    // Re-pinned 2026-06-15 (PTWG-LITE-01): lite rect/disc analytic records now
    // use paired MIS. kernelLite applies the light-sampled power heuristic and
    // connectLite intersects BSDF-sampled directions against the same liteLightTex
    // rect/disc records. RENDER-CHANGING for lite rect/disc scenes; CPU oracle
    // pins the paired-share mean.
    // Re-pinned 2026-06-11 (PTWG-MAT-01 partial): lite direct/env paths and
    // BSDF-env connection route scalar clearcoat/sheen/iridescence fields
    // through evaluateBrdfFull/brdfDirectionalPdfFull, with anisotropy forced
    // to zero because lite has no anisotropy bindings.
    // Re-pinned 2026-06-12: lite caustic/environment call signatures were
    // re-synchronized with the lite kernel material-extension call sites after
    // shader-gate caught a stale stub/caller mismatch. Caustic lite still returns
    // zero by design; environment reconnection now receives the same scalar
    // extension fields it already evaluates.
    // Re-pinned 2026-06-12: environment:'none' no longer falls through to the
    // analytic sampleSky gradient. Missing/invalid env maps now return black
    // radiance + zero env pdf; procedural-sky stays lit via the CPU-baked HDRI.
    // Re-pinned 2026-06-12: shared prologue changed `transmission` from let to
    // var so the full tier can modulate it by transmissionMap; lite still has
    // no material texture bindings and remains scalar-only.
    // Re-pinned 2026-06-12: lite MNEE/env reconnection call sites accept the
    // decoded KHR_materials_specular scalar factors to keep the scalar material
    // contract interface-aligned with full-tier shaders.
    // Re-pinned 2026-06-13: shared material.wgsl.ts appended clearcoatNormalMap
    // descriptor helpers for the full tier. Lite composes the shared text but
    // still has no material texture bindings; this is render-neutral for lite.
    // Re-pinned 2026-06-15: shared scalar material payload gained KHR volume
    // thickness and lite Beer-Lambert fallback clamps to it when authored.
    // Re-pinned 2026-06-15 (SHADOW-01 lite directional): lite directional NEE
    // decodes the signed cameraPos.w mirror so first-directional castShadow:false
    // skips the visibility ray without adding a storage-buffer binding.
    // Re-pinned 2026-06-18: H55 oracle wave fixed concentricDiscSample's
    // Shirley-Chiu signed-denominator bug. Disc-area sampling was previously
    // mirrored into the wrong quadrant for two square quadrants; render-changing
    // for lite-tier disc-area lights, now pinned by oracle.concentricDiscSample.
    expect(digest).toBe('13357933d405efd9227ea2788c70366a66337f52bc91f7b071f652a40c49a81c');
    expect(PT_WEBGPU_TRACE_LITE_WGSL.length).toBe(154047);
  });
});

describe('pt-webgpu lite WGSL contract', () => {
  it('uses the reduced binding layout (no motion / TLAS / light buffers)', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 29u;'); // VOL-THICKNESS: 28 → 29
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('motionVectorsTexture');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('varianceMomentsBuffer');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('tlasNodes');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('pointLights');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('mat.isUnlit');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('specularIntensity: f32,');
  });

  it('routes lite scalar extension lobes through full BRDF helpers with zero anisotropy', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let brdf = evaluateBrdfFull(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let brdfPdf = brdfDirectionalPdfFullSampled(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('mat.clearcoatRoughness,');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('0.0, 0.0);');
  });

  it('pairs lite rect/disc NEE with a BSDF-to-area-light connection', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn intersectLiteRectAreaLightRay(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn liteRectLightBase() -> u32');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let misWeight = powerHeuristic(lightPdf, brdfPdf);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let misWeight = powerHeuristic(bsdfPdf, bestLightPdf);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('textureLoad(liteLightTex, vec2i(i32(rb + 3u), 0), 0).rgb');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toMatch(
      /_ = throughputAtVertex;\s+return vec3f\(0\.0\);\s+}\s+fn bsdfEnvironmentConnectionContribution/,
    );
  });

  it('passes scalar clearcoat and sheen into the lite main bounce sampler', () => {
    const callIndex = PT_WEBGPU_TRACE_LITE_WGSL.lastIndexOf('let bs = sampleNextBounceDirection(');
    expect(callIndex).toBeGreaterThan(-1);
    const call = PT_WEBGPU_TRACE_LITE_WGSL.slice(callIndex, callIndex + 700);
    expect(call).toContain('mat.clearcoat,');
    expect(call).toContain('mat.clearcoatRoughness,');
    expect(call).toContain('mat.sheen,');
    expect(call).toContain('mat.sheenRoughness,');
    expect(call).toContain('mat.sheenColor,');
    expect(call).toContain('0.0, // anisotropy');
  });

  it('keeps mesh BVH trace and directional direct lighting', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn traceMeshBvh');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('params.lightDir.w');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let dirShadowDisabled = params.cameraPos.w < 0.0;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY))');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn sampleSky');
  });

  it('treats absent lite-tier environments as black, not procedural sky', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('return vec4f(0.0);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('if (lk.a <= 0.0) { return 0.0; }');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('return vec4f(sampleSky(dir), 0.0);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('if (lk.a <= 0.0) { return sampleSky(dir); }');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('Procedural-sky fallback: uniform-sphere sample + sky eval.');
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
