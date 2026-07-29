import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { activeLayerWeightRgbOracle } from './spectralScalarOracle.js';

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
    // (RGBA32F directional/point/spot/rect-area packed data). NEE loops added in kernelLite.wgsl.ts for
    // point/spot/rect-area + env importance sampling in connectLite.wgsl.ts.
    // Re-pinned 2026-06-18: liteLightTex now also prepends directional records,
    // so lite loops N-directional emitters instead of the old first-directional
    // UBO mirror.
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
    // + emitter castShadowDisabled gates in the lite NEE loops (directional
    // angularDiameter sign bit / point extra.z / spot spExtra.z / rect texel-0 .w).
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
    // Re-pinned 2026-06-12: environment:'none' no longer falls through to an
    // analytic sky gradient. Missing/invalid env maps return black radiance +
    // zero env pdf; procedural skies use their CPU-baked HDRI exclusively.
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
    // decoded the signed cameraPos.w mirror so first-directional castShadow:false
    // skipped the visibility ray without adding a storage-buffer binding.
    // Re-pinned 2026-06-18: lite directionals now prepend the same 2-vec4 records
    // used by the full tier into liteLightTex and loop params.directionalLightCount,
    // closing the former first-directional truncation gap.
    // Re-pinned 2026-06-18: H55 oracle wave fixed concentricDiscSample's
    // Shirley-Chiu signed-denominator bug. Disc-area sampling was previously
    // mirrored into the wrong quadrant for two square quadrants; render-changing
    // for lite-tier disc-area lights, now pinned by oracle.concentricDiscSample.
    // Re-pinned 2026-06-21: lite sampled indirect Fresnel now uses
    // iridescenceModifiedF0 before the shared bounce sampler. RENDER-CHANGING for
    // scalar iridescent materials on lite sampled indirect paths.
    // Re-pinned 2026-06-21: anisotropic GGX sampled/evaluated paths now use a
    // conservative projected-roughness Kulla-Conty multiscatter approximation.
    // RENDER-CHANGING for rough anisotropic materials.
    // Re-pinned 2026-06-21: ptRngFrameKey preserves PCG's old frameSeed^frameIndex
    // expression while letting Sobol use a monotonic sample key in the opt-in module.
    // Re-pinned 2026-07-23: distant-direct (environment/directional) proposal
    // selection is now separated from finite-emitter light-tree selection so
    // both proposal families retain their own normalized PDFs and MIS ownership.
    // Re-pinned 2026-07-23: lite now honors MaterialSpec.doubleSided in both
    // closest-hit and visibility traversal while preserving transmissive exits.
    // Re-pinned 2026-07-24: exact-zero delta classification and the
    // transmission-aware finite-BSDF helper are now shared with the full tier.
    // Re-pinned 2026-07-27: transmissive eval/PDF/sampling now share the same
    // coloured KHR specular/iridescence Fresnel model as the full tier.
    // Re-pinned 2026-07-27: removed unread CMF/light-direction UBO payload and
    // made cameraPos a semantic vec3f beside the live HDRI-intensity scalar.
    // Re-pinned 2026-07-27: finite-BSDF sampling now shares one finalization
    // path instead of retaining obsolete per-caller normalization helpers.
    // Re-pinned 2026-07-28: the shared BSDF now layers coherent stacks into
    // finite rough/material lobes and applies clearcoat/sheen attenuation.
    // Re-pinned 2026-07-28: invalid coherent-TMM samples are absorbed instead
    // of silently being replaced by a perfect mirror.
    // Re-pinned 2026-07-28: affine disc emitters use π·|u×v| for both NEE and
    // BSDF-connection area conversion, and BSDF connections solve the full
    // Gram system for sheared light axes.
    // Re-pinned 2026-07-28: removed four uncalled legacy BSDF wrappers.
    // Re-pinned 2026-07-28: corrupt thin-film descriptor/LUT ranges fail dark.
    // Re-pinned 2026-07-28: A1 keeps opaque roughness-zero GGX finite. A5 is
    // full-only because the lite contract rejects alpha materials before upload.
    // Re-pinned 2026-07-28: shared KHR specular/IOR semantics, map-only
    // environments, unavailable lite variance, and the 512-D Sobol table.
    // Re-pinned 2026-07-29: U11 removes continuous-specular proposal-local
    // throughput/PDF calculations overwritten by the finite finalizer.
    expect(digest).toBe('6acef3624677077dac89f0eab5d82bc551498218f167e201529aa499e66393a6');
    expect(PT_WEBGPU_TRACE_LITE_WGSL.length).toBe(213935);
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

  it('does not expose raw sample luminance as a variance auxiliary', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      'textureStore(varianceTexture, vec2i(gid.xy), vec4f(0.0));',
    );
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain(
      'vec4f(sampleLum, sampleLum, sampleLum, 1.0)',
    );
    const engineSource = readFileSync(new URL('../index.ts', import.meta.url), 'utf8');
    expect(engineSource).toContain(
      "this.#traceTier === 'full' && this.#gpu.varianceTexture != null",
    );
  });

  it('routes lite scalar extension lobes through transmission-aware full BRDF helpers with zero anisotropy', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
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

  it('keeps finite-emitter measures exact and MIS-accountable across rough transmission', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('max(PI * rradL * rradL, 1e-6)');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('max(4.0 * length(cross(ru, rv)), 1e-6)');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('max(cosLight * area, 1e-6)');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let nDotL = abs(dot(normal, wi));');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      'let offsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);',
    );
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      'envDir = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));',
    );
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('envPdf = 0.25 * INV_PI;');
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
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u)');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let dBase = liteDirBase + di * 2u;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let dDirAD = textureLoad(liteLightTex, vec2i(i32(dBase), 0), 0);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let dIrrMean = textureLoad(liteLightTex, vec2i(i32(dBase + 1u), 0), 0);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let dirShadowDisabled = angDiamRaw < 0.0;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY))');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('directLi = directLi + throughput * brdf * nDotL * dirIrrOut;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('fn sampleSky');
  });

  it('treats absent lite-tier environments as black, not procedural sky', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('return vec4f(0.0);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('if (lk.a <= 0.0) { return 0.0; }');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('return vec4f(sampleSky(dir), 0.0);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('if (lk.a <= 0.0) { return sampleSky(dir); }');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('Procedural-sky fallback: uniform-sphere sample + sky eval.');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('environmentSun.w');
  });

  it('gates emissive-on-hit to paths without an MIS-accountable prior event', () => {
    // Mirrors the full-tier gate so lite emitters are also camera-visible without
    // double-counting against the analytic BSDF↔light connection. Rough
    // transmission has a finite directional density and remains MIS-accountable;
    // only camera/delta events bypass the connection.
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('var prevSampleAllowsAreaMis = false;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('if (!prevSampleAllowsAreaMis) {');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('prevSampleAllowsAreaMis = sampleAllowsAreaMis;');
  });

  // Theme-D — the shared material FUNCS block (activeLayerWeightRgb) is composed

  it('suppresses the raw environment miss after an MIS-accounted BSDF connection', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toMatch(
      /if \(!hit\.didHit\) \{[\s\S]*?if \(!prevSampleAllowsAreaMis\) \{[\s\S]*?radiance = radiance \+ throughput \* envContribution;[\s\S]*?\}\s*break;/,
    );
    const rawMissAdds = (
      PT_WEBGPU_TRACE_LITE_WGSL.match(
        /radiance = radiance \+ throughput \* envContribution;/g,
      ) ?? []
    ).length;
    expect(rawMissAdds).toBe(1);
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      'radiance = radiance + bsdfEnvironmentConnectionContribution(',
    );
  });
  // into the lite tier too; LUMINANCE_WGSL is composed ahead of it
  // (pathTraceBruteforceLite.wgsl.ts:24), so the dedup applies in lite as well.
  it('activeLayerWeightRgb stays scalar at the hero wavelength in the lite tier', () => {
    const blue = activeLayerWeightRgbOracle([0.1, 0.2, 1], 440, true);
    const red = activeLayerWeightRgbOracle([0.1, 0.2, 1], 650, true);
    expect(blue[0]).toBe(blue[1]);
    expect(blue[1]).toBe(blue[2]);
    expect(blue[0]).toBeGreaterThan(red[0]);
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('spectralRgbFactorAtHero(layerRgb, heroLambda)');
    const inlineDots = (
      PT_WEBGPU_TRACE_LITE_WGSL.match(/dot\([^)]*vec3f\(0\.2126, 0\.7152, 0\.0722\)\)/g) ?? []
    ).length;
    expect(inlineDots).toBe(0);
  });
});
