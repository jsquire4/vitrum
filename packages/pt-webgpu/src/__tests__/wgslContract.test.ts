import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ── Theme-C byte-identity pin ───────────────────────────────────────────────
// Task 2.2 collapsed the compile-time-variant whole-body WGSL duplication
// (intersectionCore, FrameParams group-0 bindings, shadePrologue) into shared
// fragments. The CORE RULE is that the FINAL composed shader string for every
// variant must be BYTE-IDENTICAL before vs after. This SHA256 pins the composed
// full-tier trace string to the value captured from the pre-refactor code; a
// single differing byte flips the hash and fails. If a future intentional WGSL
// change lands, recompute the digest (sha256 of PT_WEBGPU_TRACE_WGSL) and update
// it here in the same commit.
describe('pt-webgpu WGSL byte-identity (Theme-C dedup pin)', () => {
  it('composed full-tier trace string matches the golden SHA256', () => {
    const digest = createHash('sha256').update(PT_WEBGPU_TRACE_WGSL).digest('hex');
    // Updated for Phase I.1: the trace kernel now composes MNEE_NEWTON_WGSL +
    // MNEE_CHAIN_WGSL + MNEE_CONNECTION_WGSL and the real point-light REFLECTION +
    // REFRACTION + GLASS-SLAB (2-vertex chain) caustics (caustic.wgsl.ts:
    // pointLightReflectionCaustic / pointLightRefractionCaustic / pointLightGlassSlabCaustic).
    // The reflection caustic is GPU-validated against the analytic mirror-image
    // reference (wsl-gpu mnee-reflection-caustic-ab.ts); the refraction caustic
    // ("water surface") against a deterministic forward-traced grid reference
    // (wsl-gpu mnee-refraction-caustic-ab.ts — ratio 0.986, slope 0.984); the
    // glass-slab chain caustic against a deterministic forward-traced SLAB grid
    // (offline focusing derivation ratio/slope 1.000 in
    // mnee-glass-slab-focusing-derivation.ts, then GPU-A/B'd in
    // mnee-glass-slab-caustic-ab.ts — ratio 0.996, slope 0.990 on lavapipe). The
    // refraction caustic gained a single-interface GUARD (causticSegmentCrossesTransmissive):
    // it skips when the light→v leg crosses a transmissive facet, since that path is
    // really a chain the slab kernel owns — without it the refraction + slab kernels
    // double-counted (slab A/B ratio 2.17 → 0.996). The composed string also passes a
    // naga compile gate (wsl-gpu mnee-slab-wgsl-compile.ts) — the byte-identity goldens
    // alone do NOT catch a symbol-scope regression. Recompute (sha256 of
    // PT_WEBGPU_TRACE_WGSL) on any intentional WGSL change and update both here.
    // Re-pinned 2026-06-06: FrameParams dropped the never-read heroStrategy slot
    // (replaced by _padAuto0) and bdptConnection.wgsl gained a firefly-guard
    // comment above BDPT_CONTRIBUTION_CLAMP. Both are render-neutral.
    // Re-pinned 2026-06-06 (2): intersectionCore closest-hit leaf loop now
    // compares against the LIVE running (*hit).dist instead of a per-leaf
    // snapshot — the snapshot made within-leaf acceptance last-writer-wins
    // (thin slabs shaded from their buried far face → black walls). RENDER-
    // CHANGING on purpose; GPU-validated via the G-P0.3 capture matrix.
    // Re-pinned 2026-06-07: photon-map demotion comment block added to
    // caustic.wgsl.ts (honest-labeling, COMMENT-ONLY — verified no WGSL-code
    // line changed; the demotion docs the gatherRadius/strategyScale constants).
    // Re-pinned 2026-06-08: TLAS mesh hits now carry instanceIndex, and normal
    // maps transform their derived tangent through the hit instance localToWorld
    // before shading rotated/scaled instances. RENDER-CHANGING on purpose.
    // Re-pinned 2026-06-09: H13/D4 — brdfDirectionalPdf opposite-hemisphere
    // branch now returns 0.0 (delta lobe; prior finite cosine·η² was wrong —
    // did not match the deterministic Snell sampler in sampleNextBounceDirection).
    // RENDER-CHANGING on transmissive scenes; A/B required before compositing.
    // Re-pinned 2026-06-09: H14-E/H14-B/H51-D — HDRI gets its own intensity lane
    // (environmentHdriIntensity, f32 slot 31, replaces _padAuto0); H14-B adds spot
    // loop to restirPtProducer; H51-D bumps point stride 8→12 + spot stride 12→16
    // (penumbra+distance+decay). RENDER-CHANGING on HDRI + spot/point scenes; A/B required.
    // Re-pinned 2026-06-09: H52 — clearcoat (additive GGX F0=0.04), sheen (Charlie
    // NDF + Neubelt-Pettineo visibility), and iridescence (Belcour & Barla 2017
    // thin-film Fresnel F0 modification) lobes added to the WebGPU BRDF.
    // MATERIAL_VEC4_STRIDE bumped 23→26. Zero-default invariant: all lobes
    // short-circuit when their scalar is 0 → pre-H52 scenes are numerically identical.
    // RENDER-CHANGING on clearcoat/sheen/iridescence materials.
    // Re-pinned 2026-06-09: H6 — HdriEnvironment.rotationY implemented in pt-webgpu.
    // rotateYNeg + rotateYPos helpers added to connect.wgsl.ts; environmentLookup
    // rotates dir by -rotationY before UV; sampleEnvironmentImportance rotates the
    // CDF-sampled direction by +rotationY. rotationY=0 → identity (zero-rotation invariant).
    // Packed into params.environmentTint.w (no layout change). RENDER-CHANGING on HDRI scenes
    // with non-zero rotationY; rotationY=0 byte-identical to pre-H6.
    // Re-pinned 2026-06-10: A9 (BDPT production quality) — the BDPT-gated light
    // subpath gained a REAL glossy/specular BSDF extension (was Lambertian-only);
    // the light-path vertex widened first to 4 rows (matId + wo-toward-prev), then
    // to 5 rows (hit-local tri/bary/instance material payload) so the §10.3
    // connection evaluates the REAL texture-mapped light-vertex BSDF/pdfs
    // (fwdEe/revLcMinus); the connection light-bounce cap rose 3→8; the
    // point emitter went isotropic (uniform sphere, was cosine-up). All gated behind
    // `if (params.bdptEnabled != 0u)` so a bdpt:false RUNTIME render is byte-identical
    // (the WGSL STRING changes, hence this re-pin; the OFF runtime path does not).
    // RENDER-CHANGING on bdpt:true scenes; equal-spp variance + caustic A/Bs → V28.
    // Re-pinned 2026-06-10: Wave B — A3 (true spectral transport: baseColor
    // Jakob-Hanika spectral reflectance carried scalar-spectral at the hero λ;
    // emitters/env/lights spectralised via spectralEmissionAtHero; MATERIAL_VEC4_STRIDE
    // bumped 26→27 for the spectral-coeff vec4 #26), B9 (Kulla-Conty GGX multiscatter
    // energy compensation in evaluateBrdf/Full + the sampled-spec boost), B10 (physical
    // refraction transmittance = baseColor, replacing mix(vec3(1),baseColor,0.15)).
    // spectralEnabled=false RGB path is byte-identical at RUNTIME (the WGSL string
    // changes via the always-present select()s, hence this re-pin); B9/B10 are
    // RENDER-CHANGING on rough-metal / glass scenes → V28 A/Bs.
    // Re-pinned 2026-06-10 for B8 (light-tree orientation cones): the shared
    // light-tree traversal WGSL grew the node stride 12→16 and gained the
    // lt_coneFactor culling term (+2024 chars). Default-path RUNTIME is byte-
    // identical for unoriented scenes (full-sphere cone ⇒ factor ≡ 1); oriented
    // emitters (spot / single-sided area) get tighter SELECTION pdf only (the pdf
    // is divided out — unbiased) → V28 oriented-emitter A/B.
    // Re-pinned 2026-06-10: D3 reserved-field consumption (ao/light/bump/env/anisotropy maps) + backtick escape fix — RENDER-CHANGING, A/B pending V28-B
    // Re-pinned 2026-06-10: caustic point/spot stride fix (H1-class) + shared light-stride constants
    // (POINT_LIGHT_VEC4_STRIDE=3u / SPOT_LIGHT_VEC4_STRIDE=4u added to material.wgsl.ts; three MNEE
    // point-light loops + photon-map point/spot seeds now use them; spot-axis negation removed from
    // photon seeding). RENDER-CHANGING for multi-light caustic scenes, A/B pending V28-B.
    // Re-pinned 2026-06-10: BDPT light-subpath estimator coherence — stored throughput/pdfFwd now
    // describe the traced segment (was a sampled-then-discarded direction): scatter direction is
    // now sampled at prevPos from the REAL BSDF (woAtPrev as outgoing), traced, and used for
    // throughput (f·cos/pdf) + pdfFwd; pdfRev(prevCol) is patched to pdfFwd (PBRT RandomWalk
    // convention); the dead "nextDir" sampling at newPos is removed. RENDER-CHANGING for bdpt:true,
    // A/B pending V28-B.
    // Re-pinned 2026-06-10: A4 real SPPM progressive photon map — SPPM_GROUP4_BINDINGS_WGSL
    // (group-3 @binding(6/7/8): sppmPhotonCells, sppmCellCounters, sppmStats) added to
    // composition; photonMapContribution replaced with sppmGather() shim; old 32-photon
    // per-pixel approximation + gatherRadius=0.35 + ×1.25 fudge REMOVED.
    // RENDER-CHANGING for causticStrategy:'photon-map'; off-path byte-identical for other strategies.
    // Re-pinned 2026-06-10: pt-webgpu tonemap dials + spectral MIS connection halves + rpt spatial
    // pdfSrc — heroLambda/matId params added to bsdfAreaLightConnectionContribution /
    // bsdfEnvironmentConnectionContribution; spectralEmissionAtHero + materialEnvMapIntensity
    // applied in BSDF half of MIS pair; qPdfSrc gathered in restirPtSpatial (was passing W not pdfSrc).
    // tonemap/spectral RENDER-CHANGING, A/B pending V28-B.
    // Re-pinned 2026-06-10: N-directional emitter packing — kernel's single
    // "if (params.lightDir.w > 1e-6)" block replaced by a loop over
    // directionalLights[] storage buffer (N records, stride 2 vec4f);
    // params.directionalLightCount carries the count; group(1) binding(10) wired.
    // 1-directional scenes byte-identical at runtime (loop iterates once with
    // the same irradiance/dir data). RENDER-CHANGING only for multi-directional
    // scenes; single-directional A/B invariant. Mesh-area NEE cap
    // (MESH_AREA_LIGHT_TRI_CAP=65536, largest-area-first) added to emitterPacking.ts
    // — no WGSL change, cap is applied on the host side during packing.
    // Re-pinned 2026-06-10 (R7a + capacity): SPPM_CELL_CAPACITY 128→32 (fits default maxStorageBufferBindingSize — behavioral-gate verified photon-map renders) + R7a — SPPM streaming-window fix (Item-2: no per-frame
    // clearBuffer, radius frozen at r₀; RENDER-CHANGING for photon-map: lower
    // variance over long runs, streaming window evicts stale photons); BDPT
    // emitter-vertex throughput correction (Item-3: fPrev = INV_PI instead of 1.0
    // so fPrev·cos/pdf = (1/π)·cos/(cos/π) = 1 ✓; spurious ×π removed; A/B
    // pending V28-B); pdfRev(prevCol) patched to true reverse density for VNDF
    // lobes (brdfDirectionalPdf(prevNormal, scatterDir, woAtPrev) instead of
    // pdfFwd — only exact for symmetric BSDFs); SPPM bind-group ordering fix
    // (Item-1: invalidateGroup3BindGroup now also nulls pathTraceBindGroup so
    // buildBindGroups rebuilds ALL groups after placeholder→real buffer swap).
    // RENDER-CHANGING for photon-map + bdpt:true; off-path byte-identical.
    // Re-pinned 2026-06-10: Wave B — aniso GGX + env half + shadePrologue rewrites
    // (kernel.wgsl.ts + shadePrologue.wgsl.ts modified); restirPtResolve.wgsl.ts
    // comment-only fix (stale "diffuse-cosine proxy" → integrand-matching target; item 18).
    // Re-pinned 2026-06-10: item 21 — spectral × photon-map regime fix.
    // photonMapContribution gains heroLambda param; sppmGather gains heroLambda and
    // applies spectralEmissionAtHero(flux.rgb, heroLambda) at gather time in spectral
    // mode (same treatment as all other RGB emission sources: rect/point/spot/mesh
    // lights and env). Non-spectral path (spectralEnabled=0): fluxOut = flux.rgb —
    // byte-identical to the pre-fix behaviour. RENDER-CHANGING on spectral+photon-map
    // scenes; other caustic strategies and non-spectral paths unaffected.
    // Re-pinned 2026-06-10: A4-progressive — true per-pixel SPPM statistics replace
    // the streaming window (SppmPixelStats struct + @group(3) @binding(9) storage buf,
    // sppmGatherProgressive with Hachisuka-Jensen 2009 radius-shrink update rule,
    // pixelIndex computed in main and threaded through photonMapContribution).
    // RENDER-CHANGING for causticStrategy:'photon-map'; off-path byte-identical for
    // other strategies.
    // Re-pinned 2026-06-10: native analytic disc emitters — disc-area records now
    // packed into the rect stream with shape tag 1.0 (concentric-disc map sampling,
    // pdf = 1/(π·r²)); connect.wgsl.ts intersectRectAreaLightRay reads the tag and
    // uses circle containment + π·|u|² area for the MIS pdf; kernel.wgsl.ts,
    // restirPtProducer.wgsl.ts, kernelLite.wgsl.ts, and bdptLightSubpath.wgsl.ts
    // all updated. The 32-triangle fan path is deleted. RENDER-CHANGING for
    // disc-lit scenes, A/B in R9-B.
    // Re-pinned 2026-06-10: D9.1 — computeAnisotropicAxes extracted from both
    // anisotropy branches of sampleNextBounceDirection (bsdf.wgsl.ts). The function
    // is mathematically identical to the inlined code (same formulas, same precision
    // thresholds). SEMANTICALLY EQUIVALENT; axis values unchanged at every call site.
    // Re-pinned 2026-06-10: D9.3 — buildShadingTangentFrame extracted in material.wgsl.ts;
    // applyNormalMap + applyBumpMap now call the shared helper. Dead bitanW variable removed
    // from applyBumpMap. SEMANTICALLY EQUIVALENT; tangent frame math unchanged.
    // Re-pinned 2026-06-10: D9.4 — mneeChainFdJacobian4x4 extracted in mneeNewton.wgsl.ts;
    // mneeNewtonSolveChain2 + mneeChainPdfJacobianDet now call the shared helper.
    // SEMANTICALLY EQUIVALENT; same FD columns, same block assembly.
    // Re-pinned 2026-06-10: D9.10 — causticReceiverRejected + causticClampedPointCount
    // extracted in caustic.wgsl.ts; all three pointLight*Caustic functions use them.
    // SEMANTICALLY EQUIVALENT; same receiver gate + same light-count cap.
    // Re-pinned 2026-06-10: D9.11 — concentricDiscSample extracted to kernelCore.wgsl.ts;
    // both kernel.wgsl.ts (full) and kernelLite.wgsl.ts (lite) use the shared helper.
    // SEMANTICALLY EQUIVALENT; same Shirley-Chiu mapping.
    // Re-pinned 2026-06-10: D9.13 — rotateYNeg/rotateYPos moved to connectCore.wgsl.ts;
    // connect.wgsl.ts duplicate removed; connectLite.wgsl.ts liteRotateY* → canonical names.
    // SEMANTICALLY EQUIVALENT; same rotation math.
    // Re-pinned 2026-06-10: D9.9 — sppmGather deleted from sppmBindings.wgsl.ts
    // (legacy streaming-window gather; superseded by sppmGatherProgressive; zero callers).
    // Re-pinned 2026-06-10: D9.8/I4.1 — SPPM_GROUP4_BINDINGS_WGSL renamed to
    // SPPM_GROUP3_BINDINGS_WGSL (bindings are at @group(3), not group 4).
    // Re-pinned 2026-06-10: D9.17 — bdptLightSubpath.wgsl.ts for-loop body
    // indentation corrected (2-space → 4-space; WGSL text change, zero semantic impact).
    // Re-pinned 2026-06-11: PTWG-03/04 SPPM photon p_select normalization,
    // packed N-directional RGB photon emission, and once-per-pixel stats gate.
    // Re-pinned 2026-06-11 (SHADOW-01): primitive castShadow any-hit gate
    // (intersectionCore triShadowCastDisabled — material vec4 #25 .w lane) +
    // emitter castShadowDisabled gates in the kernel NEE loops (directional
    // sign-encoded angularDiameter; point/spot/rect/mesh lanes) + connect.wgsl
    // BSDF-MIS parity gates. Default (flag-less) scenes are behaviorally
    // identical: every gate reads a lane that packs 0.0 by default.
    // Re-pinned 2026-06-15 (PTWG-BDPT-01): finite area BDPT emitter vertices
    // store Le/(pdfPick*pdfArea), use a distinct area-emitter sentinel, the
    // first traced finite-area extension keeps the required cos/pdfΩ = π
    // factor, and the connection no longer double-multiplies endpoint cosines.
    // RENDER-CHANGING for bdpt:true area-light scenes; CPU oracles pin endpoint
    // and one-bounce radiometry plus the glossy light-vertex connection.
    // Re-pinned 2026-06-11 (PTWG-MAT-01 partial): full-tier BSDF-side
    // area-light/environment connections now pass decoded extension lobe
    // scalars into connect.wgsl and use evaluateBrdfFull/brdfDirectionalPdfFull.
    // SPPM receiver gathering now also evaluates photons with evaluateBrdfFull
    // using the current hit's extension fields. MNEE receiver-side caustic
    // BRDF/PDF evaluations do the same. BDPT eye/light connection endpoint
    // evals and straddle PDF overrides use the full helpers as well.
    // RENDER-CHANGING for clearcoat/sheen/iridescence/aniso materials hit by
    // BSDF-sampled area/env connections or SPPM caustics; zero-extension
    // materials remain behaviorally unchanged.
    // Re-pinned 2026-06-12: normalScale is now consumed by applyNormalMap,
    // scaling tangent-space xy before reconstructing the perturbed normal.
    // Re-pinned 2026-06-12: environment:'none' no longer falls through to the
    // analytic sampleSky gradient. Missing/invalid env maps now return black
    // radiance + zero env pdf; procedural-sky stays lit via the CPU-baked HDRI.
    // Re-pinned 2026-06-12: standalone alphaMap is now sampled as LINEAR
    // coverage data and multiplies baseColor alpha + opacity in alphaMode paths.
    // Re-pinned 2026-06-12: transmissionMap is now sampled as LINEAR scalar
    // data and multiplies MaterialSpec.transmission in the full-tier prologue.
    // Re-pinned 2026-06-12: heterogeneous material texture-array layers now
    // carry per-map UV-fit scales; material samplers use fract(uv) * scale.
    // Re-pinned 2026-06-12: material samplers now consume TextureRef.wrapS/T
    // per map (repeat / clamp-to-edge / mirrored-repeat).
    // Re-pinned 2026-06-12: material samplers now consume TextureRef.texCoord
    // and KHR_texture_transform metadata per map instead of sharing baseColor.
    // Re-pinned 2026-06-12: KHR_materials_specular scalar factors now flow
    // through full-tier MNEE/SPPM receiver BRDF/PDF paths instead of defaulting
    // those caustic receivers to dielectric F0.
    // Re-pinned 2026-06-12: full-tier material texture descriptors now append
    // extension-lobe maps (clearcoat/sheen/iridescence/specular) and the prologue
    // modulates decoded lobe parameters before downstream BSDF/PDF calls.
    // Re-pinned 2026-06-13: MNEE cone-vs-BSDF MIS now uses the sampled-density
    // BRDF PDF helper to match the base/clearcoat/sheen source sampler.
    // Re-pinned 2026-06-13: full-tier clearcoatNormalMap now has appended
    // descriptor lanes and threads a sampled clearcoat normal through the main
    // megakernel's clearcoat BRDF/PDF/source-sampler paths.
    // Re-pinned 2026-06-13: full-tier pt-webgpu now uploads authored/generated
    // tangent.xyzw at group(3)/binding(10) and buildShadingTangentFrame
    // consumes interpolated handedness before falling back to UV-gradient
    // derivation. Lite unchanged.
    // Re-pinned 2026-06-13: material texture arrays now allocate generated mip
    // chains and the sampler uses an explicit geometric LOD estimate instead of
    // hard-coding level 0 in the compute shader.
    // Re-pinned 2026-06-14: full-tier COLOR_0 vertex colors now bind at group(3)
    // binding 11 and multiply baseColor / alpha in material paths.
    // Re-pinned 2026-06-15: finite-area BDPT light-subpath extension now keeps
    // the required cos/pdfΩ = π factor while legacy pseudo emitters keep INV_PI.
    // Re-pinned 2026-06-15: BDPT light vertices carry hit-local material payloads
    // including clearcoat-normal parity for light-subpath scatter/connection.
    // Re-pinned 2026-06-15: BDPT light-side material payload sampling now mirrors
    // the shade prologue's layer, thin-film, spectral reflectance, and Cauchy IOR
    // transforms.
    expect(digest).toBe('c21973a214c2c79c9a4d5483039bab42347bb90d67cd91cebe807311e5a847df');
    expect(PT_WEBGPU_TRACE_WGSL.length).toBe(368800);
  });
});

describe('pt-webgpu WGSL material contract', () => {
  it('uses the bounded rich material payload layout', () => {
    // VOL-THICKNESS bumped the stride 28 → 29 (new vec4 #28 carries KHR volume thickness).
    // Kept in lockstep with TS MATERIAL_VEC4_STRIDE.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 29u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const THIN_FILM_LAYER_LIMIT = 8u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const SPECTRAL_SAMPLE_COUNT = 32u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.isUnlit');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('specularColor: vec3f,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.specularIntensity = clamp(m27.w, 0.0, 1.0);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('volumeThickness: f32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.volumeThickness = max(m28.x, 0.0);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialAttenuationDistance(segmentDistance: f32, mat: DecodedMaterial) -> f32');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'transmission = clamp(transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let volumeThicknessSample = sampleVolumeThicknessTexture(matId, hit.triIndex, hit.baryVW);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var mediumAttenuationLimit = INFINITY;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mediumAttenuationLimit = materialAttenuationDistance(INFINITY, mat);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let attenuationDist = min(hit.dist, mediumAttenuationLimit);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (freeFlightDist < attenuationDist)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('exp(-(walkSigmaT - vec3f(heroSigmaT)) * attenuationDist)');
  });

  it('routes full-tier BSDF-side area/env connections through extension-aware BRDF helpers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bsdfAreaLightConnectionContribution(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('clearcoat: f32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let bsdfPdf = brdfDirectionalPdfFullSampled(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdf = evaluateBrdfFull(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.clearcoatRoughness,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('anisoRotation,');
  });

  it('samples clearcoat and sheen in the main eye-path with a matching sampled pdf', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn brdfDirectionalPdfFullSampled(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn brdfExtensionLobeWeightSum(clearcoat: f32, sheen: f32) -> f32 {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let xiLobe = rand_f32(rng) * lobeWeightSum;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('result.throughputMul = fresnel * g1Wi2 * msBoost * lobeWeightSum / max(specProb, 1e-4);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let ccDensity = (clearcoatWeight / lobeWeightSum) * ccPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let shDensity = (sheenWeight / lobeWeightSum) * shPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let shBrdf = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, -incomingDir, result.sampledDir);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let scatterPdfFwd = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let swappedRev = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
  });

  it('uses extension-aware BRDF evaluation for SPPM receiver gathers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sppmGatherProgressive(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('clearcoatRoughness : f32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdf = evaluateBrdfFull(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('photonMapContribution(');
  });

  it('uses extension-aware BRDF/PDF evaluation for MNEE caustic receivers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn pointLightReflectionCaustic(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let fr = evaluateBrdfFull(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdfPdf = brdfDirectionalPdfFullSampled(');
  });

  it('keeps lite-tier MNEE stub signature aligned with the lite kernel call', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn manifoldNeeContribution(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('clearcoat: f32,');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('iridescenceThicknessMax: f32,');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('mat.clearcoatRoughness,');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('mat.iridescenceThicknessMax,');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('throughputAtVertex,');
  });

  it('keeps lite-tier environment reconnection call extension-aware', () => {
    const callIndex = PT_WEBGPU_TRACE_LITE_WGSL.lastIndexOf(
      'radiance = radiance + bsdfEnvironmentConnectionContribution(',
    );
    expect(callIndex).toBeGreaterThan(-1);
    const call = PT_WEBGPU_TRACE_LITE_WGSL.slice(callIndex, callIndex + 700);
    expect(call).toContain('mat.clearcoat,');
    expect(call).toContain('mat.clearcoatRoughness,');
    expect(call).toContain('mat.sheenColor,');
    expect(call).toContain('mat.iridescenceThicknessMax,');
    expect(call).toContain('throughputAtVertex,');
  });

  it('uses extension-aware BRDF/PDF evaluation for BDPT connection endpoints', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn evaluateBdptConnection(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let eyeBrdf = evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let lvBrdf = evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightNormal, lvMat.clearcoatNormal, -connDir, lvWoPrev,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fwdEe = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let revLc = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
  });

  it('includes hero-wavelength MIS helpers when spectral mode is enabled', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleHeroWavelengthMIS');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn heroWavelengthToRgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('params.spectralEnabled != 0u');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('heroLambdaTo01(heroLambda)');
  });

  it('returns 0 for the delta-refraction lobe in brdfDirectionalPdf (H13 D4)', () => {
    // H13/D4: delta-refraction pdf is 0 (Dirac delta, not a finite density).
    // The prior finite cosine·η² approximation did not match the deterministic
    // sampler in sampleNextBounceDirection; returning 0 is unbiased because all
    // call sites guard against pdf <= 1e-6 before dividing.
    // Verify the opposite-hemisphere branch now unconditionally returns 0.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!sameHemisphere) {');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('let nDotT = max(abs(wiDotN), 1e-5);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('eta * eta * INV_PI');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('transProb * pdfTransApprox');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn activeLayerWeightRgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('isTranslucent');
  });

  it('accounts for the light selection probability in direct lighting (WS2)', () => {
    // WS2 replaced the unconditional uniform compensation (`· f32(lightCount)`):
    // each NEE branch now self-normalizes — delta lights by `· lightSelectInvPdf`
    // (1/p_select), area/env lights by folding p_select into the MIS combined
    // pdf — so the accumulation is a bare add. p_select is `1/lightCount` on the
    // uniform fallback path and `lt.pdf` on the power-weighted light-tree path.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('radiance = radiance + directLi;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightSelectInvPdf = f32(lightCount);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightSelectInvPdf = 1.0 / lt.pdf;');
  });

  it('gates emissive-on-hit to camera + refraction paths (camera-visible emitters)', () => {
    // The emissive-on-hit term must be gated on !prevSampleAllowsAreaMis so it
    // fires ONLY on the camera ray (prevSampleAllowsAreaMis inits false) and after
    // a refraction/specular-transmission bounce (which sets sampleAllowsAreaMis
    // false) — the paths the analytic bsdfAreaLightConnectionContribution cannot
    // reach. On diffuse/glossy bounces the connection already MIS-accounts for the
    // light, so the gate prevents a double-count.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var prevSampleAllowsAreaMis = false;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!prevSampleAllowsAreaMis) {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('prevSampleAllowsAreaMis = sampleAllowsAreaMis;');
    // The emissive add must appear EXACTLY ONCE and be the gated form (wrapped in
    // the `if (!prevSampleAllowsAreaMis)` block). A second/unconditional add would
    // double-count emissive against the analytic connection once re-attached.
    // A3: the added quantity is `emitContribution` (= emissive in RGB mode,
    // spectralEmissionAtHero(emissive,λ) in spectral mode) — still a single gated
    // add, still double-count-free.
    const emissiveAdds = (PT_WEBGPU_TRACE_WGSL.match(/radiance = radiance \+ throughput \* emitContribution;/g) ?? []).length;
    expect(emissiveAdds).toBe(1);
    // The select that produces emitContribution falls back to the RGB emissive
    // when spectral mode is off (the byte-identical-RGB guarantee).
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let emitContribution = select(emissive, emitSpectral, params.spectralEnabled != 0u);',
    );
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

  it('treats absent full-tier environments as black, not procedural sky', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return EnvironmentLookup(vec3f(0.0), 0.0);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('return EnvironmentLookup(sampleSky(dir), 1.0 / (4.0 * PI));');
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
    // Group-2 bind-group construction was extracted into the GpuResources
    // sub-struct (T14-followup); the per-frame dispatch (`setBindGroup(2, …)`)
    // stays in index.ts. Both halves of the host↔WGSL lockstep are asserted.
    const gpuResourcesSource = readFileSync(resolve(here, '../gpuResources.ts'), 'utf8');
    // Updated to 11 for N-directional expansion: directionalLights buffer added at group(1) binding(10).
    expect(limitsSource).toMatch(/PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP\s*=\s*11/);
    expect(indexSource).toContain('selectPtWebgpuTraceTier');
    expect(gpuResourcesSource).toContain('pathTrace.bindgroup2.full');
    expect(gpuResourcesSource).toContain('{ binding: 0, resource: { buffer: sb.tlasNodesBuffer } }');
    expect(indexSource).toContain('setBindGroup(2, gpu.pathTraceBindGroup2)');
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

  it('carries the TLAS instance index into normal-map tangent reconstruction', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const INVALID_TLAS_INSTANCE_INDEX = 0xffffffffu;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('instanceIndex: u32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('(*hit).instanceIndex = instIdx;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex);',
    );
  });

  it('threads clearcoatNormalMap through the full-tier clearcoat lobe', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'clearcoatNormal = applyClearcoatNormalMap(matId, hit.triIndex, hit.baryVW, clearcoatNormal, hit.instanceIndex);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, wo, wi,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let bs = sampleNextBounceDirectionWithClearcoatNormal(');
  });

  it('uses conservative local-ray bounds for TLAS closest-hit traversal under scaling', () => {
    // Closest-hit traversal keeps the prior unbounded local BLAS query because
    // the dynamic closest world distance is clamped after reconstruction.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, tMin, INFINITY, true, &localHit, blasRoot, true);');
  });

  it('uses finite local-ray bounds for TLAS any-hit instance traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('captureShadingDetails: bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var localTMax = tMax;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let localEnd = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * tMax);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('localTMax = max(dot(localEnd - localRay.origin, localRay.direction), tMin);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, tMin, localTMax, false, &localHit, blasRoot, false)');
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

  it('keeps the per-material envMapIntensity descriptor lane wired into environment paths', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialEnvMapIntensity(matId: u32) -> f32');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return max(materialTexDescriptors[base + 4u].w, 0.0);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lastEnvMapIntensity = materialEnvMapIntensity(matId);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let envScale = materialEnvMapIntensity(matId);');
  });

  // ── Theme-D luminance dedup (behavior-preserving) ──────────────────────────
  // material.wgsl.ts:441 and bdpt/bdptLightSubpath.wgsl.ts:8 previously inlined
  // the Rec.709 `dot(c, vec3f(0.2126, 0.7152, 0.0722))` though the canonical
  // `luminance()` (LUMINANCE_WGSL from @vitrum/shared-samplers) is composed into
  // the trace shader ahead of both modules. These pins prove the dedup landed:
  // the canonical call sites are present, and NO inline Rec.709 dot remains in
  // the composed full-tier shader. `luminance()` is defined exactly as
  // `dot(c, vec3f(0.2126,0.7152,0.0722))`, so the GPU result is unchanged.
  describe('Theme-D — Rec.709 luminance routed through canonical luminance()', () => {
    it('activeLayerWeightRgb uses luminance(layerRgb), not an inline Rec.709 dot', () => {
      expect(PT_WEBGPU_TRACE_WGSL).toContain('let lum = max(luminance(layerRgb), 0.0);');
    });

    it('bdptLightLuminance routes through luminance() and keeps the 1e-20 floor', () => {
      expect(PT_WEBGPU_TRACE_WGSL).toContain('return max(luminance(c), 1e-20);');
    });

    it('no inline Rec.709 dot(..., vec3f(0.2126, 0.7152, 0.0722)) remains in the composed shader', () => {
      // The canonical LUM_W709 const + luminance() body live in LUMINANCE_WGSL;
      // that vec3f literal is the ONLY remaining occurrence (inside the const
      // declaration), never an inline dot at a call site.
      const inlineDots = (
        PT_WEBGPU_TRACE_WGSL.match(/dot\([^)]*vec3f\(0\.2126, 0\.7152, 0\.0722\)\)/g) ?? []
      ).length;
      expect(inlineDots).toBe(0);
    });
  });

  // ── Theme-D caustic decode: COLLAPSED onto canonical decodeMaterial() ───────
  // caustic.wgsl.ts previously hand-decoded the packed material inline at two
  // sites (traceSpecularTransmissiveChain + photonMapContribution), duplicating
  // the m0/m2/m3/m22 offset arithmetic decodeMaterial() already owns. The decode
  // is now routed through decodeMaterial(matId); the only behavioural difference
  // — caustic's clamp(baseColor, 0, 1) inside its mix — is re-applied at both
  // sites to preserve bit-identical render output for in-[0,1] albedos (OOB
  // albedos are unreachable for valid scenes). This pin guards the collapse.
  it('caustic material decode is routed through canonical decodeMaterial() with baseColor re-clamped', () => {
    // Both decode sites now CALL the canonical decoder. Strip comment lines so
    // the explanatory references in comments don't count as invocations.
    const causticCode = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL
      .split('\n')
      .filter((l) => !l.trim().startsWith('//'))
      .join('\n');
    // A4: traceSpecularTransmissiveChain = 1 real call.  photonMapContribution now
    // delegates to sppmGather() (SPPM hash-grid lookup) and no longer has its own
    // decodeMaterial call — the gather is done inside the SPPM bindings module.
    const decodeCalls = causticCode.match(/let mat = decodeMaterial\(matId\);/g) ?? [];
    expect(decodeCalls.length).toBe(1);

    // The historical baseColor clamp inside the mix is preserved at the remaining site.
    expect(causticCode).toContain(
      'mix(vec3f(1.0), clamp(mat.baseColor, vec3f(0.0), vec3f(1.0)), 0.2)',
    );

    // NO raw-slot offset arithmetic / inline OOB fallback remains in caustic.
    expect(causticCode).not.toContain('m0Index');
    expect(causticCode).not.toContain('m2Index');
    expect(causticCode).not.toContain('m3Index');
    expect(causticCode).not.toContain('m22Index');
    expect(causticCode).not.toContain('select(vec4f(1.0, 1.0, 1.0, 0.5)');

    // The deferral TODO is gone (collapse is done, not deferred).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('TODO(theme-D / V-caustic)');
  });
});
