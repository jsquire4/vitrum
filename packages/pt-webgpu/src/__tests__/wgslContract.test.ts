import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import {
  FRAME_PARAMS_BYTE_SIZE,
  FrameParamsSlot,
} from '../scene/frameParamsLayout.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { activeLayerWeightRgbOracle } from './spectralScalarOracle.js';

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
    // The full-tier trace composes only the coupled, bounded 1–8 vertex MNEE
    // solver used by boundedManifoldCaustic. mneeBoundedChain.test.ts separately
    // pins every shipped full-tier composer against reintroducing the superseded
    // one-vertex and fixed two-vertex solvers. Recompute this digest on any
    // intentional WGSL change; the byte golden alone does not prove reachability.
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
    // Re-pinned 2026-06-09: H52 — clearcoat (outer GGX F0=0.04), sheen (Charlie
    // NDF + Neubelt-Pettineo visibility), and iridescence (Belcour & Barla 2017
    // thin-film Fresnel F0 modification) layers added to the WebGPU BRDF.
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
    // to 6 rows (hit-local material payload plus interface eta) so the §10.3
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
    // energy compensation in the finite full-BSDF evaluator), B10 (physical
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
    // clearBuffer, insertion hash radius frozen at r₀; RENDER-CHANGING for photon-map:
    // lower variance over long runs, streaming window evicts stale photons); BDPT
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
    // pdf = 1/(π·|u×v|)); connect.wgsl.ts intersectRectAreaLightRay reads the tag and
    // uses circle containment + π·|u×v| area for the MIS pdf; kernel.wgsl.ts,
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
    // Re-pinned 2026-07-29: the zero-consumer one-vertex/fixed-two-vertex MNEE
    // solvers and their harness-only Jacobians were removed. The live bounded
    // O(N) block-Thomas solver and estimator are unchanged.
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
    // Re-pinned 2026-06-12: environment:'none' no longer falls through to an
    // analytic sky gradient. Missing/invalid env maps return black radiance +
    // zero env pdf; procedural skies use their CPU-baked HDRI exclusively.
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
    // Re-pinned 2026-06-15: transmissive dielectrics now share the normalized
    // base/clearcoat/sheen sampled-density helper instead of returning a
    // base-only PDF/sampler before extension lobes can participate.
    // Re-pinned 2026-06-16: full-tier material descriptors gained face-selected
    // front/back layer normal map lanes, and applyNormalMap chooses those lanes
    // ahead of the top-level normal map when authored.
    // Re-pinned 2026-06-16: SPPM progressive gather now queries the stable r0
    // insertion hash grid while retaining the per-pixel shrunk physical disk.
    // Re-pinned 2026-06-16: SPPM over-capacity cells now use bounded reservoir
    // replacement plus totalInserted/storedCount density compensation.
    // Re-pinned 2026-06-16: finite rect/disc/mesh-area reflection MNEE uses the
    // bounded-chain area determinant and exact estimator ownership.
    // Re-pinned 2026-06-17: BDPT secondary connections skip the emitter endpoint
    // already covered by per-bounce NEE, avoiding direct-light double-counting.
    // Re-pinned 2026-06-17: bump maps finite-difference in raw UV space by the
    // uploaded source texel size instead of a baked 512-step barycentric nudge.
    // Re-pinned 2026-06-18: BDPT bounce-0 emitter vertices mirror
    // castShadow:false into row 4.x so eye↔light connections can skip the
    // occlusion ray for that emitter endpoint.
    // Re-pinned 2026-06-18: point-light MNEE caustics decode pointLights
    // extra.z castShadowDisabled and skip only light-leg visibility for
    // castShadow:false point emitters.
    // Re-pinned 2026-06-18: material texture descriptor constants now emit from
    // host layout exports, and extension/special-map samplers use symbolic slots
    // instead of baked vec4 offsets.
    // Re-pinned 2026-06-18: H55 oracle wave fixed concentricDiscSample's
    // Shirley-Chiu signed-denominator bug. Disc-area sampling was previously
    // mirrored into the wrong quadrant for two square quadrants; render-changing
    // for full-tier disc-area lights, now pinned by oracle.concentricDiscSample.
    // Re-pinned 2026-06-18: pt-webgpu sheen source sampling switched from the
    // documented cosine approximation to a matching Charlie half-vector sampler
    // and PDF.
    // Re-pinned 2026-06-20: material texture sampling appended per-map mip
    // policy lanes and applies `none` / nearest / linear mip LOD choices in the
    // explicit-LOD full-tier sampler.
    // Re-pinned 2026-06-20: material texture sampling appended per-map filter
    // policy lanes and uses a nearest `textureLoad` branch for regular maps.
    // Re-pinned 2026-06-21: sampled indirect PT, ReSTIR-PT source sampling, and
    // BDPT light-subpath sampling now feed iridescenceModifiedF0 into their
    // Fresnel/lobe-pdf paths, matching the direct full-BRDF evaluator.
    // RENDER-CHANGING for iridescent materials on sampled indirect paths.
    // Re-pinned 2026-06-21: anisotropic GGX sampled/evaluated paths now use a
    // conservative projected-roughness Kulla-Conty multiscatter approximation.
    // RENDER-CHANGING for rough anisotropic materials.
    // Re-pinned 2026-06-21: ptRngFrameKey preserves PCG's old frameSeed^frameIndex
    // expression while letting Sobol use a monotonic sample key in the opt-in module.
    // Re-pinned 2026-06-21: in-medium directional NEE and ReSTIR-PT suffix NEE
    // now sample signed soft-directional angularDiameter cones like the surface
    // direct-light branch instead of treating them as hard deltas.
    // Re-pinned 2026-06-22: bump-map finite-difference height reads now consume
    // the same per-map mip/filter sampler policy as regular linear material maps.
    // RENDER-CHANGING only for authored non-default bump sampler policies.
    // Re-pinned 2026-07-20 (T1-6): dedicated rgba16float emissive texture array —
    // new @group(3) @binding(17) materialTexturesEmissive + a sampleMaterialLayerEmissive
    // variant; emissive now samples the HDR emissive array (was the sRGB baseColor
    // array). RENDER-CHANGING for HDR textured emissive (values > 1.0 survive);
    // LDR emissive is visually identical (CPU sRGB→linear decode on upload).
    // Re-pinned 2026-07-21: finite dielectric rough transmission and the
    // bounded explicit-BDPT strategy-family ownership/mask are now represented
    // by the composed full-tier shader. The old unconditional 100-unit BDPT
    // contribution clamp was removed, and the bounded BDPT eye walk now keeps
    // q=1 instead of using an unreconstructable one-sided RR density. The full
    // bounded MIS recurrence and power denominator now run in log space.
    // Re-pinned 2026-07-21: two-sided BDPT medium transport and removal of the
    // non-manifold directional cone-caustic approximation changed the source.
    // Re-pinned 2026-07-22: validated TLAS traversal now restarts a stackless
    // all-instance fallback after defensive stack overflow instead of returning
    // a partial closest hit or conservative synthetic occlusion.
    // Re-pinned 2026-07-23: tangent-space normal, clearcoat-normal, and bump
    // maps now derive their frame from the descriptor-selected compact UV slot;
    // mirrored UV/TLAS handedness and anisotropy use that same authored frame.
    // Re-pinned 2026-07-23: geometric-winding double-sided traversal (including
    // mirrored TLAS/CWBVH parity), BDPT/SPPM interior-delta ownership, and
    // wavelength-carrying spectral ReSTIR reuse are intentional render changes.
    // Re-pinned 2026-07-23: finite-light, environment, SPPM, and MNEE receiver
    // evaluation now uses the authored clearcoat-normal frame.
    // Re-pinned 2026-07-24: exact-zero delta classification, stable HG sampling,
    // unbiased SPPM record accumulation, and sampled thin-film transmission are
    // now shared consistently by the full, BDPT, SPPM, and adjoint compositions.
    // Re-pinned 2026-07-27: rough dielectric eval, PDF, and source sampling now
    // share coloured specular/iridescence Fresnel across PT, BDPT, and SPPM.
    // Re-pinned 2026-07-27: removed unread CMF/light-direction UBO payload and
    // made cameraPos a semantic vec3f beside the live HDRI-intensity scalar.
    // Re-pinned 2026-07-27: soft directional medium NEE now keeps sole-family
    // ownership, and straight visibility rejects non-null IOR transitions.
    // Re-pinned 2026-07-27: ReSTIR-PT producer candidates are canonicalized
    // into the packed storage domain before target and weight evaluation.
    // Re-pinned 2026-07-27: finite-BSDF sampling now shares one finalization
    // path, and bounded MNEE uses the production caustic-event implementation.
    // Re-pinned 2026-07-27: medium NEE visibility now traverses nested null
    // boundaries while rejecting opaque or non-null dielectric blockers.
    // Re-pinned 2026-07-28: coherent stacks replace the authored bare Fresnel
    // inside finite rough/material BSDFs and all connection estimators;
    // clearcoat/sheen now attenuate the layers below them.
    // Re-pinned 2026-07-28: straight-ray medium/BDPT alpha traversal uses
    // scene-derived support, and invalid coherent-TMM samples absorb rather
    // than silently becoming perfect mirrors.
    // Re-pinned 2026-07-28: sampled-array mip operands are signed at the WGSL
    // builtin boundary, affine disc emitters use π·|u×v| area throughout, and
    // BSDF connections solve the full Gram system for sheared light axes.
    // Re-pinned 2026-07-28: removed four uncalled legacy BSDF wrappers; the
    // clearcoat-normal-aware layered entry points are the sole production path.
    // Re-pinned 2026-07-28: corrupt thin-film descriptor/LUT ranges absorb
    // instead of turning into a bright perfect-mirror fallback, and the main
    // camera walk uses the scene-derived alpha traversal bound.
    // Re-pinned 2026-07-28: A1 keeps opaque roughness-zero GGX finite and A5
    // makes every shadow traversal honor alpha masks/blends.
    // Re-pinned 2026-07-28: C28-C40 remediation removes unreachable analytic
    // sky/dead helpers, makes caustic receivers transmission-aware, and applies
    // KHR specular/IOR semantics including the IOR-zero transport surrogate.
    // C65 expands the shared Joe-Kuo Sobol table to 512 dimensions; both
    // finite-BSDF caustic receivers carry etaTOverI exactly once.
    // C35 keeps its native s=n-1 strategy helper out of the BDPT-off module;
    // the default composition retains the exact legacy explicit-strategy mask.
    // Re-pinned 2026-07-29: U11 removes every continuous-event proposal-local
    // throughput/PDF calculation overwritten by the finite finalizer.
    // The same final hygiene pass removes legacy MNEE solvers that were
    // composed but unreachable from every full-tier entry point.
    expect(digest).toBe('83c30898ae14f65773b25139456a2a2235203452e97bb18ec8e0313d7d4c1451');
    expect(PT_WEBGPU_TRACE_WGSL.length).toBe(557847);
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
      'transmission = clamp(transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let volumeThicknessSample = sampleVolumeThicknessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('materialAttenuationDistance(INFINITY, mat),');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let attenuationDist = min(hit.dist, eyeMedium.remainingDistance);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (freeFlightDist < attenuationDist)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('exp(-(walkSigmaT - vec3f(heroSigmaT)) * attenuationDist)');
  });

  it('routes full-tier BSDF-side area/env connections through extension-aware BRDF helpers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bsdfAreaLightConnectionContribution(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('clearcoat: f32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let bsdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.clearcoatRoughness,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('clearcoatNormal,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('anisoRotation,');
  });

  it('samples clearcoat and sheen with one centralized finite-event estimator', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn brdfDirectionalPdfFullSampled(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn brdfExtensionLobeWeightSum(clearcoat: f32, sheen: f32) -> f32 {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let xiLobe = rand_f32(rng) * lobeWeightSum;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '(*result).throughputMul = finiteBsdf * cosine / marginalPdf;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('let ccDensity = (clearcoatWeight / lobeWeightSum) * ccPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('let shDensity = (sheenWeight / lobeWeightSum) * shPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('result.sampledLobe = BSDF_LOBE_CLEARCOAT;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('result.sampledLobe = BSDF_LOBE_SHEEN;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let scatterPdfFwd = bs.sampledEventPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('swappedRev = bdptMarginalSurfacePdf(');
  });

  it('uses iridescence-modified F0 for sampled Fresnel and the full sampled PDF', () => {
    const fullF0Start = PT_WEBGPU_TRACE_WGSL.indexOf(
      'let f0Base = materialSpecularF0(\n' +
      '      baseColor, metallic, surfaceEtaTOverI,',
    );
    const fullKernelFresnel = PT_WEBGPU_TRACE_WGSL.slice(
      fullF0Start,
      PT_WEBGPU_TRACE_WGSL.indexOf(
        'let fresnel = materialSpecularFresnelSchlick(',
        fullF0Start,
      ) + 160,
    );
    expect(fullKernelFresnel).toContain('let f0 = iridescenceModifiedF0(');
    expect(fullKernelFresnel).toContain('mat.iridescenceThicknessMax,');

    const liteF0Start = PT_WEBGPU_TRACE_LITE_WGSL.indexOf(
      'let f0Base = materialSpecularF0(\n' +
      '      baseColor, metallic, surfaceEtaTOverI,',
    );
    const liteKernelFresnel = PT_WEBGPU_TRACE_LITE_WGSL.slice(
      liteF0Start,
      PT_WEBGPU_TRACE_LITE_WGSL.indexOf(
        'let fresnel = materialSpecularFresnelSchlick(',
        liteF0Start,
      ) + 160,
    );
    expect(liteKernelFresnel).toContain('let f0 = iridescenceModifiedF0(');
    expect(liteKernelFresnel).toContain('mat.iridescenceThicknessMax,');

    const pdfHelper = PT_WEBGPU_TRACE_WGSL.slice(
      PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfDirectionalPdfWithIridescence('),
      PT_WEBGPU_TRACE_WGSL.indexOf('fn buildOnb(', PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfDirectionalPdfWithIridescence(')),
    );
    expect(pdfHelper).toContain('iridescenceThicknessMax: f32,');
    expect(pdfHelper).toContain('let lobeWeights = brdfFiniteBaseLobeWeights(');
    expect(pdfHelper).toContain('iridescenceThicknessMin, iridescenceThicknessMax,');

    const fullPdf = PT_WEBGPU_TRACE_WGSL.slice(
      PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfDirectionalPdfFullWithClearcoatNormal('),
      PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfExtensionLobeWeightSum(', PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfDirectionalPdfFullWithClearcoatNormal(')),
    );
    expect(fullPdf).toContain('basePdf = brdfDirectionalPdfWithIridescence(');
    expect(fullPdf).toContain('iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,');
  });

  it('exports only the layered BSDF entry points that production estimators call', () => {
    for (const deadWrapper of [
      'fn evaluateBrdf(',
      'fn evaluateBrdfFull(',
      'fn brdfDirectionalPdf(',
      'fn brdfDirectionalPdfFull(',
    ]) {
      expect(PT_WEBGPU_TRACE_WGSL).not.toContain(deadWrapper);
      expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain(deadWrapper);
    }
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'fn evaluateFiniteBsdfFullWithClearcoatNormal(',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'fn brdfDirectionalPdfFullSampledWithClearcoatNormal(',
    );
  });

  it('keeps transmissive dielectric source sampling aligned with the full sampled pdf', () => {
    const pdfStart = PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    const pdfEnd = PT_WEBGPU_TRACE_WGSL.indexOf('fn brdfDirectionalPdfFullSampled(', pdfStart + 1);
    const pdfHelper = PT_WEBGPU_TRACE_WGSL.slice(pdfStart, pdfEnd);
    expect(pdfHelper).not.toContain('if (transmission > 0.0 && metallic == 0.0)');
    expect(pdfHelper).toContain(') / brdfExtensionLobeWeightSum(clearcoat, sheen);');

    const branchStart = PT_WEBGPU_TRACE_WGSL.indexOf('// Transmissive (dielectric) surface');
    const branchEnd = PT_WEBGPU_TRACE_WGSL.indexOf('// Non-transmissive surface', branchStart);
    const branch = PT_WEBGPU_TRACE_WGSL.slice(branchStart, branchEnd);
    expect(branch).toContain('let lobeWeightSum = brdfExtensionLobeWeightSum(clearcoat, sheen);');
    expect(branch).toContain('let xiLobe = rand_f32(rng) * lobeWeightSum;');
    expect(branch).toContain('let xiBase = xiLobe;');
    expect(branch).not.toContain('microfacetInterface.reflectance * g1Wi *');
    expect(branch).toContain('finalizeFiniteBounceSampleWithClearcoatNormal(');
    expect(branch).toContain(
      'baseColor * transmissionWeight *\n' +
        '            microfacetInterface.baseTransmittance *',
    );
    expect(branch).toContain(
      'let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);',
    );
    expect(branch).not.toContain('let shDensity = (sheenWeight / lobeWeightSum) * shPdf;');
    expect(branch).toContain('result.sampledLobe = BSDF_LOBE_SHEEN;');
  });

  it('uses extension-aware BRDF evaluation for SPPM receiver gathers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sppmUpdateSurfaceProgressive(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sppmCurrentProgressiveEstimate(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('clearcoatRoughness : f32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('photonMapUpdateProgressive(');
  });

  it('uses extension-aware BRDF/PDF evaluation for MNEE caustic receivers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn boundedManifoldCaustic(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let emitter = mneeSampleEmitter(rng);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let fr = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
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
    expect(PT_WEBGPU_TRACE_WGSL).toContain('eyeBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let lvBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightNormal, lvMat.clearcoatNormal, lvWoPrev, -connDir,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fwdEe = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var revLc = 0.0;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('revLc = bdptTransmissiveConnectionPdf(');
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
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var lightSelection: DirectLightSelection;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightSelection = sampleDistantDirectLight(sumDirectLighting, &rng);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightSelection = sampleCanonicalDirectLight(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let lightSelectInvPdf = lightSelection.invPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let directLightingScale = select(\n        lightSelectInvPdf, 1.0, sumDirectLighting,\n      );',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return DirectLightSelection(index, f32(lightCount));');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('1.0 / selected.pdf,');
  });

  it('keeps finite-emitter measures exact and MIS-accountable across rough transmission', () => {
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('max(PI * rradL * rradL, 1e-6)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('max(4.0 * length(cross(ru, rv)), 1e-6)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('max(cosLight * area, 1e-6)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('let worldArea = max(uvCAndMaterial.w, 1e-8);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('let denom = max(d00 * d11 - d01 * d01, 1e-20);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let nDotL = abs(dot(normal, wi));');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let offsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('roughTransmissionProposalPdf <= 0.0');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'result.sampleAllowsAreaMis = roughTransmissionProposalPdf > 0.0;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '(*result).sampleAllowsAreaMis = true;',
    );
  });

  it('gates emissive-on-hit to paths without an MIS-accountable prior event', () => {
    // The emissive-on-hit term must be gated on !prevSampleAllowsAreaMis so it
    // fires only on the camera ray (prevSampleAllowsAreaMis inits false) and
    // after a delta event that the analytic BSDF↔light connection cannot reach.
    // Rough transmission has a finite directional density and remains
    // MIS-accountable, just like diffuse/glossy reflection.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var prevSampleAllowsAreaMis = false;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!prevSampleAllowsAreaMis && !sppmOwnsCurrentEmission) {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('prevSampleAllowsAreaMis = prevEventKind != 0u && prevDirectionalPdf > 0.0;');
    // The emissive add must appear EXACTLY ONCE and be the gated form (wrapped in
    // the `if (!prevSampleAllowsAreaMis)` block). A second/unconditional add would
    // double-count emissive against the analytic connection once re-attached.
    // A3: the added quantity is `emitContribution` (= emissive in RGB mode,
    // spectralEmissionAtHero(emissive,λ) in spectral mode) — still a single gated
    // add, still double-count-free.
    const emissiveAdds = (
      PT_WEBGPU_TRACE_WGSL.match(
        /radiance = radiance \+ throughput \* emitContribution \* clearcoatEmissionAttenuation;/g,
      ) ?? []
    ).length;
    expect(emissiveAdds).toBe(1);
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '1.0 - clearcoatLayerWeight(mat.clearcoat, clearcoatNormal, -ray.direction)',
    );
    // The select that produces emitContribution falls back to the RGB emissive
    // when spectral mode is off (the byte-identical-RGB guarantee).
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let emitContribution = select(emissive, emitSpectral, params.spectralEnabled != 0u);',
    );
  });

  it('suppresses the raw environment miss after an MIS-accounted BSDF connection', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(
      /if \(!hit\.didHit\) \{[\s\S]*?if \(!prevSampleAllowsAreaMis && !sppmOwnsCurrentEmission\) \{[\s\S]*?radiance = radiance \+ throughput \* envContribution \* lastEnvMapIntensity;[\s\S]*?\}\s*break;/,
    );
    const rawMissAdds = (
      PT_WEBGPU_TRACE_WGSL.match(
        /radiance = radiance \+ throughput \* envContribution \* lastEnvMapIntensity;/g,
      ) ?? []
    ).length;
    expect(rawMissAdds).toBe(1);
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'radiance = radiance + bsdfEnvironmentConnectionContribution(',
    );
  });

  it('contains active strategy-specific caustic paths', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn causticMode() -> u32');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn photonMapUpdateProgressive');
    // Caches the strategy code in `caustic` and branches on the local, so
    // the manifold/photon dispatch sites read a single causticMode() call.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let caustic = causticMode();');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'caustic == 1u && mneeReceiverEligible',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('else if (sppmActive)');
  });

  it('threads finite coherent thin film through every connection family', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (directFamilyCount > 0u && !sppmOwnsCurrentDirect)',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      '!sppmOwnsCurrentDirect && !thinFilm.enabled',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'bdptOwnsFiniteLightFamily && !thinFilm.enabled',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'let sppmReceiverEligible = !thinFilm.enabled',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain(
      'vMat.isUnlit || vMat.thinFilmEnabled',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain(
      'if (lvMat.thinFilmEnabled) {',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'anisoStrength, anisoRotation, thinFilm, false);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'lvMat.anisotropy, lvMat.anisotropyRotation, lvMat.thinFilm, true,',
    );
  });

  it('declares INV_2PI alongside INV_PI for HDRI equirect sampling', () => {
    // pathTraceBruteforce.wgsl.ts uses INV_2PI for spherical-to-UV mapping;
    // omitting it would cause a runtime WGSL compile failure for any scene
    // with an HDRI environment.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const INV_2PI');
  });

  it('treats absent full-tier environments as black, not procedural sky', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return EnvironmentLookup(vec3f(0.0), 0.0);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('sampleSky');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('environmentSun.w');
  });

  it('FrameParams uses the exact generated 384-byte host/GPU payload', () => {
    const stride = (PT_WEBGPU_TRACE_WGSL.match(/struct FrameParams/g) ?? []).length;
    expect(stride).toBeGreaterThanOrEqual(1);
    expect(FRAME_PARAMS_BYTE_SIZE).toBe(384);
    // Verify FrameParams contains the matrix fields. Exact byte offsets
    // are pinned numerically against the generated slot table in the next test.
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
    const indexSource = readFileSync(resolve(here, '../index.ts'), 'utf8');
    // Group-2 bind-group construction was extracted into the GpuResources
    // sub-struct (T14-followup); the per-frame dispatch (`setBindGroup(2, …)`)
    // stays in index.ts. Both halves of the host↔WGSL lockstep are asserted.
    const gpuResourcesSource = readFileSync(resolve(here, '../gpuResources.ts'), 'utf8');
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
      'normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex, isFrontFace);',
    );
  });

  it('threads clearcoatNormalMap through the full-tier clearcoat lobe', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'clearcoatNormal = applyClearcoatNormalMap(matId, hit.triIndex, hit.baryVW, clearcoatNormal, hit.instanceIndex);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let bs = sampleNextBounceDirectionWithClearcoatNormal(');
  });

  it('transforms both TLAS closest-hit bounds into BLAS-local distance', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let localTMin = max(dot(localStart - localRay.origin, localRay.direction), 0.0);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'traceMeshBvh(localRay, localTMin, localTMax, true, &localHit, blasRoot, true);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('localHit.dist < (*hit).dist');
  });

  it('transforms both TLAS any-hit bounds into BLAS-local distance', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('captureShadingDetails: bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var localTMax = tMax;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let localEnd = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * tMax);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('localTMax = max(dot(localEnd - localRay.origin, localRay.direction), localTMin);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, localTMin, localTMax, false, &localHit, blasRoot, false)');
  });

  it('returns closest-hit result from traceMeshBvh when in closest mode', () => {
    // Regression guard: closest-mode traversal must return didHit so
    // traceAny fallback paths without TLAS still report mesh occlusion.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return select(false, (*hit).didHit, closest);');
  });

  it('routes traceAny through sided closest candidates before accepting occlusion', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let hit = traceClosestRaw(ray, cursor, tMax);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('materialAcceptsSidedHit(matId, hit.frontFace) &&');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('!materialShadowCastDisabled(matId)');
  });

  it('restarts any-hit and closest-hit overflow from the complete instance permutation', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn tlasClosestFallback(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn tlasAnySceneFallback(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return tlasClosestFallback(ray, tMin, tMax, hit);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return tlasAnySceneFallback(ray, tMin, tMax);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;\n        return (*hit).didHit;',
    );
  });

  it('preserves TLAS instance ids through normal and overflow-fallback closest paths', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'fn tlasResetSceneHit(hit: ptr<function, SceneHit>, tMax: f32)',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '(*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'fn tlasTraceSceneInstanceClosest(',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('(*hit).instanceIndex = instIdx;');
    const fallback = PT_WEBGPU_TRACE_WGSL.slice(
      PT_WEBGPU_TRACE_WGSL.indexOf('fn tlasClosestFallback('),
      PT_WEBGPU_TRACE_WGSL.indexOf('fn tlasTraceSceneInstanceAny('),
    );
    expect(fallback).toContain('tlasResetSceneHit(hit, tMax);');
    expect(fallback).toContain('tlasTraceSceneInstanceClosest(');
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
    it('activeLayerWeightRgb preserves RGB mode and is scalar at the hero wavelength', () => {
      expect(activeLayerWeightRgbOracle([0.8, 0.3, 0.1], 540, false)).toEqual([0.8, 0.3, 0.1]);
      const spectral = activeLayerWeightRgbOracle([0.8, 0.3, 0.1], 540, true);
      expect(spectral[0]).toBe(spectral[1]);
      expect(spectral[1]).toBe(spectral[2]);
      expect(PT_WEBGPU_TRACE_WGSL).toContain('spectralRgbFactorAtHero(layerRgb, heroLambda)');
      expect(PT_WEBGPU_TRACE_WGSL).not.toContain('heroWavelengthToRgb(heroLambda, luminance(layerRgb)');
    });

    it('BDPT source selection is exactly uniform without a luminance proxy', () => {
      expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bdptEmitterCount() -> u32');
      expect(PT_WEBGPU_TRACE_WGSL).toContain('let threshold = ((0xffffffffu % emitterCount) + 1u) % emitterCount;');
      expect(PT_WEBGPU_TRACE_WGSL).toContain('if (word >= threshold) { return word % emitterCount; }');
      expect(PT_WEBGPU_TRACE_WGSL).not.toContain('fn bdptLightLuminance(');
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

  // ── Theme-D caustic decode: canonical decodeMaterial() only ────────────────
  it('caustic material reads use canonical decodeMaterial without raw slots', () => {
    const causticCode = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL
      .split('\n')
      .filter((l) => !l.trim().startsWith('//'))
      .join('\n');
    expect(causticCode).toContain('let matId = triMaterialIds[facet.triIndex];');
    expect(causticCode).toContain('let mat = decodeMaterial(matId);');
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
