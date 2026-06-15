/**
 * Shading + ReSTIR-GI compute pass.
 *
 * Reads the spatialReservoir (DI result), re-traces the primary ray to find
 * the hit surface (since we're using primary-ray-cast mode instead of a G-buffer),
 * traces one indirect bounce (ReSTIR GI), and writes HDR color to hdrColorOut.
 *
 * This is the primary-ray-cast fallback mode.
 *
 * W4-A5 — `shadeMain` is split into one helper per lighting term:
 *   lo_emit, lo_direct, lo_sg_caustic, lo_sg_aperture, lo_indirect.
 * Each helper reads module-scope state (UBO + storage + textures) directly
 * and takes only the local surface scalars/vectors it needs as parameters.
 * `shadeMain` becomes a clean composition: sky-miss early-out → primary
 * re-cast → call each helper → AO modulate + firefly clamp → texture store.
 *
 * T5 — the two stained-glass-specific terms (sun-caustic + sky-aperture)
 * were extracted out of this general pass into `stainedGlassShade.wgsl.ts`
 * (lo_sg_caustic / lo_sg_aperture), opt-in behind ubo.stainedGlassFlags
 * (default OFF). shade no longer carries stained-glass knowledge; it just
 * calls the two helpers, which early-return vec3f(0) when their flag bit is
 * unset (mirroring the sampleCascadeC0 RC precedent). SHADE_MODULE.requires
 * lists `stainedGlassShade` so the composer emits those bodies first.
 *
 * D5.8b (resolved 2026-06-11 — raw-string interpolation, byte-identical):
 * The 8 lo_* helpers (lo_emit, lo_emitterGlow, lo_analyticNEE, lo_direct,
 * lo_sunNEE, lo_indirect, lo_transmittedGI, lo_indirectSpecular) are now
 * defined in `shadingTerms.wgsl.ts` and interpolated into SHADE_WGSL via
 * `${SHADING_TERMS_WGSL}` at the same position. A WgslModule extraction was
 * blocked because the helpers reference @group/@binding declarations that
 * the composer would emit first (see the WGSL comment block at the injection
 * site and the shadingTerms.wgsl.ts docblock for full rationale).
 * Final composed shader string is byte-identical to the original inline form.
 */

// Atlas-layout constants are consumed by ddgiSampleWgsl.ts (the canonical
// DDGI atlas sampler); shade.wgsl delegates via ddgiSampleFromBindings —
// no direct constant references needed here.
import type { WgslModule } from '../pipeline/wgslComposer.js';
import { SHADING_TERMS_WGSL } from './shadingTerms.wgsl.js';

export const SHADE_WGSL = /* wgsl */ `

@group(0) @binding(0) var gDepth:     texture_2d<f32>;
@group(0) @binding(1) var gNormal:    texture_2d<f32>;
@group(0) @binding(2) var gAlbedo:    texture_2d<f32>;
@group(0) @binding(3) var gRough:     texture_2d<f32>;
@group(0) @binding(4) var motionVec:  texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> currentReservoir:  array<u32>;
@group(0) @binding(6) var<storage, read>       previousReservoir: array<u32>;
@group(0) @binding(7) var<storage, read_write> spatialReservoir:  array<u32>;
@group(0) @binding(8) var hdrColorOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var nearestSampler: sampler;
// G-buffer write target — encoded normal in xyz (n*0.5+0.5) and primary-hit
// distance in w.  Authored here in shade and consumed by the à-trous denoiser
// in the next pass for normal+depth edge stopping.
@group(0) @binding(10) var gNormalDepthOut: texture_storage_2d<rgba16float, write>;
// Sprint 16 — ReSTIR-GI half-res reservoir. Written by risGi; read here
// for the Lo_indirect term. The dispatch grid in shade is full-res, so
// we sample at (gid.xy / 2) — each 2×2 quad shares one GI reservoir.
@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent: array<u32>;
// Sprint 18 — split indirect output for per-channel atrous-variance tuning. The
// main hdrColorOut carries direct + emit + sun caustic + sky aperture
// (already AO-modulated). The hdrIndirectOut carries Lo_indirect (× AO)
// so it can be denoised with broader sigmas before recombination. The
// combine pass downstream sums denoisedDirect + smoothedIndirect.
@group(0) @binding(12) var hdrIndirectOut: texture_storage_2d<rgba16float, write>;
// Sprint 18 follow-up — total radiance output. Welford reads this so the
// per-pixel variance estimate and the sample-budget tier reflect the full
// signal (direct + indirect), not just the direct channel.
@group(0) @binding(13) var hdrTotalOut: texture_storage_2d<rgba16float, write>;
// Item 24 — albedo demodulation (Schied 2017 §4.1). The indirect channel
// is written WITHOUT the albedo factor so the à-trous chain operates on
// pure lighting (L/albedo). hdrAlbedoOut carries the visible-point diffuse
// colour; indirectCombine re-multiplies (filtered_lighting × albedo) after
// the denoising chain.
@group(0) @binding(14) var hdrAlbedoOut: texture_storage_2d<rgba16float, write>;
// SVGF-real object ID output. 0 is sky; nonzero hit IDs combine the TLAS
// instance index and the hit triangle index so reprojection rejects history
// across independently moving objects / primitives / triangles.
@group(0) @binding(15) var svgfObjectIdOut: texture_storage_2d<r32uint, write>;

// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(4) var<storage, read> emitterCdf:   array<f32>;
// WS1 (2026-05-29) — per-tri Beer-Lambert visible color (RGBA8 packed,
// alpha=0), now an r32uint TEXTURE rather than a storage buffer so it no longer
// counts against maxStorageBuffersPerShaderStage (freeing a slot for
// bvh_normal). Read on primary glass hits to make Lo_emit reproduce PT's
// transmitted-radiance saturation. bvh_index.w stays raw attCol for receiver
// paths. Texel addressing: triIndex → vec2u(tri % BVH_BEER_TEX_WIDTH,
// tri / BVH_BEER_TEX_WIDTH); the width constant matches host bvhBeerTexture.ts.
@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
// WS1 — per-vertex world-space normals for the smooth shading-normal blend.
// Camera-visible emitters (2026-05-30) — per-triangle HDR emissive radiance Le
// (rgba32float texture). Read by lo_emitterGlow on a primary hit so emissive-mesh
// surfaces glow to the camera (the ReSTIR-DI emitter list only lights RECEIVERS;
// without this the emitter's own pixels render black). Shade-only binding.
@group(1) @binding(12) var bvh_emissive: texture_2d<f32>;
@group(1) @binding(13) var analytic_lights: texture_2d<f32>;
// B1 — per-triangle roughness+metalness (r32uint texture, binding 14). Decoded
// into the real GGX roughness/metal that feed lo_direct / lo_analyticNEE and the
// glossy/metal specular-indirect lobe (was hardcoded rough=0.85/0.05, metal=0).
@group(1) @binding(14) var bvh_material: texture_2d<u32>;
// (BVH_BEER_TEX_WIDTH is declared in surfaceTextures.wgsl, emitted earlier in
// the shade compose chain, and reused here.)
// WalkaroundUBO struct defined in COMMON_WGSL.
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// Sprint 15 — GTAO occlusion factor (rgba16float, full-res, 1-frame lagged).
// .rgb carries the per-channel Jiménez 2016 §5.2 multi-bounce AO (one
// factor per RGB channel based on the surface albedo). Multiplied into
// diffuse direct + indirect terms below to darken contact regions on a
// per-channel basis. Sky-miss and emissive light sources bypass this
// multiplier.
@group(2) @binding(1) var aoFullTexture: texture_2d<f32>;

// DDGI bind group (group 3). Atlas + sampler + grid params UBO bound here
// so the pipeline layout matches risGi.wgsl (which is the real consumer of
// the atlas at the reconnection vertex — Sprint 16 replaced shade's direct
// atlas read with reservoir consumption). Shade does not reference these
// bindings; they are declared only to keep the layout valid for layout
// compatibility checks and to leave the door open for future fallback paths.
//
// D5.1+D5.2: DDGIGridUBO struct and @group(3) @binding(3) are now provided by
// the shared ddgiGridUbo module (declared canonical in ddgiSampleWgsl.ts).
// The binding(0..2) atlas + sampler remain here for layout completeness.
// Cross-reference: ddgiGridUbo → ddgiSampleWgsl.ts (canonical source).
@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;
@group(3) @binding(2) var ddgiSampler:    sampler;

// RESERVOIR_DI_STRIDE / loadReservoirDI_rw live in COMMON_WGSL.

fn loadSpatialDI(pixelIdx: u32) -> ReservoirDI {
  return loadReservoirDI_rw(&spatialReservoir, pixelIdx);
}

fn stableSvgfObjectId(hit: IntersectionResult) -> u32 {
  let inst = hit.instanceIndex + 1u;
  let tri = hit.indices.w + 1u;
  var h = 2166136261u;
  h = (h ^ inst) * 16777619u;
  h = (h ^ tri) * 16777619u;
  return select(h, 1u, h == 0u);
}

fn storeSvgfObjectId(pix: vec2u, id: u32) {
  let dims = textureDimensions(svgfObjectIdOut);
  if (pix.x < dims.x && pix.y < dims.y) {
    textureStore(svgfObjectIdOut, pix, vec4u(id));
  }
}

// invertMat4_common + generatePrimaryRay_common live in common.wgsl;
// injected via common.wgsl (W1-R6 wgslComposer requires chain).

// ──────────────────────────────────────────────────────────────────────────
// Per-lighting-term helpers (W4-A5).
//
// Each fn computes one Lo_* term in isolation. The helpers read module-scope
// state (ubo, bvh, emitters, spatialReservoir, reservoirGiCurrent, bvh_beer)
// directly; only the per-pixel locals are passed as params. Gating (isGlass /
// isMetal) lives inside each helper so shadeMain stays a flat composition.
//
// D5.8b (deferred — composer limitation):
// The lo_* helpers cannot be extracted into a shared 'shadingTerms' WGSL
// module because they reference binding declarations (@group(N) @binding(M))
// and the WalkaroundUBO binding that are declared in THIS shader body
// (SHADE_WGSL). The wgslComposer emits required modules BEFORE the consumer's
// own source, so a 'shadingTerms' module source would appear before those
// declarations exist in the concatenated string. Moving the declarations into
// the module would duplicate them if any other shade-chain module also
// required 'shadingTerms'. Resolution requires a composer feature that supports
// post-consumer or peer-level injection ordering, which does not exist today.
// ──────────────────────────────────────────────────────────────────────────

${SHADING_TERMS_WGSL}

@compute @workgroup_size(8, 8, 1)
fn shadeMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;

  // Checkerboard sparse-shade (opt-in; OFF by default). The host COMPACTS the
  // dispatch when checkerboardOn == 1u to ~half the threads (one per
  // active-parity pixel), so gid is an index into the active-parity pixel set
  // rather than the full-res pixel. Decode it back to the true pixel pix:
  //   px = gid.x*2 + ((gid.y + frameParity) & 1u),  py = gid.y
  // This lands EXACTLY on the (px+py)&1u == frameParity set the old full-res
  // dispatch shaded (and the OLD path early-returned the complementary gap
  // pixels for resolve.wgsl to reproject). frameParity here is the SAME
  // frameCount&1 phase ResolvePass writes into ResolveUniforms.frameParity, so
  // the shaded pixels match the resolve gap-fill exactly. The compacted X grid
  // (ceil(W/2) columns) can overshoot the row's last active pixel on odd
  // widths; that overshoot lands at px >= W and is caught by the bounds guard.
  // When OFF, pix == gid.xy and the dispatch is full-res => bit-identical with
  // the pre-checkerboard kernel.
  var pix = gid.xy;
  if (ubo.checkerboardOn == 1u) {
    let startCol = (gid.y + ubo.frameParity) & 1u;
    pix = vec2u(gid.x * 2u + startCol, gid.y);
  }
  if (any(pix >= dims)) { return; }

  let pixelIdx = pix.y * dims.x + pix.x;
  var rng = pcgInit(pix.x ^ 11111u, pix.y ^ 22222u, ubo.frameSeed ^ 0xDEADu);

  // Re-trace primary ray to find hit (primary-ray-cast mode).
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(pix.x, pix.y, dims.x, dims.y, ubo.cameraPos, invVP);
  let primaryHit = traceSceneFirstHitAlphaMaskTextured(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    primaryRay, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH);

  if (!primaryHit.didHit) {
    // Sky pixel: output sky color (already written by RIS pass, but keep consistent).
    // B3 — directional IBL: sample the actual map along the camera ray so RIS miss
    // + shade miss agree (both call envRadiance(primaryRay.direction)); falls back
    // to the scalar skyTint × skyIrradiance with no HDRI (no-HDRI byte-identity).
    let skyMiss = envRadiance(primaryRay.direction);
    textureStore(hdrColorOut, pix, vec4f(skyMiss, 1.0));
    // G-buffer for sky: encoded "up" normal + depth=0.  The atrous denoiser
    // uses depth=0 as a sentinel that distinguishes sky from non-sky and
    // prevents floor radiance bleeding into sky pixels (or vice versa).
    textureStore(gNormalDepthOut, pix, vec4f(0.5, 1.0, 0.5, 0.0));
    // Item 24: sky pixels have no surface albedo. Write (1,1,1) so
    // indirectCombine's re-modulation is a no-op for sky pixels.
    textureStore(hdrAlbedoOut,   pix, vec4f(1.0, 1.0, 1.0, 1.0));
    textureStore(hdrIndirectOut, pix, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(hdrTotalOut,    pix, vec4f(skyMiss, 1.0));
    storeSvgfObjectId(pix, 0u);
    return;
  }

  // ── Primary-hit surface derivation ───────────────────────────────────────
  let pos    = primaryRay.origin + primaryRay.direction * primaryHit.dist;
  // WS1 — geometric face normal (kept for ray offsets / backface bias) vs the
  // SMOOTH barycentric shading normal (used for lighting + the G-buffer the
  // denoiser edge-stops on). V21 — the smooth normal applies in BOTH merged and
  // TLAS modes; in TLAS the LOCAL-space blend is transformed to world by the hit
  // instance's inverse-transpose (see the n_isTlas branch below).
  let geoNormal = primaryHit.normal;
  // V21 — the smooth shading normal now applies in TLAS mode too: the blended
  // barycentric normal is LOCAL-space there, so transform it to world by the hit
  // instance's inverse-transpose (world-to-local cols). Merged mode (isTlas=false)
  // leaves the blend world-space. Columns read here (binding in scope) + passed by
  // value (Naga rejects ptr<storage> params).
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = primaryHit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < arrayLength(&tlasInstanceWorldToLocal);
  let n_i = select(0u, n_base, n_ok);
  let n0 = bvh_normal[primaryHit.indices.x];
  let n1 = bvh_normal[primaryHit.indices.y];
  let n2 = bvh_normal[primaryHit.indices.z];
  let smoothNormal = smoothShadingNormal(
    primaryHit, geoNormal,
    n0.xyz, n1.xyz, n2.xyz,
    n_ok,
    tlasInstanceWorldToLocal[n_i], tlasInstanceWorldToLocal[n_i + 1u], tlasInstanceWorldToLocal[n_i + 2u],
  );
  let normal = applyNormalMapForHit(primaryHit, smoothNormal);
  let wo     = -primaryRay.direction;

  // Decode per-triangle material color from bvhIndex[triIdx].w (RGBA8 packed).
  let scalarMatColor = decodeMaterialColor(primaryHit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(primaryHit, scalarMatColor.a),
  );
  let isGlass  = matColor.a > 0.3;  // transmission > ~76/255
  let isMetal  = decodeIsMetal(primaryHit.matColorPacked);  // came / solder

  // Write the G-buffer.  Normal encoded as (n*0.5+0.5) so the atrous shader
  // can decode with n = raw*2 - 1.  Depth = primary-hit distance along ray,
  // SIGN-FLIPPED for glass primary hits to encode the surface-type
  // discriminator the atrous denoiser uses to gate bleed across the
  // panel-wall boundary.
  let depthSigned = primaryHit.dist * select(1.0, -1.0, isGlass);
  textureStore(gNormalDepthOut, pix, vec4f(normal * 0.5 + 0.5, depthSigned));
  storeSvgfObjectId(pix, stableSvgfObjectId(primaryHit));

  // Use the BVH-baked material color for ALL surfaces (glass AND room surfaces).
  let uv1 = interpolateUv1FromNormalW(primaryHit, n0, n1, n2);
  let albedo   = sampleBaseColorMap(primaryHit.indices.w, primaryHit.uv, uv1, matColor.rgb);
  // B1 — real authored roughness/metalness from the per-tri bvh_material texture
  // (was hardcoded rough=select(0.85,0.05,isGlass)/metal=0). The diffuse-default
  // invariant packs 0.85 for unspecified roughness / 0.05 for glass / metal 0,
  // so a default-diffuse scene is numerically unchanged; authored glossy/metal
  // surfaces now drive the GGX direct lobe + the specular-indirect term below.
  let rmCoord  = vec2u(primaryHit.indices.w % BVH_MATERIAL_TEX_WIDTH, primaryHit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let materialWord = textureLoad(bvh_material, vec2i(rmCoord), 0).r;
  let rm       = decodeRoughMetal(materialWord);
  let rough    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, primaryHit.uv, uv1, rm.x);
  let metal    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, primaryHit.uv, uv1, rm.y);
  let specular = sampleSpecularControls(primaryHit.indices.w);
  let clearcoat = sampleClearcoatControls(primaryHit.indices.w);
  let authoredAo = sampleAoMapFactor(primaryHit.indices.w, materialWord, primaryHit.uv, uv1);

  // GLTF-unlit — approximate KHR_materials_unlit support for walkaround:
  // output the authored base color directly, bypassing all lighting and GI.
  // This keeps the field consumed and deterministic while the realtime GI
  // backend remains an approximate material renderer rather than a full glTF
  // rasterization pipeline.
  if (decodeIsUnlitMaterial(materialWord)) {
    textureStore(hdrColorOut,    pix, vec4f(albedo,      1.0));
    textureStore(hdrIndirectOut, pix, vec4f(vec3f(0.0), 1.0));
    textureStore(hdrAlbedoOut,   pix, vec4f(albedo,      1.0));
    textureStore(hdrTotalOut,    pix, vec4f(albedo,      1.0));
    return;
  }

  // ── Per-term lighting composition ────────────────────────────────────────
  //
  // Locals are named identically to their helper outputs (Lo_emit, Lo_direct,
  // Lo_sunCaustic, Lo_skyAperture, Lo_indirect) so the structural-contract
  // tests in sprint18-indirectCombine.test.ts continue to match.
  let Lo_emit       = lo_emit(matColor, normal, isGlass, primaryHit.uv, primaryHit.matColorPacked, primaryHit.indices.w);
  // Camera-visible emitters — emissive-mesh self-emission on the primary hit.
  let Lo_emitterGlow = sampleEmissiveMap(
    primaryHit.indices.w,
    primaryHit.uv,
    uv1,
    lo_emitterGlow(primaryHit.indices.w),
  );
  // Baked camera-visible outgoing radiance. Like the PT backends, this is
  // first-hit only and additive; it does not feed ReSTIR emitter power or GI.
  let Lo_lightMap = sampleLightMap(primaryHit.indices.w, primaryHit.uv, uv1);
  let Lo_direct     = lo_direct(pixelIdx, pos, normal, geoNormal, wo, albedo, rough, metal, specular, clearcoat, isGlass, isMetal, &rng);
  // H41 — analytic point/spot NEE: additive, separate from the RIS area-emitter pool.
  // No PDF contamination: these are disjoint from the emitters[] stream.
  let Lo_analyticNEE = lo_analyticNEE(pos, normal, geoNormal, albedo, rough, metal, specular, clearcoat, wo, isGlass, isMetal);
  // item 4 (2026-06-10) — direct sun NEE: deterministic shadow ray toward the sun,
  // full BRDF (diffuse + GGX specular). Default-ON for opaque surfaces; glass skips.
  // See lo_sunNEE above for the no-double-count argument re: DDGI indirect vs direct.
  let Lo_sunNEE = lo_sunNEE(pix, pos, normal, geoNormal, albedo, rough, metal, specular, clearcoat, wo, isGlass);
  // T5 — stained-glass-specific terms now live in stainedGlassShade.wgsl.ts
  // (lo_sg_caustic / lo_sg_aperture); each early-returns vec3f(0) unless its
  // ubo.stainedGlassFlags bit is set (default OFF — generic scenes get zero
  // caustic/aperture). Same call args + same summation into directRadiance.
  let Lo_sunCaustic = lo_sg_caustic(pix, pos, normal, albedo, isGlass, isMetal);
  let Lo_skyAperture = lo_sg_aperture(pos, normal, albedo, isGlass, isMetal);
  let Lo_indirect   = lo_indirect(pix, dims, pos, normal, isGlass, isMetal);
  // B1 tail (2026-06-10) — glass refracted GI: consumption of the post-glass
  // diffuse reservoir built by risGi's 1-interface refraction walk. Returns vec3f(0)
  // for non-glass surfaces (isGlass gate). Weighted by Fresnel transmittance +
  // Beer-Lambert tint. Joins the DIRECT channel (see directRadiance below) —
  // it is not albedo-demodulated because its tint is the glass transmittance,
  // not the diffuse wall's baseColor (which was already folded into Lo by risGi).
  let Lo_transmittedGI = lo_transmittedGI(pix, dims, pos, normal, wo, matColor, isGlass, primaryHit.indices.w);
  // B1 — glossy/metal specular indirect: GGX specular lobe × the SAME ReSTIR-GI
  // reservoir sample. UN-demodulated (joins the direct channel below); fires only
  // for metal/glossy surfaces (zero on default-diffuse → invariant preserved).
  let Lo_indirectSpec = lo_indirectSpecular(pix, dims, pos, normal, wo, albedo, rough, metal, specular, clearcoat, isGlass);

  // Active terms (current pipeline state):
  //   Lo_emit           glass primary hit, deterministic per pixel
  //   Lo_direct         ReSTIR DI, atrous-denoised single sample
  //   Lo_sunNEE         direct sun NEE, deterministic per pixel (item 4, 2026-06-10, default-ON)
  //   Lo_sunCaustic     sun shadow ray through glass, deterministic (stainedGlass flag only)
  //   Lo_skyAperture    5-tap sky probe through cutout, scalar luminance (stainedGlass flag only)
  //   Lo_indirect       ReSTIR-GI half-res reservoir read (Sprint 16), per-channel split (Sprint 18)
  //   Lo_transmittedGI  glass primary hit — refracted-GI reservoir × Fresnel-T × Beer tint (B1 tail, 2026-06-10)
  //
  // Sprint 15 — GTAO modulates ALL non-emissive lighting terms.
  // - Lo_emit is the light source itself; never darken it.
  // - Direct, sun, sky-aperture, indirect all darken in concave contact
  //   regions per Jiménez 2016. Sky-miss pixels (centerDepth=0) were
  //   written 1.0 in gtao.wgsl so they pass through unmodified.
  // - The per-channel safe-fallback below replaces any pathological channel
  //   (NaN, negative, ~0 due to first-frame uninitialised sample) with 1.0
  //   so a corrupt sample never blanks the pixel. The texture is seeded
  //   with vec3(1.0) at engine init; this is defence-in-depth.
  //
  // Tier-G fix (Jiménez 2016 §5.2 per-channel multi-bounce): previously
  // shade read only the .r channel — equivalent to Bavoil-style scalar
  // AO. The upsample now writes the full per-channel multi-bounce vec3
  // into .rgb, so shade darkens each colour channel by its own factor.
  let aoRaw = textureLoad(aoFullTexture, vec2i(pix), 0).rgb;
  let aoClamped = clamp(aoRaw, vec3f(0.0), vec3f(1.0));
  let ao = vec3f(
    select(1.0, aoClamped.r, aoRaw.r > 0.001),
    select(1.0, aoClamped.g, aoRaw.g > 0.001),
    select(1.0, aoClamped.b, aoRaw.b > 0.001),
  ) * authoredAo;

  // Sprint 18 — split the radiance into a direct channel (heads to the
  // tight-sigma atrous-variance chain) and an indirect channel (heads to
  // the broader bilateral blur). The downstream combine pass sums them.
  //
  // Direct = emit (light source itself) + direct shadow-mapped terms (× AO).
  // Indirect = ReSTIR-GI Lo_indirect (× AO). Lo_emit bypasses AO because
  // the light source itself is never in self-shadow.
  //
  // Lo_skyAperture audit follow-up: the upstream computation already returns
  // skyVisScalar * skyTint * skyIrradiance * albedo * INV_PI, which is the
  // correct outgoing radiance for a Lambertian receiver under a partially-
  // visible sky dome.  Earlier code multiplied by 0.08 as an empirical trim
  // calibrated to Cornell's stained-glass ambient — that double-scaled the
  // physically-parameterised sky channel and was not portable to scenes with
  // a different skyIrradiance.  The 0.08 has been dropped; Lo_skyAperture
  // is summed at full magnitude.
  // Lo_emitterGlow (self-emission) and Lo_lightMap (baked outgoing radiance)
  // join Lo_emit OUTSIDE the AO term — emission/baked lighting already carry
  // their own visibility and should not be contact-darkened again.
  // H41 — Lo_analyticNEE is in the direct channel (same firefly-clamp tier as Lo_direct).
  // B1 — Lo_indirectSpec (glossy/metal specular reflection of GI) joins the
  // UN-demodulated direct channel (it is not albedo-proportional, so it must
  // bypass indirectCombine's albedo re-modulation). It is NOT AO-modulated:
  // GTAO is a diffuse-occlusion term; specular reflections are not darkened by
  // it. Zero for default-diffuse surfaces, so the diffuse-default invariant
  // holds byte-for-byte (the term is identically vec3f(0) there).
  // Lo_transmittedGI joins the direct channel (un-demodulated; bypasses AO —
  // the diffuse wall behind the glass is not in contact-shadow from the glass
  // pane, and GI through glass is a transmission term, not an occlusion term).
  let directRadiance = Lo_emit + Lo_emitterGlow + Lo_lightMap + Lo_indirectSpec + Lo_transmittedGI + (Lo_direct + Lo_analyticNEE + Lo_sunNEE + Lo_sunCaustic + Lo_skyAperture) * ao;
  let indirectRadiance = Lo_indirect * ao;

  // Firefly clamp — ReSTIR-DI + glancing-angle BRDF evaluations occasionally
  // produce singular radiance values (cos(θ_v) → 0 at the grazing edge of
  // a wall, near-zero RIS pdf). These propagate through the atrous-variance
  // denoiser (which would smear them spatially) and the temporal accumulator
  // (slow to bleed off).
  // Cap per-channel: physical max for an albedo-1 diffuse surface viewing
  // Le=12 ≈ 4/π × 12 ≈ 15. We clamp at 4 to suppress the grazing-edge
  // singularities (~2.8 measured at the red-wall edge stripe) while leaving
  // legitimately bright surfaces (light source itself: Lo_emit) intact —
  // those go through a separate Lo_emit branch that bypasses the BRDF
  // singularity entirely.
  // Audit B4: clamp is now UBO-driven. Default 4.0 preserves Cornell behaviour
  // (calibrated for Le=12, albedo=1, → 4 * INV_PI * Le ≈ 15 → cap at 4);
  // hosts with brighter emitters should compute ~4 * luminance(maxEmitterLe).
  let clampedDirect = min(directRadiance, vec3f(ubo.directFireflyClamp));
  // Indirect clamp is *much* tighter than direct: the atrous chain's
  // chromaticity-based color edge-stop preserves bright fireflies (center
  // bright + neighbour dark = large color delta → neighbours' smoothing
  // contribution to the bright pixel is suppressed; the spike persists).
  // Combined with ReSTIR's residual W tail (each W is capped at 4 but
  // Lo at the reconnection vertex can still be high), spikes pass through
  // atrous unchanged and admit a multi-percent jolt into the temporal
  // accumulator even at α=0.01, manifesting as a "dancing" residual noise
  // pattern.  The Cornell-tuned cap (1.0,1.0,1.0) is well above Cornell's
  // plausible converged indirect brightness (~0.3 worst case), generous
  // head-room for legitimate color-bleed peaks, but kills the firefly tail.
  // Library consumers override via HybridEngineOptions.indirectFireflyClamp
  // (per-channel vec3 so a tinted scene can clamp each channel independently).
  let clampedIndirect = min(indirectRadiance, ubo.indirectFireflyClamp);
  // Write LINEAR HDR radiance to hdrColorOut — do NOT tone-map here.
  // Tone mapping must happen AFTER the à-trous denoiser so that the denoiser
  // operates in linear HDR space. The composite pass applies ACES filmic + sRGB.
  //
  // Item 24 — albedo demodulation (Schied 2017 §4.1):
  //   hdrIndirectOut carries the albedo-demodulated lighting signal
  //   (L / albedo). indirectCombine re-applies albedo after denoising.
  //   hdrAlbedoOut carries the visible-point diffuse albedo for that re-modulation.
  //   hdrTotalOut represents the full radiance (direct + re-modulated indirect)
  //   so the Welford variance estimate reflects the actual signal energy.
  textureStore(hdrColorOut,    pix, vec4f(clampedDirect,                          1.0));
  textureStore(hdrIndirectOut, pix, vec4f(clampedIndirect,                        1.0));
  textureStore(hdrAlbedoOut,   pix, vec4f(albedo,                                 1.0));
  // Total = direct + indirect-with-albedo-restored; used only by Welford.
  textureStore(hdrTotalOut,    pix, vec4f(clampedDirect + clampedIndirect * albedo, 1.0));
}
`;

/** W1-R6 — declarative include-graph entry.
 *  Order mirrors the historical concat `COMMON_WGSL + SURFACE_TEXTURES_WGSL +
 *  DDGI_SAMPLE_WGSL + SHADE_WGSL` — surfaceTextures requires common, so the
 *  composer emits {common, surfaceTextures, ddgiSample, ...} which is
 *  byte-equivalent to that pre-R6 string.
 *
 *  T5 — `stainedGlassShade` (lo_sg_caustic / lo_sg_aperture) is appended after
 *  `sampleCascadeC0`. It requires only `common` (already emitted by the time
 *  the composer reaches it), so it contributes exactly STAINED_GLASS_SHADE_WGSL
 *  immediately before SHADE_WGSL. */
export const SHADE_MODULE: WgslModule = {
  name: 'shade',
  source: SHADE_WGSL,
  // D5.1+D5.2: ddgiSample replaced by ddgiGridUbo (which requires ddgiSample
  // transitively, and adds DDGIGridUBO struct + @group(3) @binding(3) + sampleDDGIAtPoint).
  requires: ['common', 'surfaceTextures', 'materialAtlas', 'ddgiGridUbo', 'sampleCascadeC0', 'stainedGlassShade', 'environmentSample'],
};
