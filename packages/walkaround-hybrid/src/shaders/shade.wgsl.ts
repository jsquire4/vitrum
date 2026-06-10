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
 */

// Atlas-layout constants are consumed by ddgiSampleWgsl.ts (the canonical
// DDGI atlas sampler); shade.wgsl delegates via ddgiSampleFromBindings —
// no direct constant references needed here.
import type { WgslModule } from '../pipeline/wgslComposer.js';

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
@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;
// Camera-visible emitters (2026-05-30) — per-triangle HDR emissive radiance Le
// (rgba32float texture). Read by lo_emitterGlow on a primary hit so emissive-mesh
// surfaces glow to the camera (the ReSTIR-DI emitter list only lights RECEIVERS;
// without this the emitter's own pixels render black). Shade-only binding.
@group(1) @binding(12) var bvh_emissive: texture_2d<f32>;
// H41 — analytic point/spot emitter buffer for shade NEE.
// Separate from the RIS area-emitter pool (no PDF contamination).
// Stride: 4 × vec4f = 64 bytes = 16 vec4f elements per entry.
//   element[base+0]  = position.xyz   + pad
//   element[base+4]  = color.rgb (Le = color×intensity) + pad
//   element[base+8]  = direction.xyz  + cosInner (1.0 for point)
//   element[base+12] = cosOuter + pad×3 (0.0 for point = no cone)
// arrayLength(&analytic_lights) / 16u gives the number of analytic lights.
@group(1) @binding(13) var<storage, read> analytic_lights: array<vec4f>;
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
@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;
@group(3) @binding(2) var ddgiSampler:    sampler;
struct DDGIGridUBO {
  origin:    vec3f,
  spacing:   f32,
  dimsX:     u32,
  dimsY:     u32,
  dimsZ:     u32,
  _pad0:     u32,
  irrW:      f32,
  irrH:      f32,
  visW:      f32,
  visH:      f32,
};
@group(3) @binding(3) var<uniform> ddgiGrid: DDGIGridUBO;

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

// invertMat4_common + generatePrimaryRay_common live in common.wgsl;
// injected via common.wgsl (W1-R6 wgslComposer requires chain).

// ──────────────────────────────────────────────────────────────────────────
// Per-lighting-term helpers (W4-A5).
//
// Each fn computes one Lo_* term in isolation. The helpers read module-scope
// state (ubo, bvh, emitters, spatialReservoir, reservoirGiCurrent, bvh_beer)
// directly; only the per-pixel locals are passed as params. Gating (isGlass /
// isMetal) lives inside each helper so shadeMain stays a flat composition.
// ──────────────────────────────────────────────────────────────────────────

// ── Self-emission for primary glass hits ─────────────────────────────────
//
// Le ≈ attenuationColor × transmission × sunIntensity × |sunDot| × textureMod.
// attenuationColor is read from bvh_beer (Beer-Lambert visible color =
// pow(rawAttCol, thickness/attDist)) — separate from bvhIndex.w which
// carries the RAW attCol used by emitter Le and tinted-visibility.
fn lo_emit(
  matColor:         vec4f,
  normal:           vec3f,
  isGlass:          bool,
  uv:               vec2f,
  matColorPacked:   u32,
  triIndex:         u32,
) -> vec3f {
  if (!isGlass) { return vec3f(0.0); }
  let sunDot = abs(dot(ubo.sunDirection, normal));
  if (sunDot <= 0.05) { return vec3f(0.0); }
  let trans = matColor.a;
  let texId = decodeSurfaceTextureId(matColorPacked);
  let texMod = surfaceTextureMod(uv, texId);
  // WS1 — beer is a texture now: map the triangle index to a texel.
  let beerCoord = vec2u(triIndex % BVH_BEER_TEX_WIDTH, triIndex / BVH_BEER_TEX_WIDTH);
  let beerPacked = textureLoad(bvh_beer, vec2i(beerCoord), 0).r;
  let beerAlbedo = vec3f(
    f32((beerPacked >> 24u) & 0xFFu) / 255.0,
    f32((beerPacked >> 16u) & 0xFFu) / 255.0,
    f32((beerPacked >>  8u) & 0xFFu) / 255.0,
  );
  return beerAlbedo * trans * ubo.sunIntensity * sunDot * texMod;
}

// --- Camera-visible emitters: self-emission glow on a primary hit -----------
//
// Emissive-mesh surfaces are NEE-only in walkaround (the ReSTIR-DI emitter list
// lights RECEIVERS); their own pixels render black to the camera. This returns
// the hit triangle's HDR emissive radiance Le from the per-triangle bvh_emissive
// texture so the emitter glows directly. The texel layout matches bvh_beer
// (BVH_BEER_TEX_WIDTH width; both per-triangle textures share it).
//
// No double-count: this is the emitter's OWN pixel, a different surface point
// than the receivers ReSTIR-DI shades; and lo_emit (glass Beer-Lambert) is the
// transmissive case, while packBVHEmissiveLe packs the emissive branch ONLY.
fn lo_emitterGlow(triIndex: u32) -> vec3f {
  let coord = vec2u(triIndex % BVH_BEER_TEX_WIDTH, triIndex / BVH_BEER_TEX_WIDTH);
  return textureLoad(bvh_emissive, vec2i(coord), 0).rgb;
}

// --- H41: Analytic point/spot NEE (ADDITIVE, separate from RIS area-emitter pool) ---
//
// Iterates the analytic_lights buffer for point and spot emitters. This term
// is SEPARATE from lo_direct (which samples the ReSTIR area-emitter reservoir).
// Adding it additively prevents PDF contamination — the two estimators address
// disjoint light sources (area emitters vs analytic point/spot), so their
// contributions sum without double-counting.
//
// Light model: inverse-square falloff with ε denominator floor to avoid
// division-by-zero at the light position. Spot falloff: smoothstep from
// cosOuter to cosInner. Point: cosInner=1, cosOuter=0 → smoothstep(0,1,x) = x²(3-2x) is not 1 at x≥1, so use cosOuter < cosInner always.
//
// Shadow: deterministic shadow ray (not stochastic — no variance per pixel,
// no need for reservoir denoising). skipGlass=true (same as lo_direct).
//
// Glass/metal: skip (same policy as lo_direct — their Lo_emit drives).
fn lo_analyticNEE(
  pos:      vec3f,
  normal:   vec3f,
  geoNormal: vec3f,
  albedo:   vec3f,
  rough:    f32,
  metal:    f32,
  wo:       vec3f,
  isGlass:  bool,
  isMetal:  bool,
) -> vec3f {
  // B1 — metals now receive DIRECT light (the isMetal early-out is removed):
  // evalGGX evaluates the GGX specular lobe with the real conductor F0
  // (= baseColor when metal). Glass still skips here (its Lo_emit / refracted
  // path drives it; refracted analytic NEE is out of scope this pass).
  if (isGlass) { return vec3f(0.0); }
  // V28 H41-spot FIX (2026-06-09): the struct is 4×vec4f per entry (packer
  // ANALYTIC_LIGHT_STRIDE_FLOATS=16 floats = 4 vec4s). arrayLength on an
  // array<vec4f> returns the VEC4 count, so entry count = arrayLength/4 (was /16,
  // which read float-stride as vec4-stride → count=0 for 1-3 lights → the analytic
  // DIRECT NEE returned nothing; point scenes only lit via DDGI-indirect, spot
  // scenes stayed dark). Per-entry vec4 layout: [base+0]=pos, [base+1]=color×Le,
  // [base+2]=dir.xyz+cosInner, [base+3]=cosOuter.
  let count = arrayLength(&analytic_lights) / 4u;
  if (count == 0u) { return vec3f(0.0); }
  var Lo = vec3f(0.0);
  for (var li = 0u; li < count; li++) {
    let base = li * 4u;
    let lightPos  = analytic_lights[base + 0u].xyz;
    let lightLe   = analytic_lights[base + 1u].xyz;      // pre-multiplied color×intensity
    let lightDir  = analytic_lights[base + 2u].xyz;      // toward-light direction (spot axis); (0,0,0) = point
    let cosInner  = analytic_lights[base + 2u].w;        // 1.0 for point
    let cosOuter  = analytic_lights[base + 3u].x;        // 0.0 for point

    let toL  = lightPos - pos;
    let dist = length(toL);
    if (dist < 1e-4) { continue; }
    let wi   = toL / dist;
    let nDotL = dot(normal, wi);
    if (nDotL <= 0.0) { continue; }

    // Spot cone attenuation. For a point (cosInner=1, cosOuter=0):
    //   cosTheta = dot(-lightDir, wi) but lightDir=(0,0,0) so we skip cone.
    //   cone = 1.0 for points (omnidirectional).
    var cone = 1.0;
    let hasSpot = dot(lightDir, lightDir) > 0.01;
    if (hasSpot) {
      let cosTheta = dot(-lightDir, wi);
      if (cosTheta <= cosOuter) { continue; }
      cone = smoothstep(cosOuter, cosInner, cosTheta);
    }

    // Shadow ray — same pattern as lo_direct (offset along geo normal, skipGlass=true).
    let occ = traceSceneAny(
      ubo.bvhMode, ubo.tlasNodeCount,
      &bvh_index, &bvh_position, &bvh,
      &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
      &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
      pos + geoNormal * 1e-3, wi, dist - 2e-3, ubo.triIntersectEpsilon, true);
    if (occ) { continue; }

    // Inverse-square falloff: Le / (d² + ε) · cosθ · cone · brdf
    let invDist2 = 1.0 / (dist * dist + ubo.emitterDist2Floor);
    let brdf = evalGGX(albedo, rough, metal, normal, wo, wi);
    Lo += lightLe * brdf * nDotL * cone * invDist2;
  }
  return Lo;
}

// --- Direct lighting (ReSTIR DI) ---
//
// Gated to !isGlass — on a near-mirror glass primary hit (rough=0.05) the GGX BRDF
// sample of a NEIGHBOURING emitter cell pulls in that cell's color and
// mixes it into the cell being shaded — chromatic pollution that washes
// saturated authored colors toward pastel.
// Came / solder (isMetal) skip Lo_direct: ReSTIR DI's single-sample
// variance produces high-amplitude firefly speckle on thin metallic
// strips that atrous can't smooth.
fn lo_direct(
  pixelIdx: u32,
  pos:      vec3f,
  normal:   vec3f,
  geoNormal: vec3f,
  wo:       vec3f,
  albedo:   vec3f,
  rough:    f32,
  metal:    f32,
  isGlass:  bool,
  isMetal:  bool,
  rng:      ptr<function, u32>,
) -> vec3f {
  // B1 — metals now receive DIRECT light via the GGX specular lobe (real
  // conductor F0 from baseColor). Only glass skips ReSTIR-DI here (the
  // near-mirror chromatic-pollution rationale below); refracted DI is out of
  // scope. The prior firefly-speckle rationale for skipping metals is addressed
  // by the existing direct firefly clamp + the (now physically-tight) GGX lobe.
  if (isGlass) { return vec3f(0.0); }
  let r = loadSpatialDI(pixelIdx);
  if (r.W <= 0.0 || r.M == 0u) { return vec3f(0.0); }
  let lid = r.lightId;

  // Wave 4 — ENV_SAMPLE_SENTINEL: the reservoir's winning candidate was an HDRI
  // importance-sampled direction. Decode xi → world direction and shade with the
  // env radiance × BRDF × W (no shadow ray — visibility was already tested in RIS;
  // the W already bakes in occlusion via the w_sum=0 zero-out on occlusion, matching
  // the existing emitter DI pattern where visibility is resolved in ris.wgsl before
  // storing the reservoir).
  if (lid == ENV_SAMPLE_SENTINEL) {
    if (!envHasMap()) { return vec3f(0.0); }
    let envDir = envDirFromXi(r.xi);
    let nDotL = max(0.0, dot(normal, envDir));
    if (nDotL < 1e-6) { return vec3f(0.0); }
    let envColor = envRadiance(envDir);
    let brdfE = evalGGX(albedo, rough, metal, normal, wo, envDir);
    return envColor * brdfE * r.W;
  }

  if (lid >= ubo.emitterCount) { return vec3f(0.0); }
  let e  = emitters[lid];
  // Stochastic xi instead of (0.5, 0.5). The deterministic centre-sample
  // bites hard on rect-area lights split into two triangles: the two
  // tris have different centroids, so ReSTIR flipping between them
  // produces a bimodal radiance per frame (visible flicker). Random xi
  // distributes the sample point across the triangle each frame;
  // temporalAccum integrates the variance out.
  let lsXi = vec2f(rand_f32(rng), rand_f32(rng));
  let ls = sampleEmitterPoint(e, lsXi);
  let toL = ls.pos - pos;
  let dist = length(toL);
  if (dist <= 1e-4) { return vec3f(0.0); }
  let wi    = toL / dist;
  let nDotL = max(0.0, dot(normal, wi));
  let nlDotL = max(0.0, dot(-e.normal, wi));
  if (nDotL <= 1e-6 || nlDotL <= 1e-6) { return vec3f(0.0); }
  // skipGlass=true: matches pre-canonical ReSTIR shadow-ray glass filter
  // (light passes through glass; per-channel tinted-visibility handles tint).
  // WS1 — offset the shadow-ray origin along the GEOMETRIC normal (the smooth
  // shading normal can dip below the surface near silhouettes → self-hit).
  let occ = traceSceneAny(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    pos + geoNormal * 1e-3, wi, dist - 2e-3, ubo.triIntersectEpsilon, true);
  if (occ) { return vec3f(0.0); }
  let G    = emitterGeometry(nlDotL, dist * dist, ubo.emitterDist2Floor);
  let brdf = evalGGX(albedo, rough, metal, normal, wo, wi);
  return e.Le * brdf * G * r.W;
}

// T5 — the sun-caustic + sky-aperture stained-glass-specific lighting terms
// were extracted into stainedGlassShade.wgsl.ts (lo_sg_caustic /
// lo_sg_aperture), opt-in behind ubo.stainedGlassFlags. shade no longer
// carries stained-glass knowledge; it just calls the two helpers below in
// the per-term composition. SHADE_MODULE.requires lists stainedGlassShade
// so the composer emits those bodies ahead of SHADE_WGSL.

// --- Indirect lighting (Sprint 16 — ReSTIR-GI one-bounce resampling) ---
//
// The trilinear DDGI atlas read had visible cell-grid splotches on
// smooth walls (structural single-bounce limitation). ReSTIR-GI runs
// a half-res RIS pass that picks ONE probe-direction sample per pixel
// by importance, then resamples spatially+temporally (Sprints 17-18).
// The result is per-pixel screen-space — the probe grid stops being
// the per-pixel basis, so cell artefacts go away.
//
// Lo_indirect_lighting = Lo * W * cos(N, wi) * INV_PI  (albedo-demodulated)
//   - Lo, W from the GI reservoir (half-res; bilinear-blend across 4 cells)
//   - cos × INV_PI is the receiver Lambertian BRDF response
//   - albedo is intentionally OMITTED here (Item 24 — Schied 2017 §4.1
//     albedo demodulation). The à-trous chain filters the pure lighting
//     signal; indirectCombine re-multiplies by albedo after denoising.
// Gating: glass/metal surfaces skip this (their Lo_emit drives).
//         The reservoir was empty-stored by risGi in those cases.
//
// Sprint 18 follow-up — bilinear blend across 4 surrounding half-res
// reservoirs.  The original nearest-neighbour read halfPx = gid/2u made
// every 2x2 full-res quad share one chosen sample; adjacent quads picked
// different random samples, so the indirect signal had a sharp 2-pixel
// discontinuity at every quad boundary.  risGi re-rolls samples each frame,
// so the discontinuity pattern shifted every frame and the temporal
// accumulator could not converge to a fixed point.  Blending 4 neighbours
// with bilinear weights at half-res fractional coord (gid*0.5) eliminates
// the quad grid.
fn lo_indirect(
  gid:     vec2u,
  dims:    vec2u,
  pos:     vec3f,
  normal:  vec3f,
  isGlass: bool,
  isMetal: bool,
) -> vec3f {
  if (isGlass || isMetal) { return vec3f(0.0); }
  var Lo_indirect = vec3f(0.0);
  let halfDims = dims / 2u;
  let halfPxF = vec2f(gid) * 0.5;
  let hx0 = u32(floor(halfPxF.x));
  let hy0 = u32(floor(halfPxF.y));
  let fx = halfPxF.x - f32(hx0);
  let fy = halfPxF.y - f32(hy0);
  let bw00 = (1.0 - fx) * (1.0 - fy);
  let bw10 =        fx  * (1.0 - fy);
  let bw01 = (1.0 - fx) *        fy;
  let bw11 =        fx  *        fy;
  var totalW: f32 = 0.0;
  // Confidence accumulator — bilinear-weighted ReSTIR-GI sample count over the
  // same 4 half-res reservoirs that build Lo_indirect. The reservoir M is the
  // effective number of resampled candidates the temporal+spatial passes have
  // integrated into this pixel (Bitterli 2020 / Ouyang 2021 ReSTIR-GI). A
  // higher M ⇒ lower-variance, more-trustworthy estimate. We weight each
  // reservoir's M by its bilinear contribution so the per-pixel confidence
  // matches the radiance blend exactly.
  var Maccum: f32 = 0.0;
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    var hx = hx0;
    var hy = hy0;
    var bw: f32 = 0.0;
    if      (k == 0u) { hx = hx0;          hy = hy0;          bw = bw00; }
    else if (k == 1u) { hx = hx0 + 1u;     hy = hy0;          bw = bw10; }
    else if (k == 2u) { hx = hx0;          hy = hy0 + 1u;     bw = bw01; }
    else              { hx = hx0 + 1u;     hy = hy0 + 1u;     bw = bw11; }
    if (hx >= halfDims.x) { hx = halfDims.x - 1u; }
    if (hy >= halfDims.y) { hy = halfDims.y - 1u; }
    if (bw < 1e-5) { continue; }
    let giIdx = hy * halfDims.x + hx;
    let g = loadReservoirGI_rw(&reservoirGiCurrent, giIdx);
    if (g.W <= 0.0 || g.M == 0u) { continue; }
    let toS = g.xs - pos;
    let distS = length(toS);
    if (distS <= 1e-4) { continue; }
    let wi = toS / distS;
    let cosTheta = max(0.0, dot(normal, wi));
    // Item 24: omit albedo here; indirectCombine applies it post-denoising.
    Lo_indirect = Lo_indirect + g.Lo * INV_PI * cosTheta * g.W * bw;
    Maccum = Maccum + f32(g.M) * bw;
    totalW = totalW + bw;
  }
  // Effective per-pixel ReSTIR-GI sample count (bilinear-averaged M). 0 when no
  // valid reservoir contributed — that pixel has *no* ReSTIR estimate, so the
  // confidence-ratio below hands full weight to RC (or, if RC is also off,
  // returns 0).
  var Meff: f32 = 0.0;
  if (totalW > 1e-3) {
    Lo_indirect = Lo_indirect / totalW;
    Meff = Maccum / totalW;
  }
  // W8 Phase 3 — confidence-ratio (balance-heuristic) composition with the
  // Sannikov 2023 Radiance Cascades cascade-0 estimate. Both estimators
  // integrate the SAME diffuse-indirect radiance, so any convex blend
  // (w_restir + w_rc == 1) is unbiased; we choose the blend per-pixel by each
  // estimator's reliability instead of a single host scalar.
  //
  // Confidence proxies (both ∈ [0,1]):
  //   c_restir = m            — ReSTIR-GI's normalised effective sample count
  //                             m = clamp(Meff / restirGiMClamp, 0, 1). The
  //                             temporal M-clamp is the host's "fully
  //                             converged" reference; ReSTIR variance falls
  //                             ~1/M, so m is a monotone reliability proxy
  //                             (NOT W — a high W usually means a low p̂ /
  //                             rare-sample spike, i.e. *less* reliable).
  //   c_rc = rcWeight·(1 - m) — RC is a low-variance but biased deterministic
  //                             probe integrator with no per-pixel sample
  //                             count, so its reliability is a fixed host
  //                             PRIOR (rcWeight) gated by how *unreliable*
  //                             ReSTIR is here (1 - m). When ReSTIR is well
  //                             converged (m→1) RC's weight fades out; on a
  //                             fresh disocclusion (m→0) RC's stable estimate
  //                             fills in. rcWeight stays the global RC trust
  //                             knob and the disabled-path off-switch.
  //
  // Balance heuristic: w_rc = c_rc / (c_rc + c_restir), w_restir = 1 - w_rc.
  // Degenerate guard: when neither estimator is confident (c_rc + c_restir ≈ 0,
  // i.e. no valid reservoir AND rcWeight 0) we force w_restir = 1 — Lo_indirect
  // is 0 there anyway, so the pixel stays 0.
  //
  // rc-disabled bit-identity: the host binds an all-zero rcParams placeholder
  // when RC is off (DDGIBindingState.setRCInputs(null)), so rcParams.enabled==0
  // ⇒ sampleCascadeC0 returns exactly vec3f(0), AND rcWeight==0.0 ⇒ c_rc==0 ⇒
  // w_rc==0, w_restir==1.0 exactly ⇒ result == Lo_indirect, byte-for-byte
  // identical to the pre-Phase-3 path.
  let Lo_rc = sampleCascadeC0(pos, normal);
  let m = clamp(Meff / f32(max(ubo.restirGiMClamp, 1u)), 0.0, 1.0);
  let cRestir = m;
  // RC-has-energy gate (algorithm-combination-fitness fix, 2026-06-07):
  // the confidence MIS hands RC weight rcWeight*(1-m) purely from the host
  // toggle, with no check that RC's cascades actually carry radiance. When RC
  // is disabled (rcWeight=0 ⇒ Lo_rc=0), or a cascade is empty for a given
  // pixel, a non-zero weight would REPLACE the (correct) ReSTIR-GI estimate
  // with RC's zero → black indirect. Gating cRc on the already-computed Lo_rc
  // closes that: an empty cascade forces cRc=0 ⇒ wRestirGi=1 ⇒ ReSTIR-GI keeps
  // full weight. (RC's probe cast DOES sample sun + emissive geometry + env +
  // rect-area emitter NEE since 1e893fa — see walkaround-rc probeRayCast; the
  // gate guards RC-off / empty-cascade pixels, NOT a light-model mismatch.)
  // Bit-identity preserved both ways — RC off ⇒ rcWeight=0 AND Lo_rc=0, RC
  // on-with-energy ⇒ the gate is 1.0 and the blend is unchanged.
  let rcHasEnergy = max(Lo_rc.r, max(Lo_rc.g, Lo_rc.b)) > 1e-6;
  let cRc = clamp(rcParams.rcWeight, 0.0, 1.0) * (1.0 - m) * select(0.0, 1.0, rcHasEnergy);
  let cSum = cRestir + cRc;
  // max() in the denominator keeps the (always-evaluated) select arm finite —
  // no inf/NaN to leak even though select discards it when cSum ≈ 0.
  let wRc = select(0.0, cRc / max(cSum, 1e-6), cSum > 1e-6);
  let wRestirGi = 1.0 - wRc;
  return wRestirGi * Lo_indirect + wRc * Lo_rc;
}

// --- B1: Glossy/metal SPECULAR indirect (ReSTIR-GI sample × GGX specular lobe) -
//
// The ReSTIR-GI reservoir is a DIFFUSE-irradiance cache: its candidates are
// cosine-hemisphere sampled and its target p̂ = luminance(Lo)·cosθ·INV_PI is the
// Lambertian receiver response (unchanged by B1 — preserving GRIS reuse
// correctness + the diffuse-default invariant). lo_indirect consumes that
// demodulated-diffuse channel.
//
// For glossy/metal receivers the diffuse lobe is absent (metal) or minor
// (glossy); their indirect is a SPECULAR reflection of the same reconnection
// radiance Lo arriving from xs. This term evaluates the GGX specular lobe of the
// receiver material against that stored sample (wi = normalize(xs − pos)) and
// returns it as UN-demodulated radiance — so it joins the DIRECT channel (it is
// NOT proportional to the diffuse albedo, so it must bypass indirectCombine's
// albedo re-modulation; metals carry their reflectance tint in the specular F0).
//
// Consistency: this reuses the SAME reservoir sample (xs, Lo, W) that the
// diffuse target selected — a deterministic BSDF re-weighting of a chosen
// sample (the same pattern lo_emit / lo_emitterGlow use), NOT a second
// estimator, so there is no p̂/consumption mismatch. Single nearest GI reservoir
// (no bilinear blend) — the specular lobe is higher-frequency, so the half-res
// blur of the diffuse path is undesirable here.
//
// Gate: fires only when metal > 0 OR roughness < SPEC_GI_ROUGH_MAX, so a
// default-diffuse scene (rough 0.85, metal 0) gets EXACTLY zero specular
// indirect — preserving the diffuse-default invariant byte-for-byte.
const SPEC_GI_ROUGH_MAX: f32 = 0.6;
fn lo_indirectSpecular(
  gid:    vec2u,
  dims:   vec2u,
  pos:    vec3f,
  normal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  isGlass: bool,
) -> vec3f {
  if (isGlass) { return vec3f(0.0); }
  if (metal <= 0.0 && rough >= SPEC_GI_ROUGH_MAX) { return vec3f(0.0); }
  let halfDims = dims / 2u;
  let hx = min(gid.x / 2u, halfDims.x - 1u);
  let hy = min(gid.y / 2u, halfDims.y - 1u);
  let giIdx = hy * halfDims.x + hx;
  let g = loadReservoirGI_rw(&reservoirGiCurrent, giIdx);
  if (g.W <= 0.0 || g.M == 0u) { return vec3f(0.0); }
  let toS = g.xs - pos;
  let distS = length(toS);
  if (distS <= 1e-4) { return vec3f(0.0); }
  let wi = toS / distS;
  // evalGGXSpecularOnly already includes the NdotL cosine + conductor F0.
  let specBrdf = evalGGXSpecularOnly(albedo, rough, metal, normal, wo, wi);
  return g.Lo * specBrdf * g.W;
}

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
  let primaryHit = traceSceneFirstHit(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    primaryRay, ubo.triIntersectEpsilon);

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
    textureStore(svgfObjectIdOut, pix, vec4u(0u));
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
  let normal = smoothShadingNormal(
    primaryHit, geoNormal,
    bvh_normal[primaryHit.indices.x].xyz, bvh_normal[primaryHit.indices.y].xyz, bvh_normal[primaryHit.indices.z].xyz,
    n_ok,
    tlasInstanceWorldToLocal[n_i], tlasInstanceWorldToLocal[n_i + 1u], tlasInstanceWorldToLocal[n_i + 2u],
  );
  let wo     = -primaryRay.direction;

  // Decode per-triangle material color from bvhIndex[triIdx].w (RGBA8 packed).
  let matColor = decodeMaterialColor(primaryHit.matColorPacked);
  let isGlass  = matColor.a > 0.3;  // transmission > ~76/255
  let isMetal  = decodeIsMetal(primaryHit.matColorPacked);  // came / solder

  // Write the G-buffer.  Normal encoded as (n*0.5+0.5) so the atrous shader
  // can decode with n = raw*2 - 1.  Depth = primary-hit distance along ray,
  // SIGN-FLIPPED for glass primary hits to encode the surface-type
  // discriminator the atrous denoiser uses to gate bleed across the
  // panel-wall boundary.
  let depthSigned = primaryHit.dist * select(1.0, -1.0, isGlass);
  textureStore(gNormalDepthOut, pix, vec4f(normal * 0.5 + 0.5, depthSigned));
  textureStore(svgfObjectIdOut, pix, vec4u(stableSvgfObjectId(primaryHit)));

  // Use the BVH-baked material color for ALL surfaces (glass AND room surfaces).
  let albedo   = matColor.rgb;
  // B1 — real authored roughness/metalness from the per-tri bvh_material texture
  // (was hardcoded rough=select(0.85,0.05,isGlass)/metal=0). The diffuse-default
  // invariant packs 0.85 for unspecified roughness / 0.05 for glass / metal 0,
  // so a default-diffuse scene is numerically unchanged; authored glossy/metal
  // surfaces now drive the GGX direct lobe + the specular-indirect term below.
  let rmCoord  = vec2u(primaryHit.indices.w % BVH_MATERIAL_TEX_WIDTH, primaryHit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let rm       = decodeRoughMetal(textureLoad(bvh_material, vec2i(rmCoord), 0).r);
  let rough    = rm.x;
  let metal    = rm.y;

  // ── Per-term lighting composition ────────────────────────────────────────
  //
  // Locals are named identically to their helper outputs (Lo_emit, Lo_direct,
  // Lo_sunCaustic, Lo_skyAperture, Lo_indirect) so the structural-contract
  // tests in sprint18-indirectCombine.test.ts continue to match.
  let Lo_emit       = lo_emit(matColor, normal, isGlass, primaryHit.uv, primaryHit.matColorPacked, primaryHit.indices.w);
  // Camera-visible emitters — emissive-mesh self-emission on the primary hit.
  let Lo_emitterGlow = lo_emitterGlow(primaryHit.indices.w);
  let Lo_direct     = lo_direct(pixelIdx, pos, normal, geoNormal, wo, albedo, rough, metal, isGlass, isMetal, &rng);
  // H41 — analytic point/spot NEE: additive, separate from the RIS area-emitter pool.
  // No PDF contamination: these are disjoint from the emitters[] stream.
  let Lo_analyticNEE = lo_analyticNEE(pos, normal, geoNormal, albedo, rough, metal, wo, isGlass, isMetal);
  // T5 — stained-glass-specific terms now live in stainedGlassShade.wgsl.ts
  // (lo_sg_caustic / lo_sg_aperture); each early-returns vec3f(0) unless its
  // ubo.stainedGlassFlags bit is set (default OFF — generic scenes get zero
  // caustic/aperture). Same call args + same summation into directRadiance.
  let Lo_sunCaustic = lo_sg_caustic(pix, pos, normal, albedo, isGlass, isMetal);
  let Lo_skyAperture = lo_sg_aperture(pos, normal, albedo, isGlass, isMetal);
  let Lo_indirect   = lo_indirect(pix, dims, pos, normal, isGlass, isMetal);
  // B1 — glossy/metal specular indirect: GGX specular lobe × the SAME ReSTIR-GI
  // reservoir sample. UN-demodulated (joins the direct channel below); fires only
  // for metal/glossy surfaces (zero on default-diffuse → invariant preserved).
  let Lo_indirectSpec = lo_indirectSpecular(pix, dims, pos, normal, wo, albedo, rough, metal, isGlass);

  // Active terms (current pipeline state):
  //   Lo_emit         glass primary hit, deterministic per pixel
  //   Lo_direct       ReSTIR DI, atrous-denoised single sample
  //   Lo_sunCaustic   sun shadow ray through glass, deterministic
  //   Lo_skyAperture  5-tap sky probe through cutout, scalar luminance
  //   Lo_indirect     ReSTIR-GI half-res reservoir read (Sprint 16), per-channel split (Sprint 18)
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
  );

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
  // Lo_emitterGlow (self-emission) joins Lo_emit OUTSIDE the AO term — emission
  // is not occluded by ambient occlusion.
  // H41 — Lo_analyticNEE is in the direct channel (same firefly-clamp tier as Lo_direct).
  // B1 — Lo_indirectSpec (glossy/metal specular reflection of GI) joins the
  // UN-demodulated direct channel (it is not albedo-proportional, so it must
  // bypass indirectCombine's albedo re-modulation). It is NOT AO-modulated:
  // GTAO is a diffuse-occlusion term; specular reflections are not darkened by
  // it. Zero for default-diffuse surfaces, so the diffuse-default invariant
  // holds byte-for-byte (the term is identically vec3f(0) there).
  let directRadiance = Lo_emit + Lo_emitterGlow + Lo_indirectSpec + (Lo_direct + Lo_analyticNEE + Lo_sunCaustic + Lo_skyAperture) * ao;
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
  requires: ['common', 'surfaceTextures', 'ddgiSample', 'sampleCascadeC0', 'stainedGlassShade', 'environmentSample'],
};
