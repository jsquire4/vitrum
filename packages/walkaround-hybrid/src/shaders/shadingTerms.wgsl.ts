/**
 * shadingTerms.wgsl.ts — the 8 per-lighting-term helper functions for
 * the walkaround-hybrid shade pass.
 *
 * Exported as a BARE STRING (`SHADING_TERMS_WGSL`), NOT a WgslModule.
 *
 * ── Why not a WgslModule? (D5.8b — resolved 2026-06-11) ──────────────────────
 * The lo_* helpers reference @group(N)/@binding(M) declarations and the
 * WalkaroundUBO binding that are declared in SHADE_WGSL's own body (not in any
 * required module). The wgslComposer emits required modules BEFORE the
 * consumer's own source, so a WgslModule for these helpers would be placed
 * before those declarations exist in the concatenated string → naga rejects
 * the undefined names.
 *
 * The proven alternative (the temporalGiCommon.wgsl.ts pattern) is to export
 * the text as a raw string and interpolate it into SHADE_WGSL's template
 * literal at the exact position where the inline block used to live.  The
 * final composed shader string is byte-identical to the original; only the
 * TypeScript file organisation improves.
 *
 * ── Binding / symbol contract ─────────────────────────────────────────────────
 * The includer (SHADE_WGSL) MUST have already declared, before this string is
 * interpolated:
 *
 *   Textures / buffers (group 1):
 *     bvh            : array<BVHNode>           @group(1) @binding(0)
 *     bvh_index      : array<vec4u>             @group(1) @binding(1)
 *     bvh_position   : array<vec4f>             @group(1) @binding(2)
 *     emitters       : array<EmitterTri>        @group(1) @binding(3)
 *     emitterCdf     : array<f32>               @group(1) @binding(4)
 *     bvh_beer       : texture_2d<u32>          @group(1) @binding(5)
 *     tlasNodes      : array<BVHNode>           @group(1) @binding(6)
 *     tlasInstanceIndices    : array<u32>       @group(1) @binding(7)
 *     tlasBlasRoots  : array<u32>               @group(1) @binding(8)
 *     tlasInstanceWorldToLocal : array<vec4f>   @group(1) @binding(9)
 *     tlasInstanceLocalToWorld : array<vec4f>   @group(1) @binding(10)
 *     bvh_normal     : array<vec4f>             @group(1) @binding(11)
 *     bvh_emissive   : texture_2d<f32>          @group(1) @binding(12)
 *     analytic_lights: texture_2d<f32>          @group(1) @binding(13)
 *     bvh_material   : texture_2d<u32>          @group(1) @binding(14)
 *
 *   UBO (group 2):
 *     ubo : WalkaroundUBO    @group(2) @binding(0)
 *
 *   Reservoirs (group 0):
 *     spatialReservoir      : array<u32>  @group(0) @binding(7)
 *     reservoirGiCurrent    : array<u32>  @group(0) @binding(11)
 *
 *   Structs / constants declared by required modules:
 *     BVHNode, EmitterTri, ReservoirDI, ReservoirGI, WalkaroundUBO
 *     INV_PI, ENV_SAMPLE_SENTINEL, BVH_BEER_TEX_WIDTH, BVH_MATERIAL_TEX_WIDTH
 *     SPEC_GI_ROUGH_MAX (declared in this string itself — not a pre-req)
 *
 *   Functions declared by required modules (common, surfaceTextures,
 *   ddgiGridUbo, sampleCascadeC0, stainedGlassShade, environmentSample):
 *     traceSceneAny, traceSceneAnyCastMask (SHADOW-01), traceSceneFirstHit, loadReservoirDI_rw,
 *     loadReservoirGI_rw, sampleEmitterPoint, emitterGeometry,
 *     evalGGX, evalGGXSpecularOnly, decodeSurfaceTextureId,
 *     surfaceTextureMod, decodeMaterialColor, decodeIsMetal,
 *     decodeRoughMetal, decodeIor, envHasMap, envDirFromXi, envRadiance,
 *     sampleCascadeC0, safe_normalize, rand_f32, pcgInit,
 *     smoothShadingNormal, rcParams (struct field)
 *
 *   Local helper declared in SHADE_WGSL before this interpolation:
 *     loadSpatialDI (calls loadReservoirDI_rw on spatialReservoir)
 *
 * ── D5.8b history ────────────────────────────────────────────────────────────
 * Originally the lo_* block lived inline in SHADE_WGSL and carried a D5.8b
 * deferral comment explaining why a WgslModule extraction was blocked. The
 * temporalGiCommon raw-string pattern (used by temporalGiCommon.wgsl.ts to
 * share helpers between TEMPORAL_GI_WGSL and TEMPORAL_GI_GRIS_WGSL) was the
 * proven resolution: export the text verbatim here and interpolate it with
 * `${SHADING_TERMS_WGSL}` at the same position in SHADE_WGSL's template
 * literal. Resolved 2026-06-11; byte-identical to the original composed
 * shader string.
 */
export const SHADING_TERMS_WGSL = /* wgsl */ `// ── Self-emission for primary glass hits ─────────────────────────────────
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
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  albedo:   vec3f,
  rough:    f32,
  metal:    f32,
  specular: vec4f,
  anisotropy: vec2f,
  clearcoat: vec2f,
  sheen:    vec4f,
  sheenRoughness: f32,
  wo:       vec3f,
  isGlass:  bool,
  isMetal:  bool,
) -> vec3f {
  // B1 — metals now receive DIRECT light (the isMetal early-out is removed):
  // evalGGX evaluates the GGX specular lobe with the real conductor F0
  // (= baseColor when metal). Glass still skips here (its Lo_emit / refracted
  // path drives it; refracted analytic NEE is out of scope this pass).
  if (isGlass) { return vec3f(0.0); }
  // V28/H41 analytic-light texture layout: texel 0 is a self-describing header
  // (x = light count), followed by 4×vec4f per point/spot light. The header is
  // required because the CPU side must allocate a non-empty placeholder texture
  // for zero-light scenes; deriving count from texture dimensions would treat
  // that placeholder as a real zeroed light.
  let analyticDims = textureDimensions(analytic_lights);
  let analyticHeader = textureLoad(analytic_lights, vec2i(0, 0), 0);
  let count = u32(max(analyticHeader.x, 0.0));
  if (count == 0u) { return vec3f(0.0); }
  var Lo = vec3f(0.0);
  for (var li = 0u; li < count; li++) {
    let base = 1u + li * 4u;
    let light0 = textureLoad(analytic_lights, vec2i(i32(base % analyticDims.x), i32(base / analyticDims.x)), 0);
    let light1 = textureLoad(analytic_lights, vec2i(i32((base + 1u) % analyticDims.x), i32((base + 1u) / analyticDims.x)), 0);
    let light2 = textureLoad(analytic_lights, vec2i(i32((base + 2u) % analyticDims.x), i32((base + 2u) / analyticDims.x)), 0);
    let light3 = textureLoad(analytic_lights, vec2i(i32((base + 3u) % analyticDims.x), i32((base + 3u) / analyticDims.x)), 0);
    let lightPos  = light0.xyz;
    let lightLe   = light1.xyz;
    let lightDir  = light2.xyz;
    let cosInner  = light2.w;
    let cosOuter  = light3.x;
    let castShadowDisabled = light3.y > 0.5;

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

    if (!castShadowDisabled) {
      // Shadow ray — same pattern as lo_direct (offset along geo normal, skipGlass=true).
      // SHADOW-01 — DI shadow rays skip castShadow:false geometry (bvh_material bit 0).
      let occ = traceSceneAnyCastMask(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        pos + geoNormal * 1e-3, wi, dist - 2e-3, ubo.triIntersectEpsilon, true,
        bvh_material, BVH_MATERIAL_TEX_WIDTH);
      if (occ) { continue; }
    }

    // Inverse-square falloff: Le / (d² + ε) · cosθ · cone · brdf
    let invDist2 = 1.0 / (dist * dist + ubo.emitterDist2Floor);
    let brdf = evalGGXWithSpecularClearcoatSheen(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, normal, clearcoatNormal, wo, wi);
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
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  wo:       vec3f,
  albedo:   vec3f,
  rough:    f32,
  metal:    f32,
  specular: vec4f,
  anisotropy: vec2f,
  clearcoat: vec2f,
  sheen:    vec4f,
  sheenRoughness: f32,
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
    let brdfE = evalGGXWithSpecularClearcoatSheen(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, normal, clearcoatNormal, wo, envDir);
    return envColor * brdfE * r.W;
  }

  if (lid >= ubo.emitterCount) { return vec3f(0.0); }
  let e  = emitters[lid];
  // Consume the exact sample that won the reservoir. Re-rolling a fresh point
  // here breaks the ReSTIR identity between candidate p̂, finalization p̂, and
  // shaded contribution for large close emitters.
  let ls = sampleEmitterPoint(e, r.xi);
  let toL = ls.pos - pos;
  let dist = length(toL);
  if (dist <= 1e-4) { return vec3f(0.0); }
  let wi    = toL / dist;
  let nDotL = max(0.0, dot(normal, wi));
  let nlDotL = max(0.0, dot(-e.normal, wi));
  if (nDotL <= 1e-6 || nlDotL <= 1e-6) { return vec3f(0.0); }
  if (e.castShadowDisabled < 0.5) {
    // skipGlass=true: matches pre-canonical ReSTIR shadow-ray glass filter
    // (light passes through glass; per-channel tinted-visibility handles tint).
    // WS1 — offset the shadow-ray origin along the GEOMETRIC normal (the smooth
    // shading normal can dip below the surface near silhouettes → self-hit).
    // SHADOW-01 — DI shadow rays skip castShadow:false geometry (bvh_material bit 0).
    let occ = traceSceneAnyCastMask(
      ubo.bvhMode, ubo.tlasNodeCount,
      &bvh_index, &bvh_position, &bvh,
      &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
      &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
      pos + geoNormal * 1e-3, wi, dist - 2e-3, ubo.triIntersectEpsilon, true,
      bvh_material, BVH_MATERIAL_TEX_WIDTH);
    if (occ) { return vec3f(0.0); }
  }
  let G    = emitterGeometry(nlDotL, dist * dist, ubo.emitterDist2Floor);
  let brdf = evalGGXWithSpecularClearcoatSheen(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, normal, clearcoatNormal, wo, wi);
  return e.Le * brdf * G * r.W;
}

// --- Direct sun NEE (default-on, item 4 plan/residue-closure-plan-2026-06-10.md) ----
//
// Casts one deterministic shadow ray toward the sun direction and evaluates the
// full BRDF (diffuse + GGX specular via evalGGX) when the sun lane is live
// (sunIntensity > eps). Default-ON for opaque, non-glass surfaces; glass keeps
// its existing paths (lo_emit + lo_sg_caustic when stained-glass flag set).
//
// ── No-double-count argument ───────────────────────────────────────────────────
// DDGI probes evaluate evalSunLight at the PROBE BOUNCE SURFACE (walls, floor,
// etc.), store Lo_probe = (baseColor/π) · (direct_sun_at_bounce + indirect), and
// the receiver reads that atlas to get INDIRECT irradiance = sun → wall → receiver
// (the lo_indirect term). lo_sunNEE adds sun → receiver DIRECT — a disjoint path.
//
// Concretely: every probe ray that hits a wall records the sun's contribution at
// THAT wall; the blend pass averages those records into the irradiance atlas; the
// shade pass draws from the atlas as the per-surface INDIRECT term (lo_indirect).
// lo_sunNEE below traces the ray from the PRIMARY-HIT SURFACE to the sun, which
// is a DIFFERENT surface and a DIFFERENT ray path. The two estimates do not share
// any common path segment, so they are strictly non-overlapping:
//
//   DDGI indirect: sun → bounce_wall → (probe stores E/π) → receiver reads E·albedo
//   lo_sunNEE:     sun → (shadow test along sunDir) → receiver BRDF evaluation
//
// The DDGI term carries the first-bounce INDIRECT contribution. lo_sunNEE is the
// DIRECT one-bounce term at the receiver. A double-count would only arise if the
// probe also stored "direct sun AT THE RECEIVER" in its atlas, which it cannot —
// probes are placed at grid positions, not at primary-hit surfaces.
//
// ── lo_sg_caustic relationship ────────────────────────────────────────────────
// lo_sg_caustic (SG_FLAG_SUN_CAUSTIC, stained-glass flag) also traces a ray toward
// the sun, but: (a) it only fires when the stainedGlassFlags bit is set (default
// OFF), (b) it applies a glass-tinted tinted-visibility traversal
// (bvhTraceTintedVisibility) + a causticBoost multiplier designed for stained-glass
// calibration, (c) it gates glass-or-metal (same as here). lo_sunNEE gates on
// isGlass ONLY so opaque-metal surfaces receive direct sun specular. When the
// stained-glass flag is set on a scene, both lo_sg_caustic AND lo_sunNEE can fire on
// the same opaque pixel — lo_sg_caustic adds tinted-glass-transmitted caustic light,
// lo_sunNEE adds direct (unobstructed or opaque-occluded) sun NEE. They are
// physically disjoint estimators on the same direct-sun term with different
// visibility functions: lo_sunNEE uses binary opaque occlusion (skipGlass=true so
// glass panes do NOT shadow here — the shadow test is conservative), while
// lo_sg_caustic uses the tinted multi-glass traversal. Typical stained-glass scenes
// have only glass geometry between the sun and floor/wall receivers, so the
// lo_sunNEE shadow ray through glass passes (skipGlass=true ⇒ glass is transparent
// to NEE), and lo_sg_caustic picks up the tinted coloured contribution. For generic
// opaque scenes with no stained-glass flag set, lo_sg_caustic returns vec3f(0) and
// lo_sunNEE is the sole direct-sun term.
fn lo_sunNEE(
  gid:       vec2u,
  pos:       vec3f,
  normal:    vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  albedo:    vec3f,
  rough:     f32,
  metal:     f32,
  specular:  vec4f,
  anisotropy: vec2f,
  clearcoat: vec2f,
  sheen:     vec4f,
  sheenRoughness: f32,
  wo:        vec3f,
  isGlass:   bool,
) -> vec3f {
  // Skip if sun is absent (no directional emitter or intensity essentially zero).
  if (ubo.sunIntensity < 1e-5) { return vec3f(0.0); }
  // Glass surfaces use lo_emit (Beer-Lambert self-emission) and optionally
  // lo_sg_caustic for transmitted caustics. lo_sunNEE is for opaque surfaces only.
  if (isGlass) { return vec3f(0.0); }

  // Sun direction: ubo.sunDirection is the unit vector from world origin toward
  // the sun. Apply the same soft-sun angular spread as lo_sg_caustic
  // (real sun has 0.5° angular diameter → angular radius ≈ 0.00436 rad).
  // Per-pixel deterministic: no per-frame temporal variance; same pattern as
  // lo_sg_caustic so the two terms have consistent directional sampling.
  let sunBase = ubo.sunDirection;
  let SUN_ANGULAR_RADIUS = 0.00436;
  let hx = fract(sin(f32(gid.x) * 12.9898 + f32(gid.y) * 78.233) * 43758.5453);
  let hy = fract(sin(f32(gid.x) * 93.989  + f32(gid.y) * 67.345) * 24634.6345);
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tan = safe_normalize(cross(upRef, sunBase));
  let bit = cross(sunBase, tan);
  let r2  = SUN_ANGULAR_RADIUS * sqrt(hx);
  let phi = 6.2831853 * hy;
  let toSun = safe_normalize(sunBase + tan * (r2 * cos(phi)) + bit * (r2 * sin(phi)));

  let nDotSun = dot(normal, toSun);
  if (nDotSun <= 1e-6) { return vec3f(0.0); }

  // Shadow ray — offset along geometric normal (same pattern as lo_direct / lo_analyticNEE).
  // skipGlass=true: glass panes do NOT block direct sun in this estimator —
  // lo_sg_caustic handles the tinted-glass path separately when the stained-glass
  // flag is set. This conservative transparency matches the analytic-NEE convention
  // and avoids double-counting with the tinted-visibility path in lo_sg_caustic.
  // SHADOW-01 — DI shadow rays skip castShadow:false geometry (bvh_material bit 0).
  if ((ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) == 0u) {
    let occ = traceSceneAnyCastMask(
      ubo.bvhMode, ubo.tlasNodeCount,
      &bvh_index, &bvh_position, &bvh,
      &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
      &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
      pos + geoNormal * 1e-3, toSun, 1e6, ubo.triIntersectEpsilon, true,
      bvh_material, BVH_MATERIAL_TEX_WIDTH);
    if (occ) { return vec3f(0.0); }
  }

  // Full BRDF evaluation (diffuse + GGX specular) — same pattern as lo_analyticNEE.
  // evalGGX already folds in nDotSun as its NdotL term (returns 0 when NdotL<1e-6).
  let brdf = evalGGXWithSpecularClearcoatSheen(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, normal, clearcoatNormal, wo, toSun);
  // Sun irradiance: ubo.sunIntensity is the directional emitter intensity.
  // No distance falloff — directional lights have infinite distance.
  return vec3f(ubo.sunIntensity) * brdf;
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

// --- B1 tail: Glass TRANSMITTED GI (2026-06-10) — refracted-GI reservoir consumption --
//
// When the primary hit is glass, risGi now builds a reservoir at the FIRST DIFFUSE
// surface reached by a 1-interface refraction walk (see risGi.wgsl.ts, B1 tail).
// The reservoir stores (xv=postGlassPos, nv=postGlassNormal, xs, Lo, W) — the
// same layout as the opaque-surface reservoir.
//
// Shade consumption: the stored Lo at the post-glass diffuse surface is the
// outgoing irradiance THROUGH the glass. Weight it by:
//   1. Fresnel TRANSMISSION factor at the glass interface (1 - Fresnel reflection,
//      scalar Schlick approximation at the camera-ray → glass incidence angle).
//      B1-ior-per-tri (2026-06-10): F0 is derived from the per-tri IOR (decodeIor)
//      via the physical formula F0 = ((ior−1)/(ior+1))².
//      Default IOR=1.5 → F0 = ((0.5/2.5)²) = 0.04 (preserves prior behaviour).
//   2. Beer-Lambert tint: the bvh_beer texture carries the pre-computed
//      attenuationColor^(thickness/attDist) for this glass triangle. Multiply
//      as a per-channel scale (same source as lo_emit).
//
// The result is the indirect contribution BEHIND the glass surface — it joins the
// direct channel (not the demodulated indirect channel) because its albedo is
// the glass transmittance (beerAlbedo), not the Lambertian baseColor of the receiver.
//
// Multi-interface: out of scope. A glass pixel with no valid reservoir (W=0 or M=0)
// returns vec3f(0).
fn lo_transmittedGI(
  gid:          vec2u,
  dims:         vec2u,
  pos:          vec3f,
  normal:       vec3f,
  wo:           vec3f,
  matColor:     vec4f,
  isGlass:      bool,
  triIndex:     u32,
) -> vec3f {
  if (!isGlass) { return vec3f(0.0); }
  let halfDims = dims / 2u;
  let hx = min(gid.x / 2u, halfDims.x - 1u);
  let hy = min(gid.y / 2u, halfDims.y - 1u);
  let giIdx = hy * halfDims.x + hx;
  let g = loadReservoirGI_rw(&reservoirGiCurrent, giIdx);
  if (g.W <= 0.0 || g.M == 0u) { return vec3f(0.0); }

  // Direction from the camera-primary glass hit toward the post-glass reservoir
  // vertex xv (= postGlassPos). Used only for the indirect weighting via the
  // GI reservoir weight W; no BRDF needed (the diffuse wall's BRDF was folded
  // by risGi during candidate selection).
  let toS = g.xs - g.xv;
  let distS = length(toS);
  if (distS <= 1e-4) { return vec3f(0.0); }
  let wi = toS / distS;
  let cosTheta = max(0.0, dot(g.nv, wi));

  // B1-ior-per-tri: decode per-tri IOR → compute physical Schlick F0.
  //   F0 = ((ior−1)/(ior+1))²   (normal-incidence Fresnel reflectance)
  //   Default IOR=1.5 → F0 = (0.5/2.5)² = 0.04 (matches prior GLASS_F0 constant).
  let rmCoordG = vec2u(triIndex % BVH_MATERIAL_TEX_WIDTH, triIndex / BVH_MATERIAL_TEX_WIDTH);
  let packedG = textureLoad(bvh_material, vec2i(rmCoordG), 0).r;
  let glassIor = decodeIor(packedG);
  let iorMinus1 = glassIor - 1.0;
  let iorPlus1  = glassIor + 1.0;
  let glassF0 = (iorMinus1 / iorPlus1) * (iorMinus1 / iorPlus1);

  // Fresnel transmission: scalar Schlick at camera-to-glass incidence.
  let cosI = max(0.0, dot(wo, normal));  // wo = -primaryRay.direction (camera facing)
  let fresnelR = glassF0 + (1.0 - glassF0) * pow(max(0.0, 1.0 - cosI), 5.0);
  let fresnelT = max(0.0, 1.0 - fresnelR);

  // Beer-Lambert tint (same bvh_beer read as lo_emit).
  let beerCoord = vec2u(triIndex % BVH_BEER_TEX_WIDTH, triIndex / BVH_BEER_TEX_WIDTH);
  let beerPacked = textureLoad(bvh_beer, vec2i(beerCoord), 0).r;
  let beerAlbedo = vec3f(
    f32((beerPacked >> 24u) & 0xFFu) / 255.0,
    f32((beerPacked >> 16u) & 0xFFu) / 255.0,
    f32((beerPacked >>  8u) & 0xFFu) / 255.0,
  );

  // GI contribution: Lo from the diffuse wall × Lambertian cosine response at
  // the post-glass surface × W, then multiplied by the Fresnel transmission and
  // the Beer tint to account for the glass interface attenuation.
  let Lo_transmitted = g.Lo * INV_PI * cosTheta * g.W * fresnelT * beerAlbedo;
  return Lo_transmitted;
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
  clearcoatNormal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  specular: vec4f,
  anisotropy: vec2f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  isGlass: bool,
) -> vec3f {
  if (isGlass) { return vec3f(0.0); }
  if (metal <= 0.0 && rough >= SPEC_GI_ROUGH_MAX && clearcoat.x < 1e-4 && sheen.a < 1e-4) { return vec3f(0.0); }
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
  let specBrdf = evalGGXSpecularOnlyWithSpecularClearcoatSheen(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, normal, clearcoatNormal, wo, wi);
  return g.Lo * specBrdf * g.W;
}`;
