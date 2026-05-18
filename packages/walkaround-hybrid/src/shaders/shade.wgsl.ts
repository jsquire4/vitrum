/**
 * Shading + ReSTIR-GI compute pass.
 *
 * Reads the spatialReservoir (DI result), re-traces the primary ray to find
 * the hit surface (since we're using primary-ray-cast mode instead of a G-buffer),
 * traces one indirect bounce (ReSTIR GI), and writes HDR color to hdrColorOut.
 *
 * This is the primary-ray-cast fallback mode.
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

// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(4) var<storage, read> emitterCdf:   array<f32>;
// Per-tri Beer-Lambert visible color (RGBA8 packed, alpha=0). Read on
// primary glass hits to make Lo_emit reproduce PT's transmitted-radiance
// saturation. bvh_index.w stays raw attCol for receiver paths.
@group(1) @binding(5) var<storage, read> bvh_beer:     array<u32>;

// WalkaroundUBO struct defined in COMMON_WGSL.
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// Sprint 15 — GTAO occlusion factor (r16float, full-res, 1-frame lagged).
// Multiplied into diffuse direct + indirect terms below to darken contact
// regions. Sky-miss and emissive light sources bypass this multiplier.
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

// invertMat4_common + generatePrimaryRay_common live in common.wgsl;
// they are prepended to SHADE_WGSL at compile time.

@compute @workgroup_size(8, 8, 1)
fn shadeMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let pixelIdx = gid.y * dims.x + gid.x;
  var rng = pcgInit(gid.x ^ 11111u, gid.y ^ 22222u, ubo.frameSeed ^ 0xDEADu);

  // Re-trace primary ray to find hit (primary-ray-cast mode).
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(gid.x, gid.y, dims.x, dims.y, ubo.cameraPos, invVP);
  let primaryHit = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, primaryRay, ubo.triIntersectEpsilon);

  if (!primaryHit.didHit) {
    // Sky pixel: output sky color (already written by RIS pass, but keep consistent).
    // Read from UBO so RIS miss + shade miss agree.
    let skyMiss = ubo.skyTint * ubo.skyIrradiance;
    textureStore(hdrColorOut, gid.xy, vec4f(skyMiss, 1.0));
    // G-buffer for sky: encoded "up" normal + depth=0.  The atrous denoiser
    // uses depth=0 as a sentinel that distinguishes sky from non-sky and
    // prevents floor radiance bleeding into sky pixels (or vice versa).
    textureStore(gNormalDepthOut, gid.xy, vec4f(0.5, 1.0, 0.5, 0.0));
    // Item 24: sky pixels have no surface albedo. Write (1,1,1) so
    // indirectCombine's re-modulation is a no-op for sky pixels.
    textureStore(hdrAlbedoOut,   gid.xy, vec4f(1.0, 1.0, 1.0, 1.0));
    textureStore(hdrIndirectOut, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(hdrTotalOut,    gid.xy, vec4f(skyMiss, 1.0));
    return;
  }

  let pos    = primaryRay.origin + primaryRay.direction * primaryHit.dist;
  let normal = primaryHit.normal;
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
  textureStore(gNormalDepthOut, gid.xy, vec4f(normal * 0.5 + 0.5, depthSigned));

  // matColor + isGlass already decoded above for the G-buffer write.
  // Use the BVH-baked material color for ALL surfaces (glass AND room surfaces).
  let albedo   = matColor.rgb;
  let rough    = select(0.85, 0.05, isGlass);
  let metal    = 0.0;

  // ── Self-emission for primary glass hits ─────────────────────────────────
  //
  // Le ≈ attenuationColor × transmission × sunIntensity × |sunDot| × textureMod.
  // attenuationColor is read from bvh_beer (Beer-Lambert visible color =
  // pow(rawAttCol, thickness/attDist)) — separate from bvhIndex.w which
  // carries the RAW attCol used by emitter Le and tinted-visibility.
  var Lo_emit = vec3f(0.0);
  if (isGlass) {
    let sunDot = abs(dot(ubo.sunDirection, normal));
    if (sunDot > 0.05) {
      let trans = matColor.a;
      let texId = decodeSurfaceTextureId(primaryHit.matColorPacked);
      let texMod = surfaceTextureMod(primaryHit.uv, texId);
      let beerPacked = bvh_beer[primaryHit.triIndex];
      let beerAlbedo = vec3f(
        f32((beerPacked >> 24u) & 0xFFu) / 255.0,
        f32((beerPacked >> 16u) & 0xFFu) / 255.0,
        f32((beerPacked >>  8u) & 0xFFu) / 255.0,
      );
      Lo_emit = beerAlbedo * trans * ubo.sunIntensity * sunDot * texMod;
    }
  }

  // --- Direct lighting (ReSTIR DI) ---
  //
  // Gated to !isGlass — on a near-mirror glass primary hit (rough=0.05) the GGX BRDF
  // sample of a NEIGHBOURING emitter cell pulls in that cell's color and
  // mixes it into the cell being shaded — chromatic pollution that washes
  // saturated authored colors toward pastel.
  let r = loadSpatialDI(pixelIdx);
  var Lo_direct = vec3f(0.0);
  // Came / solder (isMetal) skip Lo_direct: ReSTIR DI's single-sample
  // variance produces high-amplitude firefly speckle on thin metallic
  // strips that atrous can't smooth.
  if (!isGlass && !isMetal && r.W > 0.0 && r.M > 0u) {
    let lid = r.lightId;
    if (lid < ubo.emitterCount) {
      let e  = emitters[lid];
      // Stochastic xi instead of (0.5, 0.5). The deterministic centre-sample
      // bites hard on rect-area lights split into two triangles: the two
      // tris have different centroids, so ReSTIR flipping between them
      // produces a bimodal radiance per frame (visible flicker). Random xi
      // distributes the sample point across the triangle each frame;
      // temporalAccum integrates the variance out.
      let lsXi = vec2f(rand_f32(&rng), rand_f32(&rng));
      let ls = sampleEmitterPoint(e, lsXi);
      let toL = ls.pos - pos;
      let dist = length(toL);
      if (dist > 1e-4) {
        let wi    = toL / dist;
        let nDotL = max(0.0, dot(normal, wi));
        let nlDotL = max(0.0, dot(-e.normal, wi));
        if (nDotL > 1e-6 && nlDotL > 1e-6) {
          let occ = bvhIntersectAny(&bvh_index, &bvh_position, &bvh, pos + normal * 1e-3, wi, dist - 2e-3, ubo.triIntersectEpsilon);
          if (!occ) {
            let G    = emitterGeometry(nlDotL, dist * dist, ubo.emitterDist2Floor);
            let brdf = evalGGX(albedo, rough, metal, normal, wo, wi);
            Lo_direct = e.Le * brdf * G * r.W;
          }
        }
      }
    }
  }

  // --- Direct sun lighting with glass-aware tinted shadow ray ───────────
  //
  // Bullet 4 (caustics on receivers): the sun is treated as a directional
  // light reaching the floor/walls.  The shadow ray from the receiver
  // toward the sun walks every triangle along the path:
  //   - opaque hit  → fully shadowed (visibility = vec3f(0))
  //   - glass hit   → multiply visibility by the cell's tint factor
  //   - clear hit   → unchanged
  var Lo_sunCaustic = vec3f(0.0);
  var Lo_skyAperture = vec3f(0.0);
  // Same skip-on-metal rule: through-glass shadow rays from a came
  // bead's irregular surface produce variable visibility per pixel → speckle.
  // The ReSTIR-GI Lo_indirect term covers came illumination via the
  // half-res reservoir read further below.
  if (!isGlass && !isMetal) {
    // Direction TOWARD the sun.  ubo.sunDirection is the unit vector from
    // the world origin toward the sun.
    // Sun-cone sampling for physically-correct caustic penumbra.
    // Real sun has 0.5° angular diameter → 0.25° radius → tan ≈ 0.00436.
    //
    // Sampling strategy: PER-PIXEL DETERMINISTIC, no per-frame variance.
    // Each pixel always samples the SAME point on the sun cone (a
    // function of its (x, y) position only).
    let sunBase = ubo.sunDirection;
    let SUN_ANGULAR_RADIUS = 0.00436;
    let hx = fract(sin(f32(gid.x) * 12.9898 + f32(gid.y) * 78.233) * 43758.5453);
    let hy = fract(sin(f32(gid.x) * 93.989  + f32(gid.y) * 67.345) * 24634.6345);
    let xi = vec2f(hx, hy);
    let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
    let tan = safe_normalize(cross(upRef, sunBase));
    let bit = cross(sunBase, tan);
    let r2 = SUN_ANGULAR_RADIUS * sqrt(xi.x);
    let phi = 6.2831853 * xi.y;
    let toSun = safe_normalize(sunBase + tan * (r2 * cos(phi)) + bit * (r2 * sin(phi)));
    let nDotSun = max(0.0, dot(normal, toSun));
    if (nDotSun > 1e-6) {
      let vis = bvhTraceTintedVisibility(
        &bvh_index, &bvh_position, &bvh, &bvh_beer,
        pos + normal * 1e-3, toSun, 1e6,
      );
      // Sun irradiance × tinted visibility × Lambert(receiver) × CAUSTIC_BOOST.
      // CAUSTIC_BOOST 10 → 22: less-saturated cells (e.g., brown) Beer-Lambert
      // to pow(0.55, 6) ≈ 0.028 — caustics from those cells were below ambient
      // floor brightness, invisible against the soft DDGI cell-tint blob.
      // Audit B1: CAUSTIC_BOOST and the visibility clamp are now UBO-driven.
      // Cornell stained-glass uses 22.0 / 0.6 (the historical calibration);
      // generic scenes pass 1.0 / 1.0 (no boost, no clamp).
      let visClamped = min(vis, vec3f(ubo.causticVisClamp));
      Lo_sunCaustic = visClamped * ubo.sunIntensity * nDotSun * albedo * INV_PI * ubo.causticBoost;
    }

    // ── Multi-tap sky aperture probe ──────────────────────────────────────
    //
    // For non-glass surfaces, ambient-only DDGI doesn't deliver
    // perceptible diffuse-sky illumination. Without an explicit aperture
    // probe, the back-wall + side walls + floor outside the small caustic
    // patch render pitch black, which is un-physical for a room with a
    // daylit window.
    //
    // Probe approach: trace 5 deterministic rays — one along the
    // receiver normal + four more rotated 45° toward the sun direction
    // (a square-pyramid pattern around the surface "up axis").
    let skyTint = ubo.skyTint;
    let skyIrradiance = ubo.skyIrradiance;
    let originSky = pos + normal * 1e-3;
    let upAxis = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.y) < 0.99);
    let tangent = safe_normalize(cross(upAxis, normal));
    let bitangent = cross(normal, tangent);
    // 5 taps: centre (0°), four 45° diagonals. Each tap accumulates a
    // SCALAR luminance — opaque hit → 0, clear sky → 1, glass-tinted
    // → ~0.3 (luminance of the tint vector). Going scalar instead of
    // vec3f kills the panel-edge banding.
    let cos45 = 0.7071068;
    let sin45 = 0.7071068;
    var skyAccum = 0.0;
    var weightAccum = 0.0;
    let luminanceWeights = vec3f(0.2126, 0.7152, 0.0722);
    // Centre tap (along normal, weight 1.0).
    {
      let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, &bvh_beer, originSky, normal, 1e6);
      let lum = dot(v, luminanceWeights);
      skyAccum = skyAccum + lum * 1.0;
      weightAccum = weightAccum + 1.0;
    }
    // Four diagonal taps at 45° off-normal.
    let diag0 = safe_normalize(normal * cos45 + tangent * sin45);
    let diag1 = safe_normalize(normal * cos45 - tangent * sin45);
    let diag2 = safe_normalize(normal * cos45 + bitangent * sin45);
    let diag3 = safe_normalize(normal * cos45 - bitangent * sin45);
    {
      let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, &bvh_beer, originSky, diag0, 1e6);
      let lum = dot(v, luminanceWeights);
      skyAccum = skyAccum + lum * cos45;
      weightAccum = weightAccum + cos45;
    }
    {
      let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, &bvh_beer, originSky, diag1, 1e6);
      let lum = dot(v, luminanceWeights);
      skyAccum = skyAccum + lum * cos45;
      weightAccum = weightAccum + cos45;
    }
    {
      let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, &bvh_beer, originSky, diag2, 1e6);
      let lum = dot(v, luminanceWeights);
      skyAccum = skyAccum + lum * cos45;
      weightAccum = weightAccum + cos45;
    }
    {
      let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, &bvh_beer, originSky, diag3, 1e6);
      let lum = dot(v, luminanceWeights);
      skyAccum = skyAccum + lum * cos45;
      weightAccum = weightAccum + cos45;
    }
    let skyVisScalar = skyAccum / max(weightAccum, 1e-6);
    let skyVisAvg = vec3f(skyVisScalar);
    Lo_skyAperture = skyVisAvg * skyTint * skyIrradiance * albedo * INV_PI;
  }

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
  var Lo_indirect = vec3f(0.0);
  if (!isGlass && !isMetal) {
    let halfDims = dims / 2u;
    let halfPxF = vec2f(gid.xy) * 0.5;
    let hx0 = u32(floor(halfPxF.x));
    let hy0 = u32(floor(halfPxF.y));
    let fx = halfPxF.x - f32(hx0);
    let fy = halfPxF.y - f32(hy0);
    let bw00 = (1.0 - fx) * (1.0 - fy);
    let bw10 =        fx  * (1.0 - fy);
    let bw01 = (1.0 - fx) *        fy;
    let bw11 =        fx  *        fy;
    var totalW: f32 = 0.0;
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
      totalW = totalW + bw;
    }
    if (totalW > 1e-3) {
      Lo_indirect = Lo_indirect / totalW;
    }
  }

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
  // - The select(1.0, ao, ao > 0.001) safe-fallback prevents a corrupt
  //   first-frame AO sample (NaN, negative, huge) from blanking pixels;
  //   the texture is seeded with 1.0 at engine init but defense-in-depth.
  let aoRaw = textureLoad(aoFullTexture, vec2i(gid.xy), 0).r;
  let ao = select(1.0, clamp(aoRaw, 0.0, 1.0), aoRaw > 0.001);

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
  let directRadiance = Lo_emit + (Lo_direct + Lo_sunCaustic + Lo_skyAperture) * ao;
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
  // pattern.  Cap indirect at 1.0 — well above Cornell's plausible
  // converged indirect brightness (~0.3 worst case), generous head-room
  // for legitimate color-bleed peaks, but kills the firefly tail.
  let clampedIndirect = min(indirectRadiance, vec3f(1.0));
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
  textureStore(hdrColorOut,    gid.xy, vec4f(clampedDirect,                          1.0));
  textureStore(hdrIndirectOut, gid.xy, vec4f(clampedIndirect,                        1.0));
  textureStore(hdrAlbedoOut,   gid.xy, vec4f(albedo,                                 1.0));
  // Total = direct + indirect-with-albedo-restored; used only by Welford.
  textureStore(hdrTotalOut,    gid.xy, vec4f(clampedDirect + clampedIndirect * albedo, 1.0));
}
`;

/** W1-R6 — declarative include-graph entry.
 *  Order mirrors the historical concat `COMMON_WGSL + SURFACE_TEXTURES_WGSL +
 *  DDGI_SAMPLE_WGSL + SHADE_WGSL`. After W7-H6, `surfaceTextures` was split
 *  into the host `surfaceMods` (procedural stained-glass patterns) and the
 *  library-general `glassVisibility` (per-channel BVH shadow walker). The
 *  composer emits {common, surfaceMods, glassVisibility, ddgiSample, shade}
 *  whose concatenation equals the pre-split bytes since
 *  `STAINED_GLASS_SURFACE_MODS_WGSL + GLASS_VISIBILITY_WGSL` ===
 *  `SURFACE_TEXTURES_WGSL`. */
export const SHADE_MODULE: WgslModule = {
  name: 'shade',
  source: SHADE_WGSL,
  requires: ['common', 'surfaceMods', 'glassVisibility', 'ddgiSample'],
};
