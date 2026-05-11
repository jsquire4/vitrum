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
export const SHADE_WGSL = /* wgsl */ `

// Blend for diffuse irradiance from DDGI probes (1.0 = full contribution when isDDGIWired()).
const DDGI_DIFFUSE_BLEND: f32 = 1.0;

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

// DDGI bind group (group 3). Atlas + sampler + grid params UBO bound
// here; shade reads via ddgiSampleFromBindings. isDDGIWired() checks
// the placeholder sentinel (dimsX==1 means no real grid bound).
// Combined-sum INCLUDES Lo_ddgi at DDGI_DIFFUSE_BLEND = 1.0, gated on
// non-glass surfaces only and on isDDGIWired() — see the combined-sum
// statement at the bottom of shadeMain.
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
// @@PPG_TRAIN_BINDINGS_INSERT@@
// @@PPG_GUIDE_DECLS_INSERT@@

fn isDDGIWired() -> bool {
  // Placeholder UBO is initialized with dims (1,1,1). Any real ProbeGrid
  // produces dims >= 3 (see ProbeGrid.computeFromBounds).
  return ddgiGrid.dimsX > 1u;
}

// Thin adapter — pipelineCompiler prepends DDGI_SAMPLE_WGSL (the
// canonical implementation in ddgiSampleWgsl.ts) before SHADE_WGSL, so
// the ddgiSample helper is in scope. This function exists only to pull
// the @group(3) bindings into argument form. Single source of math.
fn ddgiSampleFromBindings(worldPos: vec3f, surfaceNormal: vec3f) -> vec3f {
  return ddgiSample(
    worldPos,
    surfaceNormal,
    ddgiIrradiance,
    ddgiVisibility,
    ddgiSampler,
    ddgiGrid.origin.x, ddgiGrid.origin.y, ddgiGrid.origin.z,
    ddgiGrid.spacing,
    ddgiGrid.dimsX, ddgiGrid.dimsY, ddgiGrid.dimsZ,
    ddgiGrid.irrW, ddgiGrid.irrH, ddgiGrid.visW, ddgiGrid.visH,
  );
}

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
  let primaryHit = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, primaryRay);

  if (!primaryHit.didHit) {
    // Sky pixel: output sky color (already written by RIS pass, but keep consistent).
    // Read from UBO so RIS miss + shade miss agree.
    let skyMiss = ubo.skyTint * ubo.skyIrradiance;
    textureStore(hdrColorOut, gid.xy, vec4f(skyMiss, 1.0));
    // G-buffer for sky: encoded "up" normal + depth=0.  The atrous denoiser
    // uses depth=0 as a sentinel that distinguishes sky from non-sky and
    // prevents floor radiance bleeding into sky pixels (or vice versa).
    textureStore(gNormalDepthOut, gid.xy, vec4f(0.5, 1.0, 0.5, 0.0));
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
          let occ = bvhIntersectAny(&bvh_index, &bvh_position, &bvh, pos + normal * 1e-3, wi, dist - 2e-3);
          if (!occ) {
            let G    = emitterGeometry(nlDotL, dist * dist);
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
  // Lo_ddgi handles came illumination.
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
      let CAUSTIC_BOOST = 22.0;
      let visClamped = min(vis, vec3f(0.6));
      Lo_sunCaustic = visClamped * ubo.sunIntensity * nDotSun * albedo * INV_PI * CAUSTIC_BOOST;
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

  // --- Indirect lighting (ReSTIR GI -- one indirect bounce) ---
  //
  // DDGI ambient on opaque receivers — diffuse-indirect irradiance
  // sampled from the probe atlas, integrated as Lambertian contribution.
  // Glass primary hits skip this — Lo_emit drives them.
  // Gated on isDDGIWired() so the placeholder atlas (dimsX=1) is ignored.
  var Lo_ddgi = vec3f(0.0);
  if (!isGlass && isDDGIWired()) {
    // The DDGI atlas blend uses pow(w, 8) directional kernel (see
    // probeUpdateBlend.wgsl). That kernel narrows the lobe — preserving
    // directional colour separation against Cornell-style gray-vs-coloured-
    // wall hemispheres — but the resulting weighted average has roughly
    // 1/N magnitude relative to the unweighted hemisphere irradiance,
    // where N is the effective lobe-coverage factor (~8 for pow8).
    // The 4.0 multiplier compensates so the indirect contribution
    // visually balances against ReSTIR DI's direct term on diffuse
    // surfaces. 16x was over-bright (boxes saturated white) and made
    // cell-grid artefacts visible in lit regions; 4x preserves visible
    // colour bleed without overwhelming direct lighting.
    Lo_ddgi = ddgiSampleFromBindings(pos, normal) * albedo * 4.0;
  }

  // Active terms (current pipeline state):
  //   Lo_emit         glass primary hit, deterministic per pixel
  //   Lo_direct       ReSTIR DI, atrous-denoised single sample
  //   Lo_sunCaustic   sun shadow ray through glass, deterministic
  //   Lo_skyAperture  5-tap sky probe through cutout, scalar luminance
  //
  // Lo_ddgi: diffuse irradiance from DDGI atlas × albedo × INV_PI (gated on isDDGIWired()).
  // @@PPG_BOUNCE_INSERT@@
  let combined = Lo_emit + Lo_direct + Lo_sunCaustic
               + Lo_skyAperture * 0.08
               + Lo_ddgi * DDGI_DIFFUSE_BLEND;
  // Firefly clamp — ReSTIR-DI + glancing-angle BRDF evaluations occasionally
  // produce singular radiance values (cos(θ_v) → 0 at the grazing edge of
  // a wall, near-zero RIS pdf). These propagate through SVGF (which would
  // smear them spatially) and the temporal accumulator (slow to bleed off).
  // Cap per-channel: physical max for an albedo-1 diffuse surface viewing
  // Le=12 ≈ 4/π × 12 ≈ 15. We clamp at 4 to suppress the grazing-edge
  // singularities (~2.8 measured at the red-wall edge stripe) while leaving
  // legitimately bright surfaces (light source itself: Lo_emit) intact —
  // those go through a separate Lo_emit branch that bypasses the BRDF
  // singularity entirely.
  let clamped = min(combined, vec3f(4.0));
  // Write LINEAR HDR radiance to hdrColorOut — do NOT tone-map here.
  // Tone mapping must happen AFTER the à-trous denoiser so that the denoiser
  // operates in linear HDR space. The composite pass applies ACES filmic + sRGB.
  // @@PPG_RECORD_INSERT@@
  textureStore(hdrColorOut, gid.xy, vec4f(clamped, 1.0));
}
`;
