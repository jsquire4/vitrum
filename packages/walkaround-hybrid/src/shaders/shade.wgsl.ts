/**
 * Shading + ReSTIR-GI compute pass.
 *
 * Reads the spatialReservoir (DI result), re-traces the primary ray to find
 * the hit surface (since we're using primary-ray-cast mode instead of a G-buffer),
 * traces one indirect bounce (ReSTIR GI), and writes HDR color to hdrColorOut.
 *
 * This is the primary-ray-cast fallback mode.
 */

import { IRR_CELL, VIS_CELL, IRR_STRIDE, VIS_STRIDE } from '../ddgi/ddgiAtlasLayout.js';

// Atlas-layout constants are template-substituted at module-load time so
// the producer (probeGrid.allocateAtlases) and the two consumers
// (this file + ddgi/ddgiSampleWgsl.ts) read the same values from
// one source of truth (ddgiAtlasLayout.ts).
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

struct WalkaroundUBO {
  viewMatrix:      mat4x4f,
  projMatrix:      mat4x4f,
  prevViewMatrix:  mat4x4f,
  cameraPos:       vec3f,
  frameSeed:       u32,
  screenSize:      vec2u,
  emitterCount:    u32,
  totalEmPower:    f32,
  sunDirection:    vec3f,
  sunIntensity:    f32,        // sun irradiance multiplier (matches BVH build)
  skyTint:         vec3f,      // diffuse sky dome RGB (from computeLightingState)
  skyIrradiance:   f32,        // sky dome brightness scalar
};
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// DDGI bind group (group 3). Atlas + sampler + grid params UBO bound
// here; shade reads via ddgiSampleFromBindings. isDDGIWired() checks
// the placeholder sentinel (dimsX==1 means no real grid bound).
// NOTE: combined-sum currently EXCLUDES Lo_ddgi as of the per-color
// asymmetry fix — see the combined-sum comment at the bottom of
// shadeMain. The DDGI binding stays plumbed for future re-enable.
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

fn isDDGIWired() -> bool {
  // Placeholder UBO is initialized with dims (1,1,1). Any real ProbeGrid
  // produces dims >= 3 (see ProbeGrid.computeFromBounds).
  return ddgiGrid.dimsX > 1u;
}

// Inlined DDGI atlas sample. Takes worldPos + surfaceNormal, returns
// vec3f diffuse-indirect irradiance via trilinear probe blend with
// Chebyshev visibility test. Math identical to ddgiSampleWgsl.ts —
// inlined here so shade.wgsl is self-contained.
fn ddgiSampleFromBindings(worldPos: vec3f, surfaceNormal: vec3f) -> vec3f {
  let gridDims = vec3u(ddgiGrid.dimsX, ddgiGrid.dimsY, ddgiGrid.dimsZ);
  let gridPos  = (worldPos - ddgiGrid.origin) / ddgiGrid.spacing;
  let baseIdx3 = vec3i(floor(gridPos));
  let frac     = fract(gridPos);

  var sumIrr   = vec3f(0.0);
  var totalWt  = 0.0;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let co  = vec3u((i & 1u), (i >> 1u) & 1u, (i >> 2u) & 1u);
    let pi3 = baseIdx3 + vec3i(co);
    if (any(pi3 < vec3i(0)) || any(pi3 >= vec3i(gridDims))) { continue; }

    let probeFlatIdx = u32(pi3.x)
                     + u32(pi3.y) * gridDims.x
                     + u32(pi3.z) * gridDims.x * gridDims.y;
    let probeWorld   = ddgiGrid.origin + vec3f(pi3) * ddgiGrid.spacing;

    // Trilinear weight.
    let tw = mix(vec3f(1.0) - frac, frac, vec3f(co));
    var w  = tw.x * tw.y * tw.z;

    // Smooth backface modulation (DDGI paper Eq. 9).
    let toProbe   = probeWorld - worldPos;
    let probeDist = length(toProbe);
    if (probeDist > 1e-3) {
      let probeDir = toProbe / probeDist;
      let nDotP    = dot(surfaceNormal, probeDir);
      let bw       = pow((nDotP + 1.0) * 0.5, 2.0) + 0.2;
      w = w * bw;
    }

    // Octahedral-encode the surface→probe direction (visibility lookup).
    let probeDirToSurf = normalize(worldPos - probeWorld);
    let dirV = -probeDirToSurf;
    let absV = abs(dirV);
    let nv   = dirV / (absV.x + absV.y + absV.z);
    var octV: vec2f;
    if (nv.z >= 0.0) { octV = nv.xy; }
    else { octV = (1.0 - abs(nv.yx)) * vec2f(sign(nv.x), sign(nv.y)); }
    octV = octV * 0.5 + 0.5;

    // Visibility atlas UV (cell + 2px border, 1px each side). Strides
    // come from ddgiAtlasLayout.ts via template substitution.
    let visStride = ${VIS_STRIDE}u;
    let visCell   = ${VIS_CELL}u;
    let visPx     = probeFlatIdx % gridDims.x;
    let visTmpY   = probeFlatIdx / gridDims.x;
    let visPy     = visTmpY % gridDims.y;
    let visPz     = visTmpY / gridDims.y;
    let visCx     = f32(visPx * visStride) + 1.0 + octV.x * f32(visCell);
    let visCy     = f32((visPy + visPz * gridDims.y) * visStride) + 1.0 + octV.y * f32(visCell);
    let visUv     = vec2f(visCx / ddgiGrid.visW, visCy / ddgiGrid.visH);
    let vis       = textureSampleLevel(ddgiVisibility, ddgiSampler, visUv, 0.0).rg;
    let mean      = vis.x;
    let variance  = abs(vis.y - mean * mean);
    let chebyshev = select(
      variance / (variance + max(0.0, probeDist - mean) * max(0.0, probeDist - mean)),
      1.0,
      probeDist <= mean,
    );
    w = w * max(chebyshev, 0.0);

    // Octahedral-encode the surface normal (irradiance lookup).
    let absN = abs(surfaceNormal);
    let nN   = surfaceNormal / (absN.x + absN.y + absN.z);
    var octN: vec2f;
    if (nN.z >= 0.0) { octN = nN.xy; }
    else { octN = (1.0 - abs(nN.yx)) * vec2f(sign(nN.x), sign(nN.y)); }
    octN = octN * 0.5 + 0.5;

    // Irradiance atlas UV (cell + 2px border, 1px each side). Strides
    // come from ddgiAtlasLayout.ts via template substitution.
    let irrStride = ${IRR_STRIDE}u;
    let irrCell   = ${IRR_CELL}u;
    let irrPx     = probeFlatIdx % gridDims.x;
    let irrTmpY   = probeFlatIdx / gridDims.x;
    let irrPy     = irrTmpY % gridDims.y;
    let irrPz     = irrTmpY / gridDims.y;
    let irrCx     = f32(irrPx * irrStride) + 1.0 + octN.x * f32(irrCell);
    let irrCy     = f32((irrPy + irrPz * gridDims.y) * irrStride) + 1.0 + octN.y * f32(irrCell);
    let irrUv     = vec2f(irrCx / ddgiGrid.irrW, irrCy / ddgiGrid.irrH);
    let irr       = textureSampleLevel(ddgiIrradiance, ddgiSampler, irrUv, 0.0).rgb;

    sumIrr  = sumIrr + irr * w;
    totalWt = totalWt + w;
  }

  if (totalWt < 1e-4) {
    return vec3f(0.0);
  }
  return sumIrr / totalWt;
}

const RESERVOIR_DI_STRIDE = 4u;

fn loadSpatialDI(pixelIdx: u32) -> ReservoirDI {
  let b = pixelIdx * RESERVOIR_DI_STRIDE;
  return ReservoirDI(spatialReservoir[b], spatialReservoir[b+1u],
                     bitcast<f32>(spatialReservoir[b+2u]), bitcast<f32>(spatialReservoir[b+3u]));
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
      let ls = sampleEmitterPoint(e, vec2f(0.5, 0.5));
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
    Lo_ddgi = ddgiSampleFromBindings(pos, normal) * albedo * INV_PI;
  }

  // Active terms (current pipeline state):
  //   Lo_emit         glass primary hit, deterministic per pixel
  //   Lo_direct       ReSTIR DI, atrous-denoised single sample
  //   Lo_sunCaustic   sun shadow ray through glass, deterministic
  //   Lo_skyAperture  5-tap sky probe through cutout, scalar luminance
  //
  // Lo_ddgi: diffuse irradiance from DDGI atlas × albedo × INV_PI (gated on isDDGIWired()).
  let combined = Lo_emit + Lo_direct + Lo_sunCaustic
               + Lo_skyAperture * 0.08
               + Lo_ddgi * DDGI_DIFFUSE_BLEND;
  // Write LINEAR HDR radiance to hdrColorOut — do NOT tone-map here.
  // Tone mapping must happen AFTER the à-trous denoiser so that the denoiser
  // operates in linear HDR space. The composite pass applies ACES filmic + sRGB.
  textureStore(hdrColorOut, gid.xy, vec4f(combined, 1.0));
}
`;
