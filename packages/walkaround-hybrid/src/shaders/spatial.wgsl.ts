/**
 * Spatial reuse compute pass.
 *
 * Combines 5 spatially-neighboring reservoirs via Poisson-disk offsets (30px radius).
 * Reads from currentReservoir, writes to spatialReservoir.
 * Two separable passes are run by the orchestrator (using the same shader twice).
 *
 * Primary-ray-cast mode: no G-buffer rasterization.  We re-cast primary
 * rays here for the center pixel + each neighbor so the target function p̂ is
 * evaluated at the CORRECT surface, not at the world origin.
 */

export const SPATIAL_WGSL = /* wgsl */ `

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

// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(4) var<storage, read> emitterCdf:   array<f32>;

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
  sunIntensity:    f32,
  skyTint:         vec3f,      // diffuse sky dome RGB
  skyIrradiance:   f32,        // sky dome brightness
};
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

const RESERVOIR_DI_STRIDE = 4u;
// NEIGHBORS = 5 (restored — was briefly 3 for perf). The spatial-2
// pass was also restored alongside.
// Per-pixel spatial-reuse drives the soft falloff / AO-like coherence
// that makes the rendering feel grounded; cutting it for perf
// produced visible "sparkles around came/solder".
const NEIGHBORS = 5u;
const RADIUS = 30.0;
const M_SCALE = 4u;

fn loadR_rw(buf: ptr<storage, array<u32>, read_write>, idx: u32) -> ReservoirDI {
  let b = idx * RESERVOIR_DI_STRIDE;
  return ReservoirDI(buf[b], buf[b+1u], bitcast<f32>(buf[b+2u]), bitcast<f32>(buf[b+3u]));
}
fn storeR_rw(buf: ptr<storage, array<u32>, read_write>, idx: u32, r: ReservoirDI) {
  let b = idx * RESERVOIR_DI_STRIDE;
  buf[b] = r.lightId; buf[b+1u] = r.M;
  buf[b+2u] = bitcast<u32>(r.w_sum); buf[b+3u] = bitcast<u32>(r.W);
}

// PrimarySurface — what we know about the surface a pixel sees, derived
// from re-casting that pixel's primary ray through the BVH.
struct PrimarySurface {
  hit:    bool,
  pos:    vec3f,
  normal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  depth:  f32,    // along-ray distance, used by the geometric similarity gate
};

fn castPrimary(px: vec2u, dims: vec2u, invVP: mat4x4f) -> PrimarySurface {
  var s: PrimarySurface;
  let ray = generatePrimaryRay_common(px.x, px.y, dims.x, dims.y, ubo.cameraPos, invVP);
  let hit = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, ray);
  s.hit = hit.didHit;
  if (!hit.didHit) {
    return s;
  }
  s.pos    = ray.origin + ray.direction * hit.dist;
  s.normal = hit.normal;
  s.wo     = -ray.direction;
  let matColor = decodeMaterialColor(hit.matColorPacked);
  let isGlass  = matColor.a > 0.3;
  s.albedo = matColor.rgb;
  s.rough  = select(0.85, 0.05, isGlass);
  s.metal  = 0.0;
  s.depth  = hit.dist;
  return s;
}

fn computePHat_s(lid: u32, surf: PrimarySurface) -> f32 {
  if (!surf.hit) { return 0.0; }
  let e = emitters[lid];
  let centroid = (e.vA + e.vB + e.vC) / 3.0;
  let toL   = centroid - surf.pos;
  let dist2 = dot(toL, toL);
  if (dist2 < 1e-8) { return 0.0; }
  let wi     = toL / sqrt(dist2);
  let nDotL  = max(0.0, dot(surf.normal, wi));
  let nlDotL = max(0.0, dot(-e.normal, wi));
  if (nDotL < 1e-6 || nlDotL < 1e-6) { return 0.0; }
  // evalGGX already multiplies by NdotL; G is emitter geometry term only.
  let G    = nlDotL / dist2;
  let brdf = evalGGX(surf.albedo, surf.rough, surf.metal, surf.normal, surf.wo, wi);
  return luminance(e.Le * brdf * G);
}

// Poisson disk offsets (normalized, scale by RADIUS in the shader).
fn poissonDisk(i: u32, rotation: f32) -> vec2f {
  var offsets: array<vec2f, 8> = array<vec2f, 8>(
    vec2f( 0.0,      1.0     ),
    vec2f( 0.866,    0.5     ),
    vec2f( 0.866,   -0.5     ),
    vec2f( 0.0,     -1.0     ),
    vec2f(-0.866,   -0.5     ),
    vec2f(-0.866,    0.5     ),
    vec2f( 0.354,    0.354   ),
    vec2f(-0.354,   -0.354   ),
  );
  let o = offsets[i % 8u];
  let s = sin(rotation);
  let c = cos(rotation);
  return vec2f(o.x * c - o.y * s, o.x * s + o.y * c);
}

@compute @workgroup_size(8, 8, 1)
fn spatialMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let pixelIdx = gid.y * dims.x + gid.x;
  var rng = pcgInit(gid.x ^ 54321u, gid.y ^ 98765u, ubo.frameSeed ^ 0xCAFEu);

  var r = loadR_rw(&currentReservoir, pixelIdx);

  // M-scale down before spatial.
  if (r.M > M_SCALE) {
    r.w_sum = r.w_sum * f32(M_SCALE) / f32(r.M);
    r.M = M_SCALE;
  }

  // Re-cast the center pixel's primary ray to get the actual surface — needed
  // both for the similarity gate (we compare against neighbor surfaces, not
  // against placeholder textures) and for evaluating p̂ at the right pos/normal.
  let vp    = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let center = castPrimary(gid.xy, dims, invVP);
  if (!center.hit) {
    // Sky pixel — no reservoir to combine; pass current through unchanged.
    storeR_rw(&spatialReservoir, pixelIdx, r);
    return;
  }

  let rotation = rand_f32(&rng) * 6.2831;

  for (var i = 0u; i < NEIGHBORS; i++) {
    let offset = poissonDisk(i, rotation);
    let nbrPx  = vec2i(gid.xy) + vec2i(vec2f(offset.x * RADIUS, offset.y * RADIUS));
    if (any(nbrPx < vec2i(0)) || any(nbrPx >= vec2i(dims))) { continue; }
    let nbrIdx = u32(nbrPx.y) * dims.x + u32(nbrPx.x);

    // Geometric similarity gate computed from BVH-cast primary surfaces.
    let nbr_surf = castPrimary(vec2u(nbrPx), dims, invVP);
    if (!nbr_surf.hit) { continue; }
    let depthDiff = abs(center.depth - nbr_surf.depth);
    // 0.10 × center.depth = relative 10% depth tolerance (more meaningful than
    // an absolute 0.15 m gate when scene scale spans tens of meters).
    let depthTol  = max(0.05, 0.10 * center.depth);
    let normalDot = dot(center.normal, nbr_surf.normal);
    if (depthDiff > depthTol || normalDot < 0.9) { continue; }

    let nbr  = loadR_rw(&currentReservoir, nbrIdx);
    let nbrM = max(1u, nbr.M / M_SCALE);

    // Re-evaluate p̂ at the CENTER surface for the neighbor's chosen light.
    let pHatNbrAtCenter = computePHat_s(nbr.lightId, center);
    let w = pHatNbrAtCenter * nbr.W * f32(nbrM);

    r.M += nbrM;
    r.w_sum += w;
    if (rand_f32(&rng) * r.w_sum < w && w > 0.0) {
      r.lightId = nbr.lightId;
    }
  }

  // Recompute W.
  let pHatZ = computePHat_s(r.lightId, center);
  r.W = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 0.0);

  storeR_rw(&spatialReservoir, pixelIdx, r);
}
`;
