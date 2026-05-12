/**
 * Temporal reuse compute pass.
 *
 * Projects current pixel through previous MVP to find the previous-frame
 * reservoir, then combines via GRIS with M-clamp = 20.
 *
 * Primary-ray-cast mode: no G-buffer rasterization.  We re-cast the
 * primary ray here to get the world-space hit `pos` and `normal`, then
 * reproject `pos` through the previous-frame view+projection matrix to find
 * the previous pixel.  This replaces the placeholder motion-vector texture
 * (which returned a constant offset, making temporal look-up land in the
 * wrong screen quadrant for ~all pixels).
 */

export const TEMPORAL_WGSL = /* wgsl */ `

// Bind groups re-declared (same layout as ris.wgsl -- required in WGSL).
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

// WalkaroundUBO struct defined in COMMON_WGSL.
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// RESERVOIR_DI_STRIDE / loadReservoirDI_{rw,ro} / storeReservoirDI_rw live in COMMON_WGSL.
// M_CLAMP is now read from the UBO (ubo.temporalMClampDI) — the compile-time
// constant is kept only as documentation of the default.
// const M_CLAMP_DEFAULT = 20u;

// PrimarySurface struct defined in COMMON_WGSL.
fn castPrimary_t(px: vec2u, dims: vec2u, camPos: vec3f, invVP: mat4x4f) -> PrimarySurface {
  var s: PrimarySurface;
  let ray = generatePrimaryRay_common(px.x, px.y, dims.x, dims.y, camPos, invVP);
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

// Reproject a world-space position through the previous-frame view+proj
// matrix.  Returns the previous pixel as ivec2 or -1 outside the frustum.
// We assume the projection matrix is unchanged between frames (FOV+aspect
// are static), so we use the current projMatrix together with the stored
// prevViewMatrix — matching how WalkaroundStage feeds the UBO
// (prevProjMatrix=projMatrix in the captureFrame path).
fn reprojectToPrev(world: vec3f, dims: vec2u) -> vec2i {
  let prevView = ubo.prevViewMatrix;
  let proj     = ubo.projMatrix;
  let clip = proj * (prevView * vec4f(world, 1.0));
  if (clip.w <= 0.0) { return vec2i(-1, -1); }
  let ndc = clip.xyz / clip.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) { return vec2i(-1, -1); }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
  let px = vec2i(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y)));
  return px;
}

fn computePHat_t(lid: u32, surf: PrimarySurface) -> f32 {
  if (!surf.hit) { return 0.0; }
  let e = emitters[lid];
  let centroid = (e.vA + e.vB + e.vC) / 3.0;
  let toL = centroid - surf.pos;
  let dist2 = dot(toL, toL);
  if (dist2 < 1e-8) { return 0.0; }
  let wi     = toL / sqrt(dist2);
  let nDotL  = max(0.0, dot(surf.normal, wi));
  let nlDotL = max(0.0, dot(-e.normal, wi));
  if (nDotL < 1e-6 || nlDotL < 1e-6) { return 0.0; }
  // evalGGX already multiplies by NdotL (receiver cosine); G is the emitter
  // geometry term only: cos(emitter) / dist².
  let G    = nlDotL / dist2;
  let brdf = evalGGX(surf.albedo, surf.rough, surf.metal, surf.normal, surf.wo, wi);
  return luminance(e.Le * brdf * G);
}

@compute @workgroup_size(8, 8, 1)
fn temporalMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let pixelIdx = gid.y * dims.x + gid.x;
  var rng = pcgInit(gid.x ^ 12345u, gid.y ^ 67890u, ubo.frameSeed ^ 0xABCDu);

  var cur = loadReservoirDI_rw(&currentReservoir, pixelIdx);

  // Re-cast current pixel's primary ray to get the actual surface.
  let vp    = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let curSurf = castPrimary_t(gid.xy, dims, ubo.cameraPos, invVP);
  if (!curSurf.hit) {
    // Sky pixel — nothing to project; pass-through.
    storeReservoirDI_rw(&currentReservoir, pixelIdx, cur);
    return;
  }

  // Reproject this surface's world position through the previous-frame view
  // matrix to find the matching previous-frame pixel.  Replaces the pre-fix
  // placeholder motion-vector lookup, which read a constant (0.5, 0.5) and
  // sent prevPx far off-screen for ~half of the frame.
  let prevPx = reprojectToPrev(curSurf.pos, dims);
  if (any(prevPx < vec2i(0)) || any(prevPx >= vec2i(dims))) {
    storeReservoirDI_rw(&currentReservoir, pixelIdx, cur);
    return;
  }

  let prevIdx = u32(prevPx.y) * dims.x + u32(prevPx.x);
  var prev = loadReservoirDI_ro(&previousReservoir, prevIdx);

  // Note: there is no explicit disocclusion gate here.  The implicit gate is
  // the p̂ re-evaluation below — if the previous reservoir's lightId is
  // occluded or back-facing at the current surface, p̂≈0 and the sample
  // contributes ~nothing (w_prev → 0).

  // M-clamp previous reservoir — UBO-driven for scene-independence.
  prev.M = min(prev.M, ubo.temporalMClampDI);

  // Evaluate p̂ at CURRENT pixel for the previous reservoir's chosen light.
  let pHatPrevAtCur = computePHat_t(prev.lightId, curSurf);
  let w_prev = pHatPrevAtCur * prev.W * f32(prev.M);

  // Combine reservoirs.
  var combined = cur;
  combined.M += prev.M;
  combined.w_sum += w_prev;
  if (rand_f32(&rng) * combined.w_sum < w_prev && w_prev > 0.0) {
    combined.lightId = prev.lightId;
  }

  // Recompute W.
  let pHatZ = computePHat_t(combined.lightId, curSurf);
  combined.W = select(0.0, combined.w_sum / (f32(combined.M) * pHatZ), pHatZ > 0.0);

  storeReservoirDI_rw(&currentReservoir, pixelIdx, combined);
}
`;
