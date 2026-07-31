/**
 * Scene-specific procedural surface textures + tinted-visibility BVH helper.
 *
 * Both items are consumed only by shade.wgsl (the BVH helper traces shadow
 * rays that respect glass tints, which calls `surfaceTextureMod` to weight
 * the per-hit attenuation by the cell's authored pattern).
 *
 * Kept out of COMMON_WGSL so the shared header stays focused on math; the
 * pipelineCompiler injects this string into shade's shader module only.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SURFACE_TEXTURES_WGSL = /* wgsl */ `

// ============================================================
// Procedural surface-texture pattern functions
// ============================================================
//
// One function per authored surface-texture name. Each takes the hit's
// interpolated UV (already in [0,1]) and returns a scalar modulation
// factor — multiplied into the cell's emission to produce visible
// per-pixel colour variation.
//
// These functions are faithful WGSL re-implementations of the GLSL
// surface bakers — the same trig + noise math, evaluated per-shader-
// invocation instead of per-texel-during-bake.  They DO NOT hardcode
// colours; they only produce the *pattern*, which the shade pass
// multiplies into the per-cell baseColor it decoded from bvhIndex.

fn _hash21(p: vec2f) -> f32 {
  // Deterministic 2D-to-1D cell hash, range ~[0,1).
  return floatCellHash(p, 0x53544631u);
}

fn _vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = _hash21(i);
  let b = _hash21(i + vec2f(1.0, 0.0));
  let c = _hash21(i + vec2f(0.0, 1.0));
  let d = _hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn _fbm(p: vec2f) -> f32 {
  // 4-octave fractal Brownian motion.
  var v = 0.0;
  var a = 0.5;
  var pp = p;
  for (var k = 0; k < 4; k = k + 1) {
    v = v + a * _vnoise(pp);
    pp = pp * 2.03;
    a  = a  * 0.5;
  }
  return v;
}

fn _waterglassMod(uv: vec2f) -> f32 {
  let k = 12.0;
  let w1 = sin(uv.x * k * 4.0 + uv.y * 2.0) * 0.4;
  let w2 = sin(uv.y * k * 5.0 - uv.x * 1.5) * 0.3;
  let w3 = sin((uv.x + uv.y) * k * 3.0) * 0.2;
  let w4 = sin((uv.x - uv.y) * k * 6.0) * 0.1;
  return 0.95 + (w1 + w2 + w3 + w4) * 0.45;
}

fn _rippleMod(uv: vec2f) -> f32 {
  let p = uv - vec2f(0.5);
  let r = length(p) * 22.0;
  let s = sin(r);
  return 0.95 + s * 0.4;
}

fn _hammeredMod(uv: vec2f) -> f32 {
  let n = _vnoise(uv * 14.0);
  let n2 = _vnoise(uv * 28.0 + vec2f(11.0, 7.0));
  return 0.85 + (n - 0.5) * 0.7 + (n2 - 0.5) * 0.25;
}

fn _graniteMod(uv: vec2f) -> f32 {
  let n = _vnoise(uv * 32.0);
  let n2 = _vnoise(uv * 64.0 + vec2f(3.0, 9.0));
  return 0.85 + (n - 0.5) * 0.45 + (n2 - 0.5) * 0.2;
}

fn _baroqueMod(uv: vec2f) -> f32 {
  let warp = vec2f(_fbm(uv * 2.0), _fbm(uv * 2.0 + vec2f(7.3, 1.7)));
  let v = _fbm(uv * 3.5 + warp * 1.5);
  return 0.7 + v * 0.7;
}

fn _catspawMod(uv: vec2f) -> f32 {
  let lattice = sin(uv.x * 18.0) * sin(uv.y * 18.0);
  let n = _vnoise(uv * 8.0);
  return 0.95 + lattice * 0.25 + (n - 0.5) * 0.3;
}

fn _flemishMod(uv: vec2f) -> f32 {
  let stripes = sin(uv.y * 28.0 + sin(uv.x * 6.0) * 1.2);
  let n = _vnoise(uv * vec2f(20.0, 4.0));
  return 0.9 + stripes * 0.3 + (n - 0.5) * 0.25;
}

/**
 * Procedural surface modulation factor for a glass cell.
 * Returns a single scalar that the shade pass multiplies into the
 * cell's emitted radiance, producing visible per-pixel patterns.
 *
 * Result is clamped to [0.2, 1.8] so cells stay distinguishably
 * coloured (no full black-out, no over-bright NaN-prone values).
 */
fn surfaceTextureMod(uv: vec2f, texId: u32) -> f32 {
  var m: f32 = 1.0;
  switch (texId) {
    case 0u: { m = 1.0; }                   // smooth — flat
    case 1u: { m = _hammeredMod(uv); }
    case 2u: { m = _rippleMod(uv); }
    case 3u: { m = _graniteMod(uv); }
    case 4u: { m = _baroqueMod(uv); }
    case 5u: { m = _waterglassMod(uv); }
    case 6u: { m = _catspawMod(uv); }
    case 7u: { m = _flemishMod(uv); }
    default: { m = 1.0; }
  }
  // Wider clamp range [0.2, 1.8] for more dramatic per-pixel texture modulation.
  return clamp(m, 0.2, 1.8);
}

// ============================================================
// Per-channel visibility (vec3f) along a ray. Used for sun-aware shadow
// queries that must tint the sunlight by every glass slab the shadow
// ray passes through, instead of either:
//   (a) the bool bvhIntersectAny path, which skips ALL glass tris and
//       therefore hands the floor full white sunlight even when colored
//       panel cells are in the path; or
//   (b) the opaque-shadow path, which would treat glass as a wall and
//       black-out the floor caustic entirely.
//
// Algorithm:
//   - repeatedly request the nearest remaining world-space hit;
//   - apply alpha/cast-shadow/interface transmission at that boundary;
//   - pair bulk entry/exit ownership by material record + TLAS instance;
//   - apply Beer-Lambert only to the geometric segment spent in that medium.
//
// tMax lets the caller cap the ray at e.g. the distance to a sampled
// emitter point. For directional-light queries pass a large value (INFINITY).
//
// traceSceneFirstHit owns merged-BVH versus TLAS dispatch and always returns a
// world-space distance, so the same continuation walk serves both layouts.
// ============================================================
// WS1 (2026-05-29) — bvh_beer is an r32uint TEXTURE (not a storage buffer):
// width matches host pipeline/bvhBeerTexture.ts. Declared here (the earliest
// scene-consuming module in the shade compose chain) so both shade and
// stainedGlassShade see it. Textures are passed to WGSL functions by handle.
const BVH_BEER_TEX_WIDTH: u32 = 4096u;

// Transparent visibility must be evaluated in hit order. The previous direct
// BVH-leaf accumulator multiplied a full authored Beer tint at every interface;
// leaf visitation order is not ray order and it cannot identify the distance
// spent inside a medium. The continuation walker below is the only tinted
// visibility implementation in this module.

fn materialShadowAuthoredBeerTint(
  hit: IntersectionResult,
  bvh_beer: texture_2d<u32>,
) -> vec3f {
  let triIdx = hit.indices.w;
  let coord = vec2u(
    triIdx % BVH_BEER_TEX_WIDTH,
    triIdx / BVH_BEER_TEX_WIDTH,
  );
  let packed = textureLoad(bvh_beer, vec2i(coord), 0).r;
  let tint = vec3f(
    f32((packed >> 24u) & 0xffu),
    f32((packed >> 16u) & 0xffu),
    f32((packed >> 8u) & 0xffu),
  ) / 255.0;
  return applyThicknessMapToBeerTint(
    triIdx,
    hit.uv,
    materialAtlasUv1ForHit(hit),
    tint,
  );
}

fn materialShadowEffectiveThickness(hit: IntersectionResult) -> f32 {
  var thickness = max(materialOpticalThickness(hit.indices.w), 0.0);
  let texel = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
  );
  if (texel.x >= 0.0) {
    thickness = thickness * clamp(texel.g, 0.0, 1.0);
  }
  return thickness;
}

fn materialShadowBeerForSegment(
  triIdx: u32,
  authoredTint: vec3f,
  authoredThickness: f32,
  segmentLength: f32,
) -> vec3f {
  if (segmentLength <= 0.0) { return vec3f(1.0); }
  if (!(authoredThickness > 0.0)) { return vec3f(0.0); }
  let rgbBeer = pow(
    clamp(authoredTint, vec3f(0.0), vec3f(1.0)),
    vec3f(segmentLength / authoredThickness),
  );
  return materialSpectralAttenuation(triIdx, segmentLength, rgbBeer);
}

fn materialShadowFaceTransmission(
  hit: IntersectionResult,
  dir: vec3f,
) -> vec3f {
  let layer = sampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
  let film = materialThinFilmResponse(
    hit.indices.w,
    hit.side >= 0.0,
    abs(dot(hit.normal, dir)),
  );
  let filmT = select(vec3f(1.0), film.transmittance, film.present != 0u);
  return faceLayerTransmission(layer) * filmT;
}

fn materialShadowCoverageForHit(
  hit: IntersectionResult,
  materialWord: u32,
) -> f32 {
  if ((materialWord & 1u) != 0u) {
    return 0.0;
  }
  let alpha = materialAlphaCoverageForHit(hit, materialWord);
  if (alpha.scalarDiscarded != 0u) {
    return 0.0;
  }
  if (alpha.mode == 0u) {
    return 1.0;
  }
  if (alpha.mode == 1u) {
    return select(0.0, 1.0, alpha.coverage >= alpha.cutoff);
  }
  if (alpha.mode == 2u) {
    return clamp(alpha.coverage, 0.0, 1.0);
  }
  return select(0.0, 1.0, alpha.coverage > 0.0);
}

fn materialShadowWorldSurfaceBudget(
  bvhMode: u32,
  tlasNodeCount: u32,
) -> u32 {
  let triangleCount = bvhIndexCount();
  if (triangleCount == 0u) { return 1u; }
  var instanceCount = 1u;
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    instanceCount = max(tlasBlasRootCount(), 1u);
  }
  if (triangleCount > 0xfffffffeu / instanceCount) {
    return 0xffffffffu;
  }
  return triangleCount * instanceCount + 1u;
}

// Shared transparent-visibility walker. When blockMaterialTransmission is
// true, material transmission is owned by the explicit specular-path
// estimator: covered glass is therefore an occluder here, while castShadow:
// false, alpha-mask holes, and the uncovered fraction of alpha-blend surfaces
// retain their ordinary visibility semantics.
fn traceSceneAlphaTintTransmittanceTexturedWithOwnership(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  bvh_beer: texture_2d<u32>,
  blockMaterialTransmission: bool,
) -> vec3f {
  var walkRay: Ray;
  walkRay.origin = origin;
  walkRay.direction = dir;
  var traveled = 0.0;
  var tau = vec3f(1.0);
  let step = max(1e-4, triEps * 4.0);
  var mediumMaterialId: array<u32, 16>;
  var mediumTri: array<u32, 16>;
  var mediumInstance: array<u32, 16>;
  var mediumTint: array<vec3f, 16>;
  var mediumThickness: array<f32, 16>;
  var mediumDepth = 0u;
  let surfaceBudget = materialShadowWorldSurfaceBudget(
    bvhMode,
    tlasNodeCount,
  );
  for (var i = 0u; i < surfaceBudget; i = i + 1u) {
    let remaining = tMax - traveled;
    if (remaining <= step || max(max(tau.x, tau.y), tau.z) <= 0.0) {
      return clamp(tau, vec3f(0.0), vec3f(1.0));
    }
    let hit = traceSceneFirstHit(
      bvhMode, tlasNodeCount,
      walkRay, triEps,
    );
    if (!hit.didHit || hit.dist >= remaining) {
      if (mediumDepth > 0u) {
        let top = mediumDepth - 1u;
        tau = tau * materialShadowBeerForSegment(
          mediumTri[top],
          mediumTint[top],
          mediumThickness[top],
          remaining,
        );
      }
      return clamp(tau, vec3f(0.0), vec3f(1.0));
    }
    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      tau = tau * materialShadowBeerForSegment(
        mediumTri[top],
        mediumTint[top],
        mediumThickness[top],
        hit.dist,
      );
      if (max(max(tau.x, tau.y), tau.z) <= 0.0) {
        return vec3f(0.0);
      }
    }
    let word = textureLoad(
      materialMask,
      vec2i(i32(hit.indices.w % materialMaskWidth), i32(hit.indices.w / materialMaskWidth)),
      0,
    ).r;
    let hasMaterialTransmission = packedMaterialHasTransmission(hit.matColorPacked);
    if (!hasMaterialTransmission) {
      let alphaT = materialShadowTransmittanceForHit(hit, word, false);
      if (alphaT <= 0.0) { return vec3f(0.0); }
      tau = tau * vec3f(alphaT);
    } else if ((word & 1u) == 0u) {
      let materialId = materialAtlasMaterialId(hit.indices.w);
      if (materialId == 0xffffffffu) {
        return vec3f(0.0);
      }
      if (
        hit.side < 0.0 &&
        mediumDepth > 0u &&
        mediumMaterialId[mediumDepth - 1u] == materialId &&
        mediumInstance[mediumDepth - 1u] == hit.instanceIndex
      ) {
        tau = tau * materialShadowFaceTransmission(hit, dir);
        mediumDepth = mediumDepth - 1u;
      } else {
        let coverage = materialShadowCoverageForHit(hit, word);
        if (coverage <= 0.0) {
          // Alpha cut-out or castShadow:false: the geometric boundary is absent.
        } else if (blockMaterialTransmission) {
          // The explicit specular estimator owns covered transmission. Preserve
          // only the uncovered fraction of alpha-blended geometry.
          tau = tau * vec3f(1.0 - coverage);
        } else {
          let scalar = decodeMaterialColor(hit.matColorPacked);
          let mappedTransmission = clamp(
            sampleTransmissionMapForHit(hit, scalar.a) *
              surfaceTextureMod(
                hit.uv,
                decodeSurfaceTextureId(hit.matColorPacked),
              ),
            0.0,
            1.0,
          );
          let interfaceTransmission =
            materialShadowFaceTransmission(hit, dir) *
            vec3f(mappedTransmission);
          let thickness = materialShadowEffectiveThickness(hit);

          // Partial alpha coverage represents a mixture of an uncovered ray
          // and a covered material ray. Treat it as a weighted interface rather
          // than incorrectly placing the entire ray inside the bulk medium.
          if (coverage < 1.0 || thickness <= 0.0) {
            tau = tau * mix(
              vec3f(1.0),
              interfaceTransmission,
              vec3f(coverage),
            );
          } else if (hit.side >= 0.0) {
            if (mediumDepth >= 16u) { return vec3f(0.0); }
            tau = tau * interfaceTransmission;
            mediumMaterialId[mediumDepth] = materialId;
            mediumTri[mediumDepth] = hit.indices.w;
            mediumInstance[mediumDepth] = hit.instanceIndex;
            mediumTint[mediumDepth] = materialShadowAuthoredBeerTint(
              hit,
              bvh_beer,
            );
            mediumThickness[mediumDepth] = thickness;
            mediumDepth = mediumDepth + 1u;
          } else if (mediumDepth > 0u) {
            // A back face that does not close the active medium is ambiguous
            // ownership (overlap, corrupt winding, or a skipped boundary).
            return vec3f(0.0);
          } else {
            // The ray starts inside this medium. There was no observed entry
            // at which to pay scalar transmission, so pay it exactly once here.
            tau = tau * interfaceTransmission *
              materialShadowBeerForSegment(
                hit.indices.w,
                materialShadowAuthoredBeerTint(hit, bvh_beer),
                thickness,
                hit.dist,
              );
          }
        }
      }
    }
    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      tau = tau * materialShadowBeerForSegment(
        mediumTri[top],
        mediumTint[top],
        mediumThickness[top],
        step,
      );
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  return vec3f(0.0);
}

fn traceSceneAlphaTintTransmittanceTextured(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  bvh_beer: texture_2d<u32>,
) -> vec3f {
  return traceSceneAlphaTintTransmittanceTexturedWithOwnership(
    bvhMode, tlasNodeCount, origin, dir, tMax, triEps,
    materialMask, materialMaskWidth, bvh_beer, false,
  );
}

fn bvhTraceTintedVisibility(
  bvh_beer:     texture_2d<u32>,
  origin: vec3f,
  dir:    vec3f,
  tMax:   f32,
) -> vec3f {
  return traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    origin,
    dir,
    tMax,
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
}
`;

/** W1-R6 — declarative include-graph entry. */
export const SURFACE_TEXTURES_MODULE: WgslModule = {
  name: 'surfaceTextures',
  source: SURFACE_TEXTURES_WGSL,
  requires: ['sceneTraversal', 'sharedPrimitives', 'materialAtlas'],
};
