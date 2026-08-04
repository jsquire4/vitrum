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
  return tint;
}

fn materialShadowAuthoredThickness(hit: IntersectionResult) -> f32 {
  // Closed-volume topology is authored by the scalar material header. A
  // spatial thickness texel may attenuate one boundary differently from its
  // paired boundary, so it must never decide whether this hit pushes/pops a
  // bulk medium or represents a reciprocal thin sheet.
  let authored = max(materialOpticalThickness(hit.indices.w), 0.0);
  // Negative one is the internal synthetic reference for bulk media whose
  // topology comes from scattering/attenuation rather than a positive authored
  // thickness. Their closed-boundary distance is authoritative.
  return select(
    -1.0,
    authored,
    materialOpticalHasAuthoredThickness(hit.indices.w) && authored > 0.0,
  );
}

fn materialShadowThicknessMapScale(hit: IntersectionResult) -> f32 {
  if (!materialOpticalHasAuthoredThickness(hit.indices.w)) {
    return 1.0;
  }
  let thicknessMap = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
  );
  if (thicknessMap.valid == 0u) { return 1.0; }
  return clamp(thicknessMap.value.g, 0.0, 1.0);
}

fn materialShadowBeerForSegment(
  triIdx: u32,
  authoredTint: vec3f,
  authoredThickness: f32,
  thicknessMapScale: f32,
  volumeScattering: vec4f,
  segmentLength: f32,
) -> vec3f {
  if (segmentLength <= 0.0) { return vec3f(1.0); }
  if (
    segmentLength != segmentLength ||
    authoredThickness != authoredThickness ||
    thicknessMapScale != thicknessMapScale ||
    any(authoredTint != authoredTint) ||
    any(volumeScattering.rgb != volumeScattering.rgb) ||
    any(abs(authoredTint) > vec3f(3.402823466e38)) ||
    any(abs(volumeScattering.rgb) > vec3f(3.402823466e38))
  ) { return vec3f(0.0); }
  // Positive optical-header thickness is an authored path-length cap scaled by
  // the actual exit-face texel. A negative header is a synthetic reference for
  // zero-thickness bulk optics: the closed-boundary segment is authoritative
  // and the thickness map is intentionally ignored.
  let hasAuthoredThickness = authoredThickness > 0.0 &&
    materialOpticalHasAuthoredThickness(triIdx);
  let referenceThickness = select(1.0, authoredThickness, hasAuthoredThickness);
  let mappedCap = referenceThickness * clamp(thicknessMapScale, 0.0, 1.0);
  let transportDistance = select(
    segmentLength,
    min(segmentLength, mappedCap),
    hasAuthoredThickness,
  );
  if (transportDistance <= 0.0) { return vec3f(1.0); }
  if (transportDistance > 3.402823466e38) {
    var absorption = vec3f(1.0);
    let header = materialOpticalHeader(triIdx);
    if (any(header != header) || any(abs(header) > vec4f(3.402823466e38))) {
      return vec3f(0.0);
    }
    if (
      round(max(header.x, 0.0)) ==
        f32(VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT)
    ) {
      absorption = vec3f(0.0);
      for (
        var spectralIndex = 0u;
        spectralIndex < VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT;
        spectralIndex = spectralIndex + 1u
      ) {
        let sample = materialOpticalLoad(
          triIdx,
          VITRUM_OPTICAL_MATERIAL_MAP_SPECTRAL_SAMPLES_TEXEL_OFFSET +
            spectralIndex,
        );
        if (any(sample != sample) || any(abs(sample) > vec4f(3.402823466e38))) {
          return vec3f(0.0);
        }
        absorption = absorption + sample.yzw *
          materialBeerTransmittanceExact(sample.x, transportDistance);
      }
      absorption = clamp(absorption, vec3f(0.0), vec3f(1.0));
    } else {
      let tint = clamp(authoredTint, vec3f(0.0), vec3f(1.0));
      absorption = vec3f(
        select(0.0, 1.0, tint.x == 1.0),
        select(0.0, 1.0, tint.y == 1.0),
        select(0.0, 1.0, tint.z == 1.0),
      );
    }
    let scatterExtinction = homogeneousBeerTransmittanceRgb(
      volumeScattering.rgb,
      transportDistance,
    );
    return absorption * scatterExtinction;
  }
  let tint = clamp(authoredTint, vec3f(0.0), vec3f(1.0));
  let exponent = transportDistance / referenceThickness;
  var rgbBeer = vec3f(0.0);
  if (exponent > MATERIAL_SHADOW_MAX_FINITE_F32) {
    // A finite segment divided by a finite-but-subnormal reference thickness
    // can still overflow. Evaluate the same d->infinity lane limit used above
    // instead of relying on pow(tint, +infinity) implementation behaviour.
    rgbBeer = vec3f(
      select(0.0, 1.0, tint.x == 1.0),
      select(0.0, 1.0, tint.y == 1.0),
      select(0.0, 1.0, tint.z == 1.0),
    );
  } else {
    rgbBeer = pow(tint, vec3f(exponent));
  }
  // materialSpectralAttenuation ignores the RGB fallback when a 32-sample
  // optical payload is present, so scale its physical distance explicitly.
  // The RGB reference was already raised to G by
  // materialShadowAuthoredBeerTint. Both representations therefore implement
  // C^(G*s/D), and G=0 is identity absorption without changing topology.
  let absorption = materialSpectralAttenuation(
    triIdx,
    transportDistance,
    rgbBeer,
  );
  let scatterExtinction = homogeneousBeerTransmittanceRgb(
    max(volumeScattering.rgb, vec3f(0.0)),
    transportDistance,
  );
  return absorption * scatterExtinction;
}

fn materialShadowSmoothNormalForHit(
  hit: IntersectionResult,
) -> vec3f {
  let isTlas = ubo.bvhMode == 1u;
  let base = hit.instanceIndex * 4u;
  let transformOk = isTlas && base + 2u < tlasWorldToLocalColumnCount();
  let transformIndex = select(0u, base, transformOk);
  return smoothShadingNormal(
    hit,
    hit.normal,
    sceneLoadBvhNormal(hit.indices.x).xyz,
    sceneLoadBvhNormal(hit.indices.y).xyz,
    sceneLoadBvhNormal(hit.indices.z).xyz,
    transformOk,
    tlasLoadWorldToLocalColumn(transformIndex),
    tlasLoadWorldToLocalColumn(transformIndex + 1u),
    tlasLoadWorldToLocalColumn(transformIndex + 2u),
  );
}

fn materialShadowFaceAbsorptionForSide(
  hit: IntersectionResult,
  frontFacing: bool,
) -> vec3f {
  let layer = sampleFaceLayerControls(hit.indices.w, frontFacing);
  return faceLayerTransmission(layer);
}

fn materialShadowThinFilmTransmissionForSide(
  hit: IntersectionResult,
  dir: vec3f,
  frontFacing: bool,
) -> vec3f {
  let mappedNormal = applyBumpMapForHit(
    hit,
    applyNormalMapForSideForHit(
      hit, materialShadowSmoothNormalForHit(hit), frontFacing,
    ),
  );
  let film = materialThinFilmResponse(
    hit.indices.w,
    frontFacing,
    abs(dot(mappedNormal, dir)),
  );
  let filmT = select(vec3f(1.0), film.transmittance, film.present != 0u);
  return filmT;
}

fn materialShadowFaceTransmissionForSide(
  hit: IntersectionResult,
  dir: vec3f,
  frontFacing: bool,
) -> vec3f {
  return materialShadowFaceAbsorptionForSide(hit, frontFacing) *
    materialShadowThinFilmTransmissionForSide(hit, dir, frontFacing);
}

fn materialShadowFaceTransmission(
  hit: IntersectionResult,
  dir: vec3f,
) -> vec3f {
  return materialShadowFaceTransmissionForSide(
    hit, dir, hit.side >= 0.0,
  );
}

fn materialShadowThinSheetTransmission(
  hit: IntersectionResult,
  dir: vec3f,
) -> vec3f {
  let incidentFrontFacing = hit.side >= 0.0;
  // A thin sheet crosses both authored SurfaceAbsorptionLayer faces but its
  // preintegrated TMM response already represents the complete film stack.
  // Select the reciprocal response for the incident side and pay it once.
  return materialShadowFaceAbsorptionForSide(hit, incidentFrontFacing) *
    materialShadowFaceAbsorptionForSide(hit, !incidentFrontFacing) *
    materialShadowThinFilmTransmissionForSide(
      hit, dir, incidentFrontFacing,
    );
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
    return materialRepresentedAlphaBlendCoverage(alpha.coverage);
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

const MATERIAL_SHADOW_MEDIUM_CAPACITY: u32 = 16u;
const MATERIAL_SHADOW_MAX_FINITE_F32: f32 = 3.402823466e38;

// Value-semantic starting state for a tinted visibility proposal. Most callers
// begin in air and use materialShadowEmptyMediumState(). Explicit path walkers
// may instead seed the ordered media already surrounding their current vertex.
// transmissionPaid distinguishes a medium entered by that path from a medium
// reconstructed around a start point: the former must not pay its scalar again
// at the paired shadow exit, while the latter owns that scalar at its first exit.
struct MaterialShadowMediumState {
  depth: u32,
  materialId: array<u32, 16>,
  tri: array<u32, 16>,
  instance: array<u32, 16>,
  tint: array<vec3f, 16>,
  thickness: array<f32, 16>,
  thicknessMapScale: array<f32, 16>,
  scattering: array<vec4f, 16>,
  albedo: array<vec3f, 16>,
  distance: array<f32, 16>,
  transmissionPaid: array<u32, 16>,
};

fn materialShadowEmptyMediumState() -> MaterialShadowMediumState {
  var state: MaterialShadowMediumState;
  state.depth = 0u;
  return state;
}

fn materialShadowMappedTransmission(hit: IntersectionResult) -> f32 {
  let scalar = decodeMaterialColor(hit.matColorPacked);
  return clamp(
    sampleTransmissionMapForHit(hit, scalar.a) *
      surfaceTextureMod(
        hit.uv,
        decodeSurfaceTextureId(hit.matColorPacked),
      ),
    0.0,
    1.0,
  );
}

struct MaterialShadowContainingMedia {
  valid: u32,
  state: MaterialShadowMediumState,
};

fn materialShadowFinite3(value: vec3f) -> bool {
  return all(value == value) &&
    all(abs(value) <= vec3f(MATERIAL_SHADOW_MAX_FINITE_F32));
}

// Reconstruct every closed medium containing the fixed launch origin by
// scanning OUTWARD along the actual first transport direction. Front events
// push a temporary LIFO and matching backs pop it; an unmatched back is an
// actual launch-inside exit and records its exit-face optical payload. Those
// exits arrive inner-to-outer and are reversed into the live outer-to-inner
// stack. The origin never moves: exact exclusive-minT progression cannot jump
// over a nearby nested boundary.
fn materialShadowClassifyContainingMedia(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  direction: vec3f,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  bvh_beer: texture_2d<u32>,
) -> MaterialShadowContainingMedia {
  var out: MaterialShadowContainingMedia;
  out.valid = 0u;
  out.state = materialShadowEmptyMediumState();
  if (materialMaskWidth == 0u) { return out; }
  let directionScale = max(
    abs(direction.x), max(abs(direction.y), abs(direction.z)),
  );
  if (
    !materialShadowFinite3(origin) ||
    !materialShadowFinite3(direction) ||
    !(directionScale > 0.0)
  ) { return out; }
  let ray = Ray(origin, direction);
  let sourceFeature = opticalSourceFeatureInvalid();
  var temporaryDepth = 0u;
  var temporaryBoundaryId: array<u32, 16>;
  var innerToOuter = materialShadowEmptyMediumState();
  var exclusiveMinT = 0.0;
  var complete = false;
  let surfaceBudget = materialShadowWorldSurfaceBudget(
    bvhMode, tlasNodeCount,
  );
  for (var surface = 0u; surface < surfaceBudget; surface = surface + 1u) {
    let event = traceSceneOpticalBoundaryEvent(
      bvhMode,
      tlasNodeCount,
      ray,
      exclusiveMinT,
      sourceFeature,
    );
    if (event.status == OPTICAL_BOUNDARY_EVENT_INVALID) { return out; }
    if (event.status == OPTICAL_BOUNDARY_EVENT_NONE) {
      complete = true;
      break;
    }
    if (!(event.t > exclusiveMinT)) { return out; }
    exclusiveMinT = event.t;
    if (event.status == OPTICAL_BOUNDARY_EVENT_TANGENT) { continue; }
    if (
      event.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      event.encodedBoundaryId == 0u ||
      !event.hit.didHit
    ) { return out; }

    if (event.side > 0.0) {
      if (temporaryDepth >= MATERIAL_SHADOW_MEDIUM_CAPACITY) { return out; }
      temporaryBoundaryId[temporaryDepth] = event.encodedBoundaryId;
      temporaryDepth += 1u;
      continue;
    }
    if (event.side >= 0.0) { return out; }
    if (temporaryDepth > 0u) {
      let top = temporaryDepth - 1u;
      if (temporaryBoundaryId[top] != event.encodedBoundaryId) { return out; }
      temporaryDepth = top;
      continue;
    }
    if (innerToOuter.depth >= MATERIAL_SHADOW_MEDIUM_CAPACITY) { return out; }

    let word = textureLoad(
      materialMask,
      vec2i(
        i32(event.hit.indices.w % materialMaskWidth),
        i32(event.hit.indices.w / materialMaskWidth),
      ),
      0,
    ).r;
    let coverage = materialShadowCoverageForHit(event.hit, word);
    if (
      coverage != 1.0 ||
      !packedMaterialHasTransmission(event.hit.matColorPacked)
    ) {
      return out;
    }
    let depth = innerToOuter.depth;
    // These legacy field names are retained for ABI stability across the
    // refractive/NRC structs, but now carry component and represented-range
    // identity rather than material-slot identity.
    innerToOuter.materialId[depth] = event.encodedBoundaryId;
    innerToOuter.instance[depth] = event.representedPrimitiveInstanceId;
    innerToOuter.tri[depth] = event.hit.indices.w;
    innerToOuter.tint[depth] = materialShadowAuthoredBeerTint(
      event.hit, bvh_beer,
    );
    innerToOuter.thickness[depth] = materialShadowAuthoredThickness(event.hit);
    innerToOuter.thicknessMapScale[depth] =
      materialShadowThicknessMapScale(event.hit);
    innerToOuter.scattering[depth] = sampleVolumeScatteringControls(
      event.hit.indices.w,
    );
    let scalar = decodeMaterialColor(event.hit.matColorPacked);
    let vertexColor = sampleVertexColorForHit(event.hit);
    innerToOuter.albedo[depth] = sampleBaseColorMap(
      event.hit,
      scalar.rgb * vertexColor.rgb,
    );
    innerToOuter.distance[depth] = 0.0;
    innerToOuter.transmissionPaid[depth] = 0u;
    innerToOuter.depth = depth + 1u;
  }
  if (!complete || temporaryDepth != 0u) { return out; }

  out.state.depth = innerToOuter.depth;
  for (var destination = 0u; destination < innerToOuter.depth; destination += 1u) {
    let source = innerToOuter.depth - 1u - destination;
    out.state.materialId[destination] = innerToOuter.materialId[source];
    out.state.instance[destination] = innerToOuter.instance[source];
    out.state.tri[destination] = innerToOuter.tri[source];
    out.state.tint[destination] = innerToOuter.tint[source];
    out.state.thickness[destination] = innerToOuter.thickness[source];
    out.state.thicknessMapScale[destination] =
      innerToOuter.thicknessMapScale[source];
    out.state.scattering[destination] = innerToOuter.scattering[source];
    out.state.albedo[destination] = innerToOuter.albedo[source];
    out.state.distance[destination] = 0.0;
    out.state.transmissionPaid[destination] =
      innerToOuter.transmissionPaid[source];
  }
  out.valid = 1u;
  return out;
}

fn materialShadowAccumulateMediumDistance(
  state: ptr<function, MaterialShadowMediumState>,
  segmentDistance: f32,
) -> bool {
  if (
    segmentDistance != segmentDistance || segmentDistance < 0.0 ||
    segmentDistance > MATERIAL_SHADOW_MAX_FINITE_F32
  ) { return false; }
  if ((*state).depth == 0u) { return true; }
  // Nested media replace their enclosing medium over the nested interval; the
  // active physical medium is the LIFO top, not every enclosing stack entry.
  let depth = (*state).depth - 1u;
  let distance = (*state).distance[depth] + segmentDistance;
  if (
    distance != distance || distance < 0.0 ||
    distance > MATERIAL_SHADOW_MAX_FINITE_F32
  ) { return false; }
  (*state).distance[depth] = distance;
  return true;
}

// A finite light endpoint may legitimately lie inside a closed medium. Match
// the live component/range stack against an outward classification from that
// endpoint, then use those actual future exit faces to finish Beer/scattering
// for every still-live medium. An infinite endpoint is valid only after every
// tracked medium has exited.
fn materialShadowEndpointTransmission(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  bvh_beer: texture_2d<u32>,
  state: MaterialShadowMediumState,
) -> vec3f {
  if (tMax != tMax || tMax < 0.0 || !materialShadowFinite3(dir)) {
    return vec3f(0.0);
  }
  if (tMax > MATERIAL_SHADOW_MAX_FINITE_F32) {
    return select(vec3f(0.0), vec3f(1.0), state.depth == 0u);
  }
  let endpoint = origin + dir * tMax;
  if (!materialShadowFinite3(endpoint)) { return vec3f(0.0); }
  let classified = materialShadowClassifyContainingMedia(
    bvhMode,
    tlasNodeCount,
    endpoint,
    dir,
    triEps,
    materialMask,
    materialMaskWidth,
    bvh_beer,
  );
  if (
    classified.valid == 0u ||
    classified.state.depth != state.depth
  ) { return vec3f(0.0); }
  var attenuation = vec3f(1.0);
  for (var depth = 0u; depth < state.depth; depth = depth + 1u) {
    if (
      classified.state.materialId[depth] != state.materialId[depth] ||
      classified.state.instance[depth] != state.instance[depth]
    ) { return vec3f(0.0); }
    attenuation = attenuation * materialShadowBeerForSegment(
      classified.state.tri[depth],
      classified.state.tint[depth],
      classified.state.thickness[depth],
      classified.state.thicknessMapScale[depth],
      classified.state.scattering[depth],
      state.distance[depth],
    );
  }
  return clamp(attenuation, vec3f(0.0), vec3f(1.0));
}

// Shared transparent-visibility walker. The starting state is copied by value,
// so one NEE proposal cannot consume or mutate another proposal's medium stack.
// When blockMaterialTransmission is true, material transmission is owned by the
// explicit specular-path estimator: covered glass is therefore an occluder here,
// while castShadow:false, alpha-mask holes, and the uncovered fraction of
// alpha-blend surfaces retain their ordinary visibility semantics.
fn traceSceneAlphaTintTransmittanceTexturedWithState(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  bvh_beer: texture_2d<u32>,
  initialMediumState: MaterialShadowMediumState,
  blockMaterialTransmission: bool,
) -> vec3f {
  var mediumState = initialMediumState;
  let walkRay = Ray(origin, dir);
  var exclusiveMinT = 0.0;
  var tau = vec3f(1.0);
  let useTlas = bvhMode == 1u && tlasNodeCount > 0u;
  let surfaceBudget = materialShadowWorldSurfaceBudget(
    bvhMode,
    tlasNodeCount,
  );
  for (var i = 0u; i < surfaceBudget; i = i + 1u) {
    if (max(max(tau.x, tau.y), tau.z) <= 0.0) {
      return clamp(tau, vec3f(0.0), vec3f(1.0));
    }
    if (tMax <= exclusiveMinT) {
      return clamp(
        tau * materialShadowEndpointTransmission(
          bvhMode, tlasNodeCount, origin, dir, tMax, triEps,
          materialMask, materialMaskWidth, bvh_beer, mediumState,
        ),
        vec3f(0.0), vec3f(1.0),
      );
    }
    var hit = traceSceneFirstHit(
      bvhMode, tlasNodeCount,
      walkRay, exclusiveMinT,
    );
    if (!hit.didHit || hit.dist >= tMax) {
      if (tMax > MATERIAL_SHADOW_MAX_FINITE_F32) {
        return select(vec3f(0.0), clamp(tau, vec3f(0.0), vec3f(1.0)), mediumState.depth == 0u);
      }
      if (!materialShadowAccumulateMediumDistance(
        &mediumState, max(tMax - exclusiveMinT, 0.0),
      )) { return vec3f(0.0); }
      return clamp(
        tau * materialShadowEndpointTransmission(
          bvhMode, tlasNodeCount, origin, dir, tMax, triEps,
          materialMask, materialMaskWidth, bvh_beer, mediumState,
        ),
        vec3f(0.0), vec3f(1.0),
      );
    }
    let word = textureLoad(
      materialMask,
      vec2i(i32(hit.indices.w % materialMaskWidth), i32(hit.indices.w / materialMaskWidth)),
      0,
    ).r;
    let hasMaterialTransmission = packedMaterialHasTransmission(hit.matColorPacked);
    var acceptedT = hit.dist;
    if (hasMaterialTransmission) {
      let exactHit = traceSceneRetraceOpticalHit(
        bvhMode, tlasNodeCount, walkRay, hit, exclusiveMinT,
      );
      let sourceFeature = sceneOpticalSourceFeatureForExactHit(
        bvhMode, tlasNodeCount, hit, exactHit,
      );
      if (
        !exactHit.hit || !(exactHit.t > exclusiveMinT) ||
        sourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
      ) { return vec3f(0.0); }
      acceptedT = exactHit.t;
      if (acceptedT >= tMax) {
        if (!materialShadowAccumulateMediumDistance(
          &mediumState, max(tMax - exclusiveMinT, 0.0),
        )) { return vec3f(0.0); }
        return clamp(
          tau * materialShadowEndpointTransmission(
            bvhMode, tlasNodeCount, origin, dir, tMax, triEps,
            materialMask, materialMaskWidth, bvh_beer, mediumState,
          ),
          vec3f(0.0), vec3f(1.0),
        );
      }
      hit.normal = exactHit.normal;
      hit.barycoord = exactHit.bary;
      hit.side = exactHit.side;
      hit.dist = exactHit.t;
      let exactTriangle = sceneLoadOpticalWorldTriangle(
        useTlas, hit.indices.w, hit.instanceIndex,
      );
      if (exactTriangle.valid == 0u) { return vec3f(0.0); }
      hit.uv = exactHit.bary.x * exactTriangle.uvA +
        exactHit.bary.y * exactTriangle.uvB +
        exactHit.bary.z * exactTriangle.uvC;
    }
    if (!materialShadowAccumulateMediumDistance(
      &mediumState, acceptedT - exclusiveMinT,
    )) { return vec3f(0.0); }
    if (!hasMaterialTransmission) {
      let alphaT = materialShadowTransmittanceForHit(hit, word, false);
      if (alphaT <= 0.0) { return vec3f(0.0); }
      tau = tau * vec3f(alphaT);
    } else {
      // Coverage owns geometric presence and must be resolved before a boundary
      // is allowed to mutate the medium stack. In particular, a cut-out exit is
      // not the paired exit of the covered entry behind it.
      let coverage = materialShadowCoverageForHit(hit, word);
      if (coverage > 0.0) {
        let boundaryId = sceneOpticalEncodedBoundaryId(
          useTlas, hit.indices.w, hit.instanceIndex,
        );
        let representedId = sceneOpticalRepresentedPrimitiveInstanceId(
          useTlas, hit.indices.w, hit.instanceIndex,
        );
        if (representedId == 0u) { return vec3f(0.0); }
        let bulkMedium = boundaryId != 0u;
        let thickness = materialShadowAuthoredThickness(hit);
        if (bulkMedium && coverage < 1.0) {
          // Fractional bulk coverage has no unique represented topology.
          return vec3f(0.0);
        }
        let pairedExit =
          bulkMedium &&
          hit.side < 0.0 &&
          mediumState.depth > 0u &&
          mediumState.materialId[mediumState.depth - 1u] == boundaryId &&
          mediumState.instance[mediumState.depth - 1u] == representedId;
        if (pairedExit) {
          let top = mediumState.depth - 1u;
          if (
            blockMaterialTransmission &&
            mediumState.transmissionPaid[top] == 0u
          ) {
            // This containing medium was reconstructed rather than entered by
            // the explicit specular estimator, so its covered exit is unowned.
            tau = tau * vec3f(1.0 - coverage);
          } else {
            tau = tau * materialShadowBeerForSegment(
              hit.indices.w,
              materialShadowAuthoredBeerTint(hit, bvh_beer),
              materialShadowAuthoredThickness(hit),
              materialShadowThicknessMapScale(hit),
              sampleVolumeScatteringControls(hit.indices.w),
              mediumState.distance[top],
            );
            var pairedExitTransmission = materialShadowFaceTransmission(hit, dir);
            if (mediumState.transmissionPaid[top] == 0u) {
              pairedExitTransmission = pairedExitTransmission *
                vec3f(materialShadowMappedTransmission(hit));
            }
            tau = tau * pairedExitTransmission;
            mediumState.depth = mediumState.depth - 1u;
          }
        } else if (blockMaterialTransmission) {
          // The explicit specular estimator owns covered transmission. Preserve
          // only the uncovered fraction of a reciprocal thin sheet.
          tau = tau * vec3f(1.0 - coverage);
        } else {
          let mappedTransmission = materialShadowMappedTransmission(hit);
          var interfaceTransmission = vec3f(mappedTransmission);
          let thicknessMapScale = materialShadowThicknessMapScale(hit);

          if (!bulkMedium) {
            // A zero-thickness sheet has no persistent inside state; fractional
            // alpha coverage is therefore a well-defined interface mixture. Its
            // virtual opposite boundary owns the reciprocal face layer/film once.
            interfaceTransmission =
              materialShadowThinSheetTransmission(hit, dir) *
              vec3f(mappedTransmission);
            tau = tau * mix(
              vec3f(1.0),
              interfaceTransmission,
              vec3f(coverage),
            );
          } else if (hit.side >= 0.0) {
            interfaceTransmission = materialShadowFaceTransmission(hit, dir) *
              vec3f(mappedTransmission);
            if (mediumState.depth >= MATERIAL_SHADOW_MEDIUM_CAPACITY) {
              return vec3f(0.0);
            }
            tau = tau * interfaceTransmission;
            mediumState.materialId[mediumState.depth] = boundaryId;
            mediumState.tri[mediumState.depth] = hit.indices.w;
            mediumState.instance[mediumState.depth] = representedId;
            mediumState.tint[mediumState.depth] = materialShadowAuthoredBeerTint(
              hit,
              bvh_beer,
            );
            mediumState.thickness[mediumState.depth] = thickness;
            mediumState.thicknessMapScale[mediumState.depth] = thicknessMapScale;
            mediumState.scattering[mediumState.depth] =
              sampleVolumeScatteringControls(hit.indices.w);
            let vertexColor = sampleVertexColorForHit(hit);
            mediumState.albedo[mediumState.depth] = sampleBaseColorMap(
              hit,
              decodeMaterialColor(hit.matColorPacked).rgb * vertexColor.rgb,
            );
            mediumState.distance[mediumState.depth] = 0.0;
            mediumState.transmissionPaid[mediumState.depth] = 1u;
            mediumState.depth = mediumState.depth + 1u;
          } else {
            // The complete containing-media classifier should have seeded every
            // valid start-inside exit. A remaining unmatched back face is broken
            // winding/overlap and must not leak light.
            return vec3f(0.0);
          }
        }
      }
    }
    exclusiveMinT = acceptedT;
  }

  return vec3f(0.0);
}

fn traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  bvh_beer: texture_2d<u32>,
  containingMedia: MaterialShadowContainingMedia,
  blockMaterialTransmission: bool,
) -> vec3f {
  // The caller-level cached classification is only an allocation/ABI seam.
  // Each NEE proposal has its own actual direction, so reconstruct from that
  // direction here before the value-semantic walk begins.
  let actualContainingMedia = materialShadowClassifyContainingMedia(
    bvhMode, tlasNodeCount, origin, dir, triEps,
    materialMask, materialMaskWidth, bvh_beer,
  );
  if (actualContainingMedia.valid == 0u) { return vec3f(0.0); }
  return traceSceneAlphaTintTransmittanceTexturedWithState(
    bvhMode, tlasNodeCount, origin, dir, tMax, triEps,
    materialMask, materialMaskWidth, bvh_beer,
    actualContainingMedia.state, blockMaterialTransmission,
  );
}

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
  let containingMedia = materialShadowClassifyContainingMedia(
    bvhMode,
    tlasNodeCount,
    origin,
    dir,
    triEps,
    materialMask,
    materialMaskWidth,
    bvh_beer,
  );
  return traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
    bvhMode, tlasNodeCount, origin, dir, tMax, triEps,
    materialMask, materialMaskWidth, bvh_beer,
    containingMedia, blockMaterialTransmission,
  );
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
