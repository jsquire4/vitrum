/**
 * TSL/WGSL helper for sampling DDGI irradiance in a fragment shader.
 *
 * Exports a single `ddgiSample` function as one self-contained wgslFn
 * source. Used by `applyDDGIShading.ts` to inject DDGI diffuse-indirect
 * into PBR materials.
 *
 * Why a single inlined function (no helpers): Three.js's wgslFn parser
 * (three.webgpu.js:38176) matches only the FIRST `fn` declaration in
 * the source via `/^[fn]*\s*([a-z_0-9]+)?\s*\(([\s\S]*?)\)/`. If the
 * source contains helper functions before the entry, the parser
 * registers the wrong function (with the wrong input count). Calling
 * the wgslFn with N arguments where the parser found a function with
 * fewer than N inputs throws `TypeError: Cannot read properties of
 * undefined (reading 'type')` from `generateInput` every frame, freezing
 * the JS thread within seconds.
 *
 * Mitigation: inline every helper (octEncodeDDGI, irrAtlasUvDDGI,
 * visAtlasUvDDGI) directly inside the single `ddgiSample` body. The
 * parser sees one entry function with exactly 15 inputs, matching the
 * call site's 15 arguments. Codec/filter helpers are appended after it.
 */

import {
  IRR_PROBE_STATE_LOCAL_X,
  IRR_PROBE_STATE_LOCAL_Y,
  IRR_STRIDE,
  VIS_CELL,
  VIS_STRIDE,
} from './ddgiAtlasLayout.js';
import { DDGI_PROBE_MAX_OFFSET_NORMALIZED } from './probeState.js';
import { DDGI_ATLAS_CODEC_WGSL } from './wgsl/ddgiAtlasCodec.wgsl.js';
import type { WgslModule } from '../wgslTypes.js';

// Atlas-layout constants are template-substituted at module-load time so
// the producer (probeGrid.allocateAtlases) and the two consumers
// (this file + engines/restir/shaders/shade.wgsl.ts) read the same values
// from one source of truth (ddgiAtlasLayout.ts).
export const DDGI_SAMPLE_WGSL = /* wgsl */`
fn ddgiSample(
  worldPos: vec3f,
  surfaceNormal: vec3f,
  irradianceAtlas: texture_2d<f32>,
  visibilityAtlas: texture_2d<f32>,
  gridOriginX: f32, gridOriginY: f32, gridOriginZ: f32,
  gridSpacing: f32,
  gridDimsX: u32, gridDimsY: u32, gridDimsZ: u32,
  irrW: f32, irrH: f32, visW: f32, visH: f32,
) -> vec3f {
  let gridOrigin = vec3f(gridOriginX, gridOriginY, gridOriginZ);
  let gridDims   = vec3u(gridDimsX, gridDimsY, gridDimsZ);
  let irradianceDimensions = textureDimensions(irradianceAtlas, 0);
  let visibilityDimensions = textureDimensions(visibilityAtlas, 0);
  if (
    !ddgiAtlasFiniteVec3(worldPos) ||
    !ddgiAtlasFiniteVec3(gridOrigin) ||
    !ddgiAtlasFiniteScalar(gridSpacing) ||
    !(gridSpacing > 0.0) ||
    !ddgiAtlasFiniteScalar(irrW) || !ddgiAtlasFiniteScalar(irrH) ||
    !ddgiAtlasFiniteScalar(visW) || !ddgiAtlasFiniteScalar(visH) ||
    !(irrW > 0.0) || !(irrH > 0.0) || !(visW > 0.0) || !(visH > 0.0) ||
    any(gridDims == vec3u(0u)) ||
    any(gridDims > vec3u(0x7fffffffu)) ||
    gridDims.y > 0xffffffffu / gridDims.x ||
    gridDims.z > 0xffffffffu / (gridDims.x * gridDims.y) ||
    gridDims.x * gridDims.y * gridDims.z > 0xffffffffu / 192u ||
    irradianceDimensions.x == 0u || irradianceDimensions.y == 0u ||
    visibilityDimensions.x == 0u || visibilityDimensions.y == 0u ||
    irrW != f32(irradianceDimensions.x) ||
    irrH != f32(irradianceDimensions.y) ||
    visW != f32(visibilityDimensions.x) ||
    visH != f32(visibilityDimensions.y)
  ) { return vec3f(0.0); }
  let receiverNormal = ddgiAtlasNormalizeOrZero(surfaceNormal);
  if (all(receiverNormal == vec3f(0.0))) { return vec3f(0.0); }

  // Surface/normal bias (Majercik 2019 §4) — offset the receiver OFF the surface
  // toward the probe interior before sampling. CRITICAL: without it a receiver
  // sampled at the EXACT surface that lies on a grid-boundary plane (Cornell
  // walls/floors — gridPos on an integer plane) has its trilinear interpolation
  // collapse onto the IN-PLANE probes, whose direction to the receiver is
  // perpendicular to the surface normal → the cosine weight max(0, dot(n, dir))
  // is 0 for all of them → totalWeight == 0 → this returns vec3f(0). That
  // zeroed the entire DDGI→ReSTIR-GI handoff (the GI reconnection points are
  // surface points), so the default realtime GI was DEAD (indirect ~0; only the
  // off-default RC path produced GI). Root-caused 2026-06-07 via gidiag A/B:
  // offsetting the sample point lifted indirect 0.0004→0.25. A quarter-spacing
  // bias clears the boundary plane while staying inside the receiver's cell; the
  // Chebyshev visibility term still guards against light leak through thin walls.
  let biasedPos = worldPos + receiverNormal * (gridSpacing * 0.25);
  if (!ddgiAtlasFiniteVec3(biasedPos)) { return vec3f(0.0); }

  let gridPos  = (biasedPos - gridOrigin) / gridSpacing;
  if (
    !ddgiAtlasFiniteVec3(gridPos) ||
    any(abs(gridPos) > vec3f(2147483520.0))
  ) { return vec3f(0.0); }
  let baseIdx3 = vec3i(floor(gridPos));
  let frac     = fract(gridPos);

  var irradianceMean = vec3f(0.0);
  var totalWeight = 0.0;
  var fallbackIrr = vec3f(0.0);
  var fallbackVisibility = 0.0;
  var fallbackDistance = DDGI_ATLAS_F32_MAX;
  var hasFallback = false;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let co  = vec3u((i & 1u), (i >> 1u) & 1u, (i >> 2u) & 1u);
    let pi3 = baseIdx3 + vec3i(co);
    if (any(pi3 < vec3i(0)) || any(pi3 >= vec3i(gridDims))) { continue; }

    let probeFlatIdx = u32(pi3.x) +
                       u32(pi3.y) * gridDims.x +
                       u32(pi3.z) * gridDims.x * gridDims.y;
    let stateCoord = vec2i(
      pi3.x * i32(${IRR_STRIDE}) + i32(${IRR_PROBE_STATE_LOCAL_X}),
      (pi3.y + pi3.z * i32(gridDims.y)) * i32(${IRR_STRIDE}) +
        i32(${IRR_PROBE_STATE_LOCAL_Y}),
    );
    let state = textureLoad(irradianceAtlas, stateCoord, 0);
    if (!ddgiAtlasFiniteScalar(state.w) || state.w < 0.5) { continue; }
    var normalizedOffset = state.xyz;
    let offsetLength2 = dot(normalizedOffset, normalizedOffset);
    let maxOffset = ${DDGI_PROBE_MAX_OFFSET_NORMALIZED};
    if (!(offsetLength2 >= 0.0) || !(offsetLength2 < 1.0e20)) {
      normalizedOffset = vec3f(0.0);
    } else if (offsetLength2 > maxOffset * maxOffset) {
      normalizedOffset =
        normalizedOffset *
        ((maxOffset - 1.0e-6) * inverseSqrt(max(offsetLength2, 1.0e-12)));
    }
    let probeWorld =
      gridOrigin + vec3f(pi3) * gridSpacing + normalizedOffset * gridSpacing;
    if (!ddgiAtlasFiniteVec3(probeWorld)) { continue; }

    // Trilinear weight.
    let tw = mix(vec3f(1.0) - frac, frac, vec3f(co));
    var w  = tw.x * tw.y * tw.z;

    // NO receiver-side per-probe cosine weight (2026-06-10 cardinal-bias fix).
    //
    // History: this site used to multiply w by the "smooth backface" probe-
    // direction cosine max(0, dot(surfaceNormal, probeDirection)) — a
    // SPATIAL probe-rejection heuristic from the OCTAHEDRAL DDGI era
    // (Majercik 2019 §6, where each octahedral cell stored a single cosine-mean
    // and the receiver weight down-weighted probes "behind" the receiver).
    //
    // It is radiometrically WRONG for the L2-SH atlas we now ship: each probe
    // stores a COMPLETE cosine-convolved irradiance field E(n) = Σ E_lm·Y_lm(n)
    // valid for ANY normal n (the cosine-weighted hemisphere integral is
    // already baked in at blend time, ddgiSH.wgsl.ts). The extra probe-direction
    // cosine then DOUBLE-applies a cosine and, worse, biases by NORMAL
    // ORIENTATION: for an axis-aligned (cardinal) normal sitting between the 8
    // cube-corner probes, the 4 probes on the far side get dot(n,probeDir) ≤ 0
    // → hard-zeroed, and the 4 near probes sit ~45° off n → dot ≈ 0.5-0.7, so
    // the trilinear blend is starved asymmetrically. DIAGONAL normals see a
    // symmetric probe-weight spread and are barely touched.
    //
    // CPU self-validating harness (ddgiReceiverFullHarness.test.ts) over the
    // 5³ / 0.4-spacing grid the GPU oracle uses, enclosed-box analytic field,
    // luminance error vs the closed-form ∫L(n·ω)dω:
    //   term                       +x      -x     diag_xy  diag_xyz
    //   hard cosine (OLD/SHIPPED) -20.9%  -20.7%  -2.6%    +1.1%
    //   wrap (d·.5+.5)²+0.2       -14.3%  -14.4%  -1.6%    +1.0%
    //   NO cosine (THIS FIX)       -6.0%   -6.5%  -0.5%    +0.9%
    // The residual ~6% at cardinals is pure spatial trilinear discretization
    // (receiver on a probe plane, quarter-cell normal bias) — diagonal-quality
    // and irreducible at this grid density. This reproduces + cures the GPU
    // oracle's 23-60% cardinal under-read (HARDWARE-VALIDATION-NEEDS.md
    // "DDGI fidelity vs ground truth"). Backface/occlusion rejection is left to
    // the PHYSICAL Chebyshev visibility term below (depth-based), not a
    // geometric normal heuristic.
    let toProbe = probeWorld - biasedPos;
    if (!ddgiAtlasFiniteVec3(toProbe)) { continue; }
    let probeDist = ddgiAtlasSafeLength(toProbe);

    // Octahedral-encode the probe→surface direction. The blend producer bins
    // depth moments under each outward probe-ray direction, so the receiver
    // must query that same hemisphere.
    let probeToSurface = -toProbe;
    // A relocated probe may exactly coincide with the biased receiver. Avoid
    // normalize(vec3f(0)) → NaN octahedral UVs; the deterministic +Y fallback
    // is visible at one zero-measure point only and keeps textureLoad finite.
    var probeDirToSurf = ddgiAtlasNormalizeOrZero(probeToSurface);
    if (all(probeDirToSurf == vec3f(0.0))) {
      probeDirToSurf = vec3f(0.0, 1.0, 0.0);
    }
    let dirV       = probeDirToSurf;
    let absV       = abs(dirV);
    let nv         = dirV / (absV.x + absV.y + absV.z);
    var octV: vec2f;
    if (nv.z >= 0.0) { octV = nv.xy; }
    else { octV = vec2f((1.0 - abs(nv.y)) * select(-1.0, 1.0, nv.x >= 0.0), (1.0 - abs(nv.x)) * select(-1.0, 1.0, nv.y >= 0.0)); }
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
    let visUv     = vec2f(visCx / visW, visCy / visH);
    let vis = ddgiAtlasSampleVisibilityBilinear(visibilityAtlas, visUv);
    if (any(vis < vec2f(0.0))) { continue; }
    let mean      = vis.x;
    let meanSquared = ddgiAtlasSaturatingMul(mean, mean);
    let variance  = max(0.0, vis.y - meanSquared);
    let occlusionDelta = max(0.0, probeDist - mean);
    var chebyshev = 1.0;
    if (probeDist > mean) {
      let deltaSquared = ddgiAtlasSaturatingMul(occlusionDelta, occlusionDelta);
      let ratioScale = max(variance, deltaSquared);
      chebyshev = 0.0;
      if (ratioScale > 0.0) {
        let varianceScaled = variance / ratioScale;
        let deltaScaled = deltaSquared / ratioScale;
        chebyshev = varianceScaled / max(varianceScaled + deltaScaled, 1.0e-8);
      }
    }
    w = clamp(w * chebyshev, 0.0, 1.0);

    // L2 SH irradiance eval (seam-free; replaces the octahedral cosine-mean
    // lookup that under-read ~33% at axis-aligned normals — the octahedral
    // seam). This probe's 9 cosine-convolved coefficients are stored in the
    // first 3x3 interior texels (coeff k at (k%3, k/3)); dot them with the SH
    // basis at the surface normal to get irradiance E directly (no *PI — the
    // cosine convolution is baked into the stored coeffs at blend time).
    let irrStride = ${IRR_STRIDE}u;
    let shPx      = probeFlatIdx % gridDims.x;
    let shTmpY    = probeFlatIdx / gridDims.x;
    let shPy      = shTmpY % gridDims.y;
    let shPz      = shTmpY / gridDims.y;
    let ix        = shPx * irrStride + 1u;
    let iy        = (shPy + shPz * gridDims.y) * irrStride + 1u;
    let encodedSh0 = textureLoad(irradianceAtlas, vec2u(ix + 0u, iy + 0u), 0);
    let encodedSh1 = textureLoad(irradianceAtlas, vec2u(ix + 1u, iy + 0u), 0);
    let encodedSh2 = textureLoad(irradianceAtlas, vec2u(ix + 2u, iy + 0u), 0);
    let encodedSh3 = textureLoad(irradianceAtlas, vec2u(ix + 0u, iy + 1u), 0);
    let encodedSh4 = textureLoad(irradianceAtlas, vec2u(ix + 1u, iy + 1u), 0);
    let encodedSh5 = textureLoad(irradianceAtlas, vec2u(ix + 2u, iy + 1u), 0);
    let encodedSh6 = textureLoad(irradianceAtlas, vec2u(ix + 0u, iy + 2u), 0);
    let encodedSh7 = textureLoad(irradianceAtlas, vec2u(ix + 1u, iy + 2u), 0);
    let encodedSh8 = textureLoad(irradianceAtlas, vec2u(ix + 2u, iy + 2u), 0);
    if (
      !ddgiAtlasIrradianceEncodingValid(encodedSh0) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh1) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh2) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh3) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh4) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh5) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh6) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh7) ||
      !ddgiAtlasIrradianceEncodingValid(encodedSh8)
    ) { continue; }
    let sh0 = ddgiAtlasDecodeIrradiance(encodedSh0);
    let sh1 = ddgiAtlasDecodeIrradiance(encodedSh1);
    let sh2 = ddgiAtlasDecodeIrradiance(encodedSh2);
    let sh3 = ddgiAtlasDecodeIrradiance(encodedSh3);
    let sh4 = ddgiAtlasDecodeIrradiance(encodedSh4);
    let sh5 = ddgiAtlasDecodeIrradiance(encodedSh5);
    let sh6 = ddgiAtlasDecodeIrradiance(encodedSh6);
    let sh7 = ddgiAtlasDecodeIrradiance(encodedSh7);
    let sh8 = ddgiAtlasDecodeIrradiance(encodedSh8);

    // Evaluate all nine signed terms in a coefficient-normalized domain. This
    // preserves cancellation when individual decoded coefficients approach
    // f32 max; only the final mathematical reconstruction can saturate.
    var coefficientScale = max(abs(sh0), abs(sh1));
    coefficientScale = max(coefficientScale, abs(sh2));
    coefficientScale = max(coefficientScale, abs(sh3));
    coefficientScale = max(coefficientScale, abs(sh4));
    coefficientScale = max(coefficientScale, abs(sh5));
    coefficientScale = max(coefficientScale, abs(sh6));
    coefficientScale = max(coefficientScale, abs(sh7));
    coefficientScale = max(coefficientScale, abs(sh8));
    let safeCoefficientScale = select(
      vec3f(1.0),
      coefficientScale,
      coefficientScale > vec3f(0.0),
    );
    let normalizedIrr =
        (sh0 / safeCoefficientScale) * 0.282095
      + (sh1 / safeCoefficientScale) * (0.488603 * receiverNormal.y)
      + (sh2 / safeCoefficientScale) * (0.488603 * receiverNormal.z)
      + (sh3 / safeCoefficientScale) * (0.488603 * receiverNormal.x)
      + (sh4 / safeCoefficientScale) * (1.092548 * receiverNormal.x * receiverNormal.y)
      + (sh5 / safeCoefficientScale) * (1.092548 * receiverNormal.y * receiverNormal.z)
      + (sh6 / safeCoefficientScale) * (0.315392 * (3.0 * receiverNormal.z * receiverNormal.z - 1.0))
      + (sh7 / safeCoefficientScale) * (1.092548 * receiverNormal.x * receiverNormal.z)
      + (sh8 / safeCoefficientScale) * (0.546274 * (receiverNormal.x * receiverNormal.x - receiverNormal.y * receiverNormal.y));
    let irr = clamp(ddgiAtlasSaturatingMulComponents(
      normalizedIrr,
      coefficientScale,
    ), vec3f(0.0), vec3f(DDGI_ATLAS_F32_MAX));

    if (probeDist < fallbackDistance) {
      fallbackDistance = probeDist;
      fallbackIrr = irr;
      fallbackVisibility = max(chebyshev, 0.0);
      hasFallback = true;
    }
    if (w > 0.0) {
      let nextTotalWeight = totalWeight + w;
      let blendWeight = w / nextTotalWeight;
      irradianceMean = ddgiAtlasSaturatingAdd3(
        ddgiAtlasSaturatingMul3(irradianceMean, 1.0 - blendWeight),
        ddgiAtlasSaturatingMul3(irr, blendWeight),
      );
      totalWeight = nextTotalWeight;
    }
  }

  if (!(totalWeight > 0.0)) {
    // If every active probe had zero trilinear/Chebyshev weight, use the
    // closest active probe only when its own visibility survives. Otherwise
    // return conservative zero; inactive probes are never reintroduced.
    return select(
      vec3f(0.0),
      ddgiAtlasSaturatingMul3(fallbackIrr, fallbackVisibility),
      hasFallback && fallbackVisibility > 0.0,
    );
  }
  // Each probe's SH eval already returns irradiance E (the cosine convolution
  // is baked into the stored coefficients at blend time), so the trilinear-
  // weighted average across probes IS the irradiance — no *PI reconstruction.
  return irradianceMean;
}
${DDGI_ATLAS_CODEC_WGSL}

fn ddgiAtlasSampleVisibilityBilinear(
  visibilityAtlas: texture_2d<f32>,
  uv: vec2f,
) -> vec2f {
  let dimensions = vec2i(textureDimensions(visibilityAtlas, 0));
  if (
    any(dimensions <= vec2i(0)) ||
    any(uv != uv) ||
    any(abs(uv) > vec2f(DDGI_ATLAS_F32_MAX))
  ) {
    return vec2f(-1.0);
  }
  let texelPosition = uv * vec2f(dimensions) - vec2f(0.5);
  if (
    !all(texelPosition == texelPosition) ||
    any(abs(texelPosition) > vec2f(2147483520.0))
  ) {
    return vec2f(-1.0);
  }
  let base = vec2i(floor(texelPosition));
  let fraction = fract(texelPosition);
  let maxCoord = dimensions - vec2i(1);
  let c00 = clamp(base, vec2i(0), maxCoord);
  let c10 = clamp(base + vec2i(1, 0), vec2i(0), maxCoord);
  let c01 = clamp(base + vec2i(0, 1), vec2i(0), maxCoord);
  let c11 = clamp(base + vec2i(1, 1), vec2i(0), maxCoord);
  let encoded00 = textureLoad(visibilityAtlas, c00, 0);
  let encoded10 = textureLoad(visibilityAtlas, c10, 0);
  let encoded01 = textureLoad(visibilityAtlas, c01, 0);
  let encoded11 = textureLoad(visibilityAtlas, c11, 0);
  if (
    !ddgiAtlasVisibilityEncodingValid(encoded00) ||
    !ddgiAtlasVisibilityEncodingValid(encoded10) ||
    !ddgiAtlasVisibilityEncodingValid(encoded01) ||
    !ddgiAtlasVisibilityEncodingValid(encoded11)
  ) { return vec2f(-1.0); }
  let m00 = ddgiAtlasDecodeVisibility(encoded00);
  let m10 = ddgiAtlasDecodeVisibility(encoded10);
  let m01 = ddgiAtlasDecodeVisibility(encoded01);
  let m11 = ddgiAtlasDecodeVisibility(encoded11);
  let top = vec2f(
    ddgiAtlasSaturatingAdd(
      ddgiAtlasSaturatingMul(m00.x, 1.0 - fraction.x),
      ddgiAtlasSaturatingMul(m10.x, fraction.x),
    ),
    ddgiAtlasSaturatingAdd(
      ddgiAtlasSaturatingMul(m00.y, 1.0 - fraction.x),
      ddgiAtlasSaturatingMul(m10.y, fraction.x),
    ),
  );
  let bottom = vec2f(
    ddgiAtlasSaturatingAdd(
      ddgiAtlasSaturatingMul(m01.x, 1.0 - fraction.x),
      ddgiAtlasSaturatingMul(m11.x, fraction.x),
    ),
    ddgiAtlasSaturatingAdd(
      ddgiAtlasSaturatingMul(m01.y, 1.0 - fraction.x),
      ddgiAtlasSaturatingMul(m11.y, fraction.x),
    ),
  );
  let filtered = vec2f(
    ddgiAtlasSaturatingAdd(
      ddgiAtlasSaturatingMul(top.x, 1.0 - fraction.y),
      ddgiAtlasSaturatingMul(bottom.x, fraction.y),
    ),
    ddgiAtlasSaturatingAdd(
      ddgiAtlasSaturatingMul(top.y, 1.0 - fraction.y),
      ddgiAtlasSaturatingMul(bottom.y, fraction.y),
    ),
  );
  return vec2f(
    filtered.x,
    max(filtered.y, ddgiAtlasSaturatingMul(filtered.x, filtered.x)),
  );
}
`;

/** W1-R6 — declarative include-graph entry. The DDGI sampler is a
 *  standalone helper function (no BVH or RNG dependence on `common`). */
export const DDGI_SAMPLE_MODULE: WgslModule = {
  name: 'ddgiSample',
  source: DDGI_SAMPLE_WGSL,
  requires: [],
};

// ─────────────────────────────────────────────────────────────────────────────
// D5.1 / D5.2 dedup (2026-06-10):
//
// DDGIGridUBO was declared character-identically in risGi.wgsl, risGiNrc.wgsl,
// and shade.wgsl. sampleDDGIAtPoint (a thin wrapper over ddgiSample) was also
// duplicated in risGi and risGiNrc (risGiNrc's header even noted "Redeclares
// sampleDDGIAtPoint"). Both are extracted here:
//
//   DDGIGridUBO struct + @group(3) @binding(3) ddgiGrid UBO declaration
//   sampleDDGIAtPoint — calls ddgiSample with all 16 arguments extracted from ddgiGrid.*
//
// Requires: ['ddgiSample'] so fn ddgiSample is in scope for sampleDDGIAtPoint.
//
// shade.wgsl: the @group(3) @binding(0..1) irradiance/visibility bindings
// remain in shade's own body; only the struct + @binding(3) are here.
// The binding sequence 0-1 (shade body) and 3 (this module, emitted first)
// is fine in a single composed string — naga sees them in one pass.
//
// Group(3) layout compat: shade uses @group(3) purely for layout compatibility
// (it does not call sampleDDGIAtPoint or read ddgiGrid fields); the binding
// declarations here keep the group slot occupied for pipeline layout validation.
// ─────────────────────────────────────────────────────────────────────────────

export const DDGI_GRID_UBO_WGSL = /* wgsl */`
// DDGIGridUBO — shared @group(3) @binding(3) layout for risGi, risGiNrc, shade.
// D5.1 dedup: extracted from three char-identical declarations (2026-06-10).
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
// sampleDDGIAtPoint — thin wrapper over ddgiSample using the ddgiGrid UBO fields.
// D5.2 dedup: extracted from duplicate definitions in risGi + risGiNrc (2026-06-10).
// The ddgiIrradiance / ddgiVisibility bindings are declared in the per-shader
// body (@group(3) @binding(0..1)) — those are in scope here
// because the composer emits this module BEFORE the consumer's own source.
fn sampleDDGIAtPoint(worldPos: vec3f, surfaceNormal: vec3f) -> vec3f {
  return ddgiSample(
    worldPos, surfaceNormal,
    ddgiIrradiance, ddgiVisibility,
    ddgiGrid.origin.x, ddgiGrid.origin.y, ddgiGrid.origin.z,
    ddgiGrid.spacing,
    ddgiGrid.dimsX, ddgiGrid.dimsY, ddgiGrid.dimsZ,
    ddgiGrid.irrW, ddgiGrid.irrH, ddgiGrid.visW, ddgiGrid.visH,
  );
}
`;

/**
 * D5.1+D5.2 — shared DDGI grid UBO struct + binding + sampleDDGIAtPoint wrapper.
 * Requires ddgiSample so fn ddgiSample is available for sampleDDGIAtPoint.
 *
 * NOTE: sampleDDGIAtPoint references ddgiIrradiance / ddgiVisibility, which
 * are declared in the CONSUMER shader's own body (@group(3) @binding(0..1)).
 * The WGSL composer emits required modules BEFORE the root's source (see
 * wgslComposer.ts), so those bindings are NOT yet declared when this module's
 * source is emitted. This means ddgiGridUbo CANNOT be a standalone composed root.
 * It must always be used as a REQUIRED module of a consumer that declares those
 * three bindings in its own source string — which is exactly the case for
 * risGi, risGiNrc, and shade. Naga resolves all declarations in the final
 * concatenated string, so forward references are fine.
 */
export const DDGI_GRID_UBO_MODULE: WgslModule = {
  name: 'ddgiGridUbo',
  source: DDGI_GRID_UBO_WGSL,
  requires: ['ddgiSample'],
};
