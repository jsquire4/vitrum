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
 * parser sees one entry function with exactly 16 inputs, matching the
 * call site's 16 arguments.
 */

import { IRR_CELL, VIS_CELL, IRR_STRIDE, VIS_STRIDE } from './ddgiAtlasLayout.js';
import type { WgslModule } from '../pipeline/wgslComposer.js';

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
  samp: sampler,
  gridOriginX: f32, gridOriginY: f32, gridOriginZ: f32,
  gridSpacing: f32,
  gridDimsX: u32, gridDimsY: u32, gridDimsZ: u32,
  irrW: f32, irrH: f32, visW: f32, visH: f32,
) -> vec3f {
  let gridOrigin = vec3f(gridOriginX, gridOriginY, gridOriginZ);
  let gridDims   = vec3u(gridDimsX, gridDimsY, gridDimsZ);

  // Surface/normal bias (Majercik 2019 §4) — offset the receiver OFF the surface
  // toward the probe interior before sampling. CRITICAL: without it a receiver
  // sampled at the EXACT surface that lies on a grid-boundary plane (Cornell
  // walls/floors — gridPos on an integer plane) has its trilinear interpolation
  // collapse onto the IN-PLANE probes, whose direction to the receiver is
  // perpendicular to the surface normal → the cosine weight max(0, dot(n, dir))
  // is 0 for all of them → totalWeight < 1e-4 → this returns vec3f(0). That
  // zeroed the entire DDGI→ReSTIR-GI handoff (the GI reconnection points are
  // surface points), so the default realtime GI was DEAD (indirect ~0; only the
  // off-default RC path produced GI). Root-caused 2026-06-07 via gidiag A/B:
  // offsetting the sample point lifted indirect 0.0004→0.25. A quarter-spacing
  // bias clears the boundary plane while staying inside the receiver's cell; the
  // Chebyshev visibility term still guards against light leak through thin walls.
  let biasedPos = worldPos + surfaceNormal * (gridSpacing * 0.25);

  let gridPos  = (biasedPos - gridOrigin) / gridSpacing;
  let baseIdx3 = vec3i(floor(gridPos));
  let frac     = fract(gridPos);

  var sum         = vec3f(0.0);
  var totalWeight = 0.0;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let co  = vec3u((i & 1u), (i >> 1u) & 1u, (i >> 2u) & 1u);
    let pi3 = baseIdx3 + vec3i(co);
    if (any(pi3 < vec3i(0)) || any(pi3 >= vec3i(gridDims))) { continue; }

    let probeFlatIdx = u32(pi3.x) +
                       u32(pi3.y) * gridDims.x +
                       u32(pi3.z) * gridDims.x * gridDims.y;
    let probeWorld   = gridOrigin + vec3f(pi3) * gridSpacing;

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
    let toProbe   = probeWorld - biasedPos;
    let probeDist = length(toProbe);

    // Octahedral-encode the surface→probe direction (visibility lookup).
    let probeDirToSurf = normalize(biasedPos - probeWorld);
    let dirV       = -probeDirToSurf;
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
    let vis       = textureSampleLevel(visibilityAtlas, samp, visUv, 0.0).rg;
    let mean      = vis.x;
    let variance  = abs(vis.y - mean * mean);
    let chebyshev = select(
      variance / (variance + max(0.0, probeDist - mean) * max(0.0, probeDist - mean)),
      1.0,
      probeDist <= mean,
    );
    w = w * max(chebyshev, 0.0);

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
    let irr =
        textureLoad(irradianceAtlas, vec2u(ix + 0u, iy + 0u), 0).rgb * 0.282095
      + textureLoad(irradianceAtlas, vec2u(ix + 1u, iy + 0u), 0).rgb * (0.488603 * surfaceNormal.y)
      + textureLoad(irradianceAtlas, vec2u(ix + 2u, iy + 0u), 0).rgb * (0.488603 * surfaceNormal.z)
      + textureLoad(irradianceAtlas, vec2u(ix + 0u, iy + 1u), 0).rgb * (0.488603 * surfaceNormal.x)
      + textureLoad(irradianceAtlas, vec2u(ix + 1u, iy + 1u), 0).rgb * (1.092548 * surfaceNormal.x * surfaceNormal.y)
      + textureLoad(irradianceAtlas, vec2u(ix + 2u, iy + 1u), 0).rgb * (1.092548 * surfaceNormal.y * surfaceNormal.z)
      + textureLoad(irradianceAtlas, vec2u(ix + 0u, iy + 2u), 0).rgb * (0.315392 * (3.0 * surfaceNormal.z * surfaceNormal.z - 1.0))
      + textureLoad(irradianceAtlas, vec2u(ix + 1u, iy + 2u), 0).rgb * (1.092548 * surfaceNormal.x * surfaceNormal.z)
      + textureLoad(irradianceAtlas, vec2u(ix + 2u, iy + 2u), 0).rgb * (0.546274 * (surfaceNormal.x * surfaceNormal.x - surfaceNormal.y * surfaceNormal.y));

    sum         = sum + irr * w;
    totalWeight = totalWeight + w;
  }

  if (totalWeight < 1e-4) {
    // No probes contributed — return zero indirect (conservative, matches
    // shade.wgsl ddgiSampleFromBindings; was vec3f(0.05) prior to consolidation).
    return vec3f(0.0);
  }
  // Each probe's SH eval already returns irradiance E (the cosine convolution
  // is baked into the stored coefficients at blend time), so the trilinear-
  // weighted average across probes IS the irradiance — no *PI reconstruction.
  return sum / totalWeight;
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
// shade.wgsl: the @group(3) @binding(0..2) irradiance/visibility/sampler
// bindings remain in shade's own body; only the struct + @binding(3) are here.
// The binding sequence 0-1-2 (shade body) and 3 (this module, emitted first)
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
// The ddgiIrradiance / ddgiVisibility / ddgiSampler bindings are declared in
// the per-shader body (@group(3) @binding(0..2)) — those are in scope here
// because the composer emits this module BEFORE the consumer's own source.
fn sampleDDGIAtPoint(worldPos: vec3f, surfaceNormal: vec3f) -> vec3f {
  return ddgiSample(
    worldPos, surfaceNormal,
    ddgiIrradiance, ddgiVisibility, ddgiSampler,
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
 * NOTE: sampleDDGIAtPoint references ddgiIrradiance / ddgiVisibility / ddgiSampler
 * which are declared in the CONSUMER shader's own body (@group(3) @binding(0..2)).
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
