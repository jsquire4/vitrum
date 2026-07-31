/**
 * reservoirPtHero.wgsl.ts — the FULL-RES (hero-stack) ReSTIR-PT / GRIS path
 * reservoir ADT for `@vitrum/pt-webgpu`, plus the hero target function and the
 * reconnection-shift module it consumes.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What this is (and what it is a port OF)
 * ════════════════════════════════════════════════════════════════════════════
 * This is the hero-stack generalization of the SHIPPING walkaround-hybrid
 * GRIS-GI reservoir (`@vitrum/walkaround-hybrid/src/shaders/reservoirGi.wgsl.ts`
 * — `struct ReservoirPT` + `RESERVOIR_GI_STRIDE` + the strided bitcast
 * load/store helpers + `updateReservoirGI` (streaming RIS) +
 * `finaliseGIReservoirWGris` (GRIS W = w_sum/p̂, NO /M)). The struct field set,
 * the bitcast-into-`array<u32>` serialization, the streaming-RIS update, and the
 * two finalize forms are MIRRORED from that proven module. This file widens the
 * struct for the hero stack (full-res, arbitrary visible-vertex material) and
 * single-homes the hero target `p̂` so the producer / temporal / resolve passes
 * all read the SAME definition.
 *
 * Ref: Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai — "Generalized
 * Resampled Importance Sampling: Foundations of ReSTIR", SIGGRAPH 2022 (GRIS);
 * Bitterli et al. 2020/2021 (ReSTIR DI/GI base reservoir + reconnection).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Implemented scope (temporal + spatial reconnection, prefix length 1)
 * ════════════════════════════════════════════════════════════════════════════
 * The reservoir stores a SINGLE-bounce reconnection sample:
 *   • xv / nv  — the VISIBLE vertex (the camera's first surface hit) and its
 *                shading normal. This is the path PREFIX (prefix length 1: the
 *                only pre-reconnection vertex is the primary hit).
 *   • xs / ns  — the RECONNECTION vertex: the surface hit by the ray bounced off
 *                xv, held FIXED in world space by the reconnection shift.
 *   • Lo       — the outgoing radiance LEAVING xs back toward xv (everything the
 *                suffix path gathers from xs onward; see the producer for the
 *                exact, energy-critical definition).
 *   • woV      — the native eye direction at xv in the sample's own domain.
 *   • primitive identity + surfaceParamV — motion-stable temporal
 *                correspondence (mesh barycentrics or analytic local position).
 *   • pdfSrc   — the ACTUAL source directional pdf that generated xv → xs
 *                (the visible-vertex BSDF sampling pdf). It is proposal metadata
 *                carried with the selected sample for shifted reuse and
 *                diagnostics; resolve never divides by it independently.
 *
 * The hero target p̂ (the resampling heuristic) is the INTEGRAND-MATCHING target —
 * the luminance of the real unshadowed reconnection contribution:
 *   p̂(z) = luminance( f_bsdf(xv; wo→wi) · max(0, cos(nv, wi)) · Lo ),  wi = xv→xs
 * (see `restirPtTargetAt`, using the visible-vertex BRDF — B3). It is a scalar
 * resampling heuristic; the W finalize divides it OUT (W = w_sum/p̂), so the
 * converged mean does NOT depend on it — only the resampling VARIANCE does, which
 * is exactly why matching the integrand (vs the old diffuse-cosine proxy) reduces
 * variance decisively for a glossy xv. The RESOLVE pass reconstructs the path
 * contribution with the SAME real BRDF (`evaluateBrdf`). (See the unbiasedness note.)
 *
 * ── Energy-consistency note ──────────────────────────────────────────────────
 * Let the producer sample the xv → xs edge from the visible-vertex BSDF with the true
 * directional pdf p_src, hit xs, and cache Lo. For that INITIAL, one-candidate
 * producer reservoir only, w = p̂/p_src and W = w_sum/p̂ = 1/p_src. Resolve then
 * forms the ordinary one-sample estimator f_bsdf·cos·Lo/p_src.
 *
 * Temporal/spatial reuse combines more than one source reservoir. Its finalized
 * W remains the GRIS normalization w_sum/p̂(selected), but is NOT generally
 * 1/pdfSrc for the selected sample. The generalized-balance weights, source
 * reservoir W values, and reconnection Jacobians are already folded into w_sum.
 * Resolve must therefore multiply the vector integrand by W exactly once and
 * must not introduce a separate /pdfSrc. The scalar target controls selection
 * variance while the GRIS normalization preserves the estimator. This holds for
 * a DIFFUSE *and* a GLOSSY visible vertex, PROVIDED:
 *   (1) the producer's candidate denominator is the REAL p_src (the
 *       visible-vertex BSDF directional pdf), not the cosine proxy, AND
 *   (2) the resolve uses the REAL evaluateBrdf at xv.
 * Both hold in this increment. The hero stack retains p_src so the producer
 * proposal remains explicit for non-Lambertian xv and so shifted reservoirs keep
 * the proposal metadata belonging to their selected sample.
 *
 * Reuse support is the stable finite, same-side reflection domain:
 *   • Opaque diffuse, glossy dielectric, metallic, clearcoat, and sheen
 *     vertices have finite connection support and are reusable.
 *   • Transmission is explicitly excluded. Delta or otherwise singular
 *     connections are also excluded by bsdfHasFiniteConnectionSupport.
 *
 * A prefix-1 shift has no randomly replayed prefix. It fixes xs and replaces
 * the source connection xq → xs with xr → xs, so the complete solid-angle
 * change of variables is exactly the half-G geometry ratio:
 *
 *     J = G(xr, xs) / G(xq, xs).
 *
 * The producer proposal density must not appear in J: the source reservoir W
 * already carries the complete prior GRIS normalization. Temporal and spatial
 * reuse re-evaluate the target in every participating domain.
 */

import { RESTIR_PT_SHIFT_WGSL } from './restirPtShift.wgsl.js';

/**
 * ReSTIR-PT runtime params UBO (the reuse unit's OWN tunables) + the reservoir
 * bind-group layout convention.
 *
 * The shared `FrameParams` (material.wgsl.ts @group(0) @binding(1)) is OWNED by
 * the megakernel and MUST NOT grow a ReSTIR-PT field for this increment (a
 * parallel agent owns that file). So the reuse passes declare their OWN small
 * uniform — `RestirPtParams` — in the ReSTIR-PT bind group (@group(4)). It
 * carries the temporal M-clamp (`restirGiMClamp` analogue). Contribution
 * weights are represented in log space by the reservoir itself; there is no
 * host-authored clamp and therefore no biased robustness mode.
 *
 * @group(4) layout (the ReSTIR-PT-specific resources, separate from the
 * inherited @group(0..3) the shared modules own):
 *   @binding(0) rpt_reservoirOut  (storage, read_write) — producer output
 *   @binding(1) rpt_reservoirCur  (storage, read_write) — temporal in/out
 *   @binding(2) rpt_reservoirPrev (storage, read)        — last frame's output
 *   @binding(3) rpt_result        (storage, read_write) — resolve output
 *   @binding(4) rptParams         (uniform)              — RestirPtParams
 * Each pass declares only the subset of @group(4) it uses (WGSL permits sparse
 * bindings); the wiring step builds the matching per-pass layout.
 */
export const RESTIR_PT_PARAMS_WGSL = /* wgsl */ `
struct RestirPtParams {
  mClamp:   u32,   // temporal M-clamp (GI restirGiMClamp analogue)
  _padA:    u32,
  _padB:    u32,
  _padC:    u32,
};
`;

/**
 * I4.3/I4.4 — Structural pin descriptor for `RestirPtParams`.
 *
 * Each entry is { name, byteOffset, type } mirroring the WGSL struct field
 * declarations above.  The host packer (`GpuResources.writeReservoirParams`)
 * writes these slots in this order; this descriptor lets a test assert:
 *   (a) the WGSL struct fields parse out in this order, and
 *   (b) the total byte size equals RESTIR_PT_PARAMS_BYTES (= 16).
 */
export const RESTIR_PT_PARAMS_FIELDS = [
  { name: 'mClamp', byteOffset: 0, type: 'u32' },
  { name: '_padA', byteOffset: 4, type: 'u32' },
  { name: '_padB', byteOffset: 8, type: 'u32' },
  { name: '_padC', byteOffset: 12, type: 'u32' },
] as const;

/** Byte size of the RestirPtParams UBO (4 × 4-byte fields). */
export const RESTIR_PT_PARAMS_BYTES = 16;

export const RESERVOIR_PT_HERO_WGSL = /* wgsl */ `// ============================================================
// ReSTIR-PT / GRIS hero reservoir.
//
// ReservoirPTHero is the rich FUNCTION-space representation used by the target
// and reuse math. Its storage representation is deliberately compact: exactly
// 64 bytes (16 × u32) per full-resolution pixel. Visible material parameters are
// deterministically rehydrated from primitive identity + surface coordinates;
// they are not duplicated in every frame-sized reservoir buffer.
// ============================================================
struct ReservoirPTHero {
  // ── reconnection sample (prefix length 1) ──
  xv:      vec3f,   // visible vertex (primary hit / path prefix)
  _pad0:   f32,
  nv:      vec3f,   // shading normal at xv
  logW:    f32,     // log RIS unbiased contribution weight (UCW)
  xs:      vec3f,   // reconnection vertex (held fixed by shift)
  logWeightSum: f32,// log running RIS weight sum
  ns:      vec3f,   // shading normal at xs
  M:       u32,     // confidence (candidate count; stored exactly through 4095)
  Lo:      vec3f,   // outgoing radiance LEAVING xs toward xv
  pdfSrc:  f32,     // producer-only proposal metadata; not serialized
  // ── native visible-domain identity (temporal correspondence) ──
  woV:               vec3f, // native eye direction at xv
  materialIdV:       u32,   // rehydrated from primitive identity
  instanceIndexV:    u32,   // TLAS instance or 0xffffffff sentinel
  roughnessV:        f32,
  metalV:            f32,
  transmissionV:     f32,
  iorV:              f32,
  // Wavelength belonging to the SELECTED reconnection sample.  It is part of
  // z, not a permanent property of the visible domain: temporal/spatial reuse
  // must carry it when a neighbour wins and re-evaluate the target domain's
  // material at this wavelength.
  heroLambdaV:       f32,
  isFrontFaceV:      bool,  // selects the authored front/back material layer
  // ── visible-vertex BRDF payload (resolve evaluates the FULL BRDF) ──
  albV:              vec3f,
  clearcoatV:        f32,
  clearcoatRoughnessV: f32,
  sheenV:            f32,
  sheenRoughnessV:   f32,
  sheenColorV:       vec3f,
  iridescenceV:      f32,
  iridescenceIorV:   f32,
  iridescenceThicknessMinV: f32,
  iridescenceThicknessMaxV: f32,
  anisotropyV:       f32,
  anisotropyRotationV: f32,
  specularColorV:    vec3f,
  specularIntensityV: f32,
  clearcoatNormalV:  vec3f,
  _padClearcoatNormalV: f32,
  // ── visible primitive-local identity ──
  triangleIndexV:    u32,
  surfaceParamV:     vec3f, // mesh bary(v,w), or normalized analytic local p
};

// Compact storage layout (64 bytes = 16 × u32):
//   [0..2] xv.xyz                  [3] logW
//   [4..6] xs.xyz                  [7] logWeightSum
//   [8]    oct16(nv)               [9] oct16(ns)
//   [10..11] shared-exponent 12-bit RGB Lo + 15-bit lambda + face + M[11:8]
//   [12]   oct16(woV)              [13] instanceIndexV
//   [14]   triangleIndexV          [15] surfaceParam[23:0] + M[7:0]
// Mesh surface coordinates use two UNORM12 barycentrics. Analytic coordinates
// use three UNORM8 values in a shape-normalized local box. M is exact through
// 4095; larger histories are deliberately saturated because the default clamp
// is 20 and confidence above 4095 is numerically counterproductive.
const RESERVOIR_PT_HERO_STRIDE: u32 = 16u;
const RPT_MAX_STORED_M: u32 = 4095u;
const RPT_MAX_FINITE_F32: f32 = 3.402823466e38;
const RPT_LOG_ZERO: f32 = -3.402823466e38;
const RPT_LOG_NUMERIC_FAILURE: f32 = 3.402823466e38;
const RPT_LOG_MAX_FINITE_F32: f32 = 88.7228390521;

fn emptyReservoirPTHero() -> ReservoirPTHero {
  var r: ReservoirPTHero;
  r.xv = vec3f(0.0); r.nv = vec3f(0,1,0);
  r.xs = vec3f(0.0); r.ns = vec3f(0,1,0);
  r.Lo = vec3f(0.0);
  r.logW = RPT_LOG_ZERO;
  r.logWeightSum = RPT_LOG_ZERO;
  r.M = 0u;
  r.pdfSrc = 0.0; r._pad0 = 0.0;
  r.woV = vec3f(0.0, 0.0, 1.0);
  r.materialIdV = 0xffffffffu;
  r.instanceIndexV = INVALID_TLAS_INSTANCE_INDEX;
  r.roughnessV = 0.0;
  r.metalV = 0.0;
  r.transmissionV = 0.0;
  r.iorV = 1.5;
  r.heroLambdaV = 550.0;
  r.isFrontFaceV = true;
  r.albV = vec3f(0.0);
  r.clearcoatV = 0.0;
  r.clearcoatRoughnessV = 0.0;
  r.sheenV = 0.0;
  r.sheenRoughnessV = 0.0;
  r.sheenColorV = vec3f(0.0);
  r.iridescenceV = 0.0;
  r.iridescenceIorV = 1.3;
  r.iridescenceThicknessMinV = 100.0;
  r.iridescenceThicknessMaxV = 400.0;
  r.anisotropyV = 0.0;
  r.anisotropyRotationV = 0.0;
  r.specularColorV = vec3f(1.0);
  r.specularIntensityV = 1.0;
  r.clearcoatNormalV = r.nv;
  r._padClearcoatNormalV = 0.0;
  r.triangleIndexV = 0xffffffffu;
  r.surfaceParamV = vec3f(0.0);
  return r;
}

struct RptVisibleMaterial {
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  clearcoatNormal: vec3f,
  isUnlit: bool,
}

// Canonical visible-material decode shared by the producer and storage
// rehydration. Keeping this in one function prevents the compact representation
// from drifting from the source sampler's texture/layer/spectral semantics.
fn rptVisibleMaterialAtSurface(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  instanceIndex: u32,
  shadingNormal: vec3f,
  isFrontFace: bool,
  heroLambda: f32,
) -> RptVisibleMaterial {
  let mat = decodeMaterial(matId);
  var out: RptVisibleMaterial;
  out.baseColor = mat.baseColor
    * sampleVertexColor(triIndex, baryVW).rgb
    * sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex).rgb;
  out.baseColor = out.baseColor
    * sampleAoFactor(matId, triIndex, baryVW, instanceIndex);
  let orm = sampleOrmTexture(matId, triIndex, baryVW, instanceIndex);
  out.roughness = clamp(mat.roughness * orm.g, 0.0, 1.0);
  out.metallic = clamp(mat.metallic * orm.b, 0.0, 1.0);
  out.transmission = clamp(
    mat.transmission
      * sampleTransmissionTexture(matId, triIndex, baryVW, instanceIndex),
    0.0, 1.0,
  );
  out.ior = mat.ior;
  if (params.spectralEnabled != 0u && mat.dispersionAbbe > 0.0) {
    out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
  }
  out.clearcoat = clamp(
    mat.clearcoat * sampleClearcoatTexture(matId, triIndex, baryVW, instanceIndex),
    0.0, 1.0,
  );
  out.clearcoatRoughness = clamp(
    mat.clearcoatRoughness
      * sampleClearcoatRoughnessTexture(matId, triIndex, baryVW, instanceIndex),
    0.0, 1.0,
  );
  out.sheen = mat.sheen;
  out.sheenRoughness = clamp(
    mat.sheenRoughness
      * sampleSheenRoughnessTexture(matId, triIndex, baryVW, instanceIndex),
    0.0, 1.0,
  );
  out.sheenColor = clamp(
    mat.sheenColor * sampleSheenColorTexture(matId, triIndex, baryVW, instanceIndex),
    vec3f(0.0), vec3f(1.0),
  );
  out.iridescence = clamp(
    mat.iridescence
      * sampleIridescenceTexture(matId, triIndex, baryVW, instanceIndex),
    0.0, 1.0,
  );
  let thicknessSample = sampleIridescenceThicknessTexture(
    matId, triIndex, baryVW, instanceIndex,
  );
  out.iridescenceThicknessMin = mat.iridescenceThicknessMin;
  out.iridescenceThicknessMax = mat.iridescenceThicknessMax;
  if (thicknessSample >= 0.0) {
    let thickness = mix(
      mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, thicknessSample,
    );
    out.iridescenceThicknessMin = thickness;
    out.iridescenceThicknessMax = thickness;
    if (thickness <= 0.0) { out.iridescence = 0.0; }
  }
  out.iridescenceIor = mat.iridescenceIor;
  out.specularColor = max(
    mat.specularColor
      * sampleSpecularColorTexture(matId, triIndex, baryVW, instanceIndex),
    vec3f(0.0),
  );
  out.specularIntensity = clamp(
    mat.specularIntensity
      * sampleSpecularIntensityTexture(matId, triIndex, baryVW, instanceIndex),
    0.0, 1.0,
  );
  if (params.spectralEnabled != 0u) {
    out.sheenColor = vec3f(spectralRgbFactorAtHero(out.sheenColor, heroLambda));
    out.specularColor =
      vec3f(spectralRgbFactorAtHero(out.specularColor, heroLambda));
  }
  out.anisotropy =
    materialAnisotropy(matId, triIndex, baryVW, instanceIndex);
  out.anisotropyRotation =
    materialAnisotropyRotation(matId, triIndex, baryVW, shadingNormal, instanceIndex);
  out.clearcoatNormal = applyClearcoatNormalMap(
    matId, triIndex, baryVW, shadingNormal, instanceIndex,
  );
  out.isUnlit = mat.isUnlit;

  let layerTx = clamp(
    select(mat.backLayerTx, mat.frontLayerTx, isFrontFace),
    vec3f(0.0), vec3f(1.0),
  );
  let layerRoughness = select(
    mat.backLayerRoughness, mat.frontLayerRoughness, isFrontFace,
  );
  if (layerRoughness >= 0.0) {
    out.roughness = clamp(layerRoughness, 0.0, 1.0);
  }
  let layerWeight = select(
    layerTx,
    activeLayerWeightRgb(layerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
  );
  out.baseColor = out.baseColor * layerWeight;
  if (params.spectralEnabled != 0u) {
    out.baseColor = vec3f(spectralCombinedReflectanceAtHero(
      out.baseColor,
      mat.baseColor,
      mat.spectralReflCoeffs,
      mat.hasSpectralReflectance,
      heroLambda,
    ));
  }
  return out;
}

// Map an analytic hit into a bounded shape-local box before its three
// coordinates are quantized. Every supported analytic intersector lies inside
// this box, so scene scale never leaks into the fixed-width representation.
fn rptNormalizeAnalyticSurfaceParam(
  analyticIndex: u32,
  localPoint: vec3f,
) -> vec3f {
  if (analyticIndex >= arrayLength(&analyticHeaders)) {
    return vec3f(0.0);
  }
  let header = analyticHeaders[analyticIndex];
  let shapeId = u32(max(header.x, 0.0));
  let paramOffset = u32(max(header.z, 0.0));
  let p0 = select(
    vec4f(0.0), analyticParams[paramOffset],
    paramOffset < arrayLength(&analyticParams),
  );
  let p1 = select(
    vec4f(0.0), analyticParams[paramOffset + 1u],
    paramOffset + 1u < arrayLength(&analyticParams),
  );
  var center = vec3f(0.0);
  var extent = vec3f(0.0);
  if (shapeId == SHAPE_SPHERE) {
    center = p0.xyz;
    extent = vec3f(p0.w);
  } else if (shapeId == SHAPE_BOX) {
    center = p0.xyz;
    extent = abs(p1.xyz);
  } else if (shapeId == SHAPE_CAPSULE) {
    center = 0.5 * p0.xyz + 0.5 * p1.xyz;
    extent = abs(0.5 * p1.xyz - 0.5 * p0.xyz) + vec3f(p1.w);
  } else if (shapeId == SHAPE_CYLINDER) {
    center = p0.xyz;
    extent = vec3f(p0.w, p1.x, p0.w);
  } else if (shapeId == SHAPE_H_CHANNEL_CAME) {
    extent = vec3f(
      0.5 * p0.x,
      0.5 * p0.z,
      0.5 * p0.y,
    );
  }
  return vec3f(
    rptRelativeCoordinate(localPoint.x, center.x, extent.x),
    rptRelativeCoordinate(localPoint.y, center.y, extent.y),
    rptRelativeCoordinate(localPoint.z, center.z, extent.z),
  );
}

fn rptPackOct16(direction: vec3f) -> u32 {
  return pack2x16snorm(octEncode(safe_normalize(direction)));
}

fn rptUnpackOct16(packed: u32) -> vec3f {
  return octDecode(unpack2x16snorm(packed));
}

struct RptDecodedLoMeta {
  Lo: vec3f,
  heroLambda: f32,
  isFrontFace: bool,
  mHigh: u32,
}

// Two words carry non-negative HDR Lo without a finite-range clamp: three
// 12-bit mantissas share an 8-bit base-2 exponent. The remaining 20 bits hold a
// 15-bit [380,780]nm wavelength, front/back layer bit, and M[11:8].
fn rptPackLoMeta(
  Lo: vec3f,
  heroLambda: f32,
  isFrontFace: bool,
  M: u32,
) -> vec2u {
  let finiteLo = select(
    vec3f(0.0), max(Lo, vec3f(0.0)),
    all(Lo == Lo) && all(abs(Lo) <= vec3f(3.402823466e38)),
  );
  let maxChannel = max(finiteLo.x, max(finiteLo.y, finiteLo.z));
  var exponentCode = 0u;
  var q = vec3u(0u);
  if (maxChannel > 0.0) {
    let exponent = clamp(
      i32(floor(log2(maxChannel))) + 1,
      // exp2(12-exponent) is the encode multiplier. -114 keeps its
      // largest value at exp2(126), finite on every conforming f32 path.
      -114,
      128,
    );
    exponentCode = u32(exponent + 127);
    let encodeScale = exp2(12.0 - f32(exponent));
    q = vec3u(clamp(
      round(finiteLo * encodeScale),
      vec3f(0.0),
      vec3f(4095.0),
    ));
  }
  let finiteHeroLambda = select(
    550.0,
    heroLambda,
    heroLambda == heroLambda && abs(heroLambda) <= 3.402823466e38,
  );
  let lambdaQ = u32(round(
    clamp((finiteHeroLambda - 380.0) / 400.0, 0.0, 1.0) * 32767.0,
  ));
  let word0 =
      (q.x & 0xfffu)
    | ((q.y & 0xfffu) << 12u)
    | ((q.z & 0xffu) << 24u);
  let word1 =
      ((q.z >> 8u) & 0xfu)
    | ((exponentCode & 0xffu) << 4u)
    | ((lambdaQ & 0x7fffu) << 12u)
    | (select(0u, 1u, isFrontFace) << 27u)
    | (((min(M, RPT_MAX_STORED_M) >> 8u) & 0xfu) << 28u);
  return vec2u(word0, word1);
}

fn rptUnpackLoMeta(word0: u32, word1: u32) -> RptDecodedLoMeta {
  var out: RptDecodedLoMeta;
  let q = vec3u(
    word0 & 0xfffu,
    (word0 >> 12u) & 0xfffu,
    ((word0 >> 24u) & 0xffu) | ((word1 & 0xfu) << 8u),
  );
  let exponentCode = (word1 >> 4u) & 0xffu;
  out.Lo = vec3f(0.0);
  if (exponentCode != 0u) {
    let exponent = i32(exponentCode) - 127;
    out.Lo = vec3f(q) * exp2(f32(exponent) - 12.0);
  }
  let lambdaQ = (word1 >> 12u) & 0x7fffu;
  out.heroLambda = 380.0 + 400.0 * (f32(lambdaQ) / 32767.0);
  out.isFrontFace = ((word1 >> 27u) & 1u) != 0u;
  out.mHigh = (word1 >> 28u) & 0xfu;
  return out;
}

fn rptCanonicalizeStoredLo(Lo: vec3f) -> vec3f {
  let packed = rptPackLoMeta(Lo, 550.0, true, 0u);
  return rptUnpackLoMeta(packed.x, packed.y).Lo;
}

fn rptPackSurfaceParam(r: ReservoirPTHero) -> u32 {
  if (r.triangleIndexV < params.triangleCount) {
    let q = vec2u(round(clamp(r.surfaceParamV.xy, vec2f(0.0), vec2f(1.0)) * 4095.0));
    return (q.x & 0xfffu) | ((q.y & 0xfffu) << 12u);
  }
  let q = vec3u(round(
    (clamp(r.surfaceParamV, vec3f(-1.0), vec3f(1.0)) * 0.5 + 0.5)
      * 255.0,
  ));
  return (q.x & 0xffu) | ((q.y & 0xffu) << 8u) | ((q.z & 0xffu) << 16u);
}

fn rptUnpackSurfaceParam(triangleIndex: u32, packed: u32) -> vec3f {
  if (triangleIndex < params.triangleCount) {
    var vw = vec2f(
      f32(packed & 0xfffu) / 4095.0,
      f32((packed >> 12u) & 0xfffu) / 4095.0,
    );
    // Independent nearest-integer quantization can move an edge sample one code
    // beyond v+w=1. Project that single-code overshoot back onto the simplex.
    let sumVW = vw.x + vw.y;
    if (sumVW > 1.0) { vw = vw / sumVW; }
    return vec3f(vw, 0.0);
  }
  return 2.0 * vec3f(
    f32(packed & 0xffu),
    f32((packed >> 8u) & 0xffu),
    f32((packed >> 16u) & 0xffu),
  ) / 255.0 - vec3f(1.0);
}

fn rptVisibleIdentityIsValid(r: ReservoirPTHero) -> bool {
  if (r.triangleIndexV < params.triangleCount) {
    if (r.triangleIndexV >= arrayLength(&indices)) { return false; }
    if (r.instanceIndexV == INVALID_TLAS_INSTANCE_INDEX) { return true; }
    if (r.instanceIndexV > 0x3fffffffu) { return false; }
    let transformBase = r.instanceIndexV * 4u;
    return transformBase + 3u < arrayLength(&tlasInstanceLocalToWorld)
        && transformBase + 3u < arrayLength(&tlasInstanceWorldToLocal);
  }
  let analyticIndex = r.triangleIndexV - params.triangleCount;
  return analyticIndex < params.analyticCount
      && analyticIndex < arrayLength(&analyticHeaders);
}

fn rptHydrateVisibleDomain(r: ptr<function, ReservoirPTHero>) {
  if ((*r).M == 0u) { return; }
  var matId = 0u;
  var baryVW = vec2f(0.0);
  if ((*r).triangleIndexV < params.triangleCount) {
    if ((*r).triangleIndexV < arrayLength(&triMaterialIds)) {
      matId = triMaterialIds[(*r).triangleIndexV];
    }
    baryVW = (*r).surfaceParamV.xy;
  } else {
    let analyticIndex = (*r).triangleIndexV - params.triangleCount;
    if (analyticIndex < arrayLength(&analyticHeaders)) {
      matId = u32(max(analyticHeaders[analyticIndex].y, 0.0));
    }
  }
  (*r).materialIdV = matId;
  let vm = rptVisibleMaterialAtSurface(
    matId,
    (*r).triangleIndexV,
    baryVW,
    (*r).instanceIndexV,
    (*r).nv,
    (*r).isFrontFaceV,
    (*r).heroLambdaV,
  );
  (*r).albV = vm.baseColor;
  (*r).roughnessV = vm.roughness;
  (*r).metalV = vm.metallic;
  (*r).transmissionV = vm.transmission;
  (*r).iorV = vm.ior;
  (*r).clearcoatV = vm.clearcoat;
  (*r).clearcoatRoughnessV = vm.clearcoatRoughness;
  (*r).sheenV = vm.sheen;
  (*r).sheenRoughnessV = vm.sheenRoughness;
  (*r).sheenColorV = vm.sheenColor;
  (*r).iridescenceV = vm.iridescence;
  (*r).iridescenceIorV = vm.iridescenceIor;
  (*r).iridescenceThicknessMinV = vm.iridescenceThicknessMin;
  (*r).iridescenceThicknessMaxV = vm.iridescenceThicknessMax;
  (*r).specularColorV = vm.specularColor;
  (*r).specularIntensityV = vm.specularIntensity;
  (*r).anisotropyV = vm.anisotropy;
  (*r).anisotropyRotationV = vm.anisotropyRotation;
  (*r).clearcoatNormalV = vm.clearcoatNormal;
}

fn loadReservoirPTHero_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirPTHero {
  let b = pixelIdx * RESERVOIR_PT_HERO_STRIDE;
  var r = emptyReservoirPTHero();
  r.xv = vec3f(
    bitcast<f32>(buf[b + 0u]),
    bitcast<f32>(buf[b + 1u]),
    bitcast<f32>(buf[b + 2u]),
  );
  r.logW = bitcast<f32>(buf[b + 3u]);
  r.xs = vec3f(
    bitcast<f32>(buf[b + 4u]),
    bitcast<f32>(buf[b + 5u]),
    bitcast<f32>(buf[b + 6u]),
  );
  r.logWeightSum = bitcast<f32>(buf[b + 7u]);
  r.nv = rptUnpackOct16(buf[b + 8u]);
  r.ns = rptUnpackOct16(buf[b + 9u]);
  let decodedLo = rptUnpackLoMeta(buf[b + 10u], buf[b + 11u]);
  r.Lo = decodedLo.Lo;
  r.heroLambdaV = decodedLo.heroLambda;
  r.isFrontFaceV = decodedLo.isFrontFace;
  r.woV = rptUnpackOct16(buf[b + 12u]);
  r.instanceIndexV = buf[b + 13u];
  r.triangleIndexV = buf[b + 14u];
  let surfaceAndM = buf[b + 15u];
  r.surfaceParamV = rptUnpackSurfaceParam(
    r.triangleIndexV, surfaceAndM & 0xffffffu,
  );
  r.M = ((decodedLo.mHigh & 0xfu) << 8u) | ((surfaceAndM >> 24u) & 0xffu);
  // pdfSrc is producer diagnostic metadata only. Reused GRIS weights consume W,
  // never a second proposal division, so it need not occupy persistent storage.
  r.pdfSrc = 1.0;
  if (r.M > 0u && !rptVisibleIdentityIsValid(r)) {
    return emptyReservoirPTHero();
  }
  rptHydrateVisibleDomain(&r);
  return r;
}

fn loadReservoirPTHero_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirPTHero {
  let b = pixelIdx * RESERVOIR_PT_HERO_STRIDE;
  var r = emptyReservoirPTHero();
  r.xv = vec3f(
    bitcast<f32>(buf[b + 0u]),
    bitcast<f32>(buf[b + 1u]),
    bitcast<f32>(buf[b + 2u]),
  );
  r.logW = bitcast<f32>(buf[b + 3u]);
  r.xs = vec3f(
    bitcast<f32>(buf[b + 4u]),
    bitcast<f32>(buf[b + 5u]),
    bitcast<f32>(buf[b + 6u]),
  );
  r.logWeightSum = bitcast<f32>(buf[b + 7u]);
  r.nv = rptUnpackOct16(buf[b + 8u]);
  r.ns = rptUnpackOct16(buf[b + 9u]);
  let decodedLo = rptUnpackLoMeta(buf[b + 10u], buf[b + 11u]);
  r.Lo = decodedLo.Lo;
  r.heroLambdaV = decodedLo.heroLambda;
  r.isFrontFaceV = decodedLo.isFrontFace;
  r.woV = rptUnpackOct16(buf[b + 12u]);
  r.instanceIndexV = buf[b + 13u];
  r.triangleIndexV = buf[b + 14u];
  let surfaceAndM = buf[b + 15u];
  r.surfaceParamV = rptUnpackSurfaceParam(
    r.triangleIndexV, surfaceAndM & 0xffffffu,
  );
  r.M = ((decodedLo.mHigh & 0xfu) << 8u) | ((surfaceAndM >> 24u) & 0xffu);
  r.pdfSrc = 1.0;
  if (r.M > 0u && !rptVisibleIdentityIsValid(r)) {
    return emptyReservoirPTHero();
  }
  rptHydrateVisibleDomain(&r);
  return r;
}

fn storeReservoirPTHero_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirPTHero) {
  let b = pixelIdx * RESERVOIR_PT_HERO_STRIDE;
  let storedM = min(r.M, RPT_MAX_STORED_M);
  let loMeta = rptPackLoMeta(
    r.Lo, r.heroLambdaV, r.isFrontFaceV, storedM,
  );
  buf[b + 0u] = bitcast<u32>(r.xv.x);
  buf[b + 1u] = bitcast<u32>(r.xv.y);
  buf[b + 2u] = bitcast<u32>(r.xv.z);
  buf[b + 3u] = bitcast<u32>(r.logW);
  buf[b + 4u] = bitcast<u32>(r.xs.x);
  buf[b + 5u] = bitcast<u32>(r.xs.y);
  buf[b + 6u] = bitcast<u32>(r.xs.z);
  buf[b + 7u] = bitcast<u32>(r.logWeightSum);
  buf[b + 8u] = rptPackOct16(r.nv);
  buf[b + 9u] = rptPackOct16(r.ns);
  buf[b + 10u] = loMeta.x;
  buf[b + 11u] = loMeta.y;
  buf[b + 12u] = rptPackOct16(r.woV);
  buf[b + 13u] = r.instanceIndexV;
  buf[b + 14u] = r.triangleIndexV;
  buf[b + 15u] =
    (rptPackSurfaceParam(r) & 0xffffffu) | ((storedM & 0xffu) << 24u);
}

// Streaming RIS reservoir update (Talbot 2005 / Bitterli 2020). Mirrors
// walkaround-hybrid's updateReservoirGI EXACTLY but carries the hero-specific
// pdfSrc alongside the chosen sample as proposal metadata for later shifted
// reuse. Resolve consumes the finalized GRIS W and never divides by pdfSrc.
// \`rand_f32\` is forward-referenced (composed earlier from PCG_WGSL).
fn rptFiniteScalar(value: f32) -> bool {
  return value == value && abs(value) <= RPT_MAX_FINITE_F32;
}

fn rptFinitePositive(value: f32) -> bool {
  return value > 0.0 && rptFiniteScalar(value);
}

fn rptLogPositive(value: f32) -> f32 {
  if (!rptFinitePositive(value)) { return RPT_LOG_ZERO; }
  return log(value);
}

fn rptLogAddExp(a: f32, b: f32) -> f32 {
  if (a == RPT_LOG_ZERO) { return b; }
  if (b == RPT_LOG_ZERO) { return a; }
  if (!rptFiniteScalar(a) || !rptFiniteScalar(b)) {
    return RPT_LOG_NUMERIC_FAILURE;
  }
  let hi = max(a, b);
  let lo = min(a, b);
  let result = hi + log(1.0 + exp(lo - hi));
  return select(RPT_LOG_NUMERIC_FAILURE, result, rptFiniteScalar(result));
}

fn rptReservoirHasNumericFailure(r: ReservoirPTHero) -> bool {
  return r.M == 0u && r.logW == RPT_LOG_NUMERIC_FAILURE;
}

fn rptMarkReservoirNumericFailure(r: ptr<function, ReservoirPTHero>) {
  (*r).M = 0u;
  (*r).logW = RPT_LOG_NUMERIC_FAILURE;
  (*r).logWeightSum = RPT_LOG_NUMERIC_FAILURE;
}

fn rptFiniteVec3(value: vec3f) -> bool {
  return rptFiniteScalar(value.x)
      && rptFiniteScalar(value.y)
      && rptFiniteScalar(value.z);
}

fn rptMaxAbs3(value: vec3f) -> f32 {
  return max(abs(value.x), max(abs(value.y), abs(value.z)));
}

// Finite Euclidean length without squaring world-space magnitudes. A zero
// return means either an exact zero vector or an unrepresentable/non-finite
// distance; callers that require a segment fail closed in both cases.
fn rptScaledLength(value: vec3f) -> f32 {
  let scale = rptMaxAbs3(value);
  if (!rptFinitePositive(scale)) { return 0.0; }
  let result = scale * length(value / scale);
  return select(0.0, result, rptFinitePositive(result));
}

// Ray offsets are relative to the f32 magnitudes that must remain
// distinguishable: the origin coordinates and the segment itself. 2^-18 is 32
// ulps around unit magnitude; bitcast(1u) is the exact smallest positive f32,
// used only when the relative product underflows.
fn rptWorldRayEpsilon(origin: vec3f, segmentLength: f32) -> f32 {
  let scale = max(rptMaxAbs3(origin), segmentLength);
  if (!rptFinitePositive(scale)) { return 0.0; }
  let epsilon = max(scale * 0.000003814697265625, bitcast<f32>(1u));
  return select(0.0, epsilon, rptFinitePositive(epsilon));
}

// |dot(delta, normal)| evaluated after equilibrating delta. This avoids both
// squared-distance overflow and tiny-scene underflow in coplanarity tests.
fn rptScaledAbsProjection(delta: vec3f, normal: vec3f) -> f32 {
  let scale = rptMaxAbs3(delta);
  if (scale == 0.0) { return 0.0; }
  if (!rptFinitePositive(scale) || !rptFiniteVec3(normal)) {
    return RPT_MAX_FINITE_F32;
  }
  let result = abs(dot(delta / scale, normal)) * scale;
  return select(RPT_MAX_FINITE_F32, result, rptFiniteScalar(result));
}

// (value-center)/extent without first subtracting possibly extreme or tiny
// authored coordinates. Invalid extents fail to the neutral identity value;
// valid analytic shapes are admitted with strictly positive finite extents.
fn rptRelativeCoordinate(value: f32, center: f32, extent: f32) -> f32 {
  let scale = max(abs(value), max(abs(center), abs(extent)));
  if (!rptFinitePositive(scale) || !rptFinitePositive(extent)) {
    return 0.0;
  }
  let denominator = extent / scale;
  if (!rptFinitePositive(denominator)) { return 0.0; }
  let result = (value / scale - center / scale) / denominator;
  return select(0.0, clamp(result, -1.0, 1.0), rptFiniteScalar(result));
}

fn rptSaturatingAddU32(a: u32, b: u32) -> u32 {
  let aStored = min(a, RPT_MAX_STORED_M);
  return aStored + min(b, RPT_MAX_STORED_M - aStored);
}

fn updateReservoirPTLog(
  r: ptr<function, ReservoirPTHero>,
  xs: vec3f, ns: vec3f, Lo: vec3f, heroLambda: f32, pdfSrc: f32,
  logWeight: f32,
  rng: ptr<function, PtRngState>,
) -> bool {
  // A finite log weight represents every positive f32-domain weight without
  // ever materializing a potentially overflowing product or quotient.
  if (!rptFiniteScalar(logWeight)
   || logWeight == RPT_LOG_ZERO
   || logWeight == RPT_LOG_NUMERIC_FAILURE
   || !rptFinitePositive(pdfSrc)
   || !rptFiniteVec3(xs)
   || !rptFiniteVec3(ns)
   || !rptFiniteVec3(Lo)
   || !rptFiniteScalar(heroLambda)
   || (params.spectralEnabled != 0u
       && (heroLambda < 380.0 || heroLambda > 780.0))
   || any(Lo < vec3f(0.0))) {
    return false;
  }
  // Recover a corrupted aggregate without discarding the current visible-domain
  // payload. The accepted candidate below necessarily replaces the stale sample
  // because its fresh sum equals its own positive weight.
  if ((*r).M == 0u || !rptFiniteScalar((*r).logWeightSum)
   || (*r).logWeightSum == RPT_LOG_NUMERIC_FAILURE) {
    (*r).M = 0u;
    (*r).logWeightSum = RPT_LOG_ZERO;
    (*r).logW = RPT_LOG_ZERO;
  }
  if ((*r).M >= RPT_MAX_STORED_M) { return false; }
  let nextLogWeightSum = rptLogAddExp((*r).logWeightSum, logWeight);
  if (!rptFiniteScalar(nextLogWeightSum)
   || nextLogWeightSum == RPT_LOG_NUMERIC_FAILURE) {
    rptMarkReservoirNumericFailure(r);
    return false;
  }
  (*r).M = (*r).M + 1u;
  (*r).logWeightSum = nextLogWeightSum;
  let selectionProbability = exp(logWeight - nextLogWeightSum);
  if (rand_f32(rng) < selectionProbability) {
    (*r).xs = xs;
    (*r).ns = ns;
    // Canonicalize before target/finalize evaluation. W is therefore computed
    // against the exact HDR value that survives the 64-byte serialization.
    (*r).Lo = rptCanonicalizeStoredLo(Lo);
    // The sampled wavelength and Lo are one indivisible sample.  Rehydrate the
    // CURRENT visible domain at the winning wavelength so the final target and
    // resolve never combine lambda_q radiance with lambda_r material factors.
    (*r).heroLambdaV = heroLambda;
    rptHydrateVisibleDomain(r);
    (*r).pdfSrc = pdfSrc;
  }
  return true;
}

fn copyReservoirPTVisibleDomain(dst: ptr<function, ReservoirPTHero>, src: ReservoirPTHero) {
  (*dst).xv = src.xv;
  (*dst).nv = src.nv;
  (*dst).woV = src.woV;
  (*dst).materialIdV = src.materialIdV;
  (*dst).instanceIndexV = src.instanceIndexV;
  (*dst).triangleIndexV = src.triangleIndexV;
  (*dst).surfaceParamV = src.surfaceParamV;
  (*dst).albV = src.albV;
  (*dst).roughnessV = src.roughnessV;
  (*dst).metalV = src.metalV;
  (*dst).transmissionV = src.transmissionV;
  (*dst).iorV = src.iorV;
  (*dst).heroLambdaV = src.heroLambdaV;
  (*dst).isFrontFaceV = src.isFrontFaceV;
  (*dst).clearcoatV = src.clearcoatV;
  (*dst).clearcoatRoughnessV = src.clearcoatRoughnessV;
  (*dst).sheenV = src.sheenV;
  (*dst).sheenRoughnessV = src.sheenRoughnessV;
  (*dst).sheenColorV = src.sheenColorV;
  (*dst).iridescenceV = src.iridescenceV;
  (*dst).iridescenceIorV = src.iridescenceIorV;
  (*dst).iridescenceThicknessMinV = src.iridescenceThicknessMinV;
  (*dst).iridescenceThicknessMaxV = src.iridescenceThicknessMaxV;
  (*dst).anisotropyV = src.anisotropyV;
  (*dst).anisotropyRotationV = src.anisotropyRotationV;
  (*dst).specularColorV = src.specularColorV;
  (*dst).specularIntensityV = src.specularIntensityV;
  (*dst).clearcoatNormalV = src.clearcoatNormalV;
}

// Reconstruct the coherent interface from compact visible-domain identity.
// Layer records live in the material table and therefore need not consume
// reservoir storage; ior/transmission are rehydrated at the selected hero
// wavelength alongside the other function-space BSDF parameters.
fn rptThinFilmForDomain(r: ReservoirPTHero) -> ThinFilmInterface {
  if (r.materialIdV == 0xffffffffu) { return bsdfNoThinFilm(); }
  let mat = decodeMaterial(r.materialIdV);
  return ThinFilmInterface(
    mat.thinFilmEnabled,
    r.materialIdV,
    mat.thinFilmLayerCountU,
    mat.thinFilmIncidentIor,
    r.iorV,
    mat.thinFilmAngleDependent,
    r.isFrontFaceV,
    params.spectralEnabled != 0u,
    r.heroLambdaV,
    r.transmissionV,
  );
}

// The hero target function p̂ in the domain whose visible vertex is xv:
//   p̂(z) = luminance( f_bsdf(xv; wo→wi) · max(0, cos(nv, wi)) · Lo ),  wi = xv→xs
// the INTEGRAND-MATCHING target (the luminance of the real unshadowed reconnection
// contribution, using the visible-vertex BRDF) — NOT the diffuse-cosine proxy the
// diffuse-only GI version uses. It is a SCALAR resampling heuristic only — W =
// w_sum/p̂ supplies the GRIS normalization, so the converged mean is INDEPENDENT
// of the target choice (for the initial one-candidate producer, this reduces to
// W = 1/p_src; reused reservoirs generally do not). The RESOLVE pass likewise
// reconstructs with the REAL evaluateBrdf at xv. Matching the integrand
// only reduces RESAMPLING VARIANCE — decisively for a GLOSSY visible vertex whose
// direction-sensitive BRDF the old cosine proxy mis-weighted (the documented prefix-1
// glossy drift, fixed here). Returns 0 on a degenerate / back-facing edge.
fn restirPtTargetAt(
  xv: vec3f,
  nv: vec3f,
  wo: vec3f,
  albV: vec3f,
  roughnessV: f32,
  metalV: f32,
  clearcoatNormalV: vec3f,
  clearcoatV: f32,
  clearcoatRoughnessV: f32,
  sheenV: f32,
  sheenRoughnessV: f32,
  sheenColorV: vec3f,
  iridescenceV: f32,
  iridescenceIorV: f32,
  iridescenceThicknessMinV: f32,
  iridescenceThicknessMaxV: f32,
  specularColorV: vec3f,
  specularIntensityV: f32,
  anisotropyV: f32,
  anisotropyRotationV: f32,
  thinFilmV: ThinFilmInterface,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  if (!rptFiniteVec3(xv) || !rptFiniteVec3(nv) || !rptFiniteVec3(wo)
   || !rptFiniteVec3(xs) || !rptFiniteVec3(Lo)) {
    return 0.0;
  }
  let d = xs - xv;
  let edgeScale = rptMaxAbs3(d);
  if (!rptFinitePositive(edgeScale)) { return 0.0; }
  let scaledEdge = d / edgeScale;
  let scaledDist2 = dot(scaledEdge, scaledEdge);
  if (!rptFinitePositive(scaledDist2)) { return 0.0; }
  let wi = scaledEdge * inverseSqrt(scaledDist2);
  let cosTheta = max(0.0, dot(nv, wi));
  if (cosTheta <= 0.0) { return 0.0; }
  let f = evaluateBrdfFullWithClearcoatNormal(
    albV, roughnessV, metalV, nv, clearcoatNormalV, wo, wi,
    clearcoatV, clearcoatRoughnessV, sheenV, sheenRoughnessV, sheenColorV,
    iridescenceV, iridescenceIorV, iridescenceThicknessMinV, iridescenceThicknessMaxV,
    specularColorV, specularIntensityV,
    anisotropyV, anisotropyRotationV, thinFilmV,
  );
  let targetValue = luminance(f * cosTheta * Lo);
  if (!rptFinitePositive(targetValue)) { return 0.0; }
  return targetValue;
}

fn restirPtTargetForDomain(r: ReservoirPTHero, wo: vec3f, xs: vec3f, Lo: vec3f) -> f32 {
  let thinFilm = rptThinFilmForDomain(r);
  return restirPtTargetAt(
    r.xv, r.nv, wo, r.albV, r.roughnessV, r.metalV,
    r.clearcoatNormalV,
    r.clearcoatV, r.clearcoatRoughnessV, r.sheenV, r.sheenRoughnessV, r.sheenColorV,
    r.iridescenceV, r.iridescenceIorV, r.iridescenceThicknessMinV, r.iridescenceThicknessMaxV,
    r.specularColorV, r.specularIntensityV,
    r.anisotropyV, r.anisotropyRotationV, thinFilm,
    xs, Lo,
  );
}

// Evaluate a sample in an arbitrary visible domain using the SAMPLE'S hero
// wavelength.  A reused sample may have been generated in another frame/pixel;
// using r.heroLambdaV here would cross-multiply Lo(lambda_sample) by a BSDF and
// texture/layer payload evaluated at lambda_domain.  RGB mode takes the cheap
// identity path.
fn restirPtTargetForDomainAtHero(
  r: ReservoirPTHero,
  sampleHeroLambda: f32,
  wo: vec3f,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  var domain = r;
  if (params.spectralEnabled != 0u) {
    if (!rptFiniteScalar(sampleHeroLambda)
     || sampleHeroLambda < 380.0 || sampleHeroLambda > 780.0) {
      return 0.0;
    }
    domain.heroLambdaV = sampleHeroLambda;
    rptHydrateVisibleDomain(&domain);
  }
  return restirPtTargetForDomain(domain, wo, xs, Lo);
}

// Prefix length one has no random-replayed bounce: xv→xs is itself the
// reconnection edge. Its solid-angle change of variables is therefore exactly
// the half-G ratio. Proposal density is already carried by the source reservoir W;
// multiplying by a BSDF-pdf ratio here would count the proposal conversion twice.
fn restirPtReconnectionJacobianForPair(
  source: ReservoirPTHero,
  targetDomain: ReservoirPTHero,
) -> f32 {
  return restirPtShiftJacobian(
    source.xv, targetDomain.xv, source.xs, source.ns,
  );
}

fn rptLogWeightedTarget(candidateCount: f32, targetValue: f32) -> f32 {
  if (!rptFinitePositive(candidateCount) || !rptFinitePositive(targetValue)) {
    return RPT_LOG_ZERO;
  }
  return log(candidateCount) + log(targetValue);
}

// Express one target-domain proxy density in the SOURCE sample's measure.
// For T_source→target this is c_target·p̂_target(T(x))·|∂T/∂x|.  These
// Jacobian-corrected terms are commensurable and therefore may share a
// generalized-balance denominator.  Omitting J compares densities in distinct
// path-space measures and breaks the required partition of unity.
fn rptLogWeightedShiftedTarget(
  candidateCount: f32,
  targetValue: f32,
  sourceToTargetJacobian: f32,
) -> f32 {
  let logWeightedTarget = rptLogWeightedTarget(candidateCount, targetValue);
  if (logWeightedTarget == RPT_LOG_ZERO
   || !rptFinitePositive(sourceToTargetJacobian)) {
    return RPT_LOG_ZERO;
  }
  let result = logWeightedTarget + log(sourceToTargetJacobian);
  return select(RPT_LOG_NUMERIC_FAILURE, result, rptFiniteScalar(result));
}

// ── Two-domain generalized-balance denominators (Lin 2022, Eq. 11). ──
// Each denominator is written in its candidate's native measure.  The native
// term has identity J; the cross-domain term carries the candidate→other-domain
// determinant.  The caller forms log(m_i) = log(native mass) - log(denom).
fn restirPtPairwiseLogDenomNeighbor(
  cR: f32, pHatR_atQsample: f32,
  qToRJacobian: f32,
  cQ: f32, pHatQ_native: f32,
) -> f32 {
  return rptLogAddExp(
    rptLogWeightedShiftedTarget(cR, pHatR_atQsample, qToRJacobian),
    rptLogWeightedTarget(cQ, pHatQ_native),
  );
}

fn restirPtPairwiseLogDenomCanonical(
  cR: f32, pHatR_native: f32,
  cQ: f32, pHatQ_atRsample: f32,
  rToQJacobian: f32,
) -> f32 {
  return rptLogAddExp(
    rptLogWeightedTarget(cR, pHatR_native),
    rptLogWeightedShiftedTarget(cQ, pHatQ_atRsample, rToQJacobian),
  );
}

// Finalise log(W) for a GRIS hero reservoir. GRIS folds each
// sample with a pairwise-MIS weight m_i where Σ m_i = 1, so the M-count does NOT
// normalise the sum — dividing by M again would under-energise the estimate.
//   log W = log(weight_sum) - log(p̂(chosen sample))
//                                      — Lin 2022 §generalized RIS, NO /M.
// Mirrors walkaround-hybrid finaliseGIReservoirWGris EXACTLY (the GRIS form),
// with the hero target p̂ via restirPtTargetAt. Keeping the normalization in
// log space avoids both overflow and any biased contribution clamp.
fn finaliseReservoirPTWGris(r: ptr<function, ReservoirPTHero>) {
  (*r).logW = RPT_LOG_ZERO;
  if ((*r).M == 0u || !rptFiniteScalar((*r).logWeightSum)
   || (*r).logWeightSum == RPT_LOG_ZERO
   || (*r).logWeightSum == RPT_LOG_NUMERIC_FAILURE) {
    (*r).M = 0u;
    return;
  }
  let pHatF = restirPtTargetForDomain((*r), (*r).woV, (*r).xs, (*r).Lo);
  if (!rptFinitePositive(pHatF) || pHatF <= 1e-9) {
    (*r).M = 0u;
    return;
  }
  let logW = (*r).logWeightSum - log(pHatF);
  if (!rptFiniteScalar(logW)) {
    rptMarkReservoirNumericFailure(r);
    return;
  }
  (*r).logW = logW;
}

// The implemented ReSTIR-PT contract has exactly one reconnection edge. Reject
// an invalid selected edge by emptying the reservoir; a transient prefix flag
// would not survive the compact 16-word store and therefore cannot be a valid
// cross-pass gate.
fn refreshReconnectionStatePT(r: ptr<function, ReservoirPTHero>) {
  let toRecon = (*r).xs - (*r).xv;
  let reconScale = max(
    abs(toRecon.x),
    max(abs(toRecon.y), abs(toRecon.z)),
  );
  if (!rptFinitePositive(reconScale)
   || (*r).M == 0u || !rptFiniteScalar((*r).logW)
   || (*r).logW == RPT_LOG_ZERO
   || (*r).logW == RPT_LOG_NUMERIC_FAILURE) {
    (*r).M = 0u;
    (*r).logW = RPT_LOG_ZERO;
    (*r).logWeightSum = RPT_LOG_ZERO;
  }
}
`;

/**
 * The reservoir module composed with its prefix-1 reconnection Jacobian.
 * `RESTIR_PT_SHIFT_WGSL` supplies the FD-validated half-G geometry ratio used
 * by temporal and spatial reuse.
 */
export const RESERVOIR_PT_HERO_WITH_SHIFT_WGSL = /* wgsl */ `
${RESTIR_PT_SHIFT_WGSL}
${RESTIR_PT_PARAMS_WGSL}
${RESERVOIR_PT_HERO_WGSL}
`;
