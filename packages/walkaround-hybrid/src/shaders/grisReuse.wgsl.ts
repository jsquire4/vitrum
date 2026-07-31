/**
 * GRIS helpers for walkaround's declared one-bounce DDGI proxy.
 *
 * Surface samples reconnect a stored DDGI suffix vertex to another receiver;
 * environment samples keep their stored direction with identity Jacobian.
 * Every viable technique is evaluated in a bounded all-domain matrix using
 * p_domain / J_domain_to_canonical. Invalid inverse maps and occluded edges have
 * exactly zero density. This is not a full path representation, does not reuse
 * multi-bounce transport, and does not claim an unbiased path-tracing estimator.
 *
 * The arithmetic is independently mirrored by the test-only
 * `__tests__/oracles/grisReuseMis.ts`.
 * Reference: Lin et al., Generalized Resampled Importance Sampling (2022).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GRIS_REUSE_WGSL = /* wgsl */ `// ============================================================
// GRIS reuse for the declared one-bounce DDGI proxy.
// Lin et al. 2022: every technique density is transformed into the
// canonical receiver measure with the inverse shift determinant.
// ============================================================

const GRIS_MAX_FINITE_F32: f32 = 3.402823466e38;
const GRIS_LOG_ZERO: f32 = -3.402823466e38;

fn grisFiniteVec3(value: vec3f) -> bool {
  return reservoirGiFinite(value.x)
    && reservoirGiFinite(value.y)
    && reservoirGiFinite(value.z);
}

fn grisSafeDirection(value: vec3f) -> vec3f {
  if (!grisFiniteVec3(value)) { return vec3f(0.0); }
  let scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
  if (!(scale > 0.0)) {
    return vec3f(0.0);
  }
  let scaled = value / scale;
  return scaled / length(scaled);
}

fn grisHistoryEpoch() -> u32 {
  return bitcast<u32>(ubo.sunAngular.y);
}

fn grisSampleKindValid(sampleKind: u32) -> bool {
  return sampleKind == GI_SAMPLE_SURFACE
    || sampleKind == GI_SAMPLE_ENVIRONMENT;
}

fn grisReconnectionGeometryTerm(x1: vec3f, x2: vec3f, n2: vec3f) -> f32 {
  if (!grisFiniteVec3(x1) || !grisFiniteVec3(x2) || !grisFiniteVec3(n2)) {
    return 0.0;
  }
  let d = x2 - x1;
  let distanceScale = max(abs(d.x), max(abs(d.y), abs(d.z)));
  let normalScale = max(abs(n2.x), max(abs(n2.y), abs(n2.z)));
  if (!(distanceScale > 0.0) || !(normalScale > 0.0)) { return 0.0; }
  let scaledDistance = d / distanceScale;
  let scaledNormal = n2 / normalScale;
  let distanceLength = length(scaledDistance);
  let normalLength = length(scaledNormal);
  let cosine = abs(dot(
    scaledNormal / normalLength,
    scaledDistance / distanceLength,
  ));
  if (!(cosine > 0.0) || !reservoirGiFinite(cosine)) { return 0.0; }
  let inverseDistanceScale = 1.0 / distanceScale;
  let result =
    cosine * inverseDistanceScale * inverseDistanceScale /
    (distanceLength * distanceLength);
  if (!reservoirGiFinite(result)) { return GRIS_MAX_FINITE_F32; }
  return max(result, 0.0);
}

fn grisDirection(sampleKind: u32, xv: vec3f, xs: vec3f, storedDirection: vec3f) -> vec3f {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
    return grisSafeDirection(storedDirection);
  }
  return grisSafeDirection(xs - xv);
}

fn grisMappedXs(sampleKind: u32, xv: vec3f, xs: vec3f, storedDirection: vec3f) -> vec3f {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
    return xv + grisSafeDirection(storedDirection) * walkaroundReconnectMaxDistance();
  }
  return xs;
}

fn grisMappedLo(sampleKind: u32, storedDirection: vec3f, storedLo: vec3f) -> vec3f {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
    let direction = grisSafeDirection(storedDirection);
    if (!(dot(direction, direction) > 0.0)) { return vec3f(0.0); }
    return envRadiance(direction);
  }
  return storedLo;
}

fn grisMaterialTargetAt(
  receiverSurface: PrimarySurface,
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
  storedLo: vec3f,
) -> f32 {
  let wi = grisDirection(sampleKind, xv, xs, storedDirection);
  var Lo = grisMappedLo(sampleKind, storedDirection, storedLo);
  if (
    sampleKind == GI_SAMPLE_ENVIRONMENT &&
    receiverSurface.hit &&
    length(receiverSurface.pos - xv) <= 5e-2
  ) {
    Lo = walkaroundScaleEnvironmentRadiance(
      Lo,
      receiverSurface.envMapIntensity,
    );
  }
  if (!grisFiniteVec3(nv) || !grisFiniteVec3(Lo)
   || !(dot(wi, wi) > 0.0)) { return 0.0; }
  let mappedXs = grisMappedXs(sampleKind, xv, xs, storedDirection);
  let result = restir_gi_receiver_phat_from_surface_or_geometry(
    receiverSurface,
    xv,
    nv,
    mappedXs,
    Lo,
  );
  return select(0.0, result, reservoirGiFinite(result) && result > 0.0);
}

fn grisProxyVisibilityAt(
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
) -> f32 {
  let wi = grisDirection(sampleKind, xv, xs, storedDirection);
  if (!grisFiniteVec3(xv) || !grisFiniteVec3(nv)
   || !(dot(wi, wi) > 0.0)) { return 0.0; }
  var tMax = INFINITY;
  if (sampleKind == GI_SAMPLE_SURFACE) {
    let d = safe_length(xs - xv);
    if (d <= walkaroundRayEndMargin()) { return 0.0; }
    tMax = d - walkaroundRayEndMargin();
  }
  if (max(0.0, dot(nv, wi)) <= 0.0) { return 0.0; }
  let tint = traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode, ubo.tlasNodeCount,

    xv + nv * walkaroundRayOriginBias(), wi, tMax, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
  );
  let visibility = luminance(tint);
  if (!reservoirGiFinite(visibility)) { return 0.0; }
  return clamp(visibility, 0.0, 1.0);
}

fn grisMaterialPHatAt(
  receiverSurface: PrimarySurface,
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
  storedLo: vec3f,
) -> f32 {
  let visibility = grisProxyVisibilityAt(xv, nv, sampleKind, xs, storedDirection);
  return grisMaterialTargetAt(
    receiverSurface,
    xv,
    nv,
    sampleKind,
    xs,
    storedDirection,
    storedLo,
  ) * visibility;
}

// |dT_domain_to_canonical|. Environment directions use the identity shift.
fn grisDomainToCanonicalJacobian(
  domainXv: vec3f,
  canonicalXv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  ns: vec3f,
) -> f32 {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) { return 1.0; }
  let gDomain = grisReconnectionGeometryTerm(domainXv, xs, ns);
  let gCanonical = grisReconnectionGeometryTerm(canonicalXv, xs, ns);
  if (gDomain <= 0.0 || gCanonical <= 0.0) { return 0.0; }
  let result = gCanonical / gDomain;
  return select(0.0, result, reservoirGiFinite(result) && result > 0.0);
}

// pHat_{<-domain}(y) = pHat_domain(T^-1(y)) * |dT^-1|
//                    = pHat_domain / |dT_domain_to_canonical|.
// Invalid inverse mappings and occluded techniques contribute exactly zero.
// log2(M * pHat / J), evaluated without first forming either quotient or
// product. A shared max-log scale is applied across the complete technique
// matrix, so extreme finite terms keep their relative mass instead of each
// saturating to the same cap.
fn grisLogWeightedTransformedDensity(
  attempts: u32,
  pHatDomain: f32,
  domainToCanonicalJacobian: f32,
  inverseValid: bool,
) -> f32 {
  if (attempts == 0u || !inverseValid
   || !reservoirGiFinite(pHatDomain) || !(pHatDomain > 0.0)
   || !reservoirGiFinite(domainToCanonicalJacobian)
   || !(domainToCanonicalJacobian > 0.0)) {
    return GRIS_LOG_ZERO;
  }
  let result =
    log2(f32(attempts))
    + log2(pHatDomain)
    - log2(domainToCanonicalJacobian);
  return select(
    GRIS_LOG_ZERO,
    result,
    reservoirGiFinite(result),
  );
}

fn grisScaledMass(logMass: f32, maxLogMass: f32) -> f32 {
  if (logMass == GRIS_LOG_ZERO || maxLogMass == GRIS_LOG_ZERO) {
    return 0.0;
  }
  let result = exp2(logMass - maxLogMass);
  return select(0.0, result, reservoirGiFinite(result) && result > 0.0);
}

fn grisLogCanonicalResamplingWeight(
  misWeight: f32,
  canonicalPHat: f32,
  sourceReservoirW: f32,
  sourceToCanonicalJacobian: f32,
) -> f32 {
  if (!reservoirGiFinite(misWeight) || !(misWeight > 0.0)
   || !reservoirGiFinite(canonicalPHat) || !(canonicalPHat > 0.0)
   || !reservoirGiFinite(sourceReservoirW) || !(sourceReservoirW > 0.0)
   || !reservoirGiFinite(sourceToCanonicalJacobian)
   || !(sourceToCanonicalJacobian > 0.0)) {
    return GRIS_LOG_ZERO;
  }
  let result =
    log2(misWeight)
    + log2(canonicalPHat)
    + log2(sourceReservoirW)
    + log2(sourceToCanonicalJacobian);
  return select(
    GRIS_LOG_ZERO,
    result,
    reservoirGiFinite(result),
  );
}

// The WRS selection sum is stored in a common max-log-scaled measure. Recover
// the estimator's unscaled W directly in log space, then apply the configured
// production cap. This keeps selection ratios exact without overflowing w_sum.
fn grisFinaliseLogScaledReservoir(
  r: ptr<function, ReservoirPT>,
  maxLogWeight: f32,
  scaledWeightSum: f32,
  wCap: f32,
) {
  (*r).W = 0.0;
  (*r).w_sum = scaledWeightSum;
  if ((*r).M == 0u
   || maxLogWeight == GRIS_LOG_ZERO
   || !reservoirGiFinite(scaledWeightSum) || !(scaledWeightSum > 0.0)
   || !reservoirGiFinite((*r).nativePHat) || !((*r).nativePHat > 0.0)
   || !reservoirGiFinite(wCap) || !(wCap > 0.0)) {
    return;
  }
  let logW =
    maxLogWeight
    + log2(scaledWeightSum)
    - log2((*r).nativePHat);
  if (!reservoirGiFinite(logW)) { return; }
  let logCap = log2(wCap);
  if (logW >= logCap) {
    (*r).W = wCap;
    return;
  }
  let value = exp2(logW);
  if (reservoirGiFinite(value) && value > 0.0) {
    (*r).W = value;
  }
}

`;

/** GRIS DDGI-proxy transformed-density WGSL include-graph entry. */
export const GRIS_REUSE_MODULE: WgslModule = {
  name: 'grisReuse',
  source: GRIS_REUSE_WGSL,
  requires: [
    'walkaroundUbo',
    'sharedPrimitives',
    'surfaceTextures',
    'environmentSample',
    'restirGiMaterial',
  ],
};
