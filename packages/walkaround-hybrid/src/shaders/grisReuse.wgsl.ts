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
const GRIS_LOG_ZERO: f32 = RESERVOIR_GI_LOG_ZERO;

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

fn grisSafeDirectionBetween(fromPoint: vec3f, toPoint: vec3f) -> vec3f {
  if (!grisFiniteVec3(fromPoint) || !grisFiniteVec3(toPoint)) {
    return vec3f(0.0);
  }
  let coordinateScale = max(
    max(abs(fromPoint.x), max(abs(fromPoint.y), abs(fromPoint.z))),
    max(abs(toPoint.x), max(abs(toPoint.y), abs(toPoint.z))),
  );
  if (!(coordinateScale > 0.0)) { return vec3f(0.0); }
  return grisSafeDirection(
    toPoint / coordinateScale - fromPoint / coordinateScale,
  );
}

// ns is face-forwarded toward the source receiver when the suffix hit is
// created. Reconnection may move the receiver within that hemisphere, but a
// move across the surface would change front/back layers, one-sided emission,
// and the oriented mapped normal. Those alternate suffix states are not stored.
fn grisSurfaceSuffixReceiverSupported(
  receiverXv: vec3f,
  xs: vec3f,
  ns: vec3f,
) -> bool {
  let towardReceiver = grisSafeDirectionBetween(xs, receiverXv);
  let suffixNormal = grisSafeDirection(ns);
  return
    dot(towardReceiver, towardReceiver) > 0.0 &&
    dot(suffixNormal, suffixNormal) > 0.0 &&
    dot(suffixNormal, towardReceiver) > 0.0;
}

fn grisHistoryEpoch() -> u32 {
  return bitcast<u32>(ubo.sunAngular.y);
}

fn grisReconnectionGeometryTerm(x1: vec3f, x2: vec3f, n2: vec3f) -> f32 {
  let logResult = grisLogReconnectionGeometryTerm(x1, x2, n2);
  return reservoirGiRepresentPositiveLog(logResult);
}

fn grisLogReconnectionGeometryTerm(x1: vec3f, x2: vec3f, n2: vec3f) -> f32 {
  if (!grisFiniteVec3(x1) || !grisFiniteVec3(x2) || !grisFiniteVec3(n2)) {
    return GRIS_LOG_ZERO;
  }
  let coordinateScale = max(
    max(abs(x1.x), max(abs(x1.y), abs(x1.z))),
    max(abs(x2.x), max(abs(x2.y), abs(x2.z))),
  );
  let normalScale = max(abs(n2.x), max(abs(n2.y), abs(n2.z)));
  if (!(coordinateScale > 0.0) || !(normalScale > 0.0)) { return GRIS_LOG_ZERO; }
  let coordinateDelta = x2 / coordinateScale - x1 / coordinateScale;
  let deltaScale = max(
    abs(coordinateDelta.x),
    max(abs(coordinateDelta.y), abs(coordinateDelta.z)),
  );
  if (!(deltaScale > 0.0)) { return GRIS_LOG_ZERO; }
  let scaledDistance = coordinateDelta / deltaScale;
  let scaledNormal = n2 / normalScale;
  let distanceLength = length(scaledDistance);
  let normalLength = length(scaledNormal);
  let cosine = abs(dot(
    scaledNormal / normalLength,
    scaledDistance / distanceLength,
  ));
  if (!(cosine > 0.0) || !reservoirGiFinite(cosine)) { return GRIS_LOG_ZERO; }
  let result =
    log2(cosine)
    - 2.0 * (log2(coordinateScale) + log2(deltaScale))
    - 2.0 * log2(distanceLength);
  return select(GRIS_LOG_ZERO, result, reservoirGiFinite(result));
}

fn grisDirection(sampleKind: u32, xv: vec3f, xs: vec3f, storedDirection: vec3f) -> vec3f {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
    return grisSafeDirection(storedDirection);
  }
  return grisSafeDirectionBetween(xv, xs);
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
  return luminance(grisMaterialContributionAt(
    receiverSurface,
    xv,
    nv,
    sampleKind,
    xs,
    storedDirection,
    storedLo,
  ));
}

fn grisMaterialContributionAt(
  receiverSurface: PrimarySurface,
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
  storedLo: vec3f,
) -> vec3f {
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
   || !(dot(wi, wi) > 0.0)) { return vec3f(0.0); }
  let mappedXs = grisMappedXs(sampleKind, xv, xs, storedDirection);
  let result = restir_gi_receiver_contribution_from_surface_or_geometry(
    receiverSurface,
    xv,
    nv,
    mappedXs,
    Lo,
  );
  return select(
    vec3f(0.0),
    result,
    grisFiniteVec3(result) && any(result > vec3f(0.0)),
  );
}

fn grisProxyTintAt(
  xv: vec3f,
  nv: vec3f,
  geoNormal: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
) -> vec3f {
  let wi = grisDirection(sampleKind, xv, xs, storedDirection);
  if (!grisFiniteVec3(xv) || !grisFiniteVec3(nv)
   || !grisFiniteVec3(geoNormal)
   || !(dot(wi, wi) > 0.0)) { return vec3f(0.0); }
  var tMax = INFINITY;
  if (sampleKind == GI_SAMPLE_SURFACE) {
    let d = safe_length(xs - xv);
    if (d <= walkaroundRayEndMargin()) { return vec3f(0.0); }
    tMax = d - walkaroundRayEndMargin();
  }
  if (max(0.0, dot(nv, wi)) <= 0.0) { return vec3f(0.0); }
  let tint = traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode, ubo.tlasNodeCount,

    xv + geoNormal * walkaroundRayOriginBias(), wi, tMax, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
  );
  if (!grisFiniteVec3(tint)) { return vec3f(0.0); }
  return clamp(tint, vec3f(0.0), vec3f(1.0));
}

fn grisProxyVisibilityAt(
  xv: vec3f,
  nv: vec3f,
  geoNormal: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
) -> f32 {
  return luminance(grisProxyTintAt(
    xv,
    nv,
    geoNormal,
    sampleKind,
    xs,
    storedDirection,
  ));
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
  return reservoirGiRepresentPositiveLog(grisLogMaterialPHatAt(
    receiverSurface,
    xv,
    nv,
    sampleKind,
    xs,
    storedDirection,
    storedLo,
  ));
}

fn grisLogMaterialPHatAt(
  receiverSurface: PrimarySurface,
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
  storedLo: vec3f,
) -> f32 {
  var receiverGeoNormal = nv;
  if (receiverSurface.hit && length(receiverSurface.pos - xv) <= 5e-2) {
    receiverGeoNormal = receiverSurface.geoNormal;
  }
  let tint = grisProxyTintAt(
    xv,
    nv,
    receiverGeoNormal,
    sampleKind,
    xs,
    storedDirection,
  );
  let materialContribution = grisMaterialContributionAt(
    receiverSurface,
    xv,
    nv,
    sampleKind,
    xs,
    storedDirection,
    storedLo,
  );
  return reservoirGiLogPositive(luminance(materialContribution * tint));
}

// |dT_domain_to_canonical|. Environment directions use the identity shift.
fn grisDomainToCanonicalJacobian(
  domainXv: vec3f,
  canonicalXv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  ns: vec3f,
) -> f32 {
  return reservoirGiRepresentPositiveLog(grisLogDomainToCanonicalJacobian(
    domainXv,
    canonicalXv,
    sampleKind,
    xs,
    ns,
  ));
}

fn grisLogDomainToCanonicalJacobian(
  domainXv: vec3f,
  canonicalXv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  ns: vec3f,
) -> f32 {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) { return 0.0; }
  if (!grisSurfaceSuffixReceiverSupported(domainXv, xs, ns)
   || !grisSurfaceSuffixReceiverSupported(canonicalXv, xs, ns)) {
    return GRIS_LOG_ZERO;
  }
  let logGDomain = grisLogReconnectionGeometryTerm(domainXv, xs, ns);
  let logGCanonical = grisLogReconnectionGeometryTerm(canonicalXv, xs, ns);
  if (!reservoirGiValidLog(logGDomain) || !reservoirGiValidLog(logGCanonical)) {
    return GRIS_LOG_ZERO;
  }
  let result = logGCanonical - logGDomain;
  return select(GRIS_LOG_ZERO, result, reservoirGiFinite(result));
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
  logPHatDomain: f32,
  logDomainToCanonicalJacobian: f32,
  inverseValid: bool,
) -> f32 {
  if (attempts == 0u || !inverseValid
   || !reservoirGiValidLog(logPHatDomain)
   || !reservoirGiValidLog(logDomainToCanonicalJacobian)) {
    return GRIS_LOG_ZERO;
  }
  let result =
    log2(f32(attempts))
    + logPHatDomain
    - logDomainToCanonicalJacobian;
  return select(
    GRIS_LOG_ZERO,
    result,
    reservoirGiFinite(result),
  );
}

fn grisScaledMass(logMass: f32, maxLogMass: f32) -> f32 {
  if (!reservoirGiValidLog(logMass) || !reservoirGiValidLog(maxLogMass)) {
    return 0.0;
  }
  let result = exp2(logMass - maxLogMass);
  return select(0.0, result, reservoirGiFinite(result) && result > 0.0);
}

fn grisLogTechniqueDenominator(maxLogMass: f32, scaledDenominator: f32) -> f32 {
  if (!reservoirGiValidLog(maxLogMass)
   || !reservoirGiFinite(scaledDenominator) || !(scaledDenominator > 0.0)) {
    return GRIS_LOG_ZERO;
  }
  let result = maxLogMass + log2(scaledDenominator);
  return select(GRIS_LOG_ZERO, result, reservoirGiFinite(result));
}

fn grisLogCanonicalResamplingWeight(
  logMisWeight: f32,
  logCanonicalPHat: f32,
  sourceH: f32,
  sourceNativeLogPHat: f32,
  logSourceToCanonicalJacobian: f32,
) -> f32 {
  if (!reservoirGiValidLog(logMisWeight)
   || !reservoirGiValidLog(logCanonicalPHat)
   || !reservoirGiValidLog(sourceH)
   || !reservoirGiValidLog(sourceNativeLogPHat)
   || !reservoirGiValidLog(logSourceToCanonicalJacobian)) {
    return GRIS_LOG_ZERO;
  }
  let sourceLogW = sourceH - sourceNativeLogPHat;
  if (!reservoirGiFinite(sourceLogW)) { return GRIS_LOG_ZERO; }
  let result =
    logMisWeight
    + logCanonicalPHat
    + sourceLogW
    + logSourceToCanonicalJacobian;
  return select(
    GRIS_LOG_ZERO,
    result,
    reservoirGiFinite(result),
  );
}

// Each generalized candidate already carries its all-technique MIS denominator,
// including every source reservoir's attempt multiplicity.  Therefore this
// finalizer persists the WRS selection correction directly, with no second M
// division: H = selectedLogWeight - log2(selectionProbability).
fn grisFinaliseRepresentedReservoir(
  r: ptr<function, ReservoirPT>,
  wrs: RepresentedWrsState,
  wCap: f32,
) {
  (*r).H = RESERVOIR_GI_LOG_ZERO;
  (*r).logW = RESERVOIR_GI_LOG_ZERO;
  if ((*r).M == 0u
   || !wrs.hasSelection
   || !reservoirGiValidLog((*r).nativeLogPHat)
   || !reservoirGiFinite(wCap) || wCap < 0.0) {
    return;
  }
  (*r).H = reservoirGiCollapseLogParts(
    representedWrsSelectedLogCorrectionParts(wrs),
  );
  reservoirGiFinaliseLogWFromH(r, wCap);
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
