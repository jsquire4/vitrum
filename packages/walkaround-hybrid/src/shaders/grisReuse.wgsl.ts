/**
 * GRIS helpers for walkaround's declared one-bounce diffuse DDGI proxy.
 *
 * Surface samples reconnect a stored DDGI suffix vertex to another receiver;
 * environment samples keep their stored direction with identity Jacobian.
 * Every viable technique is evaluated in a bounded all-domain matrix using
 * p_domain / J_domain_to_canonical. Invalid inverse maps and occluded edges have
 * exactly zero density. This is not a full path representation, does not reuse
 * glossy transport, and does not claim an unbiased path-tracing estimator.
 *
 * The arithmetic is independently mirrored by `pipeline/grisReuseMis.ts`.
 * Reference: Lin et al., Generalized Resampled Importance Sampling (2022).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GRIS_REUSE_WGSL = /* wgsl */ `// ============================================================
// GRIS reuse for the declared one-bounce diffuse DDGI proxy.
// Lin et al. 2022: every technique density is transformed into the
// canonical receiver measure with the inverse shift determinant.
// ============================================================

const GRIS_NORMAL_BIAS: f32 = 1e-3;
const GRIS_MAX_FINITE_F32: f32 = 3.402823466e38;
const GRIS_MAX_WEIGHTED_DENSITY: f32 = GRIS_MAX_FINITE_F32 / 6.0;

fn grisFiniteVec3(value: vec3f) -> bool {
  return reservoirGiFinite(value.x)
    && reservoirGiFinite(value.y)
    && reservoirGiFinite(value.z);
}

fn grisSafeDirection(value: vec3f) -> vec3f {
  if (!grisFiniteVec3(value)) { return vec3f(0.0); }
  let lengthSquared = dot(value, value);
  if (!reservoirGiFinite(lengthSquared) || !(lengthSquared > 1e-12)) {
    return vec3f(0.0);
  }
  return value * inverseSqrt(lengthSquared);
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
  let dist2 = dot(d, d);
  if (!reservoirGiFinite(dist2) || !(dist2 > 1e-12)) { return 0.0; }
  let result = abs(dot(n2, d * inverseSqrt(dist2))) / dist2;
  return select(0.0, result, reservoirGiFinite(result) && result > 0.0);
}

fn grisDirection(sampleKind: u32, xv: vec3f, xs: vec3f, storedDirection: vec3f) -> vec3f {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
    return grisSafeDirection(storedDirection);
  }
  return grisSafeDirection(xs - xv);
}

fn grisMappedXs(sampleKind: u32, xv: vec3f, xs: vec3f, storedDirection: vec3f) -> vec3f {
  if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
    return xv + grisSafeDirection(storedDirection) * 100.0;
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

fn grisProxyTargetAt(
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
  storedLo: vec3f,
) -> f32 {
  let wi = grisDirection(sampleKind, xv, xs, storedDirection);
  let Lo = grisMappedLo(sampleKind, storedDirection, storedLo);
  if (!grisFiniteVec3(nv) || !grisFiniteVec3(Lo)
   || !(dot(wi, wi) > 0.0)) { return 0.0; }
  let result = luminance(Lo) * max(0.0, dot(nv, wi)) * INV_PI;
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
  var tMax = 1e20;
  if (sampleKind == GI_SAMPLE_SURFACE) {
    let d = length(xs - xv);
    if (d <= 2.0 * GRIS_NORMAL_BIAS) { return 0.0; }
    tMax = d - 2.0 * GRIS_NORMAL_BIAS;
  }
  if (max(0.0, dot(nv, wi)) <= 0.0) { return 0.0; }
  let tint = traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode, ubo.tlasNodeCount,

    xv + nv * GRIS_NORMAL_BIAS, wi, tMax, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
  );
  let visibility = luminance(tint);
  if (!reservoirGiFinite(visibility)) { return 0.0; }
  return clamp(visibility, 0.0, 1.0);
}

fn grisProxyPHatAt(
  xv: vec3f,
  nv: vec3f,
  sampleKind: u32,
  xs: vec3f,
  storedDirection: vec3f,
  storedLo: vec3f,
) -> f32 {
  let visibility = grisProxyVisibilityAt(xv, nv, sampleKind, xs, storedDirection);
  return grisProxyTargetAt(xv, nv, sampleKind, xs, storedDirection, storedLo) * visibility;
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
fn grisTransformedDensity(
  pHatDomain: f32,
  domainToCanonicalJacobian: f32,
  inverseValid: bool,
) -> f32 {
  if (!inverseValid
   || !reservoirGiFinite(pHatDomain) || !(pHatDomain > 0.0)
   || !reservoirGiFinite(domainToCanonicalJacobian)
   || !(domainToCanonicalJacobian > 0.0)) {
    return 0.0;
  }
  let result = pHatDomain / domainToCanonicalJacobian;
  return select(0.0, result, reservoirGiFinite(result) && result > 0.0);
}

fn grisWeightedDensity(attempts: u32, density: f32) -> f32 {
  if (attempts == 0u || !reservoirGiFinite(density) || !(density > 0.0)) {
    return 0.0;
  }
  let attemptF = f32(attempts);
  if (density > GRIS_MAX_WEIGHTED_DENSITY / attemptF) {
    return GRIS_MAX_WEIGHTED_DENSITY;
  }
  return attemptF * density;
}

fn grisConfidence(m: u32, clampM: u32) -> f32 {
  return f32(min(m, clampM));
}
`;

/** GRIS DDGI-proxy transformed-density WGSL include-graph entry.
 *  The retained geometric fallback calls `luminance` (sharedPrimitives) and
 *  uses `INV_PI` (walkaroundUbo). Declared as `requires` so the topo-sort emits
 *  this module AFTER those definitions. (`inverseSqrt` is a WGSL builtin.) */
export const GRIS_REUSE_MODULE: WgslModule = {
  name: 'grisReuse',
  source: GRIS_REUSE_WGSL,
  requires: ['walkaroundUbo', 'sharedPrimitives', 'surfaceTextures', 'environmentSample'],
};
