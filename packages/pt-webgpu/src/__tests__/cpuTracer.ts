/**
 * CPU mini-ray-tracer — Test-only utility; NOT exported from the package index.
 *
 * Provides deterministic TypeScript mirrors of the core path-tracing math in
 * pathTraceBruteforce.wgsl.ts and common.wgsl.ts, plus a brute-force integrator
 * for Monte-Carlo convergence tests (33-C, Phase B).
 *
 * Design rules:
 *   - No BVH: brute-force triangle iteration (scenes ≤ 50 tris).
 *   - No Math.random(): use the lcg() helper seeded by the caller.
 *   - Mirrors cite their WGSL source location.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Vec3 = [number, number, number];
export type Rng  = { v: number };

export interface Ray { origin: Vec3; dir: Vec3 }

export interface Triangle {
  v0: Vec3; v1: Vec3; v2: Vec3;
  materialId: number;
}

export interface Material {
  /** Diffuse/specular base colour. */
  albedo: Vec3;
  roughness: number;
  metallic: number;
  transmission: number;
  ior: number;
  /** Emissive radiance; [0,0,0] for non-emissive. */
  emission: Vec3;
}

/** Simple delta-function directional light (infinite distance).
 *  File-local — consumed only by `Light` union below; tests construct
 *  `Light` values directly. */
interface DirectionalLight {
  /** Unit direction toward the light. */
  dir: Vec3;
  /** Radiance scalar (white). */
  radiance: number;
}

/** Minimal point light (position + radiance).
 *  File-local — see `DirectionalLight` rationale. */
interface PointLight { pos: Vec3; radiance: Vec3 }

export type Light = { kind: 'directional'; light: DirectionalLight }
                  | { kind: 'point';       light: PointLight };

export interface Scene {
  /** Brute-force triangle list — no BVH; keep ≤ 50 for fast CI. */
  triangles: Triangle[];
  materials: Material[];
  lights: Light[];
  /** Environment radiance in the given direction (used on miss). */
  envSample: (dir: Vec3) => Vec3;
}

export interface PathOpts {
  maxBounces: number;
}

// ---------------------------------------------------------------------------
// Deterministic LCG RNG — Park-Miller, same as energyConservation.test.ts.
// ---------------------------------------------------------------------------

/** Advance LCG state and return a uniform float in [0, 1). */
export function lcg(rng: Rng): number {
  rng.v = (Math.imul(rng.v, 1664525) + 1013904223) >>> 0;
  return rng.v / 0x100000000;
}

// ---------------------------------------------------------------------------
// Vec3 math
// ---------------------------------------------------------------------------

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function mul(a: Vec3, b: Vec3): Vec3 {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

export function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length3(v: Vec3): number {
  return Math.sqrt(dot(v, v));
}

/** Mirror of safe_normalize (common.wgsl.ts:56). */
export function safeNormalize(v: Vec3): Vec3 {
  const componentScale = Math.max(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2]));
  if (!(componentScale > 0) || componentScale > 3.402823e38) {
    return [0, 1, 0];
  }
  const scaled: Vec3 = [
    v[0] / componentScale,
    v[1] / componentScale,
    v[2] / componentScale,
  ];
  return scale(scaled, 1 / length3(scaled));
}

function maxComp(v: Vec3): number {
  return Math.max(v[0], Math.max(v[1], v[2]));
}

// ---------------------------------------------------------------------------
// safeInvDir — mirror of SAFE_INV_DIR_WGSL (shared-bvh, Williams 2005).
// Exact zero and reciprocal overflow saturate to signed f32::MAX. This avoids
// 0*Inf slab NaNs without imposing the old ±1e30 reciprocal ceiling.
// ---------------------------------------------------------------------------

function safeInvDir(d: Vec3): Vec3 {
  const maxFiniteF32 = 3.402823e38;
  function safeInv(x: number): number {
    if (x === 0) return x >= 0 ? maxFiniteF32 : -maxFiniteF32;
    return Math.max(-maxFiniteF32, Math.min(maxFiniteF32, 1.0 / x));
  }
  return [safeInv(d[0]), safeInv(d[1]), safeInv(d[2])];
}

// ---------------------------------------------------------------------------
// Möller-Trumbore triangle intersection — mirror of the canonical shared core
// `mollerTrumboreCore` (shared-bvh/wgsl/bvhIntersect.wgsl.ts, MOLLER_TRUMBORE_WGSL),
// which pt-webgpu's `intersectTriangle` (common.wgsl.ts) now wraps.
//
// Returns { t, u, v } or null on miss. Determinant rejection is a normalized
// angular test, barycentric tolerance is dimensionless, and triEps is solely
// the caller-owned ray-distance floor. The earlier mirror used the algebraically-equivalent
// h/s/q factoring with strict u<0||u>1 / v<0||u+v>1 tests; this canonical form
// uses n = cross(e1,e2); det = -dot(dir,n) and the DAO = cross(AO,dir) barycentric
// solve with signed edge tolerance, matching the WGSL it validates.
// Default 1e-5 matches the host's params.triIntersectEpsilon (index.ts:489).
// ---------------------------------------------------------------------------

function intersectTriangleMT(
  rayOrigin: Vec3,
  rayDir: Vec3,
  v0: Vec3, v1: Vec3, v2: Vec3,
  triEps = 1e-5,
): { t: number; u: number; v: number } | null {
  const e1 = sub(v1, v0);
  const e2 = sub(v2, v0);
  const n  = cross(e1, e2);
  const det = -dot(rayDir, n);
  const normalLength = Math.hypot(n[0], n[1], n[2]);
  const directionLength = Math.hypot(rayDir[0], rayDir[1], rayDir[2]);
  if (
    normalLength <= 1e-15 ||
    directionLength <= 1e-15 ||
    Math.abs(det) / (normalLength * directionLength) <= 1e-7
  ) return null;
  const invDet = 1.0 / det;
  const AO  = sub(rayOrigin, v0);
  const DAO = cross(AO, rayDir);
  const u = dot(e2, DAO) * invDet;
  const v = -dot(e1, DAO) * invDet;
  const t = dot(AO, n) * invDet;
  const w = 1.0 - u - v;
  const barycentricEpsilon = 1e-6;
  if (
    u < -barycentricEpsilon ||
    v < -barycentricEpsilon ||
    w < -barycentricEpsilon ||
    t < Math.max(triEps, 0)
  ) return null;
  return { t, u, v };
}

// ---------------------------------------------------------------------------
// Ray-AABB slab test — mirror of intersectAabb (pathTraceBruteforce.wgsl.ts:545).
// Returns { tNear, tFar } or null on miss.
// Retained as a CPU reference oracle; not all tests invoke it.
// ---------------------------------------------------------------------------

function _intersectAabb(
  rayOrigin: Vec3,
  rayDir: Vec3,
  bMin: Vec3,
  bMax: Vec3,
): { tNear: number; tFar: number } | null {
  const inv = safeInvDir(rayDir);
  const t1: Vec3 = [
    (bMin[0] - rayOrigin[0]) * inv[0],
    (bMin[1] - rayOrigin[1]) * inv[1],
    (bMin[2] - rayOrigin[2]) * inv[2],
  ];
  const t2: Vec3 = [
    (bMax[0] - rayOrigin[0]) * inv[0],
    (bMax[1] - rayOrigin[1]) * inv[1],
    (bMax[2] - rayOrigin[2]) * inv[2],
  ];
  const tNear = Math.max(
    Math.min(t1[0], t2[0]),
    Math.min(t1[1], t2[1]),
    Math.min(t1[2], t2[2]),
  );
  const tFar = Math.min(
    Math.max(t1[0], t2[0]),
    Math.max(t1[1], t2[1]),
    Math.max(t1[2], t2[2]),
  );
  return tNear > tFar ? null : { tNear, tFar };
}

// ---------------------------------------------------------------------------
// Cosine-hemisphere sample — mirror of cosineHemisphereSample (WGSL:962).
// Returns sampled direction in world-space and its PDF (cosθ/π).
// ONB is built from n using the Frisvad-style helper below.
// ---------------------------------------------------------------------------

/** Build an ONB with n as the Z-axis (Frisvad-style, no branching). */
function buildOnb(n: Vec3): { t: Vec3; b: Vec3 } {
  const lensq = n[0] * n[0] + n[1] * n[1];
  let t: Vec3;
  if (lensq > 1e-10) {
    const inv = 1.0 / Math.sqrt(lensq);
    t = [-n[1] * inv, n[0] * inv, 0.0];
  } else {
    t = [1.0, 0.0, 0.0];
  }
  const b = cross(n, t);
  return { t, b };
}

function cosineHemisphereSample(
  rng: Rng,
  n: Vec3,
): { dir: Vec3; pdf: number } {
  const u1  = lcg(rng);
  const u2  = lcg(rng);
  const r   = Math.sqrt(u1);
  const phi = 2.0 * Math.PI * u2;
  const localX = r * Math.cos(phi);
  const localY = r * Math.sin(phi);
  const localZ = Math.sqrt(Math.max(0.0, 1.0 - u1));
  const { t, b } = buildOnb(n);
  const dir = safeNormalize([
    localX * t[0] + localY * b[0] + localZ * n[0],
    localX * t[1] + localY * b[1] + localZ * n[1],
    localX * t[2] + localY * b[2] + localZ * n[2],
  ]);
  const cosTheta = Math.max(dot(dir, n), 0.0);
  const pdf = cosTheta / Math.PI;
  return { dir, pdf };
}

// ---------------------------------------------------------------------------
// GGX VNDF sampler — mirror of sampleGgxVndfTangent (WGSL:981-1004).
// Heitz 2018, Algorithm 1. Input/output in tangent-space (N = +Z).
// Retained as a CPU reference oracle; used in companion tests.
// ---------------------------------------------------------------------------

function _sampleGgxVndfTangent(
  wo: Vec3,
  alpha: number,
  rng: Rng,
): Vec3 {
  // Step 1: stretch.
  const Vh = safeNormalize([alpha * wo[0], alpha * wo[1], wo[2]]);
  // Step 2: ONB around Vh (Frisvad-style).
  const { t: T1, b: T2 } = buildOnb(Vh);
  // Step 3: sample disc.
  const u1  = lcg(rng);
  const u2  = lcg(rng);
  const r   = Math.sqrt(u1);
  const phi = 2.0 * Math.PI * u2;
  const t1  = r * Math.cos(phi);
  let   t2  = r * Math.sin(phi);
  const s   = 0.5 * (1.0 + Vh[2]);
  t2 = (1.0 - s) * Math.sqrt(Math.max(0.0, 1.0 - t1 * t1)) + s * t2;
  // Step 4: reproject and unstretch.
  const z  = Math.sqrt(Math.max(0.0, 1.0 - t1 * t1 - t2 * t2));
  const Nh: Vec3 = [
    t1 * T1[0] + t2 * T2[0] + z * Vh[0],
    t1 * T1[1] + t2 * T2[1] + z * Vh[1],
    t1 * T1[2] + t2 * T2[2] + z * Vh[2],
  ];
  return safeNormalize([alpha * Nh[0], alpha * Nh[1], Math.max(1e-6, Nh[2])]);
}

// ---------------------------------------------------------------------------
// frDielectric — mirror of WGSL:303 (PBR4e §9.3).
// Retained as a CPU reference oracle for glass-refraction tests.
// ---------------------------------------------------------------------------

function _frDielectric(cosTheta_i: number, eta: number): number {
  let ct = Math.min(Math.max(cosTheta_i, -1.0), 1.0);
  let e  = eta;
  if (ct < 0.0) { e = 1.0 / e; ct = -ct; }
  const sin2I = Math.max(0.0, 1.0 - ct * ct);
  const sin2T = sin2I / (e * e);
  if (sin2T >= 1.0) return 1.0; // TIR
  const cosT  = Math.sqrt(Math.max(0.0, 1.0 - sin2T));
  const r_par  = (e * ct - cosT) / (e * ct + cosT);
  const r_perp = (ct - e * cosT)  / (ct + e * cosT);
  return 0.5 * (r_par * r_par + r_perp * r_perp);
}

// ---------------------------------------------------------------------------
// Schlick Fresnel — mirror of fresnelSchlick (WGSL:289), scalar variant.
// Retained as a CPU reference oracle for Fresnel tests.
// ---------------------------------------------------------------------------

function _schlickFresnel(f0: number, cos: number): number {
  const m  = Math.min(Math.max(1.0 - cos, 0.0), 1.0);
  const m2 = m * m;
  const m5 = m2 * m2 * m;
  return f0 + (1.0 - f0) * m5;
}

// ---------------------------------------------------------------------------
// Power heuristic (β=2) — mirror of powerHeuristic (WGSL:332, Veach §9.2).
// Retained as a CPU reference oracle; companion tests may import it.
// ---------------------------------------------------------------------------

function _powerHeuristic(a: number, b: number): number {
  if (!(a >= 0) || !(b >= 0) || a > 3.402823466e38 || b > 3.402823466e38) {
    return 0;
  }
  const pdfScale = Math.max(a, b);
  if (!(pdfScale > 0)) return 0;
  const scaledA = a / pdfScale;
  const scaledB = b / pdfScale;
  const a2 = scaledA * scaledA;
  const b2 = scaledB * scaledB;
  return a2 / (a2 + b2);
}

// ---------------------------------------------------------------------------
// BRDF evaluation — Lambertian-only path (mirrors lambertianBrdf in tests).
// For the integrator we support purely Lambertian diffuse materials
// (roughness=1, metallic=0). Returns albedo/π.
// ---------------------------------------------------------------------------

function lambertBrdf(albedo: Vec3): Vec3 {
  const inv_pi = 1.0 / Math.PI;
  return scale(albedo, inv_pi);
}

// ---------------------------------------------------------------------------
// Scene traversal (brute-force — no BVH).
// Returns { triIdx, t, normal } or null on miss.
// ---------------------------------------------------------------------------

interface HitRecord {
  triIdx: number;
  t: number;
  normal: Vec3; // outward-facing geometric normal
  isFront: boolean;
}

function traceClosest(scene: Scene, ray: Ray, tMin: number, tMax: number): HitRecord | null {
  let best: HitRecord | null = null;
  let bestT = tMax;

  for (let i = 0; i < scene.triangles.length; i++) {
    const tri = scene.triangles[i]!;
    const hit = intersectTriangleMT(ray.origin, ray.dir, tri.v0, tri.v1, tri.v2);
    if (hit === null || hit.t < tMin || hit.t >= bestT) continue;
    bestT = hit.t;
    const e1 = sub(tri.v1, tri.v0);
    const e2 = sub(tri.v2, tri.v0);
    const geomN = safeNormalize(cross(e1, e2));
    const isFront = dot(geomN, ray.dir) < 0.0;
    const outwardN: Vec3 = isFront ? geomN : scale(geomN, -1);
    best = { triIdx: i, t: hit.t, normal: outwardN, isFront };
  }
  return best;
}

function traceAny(scene: Scene, ray: Ray, tMin: number, tMax: number): boolean {
  for (const tri of scene.triangles) {
    const hit = intersectTriangleMT(ray.origin, ray.dir, tri.v0, tri.v1, tri.v2);
    if (hit !== null && hit.t > tMin && hit.t < tMax) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Direct light sampling (NEE).
// Supports DirectionalLight and PointLight.
// ---------------------------------------------------------------------------

function sampleDirectLight(
  scene: Scene,
  hitPos: Vec3,
  normal: Vec3,
  rng: Rng,
): Vec3 {
  if (scene.lights.length === 0) return [0, 0, 0];

  // Uniform-randomly pick one light (same pattern as WGSL §1666).
  const idx   = Math.floor(lcg(rng) * scene.lights.length) % scene.lights.length;
  const entry = scene.lights[idx]!;

  let wi: Vec3;
  let Li: Vec3;
  let tMax: number;

  if (entry.kind === 'directional') {
    const dl = entry.light;
    wi   = safeNormalize(dl.dir);
    Li   = [dl.radiance, dl.radiance, dl.radiance];
    tMax = 1e30;
  } else {
    const pl      = entry.light;
    const toLight = sub(pl.pos, hitPos);
    const dist    = length3(toLight);
    wi   = scale(toLight, 1.0 / Math.max(dist, 1e-8));
    Li   = scale(pl.radiance, 1.0 / Math.max(dist * dist, 1e-5));
    tMax = dist - 2e-3;
  }

  const nDotL = dot(normal, wi);
  if (nDotL <= 0.0) return [0, 0, 0];

  // Shadow ray — offset by normal to avoid self-intersection.
  const shadowOrigin = add(hitPos, scale(normal, 1e-3));
  const shadowRay: Ray = { origin: shadowOrigin, dir: wi };
  if (traceAny(scene, shadowRay, 1e-4, tMax)) return [0, 0, 0];

  // Lambertian BSDF × NdotL × Li.  The (ρ/π)·π cancels; returns ρ·NdotL·Li.
  // We return the raw radiance contribution; the caller multiplies by throughput.
  return scale(Li, nDotL * scene.lights.length);
}

// ---------------------------------------------------------------------------
// Path integrator — NEE direct + cosine-hemisphere next-bounce, Lambertian.
// Mirrors the structure of pathTraceBruteforce.wgsl.ts `main()` (WGSL:1560+).
// Russian roulette after bounce 2 with survival ∈ [0.1, 0.95].
//
// NOTE: only Lambertian (diffuse) materials are supported in this tracer.
// Glossy/transmission materials would require the full BSDF split logic;
// the test scenes use albedo-only materials (roughness=1, metallic=0).
// ---------------------------------------------------------------------------

export function integratePath(
  scene: Scene,
  ray: Ray,
  rng: Rng,
  opts: PathOpts,
): Vec3 {
  let throughput: Vec3 = [1, 1, 1];
  let radiance: Vec3   = [0, 0, 0];
  let currentRay = ray;

  for (let bounce = 0; bounce < opts.maxBounces; bounce++) {
    const hit = traceClosest(scene, currentRay, 1e-4, 1e30);

    if (hit === null) {
      // Miss — add environment.
      radiance = add(radiance, mul(throughput, scene.envSample(currentRay.dir)));
      break;
    }

    const hitPos = add(currentRay.origin, scale(currentRay.dir, hit.t));
    const mat    = scene.materials[scene.triangles[hit.triIdx]!.materialId]!;
    const normal = hit.normal;

    // Emission.
    if (maxComp(mat.emission) > 0) {
      radiance = add(radiance, mul(throughput, mat.emission));
    }

    // NEE direct light.
    const brdf = lambertBrdf(mat.albedo);
    const directLi = sampleDirectLight(scene, hitPos, normal, rng);
    // directLi already encodes nDotL·Li; BRDF is albedo/π.
    radiance = add(radiance, mul(throughput, mul(brdf, directLi)));

    // Cosine-hemisphere next-bounce sample.
    const { dir: wi, pdf } = cosineHemisphereSample(rng, normal);
    if (pdf <= 1e-8) break;

    const _cosTheta = Math.max(dot(wi, normal), 0.0);
    // throughput *= BRDF(wi) * _cosTheta / pdf.  For Lambert: (albedo/π)·cosθ/(cosθ/π) = albedo.
    throughput = mul(throughput, mat.albedo);

    // Russian roulette after bounce 2 (mirrors WGSL:1891).
    if (bounce > 2) {
      const survival = Math.min(Math.max(maxComp(throughput), 0.1), 0.95);
      if (lcg(rng) > survival) break;
      throughput = scale(throughput, 1.0 / survival);
    }

    const nextOrigin = add(hitPos, scale(normal, 1e-3));
    currentRay = { origin: nextOrigin, dir: wi };
  }

  return radiance;
}
