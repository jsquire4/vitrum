import { PT_WEBGPU_COMMON_WGSL } from './common.wgsl.js';
import { HAMMERSLEY_WGSL } from './hammersley.wgsl.js';
import { OCTAHEDRAL_WGSL } from './octahedral.wgsl.js';

/**
 * First-pass path tracing kernel:
 * - camera ray generation from inverse VP
 * - BVH-accelerated triangle intersection
 * - simple diffuse/specular bounce integration
 * - progressive accumulation in storage buffer
 */
export const PT_WEBGPU_TRACE_WGSL = /* wgsl */ `
${PT_WEBGPU_COMMON_WGSL}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_WGSL}

struct FrameParams {
  width: u32,
  height: u32,
  frameIndex: u32,
  frameSeed: u32,
  triangleCount: u32,
  maxBounces: u32,
  bvhNodeCount: u32,
  analyticCount: u32,
  cameraPos: vec4f,
  lightDir: vec4f,
  pointLightPos: vec4f,
  pointLightRadiance: vec4f,
  spotLightPos: vec4f,
  spotLightDirection: vec4f,
  spotLightRadiance: vec4f,
  environmentTint: vec4f,
  environmentSun: vec4f,
  rectAreaPos: vec4f,
  rectAreaU: vec4f,
  rectAreaV: vec4f,
  rectAreaRadiance: vec4f,
  meshAreaTriA: vec4f,
  meshAreaTriB: vec4f,
  meshAreaTriC: vec4f,
  meshAreaRadiance: vec4f,
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
};

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: FrameParams;
@group(0) @binding(2) var<storage, read_write> accumBuffer: array<vec4f>;
@group(0) @binding(3) var<storage, read> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read> indices: array<vec4u>;
@group(0) @binding(5) var<storage, read> triMaterialIds: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<vec4f>;
@group(0) @binding(7) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4f>;
@group(0) @binding(9) var normalDepthTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var albedoTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var varianceTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var motionVectorsTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> varianceMomentsBuffer: array<vec4f>;
@group(0) @binding(14) var<storage, read> analyticHeaders: array<vec4f>;
@group(0) @binding(15) var<storage, read> analyticParams: array<vec4f>;
@group(0) @binding(16) var<storage, read> analyticLocalToWorld: array<vec4f>;
@group(0) @binding(17) var<storage, read> analyticWorldToLocal: array<vec4f>;
@group(0) @binding(18) var<storage, read> environmentMapTexels: array<vec4f>;
@group(0) @binding(19) var<storage, read> environmentMapCdf: array<f32>;

const LEAFNODE_FLAG = 0xffff0000u;

fn generatePrimaryRay(px: u32, py: u32, jitter: vec2f) -> Ray {
  let uv = (vec2f(f32(px), f32(py)) + jitter) / vec2f(f32(params.width), f32(params.height));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4 = params.invViewProj * vec4f(ndc, 1.0, 1.0);
  let near4 = params.invViewProj * vec4f(ndc, -1.0, 1.0);
  let farW = far4.xyz / far4.w;
  let nearW = near4.xyz / near4.w;
  var ray: Ray;
  ray.origin = params.cameraPos.xyz;
  ray.direction = safe_normalize(farW - nearW);
  return ray;
}

fn sampleSky(dir: vec3f) -> vec3f {
  let t = 0.5 * (dir.y + 1.0);
  var sky = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.45, 0.62, 0.95), clamp(t, 0.0, 1.0));
  let sunDir = safe_normalize(params.environmentSun.xyz);
  let sunGlow = pow(max(0.0, dot(dir, sunDir)), 512.0) * params.environmentSun.w;
  sky = sky + vec3f(1.0, 0.95, 0.85) * sunGlow;
  return sky * params.environmentTint.rgb;
}

fn hasEnvironmentMap() -> bool {
  return params.environmentTint.w > 0.5 && params.meshAreaTriB.w > 0.5 && params.meshAreaTriC.w > 0.5;
}

fn environmentDimensions() -> vec2u {
  return vec2u(u32(params.meshAreaTriB.w), u32(params.meshAreaTriC.w));
}

fn sampleEnvironmentColor(dir: vec3f) -> vec3f {
  if (!hasEnvironmentMap()) {
    return sampleSky(dir);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return sampleSky(dir);
  }
  let phi = atan2(dir.z, dir.x);
  let theta = acos(clamp(dir.y, -1.0, 1.0));
  let u = fract(phi * INV_2PI + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = min(u32(floor(u * f32(dims.x))), dims.x - 1u);
  let y = min(u32(floor(v * f32(dims.y))), dims.y - 1u);
  let idx = y * dims.x + x;
  if (idx >= arrayLength(&environmentMapTexels)) {
    return sampleSky(dir);
  }
  let texel = environmentMapTexels[idx];
  return texel.rgb * max(params.environmentSun.w, 0.0);
}

fn environmentPdf(dir: vec3f) -> f32 {
  if (!hasEnvironmentMap()) {
    return 1.0 / (4.0 * PI);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return 1.0 / (4.0 * PI);
  }
  let phi = atan2(dir.z, dir.x);
  let theta = acos(clamp(dir.y, -1.0, 1.0));
  let u = fract(phi * INV_2PI + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = min(u32(floor(u * f32(dims.x))), dims.x - 1u);
  let y = min(u32(floor(v * f32(dims.y))), dims.y - 1u);
  let idx = y * dims.x + x;
  if (idx >= arrayLength(&environmentMapTexels)) {
    return 1.0 / (4.0 * PI);
  }
  return max(environmentMapTexels[idx].w, 1e-8);
}

fn sampleEnvironmentImportance(rng: ptr<function, u32>, outDir: ptr<function, vec3f>, outColor: ptr<function, vec3f>, outPdf: ptr<function, f32>) -> bool {
  if (!hasEnvironmentMap()) {
    return false;
  }
  let dims = environmentDimensions();
  let count = dims.x * dims.y;
  if (count == 0u || arrayLength(&environmentMapCdf) < count + 1u) {
    return false;
  }
  let xi = rand_f32(rng);
  var lo = 0u;
  var hi = count;
  loop {
    if (lo + 1u >= hi) { break; }
    let mid = (lo + hi) >> 1u;
    if (environmentMapCdf[mid] <= xi) { lo = mid; } else { hi = mid; }
  }
  let idx = min(lo, count - 1u);
  let x = idx % dims.x;
  let y = idx / dims.x;
  let u = (f32(x) + 0.5) / f32(dims.x);
  let v = (f32(y) + 0.5) / f32(dims.y);
  let phi = (u - 0.5) * (2.0 * PI);
  let theta = v * PI;
  let sinTheta = sin(theta);
  let dir = vec3f(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
  let texel = environmentMapTexels[idx];
  *outDir = safe_normalize(dir);
  *outColor = texel.rgb * max(params.environmentSun.w, 0.0);
  *outPdf = max(texel.w, 1e-8);
  return true;
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

fn smithG1(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

fn powerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
  let a2 = pdfA * pdfA;
  let b2 = pdfB * pdfB;
  return a2 / max(a2 + b2, 1e-6);
}

fn evaluateBrdf(baseColor: vec3f, roughness: f32, metallic: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    return vec3f(0.0);
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  let spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
  let kd = (vec3f(1.0) - f) * (1.0 - metallic);
  let diff = kd * baseColor * INV_PI;
  return diff + spec;
}

fn brdfDirectionalPdf(baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    return 0.0;
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 1e-6);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let fresnel = fresnelSchlick(vDotH, f0);
  let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
  let baseDiffProb = max(0.0, (1.0 - metallic) * (1.0 - transmission));
  let sumProb = max(baseSpecProb + baseDiffProb, 1e-4);
  let specProb = baseSpecProb / sumProb;
  let diffProb = baseDiffProb / sumProb;
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let pdfSpec = d * nDotH / max(4.0 * vDotH, 1e-6);
  let pdfDiff = nDotL * INV_PI;
  return diffProb * pdfDiff + specProb * pdfSpec;
}

fn intersectRectAreaLightRay(rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let uAxis = params.rectAreaU.xyz;
  let vAxis = params.rectAreaV.xyz;
  let lightNormal = safe_normalize(cross(uAxis, vAxis));
  let denom = dot(lightNormal, rayDir);
  if (abs(denom) < 1e-6) {
    return false;
  }
  let t = dot(lightNormal, params.rectAreaPos.xyz - rayOrigin) / denom;
  if (t <= 1e-4) {
    return false;
  }
  let p = rayOrigin + rayDir * t;
  let rel = p - params.rectAreaPos.xyz;
  let uLen2 = max(dot(uAxis, uAxis), 1e-6);
  let vLen2 = max(dot(vAxis, vAxis), 1e-6);
  let uCoord = dot(rel, uAxis) / uLen2;
  let vCoord = dot(rel, vAxis) / vLen2;
  if (abs(uCoord) > 1.0 || abs(vCoord) > 1.0) {
    return false;
  }
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = max(4.0 * length(cross(uAxis, vAxis)), 1e-6);
  *distOut = t;
  *lightPdfOut = (t * t) / max(cosLight * area, 1e-6);
  return true;
}

fn intersectMeshAreaLightRay(rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let a = params.meshAreaTriA.xyz;
  let b = params.meshAreaTriB.xyz;
  let c = params.meshAreaTriC.xyz;
  let t = intersectTriangle(rayOrigin, rayDir, a, b, c);
  if (t <= 1e-4 || t >= INFINITY) {
    return false;
  }
  let lightNormal = safe_normalize(cross(b - a, c - a));
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
  *distOut = t;
  *lightPdfOut = (t * t) / max(cosLight * area, 1e-6);
  return true;
}

fn bsdfAreaLightConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) {
    return vec3f(0.0);
  }
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
  if (bsdfPdf <= 1e-6) {
    return vec3f(0.0);
  }
  var bestDist = INFINITY;
  var bestLightPdf = 0.0;
  var bestEmission = vec3f(0.0);
  if (params.rectAreaPos.w > 0.5) {
    var rectDist = INFINITY;
    var rectPdf = 0.0;
    if (intersectRectAreaLightRay(hitPos + normal * 1e-3, wi, &rectDist, &rectPdf)) {
      let shadowRay = Ray(hitPos + normal * 1e-3, wi);
      if (!traceAny(shadowRay, 1e-4, max(rectDist - 2e-3, 1e-3)) && rectDist < bestDist) {
        bestDist = rectDist;
        bestLightPdf = rectPdf;
        bestEmission = params.rectAreaRadiance.rgb;
      }
    }
  }
  if (params.meshAreaTriA.w > 0.5) {
    var meshDist = INFINITY;
    var meshPdf = 0.0;
    if (intersectMeshAreaLightRay(hitPos + normal * 1e-3, wi, &meshDist, &meshPdf)) {
      let shadowRay = Ray(hitPos + normal * 1e-3, wi);
      if (!traceAny(shadowRay, 1e-4, max(meshDist - 2e-3, 1e-3)) && meshDist < bestDist) {
        bestDist = meshDist;
        bestLightPdf = meshPdf;
        bestEmission = params.meshAreaRadiance.rgb;
      }
    }
  }
  if (bestDist >= INFINITY || bestLightPdf <= 1e-6) {
    return vec3f(0.0);
  }
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  let misWeight = powerHeuristic(bsdfPdf, bestLightPdf);
  return throughputAtVertex * brdf * nDotL * bestEmission * misWeight / max(bsdfPdf, 1e-6);
}

fn bsdfEnvironmentConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) return vec3f(0.0);
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
  if (bsdfPdf <= 1e-6) return vec3f(0.0);
  let shadowRay = Ray(hitPos + normal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, INFINITY)) return vec3f(0.0);
  let envPdf = environmentPdf(wi);
  let envColor = sampleEnvironmentColor(wi);
  let misWeight = powerHeuristic(bsdfPdf, envPdf);
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  return throughputAtVertex * brdf * nDotL * envColor * misWeight / max(bsdfPdf, 1e-6);
}

fn projectToNdc(pos: vec3f, vp: mat4x4f) -> vec2f {
  let clip = vp * vec4f(pos, 1.0);
  let invW = 1.0 / max(abs(clip.w), 1e-8);
  return clip.xy * invW;
}

fn intersectAabb(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32) -> bool {
  let invDir = vec3f(1.0) / ray.direction;
  let t1 = (bmin - ray.origin) * invDir;
  let t2 = (bmax - ray.origin) * invDir;
  let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let tFar = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  return !(tNear > tFar || tFar < tMin || tNear > tMax);
}

struct SceneHit {
  didHit: bool,
  dist: f32,
  triIndex: u32,
  normal: vec3f,
};

const SHAPE_SPHERE = 1u;
const SHAPE_BOX = 2u;
const SHAPE_CAPSULE = 3u;
const SHAPE_CYLINDER = 4u;
const SHAPE_H_CHANNEL_CAME = 5u;

fn transformPointCols(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, p: vec3f) -> vec3f {
  let r = c0 * p.x + c1 * p.y + c2 * p.z + c3;
  return r.xyz / max(abs(r.w), 1e-8);
}

fn transformDirectionCols(c0: vec4f, c1: vec4f, c2: vec4f, d: vec3f) -> vec3f {
  return safe_normalize((c0 * d.x + c1 * d.y + c2 * d.z).xyz);
}

fn transformNormalFromWorldToLocalCols(w2l0: vec4f, w2l1: vec4f, w2l2: vec4f, nLocal: vec3f) -> vec3f {
  return safe_normalize(vec3f(
    dot(vec3f(w2l0.x, w2l1.x, w2l2.x), nLocal),
    dot(vec3f(w2l0.y, w2l1.y, w2l2.y), nLocal),
    dot(vec3f(w2l0.z, w2l1.z, w2l2.z), nLocal),
  ));
}

fn intersectAabbDetailed(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32, nOut: ptr<function, vec3f>) -> f32 {
  let invDir = vec3f(1.0) / ray.direction;
  let t0 = (bmin - ray.origin) * invDir;
  let t1 = (bmax - ray.origin) * invDir;
  let tsm = min(t0, t1);
  let tbg = max(t0, t1);
  let tNear = max(max(tsm.x, tsm.y), tsm.z);
  let tFar = min(min(tbg.x, tbg.y), tbg.z);
  if (tNear > tFar || tFar < tMin || tNear > tMax) {
    return INFINITY;
  }
  var tHit = tNear;
  var fromFar = false;
  if (tHit < tMin) {
    tHit = tFar;
    fromFar = true;
  }
  var n = vec3f(0.0);
  let eps = 1e-4;
  if (!fromFar) {
    if (abs(tHit - tsm.x) < eps) {
      n = vec3f(select(1.0, -1.0, ray.direction.x > 0.0), 0.0, 0.0);
    } else if (abs(tHit - tsm.y) < eps) {
      n = vec3f(0.0, select(1.0, -1.0, ray.direction.y > 0.0), 0.0);
    } else {
      n = vec3f(0.0, 0.0, select(1.0, -1.0, ray.direction.z > 0.0));
    }
  } else {
    if (abs(tHit - tbg.x) < eps) {
      n = vec3f(select(-1.0, 1.0, ray.direction.x > 0.0), 0.0, 0.0);
    } else if (abs(tHit - tbg.y) < eps) {
      n = vec3f(0.0, select(-1.0, 1.0, ray.direction.y > 0.0), 0.0);
    } else {
      n = vec3f(0.0, 0.0, select(-1.0, 1.0, ray.direction.z > 0.0));
    }
  }
  *nOut = n;
  return tHit;
}

fn intersectSphereLocal(ray: Ray, center: vec3f, radius: f32, nOut: ptr<function, vec3f>) -> f32 {
  let oc = ray.origin - center;
  let a = dot(ray.direction, ray.direction);
  let b = 2.0 * dot(oc, ray.direction);
  let c = dot(oc, oc) - radius * radius;
  let disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return INFINITY;
  let s = sqrt(disc);
  let t0 = (-b - s) / (2.0 * a);
  let t1 = (-b + s) / (2.0 * a);
  var t = t0;
  if (t < 1e-5) t = t1;
  if (t < 1e-5) return INFINITY;
  let p = ray.origin + ray.direction * t;
  *nOut = safe_normalize(p - center);
  return t;
}

fn intersectCylinderLocal(ray: Ray, center: vec3f, radius: f32, halfHeight: f32, nOut: ptr<function, vec3f>) -> f32 {
  let ro = ray.origin - center;
  let rd = ray.direction;
  let a = rd.x * rd.x + rd.z * rd.z;
  let b = 2.0 * (ro.x * rd.x + ro.z * rd.z);
  let c = ro.x * ro.x + ro.z * ro.z - radius * radius;
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  let disc = b * b - 4.0 * a * c;
  if (disc >= 0.0 && abs(a) > 1e-8) {
    let s = sqrt(disc);
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    if (t0 > 1e-5) {
      let y = ro.y + rd.y * t0;
      if (abs(y) <= halfHeight) {
        bestT = t0;
        bestN = safe_normalize(vec3f(ro.x + rd.x * t0, 0.0, ro.z + rd.z * t0));
      }
    }
    if (t1 > 1e-5 && t1 < bestT) {
      let y = ro.y + rd.y * t1;
      if (abs(y) <= halfHeight) {
        bestT = t1;
        bestN = safe_normalize(vec3f(ro.x + rd.x * t1, 0.0, ro.z + rd.z * t1));
      }
    }
  }
  if (abs(rd.y) > 1e-8) {
    let topT = (halfHeight - ro.y) / rd.y;
    if (topT > 1e-5 && topT < bestT) {
      let p = ro + rd * topT;
      if (p.x * p.x + p.z * p.z <= radius * radius) {
        bestT = topT;
        bestN = vec3f(0.0, 1.0, 0.0);
      }
    }
    let bottomT = (-halfHeight - ro.y) / rd.y;
    if (bottomT > 1e-5 && bottomT < bestT) {
      let p = ro + rd * bottomT;
      if (p.x * p.x + p.z * p.z <= radius * radius) {
        bestT = bottomT;
        bestN = vec3f(0.0, -1.0, 0.0);
      }
    }
  }
  *nOut = bestN;
  return bestT;
}

fn intersectCapsuleLocal(ray: Ray, pa: vec3f, pb: vec3f, radius: f32, nOut: ptr<function, vec3f>) -> f32 {
  let ba = pb - pa;
  let oa = ray.origin - pa;
  let baba = dot(ba, ba);
  let bard = dot(ba, ray.direction);
  let baoa = dot(ba, oa);
  let rdoa = dot(ray.direction, oa);
  let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  let b = baba * rdoa - baoa * bard;
  let c = baba * oaoa - baoa * baoa - radius * radius * baba;
  let h = b * b - a * c;
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  if (h >= 0.0 && abs(a) > 1e-8) {
    let t = (-b - sqrt(h)) / a;
    let y = baoa + t * bard;
    if (t > 1e-5 && y > 0.0 && y < baba) {
      let p = oa + ray.direction * t - ba * (y / baba);
      bestT = t;
      bestN = safe_normalize(p);
    }
  }
  let ocA = ray.origin - pa;
  let bA = dot(ocA, ray.direction);
  let cA = dot(ocA, ocA) - radius * radius;
  let hA = bA * bA - cA;
  if (hA > 0.0) {
    let tA = -bA - sqrt(hA);
    if (tA > 1e-5 && tA < bestT) {
      bestT = tA;
      bestN = safe_normalize((ray.origin + ray.direction * tA) - pa);
    }
  }
  let ocB = ray.origin - pb;
  let bB = dot(ocB, ray.direction);
  let cB = dot(ocB, ocB) - radius * radius;
  let hB = bB * bB - cB;
  if (hB > 0.0) {
    let tB = -bB - sqrt(hB);
    if (tB > 1e-5 && tB < bestT) {
      bestT = tB;
      bestN = safe_normalize((ray.origin + ray.direction * tB) - pb);
    }
  }
  *nOut = bestN;
  return bestT;
}

fn intersectHChannelLocal(ray: Ray, lengthX: f32, railWidth: f32, blockHeight: f32, webThickness: f32, nOut: ptr<function, vec3f>) -> f32 {
  let hx = max(lengthX * 0.5, 1e-4);
  let hy = max(blockHeight * 0.5, 1e-4);
  let hz = max(railWidth * 0.5, 1e-4);
  let t = max(min(webThickness * 0.5, hy), 1e-4);
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  var n: vec3f;
  let railTop = intersectAabbDetailed(ray, vec3f(-hx, hy - t, -hz), vec3f(hx, hy, hz), 1e-4, INFINITY, &n);
  if (railTop < bestT) {
    bestT = railTop;
    bestN = n;
  }
  let railBottom = intersectAabbDetailed(ray, vec3f(-hx, -hy, -hz), vec3f(hx, -hy + t, hz), 1e-4, INFINITY, &n);
  if (railBottom < bestT) {
    bestT = railBottom;
    bestN = n;
  }
  let web = intersectAabbDetailed(ray, vec3f(-hx, -hy + t, -t), vec3f(hx, hy - t, t), 1e-4, INFINITY, &n);
  if (web < bestT) {
    bestT = web;
    bestN = n;
  }
  *nOut = bestN;
  return bestT;
}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  hit.didHit = false;
  hit.dist = tMax;
  hit.triIndex = 0u;
  hit.normal = vec3f(0.0, 1.0, 0.0);

  if (params.bvhNodeCount > 0u && arrayLength(&bvhNodes) > 0u) {
    var stack: array<u32, 64>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr = stackPtr + 1u;

    while (stackPtr > 0u) {
      stackPtr = stackPtr - 1u;
      let nodeIdx = stack[stackPtr];
      if (nodeIdx >= min(params.bvhNodeCount, arrayLength(&bvhNodes))) {
        continue;
      }
      let node = bvhNodes[nodeIdx];
      let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
      let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
      if (!intersectAabb(ray, bmin, bmax, tMin, hit.dist)) {
        continue;
      }

      let splitOrCount = node.splitAxisOrTriCount;
      if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
        let count = splitOrCount & 0x0000ffffu;
        let start = node.rightChildOrTriOffset;
        for (var i = 0u; i < count; i = i + 1u) {
          let t = start + i;
          if (t >= min(params.triangleCount, arrayLength(&indices))) {
            continue;
          }
          let tri = indices[t];
          if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
            continue;
          }
          let a = positions[tri.x].xyz;
          let b = positions[tri.y].xyz;
          let c = positions[tri.z].xyz;
          let hitT = intersectTriangle(ray.origin, ray.direction, a, b, c);
          if (hitT > tMin && hitT < hit.dist) {
            let p = ray.origin + ray.direction * hitT;
            let ab = b - a;
            let ac = c - a;
            let ap = p - a;
            let d00 = dot(ab, ab);
            let d01 = dot(ab, ac);
            let d11 = dot(ac, ac);
            let d20 = dot(ap, ab);
            let d21 = dot(ap, ac);
            let denom = max(d00 * d11 - d01 * d01, 1e-8);
            let v = clamp((d11 * d20 - d01 * d21) / denom, 0.0, 1.0);
            let w = clamp((d00 * d21 - d01 * d20) / denom, 0.0, 1.0);
            let u = max(0.0, 1.0 - v - w);
            var shadeNormal = safe_normalize(cross(ab, ac));
            if (tri.x < arrayLength(&normals) && tri.y < arrayLength(&normals) && tri.z < arrayLength(&normals)) {
              let na = normals[tri.x].xyz;
              let nb = normals[tri.y].xyz;
              let nc = normals[tri.z].xyz;
              shadeNormal = safe_normalize(na * u + nb * v + nc * w);
            }
            hit.didHit = true;
            hit.dist = hitT;
            hit.triIndex = t;
            hit.normal = shadeNormal;
          }
        }
      } else {
        let leftChild = nodeIdx + 1u;
        let rightChild = node.rightChildOrTriOffset;
        if (stackPtr + 2u < 64u) {
          stack[stackPtr] = rightChild;
          stackPtr = stackPtr + 1u;
          stack[stackPtr] = leftChild;
          stackPtr = stackPtr + 1u;
        }
      }
    }
  }

  let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));
  for (var ai = 0u; ai < analyticTotal; ai = ai + 1u) {
    let header = analyticHeaders[ai];
    let shapeId = u32(max(header.x, 0.0));
    let paramOffset = u32(max(header.z, 0.0));
    let matBase = ai * 4u;
    if (matBase + 3u >= arrayLength(&analyticWorldToLocal) || matBase + 3u >= arrayLength(&analyticLocalToWorld)) {
      continue;
    }
    let w2l0 = analyticWorldToLocal[matBase];
    let w2l1 = analyticWorldToLocal[matBase + 1u];
    let w2l2 = analyticWorldToLocal[matBase + 2u];
    let w2l3 = analyticWorldToLocal[matBase + 3u];
    let l2w0 = analyticLocalToWorld[matBase];
    let l2w1 = analyticLocalToWorld[matBase + 1u];
    let l2w2 = analyticLocalToWorld[matBase + 2u];
    let l2w3 = analyticLocalToWorld[matBase + 3u];
    var localRay: Ray;
    localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
    localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
    var localN = vec3f(0.0, 1.0, 0.0);
    var localT = INFINITY;
    let p0 = select(vec4f(0.0), analyticParams[paramOffset], paramOffset < arrayLength(&analyticParams));
    let p1 = select(vec4f(0.0), analyticParams[paramOffset + 1u], paramOffset + 1u < arrayLength(&analyticParams));
    if (shapeId == SHAPE_SPHERE) {
      localT = intersectSphereLocal(localRay, p0.xyz, max(p0.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_BOX) {
      localT = intersectAabbDetailed(localRay, p0.xyz - p1.xyz, p0.xyz + p1.xyz, 1e-4, INFINITY, &localN);
    } else if (shapeId == SHAPE_CAPSULE) {
      localT = intersectCapsuleLocal(localRay, p0.xyz, p1.xyz, max(p1.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_CYLINDER) {
      localT = intersectCylinderLocal(localRay, p0.xyz, max(p0.w, 1e-4), max(p1.x, 1e-4), &localN);
    } else if (shapeId == SHAPE_H_CHANNEL_CAME) {
      localT = intersectHChannelLocal(localRay, p0.x, p0.y, p0.z, p0.w, &localN);
    }
    if (localT <= tMin || localT >= INFINITY) {
      continue;
    }
    let localHitPos = localRay.origin + localRay.direction * localT;
    let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
    let worldT = dot(worldHitPos - ray.origin, ray.direction);
    if (worldT > tMin && worldT < hit.dist) {
      hit.didHit = true;
      hit.dist = worldT;
      hit.triIndex = params.triangleCount + ai;
      hit.normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localN);
    }
  }
  return hit;
}

fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  if (params.bvhNodeCount > 0u && arrayLength(&bvhNodes) > 0u) {
    var stack: array<u32, 64>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr = stackPtr + 1u;

    while (stackPtr > 0u) {
      stackPtr = stackPtr - 1u;
      let nodeIdx = stack[stackPtr];
      if (nodeIdx >= min(params.bvhNodeCount, arrayLength(&bvhNodes))) {
        continue;
      }
      let node = bvhNodes[nodeIdx];
      let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
      let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
      if (!intersectAabb(ray, bmin, bmax, tMin, tMax)) {
        continue;
      }

      let splitOrCount = node.splitAxisOrTriCount;
      if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
        let count = splitOrCount & 0x0000ffffu;
        let start = node.rightChildOrTriOffset;
        for (var i = 0u; i < count; i = i + 1u) {
          let t = start + i;
          if (t >= min(params.triangleCount, arrayLength(&indices))) {
            continue;
          }
          let tri = indices[t];
          if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
            continue;
          }
          let a = positions[tri.x].xyz;
          let b = positions[tri.y].xyz;
          let c = positions[tri.z].xyz;
          let hitT = intersectTriangle(ray.origin, ray.direction, a, b, c);
          if (hitT > tMin && hitT < tMax) {
            return true;
          }
        }
      } else {
        let leftChild = nodeIdx + 1u;
        let rightChild = node.rightChildOrTriOffset;
        if (stackPtr + 2u < 64u) {
          stack[stackPtr] = rightChild;
          stackPtr = stackPtr + 1u;
          stack[stackPtr] = leftChild;
          stackPtr = stackPtr + 1u;
        }
      }
    }
  }

  let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));
  for (var ai = 0u; ai < analyticTotal; ai = ai + 1u) {
    let header = analyticHeaders[ai];
    let shapeId = u32(max(header.x, 0.0));
    let paramOffset = u32(max(header.z, 0.0));
    let matBase = ai * 4u;
    if (matBase + 3u >= arrayLength(&analyticWorldToLocal) || matBase + 3u >= arrayLength(&analyticLocalToWorld)) {
      continue;
    }
    let w2l0 = analyticWorldToLocal[matBase];
    let w2l1 = analyticWorldToLocal[matBase + 1u];
    let w2l2 = analyticWorldToLocal[matBase + 2u];
    let w2l3 = analyticWorldToLocal[matBase + 3u];
    let l2w0 = analyticLocalToWorld[matBase];
    let l2w1 = analyticLocalToWorld[matBase + 1u];
    let l2w2 = analyticLocalToWorld[matBase + 2u];
    let l2w3 = analyticLocalToWorld[matBase + 3u];
    var localRay: Ray;
    localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
    localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
    var localN = vec3f(0.0, 1.0, 0.0);
    var localT = INFINITY;
    let p0 = select(vec4f(0.0), analyticParams[paramOffset], paramOffset < arrayLength(&analyticParams));
    let p1 = select(vec4f(0.0), analyticParams[paramOffset + 1u], paramOffset + 1u < arrayLength(&analyticParams));
    if (shapeId == SHAPE_SPHERE) {
      localT = intersectSphereLocal(localRay, p0.xyz, max(p0.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_BOX) {
      localT = intersectAabbDetailed(localRay, p0.xyz - p1.xyz, p0.xyz + p1.xyz, 1e-4, INFINITY, &localN);
    } else if (shapeId == SHAPE_CAPSULE) {
      localT = intersectCapsuleLocal(localRay, p0.xyz, p1.xyz, max(p1.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_CYLINDER) {
      localT = intersectCylinderLocal(localRay, p0.xyz, max(p0.w, 1e-4), max(p1.x, 1e-4), &localN);
    } else if (shapeId == SHAPE_H_CHANNEL_CAME) {
      localT = intersectHChannelLocal(localRay, p0.x, p0.y, p0.z, p0.w, &localN);
    }
    if (localT <= tMin || localT >= INFINITY) {
      continue;
    }
    let localHitPos = localRay.origin + localRay.direction * localT;
    let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
    let worldT = dot(worldHitPos - ray.origin, ray.direction);
    if (worldT > tMin && worldT < tMax) {
      return true;
    }
  }
  return false;
}

fn buildOnb(n: vec3f, t: ptr<function, vec3f>, b: ptr<function, vec3f>) {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  *t = normalize(cross(up, n));
  *b = cross(n, *t);
}

fn cosineHemisphereSample(rng: ptr<function, u32>, n: vec3f) -> vec3f {
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let local = vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
  var t: vec3f;
  var b: vec3f;
  buildOnb(n, &t, &b);
  return safe_normalize(local.x * t + local.y * b + local.z * n);
}

fn glossyReflectionSample(rng: ptr<function, u32>, r: vec3f, roughness: f32) -> vec3f {
  let jitterDir = cosineHemisphereSample(rng, r);
  let k = clamp(roughness * roughness, 0.0, 1.0);
  return safe_normalize(mix(r, jitterDir, k));
}

fn sampleRectAreaLight(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
  radiance: ptr<function, vec3f>,
) {
  let u = rand_f32(rng) * 2.0 - 1.0;
  let v = rand_f32(rng) * 2.0 - 1.0;
  let lp = params.rectAreaPos.xyz + params.rectAreaU.xyz * u + params.rectAreaV.xyz * v;
  let toLight = lp - hitPos;
  let dist2 = max(dot(toLight, toLight), 1e-6);
  let dist = sqrt(dist2);
  let wi = toLight / dist;
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 0.0) {
    return;
  }
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  let lightNormal = safe_normalize(cross(params.rectAreaU.xyz, params.rectAreaV.xyz));
  let cosLight = max(dot(lightNormal, -wi), 0.0);
  if (cosLight <= 0.0) {
    return;
  }
  let area = max(4.0 * length(cross(params.rectAreaU.xyz, params.rectAreaV.xyz)), 1e-6);
  let lightPdf = dist2 / max(cosLight * area, 1e-6);
  let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
  let misWeight = powerHeuristic(lightPdf, brdfPdf);
  let shadowRay = Ray(hitPos + normal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
    return;
  }
  *radiance = *radiance + throughput * brdf * nDotL * params.rectAreaRadiance.rgb * misWeight / max(lightPdf, 1e-6);
}

fn sampleMeshAreaLight(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
  radiance: ptr<function, vec3f>,
) {
  let a = params.meshAreaTriA.xyz;
  let b = params.meshAreaTriB.xyz;
  let c = params.meshAreaTriC.xyz;
  let r1 = rand_f32(rng);
  let r2 = rand_f32(rng);
  let su = sqrt(r1);
  let u = 1.0 - su;
  let v = r2 * su;
  let w = 1.0 - u - v;
  let lp = a * u + b * v + c * w;
  let toLight = lp - hitPos;
  let dist2 = max(dot(toLight, toLight), 1e-6);
  let dist = sqrt(dist2);
  let wi = toLight / dist;
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 0.0) {
    return;
  }
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  let lightNormal = safe_normalize(cross(b - a, c - a));
  let cosLight = max(dot(lightNormal, -wi), 0.0);
  if (cosLight <= 0.0) {
    return;
  }
  let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
  let lightPdf = dist2 / max(cosLight * area, 1e-6);
  let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
  let misWeight = powerHeuristic(lightPdf, brdfPdf);
  let shadowRay = Ray(hitPos + normal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
    return;
  }
  *radiance = *radiance + throughput * brdf * nDotL * params.meshAreaRadiance.rgb * misWeight / max(lightPdf, 1e-6);
}

fn causticMode() -> u32 {
  return u32(max(params.spotLightRadiance.w, 0.0));
}

fn manifoldNeeApproxContribution(
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  transmission: f32,
  ior: f32,
  throughput: vec3f,
) -> vec3f {
  if (transmission <= 1e-4 || params.lightDir.w <= 1e-6) return vec3f(0.0);
  let lightIn = -safe_normalize(params.lightDir.xyz);
  let eta = 1.0 / max(ior, 1.01);
  let refr = refract(lightIn, normal, eta);
  if (dot(refr, refr) <= 1e-8) return vec3f(0.0);
  let manifoldDir = safe_normalize(refr);
  let shadowRay = Ray(hitPos + normal * 1e-3, manifoldDir);
  if (traceAny(shadowRay, 1e-4, INFINITY)) return vec3f(0.0);
  let focus = pow(max(dot(manifoldDir, wo), 0.0), 16.0 + params.rectAreaU.w);
  let strength = transmission * (0.15 + 0.02 * params.rectAreaV.w);
  return throughput * baseColor * (params.lightDir.w * strength * focus);
}

fn photonMapApproxContribution(
  hitPos: vec3f,
  normal: vec3f,
  baseColor: vec3f,
  throughput: vec3f,
) -> vec3f {
  if (params.pointLightPos.w <= 0.5 && params.spotLightPos.w <= 0.5) return vec3f(0.0);
  var contrib = vec3f(0.0);
  if (params.pointLightPos.w > 0.5) {
    let toPoint = params.pointLightPos.xyz - hitPos;
    let dist2 = max(dot(toPoint, toPoint), 1e-4);
    let wi = toPoint / sqrt(dist2);
    let nDotL = max(dot(normal, wi), 0.0);
    let kernel = exp(-dist2 * 0.05);
    contrib = contrib + params.pointLightRadiance.rgb * nDotL * kernel / dist2;
  }
  if (params.spotLightPos.w > 0.5) {
    let toSpot = params.spotLightPos.xyz - hitPos;
    let dist2 = max(dot(toSpot, toSpot), 1e-4);
    let wi = toSpot / sqrt(dist2);
    let coneCos = dot(-wi, safe_normalize(params.spotLightDirection.xyz));
    if (coneCos >= params.spotLightDirection.w) {
      let nDotL = max(dot(normal, wi), 0.0);
      let softness = smoothstep(params.spotLightDirection.w, 1.0, coneCos);
      let kernel = exp(-dist2 * 0.05);
      contrib = contrib + params.spotLightRadiance.rgb * nDotL * softness * kernel / dist2;
    }
  }
  return throughput * baseColor * contrib * 0.12;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  var rng = pcgInit(gid.x, gid.y, params.frameSeed ^ params.frameIndex);
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var ray = generatePrimaryRay(gid.x, gid.y, jitter);

  var radiance = vec3f(0.0);
  var throughput = vec3f(1.0);
  let bounceLimit = max(1u, min(params.maxBounces, 8u));
  var firstHitValid = false;
  var firstHitPos = vec3f(0.0);
  var firstHitNormal = vec3f(0.0, 1.0, 0.0);
  var firstHitAlbedo = vec3f(0.0);
  var firstHitDepth = 0.0;

  for (var bounce = 0u; bounce < bounceLimit; bounce = bounce + 1u) {
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) {
      radiance = radiance + throughput * sampleEnvironmentColor(ray.direction);
      break;
    }

    var matId = 0u;
    if (hit.triIndex < params.triangleCount) {
      matId = select(0u, triMaterialIds[hit.triIndex], hit.triIndex < arrayLength(&triMaterialIds));
    } else {
      let analyticIndex = hit.triIndex - params.triangleCount;
      if (analyticIndex < arrayLength(&analyticHeaders)) {
        matId = u32(max(analyticHeaders[analyticIndex].y, 0.0));
      }
    }
    let m0Index = matId * 3u;
    let m1Index = m0Index + 1u;
    let m2Index = m0Index + 2u;
    let m0 = select(vec4f(0.8, 0.8, 0.8, 0.6), materials[m0Index], m0Index < arrayLength(&materials));
    let m1 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m1Index], m1Index < arrayLength(&materials));
    let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
    let baseColor = m0.rgb;
    let roughness = clamp(m0.w, 0.02, 1.0);
    let emissive = m1.rgb;
    let metallic = clamp(m1.w, 0.0, 1.0);
    let transmission = clamp(m2.x, 0.0, 1.0);
    let ior = clamp(m2.y, 1.0, 2.5);

    radiance = radiance + throughput * emissive;

    let hitPos = ray.origin + ray.direction * hit.dist;
    let normal = select(hit.normal, -hit.normal, dot(hit.normal, ray.direction) > 0.0);
    if (!firstHitValid) {
      firstHitValid = true;
      firstHitPos = hitPos;
      firstHitNormal = normal;
      firstHitAlbedo = baseColor;
      firstHitDepth = hit.dist;
    }
    let wo = -ray.direction;
    let throughputAtVertex = throughput;
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0 = mix(vec3f(0.04), baseColor, metallic);
    let fresnel = fresnelSchlick(cosThetaO, f0);

    var lightCount = 0u;
    if (params.lightDir.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    if (params.pointLightPos.w > 0.5) {
      lightCount = lightCount + 1u;
    }
    if (params.spotLightPos.w > 0.5) {
      lightCount = lightCount + 1u;
    }
    if (params.rectAreaPos.w > 0.5) {
      lightCount = lightCount + 1u;
    }
    if (params.meshAreaTriA.w > 0.5) {
      lightCount = lightCount + 1u;
    }
    if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    if (lightCount > 0u) {
      let picked = u32(min(floor(rand_f32(&rng) * f32(lightCount)), f32(lightCount - 1u)));
      var current = 0u;
      var directLi = vec3f(0.0);
      if (params.lightDir.w > 1e-6) {
        if (current == picked) {
          let lightDir = safe_normalize(params.lightDir.xyz);
          let shadowRay = Ray(hitPos + normal * 1e-3, lightDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let nDotL = max(0.0, dot(normal, lightDir));
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, lightDir);
            directLi = throughput * brdf * nDotL * (1.8 * params.lightDir.w);
          }
        }
        current = current + 1u;
      }
      if (params.pointLightPos.w > 0.5) {
        if (current == picked) {
          let toPoint = params.pointLightPos.xyz - hitPos;
          let dist2 = max(dot(toPoint, toPoint), 1e-5);
          let dist = sqrt(dist2);
          let wi = toPoint / dist;
          let pointShadowRay = Ray(hitPos + normal * 1e-3, wi);
          if (!traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
            let nDotL = max(0.0, dot(normal, wi));
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            directLi = throughput * brdf * nDotL * (params.pointLightRadiance.rgb / dist2);
          }
        }
        current = current + 1u;
      }
      if (params.spotLightPos.w > 0.5) {
        if (current == picked) {
          let toSpot = params.spotLightPos.xyz - hitPos;
          let dist2 = max(dot(toSpot, toSpot), 1e-5);
          let dist = sqrt(dist2);
          let wi = toSpot / dist;
          let coneCos = dot(-wi, safe_normalize(params.spotLightDirection.xyz));
          if (coneCos >= params.spotLightDirection.w) {
            let spotShadowRay = Ray(hitPos + normal * 1e-3, wi);
            if (!traceAny(spotShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
              let nDotL = max(0.0, dot(normal, wi));
              let softness = smoothstep(params.spotLightDirection.w, 1.0, coneCos);
              let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
              directLi = throughput * brdf * nDotL * softness * (params.spotLightRadiance.rgb / dist2);
            }
          }
        }
        current = current + 1u;
      }
      if (params.rectAreaPos.w > 0.5) {
        if (current == picked) {
          sampleRectAreaLight(&rng, hitPos, normal, wo, baseColor, roughness, metallic, transmission, throughput, &directLi);
        }
        current = current + 1u;
      }
      if (params.meshAreaTriA.w > 0.5 && current == picked) {
        sampleMeshAreaLight(&rng, hitPos, normal, wo, baseColor, roughness, metallic, transmission, throughput, &directLi);
      }
      if (params.meshAreaTriA.w > 0.5) {
        current = current + 1u;
      }
      if ((hasEnvironmentMap() || params.environmentSun.w > 1e-6) && current == picked) {
        var envDir = vec3f(0.0, 1.0, 0.0);
        var envColor = vec3f(0.0);
        var envPdf = 0.0;
        let sampled = sampleEnvironmentImportance(&rng, &envDir, &envColor, &envPdf);
        if (!sampled) {
          envDir = cosineHemisphereSample(&rng, normal);
          envColor = sampleEnvironmentColor(envDir);
          envPdf = max(environmentPdf(envDir), 1e-8);
        }
        let nDotL = max(dot(normal, envDir), 0.0);
        if (nDotL > 1e-6) {
          let shadowRay = Ray(hitPos + normal * 1e-3, envDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, envDir);
            let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, envDir);
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = throughput * brdf * nDotL * envColor * misWeight / max(envPdf, 1e-8);
          }
        }
      }
      radiance = radiance + directLi * f32(lightCount);
    }

    if (causticMode() == 1u) {
      radiance = radiance + manifoldNeeApproxContribution(
        hitPos,
        normal,
        wo,
        baseColor,
        transmission,
        ior,
        throughputAtVertex,
      );
    } else if (causticMode() == 2u) {
      radiance = radiance + photonMapApproxContribution(
        hitPos,
        normal,
        baseColor,
        throughputAtVertex,
      );
    }

    let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
    let baseTransProb = clamp(transmission * (1.0 - metallic), 0.0, 0.95);
    let baseDiffProb = max(0.0, (1.0 - metallic) * (1.0 - transmission));
    let sumProb = max(baseSpecProb + baseTransProb + baseDiffProb, 1e-4);
    let specProb = baseSpecProb / sumProb;
    let transProb = baseTransProb / sumProb;
    let diffProb = baseDiffProb / sumProb;
    let xi = rand_f32(&rng);
    let chooseTransmission = xi < transProb;
    let chooseSpecular = !chooseTransmission && (xi < transProb + specProb);
    var sampledDir = vec3f(0.0);
    var sampleAllowsAreaMis = false;
    if (chooseTransmission) {
      let frontFace = dot(ray.direction, hit.normal) < 0.0;
      let eta = select(ior, 1.0 / ior, frontFace);
      let refr = refract(ray.direction, normal, eta);
      let validRefract = dot(refr, refr) > 1e-8;
      let idealReflect = reflect(ray.direction, normal);
      let outDir = select(idealReflect, safe_normalize(refr), validRefract);
      let offsetN = select(-normal, normal, dot(outDir, normal) > 0.0);
      ray.origin = hitPos + offsetN * 1e-3;
      sampledDir = glossyReflectionSample(&rng, outDir, roughness * 0.5);
      ray.direction = sampledDir;
      throughput = throughput * mix(vec3f(1.0), baseColor, 0.15) / max(transProb, 1e-4);
    } else if (chooseSpecular) {
      let refl = reflect(ray.direction, normal);
      ray.origin = hitPos + normal * 1e-3;
      sampledDir = glossyReflectionSample(&rng, refl, roughness);
      ray.direction = sampledDir;
      sampleAllowsAreaMis = true;
      throughput = throughput * fresnel / max(specProb, 1e-4);
    } else {
      ray.origin = hitPos + normal * 1e-3;
      sampledDir = cosineHemisphereSample(&rng, normal);
      ray.direction = sampledDir;
      sampleAllowsAreaMis = true;
      let kd = (vec3f(1.0) - fresnel) * (1.0 - metallic);
      throughput = throughput * (kd * baseColor) / max(diffProb, 1e-4);
    }

    if (sampleAllowsAreaMis) {
      radiance = radiance + bsdfAreaLightConnectionContribution(
        hitPos,
        normal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        throughputAtVertex,
      );
      radiance = radiance + bsdfEnvironmentConnectionContribution(
        hitPos,
        normal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        throughputAtVertex,
      );
    }

    if (bounce > 2u) {
      let survival = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.1, 0.95);
      if (rand_f32(&rng) > survival) {
        break;
      }
      throughput = throughput / survival;
    }
  }

  let sampleColor = max(radiance, vec3f(0.0));

  let pixelIndex = gid.y * params.width + gid.x;
  var accum = accumBuffer[pixelIndex];
  accum = accum + vec4f(sampleColor, 1.0);
  accumBuffer[pixelIndex] = accum;
  let sampleLum = luminance(sampleColor);
  var moments = varianceMomentsBuffer[pixelIndex];
  moments.x = moments.x + sampleLum;
  moments.y = moments.y + sampleLum * sampleLum;
  moments.z = moments.z + 1.0;
  varianceMomentsBuffer[pixelIndex] = moments;

  let display = accum.xyz / max(accum.w, 1.0);
  let count = max(moments.z, 1.0);
  let mean = moments.x / count;
  let varL = max(0.0, moments.y / count - mean * mean);
  textureStore(outputTexture, vec2i(gid.xy), vec4f(display, 1.0));
  if (firstHitValid) {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal, firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
    let ndc = projectToNdc(firstHitPos, params.viewProj);
    let prevNdc = projectToNdc(firstHitPos, params.prevViewProj);
    let motionPx = (ndc - prevNdc) * 0.5 * vec2f(f32(params.width), f32(params.height));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(motionPx, 0.0, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
  }
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(varL, varL, varL, 1.0));
}
`;
