/**
 * adjointPass.wgsl.ts — the engine-side WS5 Phase-1 path-replay adjoint COMPUTE
 * PASS (the last V24 piece). For each pixel it re-traces the frozen-seed primary
 * ray (brute-force closest-hit), re-derives the single-bounce direct lighting
 * (point-light NEE with shadow rays — the SAME `rad/dist²` model + packing the
 * forward `kernel.wgsl.ts` uses), and accumulates `∂loss/∂θ` for the optimized
 * material parameters through the two GPU-VALIDATED adjoint stages:
 *   - the BRDF partials `dBrdf_dBaseColor` / `dBrdf_dRoughness`
 *     (`pathTraceAdjoint.wgsl.ts`, GPU == FD oracle to f32),
 *   - the chain rule + fixed-point `adjointScatter` accumulation
 *     (`adjointHarness.wgsl.ts`, analytic == on-device FD).
 *
 * It deliberately does NOT call the forward `evaluateBrdf` — the per-pixel
 * `dLoss/dRendered` is handed in by `inverseSession` (computed from the baseline
 * render vs target), so the pass only needs the DERIVATIVES of the shading.
 *
 * Scope (Phase 1, matching the differentiable set): single bounce, brute-force
 * intersection (Phase-1 inverse scenes are small — Cornell-scale), directional
 * delta + point + spot + center-sampled rect/disc/mesh-area direct lights.
 * Environment, indirect, soft-sun angular diameter, and mapped/transmissive/
 * layered/volume/spectral material terms remain deliberate finite-difference
 * fallbacks until their source terms are mirrored here and GPU-validated.
 * Direct lights are summed deterministically over all eligible lights (no MC
 * light selection: the adjoint is the deterministic expectation; finite area
 * lights are center-sampled). baseColor/roughness/metallic/specular/clearcoat
 * scalar params share this direct-light BRDF path. Map-free unlit baseColor is a primary-hit
 * contribution-level identity (`radiance += throughput * baseColor`) and is
 * scattered without requiring any light. The shading normal is faced toward the viewer
 * (the same flip the forward shade prologue applies). The primary-ray jitter
 * sequence matches the inverse baseline render:
 * sample `s` uses `frameSeed = 0x5eed5eed + s`, `frameIndex = 0`, then the pass
 * averages the per-sample derivatives. The sampled directions are FROZEN (path
 * replay differentiates only the continuous shading, never the light/BSDF
 * sampling — sidesteps visibility discontinuities).
 *
 * A FOCUSED single-group pipeline (not a forward-kernel variant): the forward
 * spends all 4 bind groups, so the adjoint binds only the read subset it needs +
 * its own I/O, in group 0.
 *
 * Ref: Vicini 2021 (Path Replay Backprop); Möller-Trumbore 1997 (intersection).
 */
import { PCG_WGSL } from '@vitrum/shared-samplers';
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from './pathTraceAdjoint.wgsl.js';

/** Field codes in the adjointParams descriptor (matches inverseSession fields). */
export const ADJOINT_FIELD_BASECOLOR = 0;
export const ADJOINT_FIELD_ROUGHNESS = 1;
/** Emissive (rgb). UNLIKE baseColor/roughness this is NOT a lit-surface NEE term:
 *  the forward adds `throughput · emissive` for the emission a CAMERA ray sees the
 *  surface emit DIRECTLY at the PRIMARY hit (shadePrologue.wgsl.ts:63, with
 *  throughput = 1 and prevSampleAllowsAreaMis false on a camera ray). Its partial
 *  ∂rendered_c/∂emissive_c = throughput_c · emissiveIntensity is a self-source — it
 *  needs no light. The descriptor carries the (fixed) emissiveIntensity in `.w`
 *  (bitcast f32), because the packed material folds intensity INTO emissive.rgb. */
export const ADJOINT_FIELD_EMISSIVE = 2;
export const ADJOINT_FIELD_SPECULAR_COLOR = 3;
export const ADJOINT_FIELD_SPECULAR_INTENSITY = 4;
export const ADJOINT_FIELD_METALLIC = 5;
export const ADJOINT_FIELD_EMISSIVE_INTENSITY = 6;
export const ADJOINT_FIELD_CLEARCOAT = 7;
export const ADJOINT_FIELD_CLEARCOAT_ROUGHNESS = 8;

/** AdjointParams UBO size in bytes (mat4 + vec4 + 3×uvec4). */
export const ADJOINT_PARAMS_UBO_BYTES = 64 + 16 + 16 + 16 + 16;

export const PT_WEBGPU_ADJOINT_PASS_WGSL = /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
// MUST match the canonical MATERIAL_VEC4_STRIDE (material.wgsl.ts / materialPacking.ts).
// This adjoint pass reads the SAME materials storage buffer the forward kernel
// uploads, so its per-material stride must equal the forward stride or every
// matId>0 material read is misaligned. Was a stale 23u (the stride at the time
// this pass was written); the forward stride has since grown through WS4/H52/A3,
// SPEC-01, and VOL-THICKNESS. matId=0 is unaffected by a stale stride because
// 0*stride=0, so single-material adjoint tests can miss this latent multi-material
// inverse-fit misalignment.
const MATERIAL_VEC4_STRIDE = 29u;

struct AdjointParams {
  invViewProj: mat4x4f,
  cameraPos:   vec4f,
  width:       u32,
  height:      u32,
  triangleCount: u32,
  pointLightCount: u32,
  paramCount:  u32,
  channels:    u32,
  rectAreaLightCount: u32,
  sampleCount: u32,
  directionalLightCount: u32,
  spotLightCount: u32,
  meshAreaLightCount: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform>             params:        AdjointParams;
@group(0) @binding(1) var<storage, read>       positions:     array<vec4f>;
@group(0) @binding(2) var<storage, read>       indices:       array<vec4u>;
@group(0) @binding(3) var<storage, read>       triMaterialIds: array<u32>;
@group(0) @binding(4) var<storage, read>       materials:     array<vec4f>;
@group(0) @binding(5) var<storage, read>       normals:       array<vec4f>;
@group(0) @binding(6) var<storage, read>       pointLights:   array<vec4f>;
@group(0) @binding(7) var<storage, read>       dLossDRendered: array<f32>;
@group(0) @binding(8) var<storage, read_write> gradAccum:     array<atomic<i32>>;
// adjointParams: per optimized param {matId, fieldCode, gradOffset, _}.
// adjointParamDescs: two vec4u records per optimized param:
//   record 0: {matId, fieldCode, gradOffset, fieldPayloadBits}
//   record 1: {payloadXBits, payloadYBits, payloadZBits, _}
// emissive uses record0.w for fixed emissiveIntensity. emissiveIntensity
// uses record1.xyz for UNFACTORED emissive RGB so intensity=0 is differentiable.
@group(0) @binding(9) var<storage, read>       adjointParamDescs: array<vec4u>;
// rect-area lights: per light {position, uAxis, vAxis, radiance} (4 vec4 stride).
@group(0) @binding(10) var<storage, read>      rectAreaLights: array<vec4f>;
// directional lights: per light {towardLight+angularDiameter, irradiance+mean} (2 vec4 stride).
@group(0) @binding(11) var<storage, read>      directionalLights: array<vec4f>;
// spot lights: per light {position, axis+cosOuter, radiance+cosInner, distance+decay+shadowFlag} (4 vec4 stride).
@group(0) @binding(12) var<storage, read>      spotLights: array<vec4f>;
// mesh-area lights: per triangle {a, b, c, radiance+shadowFlag} (4 vec4 stride).
@group(0) @binding(13) var<storage, read>      meshAreaLights: array<vec4f>;

// ── BRDF primitives ──────────────────────────────────────────────────────────
const ADJOINT_FROZEN_SEED_BASE = 0x5eed5eedu;
${PCG_WGSL}
//
// MIRROR SITE — these four functions are intentionally duplicated here from
// their canonical definitions so this adjoint-pass shader is a self-contained
// compute module (it is NOT composed with the megakernel prefix stack).
//
//   safe_normalize   → common.wgsl.ts:42
//   ggxD             → material.wgsl.ts:741  (GGX NDF, Trowbridge-Reitz)
//   smithG1          → material.wgsl.ts:747  (Smith masking/shadowing term)
//   fresnelSchlick   → material.wgsl.ts:710  (Schlick Fresnel approximation)
//
// If you change the body of any of these functions in their canonical location
// you MUST apply the same change here (and vice-versa).
fn safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-8) { return vec3f(0.0); }
  return v / l;
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
fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

// ── the GPU-validated BRDF partials + adjointScatter (gradAccum at binding 8) ──
${PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL}

// ── camera (mirror kernelCore.generatePrimaryRay) ───────────────────────────
struct Ray { origin: vec3f, direction: vec3f }
fn generatePrimaryRay(px: u32, py: u32, jitter: vec2f) -> Ray {
  let uv = (vec2f(f32(px), f32(py)) + jitter) / vec2f(f32(params.width), f32(params.height));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4 = params.invViewProj * vec4f(ndc, 1.0, 1.0);
  let near4 = params.invViewProj * vec4f(ndc, -1.0, 1.0);
  var ray: Ray;
  ray.origin = params.cameraPos.xyz;
  ray.direction = safe_normalize((far4.xyz / far4.w) - (near4.xyz / near4.w));
  return ray;
}

// ── brute-force intersection (Möller-Trumbore) ──────────────────────────────
struct Hit { valid: bool, t: f32, tri: u32, bary: vec3f }
fn closestHit(ro: vec3f, rd: vec3f) -> Hit {
  var best: Hit;
  best.valid = false;
  best.t = 1e30;
  for (var i = 0u; i < params.triangleCount; i = i + 1u) {
    let idx = indices[i];
    let v0 = positions[idx.x].xyz;
    let e1 = positions[idx.y].xyz - v0;
    let e2 = positions[idx.z].xyz - v0;
    let p = cross(rd, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) { continue; }
    let invDet = 1.0 / det;
    let tvec = ro - v0;
    let u = dot(tvec, p) * invDet;
    if (u < 0.0 || u > 1.0) { continue; }
    let q = cross(tvec, e1);
    let v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) { continue; }
    let t = dot(e2, q) * invDet;
    if (t > 1e-4 && t < best.t) {
      best.valid = true; best.t = t; best.tri = i; best.bary = vec3f(1.0 - u - v, u, v);
    }
  }
  return best;
}
fn anyHit(ro: vec3f, rd: vec3f, tMax: f32) -> bool {
  for (var i = 0u; i < params.triangleCount; i = i + 1u) {
    let idx = indices[i];
    let v0 = positions[idx.x].xyz;
    let e1 = positions[idx.y].xyz - v0;
    let e2 = positions[idx.z].xyz - v0;
    let p = cross(rd, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) { continue; }
    let invDet = 1.0 / det;
    let tvec = ro - v0;
    let u = dot(tvec, p) * invDet;
    if (u < 0.0 || u > 1.0) { continue; }
    let q = cross(tvec, e1);
    let v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) { continue; }
    let t = dot(e2, q) * invDet;
    if (t > 1e-4 && t < tMax) { return true; }
  }
  return false;
}

struct DirectLightAdjoint {
  baseColor: vec3f,
  roughness: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  metallicGrad: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
}

fn directLightAdjoint(
  dLoss_dR: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  nDotL: f32,
  Li: vec3f,
) -> DirectLightAdjoint {
  let gBaseColor = dLoss_dR * dBrdf_dBaseColorWithSpecular(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li;
  let gRough = dot(dLoss_dR, dBrdf_dRoughnessWithSpecular(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li);
  let gSpecularColor = dLoss_dR * dBrdf_dSpecularColor(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li;
  let gSpecularIntensity = dot(dLoss_dR, dBrdf_dSpecularIntensity(
    baseColor, roughness, metallic, n, wo, wi, specularColor,
  ) * nDotL * Li);
  let gMetallic = dot(dLoss_dR, dBrdf_dMetallic(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li);
  let gClearcoat = dot(dLoss_dR, dBrdf_dClearcoat(
    clearcoatRoughness, n, wo, wi,
  ) * nDotL * Li);
  let gClearcoatRoughness = dot(dLoss_dR, dBrdf_dClearcoatRoughness(
    clearcoat, clearcoatRoughness, n, wo, wi,
  ) * nDotL * Li);
  return DirectLightAdjoint(
    gBaseColor, gRough, gSpecularColor, gSpecularIntensity, gMetallic,
    gClearcoat, gClearcoatRoughness,
  );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  // Per-pixel ∂loss/∂rendered (the session computed it from baseline vs target).
  let pixel = gid.y * params.width + gid.x;
  let base = pixel * params.channels;
  let dLoss_dR = vec3f(dLossDRendered[base], dLossDRendered[base + 1u], dLossDRendered[base + 2u]);

  let replaySamples = max(params.sampleCount, 1u);
  let invReplaySamples = 1.0 / f32(replaySamples);
  for (var sampleIdx = 0u; sampleIdx < replaySamples; sampleIdx = sampleIdx + 1u) {
    var rng = pcgInit(gid.x, gid.y, ADJOINT_FROZEN_SEED_BASE + sampleIdx);
    let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
    let ray = generatePrimaryRay(gid.x, gid.y, jitter);
    let hit = closestHit(ray.origin, ray.direction);
    if (!hit.valid) { continue; }

	    let matId = triMaterialIds[hit.tri];
	    let m0 = materials[matId * MATERIAL_VEC4_STRIDE];
	    let m1 = materials[matId * MATERIAL_VEC4_STRIDE + 1u];
	    let m23 = materials[matId * MATERIAL_VEC4_STRIDE + 23u];
	    let m26 = materials[matId * MATERIAL_VEC4_STRIDE + 26u];
	    let m27 = materials[matId * MATERIAL_VEC4_STRIDE + 27u];
	    let baseColor = m0.rgb;
	    let roughness = clamp(m0.w, 0.02, 1.0);
	    let metallic = clamp(m1.w, 0.0, 1.0);
	    let isUnlit = (u32(max(m26.w, 0.0)) & 2u) != 0u;
	    let specularColor = m27.rgb;
	    let specularIntensity = clamp(m27.w, 0.0, 1.0);
	    let clearcoat = clamp(m23.x, 0.0, 1.0);
	    let clearcoatRoughness = clamp(m23.y, 0.0, 1.0);

    let idx = indices[hit.tri];
    let nGeo = safe_normalize(hit.bary.x * normals[idx.x].xyz + hit.bary.y * normals[idx.y].xyz + hit.bary.z * normals[idx.z].xyz);
    // Face the shading normal toward the viewer — the SAME flip the forward shade
    // prologue applies (shadePrologue.wgsl.ts). Without it, back-facing geometry
    // gets nDotL<=0 against an interior light and contributes no gradient.
    let n = select(-nGeo, nGeo, dot(nGeo, ray.direction) < 0.0);
    let pos = ray.origin + ray.direction * hit.t;
    let wo = -ray.direction;

    // Emissive partial — NOT a NEE term. The forward adds throughput * emissive for
    // the emission this surface is seen to emit DIRECTLY by the camera ray at THIS
    // (primary) hit (shadePrologue.wgsl.ts:63). Path-replay's primary hit has
    // throughput = 1, so d(rendered_c)/d(emissive_c) = emissiveIntensity (dContribution_
    // dEmissive with throughput = 1), and d(loss)/d(emissive_c) = dLoss_dR_c * intensity.
    // Independent of light visibility — computed here, scattered per-descriptor below
    // with that descriptor's fixed emissiveIntensity (carried in .w). It is gated by
    // the matId match in the scatter loop, so a pixel only contributes to the emissive
    // gradient when ITS primary-hit material is the optimized emissive primitive.
    let dRendered_dEmissivePerUnitIntensity = dContribution_dEmissive(vec3f(1.0), 1.0); // = (1,1,1)

    // Single-bounce direct lighting, summed deterministically over every direct
    // light source the scoped adjoint pass mirrors.
    var gBaseColor = vec3f(0.0);
    var gRough = 0.0;
	    var gSpecularColor = vec3f(0.0);
	    var gSpecularIntensity = 0.0;
	    var gMetallic = 0.0;
	    var gClearcoat = 0.0;
	    var gClearcoatRoughness = 0.0;
    for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
      let dBase = di * 2u;
      let dDirAD = directionalLights[dBase];
      let dIrrMean = directionalLights[dBase + 1u];
      if (dIrrMean.w <= 1e-6) { continue; }
      // Delta directional adjoint: exact for angularDiameter≈0. Soft-sun cone
      // scenes are kept on finite-difference by inverseSession routing; if this
      // pass is called directly anyway, use the cone center as a stable estimate.
      let wi = safe_normalize(dDirAD.xyz);
      let directionalShadowDisabled = dDirAD.w < 0.0;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (!directionalShadowDisabled && anyHit(pos + n * 1e-3, wi, 1e30)) { continue; }
      let lg = directLightAdjoint(
        dLoss_dR, baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
        clearcoat, clearcoatRoughness,
        nDotL, dIrrMean.rgb,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
    }

    for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
      let lp = pointLights[pi * 3u].xyz;
      let rad = pointLights[pi * 3u + 1u].rgb;
      let ptExtra = pointLights[pi * 3u + 2u];
      let ptMaxDist = ptExtra.x;
      let ptDecay   = ptExtra.y;
      let toPoint = lp - pos;
      let dist2 = max(dot(toPoint, toPoint), 1e-5);
      let dist = sqrt(dist2);
      if (ptMaxDist > 0.0 && dist > ptMaxDist) { continue; }
      let wi = toPoint / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (ptExtra.z <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; } // shadowed
      let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -ptDecay), ptDecay > 0.01);
      let Li = rad * attenuation;
      let lg = directLightAdjoint(
        dLoss_dR, baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
        clearcoat, clearcoatRoughness,
        nDotL, Li,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
    }

    for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
      let sb = si * 4u;
      let spos = spotLights[sb].xyz;
      let saxis = spotLights[sb + 1u];
      let sradW = spotLights[sb + 2u];
      let spExtra = spotLights[sb + 3u];
      let spotDir = safe_normalize(saxis.xyz);
      let cosOuter = saxis.w;
      let cosInner = sradW.w;
      let toSpot = spos - pos;
      let dist2 = max(dot(toSpot, toSpot), 1e-5);
      let dist = sqrt(dist2);
      if (spExtra.x > 0.0 && dist > spExtra.x) { continue; }
      let wi = toSpot / dist;
      let coneCos = dot(-wi, spotDir);
      if (coneCos < cosOuter) { continue; }
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (spExtra.z <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; }
      let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
      let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -spExtra.y), spExtra.y > 0.01);
      let Li = sradW.rgb * softness * attenuation;
      let lg = directLightAdjoint(
        dLoss_dR, baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
        clearcoat, clearcoatRoughness,
        nDotL, Li,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
    }

    // Rect-area lights: deterministic CENTER-sample of the same geometric term the
    // forward area NEE integrates (brdf·nDotL·radiance·cosLight·area/dist²). One
    // sample is a biased-but-correct-direction estimate of the area integral — good
    // enough for the gradient direction (Adam handles the magnitude). GPU-validated
    // on a camera-near rect-area light (directly lights the visible target).
    for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
      let rb = ri * 4u;
      let rpos = rectAreaLights[rb].xyz;
      let ru = rectAreaLights[rb + 1u].xyz;
      let rv = rectAreaLights[rb + 2u].xyz;
      let rshape = rectAreaLights[rb + 3u];
      let rad = rshape.rgb;
      let isDisc = abs(rshape.w - 1.0) < 0.5;
      let toLight = rpos - pos;
      let dist2 = max(dot(toLight, toLight), 1e-6);
      let dist = sqrt(dist2);
      let wi = toLight / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      let lightNormal = safe_normalize(cross(ru, rv));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight <= 0.0) { continue; }
      if (rectAreaLights[rb].w <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; } // shadowed
      let area = select(
        max(4.0 * length(cross(ru, rv)), 1e-6),
        max(PI * dot(ru, ru), 1e-6),
        isDisc,
      );
      let Li = rad * (cosLight * area / dist2);
      let lg = directLightAdjoint(
        dLoss_dR, baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
        clearcoat, clearcoatRoughness,
        nDotL, Li,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
    }

    // Mesh-area lights: deterministic CENTER-sample of each packed emissive
    // triangle. This mirrors the rect/disc scoped adjoint: it differentiates a
    // stable direct-light estimate without pretending to cover the stochastic
    // full triangle sampler or light-selection MIS yet.
    for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
      let mb = mi * 4u;
      let a = meshAreaLights[mb].xyz;
      let b = meshAreaLights[mb + 1u].xyz;
      let c = meshAreaLights[mb + 2u].xyz;
      let mr = meshAreaLights[mb + 3u];
      let center = (a + b + c) * (1.0 / 3.0);
      let edgeCross = cross(b - a, c - a);
      let area = max(0.5 * length(edgeCross), 1e-6);
      let lightNormal = safe_normalize(edgeCross);
      let toLight = center - pos;
      let dist2 = max(dot(toLight, toLight), 1e-6);
      let dist = sqrt(dist2);
      let wi = toLight / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight <= 0.0) { continue; }
      if (mr.w <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; }
      let Li = mr.rgb * (cosLight * area / dist2);
      let lg = directLightAdjoint(
        dLoss_dR, baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
        clearcoat, clearcoatRoughness,
        nDotL, Li,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
    }

    // Scatter into the gradient slot of every param that targets THIS hit's material
    // (the matId gate is what makes the emissive gradient respond to the optimized
    // primitive's own pixels — a pixel whose primary hit is a different material
    // contributes nothing to that primitive's emissive slot). Scale by
    // 1/sampleCount because the baseline render is the mean of the same frozen
    // sample sequence.
    for (var k = 0u; k < params.paramCount; k = k + 1u) {
      let descBase = k * 2u;
      let d = adjointParamDescs[descBase];
      let payload = adjointParamDescs[descBase + 1u];
      if (d.x != matId) { continue; }
      let gradOffset = d.z;
      if (d.y == ${ADJOINT_FIELD_BASECOLOR}u) {
        let gUnlitBaseColor = dLoss_dR;
        let gBase = select(gBaseColor, gUnlitBaseColor, isUnlit);
        adjointScatter(gradOffset, gBase.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gBase.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gBase.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SPECULAR_COLOR}u) {
        adjointScatter(gradOffset, gSpecularColor.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gSpecularColor.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gSpecularColor.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SPECULAR_INTENSITY}u) {
        adjointScatter(gradOffset, gSpecularIntensity * invReplaySamples);
	      } else if (d.y == ${ADJOINT_FIELD_METALLIC}u) {
	        adjointScatter(gradOffset, gMetallic * invReplaySamples);
	      } else if (d.y == ${ADJOINT_FIELD_CLEARCOAT}u) {
	        adjointScatter(gradOffset, gClearcoat * invReplaySamples);
	      } else if (d.y == ${ADJOINT_FIELD_CLEARCOAT_ROUGHNESS}u) {
	        adjointScatter(gradOffset, gClearcoatRoughness * invReplaySamples);
	      } else if (d.y == ${ADJOINT_FIELD_EMISSIVE}u) {
        // ∂loss/∂emissive_c = dLoss_dR_c · emissiveIntensity. The packed material
        // folds intensity into emissive.rgb, so the host hands the fixed
        // emissiveIntensity in the descriptor's .w (bitcast f32); the partial per
        // unit intensity is (1,1,1) at the primary hit (throughput = 1).
        let emissiveIntensity = bitcast<f32>(d.w);
        let gEmissive = dLoss_dR * dRendered_dEmissivePerUnitIntensity * emissiveIntensity;
        adjointScatter(gradOffset, gEmissive.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gEmissive.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gEmissive.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_EMISSIVE_INTENSITY}u) {
        // ∂loss/∂emissiveIntensity = Σ_c dLoss_dR_c · emissive_c. The host
        // descriptor carries unfactored emissive RGB in payload.xyz; using packed
        // material emissive/intensity would be undefined when intensity is zero.
        let emissiveRgb = vec3f(
          bitcast<f32>(payload.x),
          bitcast<f32>(payload.y),
          bitcast<f32>(payload.z),
        );
        let gIntensity = dot(
          dLoss_dR,
          dContribution_dEmissiveIntensity(vec3f(1.0), emissiveRgb),
        );
        adjointScatter(gradOffset, gIntensity * invReplaySamples);
      } else {
        adjointScatter(gradOffset, gRough * invReplaySamples);
      }
    }
  }
}
`;
