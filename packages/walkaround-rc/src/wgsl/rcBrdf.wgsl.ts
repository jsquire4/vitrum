/**
 * RC PBR BRDF lobes — raw WGSL fragment.
 *
 * Extracted verbatim (byte-identical) from `probeRayCast.wgsl.ts`. This fragment
 * carries the probe-ray direct-response BRDF: base F0 / Fresnel-Schlick,
 * iridescence-modified F0, isotropic + anisotropic GGX distribution and Smith
 * geometry, the clearcoat and sheen (Charlie) lobes, and the
 * `rcEvaluateProbeDirectResponse` combiner.
 *
 * Depends on symbols declared earlier in the assembly root: `RC_PI`, `RC_INV_PI`
 * (rcMaterialAtlas fragment), `safe_normalize` (probeRayCast body), and the
 * `RCProbeHitMaterial` struct (rcMaterialAtlas fragment). This fragment is a RAW
 * STRING interpolated into the consumer body AFTER those declarations — it is NOT
 * a standalone `WgslModule`. The composed output is byte-identical to the pre-split
 * single-file literal (pinned by `__tests__/probeRayCastByteIdentity.test.ts`).
 */

export const RC_BRDF_WGSL = /* wgsl */ `fn rcBaseMaterialF0(mat: RCProbeHitMaterial) -> vec3f {
  let dielectricF0 = vec3f(0.04) * mat.specular.rgb * mat.specular.a;
  return mix(dielectricF0, mat.albedo, clamp(mat.metalness, 0.0, 1.0));
}

fn rcFresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * f;
}

fn rcIridescenceModifiedF0(baseF0: vec3f, iridescence: vec4f, vDotH: f32) -> vec3f {
  let factor = clamp(iridescence.x, 0.0, 1.0);
  if (factor <= 1e-4) {
    return baseF0;
  }
  let thickness = max(0.0, (iridescence.z + iridescence.w) * 0.5);
  let iorShift = clamp(iridescence.y - 1.0, 0.0, 2.0) * 0.12;
  let phase = thickness * 0.012 + (1.0 - clamp(vDotH, 0.0, 1.0)) * RC_PI;
  let filmTint = clamp(
    0.5 + 0.5 * cos(vec3f(phase, phase + 2.0943951, phase + 4.1887902)) + vec3f(iorShift),
    vec3f(0.0),
    vec3f(1.0),
  );
  let filmF0 = mix(vec3f(0.04), filmTint, clamp(thickness / 1200.0, 0.0, 1.0));
  return clamp(mix(baseF0, filmF0, factor), vec3f(0.0), vec3f(1.0));
}

fn rcDistributionGGX(nDotH: f32, rough: f32) -> f32 {
  let a = rough * rough;
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(RC_PI * d * d, 1e-6);
}

fn rcGeometrySchlickGGX(nDotV: f32, rough: f32) -> f32 {
  let r = rough + 1.0;
  let k = r * r * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

fn rcGeometrySmith(nDotV: f32, nDotL: f32, rough: f32) -> f32 {
  return rcGeometrySchlickGGX(nDotV, rough) * rcGeometrySchlickGGX(nDotL, rough);
}

fn rcRotateTangentFrame(t: vec3f, b: vec3f, rotation: f32) -> mat2x3f {
  let c = cos(rotation);
  let s = sin(rotation);
  let rt = safe_normalize(t * c + b * s);
  let rb = safe_normalize(-t * s + b * c);
  return mat2x3f(rt, rb);
}

fn rcAnisotropyAxes(rough: f32, anisotropy: f32) -> vec2f {
  let a = max(0.01, rough * rough);
  let aspect = sqrt(max(0.1, 1.0 - 0.9 * clamp(anisotropy, 0.0, 1.0)));
  return vec2f(max(0.001, a / aspect), max(0.001, a * aspect));
}

fn rcDistributionGGXAnisotropic(n: vec3f, t: vec3f, b: vec3f, h: vec3f, ax: f32, ay: f32) -> f32 {
  let nDotH = max(0.0, dot(n, h));
  if (nDotH <= 1e-6) { return 0.0; }
  let tx = dot(t, h) / ax;
  let by = dot(b, h) / ay;
  let d = tx * tx + by * by + nDotH * nDotH;
  return 1.0 / max(RC_PI * ax * ay * d * d, 1e-6);
}

fn rcSmithG1Anisotropic(n: vec3f, t: vec3f, b: vec3f, v: vec3f, ax: f32, ay: f32) -> f32 {
  let nDotV = max(0.0, dot(n, v));
  if (nDotV <= 1e-6) { return 0.0; }
  let tx = dot(t, v) * ax;
  let by = dot(b, v) * ay;
  let lambda = (-1.0 + sqrt(1.0 + (tx * tx + by * by) / max(nDotV * nDotV, 1e-6))) * 0.5;
  return 1.0 / (1.0 + lambda);
}

fn rcEvalClearcoatLobe(clearcoat: vec2f, clearcoatNormal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let cc = clamp(clearcoat.x, 0.0, 1.0);
  if (cc <= 1e-4) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let nDotL = max(0.0, dot(clearcoatNormal, wi));
  let nDotV = max(1e-4, dot(clearcoatNormal, wo));
  let nDotH = max(0.0, dot(clearcoatNormal, h));
  let vDotH = max(0.0, dot(wo, h));
  if (nDotL <= 1e-6 || nDotV <= 1e-6) { return vec3f(0.0); }
  let rough = clamp(clearcoat.y, 0.01, 1.0);
  let D = rcDistributionGGX(nDotH, rough);
  let G = rcGeometrySmith(nDotV, nDotL, rough);
  let F = rcFresnelSchlick(vDotH, vec3f(0.04));
  return cc * (D * G * F) / max(4.0 * nDotV * nDotL, 1e-6) * nDotL;
}

fn rcCharlieD(nDotH: f32, alpha: f32) -> f32 {
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * RC_PI);
}

fn rcSheenVisibility(nDotL: f32, nDotV: f32) -> f32 {
  return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

fn rcEvalSheenLobe(sheen: vec4f, sheenRoughness: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let sh = clamp(sheen.a, 0.0, 1.0);
  if (sh <= 1e-4) { return vec3f(0.0); }
  let nDotL = max(0.0, dot(n, wi));
  let nDotV = max(0.0, dot(n, wo));
  if (nDotL <= 1e-6 || nDotV <= 1e-6) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let nDotH = max(0.0, dot(n, h));
  let alpha = max(clamp(sheenRoughness, 0.0, 1.0) * clamp(sheenRoughness, 0.0, 1.0), 1e-3);
  return sh * clamp(sheen.rgb, vec3f(0.0), vec3f(1.0)) * rcCharlieD(nDotH, alpha) * rcSheenVisibility(nDotL, nDotV) * nDotL;
}

fn rcEvaluateProbeDirectResponse(mat: RCProbeHitMaterial, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = max(0.0, dot(n, wi));
  if (nDotL <= 1e-6) {
    return vec3f(0.0);
  }
  let v = safe_normalize(wo);
  let l = safe_normalize(wi);
  let h = safe_normalize(v + l);
  let nDotV = max(0.0, dot(n, v));
  let nDotH = max(0.0, dot(n, h));
  let vDotH = max(0.0, dot(v, h));
  let rough = clamp(mat.roughness, 0.04, 1.0);
  var D: f32;
  var G: f32;
  let aniso = clamp(mat.anisotropy.x, 0.0, 1.0);
  if (aniso > 1e-4) {
    let frame = rcRotateTangentFrame(mat.anisotropyTangent, mat.anisotropyBitangent, mat.anisotropy.y);
    let axes = rcAnisotropyAxes(rough, aniso);
    D = rcDistributionGGXAnisotropic(n, frame[0], frame[1], h, axes.x, axes.y);
    G = rcSmithG1Anisotropic(n, frame[0], frame[1], v, axes.x, axes.y) *
        rcSmithG1Anisotropic(n, frame[0], frame[1], l, axes.x, axes.y);
  } else {
    D = rcDistributionGGX(nDotH, rough);
    G = rcGeometrySmith(nDotV, nDotL, rough);
  }
  let F0 = rcIridescenceModifiedF0(rcBaseMaterialF0(mat), mat.iridescence, vDotH);
  let F = rcFresnelSchlick(vDotH, F0);
  let spec = (D * G) * F / max(4.0 * max(nDotV, 1e-6) * nDotL, 1e-6);
  let diffuse = mat.albedo * (1.0 - clamp(mat.metalness, 0.0, 1.0)) * RC_INV_PI;
  return (diffuse + spec) * nDotL
       + rcEvalClearcoatLobe(mat.clearcoat, mat.clearcoatNormal, v, l)
       + rcEvalSheenLobe(mat.sheen, mat.sheenRoughness, n, v, l);
}`;
