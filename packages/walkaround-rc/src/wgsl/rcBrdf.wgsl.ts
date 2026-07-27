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
  let dielectricF0 = select(
    vec3f(0.04) * clamp(mat.specular.rgb, vec3f(0.0), vec3f(1.0)) * clamp(mat.specular.a, 0.0, 1.0),
    clamp(mat.specular.rgb - vec3f(1.0), vec3f(0.0), vec3f(1.0)),
    all(mat.specular.rgb >= vec3f(1.0)),
  );
  return mix(dielectricF0, mat.albedo, clamp(mat.metalness, 0.0, 1.0));
}

fn rcFresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * f;
}

fn rcIridescenceModifiedF0(baseF0: vec3f, iridescence: vec4f, vDotH: f32) -> vec3f {
  let factor = clamp(iridescence.x, 0.0, 1.0);
  if (factor <= 0.0) {
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
  if (nDotH <= 0.0 || rough <= 0.0) { return 0.0; }
  let alpha = rough * rough;
  let alpha2 = alpha * alpha;
  let d = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
  return alpha2 / (RC_PI * d * d);
}

fn rcSmithG1GGX(nDotV: f32, alpha2: f32) -> f32 {
  if (nDotV <= 0.0) { return 0.0; }
  return (2.0 * nDotV) /
    (nDotV + sqrt(alpha2 + (1.0 - alpha2) * nDotV * nDotV));
}

fn rcGeometrySmith(nDotV: f32, nDotL: f32, rough: f32) -> f32 {
  if (rough <= 0.0) { return 0.0; }
  let alpha = rough * rough;
  let alpha2 = alpha * alpha;
  return rcSmithG1GGX(nDotV, alpha2) * rcSmithG1GGX(nDotL, alpha2);
}

fn rcRotateTangentFrame(t: vec3f, b: vec3f, rotation: f32) -> mat2x3f {
  let c = cos(rotation);
  let s = sin(rotation);
  let rt = safe_normalize(t * c + b * s);
  let rb = safe_normalize(-t * s + b * c);
  return mat2x3f(rt, rb);
}

fn rcAnisotropyAxes(rough: f32, anisotropy: f32) -> vec2f {
  let alpha = rough * rough;
  let aspect = sqrt(1.0 - 0.9 * clamp(anisotropy, 0.0, 1.0));
  return vec2f(alpha / aspect, alpha * aspect);
}

fn rcDistributionGGXAnisotropic(n: vec3f, t: vec3f, b: vec3f, h: vec3f, ax: f32, ay: f32) -> f32 {
  let nDotH = dot(n, h);
  if (nDotH <= 0.0 || ax <= 0.0 || ay <= 0.0) { return 0.0; }
  let tx = dot(t, h) / ax;
  let by = dot(b, h) / ay;
  let d = tx * tx + by * by + nDotH * nDotH;
  return 1.0 / (RC_PI * ax * ay * d * d);
}

fn rcSmithG1Anisotropic(n: vec3f, t: vec3f, b: vec3f, v: vec3f, ax: f32, ay: f32) -> f32 {
  let nDotV = dot(n, v);
  if (nDotV <= 0.0 || ax <= 0.0 || ay <= 0.0) { return 0.0; }
  let tx = dot(t, v) * ax;
  let by = dot(b, v) * ay;
  let lambda = (-1.0 + sqrt(1.0 + (tx * tx + by * by) / (nDotV * nDotV))) * 0.5;
  return 1.0 / (1.0 + lambda);
}

fn rcDielectricFresnelExact(cosThetaI: f32, etaIncident: f32, etaTarget: f32) -> f32 {
  if (etaIncident == etaTarget) { return 0.0; }
  let ci = clamp(abs(cosThetaI), 0.0, 1.0);
  let eta = etaIncident / etaTarget;
  let sin2ThetaT = eta * eta * (1.0 - ci * ci);
  if (sin2ThetaT >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - sin2ThetaT);
  let rs = (etaIncident * ci - etaTarget * ct) /
    (etaIncident * ci + etaTarget * ct);
  let rp = (etaTarget * ci - etaIncident * ct) /
    (etaTarget * ci + etaIncident * ct);
  return 0.5 * (rs * rs + rp * rp);
}

fn rcSampleVisibleGgxNormal(n: vec3f, wo: vec3f, rough: f32, xi: vec2f) -> vec3f {
  if (rough <= 0.0) { return n; }
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  let woT = vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, n));
  let alpha = rough * rough;
  let vh = safe_normalize(vec3f(alpha * woT.x, alpha * woT.y, woT.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3f(1.0, 0.0, 0.0);
  if (lensq > 0.0) {
    t1 = vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(lensq);
  }
  let t2 = cross(vh, t1);
  let radius = sqrt(clamp(xi.x, 0.0, 1.0));
  let phi = 2.0 * RC_PI * clamp(xi.y, 0.0, 1.0);
  let diskX = radius * cos(phi);
  var diskY = radius * sin(phi);
  let blend = 0.5 * (1.0 + vh.z);
  diskY = (1.0 - blend) * sqrt(max(0.0, 1.0 - diskX * diskX)) +
    blend * diskY;
  let nh = diskX * t1 + diskY * t2 +
    sqrt(max(0.0, 1.0 - diskX * diskX - diskY * diskY)) * vh;
  let wmT = safe_normalize(vec3f(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
  return safe_normalize(wmT.x * tangent + wmT.y * bitangent + wmT.z * n);
}

struct RcGgxDielectricTransmissionSample {
  direction: vec3f,
  weight: f32,
  transmission: f32,
  microfacetCos: f32,
  valid: u32,
};

fn rcSampleGgxDielectricTransmission(
  n: vec3f,
  wo: vec3f,
  rough: f32,
  etaIncident: f32,
  etaTarget: f32,
  xi: vec2f,
) -> RcGgxDielectricTransmissionSample {
  var out: RcGgxDielectricTransmissionSample;
  out.direction = vec3f(0.0);
  out.weight = 0.0;
  out.transmission = 0.0;
  out.microfacetCos = 0.0;
  out.valid = 0u;
  let nDotWo = dot(n, wo);
  if (nDotWo <= 0.0 || etaIncident <= 0.0 || etaTarget <= 0.0) { return out; }
  let eta = etaIncident / etaTarget;
  let etap = etaTarget / etaIncident;
  if (rough <= 0.0) {
    let wi = refract(-wo, n, eta);
    if (dot(wi, wi) <= 0.0) { return out; }
    let interfaceT = 1.0 - rcDielectricFresnelExact(nDotWo, etaIncident, etaTarget);
    if (interfaceT <= 0.0) { return out; }
    out.direction = safe_normalize(wi);
    out.weight = interfaceT / (etap * etap);
    out.transmission = interfaceT;
    out.microfacetCos = nDotWo;
    out.valid = 1u;
    return out;
  }

  let authoredRoughness = clamp(rough, 0.0, 1.0);
  var wm = rcSampleVisibleGgxNormal(n, wo, authoredRoughness, xi);
  if (dot(wm, n) < 0.0) { wm = -wm; }
  let woDotM = dot(wo, wm);
  if (woDotM <= 0.0) { return out; }
  let wiRaw = refract(-wo, wm, eta);
  if (dot(wiRaw, wiRaw) <= 0.0) { return out; }
  let wi = safe_normalize(wiRaw);
  let nDotWiAbs = abs(dot(n, wi));
  let wiDotM = dot(wi, wm);
  let denom = wiDotM + woDotM / etap;
  if (dot(n, wi) >= 0.0 || nDotWiAbs <= 0.0 || wiDotM >= 0.0 || denom == 0.0) {
    return out;
  }
  let alpha = authoredRoughness * authoredRoughness;
  let alpha2 = alpha * alpha;
  let D = rcDistributionGGX(dot(n, wm), authoredRoughness);
  let G1o = rcSmithG1GGX(nDotWo, alpha2);
  let G = G1o * rcSmithG1GGX(nDotWiAbs, alpha2);
  let interfaceT = 1.0 - rcDielectricFresnelExact(woDotM, etaIncident, etaTarget);
  let denom2 = denom * denom;
  let pdf = (D * G1o * abs(woDotM) / nDotWo) * abs(wiDotM) / denom2;
  if (D <= 0.0 || G <= 0.0 || interfaceT <= 0.0 || pdf <= 0.0) { return out; }
  let ft = interfaceT * D * G *
    abs(wiDotM * woDotM / (nDotWiAbs * nDotWo * denom2)) /
    (etap * etap);
  out.direction = wi;
  out.weight = ft * nDotWiAbs / pdf;
  out.transmission = interfaceT;
  out.microfacetCos = woDotM;
  out.valid = 1u;
  return out;
}

fn rcEvalClearcoatLobe(clearcoat: vec2f, clearcoatNormal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let cc = clamp(clearcoat.x, 0.0, 1.0);
  let rough = clamp(clearcoat.y, 0.0, 1.0);
  if (cc <= 0.0 || rough <= 0.0) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let nDotL = dot(clearcoatNormal, wi);
  let nDotV = dot(clearcoatNormal, wo);
  if (nDotL <= 0.0 || nDotV <= 0.0) { return vec3f(0.0); }
  let nDotH = max(0.0, dot(clearcoatNormal, h));
  let vDotH = max(0.0, dot(wo, h));
  let D = rcDistributionGGX(nDotH, rough);
  let G = rcGeometrySmith(nDotV, nDotL, rough);
  let F = rcFresnelSchlick(vDotH, vec3f(0.04));
  return cc * (D * G * F) / (4.0 * nDotV * nDotL) * nDotL;
}

fn rcCharlieD(nDotH: f32, alpha: f32) -> f32 {
  if (alpha <= 0.0) { return 0.0; }
  let invAlpha = 1.0 / alpha;
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * RC_PI);
}

fn rcSheenVisibility(nDotL: f32, nDotV: f32) -> f32 {
  return 1.0 / (4.0 * (nDotL + nDotV - nDotL * nDotV));
}

fn rcEvalSheenLobe(sheen: vec4f, sheenRoughness: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let sh = clamp(sheen.a, 0.0, 1.0);
  let rough = clamp(sheenRoughness, 0.0, 1.0);
  if (sh <= 0.0 || rough <= 0.0) { return vec3f(0.0); }
  let nDotL = dot(n, wi);
  let nDotV = dot(n, wo);
  if (nDotL <= 0.0 || nDotV <= 0.0) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let nDotH = max(0.0, dot(n, h));
  let alpha = rough * rough;
  return sh * clamp(sheen.rgb, vec3f(0.0), vec3f(1.0)) * rcCharlieD(nDotH, alpha) * rcSheenVisibility(nDotL, nDotV) * nDotL;
}

fn rcEvaluateProbeDirectResponse(mat: RCProbeHitMaterial, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = dot(n, wi);
  if (nDotL <= 0.0) { return vec3f(0.0); }
  let v = safe_normalize(wo);
  let l = safe_normalize(wi);
  let h = safe_normalize(v + l);
  let nDotV = dot(n, v);
  if (nDotV <= 0.0) { return vec3f(0.0); }
  let nDotH = max(0.0, dot(n, h));
  let vDotH = max(0.0, dot(v, h));
  let rough = clamp(mat.roughness, 0.0, 1.0);
  var D: f32;
  var G: f32;
  let aniso = clamp(mat.anisotropy.x, 0.0, 1.0);
  if (aniso > 0.0) {
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
  let spec = (D * G) * F / (4.0 * nDotV * nDotL);
  let diffuse = mat.albedo * (1.0 - clamp(mat.metalness, 0.0, 1.0)) * RC_INV_PI;
  return (diffuse + spec) * nDotL
       + rcEvalClearcoatLobe(mat.clearcoat, mat.clearcoatNormal, v, l)
       + rcEvalSheenLobe(mat.sheen, mat.sheenRoughness, n, v, l);
}`;
