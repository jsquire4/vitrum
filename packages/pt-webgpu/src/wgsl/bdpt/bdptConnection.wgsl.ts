/**
 * BDPT eye↔light connection — full Veach §10.3 multi-strategy MIS (WebGPU).
 *
 * Computes the power-heuristic (β=2) MIS weight for ONE explicit connection
 * (current eye-subpath bounce `E_e` → one stored light-subpath vertex `L_c`) by
 * enumerating ALL Veach §10.3 strategy path-pdfs over the merged path
 *
 *   v[0]=L_0(emitter) … v[c]=L_c | v[c+1]=E_e … v[c+1+e]=E_0 | v[n-1]=camera
 *
 * with `n = c+e+3` and `selectedS = c+1`. This is the canonical PBRT-v4
 * `MISWeight` recurrence: a pure ratio of AREA-measure forward/reverse densities
 * walked over the actual vertices, with the per-vertex solid-angle pdf converted
 * to area on the fly via `convertDensitySAtoArea` (PBRT `Vertex::ConvertDensity`,
 * a DESTINATION-cosine-only "half-G"). The four pdfs straddling the connection
 * edge are recomputed here from the connection geometry (PBRT's pt/ptMinus/qs/
 * qsMinus pdfRev overrides); the eye-side overrides use `brdfDirectionalPdf` with
 * wo/wi as required (D1 — PBRT-correct non-symmetric reverse density). The eye
 * prefix (E_0…E_e construction-time SA pdfs + pos/normal/specular) is read from a
 * per-pixel GPU scratch buffer threaded through the eye loop. The light chain is
 * Lambertian (cosθ/π) throughout, matching the light-subpath kernel.
 *
 * This is a 1:1 port of the CPU reference `bdpt/bdptConnectionMisFull.ts`, which
 * is pinned to `@vitrum/shared-samplers`'s `bdptConnectionMIS_full` /
 * `buildBDPTStrategyPDFs_full` oracle to ~1e-12. The whole module compiles only
 * into the full-tier shader, and the kernel calls `evaluateBdptConnection` only
 * under `params.bdptEnabled != 0u`, so the BDPT-off path is bit-identical.
 *
 * References:
 *   Veach 1997 §10.3 (BDPT MIS weights), §9.2 (power heuristic, β=2),
 *     §10.3.5 (specular zero-weight rule), §8.3.2 (geometry term).
 *   Pharr et al. 2023, PBR 4e §16.3.5 Eq. 16.16; integrators.cpp MISWeight.
 *   @vitrum/shared-samplers: bdptConnectionMIS_full, buildBDPTStrategyPDFs_full.
 */
export const PT_WEBGPU_BDPT_CONNECTION_WGSL = /* wgsl */ `
const BDPT_KIND_INVALID: f32 = 3.0;
// Firefly guard for rare near-singular connection paths. This is not a
// transport term; keep it high enough that ordinary BDPT energy is unaffected.
const BDPT_CONTRIBUTION_CLAMP: f32 = 100.0;
const BDPT_MAX_MERGED: u32 = 19u; // c(<=2) + e(<=8) + 3, with headroom

// ── Eye-subpath scratch stack (D2) ──────────────────────────────────────────
// Per-pixel × params.bdptMaxEyeDepth, 2× vec4 per eye vertex:
//   slot0 = (pos.xyz, pdfFwd)   — merged forward (swapped-BSDF reverse density)
//   slot1 = (nrm.xyz, pdfRev)   — merged reverse (scatter pdf that produced E_d)
// Specular is packed as a NEGATIVE pdfFwd sentinel (pdfs are >= 0), so the load
// recovers abs(pdfFwd) and the spec flag from its sign — keeping 2 vec4/vertex.
struct BdptEyeVtx {
  pos: vec3f,
  nrm: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  spec: bool,
}

// Current pixel's linear index, set once at the top of the @compute main entry
// (bdptSetCurrentPixel) so the deeply-nested stack helpers need no threading.
var<private> bdptCurrentPixel: u32;

fn bdptSetCurrentPixel(pixelIndex: u32) {
  bdptCurrentPixel = pixelIndex;
}

fn bdptEyeStackBase(d: u32) -> u32 {
  return (bdptCurrentPixel * params.bdptMaxEyeDepth + d) * 2u;
}

fn bdptEyeStackStore(d: u32, pos: vec3f, nrm: vec3f, pdfFwd: f32, pdfRev: f32, spec: bool) {
  if (d >= params.bdptMaxEyeDepth) { return; }
  let base = bdptEyeStackBase(d);
  let signedFwd = select(max(pdfFwd, 0.0), -max(pdfFwd, 0.0) - 1e-30, spec);
  bdptEyeStack[base] = vec4f(pos, signedFwd);
  bdptEyeStack[base + 1u] = vec4f(nrm, pdfRev);
}

// Patch only the pdfFwd of an already-stored slot, preserving its position and
// its specular sign convention (specular slots keep their negative sentinel).
fn bdptEyeStackSetFwd(d: u32, pdfFwd: f32) {
  if (d >= params.bdptMaxEyeDepth) { return; }
  let base = bdptEyeStackBase(d);
  let s0 = bdptEyeStack[base];
  let wasSpec = s0.w < 0.0;
  let signedFwd = select(max(pdfFwd, 0.0), -max(pdfFwd, 0.0) - 1e-30, wasSpec);
  bdptEyeStack[base] = vec4f(s0.xyz, signedFwd);
}

fn bdptEyeStackLoad(d: u32) -> BdptEyeVtx {
  var v: BdptEyeVtx;
  let base = bdptEyeStackBase(d);
  let s0 = bdptEyeStack[base];
  let s1 = bdptEyeStack[base + 1u];
  v.pos = s0.xyz;
  v.nrm = s1.xyz;
  v.spec = s0.w < 0.0;
  v.pdfFwd = abs(s0.w);
  v.pdfRev = s1.w;
  return v;
}

fn bdptGeometricTerm(posX: vec3f, nX: vec3f, posY: vec3f, nY: vec3f) -> f32 {
  let d = posY - posX;
  let dist2 = dot(d, d);
  if (dist2 <= 1e-12) {
    return 0.0;
  }
  let w = d * inverseSqrt(dist2);
  let cosX = abs(dot(nX, w));
  let cosY = abs(dot(nY, -w));
  return (cosX * cosY) / dist2;
}

// PBRT Vertex::ConvertDensity — SA pdf → area pdf, destination-cosine only.
fn bdptConvertDensitySAtoArea(pdfSA: f32, fromPos: vec3f, destPos: vec3f, destNorm: vec3f) -> f32 {
  let d = destPos - fromPos;
  let dist2 = dot(d, d);
  if (dist2 <= 0.0) {
    return pdfSA; // coincident → unit Jacobian (endpoint guard)
  }
  let invDist = inverseSqrt(dist2);
  let cosDest = abs(dot(destNorm, d * invDist));
  return (pdfSA * cosDest) / dist2;
}

// Lambertian outgoing SA density at a light-subpath vertex along dir: |cosθ|/π.
fn bdptLambertDirPdf(n: vec3f, dir: vec3f) -> f32 {
  return abs(dot(n, normalize(dir))) * INV_PI;
}

// Merged-path vertex assembled on the fly from the light chain (texture) + the
// eye stack (scratch buffer) + the connection-induced straddle overrides.
struct BdptMergedVtx {
  pos: vec3f,
  nrm: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  spec: bool,
}

// Address one merged vertex by index i in [0, n-1]. The connection-induced
// straddle overrides are applied here so the sweep reads a coherent vertex set.
//   light side  : i <= c
//   eye side     : i in [c+1, c+1+e]  → eye depth = e - (i - (c+1))
//   camera       : i == n-1
fn bdptMergedVertex(
  i: u32,
  c: u32,        // light connection vertex index (lightChain length - 1)
  e: u32,        // eye connection vertex depth (eyeChain length - 1)
  n: u32,        // merged length
  // connection geometry / overrides
  fwdEe: f32,        // merged pdfFwd(E_e)        (light Lambert toward E_e)
  fwdEeMinus: f32,   // merged pdfFwd(E_{e-1})    (eye BSDF, wo=connDir)
  revLc: f32,        // merged pdfRev(L_c)        (eye BSDF, wi=connDir)
  revLcMinus: f32,   // merged pdfRev(L_{c-1})    (light Lambert toward L_{c-1})
  camPos: vec3f,
  camNrm: vec3f,
) -> BdptMergedVtx {
  var v: BdptMergedVtx;
  if (i <= c) {
    // Light side: read scratch-buffer column i, rows 0/1/2.
    let l0 = bdptLightPath[bdptLightPathIndex(i32(i), 0u)];
    let l1 = bdptLightPath[bdptLightPathIndex(i32(i), 1u)];
    let l2 = bdptLightPath[bdptLightPathIndex(i32(i), 2u)];
    v.pos = l0.xyz;
    v.nrm = l1.xyz;
    v.pdfFwd = l1.w;          // stored SA pdfFwd (NO baked-in G; emitter = area endpoint)
    v.pdfRev = l2.w;          // stored SA pdfRev (Lambertian construction)
    v.spec = false;           // light subpath terminates specular vertices (D4)
    if (i == c) { v.pdfRev = revLc; }
    else if (c >= 1u && i == c - 1u) { v.pdfRev = revLcMinus; }
    return v;
  }
  if (i == n - 1u) {
    v.pos = camPos;
    v.nrm = camNrm;
    v.pdfFwd = 1.0;
    v.pdfRev = 1.0;
    v.spec = false;
    return v;
  }
  // Eye side: merged i in [c+1, c+1+e] → eye scratch index d = e - (i - (c+1)).
  let off = i - (c + 1u);
  let d = e - off;
  let es = bdptEyeStackLoad(d);
  v.pos = es.pos;
  v.nrm = es.nrm;
  v.pdfFwd = es.pdfFwd;       // merged forward (swapped-BSDF reverse density)
  v.pdfRev = es.pdfRev;       // merged reverse (scatter pdf that produced E_d)
  v.spec = es.spec;
  if (off == 0u) { v.pdfFwd = fwdEe; }            // E_e   override
  else if (off == 1u) { v.pdfFwd = fwdEeMinus; }  // E_{e-1} override
  return v;
}

// Area-measure forward density of merged vertex i (edge v_{i-1}→v_i).
fn bdptFwdArea(i: u32, c: u32, e: u32, n: u32,
  fwdEe: f32, fwdEeMinus: f32, revLc: f32, revLcMinus: f32, camPos: vec3f, camNrm: vec3f) -> f32 {
  let v = bdptMergedVertex(i, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  if (i == 0u) { return v.pdfFwd; }
  let prev = bdptMergedVertex(i - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  return bdptConvertDensitySAtoArea(v.pdfFwd, prev.pos, v.pos, v.nrm);
}

// Area-measure reverse density of merged vertex i (edge v_{i+1}→v_i).
fn bdptRevArea(i: u32, c: u32, e: u32, n: u32,
  fwdEe: f32, fwdEeMinus: f32, revLc: f32, revLcMinus: f32, camPos: vec3f, camNrm: vec3f) -> f32 {
  let v = bdptMergedVertex(i, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  if (i == n - 1u) { return v.pdfRev; }
  let next = bdptMergedVertex(i + 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  return bdptConvertDensitySAtoArea(v.pdfRev, next.pos, v.pos, v.nrm);
}

// Full Veach §10.3 power-heuristic MIS weight for the selected strategy
// selectedS = c+1. Mirrors buildBDPTStrategyPDFs_full + bdptConnectionMIS_full.
fn bdptMISWeightFull(
  c: u32, e: u32, n: u32, selectedS: u32, pRef: f32,
  fwdEe: f32, fwdEeMinus: f32, revLc: f32, revLcMinus: f32, camPos: vec3f, camNrm: vec3f,
) -> f32 {
  if (pRef <= 0.0 || n == 0u || selectedS >= n) { return 0.0; }
  var pdfs: array<f32, BDPT_MAX_MERGED>;
  for (var k = 0u; k < n; k = k + 1u) { pdfs[k] = 0.0; }
  pdfs[selectedS] = pRef;

  // Left sweep (decrement s): flip v[s-1]; p_{s-1} = p_s · pRev(s-1)/pFwd(s-1).
  {
    var p = pRef;
    var s = selectedS;
    loop {
      if (s == 0u) { break; }
      let flip = bdptMergedVertex(s - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      var neighborSpec = false;
      if (s >= 2u) {
        let nb = bdptMergedVertex(s - 2u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
        neighborSpec = nb.spec;
      }
      if (flip.spec || neighborSpec) { break; }
      let pFwd = bdptFwdArea(s - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      let pRev = bdptRevArea(s - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      if (pFwd <= 0.0 || pRev <= 0.0) { break; }
      p = p * (pRev / pFwd);
      pdfs[s - 1u] = p;
      s = s - 1u;
    }
  }
  // Right sweep (increment s): flip v[s]; p_{s+1} = p_s · pFwd(s)/pRev(s).
  {
    var p = pRef;
    var s = selectedS;
    loop {
      if (s >= n - 1u) { break; }
      let flip = bdptMergedVertex(s, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      let nb = bdptMergedVertex(s + 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      if (flip.spec || nb.spec) { break; }
      let pFwd = bdptFwdArea(s, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      let pRev = bdptRevArea(s, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      if (pFwd <= 0.0 || pRev <= 0.0) { break; }
      p = p * (pFwd / pRev);
      pdfs[s + 1u] = p;
      s = s + 1u;
    }
  }

  // Power heuristic (β=2).
  var denom = 0.0;
  for (var k = 0u; k < n; k = k + 1u) {
    let pk = pdfs[k];
    if (pk > 0.0) { denom = denom + pk * pk; }
  }
  if (denom <= 0.0) { return 0.0; }
  let ps = pdfs[selectedS];
  if (ps <= 0.0) { return 0.0; }
  return (ps * ps) / denom;
}

// Forward scatter pdf at the eye vertex (the old hardcoded eyePdfFwd=1.0 is
// gone): the real per-vertex BSDF scatter densities now flow through the eye
// scratch stack (pdfRev) and the connection-induced overrides (revLc / fwdEeMinus
// via brdfDirectionalPdf), so no scalar eyePdfFwd argument is needed here.
fn evaluateBdptConnection(
  eyePos: vec3f,
  eyeNormal: vec3f,
  eyeWo: vec3f,
  eyeThroughput: vec3f,
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
  anisotropy: f32,
  anisotropyRotation: f32,
  eyeDepth: u32,
  lightVtxIdx: i32,
) -> vec3f {
  let lv0 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 0u)];
  let lv1 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 1u)];
  let lv2 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 2u)];
  if (lv0.w == BDPT_KIND_INVALID) {
    return vec3f(0.0);
  }
  let eyeIsSpecular = transmission > 0.5 && roughness < 0.05;
  if (eyeIsSpecular) {
    return vec3f(0.0);
  }
  let lightPos = lv0.xyz;
  let lightNormal = lv1.xyz;
  let lightPdfFwd = lv1.w;
  let lightThroughput = lv2.xyz;
  let toLight = lightPos - eyePos;
  let dist = length(toLight);
  if (dist < 1e-4) {
    return vec3f(0.0);
  }
  let connDir = toLight / dist;                 // E_e → L_c
  let gTerm = bdptGeometricTerm(eyePos, eyeNormal, lightPos, lightNormal);
  if (gTerm <= 0.0) {
    return vec3f(0.0);
  }
  let shadowRay = Ray(eyePos + eyeNormal * 1e-3, connDir);
  if (traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
    return vec3f(0.0);
  }
  let eyeBrdf = evaluateBrdfFull(
    baseColor, roughness, metallic, eyeNormal, eyeWo, connDir,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation,
  );
  let cosEye = max(dot(eyeNormal, connDir), 0.0);
  if (cosEye <= 0.0) {
    return vec3f(0.0);
  }
  // bdptGeometricTerm already contributes the receiver cosine for this edge.
  // Keep the BSDF value itself here so the connection does not double-count
  // cos(theta) at the eye endpoint.
  let eyeBsdfCosTheta = eyeBrdf;
  let cosLight = max(dot(lightNormal, -connDir), 0.0);
  if (cosLight <= 0.0) {
    return vec3f(0.0);
  }
  // A9 — REAL light-vertex BSDF at L_c (row 3: matId + wo-toward-prev). For a
  // surface vertex (matId >= 0) evaluate the actual BSDF scattering the incoming
  // light direction (toward L_{c-1}, = lvWoPrev) to the connection direction (L_c →
  // E_e, = -connDir); for the legacy pseudo-emitter vertex (matId == -1) keep
  // the diffuse emission/Lambertian profile cosθ/π. Finite area emitters
  // (matId == -2) already carry Le/(pdfPick·pdfArea), so their endpoint factor
  // is 1 and the geometry term owns the emitter cosine. This makes a glossy/metallic light-path
  // vertex's connection consistent with the glossy light-subpath BUILD (else the
  // BSDF mismatch between build and connect biases the estimate).
  let lv3 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 3u)];
  let lvMatId = lv3.w;
  let lvWoPrev = lv3.xyz;
  var lightBsdfCosTheta = vec3f(1.0);
  if (lvMatId == -1.0) {
    lightBsdfCosTheta = vec3f(cosLight / PI);
  }
  if (lvMatId >= 0.0) {
    let lvMat = decodeMaterial(u32(lvMatId));
    let lvBrdf = evaluateBrdfFull(
      lvMat.baseColor, max(lvMat.roughness, 0.02), lvMat.metallic,
      lightNormal, -connDir, lvWoPrev,
      lvMat.clearcoat, lvMat.clearcoatRoughness, lvMat.sheen, lvMat.sheenRoughness, lvMat.sheenColor,
      lvMat.iridescence, lvMat.iridescenceIor, lvMat.iridescenceThicknessMin, lvMat.iridescenceThicknessMax,
      0.0, 0.0,
    );
    // bdptGeometricTerm already contributes the light-vertex cosine.
    lightBsdfCosTheta = lvBrdf;
  }

  // ── Full §10.3 MIS weight ──────────────────────────────────────────────────
  let c = u32(lightVtxIdx);
  let e = eyeDepth;
  let n = c + e + 3u;
  let selectedS = c + 1u;
  if (n > BDPT_MAX_MERGED) {
    return vec3f(0.0); // depth out of scratch range (should not happen)
  }
  let camPos = params.cameraPos.xyz;
  let camNrm = normalize(camPos - eyePos);

  // Connection-induced straddle overrides (PBRT MISWeight remapping).
  let lcToE = -connDir;                          // L_c → E_e
  // A9 — forward arrival density at E_e from L_c: the REAL light-vertex BSDF pdf
  // (incoming = lvWoPrev, outgoing = lcToE) for a surface vertex; Lambertian for the
  // emitter (matId < 0). Keeps the MIS pdf bookkeeping consistent with the glossy
  // light-vertex BSDF used in lightBsdfCosTheta.
  var fwdEe = bdptLambertDirPdf(lightNormal, lcToE);
  if (lvMatId >= 0.0) {
    let lvMatF = decodeMaterial(u32(lvMatId));
    fwdEe = brdfDirectionalPdfFull(
      lvMatF.baseColor, max(lvMatF.roughness, 0.02), lvMatF.metallic,
      0.0, lvMatF.ior, lightNormal, lvWoPrev, lcToE,
      lvMatF.clearcoat, lvMatF.clearcoatRoughness, lvMatF.sheen, lvMatF.sheenRoughness,
      lvMatF.iridescence, lvMatF.iridescenceIor, lvMatF.iridescenceThicknessMin, lvMatF.iridescenceThicknessMax,
      0.0, 0.0,
    );
  }
  // E_{e-1} position from scratch (if e>=1); else camera endpoint.
  var eeMinusPos = camPos;
  if (e >= 1u) {
    let prevEye = bdptEyeStackLoad(e - 1u);
    eeMinusPos = prevEye.pos;
  }
  let eeToPrev = normalize(eeMinusPos - eyePos);  // E_e → E_{e-1} (or → camera at e=0)
  let revLc = brdfDirectionalPdfFull(
    baseColor, roughness, metallic, transmission, ior, eyeNormal, eeToPrev, connDir,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation,
  );
  var fwdEeMinus = 0.0;
  if (e >= 1u) {
    fwdEeMinus = brdfDirectionalPdfFull(
      baseColor, roughness, metallic, transmission, ior, eyeNormal, connDir, eeToPrev,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness,
      iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
      anisotropy, anisotropyRotation,
    );
  }
  // L_{c-1} override: reverse density at L_c toward L_{c-1}. REAL BSDF pdf (outgoing
  // = lcToE toward E_e, incoming = direction to L_{c-1}) for a surface vertex;
  // Lambertian for the emitter.
  var revLcMinus = 0.0;
  if (c >= 1u) {
    let lcm0 = bdptLightPath[bdptLightPathIndex(i32(c - 1u), 0u)];
    let lcToLcMinus = normalize(lcm0.xyz - lightPos);
    if (lvMatId >= 0.0) {
      let lvMatR = decodeMaterial(u32(lvMatId));
      revLcMinus = brdfDirectionalPdfFull(
        lvMatR.baseColor, max(lvMatR.roughness, 0.02), lvMatR.metallic,
        0.0, lvMatR.ior, lightNormal, lcToE, lcToLcMinus,
        lvMatR.clearcoat, lvMatR.clearcoatRoughness, lvMatR.sheen, lvMatR.sheenRoughness,
        lvMatR.iridescence, lvMatR.iridescenceIor, lvMatR.iridescenceThicknessMin, lvMatR.iridescenceThicknessMax,
        0.0, 0.0,
      );
    } else {
      revLcMinus = bdptLambertDirPdf(lightNormal, lcToLcMinus);
    }
  }

  // pRef = joint forward density of the chosen strategy = light forward
  // arrival at L_c (area) × forward arrival at E_e from L_c (area, = fwdEe·G's
  // dest cosine handled by ConvertDensity). The MIS weight is scale-invariant in
  // pRef (it cancels in the power-heuristic ratio numerator/denominator), so any
  // positive consistent pRef yields the same weight; use the area-forward of the
  // selected strategy's connection edge for numerical conditioning.
  let pRef = max(lightPdfFwd, 1e-12) * max(fwdEe, 1e-12) + 1e-30;
  let misW = bdptMISWeightFull(c, e, n, selectedS, pRef, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  if (misW <= 0.0) {
    return vec3f(0.0);
  }

  var contribution = lightThroughput * lightBsdfCosTheta * gTerm * eyeBsdfCosTheta * misW;
  contribution = contribution * eyeThroughput;
  // WGSL has no isNan/isInf builtins (those calls fail to resolve on Dawn).
  // NaN is detected by the self-inequality x != x; non-finite magnitudes are
  // caught by comparing against the largest finite f32 (≈3.4e38).
  let isNonFinite =
    any(contribution != contribution) ||
    any(abs(contribution) > vec3f(3.4e38));
  if (isNonFinite) {
    return vec3f(0.0);
  }
  return clamp(contribution, vec3f(0.0), vec3f(BDPT_CONTRIBUTION_CLAMP));
}
`;
