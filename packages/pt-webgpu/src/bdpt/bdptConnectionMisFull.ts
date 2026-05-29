/**
 * bdptConnectionMisFull.ts — full Veach §10.3 BDPT connection MIS, CPU reference.
 *
 * This module is the SINGLE SOURCE OF TRUTH for the multi-strategy BDPT
 * connection MIS weight as the GPU port computes it. It is an independent
 * reimplementation of the PBRT-v4 `MISWeight` recurrence — NOT a wrapper around
 * `@vitrum/shared-samplers`'s `bdptConnectionMIS_full`. The shared-samplers
 * oracle (`bdptMIS.ts`) is the read-only reference the GPU port must reproduce
 * to machine epsilon; this file ports the *same arithmetic* and the
 * *connection-vertex assembly* the WGSL (`bdptConnection.wgsl.ts`) and the fork
 * GLSL (`bdpt_connection.glsl.js`) shaders run, so a single test can assert the
 * three formulations agree.
 *
 * ── The merged path ──────────────────────────────────────────────────────────
 * The renderer connects ONE eye vertex (the current bounce hit `E_e`, at eye
 * depth `e`) to one stored light-subpath vertex (`L_c`, column `c`). The full
 * Veach path that gets MIS-weighted is the concatenation
 *
 *   v[0]      = L_0   (emitter endpoint)
 *   …
 *   v[c]      = L_c   (light connection vertex)            ← connection edge ─┐
 *   v[c+1]    = E_e   (eye connection vertex)              ←──────────────────┘
 *   v[c+2]    = E_{e-1}
 *   …
 *   v[c+1+e]  = E_0   (primary hit)
 *   v[c+2+e]  = camera (endpoint)
 *
 * so `n = c + e + 3` and `selectedS = c + 1` (the light-vertex count). This is
 * the index ordering `BDPTFullVertex[]` expects (v[0]=light endpoint,
 * v[n-1]=camera endpoint).
 *
 * ── Per-vertex pdfs (solid-angle measure) ───────────────────────────────────
 * Every vertex carries a forward (light→camera) and reverse (camera→light)
 * solid-angle pdf. The oracle converts each to area measure on the fly via
 * `convertDensitySAtoArea` (PBRT `Vertex::ConvertDensity`, destination-cosine
 * only). Most pdfs are the construction-time subpath densities; the FOUR pdfs
 * straddling the connection edge are CONNECTION-INDUCED and recomputed here from
 * the connection geometry, exactly as PBRT's `MISWeight` overrides
 * `pt->pdfRev`, `ptMinus->pdfRev`, `qs->pdfRev`, `qsMinus->pdfRev`:
 *
 *   merged pdfFwd(E_e)     = light-vertex Lambertian density at L_c toward E_e
 *   merged pdfFwd(E_{e-1}) = eye-material density at E_e (wo=connDir, wi→E_{e-1})
 *   merged pdfRev(L_c)     = eye-material density at E_e (wo→E_{e-1},  wi=connDir)
 *   merged pdfRev(L_{c-1}) = light-vertex Lambertian density at L_c toward L_{c-1}
 *
 * The light subpath is Lambertian (cosine hemisphere) throughout, so its
 * connection-induced densities are `|cosθ| / π`. The eye-side densities use the
 * real BSDF directional pdf (`brdfDirectionalPdf`) with wo/wi as noted — D1
 * (PBRT-correct, non-symmetric reverse density).
 *
 * References:
 *   - Veach 1997 §10.3 (BDPT MIS), §9.2 (power heuristic).
 *   - Pharr et al. 2023, PBR 4e §16.3.5 Eq. 16.16; `integrators.cpp` MISWeight.
 *
 * @module bdptConnectionMisFull
 */

export type Vec3 = readonly [number, number, number];

/** A merged-path vertex, mirroring `@vitrum/shared-samplers`'s `BDPTFullVertex`. */
export interface MergedVertex {
  readonly position: Vec3;
  readonly normal: Vec3;
  /** Forward (light→camera) solid-angle pdf. */
  readonly pdfFwd: number;
  /** Reverse (camera→light) solid-angle pdf. */
  readonly pdfRev: number;
  readonly isSpecular: boolean;
}

/** One construction-time eye-subpath vertex as carried in the GPU scratch stack. */
export interface EyeStackVertex {
  readonly position: Vec3;
  readonly normal: Vec3;
  /**
   * Merged forward (light→camera) solid-angle pdf — the swapped-direction BSDF
   * reverse density evaluated at the light-ward eye neighbour using the natural
   * next eye direction. Endpoint (camera-ward) vertices treat this as a unit
   * Jacobian (value ignored at v[n-1]).
   */
  readonly pdfFwd: number;
  /**
   * Merged reverse (camera→light) solid-angle pdf — the forward scatter pdf at
   * the camera-ward neighbour that produced this vertex.
   */
  readonly pdfRev: number;
  readonly isSpecular: boolean;
}

/** One stored light-subpath vertex (Lambertian; SA pdfs, NO baked-in G). */
export interface LightStackVertex {
  readonly position: Vec3;
  readonly normal: Vec3;
  /** Forward (light→camera) solid-angle pdf. At the emitter this is the joint
   *  emitter-area × directional density (treated as area-measure endpoint). */
  readonly pdfFwd: number;
  /** Reverse (camera→light) solid-angle pdf (Lambertian cosθ/π construction). */
  readonly pdfRev: number;
  readonly isSpecular: boolean;
}

const PI = Math.PI;
const INV_PI = 1 / Math.PI;

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function len(a: Vec3): number {
  return Math.sqrt(dot(a, a));
}
function normalize(a: Vec3): Vec3 {
  const l = len(a);
  if (l <= 0) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}

/**
 * PBRT `Vertex::ConvertDensity`: SA pdf → area pdf with a DESTINATION-cosine-only
 * Jacobian (`pdfSA · |cosθ_dest| / dist²`). Coincident → unit Jacobian. Mirrors
 * the private `convertDensitySAtoArea` in `@vitrum/shared-samplers/bdptMIS.ts`.
 */
export function convertDensitySAtoArea(
  pdfSA: number,
  fromPos: Vec3,
  destPos: Vec3,
  destNorm: Vec3,
): number {
  const d = sub(destPos, fromPos);
  const dist2 = dot(d, d);
  if (dist2 <= 0) return pdfSA;
  const invDist = 1 / Math.sqrt(dist2);
  const cosDest = Math.abs(dot(destNorm, [d[0] * invDist, d[1] * invDist, d[2] * invDist]));
  return (pdfSA * cosDest) / dist2;
}

/** Lambertian outgoing solid-angle density at `surf` along `dir`: |cosθ|/π. */
export function lambertianDirectionalPdf(surfNormal: Vec3, dir: Vec3): number {
  return Math.abs(dot(surfNormal, normalize(dir))) * INV_PI;
}

/**
 * Assemble the merged Veach path for an eye-vertex ↔ light-vertex connection,
 * applying the four PBRT connection-induced pdf overrides at the straddle
 * vertices. `eyeBrdfPdf` is the BSDF directional-pdf evaluator at the eye
 * connection vertex `E_e` (D1: pass wo/wi exactly; do NOT symmetrise).
 *
 * - `lightChain[0..c]` = L_0 (emitter) … L_c (light connection vertex), forward
 *   ordering (emitter first). Lambertian construction-time SA pdfs, no G.
 * - `eyeChain[0..e]` = E_0 (primary hit) … E_e (eye connection vertex), in
 *   camera→scene order. Construction-time merged SA pdfs (see EyeStackVertex).
 * - `camera` = the camera endpoint vertex (pos/normal; pdfs are unit-Jacobian).
 *
 * @returns merged `MergedVertex[]` of length `n = c + e + 3` and `selectedS`.
 */
export function assembleMergedConnectionPath(args: {
  readonly lightChain: ReadonlyArray<LightStackVertex>;
  readonly eyeChain: ReadonlyArray<EyeStackVertex>;
  readonly camera: { readonly position: Vec3; readonly normal: Vec3 };
  readonly eyeBrdfPdf: (wo: Vec3, wi: Vec3) => number;
}): { vertices: MergedVertex[]; selectedS: number } {
  const { lightChain, eyeChain, camera, eyeBrdfPdf } = args;
  const c = lightChain.length - 1; // light connection vertex index
  const e = eyeChain.length - 1; // eye connection vertex (current bounce) depth
  const Lc = lightChain[c]!;
  const Ee = eyeChain[e]!;

  const connDir = normalize(sub(Lc.position, Ee.position)); // E_e → L_c
  const eToLc = connDir; // wi at E_e toward light
  const lcToE = normalize(sub(Ee.position, Lc.position)); // L_c → E_e

  // ── Connection-induced overrides (PBRT MISWeight remapping) ─────────────────
  // merged pdfFwd(E_e): light-vertex outgoing density at L_c toward E_e (Lambertian).
  const fwdEe = lambertianDirectionalPdf(Lc.normal, lcToE);
  // merged pdfRev(L_c): eye-material density at E_e, wo→E_{e-1}, wi=connDir.
  let revLc: number;
  let fwdEeMinus: number | null = null;
  if (e >= 1) {
    const EeMinus = eyeChain[e - 1]!;
    const eeToEeMinus = normalize(sub(EeMinus.position, Ee.position)); // E_e → E_{e-1}
    revLc = eyeBrdfPdf(eeToEeMinus, eToLc); // wo→E_{e-1}, wi=connDir
    // merged pdfFwd(E_{e-1}): eye-material density at E_e, wo=connDir, wi→E_{e-1}.
    fwdEeMinus = eyeBrdfPdf(eToLc, eeToEeMinus);
  } else {
    // e==0: E_e is the primary hit; E_{e-1} is the camera endpoint. PBRT computes
    // qs->pdfRev = pt->Pdf(scene, ptMinus=camera, qs). The camera-ward direction
    // at E_0 is toward the camera (its wo). Use the eye vertex's view direction.
    const eeToCam = normalize(sub(camera.position, Ee.position));
    revLc = eyeBrdfPdf(eeToCam, eToLc);
  }
  // merged pdfRev(L_{c-1}): light-vertex density at L_c toward L_{c-1} (Lambertian).
  let revLcMinus: number | null = null;
  if (c >= 1) {
    const LcMinus = lightChain[c - 1]!;
    const lcToLcMinus = normalize(sub(LcMinus.position, Lc.position));
    revLcMinus = lambertianDirectionalPdf(Lc.normal, lcToLcMinus);
  }

  const vertices: MergedVertex[] = [];

  // Light side v[0..c].
  for (let i = 0; i <= c; i += 1) {
    const L = lightChain[i]!;
    let pdfRev = L.pdfRev;
    if (i === c) pdfRev = revLc;
    else if (i === c - 1 && revLcMinus != null) pdfRev = revLcMinus;
    vertices.push({
      position: L.position,
      normal: L.normal,
      pdfFwd: L.pdfFwd,
      pdfRev,
      isSpecular: L.isSpecular,
    });
  }

  // Eye side v[c+1 .. c+1+e] = E_e, E_{e-1}, …, E_0 (reverse of eyeChain order).
  for (let j = e; j >= 0; j -= 1) {
    const E = eyeChain[j]!;
    let pdfFwd = E.pdfFwd;
    if (j === e) pdfFwd = fwdEe;
    else if (j === e - 1 && fwdEeMinus != null) pdfFwd = fwdEeMinus;
    vertices.push({
      position: E.position,
      normal: E.normal,
      pdfFwd,
      pdfRev: E.pdfRev,
      isSpecular: E.isSpecular,
    });
  }

  // Camera endpoint v[n-1].
  vertices.push({
    position: camera.position,
    normal: camera.normal,
    pdfFwd: 1,
    pdfRev: 1,
    isSpecular: false,
  });

  return { vertices, selectedS: c + 1 };
}

/**
 * Enumerate all Veach §10.3 strategy path-pdfs for the merged path. Pure port of
 * `@vitrum/shared-samplers`'s `buildBDPTStrategyPDFs_full` — same area-measure
 * ratio sweep, same flipped transfer indices, same specular guard.
 */
export function buildStrategyPdfs(
  vertices: ReadonlyArray<MergedVertex>,
  selectedS: number,
  pRef: number,
): Float64Array {
  const n = vertices.length;
  const pdfs = new Float64Array(n);
  if (n === 0) return pdfs;
  pdfs[selectedS] = pRef;

  const fwdArea = (i: number): number => {
    const v = vertices[i]!;
    if (i === 0) return v.pdfFwd;
    const prev = vertices[i - 1]!;
    return convertDensitySAtoArea(v.pdfFwd, prev.position, v.position, v.normal);
  };
  const revArea = (i: number): number => {
    const v = vertices[i]!;
    if (i === n - 1) return v.pdfRev;
    const next = vertices[i + 1]!;
    return convertDensitySAtoArea(v.pdfRev, next.position, v.position, v.normal);
  };

  // Left sweep (decrement s): flip v[s-1].
  {
    let p = pRef;
    for (let s = selectedS; s > 0; s -= 1) {
      const flip = vertices[s - 1]!;
      const connNeighbor = s - 2 >= 0 ? vertices[s - 2]! : undefined;
      if (flip.isSpecular || (connNeighbor?.isSpecular ?? false)) break;
      const pFwd = fwdArea(s - 1);
      const pRev = revArea(s - 1);
      if (pFwd <= 0 || pRev <= 0) break;
      p = p * (pRev / pFwd);
      pdfs[s - 1] = p;
    }
  }
  // Right sweep (increment s): flip v[s].
  {
    let p = pRef;
    for (let s = selectedS; s < n - 1; s += 1) {
      const flip = vertices[s]!;
      const connNeighbor = vertices[s + 1]!;
      if (flip.isSpecular || connNeighbor.isSpecular) break;
      const pFwd = fwdArea(s);
      const pRev = revArea(s);
      if (pFwd <= 0 || pRev <= 0) break;
      p = p * (pFwd / pRev);
      pdfs[s + 1] = p;
    }
  }
  return pdfs;
}

/** Power-heuristic (β=2) MIS weight over the strategy pdf vector. */
export function powerHeuristicWeight(
  pdfsByStrategy: Float64Array | ReadonlyArray<number>,
  selectedS: number,
  beta = 2,
): number {
  const len_ = pdfsByStrategy.length;
  if (selectedS < 0 || selectedS >= len_) return 0;
  let denom = 0;
  for (let i = 0; i < len_; i += 1) {
    const p = pdfsByStrategy[i] ?? 0;
    if (p > 0) denom += Math.pow(p, beta);
  }
  if (denom <= 0) return 0;
  const ps = pdfsByStrategy[selectedS] ?? 0;
  if (ps <= 0) return 0;
  return Math.pow(ps, beta) / denom;
}

/**
 * End-to-end full §10.3 MIS weight for the eye↔light connection: assemble the
 * merged path, enumerate strategy pdfs, return the power-heuristic weight for the
 * selected (chosen) strategy `selectedS = c+1`. `pRef` is the path probability of
 * the chosen strategy (the joint forward density of constructing the merged path
 * by this exact strategy).
 */
export function bdptConnectionMisFull(args: {
  readonly lightChain: ReadonlyArray<LightStackVertex>;
  readonly eyeChain: ReadonlyArray<EyeStackVertex>;
  readonly camera: { readonly position: Vec3; readonly normal: Vec3 };
  readonly eyeBrdfPdf: (wo: Vec3, wi: Vec3) => number;
  readonly pRef: number;
  readonly beta?: number;
}): number {
  const { vertices, selectedS } = assembleMergedConnectionPath(args);
  const pdfs = buildStrategyPdfs(vertices, selectedS, args.pRef);
  return powerHeuristicWeight(pdfs, selectedS, args.beta ?? 2);
}
