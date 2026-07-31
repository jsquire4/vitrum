/**
 * bdptMIS.ts — BDPT MIS helpers: full Veach §10.3 (T2.H4).
 *
 * Complete strategy-PDF enumeration for a BDPT path of arbitrary length.
 * Reproduces the canonical PBRT-v4 `MISWeight` recurrence (`integrators.cpp`):
 * a pure ratio of AREA-measure forward/reverse densities, walking the actual
 * path vertices, to compute all k+1 strategy PDFs in O(k) time. The vertices
 * here carry SOLID-ANGLE pdfs, so each is converted to area measure on the fly
 * via `convertDensitySAtoArea` (PBRT's `Vertex::ConvertDensity`, a
 * destination-cosine-only "half-G"). Handles specular vertices (zero-weight)
 * and the camera/light endpoint corner cases.
 *
 * Removed 2026-05-18: the Sprint-10c `_partial` helpers were a single-
 * strategy MIS aid retained for fork-side dispatch (2-strategy case only).
 * No production consumers remained; the `_full` helpers below are the
 * canonical path.
 *
 * References:
 *   - Veach 1997, PhD thesis §9.2 (power heuristic), §10.3 (BDPT MIS weights),
 *     Algorithm 10.4 (strategy PDF enumeration).
 *   - Pharr et al. 2023, PBR 4th ed. §16.3.5, Eq. 16.16 (recursive ratio).
 *
 * @module bdptMIS
 */
import {
  requireFinite,
  requireFiniteVec3,
  requireInteger,
  requireNonNegative,
  requirePositive,
  saturatingPositiveMultiply,
} from './numericGuards.js';

type ScaledDirection = {
  readonly unit: readonly [number, number, number];
  readonly length: number;
};

function scaledDirection(
  value: readonly [number, number, number],
): ScaledDirection | null {
  const scale = Math.max(
    Math.abs(value[0]),
    Math.abs(value[1]),
    Math.abs(value[2]),
  );
  if (!(scale > 0) || !Number.isFinite(scale)) return null;
  const scaled = [
    value[0] / scale,
    value[1] / scale,
    value[2] / scale,
  ] as const;
  const scaledLength = Math.hypot(...scaled);
  if (!(scaledLength > 0) || !Number.isFinite(scaledLength)) return null;
  return {
    unit: [
      scaled[0] / scaledLength,
      scaled[1] / scaledLength,
      scaled[2] / scaledLength,
    ],
    length: scale * scaledLength,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Full Veach §10.3 implementation (T2.H4)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Compute the geometry term G(xᵢ ↔ xⱼ) between two path vertices.
 *
 * ```
 * G(xᵢ ↔ xⱼ) = |cos θᵢ · cos θⱼ| / ‖xᵢ − xⱼ‖²
 * ```
 *
 * where θᵢ is the angle at xᵢ between the connection direction and the
 * shading normal, and likewise for θⱼ.
 *
 * Returns 0 when the two positions coincide (degenerate connection) or when
 * either cosine is 0 (backface / tangent incidence).
 *
 * Reference: Veach 1997 §8.3.2, Eq. 8.10.
 *
 * @param posI      - world-space position of vertex i [x, y, z]
 * @param normalI   - unit shading normal at vertex i [x, y, z]
 * @param posJ      - world-space position of vertex j [x, y, z]
 * @param normalJ   - unit shading normal at vertex j [x, y, z]
 * @returns geometry term ≥ 0
 */
export function geometricTermG(
  posI: readonly [number, number, number],
  normalI: readonly [number, number, number],
  posJ: readonly [number, number, number],
  normalJ: readonly [number, number, number],
): number {
  // Direction from i to j (unnormalized)
  requireFiniteVec3(posI, 'geometricTermG.posI');
  requireFiniteVec3(normalI, 'geometricTermG.normalI');
  requireFiniteVec3(posJ, 'geometricTermG.posJ');
  requireFiniteVec3(normalJ, 'geometricTermG.normalJ');
  const dx = posJ[0] - posI[0];
  const dy = posJ[1] - posI[1];
  const dz = posJ[2] - posI[2];
  requireFinite(dx, 'geometricTermG.dx');
  requireFinite(dy, 'geometricTermG.dy');
  requireFinite(dz, 'geometricTermG.dz');
  const connection = scaledDirection([dx, dy, dz]);

  if (connection == null) return 0;
  const dist2 = connection.length * connection.length;

  // Unit direction i→j
  const [wx, wy, wz] = connection.unit;

  // |cos θᵢ| = |normalI · w|, |cos θⱼ| = |normalJ · (−w)|
  const normalIDirection = scaledDirection(normalI);
  const normalJDirection = scaledDirection(normalJ);
  if (normalIDirection == null || normalJDirection == null) return 0;
  const cosI = Math.abs(
    normalIDirection.unit[0] * wx +
    normalIDirection.unit[1] * wy +
    normalIDirection.unit[2] * wz,
  );
  const cosJ = Math.abs(
    normalJDirection.unit[0] * wx +
    normalJDirection.unit[1] * wy +
    normalJDirection.unit[2] * wz,
  );

  const result = (cosI * cosJ) / dist2;
  return Number.isFinite(result) ? result : Number.MAX_VALUE;
}

// ── convertDensitySAtoArea (PBRT Vertex::ConvertDensity) ──────────────────────

/**
 * Convert a solid-angle PDF for sampling vertex `dest` (as seen from `from`)
 * into the equivalent **area-measure** PDF.
 *
 * This is exactly PBRT-v4's `Vertex::ConvertDensity` (`integrators.cpp`):
 *
 * ```cpp
 * Float Vertex::ConvertDensity(Float pdf, const Vertex &next) const {
 *     Vector3f w = next.p() - p();
 *     Float invDist2 = 1 / LengthSquared(w);
 *     if (next.IsOnSurface())
 *         pdf *= AbsDot(next.ng(), w * std::sqrt(invDist2));
 *     return pdf * invDist2;
 * }
 * ```
 *
 * The Jacobian for the solid-angle → area change of variables subtends only the
 * **destination** vertex's cosine: dω = dA_dest · |cos θ_dest| / ‖from−dest‖².
 * The source-vertex cosine is NOT part of this Jacobian — it is already carried
 * inside the directional sampling density `pdfSA`. (See Veach §8.2.2.2 / PBRT
 * §16.1.1.) Using the FULL geometry term G here — which also multiplies the
 * source cosine — would over-count and produce a geometry-dependent BIAS in the
 * MIS weights. This destination-only "half-G" is the physically correct factor.
 *
 * Returns `pdfSA` unchanged when the positions coincide (degenerate; treated as
 * a unit Jacobian so endpoint/coincident cases don't blow up).
 *
 * @param pdfSA    - solid-angle PDF of sampling `dest` from `from`
 * @param fromPos  - position of the vertex the direction was sampled at
 * @param destPos  - position of the vertex being sampled (carries the cosine)
 * @param destNorm - unit shading normal at `dest`
 * @returns area-measure PDF (≥ 0)
 */
function convertDensitySAtoArea(
  pdfSA: number,
  fromPos: readonly [number, number, number],
  destPos: readonly [number, number, number],
  destNorm: readonly [number, number, number],
): number {
  const dx = destPos[0] - fromPos[0];
  const dy = destPos[1] - fromPos[1];
  const dz = destPos[2] - fromPos[2];
  const connection = scaledDirection([dx, dy, dz]);
  if (connection == null) return pdfSA;
  const dist2 = connection.length * connection.length;

  const normalDirection = scaledDirection(destNorm);
  if (normalDirection == null) return 0;
  const cosDest = Math.abs(
    normalDirection.unit[0] * connection.unit[0] +
    normalDirection.unit[1] * connection.unit[1] +
    normalDirection.unit[2] * connection.unit[2],
  );
  const result = (pdfSA * cosDest) / dist2;
  return Number.isFinite(result) ? result : Number.MAX_VALUE;
}

// ── BDPTFullVertex ────────────────────────────────────────────────────────────

/**
 * A single vertex for the full Veach §10.3 strategy enumeration.
 *
 * The full path is represented as a merged array `vertices[0..k]` where:
 *   - `vertices[0]` = light endpoint (emitter surface)
 *   - `vertices[k]` = camera endpoint (camera position or primary hit)
 *   - interior vertices carry BSDF hits from either subpath
 *
 * Index ordering follows Veach §10.3 Figure 10.4:
 *   light endpoint → scene bounces → camera endpoint
 *
 * All PDF fields are in **solid-angle measure** as required by the ratio sweep.
 * If your caller has area-measure PDFs at surface vertices, convert them with
 *   `pdfSA = pdfArea / G(xᵢ, xᵢ₊₁)`
 * before building this array.
 */
export interface BDPTFullVertex {
  /** World-space position [x, y, z]. */
  readonly position: readonly [number, number, number];
  /** Unit shading normal [x, y, z]. Required for G(x↔y) evaluation. */
  readonly normal: readonly [number, number, number];
  /**
   * Forward PDF (solid-angle) for sampling this vertex along the path's
   * forward direction (light→camera).
   */
  readonly pdfFwd: number;
  /**
   * Reverse PDF (solid-angle) for sampling this vertex in the reverse
   * direction (camera→light). Required by the recursive ratio sweep
   * (PBR4e Eq. 16.16).
   */
  readonly pdfRev: number;
  /**
   * True when this vertex lies on a specular (delta-function BSDF) surface.
   * A strategy whose explicit connection edge touches this vertex has weight 0
   * per Veach §10.3.5 — the delta PDF cannot be evaluated by a deterministic
   * connection from the other subpath. The recurrence must nevertheless walk
   * across this vertex so that farther strategies with non-delta connection
   * edges remain eligible.
   */
  readonly isSpecular: boolean;
}

// ── buildBDPTStrategyPDFs_full ────────────────────────────────────────────────

/**
 * Enumerate all Veach §10.3 strategy PDFs for a full BDPT path.
 *
 * For a path with `n = vertices.length` vertices (k = n−1 segments), there are
 * `n` strategies indexed by `s ∈ {0, …, n−1}` where `s` is the number of light
 * subpath vertices (s=0 = pure camera path, s=n−1 = pure light path). Light
 * subpath = v_0…v_{s−1}; camera subpath (traced in reverse) = v_s…v_{n−1}.
 *
 * The reference strategy `selectedS` is the one actually used to construct the
 * path. Its probability is `pRef`. All other strategy probabilities are obtained
 * by the canonical PBRT-v4 `MISWeight` recurrence (`integrators.cpp`), which is
 * a pure ratio of **area-measure** densities:
 *
 *   - Light loop:  `ri *= pdfRev_area[i] / pdfFwd_area[i]`  for i = s−1, s−2, …, 0
 *   - Camera loop: `ri *= pdfRev_area[i] / pdfFwd_area[i]`  for i = t−1, t−2, …, 1
 *
 * PBRT stores AREA-measure pdfs (G is baked in at `Vertex::ConvertDensity`), so
 * its sweep carries NO explicit G inside the loop. This repo's
 * {@link BDPTFullVertex} stores **solid-angle** pdfs (see the type doc), so we
 * convert each pdf to area measure on the fly with {@link convertDensitySAtoArea}
 * (PBRT's `ConvertDensity` — a destination-cosine-only "half-G", NOT the full
 * geometry term) and then take the same pure ratio. The two formulations are
 * identical; expressing the loop with the full two-cosine G would inject a
 * geometry-dependent bias and is wrong.
 *
 * **Transfer-vertex index.** Decrementing the strategy (`s → s−1`) flips
 * ownership of vertex `v_{s−1}` from the forward (light) chain to the reverse
 * (camera) chain, so the ratio uses `vertices[s−1]`'s densities — NOT
 * `vertices[s]`'s. Incrementing (`s → s+1`) flips `v_s`, so the ratio uses
 * `vertices[s]`'s densities. The area pdfs are:
 *
 * ```
 *   pFwd_area(i) = ConvertDensity(pdfFwd_SA(i), from = v_{i−1}, dest = v_i)
 *   pRev_area(i) = ConvertDensity(pdfRev_SA(i), from = v_{i+1}, dest = v_i)
 * ```
 *
 * (`pdfFwd` describes arriving at v_i from the forward neighbour v_{i−1};
 * `pdfRev` describes arriving at v_i from the reverse neighbour v_{i+1}.) At the
 * light endpoint (v_0) there is no v_{−1}, and at the camera endpoint (v_{n−1})
 * there is no v_n, so the missing-neighbour Jacobian is treated as 1 — matching
 * PBRT, where the endpoint pdfs are already stored in area measure.
 *
 * **Specular zero-weight rule (Veach §10.3.5):** a hypothetical strategy whose
 * connection edge touches a specular (delta-BSDF) vertex cannot be sampled by an
 * explicit connection, so its pdf is left at 0. The recurrence itself continues
 * across the delta vertex: farther strategies may place their connection on a
 * fully non-delta edge and are still valid. As in PBRT's `MISWeight`, zero
 * densities are remapped to one for the ratio walk; the connection-edge mask,
 * rather than an early loop exit, excludes the delta techniques.
 *
 * @param vertices    - merged path [v_0=light endpoint, …, v_{n-1}=camera endpoint]
 * @param selectedS   - index of the chosen strategy (0-based light vertex count)
 * @param pRef        - path probability of the selected strategy (must be > 0)
 * @returns Float64Array of per-strategy path PDFs, length `vertices.length`
 */
export function buildBDPTStrategyPDFs_full(
  vertices: ReadonlyArray<BDPTFullVertex>,
  selectedS: number,
  pRef: number,
): Float64Array {
  const n = vertices.length;
  if (n === 0) return new Float64Array(0);

  requireInteger(selectedS, 'buildBDPTStrategyPDFs_full.selectedS', 0, n - 1);
  requirePositive(pRef, 'buildBDPTStrategyPDFs_full.pRef');
  for (let i = 0; i < n; i++) {
    const v = vertices[i]!;
    requireFiniteVec3(v.position, `buildBDPTStrategyPDFs_full.vertices[${i}].position`);
    requireFiniteVec3(v.normal, `buildBDPTStrategyPDFs_full.vertices[${i}].normal`);
    requireNonNegative(v.pdfFwd, `buildBDPTStrategyPDFs_full.vertices[${i}].pdfFwd`);
    requireNonNegative(v.pdfRev, `buildBDPTStrategyPDFs_full.vertices[${i}].pdfRev`);
  }

  const pdfs = new Float64Array(n);
  pdfs[selectedS] = pRef;

  // Area-measure forward density of vertex i: pdfFwd_SA(i) converted through the
  // edge (v_{i−1} → v_i). At the light endpoint (i=0) there is no incoming edge,
  // so the pdf is already area measure (unit Jacobian).
  const fwdArea = (i: number): number => {
    const v = vertices[i]!;
    if (i === 0) return v.pdfFwd;
    const prev = vertices[i - 1]!;
    return convertDensitySAtoArea(v.pdfFwd, prev.position, v.position, v.normal);
  };

  // Area-measure reverse density of vertex i: pdfRev_SA(i) converted through the
  // edge (v_{i+1} → v_i). At the camera endpoint (i=n−1) there is no incoming
  // reverse edge, so the pdf is already area measure (unit Jacobian).
  const revArea = (i: number): number => {
    const v = vertices[i]!;
    if (i === n - 1) return v.pdfRev;
    const next = vertices[i + 1]!;
    return convertDensitySAtoArea(v.pdfRev, next.position, v.position, v.normal);
  };

  // PBRT's MISWeight uses Remap0 while walking ratios. Delta distributions have
  // no finite solid-angle density, so their stored directional PDF is commonly
  // zero; treating that sentinel as a literal zero would terminate the
  // recurrence and incorrectly discard valid strategies beyond the delta run.
  const remapZeroDensity = (pdf: number): number => pdf === 0 ? 1 : pdf;

  // ── Left sweep (decrement s): flip v_{s−1}; p_{s−1} = p_s · pRev(s−1)/pFwd(s−1) ──
  {
    let p = pRef;
    for (let s = selectedS; s > 0; s--) {
      const flip = vertices[s - 1]!; // vertex transferred light → camera

      // Strategy s−1 connects light subpath v_0…v_{s−2} to camera subpath
      // v_{s−1}…. Its connection edge is (v_{s−2}, v_{s−1}). If either endpoint
      // of that edge is specular the strategy cannot be sampled. Do not stop the
      // recurrence: a farther connection edge can be fully non-specular.
      const connNeighbor = s - 2 >= 0 ? vertices[s - 2]! : undefined;
      const connectionIsDelta =
        flip.isSpecular || (connNeighbor?.isSpecular ?? false);

      const pFwd = remapZeroDensity(fwdArea(s - 1));
      const pRev = remapZeroDensity(revArea(s - 1));
      p = saturatingPositiveMultiply(p, pRev / pFwd);
      if (!connectionIsDelta) pdfs[s - 1] = p;
    }
  }

  // ── Right sweep (increment s): flip v_s; p_{s+1} = p_s · pFwd(s)/pRev(s) ──
  {
    let p = pRef;
    for (let s = selectedS; s < n - 1; s++) {
      const flip = vertices[s]!; // vertex transferred camera → light

      // Strategy s+1 connects light subpath v_0…v_s to camera subpath v_{s+1}….
      // Its connection edge is (v_s, v_{s+1}). If either endpoint is specular the
      // strategy cannot be sampled. Continue walking so later non-delta
      // connection edges remain eligible.
      const connNeighbor = vertices[s + 1]!;
      const connectionIsDelta = flip.isSpecular || connNeighbor.isSpecular;

      const pFwd = remapZeroDensity(fwdArea(s));
      const pRev = remapZeroDensity(revArea(s));
      p = saturatingPositiveMultiply(p, pFwd / pRev);
      if (!connectionIsDelta) pdfs[s + 1] = p;
    }
  }

  return pdfs;
}

/**
 * Depth limits for the explicit-connection strategy family evaluated by a
 * bounded BDPT implementation.
 *
 * `maxLightVertices` includes the sampled emitter endpoint. `maxEyeVertices`
 * counts scene-surface vertices on the eye subpath and excludes the camera.
 * The `s=1` direct-light strategy samples its emitter independently and does
 * not require a stored light-subpath vertex budget.
 */
export interface BDPTExplicitConnectionLimits {
  readonly maxLightVertices: number;
  readonly maxEyeVertices: number;
}

function explicitConnectionStrategyIsValidUnchecked(
  pathVertexCount: number,
  strategyS: number,
  maxLightVertices: number,
  maxEyeVertices: number,
): boolean {
  // An explicit connection needs at least one light vertex and the camera
  // endpoint. s=0 is the pure-eye technique and is owned separately.
  // s=n-1 is the t=1 light-subpath-to-camera technique: it has no eye-surface
  // vertices, but it is still an explicit strategy once the renderer provides
  // a camera splat estimator. The remaining counts must fit the actually
  // allocated subpaths.
  if (pathVertexCount < 3 || strategyS < 1 || strategyS >= pathVertexCount) {
    return false;
  }
  const lightVertices = strategyS;
  const eyeVertices = pathVertexCount - strategyS - 1;
  return (
    (strategyS < 2 || lightVertices <= maxLightVertices) &&
    eyeVertices <= maxEyeVertices
  );
}

/**
 * Binding-free WGSL form of {@link bdptExplicitConnectionStrategyIsValid}.
 *
 * The pt-webgpu denominator composes this declaration directly so the CPU
 * oracle and production finite-root strategy family cannot evolve separately.
 */
export const BDPT_EXPLICIT_STRATEGY_MASK_WGSL = /* wgsl */ `
fn bdptExplicitConnectionStrategyIsValid(
  pathVertexCount: u32,
  strategyS: u32,
  maxLightVertices: u32,
  maxEyeVertices: u32,
) -> bool {
  if (pathVertexCount < 3u) {
    return false;
  }
  if (strategyS < 1u || strategyS >= pathVertexCount) {
    return false;
  }
  let lightVertices = strategyS;
  let eyeVertices = pathVertexCount - strategyS - 1u;
  return
    (strategyS < 2u || lightVertices <= maxLightVertices) &&
    eyeVertices <= maxEyeVertices;
}
`;

/** Return whether strategy `s` is sampled by the bounded explicit family. */
export function bdptExplicitConnectionStrategyIsValid(
  pathVertexCount: number,
  strategyS: number,
  limits: BDPTExplicitConnectionLimits,
): boolean {
  requireInteger(
    pathVertexCount,
    'bdptExplicitConnectionStrategyIsValid.pathVertexCount',
  );
  requireInteger(
    strategyS,
    'bdptExplicitConnectionStrategyIsValid.strategyS',
  );
  const maxLightVertices = requireInteger(
    limits.maxLightVertices,
    'bdptExplicitConnectionStrategyIsValid.maxLightVertices',
  );
  const maxEyeVertices = requireInteger(
    limits.maxEyeVertices,
    'bdptExplicitConnectionStrategyIsValid.maxEyeVertices',
  );
  return explicitConnectionStrategyIsValidUnchecked(
    pathVertexCount,
    strategyS,
    maxLightVertices,
    maxEyeVertices,
  );
}

/**
 * Copy a full recurrence vector while zeroing techniques that the bounded
 * explicit-connection kernel did not sample. Applying this before the power
 * heuristic prevents the separately-owned s=0 technique or depth-truncated
 * strategies from diluting the weights of the techniques that were actually
 * evaluated. The s=n-1/t=1 camera-splat strategy is part of this family.
 */
export function maskBDPTExplicitConnectionStrategyPDFs(
  pdfsByStrategy: ReadonlyArray<number> | Float64Array,
  limits: BDPTExplicitConnectionLimits,
): Float64Array {
  const maxLightVertices = requireInteger(
    limits.maxLightVertices,
    'maskBDPTExplicitConnectionStrategyPDFs.maxLightVertices',
  );
  const maxEyeVertices = requireInteger(
    limits.maxEyeVertices,
    'maskBDPTExplicitConnectionStrategyPDFs.maxEyeVertices',
  );
  const masked = new Float64Array(pdfsByStrategy.length);
  for (let s = 0; s < pdfsByStrategy.length; s += 1) {
    const pdf = pdfsByStrategy[s] ?? 0;
    requireNonNegative(
      pdf,
      `maskBDPTExplicitConnectionStrategyPDFs.pdfsByStrategy[${s}]`,
    );
    if (
      explicitConnectionStrategyIsValidUnchecked(
        pdfsByStrategy.length,
        s,
        maxLightVertices,
        maxEyeVertices,
      )
    ) {
      masked[s] = pdf;
    }
  }
  return masked;
}

// ── bdptConnectionMIS_full ────────────────────────────────────────────────────

/**
 * Compute the full Veach §10.3 power-heuristic MIS weight for one BDPT strategy.
 *
 * Given the per-strategy path-PDF vector from {@link buildBDPTStrategyPDFs_full},
 * returns the power-heuristic weight for `selectedS`:
 *
 * ```
 * w_s = p_s^β / Σᵢ p_i^β
 * ```
 *
 * with β=2 (Veach §9.2 recommended exponent). The standard BDPT renderer uses
 * one sample per strategy (nᵢ = 1 for all i), so the `nᵢ` factor is omitted.
 *
 * **Graceful degradation:**
 *   - All-zero PDFs: returns 0.
 *   - `selectedS` out of range: returns 0.
 *   - Selected strategy PDF is 0: returns 0.
 *
 * @param pdfsByStrategy - Float64Array from `buildBDPTStrategyPDFs_full`, length n
 * @param selectedS      - index of the strategy whose weight to compute
 * @param beta           - power heuristic exponent (default 2; β=1 = balance)
 * @returns MIS weight ∈ [0, 1]
 */
export function bdptConnectionMIS_full(
  pdfsByStrategy: ReadonlyArray<number> | Float64Array,
  selectedS: number,
  beta: number = 2,
): number {
  const len = pdfsByStrategy.length;
  if (!Number.isInteger(selectedS)) return 0;
  if (selectedS < 0 || selectedS >= len) return 0;
  requirePositive(beta, 'bdptConnectionMIS_full.beta');

  let maxPdf = 0;
  for (let i = 0; i < len; i++) {
    const p = pdfsByStrategy[i] ?? 0;
    requireNonNegative(p, `bdptConnectionMIS_full.pdfsByStrategy[${i}]`);
    maxPdf = Math.max(maxPdf, p);
  }
  if (maxPdf === 0) return 0;
  let denominator = 0;
  for (let i = 0; i < len; i++) denominator += Math.pow((pdfsByStrategy[i] ?? 0) / maxPdf, beta);

  const p_s = pdfsByStrategy[selectedS] ?? 0;
  if (p_s <= 0) return 0;

  return Math.pow(p_s / maxPdf, beta) / denominator;
}
