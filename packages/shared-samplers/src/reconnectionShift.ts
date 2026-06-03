/**
 * reconnectionShift.ts — GRIS / ReSTIR-PT reconnection-shift CPU oracle.
 *
 * This is the load-bearing reference implementation of the GRIS *reconnection
 * shift* mapping `T` and its Jacobian determinant `|∂T/∂·|`. It is THREE-free
 * and GPU-free — the same role `bdptMIS.ts` plays for BDPT MIS: a deterministic,
 * first-principles oracle that the WGSL shift + Jacobian — now live in
 * `@vitrum/walkaround-hybrid`'s `grisReuse.wgsl.ts`, composed into the spatial +
 * temporal GI reuse passes — are verified against (exactly as
 * `bdptConnectionMisFull.ts` mirrors `bdptMIS.ts`).
 *
 * GRIS evolves ReSTIR from resampling samples to resampling *paths*: a candidate
 * path generated for a neighbour pixel can be reused at this pixel only after a
 * shift map T re-roots it into this pixel's integration domain, and its
 * contribution must be re-weighted by the change-of-variables Jacobian |∂T/∂·|.
 * The *reconnection shift* is the workhorse shift: it keeps the suffix of the
 * path (everything from the reconnection vertex onward, including its cached
 * outgoing radiance) FIXED in world space, and re-connects it to the offset
 * domain's primary/visible vertex by a fresh deterministic edge.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The mapping (single reconnection edge, primary → reconnection vertex)
 * ════════════════════════════════════════════════════════════════════════════
 * A reconnection path is parameterised here by:
 *   • x1   — the primary/visible vertex (the shift's DOMAIN root; differs
 *            between base and offset pixels),
 *   • x2   — the reconnection vertex (SHARED / held fixed by the shift),
 *   • n2   — the surface normal at x2 (held fixed),
 *   • the suffix (Lo at x2, deeper bounces) — held fixed, not modeled here.
 *
 * The shift T : (x1, x2) ↦ (x1', x2) replaces the base primary vertex x1 with
 * the offset primary vertex x1' and KEEPS x2. Geometrically it swaps the
 * reconnection EDGE x1↔x2 for x1'↔x2.
 *
 * Jacobian. The reconnection-edge integrand is written in SOLID-ANGLE measure
 * about the primary vertex, but the resampling integral runs over the AREA
 * measure of the reconnection vertex (its world-space position is the shared
 * resampling coordinate). The solid-angle ⇄ area change of variables at the
 * reconnection vertex carries the geometry term
 *
 *     G(x1 ↔ x2) = |cos θ_out(x2)| / ‖x1 − x2‖²
 *
 * where θ_out(x2) is the angle at x2 between its normal n2 and the connection
 * direction. The shift's Jacobian determinant is the RATIO of the offset and
 * base geometry terms (Lin et al. 2022, "Generalized Resampled Importance
 * Sampling: Foundations of ReSTIR", SIGGRAPH 2022, Eq. 12; ReSTIR-PT
 * reconnection-shift Jacobian):
 *
 *     |∂T/∂·| = G(x1' ↔ x2) / G(x1 ↔ x2)
 *             = ( |cos θ'_out| / ‖x1' − x2‖² ) / ( |cos θ_out| / ‖x1 − x2‖² )
 *             = ( |cos θ'_out| · ‖x1 − x2‖² ) / ( |cos θ_out| · ‖x1' − x2‖² )
 *
 * (The primary-vertex BSDF/cosine factors are carried by the resampling target
 * function, NOT by this geometric Jacobian — the same destination-cosine-only
 * "half-G" distinction made in `bdptMIS.ts`'s `convertDensitySAtoArea`. The
 * source-vertex cosine is part of the integrand's directional density, so it is
 * cancelled in the ratio and does NOT appear in the Jacobian.)
 *
 * Reciprocity. The inverse shift T⁻¹ : (x1', x2) ↦ (x1, x2) is the same mapping
 * with the roles of x1 and x1' swapped, so its Jacobian is the reciprocal:
 *
 *     |∂T⁻¹/∂·| = G(x1 ↔ x2) / G(x1' ↔ x2) = 1 / |∂T/∂·|
 *
 * hence  |∂T/∂·| · |∂T⁻¹/∂·| = 1  (verified to machine ε by the tests).
 *
 * References:
 *   - Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai 2022,
 *     "Generalized Resampled Importance Sampling: Foundations of ReSTIR",
 *     ACM TOG 41(4) (SIGGRAPH 2022), §5 (shift mappings: reconnection shift)
 *     and Eq. 12 (shift Jacobian as a ratio of reconnection-edge geometry).
 *   - Bitterli et al. 2020 (ReSTIR DI) / 2021 (ReSTIR GI), reconnection sample.
 *
 * @module reconnectionShift
 */

type Vec3 = readonly [number, number, number];

// ── small vector helpers (local; no THREE) ───────────────────────────────────

function sub(a: Vec3, b: Vec3): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function len(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

// ── reconnection-path configuration ──────────────────────────────────────────

/**
 * A reconnection path configuration for the GRIS reconnection shift.
 *
 * Only the geometry the reconnection-edge Jacobian needs is modeled: the primary
 * vertex `x1`, the (shared) reconnection vertex `x2`, and the reconnection-vertex
 * normal `n2`. The path suffix (Lo at x2 and any deeper bounces) is held fixed by
 * the reconnection shift and does not enter the Jacobian, so it is not modeled.
 *
 * Maps directly onto the WGSL `ReservoirPT` fields: `x1` = `xv`, `x2` = `xs`,
 * `n2` = `ns`. (`wi_recon` = normalize(x2 − x1), `distRecon` = ‖x2 − x1‖, and
 * `cosReconOut` = |cos θ_out| are derivable from these and are what the GPU will
 * cache; see reservoirGi.wgsl.ts.)
 */
export interface ReconnectionPath {
  /** Primary/visible vertex (the shift's domain root). */
  readonly x1: Vec3;
  /** Reconnection vertex (held fixed by the shift). */
  readonly x2: Vec3;
  /** Unit surface normal at the reconnection vertex `x2`. */
  readonly n2: Vec3;
}

/**
 * Geometry term G(x1 ↔ x2) at the reconnection vertex, in the destination-cosine
 * ("half-G") convention used by the reconnection-shift Jacobian:
 *
 * ```
 * G(x1 ↔ x2) = |cos θ_out(x2)| / ‖x1 − x2‖²
 * ```
 *
 * where θ_out(x2) is the angle at x2 between its normal `n2` and the connection
 * direction x1 → x2 (equivalently x2 → x1; the absolute cosine is the same).
 * Only the reconnection-vertex cosine appears — the primary-vertex cosine is
 * carried by the integrand's directional density and cancels in the Jacobian
 * ratio (see module doc / `bdptMIS.ts` `convertDensitySAtoArea`).
 *
 * Returns 0 when x1 and x2 coincide (degenerate edge) or when the connection is
 * tangent to n2 (cos θ_out = 0).
 *
 * @param x1 - primary/visible vertex
 * @param x2 - reconnection vertex
 * @param n2 - unit normal at x2
 * @returns reconnection-edge geometry term ≥ 0
 */
export function reconnectionGeometryTerm(x1: Vec3, x2: Vec3, n2: Vec3): number {
  const d = sub(x2, x1);
  const dist = len(d);
  if (dist <= 0) return 0;
  const cosOut = Math.abs(dot(n2, d) / dist);
  return cosOut / (dist * dist);
}

// ── the reconnection shift T and its inverse T⁻¹ ──────────────────────────────

/**
 * Apply the GRIS reconnection shift `T`: re-root a BASE reconnection path onto
 * an OFFSET primary vertex, holding the reconnection vertex (and its suffix)
 * fixed.
 *
 * ```
 * T : (x1, x2, n2)  ↦  (x1', x2, n2)
 * ```
 *
 * The reconnection vertex `x2` and its normal `n2` are world-space invariants of
 * the shift; only the primary vertex changes (base `base.x1` → `offsetX1`). The
 * returned path is the offset-domain path whose contribution must be re-weighted
 * by {@link reconnectionJacobian}.
 *
 * @param base     - the base reconnection path (its `x2`/`n2` are kept)
 * @param offsetX1 - the offset domain's primary/visible vertex
 * @returns the shifted (offset-domain) reconnection path
 */
export function reconnectionShift(base: ReconnectionPath, offsetX1: Vec3): ReconnectionPath {
  return { x1: [offsetX1[0], offsetX1[1], offsetX1[2]], x2: base.x2, n2: base.n2 };
}

/**
 * Apply the INVERSE reconnection shift `T⁻¹`: map an OFFSET reconnection path
 * back onto a BASE primary vertex. Structurally identical to {@link
 * reconnectionShift} (the shift is its own form with the primary vertices
 * swapped), so `reconnectionShift(reconnectionShiftInverse(p, a), p.x1)` returns
 * `p` and vice-versa — the round-trip identity the tests pin.
 *
 * @param offset - the offset reconnection path (its `x2`/`n2` are kept)
 * @param baseX1 - the base domain's primary/visible vertex
 * @returns the un-shifted (base-domain) reconnection path
 */
export function reconnectionShiftInverse(offset: ReconnectionPath, baseX1: Vec3): ReconnectionPath {
  return { x1: [baseX1[0], baseX1[1], baseX1[2]], x2: offset.x2, n2: offset.n2 };
}

/**
 * Jacobian determinant `|∂T/∂·|` of the GRIS reconnection shift mapping `base`
 * (primary vertex `base.x1`) to `shifted` (primary vertex `shifted.x1`), with a
 * shared reconnection vertex (`base.x2 == shifted.x2`, `base.n2 == shifted.n2`).
 *
 * ```
 * |∂T/∂·| = G(x1' ↔ x2) / G(x1 ↔ x2)
 *         = ( |cos θ'_out| / ‖x1' − x2‖² ) / ( |cos θ_out| / ‖x1 − x2‖² )
 * ```
 *
 * (Lin et al. 2022 Eq. 12; ReSTIR-PT reconnection-shift Jacobian.) The base path
 * is the DENOMINATOR (it is the path whose density we are re-mapping FROM); the
 * shifted/offset path is the NUMERATOR.
 *
 * Returns 0 when the base geometry term is 0 (degenerate base edge — nothing to
 * re-map from) and `Infinity` is avoided by that guard; returns 0 when the
 * shifted geometry term is 0 (the offset edge is degenerate/tangent → the shift
 * is non-invertible there and the sample carries no weight).
 *
 * @param base    - base reconnection path (Jacobian denominator)
 * @param shifted - shifted/offset reconnection path (Jacobian numerator)
 * @returns the Jacobian determinant ≥ 0
 */
export function reconnectionJacobian(base: ReconnectionPath, shifted: ReconnectionPath): number {
  const gBase = reconnectionGeometryTerm(base.x1, base.x2, base.n2);
  if (gBase <= 0) return 0;
  const gShifted = reconnectionGeometryTerm(shifted.x1, shifted.x2, shifted.n2);
  return gShifted / gBase;
}
