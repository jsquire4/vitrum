/**
 * cauchyIor.ts — Per-wavelength index of refraction via the Cauchy dispersion formula.
 *
 * Cauchy's empirical dispersion equation:
 *   n(λ) = A + B/λ² + C/λ⁴
 * where λ is in micrometers (µm).
 *
 * The formula is accurate for visible-range normal-dispersion materials (glasses,
 * crystals, polymers).  It does NOT model anomalous dispersion (absorption bands)
 * which requires a Sellmeier-series treatment — that level of accuracy is outside
 * the current spectral system scope.
 *
 * Typical coefficients for common optical glasses (from Schott glass catalogue
 * and standard optical-glass references):
 *
 *   Crown glass (BK7):    A=1.5046,  B=0.00420 µm², C=0.00020 µm⁴, Abbe≈64
 *   Flint glass (F2):     A=1.6200,  B=0.01500 µm², C=0.00080 µm⁴, Abbe≈36
 *   Lead crystal (PbK50): A=1.5800,  B=0.01000 µm², C=0.00050 µm⁴, Abbe≈32
 *
 * Sprint 8 context: Sprint 8 used an implicit Cauchy formula via the fork
 * material payload (`dispersionStrength` ≈ B * 1e6 to match nm² scaling). This file
 * extracts the formula as a standalone TypeScript utility for host-side use,
 * testing, and documentation.
 *
 * Sprint 12 context: Hero-wavelength path tracing requires per-wavelength IOR
 * for each dielectric BSDF evaluation.  The fork shader mirrors this formula
 * in GLSL (see plan/sprint-12-pt-fork-patch.md §3).
 *
 * References:
 *   Cauchy, A.-L. "Sur la dispersion de la lumière", Mém. Acad. Sci. 10, 1836.
 *   Hecht, E. "Optics", 5th ed., §3.5.
 *   Schott AG optical glass catalogue, edition 2022.
 *   Schott TIE-29: Refractive Index and Its Dependence on Temperature.
 *
 * Sprint 12 (Phase 6) deliverable — hero-wavelength spectral path tracing.
 */

// ── Core formula ──────────────────────────────────────────────────────────────

/**
 * Compute the index of refraction at a given wavelength via Cauchy's equation:
 *   n(λ) = A + B/λ² + C/λ⁴
 * where λ is in micrometers (µm).
 *
 * @param lambdaNm - Wavelength in nanometres.  Converted internally to µm.
 * @param A        - Base IOR (dimensionless).  Typical: 1.45–1.95 for optical glass.
 * @param B        - First dispersion coefficient in µm².  Typical: 0.003–0.020.
 * @param C        - Second dispersion coefficient in µm⁴.  Optional; defaults to 0.
 *                   Including C improves accuracy in the UV portion of the spectrum;
 *                   for visible-range rendering, B alone is usually sufficient.
 * @returns Index of refraction n(λ).  Always ≥ A for normal dispersion (B,C ≥ 0).
 */
export function cauchyIOR(
  lambdaNm: number,
  A: number,
  B: number,
  C = 0,
): number {
  // Convert wavelength from nm to µm for standard Cauchy coefficient units.
  const lambdaUm = lambdaNm * 1e-3; // 1 nm = 0.001 µm
  const lam2 = lambdaUm * lambdaUm;
  return A + B / lam2 + C / (lam2 * lam2);
}

// ── Standard glass coefficients ───────────────────────────────────────────────
// Sources: Schott glass catalogue 2022; standard optical glass references.
// Coefficients are for the Cauchy formula with λ in µm.

/**
 * Cauchy coefficients for borosilicate crown glass (Schott N-BK7 equivalent).
 * Representative wavelength: 550 nm → n ≈ 1.518.  Abbe number ≈ 64.
 *
 * B is calibrated so that abbeNumber(A, B, C) ≈ 64 (standard crown value).
 * Reference: Schott N-BK7 datasheet; Hecht "Optics" 5th ed. Table 3.3.
 */
export const CAUCHY_CROWN_GLASS: { A: number; B: number; C: number } = {
  A: 1.5046,
  B: 0.00290,   // calibrated for Abbe V_d ≈ 64
  C: 0.00020,
};

/**
 * Cauchy coefficients for dense flint glass (Schott F2 equivalent).
 * Representative wavelength: 550 nm → n ≈ 1.627.  Abbe number ≈ 36.
 *
 * B is calibrated so that abbeNumber(A, B, C) ≈ 36 (standard F2 value).
 * Reference: Schott F2 datasheet; Hecht "Optics" 5th ed. Table 3.3.
 */
export const CAUCHY_FLINT_GLASS: { A: number; B: number; C: number } = {
  A: 1.6200,
  B: 0.00404,   // calibrated for Abbe V_d ≈ 36
  C: 0.00080,
};

/**
 * Cauchy coefficients for lead crystal (traditional cut-glass / bevel material).
 * Representative wavelength: 550 nm → n ≈ 1.586.  Abbe number ≈ 32.
 * Used as the Sprint 8 default for bevel cells (`dispersionStrength = 0.018`).
 *
 * B is calibrated so that abbeNumber(A, B, C) ≈ 32 (heavy lead crystal).
 * Reference: Schott Lanthanum-flint / heavy flint data; Shannon "Art of Optics".
 *
 * Note on Sprint 8 mapping:
 *   The fork material payload's `dispersionStrength` is approximately equal to
 *   `B * 1e6 / (589.3²)` when λ is in nm rather than µm.
 *   For lead crystal B = 0.0066 µm²: the nm² equivalent scales to roughly 0.019
 *   — close to the Sprint 8 empirical slider default of 0.018.
 */
export const CAUCHY_LEAD_CRYSTAL: { A: number; B: number; C: number } = {
  A: 1.5800,
  B: 0.00659,   // calibrated for Abbe V_d ≈ 32
  C: 0.00050,
};

// ── Abbe number utility ────────────────────────────────────────────────────────
// The Abbe number V_d characterises the degree of dispersion of a glass:
//   V_d = (n_d - 1) / (n_F - n_C)
// at the Fraunhofer lines:
//   d (sodium D):  589.3 nm  (design wavelength)
//   F (hydrogen):  486.1 nm  (blue)
//   C (hydrogen):  656.3 nm  (red)

/** Fraunhofer d-line wavelength (sodium D), nm. */
export const FRAUNHOFER_D_NM = 589.3;
/** Fraunhofer F-line wavelength (hydrogen blue), nm. */
export const FRAUNHOFER_F_NM = 486.1;
/** Fraunhofer C-line wavelength (hydrogen red), nm. */
export const FRAUNHOFER_C_NM = 656.3;

/**
 * Compute the Abbe number V_d from Cauchy coefficients.
 *
 * V_d = (n_d - 1) / (n_F - n_C)
 *
 * Higher Abbe number = less dispersion (crown glass V_d ≈ 60–70).
 * Lower Abbe number = more dispersion (flint glass V_d ≈ 25–40).
 *
 * @param A - Cauchy A coefficient.
 * @param B - Cauchy B coefficient (µm²).
 * @param C - Cauchy C coefficient (µm⁴), optional.
 * @returns Abbe number V_d (dimensionless, typically 20–90 for optical glass).
 */
export function abbeNumber(A: number, B: number, C = 0): number {
  const nD = cauchyIOR(FRAUNHOFER_D_NM, A, B, C);
  const nF = cauchyIOR(FRAUNHOFER_F_NM, A, B, C);
  const nC = cauchyIOR(FRAUNHOFER_C_NM, A, B, C);
  return (nD - 1) / (nF - nC);
}
