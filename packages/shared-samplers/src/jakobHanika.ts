/**
 * jakobHanika.ts — Jakob & Hanika 2019 RGB→spectrum upsampling (real solve).
 *
 * Converts a linear-sRGB colour into a 3-coefficient sigmoid model that
 * reproduces the colour when illuminated by D65 and integrated against the
 * CIE 1931 colour-matching functions.
 *
 * Reflectance model (Jakob & Hanika 2019, §3, eq. 1):
 *
 *   S(λ) = ½ + ½ · f(λ) / √(1 + f(λ)²)
 *   f(λ) = c₂·λ² + c₁·λ + c₀
 *
 * `S(λ) ∈ (0,1)` is a smooth, strictly-bounded reflectance — exactly the
 * three-parameter "sigmoid polynomial" function space the paper shows is
 * sufficient to cover the entire sRGB / Rec.2020 gamut with sub-perceptible
 * error.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Approach — per-colour Gauss–Newton solve (no multi-MB LUT)
 * ────────────────────────────────────────────────────────────────────────────
 *
 * The reference implementation (the `rgb2spec` precompute tool that ships the
 * mitsuba 3 `.coeff` tables) runs a Gauss–Newton optimiser per RGB grid cell
 * and stores the result in a 3-D table. We run the SAME optimiser, but
 * on-the-fly per colour at call time. For vitrum's use case (a handful of
 * distinct glass-cell tints upsampled once at scene build, not per fragment)
 * this trades a one-time ~24 MB table download for a few dozen microseconds of
 * Newton iteration per unique colour — a strictly better deal for a browser
 * library, and it produces the genuine paper-accurate coefficients rather than
 * an approximation.
 *
 * Objective (faithful to the reference): minimise the residual between the
 * target colour and the colour reproduced by S(λ) under D65, measured in a
 * perceptually-uniform CIE L*a*b* space so the optimiser spends its accuracy
 * budget where the eye is sensitive. The Jacobian is computed ANALYTICALLY
 * via the chain rule through the sigmoid reflectance, the CIE XYZ integral,
 * and the CIE-Lab cube-root nonlinearity (see `coeffsToLabJacobian`), giving
 * exact derivatives with no finite-difference step-size tuning.
 *
 * Numerical conditioning: λ spans [380, 780] nm, so λ² ≈ 6·10⁵. Fitting the
 * polynomial in raw nm gives a wildly ill-conditioned normal-equations matrix.
 * Following the reference, we fit in a remapped coordinate where λ is mapped
 * affinely to roughly [0, 1] (`t = (λ − λ_min)/(λ_max − λ_min)`), solve there,
 * then ALGEBRAICALLY EXPAND the coefficients back to raw-nm space so the
 * returned `[c₀, c₁, c₂]` plug directly into the GLSL `evalSpectrum(coeffs,
 * λ_nm)` mirror with no shader change.
 *
 * References:
 *   Jakob, W. & Hanika, J. 2019, "A Low-Dimensional Function Space for
 *   Efficient Spectral Upsampling", Computer Graphics Forum 38(2)
 *   (Eurographics 2019). https://rgl.epfl.ch/publications/Jakob2019Spectral
 *
 *   Reference `rgb2spec` solver (Gauss–Newton + CIE-Lab residual):
 *   https://github.com/mitsuba-renderer/rgb2spec
 *
 *   Phase 6 Sprint 8 spec: plan/archive/phase-6-roadmap.md §Sprint 8.
 *   GLSL mirror: the native WebGL2 BSDF port's `evalSpectrum` implementation.
 */

import {
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_MAX,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  CIE_D65_TABLE,
} from './cieCmf.js';

const LAMBDA_MIN = CIE_LAMBDA_MIN;
const LAMBDA_MAX = CIE_LAMBDA_MAX;

// ────────────────────────────────────────────────────────────────────────────
// Smooth sigmoid (Jakob & Hanika §3, eq. 1)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Bounded reflectance sigmoid: maps ℝ → (0, 1).
 *
 *   S = ½ + x / (2·√(1 + x²))
 *
 * Numerically safe for large |x| (no exp overflow); approaches 0/1 smoothly.
 */
function sigmoid(x: number): number {
  if (!isFinite(x)) return x > 0 ? 1 : 0;
  return 0.5 + x / (2 * Math.sqrt(1 + x * x));
}

/**
 * Derivative of the sigmoid: S'(x) = 1 / (2·(1 + x²)^{3/2}).
 *
 * Used by the analytic Jacobian of the Gauss–Newton solve.
 */
function sigmoidPrime(x: number): number {
  if (!isFinite(x)) return 0;
  const d = 1 + x * x;
  return 1 / (2 * d * Math.sqrt(d));
}

// ────────────────────────────────────────────────────────────────────────────
// Precomputed colour-system integration weights
// ────────────────────────────────────────────────────────────────────────────
//
// For a reflectance S(λ) under illuminant D65, the reproduced tristimulus is
//
//   X = (1/N) Σ_i  S(λ_i)·D65(λ_i)·x̄(λ_i)·Δλ        (and Y, Z analogously)
//   N =          Σ_i        D65(λ_i)·ȳ(λ_i)·Δλ        (luminance normaliser)
//
// so that a perfect white reflector (S ≡ 1) maps to Y = 1. We fold Δλ and the
// D65 SPD into per-sample weights once at module load. `WHITE_XYZ` is the
// tristimulus of the D65 white point under this exact discretisation — the
// reference Lab transform whitepoint, so a flat S ≡ 1 round-trips to Lab
// (100, 0, 0) and back to linear-sRGB (1, 1, 1) by construction.

interface ColorTables {
  /** Per-sample weight D65(λ)·x̄(λ)·Δλ / N. */
  readonly wx: Float64Array;
  /** Per-sample weight D65(λ)·ȳ(λ)·Δλ / N. */
  readonly wy: Float64Array;
  /** Per-sample weight D65(λ)·z̄(λ)·Δλ / N. */
  readonly wz: Float64Array;
  /** Affinely-remapped wavelength sample, t_i = (λ_i − λ_min)/(λ_max − λ_min). */
  readonly t: Float64Array;
  /** D65 white-point tristimulus under this discretisation (Y == 1). */
  readonly white: readonly [number, number, number];
}

function buildColorTables(): ColorTables {
  const n = CIE_TABLE_LENGTH;
  const wx = new Float64Array(n);
  const wy = new Float64Array(n);
  const wz = new Float64Array(n);
  const t = new Float64Array(n);

  // Luminance normaliser N = Σ D65·ȳ·Δλ (so a unit reflector has Y = 1).
  let normY = 0;
  for (let i = 0; i < n; i++) {
    normY += (CIE_D65_TABLE[i] ?? 0) * (CIE_Y_TABLE[i] ?? 0) * CIE_LAMBDA_STEP;
  }
  const invN = normY > 0 ? 1 / normY : 0;

  const span = LAMBDA_MAX - LAMBDA_MIN;
  let wX = 0;
  let wY = 0;
  let wZ = 0;
  for (let i = 0; i < n; i++) {
    const lambda = LAMBDA_MIN + i * CIE_LAMBDA_STEP;
    const d65 = CIE_D65_TABLE[i] ?? 0;
    const ex = d65 * (CIE_X_TABLE[i] ?? 0) * CIE_LAMBDA_STEP * invN;
    const ey = d65 * (CIE_Y_TABLE[i] ?? 0) * CIE_LAMBDA_STEP * invN;
    const ez = d65 * (CIE_Z_TABLE[i] ?? 0) * CIE_LAMBDA_STEP * invN;
    wx[i] = ex;
    wy[i] = ey;
    wz[i] = ez;
    t[i] = (lambda - LAMBDA_MIN) / span;
    // White point = integral of the weights themselves (S ≡ 1).
    wX += ex;
    wY += ey;
    wZ += ez;
  }

  return { wx, wy, wz, t, white: [wX, wY, wZ] };
}

const TABLES = buildColorTables();

// ────────────────────────────────────────────────────────────────────────────
// Colour-space helpers (XYZ ↔ linear sRGB, XYZ → CIE L*a*b*)
// ────────────────────────────────────────────────────────────────────────────
//
// We keep these local (rather than importing xyzToLinearSRGB from cieCmf) so
// the solver can also go linear-sRGB → XYZ, and so the Lab transform uses the
// exact discretised D65 white point above as its reference white.

/** Linear sRGB → CIE XYZ (D65), IEC 61966-2-1:1999 forward matrix. */
function linearSRGBToXYZ(r: number, g: number, b: number): [number, number, number] {
  return [
    0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
    0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
    0.0193339 * r + 0.1191920 * g + 0.9503041 * b,
  ];
}

/** CIE L*a*b* nonlinearity. */
function labF(c: number): number {
  const eps = 216 / 24389; // (6/29)³
  const kappa = 24389 / 27; // (29/3)³
  return c > eps ? Math.cbrt(c) : (kappa * c + 16) / 116;
}

/** CIE XYZ → L*a*b* relative to the discretised D65 white point. */
function xyzToLab(x: number, y: number, z: number): [number, number, number] {
  const [wX, wY, wZ] = TABLES.white;
  const fx = labF(x / (wX || 1));
  const fy = labF(y / (wY || 1));
  const fz = labF(z / (wZ || 1));
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** Derivative of the CIE L*a*b* nonlinearity f(c). */
function labFPrime(c: number): number {
  const eps = 216 / 24389; // (6/29)³
  const kappa = 24389 / 27; // (29/3)³
  return c > eps ? 1 / (3 * Math.cbrt(c * c)) : kappa / 116;
}

// ────────────────────────────────────────────────────────────────────────────
// Forward model: sigmoid coefficients (in remapped t-space) → reproduced Lab,
// with analytic Jacobian ∂Lab/∂(a,b,c)
// ────────────────────────────────────────────────────────────────────────────

interface LabAndJacobian {
  /** Reproduced colour in CIE L*a*b*. */
  readonly lab: [number, number, number];
  /** 3×3 Jacobian, J[row][col] = ∂Lab[row]/∂coeff[col], coeff = (a, b, c). */
  readonly jacobian: number[][];
}

/**
 * Reproduce the CIE L*a*b* of the reflectance S(t) = sigmoid(a·t² + b·t + c)
 * under D65 AND its analytic Jacobian w.r.t. (a, b, c).
 *
 * Chain rule:  ∂Lab/∂coeff_j = (∂Lab/∂XYZ)·(∂XYZ/∂S)·(∂S/∂arg)·(∂arg/∂coeff_j)
 * with ∂arg/∂(a,b,c) = (t², t, 1).
 */
function coeffsToLabJacobian(a: number, b: number, c: number): LabAndJacobian {
  const { wx, wy, wz, t } = TABLES;

  // Accumulate XYZ and ∂XYZ/∂coeff in one pass.
  let X = 0;
  let Y = 0;
  let Z = 0;
  // dXYZ[channel][coeff]
  const dX = [0, 0, 0];
  const dY = [0, 0, 0];
  const dZ = [0, 0, 0];

  for (let i = 0; i < CIE_TABLE_LENGTH; i++) {
    const ti = t[i]!;
    const arg = a * ti * ti + b * ti + c;
    const s = sigmoid(arg);
    const sp = sigmoidPrime(arg); // ∂S/∂arg
    const exi = wx[i]!;
    const eyi = wy[i]!;
    const ezi = wz[i]!;
    X += s * exi;
    Y += s * eyi;
    Z += s * ezi;
    // ∂arg/∂coeff = (t², t, 1)
    const dArg = [ti * ti, ti, 1];
    for (let j = 0; j < 3; j++) {
      const dS = sp * dArg[j]!;
      dX[j]! += dS * exi;
      dY[j]! += dS * eyi;
      dZ[j]! += dS * ezi;
    }
  }

  const [wX, wY, wZ] = TABLES.white;
  const xr = X / (wX || 1);
  const yr = Y / (wY || 1);
  const zr = Z / (wZ || 1);
  const fx = labF(xr);
  const fy = labF(yr);
  const fz = labF(zr);
  const lab: [number, number, number] = [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];

  // ∂f/∂coeff via ∂f/∂(normalised channel) · (1/white) · ∂XYZ/∂coeff
  const fxp = labFPrime(xr) / (wX || 1);
  const fyp = labFPrime(yr) / (wY || 1);
  const fzp = labFPrime(zr) / (wZ || 1);

  const jacobian: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let j = 0; j < 3; j++) {
    const dfx = fxp * dX[j]!;
    const dfy = fyp * dY[j]!;
    const dfz = fzp * dZ[j]!;
    jacobian[0]![j] = 116 * dfy; // ∂L*/∂coeff
    jacobian[1]![j] = 500 * (dfx - dfy); // ∂a*/∂coeff
    jacobian[2]![j] = 200 * (dfy - dfz); // ∂b*/∂coeff
  }

  return { lab, jacobian };
}

// ────────────────────────────────────────────────────────────────────────────
// Gauss–Newton solver (Jakob & Hanika §4.1; rgb2spec reference)
// ────────────────────────────────────────────────────────────────────────────
//
// Minimise ‖Lab(coeffs) − Lab_target‖² over the 3 coefficients via Newton's
// method with an analytic Jacobian and a back-tracking line search.
// The residual is the 3-vector (ΔL, Δa, Δb); the Jacobian is the 3×3 matrix
// ∂Lab/∂coeff. We solve J·δ = −r each step.

const MAX_ITER = 40;
const CONVERGE_RESIDUAL = 1e-4; // ‖r‖ (Lab ΔE units) at which we declare success

/** Solve the 3×3 linear system A·x = b by Gaussian elimination with pivoting. */
function solve3x3(A: number[][], rhs: [number, number, number]): [number, number, number] | null {
  // Copy into an augmented matrix.
  const m: number[][] = [
    [A[0]![0]!, A[0]![1]!, A[0]![2]!, rhs[0]],
    [A[1]![0]!, A[1]![1]!, A[1]![2]!, rhs[1]],
    [A[2]![0]!, A[2]![1]!, A[2]![2]!, rhs[2]],
  ];
  for (let col = 0; col < 3; col++) {
    // Partial pivot.
    let pivot = col;
    let best = Math.abs(m[col]![col]!);
    for (let row = col + 1; row < 3; row++) {
      const v = Math.abs(m[row]![col]!);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1e-18) return null; // singular
    if (pivot !== col) {
      const tmp = m[col]!;
      m[col] = m[pivot]!;
      m[pivot] = tmp;
    }
    const diag = m[col]![col]!;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = m[row]![col]! / diag;
      for (let k = col; k < 4; k++) {
        m[row]![k]! -= factor * m[col]![k]!;
      }
    }
  }
  // After full (Gauss–Jordan) elimination the matrix is diagonal; solution
  // x_i = m[i][3] / m[i][i].
  return [m[0]![3]! / m[0]![0]!, m[1]![3]! / m[1]![1]!, m[2]![3]! / m[2]![2]!];
}

/** Squared L2 length of a 3-vector. */
function norm2(v: [number, number, number]): number {
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

/**
 * Gauss–Newton fit. Returns coefficients (a, b, c) in t-space such that
 * S(t) = sigmoid(a·t² + b·t + c) reproduces `targetLab` under D65.
 *
 * Uses the analytic Jacobian ∂Lab/∂coeff (`coeffsToLabJacobian`) and a
 * back-tracking line search for guaranteed monotone descent.
 */
function gaussNewtonFit(
  targetLab: [number, number, number],
  initial: [number, number, number],
): [number, number, number] {
  let coeff = initial.slice() as [number, number, number];
  const init = coeffsToLabJacobian(coeff[0], coeff[1], coeff[2]);
  let jacobian = init.jacobian;
  let residual: [number, number, number] = [
    init.lab[0] - targetLab[0],
    init.lab[1] - targetLab[1],
    init.lab[2] - targetLab[2],
  ];
  let err = norm2(residual);

  for (let iter = 0; iter < MAX_ITER && err > CONVERGE_RESIDUAL * CONVERGE_RESIDUAL; iter++) {
    const step = solve3x3(jacobian, [-residual[0], -residual[1], -residual[2]]);
    if (step == null) break; // singular Jacobian — keep best so far

    // Back-tracking line search to guarantee monotone descent.
    let alpha = 1;
    let accepted = false;
    for (let ls = 0; ls < 24; ls++) {
      const trial: [number, number, number] = [
        coeff[0] + alpha * step[0],
        coeff[1] + alpha * step[1],
        coeff[2] + alpha * step[2],
      ];
      const eval_ = coeffsToLabJacobian(trial[0], trial[1], trial[2]);
      const r: [number, number, number] = [
        eval_.lab[0] - targetLab[0],
        eval_.lab[1] - targetLab[1],
        eval_.lab[2] - targetLab[2],
      ];
      const e = norm2(r);
      if (e < err) {
        coeff = trial;
        residual = r;
        err = e;
        jacobian = eval_.jacobian;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break; // no descent direction improved — converged to local min
  }

  return coeff;
}

// ────────────────────────────────────────────────────────────────────────────
// t-space → raw-nm coefficient expansion
// ────────────────────────────────────────────────────────────────────────────
//
// The GLSL mirror evaluates sigmoid(c₀ + c₁·λ + c₂·λ²) with λ in RAW nm. We
// fit in t = (λ − λ_min)/span for conditioning, so we must expand
//   a·t² + b·t + c   with t = (λ − λ_min)/span
// into the raw-nm polynomial  c₂·λ² + c₁·λ + c₀.
//
//   Let s = 1/span, m = λ_min.  t = s·(λ − m) = s·λ − s·m.
//   a·t² = a·s²·λ² − 2·a·s²·m·λ + a·s²·m²
//   b·t  =                b·s·λ − b·s·m
//   c    =                              c
//
//   ⇒ c₂ = a·s²
//     c₁ = −2·a·s²·m + b·s
//     c₀ = a·s²·m² − b·s·m + c
function expandToRawNm(a: number, b: number, c: number): [number, number, number] {
  const span = LAMBDA_MAX - LAMBDA_MIN;
  const s = 1 / span;
  const m = LAMBDA_MIN;
  const c2 = a * s * s;
  const c1 = -2 * a * s * s * m + b * s;
  const c0 = a * s * s * m * m - b * s * m + c;
  return [c0, c1, c2];
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/**
 * Real Jakob & Hanika 2019 RGB→spectral coefficient solve.
 *
 * Converts a linear-sRGB colour (components nominally in [0, 1]) into the
 * 3-coefficient sigmoid polynomial whose reflectance, under D65, reproduces
 * the colour. The returned coefficients are in RAW-nm space:
 *
 *   S(λ) = sigmoid(c₀ + c₁·λ + c₂·λ²),  sigmoid(x) = ½ + x / (2·√(1 + x²))
 *
 * and are directly consumable by the GLSL `evalSpectrum(coeffs, λ_nm)` mirror.
 *
 * @param r - Red channel, linear sRGB.
 * @param g - Green channel, linear sRGB.
 * @param b - Blue channel, linear sRGB.
 * @returns (c₀, c₁, c₂) raw-nm sigmoid-polynomial coefficients.
 */
function rgbToJakobHanikaCoefficients(
  r: number,
  g: number,
  b: number,
): readonly [number, number, number] {
  // Clamp to the representable sRGB cube. Components outside [0,1] (e.g. HDR
  // emission or out-of-gamut) are clamped; the sigmoid model represents
  // reflectances in [0,1] only.
  r = Math.max(0, Math.min(1, r));
  g = Math.max(0, Math.min(1, g));
  b = Math.max(0, Math.min(1, b));

  // Pure-black shortcut: S ≡ 0 ⇒ sigmoid(−∞). A large negative constant gives
  // S ≈ 0 across the band with c₁ = c₂ = 0.
  if (r <= 0 && g <= 0 && b <= 0) {
    return [-50, 0, 0] as const;
  }

  // Target colour in the perceptual residual space.
  const [tx, ty, tz] = linearSRGBToXYZ(r, g, b);
  const targetLab = xyzToLab(tx, ty, tz);

  // Initial guess in t-space: the maximally-conditioned flat S ≡ ½ (all coeffs
  // zero). At S = ½ the sigmoid derivative S'(0) = ½ is at its maximum, so the
  // Jacobian is well-conditioned and Newton can move freely toward the target
  // in any chromatic direction. Seeding a near-saturated flat spectrum instead
  // (S ≈ 1) lands in the sigmoid's flat tail where S' ≈ 0 and the optimiser
  // cannot escape — this is why the reference rgb2spec walks the grid outward
  // from the achromatic centre rather than seeding each cell at its own value.
  const initial: [number, number, number] = [0, 0, 0];

  const [a, bb, c] = gaussNewtonFit(targetLab, initial);
  return expandToRawNm(a, bb, c) as [number, number, number];
}

/**
 * Evaluate the sigmoid-polynomial spectrum at a wavelength using coefficients
 * produced by `rgbToSpectralCoefficients`.
 *
 *   S(λ) = sigmoid(c₀ + c₁·λ + c₂·λ²)
 *
 * @param coeffs   - (c₀, c₁, c₂) raw-nm coefficients.
 * @param lambdaNm - Wavelength in nm.
 * @returns Spectral reflectance in [0, 1].
 */
export function evaluateSpectrum(
  coeffs: readonly [number, number, number],
  lambdaNm: number,
): number {
  const [c0, c1, c2] = coeffs;
  return sigmoid(c0 + c1 * lambdaNm + c2 * lambdaNm * lambdaNm);
}

/**
 * Integrate a sigmoid-polynomial reflectance under D65 against the CIE CMFs
 * and convert the result to linear sRGB. This is the inverse of
 * `rgbToSpectralCoefficients` and is the exact round-trip the solver targets;
 * exported so callers (and tests) can verify upsampling accuracy.
 *
 * @param coeffs - (c₀, c₁, c₂) raw-nm coefficients.
 * @returns Reproduced [r, g, b] in linear sRGB (may fall slightly outside
 *          [0,1] for near-gamut-boundary colours; callers should clamp for
 *          display).
 */
export function spectralCoefficientsToRGB(
  coeffs: readonly [number, number, number],
): readonly [number, number, number] {
  // Re-evaluate in raw-nm space (the public coefficient convention) so this is
  // a true inverse of whatever `rgbToSpectralCoefficients` returned.
  const [c0, c1, c2] = coeffs;
  let X = 0;
  let Y = 0;
  let Z = 0;
  const { wx, wy, wz } = TABLES;
  for (let i = 0; i < CIE_TABLE_LENGTH; i++) {
    const lambda = LAMBDA_MIN + i * CIE_LAMBDA_STEP;
    const s = sigmoid(c0 + c1 * lambda + c2 * lambda * lambda);
    X += s * wx[i]!;
    Y += s * wy[i]!;
    Z += s * wz[i]!;
  }
  return [
    3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
    -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
    0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z,
  ] as const;
}

// ── Exported constants ────────────────────────────────────────────────────────

/** Visible range used by this implementation. */
export const VISIBLE_LAMBDA_MIN = LAMBDA_MIN;
export const VISIBLE_LAMBDA_MAX = LAMBDA_MAX;

/**
 * Stable public alias for the RGB→spectral coefficient solve. The genuine
 * Jakob & Hanika 2019 Gauss–Newton fit (see file-level docs).
 */
export const rgbToSpectralCoefficients = rgbToJakobHanikaCoefficients;
